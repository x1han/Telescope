//! Module 0: Transcript Evidence Resolver
//!
//! Resolves cohort-level evidence per transcript (Tier A/B/C classification,
//! subsumption detection, EM/PAL eligibility) per spec v0.2.1 F1.3 and F5.3.
//!
//! Inputs:
//! - `FlatEvidenceGraph` produced by Module 2 (`evidence::extract_evidence`).
//! - `tx_to_gene` map (TSV with header `transcript_id\tgene_id`).
//! - `cohort_size` (number of BAMs pooled). Required parameter (P0 correction).
//! - `tes_cluster_radius` (default 50 bp per F6.4).
//!
//! Outputs:
//! - `telescope.transcript_evidence.parquet` with the 10-column F5.3 schema.
//!
//! ## P0 corrections baked into this implementation
//! 1. Output field is `pal_candidate` (structural pre-filter), not
//!    `pal_eligible`. The latter is decided post-EM in `summary.rs`.
//! 2. `cohort_size` is an explicit required parameter, not inferred.
//! 3. Per-transcript molecule membership is stored as sorted `Vec<u32>`
//!    (binary search), NOT `HashSet<usize>` — avoids ~640 MB overhead on
//!    200K-transcript catalogues.
//! 4. Transcripts with no `tx_to_gene` mapping are tagged
//!    `"missing_gene_id"` and emit a single deduplicated `eprintln!`
//!    warning per transcript.
//! 5. Output schema is 10 columns per F5.3 (the table; spec narrative
//!    said "12" but the locked table has 10).
//! 6. TES-cluster gating: enforced via the proxy heuristic
//!    `is_alter_tes_marker(tx_id)` — transcripts containing `AlterTes`
//!    or `_Alter` substrings are excluded from the subsumption scan
//!    (F6.4 AlterTES protection). Fine-grained TES coordinate
//!    clustering from reference annotation is deferred (D-zone).
//!
//! ## Complexity
//! - Inverse-index build: O(N_reads × avg candidates).
//! - Per-(gene) subsumption scan: O(|TierB_gene| × |TierA_gene| ×
//!   avg |molecules(T_B)| × log avg |molecules(T_A)|). With the sorted
//!   `Vec<u32>` + binary-search representation this avoids the
//!   ~640 MB HashSet overhead that F6.5 forbids. It is NOT strictly
//!   O(K) in the absolute sense (worst case is superlinear in Tier A ×
//!   Tier B per gene) but is tractable for typical inputs (~10 tx/gene,
//!   ~100 reads/tx → ~70K ops/gene → seconds for a 20K-gene catalogue).
//!   True per-transcript 3' end clustering (F6.4 condition #2) would
//!   further prune the per-gene Tier A × Tier B product; deferred to v0.2.2.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{BooleanBuilder, Float32Builder, StringBuilder, UInt32Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::evidence::FlatEvidenceGraph;

/// Magic gene_id string written when `tx_to_gene` has no entry (F5.3 + P0 #4).
pub const MISSING_GENE_ID: &str = "missing_gene_id";

/// Structural pre-filter for `pal_candidate`: compatible reads must be at least
/// this value at the Module 0 stage. The *final* `pal_eligible` decision is
/// `weight_sum >= 10` post-EM in `summary.rs` (F5.4).
pub const PAL_CANDIDATE_MIN_READS: u32 = 10;

/// Default TES cluster radius (bp) per F6.4.
pub const DEFAULT_TES_CLUSTER_RADIUS: usize = 50;

/// Resolver configuration (P0 #2: cohort_size is required, explicit).
#[derive(Debug, Clone)]
pub struct ResolverConfig {
    /// Number of BAMs pooled into the cohort. Must be ≥ 1.
    pub cohort_size: usize,
    /// Excluded from subsumption scan iff transcript IDs contain the
    /// `AlterTes` / `_Alter` substring. The bp radius itself is
    /// structural metadata for logging; transcript-level 3' end coordinates
    /// are not yet read from reference annotations (D-zone).
    pub tes_cluster_radius: usize,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        Self {
            cohort_size: 1,
            tes_cluster_radius: DEFAULT_TES_CLUSTER_RADIUS,
        }
    }
}

/// One row of `telescope.transcript_evidence.parquet` (10 cols per F5.3).
#[derive(Debug, Clone)]
pub struct TranscriptEvidence {
    pub transcript_id: String,
    pub gene_id: String,
    pub compatible_reads: u32,
    pub unique_reads: u32,
    pub tier: String,
    pub is_subsumed: bool,
    pub subsumption_target: String,
    pub subsumption_score: f32,
    pub em_eligible: bool,
    pub pal_candidate: bool,
}

/// Build per-transcript sorted `Vec<u32>` of molecule indices.
///
/// For every molecule `m` in the graph, append `m`'s index into the
/// `Vec<u32>` of every transcript that is a candidate for `m`.
///
/// Post-condition: for any `(mol_idx, tx_idx)`, `tx_idx ∈ candidates[m]`
/// iff `mol_idx ∈ inv[tx_idx]`. This inverse index is what makes the
/// "sorted Vec<u32> + binary search" subsumption check tractable (P0 #3).
fn build_molecule_inverse_index(graph: &FlatEvidenceGraph) -> Vec<Vec<u32>> {
    let num_tx = graph.transcript_names.len();
    let mut inv: Vec<Vec<u32>> = vec![Vec::new(); num_tx];
    for (mol_idx, mol) in graph.molecules.iter().enumerate() {
        let cands = &graph.candidate_pool[mol.candidate_offset as usize
            ..(mol.candidate_offset + mol.candidate_count as u32) as usize];
        for &tx_id in cands {
            inv[tx_id as usize].push(mol_idx as u32);
        }
    }
    for v in &mut inv {
        // Sort + dedup: a mol appearing in a tx's candidate set more than
        // once would be a graph bug (Module 2 already dedups within a
        // molecule), but defensive dedup costs O(N) per tx.
        v.sort_unstable();
        v.dedup();
    }
    inv
}

/// Resolve cohort-level evidence for every transcript in the graph.
///
/// `missing_gene_warn_set` deduplicates the per-transcript warning: one
/// `eprintln!` per transcript with no `tx_to_gene` mapping. The caller
/// should pass a long-lived set so warnings don't repeat across multiple
/// invocations (e.g., for tests).
pub fn resolve_transcript_evidence(
    graph: &FlatEvidenceGraph,
    tx_to_gene: &HashMap<String, String>,
    config: &ResolverConfig,
    missing_gene_warn_set: &mut HashSet<String>,
) -> Vec<TranscriptEvidence> {
    // P1 reviewer finding: enforce basic cohort_size sanity at the resolver
    // level (not just CLI). Library embedders that bypass the CLI would
    // otherwise silently use cohort_size=0, defeating the F6.2 cohort-scope
    // guard. The CLI additionally enforces cohort_size == len(bams);
    // that cross-check cannot be performed here since the graph does not
    // carry the BAM list.
    assert!(
        config.cohort_size >= 1,
        "ResolverConfig::cohort_size must be >= 1, got {}",
        config.cohort_size
    );

    let num_tx = graph.transcript_names.len();
    let inv = build_molecule_inverse_index(graph);

    // 1. Per-transcript statistics + Tier classification (F6.2).
    let mut records: Vec<TranscriptEvidence> = Vec::with_capacity(num_tx);
    for (tx_idx, tx_id_ref) in graph.transcript_names.iter().enumerate() {
        let tx_id = tx_id_ref.as_str();
        let tx_mol_ids = &inv[tx_idx];

        let mut unique: u32 = 0;
        let mut compatible: u32 = 0;
        for &mol_idx in tx_mol_ids {
            let mol = &graph.molecules[mol_idx as usize];
            compatible += 1;
            if mol.candidate_count == 1 {
                unique += 1;
            }
        }

        let gene_id = match tx_to_gene.get(tx_id) {
            Some(g) => g.clone(),
            None => {
                let tx_id_owned = tx_id_ref.clone();
                if missing_gene_warn_set.insert(tx_id_owned.clone()) {
                    eprintln!(
                        "[telescope][WARN] transcript '{tx_id_owned}' has no gene_id mapping; \
                         emitting '{MISSING_GENE_ID}'"
                    );
                }
                MISSING_GENE_ID.to_string()
            }
        };

        let tier = classify_tier(unique, compatible);

        records.push(TranscriptEvidence {
            transcript_id: tx_id_ref.clone(),
            gene_id,
            compatible_reads: compatible,
            unique_reads: unique,
            tier,
            is_subsumed: false,
            subsumption_target: String::new(),
            subsumption_score: 0.0,
            em_eligible: false,
            pal_candidate: false,
        });
    }

    // 2. Subsumption scan within each gene (F6.4-F6.5).
    apply_subsumption_scan(&mut records, &inv);

    // 3. Finalize em_eligible / pal_candidate per F6.3.
    for r in &mut records {
        r.em_eligible = (r.tier == "TierA") || (r.tier == "TierB" && !r.is_subsumed);
        r.pal_candidate = r.em_eligible && (r.compatible_reads >= PAL_CANDIDATE_MIN_READS);
    }

    // Informational only: cohort_size must be ≥ 1; if user passes 0 they
    // are explicitly bypassing the cohort-scope guard (F6.2). Record the
    // fact in stderr rather than fail.
    if config.cohort_size == 0 {
        eprintln!(
            "[telescope][WARN] ResolverConfig.cohort_size == 0; F6.2 cohort scope guard bypassed"
        );
    }

    records
}

/// F6.2 Tier classification: high → TierA; ambiguous → TierB; none → TierC.
fn classify_tier(unique: u32, compatible: u32) -> String {
    if unique >= 1 {
        "TierA".to_string()
    } else if compatible >= 1 {
        "TierB".to_string()
    } else {
        "TierC".to_string()
    }
}

/// Proxy heuristic for F6.4 AlterTES protection: transcripts whose ID
/// contains `AlterTes` or `_Alter` represent alternative TES isoforms
/// and must not be the subject or the target of a subsumption check.
fn is_alter_tes_marker(tx_id: &str) -> bool {
    tx_id.contains("AlterTes") || tx_id.contains("_Alter")
}

/// Apply F6.4 subsumption scan within (gene_id) groups.
///
/// For v0.2.1: cluster by `gene_id` only. Transcript-level 3' end
/// coordinates (for fine-grained TES-radius clustering) would require
/// reading transcript annotation tables; documented as a v-next
/// improvement. The string proxy above enforces the AlterTES protection
/// guarantee that the spec requires.
fn apply_subsumption_scan(records: &mut [TranscriptEvidence], inv: &[Vec<u32>]) {
    // Group records by gene_id (records is parallel to `inv` by index).
    let mut by_gene: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, r) in records.iter().enumerate() {
        by_gene.entry(r.gene_id.clone()).or_default().push(i);
    }

    for indices in by_gene.values() {
        let tier_a: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|&i| {
                let r = &records[i];
                r.tier == "TierA" && !is_alter_tes_marker(&r.transcript_id)
            })
            .collect();
        let tier_b: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|&i| {
                let r = &records[i];
                r.tier == "TierB" && !is_alter_tes_marker(&r.transcript_id)
            })
            .collect();
        if tier_a.is_empty() || tier_b.is_empty() {
            continue;
        }

        // For each Tier B tx, find a Tier A tx whose molecule set is a
        // superset. First match wins (audit-only target).
        for &b_idx in &tier_b {
            let b_mols = &inv[b_idx];
            if b_mols.is_empty() {
                continue;
            }
            let mut subsumed_by: Option<(usize, f32)> = None;
            for &a_idx in &tier_a {
                let a_mols = &inv[a_idx];
                let (overlap, all_in) = subset_score(a_mols, b_mols);
                let score = overlap as f32 / b_mols.len() as f32;
                if all_in {
                    subsumed_by = Some((a_idx, score));
                    break;
                }
            }
            if let Some((a_idx, score)) = subsumed_by {
                records[b_idx].is_subsumed = true;
                records[b_idx].subsumption_target = records[a_idx].transcript_id.clone();
                records[b_idx].subsumption_score = score;
            }
        }
    }
}

/// Compute `(|a ∩ b|, |a ⊇ b|)` for two sorted `u32` slices.
///
/// - `a` = candidate molecules of the Tier A target (assumed superset candidate).
/// - `b` = candidate molecules of the Tier B subject.
///
/// `all_in` is true iff every element of `b` is also in `a` (i.e. `b ⊆ a`),
/// which is the F6.4 "fully subsumed" condition.
fn subset_score(a: &[u32], b: &[u32]) -> (usize, bool) {
    let mut overlap = 0usize;
    let mut all_in = true;
    for &x in b {
        if a.binary_search(&x).is_ok() {
            overlap += 1;
        } else {
            all_in = false;
        }
    }
    (overlap, all_in)
}

/// Write `telescope.transcript_evidence.parquet` (10-column F5.3 schema).
pub fn write_transcript_evidence_parquet<P: AsRef<Path>>(
    records: &[TranscriptEvidence],
    path: P,
) -> Result<()> {
    let mut tx_b = StringBuilder::new();
    let mut gene_b = StringBuilder::new();
    let mut comp_b = UInt32Builder::new();
    let mut uniq_b = UInt32Builder::new();
    let mut tier_b = StringBuilder::new();
    let mut sub_b = BooleanBuilder::new();
    let mut subtgt_b = StringBuilder::new();
    let mut subsc_b = Float32Builder::new();
    let mut em_b = BooleanBuilder::new();
    let mut pal_b = BooleanBuilder::new();

    for r in records {
        tx_b.append_value(&r.transcript_id);
        gene_b.append_value(&r.gene_id);
        comp_b.append_value(r.compatible_reads);
        uniq_b.append_value(r.unique_reads);
        tier_b.append_value(&r.tier);
        sub_b.append_value(r.is_subsumed);
        subtgt_b.append_value(&r.subsumption_target);
        subsc_b.append_value(r.subsumption_score);
        em_b.append_value(r.em_eligible);
        pal_b.append_value(r.pal_candidate);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("transcript_id", DataType::Utf8, false),
        Field::new("gene_id", DataType::Utf8, false),
        Field::new("compatible_reads", DataType::UInt32, false),
        Field::new("unique_reads", DataType::UInt32, false),
        Field::new("tier", DataType::Utf8, false),
        Field::new("is_subsumed", DataType::Boolean, false),
        Field::new("subsumption_target", DataType::Utf8, false),
        Field::new("subsumption_score", DataType::Float32, false),
        Field::new("em_eligible", DataType::Boolean, false),
        Field::new("pal_candidate", DataType::Boolean, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(tx_b.finish()),
            Arc::new(gene_b.finish()),
            Arc::new(comp_b.finish()),
            Arc::new(uniq_b.finish()),
            Arc::new(tier_b.finish()),
            Arc::new(sub_b.finish()),
            Arc::new(subtgt_b.finish()),
            Arc::new(subsc_b.finish()),
            Arc::new(em_b.finish()),
            Arc::new(pal_b.finish()),
        ],
    )
    .context("Failed to build transcript_evidence RecordBatch")?;

    let out_file = File::create(&path).with_context(|| {
        format!(
            "Failed to create transcript_evidence file: {:?}",
            path.as_ref()
        )
    })?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(out_file, schema, Some(props))
        .context("Failed to create ArrowWriter for transcript_evidence")?;
    writer.write(&batch).context("Failed to write batch")?;
    writer.close().context("Failed to close writer")?;
    Ok(())
}

/// Pool molecules across multiple BAMs into a single graph.
///
/// All BAMs must share the same reference catalog (`@SQ` lines). Read IDs
/// are namespaced by sample to avoid collisions across BAMs from distinct
/// samples (Module 2 already dedups within one molecule's primary +
/// supplementary alignments; this guards against the cross-BAM case).
pub fn pool_evidence_graphs<P, S>(
    bam_paths: &[(P, S)],
    tes_slack: usize,
) -> Result<(FlatEvidenceGraph, usize)>
where
    P: AsRef<Path>,
    S: AsRef<str>,
{
    use crate::evidence::extract_evidence;

    let mut pooled = FlatEvidenceGraph {
        molecules: Vec::new(),
        candidate_pool: Vec::new(),
        candidate_as: Vec::new(),
        candidate_nm: Vec::new(),
        candidate_mapq: Vec::new(),
        candidate_5p_gap: Vec::new(),
        transcript_names: Vec::new(),
    };
    let mut count: usize = 0;

    for (bam_path, sample_id) in bam_paths {
        let g = extract_evidence(bam_path.as_ref(), tes_slack)
            .with_context(|| format!("extract_evidence failed for {:?}", bam_path.as_ref()))?;
        if pooled.transcript_names.is_empty() {
            pooled.transcript_names = g.transcript_names;
        } else if pooled.transcript_names != g.transcript_names {
            anyhow::bail!(
                "BAM {:?} has a different transcript catalog than the first BAM; \
                 cohort BAMs must share the same reference.",
                bam_path.as_ref()
            );
        }
        let offset_u = pooled.candidate_pool.len() as u32;
        for mol in g.molecules {
            pooled.molecules.push(crate::evidence::MoleculeEvidence {
                read_id: format!("{}::{}", sample_id.as_ref(), mol.read_id),
                log_pal: mol.log_pal,
                candidate_offset: mol.candidate_offset + offset_u,
                candidate_count: mol.candidate_count,
            });
        }
        pooled.candidate_pool.extend(g.candidate_pool);
        pooled.candidate_as.extend(g.candidate_as);
        pooled.candidate_nm.extend(g.candidate_nm);
        pooled.candidate_mapq.extend(g.candidate_mapq);
        pooled.candidate_5p_gap.extend(g.candidate_5p_gap);
        // Lockstep invariant across pooled BAMs: the alignment-evidence pools
        // stay index-aligned with `candidate_pool`.
        //
        // THIS IS THE HIGHEST-SEVERITY SILENT-FAILURE SITE IN THE 5' CHANGE.
        // Every `extend` above MUST be present: omitting one leaves that pool
        // short while `candidate_pool` grows, and `candidate_5p_gap[o + c]` is
        // then read from the WRONG BAM's coordinate frame — a wrong 5' gap, not
        // a crash. `cargo test` stays green because the only guard is a
        // `debug_assert_eq!` on LENGTHS, and the forgotten pool is length 0 on
        // the first BAM; the assert trips only from the second BAM onward, and
        // not at all in a release build.
        debug_assert_eq!(pooled.candidate_pool.len(), pooled.candidate_as.len());
        debug_assert_eq!(pooled.candidate_pool.len(), pooled.candidate_nm.len());
        debug_assert_eq!(pooled.candidate_pool.len(), pooled.candidate_mapq.len());
        debug_assert_eq!(pooled.candidate_pool.len(), pooled.candidate_5p_gap.len());
        count += 1;
    }
    Ok((pooled, count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::{FlatEvidenceGraph, MoleculeEvidence};

    fn mk_mol(rid: &str, offset: u32, count: u16, log_pal: f32) -> MoleculeEvidence {
        MoleculeEvidence {
            read_id: rid.to_string(),
            log_pal,
            candidate_offset: offset,
            candidate_count: count,
        }
    }

    /// Guards the cohort pooling path against a MISSING `extend`.
    ///
    /// `pool_evidence_graphs` concatenates five lockstep pools across BAMs. A
    /// forgotten `extend` for any one of them leaves that pool short while
    /// `candidate_pool` grows, and every index `o + c` then reads the WRONG
    /// BAM's coordinate frame — a silently wrong 5' gap, not a crash. The
    /// in-function `debug_assert_eq!` is the only other guard and it is
    /// compiled out in release; before this test existed, deleting
    /// `pooled.candidate_5p_gap.extend(..)` left the ENTIRE suite green,
    /// because nothing called `pool_evidence_graphs` at all.
    ///
    /// Two BAMs with DISTINCT gaps are used so the assertion is about the
    /// pooled FRAME, not just the length: if the second BAM's gap overwrote or
    /// dropped the first's, the values below would not match.
    ///
    /// Mutation target: removing any one of the five `extend` calls in
    /// `pool_evidence_graphs` turns this red.
    #[test]
    fn test_pool_evidence_graphs_keeps_all_pools_aligned() {
        use crate::evidence::test_support::{synth_record_with_gap, write_synthetic_bam};

        // BAM 1: one read, g_k = 5.   BAM 2: one read, g_k = 40.
        let bam1 = write_synthetic_bam(
            "pool1",
            vec![synth_record_with_gap("p1", 0, 5, false, Some(1000), 1, 60)],
        );
        let bam2 = write_synthetic_bam(
            "pool2",
            vec![synth_record_with_gap("p2", 0, 40, false, Some(900), 2, 30)],
        );

        let pairs = [(bam1.as_path(), "S1"), (bam2.as_path(), "S2")];
        let (pooled, n) = pool_evidence_graphs(&pairs, 35).unwrap();
        let _ = std::fs::remove_file(&bam1);
        let _ = std::fs::remove_file(&bam2);

        assert_eq!(n, 2);
        assert_eq!(
            pooled.molecules.len(),
            2,
            "read ids are namespaced per sample"
        );
        let total: usize = pooled
            .molecules
            .iter()
            .map(|m| m.candidate_count as usize)
            .sum();
        assert_eq!(total, pooled.candidate_pool.len());

        // Every pool must be exactly as long as `candidate_pool` — asserted
        // unconditionally, unlike the in-function `debug_assert_eq!`, so this
        // still bites in a release build.
        assert_eq!(pooled.candidate_as.len(), pooled.candidate_pool.len());
        assert_eq!(pooled.candidate_nm.len(), pooled.candidate_pool.len());
        assert_eq!(pooled.candidate_mapq.len(), pooled.candidate_pool.len());
        assert_eq!(
            pooled.candidate_5p_gap.len(),
            pooled.candidate_pool.len(),
            "candidate_5p_gap desynchronised: a missing `extend` reads the 5' gap \
             from the wrong BAM's frame"
        );

        // The pooled 5' gaps are the two BAMs' gaps, in BAM order, and each
        // molecule's `candidate_offset` indexes its own slot.
        let mut gaps_by_sample: Vec<(String, u32)> = pooled
            .molecules
            .iter()
            .map(|m| {
                let o = m.candidate_offset as usize;
                assert_eq!(m.candidate_count, 1, "{} is single-candidate", m.read_id);
                (m.read_id.clone(), pooled.candidate_5p_gap[o])
            })
            .collect();
        gaps_by_sample.sort();
        assert_eq!(
            gaps_by_sample,
            vec![("S1::p1".to_string(), 5), ("S2::p2".to_string(), 40)],
            "pooled gaps must be the two BAMs' own gaps, each in its own frame"
        );

        // ... and the AS/NM/MAPQ pools must likewise carry each BAM's own
        // values, not a repeat of the first BAM's.
        assert_eq!(pooled.candidate_as, vec![1000, 900]);
        assert_eq!(pooled.candidate_nm, vec![1, 2]);
        assert_eq!(pooled.candidate_mapq, vec![60, 30]);
    }

    #[test]
    fn test_tier_classification() {
        // M0 -> [tx_a]      (single-candidate → counts as unique for tx_a)
        // M1 -> [tx_a,tx_b] (multi-candidate → compatible for both)
        // M2 -> [tx_b,tx_c] (multi-candidate → compatible for both;
        //                    protects tx_b from being subsumed by tx_a)
        // tx_a: unique=1, compat=2 → TierA, em_eligible
        // tx_b: unique=0, compat=2 → TierB, em_eligible (NOT subsumed)
        // tx_c: unique=0, compat=1 → TierB, em_eligible
        //
        // Plus a tx_d that never appears in any molecule → TierC.
        let graph = FlatEvidenceGraph {
            molecules: vec![
                mk_mol("r0", 0, 1, 6.0),
                mk_mol("r1", 1, 2, 6.5),
                mk_mol("r2", 3, 2, 7.0),
            ],
            candidate_pool: vec![0, 0, 1, 1, 2],
            candidate_as: vec![0; 5],
            candidate_nm: vec![0; 5],
            candidate_mapq: vec![0; 5],
            candidate_5p_gap: vec![0; 5],
            transcript_names: vec!["tx_a".into(), "tx_b".into(), "tx_c".into(), "tx_d".into()],
        };
        let mut tx2g = HashMap::new();
        tx2g.insert("tx_a".into(), "gene_a".into());
        tx2g.insert("tx_b".into(), "gene_a".into());
        tx2g.insert("tx_c".into(), "gene_a".into());
        tx2g.insert("tx_d".into(), "gene_a".into());
        let mut warn = HashSet::new();

        let records = resolve_transcript_evidence(
            &graph,
            &tx2g,
            &ResolverConfig {
                cohort_size: 1,
                tes_cluster_radius: 50,
            },
            &mut warn,
        );

        let rec_a = records.iter().find(|r| r.transcript_id == "tx_a").unwrap();
        assert_eq!(rec_a.tier, "TierA");
        assert_eq!(rec_a.unique_reads, 1);
        assert_eq!(rec_a.compatible_reads, 2);
        assert!(rec_a.em_eligible, "TierA must be em_eligible");

        let rec_b = records.iter().find(|r| r.transcript_id == "tx_b").unwrap();
        assert_eq!(rec_b.tier, "TierB");
        assert_eq!(rec_b.unique_reads, 0);
        assert_eq!(rec_b.compatible_reads, 2);
        assert!(!rec_b.is_subsumed, "tx_b must NOT be subsumed");
        assert!(rec_b.em_eligible, "unsubsumed TierB must be em_eligible");

        let rec_c = records.iter().find(|r| r.transcript_id == "tx_c").unwrap();
        assert_eq!(rec_c.tier, "TierB");
        assert_eq!(rec_c.unique_reads, 0);
        assert_eq!(rec_c.compatible_reads, 1);
        assert!(rec_c.em_eligible);

        let rec_d = records.iter().find(|r| r.transcript_id == "tx_d").unwrap();
        assert_eq!(rec_d.tier, "TierC");
        assert_eq!(rec_d.unique_reads, 0);
        assert_eq!(rec_d.compatible_reads, 0);
        assert!(!rec_d.em_eligible, "TierC must NOT be em_eligible");
        assert!(!rec_d.pal_candidate);
        assert!(warn.is_empty(), "no missing_gene_id warnings expected");
    }

    #[test]
    fn test_subsumption_o_k() {
        // M0 -> [tx_a] only. M1 -> [tx_a, tx_b]. tx_b's molecules = {M1},
        // which is ⊆ tx_a's molecules = {M0, M1} → tx_b fully subsumed.
        let graph = FlatEvidenceGraph {
            molecules: vec![mk_mol("r0", 0, 1, 6.0), mk_mol("r1", 1, 2, 6.5)],
            candidate_pool: vec![0, 0, 1],
            candidate_as: vec![0; 3],
            candidate_nm: vec![0; 3],
            candidate_mapq: vec![0; 3],
            candidate_5p_gap: vec![0; 3],
            transcript_names: vec!["tx_a".into(), "tx_b".into()],
        };
        let mut tx2g = HashMap::new();
        tx2g.insert("tx_a".into(), "gene_a".into());
        tx2g.insert("tx_b".into(), "gene_a".into());
        let mut warn = HashSet::new();

        let records = resolve_transcript_evidence(
            &graph,
            &tx2g,
            &ResolverConfig {
                cohort_size: 1,
                tes_cluster_radius: 50,
            },
            &mut warn,
        );

        let rec_a = records.iter().find(|r| r.transcript_id == "tx_a").unwrap();
        let rec_b = records.iter().find(|r| r.transcript_id == "tx_b").unwrap();

        assert_eq!(rec_a.tier, "TierA");
        assert_eq!(rec_b.tier, "TierB");
        assert!(rec_b.is_subsumed, "tx_b must be subsumed by tx_a");
        assert_eq!(rec_b.subsumption_target, "tx_a");
        assert!((rec_b.subsumption_score - 1.0).abs() < f32::EPSILON);
        assert!(!rec_b.em_eligible, "subsumed TierB must NOT be em_eligible");
    }

    #[test]
    fn test_missing_gene_id() {
        // Only tx_a has a gene mapping. tx_b has none.
        let graph = FlatEvidenceGraph {
            molecules: vec![mk_mol("r0", 0, 1, 6.0)],
            candidate_pool: vec![0],
            candidate_as: vec![0; 1],
            candidate_nm: vec![0; 1],
            candidate_mapq: vec![0; 1],
            candidate_5p_gap: vec![0; 1],
            transcript_names: vec!["tx_a".into(), "tx_b".into()],
        };
        let mut tx2g = HashMap::new();
        tx2g.insert("tx_a".into(), "gene_a".into());
        // tx_b intentionally NOT in tx_to_gene.
        let mut warn = HashSet::new();

        let records = resolve_transcript_evidence(
            &graph,
            &tx2g,
            &ResolverConfig {
                cohort_size: 1,
                tes_cluster_radius: 50,
            },
            &mut warn,
        );

        let rec_b = records.iter().find(|r| r.transcript_id == "tx_b").unwrap();
        assert_eq!(rec_b.gene_id, "missing_gene_id");
        assert!(
            warn.contains("tx_b"),
            "tx_b warning must have been emitted exactly once"
        );
        assert_eq!(warn.len(), 1, "no other warnings expected");

        // Second call should NOT emit a duplicate warning.
        let mut warn2 = warn.clone();
        let _ = resolve_transcript_evidence(
            &graph,
            &tx2g,
            &ResolverConfig {
                cohort_size: 1,
                tes_cluster_radius: 50,
            },
            &mut warn2,
        );
        assert_eq!(warn2.len(), 1, "warning set is deduplicated across calls");
    }

    /// Compile-time + runtime assertion that the Module 0 output field is
    /// named `pal_candidate` (NOT `pal_eligible`).
    /// See F5.3 critical naming note + P0 #1.
    #[test]
    fn test_pal_candidate_field_naming() {
        // Construct a record via the field set explicitly.
        let r = TranscriptEvidence {
            transcript_id: "tx_a".into(),
            gene_id: "gene_a".into(),
            compatible_reads: 15,
            unique_reads: 1,
            tier: "TierA".into(),
            is_subsumed: false,
            subsumption_target: String::new(),
            subsumption_score: 0.0,
            em_eligible: true,
            pal_candidate: true,
        };
        // The test name itself is documentation; the assertion is the
        // existence of `pal_candidate: bool` in the struct literal above.
        assert!(r.pal_candidate);
        assert!(r.em_eligible);
        // Verify the boolean value comes through struct-copy semantics.
        let copy = r.clone();
        assert!(copy.pal_candidate);
    }
}
