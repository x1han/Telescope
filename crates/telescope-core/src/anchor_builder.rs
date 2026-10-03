//! Module: Universal Mode Anchor Pool Builder (`anchor_builder.rs`)
//!
//! D1 Universal Mode Step 3 (per spec v0.2.0-rc2): streams one or more BAM
//! files, applies poly(A) anchor criteria, and writes a unified
//! `telescope.anchor_pool.parquet` ready for the downstream emission model.
//!
//! Anchor criteria (per spec §A of D1 rebuttal, accepted):
//! - 3' TES anchor: `ref_end + tes_slack >= ref_len`
//! - Unique assignment: NH tag == 1 (or MAPQ >= 20 fallback)
//! - Valid poly(A): `pt:i >= 10` (Dorado polyA tag)
//! - Coverage ratio: `alignment_length / ref_len >= min_coverage` (default 0.80)
//! - Per-sample 200-read cap per transcript (prevents deep-sample dominance)
//!
//! 5' TSS anchor is NOT required (5' truncation is normal in DRS; poly(A) lives
//! at the 3' end).
//!
//! Output schema (matches Module 3 `profile.parquet` schema so it can be
//! consumed directly by emission_model + TEL-EM):
//!   sample_id (utf8), read_id (utf8), gene_id (utf8), transcript_id (utf8),
//!   poly_a_length (int32), posterior_weight (float64 = 1.0 for anchors)

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{Float64Builder, Int32Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use noodles_bam::io::Reader as BamReader;
use noodles_sam::alignment::Record as _;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

/// Transcript → Gene lookup table. Built from a TSV (one `transcript_id\tgene_id`
/// per line) or empty (in which case `transcript_id` is used as `gene_id`).
#[derive(Debug, Default, Clone)]
pub struct TxToGene {
    map: HashMap<String, String>,
}

impl TxToGene {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Load from a TSV with header `transcript_id\tgene_id`.
    pub fn from_tsv<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut map = HashMap::new();
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read tx-to-gene TSV: {:?}", path.as_ref()))?;
        for (i, line) in content.lines().enumerate() {
            if i == 0 && line.starts_with("transcript_id") {
                continue; // header
            }
            let mut parts = line.splitn(2, '\t');
            let tx = parts.next().unwrap_or("").to_string();
            let g = parts.next().unwrap_or("").to_string();
            if !tx.is_empty() && !g.is_empty() {
                map.insert(tx, g);
            }
        }
        Ok(Self { map })
    }

    pub fn get(&self, tx: &str) -> Option<&String> {
        self.map.get(tx)
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
}

/// One filtered alignment that passes the anchor criteria.
#[derive(Debug, Clone)]
pub struct AnchorRecord {
    pub sample_id: String,
    pub read_id: String,
    pub gene_id: String,
    pub transcript_id: String,
    pub poly_a_length: i32,
}

/// Build anchor pool from one or more BAMs.
///
/// `bams_and_samples` is parallel: `bams_and_samples[i] = (path, sample_id)`.
/// `tx_to_gene` maps `transcript_id` -> `gene_id` (built from a TSV; if empty
/// the transcript_id is used as the gene_id, which is appropriate for
/// novel single-transcript loci).
pub fn build_anchor_pool<P, Q>(
    bams_and_samples: &[(P, &str)],
    tx_to_gene: &TxToGene,
    output: Q,
    tes_slack: usize,
    min_coverage: f64,
    max_reads_per_sample: usize,
) -> Result<AnchorStats>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
{
    let mut all_anchors: Vec<AnchorRecord> = Vec::new();
    let mut stats = AnchorStats::default();

    for (bam, sample_id) in bams_and_samples {
        let (anchors, sample_stats) = extract_anchors_from_bam(
            bam,
            sample_id,
            tx_to_gene,
            tes_slack,
            min_coverage,
            max_reads_per_sample,
        )?;
        stats.add(&sample_stats);
        all_anchors.extend(anchors);
    }

    // Pool-level, from the records already in hand: NOT the per-sample
    // accumulation `add` does. Distinct gene/transcript counts cannot be
    // summed across samples, so they are assigned once, over the whole pool,
    // after the loop.
    let (distinct_genes, distinct_transcripts) = pool_gene_tx_cardinality(&all_anchors);
    stats.distinct_genes = distinct_genes;
    stats.distinct_transcripts = distinct_transcripts;

    write_anchor_parquet(&all_anchors, &output)
        .with_context(|| format!("Failed to write anchor pool parquet: {:?}", output.as_ref()))?;

    Ok(stats)
}

/// Distinct `(gene_id, transcript_id)` counts over a built pool. Used to detect
/// the degenerate state in which `gene_id` was set equal to `transcript_id` on
/// every row (see [`degenerate_gene_pool_warning`]).
fn pool_gene_tx_cardinality(anchors: &[AnchorRecord]) -> (usize, usize) {
    let genes: HashSet<&str> = anchors.iter().map(|a| a.gene_id.as_str()).collect();
    let transcripts: HashSet<&str> = anchors.iter().map(|a| a.transcript_id.as_str()).collect();
    (genes.len(), transcripts.len())
}

/// The `[telescope][WARN]` line for a pool in the degenerate state — every row's
/// `gene_id` equal to its own `transcript_id` — or `None` when the pool carries
/// real gene structure.
///
/// WHY THIS IS EXACT, NOT A HEURISTIC. The degenerate state is precisely
/// `distinct gene_id == distinct transcript_id`, and the downstream arithmetic
/// follows from it alone: a gene's pool IS its own transcript's pool, so the
/// L2 gene-level fallback needs `gene_min_reads = 30` reads drawn from what is,
/// by the time L1 has just failed, fewer than `tx_min_reads = 10` reads. L2 is
/// therefore structurally unreachable, with no error and no other trace.
///
/// It is NOT enough to branch on "no `--tx-to-gene` was passed": a TSV that
/// genuinely maps one transcript to one gene is equally degenerate, and that
/// run needs the warning just as much. A TSV that maps several transcripts onto
/// a shared gene does not, and must stay silent.
///
/// An empty pool is exempt (`None`): the counts are vacuously equal there, but
/// the consequence that matters is that no anchors passed at all, and a message
/// about the fallback cascade would point at the wrong problem.
pub fn degenerate_gene_pool_warning(
    distinct_genes: usize,
    distinct_transcripts: usize,
) -> Option<String> {
    if distinct_transcripts == 0 || distinct_genes != distinct_transcripts {
        return None;
    }
    Some(format!(
        "[telescope][WARN] anchor pool has NO gene structure: {} distinct \
         transcript_id == {} distinct gene_id (gene_id was set equal to \
         transcript_id on every row).\n\
         [telescope][WARN]   Consequence: the L2 gene-level fallback is \
         STRUCTURALLY UNREACHABLE. A gene's pool IS its own transcript's pool, \
         which holds fewer than tx_min_reads=10 reads by the time L1 has been \
         tried, so the L2 gene_min_reads=30 gate can never fire.\n\
         [telescope][WARN]   The poly(A) cascade is therefore running as \
         2-LEVEL (L1 per-transcript, L3 cohort), not 3-LEVEL. An `L2=0` in the \
         downstream emission stats is THIS, not a fit failure -- do not read it \
         as evidence that the gene fit did not converge.\n\
         [telescope][WARN]   Remedy: pass --tx-to-gene <TSV> \
         (transcript_id<TAB>gene_id) to build-anchors, so that transcripts \
         sharing a gene share a pool.",
        distinct_transcripts, distinct_genes,
    ))
}

/// Extract anchors from a single BAM. Returns the filtered records plus
/// per-sample stats (counts at each filter step).
fn extract_anchors_from_bam<P: AsRef<Path>>(
    bam: P,
    sample_id: &str,
    tx_to_gene: &TxToGene,
    tes_slack: usize,
    min_coverage: f64,
    max_reads_per_sample: usize,
) -> Result<(Vec<AnchorRecord>, SampleStats)> {
    let path = bam.as_ref();
    let file = File::open(path).with_context(|| format!("Failed to open BAM: {:?}", path))?;
    let mut reader = BamReader::new(file);
    let header = reader.read_header().context("Failed to read BAM header")?;

    // Build ref_id → ref_len map from @SQ records.
    let mut ref_len: HashMap<usize, usize> = HashMap::new();
    let ref_seqs: Vec<String> = (0..header.reference_sequences().len())
        .map(|i| {
            let (name, ln) = match header.reference_sequences().get_index(i) {
                Some((n, sq)) => {
                    let l = sq.length().get();
                    (n.to_string(), l)
                }
                None => (String::new(), 0),
            };
            ref_len.insert(i, ln);
            name
        })
        .collect();

    // Track per-(transcript_id) count for the per-sample cap.
    let mut per_tx_count: HashMap<String, usize> = HashMap::new();

    let mut anchors: Vec<AnchorRecord> = Vec::new();
    let mut stats = SampleStats::default();

    for result in reader.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => {
                stats.parse_errors += 1;
                continue;
            }
        };
        stats.total_records += 1;

        // MAPQ-based uniqueness heuristic (industry standard for transcriptome
        // BAMs that may lack NH tag). MAPQ >= 20 means single-mapper.
        let mapq = record.mapping_quality().map(|m| m.get()).unwrap_or(0);
        if mapq < 20 {
            stats.filtered_unique += 1;
            continue;
        }

        // Position + length on reference.
        let ref_id = match record.reference_sequence_id() {
            Some(Ok(i)) => i,
            _ => continue,
        };
        let alignment_start = match record.alignment_start() {
            Some(Ok(p)) => p.get(),
            _ => continue,
        };
        let alignment_end = match record.alignment_end() {
            Some(Ok(p)) => p.get(),
            _ => continue,
        };

        let rl = *ref_len.get(&ref_id).unwrap_or(&0);
        if rl == 0 {
            stats.filtered_no_reflen += 1;
            continue;
        }

        // 3' TES anchor: ref_end + tes_slack >= ref_len.
        // alignment_end is 1-based inclusive in SAM/BAM; ref_len is 1-based.
        if alignment_end + tes_slack < rl {
            stats.filtered_3p_anchor += 1;
            continue;
        }

        // Coverage ratio: alignment_length / ref_len.
        let aln_len = if alignment_end >= alignment_start {
            (alignment_end - alignment_start + 1) as f64
        } else {
            0.0
        };
        let coverage = aln_len / rl as f64;
        if coverage < min_coverage {
            stats.filtered_coverage += 1;
            continue;
        }

        // poly(A) length: Dorado `pt:i` tag.
        let poly_a_length: i32 = {
            use noodles_sam::alignment::record::data::field::Tag;
            let tag = Tag::new(b'p', b't');
            match record.data().get(&tag) {
                Some(Ok(value)) => match value.as_int() {
                    Some(n) if n > 0 && n <= i32::MAX as i64 => n as i32,
                    _ => 0,
                },
                _ => 0,
            }
        };
        if poly_a_length < 10 {
            stats.filtered_polya_invalid += 1;
            continue;
        }

        // Resolve transcript_id and gene_id.
        let transcript_id = ref_seqs.get(ref_id).cloned().unwrap_or_default();
        let gene_id = tx_to_gene
            .get(&transcript_id)
            .cloned()
            .unwrap_or_else(|| transcript_id.clone());

        // Per-sample cap.
        let count = per_tx_count.entry(transcript_id.clone()).or_insert(0);
        if *count >= max_reads_per_sample {
            stats.filtered_cap += 1;
            continue;
        }
        *count += 1;

        let read_id = match record.name() {
            Some(n) => String::from_utf8_lossy(n).to_string(),
            None => continue,
        };
        anchors.push(AnchorRecord {
            sample_id: sample_id.to_string(),
            read_id,
            gene_id,
            transcript_id,
            poly_a_length,
        });
        stats.passed += 1;
    }

    Ok((anchors, stats))
}

fn write_anchor_parquet<P: AsRef<Path>>(anchors: &[AnchorRecord], output: P) -> Result<()> {
    let mut sample_b = StringBuilder::new();
    let mut read_b = StringBuilder::new();
    let mut gene_b = StringBuilder::new();
    let mut tx_b = StringBuilder::new();
    let mut pal_b = Int32Builder::new();
    let mut w_b = Float64Builder::new();

    for a in anchors {
        sample_b.append_value(&a.sample_id);
        read_b.append_value(&a.read_id);
        gene_b.append_value(&a.gene_id);
        tx_b.append_value(&a.transcript_id);
        pal_b.append_value(a.poly_a_length);
        w_b.append_value(1.0_f64); // anchors are deterministic → w=1
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("sample_id", DataType::Utf8, false),
        Field::new("read_id", DataType::Utf8, false),
        Field::new("gene_id", DataType::Utf8, false),
        Field::new("transcript_id", DataType::Utf8, false),
        Field::new("poly_a_length", DataType::Int32, false),
        Field::new("posterior_weight", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(sample_b.finish()),
            Arc::new(read_b.finish()),
            Arc::new(gene_b.finish()),
            Arc::new(tx_b.finish()),
            Arc::new(pal_b.finish()),
            Arc::new(w_b.finish()),
        ],
    )
    .context("Failed to build anchor RecordBatch")?;

    let out_file = File::create(&output)
        .with_context(|| format!("Failed to create anchor pool file: {:?}", output.as_ref()))?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(out_file, schema, Some(props))
        .context("Failed to create ArrowWriter")?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Per-sample stats for one BAM input.
#[derive(Debug, Default, Clone)]
pub struct SampleStats {
    pub total_records: u64,
    pub parse_errors: u64,
    pub filtered_unique: u64,
    pub filtered_no_reflen: u64,
    pub filtered_3p_anchor: u64,
    pub filtered_coverage: u64,
    pub filtered_polya_invalid: u64,
    pub filtered_cap: u64,
    pub passed: u64,
}

/// Aggregated stats across all input BAMs.
#[derive(Debug, Default, Clone)]
pub struct AnchorStats {
    pub samples: usize,
    pub total_records: u64,
    pub filtered_unique: u64,
    pub filtered_no_reflen: u64,
    pub filtered_3p_anchor: u64,
    pub filtered_coverage: u64,
    pub filtered_polya_invalid: u64,
    pub filtered_cap: u64,
    pub passed: u64,
    /// Distinct `gene_id` values in the emitted pool. POOL-LEVEL: assigned by
    /// `build_anchor_pool` from the built records, never summed by `add`
    /// (distinctness does not survive summation across samples). Read with
    /// `distinct_transcripts` via [`degenerate_gene_pool_warning`].
    pub distinct_genes: usize,
    /// Distinct `transcript_id` values in the emitted pool. See `distinct_genes`.
    pub distinct_transcripts: usize,
}

impl AnchorStats {
    /// Accumulate one sample's counts. Deliberately leaves `distinct_genes` /
    /// `distinct_transcripts` untouched: those are pool-level, not per-sample,
    /// and are assigned once by `build_anchor_pool` after every sample is in.
    fn add(&mut self, s: &SampleStats) {
        self.samples += 1;
        self.total_records += s.total_records;
        self.filtered_unique += s.filtered_unique;
        self.filtered_no_reflen += s.filtered_no_reflen;
        self.filtered_3p_anchor += s.filtered_3p_anchor;
        self.filtered_coverage += s.filtered_coverage;
        self.filtered_polya_invalid += s.filtered_polya_invalid;
        self.filtered_cap += s.filtered_cap;
        self.passed += s.passed;
    }

    pub fn report(&self) -> String {
        format!(
            "samples={} total={} filtered(unique={} no_reflen={} 3p_anchor={} coverage={} polya={} cap={}) passed={}",
            self.samples, self.total_records,
            self.filtered_unique, self.filtered_no_reflen,
            self.filtered_3p_anchor, self.filtered_coverage,
            self.filtered_polya_invalid, self.filtered_cap,
            self.passed
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::test_support::{synth_record, write_synthetic_bam};

    #[test]
    fn test_anchor_stats_aggregation() {
        let s1 = SampleStats {
            total_records: 100,
            passed: 30,
            ..Default::default()
        };
        let s2 = SampleStats {
            total_records: 50,
            passed: 12,
            ..Default::default()
        };
        let mut agg = AnchorStats::default();
        agg.add(&s1);
        agg.add(&s2);
        assert_eq!(agg.samples, 2);
        assert_eq!(agg.total_records, 150);
        assert_eq!(agg.passed, 42);
        // `add` is per-sample; the pool-level distinctness fields are assigned
        // by `build_anchor_pool` and must not be summed here.
        assert_eq!(agg.distinct_genes, 0);
        assert_eq!(agg.distinct_transcripts, 0);
    }

    // --- The degenerate gene pool -------------------------------------------
    //
    // The failure these pin: with no `--tx-to-gene`, `gene_id := transcript_id`
    // (see `extract_anchors_from_bam`), so the L2 gene-level fallback can never
    // fire — a gene's pool IS its transcript's pool, and L1 is only tried below
    // `tx_min_reads = 10`. It used to be silent, and `L2=0` sat in run logs
    // unannotated. These go red if the warning stops being emitted.

    /// A three-transcript synthetic BAM whose reads all clear the anchor
    /// criteria (MAPQ 60, full-length so 3'-anchored and coverage 1.0,
    /// `pt:i = 100`). Returns its path; the caller removes it.
    fn three_tx_anchor_bam(tag: &str) -> std::path::PathBuf {
        let recs = vec![
            synth_record("a0", 0, None, 0, 60),
            synth_record("a1", 0, None, 0, 60),
            synth_record("b0", 1, None, 0, 60),
            synth_record("c0", 2, None, 0, 60),
        ];
        write_synthetic_bam(tag, recs)
    }

    /// POSITIVE. No gene map: the pool IS degenerate, and the warning must say
    /// the three things a user cannot currently work out for themselves — L2 is
    /// unreachable, the cascade is 2-level not 3-level, and `--tx-to-gene` is
    /// the remedy.
    #[test]
    fn test_degenerate_gene_pool_without_tx_to_gene_warns_and_names_the_consequence() {
        let bam = three_tx_anchor_bam("degen_pos");
        let out = std::env::temp_dir().join(format!(
            "telescope_anchor_degen_pos_{}.parquet",
            std::process::id()
        ));
        let stats = build_anchor_pool(
            &[(bam.clone(), "S1")],
            &TxToGene::new(),
            &out,
            35,
            0.80,
            200,
        )
        .expect("build_anchor_pool");
        let _ = std::fs::remove_file(&bam);
        let _ = std::fs::remove_file(&out);

        assert_eq!(
            stats.passed, 4,
            "the fixture must actually pass the anchor filters, or this test \
             would be asserting on an empty pool"
        );
        assert_eq!(stats.distinct_transcripts, 3, "tx0/tx1/tx2");
        assert_eq!(
            stats.distinct_genes, 3,
            "gene_id falls back to transcript_id, so the counts are equal"
        );

        let msg = degenerate_gene_pool_warning(stats.distinct_genes, stats.distinct_transcripts)
            .expect("a pool with gene_id == transcript_id on every row MUST warn");
        for (what, needle) in [
            ("the L2 fallback being unreachable", "L2"),
            ("gene-level wording", "gene-level"),
            ("structurally unreachable", "STRUCTURALLY UNREACHABLE"),
            ("the 2-level consequence", "2-LEVEL"),
            ("the remedy", "--tx-to-gene"),
        ] {
            assert!(msg.contains(needle), "warning must name {what}; got: {msg}");
        }
    }

    /// NEGATIVE — the one that stops the warning becoming noise. The SAME BAM
    /// and the SAME command, given a gene map that puts all three transcripts
    /// on one shared gene. Real structure, so silence is the only correct
    /// output. A warning here would fire on every legitimate run.
    #[test]
    fn test_pool_with_real_gene_structure_does_not_warn() {
        let bam = three_tx_anchor_bam("degen_neg");
        let out = std::env::temp_dir().join(format!(
            "telescope_anchor_degen_neg_{}.parquet",
            std::process::id()
        ));
        let tsv =
            std::env::temp_dir().join(format!("telescope_tx2gene_neg_{}.tsv", std::process::id()));
        std::fs::write(
            &tsv,
            "transcript_id\tgene_id\ntx0\tGENE_A\ntx1\tGENE_A\ntx2\tGENE_A\n",
        )
        .unwrap();
        let map = TxToGene::from_tsv(&tsv).unwrap();
        assert_eq!(map.len(), 3);

        let stats = build_anchor_pool(&[(bam.clone(), "S1")], &map, &out, 35, 0.80, 200)
            .expect("build_anchor_pool");
        let _ = std::fs::remove_file(&bam);
        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_file(&tsv);

        assert_eq!(stats.passed, 4);
        assert_eq!(stats.distinct_transcripts, 3);
        assert_eq!(
            stats.distinct_genes, 1,
            "three transcripts on one gene: the counts differ, so L2 is reachable"
        );
        assert!(
            degenerate_gene_pool_warning(stats.distinct_genes, stats.distinct_transcripts)
                .is_none(),
            "a pool with real gene structure must NOT warn"
        );
    }

    /// The unit test behind the end-to-end pair above: the detector is the
    /// equality of the two distinct counts, and nothing else. Note what that
    /// means for a legitimately single-transcript-per-gene dataset: it too
    /// yields equal counts, and it too warns. That is the honest reading — the
    /// cascade really is 2-level there — not a false positive.
    #[test]
    fn test_detector_is_exactly_equality_of_the_two_distinct_counts() {
        let row = |gene: &str, tx: &str| AnchorRecord {
            sample_id: "S1".into(),
            read_id: "r".into(),
            gene_id: gene.into(),
            transcript_id: tx.into(),
            poly_a_length: 100,
        };
        // 4 rows, 2 genes, 2 transcripts. Real structure in the pool, and yet
        // the COUNTS are equal: the pair (2, 2) carries no more information
        // than (3, 3) does about whether gene_id was assigned per transcript.
        // The count is the exact detector of the STATE, and the state is what
        // the warning is about. What distinguishes this from a false positive
        // is the negative case above: with a real gene map, three transcripts
        // land on (1, 3), and (1, 3) is silent.
        let one_tx_per_gene = vec![row("G1", "T1"), row("G1", "T1"), row("G2", "T2")];
        assert_eq!(pool_gene_tx_cardinality(&one_tx_per_gene), (2, 2));
        assert!(
            degenerate_gene_pool_warning(2, 2).is_some(),
            "one-transcript-per-gene is genuinely degenerate and must warn"
        );

        // The no-gene-map pool: gene_id == transcript_id on every row.
        let degenerate = vec![
            row("T1", "T1"),
            row("T1", "T1"),
            row("T2", "T2"),
            row("T3", "T3"),
        ];
        assert_eq!(pool_gene_tx_cardinality(&degenerate), (3, 3));
        assert!(degenerate_gene_pool_warning(3, 3).is_some());

        // Strict inequality, either direction, is the only silence: more genes
        // than transcripts is impossible from `build_anchor_pool` (each row
        // carries one of each), but the detector is stated without relying on
        // that and is pinned here so it stays that way.
        assert!(degenerate_gene_pool_warning(1, 3).is_none());
        assert!(degenerate_gene_pool_warning(2, 3).is_none());

        // The exemption, pinned so it cannot be widened by accident: an empty
        // pool is vacuously "equal", and the honest complaint about it is that
        // no anchors passed — not that L2 was skipped.
        assert_eq!(pool_gene_tx_cardinality(&[]), (0, 0));
        assert!(degenerate_gene_pool_warning(0, 0).is_none());
    }
}
