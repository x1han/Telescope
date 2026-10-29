//! Criteria: 3' TES anchor `ref_end + tes_slack >= ref_len`; poly(A) `pt:i >= 10`; coverage
//! `alignment_length / ref_len >= min_coverage` (default 0.80); per-sample 200-read cap per
//! transcript. Ambiguity is resolved PER READ (strict argmax); no NH/MAPQ filter. Output schema:
//! sample_id, read_id, gene_id, transcript_id, poly_a_length, posterior_weight.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use arrow::array::{Float64Builder, Int32Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use noodles_bam::io::Reader as BamReader;
use noodles_sam::alignment::record::data::field::Tag;
use noodles_sam::alignment::Record as _;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

/// Dorado poly(A) tail length SAM tag: `pt:i`.
const TAG_PT: Tag = Tag::new(b'p', b't');

/// minimap2 alignment score SAM tag: `AS:i`. Drives the per-read argmax.
const TAG_AS: Tag = Tag::new(b'A', b'S');

/// Transcript → Gene lookup table: a TSV of `transcript_id\tgene_id` lines, or empty (then `transcript_id` is its own `gene_id`, and the gene relation is `Unknown`).
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

    pub fn from_tsv<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut map = HashMap::new();
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read tx-to-gene TSV: {:?}", path.as_ref()))?;
        for (i, line) in content.lines().enumerate() {
            if i == 0 && line.starts_with("transcript_id") {
                continue;
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

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AnchorParams {
    /// 3' TES slack in bp: 3'-anchored when `alignment_end + tes_slack >= ref_len`.
    pub tes_slack: usize,
    /// Minimum coverage ratio `alignment_length / ref_len`.
    pub min_coverage: f64,
    /// Per-sample cap on anchors per transcript.
    pub max_reads_per_sample: usize,
    /// Absolute floor on `delta_AS_raw` (shipped default 12.0; a single mismatch costs 6 AS). `delta_AS_raw = AS_1 - (AS_2 or 0)`, MEASURED ON C_raw — the deduped candidate set BEFORE the physical filters — so a lone raw candidate is measured against 0.
    pub gap_abs: f64,
    /// Floor on `delta_AS_raw` as a FRACTION of `AS_best` (shipped default 0.0); the two floors are combined with `max`, so a relative floor of 0.0 is inert.
    pub gap_rel: f64,
}

#[derive(Debug, Clone)]
pub struct AnchorRecord {
    pub sample_id: String,
    pub read_id: String,
    pub gene_id: String,
    pub transcript_id: String,
    pub poly_a_length: i32,
}

/// `bams_and_samples` is parallel: `bams_and_samples[i] = (path, sample_id)`. `tx_to_gene` maps `transcript_id` -> `gene_id`, falling back to the transcript itself when empty.
pub fn build_anchor_pool<P, Q>(
    bams_and_samples: &[(P, &str)],
    tx_to_gene: &TxToGene,
    output: Q,
    params: AnchorParams,
) -> Result<AnchorStats>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
{
    let mut all_anchors: Vec<AnchorRecord> = Vec::new();
    let mut stats = AnchorStats::default();

    for (bam, sample_id) in bams_and_samples {
        let (anchors, sample_stats) = extract_anchors_from_bam(bam, sample_id, tx_to_gene, params)?;
        stats.add(&sample_stats);
        all_anchors.extend(anchors);
    }

    let (distinct_genes, distinct_transcripts) = pool_gene_tx_cardinality(&all_anchors);
    stats.distinct_genes = distinct_genes;
    stats.distinct_transcripts = distinct_transcripts;

    write_anchor_parquet(&all_anchors, &output)
        .with_context(|| format!("Failed to write anchor pool parquet: {:?}", output.as_ref()))?;

    Ok(stats)
}

/// Distinct `(gene_id, transcript_id)` counts; the exact detector of the degenerate state.
fn pool_gene_tx_cardinality(anchors: &[AnchorRecord]) -> (usize, usize) {
    let genes: HashSet<&str> = anchors.iter().map(|a| a.gene_id.as_str()).collect();
    let transcripts: HashSet<&str> = anchors.iter().map(|a| a.transcript_id.as_str()).collect();
    (genes.len(), transcripts.len())
}

/// The `[telescope][WARN]` line for a pool in the degenerate state — every gene holds exactly one transcript's anchors — or `None` when the pool carries real gene structure. The degenerate state is exactly `distinct gene_id == distinct transcript_id`, so no two transcripts ever share fitted poly(A) parameters through the L2 tier.
pub fn degenerate_gene_pool_warning(
    distinct_genes: usize,
    distinct_transcripts: usize,
) -> Option<String> {
    if distinct_transcripts == 0 || distinct_genes != distinct_transcripts {
        return None;
    }
    Some(format!(
        "[telescope][WARN] anchor pool has NO gene structure: {} distinct \
         transcript_id == {} distinct gene_id, i.e. every gene here has \
         exactly one transcript.\n\
         [telescope][WARN]   Consequence: every gene's pool IS its own \
         transcript's pool, so no two transcripts ever share fitted poly(A) \
         parameters through the L2 gene-level tier. At the DEFAULT thresholds \
         (tx_min_reads=10, gene_min_reads=30) this makes L2 unreachable: \
         reaching L2 already requires n < 10, and the gate then demands 30 \
         values from a pool of at most 9.\n\
         [telescope][WARN]   Those thresholds are build-emission flags and are \
         NOT visible here. Lowering --gene-min-reads below --tx-min-reads \
         (e.g. --gene-min-reads 3) DOES make L2 fire on this pool, but it is \
         then pooling a transcript with itself, which is the same fit as L1 at \
         a different threshold -- not gene-level borrowing. An `L2=0` in the \
         downstream stats at the default thresholds is THIS, not a fit \
         failure; do not read it as evidence that the gene fit did not \
         converge.\n\
         [telescope][WARN]   Remedy: pass --tx-to-gene <TSV> \
         (transcript_id<TAB>gene_id) to build-anchors, so that transcripts \
         sharing a gene genuinely share a pool.",
        distinct_transcripts, distinct_genes,
    ))
}

/// FIXED SIZE, 20 bytes: the staging map holds one of these per surviving record.
#[derive(Debug, Clone, Copy)]
struct Staged {
    ref_id: u32,
    start: u32,
    end: u32,
    as_: i32,
    pt: i32,
}

/// One read's staging state; `cands` arrives in BAM order and is sorted in place.
struct ReadGroup {
    /// The 128-bit read-name hash this group was filed under: the per-transcript cap's sort key.
    key: u128,
    /// Byte range of this read's QNAME inside the shared name arena.
    name_off: u32,
    name_len: u32,
    cands: Vec<Staged>,
}

/// 128-bit grouping key for a read name. The guarantee is the per-record comparison in the staging loop, which fails the run if two distinct QNAMEs share a key.
fn read_key(name: &[u8]) -> u128 {
    let mut lo = DefaultHasher::new();
    name.hash(&mut lo);
    let mut hi = DefaultHasher::new();
    0xA5_u8.hash(&mut hi);
    name.hash(&mut hi);
    (u128::from(hi.finish()) << 64) | u128::from(lo.finish())
}

enum GeneRelation {
    Same,
    Unknown,
    Cross,
}

/// Resolve a transcript's gene, or fail loudly when a non-empty map does not cover it. The EMPTY map falls back to `gene_id := transcript_id`, legitimate for novel single-transcript loci; a NON-EMPTY map missing a transcript means the map and the BAM's `@SQ` header disagree.
fn resolve_gene(tx: &str, tx_to_gene: &TxToGene) -> Result<String> {
    match tx_to_gene.get(tx) {
        Some(g) => Ok(g.clone()),
        None if tx_to_gene.is_empty() => Ok(tx.to_string()),
        None => bail!(
            "--tx-to-gene has {} entries but transcript {:?} is not among them. \
             Refusing to fall back to gene_id := transcript_id: with a real map \
             present a missing transcript means the map and the BAM's @SQ header \
             disagree, and the per-read same-gene veto would then be decided on \
             an invented gene id. Fix the TSV or drop the flag.",
            tx_to_gene.len(),
            tx
        ),
    }
}

/// # THE PER-READ DECISION IS TAKEN ON C_raw
/// **C_raw** is the read's full deduped candidate set, before the physical filters. Branches, in order: no eligible candidate → `reads_no_eligible_candidate`; top two RAW candidates tie EXACTLY on AS → `rejected_tie`; raw argmax NOT eligible → `rejected_usurpation`; `delta_AS_raw < max(gap_abs, gap_rel * AS_1)` → rejected_*; otherwise cap check → emit.
fn extract_anchors_from_bam<P: AsRef<Path>>(
    bam: P,
    sample_id: &str,
    tx_to_gene: &TxToGene,
    p: AnchorParams,
) -> Result<(Vec<AnchorRecord>, SampleStats)> {
    let path = bam.as_ref();
    let file = File::open(path).with_context(|| format!("Failed to open BAM: {:?}", path))?;
    let mut reader = BamReader::new(file);
    let header = reader.read_header().context("Failed to read BAM header")?;

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

    let mut per_tx_count: HashMap<String, usize> = HashMap::new();

    let mut anchors: Vec<AnchorRecord> = Vec::new();
    let mut stats = SampleStats::default();

    let mut name_arena: Vec<u8> = Vec::new();
    let mut groups: Vec<ReadGroup> = Vec::new();
    let mut group_index: HashMap<u128, u32> = HashMap::new();
    let mut read_key_collisions: u64 = 0;

    let mut record = noodles_bam::Record::default();
    loop {
        let read = match reader.read_record(&mut record) {
            Ok(n) => n,
            Err(_) => {
                // `read_record` cannot be re-entered after a failure without risking a spin on the same truncated stream, so the scan stops here.
                stats.parse_errors += 1;
                break;
            }
        };
        if read == 0 {
            break;
        }
        stats.total_records += 1;

        // 0x4. Checked BEFORE MAPQ: every unmapped record carries MAPQ 0.
        if record.flags().is_unmapped() {
            stats.filtered_unmapped += 1;
            continue;
        }
        // 0x800. A supplementary record is one segment of a split molecule, so its poly(A) and 3' end are not the molecule's.
        if record.flags().is_supplementary() {
            stats.filtered_supplementary += 1;
            continue;
        }
        // 0x100 (secondary) is deliberately NOT dropped.

        // ⚠ THERE IS NO MAPQ GATE HERE: MAPQ was never a uniqueness criterion here, `--secondary` is PRESERVED, and `filtered_unique` stays a RETIRED compat slot at 0.
        let ref_id = match record.reference_sequence_id() {
            Some(Ok(i)) => i,
            _ => {
                stats.filtered_no_refid += 1;
                continue;
            }
        };
        let alignment_start = match record.alignment_start() {
            Some(Ok(p)) => p.get(),
            _ => {
                stats.filtered_no_position += 1;
                continue;
            }
        };
        let alignment_end = match record.alignment_end() {
            Some(Ok(p)) => p.get(),
            _ => {
                stats.filtered_no_position += 1;
                continue;
            }
        };

        // A record with no QNAME cannot be assigned to a read, so it cannot consume a cap slot.
        let name: &[u8] = match record.name() {
            Some(n) => n,
            None => {
                stats.filtered_no_read_id += 1;
                continue;
            }
        };

        let rl = *ref_len.get(&ref_id).unwrap_or(&0);
        if rl == 0 {
            stats.filtered_no_reflen += 1;
            continue;
        }

        let poly_a_length: i32 = match record.data().get(&TAG_PT) {
            Some(Ok(value)) => match value.as_int() {
                Some(n) if n > 0 && n <= i32::MAX as i64 => n as i32,
                _ => 0,
            },
            _ => 0,
        };
        // A tag-less AS reads as 0, matching `evidence.rs`'s convention: two tag-less candidates on one read tie, and a tie yields no anchor.
        let as_score: i32 = match record.data().get(&TAG_AS) {
            Some(Ok(value)) => match value.as_int() {
                Some(n) => i32::try_from(n).unwrap_or(0),
                None => 0,
            },
            _ => 0,
        };

        let key = read_key(name);
        let pos = match group_index.get(&key) {
            Some(&pos) => {
                let g = &groups[pos as usize];
                if &name_arena[g.name_off as usize..(g.name_off + g.name_len) as usize] != name {
                    read_key_collisions += 1;
                }
                pos
            }
            None => {
                let off = name_arena.len() as u32;
                name_arena.extend_from_slice(name);
                groups.push(ReadGroup {
                    key,
                    name_off: off,
                    name_len: name.len() as u32,
                    cands: Vec::new(),
                });
                let pos = (groups.len() - 1) as u32;
                group_index.insert(key, pos);
                pos
            }
        };
        groups[pos as usize].cands.push(Staged {
            ref_id: ref_id as u32,
            start: alignment_start as u32,
            end: alignment_end as u32,
            as_: as_score,
            pt: poly_a_length,
        });
    }

    // A literal duplicate check over the emitted anchors could never fire: the grouping would hide the collision first.
    assert_eq!(
        read_key_collisions, 0,
        "two distinct QNAMEs hashed to the same 128-bit grouping key, so their \
         records were merged into one read group. The per-read argmax then \
         picked one transcript for two molecules. This is the residual risk of \
         keying by hash instead of by name; see `read_key`."
    );

    // IN READ-ID-HASH ORDER: iterating the `HashMap` would make the cap depend on a per-process hash seed, and insertion order would make it depend on BAM layout. `sort_by_key`, NOT `sort_unstable_by_key`: the keys are asserted unique.
    let mut order: Vec<u32> = (0..groups.len() as u32).collect();
    order.sort_by_key(|&i| groups[i as usize].key);
    // `scratch` is C_raw: EVERY deduped run of the current read, whether or not it survived the physical filters. `.0` is the `Staged` representative.
    let mut scratch: Vec<(Staged, u64, bool)> = Vec::new();
    for &i in &order {
        let g = &mut groups[i as usize];
        stats.reads_seen += 1;

        // DEDUP BY ref_id, KEEPING MAX AS; `sort_by` is stable, so equal-AS records on one ref_id keep their BAM order — arbitrary but deterministic.
        g.cands
            .sort_by(|a, b| a.ref_id.cmp(&b.ref_id).then_with(|| b.as_.cmp(&a.as_)));

        // WALK THE RUNS, applying the physical filters once per transcript and attributing the result to EVERY record collapsed into it (`mult`), so the ledger stays closed in RECORDS. ⚠ EVERY RUN IS PUSHED: the filters decide ELIGIBILITY, not which is best.
        scratch.clear();
        let mut eligible_records: u64 = 0;
        let mut i = 0;
        while i < g.cands.len() {
            let mut j = i + 1;
            while j < g.cands.len() && g.cands[j].ref_id == g.cands[i].ref_id {
                j += 1;
            }
            let mult = (j - i) as u64;
            let rep = g.cands[i];

            let rl = *ref_len.get(&(rep.ref_id as usize)).unwrap_or(&0);
            let eligible = if rl == 0 {
                // Unreachable while pass 1 filters it: a zero makes the coverage test `NaN < min_coverage`, which is FALSE.
                stats.filtered_no_reflen += mult;
                false
            } else if (rep.end as usize + p.tes_slack) < rl {
                // 3' TES anchor: ref_end + tes_slack >= ref_len.
                stats.filtered_3p_anchor += mult;
                false
            } else {
                let aln_len = if rep.end >= rep.start {
                    (rep.end - rep.start + 1) as f64
                } else {
                    0.0
                };
                if aln_len / (rl as f64) < p.min_coverage {
                    stats.filtered_coverage += mult;
                    false
                } else if rep.pt < 10 {
                    stats.filtered_polya_invalid += mult;
                    false
                } else {
                    true
                }
            };
            if eligible {
                eligible_records += mult;
            }
            scratch.push((rep, mult, eligible));
            i = j;
        }

        if eligible_records == 0 {
            stats.reads_no_eligible_candidate += 1;
            continue;
        }
        scratch.sort_by(|a, b| b.0.as_.cmp(&a.0.as_));

        // (b) EXACT RAW TIE. An exact tie on the top two RAW AS yields NO anchor, and is checked BEFORE the winner's eligibility because a tie's verdict does not depend on which candidate sorts first. THE TIE RATE IS POPULATION-DEPENDENT.
        if scratch.len() >= 2 && scratch[0].0.as_ == scratch[1].0.as_ {
            stats.rejected_tie += 1;
            stats.filtered_eligible_not_anchor += eligible_records;
            continue;
        }

        let best = scratch[0].0;
        let runner_up: Option<Staged> = scratch.get(1).map(|e| e.0);

        // (c) RUNNER-UP USURPATION. The aligner's own best alignment for this read is not eligible and only a LOWER-AS homolog survived the physical filters.
        if !scratch[0].2 {
            stats.rejected_usurpation += 1;
            stats.filtered_eligible_not_anchor += eligible_records;
            continue;
        }

        // (d) THE GAP, MEASURED ON C_raw. `delta_AS_raw = AS_1 - (AS_2 or 0)`: an absent runner-up counts as 0, so the floor applies to a lone raw candidate too.
        let gap = (best.as_ as f64) - runner_up.map_or(0.0, |s| s.as_ as f64);
        if gap < p.gap_abs || gap < p.gap_rel * (best.as_ as f64) {
            // THE GENE RELATION IS A PROPERTY OF A PAIR: a read with no second RAW candidate has no pair, so its relation is `Unknown`.
            if let Some(second) = runner_up {
                let g_best = resolve_gene(tx_name(&ref_seqs, best.ref_id), tx_to_gene)?;
                let g_second = resolve_gene(tx_name(&ref_seqs, second.ref_id), tx_to_gene)?;
                match relation(&g_best, &g_second, tx_to_gene) {
                    GeneRelation::Same => stats.rejected_same_gene += 1,
                    // A cross-gene pair rejected on the gap is still a cross-gene pair; without this the counter would have no producer at all.
                    GeneRelation::Cross => stats.rejected_cross_gene += 1,
                    GeneRelation::Unknown => stats.rejected_gap += 1,
                }
            } else {
                stats.rejected_gap += 1;
            }
            stats.filtered_eligible_not_anchor += eligible_records;
            continue;
        }

        // THE GATE IS THE ONLY ARITHMETIC GATE: there is deliberately no unconditional same-gene veto. A read scoring AS_1 on T_1 and AS_2 << AS_1 on T_2, SIBLING ISOFORMS OF ONE GENE, has aligned over sequence T_2 does not contain. A single mismatch costs 6 AS, so a gap of even a few hundred is tens of edits.
        if let Some(second) = runner_up {
            let rel = relation(
                &resolve_gene(tx_name(&ref_seqs, best.ref_id), tx_to_gene)?,
                &resolve_gene(tx_name(&ref_seqs, second.ref_id), tx_to_gene)?,
                tx_to_gene,
            );
            if matches!(rel, GeneRelation::Unknown) {
                // NOT a read-level rejection: this read still clears the gap and will be emitted. It is counted so the ledger states the relation was unavailable.
                stats.gene_veto_disabled += 1;
                stats.same_gene_unknown += 1;
            }
        }

        // EMIT. Exactly one anchor per read: the argmax candidate.
        let transcript_id = tx_name(&ref_seqs, best.ref_id).to_string();
        let gene_id = resolve_gene(&transcript_id, tx_to_gene)?;

        let count = per_tx_count.entry(transcript_id.clone()).or_insert(0);
        if *count >= p.max_reads_per_sample {
            // THE CAP REJECTS A READ, AND A READ IS `eligible_records` RECORDS. `filtered_cap` is READ-denominated: it is the counterpart of `reads_over_cap` and the two must be equal. The remaining `eligible_records - 1` RECORDS go to `filtered_eligible_not_anchor`.
            stats.filtered_cap += 1;
            stats.filtered_eligible_not_anchor += eligible_records - 1;
            stats.reads_over_cap += 1;
            continue;
        }
        *count += 1;

        let read_id = String::from_utf8_lossy(
            &name_arena[g.name_off as usize..(g.name_off + g.name_len) as usize],
        )
        .into_owned();
        anchors.push(AnchorRecord {
            sample_id: sample_id.to_string(),
            read_id,
            gene_id,
            transcript_id,
            poly_a_length: best.pt,
        });
        stats.passed += 1;
        stats.anchors_emitted += 1;
        // Every OTHER eligible record of this read — the argmax losers and the duplicates the dedup collapsed into the winner — so the remainder is exactly `eligible_records - 1`.
        stats.filtered_eligible_not_anchor += eligible_records - 1;
    }

    // UNCONDITIONAL, not a `debug_assert!`: the release profile compiles debug assertions out, and the ledger must be closed by the PRODUCTION binary.
    assert_eq!(
        stats.filtered_cap, stats.reads_over_cap,
        "filtered_cap counts rejected READS and must stay equal to reads_over_cap; \
         {} vs {} means one of them was incremented without the other",
        stats.filtered_cap, stats.reads_over_cap
    );

    // THE READ IDENTITY ITSELF, in the release binary: `unaccounted()` proves the RECORD half closes, nothing proved the READ half did. `gene_veto_disabled` is deliberately absent: it counts reads that CLEARED the gap and were emitted.
    assert_eq!(
        stats.reads_seen,
        stats.anchors_emitted
            + stats.rejected_tie
            + stats.rejected_usurpation
            + stats.rejected_gap
            + stats.rejected_same_gene
            + stats.rejected_cross_gene
            + stats.reads_no_eligible_candidate
            + stats.reads_over_cap,
        "every read that reached the per-read stage must land in exactly one \
         outcome. reads_seen={} anchors={} tie={} usurpation={} gap={} \
         same_gene={} cross_gene={} no_eligible={} over_cap={} \
         (gene_veto_disabled={} is NOT a destination: those reads cleared the \
         gap and were emitted)",
        stats.reads_seen,
        stats.anchors_emitted,
        stats.rejected_tie,
        stats.rejected_usurpation,
        stats.rejected_gap,
        stats.rejected_same_gene,
        stats.rejected_cross_gene,
        stats.reads_no_eligible_candidate,
        stats.reads_over_cap,
        stats.gene_veto_disabled,
    );

    Ok((anchors, stats))
}

/// `ref_seqs[ref_id]`, or `""` for an id the header does not name.
fn tx_name(ref_seqs: &[String], ref_id: u32) -> &str {
    ref_seqs.get(ref_id as usize).map_or("", String::as_str)
}

/// `tx_to_gene.is_empty()` is consulted rather than inferring from the ids: with an empty map "the same id" means "the same transcript", not "the same gene".
fn relation(a: &str, b: &str, tx_to_gene: &TxToGene) -> GeneRelation {
    if tx_to_gene.is_empty() {
        GeneRelation::Unknown
    } else if a == b {
        GeneRelation::Same
    } else {
        GeneRelation::Cross
    }
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

#[derive(Debug, Default, Clone)]
pub struct SampleStats {
    pub total_records: u64,
    pub parse_errors: u64,
    /// MAPQ < 20 — NOT "non-unique" but "low MAPQ". RETIRED: no MAPQ gate any more, so this slot stays 0, but it is still in the frozen ledger schema.
    pub filtered_unique: u64,
    /// SAM `0x4`, unmapped: no alignment, so no transcript, no 3' end, no poly(A).
    pub filtered_unmapped: u64,
    /// SAM `0x800`, supplementary alignment: one segment of a split molecule.
    pub filtered_supplementary: u64,
    /// Dropped because `reference_sequence_id()` yielded `None`/Err.
    pub filtered_no_refid: u64,
    /// Same, for `alignment_start()` / `alignment_end()`: two call sites, one bucket.
    pub filtered_no_position: u64,
    /// Same, for `record.name()` being `None`: it cannot be written to the pool.
    pub filtered_no_read_id: u64,
    pub filtered_no_reflen: u64,
    pub filtered_3p_anchor: u64,
    pub filtered_coverage: u64,
    pub filtered_polya_invalid: u64,
    /// Physically eligible records that were NOT emitted as an anchor: the argmax losers, the duplicates a same-`ref_id` dedup collapsed, and every record of a rejected read.
    pub filtered_eligible_not_anchor: u64,
    /// Rejected by the per-transcript cap. **READ-DENOMINATED**: the cap rejects a read, not a record, and the `eligible_records - 1` other RECORDS go to `filtered_eligible_not_anchor`. Always equal to [`SampleStats::reads_over_cap`].
    pub filtered_cap: u64,
    /// Anchors emitted. The ledger's terminal bucket: since selection is per read, `passed` counts the RECORDS that became anchors and the reads that reached the pool.
    pub passed: u64,

    // READ-LEVEL COUNTERS. A SEPARATE DENOMINATION: everything above counts RECORDS, everything
    // below counts READS, and the two are never added together. None of these fields appears in
    // `AnchorStats::unaccounted()`; `gene_veto_disabled` is OUTSIDE because it counts reads that
    // CLEARED the gap and were EMITTED.
    /// Reads that reached the per-read decision stage. NOT "reads in the file": a read whose every record died at `0x4`, tid or POS never forms a group.
    pub reads_seen: u64,
    /// Reads that produced an anchor. Equal to `passed`.
    pub anchors_emitted: u64,
    /// Reads whose top two RAW candidates had EXACTLY equal AS, so the strict argmax had no winner. The population is C_raw, not the eligible subset.
    pub rejected_tie: u64,
    /// **RUNNER-UP USURPATION.** Reads whose raw argmax is NOT physically eligible while a strictly LOWER-AS candidate is.
    pub rejected_usurpation: u64,
    /// Reads rejected because `delta_AS_raw` was below the gap floors, AND whose gene relation could not be classified — no second raw candidate, or `tx_to_gene` empty.
    pub rejected_gap: u64,
    /// Reads whose top two RAW candidates share a gene and whose `delta_AS_raw` fell below the floors. A same-gene pair is a CLASSIFICATION of a gap failure.
    pub rejected_same_gene: u64,
    /// Reads whose top two RAW candidates are in DIFFERENT genes and whose `delta_AS_raw` fell below the floors.
    pub rejected_cross_gene: u64,
    /// Reads whose same-gene status could not be evaluated because `tx_to_gene` was empty: with an empty map `gene_id == transcript_id`, so the relation compares each transcript with itself and can never fire.
    pub gene_veto_disabled: u64,
    /// Reads ROUTED to the "unknown" class rather than to `cross_gene`. Equal to `gene_veto_disabled` in this implementation.
    pub same_gene_unknown: u64,
    /// Reads that reached the per-read stage but had no physically eligible candidate at all.
    pub reads_no_eligible_candidate: u64,
    /// Reads whose argmax transcript was already at `max_reads_per_sample`. WHICH READS SURVIVE IS A DETERMINISTIC READ-ID-HASH RULE, not BAM order; the COUNT is order-invariant.
    pub reads_over_cap: u64,
}

#[derive(Debug, Default, Clone)]
pub struct AnchorStats {
    pub samples: usize,
    pub total_records: u64,
    /// Records the BAM decoder could not read at all, summed over samples. A DISJOINT POPULATION, deliberately excluded from `unaccounted()`: it is incremented BEFORE `total_records`, so folding it in would make the deficit read -1 per parse failure.
    pub parse_errors: u64,
    pub filtered_unique: u64,
    pub filtered_unmapped: u64,
    pub filtered_supplementary: u64,
    pub filtered_no_refid: u64,
    pub filtered_no_position: u64,
    pub filtered_no_read_id: u64,
    pub filtered_no_reflen: u64,
    pub filtered_3p_anchor: u64,
    pub filtered_coverage: u64,
    pub filtered_polya_invalid: u64,
    pub filtered_eligible_not_anchor: u64,
    pub filtered_cap: u64,
    pub passed: u64,

    // --- READ-LEVEL. Summed by `add`, but NEVER folded into `unaccounted()`.
    pub reads_seen: u64,
    pub anchors_emitted: u64,
    pub rejected_tie: u64,
    pub rejected_usurpation: u64,
    pub rejected_gap: u64,
    pub rejected_same_gene: u64,
    pub rejected_cross_gene: u64,
    pub gene_veto_disabled: u64,
    pub same_gene_unknown: u64,
    pub reads_no_eligible_candidate: u64,
    pub reads_over_cap: u64,

    /// Distinct `gene_id` values in the emitted pool. POOL-LEVEL: assigned by `build_anchor_pool` from the built records, never summed by `add`.
    pub distinct_genes: usize,
    /// Distinct `transcript_id` values in the emitted pool. See `distinct_genes`.
    pub distinct_transcripts: usize,
}

impl AnchorStats {
    /// Accumulate one sample's counts. Deliberately leaves `distinct_genes` / `distinct_transcripts` untouched: those are pool-level, not per-sample.
    fn add(&mut self, s: &SampleStats) {
        self.samples += 1;
        self.total_records += s.total_records;
        self.parse_errors += s.parse_errors;
        self.filtered_unique += s.filtered_unique;
        self.filtered_unmapped += s.filtered_unmapped;
        self.filtered_supplementary += s.filtered_supplementary;
        self.filtered_no_refid += s.filtered_no_refid;
        self.filtered_no_position += s.filtered_no_position;
        self.filtered_no_read_id += s.filtered_no_read_id;
        self.filtered_no_reflen += s.filtered_no_reflen;
        self.filtered_3p_anchor += s.filtered_3p_anchor;
        self.filtered_coverage += s.filtered_coverage;
        self.filtered_polya_invalid += s.filtered_polya_invalid;
        self.filtered_eligible_not_anchor += s.filtered_eligible_not_anchor;
        self.filtered_cap += s.filtered_cap;
        self.passed += s.passed;
        self.reads_seen += s.reads_seen;
        self.anchors_emitted += s.anchors_emitted;
        self.rejected_tie += s.rejected_tie;
        self.rejected_usurpation += s.rejected_usurpation;
        self.rejected_gap += s.rejected_gap;
        self.rejected_same_gene += s.rejected_same_gene;
        self.rejected_cross_gene += s.rejected_cross_gene;
        self.gene_veto_disabled += s.gene_veto_disabled;
        self.same_gene_unknown += s.same_gene_unknown;
        self.reads_no_eligible_candidate += s.reads_no_eligible_candidate;
        self.reads_over_cap += s.reads_over_cap;
    }

    /// FIELD ORDER IS A COMPATIBILITY CONSTRAINT, DO NOT REORDER. A downstream scraper pins BOTH `filtered(unique=N no_reflen=M ... cap=K)` AND `passed=P` immediately after that closing paren, so any new field must be APPENDED after `passed=`. The `reads(...)` block after them is a DIFFERENT UNIT and says so in its own name.
    pub fn report(&self) -> String {
        format!(
            "samples={} total={} filtered(unique={} no_reflen={} 3p_anchor={} coverage={} \
             polya={} cap={}) passed={} unaccounted={} no_refid={} no_position={} no_read_id={} \
             unmapped={} supplementary={} not_anchor={} \
             reads(seen={} anchors={} tie={} gap={} same_gene={} cross_gene={} \
             veto_disabled={} same_gene_unknown={} no_eligible={} over_cap={} usurpation={})",
            self.samples,
            self.total_records,
            self.filtered_unique,
            self.filtered_no_reflen,
            self.filtered_3p_anchor,
            self.filtered_coverage,
            self.filtered_polya_invalid,
            self.filtered_cap,
            self.passed,
            self.unaccounted(),
            self.filtered_no_refid,
            self.filtered_no_position,
            self.filtered_no_read_id,
            self.filtered_unmapped,
            self.filtered_supplementary,
            self.filtered_eligible_not_anchor,
            self.reads_seen,
            self.anchors_emitted,
            self.rejected_tie,
            self.rejected_gap,
            self.rejected_same_gene,
            self.rejected_cross_gene,
            self.gene_veto_disabled,
            self.same_gene_unknown,
            self.reads_no_eligible_candidate,
            self.reads_over_cap,
            self.rejected_usurpation,
        )
    }

    /// `total_records` minus every bucket that is supposed to absorb a record; must be 0 for the ledger to be conservative. `parse_errors` is deliberately NOT in this sum: it is incremented BEFORE `total_records`.
    pub fn unaccounted(&self) -> u64 {
        let bucket_sum = self.filtered_unique
            + self.filtered_unmapped
            + self.filtered_supplementary
            + self.filtered_no_refid
            + self.filtered_no_position
            + self.filtered_no_read_id
            + self.filtered_no_reflen
            + self.filtered_3p_anchor
            + self.filtered_coverage
            + self.filtered_polya_invalid
            + self.filtered_eligible_not_anchor
            + self.filtered_cap
            + self.passed;

        // Over-count (bucket_sum > total_records) is the ONE direction the `saturating_sub` chain below cannot report: it clamps to 0 and reads as "balanced". The under-count direction is surfaced by the returned value, so assert on this one. ⚠ NOT THE GATE: the release profile compiles debug assertions out.
        debug_assert!(
            bucket_sum <= self.total_records,
            "buckets sum to {} but total_records is {}: a record reached two \
             buckets, or `passed` was bumped without a push. saturating_sub \
             would report 0 for this and hide it.",
            bucket_sum,
            self.total_records
        );

        self.total_records
            .saturating_sub(self.filtered_unique)
            .saturating_sub(self.filtered_unmapped)
            .saturating_sub(self.filtered_supplementary)
            .saturating_sub(self.filtered_no_refid)
            .saturating_sub(self.filtered_no_position)
            .saturating_sub(self.filtered_no_read_id)
            .saturating_sub(self.filtered_no_reflen)
            .saturating_sub(self.filtered_3p_anchor)
            .saturating_sub(self.filtered_coverage)
            .saturating_sub(self.filtered_polya_invalid)
            .saturating_sub(self.filtered_eligible_not_anchor)
            .saturating_sub(self.filtered_cap)
            .saturating_sub(self.passed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::test_support::{
        synth_record, synth_record_no_position, synth_record_no_read_id, synth_record_no_refid,
        synth_record_with_flags, synth_record_with_gap, write_synthetic_bam,
    };

    /// The knobs every pre-existing fixture was written against: `tes_slack` 35, `min_coverage` 0.80, cap 200, and BOTH GAP FLOORS AT 0.0.
    fn params() -> AnchorParams {
        AnchorParams {
            tes_slack: 35,
            min_coverage: 0.80,
            max_reads_per_sample: 200,
            gap_abs: 0.0,
            gap_rel: 0.0,
        }
    }

    fn params_with_gap(gap_abs: f64, gap_rel: f64) -> AnchorParams {
        AnchorParams {
            gap_abs,
            gap_rel,
            ..params()
        }
    }

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
        // `add` is per-sample; the pool-level distinctness fields must not be summed.
        assert_eq!(agg.distinct_genes, 0);
        assert_eq!(agg.distinct_transcripts, 0);
    }

    // The degenerate gene pool: with no `--tx-to-gene`, `gene_id := transcript_id`, so the L2 gene-level fallback can never fire.

    /// A three-transcript synthetic BAM whose reads all clear the anchor criteria (MAPQ 60, full-length so 3'-anchored and coverage 1.0, `pt:i = 100`).
    fn three_tx_anchor_bam(tag: &str) -> std::path::PathBuf {
        let recs = vec![
            synth_record("a0", 0, None, 0, 60),
            synth_record("a1", 0, None, 0, 60),
            synth_record("b0", 1, None, 0, 60),
            synth_record("c0", 2, None, 0, 60),
        ];
        write_synthetic_bam(tag, recs)
    }

    #[test]
    fn test_degenerate_gene_pool_without_tx_to_gene_warns_and_names_the_consequence() {
        let bam = three_tx_anchor_bam("degen_pos");
        let out = std::env::temp_dir().join(format!(
            "telescope_anchor_degen_pos_{}.parquet",
            std::process::id()
        ));
        let stats = build_anchor_pool(&[(bam.clone(), "S1")], &TxToGene::new(), &out, params())
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
            ("the L2 tier's consequence", "L2"),
            ("gene-level wording", "gene-level"),
            (
                "the consequence for shared parameters",
                "no two transcripts ever share",
            ),
            ("the default thresholds it is scoped to", "tx_min_reads=10"),
            ("the remedy", "--tx-to-gene"),
        ] {
            assert!(msg.contains(needle), "warning must name {what}; got: {msg}");
        }
        // Scoped, not absolute: `--gene-min-reads 3` makes L2 fire on this very pool.
        assert!(
            !msg.contains("STRUCTURALLY UNREACHABLE"),
            "must not claim L2 is unconditionally unreachable; the thresholds \
             are build-emission flags and can make it fire; got: {msg}"
        );
    }

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

        let stats = build_anchor_pool(&[(bam.clone(), "S1")], &map, &out, params())
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

    /// The detector is the equality of the two distinct counts, and nothing else.
    #[test]
    fn test_detector_is_exactly_equality_of_the_two_distinct_counts() {
        let row = |gene: &str, tx: &str| AnchorRecord {
            sample_id: "S1".into(),
            read_id: "r".into(),
            gene_id: gene.into(),
            transcript_id: tx.into(),
            poly_a_length: 100,
        };
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

        // Strict inequality, either direction, is the only silence.
        assert!(degenerate_gene_pool_warning(1, 3).is_none());
        assert!(degenerate_gene_pool_warning(2, 3).is_none());

        assert_eq!(pool_gene_tx_cardinality(&[]), (0, 0));
        assert!(degenerate_gene_pool_warning(0, 0).is_none());
    }

    // The identity `total_records == sum(filtered_*) + passed` is enforced. REACHABILITY IS 9/10; the
    // one path that remains is `filtered_no_reflen`, because `ref_len` is built from the header, whose lengths are `NonZero`.
    fn counted_buckets(s: &SampleStats) -> u64 {
        s.filtered_unique
            + s.filtered_unmapped
            + s.filtered_supplementary
            + s.filtered_no_refid
            + s.filtered_no_position
            + s.filtered_no_read_id
            + s.filtered_no_reflen
            + s.filtered_3p_anchor
            + s.filtered_coverage
            + s.filtered_polya_invalid
            + s.filtered_eligible_not_anchor
            + s.filtered_cap
    }

    #[test]
    fn test_anchor_record_conservation() {
        use noodles_sam::alignment::record::data::field::Tag;
        use noodles_sam::alignment::record_buf::data::field::Value;
        let nm: Tag = Tag::new(b'N', b'M');

        // No pt:i -> filtered_polya_invalid (A4, poly_a_length < 10).
        let mut nopt = synth_record("nopt", 0, None, 0, 60);
        nopt.data_mut().clear();
        nopt.data_mut().insert(nm, Value::from(1i32));

        let mut recs = vec![
            synth_record("ok0", 0, None, 0, 60),
            synth_record("ok1", 1, None, 0, 60),
            // MAPQ 10: there is NO MAPQ gate any more, so this clears every filter and becomes a third anchor. The next three records each fail exactly one physical filter.
            synth_record("lowq", 0, None, 0, 10),
            synth_record_with_gap("gap3p", 1, 100, true, None, 0, 60),
            synth_record_with_gap("shortcov", 1, 500, false, None, 0, 60),
            nopt,
        ];
        // 205 records on one transcript with the cap at 200 -> 5 over cap. ALL 205 ARE SINGLE-RECORD READS, one QNAME each, so `eligible_records == 1` for every one.
        for i in 0..205 {
            recs.push(synth_record(&format!("cap{i}"), 2, None, 0, 60));
        }

        let bam = write_synthetic_bam("conserve", recs);
        let (anchors, s) =
            extract_anchors_from_bam(&bam, "conserve", &TxToGene::new(), params()).unwrap();

        // Asserted before the per-counter expectations on purpose: an uncounted drop must be reported as the conservation failure it is.
        assert_eq!(
            s.total_records,
            counted_buckets(&s) + s.passed,
            "a record left the stream through an uncounted path: \
             total_records={} counted={} passed={}",
            s.total_records,
            counted_buckets(&s),
            s.passed
        );
        assert_eq!(
            anchors.len() as u64,
            s.passed,
            "passed counts records that never reached the pool"
        );
        // parse_errors is incremented BEFORE total_records, so it is a disjoint population and must NOT appear in the identity.
        assert_eq!(s.parse_errors, 0);

        assert_eq!(
            s.filtered_unique, 0,
            "`lowq` carries MAPQ 10 and there is no MAPQ gate any more, so the \
             retired slot stays at 0 and the record becomes an anchor"
        );
        assert_eq!(s.filtered_3p_anchor, 1, "fixture lost its 3'-anchor record");
        assert_eq!(s.filtered_coverage, 1, "fixture lost its coverage record");
        assert_eq!(
            s.filtered_polya_invalid, 1,
            "fixture lost its no-poly(A) record"
        );
        assert_eq!(s.filtered_cap, 5, "fixture lost its over-cap records");
        assert_eq!(s.passed, 203, "fixture lost its passing records");
        assert_eq!(
            s.total_records,
            1 + 1 + 1 + 5 + 203,
            "the fixture's record count changed; re-derive the expectations"
        );
        assert_eq!(s.filtered_no_refid, 0);
        assert_eq!(s.filtered_no_position, 0);
        assert_eq!(s.filtered_no_read_id, 0);
        assert_eq!(s.filtered_no_reflen, 0);

        let _ = std::fs::remove_file(&bam);
    }

    /// Reachability is 9/10. The one path this fixture does not reach is `filtered_no_reflen`, because `ref_len` is built from the header, whose lengths are `NonZero`.
    #[test]
    fn test_the_three_observable_drop_paths_really_execute() {
        let recs = vec![
            synth_record("ok0", 0, None, 0, 60),
            synth_record("ok1", 1, None, 0, 60),
            // No tid: dies at reference_sequence_id() -> filtered_no_refid
            synth_record_no_refid("norid", None, 0, 60),
            // Valid tid, POS never set -> encoder writes 0 -> no Position -> filtered_no_position
            synth_record_no_position(2, None, 0, 60),
            // No QNAME, valid tid, FLAG 0: clears every filter but the last -> filtered_no_read_id
            synth_record_no_read_id(0, None, 0, 60),
            // reverse + gap => 3' end lands short of the TES by more than tes_slack -> filtered_3p_anchor
            synth_record_with_gap("gap3p", 1, 100, true, None, 0, 60),
        ];

        let bam = write_synthetic_bam("reach", recs);
        let (anchors, s) =
            extract_anchors_from_bam(&bam, "reach", &TxToGene::new(), params()).unwrap();

        // The identity must be the thing that reports an uncounted drop rather than a fixture expectation noticing a number moved.
        assert_eq!(
            s.total_records,
            counted_buckets(&s) + s.passed,
            "a record left the stream through an uncounted path: \
             total_records={} counted={} passed={}",
            s.total_records,
            counted_buckets(&s),
            s.passed
        );

        assert_eq!(s.total_records, 6);
        assert_eq!(s.passed, 2);
        assert_eq!(
            s.filtered_no_refid, 1,
            "a record with no tid must reach filtered_no_refid, not filtered_unique"
        );
        assert_eq!(
            s.filtered_no_position, 1,
            "a record with POS=0 must reach filtered_no_position"
        );
        assert_eq!(
            s.filtered_no_read_id, 1,
            "a record with no QNAME must reach filtered_no_read_id"
        );
        assert_eq!(s.filtered_3p_anchor, 1);
        assert_eq!(anchors.len() as u64, s.passed);
        // Signed, so an OVER-count is visible too: `unaccounted()` clamps with `saturating_sub` and would read a double-count as "balanced".
        assert_eq!(
            s.total_records as i64 - counted_buckets(&s) as i64 - s.passed as i64,
            0,
            "signed conservation: a negative value means a record was counted twice"
        );

        let _ = std::fs::remove_file(&bam);
    }

    #[test]
    fn test_unaccounted_is_zero_when_every_bucket_is_populated() {
        use noodles_sam::alignment::record::data::field::Tag;
        use noodles_sam::alignment::record_buf::data::field::Value;
        let _ = std::mem::size_of::<Tag>();
        let _ = std::mem::size_of::<Value>();

        let agg = AnchorStats {
            samples: 1,
            total_records: 100,
            parse_errors: 7,
            filtered_unique: 5,
            filtered_unmapped: 12,
            filtered_supplementary: 13,
            filtered_no_refid: 3,
            filtered_no_position: 4,
            filtered_no_read_id: 5,
            filtered_no_reflen: 6,
            filtered_3p_anchor: 7,
            filtered_coverage: 8,
            filtered_polya_invalid: 9,
            filtered_eligible_not_anchor: 10,
            filtered_cap: 11,
            passed: 7,
            // A CLOSED read block: 7 + 9 + 6 + 1 + 3 + 2 + 1 + 4 + 2 == 35.
            reads_seen: 35,
            anchors_emitted: 7,
            rejected_tie: 9,
            rejected_usurpation: 6,
            rejected_gap: 1,
            rejected_same_gene: 3,
            rejected_cross_gene: 2,
            gene_veto_disabled: 1,
            same_gene_unknown: 1,
            reads_no_eligible_candidate: 4,
            reads_over_cap: 2,
            distinct_genes: 2,
            distinct_transcripts: 3,
        };
        assert_eq!(
            agg.unaccounted(),
            0,
            "twelve populated buckets summing to total_records must leave 0"
        );

        let leaking = AnchorStats {
            total_records: 101,
            ..agg.clone()
        };
        assert_eq!(
            leaking.unaccounted(),
            1,
            "unaccounted() must report the gap, not clamp it"
        );
    }

    fn tx2gene_tsv(tag: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "telescope_tx2gene_{tag}_{}.tsv",
            std::process::id()
        ));
        std::fs::write(&p, format!("transcript_id\tgene_id\n{body}")).unwrap();
        p
    }

    fn run_extract(
        tag: &str,
        recs: Vec<noodles_sam::alignment::record_buf::RecordBuf>,
        map: &TxToGene,
        p: AnchorParams,
    ) -> (Vec<AnchorRecord>, SampleStats) {
        let bam = write_synthetic_bam(tag, recs);
        let out = extract_anchors_from_bam(&bam, tag, map, p).expect("extract_anchors_from_bam");
        let _ = std::fs::remove_file(&bam);
        out
    }

    /// The conservation identity, asserted FIRST, before any per-counter expectation.
    fn record_deficit(s: &SampleStats) -> u64 {
        s.total_records - counted_buckets(s) - s.passed
    }

    fn assert_record_ledger_closes(s: &SampleStats) {
        assert_eq!(
            s.total_records,
            counted_buckets(s) + s.passed,
            "a record left the stream through an uncounted path: total_records={} \
             counted={} passed={}",
            s.total_records,
            counted_buckets(s),
            s.passed
        );
    }

    /// A SAME-GENE PAIR WHOSE GAP IS CLEAR **EMITS**: there is no unconditional same-gene veto.
    #[test]
    fn test_a_same_gene_pair_with_a_clear_gap_emits_rather_than_being_vetoed() {
        let tsv = tx2gene_tsv("samegene", "tx0\tGA\ntx1\tGA\ntx2\tGA\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();
        assert_eq!(map.len(), 3);

        let (anchors, s) = run_extract(
            "samegene",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record("r", 1, Some(90), 0, 60),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&tsv);

        assert_record_ledger_closes(&s);
        assert_eq!(
            anchors.len(),
            1,
            "AS 100 vs 90 means the read aligned over sequence the runner-up does \
             not contain; on this data a single mismatch costs 6 AS, so 10 AS is \
             real isoform evidence and the read is an anchor"
        );
        assert_eq!(anchors[0].transcript_id, "tx0");
        assert_eq!(s.passed, 1);
        assert_eq!(
            s.rejected_same_gene, 0,
            "rejected_same_gene is a CLASSIFICATION of a gap failure, not a veto; \
             nothing failed the gap here"
        );
        assert_eq!(
            s.rejected_cross_gene, 0,
            "a same-gene pair must not be counted as cross-gene"
        );
        assert_eq!(s.rejected_tie, 0, "AS 100 vs 90 is not a tie");
        assert_eq!(
            s.filtered_eligible_not_anchor, 1,
            "two records were eligible, one became the anchor"
        );
        assert_eq!(record_deficit(&s), 0);
    }

    /// ...AND WITH A GAP THAT IS **NOT** CLEAR, THE SAME PAIR IS REJECTED AND CLASSIFIED AS SAME-GENE. The floors are raised above the 10 AS the read actually has.
    #[test]
    fn test_a_same_gene_pair_rejected_by_the_gap_is_classified_as_same_gene() {
        let tsv = tx2gene_tsv("samegenegap", "tx0\tGA\ntx1\tGA\ntx2\tGA\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();

        let p = AnchorParams {
            gap_abs: 15.0,
            gap_rel: 0.0,
            ..params()
        };
        let (anchors, s) = run_extract(
            "samegenegap",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record("r", 1, Some(90), 0, 60),
            ],
            &map,
            p,
        );
        let _ = std::fs::remove_file(&tsv);

        assert_record_ledger_closes(&s);
        assert_eq!(anchors.len(), 0, "10 AS does not clear a floor of 15");
        assert_eq!(s.rejected_same_gene, 1);
        assert_eq!(s.rejected_cross_gene, 0);
        assert_eq!(s.rejected_tie, 0);
        assert_eq!(
            s.filtered_eligible_not_anchor, 2,
            "both records were eligible and neither was emitted"
        );
        assert_eq!(record_deficit(&s), 0);
    }

    /// CROSS-GENE PAIR REJECTED BY THE GAP, COUNTED SEPARATELY. `tx1` is in its own gene and the gap is raised to 15, so the pair is rejected for the gap rather than for sharing a gene.
    #[test]
    fn test_a_cross_gene_pair_rejected_by_the_gap_is_counted_separately() {
        let tsv = tx2gene_tsv("crossgene", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();

        let (anchors, s) = run_extract(
            "crossgene",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record("r", 1, Some(90), 0, 60),
            ],
            &map,
            // gap 10 < 15 -> rejected
            params_with_gap(15.0, 0.0),
        );
        let _ = std::fs::remove_file(&tsv);

        assert_record_ledger_closes(&s);
        assert_eq!(anchors.len(), 0);
        assert_eq!(s.rejected_cross_gene, 1, "tx0/GA vs tx1/GB is cross-gene");
        assert_eq!(
            s.rejected_same_gene, 0,
            "the two candidates are in DIFFERENT genes"
        );
        assert_eq!(s.rejected_gap, 0, "the gene relation was known here");
        assert_eq!(s.filtered_eligible_not_anchor, 2);
    }

    /// EXACT ARGMAX TIE. AS 100 vs AS 100 — not "close", identical. The strict argmax has no winner, so there is no anchor. Cross-gene on purpose: it proves the tie fires BEFORE the gene comparison, so a tie is not silently reclassified as a same-gene veto.
    #[test]
    fn test_an_exact_argmax_tie_yields_no_anchor() {
        let tsv = tx2gene_tsv("argmaxtie", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();

        let (anchors, s) = run_extract(
            "argmaxtie",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record("r", 1, Some(100), 0, 60),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&tsv);

        assert_record_ledger_closes(&s);
        assert_eq!(anchors.len(), 0, "an exact AS tie has no argmax");
        assert_eq!(s.rejected_tie, 1);
        assert_eq!(
            s.rejected_same_gene, 0,
            "the tie is checked first, so the gene relation never decides this"
        );
        assert_eq!(s.rejected_cross_gene, 0);
        assert_eq!(s.passed, 0);
        assert_eq!(s.filtered_eligible_not_anchor, 2);

        // And the same tie under a SAME-gene map is still a tie, not a veto.
        let tsv2 = tx2gene_tsv("argmaxtie2", "tx0\tGA\ntx1\tGA\ntx2\tGA\n");
        let map2 = TxToGene::from_tsv(&tsv2).unwrap();
        let (anchors2, s2) = run_extract(
            "argmaxtie2",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record("r", 1, Some(100), 0, 60),
            ],
            &map2,
            params(),
        );
        let _ = std::fs::remove_file(&tsv2);
        assert_eq!(anchors2.len(), 0);
        assert_eq!(s2.rejected_tie, 1);
        assert_eq!(s2.rejected_same_gene, 0, "the tie fired first");
    }

    /// THE PHYSICALLY-QUALIFIED ARGMAX IS REFUSED, NOT SILENTLY REPLACED BY ITS RUNNER-UP. tx0 (AS 100)
    /// is reverse-strand with a 100 bp 5' gap, so its 3' end lands 100 bp short of the TES — beyond the slack of 35 — and it is disqualified. tx1 (AS 50) is full length.
    #[test]
    fn test_a_physically_disqualified_argmax_is_refused_rather_than_emitted() {
        let tsv = tx2gene_tsv("usurp", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();

        let (anchors, s) = run_extract(
            "usurp",
            vec![
                synth_record_with_gap("r", 0, 100, true, Some(100), 0, 60),
                synth_record("r", 1, Some(50), 0, 60),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&tsv);

        assert_record_ledger_closes(&s);
        assert_eq!(
            anchors.len(),
            0,
            "tx1 is the read's runner-up; attributing the read to it would be a \
             different and weaker claim than 'this is the read's transcript'"
        );
        assert_eq!(s.rejected_usurpation, 1);
        assert_eq!(s.passed, 0);
        assert_eq!(
            s.filtered_3p_anchor, 1,
            "the disqualified tx0 record went to its own bucket, not to the \
             read-level rejection"
        );
        assert_eq!(
            s.filtered_eligible_not_anchor, 1,
            "tx1 WAS physically eligible and is still not emitted, so its record \
             has to be absorbed here or the record ledger leaks"
        );
        assert_eq!(s.reads_seen, 1);
        assert_eq!(s.rejected_tie, 0, "100 vs 50 is not a tie");
        assert_eq!(s.rejected_cross_gene, 0, "the gap never got to decide");
        assert_eq!(s.rejected_same_gene, 0, "the gap never got to decide");
        assert_eq!(
            s.reads_seen,
            s.rejected_usurpation
                + s.anchors_emitted
                + s.rejected_tie
                + s.rejected_gap
                + s.rejected_same_gene
                + s.rejected_cross_gene
                + s.reads_no_eligible_candidate
                + s.reads_over_cap,
            "the read identity must close with usurpation as a destination"
        );
    }

    /// A RAW TOP-TWO TIE WITH EXACTLY ONE PHYSICALLY-PASSING MEMBER IS A **TIE**, not an usurpation
    /// and not an anchor. This pins the ORDER of branches (b) and (c): the two candidates tie
    /// EXACTLY on AS, so "which of them sorts first" is arbitrary — and with the eligibility check
    /// first, the verdict of this read would depend on that arbitrary order.
    #[test]
    fn test_a_raw_top_two_tie_with_one_passing_member_is_a_tie_not_an_usurpation() {
        let tsv = tx2gene_tsv("rawtie", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();

        // tx0 AS 100 disqualified on the 3' test; tx1 AS 100 full length. The raw top two are exactly equal, so `scratch[0].2` is false while `scratch[1].2` is true — the exact state that is order-sensitive.
        let (anchors, s) = run_extract(
            "rawtie",
            vec![
                synth_record_with_gap("r", 0, 100, true, Some(100), 0, 60),
                synth_record("r", 1, Some(100), 0, 60),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&tsv);

        assert_record_ledger_closes(&s);
        assert_eq!(
            anchors.len(),
            0,
            "an exact AS tie has no argmax; emitting either member would invent \
             an isoform assignment the aligner did not make"
        );
        assert_eq!(s.rejected_tie, 1);
        assert_eq!(
            s.rejected_usurpation, 0,
            "THE BRANCH ORDER: a tie must be decided before the winner's \
             eligibility is inspected, or this read's verdict depends on which \
             of the two tied candidates happens to sort first"
        );
        assert_eq!(s.filtered_3p_anchor, 1);
        assert_eq!(s.filtered_eligible_not_anchor, 1);
        assert_eq!(s.passed, 0);
    }

    #[test]
    fn test_the_tie_fixture_differs_from_the_usurpation_fixture_by_one_as_point() {
        let tsv = tx2gene_tsv("tieadj", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();

        // tx0 AS 101 disqualified, tx1 AS 100 eligible: no tie any more.
        let (a1, s1) = run_extract(
            "tieadj",
            vec![
                synth_record_with_gap("r", 0, 100, true, Some(101), 0, 60),
                synth_record("r", 1, Some(100), 0, 60),
            ],
            &map,
            params(),
        );
        assert_record_ledger_closes(&s1);
        assert_eq!(a1.len(), 0);
        assert_eq!(
            s1.rejected_usurpation, 1,
            "one AS point of separation is the whole difference between 'tie' \
             and 'usurpation'"
        );
        assert_eq!(s1.rejected_tie, 0);

        // tx0 AS 99 disqualified, tx1 AS 100 eligible: tx1 wins outright.
        let (a2, s2) = run_extract(
            "tieadj2",
            vec![
                synth_record_with_gap("r", 0, 100, true, Some(99), 0, 60),
                synth_record("r", 1, Some(100), 0, 60),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&tsv);
        assert_record_ledger_closes(&s2);
        assert_eq!(a2.len(), 1, "the eligible candidate IS the argmax");
        assert_eq!(a2[0].transcript_id, "tx1");
        assert_eq!(s2.rejected_usurpation, 0);
        assert_eq!(s2.rejected_tie, 0);
    }

    /// A READ WITH **ONE** RAW CANDIDATE IS MEASURED AGAINST THE FLOOR: `delta_AS_raw = AS_1 - 0`. Two reads identical except for their single candidate's AS: 11 (below a floor of 12) and 12 (exactly on it).
    #[test]
    fn test_a_read_with_one_raw_candidate_is_still_measured_against_the_gap_floor() {
        let tsv = tx2gene_tsv("solo_gap", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();

        let (anchors, s) = run_extract(
            "solo_gap",
            vec![
                // 11 < 12 -> rejected into `rejected_gap` (no pair, so Unknown).
                synth_record("under", 0, Some(11), 0, 60),
                // 12 == 12 -> the floor is inclusive, so this emits.
                synth_record("on", 1, Some(12), 0, 60),
            ],
            &map,
            params_with_gap(12.0, 0.0),
        );
        let _ = std::fs::remove_file(&tsv);

        assert_record_ledger_closes(&s);
        assert_eq!(anchors.len(), 1, "only the read exactly ON the floor emits");
        assert_eq!(anchors[0].read_id, "on");
        assert_eq!(
            s.rejected_gap, 1,
            "there is no second raw candidate, so there is no gene relation to \
             classify and the gap is the only thing left to name it by"
        );
        assert_eq!(s.rejected_usurpation, 0, "both argmaxes are eligible");
        assert_eq!(s.rejected_tie, 0, "one candidate each");
        assert_eq!(s.filtered_eligible_not_anchor, 1);
    }

    /// THE GAP IS MEASURED AGAINST THE **RAW** RUNNER-UP, EVEN WHEN THAT RUNNER-UP IS INELIGIBLE.
    /// tx0 AS 100 is eligible, tx1 AS 92 is full length but its poly(A) fails (`pt < 10`). On
    /// C_raw the runner-up exists and is 8 AS behind, so a 15-point floor rejects the read into
    /// `rejected_cross_gene`.
    #[test]
    fn test_the_gap_is_measured_against_an_ineligible_raw_runner_up() {
        let tsv = tx2gene_tsv("rawrunnerup", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();

        let (anchors, s) = run_extract(
            "rawrunnerup",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                // 5' gap of 900 bp on a 2000 bp reference: coverage 0.55 < 0.80.
                synth_record_with_gap("r", 1, 900, false, Some(92), 0, 60),
            ],
            &map,
            params_with_gap(15.0, 0.0),
        );
        let _ = std::fs::remove_file(&tsv);

        assert_record_ledger_closes(&s);
        assert_eq!(
            anchors.len(),
            0,
            "the read's second-best RAW alignment is 8 AS behind, which is \
             below the floor of 15; a candidate that fails the coverage filter \
             is disqualified, not deleted"
        );
        assert_eq!(s.rejected_cross_gene, 1, "tx0/GA vs tx1/GB");
        assert_eq!(
            s.filtered_coverage, 1,
            "the runner-up is also accounted as the record it is"
        );
        assert_eq!(
            s.filtered_eligible_not_anchor, 1,
            "tx0 was eligible and not emitted"
        );
    }

    /// GAP BOUNDARY, AT THE ARITHMETIC'S OWN EDGE. Two reads, identical in every way except the
    /// AS gap: 10 and 20. The floor is 15. One gap-1 either side decides the run, so the
    /// comparison cannot be `<=` on one side and `<` on the other.
    #[test]
    fn test_the_gap_boundary_is_inclusive_and_exact() {
        let tsv = tx2gene_tsv("gapbound", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();

        let (anchors, s) = run_extract(
            "gapbound",
            vec![
                // gap 10: one BELOW the floor of 15 -> rejected
                synth_record("narrow", 0, Some(100), 0, 60),
                synth_record("narrow", 1, Some(90), 0, 60),
                // gap 20: five above -> accepted, and tx0/GA beats tx1/GB
                synth_record("wide", 0, Some(100), 0, 60),
                synth_record("wide", 1, Some(80), 0, 60),
                // gap exactly 15: ON the floor -> accepted, since the rule is `gap >= floor`.
                synth_record("exact", 0, Some(100), 0, 60),
                synth_record("exact", 1, Some(85), 0, 60),
            ],
            &map,
            params_with_gap(15.0, 0.0),
        );
        let _ = std::fs::remove_file(&tsv);

        assert_record_ledger_closes(&s);
        assert_eq!(
            anchors.len(),
            2,
            "gap 20 and gap 15 clear a floor of 15; gap 10 does not"
        );
        assert_eq!(
            s.rejected_cross_gene, 1,
            "exactly one cross-gene pair was rejected on the gap"
        );
        assert_eq!(s.passed, 2);
        let ids: Vec<&str> = anchors.iter().map(|a| a.read_id.as_str()).collect();
        assert!(
            ids.contains(&"wide") && ids.contains(&"exact"),
            "the two accepted reads must be the wide and exact ones; got {ids:?}"
        );
        assert!(!ids.contains(&"narrow"), "gap 10 is below the floor of 15");

        // THE OTHER FLOOR, exercised on its own: `gap_rel` is a fraction of AS_best, so with AS_best = 100 a floor of 0.05 is a gap of 5.
        let (a2, s2) = run_extract(
            "gaprel",
            vec![
                synth_record("relnarrow", 0, Some(100), 0, 60),
                synth_record("relnarrow", 1, Some(96), 0, 60), // gap 4 < 5
            ],
            &map,
            params_with_gap(0.0, 0.05),
        );
        assert_eq!(a2.len(), 0, "gap 4 is below 5% of AS_best = 100");
        assert_eq!(s2.rejected_cross_gene, 1);
        assert_record_ledger_closes(&s2);
    }

    /// DEDUP BY ref_id, KEEPING MAX AS. Two records on the SAME transcript are one candidate,
    /// not two, so they cannot produce the cross-transcript argmax tie, and only the highest-scoring one is put through the physical filters.
    #[test]
    fn test_two_records_to_the_same_reference_dedup_to_the_max_as() {
        let tsv = tx2gene_tsv("dedup", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();

        // Read 1: on tx0 only. The AS-100 record is reverse-strand with a 100 bp 5' gap, so its 3'
        // end lands 100 bp short of the TES — beyond the slack of 35 — and it is the max-AS record, hence the one evaluated.
        let (anchors, s) = run_extract(
            "dedup",
            vec![
                synth_record_with_gap("dupe", 0, 100, true, Some(100), 0, 60),
                synth_record("dupe", 0, Some(50), 0, 60),
            ],
            &map,
            params(),
        );
        assert_record_ledger_closes(&s);
        assert_eq!(
            s.filtered_3p_anchor, 2,
            "the max-AS representative failed the 3' filter, and it stands for \
             BOTH collapsed records — one deduped representative, two records"
        );
        assert_eq!(anchors.len(), 0, "no surviving candidate");
        assert_eq!(
            s.filtered_eligible_not_anchor, 0,
            "the records were not ELIGIBLE — they died on a physical filter, so \
             counting them here would misfile them"
        );

        // Read 2: three identical-AS records on one transcript. Dedup collapses them to one candidate, so this read has no argmax tie with itself.
        let (anchors2, s2) = run_extract(
            "dedup2",
            vec![
                synth_record("triple", 0, Some(100), 0, 60),
                synth_record("triple", 0, Some(100), 0, 60),
                synth_record("triple", 0, Some(100), 0, 60),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&tsv);
        assert_record_ledger_closes(&s2);
        assert_eq!(
            anchors2.len(),
            1,
            "three records to ONE reference are one candidate, not a tie"
        );
        assert_eq!(s2.rejected_tie, 0);
        assert_eq!(s2.passed, 1);
        assert_eq!(
            s2.filtered_eligible_not_anchor, 2,
            "the two collapsed duplicates were eligible and were not emitted"
        );
        assert_eq!(s2.reads_seen, 1, "three records, ONE read");
    }

    /// DEDUP GROUPS NON-ADJACENT RECORDS, WHICH IS THE ONLY WAY TO SEE IT. Here the same reference
    /// is split across the read by another one, so the two `tx0` records only become one candidate
    /// if the sort really ran. Sorted: the `tx0` representative is the TRUNCATED one (AS 100), so
    /// it fails the 3' filter and stands for BOTH of its records (`filtered_3p_anchor == 2`). It is
    /// ALSO the read's raw argmax, and it is ineligible, so under the C_raw rule the read is a
    /// runner-up usurpation and `tx1` is not emitted.
    #[test]
    fn test_dedup_groups_records_that_are_not_adjacent_in_the_bam() {
        let tsv = tx2gene_tsv("dedupadj", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();

        let (anchors, s) = run_extract(
            "dedupadj",
            vec![
                synth_record_with_gap("split", 0, 100, true, Some(100), 0, 60),
                synth_record("split", 1, Some(50), 0, 60),
                synth_record("split", 0, Some(50), 0, 60),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&tsv);

        assert_record_ledger_closes(&s);
        assert_eq!(
            s.filtered_3p_anchor, 2,
            "the tx0 representative fails the 3' filter and stands for BOTH of \
             the tx0 records, wherever they sit in the BAM"
        );
        assert_eq!(
            anchors.len(),
            0,
            "the read's raw argmax is the TRUNCATED tx0 record, which is not \
             physically eligible; emitting the lower-AS tx1 instead is exactly \
             the runner-up usurpation the C_raw rule exists to refuse"
        );
        assert_eq!(s.rejected_usurpation, 1);
        assert_eq!(
            s.rejected_tie, 0,
            "the two tx0 records are one candidate, so the raw top two are \
             tx0@100 and tx1@50, which is not a tie"
        );
        assert_eq!(
            s.filtered_eligible_not_anchor, 1,
            "the one physically-passing record of this read (tx1) is absorbed"
        );
        assert_eq!(s.passed, 0);
    }

    /// SUPPLEMENTARY AND UNMAPPED RECORDS DO NOT REACH THE POOL — INCLUDING AT MAPQ >= 20: a MAPQ
    /// filter alone would let a split segment into the pool as if it were a complete molecule.
    #[test]
    fn test_supplementary_and_unmapped_records_never_become_anchors_even_at_high_mapq() {
        use noodles_sam::alignment::record::Flags;

        let (anchors, s) = run_extract(
            "flagged",
            vec![
                synth_record_with_flags("prim", 0, Flags::empty(), Some(100)),
                synth_record_with_flags("supp", 0, Flags::SUPPLEMENTARY, Some(100)),
                synth_record_with_flags("unmapped", 1, Flags::UNMAPPED, Some(100)),
            ],
            &TxToGene::new(),
            params(),
        );

        assert_record_ledger_closes(&s);
        assert_eq!(
            anchors.len(),
            1,
            "only the primary alignment may reach the pool"
        );
        assert_eq!(s.filtered_supplementary, 1);
        assert_eq!(s.filtered_unmapped, 1);
        assert_eq!(
            s.filtered_unique, 0,
            "every fixture record here has MAPQ 60; the two flagged records must \
             have been caught by their FLAG, not by the MAPQ gate"
        );
        assert_eq!(
            s.filtered_eligible_not_anchor, 0,
            "a supplementary record is not 'eligible but not selected' — it was \
             never a molecule"
        );
        assert_eq!(s.passed, 1);
        assert_eq!(
            s.reads_seen, 1,
            "the two flagged records never formed a read"
        );
    }

    /// SECONDARY ALIGNMENTS ARE ADMITTED, and the record ledger still closes with them in the stream.
    /// `0x100` is the flag the input contract requires be preserved.
    #[test]
    fn test_the_record_ledger_closes_while_secondaries_are_admitted() {
        use noodles_sam::alignment::record::Flags;

        let tsv = tx2gene_tsv("secondary", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();
        let (anchors, s) = run_extract(
            "secondary",
            vec![
                synth_record("p0", 0, Some(100), 0, 60),
                // The same read, with its SECONDARY on tx1 losing on AS.
                synth_record_with_flags("p0", 1, Flags::SECONDARY, Some(40)),
                // A read whose ONLY alignment is secondary: it must still win.
                synth_record_with_flags("s0", 1, Flags::SECONDARY, Some(100)),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&tsv);

        assert_record_ledger_closes(&s);
        assert_eq!(record_deficit(&s), 0);
        assert_eq!(s.total_records, 3);
        assert_eq!(
            anchors.len(),
            2,
            "p0 keeps tx0 on AS and s0 is anchored by a secondary alone"
        );
        assert_eq!(s.passed, 2);
        assert_eq!(
            s.filtered_eligible_not_anchor, 1,
            "the AS-40 secondary lost"
        );
        assert_eq!(
            s.filtered_unique, 0,
            "a secondary with MAPQ 60 must not be counted as low-MAPQ"
        );
        // SET-WISE, NOT SEQUENCE-WISE. Anchor ROW ORDER is a deterministic function of the read's
        // 128-bit name hash (see the pass-2 ordering comment), not of BAM order. Nothing downstream
        // may depend on the order: the emission model accumulates every row into per-transcript
        // collections and never indexes across rows.
        let mut tx: Vec<&str> = anchors.iter().map(|a| a.transcript_id.as_str()).collect();
        tx.sort_unstable();
        assert_eq!(tx, vec!["tx0", "tx1"]);
        let mut ids: Vec<&str> = anchors.iter().map(|a| a.read_id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec!["p0", "s0"]);
    }

    /// ANCHORS EMITTED <= READS SEEN, AND THE READ BLOCK CLOSES. The inequality is the one that can only break if selection ever stops being per read.
    #[test]
    fn test_anchors_emitted_never_exceed_reads_seen_and_the_read_block_closes() {
        use noodles_sam::alignment::record::Flags;
        let tsv = tx2gene_tsv("closure", "tx0\tGA\ntx1\tGB\ntx2\tGA\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();

        let (anchors, s) = run_extract(
            "closure",
            vec![
                synth_record("solo", 0, Some(100), 0, 60),
                synth_record("cross", 0, Some(100), 0, 60),
                synth_record("cross", 1, Some(80), 0, 60),
                synth_record("tie", 0, Some(100), 0, 60),
                synth_record("tie", 1, Some(100), 0, 60),
                // same gene (tx0/GA, tx2/GA) but gap 30 > 0 -> EMITS.
                synth_record("samegene", 0, Some(100), 0, 60),
                synth_record("samegene", 2, Some(70), 0, 60),
                // physically ineligible everywhere -> no eligible candidate
                synth_record_with_gap("dead", 1, 100, true, Some(100), 0, 60),
                // dropped before grouping, so NOT a read
                synth_record_with_flags("gone", 0, Flags::UNMAPPED, Some(100)),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&tsv);

        assert_record_ledger_closes(&s);
        assert_eq!(
            s.reads_seen, 5,
            "the unmapped record never formed a group, so there are 5 reads"
        );
        assert_eq!(s.anchors_emitted, 3);
        assert_eq!(anchors.len() as u64, s.anchors_emitted);
        assert!(
            s.anchors_emitted <= s.reads_seen,
            "one anchor per read at most: {} anchors from {} reads",
            s.anchors_emitted,
            s.reads_seen
        );
        assert_eq!(
            s.passed, s.anchors_emitted,
            "`passed` means anchors emitted"
        );

        assert_eq!(s.rejected_tie, 1);
        // `rejected_same_gene` is only how a GAP failure is CLASSIFIED, never an independent veto.
        // With `params()` the floors are 0.0, nothing fails the gap, so both same-gene and
        // cross-gene are 0.
        assert_eq!(s.rejected_same_gene, 0);
        assert_eq!(s.rejected_cross_gene, 0);
        assert_eq!(s.gene_veto_disabled, 0);
        assert_eq!(s.reads_no_eligible_candidate, 1);

        assert_eq!(
            s.reads_seen,
            s.anchors_emitted
                + s.rejected_tie
                + s.rejected_usurpation
                + s.rejected_gap
                + s.rejected_same_gene
                + s.rejected_cross_gene
                + s.reads_no_eligible_candidate
                + s.reads_over_cap,
            "every read that reached the per-read stage must land in exactly one \
             outcome; reads_seen={} tie={} usurpation={} gap={} same_gene={} \
             cross_gene={} no_eligible={} over_cap={} anchors={} \
             (gene_veto_disabled={} is NOT a destination: those reads cleared \
             the gap and were emitted)",
            s.reads_seen,
            s.rejected_tie,
            s.rejected_usurpation,
            s.rejected_gap,
            s.rejected_same_gene,
            s.rejected_cross_gene,
            s.gene_veto_disabled,
            s.reads_no_eligible_candidate,
            s.reads_over_cap,
            s.anchors_emitted,
        );
        // `reads_over_cap` and `rejected_usurpation` are both 0 on this fixture, so adding them above is arithmetically inert.
        assert_eq!(
            s.reads_over_cap, 0,
            "no transcript reaches the cap of 200 here, so this fixture cannot \
             witness reads_over_cap -- do not read the identity above as \
             covering it"
        );
    }

    /// THE CAP BRANCH ABSORBS EVERY RECORD THE REJECTED READ CONTRIBUTED. The fixture is three reads
    /// of TWO records each, so `eligible_records - 1` is 1 and a record leak is a whole record.
    /// ORDER-INDEPENDENT ON PURPOSE: the cap keeps the first `max_reads_per_sample` reads in
    /// read-id-hash order, so which of the three competing reads is rejected is not knowable in
    /// advance. All three have `eligible_records == 2`, so the assertion holds whichever loses.
    #[test]
    fn test_a_multi_record_read_over_the_cap_absorbs_every_record_it_contributed() {
        let tsv = tx2gene_tsv("capleak", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();
        assert_eq!(map.len(), 3, "a cross-gene map, so the veto cannot fire");

        let mut recs = Vec::new();
        for i in 0..3 {
            // Two RECORDS, two TRANSCRIPTS, so the dedup cannot collapse them and `eligible_records`
            // is genuinely 2. The gap is 50 and the pair is cross-gene, so the read reaches the cap check.
            recs.push(synth_record(&format!("sat{i}"), 2, Some(100), 0, 60));
            recs.push(synth_record(&format!("sat{i}"), 0, Some(50), 0, 60));
        }
        // MAPQ 5: cleared by every remaining filter, becomes a fourth anchor.
        recs.push(synth_record("lowq", 1, Some(100), 0, 5));
        assert_eq!(recs.len(), 7);

        let mut p = params();
        p.max_reads_per_sample = 2;
        let (anchors, s) = run_extract("capleak", recs, &map, p);
        let _ = std::fs::remove_file(&tsv);

        assert_record_ledger_closes(&s);
        assert_eq!(record_deficit(&s), 0);

        assert_eq!(s.total_records, 7);
        assert_eq!(
            s.reads_seen, 4,
            "three saturating reads plus the low-MAPQ read"
        );

        assert_eq!(s.reads_over_cap, 1, "the third read on tx2 is over the cap");
        assert_eq!(
            s.reads_seen,
            s.anchors_emitted
                + s.rejected_tie
                + s.rejected_usurpation
                + s.rejected_gap
                + s.rejected_same_gene
                + s.rejected_cross_gene
                + s.gene_veto_disabled
                + s.reads_no_eligible_candidate
                + s.reads_over_cap,
            "the cap is a READ-level outcome and has to appear in the read identity"
        );
        assert_eq!(
            s.anchors_emitted, 3,
            "two cap winners plus the low-MAPQ read"
        );
        assert_eq!(s.passed, 3);
        assert_eq!(anchors.len(), 3);

        assert_eq!(
            s.filtered_cap, 1,
            "filtered_cap is READ-denominated: one rejected read"
        );
        assert_eq!(
            s.filtered_cap, s.reads_over_cap,
            "the two names must count the same population"
        );
        assert_eq!(
            s.filtered_eligible_not_anchor, 3,
            "the two emitted reads each lose one argmax loser (2), and the \
             rejected read contributes eligible_records - 1 == 1. Without that \
             last term this is 2 and the record ledger is short by exactly 1."
        );

        assert_eq!(
            s.filtered_unique, 0,
            "MAPQ 5 clears every filter now; the retired slot must stay at 0"
        );
        assert_eq!(s.filtered_unmapped, 0);
        assert_eq!(s.filtered_supplementary, 0);
        assert_eq!(s.filtered_3p_anchor, 0);
        assert_eq!(s.filtered_coverage, 0);
        assert_eq!(s.filtered_polya_invalid, 0);
        assert_eq!(s.rejected_tie, 0, "100 vs 50 is not a tie");
        assert_eq!(
            s.rejected_usurpation, 0,
            "every candidate here is physically eligible, so no argmax is \
             disqualified"
        );
        assert_eq!(s.rejected_same_gene, 0);
        assert_eq!(s.rejected_cross_gene, 0, "the gap floors are off");
        assert_eq!(s.gene_veto_disabled, 0);
        assert_eq!(s.reads_no_eligible_candidate, 0);

        // The anchors are the two cap winners on tx2 plus the low-MAPQ read on tx1. WHICH two sat
        // reads won is hash order and is deliberately not asserted.
        let mut ids: Vec<&str> = anchors.iter().map(|a| a.read_id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec!["lowq", "sat1", "sat2"]);
        let mut txs: Vec<&str> = anchors.iter().map(|a| a.transcript_id.as_str()).collect();
        txs.sort_unstable();
        assert_eq!(txs, vec!["tx1", "tx2", "tx2"]);
    }

    /// AN EMPTY `tx_to_gene` DOES NOT SILENTLY BECOME "CROSS-GENE". With no map, `gene_id ==
    /// transcript_id` for every candidate, so a naive same-gene check compares each transcript with
    /// itself and can NEVER fire. `gene_veto_disabled` and `same_gene_unknown` are non-zero, and
    /// `rejected_cross_gene` is zero.
    #[test]
    fn test_an_empty_tx_to_gene_emits_but_records_the_unknown_relation() {
        let (anchors, s) = run_extract(
            "vetooff",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record("r", 1, Some(90), 0, 60),
                // A solo read is unaffected: with one candidate there is no pair to classify.
                synth_record("solo", 2, Some(100), 0, 60),
            ],
            &TxToGene::new(),
            params(),
        );

        assert_record_ledger_closes(&s);
        // Universal Mode does not refuse a read for having an unclassifiable rival: it applies the same score/gap criteria as the mapped case and counts the fact that the RELATION was unavailable.
        assert_eq!(
            anchors.len(),
            2,
            "the pair clears the gap, so it emits; the missing gene map is \
             recorded, not used as grounds for refusal"
        );
        assert_eq!(s.gene_veto_disabled, 1);
        assert_eq!(
            s.same_gene_unknown, 1,
            "the read must be recorded in the UNKNOWN class, not to cross_gene"
        );
        assert_eq!(
            s.rejected_cross_gene, 0,
            "with an empty map the gene relation is unknown, and 'unknown' is \
             not 'cross-gene' — this is the whole point of the counter"
        );
        assert_eq!(
            s.rejected_same_gene, 0,
            "nothing failed the gap, and 'unknown' must not be invented into \
             'same gene' either"
        );
        assert_eq!(s.filtered_eligible_not_anchor, 1);

        // And the empty-map fallback for the EMITTED gene_id is unchanged: it is still `gene_id := transcript_id`, which is what the degenerate gene pool warning keys on.
        assert_eq!(anchors[0].gene_id, anchors[0].transcript_id);
    }

    /// ...AND WITH NO MAP **AND** A GAP THAT IS NOT CLEAR, THE READ IS REJECTED AND LANDS IN
    /// `rejected_gap` — the Unknown arm of the gap classification. With a map supplied the relation is
    /// always Same or Cross, so `rejected_gap` can only be non-zero in Universal Mode.
    #[test]
    fn test_an_empty_tx_to_gene_with_an_unmet_gap_is_rejected_as_rejected_gap() {
        let p = AnchorParams {
            gap_abs: 15.0,
            gap_rel: 0.0,
            ..params()
        };
        let (anchors, s) = run_extract(
            "vetooffgap",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record("r", 1, Some(90), 0, 60),
            ],
            &TxToGene::new(),
            p,
        );

        assert_record_ledger_closes(&s);
        assert_eq!(anchors.len(), 0, "10 AS does not clear a floor of 15");
        assert_eq!(s.rejected_gap, 1, "the Unknown arm of the gap block");
        assert_eq!(s.rejected_same_gene, 0, "no map means never Same");
        assert_eq!(s.rejected_cross_gene, 0, "no map means never Cross");
        assert_eq!(s.gene_veto_disabled, 0, "it never cleared the gap");
        assert_eq!(s.filtered_eligible_not_anchor, 2);
    }

    /// A NON-EMPTY `tx_to_gene` THAT DOES NOT COVER THE BAM ERRORS OUT. The empty-map fallback
    /// (`gene_id := transcript_id`) is legitimate and stays; the non-empty case is a different fact:
    /// the TSV and the BAM's `@SQ` header disagree.
    #[test]
    fn test_a_non_empty_gene_map_that_misses_a_transcript_is_an_error() {
        let tsv = tx2gene_tsv("partial", "tx0\tGA\ntx1\tGB\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();
        assert_eq!(map.len(), 2, "the map is NOT empty — that is the point");

        let bam = write_synthetic_bam(
            "partial",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                // tx2 is in the header and absent from the TSV.
                synth_record("r", 2, Some(90), 0, 60),
            ],
        );
        let err = extract_anchors_from_bam(&bam, "partial", &map, params())
            .expect_err("a non-empty map must not silently cover a missing transcript");
        let _ = std::fs::remove_file(&bam);
        let _ = std::fs::remove_file(&tsv);

        let msg = err.to_string();
        assert!(
            msg.contains("tx2"),
            "the error must name the transcript it could not resolve; got: {msg}"
        );
        assert!(
            msg.contains("Refusing"),
            "the error must say it refused rather than fell back; got: {msg}"
        );
    }

    /// THE `reads(...)` BLOCK APPEARS IN `report()` AND SAYS IT IS READS. `report()`'s prefix is
    /// pinned byte-for-byte by a downstream scraper's regex, so new material can only go at the end.
    #[test]
    fn test_report_keeps_the_pinned_prefix_and_appends_the_new_buckets() {
        let agg = AnchorStats {
            samples: 1,
            total_records: 100,
            filtered_unmapped: 12,
            filtered_supplementary: 13,
            filtered_eligible_not_anchor: 10,
            reads_seen: 27,
            anchors_emitted: 7,
            rejected_tie: 9,
            reads_over_cap: 4,
            rejected_usurpation: 6,
            ..Default::default()
        };
        let line = agg.report();

        assert!(
            line.starts_with("samples=1 total=100 filtered(unique=0 no_reflen=0 3p_anchor=0 coverage=0 polya=0 cap=0) passed=0 unaccounted=65 no_refid=0 no_position=0 no_read_id=0"),
            "the pinned prefix changed, which breaks ANCHOR_STATS_RE:\n{line}"
        );
        for token in [
            "unmapped=12",
            "supplementary=13",
            "not_anchor=10",
            "reads(seen=27 anchors=7 tie=9",
            // `over_cap` and `usurpation` belong INSIDE reads(...) and at its END: inserting either
            // into `filtered(...)` would sit inside the region the regex pins. They are DISTINCTIVE
            // (4 and 6), so a transposition between them is visible.
            "no_eligible=0 over_cap=4 usurpation=6)",
        ] {
            assert!(
                line.contains(token),
                "report must carry {token}; got: {line}"
            );
        }
        assert!(
            line.contains("reads("),
            "the read block must be labelled `reads` so its unit is not read as \
             the record ledger's unit; got: {line}"
        );
    }
}
