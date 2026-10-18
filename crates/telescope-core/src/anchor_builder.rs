//! Module: Universal Mode Anchor Pool Builder (`anchor_builder.rs`)
//!
//! D1 Universal Mode Step 3 (per spec v0.2.0-rc2): streams one or more BAM
//! files, applies poly(A) anchor criteria, and writes a unified
//! `telescope.anchor_pool.parquet` ready for the downstream emission model.
//!
//! Anchor criteria (per spec §A of D1 rebuttal, accepted):
//! - 3' TES anchor: `ref_end + tes_slack >= ref_len`
//! - Ambiguity is resolved PER READ (strict argmax over the read's eligible
//!   candidates), NOT by an NH/MAPQ record filter. See `extract_anchors_from_bam`.
//! - Valid poly(A): `pt:i >= 10` (Dorado polyA tag)
//! - Coverage ratio: `alignment_length / ref_len >= min_coverage` (default 0.80)
//! - Per-sample 200-read cap per transcript (prevents deep-sample dominance)
//!
//! 5' TSS anchor is NOT required (5' truncation is normal in DRS; poly(A) lives
//! at the 3' end).
//!
//! # SELECTION IS PER READ, NOT PER RECORD (2026-10-18)
//!
//! The five criteria above are all PER RECORD and none of them asks whether a
//! read is *unambiguous*. A read therefore used to produce one anchor per
//! record, so a read aligning to three transcripts emitted three anchors, and a
//! read whose best two alignment scores were exactly equal — which is what a
//! read sitting in sequence shared by both isoforms looks like — emitted its
//! anchor off an arbitrary tie-break. Anchor selection now runs per read over
//! all of that read's records at once (see [`extract_anchors_from_bam`]).
//!
//! This is forced by the input contract and is not a design choice: the
//! production BAM is `SO:coordinate`, a single read's records sit at line 11
//! and line 23,282,041 of 39,369,625, and no sliding window can span that.
//! Whole-file in-memory grouping is the only option that does not change the
//! input contract (no samtools collate, no `sort -n`).
//!
//! Output schema (matches Module 3 `profile.parquet` schema so it can be
//! consumed directly by emission_model + TEL-EM):
//!   sample_id (utf8), read_id (utf8), gene_id (utf8), transcript_id (utf8),
//!   poly_a_length (int32), posterior_weight (float64 = 1.0 for anchors)

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

    /// Whether the map carries no entries at all.
    ///
    /// This is load-bearing and not a convenience accessor. With an EMPTY map
    /// every transcript is its own gene, so the per-read same-gene veto in
    /// [`extract_anchors_from_bam`] compares each transcript against itself
    /// and can never fire — every intra-gene multi-candidate read would be
    /// filed as cross-gene, which is the exact opposite of the truth. The veto
    /// reads this to route those reads to `same_gene_unknown` and to count
    /// `gene_veto_disabled` instead of pretending it decided.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Per-run knobs for [`build_anchor_pool`].
///
/// GROUPED, not a scalar list: with the two gap parameters added this entry
/// point takes 8 positional arguments, which is over clippy's
/// `too_many_arguments` threshold of 7 and would have added a lint site to a
/// crate whose standing budget is "no NEW sites".
#[derive(Debug, Clone, Copy)]
pub struct AnchorParams {
    /// 3' TES slack in bp: an alignment is 3'-anchored when
    /// `alignment_end + tes_slack >= ref_len`.
    pub tes_slack: usize,
    /// Minimum coverage ratio `alignment_length / ref_len`.
    pub min_coverage: f64,
    /// Per-sample cap on anchors per transcript.
    pub max_reads_per_sample: usize,
    /// Absolute floor on `AS_best - AS_second`.
    ///
    /// ⚠ **THE DEFAULT 0.0 IS NOT A CALIBRATED PRODUCTION VALUE. IT MEANS "NO
    /// GAP REQUIREMENT".** Nothing in this repository measures what this
    /// threshold should be; the owner explicitly declined to ship an
    /// uncalibrated default, so the flag exists, the arithmetic is exact and
    /// tested at the boundary, and the default is the identity. Read
    /// `gap=0` as "this criterion is switched off", never as "zero gap is
    /// correct".
    pub gap_abs: f64,
    /// Floor on `AS_best - AS_second` as a FRACTION of `AS_best`. Same
    /// default caveat as [`AnchorParams::gap_abs`], and same reading: 0.0 is
    /// "off", not "right".
    pub gap_rel: f64,
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
/// novel single-transcript loci — but see [`TxToGene::is_empty`] for what that
/// costs the same-gene veto).
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
/// `distinct gene_id == distinct transcript_id`, and the consequence follows
/// from that alone: every gene holds exactly one transcript's anchors, so no
/// two transcripts ever share fitted poly(A) parameters through the L2 tier.
///
/// Note what the counts do and do not prove. They show the gene -> transcript
/// relation is INJECTIVE; they do not show `gene_id` is literally the
/// `transcript_id` string. A legitimate 1:1 map that renames (`T1 -> G1`) is
/// equally degenerate and equally deserves the warning, so the message must not
/// claim the ids were equal — for that user that would be false. The repo's own
/// `test_detector_is_exactly_equality_of_the_two_distinct_counts` builds
/// `G1`/`T1` and expects a warning, which is the case that makes this a real
/// hazard rather than a theoretical one.
///
/// The thresholds are `build-emission` flags that this function cannot see, so
/// the message must not state an unconditional claim about them. At the
/// DEFAULTS (`tx_min_reads = 10`, `gene_min_reads = 30`) L2 is unreachable —
/// reaching it already requires `n < 10`, and the gate then demands 30 values
/// from a pool of at most 9 — and the warning says exactly that, scoped to the
/// defaults. It does not say "always", because `--gene-min-reads 3` does make
/// L2 fire on this pool. Measured on HC2: the same degenerate anchor pool
/// yields L2=0 at the default threshold and L2=3424 at `--gene-min-reads 3`.
/// That configuration is not a repair, though — the pooled set is the
/// transcript's own reads, so L2 there is L1 at a different threshold, and the
/// warning says so rather than inviting the reader to think a threshold tweak
/// substitutes for a gene map.
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

/// One alignment that survived the record-level filter chain, reduced to the
/// five fields the per-read decision needs.
///
/// FIXED SIZE, 20 bytes. The staging map holds one of these per surviving
/// record — 37.7 M of them on HC2 — so anything variable-length here would be
/// multiplied by 37.7 M.
#[derive(Debug, Clone, Copy)]
struct Staged {
    ref_id: u32,
    start: u32,
    end: u32,
    as_: i32,
    pt: i32,
}

/// One read's staging state. `cands` is in BAM order on entry from the
/// streaming pass; the per-read pass sorts it in place.
struct ReadGroup {
    /// The 128-bit read-name hash this group was filed under (see [`read_key`]).
    ///
    /// It is STORED rather than recomputed in pass 2 because it is the cap's
    /// sort key: which reads survive the per-transcript cap has to be a
    /// deterministic function of the read's own name, and the name lives in the
    /// arena, not in the `HashMap` that pass 1 used. Recomputing would mean
    /// re-hashing 2.8 M names on the floor for no gain — 16 bytes here is
    /// cheaper than the re-hash and it is the same value.
    key: u128,
    /// Byte range of this read's QNAME inside the shared name arena. Keeping
    /// NAMES as bytes in one arena rather than as a `String` per read is the
    /// whole point: a `String` per read was measured at ~2 GB on HC2, and only
    /// the ~600 k reads that actually become anchors ever need one (see
    /// `AnchorRecord.read_id` construction below).
    name_off: u32,
    name_len: u32,
    cands: Vec<Staged>,
}

/// 128-bit grouping key for a read name.
///
/// WHY A HASH AND NOT THE NAME. The name itself as a `HashMap` key was
/// measured at ~1.77 GB of staging cost on HC2, almost all of it one heap
/// allocation per read. A `u128` key is 16 bytes, lives inline in the map, and
/// allocates nothing. Two SipHash-1-3 runs over the name, distinguished by a
/// leading byte, give the two halves; the birthday probability of ANY collision
/// among 5.17 M reads is ~1e-25, and [`read_key_collisions`] turns the residual
/// into a checked fact rather than an assumption.
///
/// The two halves are not independent — they are the same keyed function over
/// two different messages — so the real guarantee is not a probability
/// estimate but the per-record comparison in the staging loop, which fails the
/// run outright if any two distinct QNAMEs ever land on one key.
fn read_key(name: &[u8]) -> u128 {
    let mut lo = DefaultHasher::new();
    name.hash(&mut lo);
    let mut hi = DefaultHasher::new();
    0xA5_u8.hash(&mut hi);
    name.hash(&mut hi);
    (u128::from(hi.finish()) << 64) | u128::from(lo.finish())
}

/// How a read's top two candidates relate under `tx_to_gene`.
enum GeneRelation {
    /// Both transcripts resolve to the same gene.
    Same,
    /// The map is empty, so `gene_id == transcript_id` for both and the
    /// relation is genuinely unknown — NOT the same as `Cross`, which is a
    /// statement the data supports.
    Unknown,
    /// The two transcripts resolve to different genes.
    Cross,
}

/// Resolve a transcript's gene, or fail loudly when a non-empty map does not
/// cover it.
///
/// R6. The EMPTY map falls back to `gene_id := transcript_id`, which is
/// legitimate for novel single-transcript loci and which two existing tests
/// pin. The NON-EMPTY map missing a transcript is a different situation
/// entirely: the map and the BAM's `@SQ` header disagree, and the silent
/// fallback would then decide the same-gene veto on a gene id that was
/// invented on the spot — a veto that never fires reads exactly like a veto
/// that correctly found no shared gene.
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

/// Extract anchors from a single BAM. Returns the emitted anchors plus
/// per-sample stats (counts at each filter step).
///
/// # TWO PASSES, AND WHY TWO
///
/// **Pass 1** streams the file once through `read_record` — the zero-copy form
/// `evidence.rs:398` uses, over ONE reused record buffer, because
/// `bam::io::Records::next()` deep-clones every record. Records that clear the
/// record-level chain are reduced to a 20-byte [`Staged`] and appended to their
/// read's group; the QNAME is copied as BYTES into a shared arena and only
/// turned into a `String` for a read that actually wins an anchor.
///
/// **Pass 2** walks the groups in READ-ID-HASH order — a total order derived
/// from the same 128-bit key that groups them, NOT from `HashMap` iteration
/// order and NOT from BAM order. Cap membership must be a deterministic
/// function of the read's own name, or two runs over the same file would emit
/// two different pools.
///
/// # THE ORDER OF THE CHECKS IS THE SPECIFICATION, not an accident
///
/// unmapped → supplementary → refid → position → QNAME → ref_len (pass 1; the
/// MAPQ gate that used to sit between `supplementary` and `refid` was REMOVED
/// in R139 and `filtered_unique` is now a retired slot that stays 0), then per
/// read: dedup by ref_id keeping max AS → 3' TES, coverage, poly(A) per
/// candidate → strict argmax → gap → same-gene veto → per-transcript cap.
///
/// SECONDARY ALIGNMENTS ARE KEPT (`0x100`). Dropping them would reintroduce
/// exactly the defect this pass exists to fix.
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

    // ===================== PASS 1: per RECORD =============================
    let mut name_arena: Vec<u8> = Vec::new();
    let mut groups: Vec<ReadGroup> = Vec::new();
    let mut group_index: HashMap<u128, u32> = HashMap::new();
    let mut read_key_collisions: u64 = 0;

    let mut record = noodles_bam::Record::default();
    loop {
        let read = match reader.read_record(&mut record) {
            Ok(n) => n,
            Err(_) => {
                // The old `reader.records()` iterator skipped a bad record and
                // asked for the next one; `read_record` cannot be re-entered
                // after a failure without risking a spin on the same truncated
                // stream, so the scan stops here. `parse_errors` stays a
                // DISJOINT population (see `AnchorStats::parse_errors`), so the
                // ledger identity is unaffected either way.
                stats.parse_errors += 1;
                break;
            }
        };
        if read == 0 {
            break;
        }
        stats.total_records += 1;

        // 0x4. Checked BEFORE MAPQ, which is the point of the move: every
        // unmapped record carries MAPQ 0 and used to die at the MAPQ gate
        // counted as "non-unique", so the ledger could not tell a record with
        // no alignment from an ambiguously-placed one. Measured on HC2:
        // 1,654,825 unmapped records were in that bucket.
        if record.flags().is_unmapped() {
            stats.filtered_unmapped += 1;
            continue;
        }
        // 0x800. Same reason, smaller number, and it is a LIVE defect rather
        // than a cosmetic one: 25,077 supplementary records on HC2, of which
        // 33.43% carry MAPQ >= 20 and were therefore entering the anchor pool
        // as if they were complete molecules.
        if record.flags().is_supplementary() {
            stats.filtered_supplementary += 1;
            continue;
        }
        // 0x100 (secondary) is deliberately NOT dropped.

        // ⚠ THERE IS NO MAPQ GATE HERE, AND ITS ABSENCE IS THE POINT (R139).
        //
        // `if mapq < 20 { stats.filtered_unique += 1; continue; }` used to sit
        // here, and it made the entire per-read stage dead code on real data:
        // HC2's secondaries ALL carry MAPQ 0 (measured over all 39,369,625
        // records), so every surviving record was a primary, every read held
        // exactly one candidate, `scratch.len() >= 2` was never true, and
        // strict argmax / gap / same-gene veto never executed even once. Five
        // counters read 0 and — that is what named it — did not add back up to
        // `reads_seen`.
        //
        // `filtered_unique` therefore stays as a RETIRED compat slot at 0. It is
        // still emitted, still inside the twelve-key `bucket_sum`, and still in
        // the frozen `telescope.filter_ledger/v1` schema; removing the key would
        // break a published artifact for no gain. Nothing increments it, which is
        // why a fixture asserts it is 0 rather than merely being absent — a
        // re-added `+= 1` would otherwise be invisible, since no test in
        // `ledger_json.rs` reads its value.
        //
        // MAPQ was never a uniqueness criterion for this pipeline. `--secondary`
        // is PRESERVED (spec F-BAM-Alignment), so MAPQ 0 on a secondary is the
        // aligner reporting that it did not claim the placement, not that the
        // placement does not exist; discarding on that basis discarded 32.5 M
        // of HC2's 39.4 M records and, with them, the ambiguity this whole pass
        // exists to resolve.
        //
        // Position + length on reference.
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

        // NOTE THE POSITION OF THIS CHECK. It used to be LAST, after the
        // per-transcript cap had already been incremented, so that a record
        // with no QNAME consumed a cap slot. Grouping by QNAME makes that
        // impossible to preserve: a record with no name cannot be assigned to
        // a read at all. The counter is unchanged and the ledger still closes;
        // what changes is only which of `filtered_no_read_id` and
        // `filtered_cap` such a record lands in, and
        // `recompute_anchor_filters` in the Dataset A scraper has been updated
        // to match.
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
        // A tag-less AS reads as 0, matching `evidence.rs`'s convention
        // (measured there at 0 of 600,000 records). Two tag-less candidates on
        // one read therefore tie, and a tie yields no anchor — which is the
        // honest answer for a pair the aligner gave no score to distinguish.
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

    // The falsifiable form of "no two anchors share a read_id". One anchor per
    // GROUP is emitted by construction, so a literal duplicate check over the
    // emitted anchors could never fire — the grouping would have hidden the
    // collision before it could produce one. Checking the hash instead, on
    // every record, catches the same defect at the moment it happens, and
    // fails the run rather than silently merging two molecules' anchors.
    assert_eq!(
        read_key_collisions, 0,
        "two distinct QNAMEs hashed to the same 128-bit grouping key, so their \
         records were merged into one read group. The per-read argmax then \
         picked one transcript for two molecules. This is the residual risk of \
         keying by hash instead of by name; see `read_key`."
    );

    // ===================== PASS 2: per READ ===============================
    //
    // IN READ-ID-HASH ORDER, AND THAT IS THE POINT. Pass 1 fully materialised
    // `groups`, so the visiting order is free — and "free" is what makes it
    // worth being explicit. The alternatives are both wrong in different ways:
    // iterating the `HashMap` makes the cap depend on a per-process hash seed,
    // so two runs over the same BAM emit two different pools; iterating `groups`
    // in insertion order makes it depend on BAM layout, so re-sorting the input
    // silently changes which reads a deep transcript keeps.
    //
    // `sort_by_key`, NOT `sort_unstable_by_key`. The keys are asserted unique
    // (see `read_key_collisions`), so an unstable sort would be deterministic
    // TODAY — and the sort would silently stop being deterministic the moment
    // that assert was relaxed or a duplicate key appeared. A stable sort costs
    // one allocation's worth of scratch and removes the dependency.
    let mut order: Vec<u32> = (0..groups.len() as u32).collect();
    order.sort_by_key(|&i| groups[i as usize].key);
    let mut scratch: Vec<(Staged, u64)> = Vec::new();
    for &i in &order {
        let g = &mut groups[i as usize];
        stats.reads_seen += 1;

        // DEDUP BY ref_id, KEEPING MAX AS. `sort_by` is stable, so two records
        // on the same ref_id with the SAME AS keep their BAM order — a choice
        // that is arbitrary but deterministic, and never decisive, because a
        // same-ref_id pair is one transcript by construction and so cannot
        // produce the cross-transcript argmax tie checked below.
        g.cands
            .sort_by(|a, b| a.ref_id.cmp(&b.ref_id).then_with(|| b.as_.cmp(&a.as_)));

        // WALK THE RUNS, applying the physical filters once per surviving
        // transcript and attributing the result to EVERY record collapsed into
        // it (`mult`). That is what keeps the ledger closed in RECORDS: a read
        // with three identical records on one transcript that fails the 3' test
        // has lost three records, not one.
        scratch.clear();
        let mut i = 0;
        while i < g.cands.len() {
            let mut j = i + 1;
            while j < g.cands.len() && g.cands[j].ref_id == g.cands[i].ref_id {
                j += 1;
            }
            let mult = (j - i) as u64;
            let rep = g.cands[i];

            let rl = *ref_len.get(&(rep.ref_id as usize)).unwrap_or(&0);
            if rl == 0 {
                // Unreachable while pass 1 filters it, and kept anyway: a zero
                // here would make the coverage test `NaN < min_coverage`, which
                // is FALSE, so the record would sail through as eligible.
                stats.filtered_no_reflen += mult;
            } else if (rep.end as usize + p.tes_slack) < rl {
                // 3' TES anchor: ref_end + tes_slack >= ref_len.
                stats.filtered_3p_anchor += mult;
            } else {
                let aln_len = if rep.end >= rep.start {
                    (rep.end - rep.start + 1) as f64
                } else {
                    0.0
                };
                if aln_len / (rl as f64) < p.min_coverage {
                    stats.filtered_coverage += mult;
                } else if rep.pt < 10 {
                    stats.filtered_polya_invalid += mult;
                } else {
                    scratch.push((rep, mult));
                }
            }
            i = j;
        }

        if scratch.is_empty() {
            // Every record this read contributed is already in a physical
            // bucket. Nothing to reject at the read level.
            stats.reads_no_eligible_candidate += 1;
            continue;
        }
        // Sort by AS descending. Stable, so a tie keeps the dedup order.
        scratch.sort_by(|a, b| b.0.as_.cmp(&a.0.as_));
        let eligible_records: u64 = scratch.iter().map(|(_, m)| *m).sum();

        // STRICT ARGMAX. An exact tie on the top AS yields NO anchor.
        //
        // THE TIE RATE IS **POPULATION-DEPENDENT**, AND THE POPULATION MUST BE
        // NAMED. This comment used to read "56.0% of reads with two or more
        // PHYSICALLY-ELIGIBLE candidates have their top two AS values EXACTLY
        // equal", which is false: 56.0% (and the reproduced 56.79% below) is
        // the PRE-physical-filter population. Read the right number off the
        // wrong population and the tie veto looks like it fires on most reads
        // when in fact it fires on roughly one in eight.
        //
        // Same 4,000,000-record prefix of HC2.sorted.bam, one variable — whether
        // tes35 / cov0.80 / pt10 is applied:
        //
        //   NO physical filter         multi 558,471   tie 317,145   = 56.79%
        //   WITH tes35/cov0.80/pt10    multi 238,417   tie  26,590   = 11.15%
        //
        // and on the FULL BAM with the MAPQ gate removed (39,369,625 records,
        // 17,782,042 of them carrying an `AS`):
        //
        //   multi 2,479,569            tie 314,613      = 12.69%
        //
        // Reproduce each row with the script that produced it — they are
        // DIFFERENT populations and one script does not cover both:
        //
        //   # the two prefix rows above
        //   /datf/hanxi/software/miniconda3/envs/DRS/bin/python \
        //     workspace/validation/benchmark/anchor_qualification/measure_tie_rate.py
        //   # the full-BAM row
        //   /datf/hanxi/software/miniconda3/envs/DRS/bin/python \
        //     workspace/validation/benchmark/anchor_qualification/measure_nogate_candidates.py
        //
        // ⚠ `measure_nogate_candidates.py` carried a LOWERCASE `has_tag("as")`
        // until 2026-10-18. pysam's `has_tag` is case-sensitive and minimap2
        // writes `AS`, so the lookup missed every record, every candidate scored
        // 0, and the script reported tie == multi == 100% — it could not
        // reproduce its own saved output. Fixed, and an `assert n_as > 0` now
        // stops a parse failure from printing prose that describes it. This is
        // the SAME defect class as the fake 100.00% tie rate, and it is why the
        // script asserts rather than defaults.
        //
        // THE TOOL AGREES WITH THE NUMERATOR. `rejected_tie` on the full HC2 BAM
        // is **314,613**, the number above exactly. The tool has no "multi"
        // counter, so the DENOMINATOR is the Python population's alone; it also
        // applies the physical filters per RECORD before the same-`ref_id` dedup
        // while this function dedups first and filters the max-AS
        // representative, so the two "multi" populations are not the same set.
        //
        // MECHANISM, not coincidence: a tie means the read sits in sequence its
        // top candidates SHARE, which is overwhelmingly INTERNAL, and the 3' TES
        // test requires the read to reach the transcript's 3' end. The physical
        // filters therefore remove precisely the reads that tie.
        //
        // Vetoing ties is still the right behaviour on the surviving population:
        // such a read carries no isoform information, and a tie-break would
        // invent an isoform assignment out of nothing. What the veto no longer
        // looks like is "the common case" — it is not, and saying so would be
        // the same category of error as the number.
        if scratch.len() >= 2 && scratch[0].0.as_ == scratch[1].0.as_ {
            stats.rejected_tie += 1;
            stats.filtered_eligible_not_anchor += eligible_records;
            continue;
        }

        let best = scratch[0].0;
        if scratch.len() >= 2 {
            let second = scratch[1].0;
            let gap = (best.as_ - second.as_) as f64;

            // GAP. Both floors must hold. With the shipped defaults
            // (0.0 / 0.0) this is the identity — see `AnchorParams::gap_abs`
            // for why that is "switched off" and not "calibrated".
            if gap < p.gap_abs || gap < p.gap_rel * (best.as_ as f64) {
                let g_best = resolve_gene(tx_name(&ref_seqs, best.ref_id), tx_to_gene)?;
                let g_second = resolve_gene(tx_name(&ref_seqs, second.ref_id), tx_to_gene)?;
                match relation(&g_best, &g_second, tx_to_gene) {
                    GeneRelation::Same => stats.rejected_same_gene += 1,
                    // A cross-gene pair rejected on the gap is still a
                    // cross-gene pair; without this the counter would have no
                    // producer at all and would report 0 forever.
                    GeneRelation::Cross => stats.rejected_cross_gene += 1,
                    // Gene relation unknown: the map is empty, so there is
                    // nothing to attribute the rejection to but the gap.
                    GeneRelation::Unknown => stats.rejected_gap += 1,
                }
                stats.filtered_eligible_not_anchor += eligible_records;
                continue;
            }

            // SAME-GENE VETO. With an EMPTY map `gene_id == transcript_id` for
            // every candidate, so this comparison can NEVER fire and every
            // intra-gene multi-candidate read would be filed as cross-gene —
            // the precise inverse of the truth. That state is detected and
            // counted, not worked around: `gene_veto_disabled` says the veto
            // was unavailable, and the read is routed to `same_gene_unknown`
            // rather than to a cross-gene class it does not belong to.
            match relation(
                &resolve_gene(tx_name(&ref_seqs, best.ref_id), tx_to_gene)?,
                &resolve_gene(tx_name(&ref_seqs, second.ref_id), tx_to_gene)?,
                tx_to_gene,
            ) {
                GeneRelation::Same => {
                    stats.rejected_same_gene += 1;
                    stats.filtered_eligible_not_anchor += eligible_records;
                    continue;
                }
                GeneRelation::Cross => {}
                GeneRelation::Unknown => {
                    stats.gene_veto_disabled += 1;
                    stats.same_gene_unknown += 1;
                    stats.filtered_eligible_not_anchor += eligible_records;
                    continue;
                }
            }
        }

        // EMIT. Exactly one anchor per read: the argmax candidate.
        let transcript_id = tx_name(&ref_seqs, best.ref_id).to_string();
        let gene_id = resolve_gene(&transcript_id, tx_to_gene)?;

        let count = per_tx_count.entry(transcript_id.clone()).or_insert(0);
        if *count >= p.max_reads_per_sample {
            // THE CAP REJECTS A READ, AND A READ IS `eligible_records` RECORDS.
            //
            // `filtered_cap += 1` alone is a record-denominated bucket absorbing
            // a read-denominated event, and it is only exact when
            // `eligible_records == 1`. It was, for as long as the MAPQ gate made
            // every surviving record a primary — one record per read — which is
            // why the leak sat there unnoticed with `deficit 0` on HC2. It is
            // not now: a read reaching here can carry the argmax record plus
            // every argmax loser plus every duplicate a same-`ref_id` dedup
            // collapsed, and all of them have now left the stream.
            //
            // The split is deliberately ASYMMETRIC, which is the other half of
            // the fix:
            //   * `filtered_cap` stays READ-denominated — it is the counterpart of
            //     `reads_over_cap`, and the two must be equal, so it counts
            //     rejected READS. Making it absorb records would double it.
            //   * `filtered_eligible_not_anchor` absorbs the remaining
            //     `eligible_records - 1` RECORDS, exactly as the tie / gap /
            //     veto branches absorb `eligible_records` and the emit branch
            //     absorbs `eligible_records - 1`.
            //
            // `eligible_records >= 1` always: this code is only reachable with a
            // non-empty `scratch`, and `eligible_records` is that slice's sum.
            stats.filtered_cap += 1;
            stats.filtered_eligible_not_anchor += eligible_records - 1;
            stats.reads_over_cap += 1;
            continue;
        }
        *count += 1;

        // THE ONE `String` PER READ THAT THIS DESIGN BUYS. Everything above
        // worked on bytes; this is where the name becomes owned text, and it
        // happens once per read that actually reaches the pool.
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
        // Every OTHER eligible record of this read — the argmax losers and the
        // duplicates the dedup collapsed into the winner. One record became
        // `passed`, so the remainder is exactly `eligible_records - 1`.
        stats.filtered_eligible_not_anchor += eligible_records - 1;
    }

    // `filtered_cap` is read-denominated and `reads_over_cap` is the same
    // population written down twice under the two names the two denominations
    // need. They are incremented on one line pair and nowhere else, so this
    // cannot fire today; it is here because a future edit that adds a second
    // `filtered_cap += 1` — a new terminal branch, a diagnostic re-bucketing —
    // would break the read identity in the READ direction, and `unaccounted()`
    // cannot see that at all because it is denominated in records.
    //
    // Dev-time tripwire only, like the `debug_assert!` in `unaccounted()`:
    // `ci/gates.sh` runs `--release`, which compiles it out. The load-bearing
    // version is
    // `test_a_multi_record_read_over_the_cap_absorbs_every_record_it_contributed`.
    debug_assert_eq!(
        stats.filtered_cap, stats.reads_over_cap,
        "filtered_cap counts rejected READS and must stay equal to reads_over_cap; \
         {} vs {} means one of them was incremented without the other",
        stats.filtered_cap, stats.reads_over_cap
    );

    Ok((anchors, stats))
}

/// `ref_seqs[ref_id]`, or `""` for an id the header does not name.
fn tx_name(ref_seqs: &[String], ref_id: u32) -> &str {
    ref_seqs.get(ref_id as usize).map_or("", String::as_str)
}

/// Classify two resolved gene ids.
///
/// `tx_to_gene.is_empty()` is consulted rather than inferring from the ids,
/// because with an empty map `gene_id == transcript_id` and "the same id" then
/// means "the same transcript", not "the same gene" — a distinction this whole
/// function exists to keep.
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

/// Per-sample stats for one BAM input.
#[derive(Debug, Default, Clone)]
pub struct SampleStats {
    pub total_records: u64,
    pub parse_errors: u64,
    /// MAPQ < 20. NOTE this is NOT "non-unique" -- it is "low MAPQ".
    ///
    /// The docstring that used to sit here (and which is still true of the
    /// MEASUREMENT) recorded that on HC2 this bucket swallowed 38,163,664
    /// records, of which 1,654,825 were unmapped and 25,077 supplementary --
    /// every one of them MAPQ 0 -- and concluded that unmapped records were
    /// counted here rather than in a bucket of their own, "splitting the two is
    /// deferred". It is no longer deferred: `0x4` and `0x800` are now tested
    /// FIRST and have their own buckets, because a record with no alignment and
    /// a record ambiguously placed are different facts and the ledger said they
    /// were the same one.
    ///
    /// Reproduce the measurement (HC2, 2026-10-13):
    ///
    /// ```bash
    /// B=/datg/hanxi/project/Telescope/aidrs_pipeline/trans_align/HC2/HC2.sorted.bam
    /// samtools flagstat "$B" | grep -E 'in total|mapped'      # 39369625 / 37714800
    /// samtools view "$B" | awk '$5 < 20' | wc -l              # 38163664
    /// samtools view -f 4 "$B" | awk '{print $5}' | sort -u    # 0  (every unmapped record)
    /// samtools view -f 4 "$B" | awk '$5 >= 20' | wc -l        # 0
    /// ```
    pub filtered_unique: u64,
    /// SAM `0x4`, unmapped. New bucket (2026-10-18). A record with no
    /// alignment has no transcript, no 3' end and no poly(A) to anchor on; the
    /// MAPQ gate used to absorb it because every unmapped record also carries
    /// MAPQ 0, which made the ledger report an unplaced record as an
    /// ambiguously-placed one.
    pub filtered_unmapped: u64,
    /// SAM `0x800`, supplementary alignment. New bucket (2026-10-18).
    ///
    /// THIS ONE CHANGED BEHAVIOUR, NOT JUST BOOKKEEPING. A supplementary
    /// record is one segment of a split molecule, so its poly(A) and its 3' end
    /// are not the molecule's. They were entering the anchor pool whenever they
    /// carried MAPQ >= 20, and 33.43% of HC2's 25,077 supplementary records do.
    /// `evidence.rs` has always dropped them for the same reason.
    pub filtered_supplementary: u64,
    /// Dropped because `reference_sequence_id()` yielded `None`/Err AFTER
    /// passing the MAPQ gate. Added 2026-10-13: this path used to `continue`
    /// with no counter at all, so records leaving here were invisible.
    pub filtered_no_refid: u64,
    /// Same, for `alignment_start()` / `alignment_end()`. Two call sites, one
    /// bucket: a record can reach only one of them.
    pub filtered_no_position: u64,
    /// Same, for `record.name()` being `None`. A record with no name cannot be
    /// written to the pool, so this is not a discard we could avoid.
    pub filtered_no_read_id: u64,
    pub filtered_no_reflen: u64,
    pub filtered_3p_anchor: u64,
    pub filtered_coverage: u64,
    pub filtered_polya_invalid: u64,
    /// Physically eligible records that were NOT emitted as an anchor. New
    /// bucket (2026-10-18), and it is the bucket that makes the per-read
    /// decision closeable in RECORDS.
    ///
    /// It absorbs three distinct populations, which is why the read-level
    /// counters exist alongside it: the argmax losers of a read that did win,
    /// the duplicates a same-`ref_id` dedup collapsed, and every record of a
    /// read rejected by the tie / gap / same-gene rules.
    pub filtered_eligible_not_anchor: u64,
    /// Rejected by the per-transcript cap. **READ-DENOMINATED**, despite living
    /// among the record buckets: the cap rejects a read, not a record.
    ///
    /// The records that read carried but did not contribute — `eligible_records
    /// - 1` of them — go to `filtered_eligible_not_anchor`, so the two together
    /// absorb every record the read brought. Always equal to
    /// [`SampleStats::reads_over_cap`].
    pub filtered_cap: u64,
    /// Anchors emitted. This is the ledger's terminal bucket: `passed`
    /// counts RECORDS that became anchors, and since selection is per read it
    /// also counts how many reads reached the pool.
    pub passed: u64,

    // ------------------------------------------------------------------
    // READ-LEVEL COUNTERS. A SEPARATE DENOMINATION.
    //
    // Everything above counts RECORDS. Everything below counts READS. They are
    // never added together and none of these fields appears in
    // `AnchorStats::unaccounted()` — mixing the two is how a number stops
    // meaning anything. The read-level identity, when the list is closed, is
    //
    //   reads_seen == anchors_emitted
    //               + rejected_tie + rejected_gap + rejected_same_gene
    //               + rejected_cross_gene + gene_veto_disabled
    //               + reads_no_eligible_candidate + reads_over_cap
    //
    // which `test_anchors_emitted_never_exceed_reads_seen_and_the_read_block_closes`
    // asserts. `same_gene_unknown` is deliberately OUTSIDE that identity: see
    // its own doc comment. `reads_over_cap` is INSIDE it, and `filtered_cap` is
    // NOT — those two name the same population in the two denominations, and
    // only the read-denominated one belongs in a read identity.
    // ------------------------------------------------------------------
    /// Reads that reached the per-read decision stage — i.e. that contributed
    /// at least one record clearing the record-level chain.
    ///
    /// NOT "reads in the file". A read whose every record died at `0x4`,
    /// MAPQ, tid or POS never forms a group and is not counted here; its
    /// records are already accounted in the record buckets, and counting the
    /// read as well would double it.
    pub reads_seen: u64,
    /// Reads that produced an anchor. Equal to `passed`.
    pub anchors_emitted: u64,
    /// Reads whose top two eligible candidates had EXACTLY equal AS, so the
    /// strict argmax had no winner.
    pub rejected_tie: u64,
    /// Reads rejected because `AS_best - AS_second` was below the gap floors,
    /// AND whose gene relation could not be classified (empty `tx_to_gene`).
    ///
    /// Scoped to the unclassifiable case on purpose: when the gene relation IS
    /// known, a gap rejection is attributed to `rejected_same_gene` or
    /// `rejected_cross_gene` instead, because "these two are the same gene and
    /// the gap was too small" is the more specific fact and the one a reader
    /// needs. This counter keeps a producer only in the state where no more
    /// specific class exists.
    pub rejected_gap: u64,
    /// Reads whose top two eligible candidates share a gene, and which were
    /// therefore not eligible to produce an anchor — either the gap rejected
    /// them first, or the same-gene veto fired.
    pub rejected_same_gene: u64,
    /// Reads whose top two eligible candidates are in DIFFERENT genes and which
    /// the gap still rejected.
    ///
    /// A cross-gene pair that clears the gap becomes an anchor and is counted
    /// by `anchors_emitted`, so this counter exists to keep the cross-gene
    /// rejection visible. Without it the cross-gene class would have no
    /// producer at all and would print 0 forever.
    pub rejected_cross_gene: u64,
    /// Reads whose same-gene status could not be evaluated because
    /// `tx_to_gene` was empty, and which therefore never reached a
    /// same-gene/cross-gene verdict.
    ///
    /// With an empty map `gene_id == transcript_id`, so the veto compares each
    /// transcript with itself and can never fire; without this counter that
    /// reads as a veto that correctly found nothing.
    pub gene_veto_disabled: u64,
    /// Reads ROUTED to the "unknown" class rather than to `cross_gene`. Equal
    /// to `gene_veto_disabled` in this implementation, and the two names are
    /// kept because they answer different questions: this one says where the
    /// read WENT, the other says why the decision was unavailable.
    /// `test_empty_tx_to_gene_routes_to_the_unknown_class` pins the equality.
    pub same_gene_unknown: u64,
    /// Reads that reached the per-read stage but had no physically eligible
    /// candidate at all. Without this counter the read-level identity would be
    /// short by exactly these reads, and short by a number that grows with how
    /// aggressive the physical filters are.
    pub reads_no_eligible_candidate: u64,
    /// Reads whose argmax transcript was already at `max_reads_per_sample`, so
    /// the per-transcript cap rejected them (2026-10-18).
    ///
    /// Added because the read identity had a hole with no counter in it: the cap
    /// is a READ-level rejection, and before this field existed the gap showed up
    /// as `reads_seen` failing to equal the sum of the seven read destinations —
    /// a shortfall of exactly `filtered_cap`, which is what named the defect on
    /// HC2. The `filtered_cap` value it equals is the record-denominated twin.
    ///
    /// WHICH READS SURVIVE IS A DETERMINISTIC READ-ID-HASH RULE, not
    /// coordinate-sorted BAM order: see the pass-2 ordering comment. The COUNT
    /// is order-invariant either way — it is `max(0, n_reads_argmaxing_on_tx -
    /// cap)` — so only the identity of the survivors depends on the rule, not
    /// this number.
    pub reads_over_cap: u64,
}

/// Aggregated stats across all input BAMs.
#[derive(Debug, Default, Clone)]
pub struct AnchorStats {
    pub samples: usize,
    pub total_records: u64,
    /// Records the BAM decoder could not read at all, summed over samples.
    ///
    /// A DISJOINT POPULATION, deliberately excluded from `unaccounted()`:
    /// it is incremented BEFORE `total_records`, so folding it into the
    /// identity's right-hand side would make the deficit read -1 for every
    /// parse failure -- a number that looks like a tunable bias rather than a
    /// bookkeeping mistake. Round 65 decision 2.
    ///
    /// Added 2026-10-13 (Ticket 2C) because the frozen ledger schema requires
    /// it and `SampleStats` already counted it per sample; without this field
    /// the cohort-level ledger was blind to a population that leaves the
    /// stream before the LHS is even incremented.
    pub parse_errors: u64,
    pub filtered_unique: u64,
    /// See [`SampleStats::filtered_unmapped`].
    pub filtered_unmapped: u64,
    /// See [`SampleStats::filtered_supplementary`].
    pub filtered_supplementary: u64,
    /// See [`SampleStats::filtered_no_refid`].
    pub filtered_no_refid: u64,
    /// See [`SampleStats::filtered_no_position`].
    pub filtered_no_position: u64,
    /// See [`SampleStats::filtered_no_read_id`].
    pub filtered_no_read_id: u64,
    pub filtered_no_reflen: u64,
    pub filtered_3p_anchor: u64,
    pub filtered_coverage: u64,
    pub filtered_polya_invalid: u64,
    /// See [`SampleStats::filtered_eligible_not_anchor`].
    pub filtered_eligible_not_anchor: u64,
    pub filtered_cap: u64,
    pub passed: u64,

    // --- READ-LEVEL. Summed by `add`, but NEVER folded into `unaccounted()`;
    // see the block comment on [`SampleStats::reads_seen`]. ------------------------
    pub reads_seen: u64,
    /// See [`SampleStats::anchors_emitted`].
    pub anchors_emitted: u64,
    /// See [`SampleStats::rejected_tie`].
    pub rejected_tie: u64,
    /// See [`SampleStats::rejected_gap`].
    pub rejected_gap: u64,
    /// See [`SampleStats::rejected_same_gene`].
    pub rejected_same_gene: u64,
    /// See [`SampleStats::rejected_cross_gene`].
    pub rejected_cross_gene: u64,
    /// See [`SampleStats::gene_veto_disabled`].
    pub gene_veto_disabled: u64,
    /// See [`SampleStats::same_gene_unknown`].
    pub same_gene_unknown: u64,
    /// See [`SampleStats::reads_no_eligible_candidate`].
    pub reads_no_eligible_candidate: u64,
    /// See [`SampleStats::reads_over_cap`].
    pub reads_over_cap: u64,

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
        self.rejected_gap += s.rejected_gap;
        self.rejected_same_gene += s.rejected_same_gene;
        self.rejected_cross_gene += s.rejected_cross_gene;
        self.gene_veto_disabled += s.gene_veto_disabled;
        self.same_gene_unknown += s.same_gene_unknown;
        self.reads_no_eligible_candidate += s.reads_no_eligible_candidate;
        self.reads_over_cap += s.reads_over_cap;
    }

    /// Render the stats line.
    ///
    /// FIELD ORDER IS A COMPATIBILITY CONSTRAINT, DO NOT REORDER.
    /// `workspace/validation/benchmark/dseta_step0_5/run_dseta_step0_5.py`
    /// (plus five archived copies) scrapes this line with a regex that pins
    /// BOTH `filtered(unique=N no_reflen=M ... cap=K)` AND `passed=P`
    /// immediately after that closing paren, then computes its own
    /// `deficit = total - (filtered + passed)`. Any new field inserted inside
    /// the group OR between `)` and `passed=` makes `must()` fail and `die()`.
    ///
    /// ⇒ the new counters live after `passed=`. Verified against the real
    /// regex, not reasoned about: an earlier attempt put them after `cap=`
    /// and still failed to match.
    ///
    /// ⚠ THAT SCRAPER IS NOW STALE, AND IT FAILS LOUDLY IN A CONFUSING WAY.
    /// It recomputes `deficit` over the OLD bucket set. If any of
    /// `no_refid` / `no_position` / `no_read_id` is non-zero it matches the
    /// line, then `die()`s claiming "a record left the stream through an
    /// unaccounted path" about a ledger that is in fact balanced. It must be
    /// updated to read `unaccounted=` before Ticket 2C relies on it.
    ///
    /// This was NOT the only place the identity was ever evaluated, and this
    /// is not the first version of it: see `em.rs`'s `deficit()` /
    /// `terminal_sum()`, which enforces the same rule on the E-step after the
    /// `ΔAS` prune reached no bucket on 2026-10-06.
    ///
    /// What actually kept the Dataset A scraper green was simpler than a
    /// bookkeeping accident: on the BAMs this project actually has, those four
    /// paths do not fire. That is a statement about PRODUCTION DATA, not about
    /// what is constructible -- the two came apart on 2026-10-13, when three of
    /// the four were shown to be reachable through ordinary fixtures, and one
    /// (`filtered_no_reflen`) still has no attempt against it. See the
    /// CORRECTION block in this file's test module.
    ///
    /// The identity was observably holding on this data. What it was NOT was
    /// *guaranteed* -- and, before Ticket 2B, no test asserted it.
    ///
    /// ### 2026-10-18: three more buckets, and a whole second denomination
    ///
    /// The `filtered(...)` group and everything up to `no_read_id=` are
    /// **UNCHANGED, BYTE FOR BYTE**, because `ANCHOR_STATS_RE` in
    /// `run_dseta_step0_5.py` pins that prefix. The three new record buckets
    /// (`unmapped`, `supplementary`, `not_anchor`) are appended AFTER
    /// `no_read_id=` for the same reason the Ticket-2A buckets were: the regex
    /// is `re.search`, not `re.fullmatch`, so appending cannot break it, while
    /// inserting inside the pinned region would.
    ///
    /// The `reads(...)` block after them is a DIFFERENT UNIT and says so in
    /// its own name. It is read-denominated and takes no part in
    /// `unaccounted()`; putting read counts inside the record ledger is how a
    /// deficit starts reading as a bias.
    ///
    /// ### 2026-10-18 (R139): `over_cap=` inside the reads block, for the
    /// same append-only reason
    ///
    /// `over_cap` is the read-denominated half of the cap. Without it the read
    /// block could not close on any run where the cap actually bit, and the
    /// shortfall was indistinguishable from a bug in the veto counters — which
    /// is precisely how the dead MAPQ gate stayed hidden. It goes at the END of
    /// `reads(...)`, never inside `filtered(...)`: `cap=` there is the record-
    /// bucketed twin and the two are different numbers only because the branch
    /// splits them (`eligible_records - 1` records go to `not_anchor`).
    pub fn report(&self) -> String {
        format!(
            "samples={} total={} filtered(unique={} no_reflen={} 3p_anchor={} coverage={} \
             polya={} cap={}) passed={} unaccounted={} no_refid={} no_position={} no_read_id={} \
             unmapped={} supplementary={} not_anchor={} \
             reads(seen={} anchors={} tie={} gap={} same_gene={} cross_gene={} \
             veto_disabled={} same_gene_unknown={} no_eligible={} over_cap={})",
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
        )
    }

    /// `total_records` minus every bucket that is supposed to absorb a record.
    ///
    /// This is the number that must be 0 for the ledger to be conservative.
    /// Ticket 2B added the test that enforces it (`249675a` widened the
    /// covered paths to 9/10); the `debug_assert!` below is a dev-time
    /// tripwire for the one direction `saturating_sub` cannot report, and it
    /// compiles out under `--release`, which is how `ci/gates.sh` runs.
    ///
    /// `parse_errors` is deliberately NOT in this sum. It IS aggregated onto
    /// this struct now (Ticket 2C, for the frozen ledger schema), so a reader
    /// meeting the field in `filter_ledger.json` will not conclude it was
    /// never counted -- it was counted and then deliberately excluded, because
    /// it is incremented BEFORE `total_records` and is therefore a disjoint
    /// population.
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

        // Over-count (bucket_sum > total_records) is the ONE direction the
        // `saturating_sub` chain below cannot report: it clamps to 0 and reads
        // as "balanced". The under-count direction is surfaced correctly by
        // the returned value itself, so assert on this one only.
        //
        // ⚠ NOT THE GATE. `ci/gates.sh` runs `cargo test --release`, and the
        // release profile does not set `debug-assertions`, so this compiles out
        // there. Ticket 2B's test must carry its own `assert_eq!` in the test
        // body; this line is a dev-time tripwire, nothing more.
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

    /// The knobs every pre-existing fixture was written against: `tes_slack`
    /// 35, `min_coverage` 0.80, cap 200, and BOTH GAP FLOORS AT 0.0.
    ///
    /// The 0.0s are the shipped defaults and they mean "criterion off", not
    /// "correct value" — see [`AnchorParams::gap_abs`]. Spelling them out here
    /// rather than inheriting a `Default` impl is deliberate: a fixture that
    /// read its thresholds from somewhere silent could keep passing while the
    /// thresholds changed underneath it.
    fn params() -> AnchorParams {
        AnchorParams {
            tes_slack: 35,
            min_coverage: 0.80,
            max_reads_per_sample: 200,
            gap_abs: 0.0,
            gap_rel: 0.0,
        }
    }

    /// `params()` with the two gap floors opened up to a specific value.
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
        // Scoped, not absolute. `--gene-min-reads 3` makes L2 fire on this very
        // pool (measured: L2=3424 on HC2 vs L2=0 at the default), so a message
        // asserting unconditional unreachability would be false.
        assert!(
            !msg.contains("STRUCTURALLY UNREACHABLE"),
            "must not claim L2 is unconditionally unreachable; the thresholds \
             are build-emission flags and can make it fire; got: {msg}"
        );
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

    // === Ticket 2B: the conservation invariant, made enforced ============
    //
    // Before this test the identity `total_records == sum(filtered_*) + passed`
    // held only ACCIDENTALLY, on data that happened not to trigger four of the
    // drop paths. Measured on real HC2 (39,369,625 records) it closed exactly,
    // with all three new buckets at 0 -- but nothing enforced it. The gap was
    // auditability, not a wrong number.
    //
    // === CORRECTION 2026-10-13 (Ticket 2B-Fix) ============================
    //
    // The paragraph below used to read "Six of the ten terminal paths are
    // reachable from a synthetic BAM. The other four are NOT." THAT WAS WRONG,
    // TWICE, and both errors are worth keeping because they are the same error.
    //
    //   ERROR 1. The `bam::io::Writer` error I cited --
    //   `InvalidReferenceSequenceId(MissingEntry { actual: 99, expected: 3 })`
    //   -- is the `Some(id)`-OUT-OF-RANGE branch. Having NO tid is a separate,
    //   supported encoding (tid = -1), and it is what `RNAME=*` produces.
    //
    //   ERROR 2. Having corrected error 1 I then wrote that
    //   `filtered_no_position` was still unreachable, reasoning that "POS is a
    //   fixed-width mandatory BAM field, so 'absent' has no encoding". That is
    //   wrong too. The constraint is on the RUST CONSTRUCTOR --
    //   `Position::new(0)` is invalid because positions are 1-based -- not on
    //   the encoding. An unset field is written as 0, and 0 is precisely what
    //   `alignment_start()` cannot turn into a `Position`.
    //
    //   In both cases I read a real, specific library fact and generalised it
    //   one step past what it said. The error message described a different
    //   branch; the invalid constructor described a different layer. Specific
    //   is not the same as correct.
    //
    // REACHABILITY IS NOW 9/10, enforced by
    // `test_the_three_observable_drop_paths_really_execute` below.
    //
    // WHAT THIS FIXTURE COVERS. Six of the ten terminal paths. The other four:
    //
    //   filtered_no_refid     -- REACHABLE, covered by the 9/10 test
    //                             (`synth_record_no_refid`).
    //   filtered_no_position  -- REACHABLE, covered by the 9/10 test
    //                             (`synth_record_no_position`).
    //   filtered_no_read_id   -- REACHABLE, covered by the 9/10 test
    //                             (`synth_record_no_read_id`).
    //   filtered_no_reflen    -- NOT TESTED. No attempt was made. `ref_len` is
    //                             built from the header, whose lengths are
    //                             `NonZero`; that is a statement about a
    //                             constructor, not a failed attempt, and it is
    //                             labelled as one.
    //
    // The remaining one is covered statically by
    // `workspace/comms/2026-10-13-round68-closeout/pkg/scripts/audit_record_conservation.py`,
    // which does not depend on being able to construct a record. Neither tool is
    // in `gates.sh` today; that is a deliberate standing decision, not an
    // oversight.

    /// Σ of every bucket that is supposed to absorb a record.
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
            synth_record("ok0", 0, None, 0, 60), // -> passed
            synth_record("ok1", 1, None, 0, 60), // -> passed
            // MAPQ 10. There is NO MAPQ gate any more (R139), so this used to
            // land in `filtered_unique` and now clears every filter and becomes
            // a third anchor. It is kept, rather than deleted, precisely so the
            // retirement of that bucket is witnessed by this fixture instead of
            // asserted on an input where it was never in play.
            synth_record("lowq", 0, None, 0, 10), // -> passed (was filtered_unique)
            synth_record_with_gap("gap3p", 1, 100, true, None, 0, 60), // 3' too short
            synth_record_with_gap("shortcov", 1, 500, false, None, 0, 60), // coverage
            nopt,                                 // no poly(A)
        ];
        // 205 records on one transcript with the cap at 200 -> 5 over cap.
        //
        // ⚠ ALL 205 ARE SINGLE-RECORD READS, one QNAME each, which is exactly
        // why the cap branch's record leak was invisible here for so long:
        // `eligible_records == 1` for every one of them, so
        // `filtered_cap += 1` absorbed the whole read and the missing
        // `eligible_records - 1` term was zero. The multi-record counterpart is
        // `test_a_multi_record_read_over_the_cap_absorbs_every_record_it_contributed`.
        // Do not "tidy" this into a multi-record fixture: the arithmetic pinned
        // above (203 passed / 5 over cap) is easier to read here than there.
        for i in 0..205 {
            recs.push(synth_record(&format!("cap{i}"), 2, None, 0, 60));
        }

        let bam = write_synthetic_bam("conserve", recs);
        let (anchors, s) =
            extract_anchors_from_bam(&bam, "conserve", &TxToGene::new(), params()).unwrap();

        // --- THE INVARIANT, FIRST ------------------------------------------
        // Asserted before the per-counter expectations on purpose. The
        // invariant holds for ANY input as long as every drop is counted, so a
        // fixture drift keeps it green and is caught by the per-counter
        // asserts below; an uncounted drop breaks it and must be reported as
        // the conservation failure it is, not as "the fixture lost a record".
        // With the order reversed, measured 2026-10-13, the fixture assertion
        // fired first every time and the invariant was never the thing that
        // reported the failure -- which made the load-bearing claim untestable.
        assert_eq!(
            s.total_records,
            counted_buckets(&s) + s.passed,
            "a record left the stream through an uncounted path: \
             total_records={} counted={} passed={}",
            s.total_records,
            counted_buckets(&s),
            s.passed
        );
        // The terminal really is `passed`.
        assert_eq!(
            anchors.len() as u64,
            s.passed,
            "passed counts records that never reached the pool"
        );
        // parse_errors is incremented BEFORE total_records, so it is a disjoint
        // population and must NOT appear in the identity. Asserted so a future
        // edit that adds it to the sum is caught rather than silently accepted.
        assert_eq!(s.parse_errors, 0);

        // --- the fixture really exercises what it claims to ----------------
        // Without these the test could pass on a fixture that filters nothing,
        // which is the shape of vacuity this project keeps shipping.
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
        // The four paths this fixture cannot reach are pinned at zero, so a
        // future change that makes one fire shows up here as a count change
        // rather than as a silently different total.
        assert_eq!(s.filtered_no_refid, 0);
        assert_eq!(s.filtered_no_position, 0);
        assert_eq!(s.filtered_no_read_id, 0);
        assert_eq!(s.filtered_no_reflen, 0);

        let _ = std::fs::remove_file(&bam);
    }

    /// Reachability is 9/10, and this is the test that says so.
    ///
    /// Ticket 2B asserted that FOUR of the ten terminal paths could not be
    /// reached by any fixture, on the strength of `bam::io::Writer` raising
    /// `InvalidReferenceSequenceId`. That error is the `Some(id)`-out-of-range
    /// branch; having NO tid is a different, supported encoding. Ticket 2B-Fix
    /// then recorded `filtered_no_position` as still unreachable, reasoning
    /// that POS has no encoding for "absent". **That reasoning was also wrong**
    /// — see `synth_record_no_position` for why, and for the fix. Three of the
    /// four execute; the test below exercises all three in one composition:
    ///
    /// ```text
    /// total=6  no_refid=1  no_position=1  no_read_id=1  3p_anchor=1  passed=2
    /// ```
    ///
    /// ## The one that remains
    ///
    /// `filtered_no_reflen` fires when `ref_len.get(&ref_id)` returns 0. It is
    /// NOT TESTED — no attempt was made — and the reason it is not merely
    /// "hard" is that `ref_len` is built from the header, whose lengths are
    /// `NonZero`. **That is a library statement about a constructor, not a
    /// proof that I tried**, and it is labelled as one. If a route exists, this
    /// comment is where it belongs.
    #[test]
    fn test_the_three_observable_drop_paths_really_execute() {
        let recs = vec![
            synth_record("ok0", 0, None, 0, 60), // -> passed
            synth_record("ok1", 1, None, 0, 60), // -> passed
            // No tid. MAPQ 60 clears the MAPQ gate, so it dies at
            // reference_sequence_id() rather than one line earlier.
            synth_record_no_refid("norid", None, 0, 60), // -> filtered_no_refid
            // Valid tid, MAPQ 60, POS never set -> encoder writes 0 ->
            // alignment_start() cannot make a Position.
            synth_record_no_position(2, None, 0, 60), // -> filtered_no_position
            // No QNAME, valid tid, FLAG 0: clears every filter but the last.
            synth_record_no_read_id(0, None, 0, 60), // -> filtered_no_read_id
            // reverse + gap => 3' end lands short of the TES by more than
            // tes_slack, so this exercises a FOURTH bucket and keeps the
            // per-bucket expectations from being three-field trivial.
            synth_record_with_gap("gap3p", 1, 100, true, None, 0, 60), // -> filtered_3p_anchor
        ];

        let bam = write_synthetic_bam("reach", recs);
        let (anchors, s) =
            extract_anchors_from_bam(&bam, "reach", &TxToGene::new(), params()).unwrap();

        // --- THE INVARIANT, FIRST ------------------------------------------
        // Same ordering rule as `test_anchor_record_conservation`, and for the
        // same reason: the identity holds for ANY input as long as every drop
        // is counted, so it must be the thing that reports an uncounted drop
        // rather than a fixture expectation noticing a number moved.
        assert_eq!(
            s.total_records,
            counted_buckets(&s) + s.passed,
            "a record left the stream through an uncounted path: \
             total_records={} counted={} passed={}",
            s.total_records,
            counted_buckets(&s),
            s.passed
        );

        // The whole point of this test: these three are NON-ZERO on a real BAM.
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
        // Signed, so an OVER-count is visible too. `AnchorStats::unaccounted()`
        // clamps with `saturating_sub`, which would read a record bumped into
        // two buckets as "balanced" -- this cannot, because the sum is
        // subtracted as a whole rather than clamped per step.
        assert_eq!(
            s.total_records as i64 - counted_buckets(&s) as i64 - s.passed as i64,
            0,
            "signed conservation: a negative value means a record was counted twice"
        );

        let _ = std::fs::remove_file(&bam);
    }

    /// The SAME identity, evaluated over a `SampleStats` built by hand.
    ///
    /// As of Ticket 2B-Fix (2026-10-13) this is no longer the only place the
    /// three counters appear in a runtime assertion --
    /// `test_the_three_observable_drop_paths_really_execute` reaches all three
    /// through a real BAM. What is left for THIS test is the arithmetic on a
    /// counter no fixture populates (`filtered_no_reflen`, untested): it
    /// asserts that a stats value carrying all twelve buckets at once still
    /// balances, and that a leak is reported rather than absorbed.
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
            // A CLOSED read block: 7 + 9 + 1 + 3 + 2 + 1 + 4 + 2 == 29.
            reads_seen: 29,
            anchors_emitted: 7,
            rejected_tie: 9,
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

        // One record lost: the helper must surface it, not absorb it.
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

    // =================================================================
    // PER-READ SELECTION (2026-10-18)
    //
    // Every test below is written so that the PRE-SELECTION code — one anchor
    // per record, no vetoes — makes it go red. A fixture that only proves the
    // new code runs is a fixture that cannot catch the new code being wrong.
    // =================================================================

    /// A `tx_to_gene` TSV, removed with the rest of the scratch files.
    fn tx2gene_tsv(tag: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "telescope_tx2gene_{tag}_{}.tsv",
            std::process::id()
        ));
        std::fs::write(&p, format!("transcript_id\tgene_id\n{body}")).unwrap();
        p
    }

    /// Run the extractor over a fixture and hand back the stats, cleaning the
    /// BAM up.
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

    /// The conservation identity, asserted the way `test_anchor_record_conservation`
    /// asserts it: FIRST, before any per-counter expectation, so an uncounted
    /// drop is reported as the ledger failure it is rather than as a fixture
    /// that quietly changed.
    /// `total_records - counted_buckets - passed`, signed-free because the
    /// identity assert above it has already ruled out an over-count.
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

    /// SAME-GENE VETO. The read's best alignment is 10 AS ahead of its second,
    /// which clears the gap floors — so the ONLY thing that can reject it is
    /// the veto, and if the veto did not fire this read would become an anchor.
    ///
    /// The pre-selection code emitted an anchor here, because it never looked
    /// at the second candidate at all.
    #[test]
    fn test_a_same_gene_pair_is_vetoed_even_when_the_gap_is_clear() {
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
            0,
            "both candidates are in one gene: the read cannot choose an isoform"
        );
        assert_eq!(s.passed, 0);
        assert_eq!(s.rejected_same_gene, 1);
        assert_eq!(
            s.rejected_cross_gene, 0,
            "a same-gene pair must not be counted as cross-gene"
        );
        assert_eq!(s.rejected_tie, 0, "AS 100 vs 90 is not a tie");
        assert_eq!(
            s.filtered_eligible_not_anchor, 2,
            "both records were physically eligible and neither was emitted"
        );
        assert_eq!(record_deficit(&s), 0);
    }

    /// CROSS-GENE PAIR REJECTED BY THE GAP, COUNTED SEPARATELY. Same shape as
    /// the test above with one fact changed — `tx1` moves to its own gene — and
    /// the gap raised to 15 so the pair is rejected for the gap rather than
    /// for sharing a gene.
    ///
    /// This is the only thing that gives `rejected_cross_gene` a producer. A
    /// cross-gene pair that CLEARS the gap becomes an anchor, so without this
    /// the counter would print 0 forever and a reader would take that for "no
    /// cross-gene pair was ever rejected".
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

    /// EXACT ARGMAX TIE. AS 100 vs AS 100 — not "close", identical. The
    /// strict argmax has no winner, so there is no anchor.
    ///
    /// WHY THIS IS THE COMMON CASE AND NOT A CORNER CASE: on HC2, 56.0% of
    /// reads with two or more physically-eligible candidates have their top two
    /// AS values exactly equal, and 98.7% of those reads have every candidate
    /// inside one gene. The read lies in sequence the candidates share, so it
    /// carries no isoform information at all. Emitting an anchor anyway would
    /// be inventing an isoform assignment, which is what the pre-selection
    /// code did on every one of them.
    ///
    /// Cross-gene on purpose: it proves the tie fires BEFORE the gene
    /// comparison, so a tie is not silently reclassified as a same-gene veto.
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
        // The rule that fires is the one the data triggers, whichever map is
        // supplied.
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

    /// GAP BOUNDARY, AT THE ARITHMETIC'S OWN EDGE.
    ///
    /// Two reads, identical in every way except the AS gap: 10 and 20. The
    /// floor is 15. One gap-1 either side decides the run, so the comparison
    /// cannot be `<=` on one side and `<` on the other without this going red.
    ///
    /// A cross-gene map, because with a same-gene map the veto would reject
    /// both reads and the gap would never be the deciding rule.
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
                // gap exactly 15: ON the floor -> accepted, since the rule is
                // `gap >= floor`.
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

        // THE OTHER FLOOR, exercised on its own. `gap_rel` is a fraction of
        // AS_best, so with AS_best = 100 a floor of 0.05 is a gap of 5.
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

    /// DEDUP BY ref_id, KEEPING MAX AS. Two records on the SAME transcript
    /// are one candidate, not two, so they cannot produce the cross-transcript
    /// argmax tie, and only the highest-scoring one is put through the
    /// physical filters.
    ///
    /// The fixture makes the ORDER of those two steps observable, which a
    /// simple "three identical records -> one anchor" fixture does not: the
    /// 3'-truncated record carries the HIGHER AS, so dedup-first evaluates the
    /// truncated alignment and rejects the read, while filters-first would have
    /// dropped the truncated one and let the full-length one through. If the
    /// order is ever swapped, the two buckets below swap with it.
    #[test]
    fn test_two_records_to_the_same_reference_dedup_to_the_max_as() {
        let tsv = tx2gene_tsv("dedup", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();

        // Read 1: on tx0 only. The AS-100 record is reverse-strand with a 100 bp
        // 5' gap, so its 3' end lands 100 bp short of the TES — beyond the
        // slack of 35 — and it is the max-AS record, hence the one evaluated.
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

        // Read 2: three identical-AS records on one transcript. Dedup collapses
        // them to one candidate, so this read has no argmax tie with itself and
        // emits exactly one anchor. Three records in, one out.
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

    /// DEDUP GROUPS NON-ADJACENT RECORDS, WHICH IS THE ONLY WAY TO SEE IT.
    ///
    /// ⚠ THIS CASE EXISTS BECAUSE THE TWO ABOVE DID NOT TEST DEDUP. Both put a
    /// read's records to one `ref_id` ADJACENT in the BAM, and the per-read pass
    /// collapses runs of equal `ref_id` — so they collapse whether or not the
    /// group was sorted first. Deleting the sort entirely left the whole suite
    /// GREEN (mutation battery, 2026-10-18). The two cases above are still
    /// worth having — they pin the multiplicity accounting — but on their own
    /// they do not pin dedup.
    ///
    /// Here the same reference is split across the read by another one, so the
    /// two `tx0` records only become one candidate if the sort really ran:
    ///
    /// ```text
    ///   tx0 AS 100, 3' truncated   ─┐  tx0 rep: mult 2, max AS 100
    ///   tx1 AS  50                 ─┤  tx1 rep: mult 1, AS 50
    ///   tx0 AS  50, full length    ─┘
    /// ```
    ///
    /// Sorted: the `tx0` representative is the TRUNCATED one (AS 100), so it
    /// fails the 3' filter and stands for BOTH of its records
    /// (`filtered_3p_anchor == 2`), leaving `tx1` as the read's single eligible
    /// candidate → one anchor.
    ///
    /// Unsorted: the truncated record is a representative on its own
    /// (`filtered_3p_anchor == 1`), the full-length `tx0` survives as a second
    /// candidate, and it ties with `tx1` on AS 50 → no anchor at all. Three
    /// different counters move, so the sort cannot be deleted silently.
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
            1,
            "tx1 is the read's only eligible candidate, so it is the anchor"
        );
        assert_eq!(anchors[0].transcript_id, "tx1");
        assert_eq!(
            s.rejected_tie, 0,
            "the two tx0 records are one candidate, so they cannot tie with \
             anything; only one candidate survives here at all"
        );
        assert_eq!(s.filtered_eligible_not_anchor, 0);
        assert_eq!(s.passed, 1);
    }

    /// SUPPLEMENTARY AND UNMAPPED RECORDS DO NOT REACH THE POOL — INCLUDING
    /// AT MAPQ >= 20.
    ///
    /// THIS IS A LIVE DEFECT, NOT A NEW RULE. 33.43% of HC2's 25,077
    /// supplementary records carry MAPQ >= 20, so they cleared the old MAPQ
    /// gate and were written to the anchor pool as if a split segment were a
    /// complete molecule. `filtered_unique == 0` below is the load-bearing
    /// assert: it says the records did NOT die at the MAPQ gate, which is the
    /// route they used to take.
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

    /// SECONDARY ALIGNMENTS ARE ADMITTED, and the record ledger still closes
    /// with them in the stream.
    ///
    /// `0x100` is the flag the input contract (spec F-BAM-Alignment) requires
    /// be preserved. A secondary with MAPQ 60 is a real alignment to a real
    /// transcript and must be allowed to win an argmax; dropping it would
    /// reintroduce exactly the ambiguity this pass exists to resolve.
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
        // SET-WISE, NOT SEQUENCE-WISE. Anchor ROW ORDER is a deterministic
        // function of the read's 128-bit name hash (see the pass-2 ordering
        // comment), not of BAM order: `p0` and `s0` are emitted in whichever
        // order their keys sort into, which is not the order they were written
        // to the fixture. Asserting the input order here would pin a property
        // the code deliberately does not have — and it went red for exactly that
        // reason when the cap order was changed from BAM order to hash order.
        //
        // Nothing downstream may depend on the order: `emission_model.rs`'s
        // `read_anchor_pool_parquet` accumulates every row into per-transcript
        // collections and never indexes across rows, so a permutation of the
        // pool is invisible to it.
        let mut tx: Vec<&str> = anchors.iter().map(|a| a.transcript_id.as_str()).collect();
        tx.sort_unstable();
        assert_eq!(tx, vec!["tx0", "tx1"]);
        let mut ids: Vec<&str> = anchors.iter().map(|a| a.read_id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec!["p0", "s0"]);
    }

    /// ANCHORS EMITTED <= READS SEEN, AND THE READ BLOCK CLOSES.
    ///
    /// The inequality is the one that can only break if selection ever stops
    /// being per read — a re-introduced per-record push would put two anchors
    /// on one read and break it. The closure is the denominator check: it says
    /// every read that reached the per-read stage landed in exactly one
    /// outcome, so a rejection cannot be counted twice or lost.
    #[test]
    fn test_anchors_emitted_never_exceed_reads_seen_and_the_read_block_closes() {
        use noodles_sam::alignment::record::Flags;
        let tsv = tx2gene_tsv("closure", "tx0\tGA\ntx1\tGB\ntx2\tGA\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();

        let (anchors, s) = run_extract(
            "closure",
            vec![
                // 1. one candidate -> anchor
                synth_record("solo", 0, Some(100), 0, 60),
                // 2. cross-gene, gap clear -> anchor
                synth_record("cross", 0, Some(100), 0, 60),
                synth_record("cross", 1, Some(80), 0, 60),
                // 3. exact tie -> rejected_tie
                synth_record("tie", 0, Some(100), 0, 60),
                synth_record("tie", 1, Some(100), 0, 60),
                // 4. same gene (tx0/GA, tx2/GA), gap clear -> vetoed
                synth_record("samegene", 0, Some(100), 0, 60),
                synth_record("samegene", 2, Some(70), 0, 60),
                // 5. physically ineligible everywhere -> no eligible candidate
                synth_record_with_gap("dead", 1, 100, true, Some(100), 0, 60),
                // 6. dropped before grouping, so NOT a read
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
        assert_eq!(s.anchors_emitted, 2);
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
        assert_eq!(s.rejected_same_gene, 1);
        assert_eq!(s.rejected_cross_gene, 0);
        assert_eq!(s.gene_veto_disabled, 0);
        assert_eq!(s.reads_no_eligible_candidate, 1);

        assert_eq!(
            s.reads_seen,
            s.anchors_emitted
                + s.rejected_tie
                + s.rejected_gap
                + s.rejected_same_gene
                + s.rejected_cross_gene
                + s.gene_veto_disabled
                + s.reads_no_eligible_candidate
                + s.reads_over_cap,
            "every read that reached the per-read stage must land in exactly one \
             outcome; reads_seen={} tie={} gap={} same_gene={} cross_gene={} \
             veto_disabled={} no_eligible={} over_cap={} anchors={}",
            s.reads_seen,
            s.rejected_tie,
            s.rejected_gap,
            s.rejected_same_gene,
            s.rejected_cross_gene,
            s.gene_veto_disabled,
            s.reads_no_eligible_candidate,
            s.reads_over_cap,
            s.anchors_emitted,
        );
        // `reads_over_cap` is 0 on this fixture, so adding it above is
        // arithmetically inert and would leave the identity untested in the very
        // direction it was added for. The fixture that makes it non-zero is
        // `test_a_multi_record_read_over_the_cap_absorbs_every_record_it_contributed`.
        assert_eq!(
            s.reads_over_cap, 0,
            "no transcript reaches the cap of 200 here, so this fixture cannot \
             witness reads_over_cap -- do not read the identity above as \
             covering it"
        );
    }

    /// THE CAP BRANCH ABSORBS EVERY RECORD THE REJECTED READ CONTRIBUTED.
    ///
    /// THE FIXTURE IS DELIBERATELY MULTI-RECORD, AND THAT IS THE WHOLE POINT.
    /// The pre-existing cap fixture
    /// (`test_anchor_record_conservation`) is 205 reads of ONE record each, so
    /// `eligible_records == 1` everywhere and `filtered_cap += 1` happened to
    /// be exact. On HC2 with the MAPQ gate in place it was also exact, for the
    /// same reason and at a scale of 412,478 reads. A test built from
    /// single-record reads cannot see the defect at all: it proves the READ
    /// identity and silently misses the RECORD one. This fixture is three reads
    /// of TWO records each, which makes `eligible_records - 1` equal to 1 and
    /// the leak a whole record.
    ///
    /// THE FIXTURE IS ALSO ORDER-INDEPENDENT ON PURPOSE. The cap keeps the
    /// first `max_reads_per_sample` reads in read-id-hash order, so with a cap
    /// of 2 over 3 competing reads it is not knowable in advance WHICH read is
    /// rejected. Every one of the three has `eligible_records == 2`, so the
    /// assertion holds whichever read loses — and a fixture that named the
    /// winner would have to encode a hash order in a test file, which is the
    /// thing the hash rule exists to avoid depending on.
    ///
    /// Red before the R139 fix: the cap branch absorbed 1 of 2 records, so
    /// `filtered_eligible_not_anchor + filtered_cap + passed` was 5 against a
    /// `total_records` of 7, and `unaccounted()` reported 1. That is the
    /// UNDER-count direction, which `saturating_sub` reports honestly; the
    /// over-count direction is the one it clamps to 0.
    ///
    /// `lowq` is here for a second, unrelated reason: it carries MAPQ 5 and
    /// would have died at the retired gate, so the fixture also witnesses
    /// `filtered_unique == 0` on an input where the gate USED to fire.
    #[test]
    fn test_a_multi_record_read_over_the_cap_absorbs_every_record_it_contributed() {
        let tsv = tx2gene_tsv("capleak", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = TxToGene::from_tsv(&tsv).unwrap();
        assert_eq!(map.len(), 3, "a cross-gene map, so the veto cannot fire");

        let mut recs = Vec::new();
        for i in 0..3 {
            // Two RECORDS, two TRANSCRIPTS, so the dedup cannot collapse them
            // and `eligible_records` is genuinely 2. AS 100 on tx2 wins the
            // argmax, AS 50 on tx0 loses; the gap is 50 and the pair is
            // cross-gene, so the read reaches the cap check rather than being
            // rejected earlier.
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

        // --- THE RECORD IDENTITY, FIRST -------------------------------------
        assert_record_ledger_closes(&s);
        assert_eq!(record_deficit(&s), 0);

        assert_eq!(s.total_records, 7);
        assert_eq!(
            s.reads_seen, 4,
            "three saturating reads plus the low-MAPQ read"
        );

        // --- THE READ IDENTITY, WITH A NON-ZERO CAP TERM --------------------
        assert_eq!(s.reads_over_cap, 1, "the third read on tx2 is over the cap");
        assert_eq!(
            s.reads_seen,
            s.anchors_emitted
                + s.rejected_tie
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

        // --- THE LEAK ITSELF, IN RECORDS ------------------------------------
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

        // --- THE MAPQ GATE IS GONE, WITNESSED WHERE IT USED TO FIRE ---------
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
        assert_eq!(s.rejected_same_gene, 0);
        assert_eq!(s.rejected_cross_gene, 0, "the gap floors are off");
        assert_eq!(s.gene_veto_disabled, 0);
        assert_eq!(s.reads_no_eligible_candidate, 0);

        // The anchors are the two cap winners on tx2 plus the low-MAPQ read on
        // tx1. WHICH two sat reads won is hash order and is deliberately not
        // asserted; the read_ids must still be distinct, or the per-read
        // grouping itself is broken.
        let mut ids: Vec<&str> = anchors.iter().map(|a| a.read_id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec!["lowq", "sat1", "sat2"]);
        let mut txs: Vec<&str> = anchors.iter().map(|a| a.transcript_id.as_str()).collect();
        txs.sort_unstable();
        assert_eq!(txs, vec!["tx1", "tx2", "tx2"]);
    }

    /// AN EMPTY `tx_to_gene` DOES NOT SILENTLY BECOME "CROSS-GENE".
    ///
    /// With no map, `gene_id == transcript_id` for every candidate, so a naive
    /// same-gene check compares each transcript with itself and can NEVER fire.
    /// Every intra-gene multi-candidate read would then be filed as cross-gene —
    /// and it would be filed so with total confidence, because the comparison
    /// genuinely ran and genuinely said "different". This test pins the two
    /// facts that make that state visible instead: `gene_veto_disabled` and
    /// `same_gene_unknown` are non-zero, and `rejected_cross_gene` is zero.
    #[test]
    fn test_an_empty_tx_to_gene_routes_multi_candidate_reads_to_the_unknown_class() {
        let (anchors, s) = run_extract(
            "vetooff",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record("r", 1, Some(90), 0, 60),
                // A solo read is unaffected: with one candidate there is no
                // pair to classify, so the veto has nothing to do.
                synth_record("solo", 2, Some(100), 0, 60),
            ],
            &TxToGene::new(),
            params(),
        );

        assert_record_ledger_closes(&s);
        assert_eq!(
            anchors.len(),
            1,
            "only the single-candidate read is emitted"
        );
        assert_eq!(s.gene_veto_disabled, 1);
        assert_eq!(
            s.same_gene_unknown, 1,
            "the read must be routed to the UNKNOWN class, not to cross_gene"
        );
        assert_eq!(
            s.rejected_cross_gene, 0,
            "with an empty map the gene relation is unknown, and 'unknown' is \
             not 'cross-gene' — this is the whole point of the counter"
        );
        assert_eq!(
            s.rejected_same_gene, 0,
            "the veto cannot fire on an empty map"
        );
        assert_eq!(s.filtered_eligible_not_anchor, 2);

        // And the empty-map fallback for the EMITTED gene_id is unchanged:
        // it is still `gene_id := transcript_id`, which is what the degenerate
        // gene pool warning keys on. That behaviour is orthogonal to the veto
        // and two existing tests depend on it.
        assert_eq!(anchors[0].gene_id, anchors[0].transcript_id);
    }

    /// A NON-EMPTY `tx_to_gene` THAT DOES NOT COVER THE BAM ERRORS OUT.
    ///
    /// The empty-map fallback (`gene_id := transcript_id`) is legitimate for a
    /// novel single-transcript locus and stays. The non-empty case is a
    /// different fact: the TSV and the BAM's `@SQ` header disagree, and the
    /// silent fallback would then decide the same-gene veto on an invented
    /// gene id — a veto that can never fire reads exactly like a veto that
    /// correctly found nothing.
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

    /// THE `reads(...)` BLOCK APPEARS IN `report()` AND SAYS IT IS READS.
    ///
    /// `report()`'s prefix is pinned byte-for-byte by the Dataset A scraper's
    /// regex, so the new material can only go at the end — and going at the
    /// end is also what keeps the two denominations from being read as one.
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
            ..Default::default()
        };
        let line = agg.report();

        // The pinned prefix, exactly as the regex expects it.
        assert!(
            line.starts_with("samples=1 total=100 filtered(unique=0 no_reflen=0 3p_anchor=0 coverage=0 polya=0 cap=0) passed=0 unaccounted=65 no_refid=0 no_position=0 no_read_id=0"),
            "the pinned prefix changed, which breaks ANCHOR_STATS_RE:\n{line}"
        );
        for token in [
            "unmapped=12",
            "supplementary=13",
            "not_anchor=10",
            "reads(seen=27 anchors=7 tie=9",
            // `over_cap` belongs INSIDE reads(...) and at its END: inserting it
            // into `filtered(...)` would sit inside the region ANCHOR_STATS_RE
            // pins, and putting it before `reads(` would make a read count
            // readable as a record count.
            "no_eligible=0 over_cap=4)",
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
