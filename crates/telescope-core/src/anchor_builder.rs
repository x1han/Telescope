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
            None => {
                stats.filtered_no_read_id += 1;
                continue;
            }
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
    /// MAPQ < 20. NOTE this is NOT "non-unique" -- it is "low MAPQ", and on
    /// these BAMs it swallows nearly everything. Reproduce (HC2, 2026-10-13):
    ///
    /// ```bash
    /// B=/datg/hanxi/project/Telescope/aidrs_pipeline/trans_align/HC2/HC2.sorted.bam
    /// samtools flagstat "$B" | grep -E 'in total|mapped'      # 39369625 / 37714800
    /// samtools view "$B" | awk '$5 < 20' | wc -l              # 38163664
    /// samtools view -f 4 "$B" | awk '{print $5}' | sort -u    # 0  (every unmapped record)
    /// samtools view -f 4 "$B" | awk '$5 >= 20' | wc -l        # 0
    /// ```
    ///
    /// ⇒ unmapped records are counted HERE, not in `filtered_no_refid`: they
    /// never survive the MAPQ gate. Splitting the two is deferred -- it changes
    /// what the ledger asserts, so it is a decision, not a repair.
    pub filtered_unique: u64,
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
    pub filtered_cap: u64,
    pub passed: u64,
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
        self.parse_errors += s.parse_errors;
        self.filtered_unique += s.filtered_unique;
        self.filtered_no_refid += s.filtered_no_refid;
        self.filtered_no_position += s.filtered_no_position;
        self.filtered_no_read_id += s.filtered_no_read_id;
        self.filtered_no_reflen += s.filtered_no_reflen;
        self.filtered_3p_anchor += s.filtered_3p_anchor;
        self.filtered_coverage += s.filtered_coverage;
        self.filtered_polya_invalid += s.filtered_polya_invalid;
        self.filtered_cap += s.filtered_cap;
        self.passed += s.passed;
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
    /// The production claim rests on measured facts recorded at `SampleStats`:
    /// every unmapped record carries MAPQ 0, so it dies at the MAPQ gate before
    /// reaching `reference_sequence_id()`. Reproduce with the `samtools`
    /// commands quoted on that field. On HC2 all three new buckets are 0.
    ///
    /// The identity was observably holding on this data. What it was NOT was
    /// *guaranteed* -- and, before Ticket 2B, no test asserted it.
    pub fn report(&self) -> String {
        format!(
            "samples={} total={} filtered(unique={} no_reflen={} 3p_anchor={} coverage={} \
             polya={} cap={}) passed={} unaccounted={} no_refid={} no_position={} no_read_id={}",
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
            self.filtered_no_read_id
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
            + self.filtered_no_refid
            + self.filtered_no_position
            + self.filtered_no_read_id
            + self.filtered_no_reflen
            + self.filtered_3p_anchor
            + self.filtered_coverage
            + self.filtered_polya_invalid
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
            .saturating_sub(self.filtered_no_refid)
            .saturating_sub(self.filtered_no_position)
            .saturating_sub(self.filtered_no_read_id)
            .saturating_sub(self.filtered_no_reflen)
            .saturating_sub(self.filtered_3p_anchor)
            .saturating_sub(self.filtered_coverage)
            .saturating_sub(self.filtered_polya_invalid)
            .saturating_sub(self.filtered_cap)
            .saturating_sub(self.passed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::test_support::{
        synth_record, synth_record_no_position, synth_record_no_read_id, synth_record_no_refid,
        synth_record_with_gap, write_synthetic_bam,
    };

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
            + s.filtered_no_refid
            + s.filtered_no_position
            + s.filtered_no_read_id
            + s.filtered_no_reflen
            + s.filtered_3p_anchor
            + s.filtered_coverage
            + s.filtered_polya_invalid
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
            synth_record("ok0", 0, None, 0, 60),  // -> passed
            synth_record("ok1", 1, None, 0, 60),  // -> passed
            synth_record("lowq", 0, None, 0, 10), // MAPQ < 20
            synth_record_with_gap("gap3p", 1, 100, true, None, 0, 60), // 3' too short
            synth_record_with_gap("shortcov", 1, 500, false, None, 0, 60), // coverage
            nopt,                                 // no poly(A)
        ];
        // 205 records on one transcript with the cap at 200 -> 5 over cap.
        for i in 0..205 {
            recs.push(synth_record(&format!("cap{i}"), 2, None, 0, 60));
        }

        let bam = write_synthetic_bam("conserve", recs);
        let (anchors, s) =
            extract_anchors_from_bam(&bam, "conserve", &TxToGene::new(), 35, 0.80, 200).unwrap();

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
        assert_eq!(s.filtered_unique, 1, "fixture lost its MAPQ<20 record");
        assert_eq!(s.filtered_3p_anchor, 1, "fixture lost its 3'-anchor record");
        assert_eq!(s.filtered_coverage, 1, "fixture lost its coverage record");
        assert_eq!(
            s.filtered_polya_invalid, 1,
            "fixture lost its no-poly(A) record"
        );
        assert_eq!(s.filtered_cap, 5, "fixture lost its over-cap records");
        assert_eq!(s.passed, 202, "fixture lost its passing records");
        assert_eq!(
            s.total_records,
            1 + 1 + 1 + 1 + 5 + 202,
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
            extract_anchors_from_bam(&bam, "reach", &TxToGene::new(), 35, 0.80, 200).unwrap();

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
    /// asserts that a stats value carrying all ten buckets at once still
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
            filtered_unique: 10,
            filtered_no_refid: 3,
            filtered_no_position: 4,
            filtered_no_read_id: 5,
            filtered_no_reflen: 6,
            filtered_3p_anchor: 7,
            filtered_coverage: 8,
            filtered_polya_invalid: 9,
            filtered_cap: 11,
            passed: 37,
            distinct_genes: 2,
            distinct_transcripts: 3,
        };
        assert_eq!(
            agg.unaccounted(),
            0,
            "ten populated buckets summing to total_records must leave 0"
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
}
