//! Module 1: Emission Model Builder
//!
//! Two ways to construct the per-transcript emission model
//! $P(L \mid T_k)$ — a Gaussian in $\log_2(\text{PAL})$
//! space — for TEL-EM:
//!
//! - **Universal mode (the only one reachable from a CLI
//!   subcommand)**: parses the per-read
//!   `telescope.anchor_pool.parquet` (produced by
//!   `anchor_builder::build_anchor_pool`), optionally joins
//!   with `telescope.transcript_evidence.parquet` to apply
//!   `em_eligible` filtering, then fits the emission model
//!   with a 3-level fallback cascade (Tx → Gene → Cohort).
//!   Entry point: `build_emission_from_anchors`, dispatched
//!   from `telescope build-emission --anchor-pool`.
//!
//! - **External poly(A) initializer (DEFERRED, not wired to
//!   any CLI subcommand)**: parses an AIDRS-shaped pair
//!   (`aidrs.transcript_polyA_len.parquet` +
//!   `aidrs.transcript.assessment.tsv`) and builds the same
//!   emission model. The three functions
//!   [`AssessmentQuality`], [`load_assessment_quality`] and
//!   [`build_emission_parquet`] are RETAINED, working and
//!   tested, but no CLI entry point currently reaches them:
//!   the `--polya-parquet` / `--assessment-tsv`
//!   arguments of `telescope build-emission` were deleted on
//!   2026-10-03, and `--anchor-pool` is now that subcommand's
//!   sole input.
//!
//!   WHY IT IS DEFERRED, NOT DELETED. This was once "AIDRS
//!   Mode", a peer of Universal mode with its own CLI surface,
//!   and it was removed in that role because the two paths
//!   were never actually compared. Whether AIDRS's upstream
//!   read→transcript assignment is BETTER OR WORSE than
//!   Universal's own is a measurement that HAS NEVER BEEN
//!   MADE. Until it is made, the initializer design is frozen
//!   in the 解冻区 (thaw region): keeping the code costs
//!   nothing, but re-exposing the CLI would be a decision
//!   dressed as a convenience. The one thing the frozen code
//!   does establish is agreement, not superiority — see
//!   `test_aidrs_universal_equivalence` below.
//!
//! Output: `telescope.emission.parquet` with the schema
//! declared in `output_schema()`. Each row is one transcript
//! $T_k$ with computed emission parameters
//! $(\mu_{k,\text{cohort}}, \sigma^2_{k,\text{cohort}})$.
//!
//! ## The fitted object is a DISTRIBUTION, not a scalar
//!
//! A row of the anchor pool is one (sample, transcript, read)
//! triple carrying one observed `poly_a_length`. A
//! transcript's emission is therefore fitted to a
//! **per-read observation set**, and $\mu_k$ / $\sigma^2_k$
//! are DERIVED PARAMETERS of that fit — summary statistics,
//! not measurements. A transcript with 1, 3 or 400 anchor
//! reads has the same kind of row here; only the width of the
//! observation set behind it differs. The shipped anchor
//! pool is 90,703 rows over 15,427 transcripts, of which
//! 49.32% (7,609) carry more than one distinct observed
//! length, so the scalar reading of a "transcript's
//! poly(A) length" is wrong for roughly half the
//! transcriptome and the distribution reading is the one
//! to use. Both figures, and the command that produces
//! them, are in `workspace/notes/EXPERIMENT.md` under
//! "2026-10-03 (measurement provenance)"; the row and
//! transcript counts also appear in the 2026-10-02
//! anchor-pool fan-out entry.
//!
//! ## `polyA_frac` + `is_polyA_competent` semantics
//!
//! - **External poly(A) initializer (deferred)**: state is
//!   lifted from `aidrs.transcript.assessment.tsv` (State
//!   1/2/3 → 3-way classification, see
//!   `load_assessment_quality`).
//! - **Universal mode**: there is no equivalent
//!   `assessment.tsv`. We define:
//!   - `polyA_frac` = fraction of anchor reads for the
//!     transcript with `poly_a_length >= 10`. Since the
//!     anchor filter already requires `pt:i >= 10`, this
//!     fraction is **1.0 by construction**.
//!   - `is_polyA_competent` = `polyA_frac > 0 AND n_reads > 0`.
//!     Always true for non-empty transcripts.
//!
//! This semantic gap is the reason the equivalence test
//! (`test_aidrs_universal_equivalence`) compares only
//! `mu_logPAL`, `sigma2_logPAL`, and `n_cohort_flnc`, and
//! documents `polyA_frac` / `is_polyA_competent` as
//! "by-construction-equal" rather than asserted.
//!
//! Source spec: `workspace/specs/telescope-v0.2.1.md`
//! 常温区 Module 1 (Mode B = Universal) + F2/F3 + Step 3.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};

use crate::anchor_builder::TxToGene;
use arrow::array::{
    Array, BooleanArray, BooleanBuilder, Float64Builder, Int32Array, ListArray, StringArray,
    StringBuilder, UInt32Builder,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

/// Physical instrumentation floor on cohort variance.
///
/// Why this constant exists (per `specs/telescope-v0.1.3.md` F3 and
/// `notes/spec-self-audit-v0.1.3.md`):
///
/// Even with the Weak-MAP weak prior in TEL-EM (denominator
/// `+2*nu_0+3 = 5`), the variance update hits a degenerate corner
/// if a single-molecule cluster assigns a read to a transcript
/// whose cohort emission model has `var(y) == 0` *and* the new read
/// has `y_r == mu_k` exactly. In that case
/// `sigma^2(m+1) = 0 / (w_rk + 5) = 0`, locking sigma at 0 and
/// starving subsequent E-steps of the Gaussian likelihood.
///
/// The floor is a **physical** quantity, not a numerical convenience, and the
/// name says so. Renamed from `MIN_COHORT_VARIANCE` on 2026-10-02 by expert
/// ruling, which required the constant to carry its justification rather than
/// a bare number.
///
/// # Why 0.04
///
/// The floor lives in `log2(PAL)` space — `evidence.rs` computes
/// `log_pal = (poly_a_len as f32).log2()`, so `y` is base 2 throughout. With
/// `sigma^2 = 0.04` we have `sigma = 0.20` in log2 units, and a lognormal with
/// log2-scale sd `sigma` has a physical CV of
///
/// ```text
///     CV = 2^sigma - 1 = 2^0.20 - 1 = 0.1487
/// ```
///
/// i.e. **~15%**. That matches the instrument: a poly(A) tail measured by
/// nanopore current (Dorado, tailfindr, Nanopolish) carries a signal-noise
/// floor of roughly +/- 5-10 nt, and on a 60-80 nt tail that is a 10-15% CV.
/// The floor is that floor, expressed in the units the model works in.
///
/// # Why it must not be removed
///
/// The inverse-gamma prior already prevents the *posterior mean* from
/// collapsing, so the floor is partly redundant with it — a MAP estimate under
/// `nu_0`/`beta_0` cannot reach zero no matter how few reads land on a
/// transcript. What the prior does not do is stop the LIKELIHOOD from becoming
/// degenerate: with `sigma^2 -> 0` and `y_r == mu_k` exactly, the Gaussian
/// density is unbounded, and a single coincidental read can dominate the
/// E-step for that transcript.
///
/// This is the same philosophy as `alevin-fry`'s `MIN_OUTPUT_ALPHA = 0.01`:
/// always keep the posterior non-degenerate, whatever the data does.
///
/// Renamed, NOT removed. The removal argument is a testable claim and is
/// recorded as one rather than settled by assertion — see
/// `workspace/notes/EXPERIMENT.md`, 2026-10-02.
pub const INSTRUMENTAL_PAL_NOISE_FLOOR: f64 = 0.04;

/// MAD-to-sigma scale factor for normal-consistency.
///
/// For a standard normal, `median(|X - median(X)|) * 1.4826 ≈ σ`.
/// We use this to derive a robust cohort-level variance from
/// the median absolute deviation without parametric assumptions.
const MAD_TO_SIGMA: f64 = 1.4826;

/// Fallback thresholds for the 3-level Tx → Gene → Cohort
/// cascade in Universal Mode.
///
/// Per expert ruling (v0.2.1 常温区 Step 3 Part 2), **every
/// threshold MUST be CLI-configurable** ("严禁硬编码常数").
/// The defaults below are the v0.2.1 spec defaults but can be
/// overridden per-invocation via CLI flags
/// (`--tx-min-reads`, `--gene-min-reads`,
/// `--cohort-min-reads`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FallbackConfig {
    /// L1 Tx-level: minimum reads to fit log-normal at the
    /// transcript level. Below this, fall to L2 (gene) or L3
    /// (cohort).
    pub tx_min_reads: usize,
    /// L2 Gene-level: minimum pooled reads in the same
    /// `gene_id` to fit at the gene level. Below this, fall
    /// to L3 (cohort).
    pub gene_min_reads: usize,
    /// L3 Cohort-level: informational minimum for the *total*
    /// cohort read count. **Does not gate** the L3 fit —
    /// cohort fallback is always available as long as ≥1 read
    /// exists — but is logged when the cohort is below the
    /// threshold so that downstream consumers can flag
    /// low-cohort cases.
    pub cohort_min_reads: usize,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            tx_min_reads: 10,
            gene_min_reads: 30,
            cohort_min_reads: 50,
        }
    }
}

/// Per-transcript poly(A) state from `aidrs.transcript.assessment.tsv`.
///
/// Part of the DEFERRED external poly(A) initializer; see the module
/// docs. Retained, not deprecated: no CLI subcommand reaches it, and it
/// is expected to be re-wired only after the read→transcript assignment
/// comparison the module docs name has been made.
#[derive(Debug, Clone, Copy)]
pub struct AssessmentQuality {
    pub poly_a_frac: f64,
    /// True iff `polyA_frac > 0` AND AIDRS state was reachable.
    /// False for State 2 (`polyA_frac` effectively 0) and State 3
    /// (NaN — Dorado `pt:i` tag entirely missing).
    pub is_poly_a_competent: bool,
}

/// Reads `aidrs.transcript.assessment.tsv` and parses the
/// `(TrID, polyA_frac)` columns into a HashMap keyed by TrID.
///
/// State 3 (NaN / missing `polyA_frac`) and State 2
/// (`polyA_frac == 0` or non-numeric) both become
/// `(0.0, is_poly_a_competent=false)`. Numeric non-zero values
/// keep their original `polyA_frac` and flag the transcript as
/// polyA-competent.
///
/// Part of the DEFERRED external poly(A) initializer; see the module
/// docs. Not reachable from any CLI subcommand.
pub fn load_assessment_quality<P: AsRef<Path>>(
    tsv_path: P,
) -> Result<HashMap<String, AssessmentQuality>> {
    let file =
        File::open(&tsv_path).with_context(|| format!("Failed to open {:?}", tsv_path.as_ref()))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let header_line = match lines.next() {
        Some(h) => h?,
        None => bail!("Assessment TSV is empty"),
    };

    let headers: Vec<&str> = header_line.split('\t').collect();
    let tr_id_idx = headers
        .iter()
        .position(|&h| h == "TrID")
        .context("Missing 'TrID' column in assessment.tsv")?;
    let frac_idx = headers
        .iter()
        .position(|&h| h == "polyA_frac")
        .context("Missing 'polyA_frac' column in assessment.tsv")?;

    let mut map = HashMap::new();
    for line in lines {
        let l = line?;
        if l.is_empty() {
            continue;
        }
        let fields: Vec<&str> = l.split('\t').collect();
        if fields.len() <= frac_idx || fields.len() <= tr_id_idx {
            continue;
        }

        let tr_id = fields[tr_id_idx].to_string();
        let frac_str = fields[frac_idx];

        let (poly_a_frac, is_poly_a_competent) = match frac_str.parse::<f64>() {
            Ok(val) if val.is_nan() => (0.0, false), // State 3
            Ok(val) if val <= 0.0 => (0.0, false),   // State 2
            Ok(val) => (val, true),                  // State 1
            Err(_) => (0.0, false),                  // unparseable
        };

        map.insert(
            tr_id,
            AssessmentQuality {
                poly_a_frac,
                is_poly_a_competent,
            },
        );
    }

    Ok(map)
}

/// Schema for the output `telescope.emission.parquet`.
/// Exposed for downstream callers (Module 3) that need to read
/// what Module 1 produced.
pub fn output_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("transcript_id", DataType::Utf8, false),
        Field::new("gene_id", DataType::Utf8, false),
        Field::new("polyA_frac", DataType::Float64, false),
        Field::new("is_polyA_competent", DataType::Boolean, false),
        // WHAT THIS COLUMN ACTUALLY HOLDS. Despite the name, `n_cohort_flnc`
        // is NOT a cohort-fallback count and is not a count of distinct
        // lengths. It is the number of anchor OBSERVATIONS for the
        // transcript: `acc.len()`, one per (sample, transcript, read) row in
        // the anchor pool. The anchor pool carries the same poly_a_length on
        // two different reads as TWO rows, and this column counts 2.
        //
        // The L1/L2/L3 level is therefore NOT stored in the file. It is
        // INFERRED downstream by comparing this value against the
        // `tx_min_reads` / `gene_min_reads` thresholds that
        // `build_emission_from_anchors` was run with: `>= tx_min_reads` is
        // L1, and anything smaller is L2 or L3 depending on its gene's pooled
        // count. Those thresholds are not recorded in the parquet, so this
        // reconstruction is only exact when the reader knows them.
        //
        // The column is NOT renamed and the schema is NOT changed: it is part
        // of the published contract that Module 3 and downstream consumers
        // read. `test_n_cohort_flnc_counts_observations_not_distinct_lengths`
        // pins the observation-counting semantics.
        Field::new("n_cohort_flnc", DataType::UInt32, false),
        Field::new("mu_logPAL", DataType::Float64, false),
        Field::new("sigma2_logPAL", DataType::Float64, false),
    ]))
}

/// Build `telescope.emission.parquet` from AIDRS outputs.
///
/// Reads `aidrs.transcript_polyA_len.parquet` row-by-row (chunked),
/// joins each row's `TrID` against the assessment TSV to get
/// `polyA_frac` and `is_polyA_competent`, computes the log2(pal)
/// mean/variance per transcript, applies `INSTRUMENTAL_PAL_NOISE_FLOOR`
/// floor, and writes the 7-column output schema above.
///
/// Part of the DEFERRED external poly(A) initializer; see the module
/// docs. No CLI subcommand reaches it as of 2026-10-03 — the
/// `build-emission` command that used to dispatch here takes
/// `--anchor-pool` only. Retained so the initializer stays a
/// re-wirable option once the comparison the module docs name has been
/// made; it is covered live by `test_aidrs_universal_equivalence`.
pub fn build_emission_parquet<P: AsRef<Path>>(
    poly_a_parquet_path: P,
    assessment_tsv_path: P,
    output_emission_path: P,
) -> Result<()> {
    let quality_map = load_assessment_quality(assessment_tsv_path)?;

    // Open Parquet reader.
    let file = File::open(&poly_a_parquet_path)
        .with_context(|| format!("Failed to open {:?}", poly_a_parquet_path.as_ref()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to build ParquetRecordBatchReaderBuilder")?;
    let mut reader = builder.with_batch_size(8192).build()?;

    // Output column builders.
    let mut tr_id_builder = StringBuilder::new();
    let mut gene_id_builder = StringBuilder::new();
    let mut frac_builder = Float64Builder::new();
    let mut competent_builder = BooleanBuilder::new();
    let mut count_builder = UInt32Builder::new();
    let mut mu_builder = Float64Builder::new();
    let mut sigma2_builder = Float64Builder::new();

    while let Some(batch_result) = reader.next() {
        let batch = batch_result.context("Failed to read next Parquet batch")?;

        let tr_col = batch
            .column_by_name("TrID")
            .context("Missing 'TrID' column in polyA parquet")?
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .context("Failed to downcast 'TrID' to StringArray")?;

        let gene_col = batch
            .column_by_name("GeneID")
            .context("Missing 'GeneID' column in polyA parquet")?
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .context("Failed to downcast 'GeneID' to StringArray")?;

        let raw_col = batch
            .column_by_name("raw_polyA_lengths")
            .context("Missing 'raw_polyA_lengths' column in polyA parquet")?
            .as_any()
            .downcast_ref::<ListArray>()
            .context("Failed to downcast 'raw_polyA_lengths' to ListArray")?;

        for i in 0..batch.num_rows() {
            let tr_id = tr_col.value(i);
            let gene_id = gene_col.value(i);

            // Pull the i-th row's List<Int32> child array.
            // `raw_col.value(i)` returns `Arc<dyn Array>` (NOT &Arc).
            let values_arc: &arrow::array::ArrayRef = &raw_col.value(i);
            let int_values = (**values_arc)
                .as_any()
                .downcast_ref::<arrow::array::PrimitiveArray<Int32Type>>()
                .context("List elements are not int32")?;

            let lengths: Vec<i32> = (0..int_values.len())
                .filter(|&idx| !int_values.is_null(idx))
                .map(|idx| int_values.value(idx))
                .filter(|&l| l > 0)
                .collect();

            let quality = quality_map.get(tr_id);
            let poly_a_frac = quality.map(|q| q.poly_a_frac).unwrap_or(0.0);
            let mut is_competent = quality.map(|q| q.is_poly_a_competent).unwrap_or(false);

            let n_flnc = lengths.len() as u32;
            if n_flnc == 0 {
                is_competent = false;
            }

            let (mu, sigma2) = if is_competent && n_flnc > 0 {
                let log_lens: Vec<f64> = lengths.iter().map(|&x| (x as f64).log2()).collect();
                let mean = log_lens.iter().sum::<f64>() / (n_flnc as f64);
                let variance = if n_flnc > 1 {
                    let sq_diff_sum: f64 = log_lens.iter().map(|&x| (x - mean).powi(2)).sum();
                    sq_diff_sum / ((n_flnc - 1) as f64) // sample variance
                } else {
                    INSTRUMENTAL_PAL_NOISE_FLOOR
                };
                (mean, variance.max(INSTRUMENTAL_PAL_NOISE_FLOOR))
            } else {
                // State 2/3 -> emit placeholder mu=0 and floored sigma.
                (0.0, INSTRUMENTAL_PAL_NOISE_FLOOR)
            };

            tr_id_builder.append_value(tr_id);
            gene_id_builder.append_value(gene_id);
            frac_builder.append_value(poly_a_frac);
            competent_builder.append_value(is_competent);
            count_builder.append_value(n_flnc);
            mu_builder.append_value(mu);
            sigma2_builder.append_value(sigma2);
        }
    }

    // Materialise output batch.
    let schema = output_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(tr_id_builder.finish()),
            Arc::new(gene_id_builder.finish()),
            Arc::new(frac_builder.finish()),
            Arc::new(competent_builder.finish()),
            Arc::new(count_builder.finish()),
            Arc::new(mu_builder.finish()),
            Arc::new(sigma2_builder.finish()),
        ],
    )
    .context("Failed to build output RecordBatch")?;

    // Write to disk.
    let out_file = File::create(&output_emission_path).with_context(|| {
        format!(
            "Failed to create output {:?}",
            output_emission_path.as_ref()
        )
    })?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(out_file, schema, Some(props))
        .context("Failed to create ArrowWriter")?;
    writer.write(&batch).context("Failed to write batch")?;
    writer.close().context("Failed to close writer")?;

    Ok(())
}

// =============================================================================
// Universal Mode (Mode B) — anchor_pool → emission model
// =============================================================================

/// Per-transcript accumulator of `log2(PAL)` values from the
/// anchor pool.
#[derive(Debug, Clone)]
struct TxAccumulator {
    gene_id: String,
    /// `log2(poly_a_length)` for every anchor read assigned to
    /// this transcript. (Anchor filter already requires
    /// `poly_a_length >= 10`, so values are finite positives.)
    log2_pals: Vec<f64>,
}

impl TxAccumulator {
    fn new(gene_id: String) -> Self {
        Self {
            gene_id,
            log2_pals: Vec::new(),
        }
    }
    fn len(&self) -> usize {
        self.log2_pals.len()
    }
}

/// Result of fitting a Gaussian in `log2(PAL)` space.
///
/// Both AIDRS and Universal modes produce identical shape:
/// `mu` is the arithmetic mean (L1) or median (L3) of
/// `log2(PAL)`; `sigma2` is the sample variance (L1/L2) or
/// `(MAD * 1.4826)^2` (L3), floored at `INSTRUMENTAL_PAL_NOISE_FLOOR`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct LogNormalFit {
    mu: f64,
    sigma2: f64,
}

/// Fit `(mean, sample variance floored at INSTRUMENTAL_PAL_NOISE_FLOOR)`
/// from a slice of `log2(PAL)` values. Returns `None` iff
/// `values` is empty.
///
/// The result is a function of the VALUES, not of the order they arrive in.
/// `f64` addition is not associative, so the sum of a multiset depends on the
/// order it is accumulated in, and the L2 gene pool is assembled by iterating
/// `tx_accums` -- a `HashMap`, whose iteration order is a function of the
/// insertion history and the per-process random seed. Summing in that order
/// made the same anchor pool emit L2 rows differing in the last 1-2 mantissa
/// bits between runs (~1e-16 relative, measured over 12 runs; L1 and L3 were
/// already bit-reproducible). Sorting first costs microseconds on a few
/// thousand values and removes the seed from the result. Pinned by
/// `test_l2_gene_fit_is_bit_identical_across_accumulator_insertion_orders`.
fn fit_log_normal(values: &[f64]) -> Option<LogNormalFit> {
    let n = values.len();
    if n == 0 {
        return None;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        let sq_diff_sum: f64 = sorted.iter().map(|&x| (x - mean).powi(2)).sum();
        sq_diff_sum / ((n - 1) as f64) // Bessel-corrected sample variance
    } else {
        INSTRUMENTAL_PAL_NOISE_FLOOR
    };
    Some(LogNormalFit {
        mu: mean,
        sigma2: variance.max(INSTRUMENTAL_PAL_NOISE_FLOOR),
    })
}

/// Cohort-level robust fit: `mu` = median(log2(PAL)),
/// `sigma2` = `(MAD * 1.4826)^2`, floored at
/// `INSTRUMENTAL_PAL_NOISE_FLOOR`. Returns `None` iff `values` is empty.
///
/// Order-independent by construction, and for a different reason than
/// `fit_log_normal` needs its sort: both statistics are ORDER STATISTICS read
/// off a sorted copy, and nothing here is summed, so no accumulation order
/// reaches the result. The sort is load-bearing -- do not remove it as a
/// micro-optimisation.
fn cohort_log_normal_fit(values: &[f64]) -> Option<LogNormalFit> {
    let n = values.len();
    if n == 0 {
        return None;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    };

    let mut abs_devs: Vec<f64> = sorted.iter().map(|&x| (x - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if abs_devs.len() % 2 == 1 {
        abs_devs[abs_devs.len() / 2]
    } else {
        0.5 * (abs_devs[abs_devs.len() / 2 - 1] + abs_devs[abs_devs.len() / 2])
    };

    let sigma = MAD_TO_SIGMA * mad;
    let sigma2 = (sigma * sigma).max(INSTRUMENTAL_PAL_NOISE_FLOOR);
    Some(LogNormalFit { mu: median, sigma2 })
}

/// Load the set of `transcript_id`s with `em_eligible == true`
/// from `telescope.transcript_evidence.parquet` (Module 0
/// output, F5.3 schema).
///
/// This is the entry point for Universal Mode's optional
/// `em_eligible` filtering (resolver F6.3 + cohort pre-filter).
pub fn load_em_eligible_set<P: AsRef<Path>>(path: P) -> Result<HashSet<String>> {
    let file = File::open(&path).with_context(|| format!("Failed to open {:?}", path.as_ref()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to build ParquetRecordBatchReaderBuilder for transcript_evidence")?;
    let mut reader = builder.with_batch_size(8192).build()?;

    let mut set: HashSet<String> = HashSet::new();
    while let Some(batch_result) = reader.next() {
        let batch = batch_result.context("Failed to read transcript_evidence batch")?;
        let tx_col = batch
            .column_by_name("transcript_id")
            .context("Missing 'transcript_id' column in transcript_evidence")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Failed to downcast 'transcript_id' to StringArray")?;
        let em_col = batch
            .column_by_name("em_eligible")
            .context("Missing 'em_eligible' column in transcript_evidence")?
            .as_any()
            .downcast_ref::<BooleanArray>()
            .context("Failed to downcast 'em_eligible' to BooleanArray")?;
        for i in 0..batch.num_rows() {
            // O(1) per row: `value(i)` on a non-null BooleanArray is a
            // direct bit read. Avoided `iter().nth(i)` which was O(i)
            // per row → O(n²) total. `em_eligible` is non-null per F5.3
            // schema; `value()` panics on null which is the intended
            // contract for schema-conformant inputs.
            if em_col.value(i) {
                set.insert(tx_col.value(i).to_string());
            }
        }
    }
    Ok(set)
}

/// Stream `telescope.anchor_pool.parquet` (anchor_builder
/// output schema) into per-transcript accumulators + a flat
/// cohort-level `log2(PAL)` vector.
fn read_anchor_pool_parquet<P: AsRef<Path>>(
    path: P,
    tx_accums: &mut HashMap<String, TxAccumulator>,
    cohort_log2_pals: &mut Vec<f64>,
) -> Result<()> {
    let file = File::open(&path).with_context(|| format!("Failed to open {:?}", path.as_ref()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to build ParquetRecordBatchReaderBuilder for anchor_pool")?;
    let mut reader = builder.with_batch_size(8192).build()?;

    while let Some(batch_result) = reader.next() {
        let batch = batch_result.context("Failed to read anchor_pool batch")?;

        let tx_col = batch
            .column_by_name("transcript_id")
            .context("Missing 'transcript_id' column in anchor_pool")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Failed to downcast 'transcript_id' to StringArray")?;
        let gene_col = batch
            .column_by_name("gene_id")
            .context("Missing 'gene_id' column in anchor_pool")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Failed to downcast 'gene_id' to StringArray")?;
        let pal_col = batch
            .column_by_name("poly_a_length")
            .context("Missing 'poly_a_length' column in anchor_pool")?
            .as_any()
            .downcast_ref::<Int32Array>()
            .context("Failed to downcast 'poly_a_length' to Int32Array")?;

        for i in 0..batch.num_rows() {
            let tx_id = tx_col.value(i).to_string();
            let gene_id = gene_col.value(i).to_string();
            let pal = pal_col.value(i);
            let log2_pal = (pal as f64).log2();
            // `or_insert_with` only fires when the key is new,
            // so we capture `gene_id` only on first sight of each tx.
            let entry = tx_accums
                .entry(tx_id)
                .or_insert_with(|| TxAccumulator::new(gene_id));
            entry.log2_pals.push(log2_pal);
            cohort_log2_pals.push(log2_pal);
        }
    }
    Ok(())
}

/// How many transcripts each level of the Tx → Gene → Cohort fallback
/// supplied a fit for.
///
/// This exists because the split is a RESULT, not a log line. It was computed
/// correctly all along and printed to stderr, but nothing outside
/// `build_emission_from_anchors` could read it, so `n_l1 += 1` -> `n_l1 += 0`
/// left the whole test suite green while changing the number a reader of the
/// run compares across arms. The counters are returned so that both the CLI and
/// a test read the same value from the same place.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FallbackTierCounts {
    /// L1: `n_reads_tx >= fallback.tx_min_reads` — per-transcript fit.
    pub l1: usize,
    /// L2: pooled gene reads `>= fallback.gene_min_reads` — the shared gene fit.
    pub l2: usize,
    /// L3: cohort-level fit (always available when ≥1 read).
    pub l3: usize,
    /// Rows IMPUTED to reach the reference universe: a transcript named by
    /// `@SQ` that the anchor pool never observed. Zero unless a
    /// `reference_universe` was supplied.
    ///
    /// This is deliberately NOT folded into `l3`. Folding it there would keep
    /// `l1 + l2 + l3 == rows.len()` true, but it would also silently rebase
    /// the published tier percentages onto a 70,511-row denominator, turning
    /// a previously printed `L3=17327 (51.8%)` into `(71.5%)` with no flag and
    /// no doc change. Keeping it separate means the L1/L2/L3 percentages keep
    /// describing the ANCHORED population — comparable with every earlier run —
    /// and a reader who wants the whole table subtracts `n_imputed`.
    ///
    /// The partition invariant is therefore four terms, not three:
    /// `l1 + l2 + l3 + n_imputed == rows.len()`. Pinned by
    /// `test_l1_l2_l3_tier_attribution_and_split` and by
    /// `test_padded_rows_are_cohort_fits_not_a_fourth_tier`.
    pub n_imputed: usize,
    /// Transcripts named by `reference_universe`. Zero when none was supplied.
    pub n_reference: usize,
    /// Anchored transcripts that the reference universe does NOT name.
    ///
    /// Non-zero means the model and the alignment reference disagree about
    /// which transcripts exist, which makes the universe the wrong authority
    /// and the emitted row count NOT equal to `n_reference`. The CLI warns
    /// rather than silently reporting a `total` that is not the universe size.
    pub n_anchored_not_in_reference: usize,
    /// Absolute path of the per-transcript provenance sidecar just written.
    ///
    /// Returned rather than only logged: the sidecar is a product of this
    /// function, and a product whose path only exists on stderr cannot be
    /// opened by the test that is supposed to check it.
    pub provenance_path: std::path::PathBuf,
}

/// The authoritative reference universe: every `@SQ SN` in an alignment
/// BAM's header, in header order.
///
/// WHY A BAM, AND WHY IT IS THE RIGHT AUTHORITY
/// ------------------------------------------
/// The emission model is keyed by the transcripts an alignment can name, so
/// the set of transcripts the alignment CAN name is the only universe that
/// makes a candidate evaluable. `tx2gene` is not a universe (it maps
/// transcripts that exist; it does not say which exist), and a hand-kept list
/// drifts from the reference it claims to describe. HC2 measured
/// `n(model) - n(@SQ)` drift at exactly 0, which is what makes this
/// checkable rather than merely defensible.
///
/// The BAM is read HEADER-ONLY. No record is touched, so this costs a few
/// kilobytes of I/O on a multi-gigabyte file.
pub fn read_reference_universe<P: AsRef<Path>>(bam_path: P) -> Result<Vec<String>> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(bam_path.as_ref())
        .with_context(|| format!("Failed to open reference BAM: {:?}", bam_path.as_ref()))?;
    let mut reader = noodles_bam::io::Reader::new(BufReader::new(file));
    let header = reader
        .read_header()
        .with_context(|| format!("Failed to read BAM header: {:?}", bam_path.as_ref()))?;

    // `reference_sequences()` is keyed by ReferenceSequenceId and its
    // iteration order matches the `@SQ` lines, which is what makes the result
    // deterministic. Names are NOT de-duplicated: SAM requires `@SQ SN` to be
    // unique, so a duplicate means a malformed file, and silently collapsing
    // one would turn a corrupt reference into a plausible-looking universe.
    Ok((0..header.reference_sequences().len())
        .map(|i| match header.reference_sequences().get_index(i) {
            Some((name, _)) => name.to_string(),
            None => String::new(),
        })
        .filter(|n| !n.is_empty())
        .collect())
}

/// The four provenance tiers, exactly as written into
/// `<stem>.provenance.parquet`'s `tier` column.
///
/// WHY THESE FOUR AND NOT THREE. The L1/L2/L3 cascade has three levels, and
/// an IMPUTED row is not one of them: it has no anchors at all, so it never
/// entered the cascade. Folding it into `L3` would make "L3" mean both "this
/// transcript's own gene pool was too small" and "this transcript was never
/// observed", which are different facts with different consequences downstream.
/// `L3_imputed` and `L3_anchored` are therefore separate enum values even
/// though both carry the cohort fit.
pub const TIER_L1: &str = "L1";
pub const TIER_L2: &str = "L2";
pub const TIER_L3_ANCHORED: &str = "L3_anchored";
pub const TIER_L3_IMPUTED: &str = "L3_imputed";

/// Schema of the per-transcript provenance sidecar: exactly four columns.
///
/// Deliberately NOT a column of `telescope.emission.parquet`. That schema is
/// frozen at seven columns, and the question this table answers — which
/// fallback level produced this row's `(mu, sigma2)` — is not answerable from
/// `n_cohort_flnc` alone, because L2 is decided by the GENE's pooled count
/// while the parquet only carries the TRANSCRIPT's own count. The
/// discriminators have to live outside the frozen file.
pub fn provenance_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("transcript_id", DataType::Utf8, false),
        Field::new("gene_id", DataType::Utf8, false),
        Field::new("tier", DataType::Utf8, false),
        Field::new("n_anchor", DataType::UInt32, false),
    ]))
}

/// Write the per-transcript provenance sidecar next to the emission parquet.
///
/// `rows` MUST already be sorted strictly ascending by `transcript_id`, and
/// MUST be the same key set as the emission file, in the same order. Both
/// invariants hold by construction at the call site in
/// [`build_emission_from_anchors`], where the row loop iterates one sorted
/// `tx_keys` vector and appends to both builders in lockstep.
///
/// The file name is derived from `emission_path` by replacing `.parquet`
/// with `.provenance.parquet`, so `<stem>.emission.parquet` produces
/// `<stem>.emission.provenance.parquet`. The path is RETURNED as well as
/// written, because a path that only reaches stderr is a path no test can
/// open.
pub fn write_provenance_parquet<P: AsRef<Path>>(
    emission_path: P,
    rows: &[(String, String, &'static str, u32)],
) -> Result<std::path::PathBuf> {
    let emission_path = emission_path.as_ref();
    let out_path = emission_path.with_extension("provenance.parquet");

    let mut tx_b = StringBuilder::new();
    let mut gene_b = StringBuilder::new();
    let mut tier_b = StringBuilder::new();
    let mut n_b = UInt32Builder::new();
    for (tx, gene, tier, n) in rows {
        tx_b.append_value(tx);
        gene_b.append_value(gene);
        tier_b.append_value(*tier);
        n_b.append_value(*n);
    }

    let schema = provenance_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(tx_b.finish()),
            Arc::new(gene_b.finish()),
            Arc::new(tier_b.finish()),
            Arc::new(n_b.finish()),
        ],
    )
    .context("Failed to build provenance RecordBatch")?;
    let out_file = File::create(&out_path)
        .with_context(|| format!("Failed to create provenance parquet {:?}", out_path))?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(out_file, schema, Some(props))
        .context("Failed to create provenance ArrowWriter")?;
    writer
        .write(&batch)
        .context("Failed to write provenance batch")?;
    writer
        .close()
        .context("Failed to close provenance writer")?;
    Ok(out_path)
}

/// Build `telescope.emission.parquet` from a Universal Mode
/// `anchor_pool.parquet` (per-read rows).
///
/// Returns the L1/L2/L3 split — see [`FallbackTierCounts`].
///
/// Algorithm (per spec 常温区 Step 3 Part 2):
/// 1. Read `anchor_pool.parquet` (per-read `poly_a_length`,
///    `gene_id`, `transcript_id`).
/// 2. If `transcript_evidence_path` is provided, drop reads
///    whose `transcript_id` has `em_eligible == false`.
/// 3. Compute cohort-level robust fit (median + MAD) across
///    all surviving reads.
/// 4. For each transcript, pick the best fit:
///    - L1 Tx-level if `n_reads_tx >= fallback.tx_min_reads`.
///    - L2 Gene-level if pooled gene reads
///      `>= fallback.gene_min_reads`. The same `(mu, sigma2)`
///      is shared across all underpowered transcripts in the
///      same gene.
///    - L3 Cohort-level fit (always available when ≥1 read).
/// 5. `polyA_frac = 1.0` by construction (anchor filter
///    requires `pt:i >= 10`); `is_polyA_competent = true` for
///    every non-empty transcript. See module docstring.
pub fn build_emission_from_anchors<P, Q>(
    anchor_pool_path: P,
    transcript_evidence_path: Option<Q>,
    output_path: P,
    fallback: &FallbackConfig,
    reference_universe: Option<&[String]>,
    tx_to_gene: Option<&TxToGene>,
) -> Result<FallbackTierCounts>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
{
    // 1. Read anchor pool.
    let mut tx_accums: HashMap<String, TxAccumulator> = HashMap::new();
    let mut cohort_log2_pals: Vec<f64> = Vec::new();
    read_anchor_pool_parquet(
        anchor_pool_path.as_ref(),
        &mut tx_accums,
        &mut cohort_log2_pals,
    )?;

    // 2. Optional em_eligible filter.
    //
    // `em_suppressed` records what the filter removed, BY MOVE -- the loop
    // consumes `tx_accums`, so cloning the dropped keys would cost a clone per
    // ineligible transcript. It is consumed later, when the reference universe
    // pads the row set: a suppressed transcript that the universe names is
    // still emitted, but as an INCOMPETENT row rather than a competent one.
    // Dropping the row instead would leave `rows.len() < n_reference` with no
    // counter able to see it, and would report the transcript as `n_missing`
    // in the candidate ledger -- whose documented question is "would padding
    // have saved this read?" (em.rs `CandidateAvailability`) -- when the truth
    // is that the suppression was deliberate. See the match arm below.
    let mut em_suppressed: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Some(te_path) = transcript_evidence_path {
        let em_set = load_em_eligible_set(&te_path)?;
        let mut filtered_accums: HashMap<String, TxAccumulator> = HashMap::new();
        let mut filtered_log2: Vec<f64> = Vec::new();
        for (tx, acc) in tx_accums {
            if em_set.contains(&tx) {
                for &v in &acc.log2_pals {
                    filtered_log2.push(v);
                }
                filtered_accums.insert(tx, acc);
            } else {
                em_suppressed.insert(tx);
            }
        }
        tx_accums = filtered_accums;
        cohort_log2_pals = filtered_log2;
    }

    if cohort_log2_pals.is_empty() {
        bail!(
            "No anchor reads remain after em_eligible filtering; \
             cannot compute cohort-level fallback"
        );
    }

    // 3. Cohort-level fit (always computed; used as L3 fallback).
    let cohort_fit = cohort_log_normal_fit(&cohort_log2_pals)
        .context("cohort_log_normal_fit failed despite non-empty input")?;

    // Informational only: spec says L3 is always available, but
    // log a warning when the cohort is suspiciously small.
    if cohort_log2_pals.len() < fallback.cohort_min_reads {
        eprintln!(
            "[telescope][INFO] cohort reads ({}) < cohort_min_reads ({}); \
             L3 fallback applied anyway",
            cohort_log2_pals.len(),
            fallback.cohort_min_reads
        );
    }

    // 4. Per-gene pool + fit (used as L2 fallback).
    let mut gene_pool: HashMap<String, Vec<f64>> = HashMap::new();
    for acc in tx_accums.values() {
        gene_pool
            .entry(acc.gene_id.clone())
            .or_default()
            .extend_from_slice(&acc.log2_pals);
    }
    let mut gene_fit: HashMap<String, LogNormalFit> = HashMap::new();
    for (gene_id, vals) in gene_pool {
        if vals.len() >= fallback.gene_min_reads {
            if let Some(fit) = fit_log_normal(&vals) {
                gene_fit.insert(gene_id, fit);
            }
        }
    }

    // 5+6. Per-transcript fit + output row assembly.
    let mut tx_id_builder = StringBuilder::new();
    let mut gene_id_builder = StringBuilder::new();
    let mut frac_builder = Float64Builder::new();
    let mut competent_builder = BooleanBuilder::new();
    let mut count_builder = UInt32Builder::new();
    let mut mu_builder = Float64Builder::new();
    let mut sigma2_builder = Float64Builder::new();
    // Provenance sidecar, built in lockstep with the emission row below. It is
    // a SEPARATE FILE, not an eighth column: the 7-column emission schema is
    // frozen, and `n_cohort_flnc == 0` is the only in-schema marker of an
    // imputed row, so the fallback LEVEL that produced an ANCHORED row's
    // `(mu, sigma2)` is not recoverable from the parquet itself.
    let mut tier_builder = StringBuilder::new();

    // THE EMITTED KEY SET. Normally this is the anchor pool's transcripts and
    // nothing else, which is precisely the defect that padding exists to fix:
    // a read whose candidate is absent from the table cannot be evaluated, so
    // it is dropped with no profile row. With a reference universe the set
    // becomes the UNION, and a universe key with no anchors becomes an imputed
    // row.
    //
    // A universe key that DOES have anchors is reached through `tx_accums`
    // and therefore takes the anchored path — the union makes a double emit
    // structurally impossible rather than something a test has to catch.
    let mut tx_keys: Vec<&String> = tx_accums.keys().collect();
    let mut imputed_keys: std::collections::HashSet<&String> = std::collections::HashSet::new();
    let mut n_reference = 0usize;
    let mut n_anchored_not_in_reference = 0usize;
    if let Some(universe) = reference_universe {
        n_reference = universe.len();
        let anchored: std::collections::HashSet<&String> = tx_accums.keys().collect();
        for tx in universe {
            if anchored.contains(tx) {
                continue;
            }
            imputed_keys.insert(tx);
            tx_keys.push(tx);
        }
        n_anchored_not_in_reference = anchored.iter().filter(|tx| !universe.contains(tx)).count();
    }

    // Deterministic output ordering: sort by transcript_id. The union adds
    // keys that were not in the map, so this sort is what makes the padded
    // output byte-stable across runs and HashMap iteration orders.
    tx_keys.sort();

    let mut n_l1 = 0usize;
    let mut n_l2 = 0usize;
    let mut n_l3 = 0usize;

    // Captured before the loop consumes the vector: the emitted row count is
    // the union's size, and it is what the run header must report once a
    // reference universe makes it differ from `tx_accums.len()`.
    let n_rows_emitted = tx_keys.len();

    // The same rows, in the same order, as a plain Vec. This is the ONLY place
    // the two tables are kept in step, which is why the sidecar is written
    // from here rather than re-derived by re-reading the parquet back: a
    // re-read would be a second code path that could disagree with the first
    // and there would be nothing to catch it.
    let mut provenance_rows: Vec<(String, String, &'static str, u32)> =
        Vec::with_capacity(n_rows_emitted);

    for tx_id in tx_keys {
        // IMPUTED ROW. Reached only for a universe key with no entry in
        // `tx_accums`, which the union above guarantees. Two distinct reasons
        // land here, and they are NOT the same row:
        //
        //   (a) never had anchors -- `is_polyA_competent = true`. That is the
        //       load-bearing choice, and it is NOT a claim this transcript was
        //       measured to have a poly(A) tail; it says the row is evaluable.
        //       The F3 axiom isolates transcripts with NO poly(A), and a
        //       transcript absent from the anchor pool has not been shown to be
        //       one. HC2 measured `pt >= 10` on 98.23% of the reads landing on
        //       exactly this set, so the isolated case is not what is padded.
        //
        //   (b) HAD anchors and was removed by the `em_eligible` filter --
        //       `is_polyA_competent = false`. Marking this row true would let a
        //       transcript the resolver deliberately suppressed win posterior
        //       mass, which is the bug this arm exists to close: the reads of a
        //       TierC transcript (zero poly(A)-compatible reads BY CONSTRUCTION,
        //       `resolver::classify_tier`) or of a TierB transcript superseded by
        //       its TierA isoform would stay attributed to it instead of to the
        //       transcript that supersedes it.
        //
        // `false` is not a demotion, it is the E-step's own switch: an
        // incompetent candidate is pushed as `-inf` and cannot win any mass, so
        // the row is inert while still present. Keeping the row is what keeps
        // `rows.len() == n_reference` and routes the candidate to the ledger's
        // `n_incompetent` bucket -- whose question is "where did this candidate
        // go?" -- rather than to `n_missing`, whose question is "would padding
        // have saved this read?" (em.rs `CandidateAvailability`).
        //
        // In both cases the fit is the cohort fit and `n_cohort_flnc = 0` is the
        // in-schema marker that says "imputed, not observed" — the 7-column
        // schema is frozen and gets no eighth column for this.
        let (gene_id, poly_a_frac, is_poly_a_competent, n_obs, fit, tier) =
            match tx_accums.get(tx_id.as_str()) {
                None if em_suppressed.contains(tx_id.as_str()) => {
                    let gene = tx_to_gene
                        .and_then(|m| m.get(tx_id).cloned())
                        .unwrap_or_else(|| (*tx_id).clone());
                    (
                        gene,
                        1.0_f64,
                        false,
                        0_u32,
                        LogNormalFit {
                            mu: cohort_fit.mu,
                            sigma2: cohort_fit.sigma2,
                        },
                        TIER_L3_IMPUTED,
                    )
                }
                None => {
                    let gene = tx_to_gene
                        .and_then(|m| m.get(tx_id).cloned())
                        .unwrap_or_else(|| (*tx_id).clone());
                    (
                        gene,
                        1.0_f64,
                        true,
                        0_u32,
                        LogNormalFit {
                            mu: cohort_fit.mu,
                            sigma2: cohort_fit.sigma2,
                        },
                        TIER_L3_IMPUTED,
                    )
                }
                Some(acc) => {
                    let n = acc.len();
                    // Universal-Mode polyA_frac: anchor filter enforces
                    // pt:i >= 10, so by construction this is 1.0 for any
                    // non-empty transcript. Documented in module docstring.
                    let (fit, tier) = if n >= fallback.tx_min_reads {
                        n_l1 += 1;
                        // L1: per-tx fit. `n >= tx_min_reads >= 1` implies
                        // `log2_pals` is non-empty, so unwrap is safe.
                        (
                            fit_log_normal(&acc.log2_pals)
                                .expect("L1 fit must succeed when n >= tx_min_reads >= 1"),
                            TIER_L1,
                        )
                    } else if let Some(gf) = gene_fit.get(&acc.gene_id) {
                        n_l2 += 1;
                        // L2: per-gene fit. Shared across all underpowered txs in gene.
                        (*gf, TIER_L2)
                    } else {
                        n_l3 += 1;
                        // L3: cohort-level fallback (always available).
                        (cohort_fit, TIER_L3_ANCHORED)
                    };
                    (acc.gene_id.clone(), 1.0_f64, n > 0, n as u32, fit, tier)
                }
            };

        tx_id_builder.append_value(tx_id);
        gene_id_builder.append_value(&gene_id);
        frac_builder.append_value(poly_a_frac);
        competent_builder.append_value(is_poly_a_competent);
        // `n_cohort_flnc` := the number of anchor OBSERVATIONS for this
        // transcript, NOT a count of distinct poly(A) lengths and NOT a
        // cohort-fallback count. Two reads carrying the same `poly_a_length`
        // are two anchor rows and this value is 2.
        //
        // WHICH FALLBACK LEVEL produced the row's `(mu, sigma2)` is not
        // recorded here; it is inferred by comparing `n` against
        // `tx_min_reads` / `gene_min_reads`. Pinned by
        // `test_n_cohort_flnc_counts_observations_not_distinct_lengths`.
        //
        // `0` NOW HAS A SECOND MEANING, and that is deliberate: it marks an
        // IMPUTED row, emitted to reach the reference universe for a
        // transcript the anchor pool never observed. Before reference-universe
        // padding, `0` was unreachable in this file -- a transcript with no
        // anchors had no row at all -- so nothing downstream could have been
        // relying on it. Inside the parquet an imputed row is
        // indistinguishable from a transcript observed zero times; that is
        // the accepted cost of the frozen 7-column schema, and the
        // discriminators live outside it (`FallbackTierCounts::n_imputed` and
        // the CLI's provenance sidecar).
        count_builder.append_value(n_obs);
        mu_builder.append_value(fit.mu);
        sigma2_builder.append_value(fit.sigma2);
        tier_builder.append_value(tier);
        provenance_rows.push((tx_id.clone(), gene_id.clone(), tier, n_obs));
    }

    let schema = output_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(tx_id_builder.finish()),
            Arc::new(gene_id_builder.finish()),
            Arc::new(frac_builder.finish()),
            Arc::new(competent_builder.finish()),
            Arc::new(count_builder.finish()),
            Arc::new(mu_builder.finish()),
            Arc::new(sigma2_builder.finish()),
        ],
    )
    .context("Failed to build Universal-Mode output RecordBatch")?;

    let out_file = File::create(&output_path)
        .with_context(|| format!("Failed to create output {:?}", output_path.as_ref()))?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(out_file, schema, Some(props))
        .context("Failed to create ArrowWriter")?;
    writer.write(&batch).context("Failed to write batch")?;
    writer.close().context("Failed to close writer")?;

    // The L1/L2/L3 COUNTS are what the fallback cascade chose; the FRACTIONS
    // are what it means for the run, and the two answer different questions.
    // A run that is 80% L1 and a run that is 80% L3 both print a valid triple,
    // but the second is a run in which most transcripts carry no
    // transcript-specific poly(A) distribution at all — the cohort fit is
    // standing in for them. The denominator is the number of output rows, and
    // the caller formats it: the split is RETURNED (see `FallbackTierCounts`)
    // rather than formatted here, because a number that only reaches stderr is
    // a number no test can read.
    //
    // The run header stays here because only this function knows how many
    // cohort reads survived the `em_eligible` filter. The THIRD number is the
    // emitted row count: it is NOT `tx_accums.len()` once a reference universe
    // is supplied, and printing the anchored count alone beside a padded table
    // would state a row count the file does not have.
    eprintln!(
        "[telescope]  Universal mode: {} anchored transcripts, {} cohort reads, \
         {} rows emitted",
        tx_accums.len(),
        cohort_log2_pals.len(),
        n_rows_emitted,
    );

    // Written AFTER the emission parquet so that a failure here cannot leave a
    // provenance file describing a model that was never emitted.
    let provenance_path = write_provenance_parquet(output_path.as_ref(), &provenance_rows)?;

    Ok(FallbackTierCounts {
        l1: n_l1,
        l2: n_l2,
        l3: n_l3,
        n_imputed: imputed_keys.len(),
        n_reference,
        n_anchored_not_in_reference,
        provenance_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Builder, ListBuilder, UInt32Array};
    use std::path::PathBuf;

    /// A private scratch directory that no other process can collide with.
    ///
    /// Two things are wrong with `temp_dir().join("telescope_test_<x>")`:
    ///
    /// 1. `$TMPDIR` here is `/datg/hanxi/tmp`, which is **nfs4**. A file
    ///    written and immediately re-read under concurrent I/O can come back
    ///    0 bytes. That already flaked a different test in this crate once
    ///    (`EOF: Parquet file too small`); the same helper exists in `em.rs`
    ///    as `local_scratch`.
    /// 2. A FIXED name is shared by every concurrent `cargo test`, and each
    ///    test `remove_dir_all`s its directory at the START. Two runs
    ///    therefore delete each other's fixtures: measured on 2026-10-06 as
    ///    `open provenance parquet: No such file or directory`, in a test
    ///    that had passed minutes earlier in the same day. A test that is
    ///    green once and red once is worse than one that is red always,
    ///    because it trains the reader to distrust the suite.
    ///
    /// Local xfs `/tmp` plus the PID fixes both: distinct per process, and not
    /// on the network filesystem.
    fn local_scratch(name: &str) -> PathBuf {
        PathBuf::from("/tmp").join(format!("telescope_em_{}_{}", name, std::process::id()))
    }

    #[test]
    fn min_cohort_variance_value() {
        // Pinned in spec F3 / spec-self-audit-v0.1.3 R1.
        assert!((INSTRUMENTAL_PAL_NOISE_FLOOR - 0.04).abs() < f64::EPSILON);
    }

    #[test]
    fn output_schema_has_seven_columns() {
        let s = output_schema();
        assert_eq!(s.fields().len(), 7);
        let names: Vec<&str> = s.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec![
                "transcript_id",
                "gene_id",
                "polyA_frac",
                "is_polyA_competent",
                "n_cohort_flnc",
                "mu_logPAL",
                "sigma2_logPAL",
            ]
        );
    }

    // ===================================================================
    // Universal Mode test fixtures + the 3 mandated tests
    // ===================================================================

    /// Logical fixture row shared across all 3 Universal-Mode tests.
    /// Each `(gene_id, tx_id)` may appear at most once.
    struct TxFixture {
        gene_id: String,
        tx_id: String,
        pals: Vec<i32>,
    }

    /// Write an AIDRS-format `aidrs.transcript_polyA_len.parquet`
    /// (3-column subset) from a list of `TxFixture`.
    ///
    /// This is the fixture writer for the DEFERRED external poly(A)
    /// initializer, and it is RETAINED rather than deleted: it is what
    /// `test_aidrs_universal_equivalence` feeds `build_emission_parquet`
    /// with, and that test is required to keep running. Deleting the
    /// helper would delete the test with it.
    fn write_aidrs_parquet(path: &Path, fxs: &[TxFixture]) {
        let mut tx_b = StringBuilder::new();
        let mut gene_b = StringBuilder::new();
        let mut list_b = ListBuilder::new(Int32Builder::new());

        for fx in fxs {
            tx_b.append_value(&fx.tx_id);
            gene_b.append_value(&fx.gene_id);
            let inner = list_b.values();
            for &pal in &fx.pals {
                inner.append_value(pal);
            }
            list_b.append(true);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("TrID", DataType::Utf8, false),
            Field::new("GeneID", DataType::Utf8, false),
            Field::new(
                "raw_polyA_lengths",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                false,
            ),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(tx_b.finish()),
                Arc::new(gene_b.finish()),
                Arc::new(list_b.finish()),
            ],
        )
        .expect("build AIDRS test fixture batch");
        let out = File::create(path).expect("create AIDRS test parquet");
        let props = WriterProperties::builder().build();
        let mut w = ArrowWriter::try_new(out, schema, Some(props))
            .expect("init ArrowWriter for AIDRS fixture");
        w.write(&batch).expect("write AIDRS fixture batch");
        w.close().expect("close AIDRS fixture writer");
    }

    /// Write an AIDRS-format assessment.tsv (`TrID\tpolyA_frac`).
    /// All entries get `polyA_frac = 1.0` (State 1).
    ///
    /// RETAINED, not deleted: fixture writer for the DEFERRED external
    /// poly(A) initializer, used only by
    /// `test_aidrs_universal_equivalence`. See `write_aidrs_parquet`.
    fn write_assessment_tsv(path: &Path, fxs: &[TxFixture]) {
        use std::fmt::Write as _;
        let mut s = String::from("TrID\tpolyA_frac\n");
        for fx in fxs {
            writeln!(s, "{}\t1.0", fx.tx_id).unwrap();
        }
        std::fs::write(path, s).expect("write assessment.tsv fixture");
    }

    /// Write a Universal-Mode `anchor_pool.parquet` from the
    /// same logical fixtures: one row per `(tx_id, pal)`.
    fn write_anchor_pool_parquet(path: &Path, fxs: &[TxFixture]) {
        let mut sample_b = StringBuilder::new();
        let mut read_b = StringBuilder::new();
        let mut gene_b = StringBuilder::new();
        let mut tx_b = StringBuilder::new();
        let mut pal_b = Int32Builder::new();
        let mut weight_b = Float64Builder::new();

        for fx in fxs {
            for (i, &pal) in fx.pals.iter().enumerate() {
                sample_b.append_value("sample_1");
                read_b.append_value(format!("{}_r{}", fx.tx_id, i));
                gene_b.append_value(&fx.gene_id);
                tx_b.append_value(&fx.tx_id);
                pal_b.append_value(pal);
                weight_b.append_value(1.0_f64);
            }
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
                Arc::new(weight_b.finish()),
            ],
        )
        .expect("build anchor_pool test fixture batch");
        let out = File::create(path).expect("create anchor_pool test parquet");
        let props = WriterProperties::builder().build();
        let mut w = ArrowWriter::try_new(out, schema, Some(props))
            .expect("init ArrowWriter for anchor_pool fixture");
        w.write(&batch).expect("write anchor_pool fixture batch");
        w.close().expect("close anchor_pool fixture writer");
    }

    /// Write a Module-0 `transcript_evidence.parquet` fixture
    /// containing only the 2 columns consumed by
    /// `load_em_eligible_set` (`transcript_id`, `em_eligible`).
    fn write_transcript_evidence_minimal(path: &Path, entries: &[(String, bool)]) {
        let mut tx_b = StringBuilder::new();
        let mut em_b = BooleanBuilder::new();
        for (tx, em) in entries {
            tx_b.append_value(tx);
            em_b.append_value(*em);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("transcript_id", DataType::Utf8, false),
            Field::new("em_eligible", DataType::Boolean, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(tx_b.finish()), Arc::new(em_b.finish())],
        )
        .expect("build transcript_evidence fixture batch");
        let out = File::create(path).expect("create transcript_evidence test parquet");
        let props = WriterProperties::builder().build();
        let mut w = ArrowWriter::try_new(out, schema, Some(props))
            .expect("init ArrowWriter for transcript_evidence fixture");
        w.write(&batch)
            .expect("write transcript_evidence fixture batch");
        w.close().expect("close transcript_evidence fixture writer");
    }

    /// Read `telescope.emission.parquet` back into a
    /// `Vec<(transcript_id, mu, sigma2, n_cohort_flnc, polyA_frac, is_polyA_competent)>`
    /// for assertions. Sorted by `transcript_id`.
    /// Read the provenance sidecar back as (transcript_id, gene_id, tier,
    /// n_anchor). Deliberately a SEPARATE reader from `read_emission_parquet`
    /// so the test is not comparing one in-memory vector against itself.
    fn read_provenance_parquet(path: &Path) -> Vec<(String, String, String, u32)> {
        let f = File::open(path).expect("open provenance parquet");
        let b = ParquetRecordBatchReaderBuilder::try_new(f).expect("init provenance reader");
        let reader = b.with_batch_size(8192).build().expect("build");
        let mut out = Vec::new();
        for batch in reader {
            let batch = batch.expect("provenance batch");
            let tx = batch
                .column_by_name("transcript_id")
                .expect("tx")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("tx str");
            let gene = batch
                .column_by_name("gene_id")
                .expect("gene")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("gene str");
            let tier = batch
                .column_by_name("tier")
                .expect("tier")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("tier str");
            let n = batch
                .column_by_name("n_anchor")
                .expect("n_anchor")
                .as_any()
                .downcast_ref::<UInt32Array>()
                .expect("n_anchor u32");
            for i in 0..batch.num_rows() {
                out.push((
                    tx.value(i).to_string(),
                    gene.value(i).to_string(),
                    tier.value(i).to_string(),
                    n.value(i),
                ));
            }
        }
        out
    }

    fn read_emission_parquet(path: &Path) -> Vec<(String, String, f64, bool, u32, f64, f64)> {
        let f = File::open(path).expect("open emission parquet");
        let b = ParquetRecordBatchReaderBuilder::try_new(f).expect("init reader");
        let mut reader = b.with_batch_size(8192).build().expect("build reader");
        let mut out = Vec::new();
        while let Some(batch) = reader.next() {
            let batch = batch.expect("read batch");
            let tx = batch
                .column_by_name("transcript_id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let gene = batch
                .column_by_name("gene_id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let frac = batch
                .column_by_name("polyA_frac")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap();
            let comp = batch
                .column_by_name("is_polyA_competent")
                .unwrap()
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap();
            let n = batch
                .column_by_name("n_cohort_flnc")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::UInt32Array>()
                .unwrap();
            let mu = batch
                .column_by_name("mu_logPAL")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap();
            let sigma2 = batch
                .column_by_name("sigma2_logPAL")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                out.push((
                    tx.value(i).to_string(),
                    gene.value(i).to_string(),
                    frac.value(i),
                    comp.value(i),
                    n.value(i),
                    mu.value(i),
                    sigma2.value(i),
                ));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Per spec 常温区 Step 3 Part 2 + final ruling: the
    /// statistical equivalence test between the external
    /// poly(A) initializer and Universal mode is the acceptance
    /// criterion for Step 3 Part 2.
    ///
    /// THIS TEST KEEPS RUNNING after the initializer lost its CLI
    /// surface on 2026-10-03. It is the live proof that
    /// `build_emission_parquet` (deferred, no CLI entry point
    /// reaches it) and `build_emission_from_anchors` (the only
    /// CLI-reachable path) AGREE — to <1e-9 relative on `mu_logPAL`
    /// and `sigma2_logPAL`, and EXACTLY on `n_cohort_flnc`. Do not
    /// convert it to prose and do not loosen its tolerances: it is
    /// the only executable evidence that the retained initializer
    /// is a real option rather than dead code, and the only thing
    /// pinning the fit is format-independent.
    ///
    /// Note what it does NOT show: that either path assigns reads
    /// to transcripts better than the other. Both consume the same
    /// (tx, pal) pairs here, so upstream read→transcript
    /// assignment — the actual open question — is held fixed and
    /// unmeasured. See the module docs on why the initializer is
    /// deferred.
    ///
    /// Test fixture: 3 transcripts, 100 reads each, all reads
    /// uniquely mapping to 1 transcript. Reads sampled from a
    /// log-normal in `log2(PAL)` space (synthetic, deterministic
    /// pal values). The SAME data is laid out in two formats
    /// (initializer-format list-per-tx vs Universal-format
    /// per-read rows) and processed by the two paths. Per-transcript
    /// `mu_logPAL`, `sigma2_logPAL`, and `n_cohort_flnc` MUST
    /// agree to machine precision. `polyA_frac` and
    /// `is_polyA_competent` are documented to differ conceptually
    /// (Universal mode is always State 1 by construction).
    #[test]
    fn test_aidrs_universal_equivalence() {
        // Deterministic fixture: 3 transcripts × 100 reads each.
        // Use pal values [a, a+1, ..., a+99] per tx — the actual
        // distribution shape is irrelevant; what matters is that
        // both AIDRS and Universal see IDENTICAL (tx, pal) pairs.
        let fxs = vec![
            TxFixture {
                gene_id: "gene_1".into(),
                tx_id: "tx_1".into(),
                pals: (50..150).collect::<Vec<i32>>(), // 50..149
            },
            TxFixture {
                gene_id: "gene_1".into(),
                tx_id: "tx_2".into(),
                pals: (30..130).collect::<Vec<i32>>(), // 30..129
            },
            TxFixture {
                gene_id: "gene_2".into(),
                tx_id: "tx_3".into(),
                pals: (70..170).collect::<Vec<i32>>(), // 70..169
            },
        ];

        let tmp = local_scratch("telescope_test_aidrs_universal_equiv");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir tmp");

        let aidrs_poly_a = tmp.join("polyA.parquet");
        let aidrs_assess = tmp.join("assessment.tsv");
        let anchor_pool = tmp.join("anchor_pool.parquet");
        let out_aidrs = tmp.join("emission_aidrs.parquet");
        let out_univ = tmp.join("emission_universal.parquet");

        write_aidrs_parquet(&aidrs_poly_a, &fxs);
        write_assessment_tsv(&aidrs_assess, &fxs);
        write_anchor_pool_parquet(&anchor_pool, &fxs);

        // Run both modes.
        build_emission_parquet(&aidrs_poly_a, &aidrs_assess, &out_aidrs).expect("AIDRS mode run");
        build_emission_from_anchors(
            &anchor_pool,
            None::<&str>,
            &out_univ,
            &FallbackConfig::default(),
            None,
            None,
        )
        .expect("Universal mode run");

        let rows_aidrs = read_emission_parquet(&out_aidrs);
        let rows_univ = read_emission_parquet(&out_univ);

        // Both outputs must have the same set of transcripts in
        // the same order (sort-by-tx_id on the read side).
        assert_eq!(rows_aidrs.len(), 3, "AIDRS: 3 transcript rows expected");
        assert_eq!(rows_univ.len(), 3, "Universal: 3 transcript rows expected");
        let aidrs_tx: Vec<&str> = rows_aidrs.iter().map(|r| r.0.as_str()).collect();
        let univ_tx: Vec<&str> = rows_univ.iter().map(|r| r.0.as_str()).collect();
        assert_eq!(aidrs_tx, univ_tx, "transcript_id sets must match");

        // Per-tx equivalence within machine precision.
        let tol = 1e-9_f64; // spec-mandated
        for (a, u) in rows_aidrs.iter().zip(rows_univ.iter()) {
            assert_eq!(a.0, u.0, "transcript_id order mismatch");
            assert_eq!(a.1, u.1, "gene_id must match for {}", a.0);
            assert_eq!(a.4, u.4, "n_cohort_flnc must match exactly for {}", a.0);
            let mu_rel = if a.5.abs() < 1e-300 {
                0.0
            } else {
                ((a.5 - u.5) / a.5).abs()
            };
            let s2_rel = if a.6.abs() < 1e-300 {
                0.0
            } else {
                ((a.6 - u.6) / a.6).abs()
            };
            assert!(
                mu_rel < tol,
                "mu_logPAL relative diff for {} = {:.3e} (AIDRS={:.16} Univ={:.16})",
                a.0,
                mu_rel,
                a.5,
                u.5
            );
            assert!(
                s2_rel < tol,
                "sigma2_logPAL relative diff for {} = {:.3e} (AIDRS={:.16} Univ={:.16})",
                a.0,
                s2_rel,
                a.6,
                u.6
            );
            // Documented conceptual diff (always State 1 in Universal).
            assert_eq!(u.2, 1.0, "Universal polyA_frac must be 1.0 by construction");
            assert!(
                u.3,
                "Universal is_polyA_competent must be true for non-empty tx"
            );
            assert_eq!(a.2, 1.0, "AIDRS polyA_frac = 1.0 (State 1)");
            assert!(a.3, "AIDRS is_polyA_competent = true (State 1)");
        }
    }

    /// Verify the 3-level fallback (Tx → Gene → Cohort) picks
    /// the correct level per transcript given the configured
    /// thresholds.
    #[test]
    fn test_3level_fallback_thresholds() {
        // T1: 50 reads in gene_A → L1 (n >= 10).
        // T2: 5 reads in gene_A → L2 (gene_A pool = 55 >= 30).
        // T3: 5 reads in gene_A → L2 (same gene as T2 → same fit as T2).
        // T4: 3 reads in gene_B → L3 (gene_B pool = 3 < 30 → cohort fallback).
        let fxs = vec![
            TxFixture {
                gene_id: "gene_A".into(),
                tx_id: "tx_1".into(),
                pals: (50..100).collect::<Vec<i32>>(), // 50 reads
            },
            TxFixture {
                gene_id: "gene_A".into(),
                tx_id: "tx_2".into(),
                pals: (200..205).collect::<Vec<i32>>(), // 5 reads
            },
            TxFixture {
                gene_id: "gene_A".into(),
                tx_id: "tx_3".into(),
                pals: (300..305).collect::<Vec<i32>>(), // 5 reads
            },
            TxFixture {
                gene_id: "gene_B".into(),
                tx_id: "tx_4".into(),
                pals: (400..403).collect::<Vec<i32>>(), // 3 reads
            },
        ];

        let tmp = local_scratch("telescope_test_3level_fallback");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir tmp");

        let anchor_pool = tmp.join("anchor_pool.parquet");
        let out = tmp.join("emission.parquet");
        write_anchor_pool_parquet(&anchor_pool, &fxs);

        let fb = FallbackConfig {
            tx_min_reads: 10,
            gene_min_reads: 30,
            cohort_min_reads: 50,
        };
        build_emission_from_anchors(&anchor_pool, None::<&str>, &out, &fb, None, None)
            .expect("Universal mode run");

        let rows = read_emission_parquet(&out);
        assert_eq!(rows.len(), 4);
        let by_tx: HashMap<&str, &(String, String, f64, bool, u32, f64, f64)> =
            rows.iter().map(|r| (r.0.as_str(), r)).collect();

        // L1: tx_1 uses its OWN reads.
        let r1 = by_tx["tx_1"];
        let expected_l1_vals: Vec<f64> = (50..100).map(|p| (p as f64).log2()).collect();
        let exp_mu_1 = expected_l1_vals.iter().sum::<f64>() / expected_l1_vals.len() as f64;
        let exp_var_1 = {
            let sq: f64 = expected_l1_vals
                .iter()
                .map(|&x| (x - exp_mu_1).powi(2))
                .sum();
            sq / (expected_l1_vals.len() as f64 - 1.0)
        }
        .max(INSTRUMENTAL_PAL_NOISE_FLOOR);
        assert!((r1.5 - exp_mu_1).abs() < 1e-12, "L1 mu for tx_1");
        assert!((r1.6 - exp_var_1).abs() < 1e-12, "L1 sigma2 for tx_1");

        // L2: tx_2 and tx_3 share the gene_A pool (50 + 5 + 5 = 60 reads).
        let r2 = by_tx["tx_2"];
        let r3 = by_tx["tx_3"];
        assert!((r2.5 - r3.5).abs() < 1e-12, "L2 mu shared across tx_2/tx_3");
        assert!(
            (r2.6 - r3.6).abs() < 1e-12,
            "L2 sigma2 shared across tx_2/tx_3"
        );
        // Verify the gene-pool fit is what we expect.
        let gene_a_vals: Vec<f64> = fxs
            .iter()
            .filter(|f| f.gene_id == "gene_A")
            .flat_map(|f| f.pals.iter().map(|&p| (p as f64).log2()))
            .collect();
        let exp_mu_g = gene_a_vals.iter().sum::<f64>() / gene_a_vals.len() as f64;
        let exp_var_g = {
            let sq: f64 = gene_a_vals.iter().map(|&x| (x - exp_mu_g).powi(2)).sum();
            sq / (gene_a_vals.len() as f64 - 1.0)
        }
        .max(INSTRUMENTAL_PAL_NOISE_FLOOR);
        assert!((r2.5 - exp_mu_g).abs() < 1e-12, "L2 mu == gene_A mean");
        assert!(
            (r2.6 - exp_var_g).abs() < 1e-12,
            "L2 sigma2 == gene_A sample variance"
        );

        // L3: tx_4 uses cohort-level robust fit.
        let r4 = by_tx["tx_4"];
        // L3 mu = median of ALL cohort log2(PAL).
        let mut all: Vec<f64> = fxs
            .iter()
            .flat_map(|f| f.pals.iter().map(|&p| (p as f64).log2()))
            .collect();
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let exp_mu_c = if all.len() % 2 == 1 {
            all[all.len() / 2]
        } else {
            0.5 * (all[all.len() / 2 - 1] + all[all.len() / 2])
        };
        let mut devs: Vec<f64> = all.iter().map(|&x| (x - exp_mu_c).abs()).collect();
        devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = if devs.len() % 2 == 1 {
            devs[devs.len() / 2]
        } else {
            0.5 * (devs[devs.len() / 2 - 1] + devs[devs.len() / 2])
        };
        let exp_sigma2_c = (MAD_TO_SIGMA * mad)
            .powi(2)
            .max(INSTRUMENTAL_PAL_NOISE_FLOOR);
        assert!((r4.5 - exp_mu_c).abs() < 1e-12, "L3 mu == cohort median");
        assert!(
            (r4.6 - exp_sigma2_c).abs() < 1e-12,
            "L3 sigma2 == cohort MAD^2 * 1.4826^2"
        );
    }

    /// Verify that `transcript_evidence.parquet` `em_eligible`
    /// filtering drops reads whose transcript is not eligible.
    /// Specifically, even though `tx_filtered` has reads in the
    /// anchor pool, those reads must NOT contribute to the
    /// output (its emission row is dropped entirely).
    #[test]
    fn test_universal_em_eligible_filtering() {
        let fxs = vec![
            TxFixture {
                gene_id: "gene_X".into(),
                tx_id: "tx_keep".into(),
                pals: (10..20).collect::<Vec<i32>>(), // 10 reads
            },
            TxFixture {
                gene_id: "gene_X".into(),
                tx_id: "tx_filtered".into(),
                pals: (30..40).collect::<Vec<i32>>(), // 10 reads → all dropped
            },
            TxFixture {
                gene_id: "gene_Y".into(),
                tx_id: "tx_keep2".into(),
                pals: (50..60).collect::<Vec<i32>>(), // 10 reads
            },
        ];

        let tmp = local_scratch("telescope_test_em_eligible_filter");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir tmp");

        let anchor_pool = tmp.join("anchor_pool.parquet");
        let te_path = tmp.join("transcript_evidence.parquet");
        let out = tmp.join("emission.parquet");
        write_anchor_pool_parquet(&anchor_pool, &fxs);
        write_transcript_evidence_minimal(
            &te_path,
            &[
                ("tx_keep".into(), true),
                ("tx_filtered".into(), false),
                ("tx_keep2".into(), true),
            ],
        );

        let fb = FallbackConfig::default();
        build_emission_from_anchors(&anchor_pool, Some(&te_path), &out, &fb, None, None)
            .expect("Universal mode run with em_eligible filter");

        let rows = read_emission_parquet(&out);
        // tx_filtered must be absent; the other two must be present.
        let tx_ids: Vec<&str> = rows.iter().map(|r| r.0.as_str()).collect();
        assert_eq!(tx_ids, vec!["tx_keep", "tx_keep2"]);

        // n_cohort_flnc for each kept tx is 10 (its own reads),
        // NOT 10 + 10 = 20 (would happen if filter was bypassed).
        let by_tx: HashMap<&str, u32> = rows.iter().map(|r| (r.0.as_str(), r.4)).collect();
        assert_eq!(by_tx["tx_keep"], 10);
        assert_eq!(by_tx["tx_keep2"], 10);
    }

    /// `n_cohort_flnc` counts OBSERVATIONS, not distinct lengths.
    ///
    /// The column is named as though it were a cohort-fallback count and is
    /// easy to read as a count of distinct values; it is neither. It is
    /// `acc.len()` — the number of anchor rows for the transcript, one per
    /// (sample, transcript, read). A transcript whose 2 reads carry the SAME
    /// `poly_a_length` therefore reports 2, not 1.
    ///
    /// This is the ONLY assertion that separates the two readings. The
    /// distinct-values case is already pinned by
    /// `test_universal_em_eligible_filtering` above (10 reads, 10 distinct
    /// lengths, `n_cohort_flnc == 10`); that case is consistent with BOTH
    /// "counts observations" and "counts distinct values", so it cannot
    /// discriminate. Repeating it here would add no coverage. Degenerating
    /// the two reads to one value is what makes the readings differ.
    ///
    /// A mutation that made this column count distinct lengths would leave
    /// the whole suite green without this test.
    #[test]
    fn test_n_cohort_flnc_counts_observations_not_distinct_lengths() {
        let tmp = local_scratch("telescope_test_ncohort_obs");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir tmp");

        // Two reads, SAME poly_a_length on both. `pals` is the per-read list,
        // and `write_anchor_pool_parquet` emits one row per element, so this
        // is genuinely two observations of one value.
        let fxs = vec![TxFixture {
            gene_id: "gene_dup".into(),
            tx_id: "tx_dup".into(),
            pals: vec![80, 80],
        }];

        let anchor_pool = tmp.join("anchor_pool.parquet");
        let out = tmp.join("emission.parquet");
        write_anchor_pool_parquet(&anchor_pool, &fxs);
        build_emission_from_anchors(
            &anchor_pool,
            None::<&str>,
            &out,
            &FallbackConfig::default(),
            None,
            None,
        )
        .expect("Universal mode run on a duplicate-length transcript");

        let rows = read_emission_parquet(&out);
        assert_eq!(rows.len(), 1, "one transcript row expected");
        assert_eq!(
            rows[0].4, 2,
            "n_cohort_flnc must be the OBSERVATION count (2 reads, both 80 nt), \
             not a count of distinct poly(A) lengths (which would be 1) and not a \
             cohort-fallback count"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// L1 / L2 / L3 TIER ATTRIBUTION and SPLIT, with all three tiers non-empty.
    ///
    /// TWO THINGS ARE ASSERTED, and they fail independently.
    ///
    /// (1) THE SPLIT, read off `build_emission_from_anchors`'s return value:
    /// `(l1, l2, l3) == (1, 4, 1)`. This is the assertion that pins the
    /// counters themselves. `n_l1 += 1` -> `n_l1 += 0` used to leave the whole
    /// suite green, because the counters were function-local and reached only
    /// an `eprintln!`; they are now returned as [`FallbackTierCounts`] and read
    /// from here and from the CLI. The counters sum to the number of emitted
    /// rows, which is asserted too — a cascade that counted a transcript twice
    /// or dropped one would break it.
    ///
    /// (2) THE ATTRIBUTION: which fit each transcript was actually given.
    /// `test_3level_fallback_thresholds` above already pins the `(mu, sigma2)`
    /// each level produces, but it does not pin WHICH transcripts sit in each
    /// level, nor that all four L2 transcripts share one bit-identical fit.
    /// Collapsing the L2 branch into L3 turns (2) red.
    ///
    /// THE FIXTURE. L2 is the tier that a naive fixture cannot reach, because
    /// it needs a gene pool strictly larger than any single transcript's pool.
    /// `write_anchor_pool_parquet` DOES take a `gene_id` per `TxFixture` and
    /// emits it verbatim on every row, so several DISTINCT transcripts can be
    /// given one SHARED `gene_id` — which is what `test_3level_fallback_thresholds`
    /// already relies on, and it is unrelated to the `anchor_builder.rs`
    /// `gene_id := transcript_id` fallback (that is about the production
    /// anchor builder, not about this fixture writer). Here `tx_min_reads = 10`
    /// and `gene_min_reads = 30`: each L2 transcript is capped at 9 reads, so
    /// four of them are needed to clear 30 in the pool.
    #[test]
    fn test_l1_l2_l3_tier_attribution_and_split() {
        let fxs = vec![
            // L1: 12 >= tx_min_reads -> its OWN reads. gene_l1's pool is 12,
            // below gene_min_reads, which is irrelevant: L1 is decided first.
            TxFixture {
                gene_id: "gene_l1".into(),
                tx_id: "tx_l1".into(),
                pals: (100..112).collect::<Vec<i32>>(), // 12 reads
            },
            // L2: each of these is 9 < 10, but they SHARE gene_l2, whose pool
            // is 4 x 9 = 36 >= 30. One transcript alone could never do this.
            TxFixture {
                gene_id: "gene_l2".into(),
                tx_id: "tx_l2_a".into(),
                pals: (300..309).collect::<Vec<i32>>(), // 9 reads
            },
            TxFixture {
                gene_id: "gene_l2".into(),
                tx_id: "tx_l2_b".into(),
                pals: (400..409).collect::<Vec<i32>>(), // 9 reads
            },
            TxFixture {
                gene_id: "gene_l2".into(),
                tx_id: "tx_l2_c".into(),
                pals: (500..509).collect::<Vec<i32>>(), // 9 reads
            },
            TxFixture {
                gene_id: "gene_l2".into(),
                tx_id: "tx_l2_d".into(),
                pals: (600..609).collect::<Vec<i32>>(), // 9 reads
            },
            // L3: 3 < 10 and gene_l3's pool is 3 < 30 -> cohort fit.
            TxFixture {
                gene_id: "gene_l3".into(),
                tx_id: "tx_l3".into(),
                pals: (900..903).collect::<Vec<i32>>(), // 3 reads
            },
        ];

        let tmp = local_scratch("telescope_test_l123_tiers");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir tmp");

        let anchor_pool = tmp.join("anchor_pool.parquet");
        let out = tmp.join("emission.parquet");
        write_anchor_pool_parquet(&anchor_pool, &fxs);

        let fb = FallbackConfig {
            tx_min_reads: 10,
            gene_min_reads: 30,
            cohort_min_reads: 50,
        };
        let tiers = build_emission_from_anchors(&anchor_pool, None::<&str>, &out, &fb, None, None)
            .expect("Universal mode run");

        // (1) THE SPLIT — read off the return value. This is the assertion
        // that makes `n_l1 += 1` -> `n_l1 += 0` a red test, which it was not
        // while the counters reached nothing but an `eprintln!`.
        assert_eq!(
            tiers,
            FallbackTierCounts {
                l1: 1,
                l2: 4,
                l3: 1,
                n_imputed: 0,
                n_reference: 0,
                n_anchored_not_in_reference: 0,
                // Pinned against the naming rule rather than against the
                // value the function returned, which would be a tautology.
                provenance_path: out.with_extension("provenance.parquet"),
            },
            "the L1/L2/L3 split: 1 transcript at/above tx_min_reads, 4 sharing a \
             qualifying gene pool, 1 in neither"
        );

        let rows = read_emission_parquet(&out);
        assert_eq!(rows.len(), 6, "one output row per transcript: {rows:?}");
        assert_eq!(
            tiers.l1 + tiers.l2 + tiers.l3,
            rows.len(),
            "the three tiers must partition the output rows: a transcript counted \
             twice, or not at all, is not a partition"
        );

        // The three reference fits, computed here by calling the SAME private
        // helpers the cascade calls, so the expectations cannot drift from the
        // implementation on a detail like Bessel correction or the noise floor.
        let logs = |tx: &str| -> Vec<f64> {
            fxs.iter()
                .find(|f| f.tx_id == tx)
                .expect("fixture transcript")
                .pals
                .iter()
                .map(|&p| (p as f64).log2())
                .collect()
        };
        let l1_fit = fit_log_normal(&logs("tx_l1")).expect("L1 fit is non-empty");
        let l2_fit = fit_log_normal(
            &["tx_l2_a", "tx_l2_b", "tx_l2_c", "tx_l2_d"]
                .iter()
                .flat_map(|t| logs(t))
                .collect::<Vec<f64>>(),
        )
        .expect("gene_l2 pool is non-empty");
        let l3_fit = cohort_log_normal_fit(
            &fxs.iter()
                .flat_map(|f| f.pals.iter().map(|&p| (p as f64).log2()))
                .collect::<Vec<f64>>(),
        )
        .expect("cohort is non-empty");

        // The fixture is only a tier test if the three fits are actually
        // distinguishable. Asserted up front so a fixture edit that collapses
        // two of them fails HERE, with a clear message, instead of silently
        // making the per-transcript comparisons below vacuous.
        assert!(
            (l1_fit.mu - l2_fit.mu).abs() > 1e-6
                && (l1_fit.mu - l3_fit.mu).abs() > 1e-6
                && (l2_fit.mu - l3_fit.mu).abs() > 1e-6
                && (l1_fit.sigma2 - l2_fit.sigma2).abs() > 1e-6
                && (l1_fit.sigma2 - l3_fit.sigma2).abs() > 1e-6
                && (l2_fit.sigma2 - l3_fit.sigma2).abs() > 1e-6,
            "the three tiers must be distinguishable in BOTH components, so the \
             1e-9 match tolerance below cannot overlap two of them: \
             L1={l1_fit:?} L2={l2_fit:?} L3={l3_fit:?}"
        );

        // Classify each emitted row by the fit it carries.
        //
        // The match tolerance is 1e-9, not exact, because of the L2
        // HashMap-ordering effect described further down: a re-derived gene
        // fit can differ from the stored one by a few ULPs. 1e-9 sits ~8
        // orders of magnitude below the smallest gap between two tiers (~0.12
        // on this fixture), which the assertion above pins.
        let tier_of = |tx: &str| -> &'static str {
            let (_, _, _, _, _, mu, sigma2) = rows
                .iter()
                .find(|r| r.0 == tx)
                .unwrap_or_else(|| panic!("{tx} missing from the emission table: {rows:?}"));
            let same =
                |f: &LogNormalFit| (mu - f.mu).abs() < 1e-9 && (sigma2 - f.sigma2).abs() < 1e-9;
            match (same(&l1_fit), same(&l2_fit), same(&l3_fit)) {
                (true, false, false) => "L1",
                (false, true, false) => "L2",
                (false, false, true) => "L3",
                // Two or zero matches: either a tier produced an unexpected fit
                // or the levels are indistinguishable. Both are failures.
                m => panic!(
                    "{tx} does not match exactly one tier's fit: \
                     (l1,l2,l3) matched = {m:?}; row = ({mu}, {sigma2}); \
                     L1={l1_fit:?} L2={l2_fit:?} L3={l3_fit:?}"
                ),
            }
        };

        assert_eq!(tier_of("tx_l1"), "L1", "12 >= tx_min_reads is tx-level");
        for tx in ["tx_l2_a", "tx_l2_b", "tx_l2_c", "tx_l2_d"] {
            assert_eq!(
                tier_of(tx),
                "L2",
                "{tx}: 9 < tx_min_reads but gene_l2 pools 36 >= 30, so it must take \
                 the SHARED gene fit, not the cohort fit"
            );
        }
        assert_eq!(
            tier_of("tx_l3"),
            "L3",
            "3 < tx_min_reads and gene_l3 pools only 3 < 30, so it must fall to the cohort"
        );

        // The four L2 transcripts must share ONE fit, bit for bit — that
        // sharing is the mechanism behind `n_polya_zero_power` in Module 3, and
        // an L2 that recomputed per transcript would break that metric.
        //
        // BIT-IDENTICAL TO EACH OTHER, and now also bit-identical to
        // `l2_fit` as recomputed here. The gene pool is assembled by iterating
        // `tx_accums`, a `HashMap`, so the order the 36 values arrive in is not
        // fixed and `fit_log_normal`'s `sum::<f64>()` is not associative --
        // which is why `fit_log_normal` sorts before summing. The 1e-9
        // tolerance below is therefore slack the code no longer needs; it is
        // kept because the re-derivation is a second, independent call and a
        // ULP-level assertion is a poor failure message. The exact-equality
        // claim lives in
        // `test_l2_gene_fit_is_bit_identical_across_accumulator_insertion_orders`.
        let l2_rows: Vec<&(String, String, f64, bool, u32, f64, f64)> =
            rows.iter().filter(|r| r.0.starts_with("tx_l2_")).collect();
        assert_eq!(l2_rows.len(), 4);
        let l2_shared = (l2_rows[0].5, l2_rows[0].6);
        for r in &l2_rows {
            assert_eq!(
                (r.5, r.6),
                l2_shared,
                "{} must carry the IDENTICAL shared gene fit, not a per-transcript one",
                r.0
            );
            assert!(
                (r.5 - l2_fit.mu).abs() < 1e-9 && (r.6 - l2_fit.sigma2).abs() < 1e-9,
                "{} must carry the GENE-POOL fit, not its own: got ({}, {}), \
                 gene fit ({}, {})",
                r.0,
                r.5,
                r.6,
                l2_fit.mu,
                l2_fit.sigma2
            );
        }

        // (2) CROSS-CHECK: the triple the cascade RETURNED must agree with the
        // triple recovered independently from the per-transcript attribution
        // above. The returned value and the emitted fits are produced by two
        // different branches of the same loop; a cascade that incremented the
        // wrong counter would pass a test that only read the return value, and
        // one that fitted the wrong tier would pass a test that only counted.
        let (mut n_l1, mut n_l2, mut n_l3) = (0usize, 0usize, 0usize);
        for r in &rows {
            match tier_of(&r.0) {
                "L1" => n_l1 += 1,
                "L2" => n_l2 += 1,
                _ => n_l3 += 1,
            }
        }
        assert_eq!(
            (n_l1, n_l2, n_l3),
            (tiers.l1, tiers.l2, tiers.l3),
            "the returned split must match the split implied by the fits each \
             transcript was actually given"
        );

        // The `.max(1)` zero guard the CLI applies to the printed denominator:
        // it only bites when there are no rows at all, and it must not change
        // these.
        let total = (tiers.l1 + tiers.l2 + tiers.l3).max(1);
        assert_eq!(total, 6, "one denominator row per emitted transcript");
        assert!((tiers.l1 as f64 * 100.0 / total as f64 - 100.0 / 6.0).abs() < 1e-12);
        assert!((tiers.l2 as f64 * 100.0 / total as f64 - 400.0 / 6.0).abs() < 1e-12);
        assert!((tiers.l3 as f64 * 100.0 / total as f64 - 100.0 / 6.0).abs() < 1e-12);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// L2 must be bit-reproducible across processes.
    ///
    /// The L2 fit is the only one that reads a `HashMap`-derived slice: the
    /// gene pool is assembled by iterating `tx_accums`, so the order its
    /// `log2(PAL)` values are handed to `fit_log_normal` depends on the
    /// insertion history and the per-process random seed. `f64` addition is not
    /// associative, so before `fit_log_normal` sorted, the same anchor pool
    /// emitted L2 rows differing in the last 1-2 mantissa bits between runs
    /// (~1e-16 relative, measured over 12 runs on a fixture; L1 and L3 were
    /// already bit-reproducible and stayed that way).
    ///
    /// WHY THIS TEST IS NOT "RUN THE BUILD TWICE". `HashMap` iteration order is
    /// STABLE WITHIN A PROCESS, so re-running the same build in the same
    /// process reproduces the non-deterministic value exactly and passes with
    /// the bug present. The only way to reach two different iteration orders
    /// from one process is to build two maps with different insertion
    /// histories, which is what the three orders below do. (The alternative —
    /// hard-coding the expected bits — was rejected: it pins the summation
    /// order, so any later change to how the mean is accumulated, e.g.
    /// `powi(2)` -> `x * x`, would fail for a reason that has nothing to do
    /// with determinism.)
    ///
    /// `to_bits()`, not `==`: `PartialEq for f64` is a numeric comparison, and
    /// the point of the test is the bit pattern, not the value.
    #[test]
    fn test_l2_gene_fit_is_bit_identical_across_accumulator_insertion_orders() {
        // A realistic pool, and deliberately large: PAL lengths spread over
        // 10..=400 (~2.3 nats of range) over 3000 values. The order of a
        // 3000-term f64 sum of numbers this size IS observable in the last
        // mantissa bits; a handful of values would sum identically in any
        // order and this test would pass with the bug.
        let per_tx: Vec<Vec<f64>> = ["tx_a", "tx_b", "tx_c", "tx_d"]
            .iter()
            .enumerate()
            .map(|(i, _)| {
                (0..750)
                    .map(|j| ((10 + ((i * 750 + j) * 37) % 391) as f64).log2())
                    .collect()
            })
            .collect();

        // The production assembly, verbatim: one accumulator per transcript,
        // then pool the accumulators in map order, then fit. Reproduced here
        // rather than calling `build_emission_from_anchors` so that the ONLY
        // thing varying between the three calls is the insertion order.
        let tx_names = ["tx_a", "tx_b", "tx_c", "tx_d"];
        let fit_for_order = |order: &[usize]| -> LogNormalFit {
            let mut accums: HashMap<String, TxAccumulator> = HashMap::new();
            for &i in order {
                accums
                    .entry(tx_names[i].to_string())
                    .or_insert_with(|| TxAccumulator::new(format!("gene_{i}")))
                    .log2_pals
                    .extend_from_slice(&per_tx[i]);
            }
            let mut gene_pool: Vec<f64> = Vec::new();
            for acc in accums.values() {
                gene_pool.extend_from_slice(&acc.log2_pals);
            }
            assert_eq!(gene_pool.len(), 3000, "every value reaches the pool");
            fit_log_normal(&gene_pool).expect("gene pool is non-empty")
        };

        let orders: [&[usize]; 3] = [&[0, 1, 2, 3], &[3, 2, 1, 0], &[1, 3, 0, 2]];
        let reference = fit_for_order(orders[0]);
        for order in orders.iter().copied().skip(1) {
            let other = fit_for_order(order);
            assert_eq!(
                (other.mu.to_bits(), other.sigma2.to_bits()),
                (reference.mu.to_bits(), reference.sigma2.to_bits()),
                "accumulator insertion order {order:?} changed the L2 gene fit by \
                 at least one mantissa bit: {other:?} vs {reference:?}. \
                 `fit_log_normal` must sort before summing."
            );
        }

        // L3 shares the fixture and the concern but not the mechanism: it reads
        // order STATISTICS off an already-sorted copy and sums nothing, so it
        // was bit-reproducible before the fix and must stay that way. Cheap to
        // assert, and it stops the sort in `cohort_log_normal_fit` from being
        // removed as a micro-optimisation on the strength of "it sums
        // nothing, the sort is redundant".
        let flat: Vec<f64> = per_tx.concat();
        for order in orders.iter().copied().skip(1) {
            let mut reordered: Vec<f64> = Vec::with_capacity(flat.len());
            for &i in order {
                reordered.extend_from_slice(&per_tx[i]);
            }
            let a = cohort_log_normal_fit(&flat).expect("cohort is non-empty");
            let b = cohort_log_normal_fit(&reordered).expect("cohort is non-empty");
            assert_eq!(
                (b.mu.to_bits(), b.sigma2.to_bits()),
                (a.mu.to_bits(), a.sigma2.to_bits()),
                "L3 must not depend on input order either, order {order:?}: \
                 {b:?} vs {a:?}"
            );
        }
    }
    /// PADDING. The emitted row set is the UNION of the anchor pool and the
    /// reference universe, and an imputed row is the cohort fit marked by
    /// `n_cohort_flnc == 0`.
    ///
    /// These are the assertions the whole feature rests on, and each one has a
    /// mutation that breaks it:
    ///
    /// | claim | breaking mutation |
    /// |---|---|
    /// | `rows.len() == |universe|` | drop the imputed arm of the row match |
    /// | sorted by `transcript_id` | remove `tx_keys.sort()` |
    /// | imputed rows carry the COHORT fit | fit them from something else |
    /// | `n_cohort_flnc == 0` on imputed rows | write `1` |
    /// | an anchored+universe key is emitted ONCE | `if anchored.contains(tx)` -> `if false` |
    /// | `l1+l2+l3+n_imputed == rows.len()` | count imputed rows in `l3` instead |
    /// THE PROVENANCE SIDECAR IS THE ONLY PLACE THE FALLBACK LEVEL IS WRITTEN
    /// DOWN. `n_cohort_flnc` cannot recover it: L2 is decided by the GENE's
    /// pooled count, and the frozen parquet carries only the TRANSCRIPT's own
    /// count. So if the tier column is wrong, every downstream `tier` join is
    /// wrong, and nothing else in the file would notice.
    ///
    /// This reads the sidecar back from disk with its own reader and checks
    /// three things separately, because they fail differently:
    ///   * the key set and ORDER match the emission parquet exactly (an
    ///     unordered or short sidecar turns the downstream join into a
    ///     different, wrong join);
    ///   * `n_anchor == 0` if and only if the tier is `L3_imputed` (the two
    ///     encodings of "imputed" must not drift apart);
    ///   * the tier histogram reproduces `FallbackTierCounts` (the counters the
    ///     CLI prints must describe the same partition the file records).
    #[test]

    fn test_provenance_sidecar_agrees_with_the_emission_rows_row_by_row() {
        let tmp = local_scratch("telescope_test_provenance");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir tmp");

        // All three anchored tiers in one fixture, plus two imputed keys.
        //
        // L2 is the hard one to reach and getting it wrong is silent: it needs
        // the GENE's pooled count over `gene_min_reads` while the TRANSCRIPT's
        // own count is under `tx_min_reads`. An earlier version of this fixture
        // had 12 + 3 reads in gene_a against a default `gene_min_reads` of 30,
        // so no gene fit existed at all and `tx_l2` silently became L3_anchored.
        // The row-by-row assertions still passed; only the `n_l2 > 0` CONTROL
        // below caught it, which is the reason that control is here.
        //
        //   tx_l1  12 reads           >= 10            -> L1
        //   tx_l3  20 reads           >= 10            -> L1
        //   tx_l2   3 reads           <  10, gene_a    -> L2
        //   gene_a pool 12+3+20 = 35 >= 30
        let fxs = vec![
            TxFixture {
                gene_id: "gene_a".into(),
                tx_id: "tx_l1".into(),
                pals: (80..=91).collect::<Vec<i32>>(),
            },
            TxFixture {
                gene_id: "gene_a".into(),
                tx_id: "tx_l2".into(),
                pals: vec![80, 81, 82],
            },
            TxFixture {
                gene_id: "gene_a".into(),
                tx_id: "tx_l3".into(),
                pals: (100..=119).collect::<Vec<i32>>(),
            },
        ];
        let anchor_pool = tmp.join("anchor_pool.parquet");
        let out = tmp.join("emission.parquet");
        write_anchor_pool_parquet(&anchor_pool, &fxs);

        let universe: Vec<String> = vec![
            "tx_l1".into(),
            "tx_l2".into(),
            "tx_l3".into(),
            "tx_pad_b".into(),
            "tx_pad_a".into(),
        ];
        let tiers = build_emission_from_anchors(
            &anchor_pool,
            None::<&str>,
            &out,
            &FallbackConfig::default(),
            Some(&universe),
            None,
        )
        .expect("Universal mode run");

        assert_eq!(
            tiers.provenance_path,
            out.with_extension("provenance.parquet"),
            "the sidecar is named from the emission file name"
        );

        let prov = read_provenance_parquet(&tiers.provenance_path);
        let emission = read_emission_parquet(&out);

        assert_eq!(
            prov.len(),
            emission.len(),
            "the sidecar must have exactly one row per emission row"
        );
        for (p, e) in prov.iter().zip(emission.iter()) {
            assert_eq!(
                p.0, e.0,
                "sidecar and emission must agree on transcript_id IN ORDER; a \
                 sidecar sorted differently turns every downstream join into a \
                 different join"
            );
            assert_eq!(p.1, e.1, "gene_id must agree row by row");
        }
        assert!(
            prov.windows(2).all(|w| w[0].0 < w[1].0),
            "the sidecar must be STRICTLY ascending by transcript_id: {:?}",
            prov
        );

        for (tx, gene, tier, n_anchor) in &prov {
            let imputed = *tier == TIER_L3_IMPUTED;
            assert_eq!(
                *n_anchor == 0,
                imputed,
                "{tx}: n_anchor == 0 must mean exactly 'imputed', and this row \
                 is tier {tier} with n_anchor {n_anchor} (gene {gene})"
            );
        }

        let n_l1 = prov.iter().filter(|r| r.2 == TIER_L1).count();
        let n_l2 = prov.iter().filter(|r| r.2 == TIER_L2).count();
        let n_l3a = prov.iter().filter(|r| r.2 == TIER_L3_ANCHORED).count();
        let n_l3i = prov.iter().filter(|r| r.2 == TIER_L3_IMPUTED).count();
        assert_eq!(
            (n_l1, n_l2, n_l3a, n_l3i),
            (tiers.l1, tiers.l2, tiers.l3, tiers.n_imputed),
            "the tier histogram in the file must reproduce the counters the CLI \
             prints. counters=({}, {}, {}, {}) file=({}, {}, {}, {})",
            tiers.l1,
            tiers.l2,
            tiers.l3,
            tiers.n_imputed,
            n_l1,
            n_l2,
            n_l3a,
            n_l3i
        );
        assert_eq!(n_l3i, 2, "the fixture pads exactly two transcripts");
        assert!(
            n_l1 > 0 && n_l2 > 0,
            "CONTROL: the fixture must exercise L1 and L2, or the L2 row above \
             is not actually testing the L2 label. l1={n_l1} l2={n_l2}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_padded_rows_are_cohort_fits_marked_by_a_zero_observation_count() {
        let tmp = local_scratch("telescope_test_padding");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir tmp");

        // One well-observed transcript -> L1 (>= tx_min_reads = 10).
        let fxs = vec![TxFixture {
            gene_id: "gene_a".into(),
            tx_id: "tx_anchored".into(),
            pals: vec![80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91],
        }];
        let anchor_pool = tmp.join("anchor_pool.parquet");
        let out = tmp.join("emission.parquet");
        write_anchor_pool_parquet(&anchor_pool, &fxs);

        // Deliberately NOT sorted, and `tx_anchored` IS in the universe, so
        // the test also covers the intersection rather than only the padding.
        let universe: Vec<String> =
            vec!["tx_anchored".into(), "tx_pad_b".into(), "tx_pad_a".into()];
        let tiers = build_emission_from_anchors(
            &anchor_pool,
            None::<&str>,
            &out,
            &FallbackConfig::default(),
            Some(&universe),
            None,
        )
        .expect("Universal mode run with a reference universe");

        let rows = read_emission_parquet(&out);
        assert_eq!(
            rows.len(),
            3,
            "the emitted set is the UNION: 1 anchored + 2 imputed, and the \\
             intersection transcript must not be emitted twice. rows={rows:?}"
        );
        assert_eq!(tiers.n_imputed, 2);
        assert_eq!(tiers.n_reference, 3);
        assert_eq!(tiers.n_anchored_not_in_reference, 0);

        // Deterministic ordering. The union adds keys the map never had, so
        // this sort is what makes a padded build byte-stable.
        let ids: Vec<&str> = rows.iter().map(|r| r.0.as_str()).collect();
        assert_eq!(
            ids,
            vec!["tx_anchored", "tx_pad_a", "tx_pad_b"],
            "output must be sorted by transcript_id regardless of the universe's \\
             input order"
        );

        // The anchored row keeps its own fit and its own observation count.
        assert_eq!(rows[0].4, 12, "12 anchor observations");
        assert!(rows[0].3, "observed transcripts are competent");

        // The cohort fit, computed by calling the SAME private helper the
        // build calls, so the expectation cannot drift from the code.
        let cohort_in: Vec<f64> = (80..=91).map(|p| (p as f64).log2()).collect();
        let cohort = cohort_log_normal_fit(&cohort_in).expect("cohort fit");
        for r in &rows[1..] {
            assert_eq!(
                r.4, 0,
                "an IMPUTED row is marked by n_cohort_flnc == 0, which is the only \\
                 in-schema signal available under the frozen 7-column schema"
            );
            assert!(
                r.3,
                "imputed rows are is_polyA_competent == true: the row asserts the \\
                 transcript is EVALUABLE, not that its poly(A) was measured"
            );
            assert_eq!(r.1, r.0, "no tx2gene supplied, so gene_id := transcript_id");
            assert_eq!(r.2, 1.0, "polyA_frac");
            assert!(
                (r.5 - cohort.mu).abs() < 1e-12,
                "an imputed row carries the COHORT mu, not a fit of its own. \
                 got {} want {}",
                r.5,
                cohort.mu
            );
            assert!(
                (r.6 - cohort.sigma2).abs() < 1e-12,
                "an imputed row carries the COHORT sigma2. got {} want {}",
                r.6,
                cohort.sigma2
            );
        }
        // And the imputed rows are NOT the anchored transcript's fit: if they
        // were, the assertion above would pass for the wrong reason whenever
        // the two happened to coincide.
        assert!(
            (rows[0].5 - rows[1].5).abs() > 1e-9,
            "CONTROL: the anchored mu must differ from the cohort mu, or the \\
             cohort assertion above is vacuous. anchored={} cohort={}",
            rows[0].5,
            rows[1].5
        );

        // THE PARTITION IS FOUR TERMS NOW, not three. This is the invariant
        // that keeps `n_imputed` from being double-counted or dropped.
        assert_eq!(
            tiers.l1 + tiers.l2 + tiers.l3 + tiers.n_imputed,
            rows.len(),
            "l1 + l2 + l3 + n_imputed must partition the output rows"
        );
        assert_eq!(
            tiers.l1 + tiers.l2 + tiers.l3,
            1,
            "the tier counters describe the ANCHORED population only. Folding \\
             imputed rows into l3 would silently rebase every published tier \\
             percentage onto the padded denominator."
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The universe and the model can disagree about which transcripts exist.
    /// When they do, `n_emission_after` is NOT the universe size, and saying so
    /// is the difference between a warning and a wrong `total` nobody checks.
    #[test]
    fn test_anchored_transcripts_absent_from_the_universe_are_counted_not_dropped() {
        let tmp = local_scratch("telescope_test_drift");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir tmp");

        let fxs = vec![TxFixture {
            gene_id: "gene_a".into(),
            tx_id: "tx_anchored".into(),
            pals: vec![80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91],
        }];
        let anchor_pool = tmp.join("anchor_pool.parquet");
        let out = tmp.join("emission.parquet");
        write_anchor_pool_parquet(&anchor_pool, &fxs);

        // The universe knows nothing about the one anchored transcript.
        let universe: Vec<String> = vec!["tx_pad_a".into()];
        let tiers = build_emission_from_anchors(
            &anchor_pool,
            None::<&str>,
            &out,
            &FallbackConfig::default(),
            Some(&universe),
            None,
        )
        .expect("Universal mode run with a disagreeing universe");

        assert_eq!(
            tiers.n_anchored_not_in_reference, 1,
            "the drift is COUNTED. Silently dropping the anchored row would make \\
             the emitted table disagree with the model that produced it."
        );
        assert_eq!(tiers.n_reference, 1);
        assert_eq!(tiers.n_imputed, 1);
        let rows = read_emission_parquet(&out);
        assert_eq!(
            rows.len(),
            2,
            "the anchored row is still emitted: the union is a UNION. rows={rows:?}"
        );
        assert_eq!(rows[0].0, "tx_anchored", "sorted first");
        assert_eq!(rows[1].4, 0, "tx_pad_a is the imputed one");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A transcript the `em_eligible` filter removed and a transcript the
    /// universe has never heard of both arrive at the row-assembly match with
    /// `tx_accums.get(..) == None`, but they are not the same thing, and the
    /// emitted row is where the difference has to survive.
    ///
    /// Before this test's production change both took the same arm and both
    /// were written `is_poly_a_competent = true`, which told the EM solver that
    /// a deliberately suppressed transcript had a flawless poly(A)
    /// distribution. The suppression was therefore silently undone downstream,
    /// and the candidate ledger filed the transcript under `rows_emitted`
    /// rather than `n_incompetent` -- `n_incompetent` being the bucket whose
    /// documented question is "present, but pushed to `-inf`".
    ///
    /// `tx_anchorless` is the CONTROL and it is not decoration. It is in the
    /// universe but was never anchored, so the filter never saw it and it must
    /// stay competent. Without that row a mutation that marked *every*
    /// `None` transcript incompetent would pass this test, while
    /// `test_universal_em_eligible_filtering` stays green because it supplies
    /// no universe.
    #[test]
    fn test_suppressed_transcripts_emitted_as_incompetent() {
        let tmp = local_scratch("telescope_test_suppressed_incompetent");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir tmp");

        let fxs = vec![
            TxFixture {
                gene_id: "gene_A".into(),
                tx_id: "tx_keep".into(),
                pals: (10..20).collect::<Vec<i32>>(),
            },
            TxFixture {
                gene_id: "gene_A".into(),
                tx_id: "tx_filtered".into(),
                pals: (30..40).collect::<Vec<i32>>(),
            },
        ];

        let anchor_pool = tmp.join("anchor_pool.parquet");
        let te_path = tmp.join("transcript_evidence.parquet");
        let out = tmp.join("emission.parquet");
        write_anchor_pool_parquet(&anchor_pool, &fxs);
        // `tx_anchorless` is deliberately ABSENT here. `load_em_eligible_set`
        // inserts only rows where `em_eligible == true`, so "not listed" and
        // "listed false" are the same thing to the filter -- which is why an
        // anchor-less transcript is never suppressed and the control below
        // still means what it claims to.
        write_transcript_evidence_minimal(
            &te_path,
            &[("tx_keep".into(), true), ("tx_filtered".into(), false)],
        );
        let universe: Vec<String> = vec![
            "tx_keep".into(),
            "tx_filtered".into(),
            "tx_anchorless".into(),
        ];

        let tiers = build_emission_from_anchors(
            &anchor_pool,
            Some(&te_path),
            &out,
            &FallbackConfig::default(),
            Some(&universe),
            None,
        )
        .expect("em_eligible filter + reference universe");

        let rows = read_emission_parquet(&out);
        let by_tx: HashMap<&str, _> = rows.iter().map(|r| (r.0.as_str(), r)).collect();

        // (1) The suppressed row is PRESENT and the count is the universe size.
        assert_eq!(tiers.n_reference, 3);
        assert_eq!(
            rows.len(),
            3,
            "the union is anchored UNION universe, so the row count is \\
             n_reference here (both anchored transcripts are in the universe). \\
             rows={rows:?}"
        );
        assert_eq!(tiers.n_anchored_not_in_reference, 0);
        // The law `telescope-cli` asserts on the production path, evaluated
        // against the same numbers. See the `ensure!` in main.rs.
        assert_eq!(
            rows.len(),
            tiers.n_reference + tiers.n_anchored_not_in_reference,
            "emission conservation"
        );

        let sup = by_tx.get("tx_filtered").expect(
            "the suppressed transcript must still be emitted, just \\
                     marked incompetent -- dropping the row would leave \\
                     rows.len() < n_reference with no counter able to see it",
        );
        assert_eq!(sup.4, 0, "its anchors were filtered out");
        // (2) The whole point: strictly false, not merely "false because zero".
        assert!(
            !sup.3,
            "an em_eligible-suppressed transcript must be emitted \\
             INCOMPETENT. competent=true here claims a flawless poly(A) \\
             distribution for a transcript the filter rejected on purpose, \\
             and hands it a non-zero posterior in the EM solve."
        );

        // CONTROL: never anchored, never seen by the filter, still competent.
        let anchorless = by_tx
            .get("tx_anchorless")
            .expect("the universe key with no anchors is emitted as imputed");
        assert_eq!(anchorless.4, 0);
        assert!(
            anchorless.3,
            "CONTROL: a universe transcript with no anchors is imputed for \\
             COMPLETENESS, not suppressed, so it stays competent. If this is \\
             false the match arm is matching on `None` rather than on being \\
             suppressed."
        );

        let keep = by_tx.get("tx_keep").expect("the eligible transcript");
        assert!(keep.3);
        assert_eq!(
            keep.4, 10,
            "the eligible transcript keeps its own 10 observations"
        );

        // The partition stays four terms: neither imputed row is folded into
        // the anchored counters, so every published tier percentage still
        // describes the anchored population.
        assert_eq!(tiers.n_imputed, 2);
        assert_eq!(tiers.l1 + tiers.l2 + tiers.l3, 1);
        assert_eq!(tiers.l1 + tiers.l2 + tiers.l3 + tiers.n_imputed, rows.len());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `n == 0` inside the parquet is now ambiguous on purpose, and this pins
    /// the ambiguity rather than leaving it to a reader to discover: with a
    /// universe, `0` means "imputed"; without one, it cannot occur at all.
    ///
    /// The two conditions are DISJOINT by construction, which is what makes a
    /// single marker usable: there is no run in which a `0` is ambiguous
    /// between "imputed" and "observed zero times", because the second cannot
    /// occur while the first is possible.
    #[test]
    fn test_without_a_universe_no_row_is_imputed() {
        let tmp = local_scratch("telescope_test_nopad");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir tmp");

        let fxs = vec![TxFixture {
            gene_id: "gene_a".into(),
            tx_id: "tx_only".into(),
            pals: vec![80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91],
        }];
        let anchor_pool = tmp.join("anchor_pool.parquet");
        let out = tmp.join("emission.parquet");
        write_anchor_pool_parquet(&anchor_pool, &fxs);
        let tiers = build_emission_from_anchors(
            &anchor_pool,
            None::<&str>,
            &out,
            &FallbackConfig::default(),
            None,
            None,
        )
        .expect("Universal mode run without a universe");

        assert_eq!(tiers.n_imputed, 0);
        assert_eq!(tiers.n_reference, 0);
        assert_eq!(tiers.n_anchored_not_in_reference, 0);
        let rows = read_emission_parquet(&out);
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].4, 12,
            "12 anchor observations, so this column is not 0 in an unpadded build"
        );
        assert!(
            rows.iter().all(|r| r.4 > 0),
            "every row in an unpadded build has at least one observation"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
