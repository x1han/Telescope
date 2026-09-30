//! Module 1: Emission Model Builder
//!
//! Two input modes (mutually exclusive at the CLI level):
//!
//! - **AIDRS Mode (legacy)**: parses AIDRS outputs
//!   (`aidrs.transcript_polyA_len.parquet` +
//!   `aidrs.transcript.assessment.tsv`)
//!   and constructs the per-transcript emission model
//!   $P(L \mid T_k)$ — a Gaussian in $\log_2(\text{PAL})$
//!   space — for TEL-EM.
//! - **Universal Mode (D1 Step 3)**: parses the per-read
//!   `telescope.anchor_pool.parquet` (produced by
//!   `anchor_builder::build_anchor_pool`), optionally joins
//!   with `telescope.transcript_evidence.parquet` to apply
//!   `em_eligible` filtering, then fits the same emission
//!   model with a 3-level fallback cascade
//!   (Tx → Gene → Cohort).
//!
//! Output: `telescope.emission.parquet` with the schema
//! declared in `output_schema()`. Each row is one transcript
//! $T_k$ with computed emission parameters
//! $(\mu_{k,\text{cohort}}, \sigma^2_{k,\text{cohort}})$.
//!
//! ## `polyA_frac` + `is_polyA_competent` semantics
//!
//! - **AIDRS Mode**: state is lifted from
//!   `aidrs.transcript.assessment.tsv` (State 1/2/3 → 3-way
//!   classification, see `load_assessment_quality`).
//! - **Universal Mode**: there is no equivalent
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
/// We pin a numerical floor of 0.04 in `log2(PAL)` space, which
/// corresponds to a per-transcript CV of ~15% — roughly the ONT
/// physical measurement noise floor. This is consistent with
/// `alevin-fry`'s `MIN_OUTPUT_ALPHA=0.01` philosophy of always
/// keeping a non-degenerate posterior.
pub const MIN_COHORT_VARIANCE: f64 = 0.04;

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
/// mean/variance per transcript, applies `MIN_COHORT_VARIANCE`
/// floor, and writes the 7-column output schema above.
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
                    MIN_COHORT_VARIANCE
                };
                (mean, variance.max(MIN_COHORT_VARIANCE))
            } else {
                // State 2/3 -> emit placeholder mu=0 and floored sigma.
                (0.0, MIN_COHORT_VARIANCE)
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
/// `(MAD * 1.4826)^2` (L3), floored at `MIN_COHORT_VARIANCE`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct LogNormalFit {
    mu: f64,
    sigma2: f64,
}

/// Fit `(mean, sample variance floored at MIN_COHORT_VARIANCE)`
/// from a slice of `log2(PAL)` values. Returns `None` iff
/// `values` is empty.
fn fit_log_normal(values: &[f64]) -> Option<LogNormalFit> {
    let n = values.len();
    if n == 0 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        let sq_diff_sum: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum();
        sq_diff_sum / ((n - 1) as f64) // Bessel-corrected sample variance
    } else {
        MIN_COHORT_VARIANCE
    };
    Some(LogNormalFit {
        mu: mean,
        sigma2: variance.max(MIN_COHORT_VARIANCE),
    })
}

/// Cohort-level robust fit: `mu` = median(log2(PAL)),
/// `sigma2` = `(MAD * 1.4826)^2`, floored at
/// `MIN_COHORT_VARIANCE`. Returns `None` iff `values` is empty.
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
    let sigma2 = (sigma * sigma).max(MIN_COHORT_VARIANCE);
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

/// Build `telescope.emission.parquet` from a Universal Mode
/// `anchor_pool.parquet` (per-read rows).
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
) -> Result<()>
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

    // Deterministic output ordering: sort by transcript_id.
    let mut tx_keys: Vec<&String> = tx_accums.keys().collect();
    tx_keys.sort();

    let mut n_l1 = 0usize;
    let mut n_l2 = 0usize;
    let mut n_l3 = 0usize;

    for tx_id in tx_keys {
        let acc = &tx_accums[tx_id];
        let n = acc.len();
        // Universal-Mode polyA_frac: anchor filter enforces
        // pt:i >= 10, so by construction this is 1.0 for any
        // non-empty transcript. Documented in module docstring.
        let poly_a_frac = 1.0_f64;
        let is_poly_a_competent = n > 0;

        let fit = if n >= fallback.tx_min_reads {
            n_l1 += 1;
            // L1: per-tx fit. `n >= tx_min_reads >= 1` implies
            // `log2_pals` is non-empty, so unwrap is safe.
            fit_log_normal(&acc.log2_pals).expect("L1 fit must succeed when n >= tx_min_reads >= 1")
        } else if let Some(gf) = gene_fit.get(&acc.gene_id) {
            n_l2 += 1;
            // L2: per-gene fit. Shared across all underpowered txs in gene.
            *gf
        } else {
            n_l3 += 1;
            // L3: cohort-level fallback (always available).
            cohort_fit
        };

        tx_id_builder.append_value(tx_id);
        gene_id_builder.append_value(&acc.gene_id);
        frac_builder.append_value(poly_a_frac);
        competent_builder.append_value(is_poly_a_competent);
        count_builder.append_value(n as u32);
        mu_builder.append_value(fit.mu);
        sigma2_builder.append_value(fit.sigma2);
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

    eprintln!(
        "[telescope]  Universal mode: {} transcripts, {} cohort reads, \
         L1={} L2={} L3={}",
        tx_accums.len(),
        cohort_log2_pals.len(),
        n_l1,
        n_l2,
        n_l3,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Builder, ListBuilder};

    #[test]
    fn min_cohort_variance_value() {
        // Pinned in spec F3 / spec-self-audit-v0.1.3 R1.
        assert!((MIN_COHORT_VARIANCE - 0.04).abs() < f64::EPSILON);
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
    /// statistical equivalence test between AIDRS mode and
    /// Universal mode is the acceptance criterion for Step 3
    /// Part 2.
    ///
    /// Test fixture: 3 transcripts, 100 reads each, all reads
    /// uniquely mapping to 1 transcript. Reads sampled from a
    /// log-normal in `log2(PAL)` space (synthetic, deterministic
    /// pal values). The SAME data is laid out in two formats
    /// (AIDRS-format list-per-tx vs Universal-format per-read
    /// rows) and processed by the two modes. Per-transcript
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

        let tmp = std::env::temp_dir().join("telescope_test_aidrs_universal_equiv");
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

        let tmp = std::env::temp_dir().join("telescope_test_3level_fallback");
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
        build_emission_from_anchors(&anchor_pool, None::<&str>, &out, &fb)
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
        .max(MIN_COHORT_VARIANCE);
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
        .max(MIN_COHORT_VARIANCE);
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
        let exp_sigma2_c = (MAD_TO_SIGMA * mad).powi(2).max(MIN_COHORT_VARIANCE);
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

        let tmp = std::env::temp_dir().join("telescope_test_em_eligible_filter");
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
        build_emission_from_anchors(&anchor_pool, Some(&te_path), &out, &fb)
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
}
