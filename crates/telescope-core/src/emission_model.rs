//! Per-transcript poly(A) emission model $P(L \mid T_k)$ for TEL-EM: a
//! Gaussian in $\log_2(\text{PAL})$ space.
//!
//! - `build_emission_from_anchors` — Universal mode, the only CLI-reachable
//!   path; fits from the anchor pool with a 3-level fallback cascade
//!   (Tx → Gene → Cohort).
//! - `build_emission_parquet` — external poly(A) initializer, deferred and
//!   not reachable from any CLI subcommand: its read→transcript assignment
//!   has never been compared with Universal's, so it is kept working and
//!   tested but not re-exposed.
//!
//! Output: `telescope.emission.parquet` (schema in `output_schema()`). The
//! fitted object is a DISTRIBUTION, not a scalar: $\mu$ and $\sigma^2$ are
//! derived parameters of a fit over that transcript's per-read anchor
//! observations.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};

use crate::anchor_builder::TxToGene;
use arrow::array::{
    Array, ArrayBuilder, BooleanArray, BooleanBuilder, Float64Builder, Int32Array, ListArray,
    StringArray, StringBuilder, UInt32Builder,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

/// Physical instrumentation floor on cohort variance, in `log2(PAL)` space.
///
/// `sigma^2 = 0.04` is `sigma = 0.20` log2 units, i.e. a physical CV of
/// `2^0.20 - 1 ≈ 0.149` (~15%): the signal-noise floor of a nanopore poly(A)
/// tail. It must not be removed — with `sigma^2 -> 0` and `y_r == mu_k`
/// exactly, the Gaussian density is unbounded and one coincidental read can
/// dominate the E-step for that transcript.
pub const INSTRUMENTAL_PAL_NOISE_FLOOR: f64 = 0.04;

/// MAD-to-sigma scale factor for normal-consistency.
///
/// For a standard normal, `median(|X - median(X)|) * 1.4826 ≈ σ`. Used to
/// derive a robust cohort-level variance from the median absolute deviation
/// without parametric assumptions.
const MAD_TO_SIGMA: f64 = 1.4826;

/// Fallback thresholds for the 3-level Tx → Gene → Cohort cascade.
///
/// Every threshold is CLI-configurable and must stay so; the values below are
/// the defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FallbackConfig {
    /// L1 Tx-level: minimum reads to fit at the transcript level.
    /// Below this, fall to L2 (gene) or L3 (cohort).
    pub tx_min_reads: usize,
    /// L2 Gene-level: minimum pooled reads in the same `gene_id`.
    /// Below this, fall to L3 (cohort).
    pub gene_min_reads: usize,
    /// L3 Cohort-level: informational minimum for the *total* cohort read
    /// count. **Does not gate** the L3 fit — cohort fallback is always
    /// available as long as ≥1 read exists — but a below-threshold cohort is
    /// logged so downstream consumers can flag it.
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

/// Per-transcript poly(A) state from the external initializer's assessment
/// table.
#[derive(Debug, Clone, Copy)]
pub struct AssessmentQuality {
    pub poly_a_frac: f64,
    /// True iff `polyA_frac > 0` AND the assessment state was reachable.
    /// False for State 2 (`polyA_frac` effectively 0) and State 3
    /// (NaN — Dorado `pt:i` tag entirely missing).
    pub is_poly_a_competent: bool,
}

/// Reads `aidrs.transcript.assessment.tsv` and parses the `(TrID, polyA_frac)`
/// columns into a HashMap keyed by TrID.
///
/// State 3 (NaN / missing `polyA_frac`) and State 2 (`polyA_frac == 0` or
/// non-numeric) both become `(0.0, is_poly_a_competent = false)`. A numeric
/// non-zero value keeps its `polyA_frac` and flags the transcript as
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
        // `build_emission_from_anchors` was run with. Those thresholds are
        // not recorded in the parquet, so this reconstruction is only exact
        // when the reader knows them.
        Field::new("n_cohort_flnc", DataType::UInt32, false),
        Field::new("mu_logPAL", DataType::Float64, false),
        Field::new("sigma2_logPAL", DataType::Float64, false),
    ]))
}

/// Build `telescope.emission.parquet` from the external initializer's inputs.
///
/// Reads `aidrs.transcript_polyA_len.parquet` row-by-row (chunked), joins
/// each row's `TrID` against the assessment TSV to get `polyA_frac` and
/// `is_polyA_competent`, computes the log2(pal) mean/variance per
/// transcript, applies the `INSTRUMENTAL_PAL_NOISE_FLOOR` floor, and writes
/// the 7-column output schema above.
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

/// Per-transcript accumulator of `log2(PAL)` values from the anchor pool.
#[derive(Debug, Clone)]
struct TxAccumulator {
    gene_id: String,
    /// `log2(poly_a_length)` for every anchor read assigned to this
    /// transcript. (The anchor filter already requires `poly_a_length >= 10`,
    /// so values are finite positives.)
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
/// Both the external initializer and Universal mode produce the same shape:
/// `mu` is the arithmetic mean (L1) or median (L3) of `log2(PAL)`; `sigma2`
/// is the sample variance (L1/L2) or `(MAD * 1.4826)^2` (L3), floored at
/// `INSTRUMENTAL_PAL_NOISE_FLOOR`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct LogNormalFit {
    mu: f64,
    sigma2: f64,
}

/// Fit `(mean, sample variance floored at INSTRUMENTAL_PAL_NOISE_FLOOR)` from
/// a slice of `log2(PAL)` values. Returns `None` iff `values` is empty.
///
/// The values are SORTED before summing: `f64` addition is not associative,
/// so the order a multiset is summed in changes the result, and the L2 gene
/// pool is assembled by iterating a `HashMap`. Sorting removes the
/// per-process seed from the result.
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

/// Cohort-level robust fit: `mu` = median(log2(PAL)), `sigma2` =
/// `(MAD * 1.4826)^2`, floored at `INSTRUMENTAL_PAL_NOISE_FLOOR`. Returns
/// `None` iff `values` is empty.
///
/// The sort is load-bearing — do not remove it as a micro-optimisation.
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

/// Load the set of `transcript_id`s with `em_eligible == true` from
/// `telescope.transcript_evidence.parquet`.
///
/// This is the entry point for Universal mode's optional `em_eligible`
/// filtering.
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
            // per row → O(n²) total. `em_eligible` is non-null in the
            // schema; `value()` panics on null which is the intended
            // contract for schema-conformant inputs.
            if em_col.value(i) {
                set.insert(tx_col.value(i).to_string());
            }
        }
    }
    Ok(set)
}

/// Stream `telescope.anchor_pool.parquet` into per-transcript accumulators
/// plus a flat cohort-level `log2(PAL)` vector.
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
/// supplied a fit for, plus the reference-universe bookkeeping.
///
/// The counters are RETURNED, not only printed: a number that reaches nothing
/// but stderr cannot be read by a caller or a test.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FallbackTierCounts {
    /// L1: `n_reads_tx >= fallback.tx_min_reads` — per-transcript fit.
    pub l1: usize,
    /// L2: pooled gene reads `>= fallback.gene_min_reads` — the shared gene fit.
    pub l2: usize,
    /// L3: cohort-level fit (always available when ≥1 read).
    pub l3: usize,
    /// Rows IMPUTED to reach the reference universe: a transcript named by the
    /// reference that the anchor pool never observed. Zero unless a
    /// `reference_universe` was supplied.
    ///
    /// Deliberately NOT folded into `l3`: folding it there would silently
    /// rebase the published tier percentages onto the padded denominator.
    /// The partition invariant is therefore four terms, not three:
    /// `l1 + l2 + l3 + n_imputed == rows.len()`.
    ///
    /// `n_imputed` is the SUM of the two fields below and of nothing else.
    pub n_imputed: usize,
    /// Imputed rows that were never anchored: named by the reference universe
    /// and absent from the anchor pool, so nothing was observed about them.
    ///
    /// `is_polyA_competent = true` on these rows. They are padded for
    /// COMPLETENESS — a transcript the anchor pool never observed has not
    /// been shown to have no poly(A).
    pub n_imputed_padded: usize,
    /// Imputed rows that HAD anchors and were removed by the `em_eligible`
    /// filter before row assembly.
    ///
    /// `is_polyA_competent = false` on these rows. The suppression was
    /// DELIBERATE, which is a different fact from "never observed".
    pub n_imputed_suppressed: usize,
    /// Transcripts named by `reference_universe`. Zero when none was supplied.
    pub n_reference: usize,
    /// Anchored transcripts that the reference universe does NOT name.
    ///
    /// Non-zero means the model and the alignment reference disagree about
    /// which transcripts exist, so the emitted row count is NOT equal to
    /// `n_reference`.
    pub n_anchored_not_in_reference: usize,
    /// Absolute path of the per-transcript provenance sidecar just written.
    ///
    /// Returned rather than only logged, so a caller can open it.
    pub provenance_path: std::path::PathBuf,
}

/// The authoritative reference universe: every `@SQ SN` in an alignment BAM's
/// header, in header order.
///
/// A BAM is the authority because the emission model is keyed by the
/// transcripts an alignment can name; `tx_to_gene` maps transcripts that
/// exist but does not say which exist. The BAM is read HEADER-ONLY — no
/// record is touched.
///
/// TRAP: the header reader is a name-keyed MAP, so a header naming the same
/// transcript twice arrives here already collapsed, and the duplicate is not
/// visible to this function. `build_emission_from_anchors` therefore
/// re-checks the universe for duplicates itself and fails closed.
pub fn read_reference_universe<P: AsRef<Path>>(bam_path: P) -> Result<Vec<String>> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(bam_path.as_ref())
        .with_context(|| format!("Failed to open reference BAM: {:?}", bam_path.as_ref()))?;
    let mut reader = noodles_bam::io::Reader::new(BufReader::new(file));
    let header = reader
        .read_header()
        .with_context(|| format!("Failed to read BAM header: {:?}", bam_path.as_ref()))?;

    // `reference_sequences()` is keyed by ReferenceSequenceId and its iteration
    // order matches the `@SQ` lines, which is what makes the result
    // deterministic.
    Ok((0..header.reference_sequences().len())
        .map(|i| match header.reference_sequences().get_index(i) {
            Some((name, _)) => name.to_string(),
            None => String::new(),
        })
        .filter(|n| !n.is_empty())
        .collect())
}

/// The five provenance tiers, exactly as written into
/// `<stem>.provenance.parquet`'s `tier` column.
///
/// An IMPUTED row is not a level of the L1/L2/L3 cascade: it has no anchors
/// at all. `L3_imputed` (the gene was measured) and `L3_unexpressed` (it was
/// not) are therefore separate values even though both carry the cohort fit.
pub const TIER_L1: &str = "L1";
pub const TIER_L2: &str = "L2";
pub const TIER_L3_ANCHORED: &str = "L3_anchored";
pub const TIER_L3_IMPUTED: &str = "L3_imputed";
pub const TIER_L3_UNEXPRESSED: &str = "L3_unexpressed";

/// Schema of the per-transcript provenance sidecar: exactly four columns.
///
/// Deliberately NOT a column of `telescope.emission.parquet`: the fallback
/// level that produced a row's `(mu, sigma2)` is not answerable from
/// `n_cohort_flnc` alone, because L2 is decided by the GENE's pooled count
/// while the parquet carries only the TRANSCRIPT's own count.
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
/// The file name is derived from `emission_path` by replacing `.parquet` with
/// `.provenance.parquet`. The path is RETURNED as well as written.
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

/// Build `telescope.emission.parquet` from a Universal-mode
/// `anchor_pool.parquet` (per-read rows), fitting each transcript at the Tx,
/// Gene or Cohort level and returning the L1/L2/L3 split.
///
/// See [`FallbackTierCounts`] for the returned counts.
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
    // Fail closed on a reference universe that names a transcript twice: a
    // duplicate key would be pushed into the emitted row set twice while
    // `n_imputed` (a HashSet) counts it once, so the CLI's conservation law —
    // an identity on two SETS — would be violated by construction.
    if let Some(universe) = reference_universe {
        let mut seen = std::collections::HashSet::with_capacity(universe.len());
        for tx in universe {
            if !seen.insert(tx.as_str()) {
                anyhow::bail!(
                    "reference universe names transcript {tx:?} more than once; \
                     a duplicate key is pushed into the emitted row set twice while \
                     `n_imputed` (a HashSet) counts it once, so the conservation law \
                     is violated by construction rather than by accident",
                );
            }
        }
    }

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
    // `em_suppressed` records what the filter removed BY MOVE — the loop
    // consumes `tx_accums`, so cloning the dropped keys would cost a clone per
    // ineligible transcript. A suppressed transcript that the reference
    // universe names is still emitted, but as an INCOMPETENT row; dropping the
    // row instead would leave `rows.len() < n_reference` with no counter able
    // to see it.
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

    // Informational only: L3 is always available, but log a warning when the
    // cohort is suspiciously small.
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

    // 5+6. Per-transcript fit + output row assembly. Each transcript takes
    // the best of: L1 Tx-level if `n_reads_tx >= fallback.tx_min_reads`;
    // L2 Gene-level if pooled gene reads `>= fallback.gene_min_reads` (the
    // same `(mu, sigma2)` is shared across all underpowered transcripts in
    // that gene); L3 Cohort-level fit (always available when ≥1 read).
    // `polyA_frac` is 1.0 by construction (the anchor filter requires
    // `pt:i >= 10`) and `is_polyA_competent` is true for every non-empty
    // transcript.
    let mut tx_id_builder = StringBuilder::new();
    let mut gene_id_builder = StringBuilder::new();
    let mut frac_builder = Float64Builder::new();
    let mut competent_builder = BooleanBuilder::new();
    let mut count_builder = UInt32Builder::new();
    let mut mu_builder = Float64Builder::new();
    let mut sigma2_builder = Float64Builder::new();
    // Provenance sidecar, built in lockstep with the emission row below: a
    // SEPARATE FILE, not an eighth column, because the 7-column emission
    // schema is frozen and the fallback LEVEL that produced an ANCHORED row's
    // `(mu, sigma2)` is not recoverable from the parquet itself.
    let mut tier_builder = StringBuilder::new();

    // THE EMITTED KEY SET: normally the anchor pool's transcripts, and with a
    // reference universe the UNION of pool and universe. A universe key with
    // no anchors becomes an imputed row; a universe key that HAS anchors is
    // reached through `tx_accums`, so the union makes a double emit
    // structurally impossible.
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
    // re-read would be a second code path that could disagree with the first.
    let mut provenance_rows: Vec<(String, String, &'static str, u32)> =
        Vec::with_capacity(n_rows_emitted);

    for tx_id in tx_keys {
        // IMPUTED ROW. Reached only for a universe key with no entry in
        // `tx_accums`. Two distinct reasons land here, and they are NOT the
        // same row:
        //
        //   (a) never had anchors -- `is_polyA_competent = true`. That says
        //       the row is EVALUABLE, not that the transcript was measured to
        //       have a poly(A) tail.
        //   (b) HAD anchors and was removed by the `em_eligible` filter --
        //       `is_polyA_competent = false`. Marking this row true would let
        //       a transcript the resolver deliberately suppressed win
        //       posterior mass.
        //
        // `false` is not a demotion, it is the E-step's own switch: an
        // incompetent candidate is pushed as `-inf` and cannot win any mass.
        //
        // In both cases the fit is the cohort fit and `n_cohort_flnc = 0` is the
        // in-schema marker that says "imputed, not observed".
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
                    // LABEL ONLY: the fitted values, `is_polyA_competent` and
                    // the row count are identical in both arms — this changes
                    // what the table SAYS, not what it computes. `gene_fit`
                    // holds a fit for every gene whose POOLED anchors reached
                    // `gene_min_reads`, so it answers exactly one question
                    // about a never-anchored transcript: did anything in this
                    // dataset measure this LOCUS?
                    let tier = if gene_fit.contains_key(&gene) {
                        TIER_L3_IMPUTED
                    } else {
                        TIER_L3_UNEXPRESSED
                    };
                    (
                        gene,
                        1.0_f64,
                        true,
                        0_u32,
                        LogNormalFit {
                            mu: cohort_fit.mu,
                            sigma2: cohort_fit.sigma2,
                        },
                        tier,
                    )
                }
                Some(acc) => {
                    let n = acc.len();
                    // Universal-mode polyA_frac: the anchor filter enforces
                    // pt:i >= 10, so by construction this is 1.0 for any
                    // non-empty transcript.
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
        // `tx_min_reads` / `gene_min_reads`.
        //
        // `0` marks an IMPUTED row, emitted to reach the reference universe for
        // a transcript the anchor pool never observed. Inside the frozen
        // 7-column schema an imputed row is otherwise indistinguishable from a
        // transcript observed zero times; the discriminators live outside it
        // (`FallbackTierCounts::n_imputed` and the provenance sidecar).
        count_builder.append_value(n_obs);
        mu_builder.append_value(fit.mu);
        sigma2_builder.append_value(fit.sigma2);
        tier_builder.append_value(tier);
        provenance_rows.push((tx_id.clone(), gene_id.clone(), tier, n_obs));
    }

    // The physical builder must have received exactly one row per key the
    // solver decided to emit. `tx_keys` was MOVED by the emission loop above, so
    // `tx_keys.len()` is unreachable here; `n_rows_emitted` is its pre-move
    // capture, and the builder is still live because `finish()` has not run.
    //
    // This is the check the CLI conservation `ensure!` cannot make: that law
    // compares two cardinalities of `anchored UNION universe`, an identity on
    // two SETS, while a builder that dropped or duplicated a row changes the
    // emitted ARRAYS.
    anyhow::ensure!(
        tx_id_builder.len() == n_rows_emitted,
        "emission builder row count {} != emitted key count {}",
        tx_id_builder.len(),
        n_rows_emitted,
    );

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
    // are what it means for the run, and the two answer different questions: a
    // run that is 80% L3 is a run in which most transcripts carry no
    // transcript-specific poly(A) distribution at all.
    //
    // The split is RETURNED (see `FallbackTierCounts`) rather than formatted
    // here: a number that only reaches stderr is a number no test can read.
    //
    // The third number is the emitted row count: it is NOT `tx_accums.len()`
    // once a reference universe is supplied, and printing the anchored count
    // alone beside a padded table would state a row count the file does not
    // have.
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

    // `n_imputed_padded` and `n_imputed_suppressed` are computed by two
    // independent filters over `imputed_keys`, and `em_suppressed` is a
    // HashSet, so the two predicates are exact complements and the sum is
    // `imputed_keys.len()` by construction.
    let n_imputed = imputed_keys.len();
    let n_imputed_padded = imputed_keys
        .iter()
        .filter(|tx| !em_suppressed.contains(**tx))
        .count();
    let n_imputed_suppressed = imputed_keys
        .iter()
        .filter(|tx| em_suppressed.contains(**tx))
        .count();
    anyhow::ensure!(
        n_imputed == n_imputed_padded + n_imputed_suppressed,
        "imputed split is not a partition: n_imputed={n_imputed} but \
         padded={n_imputed_padded} + suppressed={n_imputed_suppressed}",
    );

    Ok(FallbackTierCounts {
        l1: n_l1,
        l2: n_l2,
        l3: n_l3,
        n_imputed,
        n_imputed_padded,
        n_imputed_suppressed,
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
    /// Local xfs `/tmp`, NOT `$TMPDIR`: on a network filesystem a file written
    /// and immediately re-read under concurrent I/O can come back 0 bytes.
    /// Plus the PID: a FIXED name is shared by every concurrent `cargo test`,
    /// and each test removes its directory at the START, so two runs delete
    /// each other's fixtures.
    fn local_scratch(name: &str) -> PathBuf {
        PathBuf::from("/tmp").join(format!("telescope_em_{}_{}", name, std::process::id()))
    }

    #[test]
    fn min_cohort_variance_value() {
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
    // Universal-mode test fixtures
    // ===================================================================

    /// Logical fixture row shared across the Universal-mode tests.
    /// Each `(gene_id, tx_id)` may appear at most once.
    struct TxFixture {
        gene_id: String,
        tx_id: String,
        pals: Vec<i32>,
    }

    /// Write an AIDRS-format `aidrs.transcript_polyA_len.parquet`
    /// (3-column subset) from a list of `TxFixture`.
    ///
    /// Retained with the deferred initializer it exercises: deleting it would
    /// delete `test_aidrs_universal_equivalence` with it.
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
    /// Retained with the deferred initializer it exercises; see
    /// `write_aidrs_parquet`.
    fn write_assessment_tsv(path: &Path, fxs: &[TxFixture]) {
        use std::fmt::Write as _;
        let mut s = String::from("TrID\tpolyA_frac\n");
        for fx in fxs {
            writeln!(s, "{}\t1.0", fx.tx_id).unwrap();
        }
        std::fs::write(path, s).expect("write assessment.tsv fixture");
    }

    /// Write a Universal-mode `anchor_pool.parquet` from the same logical
    /// fixtures: one row per `(tx_id, pal)`.
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

    /// Write a `transcript_evidence.parquet` fixture containing only the 2
    /// columns consumed by `load_em_eligible_set` (`transcript_id`,
    /// `em_eligible`).
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

    /// Read `telescope.emission.parquet` back into a
    /// `Vec<(transcript_id, gene_id, polyA_frac, is_polyA_competent,
    /// n_cohort_flnc, mu_logPAL, sigma2_logPAL)>` for assertions. Sorted by
    /// `transcript_id`.
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

    /// The deferred external initializer and Universal mode must AGREE: to
    /// <1e-9 relative on `mu_logPAL` and `sigma2_logPAL`, and EXACTLY on
    /// `n_cohort_flnc`.
    ///
    /// Do not convert this to prose and do not loosen its tolerances: it is
    /// the only executable evidence that the retained initializer is a real
    /// option rather than dead code, and the only thing pinning the fit as
    /// format-independent.
    ///
    /// Note what it does NOT show: that either path assigns reads to
    /// transcripts better than the other. Both consume the same (tx, pal)
    /// pairs, so upstream read→transcript assignment is held fixed and
    /// unmeasured.
    ///
    /// Fixture: 3 transcripts, 100 reads each. The SAME data is laid out in
    /// two formats (list-per-tx vs per-read rows) and processed by the two
    /// paths. `polyA_frac` and `is_polyA_competent` are documented to differ
    /// conceptually (Universal mode is always State 1 by construction).
    #[test]
    fn test_aidrs_universal_equivalence() {
        // Deterministic fixture: 3 transcripts × 100 reads each.
        // Use pal values [a, a+1, ..., a+99] per tx — the actual
        // distribution shape is irrelevant; what matters is that
        // both paths see IDENTICAL (tx, pal) pairs.
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
        let tol = 1e-9_f64; // relative tolerance for the two paths to agree
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

    /// `em_eligible` filtering drops reads whose transcript is not eligible:
    /// even though `tx_filtered` has reads in the anchor pool, those reads
    /// must NOT contribute to the output (its emission row is dropped).
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
    /// The column is easy to read as a count of distinct values; it is
    /// `acc.len()` — the number of anchor rows for the transcript, one per
    /// (sample, transcript, read). A transcript whose 2 reads carry the SAME
    /// `poly_a_length` therefore reports 2, not 1.
    ///
    /// The distinct-values case is consistent with BOTH readings, so only the
    /// degenerate case discriminates: a mutation that made this column count
    /// distinct lengths would otherwise leave the suite green.
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
    /// TWO THINGS ARE ASSERTED, and they fail independently: (1) the SPLIT
    /// `(l1, l2, l3) == (1, 4, 1)` read off the return value, which is what
    /// pins the counters themselves, and (2) the ATTRIBUTION — which fit each
    /// transcript was actually given.
    ///
    /// THE FIXTURE. L2 is the tier a naive fixture cannot reach: it needs a
    /// gene pool strictly larger than any single transcript's pool, so several
    /// DISTINCT transcripts share one `gene_id`. Here `tx_min_reads = 10` and
    /// `gene_min_reads = 30`: each L2 transcript is capped at 9 reads, so four
    /// of them are needed to clear 30 in the pool.
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
                n_imputed_padded: 0,
                n_imputed_suppressed: 0,
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
        // sharing is the mechanism behind `n_polya_zero_power` in the solver,
        // and an L2 that recomputed per transcript would break that metric.
        //
        // The gene pool is assembled by iterating `tx_accums`, a `HashMap`, so
        // the order the 36 values arrive in is not fixed and `fit_log_normal`'s
        // `sum::<f64>()` is not associative — which is why `fit_log_normal`
        // sorts before summing. The 1e-9 tolerance below is therefore slack
        // the code no longer needs; it is kept because the re-derivation is a
        // second, independent call and a ULP-level assertion is a poor
        // failure message. The exact-equality claim lives in
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
    /// emitted L2 rows differing in the last 1-2 mantissa bits between runs.
    ///
    /// WHY THIS TEST IS NOT "RUN THE BUILD TWICE". `HashMap` iteration order is
    /// STABLE WITHIN A PROCESS, so re-running the same build in the same
    /// process reproduces the non-deterministic value exactly and passes with
    /// the bug present. The only way to reach two different iteration orders
    /// from one process is to build two maps with different insertion
    /// histories, which is what the three orders below do.
    ///
    /// `to_bits()`, not `==`: the point of the test is the bit pattern, not the
    /// value.
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
        // removed as a micro-optimisation.
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
    ///   * `n_anchor == 0` if and only if the tier is `L3_imputed` OR
    ///     `L3_unexpressed` — "a row with no anchors of its own" has TWO
    ///     labels, and the two encodings of "imputed" must not drift apart;
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
        // own count is under `tx_min_reads`. Only the `n_l2 > 0` CONTROL below
        // catches that, which is the reason that control is here.
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
            // `n_anchor == 0` marks a row with NO anchors of its own. That
            // population has TWO tiers, not one: `L3_imputed` (the gene was
            // measured) and `L3_unexpressed` (it was not). The invariant is
            // therefore over the SET, and this fixture has no gene map, so
            // every padded row is `L3_unexpressed` — which is why this test
            // exercises the label but not the split.
            let never_anchored = *tier == TIER_L3_IMPUTED || *tier == TIER_L3_UNEXPRESSED;
            assert_eq!(
                *n_anchor == 0,
                never_anchored,
                "{tx}: n_anchor == 0 must mean exactly 'a row with no anchors of \
                 its own' (L3_imputed or L3_unexpressed), and this row is tier \
                 {tier} with n_anchor {n_anchor} (gene {gene})"
            );
        }

        let n_l1 = prov.iter().filter(|r| r.2 == TIER_L1).count();
        let n_l2 = prov.iter().filter(|r| r.2 == TIER_L2).count();
        let n_l3a = prov.iter().filter(|r| r.2 == TIER_L3_ANCHORED).count();
        // `n_imputed` counts ROWS with no anchors of their own, and that
        // population carries two labels. The histogram is over ROWS, so the
        // imputed bucket is the SUM — counting only `L3_imputed` would make
        // this test assert 0 against a counter that says 2, and "fixing" it by
        // relaxing the comparison would leave the histogram unchecked.
        let n_l3i = prov
            .iter()
            .filter(|r| r.2 == TIER_L3_IMPUTED || r.2 == TIER_L3_UNEXPRESSED)
            .count();
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
        // This fixture supplies NO gene map, so every padded transcript's gene
        // is its own id and nothing anchors it: the two rows are BOTH
        // `L3_unexpressed`. The split itself is covered by
        // `the_tier_label_splits_a_padded_transcript_by_whether_its_gene_was_measured`.
        assert_eq!(
            prov.iter().filter(|r| r.2 == TIER_L3_UNEXPRESSED).count(),
            2,
            "with no gene map, no padded transcript has a measured gene"
        );
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

    /// A never-anchored transcript is `L3_imputed` when its OWN GENE has an
    /// entry in `gene_fit`, and `L3_unexpressed` when it does not.
    ///
    /// `gene_fit` is built from genes whose POOLED anchors reach
    /// `gene_min_reads`, so its membership answers exactly one question about
    /// a transcript with no anchors of its own: **did this dataset measure this
    /// locus at all?** A gene that made the cut has real measurements behind
    /// it, so filling in one of its isoforms is a derivation; a gene that did
    /// not has nothing but the cohort-wide prior.
    ///
    /// IT IS A LABEL AND NOTHING ELSE. Every fitted value,
    /// `is_polyA_competent` and the row COUNT is asserted identical across the
    /// two arms: a change that quietly started fitting the gene instead of the
    /// cohort would keep the tier label and move `(mu, sigma2)`.
    ///
    /// THE FIXTURE uses a real `tx_to_gene`, because with no map
    /// `gene_id := transcript_id` and every padded transcript's gene is its own
    /// id, which nobody anchors — so the split would collapse to "everything
    /// is unexpressed" and the test would pass for the wrong reason.
    #[test]
    fn the_tier_label_splits_a_padded_transcript_by_whether_its_gene_was_measured() {
        let tmp = local_scratch("telescope_test_tiersplit");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir tmp");

        // `FallbackConfig::default()` gene_min_reads is 30; gene_a pools 15
        // here, so give gene_a enough anchors by adding a second transcript
        // under the same gene.
        let fxs = vec![
            TxFixture {
                gene_id: "gene_a".into(),
                tx_id: "tx_a1".into(),
                pals: (80..=95).collect(),
            },
            TxFixture {
                gene_id: "gene_a".into(),
                tx_id: "tx_a2".into(),
                pals: (81..=94).collect(),
            },
            TxFixture {
                gene_id: "gene_b".into(),
                tx_id: "tx_b1".into(),
                pals: vec![85, 86],
            },
        ];
        let anchor_pool = tmp.join("anchor_pool.parquet");
        let out = tmp.join("emission.parquet");
        write_anchor_pool_parquet(&anchor_pool, &fxs);

        // gene_a: 16 + 14 = 30 pooled anchors == gene_min_reads -> a gene_fit
        // entry. gene_b: 2 -> none.
        let tsv = tmp.join("tx_to_gene.tsv");
        std::fs::write(
            &tsv,
            "transcript_id\tgene_id\n\
             tx_a1\tgene_a\n\
             tx_a2\tgene_a\n\
             tx_pad_a\tgene_a\n\
             tx_b1\tgene_b\n\
             tx_pad_b\tgene_b\n",
        )
        .expect("write tx2gene");
        let map = TxToGene::from_tsv(&tsv).expect("parse tx2gene");
        assert_eq!(
            map.len(),
            5,
            "the map must not be empty, or nothing classifies"
        );
        let universe: Vec<String> = vec![
            "tx_a1".into(),
            "tx_a2".into(),
            "tx_b1".into(),
            "tx_pad_a".into(),
            "tx_pad_b".into(),
        ];
        let tiers = build_emission_from_anchors(
            &anchor_pool,
            None::<&str>,
            &out,
            &FallbackConfig::default(),
            Some(&universe),
            Some(&map),
        )
        .expect("Universal mode run with a reference universe");

        let prov = read_provenance_parquet(&tiers.provenance_path);
        let tier_of = |tx: &str| -> String {
            prov.iter()
                .find(|r| r.0 == tx)
                .unwrap_or_else(|| panic!("{tx} missing from the provenance table"))
                .2
                .to_string()
        };

        assert_eq!(
            tier_of("tx_pad_a"),
            TIER_L3_IMPUTED,
            "gene_a pooled 30 anchors and reached gene_min_reads, so this locus \
             WAS measured — filling in one of its isoforms is a legal imputation"
        );
        assert_eq!(
            tier_of("tx_pad_b"),
            TIER_L3_UNEXPRESSED,
            "gene_b pooled 2 anchors and has no gene_fit entry, so nothing in \
             this dataset informs tx_pad_b beyond the cohort-wide prior"
        );
        assert_ne!(
            tier_of("tx_pad_a"),
            tier_of("tx_pad_b"),
            "the whole point of the third status: two identical cohort fits, \
             labelled by what is known about their loci"
        );

        // And it really is label-only: same fit, same competence, same count.
        let rows = read_emission_parquet(&out);
        let idx = |tx: &str| rows.iter().position(|r| r.0 == tx).expect("row");
        let (a, b) = (idx("tx_pad_a"), idx("tx_pad_b"));
        assert_eq!(rows[a].5, rows[b].5, "mu unchanged by the label");
        assert_eq!(rows[a].6, rows[b].6, "sigma2 unchanged by the label");
        assert_eq!(rows[a].3, rows[b].3, "is_polyA_competent unchanged");
        assert_eq!(rows[a].4, 0, "n_cohort_flnc unchanged");
        assert_eq!(tiers.n_imputed, 2);

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
    /// Writing both as `is_poly_a_competent = true` would tell the EM solver
    /// that a deliberately suppressed transcript has a flawless poly(A)
    /// distribution, silently undoing the suppression downstream.
    ///
    /// `tx_anchorless` is the CONTROL and it is not decoration. It is in the
    /// universe but was never anchored, so the filter never saw it and it must
    /// stay competent. Without that row a mutation that marked *every* `None`
    /// transcript incompetent would pass this test, while
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

        // The split of `n_imputed` into its two detail counts.
        // `n_imputed` stays the backwards-compatible TOTAL, and
        // the two detail fields partition it. Asserted as an identity rather
        // than trusted, because the split is computed from the same key set and
        // a mis-filtered `em_suppressed` lookup would make both halves add up
        // while putting the wrong transcript in each.
        assert_eq!(
            tiers.n_imputed_padded, 1,
            "only tx_anchorless was never observed"
        );
        assert_eq!(
            tiers.n_imputed_suppressed, 1,
            "only tx_filtered was suppressed by em_eligible"
        );
        assert_eq!(
            tiers.n_imputed_padded + tiers.n_imputed_suppressed,
            tiers.n_imputed,
            "the two detail fields must PARTITION n_imputed, not extend it"
        );
        // A no-universe run has no imputed rows of either kind.
        assert_eq!(tiers.n_imputed_suppressed, 1, "exactly the suppressed one");

        // The suppressed row carries the COHORT fit, not a fit of its own --
        // it had anchors once, but they were filtered out before any fitting
        // happened, so the cohort is the only thing that can have fitted it.
        let cohort_in: Vec<f64> = (10..20).map(|p| (p as f64).log2()).collect();
        let cohort = cohort_log_normal_fit(&cohort_in).expect("cohort fit");
        assert_eq!(sup.4, 0, "repeated for the reason stated at the assert");
        assert!(
            (sup.5 - cohort.mu).abs() < 1e-12,
            "the suppressed row carries the COHORT mu. got {} want {}",
            sup.5,
            cohort.mu
        );
        assert!(
            (sup.6 - cohort.sigma2).abs() < 1e-12,
            "the suppressed row carries the COHORT sigma2. got {} want {}",
            sup.6,
            cohort.sigma2
        );
        assert_eq!(sup.2, 1.0, "polyA_frac");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A reference universe that names a transcript twice is REJECTED at the
    /// entry point, not de-duplicated.
    ///
    /// A duplicate key would push the row into `tx_keys` once per occurrence,
    /// so the row was emitted TWICE with byte-identical contents while
    /// `imputed_keys` -- being a `HashSet` -- counted it once. The CLI
    /// conservation law is the only thing that could notice, and it could not:
    /// both sides of that law are the cardinality of `anchored UNION universe`,
    /// an identity on two SETS.
    ///
    /// The final control rejects nothing: a distinct universe must still
    /// build, so the check cannot be satisfied by refusing everything.
    #[test]
    fn a_duplicated_universe_key_is_rejected_at_the_entry_point() {
        let tmp = local_scratch("telescope_test_dup_universe");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir tmp");

        let fxs = vec![TxFixture {
            gene_id: "gene_a".into(),
            tx_id: "tx_only".into(),
            pals: (80..92).collect::<Vec<i32>>(),
        }];
        let anchor_pool = tmp.join("anchor_pool.parquet");
        let out = tmp.join("emission.parquet");
        write_anchor_pool_parquet(&anchor_pool, &fxs);

        // `tx_pad` appears TWICE: the input that used to emit two identical rows
        // for one transcript while `n_imputed` counted it once.
        let universe: Vec<String> = vec!["tx_only".into(), "tx_pad".into(), "tx_pad".into()];
        let err = build_emission_from_anchors(
            &anchor_pool,
            None::<&str>,
            &out,
            &FallbackConfig::default(),
            Some(&universe),
            None,
        )
        .expect_err("a duplicated universe must be REJECTED, not silently de-duplicated");

        let msg = err.to_string();
        assert!(
            msg.contains("tx_pad"),
            "the error must NAME the offending transcript; an operator holding a \
             70k-entry universe cannot otherwise tell which entry to fix. got: {msg}"
        );

        // CONTROL: the rejection precedes every write, so the corrupt table this
        // input used to produce cannot exist on disk at all.
        assert!(
            !out.exists(),
            "no output parquet may be written when the universe is rejected; the \
             old behaviour wrote one containing two identical tx_pad rows"
        );

        // CONTROL: a well-formed universe is still accepted. Without this the
        // check would pass by refusing everything, which is the other failure
        // mode a fail-closed guard has.
        let good: Vec<String> = vec!["tx_only".into(), "tx_pad".into()];
        build_emission_from_anchors(
            &anchor_pool,
            None::<&str>,
            &out,
            &FallbackConfig::default(),
            Some(&good),
            None,
        )
        .expect("a universe with distinct keys must still build");
        let rows = read_emission_parquet(&out);
        assert_eq!(rows.len(), 2, "1 anchored + 1 padded; got rows={rows:?}");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `n == 0` inside the parquet is ambiguous on purpose, and this pins the
    /// ambiguity rather than leaving it to a reader to discover: with a
    /// universe, `0` means "imputed"; without one, it cannot occur at all.
    ///
    /// The two conditions are DISJOINT by construction, which is what makes a
    /// single marker usable: there is no run in which a `0` is ambiguous
    /// between "imputed" and "observed zero times".
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
