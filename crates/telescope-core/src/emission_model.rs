//! Module 1: Emission Model Builder
//!
//! Parses AIDRS outputs
//! (`aidrs.transcript_polyA_len.parquet` +
//! `aidrs.transcript.assessment.tsv`)
//! and constructs the per-transcript emission model
//! $P(L \mid T_k)$ — a Gaussian in $\log_2(\text{PAL})$
//! space — for TEL-EM.
//!
//! Output: `telescope.emission.parquet` with the schema
//! declared in `output_schema()`. Each row is one transcript
//! $T_k$ with computed emission parameters
//! $(\mu_{k,\text{cohort}}, \sigma^2_{k,\text{cohort}})$ and
//! the `is_polyA_competent` classification flag from AIDRS
//! quality.
//!
//! Source spec: `workspace/specs/telescope-v0.1.3.md` F2 + F3.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use arrow::array::{
    Array, BooleanBuilder, Float64Builder, ListArray, StringBuilder, UInt32Builder,
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
    let file = File::open(&tsv_path)
        .with_context(|| format!("Failed to open {:?}", tsv_path.as_ref()))?;
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
            Ok(val) if val.is_nan() => (0.0, false),    // State 3
            Ok(val) if val <= 0.0 => (0.0, false),     // State 2
            Ok(val) => (val, true),                    // State 1
            Err(_) => (0.0, false),                    // unparseable
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
    let file = File::open(&poly_a_parquet_path).with_context(|| {
        format!("Failed to open {:?}", poly_a_parquet_path.as_ref())
    })?;
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
                let log_lens: Vec<f64> =
                    lengths.iter().map(|&x| (x as f64).log2()).collect();
                let mean = log_lens.iter().sum::<f64>() / (n_flnc as f64);
                let variance = if n_flnc > 1 {
                    let sq_diff_sum: f64 =
                        log_lens.iter().map(|&x| (x - mean).powi(2)).sum();
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
    let out_file = File::create(&output_emission_path)
        .with_context(|| format!("Failed to create output {:?}", output_emission_path.as_ref()))?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(out_file, schema, Some(props))
        .context("Failed to create ArrowWriter")?;
    writer.write(&batch).context("Failed to write batch")?;
    writer.close().context("Failed to close writer")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}