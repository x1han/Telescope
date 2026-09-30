//! Module: Profile Summary Engine (`summary.rs`)
//!
//! Streams `telescope.profile.parquet`, performs memory-efficient aggregation,
//! computes Kish's effective sample size $n_{\text{eff}}$ and posterior-weighted
//! moments per (sample × gene × transcript).
//!
//! Architecture (spec v0.2.0 F5):
//! - Output: `telescope.sample_summary.parquet`.
//! - Memory profile: **O(K_txs)**, not O(N_reads). Sufficient statistics only;
//!   individual read points are NOT stored.
//! - Quantile columns (q05, q25, q50, q75, q95) are deliberately NOT in the
//!   locked schema; they move to a separate `distribution_summary.parquet`
//!   gated by `--with-quantiles` (D7, not yet implemented).
//!
//! Math:
//! - Kish $n_{\text{eff}} = (\sum w)^2 / \sum w^2$
//! - Weighted mean: $\bar{y}_w = \sum w \log_2 L / \sum w$
//! - Posterior-weighted variance (reliability weights):
//!   $\sigma^2_w = \frac{\sum w (y - \bar{y}_w)^2}{\sum w - \sum w^2 / \sum w}$
//!   computed analytically from sufficient statistics; no point iteration.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{Float64Array, Float64Builder, Int32Array, StringArray, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

/// Per-(sample, gene, transcript) accumulator. Streaming: only 7 f64 scalars.
#[derive(Debug, Default, Clone)]
struct TranscriptAccumulator {
    weight_sum: f64,
    sum_w2: f64,
    sum_w_log: f64,
    sum_w_log2: f64,
}

impl TranscriptAccumulator {
    #[inline]
    fn add(&mut self, length: i32, weight: f64) {
        if weight <= 0.0 || length <= 0 {
            return;
        }
        let log_pal = (length as f64).log2();
        self.weight_sum += weight;
        self.sum_w2 += weight * weight;
        self.sum_w_log += weight * log_pal;
        self.sum_w_log2 += weight * log_pal * log_pal;
    }

    /// Kish's effective sample size: $n_{\text{eff}} = (\sum w)^2 / \sum w^2$.
    /// Equals $N$ when all weights are equal; approaches 1 when one weight
    /// dominates. Captures the *statistical* information content of the
    /// weighted sample, not the raw read count.
    #[inline]
    fn effective_n(&self) -> f64 {
        if self.sum_w2 > 0.0 {
            (self.weight_sum * self.weight_sum) / self.sum_w2
        } else {
            0.0
        }
    }

    #[inline]
    fn weighted_mean(&self) -> f64 {
        if self.weight_sum > 0.0 {
            self.sum_w_log / self.weight_sum
        } else {
            0.0
        }
    }

    /// Posterior-weighted dispersion (reliability-weight variance).
    /// Posterior weights $w_{rk}$ are NOT sampling weights — they are
    /// probabilities of transcript assignment from TEL-EM. So this is a
    /// *posterior-weighted dispersion estimate*, not an unbiased sample
    /// variance in the classical frequentist sense.
    ///
    /// Algebraically closed form (no per-point iteration):
    /// $\sigma^2_w = (\sum w y^2 - (\sum w y)^2 / \sum w)
    ///              / (\sum w - \sum w^2 / \sum w)$
    #[inline]
    fn posterior_weighted_variance(&self) -> f64 {
        let sw = self.weight_sum;
        if sw <= 0.0 || self.sum_w2 <= 0.0 {
            return 0.0;
        }
        let swy2_minus = self.sum_w_log2 - (self.sum_w_log * self.sum_w_log) / sw;
        let denom = sw - self.sum_w2 / sw;
        if denom > 0.0 && swy2_minus > 0.0 {
            swy2_minus / denom
        } else {
            0.0
        }
    }
}

/// Stream `telescope.profile.parquet` and write `telescope.sample_summary.parquet`.
///
/// HashMap key: `(sample_id, gene_id, transcript_id)` — gene_id is part of the
/// key to prevent silent overwrites when AIDRS emits the same transcript_id
/// under multiple gene_ids (e.g. novel transcripts at gene boundaries).
pub fn summarize_profile_parquet<P, Q>(
    profile_parquet_path: P,
    output_summary_path: Q,
) -> Result<()>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
{
    let file = File::open(&profile_parquet_path).with_context(|| {
        format!(
            "Failed to open profile parquet: {:?}",
            profile_parquet_path.as_ref()
        )
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to build ParquetRecordBatchReaderBuilder")?;
    let mut reader = builder
        .with_batch_size(65536)
        .build()
        .context("Failed to build parquet reader")?;

    // Key: (sample_id, gene_id, transcript_id). Triple key avoids silent
    // gene_id overwrite when transcript_id is reused across gene boundaries.
    let mut map: HashMap<(String, String, String), TranscriptAccumulator> = HashMap::new();

    while let Some(batch_result) = reader.next() {
        let batch = batch_result.context("Failed to read profile batch")?;

        let sample_col = batch
            .column_by_name("sample_id")
            .context("Missing 'sample_id' column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("'sample_id' is not a StringArray")?;
        let gene_col = batch
            .column_by_name("gene_id")
            .context("Missing 'gene_id' column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("'gene_id' is not a StringArray")?;
        let tx_col = batch
            .column_by_name("transcript_id")
            .context("Missing 'transcript_id' column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("'transcript_id' is not a StringArray")?;
        let pal_col = batch
            .column_by_name("polyA_length")
            .context("Missing 'polyA_length' column")?
            .as_any()
            .downcast_ref::<Int32Array>()
            .context("'polyA_length' is not an Int32Array")?;
        let weight_col = batch
            .column_by_name("posterior_weight")
            .context("Missing 'posterior_weight' column")?
            .as_any()
            .downcast_ref::<Float64Array>()
            .context("'posterior_weight' is not a Float64Array")?;

        for i in 0..batch.num_rows() {
            let sample_id = sample_col.value(i).to_string();
            let gene_id = gene_col.value(i).to_string();
            let tx_id = tx_col.value(i).to_string();
            let length = pal_col.value(i);
            let weight = weight_col.value(i);

            let key = (sample_id, gene_id, tx_id);
            map.entry(key)
                .or_insert_with(TranscriptAccumulator::default)
                .add(length, weight);
        }
    }

    // Build output parquet (9 columns, locked schema).
    let mut sample_b = StringBuilder::new();
    let mut gene_b = StringBuilder::new();
    let mut tx_b = StringBuilder::new();
    let mut wsum_b = Float64Builder::new();
    let mut neff_b = Float64Builder::new();
    let mut swlog_b = Float64Builder::new();
    let mut swlog2_b = Float64Builder::new();
    let mut mean_b = Float64Builder::new();
    let mut var_b = Float64Builder::new();

    for ((sample_id, gene_id, tx_id), acc) in map {
        if acc.weight_sum <= 0.0 {
            continue;
        }
        sample_b.append_value(&sample_id);
        gene_b.append_value(&gene_id);
        tx_b.append_value(&tx_id);
        wsum_b.append_value(acc.weight_sum);
        neff_b.append_value(acc.effective_n());
        swlog_b.append_value(acc.sum_w_log);
        swlog2_b.append_value(acc.sum_w_log2);
        mean_b.append_value(acc.weighted_mean());
        var_b.append_value(acc.posterior_weighted_variance());
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("sample_id", DataType::Utf8, false),
        Field::new("gene_id", DataType::Utf8, false),
        Field::new("transcript_id", DataType::Utf8, false),
        Field::new("weight_sum", DataType::Float64, false),
        Field::new("effective_n", DataType::Float64, false),
        Field::new("sum_w_logPAL", DataType::Float64, false),
        Field::new("sum_w_logPAL2", DataType::Float64, false),
        Field::new("weighted_mean_logPAL", DataType::Float64, false),
        Field::new("posterior_weighted_variance", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(sample_b.finish()),
            Arc::new(gene_b.finish()),
            Arc::new(tx_b.finish()),
            Arc::new(wsum_b.finish()),
            Arc::new(neff_b.finish()),
            Arc::new(swlog_b.finish()),
            Arc::new(swlog2_b.finish()),
            Arc::new(mean_b.finish()),
            Arc::new(var_b.finish()),
        ],
    )
    .context("Failed to build sample_summary RecordBatch")?;

    let out_file = File::create(&output_summary_path).with_context(|| {
        format!(
            "Failed to create output parquet: {:?}",
            output_summary_path.as_ref()
        )
    })?;
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
    fn test_kish_effective_n_uniform_weights() {
        // 4 reads, weights 1.0 → n_eff = 4.0
        let mut acc = TranscriptAccumulator::default();
        for &pal in &[50_i32, 80, 100, 150] {
            acc.add(pal, 1.0);
        }
        assert!((acc.effective_n() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_kish_effective_n_skewed_weights() {
        // 100 reads, but 1 read has weight 100x the rest → n_eff < 100
        let mut acc = TranscriptAccumulator::default();
        for _ in 0..99 {
            acc.add(100, 1.0);
        }
        acc.add(100, 100.0);
        // sum_w = 99*1 + 100 = 199; sum_w2 = 99*1 + 10000 = 10099
        // n_eff = 199^2 / 10099 = 39601 / 10099 ≈ 3.92
        assert!(acc.effective_n() < 4.0);
        assert!(acc.effective_n() > 3.8);
    }

    #[test]
    fn test_posterior_weighted_variance_known_case() {
        // Hand-computed: weights 1, lengths log2 = 5, 7
        // sum_w = 2, sum_w2 = 2, sum_w_log = 12, sum_w_log2 = 25+49 = 74
        // mean = 12/2 = 6
        // var_num = 74 - 144/2 = 74 - 72 = 2
        // denom = 2 - 2/2 = 1
        // var = 2 / 1 = 2
        let mut acc = TranscriptAccumulator::default();
        acc.add(32, 1.0); // log2(32) = 5
        acc.add(128, 1.0); // log2(128) = 7
        assert!((acc.posterior_weighted_variance() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_zero_weight_safe() {
        let mut acc = TranscriptAccumulator::default();
        acc.add(100, 0.0);
        acc.add(100, 1.0);
        assert_eq!(acc.effective_n(), 1.0);
        assert_eq!(acc.posterior_weighted_variance(), 0.0);
    }
}
