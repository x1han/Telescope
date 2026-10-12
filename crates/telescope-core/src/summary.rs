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

use anyhow::{bail, Context, Result};
use arrow::array::{
    Float64Array, Float64Builder, Int32Array, StringArray, StringBuilder, UInt32Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

/// One row of the per-transcript provenance sidecar written by
/// `build-emission`. See `emission_model::provenance_schema`.
struct ProvenanceRow {
    gene_id: String,
    tier: &'static str,
    n_anchor: u32,
}

/// Read the provenance sidecar into `transcript_id -> ProvenanceRow`.
///
/// The sidecar is the AUTHORITY on which transcripts exist. `telescope quant`
/// can only ever emit a profile row for a transcript the emission model
/// evaluated, so the profile is a strict subset of the universe, and a
/// transcript with zero weight is invisible in the profile. This is what makes
/// an inner join silently lose 1,334 HC2 rows.
fn load_provenance<P: AsRef<Path>>(path: P) -> Result<Vec<(String, ProvenanceRow)>> {
    let file = File::open(&path)
        .with_context(|| format!("Failed to open provenance parquet: {:?}", path.as_ref()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to build provenance ParquetRecordBatchReaderBuilder")?
        .with_batch_size(65536)
        .build()
        .context("Failed to build provenance parquet reader")?;

    let mut out: Vec<(String, ProvenanceRow)> = Vec::new();
    for batch in reader {
        let batch = batch.context("Failed to read provenance batch")?;
        let tx = batch
            .column_by_name("transcript_id")
            .context("Missing 'transcript_id' in provenance")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("provenance 'transcript_id' is not a StringArray")?;
        let gene = batch
            .column_by_name("gene_id")
            .context("Missing 'gene_id' in provenance")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("provenance 'gene_id' is not a StringArray")?;
        let tier = batch
            .column_by_name("tier")
            .context("Missing 'tier' in provenance")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("provenance 'tier' is not a StringArray")?;
        let n_anchor = batch
            .column_by_name("n_anchor")
            .context("Missing 'n_anchor' in provenance")?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("provenance 'n_anchor' is not a UInt32Array")?;
        for i in 0..batch.num_rows() {
            // `tier` is an enum in the writer; an unknown value here means the
            // sidecar came from a different version, and defaulting it to
            // something plausible is exactly the silent-wrong-answer this
            // table exists to prevent.
            let t = match tier.value(i) {
                "L1" => "L1",
                "L2" => "L2",
                "L3_anchored" => "L3_anchored",
                "L3_imputed" => "L3_imputed",
                other => bail!("provenance 'tier' has unknown value {:?}", other),
            };
            out.push((
                tx.value(i).to_string(),
                ProvenanceRow {
                    gene_id: gene.value(i).to_string(),
                    tier: t,
                    n_anchor: n_anchor.value(i),
                },
            ));
        }
    }
    Ok(out)
}

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
///
/// `provenance_path`, when given, is the per-transcript sidecar written by
/// `build-emission`. It changes the output in three ways, all of which are
/// the reason it exists:
///
/// 1. **The key set becomes the reference universe, not the profile.** Every
///    transcript in the sidecar gets a row. Without it, a transcript that no
///    read landed on is absent from the output entirely -- an inner join, which
///    on HC2 silently drops 1,334 of 70,511 transcripts.
/// 2. **`tier` and `n_anchor` columns appear.** These are not recoverable from
///    the profile or the frozen 7-column emission parquet.
/// 3. **Row order becomes deterministic.** See the sort below.
///
/// When it is `None` the function behaves exactly as it did before: one row
/// per (sample, gene, transcript) that received weight, in unspecified order.
///
/// A `sample_id` is required for the zero-filled rows and there is no way to
/// derive one: the sidecar is per-transcript and carries no sample column. It
/// is taken from the profile, and a profile covering MORE THAN ONE sample is
/// rejected rather than guessed, because attributing 1,334 zero-weight
/// transcripts to an arbitrary sample is a silent wrong answer.
pub fn summarize_profile_parquet<P, Q>(
    profile_parquet_path: P,
    output_summary_path: Q,
    provenance_path: Option<&Path>,
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

    // Resolve the output key set and fill in the zero-weight rows.
    //
    // `rows` is a Vec of fully-resolved output rows rather than the HashMap,
    // for two reasons that are both correctness requirements and not style:
    // the universe must be able to contribute rows the HashMap has no entry
    // for, and the output order must not depend on HashMap iteration order
    // (Rust's default hasher is seeded per process, so `for (k, v) in map`
    // yields a different order on every run).
    let provenance = match provenance_path {
        Some(p) => Some(load_provenance(p)?),
        None => None,
    };

    let mut rows: Vec<(
        String,
        String,
        String,
        TranscriptAccumulator,
        &'static str,
        u32,
    )> = Vec::new();

    if let Some(prov) = &provenance {
        let mut prov_index: HashMap<&str, &ProvenanceRow> = HashMap::with_capacity(prov.len());
        for (tx, r) in prov {
            prov_index.insert(tx.as_str(), r);
        }

        let mut sample_ids: Vec<&str> = map.keys().map(|(s, _, _)| s.as_str()).collect();
        sample_ids.sort_unstable();
        sample_ids.dedup();
        if sample_ids.len() != 1 {
            bail!(
                "summarize with --provenance requires a single-sample profile, \
                 because a zero-weight transcript has no sample to belong to. \
                 Found {}: {:?}. Run one sample at a time, or drop --provenance.",
                sample_ids.len(),
                sample_ids
            );
        }
        let sample_id = sample_ids[0].to_string();

        // A profile row outside the universe means the two artifacts disagree
        // about which transcripts exist. That is exactly the class of bug the
        // provenance chain exists to catch, so it is a hard error rather than
        // a row with a guessed tier.
        let orphans: Vec<&str> = map
            .keys()
            .map(|(_, _, tx)| tx.as_str())
            .filter(|tx| !prov_index.contains_key(tx))
            .collect();
        if !orphans.is_empty() {
            let mut shown: Vec<&str> = orphans.clone();
            shown.sort_unstable();
            shown.truncate(5);
            bail!(
                "{} transcript(s) carry weight in the profile but are absent from \
                 the provenance sidecar: {:?}{}. The emission universe and the \
                 profile disagree about which transcripts exist.",
                orphans.len(),
                shown,
                if orphans.len() > shown.len() {
                    ", ..."
                } else {
                    ""
                }
            );
        }

        // Iterate the UNIVERSE, in the sidecar's order, so the output is the
        // full 70,511-row closure with the profile's aggregates left-joined
        // onto it. A transcript no read landed on simply has no HashMap entry
        // and keeps the default accumulator: weight_sum, effective_n and cpm
        // are all 0.0 by construction, not by special-casing.
        for (tx, p) in prov {
            let acc = map
                .get(&(sample_id.clone(), p.gene_id.clone(), tx.clone()))
                .or_else(|| {
                    // The HashMap key is (sample, gene, tx) and a transcript's
                    // gene_id can, in principle, differ between the profile and
                    // the sidecar. Falling back to any entry for this
                    // transcript keeps the aggregate rather than silently
                    // zeroing a transcript that demonstrably has weight.
                    map.iter()
                        .find(|((s, _, t), _)| s == &sample_id && t == tx)
                        .map(|(_, a)| a)
                })
                .cloned()
                .unwrap_or_default();
            rows.push((
                sample_id.clone(),
                p.gene_id.clone(),
                tx.clone(),
                acc,
                p.tier,
                p.n_anchor,
            ));
        }
    } else {
        for ((sample_id, gene_id, tx_id), acc) in map {
            if acc.weight_sum <= 0.0 {
                continue;
            }
            rows.push((sample_id, gene_id, tx_id, acc, "L1_ungoverned", 0));
        }
    }

    // DETERMINISTIC ORDER. Not cosmetic: a byte-comparison between two runs on
    // the same input is only meaningful if the row order is fixed, and
    // HashMap iteration order is not fixed across processes.
    rows.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.1.cmp(&b.1))
    });

    // CPM denominator: the sample's total posterior weight. Zero-weight rows
    // contribute 0 to it and receive 0.0 back, so a transcript that was never
    // observed does not dilute the others' CPM.
    //
    // cpm = 1e6 * weight_sum / SUM(weight_sum over the sample)
    //
    // There is NO transcript-length normalisation anywhere in this path, and
    // that is correct rather than an oversight. Direct RNA sequencing samples
    // native molecules one read per molecule; it does not shear transcripts
    // into length-biased fragments, so the fragment-count-per-length term that
    // defines RNA-seq TPM has no physical referent here. The column was
    // named `tpm` until 2026-10-12 and was mislabelled: arithmetically it has
    // always been CPM. Renamed rather than aliased so the unit in the schema
    // matches the quantity in the formula.
    let total_weight: f64 = rows.iter().map(|r| r.3.weight_sum).sum();
    let cpm_scale = if total_weight > 0.0 {
        1.0e6 / total_weight
    } else {
        0.0
    };

    // Build output parquet (12 columns).
    let mut sample_b = StringBuilder::new();
    let mut gene_b = StringBuilder::new();
    let mut tx_b = StringBuilder::new();
    let mut wsum_b = Float64Builder::new();
    let mut neff_b = Float64Builder::new();
    let mut swlog_b = Float64Builder::new();
    let mut swlog2_b = Float64Builder::new();
    let mut mean_b = Float64Builder::new();
    let mut var_b = Float64Builder::new();
    let mut tier_b = StringBuilder::new();
    let mut n_anchor_b = arrow::array::UInt32Builder::new();
    let mut cpm_b = Float64Builder::new();

    for (sample_id, gene_id, tx_id, acc, tier, n_anchor) in rows {
        sample_b.append_value(&sample_id);
        gene_b.append_value(&gene_id);
        tx_b.append_value(&tx_id);
        wsum_b.append_value(acc.weight_sum);
        neff_b.append_value(acc.effective_n());
        swlog_b.append_value(acc.sum_w_log);
        swlog2_b.append_value(acc.sum_w_log2);
        mean_b.append_value(acc.weighted_mean());
        var_b.append_value(acc.posterior_weighted_variance());
        tier_b.append_value(tier);
        n_anchor_b.append_value(n_anchor);
        cpm_b.append_value(acc.weight_sum * cpm_scale);
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
        // Added 2026-10-06. The previous 9-column schema is NOT preserved as
        // an alternative: `tier` is the whole point of the provenance work,
        // and `cpm` is what a downstream differential pipeline consumes.
        // Both are described where they are produced.
        Field::new("tier", DataType::Utf8, false),
        Field::new("n_anchor", DataType::UInt32, false),
        Field::new("cpm", DataType::Float64, false),
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
            Arc::new(tier_b.finish()),
            Arc::new(n_anchor_b.finish()),
            Arc::new(cpm_b.finish()),
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

    // ---- fixtures for the provenance / closure tests ---------------------
    //
    // Local xfs `/tmp`, NOT `$TMPDIR`: `$TMPDIR` here is `/datg/hanxi/tmp` on
    // nfs4, where a parquet written and immediately re-read under concurrent
    // test load comes back as 0 bytes ("EOF: Parquet file too small"). That
    // flaked a different test in this crate once already; see the
    // `local_scratch` note in em.rs.

    fn scratch(name: &str) -> std::path::PathBuf {
        std::path::PathBuf::from("/tmp").join(format!(
            "telescope_summary_{}_{}",
            name,
            std::process::id()
        ))
    }

    /// Write a profile parquet. `rows` is `(transcript_id, gene_id, pal, weight)`.
    fn write_profile(path: &std::path::Path, sample: &str, rows: &[(&str, &str, i32, f64)]) {
        use arrow::array::BooleanArray;
        use arrow::array::Int32Builder;
        let schema = Arc::new(Schema::new(vec![
            Field::new("sample_id", DataType::Utf8, false),
            Field::new("gene_id", DataType::Utf8, false),
            Field::new("transcript_id", DataType::Utf8, false),
            Field::new("polyA_length", DataType::Int32, false),
            Field::new("posterior_weight", DataType::Float64, false),
            // Present but unread: the summary engine must not care.
            Field::new("unused_flag", DataType::Boolean, false),
        ]));
        let mut pal_b = Int32Builder::new();
        let mut w_b = Float64Builder::new();
        for &(_, _, pal, w) in rows {
            pal_b.append_value(pal);
            w_b.append_value(w);
        }
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(
                    rows.iter().map(|_| sample).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|r| r.1).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|r| r.0).collect::<Vec<_>>(),
                )),
                Arc::new(pal_b.finish()),
                Arc::new(w_b.finish()),
                Arc::new(BooleanArray::from(vec![true; rows.len()])),
            ],
        )
        .expect("profile batch");
        let f = File::create(path).expect("create profile");
        let mut w = ArrowWriter::try_new(f, schema, Some(WriterProperties::builder().build()))
            .expect("profile writer");
        w.write(&batch).expect("write profile");
        w.close().expect("close profile");
    }

    /// Write a provenance sidecar in the 4-column shape `build-emission` emits.
    fn write_provenance(path: &std::path::Path, rows: &[(&str, &str, &str, u32)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("transcript_id", DataType::Utf8, false),
            Field::new("gene_id", DataType::Utf8, false),
            Field::new("tier", DataType::Utf8, false),
            Field::new("n_anchor", DataType::UInt32, false),
        ]));
        let mut n_b = arrow::array::UInt32Builder::new();
        for &(_, _, _, n) in rows {
            n_b.append_value(n);
        }
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(
                    rows.iter().map(|r| r.0).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|r| r.1).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|r| r.2).collect::<Vec<_>>(),
                )),
                Arc::new(n_b.finish()),
            ],
        )
        .expect("provenance batch");
        let f = File::create(path).expect("create provenance");
        let mut w = ArrowWriter::try_new(f, schema, Some(WriterProperties::builder().build()))
            .expect("provenance writer");
        w.write(&batch).expect("write provenance");
        w.close().expect("close provenance");
    }

    /// Read the summary parquet back as `(transcript_id, tier, weight_sum,
    /// effective_n, cpm)`, in FILE order -- file order is the thing under test,
    /// so it must not be re-sorted on the way in.
    fn read_summary(path: &std::path::Path) -> Vec<(String, String, f64, f64, f64)> {
        let f = File::open(path).expect("open summary");
        let r = ParquetRecordBatchReaderBuilder::try_new(f)
            .expect("summary builder")
            .build()
            .expect("summary reader");
        let mut out = Vec::new();
        for b in r {
            let b = b.expect("summary batch");
            let tx = b
                .column_by_name("transcript_id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let tier = b
                .column_by_name("tier")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let ws = b
                .column_by_name("weight_sum")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let ne = b
                .column_by_name("effective_n")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let tp = b
                .column_by_name("cpm")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            for i in 0..b.num_rows() {
                out.push((
                    tx.value(i).to_string(),
                    tier.value(i).to_string(),
                    ws.value(i),
                    ne.value(i),
                    tp.value(i),
                ));
            }
        }
        out
    }

    /// THE CLOSURE. The profile covers 2 of 4 universe transcripts; the output
    /// must carry all 4, with the 2 unobserved ones zeroed and still labelled.
    ///
    /// This is the test the expert's A-0 ruling turns on. Without the
    /// provenance argument the engine is an INNER JOIN and the two zero-weight
    /// transcripts simply do not appear -- which is a plausible-looking file,
    /// not an error, and is exactly how 1,334 HC2 transcripts went missing.
    #[test]
    fn test_provenance_turns_an_inner_join_into_a_full_universe_closure() {
        let p = scratch("closure_profile");
        let v = scratch("closure_prov");
        let o = scratch("closure_out");
        write_profile(
            &p,
            "S1",
            &[("tx_a", "g1", 100, 1.0), ("tx_b", "g1", 120, 3.0)],
        );
        write_provenance(
            &v,
            &[
                ("tx_a", "g1", "L1", 50),
                ("tx_b", "g1", "L2", 4),
                ("tx_c", "g2", "L3_imputed", 0),
                ("tx_d", "g2", "L3_imputed", 0),
            ],
        );

        summarize_profile_parquet(&p, &o, Some(&v)).expect("summarize with provenance");
        let rows = read_summary(&o);
        assert_eq!(
            rows.len(),
            4,
            "every universe transcript must appear, not only the ones a read hit: {:?}",
            rows
        );
        let by_tx: std::collections::HashMap<&str, &(String, String, f64, f64, f64)> =
            rows.iter().map(|r| (r.0.as_str(), r)).collect();
        for tx in ["tx_c", "tx_d"] {
            let r = by_tx.get(tx).unwrap_or_else(|| panic!("{} missing", tx));
            assert_eq!(r.1, "L3_imputed", "{} must keep its tier", tx);
            assert_eq!(r.2, 0.0, "{} weight_sum must be 0.0, not absent", tx);
            assert_eq!(r.3, 0.0, "{} effective_n must be 0.0, not NaN", tx);
            assert_eq!(r.4, 0.0, "{} cpm must be 0.0", tx);
        }
        // The two observed transcripts keep their real aggregates.
        assert_eq!(by_tx["tx_a"].2, 1.0);
        assert_eq!(by_tx["tx_b"].2, 3.0);
        assert_eq!(by_tx["tx_b"].1, "L2");

        for f in [p, v, o] {
            let _ = std::fs::remove_file(f);
        }
    }

    /// WITHOUT the sidecar the key set is the profile's, and the zero-weight
    /// transcripts are gone. Pinned so the closure test above cannot be passed
    /// by a change that always emits the profile's rows.
    #[test]
    fn test_without_provenance_the_key_set_is_still_the_profiles() {
        let p = scratch("noprofile_prov");
        let o = scratch("noprofile_out");
        write_profile(
            &p,
            "S1",
            &[("tx_a", "g1", 100, 1.0), ("tx_b", "g1", 120, 3.0)],
        );
        summarize_profile_parquet(&p, &o, None).expect("summarize without provenance");
        let rows = read_summary(&o);
        assert_eq!(rows.len(), 2, "no sidecar means no universe: {:?}", rows);

        for f in [p, o] {
            let _ = std::fs::remove_file(f);
        }
    }

    /// CPM is a within-sample share, so the shares must sum to 1e6. Without
    /// this the denominator could be anything and still look plausible.
    #[test]
    fn test_cpm_sums_to_one_million_across_the_weighted_transcripts() {
        let p = scratch("cpm_profile");
        let v = scratch("cpm_prov");
        let o = scratch("cpm_out");
        write_profile(
            &p,
            "S1",
            &[
                ("tx_a", "g1", 100, 1.0),
                ("tx_b", "g1", 120, 3.0),
                ("tx_c", "g2", 90, 0.0),
            ],
        );
        write_provenance(
            &v,
            &[
                ("tx_a", "g1", "L1", 50),
                ("tx_b", "g1", "L1", 50),
                ("tx_c", "g2", "L1", 50),
            ],
        );
        summarize_profile_parquet(&p, &o, Some(&v)).expect("summarize");
        let rows = read_summary(&o);
        let total: f64 = rows.iter().map(|r| r.4).sum();
        assert!(
            (total - 1.0e6).abs() < 1e-6,
            "cpm must be a partition of 1e6; got {}",
            total
        );

        for f in [p, v, o] {
            let _ = std::fs::remove_file(f);
        }
    }

    /// DETERMINISM. Same rows, opposite insertion order, byte-identical
    /// output. Rust's default hasher is seeded per process and `HashMap`
    /// iteration order is not fixed, so an unsorted `for (k, v) in map` passes
    /// every assertion above while emitting a different file order on every
    /// run -- and a byte comparison between two runs then means nothing.
    #[test]
    fn test_output_order_does_not_depend_on_profile_row_order() {
        let v = scratch("order_prov");
        let mut rows: Vec<(&str, &str, &str, u32)> = vec![
            ("tx_a", "g1", "L1", 10),
            ("tx_b", "g1", "L1", 10),
            ("tx_c", "g2", "L1", 10),
            ("tx_d", "g2", "L1", 10),
            ("tx_e", "g2", "L1", 10),
        ];
        write_provenance(&v, &rows);
        rows.reverse();
        let prov_rev = scratch("order_prov_rev");
        write_provenance(&prov_rev, &rows);

        let fwd = [
            ("tx_e", "g2", 90, 1.0),
            ("tx_c", "g2", 90, 2.0),
            ("tx_a", "g1", 100, 3.0),
            ("tx_d", "g2", 90, 4.0),
            ("tx_b", "g1", 120, 5.0),
        ];
        let rev: [(&str, &str, i32, f64); 5] = [
            ("tx_b", "g1", 120, 5.0),
            ("tx_d", "g2", 90, 4.0),
            ("tx_a", "g1", 100, 3.0),
            ("tx_c", "g2", 90, 2.0),
            ("tx_e", "g2", 90, 1.0),
        ];

        let p1 = scratch("order_profile_fwd");
        let p2 = scratch("order_profile_rev");
        let o1 = scratch("order_out_fwd");
        let o2 = scratch("order_out_rev");
        write_profile(&p1, "S1", &fwd);
        write_profile(&p2, "S1", &rev);
        summarize_profile_parquet(&p1, &o1, Some(&v)).expect("summarize fwd");
        summarize_profile_parquet(&p2, &o2, Some(&prov_rev)).expect("summarize rev");

        let r1 = read_summary(&o1);
        let r2 = read_summary(&o2);
        assert_eq!(
            r1, r2,
            "row order must be a function of the transcript ids, not of the \
             order the profile happened to store them in"
        );
        let ids: Vec<&str> = r1.iter().map(|r| r.0.as_str()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(
            ids, sorted,
            "output must be sorted by transcript_id: {:?}",
            ids
        );

        for f in [p1, p2, o1, o2, v, prov_rev] {
            let _ = std::fs::remove_file(f);
        }
    }

    /// A zero-weight transcript has no sample to belong to, so a multi-sample
    /// profile is refused rather than attributed to an arbitrary one.
    #[test]
    fn test_provenance_rejects_a_multi_sample_profile() {
        let p = scratch("multi_profile");
        let v = scratch("multi_prov");
        let o = scratch("multi_out");
        write_profile(&p, "S1", &[("tx_a", "g1", 100, 1.0)]);
        write_provenance(
            &v,
            &[("tx_a", "g1", "L1", 10), ("tx_z", "g1", "L3_imputed", 0)],
        );
        // A second sample, written by hand since write_profile is single-sample.
        let p2 = scratch("multi_profile2");
        write_profile(&p2, "S2", &[("tx_a", "g1", 100, 1.0)]);

        // Single sample first: this must SUCCEED, so the rejection below is
        // about the multi-sample case and not a fixture that never works.
        summarize_profile_parquet(&p, &o, Some(&v)).expect("single sample must work");

        let merged = scratch("multi_profile_merged");
        merge_two_samples(&p, &p2, &merged);
        let err = summarize_profile_parquet(&merged, &o, Some(&v))
            .expect_err("two samples must be rejected, not silently attributed");
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("single-sample"),
            "the error must say WHY; got: {}",
            msg
        );

        for f in [p, p2, merged, v, o] {
            let _ = std::fs::remove_file(f);
        }
    }

    /// Concatenate two single-sample profiles into one two-sample profile.
    fn merge_two_samples(a: &std::path::Path, b: &std::path::Path, out: &std::path::Path) {
        let mut rows: Vec<(String, String, String, i32, f64)> = Vec::new();
        for src in [a, b] {
            let f = File::open(src).expect("open");
            let r = ParquetRecordBatchReaderBuilder::try_new(f)
                .expect("builder")
                .build()
                .expect("reader");
            for batch in r {
                let batch = batch.expect("batch");
                let s = batch
                    .column_by_name("sample_id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                let g = batch
                    .column_by_name("gene_id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                let t = batch
                    .column_by_name("transcript_id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                let pal = batch
                    .column_by_name("polyA_length")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                let w = batch
                    .column_by_name("posterior_weight")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                for i in 0..batch.num_rows() {
                    rows.push((
                        s.value(i).to_string(),
                        g.value(i).to_string(),
                        t.value(i).to_string(),
                        pal.value(i),
                        w.value(i),
                    ));
                }
            }
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("sample_id", DataType::Utf8, false),
            Field::new("gene_id", DataType::Utf8, false),
            Field::new("transcript_id", DataType::Utf8, false),
            Field::new("polyA_length", DataType::Int32, false),
            Field::new("posterior_weight", DataType::Float64, false),
        ]));
        let mut pal_b = arrow::array::Int32Builder::new();
        let mut w_b = Float64Builder::new();
        for r in &rows {
            pal_b.append_value(r.3);
            w_b.append_value(r.4);
        }
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(
                    rows.iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|r| r.1.clone()).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|r| r.2.clone()).collect::<Vec<_>>(),
                )),
                Arc::new(pal_b.finish()),
                Arc::new(w_b.finish()),
            ],
        )
        .expect("merged batch");
        let f = File::create(out).expect("create merged");
        let mut w = ArrowWriter::try_new(f, schema, Some(WriterProperties::builder().build()))
            .expect("merged writer");
        w.write(&batch).expect("write merged");
        w.close().expect("close merged");
    }

    /// A profile transcript the universe does not name means the two artifacts
    /// disagree about which transcripts exist. Refused, not given a guessed tier.
    #[test]
    fn test_provenance_rejects_a_transcript_outside_the_universe() {
        let p = scratch("orphan_profile");
        let v = scratch("orphan_prov");
        let o = scratch("orphan_out");
        write_profile(
            &p,
            "S1",
            &[("tx_a", "g1", 100, 1.0), ("tx_ghost", "g1", 100, 1.0)],
        );
        write_provenance(&v, &[("tx_a", "g1", "L1", 10)]);
        let err = summarize_profile_parquet(&p, &o, Some(&v))
            .expect_err("a transcript outside the universe must be refused");
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("tx_ghost"),
            "the error must NAME the offending transcript; got: {}",
            msg
        );

        for f in [p, v, o] {
            let _ = std::fs::remove_file(f);
        }
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
