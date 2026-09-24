//! Module 3: TEL-EM Solver
//!
//! Executes the bifurcated Terminal-aware Expectation Maximization (TEL-EM)
//! algorithm on the flat evidence graph produced by Module 2.
//!
//! Bifurcated inference (spec v0.1.7 F2.1):
//! - Unambiguous reads (candidate_count == 1) get w_rk ≡ 1.0 (deterministic).
//! - Ambiguous reads (candidate_count > 1) enter the EM loop.
//!
//! Real-data profile (Tongji 2026-09-25): 99.7% unambiguous, ~0.3% ambiguous
//! (~7,000 reads). EM therefore runs on a tiny subset.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{BooleanArray, Float64Array, Float64Builder, Int32Builder, StringArray, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

use crate::emission_model::MIN_COHORT_VARIANCE;
use crate::evidence::FlatEvidenceGraph;

/// Per-transcript parameters loaded from `telescope.emission.parquet`.
#[derive(Debug, Clone)]
pub struct TranscriptParam {
    pub transcript_id: String,
    pub gene_id: String,
    pub is_competent: bool,
    pub mu: f64,
    pub sigma2_cohort: f64,
    /// Current EM iteration σ² (initialized to cohort, updated each M-step).
    pub sigma2: f64,
    /// Mixing proportion π_k (initialized at iter 0, updated each M-step).
    pub alpha: f64,
}

/// Load `telescope.emission.parquet` produced by Module 1.
pub fn load_emission_parameters<P: AsRef<Path>>(
    emission_path: P,
) -> Result<HashMap<String, TranscriptParam>> {
    let file = File::open(&emission_path)
        .with_context(|| format!("Failed to open emission parquet: {:?}", emission_path.as_ref()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to build ParquetRecordBatchReaderBuilder")?;
    let mut reader = builder.with_batch_size(8192).build()?;

    let mut map = HashMap::new();

    while let Some(batch_result) = reader.next() {
        let batch = batch_result.context("Failed to read emission batch")?;

        let tr_col = batch
            .column_by_name("transcript_id")
            .context("Missing 'transcript_id'")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("transcript_id is not StringArray")?;
        let gene_col = batch
            .column_by_name("gene_id")
            .context("Missing 'gene_id'")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("gene_id is not StringArray")?;
        let comp_col = batch
            .column_by_name("is_polyA_competent")
            .context("Missing 'is_polyA_competent'")?
            .as_any()
            .downcast_ref::<BooleanArray>()
            .context("is_polyA_competent is not BooleanArray")?;
        let mu_col = batch
            .column_by_name("mu_logPAL")
            .context("Missing 'mu_logPAL'")?
            .as_any()
            .downcast_ref::<Float64Array>()
            .context("mu_logPAL is not Float64Array")?;
        let sigma2_col = batch
            .column_by_name("sigma2_logPAL")
            .context("Missing 'sigma2_logPAL'")?
            .as_any()
            .downcast_ref::<Float64Array>()
            .context("sigma2_logPAL is not Float64Array")?;

        for i in 0..batch.num_rows() {
            let tr_id = tr_col.value(i).to_string();
            let gene_id = gene_col.value(i).to_string();
            let is_competent = comp_col.value(i);
            let mu = mu_col.value(i);
            let sigma2_cohort = sigma2_col.value(i);

            map.insert(
                tr_id.clone(),
                TranscriptParam {
                    transcript_id: tr_id,
                    gene_id,
                    is_competent,
                    mu,
                    sigma2_cohort,
                    sigma2: sigma2_cohort,
                    alpha: 0.0, // initialized in solve_and_write_profile
                },
            );
        }
    }

    Ok(map)
}

/// Run TEL-EM and write `telescope.profile.parquet`.
///
/// Returns: stats `(n_unambiguous, n_ambiguous_em, n_ambiguous_iter, n_dropped_incompetent)`.
pub fn solve_and_write_profile<P: AsRef<Path>>(
    graph: FlatEvidenceGraph,
    emission_path: P,
    sample_id: &str,
    output_profile_path: P,
    tol: f64,
    max_iter: usize,
) -> Result<TelEmStats> {
    let mut param_map = load_emission_parameters(emission_path)?;
    let num_transcripts = graph.transcript_names.len();

    // Index transcripts by position in `graph.transcript_names`.
    let mut params: Vec<Option<TranscriptParam>> = Vec::with_capacity(num_transcripts);
    for name in &graph.transcript_names {
        params.push(param_map.remove(name));
    }

    // 1. Split molecules by candidate_count.
    let mut unambiguous_indices: Vec<usize> = Vec::new();
    let mut ambiguous_indices: Vec<usize> = Vec::new();
    for (idx, mol) in graph.molecules.iter().enumerate() {
        match mol.candidate_count {
            0 => {}
            1 => unambiguous_indices.push(idx),
            _ => ambiguous_indices.push(idx),
        }
    }
    let n_total_molecules = graph.molecules.len() as f64;
    let n_unambiguous = unambiguous_indices.len();
    let n_ambiguous = ambiguous_indices.len();

    // 2. Initialize π_k from candidate evidence counts (Laplace smoothing ε=0.01).
    let eps = 0.01_f64;
    let mut init_counts = vec![eps; num_transcripts];
    for &idx in &unambiguous_indices {
        let mol = &graph.molecules[idx];
        let tx_id = graph.candidate_pool[mol.candidate_offset as usize] as usize;
        init_counts[tx_id] += 1.0;
    }
    for &idx in &ambiguous_indices {
        let mol = &graph.molecules[idx];
        let share = 1.0 / (mol.candidate_count as f64);
        for c in 0..mol.candidate_count {
            let tx_id = graph.candidate_pool[(mol.candidate_offset + c as u32) as usize] as usize;
            init_counts[tx_id] += share;
        }
    }
    let total_init: f64 = init_counts.iter().sum();
    for i in 0..num_transcripts {
        if let Some(ref mut p) = params[i] {
            p.alpha = init_counts[i] / total_init;
        }
    }

    // 3. TEL-EM loop (only iterates over ambiguous reads).
    let nu_0 = 1.0_f64;
    let beta_0 = 1.0_f64;
    let mut prev_ll = f64::NEG_INFINITY;
    let mut final_weights: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_ambiguous];
    let mut n_em_iter = 0_usize;

    for iter in 0..max_iter {
        // E-step: parallel over ambiguous reads. Compute log-sum-exp normalized weights
        // AND per-read log-likelihood.
        let e_results: Vec<(f64, Vec<(usize, f64)>)> = ambiguous_indices
            .par_iter()
            .map(|&mol_idx| {
                let mol = &graph.molecules[mol_idx];
                let y = mol.log_pal as f64;

                // Compute raw log responsibilities (with alpha + Gaussian).
                let mut raw: Vec<(usize, f64)> = Vec::with_capacity(mol.candidate_count as usize);
                for c in 0..mol.candidate_count {
                    let tx_id =
                        graph.candidate_pool[(mol.candidate_offset + c as u32) as usize] as usize;
                    if let Some(ref p) = params[tx_id] {
                        if !p.is_competent || p.alpha <= 0.0 {
                            // Push -inf so normalization kills this candidate.
                            raw.push((tx_id, f64::NEG_INFINITY));
                        } else {
                            let ll = p.alpha.ln()
                                - (p.sigma2.sqrt()).ln()
                                - ((y - p.mu).powi(2) / (2.0 * p.sigma2));
                            raw.push((tx_id, ll));
                        }
                    }
                }

                // Log-sum-exp over raw.
                let max_ll = raw
                    .iter()
                    .map(|&(_, ll)| ll)
                    .fold(f64::NEG_INFINITY, f64::max);
                if max_ll == f64::NEG_INFINITY {
                    return (0.0, Vec::new());
                }
                let sum_exp: f64 = raw.iter().map(|&(_, ll)| (ll - max_ll).exp()).sum();
                let log_z = max_ll + sum_exp.ln();

                let weights: Vec<(usize, f64)> = raw
                    .into_iter()
                    .filter(|&(_, ll)| ll > f64::NEG_INFINITY)
                    .map(|(tx_id, ll)| (tx_id, (ll - log_z).exp()))
                    .collect();
                (log_z, weights)
            })
            .collect();

        let current_ll: f64 = e_results.iter().map(|&(ll, _)| ll).sum();
        n_em_iter = iter + 1;

        // Convergence check (store weights + break if converged).
        if (current_ll - prev_ll).abs() < tol {
            for (i, (_, w)) in e_results.into_iter().enumerate() {
                final_weights[i] = w;
            }
            break;
        }
        prev_ll = current_ll;

        // Collect weights for M-step (always — even on non-final iterations).
        for (i, (_, w)) in e_results.into_iter().enumerate() {
            final_weights[i] = w;
        }

        // M-step: update π_k and σ²_k.
        let mut accum_w = vec![0.0_f64; num_transcripts];
        let mut accum_sq = vec![0.0_f64; num_transcripts];

        // Unambiguous: w=1.0 always.
        for &idx in &unambiguous_indices {
            let mol = &graph.molecules[idx];
            let tx_id = graph.candidate_pool[mol.candidate_offset as usize] as usize;
            if let Some(ref p) = params[tx_id] {
                accum_w[tx_id] += 1.0;
                accum_sq[tx_id] += (mol.log_pal as f64 - p.mu).powi(2);
            }
        }

        // Ambiguous: use current weights.
        for (i, weights) in final_weights.iter().enumerate() {
            let mol_idx = ambiguous_indices[i];
            let mol = &graph.molecules[mol_idx];
            let y = mol.log_pal as f64;
            for &(tx_id, w) in weights {
                if let Some(ref p) = params[tx_id] {
                    accum_w[tx_id] += w;
                    accum_sq[tx_id] += w * (y - p.mu).powi(2);
                }
            }
        }

        for i in 0..num_transcripts {
            if let Some(ref mut p) = params[i] {
                let w_sum = accum_w[i];
                p.alpha = w_sum / n_total_molecules;
                p.sigma2 = (accum_sq[i] + 2.0 * beta_0 * p.sigma2_cohort)
                    / (w_sum + 2.0 * nu_0 + 3.0);
                p.sigma2 = p.sigma2.max(MIN_COHORT_VARIANCE);
            }
        }
    }

    // 4. Build output parquet.
    let mut sample_b = StringBuilder::new();
    let mut read_b = StringBuilder::new();
    let mut gene_b = StringBuilder::new();
    let mut tx_b = StringBuilder::new();
    let mut pal_b = Int32Builder::new();
    let mut w_b = Float64Builder::new();
    let mut n_dropped_incompetent = 0_u64;

    // Unambiguous rows: 1 per read.
    for &idx in &unambiguous_indices {
        let mol = &graph.molecules[idx];
        let tx_id = graph.candidate_pool[mol.candidate_offset as usize] as usize;
        let pal_i32 = poly_a_length_from_log2(mol.log_pal);
        if let Some(ref p) = params[tx_id] {
            if !p.is_competent {
                n_dropped_incompetent += 1;
                continue;
            }
            sample_b.append_value(sample_id);
            read_b.append_value(&mol.read_id);
            gene_b.append_value(&p.gene_id);
            tx_b.append_value(&p.transcript_id);
            pal_b.append_value(pal_i32);
            w_b.append_value(1.0);
        } else {
            // transcript missing from emission parquet → skip
            n_dropped_incompetent += 1;
        }
    }

    // Ambiguous rows: per (read × candidate) with posterior ≥ 0.01.
    let min_weight = 0.01_f64;
    for (i, weights) in final_weights.into_iter().enumerate() {
        let mol_idx = ambiguous_indices[i];
        let mol = &graph.molecules[mol_idx];
        let pal_i32 = poly_a_length_from_log2(mol.log_pal);
        for (tx_id, w) in weights {
            if w < min_weight {
                continue;
            }
            if let Some(ref p) = params[tx_id] {
                if !p.is_competent {
                    n_dropped_incompetent += 1;
                    continue;
                }
                sample_b.append_value(sample_id);
                read_b.append_value(&mol.read_id);
                gene_b.append_value(&p.gene_id);
                tx_b.append_value(&p.transcript_id);
                pal_b.append_value(pal_i32);
                w_b.append_value(w);
            } else {
                n_dropped_incompetent += 1;
            }
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("sample_id", DataType::Utf8, false),
        Field::new("read_id", DataType::Utf8, false),
        Field::new("gene_id", DataType::Utf8, false),
        Field::new("transcript_id", DataType::Utf8, false),
        Field::new("polyA_length", DataType::Int32, false),
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
    .context("Failed to build output RecordBatch")?;

    let out_file = File::create(output_profile_path.as_ref())
        .with_context(|| format!("Failed to create output {:?}", output_profile_path.as_ref()))?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(out_file, schema, Some(props))
        .context("Failed to create ArrowWriter")?;
    writer.write(&batch).context("Failed to write batch")?;
    writer.close().context("Failed to close writer")?;

    Ok(TelEmStats {
        n_unambiguous,
        n_ambiguous,
        n_em_iter,
        n_dropped_incompetent,
    })
}

/// Round-trip polyA length through log2.
fn poly_a_length_from_log2(log_pal: f32) -> i32 {
    log_pal.exp2().round() as i32
}

/// TEL-EM execution stats for logging / telemetry.
#[derive(Debug, Clone, Copy)]
pub struct TelEmStats {
    pub n_unambiguous: usize,
    pub n_ambiguous: usize,
    pub n_em_iter: usize,
    pub n_dropped_incompetent: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poly_a_length_round_trip() {
        // log2(120) ≈ 6.9069, exp2 = 120
        assert_eq!(poly_a_length_from_log2(6.9069), 120);
        assert_eq!(poly_a_length_from_log2(7.0), 128);
    }
}