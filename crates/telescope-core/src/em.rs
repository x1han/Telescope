//! Module 3: TEL-EM Solver
//!
//! Executes the bifurcated Terminal-aware Expectation Maximization (TEL-EM)
//! algorithm on the flat evidence graph produced by Module 2.
//!
//! Bifurcated inference (spec v0.1.7 F2.1):
//! - Unambiguous reads (candidate_count == 1) get w_rk ≡ 1.0 (deterministic).
//! - Ambiguous reads (candidate_count > 1) enter the EM loop, which carries the
//!   joint `w_rk ∝ π_k · A_rk · N(y_r; μ_k, σ_k²)` term of spec v0.2.7 F-EM-4;
//!   `ln A_rk = (AS_rk − max_j AS_rj) / λ` comes from Module 2's `AS:i` pool.
//!
//! Real-data profile (Tongji 2026-09-25): 99.7% unambiguous, ~0.3% ambiguous
//! (~7,000 reads). EM therefore runs on a tiny subset.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{
    BooleanArray, Float32Builder, Float64Array, Float64Builder, Int32Builder, StringArray,
    StringBuilder, UInt16Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

use crate::emission_model::MIN_COHORT_VARIANCE;
use crate::evidence::{FlatEvidenceGraph, MoleculeEvidence};

/// Per-transcript parameters loaded from `telescope.emission.parquet`.
#[derive(Debug, Clone)]
pub struct TranscriptParam {
    pub transcript_id: String,
    pub gene_id: String,
    pub is_competent: bool,
    /// Cohort-level μ used as M-step MAP prior (fixed across iterations).
    pub mu_cohort: f64,
    /// Current μ estimate (initialized to mu_cohort, updated each M-step).
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
    let file = File::open(&emission_path).with_context(|| {
        format!(
            "Failed to open emission parquet: {:?}",
            emission_path.as_ref()
        )
    })?;
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
            let mu_cohort = mu_col.value(i);
            let mu = mu_cohort; // Initial value = cohort; M-step will update
            let sigma2_cohort = sigma2_col.value(i);

            map.insert(
                tr_id.clone(),
                TranscriptParam {
                    transcript_id: tr_id,
                    gene_id,
                    is_competent,
                    mu_cohort,
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
/// Returns: stats `(n_unambiguous, n_ambiguous_em, n_ambiguous_iter, n_dropped_incompetent)`
/// plus Phase A telemetry (entropy 3-tier, ghost-drop counter, EM log-L trace).
///
/// `aln_weight_lambda` is the scale λ of the spec v0.2.7 F-EM-4 alignment term
/// `ln A_rk = (AS_rk − max_j AS_rj) / λ`. It is only reachable for reads with
/// `candidate_count > 1`; single-candidate reads take the deterministic
/// `w = 1.0` branch. Must be strictly positive (λ = 0 → division by zero → NaN).
///
/// `output_em_trace`: if `Some`, receives one tuple per EM iteration
/// `(iter_index, log_likelihood, mean_entropy)` (mean entropy = 0 if no
/// ambiguous reads). Pass `None` to skip trace collection.
// `aln_weight_lambda` pushed the signature to 8 arguments; collapsing the EM
// hyperparameters into a config struct is a separate refactor, out of scope.
#[allow(clippy::too_many_arguments)]
pub fn solve_and_write_profile<P: AsRef<Path>>(
    graph: FlatEvidenceGraph,
    emission_path: P,
    sample_id: &str,
    output_profile_path: P,
    tol: f64,
    max_iter: usize,
    aln_weight_lambda: f64,
    mut output_em_trace: Option<&mut Vec<(usize, f64, f64)>>,
) -> Result<TelEmStats> {
    // NaN-safe equivalent of the CLI's `!(aln_weight_lambda > 0.0)`: `<= 0.0`
    // alone would let NaN through (every comparison with NaN is false) and NaN λ
    // emits NaN posterior weights instead of failing. +inf stays accepted, as
    // on the CLI. Spelled this way to stay clear of `neg_cmp_op_on_partial_ord`.
    if aln_weight_lambda <= 0.0 || aln_weight_lambda.is_nan() {
        anyhow::bail!(
            "aln_weight_lambda must be > 0 (got {}); it scales ln A_rk = (AS_rk − \
             max_j AS_rj)/λ, so 0 and NaN would emit NaN posterior weights",
            aln_weight_lambda
        );
    }

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
    let tau_0 = 2.0_f64; // MAP pseudo-count damping for μ update (spec v0.2.6 §F2).
    let mut prev_ll = f64::NEG_INFINITY;
    let mut final_weights: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_ambiguous];
    let mut n_em_iter = 0_usize;

    for iter in 0..max_iter {
        // E-step: parallel over ambiguous reads. Compute log-sum-exp normalized weights,
        // per-read log-likelihood, and per-read posterior entropy.
        let e_results: Vec<(f64, Vec<(usize, f64)>, f64)> = ambiguous_indices
            .par_iter()
            .map(|&mol_idx| {
                let mol = &graph.molecules[mol_idx];
                let y = mol.log_pal as f64;

                // Alignment-score reference point for this read. Taken over the
                // full `0..candidate_count` range — i.e. before competence
                // masking, and over candidates that the `if let Some(ref p) =
                // params[tx_id]` arm below may skip entirely (no emission row →
                // never pushed into `raw`).
                //
                // NOTE: this is therefore NOT the same range as `max_ll` below,
                // which is taken over `raw`. A skipped candidate moves `max_as`
                // but not `max_ll`. The mismatch is a single constant offset
                // shared by every surviving candidate, which cancels inside the
                // log-sum-exp, so the emitted posteriors are unaffected — but
                // the ranges differ and must not be described as identical.
                let max_as = (0..mol.candidate_count)
                    .map(|c| graph.candidate_as[(mol.candidate_offset + c as u32) as usize])
                    .fold(i32::MIN, i32::max);

                // Compute raw log responsibilities (with alpha + Gaussian + A_rk).
                let mut raw: Vec<(usize, f64)> = Vec::with_capacity(mol.candidate_count as usize);
                for c in 0..mol.candidate_count {
                    let idx = (mol.candidate_offset + c as u32) as usize;
                    let tx_id = graph.candidate_pool[idx] as usize;
                    // ln A_rk = (AS_rk − max_j AS_rj) / λ (spec v0.2.7 F-EM-4).
                    // Added to `ll` *before* the `max_ll` log-sum-exp shift, so
                    // any λ is numerically safe — no clamping needed.
                    let ln_a = (graph.candidate_as[idx] - max_as) as f64 / aln_weight_lambda;
                    if let Some(ref p) = params[tx_id] {
                        if !p.is_competent || p.alpha <= 0.0 {
                            // Push -inf so normalization kills this candidate.
                            raw.push((tx_id, f64::NEG_INFINITY));
                        } else {
                            let ll = p.alpha.ln()
                                - (p.sigma2.sqrt()).ln()
                                - ((y - p.mu).powi(2) / (2.0 * p.sigma2))
                                + ln_a;
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
                    // No competent candidate — read becomes a ghost-drop.
                    return (0.0, Vec::new(), 0.0);
                }
                let sum_exp: f64 = raw.iter().map(|&(_, ll)| (ll - max_ll).exp()).sum();
                let log_z = max_ll + sum_exp.ln();

                let weights: Vec<(usize, f64)> = raw
                    .into_iter()
                    .filter(|&(_, ll)| ll > f64::NEG_INFINITY)
                    .map(|(tx_id, ll)| (tx_id, (ll - log_z).exp()))
                    .collect();
                let entropy = posterior_entropy(&weights);
                (log_z, weights, entropy)
            })
            .collect();

        let current_ll: f64 = e_results.iter().map(|&(ll, _, _)| ll).sum();
        n_em_iter = iter + 1;

        // Record EM trace (per-iter log-L + mean posterior entropy).
        if let Some(ref mut trace) = output_em_trace {
            // P1 fix: exclude ghost drops (h == 0) from the denominator so the
            // reported mean_entropy reflects reads that actually participated
            // in E-step weight computation, not reads whose posteriors were
            // empty due to no competent candidates.
            let n_with_entropy = e_results.iter().filter(|&&(_, _, h)| h > 0.0).count();
            let mean_entropy = if n_with_entropy > 0 {
                e_results.iter().map(|&(_, _, h)| h).sum::<f64>() / n_with_entropy as f64
            } else {
                0.0
            };
            trace.push((iter + 1, current_ll, mean_entropy));
        }

        // Convergence check: per-molecule average log-likelihood delta.
        //
        // The absolute delta |log L_new - log L_old| has summation noise of
        // order sqrt(N_ambig) * eps * |log L| from parallel (Rayon) summation
        // reordering in IEEE 754. For N_ambig ≈ 12 000 and |log L| ≈ 150 000,
        // this noise floor is ~3 × 10⁻⁹ — strictly below the spec threshold
        // 10⁻⁷ but the *cumulative* effect across 1000 iterations produces
        // jitter at the 10⁻⁷ to 10⁻⁶ scale, which masquerades as oscillation.
        // Normalizing by N_ambig gives a per-molecule metric that is
        // numerically well-behaved: at convergence the per-molecule
        // improvement is well below 10⁻⁶, which on the absolute scale is
        // `N_ambig * 1e-6` (≈ 0.012 for 12 K reads).
        let n_ambig = e_results.len() as f64;
        let delta_avg_ll = if n_ambig > 0.0 {
            (current_ll - prev_ll).abs() / n_ambig
        } else {
            0.0
        };
        if delta_avg_ll < tol {
            for (i, (_, w, _)) in e_results.into_iter().enumerate() {
                final_weights[i] = w;
            }
            break;
        }
        prev_ll = current_ll;

        // Collect weights for M-step (always — even on non-final iterations).
        for (i, (_, w, _)) in e_results.into_iter().enumerate() {
            final_weights[i] = w;
        }

        // M-step: single-pass algebraic closed-form (spec v0.2.6 §F2 F-EM-4 Locked).
        //
        // Pre-M-step accumulators collect only (W, S1=Σ w·y, S2=Σ w·y²) per
        // transcript. The σ² deviation Σ w·(y − μ_new)² is reconstructed
        // algebraically AFTER μ is updated — avoiding the v0.2.4 bug of
        // squaring deviations around an old μ that contradicts the new one.
        let mut accum_w = vec![0.0_f64; num_transcripts];
        let mut accum_w_y = vec![0.0_f64; num_transcripts]; // S1
        let mut accum_w_y2 = vec![0.0_f64; num_transcripts]; // S2

        // Unambiguous: w=1.0 always.
        for &idx in &unambiguous_indices {
            let mol = &graph.molecules[idx];
            let tx_id = graph.candidate_pool[mol.candidate_offset as usize] as usize;
            let y = mol.log_pal as f64;
            if params[tx_id].is_some() {
                accum_w[tx_id] += 1.0;
                accum_w_y[tx_id] += y;
                accum_w_y2[tx_id] += y * y;
            }
        }

        // Ambiguous: use current weights.
        for (i, weights) in final_weights.iter().enumerate() {
            let mol_idx = ambiguous_indices[i];
            let mol = &graph.molecules[mol_idx];
            let y = mol.log_pal as f64;
            for &(tx_id, w) in weights {
                if params[tx_id].is_some() {
                    accum_w[tx_id] += w;
                    accum_w_y[tx_id] += w * y;
                    accum_w_y2[tx_id] += w * y * y;
                }
            }
        }

        // 3-step update: μ → σ² (around μ_new) → π. Order is locked (spec §F2).
        for i in 0..num_transcripts {
            if let Some(ref mut p) = params[i] {
                let w_sum = accum_w[i];

                // STEP 1: μ update — weak empirical Bayes shrinkage (τ₀=2).
                p.mu = (accum_w_y[i] + tau_0 * p.mu_cohort) / (w_sum + tau_0);

                // STEP 2: σ² update — algebraic closed-form around μ_new.
                //   Σ w·(y − μ_new)² = S2 − 2·μ_new·S1 + μ_new²·W   (identity).
                let s1 = accum_w_y[i];
                let s2 = accum_w_y2[i];
                let sq_dev_new = (s2 - 2.0 * p.mu * s1 + p.mu * p.mu * w_sum).max(0.0);
                p.sigma2 =
                    (sq_dev_new + 2.0 * beta_0 * p.sigma2_cohort) / (w_sum + 2.0 * nu_0 + 3.0);
                p.sigma2 = p.sigma2.max(MIN_COHORT_VARIANCE);

                // STEP 3: π update.
                // P1 fix: defensive check for n_total_molecules == 0 (BAM
                // with zero mapped reads). Without this, every p.alpha would
                // become 0/0 = NaN. In practice the E-step would not iterate
                // for an empty graph (delta_avg_ll = 0 < tol on iter 0),
                // but a fully-empty BAM hitting quant() would emit NaN rows.
                p.alpha = if n_total_molecules > 0.0 {
                    w_sum / n_total_molecules
                } else {
                    0.0
                };
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
    let mut ncand_b = UInt16Builder::new();
    let mut as_ratio_b = Float32Builder::new();
    let mut n_dropped_incompetent = 0_u64;
    let mut n_dropped_no_competent_candidates = 0_u64;
    let mut n_entropy_clean = 0_u64;
    let mut n_entropy_partial = 0_u64;
    let mut n_entropy_degenerate = 0_u64;

    // Unambiguous rows: 1 per read.
    for &idx in &unambiguous_indices {
        let mol = &graph.molecules[idx];
        let tx_id = graph.candidate_pool[mol.candidate_offset as usize] as usize;
        let pal_i32 = poly_a_length_from_log2(mol.log_pal);
        // Observational columns. Both are per READ properties, so they are
        // derived once here and repeated verbatim on every row of this read
        // (below) — they do NOT vary with the candidate a row describes.
        // `candidate_count` is read off the graph as-is, i.e. BEFORE the
        // `min_weight` truncation in the ambiguous branch below can drop rows.
        let ncand = mol.candidate_count;
        let as_ratio = read_max_as_ratio(&graph, mol);
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
            ncand_b.append_value(ncand);
            as_ratio_b.append_value(as_ratio);
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
        // Observational columns, per READ — computed once, before the
        // `min_weight` truncation below can drop rows, and repeated verbatim
        // on every emitted row of this read. `candidate_count` is the ORIGINAL
        // count off the graph (what the E-step saw), NOT the number of rows
        // this loop emits.
        let ncand = mol.candidate_count;
        let as_ratio = read_max_as_ratio(&graph, mol);

        // Phase A entropy 3-tier + ghost-drop telemetry.
        if weights.is_empty() {
            n_dropped_no_competent_candidates += 1;
        } else {
            let h = posterior_entropy(&weights);
            // P1 fix: upper bound raised to 0.70 (was 0.60) so that a
            // 2-way ambiguous read with balanced 50/50 weights (H = ln(2)
            // ≈ 0.693) is correctly classified as 'partial' rather than
            // 'degenerate', honoring the docstring semantics.
            if h < 0.20 {
                n_entropy_clean += 1;
            } else if h <= 0.70 {
                n_entropy_partial += 1;
            } else {
                n_entropy_degenerate += 1;
            }
        }

        // Renormalize the retained weights so each read's emitted posterior
        // weights sum to 1.0. The denominator is taken over the rows actually
        // emitted, i.e. `{w >= min_weight}` — the truncation above discards
        // real probability mass, and without this step the emitted weights sum
        // to less than 1. This is a *pure output-stage* correction: the M-step
        // consumed the untruncated `final_weights` before this point, so the EM
        // fit itself is unaffected.
        //
        // COUPLING: `{w >= min_weight}` is the right denominator only because
        // the E-step's `if let Some` (above) *skips* candidates with missing
        // params instead of pushing `-inf`. If that ever changes to push `-inf`
        // for missing params, the two later filters here (`!p.is_competent`,
        // missing `params[tx_id]`) become reachable and this denominator
        // silently breaks. The same `if let Some` also widens `max_as` relative
        // to `max_ll` in the E-step — harmless there (constant shift inside a
        // softmax), but it means "the weights" and "the raw log
        // responsibilities" are not the same candidate set.
        let retained_sum: f64 = weights
            .iter()
            .filter(|&(_, w)| *w >= min_weight)
            .map(|&(_, w)| w)
            .sum();

        for (tx_id, w) in weights {
            if w < min_weight {
                continue;
            }
            // Safe: this body only runs when at least one weight is retained,
            // hence `retained_sum >= min_weight > 0`.
            let w_norm = w / retained_sum;
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
                w_b.append_value(w_norm);
                ncand_b.append_value(ncand);
                as_ratio_b.append_value(as_ratio);
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
        // The two columns below are OBSERVATIONAL ONLY — nothing in TEL-EM
        // reads them back, and they do not participate in A_rk, the `min_weight`
        // cutoff, or the renormalization. Both are per READ properties and are
        // repeated identically on every row of that read; a reader must not
        // treat them as per-candidate values.
        //
        // Rationale for recording them before deciding anything: an expert
        // panel measured `max_as_ratio` separating full-length reads (median
        // 0.906) from truncated reads (median 0.998) on PGAM1. The panel
        // REJECTED a hard gate on that ratio, but asked for the cross-gene
        // distribution to be emitted so the decision can be made on real data
        // rather than on one gene. These columns are that record.
        //
        // CAVEAT — THE EMITTED DISTRIBUTION IS SURVIVAL-BIASED. Both columns
        // ride on rows that survive the `min_weight` cutoff, so a read whose
        // top candidate clears the cutoff while every competitor falls below it
        // emits exactly one row, and a read all of whose candidates fall below
        // emits none at all. Reads with decisive evidence are therefore
        // over-represented relative to genuinely contested reads — which is
        // the opposite of the population a score-ratio gate would be judged on.
        // Any analysis of these columns MUST group by `read_id` and report the
        // read-level coverage rate alongside the distribution; do not read the
        // row-level histogram as an unbiased per-read statistic.
        //
        // The molecule's ORIGINAL candidate count (`MoleculeEvidence::candidate_count`
        // as it stands on the graph), captured before the `min_weight`
        // truncation drops rows. Differs from the number of emitted rows exactly
        // when the truncation fired.
        Field::new("n_candidates_before_cutoff", DataType::UInt16, false),
        // Second-largest / largest `AS:i` over the read's full candidate range.
        // 0.0 when the read has no second candidate or when no candidate carries
        // an `AS:i` tag. See `read_max_as_ratio` for the exact edge cases.
        Field::new("max_as_ratio", DataType::Float32, false),
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
            Arc::new(ncand_b.finish()),
            Arc::new(as_ratio_b.finish()),
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
        n_dropped_no_competent_candidates,
        n_entropy_clean,
        n_entropy_partial,
        n_entropy_degenerate,
        em_trace: output_em_trace.map(|t| t.clone()).unwrap_or_default(),
    })
}

/// Round-trip polyA length through log2.
fn poly_a_length_from_log2(log_pal: f32) -> i32 {
    log_pal.exp2().round() as i32
}

/// Per-read observational statistic: `second_largest_AS / largest_AS` over the
/// read's FULL candidate range `candidate_as[offset .. offset+candidate_count]`,
/// emitted on every row of that read.
///
/// Three defined cases:
/// - `candidate_count == 1` → `0.0`. There is no second candidate, so the ratio
///   carries no information about competition between candidates.
/// - `largest_AS <= 0` → `0.0`. Every candidate lacks an `AS:i` tag (Module 2
///   falls back to 0), so the ratio would be `0/0` — undefined. `0.0` here means
///   "this read has no alignment evidence", not "its top two candidates tie".
/// - `second_AS < 0` → `0.0`. `AS` is a raw `i32` tag; a malformed or
///   non-minimap2 BAM can carry a negative value, which would otherwise yield
///   a negative "proportion" (`[100, -5] -> -0.05`, `[i32::MIN, 5] -> -4.3e8`).
///   A negative alignment score is not evidence, so it is treated the same way
///   as a missing one rather than being silently emitted.
/// - otherwise → `second_largest / largest`, in `[0, 1]`. `AS` is an `i32` and
///   the ratio is stored as `Float32`, so the division is done in `f64` and cast
///   down; an `f32` division of two `i32`s would lose precision before the
///   narrowing even needs to happen.
///
/// NOTE ON THE `0.0` SENTINEL: it is deliberately overloaded across three
/// distinct situations — an unambiguous read, a read with no usable alignment
/// score, and a read whose runner-up is unusable. That keeps the column
/// non-nullable and its type simple, but it means a downstream query of the
/// form "fraction of reads with `max_as_ratio < 0.5`" silently pools all three.
/// Such a query is only meaningful after separating them, e.g. by also
/// conditioning on `n_candidates_before_cutoff >= 2`.
///
/// NOT USED BY THE SOLVER. This is recorded, not acted upon: an expert panel
/// measured it separating full-length reads (median 0.906) from truncated reads
/// (median 0.998) on PGAM1, rejected a hard gate on it, and asked for the
/// cross-gene distribution to be emitted before deciding whether it becomes one.
/// No threshold reads it, and it is deliberately NOT fed into `A_rk`.
fn read_max_as_ratio(graph: &FlatEvidenceGraph, mol: &MoleculeEvidence) -> f32 {
    if mol.candidate_count < 2 {
        return 0.0;
    }
    let o = mol.candidate_offset as usize;
    let n = mol.candidate_count as usize;
    let slice = &graph.candidate_as[o..o + n];

    // Two largest in one pass; `second` is the runner-up, so a tie at the top
    // still yields a defined ratio (equal AS → 1.0) rather than a dropped entry.
    let mut largest = i32::MIN;
    let mut second = i32::MIN;
    for &as_score in slice {
        if as_score >= largest {
            second = largest;
            largest = as_score;
        } else if as_score > second {
            second = as_score;
        }
    }

    if largest <= 0 || second < 0 {
        return 0.0;
    }
    (second as f64 / largest as f64) as f32
}

/// Posterior entropy (in nats) of a single ambiguous read over its
/// non-zero weighted candidates: `H = -Σ_k w_k * ln(w_k)`.
/// Returns 0.0 for an empty / degenerate weight vector.
fn posterior_entropy(weights: &[(usize, f64)]) -> f64 {
    weights
        .iter()
        .map(|&(_, w)| if w > 0.0 { -w * w.ln() } else { 0.0 })
        .sum()
}

/// TEL-EM execution stats for logging / telemetry.
#[derive(Debug, Clone)]
pub struct TelEmStats {
    pub n_unambiguous: usize,
    pub n_ambiguous: usize,
    pub n_em_iter: usize,
    pub n_dropped_incompetent: u64,
    /// Reads dropped because all candidates were `is_polyA_competent == false`
    /// (ghost transcripts / n_flnc==0). Phase A telemetry.
    pub n_dropped_no_competent_candidates: u64,
    /// Posterior entropy 3-tier counters (per ambiguous read, in nats).
    /// clean: H < 0.20 nats (near-deterministic assignment).
    pub n_entropy_clean: u64,
    /// partial: 0.20 <= H <= 0.60 nats (two-three-way partial ambiguity).
    pub n_entropy_partial: u64,
    /// degenerate: H > 0.60 nats (effectively uniform over candidates).
    pub n_entropy_degenerate: u64,
    /// EM trace: per-iter `(iter_index, log_likelihood, mean_entropy)`.
    pub em_trace: Vec<(usize, f64, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    use arrow::array::BooleanBuilder;

    use crate::evidence::MoleculeEvidence;

    /// Output-stage truncation threshold mirrored from `solve_and_write_profile`.
    const MIN_WEIGHT: f64 = 0.01;

    #[test]
    fn test_poly_a_length_round_trip() {
        // log2(120) ≈ 6.9069, exp2 = 120
        assert_eq!(poly_a_length_from_log2(6.9069), 120);
        assert_eq!(poly_a_length_from_log2(7.0), 128);
    }

    /// Unique scratch path under the system temp dir. Callers remove it.
    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("telescope_em_{}_{}", name, std::process::id()))
    }

    /// Write a minimal `telescope.emission.parquet`: one poly(A)-competent row
    /// per `(transcript_id, gene_id, mu_logPAL, sigma2_logPAL)`.
    fn write_test_emission(path: &Path, rows: &[(&str, &str, f64, f64)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("transcript_id", DataType::Utf8, false),
            Field::new("gene_id", DataType::Utf8, false),
            Field::new("is_polyA_competent", DataType::Boolean, false),
            Field::new("mu_logPAL", DataType::Float64, false),
            Field::new("sigma2_logPAL", DataType::Float64, false),
        ]));
        let mut tx_b = StringBuilder::new();
        let mut gene_b = StringBuilder::new();
        let mut comp_b = BooleanBuilder::new();
        let mut mu_b = Float64Builder::new();
        let mut s2_b = Float64Builder::new();
        for &(tx, gene, mu, sigma2) in rows {
            tx_b.append_value(tx);
            gene_b.append_value(gene);
            comp_b.append_value(true);
            mu_b.append_value(mu);
            s2_b.append_value(sigma2);
        }
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(tx_b.finish()),
                Arc::new(gene_b.finish()),
                Arc::new(comp_b.finish()),
                Arc::new(mu_b.finish()),
                Arc::new(s2_b.finish()),
            ],
        )
        .unwrap();
        let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Build a one-molecule `FlatEvidenceGraph`: read `r1` with poly(A) length
    /// `log_pal` and one candidate per `(transcript_id, AS:i)` entry.
    fn synth_graph(log_pal: f32, candidates: &[(&str, i32)]) -> FlatEvidenceGraph {
        let mut transcript_names = Vec::new();
        let mut candidate_pool = Vec::new();
        let mut candidate_as = Vec::new();
        for (tx, as_score) in candidates {
            candidate_pool.push(transcript_names.len() as u32);
            transcript_names.push(tx.to_string());
            candidate_as.push(*as_score);
        }
        let n = candidates.len();
        FlatEvidenceGraph {
            molecules: vec![MoleculeEvidence {
                read_id: "r1".to_string(),
                log_pal,
                candidate_offset: 0,
                candidate_count: n as u16,
            }],
            candidate_nm: vec![0; n],
            candidate_mapq: vec![60; n],
            candidate_pool,
            candidate_as,
            transcript_names,
        }
    }

    /// Read back `telescope.profile.parquet` as `(transcript_id, weight)` rows
    /// in file order.
    fn read_profile(path: &Path) -> Vec<(String, f64)> {
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
            .unwrap()
            .with_batch_size(8192)
            .build()
            .unwrap();
        let mut out = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let tx = batch
                .column_by_name("transcript_id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let w = batch
                .column_by_name("posterior_weight")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                out.push((tx.value(i).to_string(), w.value(i)));
            }
        }
        out
    }

    fn weight_of(rows: &[(String, f64)], transcript_id: &str) -> f64 {
        rows.iter()
            .find(|(tx, _)| tx == transcript_id)
            .map(|&(_, w)| w)
            .unwrap_or_else(|| panic!("{transcript_id} missing from profile: {rows:?}"))
    }

    /// Regression test for the F-EM-4 A_rk term: it must reach the posterior.
    ///
    /// Both candidates carry IDENTICAL emission parameters (same μ, same σ²)
    /// and identical π at initialization, so `π_k · N(y; μ_k, σ_k²)` is exactly
    /// equal for the two and A_rk is the only term able to separate them.
    /// ΔAS = 200 at λ = 100 is a 2.0-nat gap → posterior 0.8808 / 0.1192.
    ///
    /// Mutation target: dropping `+ ln_a` from the E-step flattens both
    /// candidates to 0.5 and this test goes red.
    ///
    /// `max_iter = 1` runs exactly one E-step from the symmetric-π
    /// initialization, so the M-step feedback (which on its own would push π
    /// further towards whichever transcript won the first E-step) cannot be
    /// mistaken for the A_rk effect.
    #[test]
    fn test_a_rk_breaks_weight_tie_by_alignment_score() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("a_rk_emission.parquet");
        let profile = scratch("a_rk_profile.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", log_pal as f64, 0.04),
                ("tx1", "gene1", log_pal as f64, 0.04),
            ],
        );
        // One read aligning to tx0 with AS 1000 and to tx1 with AS 800.
        let graph = synth_graph(log_pal, &[("tx0", 1000), ("tx1", 800)]);

        solve_and_write_profile(graph, &emission, "TEST", &profile, 1e-6, 1, 100.0, None).unwrap();

        let rows = read_profile(&profile);
        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);

        let w_high_as = weight_of(&rows, "tx0");
        let w_low_as = weight_of(&rows, "tx1");
        assert!(
            w_high_as > w_low_as,
            "A_rk must favour the higher-AS candidate: tx0={w_high_as}, tx1={w_low_as}"
        );
        assert!(
            (w_high_as - 0.880_797).abs() < 1e-6,
            "a 2.0-nat A_rk gap must give 0.8808, got {w_high_as}"
        );
    }

    /// Unit test for the observational `max_as_ratio` statistic, covering the
    /// three defined cases of its contract.
    ///
    /// `synth_graph` puts read `r1`'s candidates at offset 0 in pool order, so
    /// the `AS:i` slice the helper reads is exactly the `candidates` argument.
    #[test]
    fn test_max_as_ratio_defined_cases() {
        let log_pal = (100.0_f32).log2();

        // (a) candidate_count == 1 → no second candidate → 0.0.
        let single = synth_graph(log_pal, &[("tx0", 3206)]);
        assert_eq!(
            read_max_as_ratio(&single, &single.molecules[0]),
            0.0,
            "a single candidate carries no runner-up, so the ratio is 0.0"
        );

        // (b) largest AS == 0 (every candidate tag-less) → 0.0, NOT 0/0 = NaN.
        let tagless = synth_graph(log_pal, &[("tx0", 0), ("tx1", 0), ("tx2", 0)]);
        assert_eq!(
            read_max_as_ratio(&tagless, &tagless.molecules[0]),
            0.0,
            "tag-less candidates must not produce a NaN from 0/0"
        );

        // (c) Normal case: runner-up over the top AS, in f64 then narrowed.
        //     3206/3338 = 0.96042... — the runner-up is deliberately placed LAST
        //     in pool order and the maximum FIRST, so a naive "first element vs
        //     max" implementation returns 1.0 and fails here.
        let mixed = synth_graph(log_pal, &[("tx0", 3338), ("tx1", 1654), ("tx2", 3206)]);
        let ratio = read_max_as_ratio(&mixed, &mixed.molecules[0]);
        assert!(
            (ratio - (3206.0_f64 / 3338.0) as f32).abs() < 1e-7,
            "ratio must be second-largest / largest, got {ratio}"
        );
        assert!(
            (0.0..=1.0).contains(&ratio),
            "the ratio is a proportion, got {ratio}"
        );

        // (d) Negative AS in the runner-up slot → 0.0, not a negative ratio.
        //     `AS` is a raw i32 tag and a malformed BAM can carry a negative
        //     value; `[100, -5]` would otherwise emit -0.05.
        let negative = synth_graph(log_pal, &[("tx0", 100), ("tx1", -5)]);
        assert_eq!(
            read_max_as_ratio(&negative, &negative.molecules[0]),
            0.0,
            "a negative runner-up AS is not evidence and must not yield a negative ratio"
        );
    }

    /// Regression test for the output-stage renormalization after the
    /// `min_weight` truncation.
    ///
    /// tx1's emission term is 5 nats worse than tx0's, landing it at
    /// `e⁻⁵/(1+e⁻⁵) = 0.0067` — below `MIN_WEIGHT`, so it is truncated away and
    /// 0.0067 of probability mass is destroyed. The rows actually emitted must
    /// still sum to exactly 1.0.
    ///
    /// Mutation target: summing the denominator over ALL weights instead of the
    /// retained ones leaves the surviving row at 0.9933 and this test goes red.
    /// Equal AS on both candidates keeps A_rk ≡ 1, so the split is produced by
    /// the emission term alone and this test stays independent of λ.
    #[test]
    fn test_emitted_weights_renormalize_after_min_weight_truncation() {
        let log_pal = (100.0_f32).log2();
        let sigma2 = 0.04;
        let emission = scratch("renorm_emission.parquet");
        let profile = scratch("renorm_profile.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", log_pal as f64, sigma2),
                // μ offset sized so `(y − μ)² / (2σ²)` = 5.0 nats.
                (
                    "tx1",
                    "gene1",
                    log_pal as f64 - (2.0 * sigma2 * 5.0).sqrt(),
                    sigma2,
                ),
            ],
        );
        let graph = synth_graph(log_pal, &[("tx0", 900), ("tx1", 900)]);

        solve_and_write_profile(graph, &emission, "TEST", &profile, 1e-6, 1, 100.0, None).unwrap();

        let rows = read_profile(&profile);
        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);

        assert_eq!(
            rows.len(),
            1,
            "the sub-{:0} candidate must be truncated away: {rows:?}",
            MIN_WEIGHT
        );
        assert_eq!(
            rows[0].1, 1.0,
            "the single retained row must absorb all posterior mass"
        );
        assert_eq!(
            rows.iter().map(|&(_, w)| w).sum::<f64>(),
            1.0,
            "emitted posterior weights must sum to exactly 1.0"
        );
    }

    /// Regression test for the *non-degenerate* renormalization path.
    ///
    /// The test above only ever retains one row, so its emitted weight is
    /// `w / w_max = 1.0` by construction and the interesting arithmetic —
    /// dividing **several** surviving rows by a `retained_sum < 1` — is never
    /// exercised. Three candidates split the posterior in the ratio
    /// 0.80 : 0.15 : 0.005. The third is below `MIN_WEIGHT` (0.01) and is
    /// truncated, so 0.005 of probability mass is destroyed and the denominator
    /// drops to the retained 0.80 + 0.15. The emitted rows must therefore be
    /// ≈ 0.8421 / 0.1579 — not 0.80 / 0.15 — and sum to 1.0.
    ///
    /// Mutation target: summing the denominator over ALL weights instead of the
    /// retained ones leaves the emitted rows at 0.8377 / 0.1571, summing to
    /// ≈ 0.9948, and this test goes red.
    ///
    /// Equal AS on all three candidates keeps A_rk ≡ 1, so the split is
    /// produced by the emission term alone and this test stays independent
    /// of λ, just like its degenerate sibling above.
    #[test]
    fn test_emitted_weights_renormalize_across_multiple_retained_rows() {
        let log_pal = (100.0_f32).log2();
        let sigma2 = 0.04;
        let emission = scratch("renorm3_emission.parquet");
        let profile = scratch("renorm3_profile.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", log_pal as f64, sigma2),
                // μ offset sized so `(y − μ)² / (2σ²)` = ln(0.80 / 0.15) nats.
                (
                    "tx1",
                    "gene1",
                    log_pal as f64 - (2.0 * sigma2 * (0.80_f64 / 0.15).ln()).sqrt(),
                    sigma2,
                ),
                // ... = ln(0.80 / 0.005) nats — lands below `MIN_WEIGHT`.
                (
                    "tx2",
                    "gene2",
                    log_pal as f64 - (2.0 * sigma2 * (0.80_f64 / 0.005).ln()).sqrt(),
                    sigma2,
                ),
            ],
        );
        let graph = synth_graph(log_pal, &[("tx0", 900), ("tx1", 900), ("tx2", 900)]);

        solve_and_write_profile(graph, &emission, "TEST", &profile, 1e-6, 1, 100.0, None).unwrap();

        let rows = read_profile(&profile);
        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);

        assert_eq!(
            rows.len(),
            2,
            "only the sub-{:0} candidate may be truncated away: {rows:?}",
            MIN_WEIGHT
        );

        let w0 = weight_of(&rows, "tx0");
        let w1 = weight_of(&rows, "tx1");
        // 0.80 / 0.95 and 0.15 / 0.95 up to the shared 0.005 that was truncated
        // out of the normalization (exact values: 0.842105 / 0.157895).
        assert!(
            (w0 - 0.842_105).abs() < 1e-5,
            "the dominant row must absorb the truncated mass: {w0}"
        );
        assert!(
            (w1 - 0.157_895).abs() < 1e-5,
            "the runner-up must be renormalized by the same denominator: {w1}"
        );
        let sum: f64 = rows.iter().map(|&(_, w)| w).sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "emitted posterior weights must sum to 1.0, got {sum}"
        );
    }
}
