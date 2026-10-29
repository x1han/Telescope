//! TEL-EM solver: the bifurcated Terminal-aware Expectation-Maximization solver.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{
    BooleanArray, Float32Builder, Float64Array, Float64Builder, Int32Builder, StringArray,
    StringBuilder, UInt16Builder, UInt32Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

use crate::emission_model::INSTRUMENTAL_PAL_NOISE_FLOOR;
use crate::evidence::{FlatEvidenceGraph, MoleculeEvidence};

pub const NU_0: f64 = 1.0;
pub const BETA_0: f64 = 1.0;
pub const TAU_0: f64 = 2.0;
/// Degrees-of-freedom offset `c` in the σ² M-step denominator
/// `(S(mu) + 2*beta_0*sigma2_cohort) / (W + 2*nu_0 + c)`; `c = 2` IS the
/// Model A estimator and `c != 2` is NEVER Model B.
pub const SIGMA2_DOF_OFFSET: f64 = 2.0;

#[derive(Debug, Clone)]
pub struct TranscriptParam {
    pub transcript_id: String,
    pub gene_id: String,
    pub is_competent: bool,
    /// Shrinkage-prior μ, bound at load from the emission parquet's own
    /// `mu_logPAL` column. "cohort" = the tier the value was resolved at.
    pub mu_cohort: f64,
    pub mu: f64,
    /// Shrinkage-prior σ², bound at load from the emission parquet's own
    /// `sigma2_logPAL` column.
    pub sigma2_cohort: f64,
    pub sigma2: f64,
    pub alpha: f64,
    /// The M-step's W for this transcript: the E-step weight sum, pre-truncation
    /// and pre-renormalisation, stamped after the EM loop from the FINAL E-step's
    /// weights. NOT recoverable from `alpha` (W divided by the molecule count).
    pub em_accum_weight: f64,
    /// TRUE when at least one M-step completed, so `alpha`, `mu` and `sigma2` are
    /// an M-step's output rather than the Laplace initialisation. Convergence breaks
    /// BEFORE the M-step, so a first-iteration convergence exports the init.
    pub em_finalized: bool,
    /// TRUE when this row was IMPUTED to reach the reference universe, i.e.
    /// `n_cohort_flnc == 0` in the emission parquet; it carries the cohort fit.
    pub is_imputed: bool,
}

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
        let nobs_col = batch
            .column_by_name("n_cohort_flnc")
            .context("Missing 'n_cohort_flnc'")?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("n_cohort_flnc is not UInt32Array")?;

        for i in 0..batch.num_rows() {
            let tr_id = tr_col.value(i).to_string();
            let gene_id = gene_col.value(i).to_string();
            let is_competent = comp_col.value(i);
            let mu_cohort = mu_col.value(i);
            let mu = mu_cohort; // Initial value = cohort; M-step will update
            let sigma2_cohort = sigma2_col.value(i);

            // `sigma2_cohort` is bound from this transcript's own `sigma2_logPAL`
            // and is a VARIANCE: it must arrive finite and strictly positive, or
            // it poisons `p.sigma2` for every transcript that borrows it.
            anyhow::ensure!(
                sigma2_cohort.is_finite() && sigma2_cohort > 0.0,
                "emission row {i} (transcript {}) has sigma2_logPAL = {}, which is \
                 not a finite positive variance; it is about to become the M-step \
                 prior scale beta_0 * sigma2_cohort",
                tr_id,
                sigma2_cohort
            );

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
                    alpha: 0.0,           // initialized in solve_and_write_profile
                    em_accum_weight: 0.0, // stamped after the EM loop; see the field doc
                    em_finalized: false,  // set after the EM loop
                    is_imputed: nobs_col.value(i) == 0,
                },
            );
        }
    }

    Ok(map)
}

/// Run TEL-EM and write `telescope.profile.parquet`. `aln_weight_lambda` must be > 0.
///
/// Evidence terms, in the form that must not be re-derived:
///     `g_k` = `s_k - 1`, the 1-BASED start of the read's BEST alignment to candidate
///     k, in candidate k's OWN cDNA coordinates (never a genomic span).
///     `ln_5p_k` = `five_prime_weight` if `g_k <= five_prime_window` else 0.0.
///     prune: drop `Delta-AS_k > prune_max_delta_as`; `w_k` ∝ `pi_k * exp(ln_a_k + ln_5p_k)`.
#[allow(clippy::too_many_arguments)]
pub fn solve_and_write_profile<P: AsRef<Path>>(
    graph: FlatEvidenceGraph,
    emission_path: P,
    sample_id: &str,
    output_profile_path: P,
    tol: f64,
    max_iter: usize,
    aln_weight_lambda: f64,
    five_prime_weight: f64,
    poly_a_weight: f64,
    five_prime_window: u32,
    prune_max_delta_as: i32,
    min_weight: f64,
    output_em_trace: Option<&mut Vec<(usize, f64, f64)>>,
    output_emission_params: Option<&mut Vec<Option<TranscriptParam>>>,
    output_read_assignments: Option<&mut Vec<ReadAssignment>>,
) -> Result<TelEmStats> {
    solve_and_write_profile_impl(
        graph,
        emission_path,
        sample_id,
        output_profile_path,
        tol,
        max_iter,
        aln_weight_lambda,
        five_prime_weight,
        poly_a_weight,
        five_prime_window,
        prune_max_delta_as,
        min_weight,
        output_em_trace,
        output_emission_params,
        output_read_assignments,
        SIGMA2_DOF_OFFSET,
    )
}

/// `solve_and_write_profile` with the σ² regularisation degrees-of-freedom offset
/// `c` supplied by the caller (same arguments, in the same order, plus this one).
/// `c` is the trailing `+ c` of the σ² denominator; `c = 3` is NOT the joint NIG
/// MAP -- read `SIGMA2_DOF_OFFSET`.
#[allow(clippy::too_many_arguments)]
pub fn solve_and_write_profile_with_dof_offset<P: AsRef<Path>>(
    graph: FlatEvidenceGraph,
    emission_path: P,
    sample_id: &str,
    output_profile_path: P,
    tol: f64,
    max_iter: usize,
    aln_weight_lambda: f64,
    five_prime_weight: f64,
    poly_a_weight: f64,
    five_prime_window: u32,
    prune_max_delta_as: i32,
    min_weight: f64,
    output_em_trace: Option<&mut Vec<(usize, f64, f64)>>,
    output_emission_params: Option<&mut Vec<Option<TranscriptParam>>>,
    output_read_assignments: Option<&mut Vec<ReadAssignment>>,
    sigma2_dof_offset: f64,
) -> Result<TelEmStats> {
    assert!(
        sigma2_dof_offset.is_finite() && sigma2_dof_offset > 0.0,
        "sigma2_dof_offset must be finite and positive, got {sigma2_dof_offset}"
    );
    solve_and_write_profile_impl(
        graph,
        emission_path,
        sample_id,
        output_profile_path,
        tol,
        max_iter,
        aln_weight_lambda,
        five_prime_weight,
        poly_a_weight,
        five_prime_window,
        prune_max_delta_as,
        min_weight,
        output_em_trace,
        output_emission_params,
        output_read_assignments,
        sigma2_dof_offset,
    )
}

#[allow(clippy::too_many_arguments)]
fn solve_and_write_profile_impl<P: AsRef<Path>>(
    graph: FlatEvidenceGraph,
    emission_path: P,
    sample_id: &str,
    output_profile_path: P,
    tol: f64,
    max_iter: usize,
    aln_weight_lambda: f64,
    five_prime_weight: f64,
    // Scale on the poly(A) mixture term `ln alpha_k - ln sigma2_k - (y - mu_k)^2
    // / (2 sigma2_k)`. 1.0 is the shipped model; 0.0 removes the poly(A)
    // evidence, but the abundance prior `ln alpha_k` survives at any weight.
    poly_a_weight: f64,
    five_prime_window: u32,
    prune_max_delta_as: i32,
    min_weight: f64,
    mut output_em_trace: Option<&mut Vec<(usize, f64, f64)>>,
    mut output_emission_params: Option<&mut Vec<Option<TranscriptParam>>>,
    output_read_assignments: Option<&mut Vec<ReadAssignment>>,
    // Trailing constant `c` of the σ² denominator. Callers reach this through
    // `solve_and_write_profile` or `solve_and_write_profile_with_dof_offset`.
    sigma2_dof_offset: f64,
) -> Result<TelEmStats> {
    // API-LEVEL DEFENCE: the CLI forces `poly_a_weight = 0` when the graph has no
    // poly(A), but a library caller never goes through the CLI. Pairing a
    // placeholder graph with a positive weight feeds a constant emission into the
    // E-step and silently distorts every posterior.
    if poly_a_weight > 0.0 && graph.poly_a_is_placeholder {
        anyhow::bail!(
            "FATAL: poly_a_weight = {poly_a_weight} on a graph built with \
             --polyA-source none. Every molecule's log_pal is the placeholder, \
             not an observed tail length, so the Gaussian would score a constant \
             emission and distort every posterior. Re-run with \
             --polyA-source bam_tag:<TAG>, or set --poly-a-weight 0."
        );
    }

    if aln_weight_lambda <= 0.0 || aln_weight_lambda.is_nan() {
        anyhow::bail!(
            "aln_weight_lambda must be > 0 (got {}); it scales ln A_rk = (AS_rk − \
             max_j AS_rj)/λ, so 0 and NaN would emit NaN posterior weights",
            aln_weight_lambda
        );
    }
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !poly_a_weight.is_finite() || poly_a_weight < 0.0 {
        anyhow::bail!(
            "poly_a_weight must be finite and >= 0 (got {}); 0.0 is the valid \
             poly(A)-off switch (the abundance prior and alignment terms remain), \
             but a negative weight inverts the poly(A) \
             term and NaN would emit NaN posterior weights",
            poly_a_weight
        );
    }
    if !min_weight.is_finite() || !(0.0..1.0).contains(&min_weight) {
        anyhow::bail!(
            "min_weight must be finite and in [0, 1) (got {}); it is the output-stage \
             posterior truncation threshold, and 1.0 would retain only a read's \
             argmax, silently turning the profile into a hard-assignment table",
            min_weight
        );
    }
    // 5'-anchoring weight. 0.0 is a LEGAL off switch: it makes `ln_5p ≡ 0`, so the
    // term leaves the softmax EXACTLY. Only a negative weight or NaN are rejected;
    // `is_finite()`, because +inf poisons the log-sum-exp with `inf - inf`.
    if !five_prime_weight.is_finite() || five_prime_weight < 0.0 {
        anyhow::bail!(
            "five_prime_weight must be >= 0 and non-NaN (got {}); it is the log-space \
             bonus ln_5p awarded to a candidate whose observed 5' gap g_k <= \
             five_prime_window, so 0.0 (the off switch) is valid but a negative \
             weight would invert the term",
            five_prime_weight
        );
    }
    if prune_max_delta_as < 0 {
        anyhow::bail!(
            "prune_max_delta_as must be >= 0 (got {}); it drops candidates whose \
             alignment score is more than this far below the read's best, and a \
             negative threshold would drop the argmax too, emptying every read",
            prune_max_delta_as
        );
    }

    if max_iter == 0 {
        anyhow::bail!(
            "max_iter must be >= 1 (got {}); with 0 iterations no E-step runs, no posterior is \
             produced, and every candidate would be reported as having no emission row",
            max_iter
        );
    }

    let mut param_map = load_emission_parameters(emission_path)?;
    let num_transcripts = graph.transcript_names.len();
    let n_emission = param_map.len();

    let mut params: Vec<Option<TranscriptParam>> = Vec::with_capacity(num_transcripts);
    for name in &graph.transcript_names {
        params.push(param_map.remove(name));
    }
    let n_matched = params.iter().filter(|p| p.is_some()).count();

    // ID-SPACE GUARD. `graph.transcript_names` comes from the BAM @SQ header; the
    // emission parquet is keyed by whatever `build-emission` was handed, and the
    // two can be DIFFERENT SPACES. A zero overlap makes every `params[i]` `None`, so
    // every read takes the ghost-drop path and the run still exits 0 -- refuse it.
    if num_transcripts > 0 && n_matched == 0 {
        let bam_sample = graph
            .transcript_names
            .first()
            .map(String::as_str)
            .unwrap_or("");
        let em_sample = param_map.keys().next().map(String::as_str).unwrap_or("");
        anyhow::bail!(
            "ID-SPACE MISMATCH: none of the {num_transcripts} transcripts in the BAM are \
             present among the {n_emission} rows of the emission parquet (BAM id e.g. \
             {bam_sample:?}, emission id e.g. {em_sample:?}). Every read would be \
             silently dropped and an EMPTY profile written. The configuration that \
             produces this is a BAM keyed by one id space (e.g. `tx_1`, `tx_2`, ...) \
             while the emission tables are keyed by another (`ENST`, `ENSG_NovelTr`); \
             it was bridged with `--tx-id-mapping <bam_id<TAB>original_id tsv>`. \
             Universal Mode (`build-anchors`) needs no bridge, because its ids come \
             from the BAM itself."
        );
    }

    let mut unambiguous_indices: Vec<usize> = Vec::new();
    let mut ambiguous_indices: Vec<usize> = Vec::new();
    let mut ledger = CandidateLedger::default();
    for (idx, mol) in graph.molecules.iter().enumerate() {
        match mol.candidate_count {
            0 => {}
            n => {
                ledger.total_candidates += u64::from(n);
                if n == 1 {
                    unambiguous_indices.push(idx);
                } else {
                    ambiguous_indices.push(idx);
                }
            }
        }
    }
    let n_total_molecules = graph.molecules.len() as f64;
    let n_unambiguous = unambiguous_indices.len();
    let n_ambiguous = ambiguous_indices.len();

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

    use crate::em::{BETA_0, NU_0, TAU_0};
    let nu_0 = NU_0;
    let beta_0 = BETA_0;
    let tau_0 = TAU_0;
    let mut prev_ll = f64::NEG_INFINITY;
    let mut final_weights: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_ambiguous];
    let mut n_em_iter = 0_usize;

    // The `(mu, sigma2)` the LAST E-step actually consumed. The M-step rewrites
    // `p.mu`/`p.sigma2` under `poly_a_weight > 0.0`, so the telemetry must read
    // the PRE-M-step state.
    let mut last_estep_params: Vec<Option<TranscriptParam>> = Vec::new();

    let mut m_step_ran = false;

    // The W the LAST M-step actually consumed. `p.em_accum_weight` is stamped at
    // the end of the run from a FRESH suffstats pass over the post-loop
    // `final_weights`. The two are one E-step apart.
    let mut last_mstep_w: Vec<f64> = Vec::new();

    for iter in 0..max_iter {
        let e_results: Vec<(f64, Vec<(usize, f64)>, f64)> = ambiguous_indices
            .par_iter()
            .map(|&mol_idx| {
                let mol = &graph.molecules[mol_idx];
                let y = mol.log_pal as f64;

                // Alignment-score reference point for this read. Taken over the
                // FULL `0..candidate_count` range -- before competence masking, and
                // over candidates the `params` lookup below skips entirely.
                let max_as = (0..mol.candidate_count)
                    .map(|c| graph.candidate_as[(mol.candidate_offset + c as u32) as usize])
                    .fold(i32::MIN, i32::max);

                let mut raw: Vec<(usize, f64)> = Vec::with_capacity(mol.candidate_count as usize);
                for c in 0..mol.candidate_count {
                    let idx = (mol.candidate_offset + c as u32) as usize;
                    let tx_id = graph.candidate_pool[idx] as usize;

                    // ΔAS is a MAGNITUDE: the read's best score minus this
                    // candidate's, so ΔAS >= 0. The prune thresholds the magnitude,
                    // the log term negates it. `saturating_sub`, not `-`: a plain
                    // subtraction WRAPS under `--release` into a negative ΔAS.
                    let delta_as = max_as.saturating_sub(graph.candidate_as[idx]);

                    // ΔAS prune: drop candidates whose alignment score is more than
                    // `prune_max_delta_as` below the read's best. `-inf` rather than a
                    // bare `continue`, matching the competence mask below: the two are
                    // observationally IDENTICAL here because `weights` is built with
                    // `filter(ll > -inf)`. It cannot empty a read: the argmax has
                    // `delta_as == 0`.
                    if delta_as > prune_max_delta_as {
                        raw.push((tx_id, f64::NEG_INFINITY));
                        continue;
                    }

                    let ln_a = -(delta_as as f64) / aln_weight_lambda;

                    // ln_5p_k = β if g_k ≤ window else 0. `g_k` is the observed 5' gap
                    // of the read's best alignment to THIS candidate, in this
                    // candidate's own cDNA coordinates -- a PRIOR, not an EM parameter.
                    let ln_5p = if graph.candidate_5p_gap[idx] <= five_prime_window {
                        five_prime_weight
                    } else {
                        0.0
                    };

                    if let Some(ref p) = params[tx_id] {
                        if !p.is_competent || p.alpha <= 0.0 {
                            raw.push((tx_id, f64::NEG_INFINITY));
                        } else {
                            let ll = p.alpha.ln()
                                + poly_a_weight
                                    * (-(p.sigma2.sqrt()).ln()
                                        - ((y - p.mu).powi(2) / (2.0 * p.sigma2)))
                                + ln_a
                                + ln_5p;
                            raw.push((tx_id, ll));
                        }
                    }
                }

                let max_ll = raw
                    .iter()
                    .map(|&(_, ll)| ll)
                    .fold(f64::NEG_INFINITY, f64::max);
                if max_ll == f64::NEG_INFINITY {
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

        last_estep_params = params.clone();

        let current_ll: f64 = e_results.iter().map(|&(ll, _, _)| ll).sum();
        n_em_iter = iter + 1;

        if let Some(ref mut trace) = output_em_trace {
            let n_with_entropy = e_results.iter().filter(|&&(_, _, h)| h > 0.0).count();
            let mean_entropy = if n_with_entropy > 0 {
                e_results.iter().map(|&(_, _, h)| h).sum::<f64>() / n_with_entropy as f64
            } else {
                0.0
            };
            trace.push((iter + 1, current_ll, mean_entropy));
        }

        // Convergence check: per-molecule average log-likelihood delta. The ABSOLUTE
        // delta has summation noise of order sqrt(N_ambig) * eps * |log L| from Rayon.
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

        for (i, (_, w, _)) in e_results.into_iter().enumerate() {
            final_weights[i] = w;
        }

        // M-step: single-pass algebraic closed-form. Pre-M-step accumulators collect
        // only (W, S1=Σ w·y, S2=Σ w·y²) per transcript. The σ² deviation
        // Σ w·(y − μ_new)² is reconstructed AFTER μ is updated, never around a stale μ.
        m_step_ran = true; // "an M-step ran" means this flag, not `n_em_iter`.
        let (accum_w, accum_w_y, accum_w_y2) = accumulate_mstep_suffstats(
            &params,
            &graph,
            &unambiguous_indices,
            &ambiguous_indices,
            &final_weights,
            num_transcripts,
        );

        // 3-step update: μ → σ² (around μ_new) → π. The order is locked. STEPS 1 AND 2
        // ARE SKIPPED AT `poly_a_weight == 0`: guarding only the two `accum_w_y*` lines
        // drags μ to 0 and σ² onto the noise floor. STEP 3 is NOT skipped.
        for i in 0..num_transcripts {
            if let Some(ref mut p) = params[i] {
                let w_sum = accum_w[i];

                // STEP 0: record W itself, OUTSIDE the poly(A) guard below: at
                // `poly_a_weight == 0` W is still what `alpha` is built from.
                p.em_accum_weight = w_sum;

                if poly_a_weight > 0.0 {
                    p.mu = (accum_w_y[i] + tau_0 * p.mu_cohort) / (w_sum + tau_0);

                    let s1 = accum_w_y[i];
                    let s2 = accum_w_y2[i];
                    let sq_dev_new = (s2 - 2.0 * p.mu * s1 + p.mu * p.mu * w_sum).max(0.0);
                    // MODEL A — fixed-absolute-precision prior.
                    //
                    // Model B (the joint NIG MAP) would put `tau_0*(mu - mu_cohort)^2` back
                    // in the NUMERATOR and `+3` in the denominator. Both changes go
                    // together: keeping one without the other is a different estimator.
                    // The trailing `+ 2` is `sigma2_dof_offset`, pinned to
                    // `SIGMA2_DOF_OFFSET == 2.0`.
                    p.sigma2 = (sq_dev_new + 2.0 * beta_0 * p.sigma2_cohort)
                        / (w_sum + 2.0 * nu_0 + sigma2_dof_offset);
                    p.sigma2 = p.sigma2.max(INSTRUMENTAL_PAL_NOISE_FLOOR);
                }

                p.alpha = if n_total_molecules > 0.0 {
                    w_sum / n_total_molecules
                } else {
                    0.0
                };
            }
        }

        last_mstep_w = accum_w;
    }

    // STAMP THE RUN'S ACTUAL W AND FINALISATION STATE INTO THE SNAPSHOT.
    //
    // The snapshot the sidecar is written from is taken BEFORE the M-step runs, and
    // on convergence the loop breaks before the M-step too -- so the M-step's own
    // assignment is not enough. The convergence `break` fills `final_weights`.
    let mut max_dw_phase_abs = 0.0_f64;
    let mut max_dw_phase_rel = 0.0_f64;
    {
        let (final_w, _, _) = accumulate_mstep_suffstats(
            &params,
            &graph,
            &unambiguous_indices,
            &ambiguous_indices,
            &final_weights,
            num_transcripts,
        );
        for (i, slot) in last_estep_params.iter_mut().enumerate() {
            if let Some(ref mut p) = slot {
                p.em_accum_weight = final_w[i];
                p.alpha = if n_total_molecules > 0.0 {
                    final_w[i] / n_total_molecules
                } else {
                    0.0
                };
                p.em_finalized = m_step_ran;
            }
        }

        if !last_mstep_w.is_empty() {
            for i in 0..num_transcripts {
                let m_step_i = last_mstep_w[i];
                let d = (final_w[i] - m_step_i).abs();
                if d > max_dw_phase_abs {
                    max_dw_phase_abs = d;
                }
                let scale = final_w[i].abs().max(m_step_i.abs());
                if scale > 0.0 {
                    let r = d / scale;
                    if r > max_dw_phase_rel {
                        max_dw_phase_rel = r;
                    }
                }
            }
        }
    }

    let want_read_assignments = output_read_assignments.is_some();

    let mut sample_b = StringBuilder::new();
    let mut read_b = StringBuilder::new();
    let mut gene_b = StringBuilder::new();
    let mut tx_b = StringBuilder::new();
    let mut pal_b = Int32Builder::new();
    let mut w_b = Float64Builder::new();
    let mut ncand_b = UInt16Builder::new();
    let mut as_ratio_b = Float32Builder::new();
    let mut n_unambiguous_dropped_incompetent = 0_u64;
    let mut n_unambiguous_dropped_missing_emission = 0_u64;
    let mut n_ghost_read_all_missing = 0_u64;
    let mut n_ghost_read_all_incompetent = 0_u64;
    let mut n_ghost_read_all_alpha_zero = 0_u64;
    let mut n_dropped_output_row_competence = 0_u64;
    let mut n_dropped_no_competent_candidates = 0_u64;
    let mut n_reads_any_emission_candidate = 0_u64;
    let mut n_reads_single_emission_candidate = 0_u64;
    let mut n_reads_multi_emission_candidate = 0_u64;
    let mut n_polya_zero_power_resolved = 0_u64;
    let mut n_entropy_clean = 0_u64;
    let mut n_entropy_partial = 0_u64;
    let mut n_entropy_degenerate = 0_u64;
    let mut n_degenerate_all_imputed = 0_u64;
    let mut n_degenerate_with_anchored = 0_u64;
    let mut n_polya_zero_power = 0_u64;

    // Per-read assignments are filled into a molecule-indexed slot vector and
    // flattened at the end, so the emitted table is in MOLECULE order, one row per
    // read. `final_weights` is indexed by AMBIGUOUS RANK.
    let mut read_assign: Vec<Option<ReadAssignment>> = if want_read_assignments {
        vec![None; graph.molecules.len()]
    } else {
        Vec::new()
    };

    for &idx in &unambiguous_indices {
        let mol = &graph.molecules[idx];
        let tx_id = graph.candidate_pool[mol.candidate_offset as usize] as usize;
        let pal_i32 = poly_a_length_from_log2(mol.log_pal);
        // Observational columns. Both are per READ properties, derived once here and
        // repeated verbatim on every row. `candidate_count` is the PRE-prune count.
        let ncand = mol.candidate_count;
        let as_ratio = read_max_as_ratio(&graph, mol);
        // EMISSION COVERAGE, counted over ALL reads. An unambiguous read has exactly
        // one candidate, so its `n_with_emission` is 0 or 1.
        let un_with_emission = params[tx_id].is_some();
        n_reads_any_emission_candidate += u64::from(un_with_emission);
        n_reads_single_emission_candidate += u64::from(un_with_emission);
        if let Some(ref p) = params[tx_id] {
            if !p.is_competent {
                n_unambiguous_dropped_incompetent += 1;
                ledger.unambig_incompetent += 1;
                continue;
            }
            if want_read_assignments {
                read_assign[idx] = Some(ReadAssignment {
                    read_id: mol.read_id.clone(),
                    n_candidates_before_cutoff: ncand as usize,
                    top_tx_id: Some(p.transcript_id.clone()),
                    top_weight: 1.0,
                    entropy: 0.0,
                });
            }
            sample_b.append_value(sample_id);
            read_b.append_value(&mol.read_id);
            gene_b.append_value(&p.gene_id);
            tx_b.append_value(&p.transcript_id);
            pal_b.append_value(pal_i32);
            w_b.append_value(1.0);
            ncand_b.append_value(ncand);
            as_ratio_b.append_value(as_ratio);
            ledger.unambig_emitted += 1;
        } else {
            n_unambiguous_dropped_missing_emission += 1;
            ledger.unambig_missing += 1;
        }
    }

    // Ambiguous rows: per (read × candidate) with posterior ≥ `min_weight`, a
    // CALLER-SUPPLIED parameter: the losing candidate's row is dropped at the
    // shipped 0.01 and the survivors are renormalised to 1.0.
    for (i, weights) in final_weights.into_iter().enumerate() {
        let mol_idx = ambiguous_indices[i];
        let mol = &graph.molecules[mol_idx];
        let pal_i32 = poly_a_length_from_log2(mol.log_pal);
        // Observational columns, per READ — computed once, before the prune and the
        // truncation can drop rows. `candidate_count` is the ORIGINAL count.
        let ncand = mol.candidate_count;
        let as_ratio = read_max_as_ratio(&graph, mol);

        // DEGRADATION TELEMETRY — how often poly(A) had no say at all. See
        // `polya_emission_count_and_identity`. `last_estep_params`, NOT `params`.
        let (n_with_emission, all_identical) =
            polya_emission_count_and_identity(&graph, mol, &last_estep_params);
        n_reads_any_emission_candidate += u64::from(n_with_emission > 0);
        n_reads_single_emission_candidate += u64::from(n_with_emission == 1);
        if n_with_emission >= ZERO_POWER_MIN_EMISSION_CANDIDATES {
            n_reads_multi_emission_candidate += 1;
            if all_identical {
                n_polya_zero_power_resolved += 1;
            }
        }
        if n_with_emission > 0 && all_identical {
            n_polya_zero_power += 1;
        }

        // CANDIDATE CONSERVATION LEDGER. Taken for EVERY contested read, not only for
        // ghosts: a counter inside the ghost branch accumulates over that population.
        let av = candidate_availability(&graph, mol, &last_estep_params, prune_max_delta_as);
        ledger.pruned_delta_as += av.n_pruned_delta_as as u64;
        ledger.missing += av.n_missing as u64;
        ledger.incompetent += av.n_incompetent as u64;
        ledger.alpha_zero += av.n_alpha_zero as u64;
        ledger.live_candidates += av.n_live as u64;

        if weights.is_empty() {
            n_dropped_no_competent_candidates += 1;
            // GHOST DECOMPOSITION. `weights.is_empty()` on its own conflates three
            // unrelated causes -- absent emission row, `is_polyA_competent == false`,
            // and `alpha <= 0.0` -- and cannot be attributed without a census.
            match av.uniform_cause(mol.candidate_count as usize) {
                Some(GhostCause::Missing) => n_ghost_read_all_missing += 1,
                Some(GhostCause::Incompetent) => n_ghost_read_all_incompetent += 1,
                Some(GhostCause::AlphaZero) => n_ghost_read_all_alpha_zero += 1,
                None => {}
            }
        } else {
            let h = posterior_entropy(&weights);
            if h < 0.20 {
                n_entropy_clean += 1;
            } else if h <= 0.70 {
                n_entropy_partial += 1;
            } else {
                n_entropy_degenerate += 1;
                // Split the degenerate reads by WHY they are flat. `weights` carries the
                // GLOBAL transcript id of every survivor, and `params` is indexed by it.
                let mut all_imputed = true;
                for &(tx_idx, w) in &weights {
                    if w <= 0.0 {
                        continue;
                    }
                    let imputed = params[tx_idx]
                        .as_ref()
                        .map(|p| p.is_imputed)
                        .unwrap_or(true);
                    if !imputed {
                        all_imputed = false;
                        break;
                    }
                }
                if all_imputed {
                    n_degenerate_all_imputed += 1;
                } else {
                    n_degenerate_with_anchored += 1;
                }
            }
        }

        // THE INVARIANT this table has to keep is one-directional:
        //
        //     top_tx_id == Some(..)  IMPLIES  that read has a profile row.
        //
        // A GHOST read has no profile row (`top_tx_id: None`); a `min_weight`-culled
        // read HAS a real argmax and IS published under it at weight 1.0. The argmax
        // is computed ONCE, here: a second copy would drift from this one.
        let (top, top_w) = weights.iter().fold((None, 0.0f64), |acc, &(tx, w)| {
            // Strict `>` (not `>=`) => on an exact tie the FIRST candidate in pool order
            // wins, deterministically.
            if w > acc.1 {
                (Some(tx), w)
            } else {
                acc
            }
        });

        if want_read_assignments {
            // `weights` is pre-truncation and pre-renormalisation, so `top_weight` is the
            // RAW E-step weight; the profile's `posterior_weight` is `w / retained_sum`.
            read_assign[mol_idx] = Some(ReadAssignment {
                read_id: mol.read_id.clone(),
                n_candidates_before_cutoff: ncand as usize,
                // `None` only on the ghost path: the fold never ran, so there is no
                // candidate to name. That rests on the E-step pushing `-inf` for any
                // candidate with `!p.is_competent || p.alpha <= 0.0`.
                top_tx_id: top.and_then(|tx| {
                    params
                        .get(tx)
                        .and_then(|o| o.as_ref())
                        .map(|p| p.transcript_id.clone())
                }),
                top_weight: top_w,
                entropy: if weights.is_empty() {
                    0.0
                } else {
                    posterior_entropy(&weights)
                },
            });
        }

        // Renormalize the retained weights so each read's emitted posterior weights sum
        // to 1.0. The denominator is over the rows actually emitted, i.e.
        // `{w >= min_weight}` -- the truncation above discards real probability mass.
        //
        // COUPLING: that denominator is right only because the E-step's `if let Some`
        // (above) *skips* candidates with missing params instead of pushing `-inf`.
        //
        // THE COUPLED PREDICATE: this filter and the `w < min_weight` test in the emit
        // loop below are two spellings of ONE decision. If this returned 0 while the
        // loop still emitted the argmax, `w_norm` would be `w / 0.0`.
        let retained_sum: f64 = {
            let s: f64 = weights
                .iter()
                .filter(|&(_, w)| *w >= min_weight)
                .map(|&(_, w)| w)
                .sum();
            if s > 0.0 {
                s
            } else {
                top_w
            }
        };

        ledger.live_candidates_entered += weights.len() as u64;

        let argmax_only = weights.iter().all(|&(_, w)| w < min_weight);

        for (tx_id, w) in weights {
            if w < min_weight && !(argmax_only && Some(tx_id) == top) {
                // Provably 0 at `min_weight = 0.0`: every posterior is
                // `exp(ll - log_z) >= 0`, so `w < 0.0` is never true. The argmax is the
                // ONE exception: it is culled from the prune count, not culled -- it is
                // emitted, one line below.
                ledger.pruned_min_weight += 1;
                continue;
            }
            // Safe: this body only runs when at least one weight is retained --
            // `retained_sum > 0` follows from the softmax summing to 1.
            let w_norm = w / retained_sum;
            if let Some(ref p) = params[tx_id] {
                if !p.is_competent {
                    n_dropped_output_row_competence += 1;
                    ledger.dropped_output_competence += 1;
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
                ledger.rows_emitted += 1;
                if argmax_only && Some(tx_id) == top {
                    ledger.reads_retained_by_argmax += 1;
                }
            } else {
                n_dropped_output_row_competence += 1;
                ledger.dropped_output_competence += 1;
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
        // The two columns below are OBSERVATIONAL ONLY -- nothing in TEL-EM reads them
        // back, and they do not participate in A_rk, the cutoff, or the renormalization.
        //
        // CAVEAT -- THE EMITTED DISTRIBUTION IS SURVIVAL-BIASED: both columns ride on
        // rows that survive the `Delta-AS > prune_max_delta_as` prune AND the
        // `min_weight` cutoff. Any analysis MUST group by `read_id` and report the
        // read-level coverage rate.
        Field::new("n_candidates_before_cutoff", DataType::UInt16, false),
        // Second-largest / largest `AS:i` over the read's full candidate range. 0.0
        // when the read has no second candidate or no candidate carries an `AS:i` tag.
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

    // THE CANDIDATE CONSERVATION TRIPWIRE.
    //
    // Deliberately AFTER the parquet is closed: the profile is the expensive artefact,
    // and deleting it over a counter disagreement would destroy the evidence needed
    // to find out why.
    //
    // A non-zero value means one of two things: (a) the census and the E-step have
    // drifted apart, so a class of candidate reached no terminal bucket; or (b) a
    // candidate produced a non-finite posterior that `filter(ll > -inf)` dropped.
    anyhow::ensure!(
        ledger.deficit() == 0,
        "candidate ledger does not close: deficit {} (candidates {} vs terminal fates {}). \
         Either the census and the E-step have drifted apart, or a candidate produced a \
         non-finite posterior (a hand-built emission parquet with sigma2 <= 0 will do it). \
         NOT lost reads -- `min_weight` culling is accounted for. \
         The profile at {:?} was written in full and is intact; keep it and use it to find \
         the unaccounted class. Per-bucket ledger: {:?}",
        ledger.deficit(),
        ledger.total_candidates,
        ledger.terminal_sum(),
        output_profile_path.as_ref(),
        ledger,
    );
    anyhow::ensure!(
        ledger.census_vs_estep_deficit() == 0,
        "candidate census disagrees with the E-step by {} candidates: the census replayed \
         {} live candidates but the E-step carried {} into `weights`. These are two \
         implementations of one predicate (max_as, the saturating_sub, the strict `>`) and \
         they have drifted apart. Do NOT loosen this assert -- the census is a reimplementation \
         and the drift, not the assert, is the bug. NOTE: algebraically the same condition \
         `deficit()` checks, kept because it names the cause. Ledger: {:?}",
        ledger.census_vs_estep_deficit(),
        ledger.live_candidates,
        ledger.live_candidates_entered,
        ledger,
    );

    if let Some(out) = output_emission_params.as_deref_mut() {
        *out = std::mem::take(&mut last_estep_params);
    }

    if let Some(out) = output_read_assignments {
        // `flatten()` drops the `None` slots. Every molecule is visited by exactly one
        // of the two loops above, and what survives obeys the weaker, one-directional
        // invariant the cross-run join needs: every emitted assignment either HAS a
        // profile row, or is a ghost with `top_tx_id = None`.
        *out = read_assign.into_iter().flatten().collect();
    }

    Ok(TelEmStats {
        n_unambiguous,
        n_ambiguous,
        n_em_iter,
        ledger,
        n_unambiguous_dropped_incompetent,
        n_unambiguous_dropped_missing_emission,
        n_ghost_read_all_missing,
        n_ghost_read_all_incompetent,
        n_ghost_read_all_alpha_zero,
        n_dropped_output_row_competence,
        n_dropped_no_competent_candidates,
        n_entropy_clean,
        n_entropy_partial,
        n_entropy_degenerate,
        n_degenerate_all_imputed,
        n_degenerate_with_anchored,
        n_polya_zero_power,
        n_polya_zero_power_resolved,
        n_reads_multi_emission_candidate,
        n_reads_any_emission_candidate,
        n_reads_single_emission_candidate,
        em_trace: output_em_trace.map(|t| t.clone()).unwrap_or_default(),
        max_dw_phase_abs,
        max_dw_phase_rel,
    })
}

fn poly_a_length_from_log2(log_pal: f32) -> i32 {
    log_pal.exp2().round() as i32
}

/// DEGRADATION TELEMETRY. Returns `(n_with_emission, all_identical)` for the
/// read's candidate set. `all_identical` is true iff every candidate that carries
/// an emission row shares one identical `(mu, sigma2)` pair, so the poly(A) term
/// cancels exactly in every pairwise odds ratio.
///
/// The comparison is EXACT `==`, NOT A TOLERANCE: the shared fits are
/// bit-identical copies by construction. A candidate with NO emission row is skipped
/// rather than treated as a mismatch: a ghost transcript has no `(mu, sigma2)` to
/// disagree with. No emission row at all returns `(0, false)`; exactly one returns
/// `(1, true)`. The read's FULL candidate range is used.
fn polya_emission_count_and_identity(
    graph: &FlatEvidenceGraph,
    mol: &MoleculeEvidence,
    params: &[Option<TranscriptParam>],
) -> (usize, bool) {
    let off = mol.candidate_offset as usize;
    let end = off + mol.candidate_count as usize;
    let mut first: Option<(f64, f64)> = None;
    let mut n_with_emission = 0usize;
    let mut disagrees = false;
    for &tx in &graph.candidate_pool[off..end] {
        let Some(p) = params.get(tx as usize).and_then(|o| o.as_ref()) else {
            continue;
        };
        n_with_emission += 1;
        match first {
            None => first = Some((p.mu, p.sigma2)),
            Some((mu0, s20)) if mu0 == p.mu && s20 == p.sigma2 => {}
            Some(_) => disagrees = true,
        }
    }
    (n_with_emission, first.is_some() && !disagrees)
}

/// How many candidates must carry an emission row before "they all share one `mu`"
/// is a MEASURED statement rather than a comparison that never happened.
const ZERO_POWER_MIN_EMISSION_CANDIDATES: usize = 2;

/// Per-read observational statistic: `second_largest_AS / largest_AS` over the read's
/// FULL candidate range, emitted on every row of that read.
///
/// Three defined cases, all yielding `0.0`: `candidate_count == 1` (no second
/// candidate); `largest_AS <= 0` (no candidate carries an `AS:i` tag, so `0.0`
/// means "no alignment evidence", not "a tie"); and `second_AS < 0` (`AS` is a raw
/// `i32` and a malformed BAM can carry a negative value).
///
/// Otherwise `second_largest / largest`, in `[0, 1]`, computed in `f64`. NOT USED BY
/// THE SOLVER.
fn read_max_as_ratio(graph: &FlatEvidenceGraph, mol: &MoleculeEvidence) -> f32 {
    if mol.candidate_count < 2 {
        return 0.0;
    }
    let o = mol.candidate_offset as usize;
    let n = mol.candidate_count as usize;
    let slice = &graph.candidate_as[o..o + n];

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

/// Why a read's candidate set contains nothing that could carry posterior mass.
///
/// The census is taken over the read's ORIGINAL candidate range, and each bucket is
/// defined by the E-step's own decision procedure: a read is a ghost exactly when
/// the E-step's `max_ll` came out `-inf`.
///
/// The two ways a candidate can be absent from the E-step's `raw` vector must NOT
/// share a bucket: **pruned** (`delta_as > prune_max_delta_as`, pushed as `-inf`, so
/// the entry IS in `raw`) and **no emission row** (the `if let Some(ref p) =
/// params[tx_id]` arm has no `else`, so the entry is never pushed).
struct CandidateAvailability {
    n_missing: usize,
    n_incompetent: usize,
    /// Present and competent, but `alpha <= 0.0` — pushed as `-inf`. The M-step can
    /// drive `alpha` to exactly 0 without touching `is_competent`.
    n_alpha_zero: usize,
    /// Killed by the E-step's `ΔAS` prune — `delta_as > prune_max_delta_as`.
    ///
    /// This bucket is what makes the candidate ledger closable. It is NOT consulted
    /// by [`CandidateAvailability::uniform_cause`]: a pruned candidate is dead for a
    /// reason no amount of emission padding addresses.
    n_pruned_delta_as: usize,
    /// Carried posterior mass. A ghost read has `n_live == 0`. NOT a terminal fate:
    /// live candidates go on to be split by `min_weight` and by the output stage's
    /// own competence recheck, so this is EXCLUDED from [`CandidateLedger::deficit`].
    n_live: usize,
}

impl CandidateAvailability {
    /// The single cause shared by EVERY candidate, when exactly one cause accounts for
    /// all of them; `None` when they died for a mix of reasons, or one was pruned.
    fn uniform_cause(&self, candidate_count: usize) -> Option<GhostCause> {
        if candidate_count == 0 {
            return None;
        }
        if self.n_missing == candidate_count {
            Some(GhostCause::Missing)
        } else if self.n_incompetent == candidate_count {
            Some(GhostCause::Incompetent)
        } else if self.n_alpha_zero == candidate_count {
            Some(GhostCause::AlphaZero)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GhostCause {
    Missing,
    Incompetent,
    AlphaZero,
}

/// One walk of a read's candidate range, classifying each candidate by the
/// E-step's own filter. `params` must be `last_estep_params`, NOT `params`: the
/// census explains a `final_weights` produced from the pre-M-step model, and
/// `alpha` is the field the M-step rewrites.
fn candidate_availability(
    graph: &FlatEvidenceGraph,
    mol: &MoleculeEvidence,
    params: &[Option<TranscriptParam>],
    prune_max_delta_as: i32,
) -> CandidateAvailability {
    let off = mol.candidate_offset as usize;
    let n = mol.candidate_count as usize;
    let mut av = CandidateAvailability {
        n_missing: 0,
        n_incompetent: 0,
        n_alpha_zero: 0,
        n_pruned_delta_as: 0,
        n_live: 0,
    };
    // Same reference point as the E-step: the read's best `AS` over its FULL
    // candidate range, taken before competence masking.
    let max_as = graph.candidate_as[off..off + n]
        .iter()
        .fold(i32::MIN, |a, &b| a.max(b));
    for c in 0..n {
        let idx = off + c;
        let tx_id = graph.candidate_pool[idx] as usize;
        let delta_as = max_as.saturating_sub(graph.candidate_as[idx]);
        if delta_as > prune_max_delta_as {
            // Counted, but not attributed: the E-step pushes this candidate as
            // `-inf`, so without this increment it is a candidate no bucket claims.
            av.n_pruned_delta_as += 1;
            continue;
        }
        let Some(p) = params.get(tx_id).and_then(|o| o.as_ref()) else {
            av.n_missing += 1;
            continue;
        };
        if !p.is_competent {
            av.n_incompetent += 1;
        } else if p.alpha <= 0.0 {
            av.n_alpha_zero += 1;
        } else {
            av.n_live += 1;
        }
    }
    av
}

/// The candidate conservation ledger: every candidate the evidence graph
/// produced is accounted for by exactly one terminal fate.
///
/// WHAT "TERMINAL FATE" MEANS. A candidate is counted exactly once, at the point
/// where it stops moving. The E-step census splits the contested read's candidates
/// four ways, and the survivor bucket is split three ways at the output stage --
/// so the survivors are counted there and NOT here. `live_candidates` is recorded
/// because [`CandidateLedger::census_vs_estep_deficit`] needs it, but
/// [`CandidateLedger::deficit`] deliberately does not sum it.
///
/// SCOPE. `total_candidates` counts every candidate of every molecule with
/// `candidate_count >= 1`, including those on reads that emitted nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CandidateLedger {
    pub total_candidates: u64,

    pub pruned_delta_as: u64,
    pub missing: u64,
    pub incompetent: u64,
    pub alpha_zero: u64,
    pub live_candidates: u64,

    /// Live candidates that were culled by `min_weight` AND not emitted. The argmax
    /// of an all-culled read is culled-but-emitted, so it is NOT counted here; see
    /// `reads_retained_by_argmax`.
    ///
    /// Not a constant: `min_weight` is a caller-supplied parameter, and at
    /// `min_weight = 0.0` this is provably exactly 0.
    pub pruned_min_weight: u64,
    /// Reads whose every live candidate fell below `min_weight`, and whose argmax
    /// was therefore retained and emitted with `posterior_weight = 1.0` anyway.
    ///
    /// THIS COUNTS READS, NOT CANDIDATES, so it must NOT appear in `terminal_sum()`.
    pub reads_retained_by_argmax: u64,
    pub dropped_output_competence: u64,
    pub rows_emitted: u64,

    pub unambig_incompetent: u64,
    pub unambig_missing: u64,
    pub unambig_emitted: u64,

    pub live_candidates_entered: u64,
}

impl CandidateLedger {
    fn terminal_sum(&self) -> u64 {
        self.pruned_delta_as
            + self.missing
            + self.incompetent
            + self.alpha_zero
            + self.pruned_min_weight
            + self.dropped_output_competence
            + self.rows_emitted
            + self.unambig_incompetent
            + self.unambig_missing
            + self.unambig_emitted
    }

    /// `total_candidates` minus the sum of the terminal fates.
    ///
    /// **Must be exactly 0.** A non-zero value is a real instrumentation defect.
    pub fn deficit(&self) -> i64 {
        self.total_candidates as i64 - self.terminal_sum() as i64
    }

    /// `live_candidates` minus `live_candidates_entered`.
    ///
    /// **NOT an independent check on `deficit()`.** It is algebraically the same
    /// condition; when it fires, it says *which* quantity disagreed.
    pub fn census_vs_estep_deficit(&self) -> i64 {
        self.live_candidates as i64 - self.live_candidates_entered as i64
    }
}

fn posterior_entropy(weights: &[(usize, f64)]) -> f64 {
    weights
        .iter()
        .map(|&(_, w)| if w > 0.0 { -w * w.ln() } else { 0.0 })
        .sum()
}

/// Per-read posterior summary, for callers that need read-level outcomes the per
/// (read, candidate) profile cannot express (top-1 switch rate, entropy shift).
///
/// Three decisions here are load-bearing:
/// - `top_tx_id` is the transcript NAME, not the pool index: the index is only
///   meaningful relative to one run's `transcript_names` order, and this table is
///   joined against a SECOND run's table.
/// - `n_candidates_before_cutoff` is `MoleculeEvidence::candidate_count`, i.e. the
///   fan-out BEFORE the `Delta-AS` prune, the `is_competent` filter and the
///   `min_weight` truncation.
/// - `top_tx_id` is `None` for a GHOST read -- no competent candidate, empty
///   `weights`. The ghost has no profile row either, so
///
///   ```text
///   top_tx_id == Some(..)  IMPLIES  that read has a profile row.
///   ```
#[derive(Debug, Clone)]
pub struct ReadAssignment {
    pub read_id: String,
    pub n_candidates_before_cutoff: usize,
    pub top_tx_id: Option<String>,
    pub top_weight: f64,
    pub entropy: f64,
}

/// Write the per-read posterior summary as Parquet, matching every other output
/// this tool produces (profile, emission, anchors, summary are all Parquet).
///
/// `top_tx_id` is written as a nullable string rather than an index: a pool index
/// only means something relative to one run's `transcript_names`.
fn accumulate_mstep_suffstats(
    params: &[Option<TranscriptParam>],
    graph: &FlatEvidenceGraph,
    unambiguous_indices: &[usize],
    ambiguous_indices: &[usize],
    final_weights: &[Vec<(usize, f64)>],
    num_transcripts: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut accum_w = vec![0.0_f64; num_transcripts];
    let mut accum_w_y = vec![0.0_f64; num_transcripts]; // S1
    let mut accum_w_y2 = vec![0.0_f64; num_transcripts]; // S2

    for &idx in unambiguous_indices {
        let mol = &graph.molecules[idx];
        let tx_id = graph.candidate_pool[mol.candidate_offset as usize] as usize;
        let y = mol.log_pal as f64;
        if params[tx_id].is_some() {
            accum_w[tx_id] += 1.0;
            accum_w_y[tx_id] += y;
            accum_w_y2[tx_id] += y * y;
        }
    }

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
    (accum_w, accum_w_y, accum_w_y2)
}

pub fn write_read_assignments_parquet(
    rows: &[ReadAssignment],
    path: &std::path::Path,
) -> Result<()> {
    use std::sync::Arc;

    use arrow::array::{Float64Builder, Int64Builder, StringBuilder};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let mut read_b = StringBuilder::new();
    let mut ncand_b = Int64Builder::new();
    let mut top_b = StringBuilder::new();
    let mut w_b = Float64Builder::new();
    let mut h_b = Float64Builder::new();
    for r in rows {
        read_b.append_value(&r.read_id);
        ncand_b.append_value(r.n_candidates_before_cutoff as i64);
        top_b.append_option(r.top_tx_id.as_deref());
        w_b.append_value(r.top_weight);
        h_b.append_value(r.entropy);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("read_id", DataType::Utf8, false),
        Field::new("n_candidates_before_cutoff", DataType::Int64, false),
        Field::new("top_tx_id", DataType::Utf8, true),
        Field::new("top_weight", DataType::Float64, false),
        Field::new("entropy", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(read_b.finish()),
            Arc::new(ncand_b.finish()),
            Arc::new(top_b.finish()),
            Arc::new(w_b.finish()),
            Arc::new(h_b.finish()),
        ],
    )
    .context("Failed to build read-assignments RecordBatch")?;

    let f = std::fs::File::create(path).with_context(|| format!("Failed to create {:?}", path))?;
    let mut w = ArrowWriter::try_new(f, schema, Some(WriterProperties::builder().build()))
        .context("Failed to create ArrowWriter for read assignments")?;
    w.write(&batch)
        .context("Failed to write read-assignments batch")?;
    w.close()
        .context("Failed to close read-assignments writer")?;
    Ok(())
}

/// Write the `(mu, sigma2)` the last E-step consumed, as a DIAGNOSTIC sidecar.
///
/// The first five columns are named and typed to match the emission parquet, so the
/// result loads back through `load_emission_parameters` unchanged.
///
/// The remaining columns exist so the file can be audited: `mu_cohort` (the
/// shrinkage prior, so a reader can see that re-loading RE-ANCHORS the prior -- the
/// loader sets `mu_cohort = mu_logPAL` AND `sigma2_cohort = sigma2_logPAL`, so BOTH
/// priors move to the fitted values and tau_0 = 2 / 2*beta_0 are applied a second
/// time); `alpha` (the mixture weight the last E-step used); `state` (which iterate
/// this row holds); `em_accum_weight` (the E-step weight sum W, in molecule weight
/// units); `mu_fitted` / `sigma2_fitted` (ALIASES of `mu_logPAL` / `sigma2_logPAL`,
/// same values, same iteration); and `em_finalized` (whether an M-step ran).
pub fn write_emission_params_parquet(
    params: &[Option<TranscriptParam>],
    path: &std::path::Path,
) -> Result<()> {
    use arrow::array::BooleanBuilder;

    let mut tr_b = StringBuilder::new();
    let mut gene_b = StringBuilder::new();
    let mut comp_b = BooleanBuilder::new();
    let mut mu_b = Float64Builder::new();
    let mut s2_b = Float64Builder::new();
    let mut cohort_b = Float64Builder::new();
    let mut alpha_b = Float64Builder::new();
    let mut state_b = StringBuilder::new();
    let mut accum_w_b = Float64Builder::new();
    let mut finalized_b = BooleanBuilder::new();
    let mut mu_fit_b = Float64Builder::new();
    let mut s2_fit_b = Float64Builder::new();

    // A transcript with NO emission row is `None` and gets NO row here, exactly
    // as in the emission parquet.
    for p in params.iter().flatten() {
        tr_b.append_value(&p.transcript_id);
        gene_b.append_value(&p.gene_id);
        comp_b.append_value(p.is_competent);
        mu_b.append_value(p.mu);
        s2_b.append_value(p.sigma2);
        cohort_b.append_value(p.mu_cohort);
        alpha_b.append_value(p.alpha);
        state_b.append_value("e_step_final");
        accum_w_b.append_value(p.em_accum_weight);
        finalized_b.append_value(p.em_finalized);
        mu_fit_b.append_value(p.mu);
        s2_fit_b.append_value(p.sigma2);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("transcript_id", DataType::Utf8, false),
        Field::new("gene_id", DataType::Utf8, false),
        Field::new("is_polyA_competent", DataType::Boolean, false),
        Field::new("mu_logPAL", DataType::Float64, false),
        Field::new("sigma2_logPAL", DataType::Float64, false),
        Field::new("mu_cohort", DataType::Float64, false),
        Field::new("alpha", DataType::Float64, false),
        Field::new("state", DataType::Utf8, false),
        Field::new("em_accum_weight", DataType::Float64, false),
        Field::new("em_finalized", DataType::Boolean, false),
        Field::new("mu_fitted", DataType::Float64, false),
        Field::new("sigma2_fitted", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(tr_b.finish()),
            Arc::new(gene_b.finish()),
            Arc::new(comp_b.finish()),
            Arc::new(mu_b.finish()),
            Arc::new(s2_b.finish()),
            Arc::new(cohort_b.finish()),
            Arc::new(alpha_b.finish()),
            Arc::new(state_b.finish()),
            Arc::new(accum_w_b.finish()),
            Arc::new(finalized_b.finish()),
            Arc::new(mu_fit_b.finish()),
            Arc::new(s2_fit_b.finish()),
        ],
    )
    .context("Failed to build emission-params RecordBatch")?;

    let f = std::fs::File::create(path).with_context(|| format!("Failed to create {:?}", path))?;
    let mut w = ArrowWriter::try_new(f, schema, Some(WriterProperties::builder().build()))
        .context("Failed to create ArrowWriter for emission params")?;
    w.write(&batch)
        .context("Failed to write emission-params batch")?;
    w.close()
        .context("Failed to close emission-params writer")?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct TelEmStats {
    /// Candidate conservation ledger. `deficit()` is asserted to be 0 inside
    /// `solve_and_write_profile` before this is ever returned.
    pub ledger: CandidateLedger,
    pub n_unambiguous: usize,
    pub n_ambiguous: usize,
    pub n_em_iter: usize,
    /// READ-LEVEL, unambiguous population. Denominator `n_unambiguous`. An
    /// unambiguous read has one candidate: present and `is_polyA_competent == false`.
    pub n_unambiguous_dropped_incompetent: u64,
    /// READ-LEVEL, unambiguous population. Denominator `n_unambiguous`.
    /// The read's single candidate has no emission row at all. There is deliberately
    /// NO unambiguous counterpart to `n_ghost_read_all_alpha_zero`.
    pub n_unambiguous_dropped_missing_emission: u64,
    pub n_ghost_read_all_missing: u64,
    pub n_ghost_read_all_incompetent: u64,
    /// READ-LEVEL, ghost population. Denominator
    /// `n_dropped_no_competent_candidates`. Contested reads with no posterior
    /// whose entire candidate range was present and competent but had
    /// `alpha <= 0.0` -- the abundance prior the M-step drove to zero.
    ///
    /// A genuinely distinct cause from the other two: `is_competent` is
    /// M-step-invariant while `alpha` is exactly the field the M-step rewrites.
    pub n_ghost_read_all_alpha_zero: u64,
    /// ROW-LEVEL, and **not** comparable with any of the read-level counters above.
    /// Incremented once per (read × candidate) row when a row's transcript is
    /// `!is_competent` or absent from `params`. It is a TRIPWIRE: if the E-step's
    /// `-inf` mask is ever removed, this fires.
    pub n_dropped_output_row_competence: u64,
    /// READ-LEVEL, ghost population. Contested reads (>= 2 candidates) with NO
    /// posterior at all -- `weights.is_empty()`. This is the TOTAL for the
    /// decomposition above.
    ///
    /// `n_ghost_read_all_missing + n_ghost_read_all_incompetent +
    /// n_ghost_read_all_alpha_zero` is a PARTIAL decomposition of it.
    pub n_dropped_no_competent_candidates: u64,
    pub n_entropy_clean: u64,
    pub n_entropy_partial: u64,
    /// degenerate: H > 0.70 nats (effectively uniform over candidates).
    /// The upper bounds are 0.20 / 0.70, NOT 0.20 / 0.60: re-derive these tiers
    /// from the comparison in the E-step.
    pub n_entropy_degenerate: u64,
    /// Of the `n_entropy_degenerate` reads, the ones whose ENTIRE surviving
    /// candidate set is IMPUTED (`n_cohort_flnc == 0`, carrying the shared cohort
    /// fit). Unit: READS. Denominator: `n_entropy_degenerate`.
    ///
    /// Those reads are flat for a MECHANISM that is not a genuine tie: every
    /// candidate shares one identical density.
    pub n_degenerate_all_imputed: u64,
    /// `n_degenerate_all_imputed`'s complement inside `n_entropy_degenerate`:
    /// reads that are degenerate with at least one ANCHORED candidate in the set.
    /// Unit: READS. Denominator: `n_entropy_degenerate`.
    ///
    /// `n_degenerate_all_imputed + n_degenerate_with_anchored` must equal
    /// `n_entropy_degenerate` EXACTLY.
    pub n_degenerate_with_anchored: u64,
    /// DEGRADATION TELEMETRY. Contested reads (>= 2 candidates) whose candidate set
    /// carries ZERO poly(A) discriminative power: every candidate that has an
    /// emission row shares one identical `(mu, sigma2)` pair, so the poly(A) term
    /// cancels EXACTLY in every pairwise odds ratio.
    ///
    /// Read `n_polya_zero_power / n_ambiguous` for the fraction. The denominator is
    /// deliberately `n_ambiguous` and not `n_total`.
    ///
    /// THE COMPARISON IS EXACT `==`, NOT A TOLERANCE: candidates that disagree by as
    /// little as one ULP are DISCRIMINATING, so the number UNDER-reports zero
    /// poly(A) power.
    pub n_polya_zero_power: u64,
    /// DEGRADATION TELEMETRY, second metric. Contested reads carrying at least
    /// `ZERO_POWER_MIN_EMISSION_CANDIDATES` emission rows in which all of them
    /// share one identical `(mu, sigma2)`. Unlike `n_polya_zero_power` this cannot
    /// be satisfied by a read that had nothing to compare.
    ///
    /// Read `n_polya_zero_power_resolved / n_reads_multi_emission_candidate`. Do
    /// NOT divide by `n_ambiguous`.
    pub n_polya_zero_power_resolved: u64,
    /// DENOMINATOR for `n_polya_zero_power_resolved`: contested reads with at
    /// least `ZERO_POWER_MIN_EMISSION_CANDIDATES` candidates carrying an
    /// emission row. Can legitimately be 0 on a sparse model.
    pub n_reads_multi_emission_candidate: u64,
    /// NUMERATOR for emission coverage. Counts over ALL reads with at least one
    /// candidate, not just contested ones. Divide by `n_unambiguous + n_ambiguous`
    /// and label the denominator "reads with >=1 candidate".
    pub n_reads_any_emission_candidate: u64,
    /// NUMERATOR for the single-candidate share: reads with exactly ONE candidate
    /// carrying an emission row. Same denominator as
    /// `n_reads_any_emission_candidate`.
    pub n_reads_single_emission_candidate: u64,
    pub em_trace: Vec<(usize, f64, f64)>,
    /// Largest absolute disagreement, over all transcripts, between the W the last
    /// M-step CONSUMED and the W the sidecar PUBLISHES as `em_accum_weight`. Unit:
    /// molecules. Denominator: none -- this is a max, not a rate. `0.0` when no
    /// M-step ran.
    ///
    /// Non-zero means the run's published W is one E-step behind the W that
    /// produced the published `sigma2`.
    pub max_dw_phase_abs: f64,
    /// The same disagreement divided, per transcript, by the larger of the
    /// two arms; `0.0` when no M-step ran or when a transcript is zero in both.
    /// Dimensionless.
    pub max_dw_phase_rel: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    use arrow::array::{Array, BooleanBuilder, Int64Array, UInt32Builder};

    use crate::evidence::MoleculeEvidence;

    /// The M-step's three prior constants are pinned BY VALUE -- exact equality,
    /// not approximate: these are hyper-parameters, not measurements.
    #[test]
    fn test_m_step_prior_constants_are_pinned() {
        assert_eq!(
            NU_0, 1.0,
            "NU_0 changed; the InvGamma shape enters the sigma2 denominator"
        );
        assert_eq!(
            BETA_0, 1.0,
            "BETA_0 changed; it multiplies sigma2_cohort in the sigma2 numerator"
        );
        assert_eq!(
            TAU_0, 2.0,
            "TAU_0 changed; it is the pseudo-count damping on the mu update \
             AND the kappa_0 of the mu|v conditional prior in the sigma2 \
             numerator. Changing it moves every shrinkage in the model."
        );
        assert_eq!(
            INSTRUMENTAL_PAL_NOISE_FLOOR, 0.04,
            "the sigma2 floor changed; it silently truncates every small-variance \
             transcript and the oracle ranks models against it"
        );
        assert_eq!(
            SIGMA2_DOF_OFFSET, 2.0,
            "SIGMA2_DOF_OFFSET changed; it is the trailing constant of the sigma2 \
             denominator and 2.0 IS Model A, which drops the \
             numerator term AND the -0.5 ln v normaliser together, so +2 and \
             +3 are not the same model with a knob turned. The knob exists for \
             the simulation harness; moving it moves the shipped estimator."
        );
    }

    /// The "only naming site" invariant, pinned with a red proof.
    ///
    /// Counts naming sites of `SIGMA2_DOF_OFFSET` in PRODUCTION code and pins the
    /// count at 2 -- the declaration and the one call. Zero means a literal crept
    /// into the shipped path; three or more means a second path can supply `c`.
    #[test]
    fn test_the_shipped_entry_point_is_the_only_naming_site_of_the_dof_offset() {
        let source = include_str!("em.rs");
        let cut = source
            .rfind("#[cfg(test)]\nmod tests {")
            .expect("em.rs has exactly one #[cfg(test)] mod tests block");
        let production = &source[..cut];

        let mut code = String::new();
        for line in production.lines() {
            // A `//` reached before any quote is a comment. KNOWN LIMIT: this is not a
            // Rust lexer, so a line with `//` inside a string literal would be truncated.
            let head = match line.find("//") {
                Some(at) if !line[..at].contains('"') => &line[..at],
                _ => line,
            };
            code.push_str(head);
            code.push('\n');
        }

        let sites = code.matches("SIGMA2_DOF_OFFSET").count();
        assert_eq!(
            sites, 2,
            "expected exactly 2 naming sites in production code -- the `pub const` \
             and the single call in `solve_and_write_profile` -- but found {sites}. \
             1 means the shipped path passes a literal and no longer reads the \
             declared constant. Comments are excluded, so a doc mention cannot \
             inflate this count. NOTE: a count of 2 alone does NOT rule out a \
             third path supplying its own `c` from a literal -- that is what the \
             call-site assertion below is for."
        );

        let callers = code.matches("solve_and_write_profile_impl(").count();
        assert_eq!(
            callers, 2,
            "expected exactly 2 callers of solve_and_write_profile_impl -- the \
             shipped forwarder and the parameterised entry point -- but found \
             {callers}. A third caller can supply a literal `c` without ever \
             naming SIGMA2_DOF_OFFSET, which is invisible to the count above."
        );
    }

    const MIN_WEIGHT: f64 = 0.01;

    const FIVE_PRIME_WEIGHT: f64 = 1.8;
    const FIVE_PRIME_WINDOW: u32 = 20;
    const PRUNE_MAX_DELTA_AS: i32 = 50;
    /// The 5' term switched off. `ln_5p == 0` exactly, so any test passing this
    /// is provably independent of `candidate_5p_gap`.
    const POLY_A_ON: f64 = 1.0;
    const FIVE_PRIME_OFF: f64 = 0.0;
    /// The `ΔAS` prune switched off. Legal (`i32::MAX` is a valid threshold).
    const PRUNE_OFF: i32 = i32::MAX;

    fn synth_graph_multi(mols: &[(f32, Vec<(&str, i32, u32)>)]) -> FlatEvidenceGraph {
        let mut molecules = Vec::new();
        let (mut candidate_pool, mut candidate_as, mut candidate_5p_gap) =
            (Vec::new(), Vec::new(), Vec::new());
        let (mut candidate_nm, mut candidate_mapq) = (Vec::new(), Vec::new());
        let mut transcript_names: Vec<String> = Vec::new();
        for (i, (log_pal, cands)) in mols.iter().enumerate() {
            let offset = candidate_pool.len() as u32;
            for &(tx, as_, gap) in cands {
                let id = match transcript_names.iter().position(|n| n == tx) {
                    Some(p) => p,
                    None => {
                        transcript_names.push(tx.to_string());
                        transcript_names.len() - 1
                    }
                };
                candidate_pool.push(id as u32);
                candidate_as.push(as_);
                candidate_5p_gap.push(gap);
                candidate_nm.push(0);
                candidate_mapq.push(60);
            }
            molecules.push(MoleculeEvidence {
                read_id: format!("r{}", i + 1),
                log_pal: *log_pal,
                candidate_offset: offset,
                candidate_count: cands.len() as u16,
            });
        }
        FlatEvidenceGraph {
            molecules,
            candidate_pool,
            candidate_as,
            candidate_nm,
            candidate_mapq,
            candidate_5p_gap,
            transcript_names,
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        }
    }

    /// Joint log-posterior of MODEL B -- the NIG prior.
    ///
    /// NOT A DESCRIPTION OF PRODUCTION: the M-step is Model A.
    ///
    ///     Q = -(W/2) ln(2 pi v) - S(mu)/(2v) - (nu_0+1) ln v - beta_0/v
    ///         - 0.5 ln v - kappa_0 (mu - mu_cohort)^2 / (2v)
    ///
    /// The `-0.5 ln v` normaliser of `mu | v ~ N(mu_cohort, v/kappa_0)` is what
    /// puts the `+1` into the sigma^2 denominator.
    fn nig_joint_logpost(
        mu: f64,
        v: f64,
        w: f64,
        s1: f64,
        s2: f64,
        mu_cohort: f64,
        kappa_0: f64,
        nu_0: f64,
        beta_0: f64,
        sigma2_cohort: f64,
    ) -> f64 {
        let s_mu = s2 - 2.0 * mu * s1 + mu * mu * w;
        -(w / 2.0) * (2.0 * std::f64::consts::PI * v).ln() - s_mu / (2.0 * v)
            - (nu_0 + 1.0) * v.ln()
            // The Inverse-Gamma SCALE is `beta_0 * sigma2_cohort`, not `beta_0`:
            // InvGamma(a, b) has density v^{-(a+1)} exp(-b/v), and `em.rs` puts
            // `beta_0 * sigma2_cohort` in the exponent.
            - beta_0 * sigma2_cohort / v
            - 0.5 * v.ln()
            - kappa_0 * (mu - mu_cohort).powi(2) / (2.0 * v)
    }

    fn golden_max<F: Fn(f64) -> f64>(f: F, mut lo: f64, mut hi: f64, iters: usize) -> f64 {
        let inv_phi = 0.618_033_988_749_894_9_f64;
        for _ in 0..iters {
            let a = hi - (hi - lo) * inv_phi;
            let b = lo + (hi - lo) * inv_phi;
            if f(a) < f(b) {
                lo = a;
            } else {
                hi = b;
            }
        }
        0.5 * (lo + hi)
    }

    /// THE M-STEP IS MODEL A, AND THIS IS THE TEST THAT PROVES IT.
    ///   (a) production is strictly below Model B's numeric argmax;
    ///   (b) production equals Model A's closed form, recomputed
    ///       here from the accumulators with no shared code.
    #[test]
    fn test_mstep_is_model_a_and_is_not_model_b() {
        // NU_0 / BETA_0 / TAU_0 are deliberately NOT restated here. This test is
        // named for production Model A, so it must read production's own
        // constants -- a local transcription guarantees about the test, not the crate.
        const MU_COHORT: f64 = 8.5; // deliberately NOT near the observations
        const SIGMA2_COHORT: f64 = 0.5;
        const FLOOR: f64 = crate::emission_model::INSTRUMENTAL_PAL_NOISE_FLOOR;

        let lens: [f32; 3] = [64.0, 100.0, 256.0];
        let mut mols: Vec<(f32, Vec<(&str, i32, u32)>)> = lens
            .iter()
            .map(|&l| (l.log2(), vec![("tx0", 1000, 0u32)]))
            .collect();
        mols.push((256.0f32.log2(), vec![("tx1", 1000, 0), ("tx2", 1000, 0)]));

        let ys: Vec<f64> = lens.iter().map(|&l| l.log2() as f64).collect();
        let w: f64 = ys.len() as f64;
        let s1: f64 = ys.iter().sum();
        let s2: f64 = ys.iter().map(|&y| y * y).sum();

        assert!(
            w * s2 - s1 * s1 > 0.0,
            "fixture must have real data variance: W*S2 - S1^2 = {}",
            w * s2 - s1 * s1
        );

        let emission = scratch("mstep_objective_emission.parquet");
        let profile = scratch("mstep_objective_profile.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", MU_COHORT, SIGMA2_COHORT),
                ("tx1", "gene1", 7.0, SIGMA2_COHORT),
                ("tx2", "gene2", 7.0, SIGMA2_COHORT),
            ],
        );

        let mut params = Vec::new();
        solve_and_write_profile(
            synth_graph_multi(&mols),
            &emission,
            "TEST",
            &profile,
            1e-9,
            3,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            Some(&mut params),
            None,
        )
        .unwrap();

        let p = params[0].as_ref().expect("tx0 must have params");
        let (mu_prod, v_prod) = (p.mu, p.sigma2);

        assert!(
            (mu_prod - MU_COHORT).abs() > 1e-6,
            "the M-step did not run: mu is still the emission value {MU_COHORT}. \
             The fixture must supply at least one AMBIGUOUS molecule, or the \
             convergence check breaks before the M-step is ever reached."
        );
        assert!(
            (v_prod - SIGMA2_COHORT).abs() > 1e-9,
            "the M-step did not run: sigma2 is still the emission value \
             {SIGMA2_COHORT}."
        );

        assert!(
            v_prod > FLOOR * 10.0,
            "sigma2 = {v_prod} is near INSTRUMENTAL_PAL_NOISE_FLOOR ({FLOOR}); the \
             floor is clamping and this fixture no longer tests the closed form."
        );

        let q = |mu: f64, v: f64| {
            nig_joint_logpost(
                mu,
                v,
                w,
                s1,
                s2,
                MU_COHORT,
                TAU_0,
                NU_0,
                BETA_0,
                SIGMA2_COHORT,
            )
        };
        let best_v_for = |mu: f64| golden_max(|v| q(mu, v), 1e-9, 1e2, 300);
        let mu_star = golden_max(
            |mu| {
                let v = best_v_for(mu);
                q(mu, v)
            },
            ys.iter().cloned().fold(f64::INFINITY, f64::min) - 1.0,
            ys.iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                .max(MU_COHORT)
                + 1.0,
            300,
        );
        let v_star = best_v_for(mu_star);

        let q_prod = q(mu_prod, v_prod);
        let q_best = q(mu_star, v_star);
        assert!(
            q_prod < q_best - 1e-6,
            "production is still sitting at Model B's joint-MAP optimum.\n  \
             production (mu, sigma2) = ({mu_prod:.17}, {v_prod:.17})\n  \
             Model B argmax (mu, sigma2) = ({mu_star:.17}, {v_star:.17})\n  \
             Q_B(production) - Q_B(argmax) = {:.3e}, which is not the expected \
             NEGATIVE gap.\n  \
             The production M-step is Model A: numerator \
             without the kappa_0*(mu-mu_cohort)^2 term, denominator +2 not +3. \
             If this fires, Model B is back.",
            q_prod - q_best
        );

        let mu_indep = (s1 + TAU_0 * MU_COHORT) / (w + TAU_0);
        let s_mu_indep = s2 - 2.0 * mu_indep * s1 + mu_indep * mu_indep * w;
        let sigma2_model_a =
            ((s_mu_indep + 2.0 * BETA_0 * SIGMA2_COHORT) / (w + 2.0 * NU_0 + 2.0)).max(FLOOR);
        assert!(
            (mu_prod - mu_indep).abs() < 1e-9,
            "production mu {:.17} != the independent closed form {:.17}.",
            mu_prod,
            mu_indep
        );
        assert!(
            (v_prod - sigma2_model_a).abs() <= 1e-12 * sigma2_model_a.max(1.0),
            "production sigma2 {:.17} != Model A's independent closed form \
             {:.17}.\n  production = (S(mu) + 2*beta_0*sigma2_cohort) / \
             (W + 2*nu_0 + 2), floored.",
            v_prod,
            sigma2_model_a
        );

        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);
    }

    #[test]
    fn test_poly_a_length_round_trip() {
        assert_eq!(poly_a_length_from_log2(6.9069), 120);
        assert_eq!(poly_a_length_from_log2(7.0), 128);
    }

    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("telescope_em_{}_{}", name, std::process::id()))
    }

    fn local_scratch(name: &str) -> PathBuf {
        PathBuf::from("/tmp").join(format!("telescope_em_{}_{}", name, std::process::id()))
    }

    fn write_test_emission(path: &Path, rows: &[(&str, &str, f64, f64)]) {
        let all_competent: Vec<(&str, &str, f64, f64, bool)> = rows
            .iter()
            .map(|&(a, b, c, d)| (a, b, c, d, true))
            .collect();
        write_test_emission_with_competence(path, &all_competent);
    }

    /// As `write_test_emission`, but each row also carries its
    /// `is_polyA_competent` flag.
    ///
    /// `n_cohort_flnc` is written as a NON-ZERO constant for every row, so no row
    /// is imputed. It is written at all because the loader REQUIRES the column.
    fn write_test_emission_with_competence(path: &Path, rows: &[(&str, &str, f64, f64, bool)]) {
        let with_nobs: Vec<(&str, &str, f64, f64, bool, u32)> = rows
            .iter()
            .map(|&(tx, g, mu, s2, c)| (tx, g, mu, s2, c, 10))
            .collect();
        write_test_emission_with_nobs(path, &with_nobs)
    }

    /// Same as `write_test_emission_with_competence` but with an explicit
    /// `n_cohort_flnc` per row, so a test can mark a row IMPUTED with 0.
    fn write_test_emission_with_nobs(path: &Path, rows: &[(&str, &str, f64, f64, bool, u32)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("transcript_id", DataType::Utf8, false),
            Field::new("gene_id", DataType::Utf8, false),
            Field::new("is_polyA_competent", DataType::Boolean, false),
            Field::new("mu_logPAL", DataType::Float64, false),
            Field::new("sigma2_logPAL", DataType::Float64, false),
            Field::new("n_cohort_flnc", DataType::UInt32, false),
        ]));
        let mut tx_b = StringBuilder::new();
        let mut gene_b = StringBuilder::new();
        let mut comp_b = BooleanBuilder::new();
        let mut mu_b = Float64Builder::new();
        let mut s2_b = Float64Builder::new();
        let mut nobs_b = UInt32Builder::new();
        for &(tx, gene, mu, sigma2, is_competent, nobs) in rows {
            tx_b.append_value(tx);
            gene_b.append_value(gene);
            comp_b.append_value(is_competent);
            mu_b.append_value(mu);
            s2_b.append_value(sigma2);
            nobs_b.append_value(nobs);
        }
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(tx_b.finish()),
                Arc::new(gene_b.finish()),
                Arc::new(comp_b.finish()),
                Arc::new(mu_b.finish()),
                Arc::new(s2_b.finish()),
                Arc::new(nobs_b.finish()),
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
        let with_gaps: Vec<(&str, i32, u32)> =
            candidates.iter().map(|&(tx, as_)| (tx, as_, 0)).collect();
        synth_graph_with_gaps(log_pal, &with_gaps)
    }

    /// At `poly_a_weight == 0` the M-step must FREEZE mu and sigma2, not
    /// recompute them from a placeholder `y`. With `accum_w` (W) still
    /// accumulating and S1 forced to 0:
    ///
    ///     mu     = (0 + tau_0 * mu_cohort) / (w_sum + tau_0)  ->  0
    ///     sigma2 -> INSTRUMENTAL_PAL_NOISE_FLOOR
    #[test]
    fn test_poly_a_weight_zero_freezes_mu_and_sigma2_instead_of_dragging_them() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("mstep_freeze_emission.parquet");
        let profile = scratch("mstep_freeze_profile.parquet");
        write_test_emission(
            &emission,
            &[("tx0", "gene0", 7.0, 0.5), ("tx1", "gene1", 7.0, 0.5)],
        );
        let cands: [(&str, i32, u32); 2] = [("tx0", 1000, 0), ("tx1", 1000, 0)];

        let mut params_on = Vec::new();
        solve_and_write_profile(
            synth_graph_with_gaps(log_pal, &cands),
            &emission,
            "TEST",
            &profile,
            1e-6,
            5,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            Some(&mut params_on),
            None,
        )
        .unwrap();
        let mu_on = params_on[0].as_ref().expect("tx0 must have params").mu;
        assert!(
            (mu_on - 7.0).abs() > 1e-6,
            "CONTROL: with poly(A) on, mu must be updated away from its initial \
             7.0, otherwise this test proves nothing. mu_on = {mu_on}"
        );

        let mut params_off = Vec::new();
        solve_and_write_profile(
            synth_graph_with_gaps(log_pal, &cands),
            &emission,
            "TEST",
            &profile,
            1e-6,
            5,
            20.0,
            FIVE_PRIME_OFF,
            0.0,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            Some(&mut params_off),
            None,
        )
        .unwrap();

        for (i, tx) in ["tx0", "tx1"].iter().enumerate() {
            let p = params_off[i]
                .as_ref()
                .unwrap_or_else(|| panic!("{tx} must have params"));
            assert!(
                (p.mu - 7.0).abs() < 1e-12,
                "{tx}: mu must stay at its loaded value 7.0 at poly_a_weight = 0, \
                 got {} -- the placeholder y reached the M-step",
                p.mu
            );
            assert!(
                (p.sigma2 - 0.5).abs() < 1e-12,
                "{tx}: sigma2 must stay at its loaded value 0.5 at poly_a_weight = 0, \
                 got {} -- expected the INSTRUMENTAL_PAL_NOISE_FLOOR drag",
                p.sigma2
            );
            assert!(p.mu.is_finite() && p.sigma2.is_finite());
        }

        let alpha_off = params_off[0].as_ref().unwrap().alpha;
        assert!(
            alpha_off > 0.0,
            "STEP 3 (alpha) must still run at poly_a_weight = 0: it is the \
             abundance prior, not a poly(A) term"
        );

        let accum_w_off = params_off[0].as_ref().unwrap().em_accum_weight;
        assert!(
            accum_w_off > 0.0,
            "W must still accumulate at poly_a_weight = 0: it is the weight sum \
             alpha is divided by, and it is NOT a poly(A) term. Got {accum_w_off}."
        );

        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);
    }

    /// STEP 1 and STEP 2 of the M-step, asserted as VALUES at the one setting
    /// where they actually run. Tolerance 1e-12: this is a bit-for-bit closed
    /// form, not a numeric estimate, so any wider tolerance buys tolerance for a
    /// bug.
    ///
    /// `max_iter = 2` AND NOT MORE: `output_emission_params` is fed from
    /// `last_estep_params`, snapshotted after the E-step consumed `params` and
    /// before the M-step rewrote it, so the exported value is one M-step BEHIND
    /// `params`.
    ///
    ///     sigma2 = (sq_dev + 2*beta_0*sigma2_cohort) / (w_sum + 2*nu_0 + 2)
    ///             = (0.040588324521667830 + 1.0) / 4.5 = 0.23124184989370397
    #[test]
    fn test_mu_cohort_is_immutable_across_em_iterations() {
        const MU_COHORT: f64 = 7.0;
        let y = (100.0_f32).log2() as f64;

        let emission = scratch("mu_cohort_frozen_emission.parquet");
        let profile = scratch("mu_cohort_frozen_profile.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", MU_COHORT, 0.5),
                ("tx1", "gene1", MU_COHORT, 0.5),
            ],
        );
        let cands: [(&str, i32, u32); 2] = [("tx0", 1000, 0), ("tx1", 1000, 0)];

        let mut params = Vec::new();
        solve_and_write_profile(
            synth_graph_with_gaps(y as f32, &cands),
            &emission,
            "TEST",
            &profile,
            1e-6,
            2,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            Some(&mut params),
            None,
        )
        .unwrap();

        for (i, tx) in ["tx0", "tx1"].iter().enumerate() {
            let p = params[i]
                .as_ref()
                .unwrap_or_else(|| panic!("{tx} must have params"));

            assert!(
                (p.mu - MU_COHORT).abs() > 1e-9,
                "{tx}: mu ({}) must differ from the prior ({MU_COHORT}) for this \
                 test to mean anything; equal values make the next assertion \
                 vacuous",
                p.mu
            );

            assert_eq!(
                p.mu_cohort, MU_COHORT,
                "{tx}: mu_cohort MUTATED during the EM run (prior={MU_COHORT}, \
                 now={}). The shrinkage target must stay frozen at the loaded \
                 value for every iteration; a prior that chases the current \
                 iterate turns the M-step into a self-anchoring fixed point.",
                p.mu_cohort
            );
        }
    }

    #[test]
    fn test_mstep_mu_and_sigma2_match_their_closed_forms_when_poly_a_is_on() {
        const EXPECTED_MU: f64 = 6.928_771_209_716_797;
        // sigma^2 for the SAME fixture, recomputed from the documented operands
        // rather than read off a run.
        //
        //   One molecule, two SYMMETRIC candidates: the posterior is exactly
        //   (0.5, 0.5), so each transcript sees W = 0.5, S1 = 0.5*y, S2 = 0.5*y^2
        //   with y = (100f32).log2() as f64 = 6.6438560485839844
        //   => S1 = 3.3219280242919922, S2 = 22.070411597152997
        //   mu = (S1 + tau_0*mu_cohort)/(W + tau_0) = 6.9287712097167971
        //   S(mu) = S2 - 2*mu*S1 + mu^2*W          = 0.040588324521667829
        //
        //   MODEL A: no kappa_0*(mu - mu_cohort)^2 term, and the
        //   denominator loses the `+1` that term's normaliser contributed:
        //   numerator   = S(mu) + 2*beta_0*sigma2_cohort = 1.040588324521667829
        //   denominator = W + 2*nu_0 + 2                  = 4.5
        //   sigma2 = 1.040588324521667829 / 4.5            = 0.23124184989370397
        //
        // READ THIS NUMBER OFF A RUN, NOT OFF PAPER: recomputing it in f64 from
        // the printed operands gives 0.23124184274217108 -- eight digits, not 17.
        const EXPECTED_SIGMA2: f64 = 0.231_241_849_893_703_97;
        const TOL: f64 = 1e-12;

        let y = (100.0_f32).log2() as f64;
        const MU_COHORT: f64 = 7.0;

        let emission = scratch("mstep_closed_form_emission.parquet");
        let profile = scratch("mstep_closed_form_profile.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", MU_COHORT, 0.5),
                ("tx1", "gene1", MU_COHORT, 0.5),
            ],
        );
        let cands: [(&str, i32, u32); 2] = [("tx0", 1000, 0), ("tx1", 1000, 0)];

        let mut params_on = Vec::new();
        solve_and_write_profile(
            synth_graph_with_gaps((100.0_f32).log2(), &cands),
            &emission,
            "TEST",
            &profile,
            1e-6,
            2, // max_iter -- see the note above; do NOT raise.
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            Some(&mut params_on),
            None,
        )
        .unwrap();

        for (i, tx) in ["tx0", "tx1"].iter().enumerate() {
            let p = params_on[i]
                .as_ref()
                .unwrap_or_else(|| panic!("{tx} must have params"));

            assert!(
                p.mu > y.min(MU_COHORT) && p.mu < y.max(MU_COHORT),
                "{tx}: the empirical-Bayes shrinkage must pull mu strictly between \
                 the observation {y} and the cohort prior {MU_COHORT}; got {}. \
                 Equal to one endpoint means one of the two terms was dropped.",
                p.mu
            );
            assert!(
                (p.mu - EXPECTED_MU).abs() < TOL,
                "M-STEP mu REGRESSION on {tx}: got {:.17}, expected {EXPECTED_MU:.17}. \
                 The expected value is the f32-widened closed form derived on this \
                 test. If the prior weight, the weights, or the placeholder moved, \
                 re-derive it from that derivation -- do NOT widen the tolerance, \
                 which is the one thing this test exists to prevent.",
                p.mu
            );
            assert!(
                (p.sigma2 - EXPECTED_SIGMA2).abs() < TOL,
                "M-STEP sigma2 REGRESSION on {tx}: got {:.17}, expected \
                 {EXPECTED_SIGMA2:.17}. A value of 0.04 means \
                 INSTRUMENTAL_PAL_NOISE_FLOOR clamped the closed form, so this \
                 fixture stopped exercising it.",
                p.sigma2
            );
        }

        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);
    }

    const DOF_OFFSET_MU_COHORT: f64 = 7.0;
    const DOF_OFFSET_SIGMA2_COHORT: f64 = 0.5;
    const DOF_OFFSET_W: f64 = 0.5;

    fn dof_offset_closed_form(c: f64) -> f64 {
        let y = (100.0_f32).log2() as f64;
        let mu = (DOF_OFFSET_W * y + TAU_0 * DOF_OFFSET_MU_COHORT) / (DOF_OFFSET_W + TAU_0);
        let s1 = DOF_OFFSET_W * y;
        let s2 = DOF_OFFSET_W * y * y;
        let sq_dev_new = (s2 - 2.0 * mu * s1 + mu * mu * DOF_OFFSET_W).max(0.0);
        let numerator = sq_dev_new + 2.0 * BETA_0 * DOF_OFFSET_SIGMA2_COHORT;
        numerator / (DOF_OFFSET_W + 2.0 * NU_0 + c)
    }

    #[allow(clippy::type_complexity)]
    fn dof_offset_run(
        emission: &Path,
        profile_tag: &str,
        explicit_c: Option<f64>,
    ) -> (
        TelEmStats,
        Vec<Option<TranscriptParam>>,
        Vec<(usize, f64, f64)>,
        Vec<ReadAssignment>,
    ) {
        let profile = scratch(&format!("dof_offset_profile_{profile_tag}.parquet"));
        let cands: [(&str, i32, u32); 2] = [("tx0", 1000, 0), ("tx1", 1000, 0)];
        let mut params = Vec::new();
        let mut trace = Vec::new();
        let mut reads = Vec::new();
        let stats = match explicit_c {
            Some(c) => solve_and_write_profile_with_dof_offset(
                synth_graph_with_gaps((100.0_f32).log2(), &cands),
                emission,
                "TEST",
                &profile,
                1e-6,
                2, // max_iter -- see the closed-form test's note; do NOT raise.
                20.0,
                FIVE_PRIME_OFF,
                POLY_A_ON,
                FIVE_PRIME_WINDOW,
                PRUNE_MAX_DELTA_AS,
                MIN_WEIGHT,
                Some(&mut trace),
                Some(&mut params),
                Some(&mut reads),
                c,
            ),
            None => solve_and_write_profile(
                synth_graph_with_gaps((100.0_f32).log2(), &cands),
                emission,
                "TEST",
                &profile,
                1e-6,
                2,
                20.0,
                FIVE_PRIME_OFF,
                POLY_A_ON,
                FIVE_PRIME_WINDOW,
                PRUNE_MAX_DELTA_AS,
                MIN_WEIGHT,
                Some(&mut trace),
                Some(&mut params),
                Some(&mut reads),
            ),
        }
        .unwrap();
        let _ = std::fs::remove_file(&profile);
        (stats, params, trace, reads)
    }

    fn dof_offset_emission(tag: &str) -> PathBuf {
        let emission = scratch(&format!("dof_offset_emission_{tag}.parquet"));
        write_test_emission(
            &emission,
            &[
                (
                    "tx0",
                    "gene0",
                    DOF_OFFSET_MU_COHORT,
                    DOF_OFFSET_SIGMA2_COHORT,
                ),
                (
                    "tx1",
                    "gene1",
                    DOF_OFFSET_MU_COHORT,
                    DOF_OFFSET_SIGMA2_COHORT,
                ),
            ],
        );
        emission
    }

    /// The default must be the shipped Model A estimator: run the SAME fixture
    /// through the parameterised entry point with `c = 2` explicitly and through
    /// the default entry point, and require the two to agree to the LAST BIT on
    /// every observable.
    ///
    /// EXACT `to_bits()` EQUALITY, NO TOLERANCE -- a `< 1e-15` bound here would
    /// still pass for every `c` within an ULP of 2.
    #[test]
    fn test_default_sigma2_dof_offset_is_bit_identical_to_the_named_parameter() {
        let emission = dof_offset_emission("default");
        let (stats_a, params_a, trace_a, reads_a) = dof_offset_run(&emission, "default", None);
        let (stats_b, params_b, trace_b, reads_b) =
            dof_offset_run(&emission, "default_explicit", Some(2.0));
        let _ = std::fs::remove_file(&emission);

        assert_eq!(
            params_a.len(),
            2,
            "arm A must produce both transcripts or the bit-comparison below is vacuous"
        );
        assert_eq!(
            params_b.len(),
            2,
            "arm B must produce both transcripts or the bit-comparison below is vacuous"
        );
        assert!(
            !trace_a.is_empty() && !trace_b.is_empty(),
            "both arms must record an EM trace or the trace comparison is vacuous"
        );
        assert_eq!(
            reads_a.len(),
            1,
            "arm A must produce the read assignment or that comparison is vacuous"
        );
        assert_eq!(
            reads_b.len(),
            1,
            "arm B must produce the read assignment or that comparison is vacuous"
        );

        for (i, (pa, pb)) in params_a.iter().zip(params_b.iter()).enumerate() {
            let pa = pa
                .as_ref()
                .unwrap_or_else(|| panic!("arm A row {i} must have params"));
            let pb = pb
                .as_ref()
                .unwrap_or_else(|| panic!("arm B row {i} must have params"));
            assert_eq!(pa.transcript_id, pb.transcript_id, "row {i} transcript_id");
            assert_eq!(pa.gene_id, pb.gene_id, "row {i} gene_id");
            assert_eq!(pa.is_competent, pb.is_competent, "row {i} is_competent");
            assert_eq!(pa.em_finalized, pb.em_finalized, "row {i} em_finalized");
            assert_eq!(pa.is_imputed, pb.is_imputed, "row {i} is_imputed");
            assert_eq!(pa.mu.to_bits(), pb.mu.to_bits(), "row {i} mu (bitwise)");
            assert_eq!(
                pa.sigma2.to_bits(),
                pb.sigma2.to_bits(),
                "row {i} sigma2 (bitwise); THE DEFAULT MUST BE c = 2 BIT FOR BIT"
            );
            assert_eq!(
                pa.sigma2_cohort.to_bits(),
                pb.sigma2_cohort.to_bits(),
                "row {i} sigma2_cohort (bitwise)"
            );
            assert_eq!(
                pa.mu_cohort.to_bits(),
                pb.mu_cohort.to_bits(),
                "row {i} mu_cohort (bitwise)"
            );
            assert_eq!(
                pa.alpha.to_bits(),
                pb.alpha.to_bits(),
                "row {i} alpha (bitwise)"
            );
            assert_eq!(
                pa.em_accum_weight.to_bits(),
                pb.em_accum_weight.to_bits(),
                "row {i} em_accum_weight (bitwise)"
            );
        }

        assert_eq!(
            trace_a.len(),
            trace_b.len(),
            "EM trace length differs: {} vs {}",
            trace_a.len(),
            trace_b.len()
        );
        for (i, ((ai, al, ae), (bi, bl, be))) in trace_a.iter().zip(trace_b.iter()).enumerate() {
            assert_eq!(ai, bi, "trace[{i}] iteration index");
            assert_eq!(
                al.to_bits(),
                bl.to_bits(),
                "trace[{i}] log-likelihood (bitwise)"
            );
            assert_eq!(
                ae.to_bits(),
                be.to_bits(),
                "trace[{i}] mean entropy (bitwise)"
            );
        }

        for (i, (ra, rb)) in reads_a.iter().zip(reads_b.iter()).enumerate() {
            assert_eq!(ra.read_id, rb.read_id, "read[{i}] read_id");
            assert_eq!(
                ra.n_candidates_before_cutoff, rb.n_candidates_before_cutoff,
                "read[{i}] n_candidates_before_cutoff"
            );
            assert_eq!(ra.top_tx_id, rb.top_tx_id, "read[{i}] top_tx_id");
            assert_eq!(
                ra.top_weight.to_bits(),
                rb.top_weight.to_bits(),
                "read[{i}] top_weight (bitwise)"
            );
            assert_eq!(
                ra.entropy.to_bits(),
                rb.entropy.to_bits(),
                "read[{i}] entropy (bitwise)"
            );
        }

        assert_eq!(
            stats_a.ledger, stats_b.ledger,
            "the candidate ledger must be identical between the two arms"
        );
        assert_eq!(
            format!("{stats_a:?}"),
            format!("{stats_b:?}"),
            "TelEmStats must be identical between the default and the explicit c = 2 arm"
        );
    }

    /// The pre/post receipt. These two bit patterns are what production produced
    /// BEFORE `sigma2_dof_offset` became a parameter.
    ///
    /// `0x3fcd99553b6147ab` is `EXPECTED_SIGMA2.to_bits()` for the closed-form
    /// test above, and `0x401bb70fcccccccd` is `EXPECTED_MU.to_bits()`.
    #[test]
    fn test_closed_form_fixture_sigma2_and_mu_are_bit_exact() {
        let emission = dof_offset_emission("bits");
        let (_stats, params, _trace, _reads) = dof_offset_run(&emission, "bits", None);
        let _ = std::fs::remove_file(&emission);
        let p = params[0]
            .as_ref()
            .expect("tx0 must have params in the closed-form fixture");

        assert_eq!(
            p.sigma2.to_bits(),
            0x3fcd_9955_3b61_47abu64,
            "sigma2 bit pattern CHANGED. Measured pre-refactor: 0x3fcd99553b6147ab \
             (got {:#018x}, {:.17}); report the ULP distance between them, do NOT \
             widen this assertion",
            p.sigma2.to_bits(),
            p.sigma2
        );
        assert_eq!(
            p.mu.to_bits(),
            0x401b_b70f_cccc_cccd_u64,
            "mu bit pattern CHANGED. Measured pre-refactor: 0x401bb70fcccccccd \
             (got {:#018x}, {:.17}); report the ULP distance between them, do NOT \
             widen this assertion",
            p.mu.to_bits(),
            p.mu
        );
    }

    /// What `c = 3` actually computes -- and, more usefully, what it does NOT
    /// compute.
    ///
    /// * the value is compared BITWISE against `dof_offset_closed_form(3.0)`,
    ///   computed in the test from the fixture's documented operands; comparing
    ///   only against sigma2(2) being smaller would be satisfied by any
    ///   monotone-in-`c` implementation, including one that ignores `c`.
    /// * `c = 3` is NOT Model B, asserted with a number rather than prose:
    ///   Model B also restores `tau_0 * (mu - mu_cohort)^2` in the
    ///   NUMERATOR, while the `c` knob moves the denominator only.
    #[test]
    fn test_sigma2_dof_offset_three_is_the_third_estimator_not_model_b() {
        let emission = dof_offset_emission("c3");
        let (_s3, params3, _t3, _r3) = dof_offset_run(&emission, "c3", Some(3.0));
        let (_s2, params2, _t2, _r2) = dof_offset_run(&emission, "c2", Some(2.0));
        let _ = std::fs::remove_file(&emission);
        let s3 = params3[0].as_ref().expect("tx0 at c=3").sigma2;
        let s2 = params2[0].as_ref().expect("tx0 at c=2").sigma2;

        assert_eq!(
            s3.to_bits(),
            dof_offset_closed_form(3.0).to_bits(),
            "sigma2 at c = 3 must equal the closed form \
             (S(mu) + 2*beta_0*sigma2_cohort) / (W + 2*nu_0 + 3) BIT FOR BIT; \
             got {s3:.17}, closed form {:.17}",
            dof_offset_closed_form(3.0)
        );
        assert_eq!(
            s3.to_bits(),
            0x3fc8_37a2_d37e_235du64,
            "sigma2 at c = 3 moved: got {s3:.17} ({:#018x}), expected 0.18919787718575778",
            s3.to_bits()
        );

        // `c = 3` is NOT Model B. Recomputed here rather than quoted, so
        // the distinction survives an edit to the constant or to a prior.
        let y = (100.0_f32).log2() as f64;
        let mu = (DOF_OFFSET_W * y + TAU_0 * DOF_OFFSET_MU_COHORT) / (DOF_OFFSET_W + TAU_0);
        let s1 = DOF_OFFSET_W * y;
        let s2_sum = DOF_OFFSET_W * y * y;
        let sq_dev_new = (s2_sum - 2.0 * mu * s1 + mu * mu * DOF_OFFSET_W).max(0.0);
        let model_b = (sq_dev_new
            + TAU_0 * (mu - DOF_OFFSET_MU_COHORT).powi(2)
            + 2.0 * BETA_0 * DOF_OFFSET_SIGMA2_COHORT)
            / (DOF_OFFSET_W + 2.0 * NU_0 + 3.0);
        assert_eq!(
            model_b.to_bits(),
            0x3fc8_7417_2b46_6663u64,
            "the Model B reference moved: got {model_b:.17}. Re-derive it \
             before trusting the comparison below"
        );
        assert_ne!(
            s3.to_bits(),
            model_b.to_bits(),
            "c = 3 has become Model B. The knob moves the DENOMINATOR only; \
             Model B also needs tau_0*(mu - mu_cohort)^2 back in the \
             numerator, and the two land on different values ({s3} vs {model_b}). \
             Either the numerator was parameterised too, or this arm is no longer the \
             estimator it claims."
        );

        assert!(
            s3 < s2,
            "sigma2 must SHRINK as c grows: sigma2(3) = {s3} is not below \
             sigma2(2) = {s2}"
        );

        assert!(
            s3 > INSTRUMENTAL_PAL_NOISE_FLOOR * 2.0,
            "sigma2 at c = 3 is {s3}, within 2x INSTRUMENTAL_PAL_NOISE_FLOOR \
             ({INSTRUMENTAL_PAL_NOISE_FLOOR}); the floor would be clamping and this \
             arm would be measuring the floor rather than the closed form"
        );
    }

    /// sigma^2 is strictly decreasing in `c` over [1, 5], and every point equals
    /// the closed form bit for bit.
    ///
    /// A NON-MONOTONE SWEEP IS A BUG, NOT A RESULT. The arithmetic says
    /// `sigma2 = N / (W + 2*nu_0 + c)` with `N` positive and `c`-independent,
    /// hence strictly decreasing; any other shape means `c` reached something it
    /// should not have -- the E-step's weights, `mu`, or the floor.
    #[test]
    fn test_sigma2_is_strictly_decreasing_over_the_c_sweep() {
        let sweep = [1.0f64, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let emission = dof_offset_emission("sweep");
        let mut measured = Vec::new();
        for (i, c) in sweep.iter().enumerate() {
            let (_stats, params, _trace, _reads) =
                dof_offset_run(&emission, &format!("sweep{i}"), Some(*c));
            let s = params[0].as_ref().expect("tx0").sigma2;
            assert_eq!(
                s.to_bits(),
                dof_offset_closed_form(*c).to_bits(),
                "sigma2 at c = {c} must equal the closed form BIT FOR BIT; got \
                 {s:.17}, closed form {:.17}",
                dof_offset_closed_form(*c)
            );
            measured.push((c, s));
        }
        let _ = std::fs::remove_file(&emission);

        for w in measured.windows(2) {
            let (c_lo, s_lo) = w[0];
            let (c_hi, s_hi) = w[1];
            assert!(
                s_hi < s_lo,
                "sigma2 must be STRICTLY decreasing in c, but sigma2({c_hi}) = {s_hi} \
                 is not below sigma2({c_lo}) = {s_lo}. A flat or rising step means `c` \
                 reached the E-step, mu, or the noise floor -- investigate before \
                 writing this down as a result."
            );
        }
        let floor_ratio =
            measured.last().expect("the sweep is non-empty").1 / INSTRUMENTAL_PAL_NOISE_FLOOR;
        assert!(
            floor_ratio > 2.0,
            "the sweep's minimum sigma2 is {floor_ratio}x the noise floor; below \
             roughly 2x, the low end of this sweep would be measuring \
             INSTRUMENTAL_PAL_NOISE_FLOOR rather than the closed form"
        );
    }

    /// A transcript that wins NO posterior mass still gets an M-step, and the
    /// sigma^2 it lands on is `2*beta_0*sigma2_cohort / (2*nu_0 + c)` -- NOT the
    /// prior. The M-step loop iterates EVERY graph transcript, so `w_sum == 0` is
    /// a state the estimator genuinely visits.
    ///
    /// The ratio to the prior is `2 / (2*nu_0 + c)`, so it TRACKS `c` -- 0.500x at
    /// the production value `c = 2`. `c = 0` -- which would make the ratio exactly
    /// 1.000 and reduce the M-step to a no-op at W = 0 -- is UNREACHABLE, because
    /// the parameterised entry point's guard is `is_finite() && > 0.0`.
    #[test]
    fn test_zero_weight_transcript_sigma2_is_two_beta0_prior_over_two_nu0_plus_c() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("w0_emission.parquet");
        write_test_emission_with_competence(
            &emission,
            &[
                (
                    "tx0",
                    "gene0",
                    DOF_OFFSET_MU_COHORT,
                    DOF_OFFSET_SIGMA2_COHORT,
                    true,
                ),
                (
                    "tx1",
                    "gene1",
                    DOF_OFFSET_MU_COHORT,
                    DOF_OFFSET_SIGMA2_COHORT,
                    true,
                ),
                (
                    "tx2",
                    "gene2",
                    DOF_OFFSET_MU_COHORT,
                    DOF_OFFSET_SIGMA2_COHORT,
                    false,
                ),
            ],
        );

        let sweep = [0.5f64, 1.0, 2.0, 3.0, 5.0];
        let mut measured = Vec::new();
        for c in sweep {
            let profile = scratch(&format!("w0_profile_{c}.parquet"));
            let mut params = Vec::new();
            solve_and_write_profile_with_dof_offset(
                synth_graph(log_pal, &[("tx0", 900), ("tx1", 900), ("tx2", 900)]),
                &emission,
                "TEST",
                &profile,
                1e-6,
                2,
                100.0,
                FIVE_PRIME_OFF,
                POLY_A_ON,
                FIVE_PRIME_WINDOW,
                PRUNE_MAX_DELTA_AS,
                MIN_WEIGHT,
                None,
                Some(&mut params),
                None,
                c,
            )
            .unwrap();
            let _ = std::fs::remove_file(&profile);

            let w0 = params
                .iter()
                .flatten()
                .find(|p| p.transcript_id == "tx2")
                .expect(
                    "tx2 must have params: it is in the graph, so the M-step \
                         visits it regardless of the weight it won",
                );
            let live = params
                .iter()
                .flatten()
                .find(|p| p.transcript_id == "tx0")
                .expect("tx0 must have params");

            assert_eq!(
                w0.em_accum_weight, 0.0,
                "tx2 is pushed to -inf, so its M-step W must be exactly 0.0; \
                 got {}. If this is non-zero the fixture no longer produces a \
                 W = 0 transcript and the sigma2 assertions below are measuring \
                 something else.",
                w0.em_accum_weight
            );
            assert!(
                live.em_accum_weight > 0.0,
                "CONTROL: tx0 must win posterior mass ({}) or the zero above says \
                 nothing about incompetence",
                live.em_accum_weight
            );

            let want = 2.0 * BETA_0 * DOF_OFFSET_SIGMA2_COHORT / (2.0 * NU_0 + c);
            assert_eq!(
                w0.sigma2.to_bits(),
                want.to_bits(),
                "sigma2 at W = 0 and c = {c} must equal \
                 2*beta_0*sigma2_cohort / (2*nu_0 + c) BIT FOR BIT; got {:.17}, \
                 want {:.17}",
                w0.sigma2,
                want
            );

            assert_eq!(
                w0.mu.to_bits(),
                DOF_OFFSET_MU_COHORT.to_bits(),
                "at W = 0 mu must land EXACTLY on the cohort prior, not near it; \
                 got {:.17}",
                w0.mu
            );

            assert!(
                w0.sigma2 > INSTRUMENTAL_PAL_NOISE_FLOOR,
                "sigma2 at c = {c} is {}, at or below INSTRUMENTAL_PAL_NOISE_FLOOR \
                 ({INSTRUMENTAL_PAL_NOISE_FLOOR}); the floor is clamping and the \
                 bit-exact comparison is vacuous",
                w0.sigma2
            );

            let ratio = w0.sigma2 / DOF_OFFSET_SIGMA2_COHORT;
            let want_ratio = 2.0 / (2.0 * NU_0 + c);
            assert_eq!(
                ratio.to_bits(),
                want_ratio.to_bits(),
                "the W = 0 ratio to the prior must be exactly 2/(2*nu_0 + c) = \
                 {want_ratio:.6} at c = {c}; got {ratio:.17}. The factor TRACKS c, \
                 so \"sigma2 is halved with no evidence\" holds only at c = 2."
            );
            measured.push((c, ratio));
        }
        let _ = std::fs::remove_file(&emission);

        let prod = measured
            .iter()
            .find(|(c, _)| *c == 2.0)
            .expect("the sweep contains the production c = 2");
        assert_eq!(
            prod.1.to_bits(),
            0.5f64.to_bits(),
            "at the production c = 2 the W = 0 sigma2 is exactly HALF the prior; \
             got {:.17}. If this moved, the closed form changed.",
            prod.1
        );
        for w in measured.windows(2) {
            assert!(
                w[1].1 < w[0].1,
                "the W = 0 ratio must be STRICTLY decreasing in c, but \
                 ratio({}) = {:.6} is not below ratio({}) = {:.6}. Investigate \
                 before writing this down as a result.",
                w[1].0,
                w[1].1,
                w[0].0,
                w[0].1
            );
        }
    }

    #[test]
    #[should_panic(expected = "sigma2_dof_offset must be finite and positive")]
    fn test_sigma2_dof_offset_guard_rejects_the_exact_denominator_zero() {
        let emission = dof_offset_emission("guard");
        let _ = dof_offset_run(&emission, "guard", Some(-2.5));
    }

    #[test]
    #[should_panic(expected = "sigma2_dof_offset must be finite and positive")]
    fn test_sigma2_dof_offset_guard_rejects_nan() {
        let emission = dof_offset_emission("guard_nan");
        let _ = dof_offset_run(&emission, "guard_nan", Some(f64::NAN));
    }

    #[test]
    #[should_panic(expected = "sigma2_dof_offset must be finite and positive")]
    fn test_sigma2_dof_offset_guard_rejects_infinity() {
        let emission = dof_offset_emission("guard_inf");
        let _ = dof_offset_run(&emission, "guard_inf", Some(f64::INFINITY));
    }

    #[allow(clippy::type_complexity)]
    fn read_sidecar(path: &Path) -> (Vec<String>, Vec<(String, Vec<(String, Option<f64>)>)>) {
        let file = File::open(path).expect("sidecar file must exist");
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("sidecar must be readable as parquet")
            .with_batch_size(8192)
            .build()
            .expect("sidecar reader must build");

        let mut names: Vec<String> = Vec::new();
        let mut rows: Vec<(String, Vec<(String, Option<f64>)>)> = Vec::new();
        for batch in reader {
            let batch = batch.expect("sidecar batch must read");
            if names.is_empty() {
                names = batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect();
            }
            let tr = batch
                .column_by_name("transcript_id")
                .expect("sidecar must carry transcript_id")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("transcript_id must be Utf8");
            for i in 0..batch.num_rows() {
                let mut cells = Vec::new();
                for (idx, field) in batch.schema().fields().iter().enumerate() {
                    if field.name() == "transcript_id" {
                        continue;
                    }
                    let v = batch
                        .column(idx)
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .map(|a| a.value(i));
                    cells.push((field.name().clone(), v));
                }
                rows.push((tr.value(i).to_string(), cells));
            }
        }
        (names, rows)
    }

    fn sidecar_cell(row: &(String, Vec<(String, Option<f64>)>), col: &str) -> Option<f64> {
        row.1.iter().find(|(n, _)| n == col).and_then(|(_, v)| *v)
    }

    fn sidecar_finalized(path: &Path) -> Vec<(String, bool)> {
        let file = File::open(path).expect("sidecar file must exist");
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("sidecar must be readable as parquet")
            .with_batch_size(8192)
            .build()
            .expect("sidecar reader must build");

        let mut out = Vec::new();
        for batch in reader {
            let batch = batch.expect("sidecar batch must read");
            let tr = batch
                .column_by_name("transcript_id")
                .expect("sidecar must carry transcript_id")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("transcript_id must be Utf8");
            let fin = batch
                .column_by_name("em_finalized")
                .expect("the sidecar must carry an `em_finalized` column")
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("em_finalized must be Boolean, not another type");
            for i in 0..batch.num_rows() {
                out.push((tr.value(i).to_string(), fin.value(i)));
            }
        }
        assert!(!out.is_empty(), "sidecar must emit at least one row");
        out
    }

    #[test]
    fn test_sidecar_em_accum_weight_is_the_estep_weight_sum_in_molecules() {
        const TOL: f64 = 1e-12;
        const EXPECTED_W: f64 = 0.5;
        const EXPECTED_ALPHA: f64 = 0.25;

        let emission = scratch("accum_w_emission.parquet");
        let profile = scratch("accum_w_profile.parquet");
        let sidecar = scratch("accum_w_sidecar.parquet");

        let graph = multi_molecule_graph(
            (100.0_f32).log2(),
            &[
                vec![("tx0", 1000, 0), ("tx1", 1000, 0)],
                vec![("solo", 1000, 0)],
            ],
        );
        multi_molecule_emission(&emission, &["r0_tx0", "r0_tx1", "r1_solo"]);

        let mut params = Vec::new();
        // max_iter = 2 so an M-step has run; with 1 the sidecar carries the Laplace init.
        solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            2, // max_iter -- see the note above; do NOT lower to 1.
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            Some(&mut params),
            None,
        )
        .unwrap();

        write_emission_params_parquet(&params, &sidecar).expect("sidecar must write");
        let (_, rows) = read_sidecar(&sidecar);
        assert_eq!(rows.len(), 3, "every emitted transcript gets a row");

        let fin = sidecar_finalized(&sidecar);
        assert!(
            fin.iter().all(|(_, v)| *v),
            "max_iter = 2 with a contested molecule MUST run an M-step, so every \
             row must be em_finalized; got {:?}",
            fin.iter()
                .filter(|(_, v)| !*v)
                .map(|(t, _)| t)
                .collect::<Vec<_>>()
        );

        for tx in ["r0_tx0", "r0_tx1"] {
            let row = rows
                .iter()
                .find(|(id, _)| id == tx)
                .unwrap_or_else(|| panic!("{tx} must have a sidecar row"));
            let w = sidecar_cell(row, "em_accum_weight")
                .unwrap_or_else(|| panic!("{tx}: sidecar must carry an em_accum_weight column"));
            let alpha = sidecar_cell(row, "alpha").unwrap();

            assert!(
                (w - EXPECTED_W).abs() < TOL,
                "{tx}: em_accum_weight REGRESSION: got {w:.17}, expected \
                 {EXPECTED_W}. em_accum_weight is the E-step weight sum, in \
                 MOLECULE weight units -- not alpha, not the per-read posterior \
                 the profile parquet stores. Derivation is on this test."
            );
            assert!(
                (w - alpha).abs() > 1e-6,
                "{tx}: em_accum_weight ({w}) and alpha ({alpha}) are the same \
                 number, so this fixture cannot tell W from the abundance share \
                 and the assertion above proved nothing"
            );
            assert!(
                (alpha - EXPECTED_ALPHA).abs() < TOL,
                "{tx}: alpha REGRESSION: got {alpha}, expected {EXPECTED_ALPHA} \
                 (W 0.5 over 2 molecules). If this fails the fixture no longer \
                 separates W from the share."
            );
        }

        let solo = rows
            .iter()
            .find(|(id, _)| id == "r1_solo")
            .expect("the unambiguous transcript must have a row");
        assert!(
            (sidecar_cell(solo, "em_accum_weight").unwrap() - 1.0).abs() < TOL,
            "an unambiguous read whose only candidate is present contributes \
             exactly 1.0 to W; got {:?}",
            sidecar_cell(solo, "em_accum_weight")
        );

        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);
        let _ = std::fs::remove_file(&sidecar);
    }

    #[test]
    fn test_write_emission_params_parquet_emits_new_columns_and_omits_none_rows() {
        let sidecar = scratch("writer_sidecar.parquet");
        let rows_in: Vec<Option<TranscriptParam>> = vec![
            Some(TranscriptParam {
                transcript_id: "tA".into(),
                gene_id: "gA".into(),
                is_competent: true,
                mu_cohort: 7.0,
                mu: 6.5,
                sigma2_cohort: 0.5,
                sigma2: 0.25,
                alpha: 0.25,
                em_accum_weight: 0.5,
                em_finalized: true,
                is_imputed: false,
            }),
            None,
            Some(TranscriptParam {
                transcript_id: "tB".into(),
                gene_id: "gB".into(),
                is_competent: false,
                mu_cohort: 7.0,
                mu: 7.25,
                sigma2_cohort: 0.5,
                sigma2: 0.125,
                alpha: 0.75,
                em_accum_weight: 1.5,
                em_finalized: true,
                is_imputed: true,
            }),
        ];

        write_emission_params_parquet(&rows_in, &sidecar).expect("writer must succeed");
        let (names, rows) = read_sidecar(&sidecar);

        for col in [
            "em_accum_weight",
            "em_finalized",
            "mu_fitted",
            "sigma2_fitted",
        ] {
            assert!(
                names.iter().any(|n| n == col),
                "the sidecar must carry a `{col}` column; it has {names:?}"
            );
        }
        for col in [
            "transcript_id",
            "gene_id",
            "is_polyA_competent",
            "mu_logPAL",
            "sigma2_logPAL",
            "mu_cohort",
            "alpha",
            "state",
        ] {
            assert!(
                names.iter().any(|n| n == col),
                "the sidecar must keep its pre-existing `{col}` column; it has \
                 {names:?}"
            );
        }

        assert_eq!(
            rows.len(),
            2,
            "the `None` transcript must get NO row: a row for it would invent a \
             parameter the model never scored. Read back: {:?}",
            rows.iter().map(|(id, _)| id).collect::<Vec<_>>()
        );

        for (tx, em_accum_weight, alpha, mu, sigma2) in
            [("tA", 0.5, 0.25, 6.5, 0.25), ("tB", 1.5, 0.75, 7.25, 0.125)]
        {
            let row = rows
                .iter()
                .find(|(id, _)| id == tx)
                .unwrap_or_else(|| panic!("{tx} must have a row"));
            for (col, want) in [
                ("em_accum_weight", em_accum_weight),
                ("alpha", alpha),
                ("mu_fitted", mu),
                ("sigma2_fitted", sigma2),
                ("mu_logPAL", mu),
                ("sigma2_logPAL", sigma2),
            ] {
                assert_eq!(
                    sidecar_cell(row, col),
                    Some(want),
                    "{tx}: column `{col}` must carry the value written"
                );
            }
        }

        let fin: std::collections::HashMap<_, _> =
            sidecar_finalized(&sidecar).into_iter().collect();
        assert_eq!(
            fin.get("tA"),
            Some(&true),
            "tA: em_finalized must be written"
        );
        assert_eq!(
            fin.get("tB"),
            Some(&true),
            "tB: em_finalized must be written"
        );
        assert_eq!(
            fin.len(),
            2,
            "only the two `Some` transcripts get a row, so the None one must \
             contribute no em_finalized entry either; got {:?}",
            fin.keys().collect::<Vec<_>>()
        );

        let _ = std::fs::remove_file(&sidecar);
    }

    #[test]
    fn test_the_sidecar_cannot_be_reloaded_as_an_emission_model() {
        let sidecar = scratch("reload_probe_sidecar.parquet");
        let rows_in: Vec<Option<TranscriptParam>> = vec![Some(TranscriptParam {
            transcript_id: "tA".into(),
            gene_id: "gA".into(),
            is_competent: true,
            mu_cohort: 7.0,
            mu: 6.5,
            sigma2_cohort: 0.5,
            sigma2: 0.25,
            alpha: 0.25,
            em_accum_weight: 0.5,
            em_finalized: true,
            is_imputed: false,
        })];

        write_emission_params_parquet(&rows_in, &sidecar).expect("writer must succeed");

        let err = load_emission_parameters(&sidecar).err().expect(
            "loading a sidecar as an emission model must FAIL. If this now \
                 succeeds, `n_cohort_flnc` reached the sidecar: the CLI help text \
                 at `--dump-emission-params` and the prior-reanchoring hazard it \
                 describes both have to be re-decided before this is turned back on.",
        );
        let msg = format!("{err:#}");
        assert!(
            msg.contains("n_cohort_flnc"),
            "the round trip must die on the REQUIRED `n_cohort_flnc` column, so the \
             help text names the right reason; got: {msg}"
        );

        let _ = std::fs::remove_file(&sidecar);
    }

    #[test]
    fn test_sidecar_accum_w_is_consistent_with_alpha_times_the_molecule_count() {
        const TOL: f64 = 1e-12;

        let emission = scratch("accum_w_identity_emission.parquet");
        let profile = scratch("accum_w_identity_profile.parquet");
        let sidecar = scratch("accum_w_identity_sidecar.parquet");

        let graph = multi_molecule_graph(
            (100.0_f32).log2(),
            &[
                vec![("tx0", 1000, 0), ("tx1", 1000, 0)],
                vec![("solo", 1000, 0)],
                vec![("a", 1000, 0), ("b", 1000, 0)],
            ],
        );
        multi_molecule_emission(&emission, &["r0_tx0", "r0_tx1", "r1_solo", "r2_a", "r2_b"]);

        let mut params = Vec::new();
        // max_iter = 2 so an M-step has run; with 1 the sidecar carries the Laplace init.
        solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            2, // max_iter -- >= 2 so an M-step has run; see the note above.
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            Some(&mut params),
            None,
        )
        .unwrap();
        write_emission_params_parquet(&params, &sidecar).expect("sidecar must write");

        let (_, rows) = read_sidecar(&sidecar);
        let sum_w: f64 = rows
            .iter()
            .map(|r| sidecar_cell(r, "em_accum_weight").unwrap())
            .sum();
        let sum_alpha: f64 = rows.iter().map(|r| sidecar_cell(r, "alpha").unwrap()).sum();
        let n_total_molecules = 3.0_f64;

        assert!(
            sidecar_finalized(&sidecar).iter().all(|(_, v)| *v),
            "this fixture has contested molecules and max_iter = 2, so an \
             M-step ran and every row must be em_finalized"
        );

        assert!(
            (sum_w - n_total_molecules * sum_alpha).abs() < TOL,
            "sum(em_accum_weight) = {sum_w} but n_total_molecules * sum(alpha) = {}; \
             the two columns come from the same M-step and must agree",
            n_total_molecules * sum_alpha
        );
        assert!(
            sum_w > 1.0 && (sum_alpha - 1.0).abs() < TOL,
            "non-degenerate guard: sum(em_accum_weight) = {sum_w}, sum(alpha) = {sum_alpha}"
        );

        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);
        let _ = std::fs::remove_file(&sidecar);
    }

    #[test]
    fn test_c2_stamping_is_atomic_where_w_actually_moves_between_iterations() {
        const TOL: f64 = 1e-12;
        const N_MOLECULES: f64 = 3.0;
        const RUNS: [usize; 4] = [1, 2, 3, 4];

        let emission = scratch("c2_dynamic_emission.parquet");
        write_test_emission(
            &emission,
            &[
                ("r0_tx0", "g", 5.0, 0.5),
                ("r0_tx1", "g", 8.0, 0.5),
                ("r1_solo", "g", 6.5, 0.5),
                ("r2_a", "g", 5.5, 0.5),
                ("r2_b", "g", 7.5, 0.5),
            ],
        );

        let build_graph = || {
            multi_molecule_graph(
                (100.0_f32).log2(),
                &[
                    vec![("tx0", 900, 0), ("tx1", 870, 0)],
                    vec![("solo", 1000, 0)],
                    vec![("a", 1000, 0), ("b", 1000, 0)],
                ],
            )
        };

        let stamped_at = |max_iter: usize| -> Vec<(String, f64, f64, bool)> {
            let profile = scratch(&format!("c2_dynamic_profile_{max_iter}.parquet"));
            let mut params: Vec<Option<TranscriptParam>> = Vec::new();
            solve_and_write_profile(
                build_graph(),
                &emission,
                "TEST",
                &profile,
                1e-6,
                max_iter,
                20.0,
                FIVE_PRIME_OFF,
                POLY_A_ON,
                FIVE_PRIME_WINDOW,
                PRUNE_MAX_DELTA_AS,
                MIN_WEIGHT,
                None,
                Some(&mut params),
                None,
            )
            .unwrap_or_else(|e| panic!("solve failed at max_iter={max_iter}: {e}"));
            let _ = std::fs::remove_file(&profile);
            let out: Vec<(String, f64, f64, bool)> = params
                .into_iter()
                .flatten()
                .map(|p| (p.transcript_id, p.em_accum_weight, p.alpha, p.em_finalized))
                .collect();
            assert_eq!(
                out.len(),
                5,
                "all five fixture transcripts must carry a parameter at \
                 max_iter={max_iter}; got {:?}",
                out.iter().map(|r| &r.0).collect::<Vec<_>>()
            );
            out
        };

        let runs: Vec<Vec<(String, f64, f64, bool)>> =
            RUNS.iter().map(|&n| stamped_at(n)).collect();

        let mut movers: Vec<String> = Vec::new();
        for pair in runs.windows(2) {
            for ((id, w0, _, _), (id1, w1, _, _)) in pair[0].iter().zip(&pair[1]) {
                assert_eq!(
                    id, id1,
                    "transcript order must not depend on max_iter; a reorder \
                     would make the run-to-run comparison below meaningless"
                );
                if (w0 - w1).abs() > TOL {
                    movers.push(id.clone());
                }
            }
        }
        assert!(
            !movers.is_empty(),
            "this fixture is supposed to be DYNAMIC, but W is identical at every \
             max_iter in {RUNS:?}. A fixed point means W(stamped one iteration \
             late) == W(stamped now), so the identity asserted below would hold \
             no matter which iteration each column came from, and this test \
             would be green against the very defect it exists to catch. Rebuild \
             it with candidates whose mu differ from each other."
        );
        assert!(
            movers.iter().any(|m| m == "r0_tx0" || m == "r0_tx1"),
            "W must move on the AS-contested molecule specifically; only it has \
             two candidates whose posterior the fit can actually shift. Moved: {movers:?}"
        );

        for (n, run) in RUNS.iter().zip(&runs) {
            let sum_w: f64 = run.iter().map(|(_, w, _, _)| *w).sum();
            let sum_alpha: f64 = run.iter().map(|(_, _, a, _)| *a).sum();
            assert!(
                (sum_w - N_MOLECULES * sum_alpha).abs() < TOL,
                "max_iter={n}: sum(em_accum_weight) = {sum_w} but \
                 n_total_molecules * sum(alpha) = {}",
                N_MOLECULES * sum_alpha
            );
            for (id, w, a, _) in run {
                assert!(
                    (w - N_MOLECULES * a).abs() < TOL,
                    "max_iter={n}: {id} carries em_accum_weight = {w} and alpha = \
                     {a} (implying {}); the two columns are the same quantity \
                     and must be stamped from the same final E-step",
                    N_MOLECULES * a
                );
                assert!(
                    *w > 0.0,
                    "max_iter={n}: {id} has W = {w}. A zero here would satisfy \
                     every assertion above for the wrong reason."
                );
            }
        }

        for (n, run) in RUNS.iter().zip(&runs).skip(1) {
            for (id, _, _, finalized) in run {
                assert!(
                    *finalized,
                    "max_iter={n}: {id} must be em_finalized -- max_iter = 1 was \
                     its own run, and >= 2 means an M-step ran"
                );
            }
        }

        let _ = std::fs::remove_file(&emission);
    }

    #[test]
    fn test_sidecar_em_accum_weight_drops_molecules_whose_candidates_are_all_missing() {
        let emission = scratch("accum_w_gap_emission.parquet");
        let profile = scratch("accum_w_gap_profile.parquet");
        let sidecar = scratch("accum_w_gap_sidecar.parquet");

        let graph = multi_molecule_graph(
            10.0,
            &[
                vec![("a", 1000, 0), ("b", 1000, 0)],
                vec![("gone", 1000, 0)],
            ],
        );
        multi_molecule_emission(&emission, &["r0_a", "r0_b"]);

        let mut params = Vec::new();
        solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            2,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            Some(&mut params),
            None,
        )
        .unwrap();
        write_emission_params_parquet(&params, &sidecar).expect("sidecar must write");

        let (_, rows) = read_sidecar(&sidecar);
        let sum_w: f64 = rows
            .iter()
            .map(|r| sidecar_cell(r, "em_accum_weight").unwrap())
            .sum();
        let sum_alpha: f64 = rows.iter().map(|r| sidecar_cell(r, "alpha").unwrap()).sum();
        let n_total_molecules = 2.0_f64;

        assert!(
            sum_w < n_total_molecules,
            "a molecule whose every candidate is missing from the emission \
             parquet contributes 0 to W while still counting in \
             n_total_molecules, so sum(em_accum_weight) ({sum_w}) MUST fall short of {n_total_molecules}"
        );
        assert!(
            (sum_w - 1.0).abs() < 1e-12 && sum_alpha < 1.0,
            "exactly r0's weight survives: sum(em_accum_weight) = {sum_w}, \
             sum(alpha) = {sum_alpha}"
        );
        assert!(
            n_total_molecules > sum_w,
            "the missing-candidate molecule still counts as a molecule"
        );

        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);
        let _ = std::fs::remove_file(&sidecar);
    }

    #[test]
    fn test_em_finalized_is_false_when_the_run_never_reaches_an_m_step() {
        const TOL: f64 = 1e-12;

        {
            let emission = scratch("finalized_a_emission.parquet");
            let profile = scratch("finalized_a_profile.parquet");
            let sidecar = scratch("finalized_a_sidecar.parquet");

            let graph = multi_molecule_graph(
                10.0,
                &[
                    vec![("a", 1000, 0)],
                    vec![("b", 1000, 0)],
                    vec![("c", 1000, 0)],
                ],
            );
            multi_molecule_emission(&emission, &["r0_a", "r1_b", "r2_c"]);

            let mut params = Vec::new();
            solve_and_write_profile(
                graph,
                &emission,
                "TEST",
                &profile,
                1e-6,
                5, // max_iter -- generous, so a FALSE here cannot be blamed on it
                20.0,
                FIVE_PRIME_OFF,
                POLY_A_ON,
                FIVE_PRIME_WINDOW,
                PRUNE_MAX_DELTA_AS,
                MIN_WEIGHT,
                None,
                Some(&mut params),
                None,
            )
            .unwrap();
            write_emission_params_parquet(&params, &sidecar).expect("sidecar must write");

            let fin = sidecar_finalized(&sidecar);
            assert_eq!(fin.len(), 3, "three transcripts, three sidecar rows");
            assert!(
                fin.iter().all(|(_, v)| !*v),
                "n_ambig == 0 makes delta_avg_ll 0.0, so the loop breaks at \
                 iteration 0 with NO M-step: every row is the Laplace \
                 initialisation and must be em_finalized = false. Got {:?}",
                fin.iter()
                    .filter(|(_, v)| *v)
                    .map(|(t, _)| t)
                    .collect::<Vec<_>>()
            );

            let (_, rows) = read_sidecar(&sidecar);
            let sum_w: f64 = rows
                .iter()
                .map(|r| sidecar_cell(r, "em_accum_weight").unwrap())
                .sum();
            assert!(
                (sum_w - 3.0).abs() < TOL,
                "W is NOT gated on em_finalized: each unambiguous molecule \
                 contributes exactly 1.0, so sum(em_accum_weight) = 3 even \
                 though no M-step ran. Got {sum_w}"
            );
            let sum_alpha: f64 = rows.iter().map(|r| sidecar_cell(r, "alpha").unwrap()).sum();
            assert!(
                (sum_alpha - 1.0).abs() < TOL,
                "and the alpha in those same rows is still the Laplace \
                 initialisation, which is exactly what em_finalized = false \
                 is warning about; sum(alpha) = {sum_alpha}"
            );

            let _ = std::fs::remove_file(&emission);
            let _ = std::fs::remove_file(&profile);
            let _ = std::fs::remove_file(&sidecar);
        }

        {
            let emission = scratch("finalized_b_emission.parquet");
            let profile = scratch("finalized_b_profile.parquet");
            let sidecar = scratch("finalized_b_sidecar.parquet");

            let graph = multi_molecule_graph(10.0, &[vec![("a", 1000, 0), ("b", 1000, 0)]]);
            multi_molecule_emission(&emission, &["r0_a", "r0_b"]);

            let mut params = Vec::new();
            solve_and_write_profile(
                graph,
                &emission,
                "TEST",
                &profile,
                1e-6,
                1, // max_iter -- the arm `n_em_iter >= 2` gets wrong
                20.0,
                FIVE_PRIME_OFF,
                POLY_A_ON,
                FIVE_PRIME_WINDOW,
                PRUNE_MAX_DELTA_AS,
                MIN_WEIGHT,
                None,
                Some(&mut params),
                None,
            )
            .unwrap();
            write_emission_params_parquet(&params, &sidecar).expect("sidecar must write");

            let fin = sidecar_finalized(&sidecar);
            assert_eq!(fin.len(), 2, "two candidates, two sidecar rows");
            assert!(
                fin.iter().all(|(_, v)| *v),
                "a contested molecule cannot converge on iteration 0 \
                 (prev_ll is -inf, so the delta is inf), so max_iter = 1 runs \
                 exactly one M-step and n_em_iter is 1. A rule of \
                 `n_em_iter >= 2` reports this fit as an initialisation. Got {:?}",
                fin.iter()
                    .filter(|(_, v)| !*v)
                    .map(|(t, _)| t)
                    .collect::<Vec<_>>()
            );

            let _ = std::fs::remove_file(&emission);
            let _ = std::fs::remove_file(&profile);
            let _ = std::fs::remove_file(&sidecar);
        }
    }

    #[test]
    fn test_em_accum_weight_tracks_the_final_estep_not_the_m_step_before_it() {
        const TOL: f64 = 1e-12;

        let run = |tag: &str, max_iter: usize| -> (Vec<(String, f64)>, Vec<(String, f64)>) {
            let emission = scratch(&format!("stale_{tag}_emission.parquet"));
            let profile = scratch(&format!("stale_{tag}_profile.parquet"));
            let sidecar = scratch(&format!("stale_{tag}_sidecar.parquet"));

            let graph =
                multi_molecule_graph((100.0_f32).log2(), &[vec![("x", 1000, 0), ("y", 1000, 0)]]);
            write_test_emission(
                &emission,
                &[("r0_x", "g", 9.0, 0.5), ("r0_y", "g", 5.0, 0.5)],
            );

            let mut params = Vec::new();
            solve_and_write_profile(
                graph,
                &emission,
                "TEST",
                &profile,
                1e-6,
                max_iter,
                20.0,
                FIVE_PRIME_OFF,
                POLY_A_ON,
                FIVE_PRIME_WINDOW,
                PRUNE_MAX_DELTA_AS,
                0.0,
                None,
                Some(&mut params),
                None,
            )
            .unwrap();
            write_emission_params_parquet(&params, &sidecar).expect("sidecar must write");

            let mut from_profile: Vec<(String, f64)> = Vec::new();
            for (_, tx, w) in read_profile_by_read(&profile) {
                match from_profile.iter_mut().find(|(t, _)| *t == tx) {
                    Some(slot) => slot.1 += w,
                    None => from_profile.push((tx, w)),
                }
            }
            from_profile.sort_by(|a, b| a.0.cmp(&b.0));

            let (_, rows) = read_sidecar(&sidecar);
            let mut from_sidecar: Vec<(String, f64)> = rows
                .iter()
                .map(|row| {
                    (
                        row.0.clone(),
                        sidecar_cell(row, "em_accum_weight")
                            .unwrap_or_else(|| panic!("{} must carry em_accum_weight", row.0)),
                    )
                })
                .collect();
            from_sidecar.sort_by(|a, b| a.0.cmp(&b.0));

            let _ = std::fs::remove_file(&emission);
            let _ = std::fs::remove_file(&profile);
            let _ = std::fs::remove_file(&sidecar);
            (from_sidecar, from_profile)
        };

        let (sidecar_it1, _) = run("it1", 1);
        let (sidecar_it2, profile_it2) = run("it2", 2);

        let moved = sidecar_it1
            .iter()
            .zip(sidecar_it2.iter())
            .any(|((a, x), (b, y))| a == b && (x - y).abs() > 1e-9);
        assert!(
            moved,
            "VACUITY GUARD: this fixture must move W between its first and second \
             E-step, or it cannot tell the final iteration's W from the previous \
             one's. sidecar(max_iter=1) = {sidecar_it1:?}, \
             sidecar(max_iter=2) = {sidecar_it2:?}"
        );

        assert_eq!(
            profile_it2.len(),
            2,
            "both candidates must be emitted, or posterior_weight carries \
             w/retained_sum rather than w and this cross-check is meaningless"
        );
        assert_eq!(sidecar_it2.len(), 2, "two candidates, two sidecar rows");

        for ((tx, sidecar_w), (tx2, profile_w)) in sidecar_it2.iter().zip(profile_it2.iter()) {
            assert_eq!(tx, tx2, "the two reads must cover the same transcripts");
            assert!(
                (sidecar_w - profile_w).abs() < TOL,
                "{tx}: em_accum_weight = {sidecar_w:.17} but the profile's own \
                 per-read weights sum to {profile_w:.17}. The sidecar is fed \
                 from `last_estep_params`, whose snapshot predates the M-step, \
                 so the M-step's own assignment is ONE ITERATION BEHIND the \
                 weights this profile was written from. max_iter = 2 here, so \
                 these are the SECOND E-step's weights."
            );
        }
    }

    #[test]
    fn test_dw_phase_metrics_are_exactly_zero_when_there_is_no_phase_and_bounded_otherwise() {
        let solve = |tag: &str, max_iter: usize| {
            let emission = scratch(&format!("dwphase_{tag}_emission.parquet"));
            let profile = scratch(&format!("dwphase_{tag}_profile.parquet"));
            let graph =
                multi_molecule_graph((100.0_f32).log2(), &[vec![("x", 1000, 0), ("y", 1000, 0)]]);
            write_test_emission(
                &emission,
                &[("r0_x", "g", 9.0, 0.5), ("r0_y", "g", 5.0, 0.5)],
            );
            let mut params = Vec::new();
            let stats = solve_and_write_profile(
                graph,
                &emission,
                "TEST",
                &profile,
                1e-6,
                max_iter,
                20.0,
                FIVE_PRIME_OFF,
                POLY_A_ON,
                FIVE_PRIME_WINDOW,
                PRUNE_MAX_DELTA_AS,
                0.0,
                None,
                Some(&mut params),
                None,
            )
            .expect("solve must succeed");
            let _ = std::fs::remove_file(&emission);
            let _ = std::fs::remove_file(&profile);
            (
                stats.max_dw_phase_abs,
                stats.max_dw_phase_rel,
                stats.n_em_iter,
            )
        };

        let (abs1, rel1, iter1) = solve("one", 1);
        assert_eq!(iter1, 1, "the max_iter=1 arm must really run one iteration");
        assert_eq!(
            abs1, 0.0,
            "with a single E-step the M-step's W and the published W are the same \
             numbers, so the phase metric must be EXACTLY zero; got {abs1}"
        );
        assert_eq!(
            rel1, 0.0,
            "same arm, relative form must be zero too; got {rel1}"
        );

        let (abs_n, rel_n, iter_n) = solve("many", 20);
        assert!(
            iter_n > 1,
            "this fixture must actually iterate, or the assertion below is \
             measuring a fixed point and would pass for the wrong reason; got {iter_n}"
        );
        assert!(
            abs_n > 0.0,
            "an EM that moved between E-steps must report a non-zero phase gap; \
             got exactly 0.0, which means `last_mstep_w` was never populated"
        );
        assert!(
            (0.0..=1.0).contains(&rel_n),
            "the relative form divides by the LARGER of the two arms, so it is \
             bounded by 1 by construction; got {rel_n}"
        );
    }

    #[test]
    fn test_emission_prior_fields_are_bound_from_the_emission_parquet() {
        let emission = scratch("prior_binding_emission.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx_a", "g", 9.0, 0.5),
                ("tx_b", "g", 5.0, 1.25),
                ("tx_c", "g", 7.25, 3.75),
            ],
        );

        let params = load_emission_parameters(&emission).expect("emission must load");
        let get = |id: &str| {
            params
                .get(id)
                .unwrap_or_else(|| panic!("{id} must be present in the loaded params"))
        };

        assert_eq!(
            get("tx_a").sigma2_cohort.to_bits(),
            0.5_f64.to_bits(),
            "tx_a must carry ITS OWN sigma2_logPAL"
        );
        assert_eq!(
            get("tx_b").sigma2_cohort.to_bits(),
            1.25_f64.to_bits(),
            "tx_b must carry ITS OWN sigma2_logPAL, not tx_a's"
        );
        assert_eq!(
            get("tx_c").sigma2_cohort.to_bits(),
            3.75_f64.to_bits(),
            "tx_c must carry ITS OWN sigma2_logPAL, not a cohort-wide constant"
        );
        assert_eq!(get("tx_a").mu_cohort.to_bits(), 9.0_f64.to_bits());
        assert_eq!(get("tx_b").mu_cohort.to_bits(), 5.0_f64.to_bits());
        assert_eq!(get("tx_c").mu_cohort.to_bits(), 7.25_f64.to_bits());

        assert_eq!(get("tx_a").sigma2.to_bits(), 0.5_f64.to_bits());

        let s2s = [
            get("tx_a").sigma2_cohort,
            get("tx_b").sigma2_cohort,
            get("tx_c").sigma2_cohort,
        ];
        assert!(
            s2s[0] != s2s[1] && s2s[1] != s2s[2] && s2s[0] != s2s[2],
            "the three transcripts were given three different sigma2_logPAL values, \
             so the loaded priors must be three different numbers; got {s2s:?}"
        );

        let _ = std::fs::remove_file(&emission);
    }

    #[test]
    fn test_emission_loader_rejects_a_non_positive_or_non_finite_sigma2_prior() {
        for (label, bad) in [("zero", 0.0_f64), ("negative", -1.0), ("nan", f64::NAN)] {
            let emission = scratch(&format!("prior_bad_{label}_emission.parquet"));
            write_test_emission(
                &emission,
                &[("tx_ok", "g", 7.0, 0.5), ("tx_bad", "g", 6.0, bad)],
            );

            let err = load_emission_parameters(&emission)
                .expect_err(&format!("sigma2_logPAL = {bad} must be rejected at load"));
            let msg = format!("{err:#}");
            assert!(
                msg.contains("sigma2_logPAL"),
                "the error must name the offending COLUMN, not just the row; got: {msg}"
            );
            assert!(
                msg.contains("tx_bad"),
                "the error must name the offending TRANSCRIPT so an operator can \
                 find it in a 70k-row model; got: {msg}"
            );
            let _ = std::fs::remove_file(&emission);
        }
    }

    #[test]
    fn test_quant_refuses_a_placeholder_graph_with_a_positive_poly_a_weight() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("placeholder_guard_emission.parquet");
        let profile = scratch("placeholder_guard_profile.parquet");
        write_test_emission(&emission, &[("tx0", "gene0", log_pal as f64, 0.04)]);

        let mut graph = synth_graph(log_pal, &[("tx0", 1000)]);
        graph.poly_a_is_placeholder = true;

        let err = solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            1,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            None,
        )
        .expect_err("a placeholder graph at poly_a_weight > 0 is the silently-wrong pairing");

        let msg = format!("{err:#}");
        assert!(
            msg.contains("FATAL") && msg.contains("--poly-a-weight 0"),
            "the error must name the fix, or the caller cannot act on it: {msg}"
        );

        let mut graph = synth_graph(log_pal, &[("tx0", 1000)]);
        graph.poly_a_is_placeholder = true;
        solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            1,
            20.0,
            FIVE_PRIME_OFF,
            0.0,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            None,
        )
        .expect("a placeholder graph at weight 0 is the supported alignment-only run");

        let graph = synth_graph(log_pal, &[("tx0", 1000)]);
        solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            1,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            None,
        )
        .expect("a measured graph at a positive weight is the normal case");

        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);
    }

    fn synth_graph_with_gaps(log_pal: f32, candidates: &[(&str, i32, u32)]) -> FlatEvidenceGraph {
        let mut transcript_names = Vec::new();
        let mut candidate_pool = Vec::new();
        let mut candidate_as = Vec::new();
        let mut candidate_5p_gap = Vec::new();
        for (tx, as_score, gap) in candidates {
            candidate_pool.push(transcript_names.len() as u32);
            transcript_names.push(tx.to_string());
            candidate_as.push(*as_score);
            candidate_5p_gap.push(*gap);
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
            candidate_5p_gap,
            transcript_names,
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        }
    }

    fn census_fixture(spec: &[(i32, bool, f64)], present: &[bool]) -> CandidateAvailability {
        let with_gaps: Vec<(&str, i32, u32)> =
            spec.iter().map(|&(as_, _, _)| ("t", as_, 0)).collect();
        let graph = synth_graph_with_gaps(7.0, &with_gaps);
        let params: Vec<Option<TranscriptParam>> = spec
            .iter()
            .zip(present)
            .map(|(&(_, competent, alpha), &keep)| {
                keep.then(|| TranscriptParam {
                    transcript_id: "t".into(),
                    gene_id: "g".into(),
                    is_competent: competent,
                    mu_cohort: 7.0,
                    mu: 7.0,
                    sigma2_cohort: 0.04,
                    sigma2: 0.04,
                    alpha,
                    em_accum_weight: 0.0, // fixture never runs an M-step
                    em_finalized: false,
                    is_imputed: false, // test fixture: every row is anchored
                })
            })
            .collect();
        candidate_availability(&graph, &graph.molecules[0], &params, PRUNE_MAX_DELTA_AS)
    }

    fn multi_molecule_graph(
        log_pal: f32,
        molecules: &[Vec<(&str, i32, u32)>],
    ) -> FlatEvidenceGraph {
        let mut g = FlatEvidenceGraph {
            molecules: Vec::new(),
            candidate_nm: Vec::new(),
            candidate_mapq: Vec::new(),
            candidate_pool: Vec::new(),
            candidate_as: Vec::new(),
            candidate_5p_gap: Vec::new(),
            transcript_names: Vec::new(),
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        };
        for (mi, cands) in molecules.iter().enumerate() {
            let offset = g.candidate_pool.len() as u32;
            for (tx, as_score, gap) in cands {
                let name = format!("r{mi}_{tx}");
                g.candidate_pool.push(g.transcript_names.len() as u32);
                g.transcript_names.push(name);
                g.candidate_as.push(*as_score);
                g.candidate_5p_gap.push(*gap);
                g.candidate_nm.push(0);
                g.candidate_mapq.push(60);
            }
            g.molecules.push(MoleculeEvidence {
                read_id: format!("r{mi}"),
                log_pal,
                candidate_offset: offset,
                candidate_count: cands.len() as u16,
            });
        }
        g
    }

    fn multi_molecule_emission(path: &Path, wants: &[&str]) {
        let rows: Vec<(&str, &str, f64, f64)> = wants.iter().map(|t| (*t, "g", 7.0, 0.5)).collect();
        write_test_emission(path, &rows);
    }

    #[test]
    fn test_candidate_ledger_claims_every_candidate_exactly_once() {
        let emission = local_scratch("ledger_emission.parquet");
        let profile = local_scratch("ledger_profile.parquet");
        let graph = multi_molecule_graph(
            10.0,
            &[
                vec![("solo", 1000, 0)],
                vec![("a", 1000, 0), ("b", 1000, 0)],
                vec![("live", 1000, 0), ("bad", 500, 0)],
                vec![("gone", 1000, 0)],
            ],
        );
        multi_molecule_emission(&emission, &["r0_solo", "r1_a", "r1_b", "r2_live"]);

        let stats = solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            5,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            None,
        )
        .expect("solve must succeed; the in-library tripwire would have fired otherwise");
        let l = stats.ledger;

        assert_eq!(
            l.deficit(),
            0,
            "every candidate must have exactly one terminal fate. Ledger: {l:?}"
        );
        assert_eq!(l.total_candidates, 6, "1 + 2 + 2 + 1 candidates");
        assert_eq!(l.pruned_delta_as, 1, "r2_bad: delta_as = 500 > 50");
        assert_eq!(l.missing, 0, "the pruned candidate is never looked up");
        assert_eq!(l.incompetent, 0);
        assert_eq!(l.alpha_zero, 0);
        assert_eq!(l.pruned_min_weight, 0, "MIN_WEIGHT = 0.01 on 0.5/0.5 pairs");
        assert_eq!(l.dropped_output_competence, 0);
        assert_eq!(l.rows_emitted, 3, "r1_a, r1_b, r2_live");
        assert_eq!(l.unambig_emitted, 1, "r0_solo");
        assert_eq!(l.unambig_missing, 1, "r3_gone");
        assert_eq!(l.unambig_incompetent, 0);
        assert_eq!(
            l.census_vs_estep_deficit(),
            0,
            "the census replayed {} live candidates, the E-step carried {}",
            l.live_candidates,
            l.live_candidates_entered
        );
        assert_eq!(l.live_candidates, 3, "r1_a, r1_b, r2_live");
    }

    #[test]
    fn test_candidate_ledger_attributes_min_weight_culls() {
        let emission = local_scratch("ledger_mw_emission.parquet");
        let profile = local_scratch("ledger_mw_profile.parquet");
        let graph = multi_molecule_graph(
            10.0,
            &[vec![("a", 1000, 0)], vec![("x", 1000, 0), ("y", 1000, 0)]],
        );
        multi_molecule_emission(&emission, &["r0_a", "r1_x", "r1_y"]);

        let stats = solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            5,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            0.6,
            None,
            None,
            None,
        )
        .unwrap();
        let l = stats.ledger;
        assert_eq!(l.deficit(), 0, "ledger: {l:?}");
        assert_eq!(
            l.pruned_min_weight, 1,
            "only the LOSING 0.5-weight candidate is culled-and-not-emitted; \
             the argmax is culled-but-emitted and must not be counted here"
        );
        assert_eq!(l.rows_emitted, 1, "the all-culled read is still published");
        assert_eq!(
            l.reads_retained_by_argmax, 1,
            "and the row it was published under is a forced one"
        );
        assert_eq!(l.unambig_emitted, 1, "the unambiguous read is unaffected");
    }

    #[test]
    fn test_all_culled_read_publishes_unit_mass_not_nan() {
        let emission = local_scratch("argmax_mass_emission.parquet");
        let profile = local_scratch("argmax_mass_profile.parquet");
        let graph = multi_molecule_graph(10.0, &[vec![("x", 1000, 0), ("y", 1000, 0)]]);
        multi_molecule_emission(&emission, &["r0_x", "r0_y"]);

        solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            5,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            0.6,
            None,
            None,
            None,
        )
        .unwrap();

        let rows: Vec<(String, f64)> = read_profile(&profile);
        assert_eq!(rows.len(), 1, "one forced row: {rows:?}");
        let (tx, w) = &rows[0];
        assert!(
            w.is_finite(),
            "posterior_weight must never be NaN/inf; got {w} for {tx}. A NaN here \
             would flow into every CPM in the summary without turning any test red"
        );
        assert_eq!(
            *w, 1.0,
            "the retained argmax carries the read's whole unit of mass"
        );
    }

    #[test]
    fn test_candidate_ledger_min_weight_zero_prunes_nothing() {
        let emission = local_scratch("ledger_mw0_emission.parquet");
        let profile = local_scratch("ledger_mw0_profile.parquet");
        let graph = multi_molecule_graph(
            10.0,
            &[vec![("a", 1000, 0)], vec![("x", 1000, 0), ("y", 1000, 0)]],
        );
        multi_molecule_emission(&emission, &["r0_a", "r1_x", "r1_y"]);
        let stats = solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            5,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            0.0,
            None,
            None,
            None,
        )
        .unwrap();
        let l = stats.ledger;
        assert_eq!(l.pruned_min_weight, 0, "w < 0.0 is never true");
        assert_eq!(l.deficit(), 0, "ledger: {l:?}");
        assert_eq!(l.rows_emitted, 2, "both contested candidates emit");
    }

    #[test]
    fn test_candidate_ledger_attributes_an_incompetent_candidate_end_to_end() {
        let emission = local_scratch("ledger_cause_emission.parquet");
        let profile = local_scratch("ledger_cause_profile.parquet");
        let graph = multi_molecule_graph(
            10.0,
            &[vec![("incomp", 1000, 0), ("ok", 1000, 0), ("ok2", 1000, 0)]],
        );
        let rows: [(&str, &str, f64, f64, bool); 3] = [
            ("r0_incomp", "g", 7.0, 0.5, false),
            ("r0_ok", "g", 7.0, 0.5, true),
            ("r0_ok2", "g", 7.0, 0.5, true),
        ];
        write_test_emission_with_competence(&emission, &rows);

        let mut params = Vec::new();
        let stats = solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            5,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            Some(&mut params),
            None,
        )
        .unwrap();
        let l = stats.ledger;
        assert_eq!(l.deficit(), 0, "ledger: {l:?}");
        assert_eq!(
            l.incompetent, 1,
            "r0_incomp: present but is_competent == false. If this reads 0, the \
             increment was transposed with `alpha_zero` -- that is mutation M25."
        );
        assert_eq!(l.missing, 0, "every transcript has an emission row");
        assert_eq!(
            l.alpha_zero, 0,
            "structurally unreachable through the solver; see the doc comment. \
             Asserted explicitly so a future change that makes it non-zero is \
             noticed rather than silently absorbed into a passing sum."
        );
        assert_eq!(l.live_candidates, 2, "the two competent candidates");
        assert_eq!(l.rows_emitted, 2);
    }

    #[test]
    fn test_census_counts_the_prune_without_attributing_to_it() {
        let av = census_fixture(
            &[
                (100, true, 0.5), // live
                (20, true, 0.5),  // pruned: delta_as = 80 > 50
            ],
            &[true, true],
        );
        assert_eq!(av.n_pruned_delta_as, 1, "candidate 1 is pruned");
        assert_eq!(av.n_live, 1, "and it is not live");
        assert_eq!(
            av.n_pruned_delta_as + av.n_live + av.n_missing + av.n_incompetent + av.n_alpha_zero,
            2,
            "the five buckets partition the candidate range"
        );
        assert_eq!(
            av.uniform_cause(2),
            None,
            "one live + one pruned is not a uniform cause"
        );
    }

    #[test]
    fn test_max_iter_zero_is_rejected_rather_than_reported_as_all_missing() {
        let emission = local_scratch("ledger_iter0_emission.parquet");
        let profile = local_scratch("ledger_iter0_profile.parquet");
        let graph = multi_molecule_graph(
            10.0,
            &[vec![("a", 1000, 0)], vec![("x", 1000, 0), ("y", 1000, 0)]],
        );
        multi_molecule_emission(&emission, &["r0_a", "r1_x", "r1_y"]);
        let err = solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            0,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            None,
        )
        .expect_err("max_iter = 0 must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("max_iter must be >= 1"),
            "error must name the cause, got: {msg}"
        );
    }

    #[test]
    fn test_candidate_availability_keeps_four_death_causes_apart() {
        let av = census_fixture(
            &[
                (100, true, 0.5),  // 0 live
                (100, true, 0.5),  // 1 live
                (100, false, 0.5), // 2 incompetent
                (100, true, 0.0),  // 3 present, competent, alpha == 0
                (20, true, 0.5),   // 4 LIVE BUT PRUNED
                (100, true, 0.5),  // 5 no emission row
                // 7 present, incompetent AND alpha == 0: dead twice over. It must be
                // attributed to incompetence, matching the E-step's
                // `!p.is_competent || p.alpha <= 0.0` short-circuit. Without this candidate
                // the two arms could be swapped and no test would notice.
                (100, false, 0.0),
            ],
            &[true, true, true, true, true, false, true],
        );
        assert_eq!(av.n_live, 2, "0 and 1 only: candidate 4 is pruned");
        assert_eq!(av.n_missing, 1, "candidate 5 has no emission row");
        assert_eq!(
            av.n_incompetent, 2,
            "candidates 2 and 7. Candidate 7 is incompetent AND alpha == 0, and the
             E-step tests competence first, so it lands here and not in
             `n_alpha_zero` -- see the note on the order of the two arms."
        );
        assert_eq!(
            av.n_alpha_zero, 1,
            "candidate 3 is competent but alpha == 0"
        );
        assert_eq!(
            av.uniform_cause(7),
            None,
            "four different causes are present, so no single one accounts for the \
             read. Reporting any of them here would over-claim what padding fixes."
        );

        assert_eq!(
            census_fixture(&[(100, true, 0.5), (100, true, 0.5)], &[false, false]).uniform_cause(2),
            Some(GhostCause::Missing)
        );
        assert_eq!(
            census_fixture(&[(100, false, 0.5), (100, false, 0.5)], &[true, true]).uniform_cause(2),
            Some(GhostCause::Incompetent)
        );
        assert_eq!(
            census_fixture(&[(100, true, 0.0), (100, true, 0.0)], &[true, true]).uniform_cause(2),
            Some(GhostCause::AlphaZero)
        );
        assert_eq!(
            census_fixture(&[(100, true, 0.5), (20, true, 0.5)], &[false, true]).uniform_cause(2),
            None,
            "one candidate is missing and the other is pruned: the read is NOT \
             rescued by padding, because the pruned candidate dies either way"
        );
    }

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
        let graph = synth_graph(log_pal, &[("tx0", 1000), ("tx1", 800)]);

        solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            1,
            100.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_OFF,
            MIN_WEIGHT,
            None,
            None,
            None,
        )
        .unwrap();

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

    #[test]
    fn test_max_as_ratio_defined_cases() {
        let log_pal = (100.0_f32).log2();

        let single = synth_graph(log_pal, &[("tx0", 3206)]);
        assert_eq!(
            read_max_as_ratio(&single, &single.molecules[0]),
            0.0,
            "a single candidate carries no runner-up, so the ratio is 0.0"
        );

        let tagless = synth_graph(log_pal, &[("tx0", 0), ("tx1", 0), ("tx2", 0)]);
        assert_eq!(
            read_max_as_ratio(&tagless, &tagless.molecules[0]),
            0.0,
            "tag-less candidates must not produce a NaN from 0/0"
        );

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

        let negative = synth_graph(log_pal, &[("tx0", 100), ("tx1", -5)]);
        assert_eq!(
            read_max_as_ratio(&negative, &negative.molecules[0]),
            0.0,
            "a negative runner-up AS is not evidence and must not yield a negative ratio"
        );
    }

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
                (
                    "tx1",
                    "gene1",
                    log_pal as f64 - (2.0 * sigma2 * 5.0).sqrt(),
                    sigma2,
                ),
            ],
        );
        let graph = synth_graph(log_pal, &[("tx0", 900), ("tx1", 900)]);

        solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            1,
            100.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            None,
        )
        .unwrap();

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

    fn read_profile_by_read(path: &Path) -> Vec<(String, String, f64)> {
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
            let rd = batch
                .column_by_name("read_id")
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
                out.push((rd.value(i).to_string(), tx.value(i).to_string(), w.value(i)));
            }
        }
        out
    }

    fn prior_biased_graph() -> FlatEvidenceGraph {
        let log_pal = (100.0_f32).log2();
        let mut molecules = Vec::new();
        let mut candidate_pool: Vec<u32> = Vec::new();
        let mut candidate_as: Vec<i32> = Vec::new();
        let mut n = 0usize;
        for (tx, count) in [(0u32, 30usize), (1, 3usize)] {
            for i in 0..count {
                molecules.push(MoleculeEvidence {
                    read_id: format!("u{tx}_{i}"),
                    log_pal,
                    candidate_offset: n as u32,
                    candidate_count: 1,
                });
                candidate_pool.push(tx);
                candidate_as.push(900);
                n += 1;
            }
        }
        molecules.push(MoleculeEvidence {
            read_id: "PROBE".to_string(),
            log_pal,
            candidate_offset: n as u32,
            candidate_count: 2,
        });
        candidate_pool.extend_from_slice(&[0, 1]); // tx0, tx1
        candidate_as.extend_from_slice(&[900, 900]); // EQUAL alignment score
        n += 2;

        let graph = FlatEvidenceGraph {
            molecules,
            candidate_nm: vec![0; n],
            candidate_mapq: vec![60; n],
            candidate_5p_gap: vec![0; n],
            candidate_pool,
            candidate_as,
            transcript_names: vec!["tx0".to_string(), "tx1".to_string()],
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        };
        assert_eq!(graph.candidate_pool.len(), n);
        assert_eq!(graph.candidate_as.len(), n);
        assert_eq!(graph.candidate_5p_gap.len(), n);
        graph
    }

    #[test]
    fn test_e_step_reduces_to_the_abundance_prior_when_all_evidence_is_tied() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("identity_emission.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", log_pal as f64, 0.04),
                ("tx1", "gene1", log_pal as f64, 0.04),
            ],
        );

        let arms: [(f64, f64, f64, &str); 6] = [
            (0.0, FIVE_PRIME_OFF, 100.0, "prior+aln only"),
            (0.0, FIVE_PRIME_OFF, 10.0, "prior+aln only, λ=10"),
            (0.0, FIVE_PRIME_WEIGHT, 100.0, "no-polyA, 5' on"),
            (POLY_A_ON, FIVE_PRIME_OFF, 100.0, "polyA on, 5' off"),
            (POLY_A_ON, FIVE_PRIME_WEIGHT, 10.0, "Full, λ=10"),
            (POLY_A_ON, FIVE_PRIME_WEIGHT, 100.0, "Full"),
        ];

        let mut reference_odds: Option<f64> = None;
        for (poly_a_weight, five_prime_weight, lambda, label) in arms {
            let profile = scratch(&format!("identity_profile_{label}.parquet"));
            solve_and_write_profile(
                prior_biased_graph(),
                &emission,
                "TEST",
                &profile,
                1e-6,
                1, // one E-step: the posterior reflects the INIT prior
                lambda,
                five_prime_weight,
                poly_a_weight,
                FIVE_PRIME_WINDOW,
                PRUNE_MAX_DELTA_AS,
                0.0,
                None,
                None,
                None,
            )
            .unwrap();

            let rows = read_profile_by_read(&profile);
            let w = |tx: &str| -> f64 {
                rows.iter()
                    .filter(|(r, t, _)| r == "PROBE" && t == tx)
                    .map(|&(_, _, w)| w)
                    .sum()
            };
            let (w0, w1) = (w("tx0"), w("tx1"));

            assert!(
                w0 > 0.7 && w1 < 0.3,
                "[{label}] all evidence is tied, so the PROBE read must follow the \
                 abundance prior alone (~0.90 onto tx0). It got w0={w0:.4} w1={w1:.4}. \
                 A value near 0.5/0.5 means `ln alpha` is not reaching this arm's \
                 softmax: the M-step re-estimates abundances that no posterior can \
                 see, and the EM is inert."
            );

            let odds = w0 / w1;
            match reference_odds {
                None => reference_odds = Some(odds),
                Some(ref0) => assert!(
                    (odds / ref0 - 1.0).abs() < 1e-6,
                    "[{label}] posterior odds {odds:.6} differ from the reference arm's \
                     {ref0:.6}. With every evidence term tied, each weight is a constant \
                     that must cancel, so no arm may change the odds. A weight is \
                     reaching a term it must not."
                ),
            }
        }
    }

    #[test]
    fn test_m_step_alpha_is_normalised_by_the_total_molecule_count() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("alpha_norm_emission.parquet");
        let profile = scratch("alpha_norm_profile.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", log_pal as f64, 0.04),
                ("tx1", "gene1", log_pal as f64, 0.04),
            ],
        );

        let mut params = Vec::new();
        solve_and_write_profile(
            prior_biased_graph(),
            &emission,
            "TEST",
            &profile,
            1e-9,
            5,
            100.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            Some(&mut params),
            None,
        )
        .unwrap();
        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);

        let alpha: Vec<f64> = params
            .iter()
            .map(|p| {
                p.as_ref()
                    .expect("both transcripts are in the emission")
                    .alpha
            })
            .collect();

        let init_alpha: Vec<f64> = [30.01 + 0.5, 3.01 + 0.5]
            .iter()
            .map(|&c| c / (30.01 + 3.01 + 1.0))
            .collect();
        assert!(
            alpha
                .iter()
                .zip(&init_alpha)
                .any(|(a, i)| (a - i).abs() > 1e-6),
            "CONTROL: alpha must have been RE-ESTIMATED by the M-step, not left \
             at its init value. alpha={alpha:?} init={init_alpha:?}. Without this \
             the sum assertion below would be satisfied by the initialiser."
        );

        let sum: f64 = alpha.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "alpha is a mixture weight vector: the M-step divides by the TOTAL \
             molecule count (34 here) precisely so the vector sums to 1. It sums \
             to {sum}. A divisor of `n_total_molecules - 5.0` would rescale every \
             component by 34/29 = {:.4} and leave the vector unnormalised, which is \
             invisible to the E-step softmax and to every emitted column but is \
             still a wrong abundance.",
            34.0 / 29.0
        );

        assert!(
            alpha[0] > alpha[1] && (alpha[0] - alpha[1]).abs() > 0.5,
            "tx0 carries 30 unambiguous molecules against tx1's 3, so alpha must \
             stay strongly ordered: {alpha:?}"
        );
    }

    fn duplicate_every_molecule(g: &FlatEvidenceGraph) -> FlatEvidenceGraph {
        let mut molecules = Vec::with_capacity(2 * g.molecules.len());
        let mut candidate_pool = Vec::with_capacity(2 * g.candidate_pool.len());
        let mut candidate_as = Vec::with_capacity(2 * g.candidate_as.len());
        let mut candidate_nm = Vec::with_capacity(2 * g.candidate_nm.len());
        let mut candidate_mapq = Vec::with_capacity(2 * g.candidate_mapq.len());
        let mut candidate_5p_gap = Vec::with_capacity(2 * g.candidate_5p_gap.len());
        let slice = |off: u32, cnt: u16| {
            let (o, n) = (off as usize, cnt as usize);
            (o, o + n)
        };
        for copy in 0..2u32 {
            for m in &g.molecules {
                let (o, e) = slice(m.candidate_offset, m.candidate_count);
                let offset = candidate_pool.len() as u32;
                candidate_pool.extend_from_slice(&g.candidate_pool[o..e]);
                candidate_as.extend_from_slice(&g.candidate_as[o..e]);
                candidate_nm.extend_from_slice(&g.candidate_nm[o..e]);
                candidate_mapq.extend_from_slice(&g.candidate_mapq[o..e]);
                candidate_5p_gap.extend_from_slice(&g.candidate_5p_gap[o..e]);
                molecules.push(MoleculeEvidence {
                    read_id: if copy == 0 {
                        m.read_id.clone()
                    } else {
                        format!("{}_dup", m.read_id)
                    },
                    log_pal: m.log_pal,
                    candidate_offset: offset,
                    candidate_count: m.candidate_count,
                });
            }
        }
        FlatEvidenceGraph {
            molecules,
            candidate_pool,
            candidate_as,
            candidate_nm,
            candidate_mapq,
            candidate_5p_gap,
            transcript_names: g.transcript_names.clone(),
            poly_a_is_placeholder: g.poly_a_is_placeholder,
            stats: g.stats.clone(),
        }
    }

    #[test]
    fn test_alpha_is_invariant_to_duplicating_every_molecule() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("alpha_scale_emission.parquet");
        let profile = scratch("alpha_scale_profile.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", log_pal as f64, 0.04),
                ("tx1", "gene1", log_pal as f64, 0.04),
            ],
        );

        let once = prior_biased_graph();
        let twice = duplicate_every_molecule(&once);

        assert_eq!(
            once.molecules.len() * 2,
            twice.molecules.len(),
            "duplication produced {} molecules, expected {}",
            twice.molecules.len(),
            once.molecules.len() * 2
        );
        assert_eq!(2 * once.candidate_pool.len(), twice.candidate_pool.len());
        for (i, m) in twice.molecules.iter().enumerate() {
            let (o, e) = (
                m.candidate_offset as usize,
                m.candidate_offset as usize + m.candidate_count as usize,
            );
            assert!(
                e <= twice.candidate_pool.len(),
                "molecule {i} points at candidates {o}..{e} of {}",
                twice.candidate_pool.len()
            );
            let orig = &once.molecules[i % once.molecules.len()];
            assert_eq!(m.log_pal, orig.log_pal, "the copy changed log_pal");
            assert_eq!(
                m.candidate_count, orig.candidate_count,
                "the copy lost candidates"
            );
        }

        let mut params_once = Vec::new();
        solve_and_write_profile(
            once,
            &emission,
            "TEST",
            &profile,
            1e-9,
            5,
            100.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            Some(&mut params_once),
            None,
        )
        .unwrap();
        let mut params_twice = Vec::new();
        solve_and_write_profile(
            twice,
            &emission,
            "TEST",
            &profile,
            1e-9,
            5,
            100.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            Some(&mut params_twice),
            None,
        )
        .unwrap();
        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);

        let mut once_a: Vec<f64> = Vec::new();
        let mut twice_a: Vec<f64> = Vec::new();
        assert_eq!(
            params_once.len(),
            params_twice.len(),
            "duplication changed the number of transcripts with parameters"
        );
        for i in 0..params_once.len() {
            once_a.push(
                params_once[i]
                    .as_ref()
                    .expect("both transcripts are in the emission")
                    .alpha,
            );
            twice_a.push(
                params_twice[i]
                    .as_ref()
                    .expect("both transcripts are in the emission")
                    .alpha,
            );
        }

        const TOL: f64 = 1e-4;
        for (k, (a, b)) in once_a.iter().zip(&twice_a).enumerate() {
            assert!(
                (a - b).abs() <= TOL,
                "alpha is a PER-MOLECULE quantity and must not depend on how many \
                 molecules the run contained. Duplicating every read moved \
                 transcript {k} from {a:.17} to {b:.17} (delta {:.3e}, tol {TOL:e}).\n  \
                 alpha_k = W_k / N, and both W_k and N double, so the quotient is \
                 unchanged up to the prior's fixed weight (3.4e-6 measured). A \
                 divisor that is not a common factor of the total count -- \
                 `N - 5.0` gives {once_a:?} against {twice_a:?} here, an 8.6e-2 \
                 shift -- rescales every component by N/(N-5) against 2N/(2N-5) \
                 and moves abundances with sequencing depth.",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn test_poly_a_weight_zero_keeps_the_abundance_prior_in_the_e_step() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("prior_emission.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", log_pal as f64, 0.04),
                ("tx1", "gene1", log_pal as f64, 0.04),
            ],
        );

        let profile = scratch("prior_profile.parquet");
        solve_and_write_profile(
            prior_biased_graph(),
            &emission,
            "TEST",
            &profile,
            1e-6,
            1,
            100.0,
            FIVE_PRIME_OFF,
            0.0,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            0.0,
            None,
            None,
            None,
        )
        .unwrap();

        let rows = read_profile_by_read(&profile);
        let probe: f64 = rows
            .iter()
            .filter(|(r, t, _)| r == "PROBE" && t == "tx0")
            .map(|&(_, _, w)| w)
            .sum();
        assert!(
            probe > 0.7,
            "with poly(A) OFF the ambiguous PROBE read has no evidence of its own, \
             so it must follow the abundance prior (~0.9 onto tx0). It got {probe}. \
             If this is ~0.5, `ln alpha` is being scaled by poly_a_weight and the \
             M-step cannot influence any posterior -- the EM is inert."
        );
    }

    #[test]
    fn test_poly_a_weight_zero_removes_the_poly_a_term_entirely() {
        let log_pal = (100.0_f32).log2();
        let sigma2 = 0.04;
        let emission = scratch("paw_emission.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", log_pal as f64, sigma2),
                ("tx1", "gene1", log_pal as f64 + 4.0, sigma2),
            ],
        );
        let cands = [("tx0", 900), ("tx1", 900)];

        let pi_tx0 = |poly_a_weight: f64, tag: &str| -> f64 {
            let profile = scratch(&format!("paw_profile_{tag}.parquet"));
            solve_and_write_profile(
                synth_graph(log_pal, &cands),
                &emission,
                "TEST",
                &profile,
                1e-6,
                1,
                100.0,
                FIVE_PRIME_OFF,
                poly_a_weight,
                FIVE_PRIME_WINDOW,
                PRUNE_MAX_DELTA_AS,
                0.0,
                None,
                None,
                None,
            )
            .unwrap();
            let rows = read_profile(&profile);
            let tot: f64 = rows.iter().map(|&(_, w)| w).sum();
            assert!(tot > 0.0, "{tag}: profile is empty, every read was dropped");
            rows.iter()
                .filter(|(t, _)| t == "tx0")
                .map(|&(_, w)| w)
                .sum::<f64>()
                / tot
        };

        let on = pi_tx0(POLY_A_ON, "on");
        let off = pi_tx0(0.0, "off");

        assert!(
            on > 0.9,
            "with poly(A) ON, tx0 (whose mu matches the read) must win decisively, got {on}"
        );
        assert!(
            (off - 0.5).abs() < 0.02,
            "with poly_a_weight=0 the poly(A) term must vanish and the posterior \
             must be a dead heat, got {off}. If this fails the parameter is being \
             ignored or applied to the wrong term, and the alignment-only \
             baseline would silently still contain the poly(A) evidence."
        );
    }

    #[test]
    fn test_zero_min_weight_disables_truncation_and_still_renormalizes() {
        let log_pal = (100.0_f32).log2();
        let sigma2 = 0.04;
        let emission = scratch("mw0_emission.parquet");
        let profile_cut = scratch("mw0_profile_cut.parquet");
        let profile_zero = scratch("mw0_profile_zero.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", log_pal as f64, sigma2),
                (
                    "tx1",
                    "gene1",
                    log_pal as f64 - (2.0 * sigma2 * 5.0).sqrt(),
                    sigma2,
                ),
            ],
        );
        let cands = [("tx0", 900), ("tx1", 900)];

        for (path, min_weight) in [(&profile_cut, MIN_WEIGHT), (&profile_zero, 0.0)] {
            solve_and_write_profile(
                synth_graph(log_pal, &cands),
                &emission,
                "TEST",
                path,
                1e-6,
                1,
                100.0,
                FIVE_PRIME_OFF,
                POLY_A_ON,
                FIVE_PRIME_WINDOW,
                PRUNE_MAX_DELTA_AS,
                min_weight,
                None,
                None,
                None,
            )
            .unwrap();
        }

        let cut = read_profile(&profile_cut);
        let zero = read_profile(&profile_zero);
        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile_cut);
        let _ = std::fs::remove_file(&profile_zero);

        assert_eq!(
            zero.len(),
            cut.len() + 1,
            "min_weight = 0 must retain the candidate the default cutoff drops \
             (cut={cut:?}, zero={zero:?})"
        );
        assert!(
            zero.iter().any(|&(_, w)| w < MIN_WEIGHT),
            "at min_weight = 0 a sub-cutoff weight must be emitted verbatim, \
             otherwise 0 is not actually 'no truncation': {zero:?}"
        );
        assert_eq!(
            zero.iter().map(|&(_, w)| w).sum::<f64>(),
            1.0,
            "even with no truncation the per-read posterior must sum to 1.0"
        );
    }

    #[test]
    fn test_incompetent_candidate_is_excluded_from_the_renormalisation_base() {
        let log_pal = (100.0_f32).log2();
        let sigma2 = 0.04;
        let emission = scratch("incomp_emission.parquet");
        let profile = scratch("incomp_profile.parquet");
        write_test_emission_with_competence(
            &emission,
            &[
                (
                    "tx0",
                    "gene0",
                    log_pal as f64 - (2.0_f64 * sigma2 * 5.0_f64).sqrt(),
                    sigma2,
                    true,
                ),
                ("tx1", "gene1", log_pal as f64, sigma2, false),
            ],
        );
        let graph = synth_graph(log_pal, &[("tx0", 900), ("tx1", 900)]);

        let stats = solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            1,
            100.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            None,
        )
        .unwrap();

        let rows = read_profile(&profile);
        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);

        assert_eq!(
            rows.len(),
            1,
            "the incompetent candidate must not be emitted: {rows:?}"
        );
        assert!(
            !rows.iter().any(|(t, _)| t == "tx1"),
            "tx1 is is_polyA_competent=false and must never reach the profile"
        );
        assert_eq!(
            rows.iter().map(|&(_, w)| w).sum::<f64>(),
            1.0,
            "a dropped-incompetent candidate must not leave stranded mass in the \
             denominator -- emitted weights would sum to {} instead of 1.0",
            rows.iter().map(|&(_, w)| w).sum::<f64>()
        );
        assert_eq!(
            rows[0].1, 1.0,
            "the lone survivor must absorb the whole posterior, not 0.9933"
        );
        let _ = stats;
    }

    #[test]
    fn test_retained_sum_covers_exactly_the_rows_emitted_when_all_three_exclusions_fire() {
        let log_pal = (100.0_f32).log2();
        let sigma2 = 0.04;
        let emission = scratch("retained_emission.parquet");
        let profile = scratch("retained_profile.parquet");
        write_test_emission_with_competence(
            &emission,
            &[
                (
                    "tx0",
                    "gene0",
                    log_pal as f64 - (2.0_f64 * sigma2 * 5.0).sqrt(),
                    sigma2,
                    true,
                ),
                (
                    "tx1",
                    "gene1",
                    log_pal as f64 - (2.0 * sigma2 * (5.0 + (0.80_f64 / 0.15).ln())).sqrt(),
                    sigma2,
                    true,
                ),
                (
                    "tx2",
                    "gene2",
                    log_pal as f64 - (2.0 * sigma2 * (5.0 + (0.80_f64 / 0.005).ln())).sqrt(),
                    sigma2,
                    true,
                ),
                ("tx3", "gene3", log_pal as f64, sigma2, false),
            ],
        );
        let graph = synth_graph(
            log_pal,
            &[
                ("tx0", 900),
                ("tx1", 900),
                ("tx2", 900),
                ("tx3", 900),
                ("tx4", 900),
            ],
        );

        solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            1,
            100.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            None,
        )
        .unwrap();

        let rows = read_profile_by_read(&profile);
        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);

        let w = |tx: &str| -> f64 {
            rows.iter()
                .filter(|(r, t, _)| r == "r1" && t == tx)
                .map(|&(_, _, w)| w)
                .sum()
        };
        let emitted: Vec<&str> = rows.iter().map(|(_, t, _)| t.as_str()).collect();
        assert_eq!(
            emitted.len(),
            2,
            "exactly tx0 and tx1 may be emitted -- tx2 is truncated, tx3 is \
             incompetent, tx4 has no emission row: {rows:?}"
        );
        assert!(
            !rows.iter().any(|(_, t, _)| t == "tx2"),
            "tx2's weight is 0.005/0.955 = 0.0052, below MIN_WEIGHT = {MIN_WEIGHT}, \
             so it must be truncated: {rows:?}"
        );
        assert!(
            !rows.iter().any(|(_, t, _)| t == "tx3"),
            "tx3 is is_polyA_competent=false and must never reach the profile"
        );
        assert!(
            !rows.iter().any(|(_, t, _)| t == "tx4"),
            "tx4 has no emission row, so it can carry neither weight nor mass"
        );

        let sum: f64 = rows.iter().map(|&(_, _, w)| w).sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "the emitted posterior must sum to 1.0, got {sum}. A denominator \
             taken over all of `weights` instead of the retained ones leaves the \
             truncated 0.005 stranded here."
        );

        assert!(
            (w("tx0") - 0.842_105).abs() < 1e-5,
            "tx0 must absorb the truncated mass: 0.80/0.95 = 0.842105, got {}",
            w("tx0")
        );
        assert!(
            (w("tx1") - 0.157_895).abs() < 1e-5,
            "tx1 must be renormalised by the same denominator: 0.15/0.95 = \
             0.157895, got {}",
            w("tx1")
        );
    }

    #[test]
    fn test_disjoint_id_spaces_are_refused_before_any_profile_is_written() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("idspace_emission.parquet");
        let profile = scratch("idspace_profile.parquet");
        write_test_emission(&emission, &[("TX_999", "gene9", log_pal as f64, 0.04)]);
        let graph = synth_graph(log_pal, &[("tx0", 900)]);

        let err = solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            1,
            100.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            None,
        )
        .unwrap_err();
        let _ = std::fs::remove_file(&emission);

        let msg = err.to_string();
        assert!(
            msg.contains("ID-SPACE MISMATCH"),
            "a zero-overlap emission must be refused by name, got: {msg}"
        );
        assert!(
            !profile.exists(),
            "the refusal must happen before the profile writer opens the file, \
             otherwise the empty profile is already on disk by the time we bail"
        );
    }

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
                (
                    "tx1",
                    "gene1",
                    log_pal as f64 - (2.0 * sigma2 * (0.80_f64 / 0.15).ln()).sqrt(),
                    sigma2,
                ),
                (
                    "tx2",
                    "gene2",
                    log_pal as f64 - (2.0 * sigma2 * (0.80_f64 / 0.005).ln()).sqrt(),
                    sigma2,
                ),
            ],
        );
        let graph = synth_graph(log_pal, &[("tx0", 900), ("tx1", 900), ("tx2", 900)]);

        solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            1,
            100.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            None,
        )
        .unwrap();

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

    #[test]
    fn test_parameter_validation_rejects_bad_values_and_accepts_the_off_switch() {
        let log_pal = (100.0_f32).log2();
        let missing = scratch("definitely_not_here.parquet");
        let profile = scratch("validation_profile.parquet");

        let run = |lambda: f64, w5p: f64, window: u32, prune: i32| {
            solve_and_write_profile(
                synth_graph(log_pal, &[("tx0", 900), ("tx1", 800)]),
                &missing,
                "TEST",
                &profile,
                1e-6,
                1,
                lambda,
                w5p,
                POLY_A_ON,
                window,
                prune,
                MIN_WEIGHT,
                None,
                None,
                None,
            )
            .map(|_| ())
            .map_err(|e| e.to_string())
        };

        for (lambda, w5p, window, prune, label) in [
            (20.0, 0.0, 20, 50, "5' off switch"),
            (20.0, 1.8, 20, i32::MAX, "prune off"),
            (
                f64::INFINITY,
                0.0,
                20,
                50,
                "infinite lambda (as on the CLI)",
            ),
        ] {
            let err = run(lambda, w5p, window, prune).unwrap_err();
            assert!(
                !err.contains("must be"),
                "{label} must pass validation, got: {err}"
            );
        }

        for (lambda, w5p, window, prune, needle, label) in [
            (0.0, 1.8, 20, 50, "> 0", "λ = 0"),
            (f64::NAN, 1.8, 20, 50, "> 0", "λ = NaN"),
            (20.0, -1.0, 20, 50, ">= 0", "negative 5' weight"),
            (20.0, f64::NAN, 20, 50, ">= 0", "NaN 5' weight"),
            (20.0, 1.8, 20, -1, ">= 0", "negative prune threshold"),
        ] {
            let err = run(lambda, w5p, window, prune).unwrap_err();
            assert!(
                err.contains(needle),
                "{label} must be rejected with a message containing {needle:?}, got: {err}"
            );
        }
    }

    fn two_candidate_posteriors(
        name: &str,
        candidates: &[(&str, i32, u32)],
        aln_weight_lambda: f64,
        five_prime_weight: f64,
        prune_max_delta_as: i32,
    ) -> Vec<(String, f64)> {
        let log_pal = (100.0_f32).log2();
        let emission = scratch(&format!("{name}_emission.parquet"));
        let profile = scratch(&format!("{name}_profile.parquet"));
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", log_pal as f64, 0.04),
                ("tx1", "gene1", log_pal as f64, 0.04),
            ],
        );
        let graph = synth_graph_with_gaps(log_pal, candidates);
        solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            1,
            aln_weight_lambda,
            five_prime_weight,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            prune_max_delta_as,
            MIN_WEIGHT,
            None,
            None,
            None,
        )
        .unwrap();
        let rows = read_profile(&profile);
        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);
        rows
    }

    #[test]
    fn test_five_prime_term_shifts_posterior_toward_the_anchored_candidate() {
        let rows = two_candidate_posteriors(
            "five_prime_on",
            &[("tx0", 900, 5), ("tx1", 900, 500)],
            20.0,
            FIVE_PRIME_WEIGHT,
            PRUNE_MAX_DELTA_AS,
        );
        assert_eq!(
            rows.len(),
            2,
            "equal AS means neither candidate is pruned or truncated: {rows:?}"
        );
        let w_anchored = weight_of(&rows, "tx0");
        let w_anchored_as_lo = weight_of(&rows, "tx1");
        assert!(
            (w_anchored - 0.858_149).abs() < 1e-5,
            "a 1.8-nat 5' bonus must give 0.8581, got {w_anchored}"
        );
        assert!(
            (w_anchored_as_lo - 0.141_851).abs() < 1e-5,
            "the 3'-anchored candidate must take the complement, got {w_anchored_as_lo}"
        );
        assert!(
            (w_anchored + w_anchored_as_lo - 1.0).abs() < 1e-12,
            "emitted posterior weights must sum to 1.0"
        );
    }

    #[test]
    fn test_redrawing_the_5p_start_restores_the_flat_posterior() {
        let reference_rows = two_candidate_posteriors(
            "five_prime_off_ref",
            &[("tx0", 900, 500), ("tx1", 900, 500)],
            20.0,
            FIVE_PRIME_OFF,
            PRUNE_MAX_DELTA_AS,
        );
        let (r0, r1) = (
            weight_of(&reference_rows, "tx0"),
            weight_of(&reference_rows, "tx1"),
        );
        assert!(
            (r0 - 0.5).abs() < 1e-12 && (r1 - 0.5).abs() < 1e-12,
            "with the 5' term off and equal AS the posterior must be flat, got {r0} / {r1}"
        );

        let outside = two_candidate_posteriors(
            "five_prime_redraw_outside",
            &[("tx0", 900, 500), ("tx1", 900, 500)],
            20.0,
            FIVE_PRIME_WEIGHT,
            PRUNE_MAX_DELTA_AS,
        );
        let (o0, o1) = (weight_of(&outside, "tx0"), weight_of(&outside, "tx1"));
        assert!(
            (o0 - 0.5).abs() < 1e-12 && (o1 - 0.5).abs() < 1e-12,
            "a redraw outside the window must restore the flat posterior, got {o0} / {o1}"
        );

        let inside = two_candidate_posteriors(
            "five_prime_redraw_inside",
            &[("tx0", 900, 7), ("tx1", 900, 7)],
            20.0,
            FIVE_PRIME_WEIGHT,
            PRUNE_MAX_DELTA_AS,
        );
        let (i0, i1) = (weight_of(&inside, "tx0"), weight_of(&inside, "tx1"));
        assert!(
            (i0 - 0.5).abs() < 1e-12 && (i1 - 0.5).abs() < 1e-12,
            "a constant ln_5p must cancel inside the log-sum-exp, got {i0} / {i1}"
        );
    }

    #[test]
    fn test_five_prime_window_is_inclusive_at_20() {
        let at_edge = two_candidate_posteriors(
            "gap_window_edge",
            &[("tx0", 900, 20), ("tx1", 900, 900)],
            20.0,
            FIVE_PRIME_WEIGHT,
            PRUNE_MAX_DELTA_AS,
        );
        let w_edge = weight_of(&at_edge, "tx0");
        assert!(
            (w_edge - 0.858_149).abs() < 1e-5,
            "g_k = 20 is inside the `<= 20` window and must earn the bonus, got {w_edge}"
        );

        let just_outside = two_candidate_posteriors(
            "gap_window_21",
            &[("tx0", 900, 21), ("tx1", 900, 900)],
            20.0,
            FIVE_PRIME_WEIGHT,
            PRUNE_MAX_DELTA_AS,
        );
        let w_21 = weight_of(&just_outside, "tx0");
        assert!(
            (w_21 - 0.5).abs() < 1e-12,
            "g_k = 21 is outside the `<= 20` window and must earn nothing, got {w_21}"
        );
    }

    #[test]
    fn test_lambda_20_pins_the_alignment_posterior() {
        let rows = two_candidate_posteriors(
            "lambda20",
            &[("tx0", 1000, 900), ("tx1", 960, 900)],
            20.0,
            FIVE_PRIME_WEIGHT,
            PRUNE_MAX_DELTA_AS,
        );
        assert_eq!(rows.len(), 2, "ΔAS = 40 must not be pruned: {rows:?}");
        let w_high = weight_of(&rows, "tx0");
        let w_low = weight_of(&rows, "tx1");
        assert!(
            (w_high - 0.880_797).abs() < 1e-5,
            "ΔAS = 40 at λ = 20 is a 2.0-nat gap → 0.8808, got {w_high}"
        );
        assert!(
            (w_high + w_low - 1.0).abs() < 1e-12,
            "weights must sum to 1.0"
        );
    }

    #[test]
    fn test_delta_as_prune_drops_the_candidate_and_keeps_weights_normalised() {
        let candidates: &[(&str, i32, u32)] = &[("tx0", 1000, 900), ("tx1", 800, 900)];

        let pruned_rows = two_candidate_posteriors(
            "prune_on",
            candidates,
            20.0,
            FIVE_PRIME_OFF,
            PRUNE_MAX_DELTA_AS,
        );
        assert_eq!(
            pruned_rows.len(),
            1,
            "ΔAS = 200 > 50 must delete the weaker candidate outright: {pruned_rows:?}"
        );
        let w_kept = weight_of(&pruned_rows, "tx0");
        assert!(
            (w_kept - 1.0).abs() < 1e-12,
            "the surviving candidate must absorb all mass, got {w_kept}"
        );
        assert!(
            (pruned_rows.iter().map(|&(_, w)| w).sum::<f64>() - 1.0).abs() < 1e-12,
            "a pruned candidate must not leak mass into the retained denominator"
        );

        let unpruned_rows =
            two_candidate_posteriors("prune_off", candidates, 20.0, FIVE_PRIME_OFF, PRUNE_OFF);
        assert_eq!(
            unpruned_rows.len(),
            1,
            "ΔAS = 200 is 10 nats, below MIN_WEIGHT whether or not it is pruned: \
             {unpruned_rows:?}"
        );
        assert!(
            (weight_of(&unpruned_rows, "tx0") - 1.0).abs() < 1e-12,
            "truncation also renormalises to 1.0: {unpruned_rows:?}"
        );

        let keepable: &[(&str, i32, u32)] = &[("tx0", 1000, 900), ("tx1", 940, 900)];
        let pruned_60 = two_candidate_posteriors(
            "prune_60_on",
            keepable,
            20.0,
            FIVE_PRIME_OFF,
            PRUNE_MAX_DELTA_AS,
        );
        assert_eq!(
            pruned_60.len(),
            1,
            "ΔAS = 60 > 50 must be pruned even though A_rk alone would keep it: \
             {pruned_60:?}"
        );
        assert!(
            (weight_of(&pruned_60, "tx0") - 1.0).abs() < 1e-12,
            "the survivor absorbs everything: {pruned_60:?}"
        );

        let unpruned_60 =
            two_candidate_posteriors("prune_60_off", keepable, 20.0, FIVE_PRIME_OFF, PRUNE_OFF);
        assert_eq!(
            unpruned_60.len(),
            2,
            "without the prune the same candidate is emitted as a live hypothesis: \
             {unpruned_60:?}"
        );
        let w0 = weight_of(&unpruned_60, "tx0");
        let w1 = weight_of(&unpruned_60, "tx1");
        assert!(
            (w0 - 0.952_574).abs() < 1e-5,
            "−3.0 nats of A_rk must give 0.9526 on its own, got {w0}"
        );
        assert!(
            (w0 + w1 - 1.0).abs() < 1e-12,
            "emitted weights must sum to 1.0, got {w0} + {w1}"
        );
    }

    #[test]
    fn test_prune_boundary_is_strictly_greater_than() {
        let rows_at = two_candidate_posteriors(
            "prune_at_50",
            &[("tx0", 1000, 900), ("tx1", 950, 900)],
            20.0,
            FIVE_PRIME_OFF,
            PRUNE_MAX_DELTA_AS,
        );
        assert_eq!(
            rows_at.len(),
            2,
            "ΔAS = 50 is NOT pruned (`> 50`, not `>= 50`): {rows_at:?}"
        );
        let w_at = weight_of(&rows_at, "tx0");
        assert!(
            (w_at - 0.924_142).abs() < 1e-5,
            "a surviving ΔAS = 50 candidate must still cost 2.5 nats, got {w_at}"
        );
        assert!(
            (w_at + weight_of(&rows_at, "tx1") - 1.0).abs() < 1e-12,
            "the retained pair must sum to 1.0: {rows_at:?}"
        );

        let rows_51 = two_candidate_posteriors(
            "prune_at_51",
            &[("tx0", 1000, 900), ("tx1", 949, 900)],
            20.0,
            FIVE_PRIME_OFF,
            PRUNE_MAX_DELTA_AS,
        );
        assert_eq!(rows_51.len(), 1, "ΔAS = 51 must be pruned: {rows_51:?}");
    }

    fn mixed_unambiguous_then_ambiguous(log_pal: f32) -> FlatEvidenceGraph {
        let mut molecules = vec![
            MoleculeEvidence {
                read_id: "UNAMB".to_string(),
                log_pal,
                candidate_offset: 0,
                candidate_count: 1,
            },
            MoleculeEvidence {
                read_id: "AMB_TX0".to_string(),
                log_pal,
                candidate_offset: 1,
                candidate_count: 2,
            },
            MoleculeEvidence {
                read_id: "AMB_TX1".to_string(),
                log_pal,
                candidate_offset: 3,
                candidate_count: 2,
            },
        ];
        molecules.shrink_to_fit();
        let mut candidate_pool = vec![2, 0, 1, 1, 0];
        candidate_pool.shrink_to_fit();
        let candidate_as = vec![1000, 900, 870, 900, 860];
        let n = candidate_pool.len();
        FlatEvidenceGraph {
            molecules,
            candidate_nm: vec![0; n],
            candidate_mapq: vec![60; n],
            candidate_5p_gap: vec![0; n],
            candidate_pool,
            candidate_as,
            transcript_names: vec!["tx0".to_string(), "tx1".to_string(), "tx2".to_string()],
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        }
    }

    fn plain_and_min_weight_culled(log_pal: f32) -> FlatEvidenceGraph {
        let mut molecules = vec![
            MoleculeEvidence {
                read_id: "PLAIN".to_string(),
                log_pal,
                candidate_offset: 0,
                candidate_count: 1,
            },
            MoleculeEvidence {
                read_id: "CULLED".to_string(),
                log_pal,
                candidate_offset: 1,
                candidate_count: 2,
            },
        ];
        molecules.shrink_to_fit();
        let mut candidate_pool = vec![0, 1, 2];
        candidate_pool.shrink_to_fit();
        let n = candidate_pool.len();
        FlatEvidenceGraph {
            molecules,
            candidate_nm: vec![0; n],
            candidate_mapq: vec![60; n],
            candidate_5p_gap: vec![0; n],
            candidate_pool,
            candidate_as: vec![900, 900, 900],
            transcript_names: vec!["tx0".to_string(), "tx1".to_string(), "tx2".to_string()],
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        }
    }

    fn ghost_read_beside_a_live_one(log_pal: f32) -> FlatEvidenceGraph {
        let mut molecules = vec![
            MoleculeEvidence {
                read_id: "LIVE".to_string(),
                log_pal,
                candidate_offset: 0,
                candidate_count: 1,
            },
            MoleculeEvidence {
                read_id: "GHOST".to_string(),
                log_pal,
                candidate_offset: 1,
                candidate_count: 2,
            },
        ];
        molecules.shrink_to_fit();
        let mut candidate_pool = vec![0, 1, 2];
        candidate_pool.shrink_to_fit();
        let n = candidate_pool.len();
        FlatEvidenceGraph {
            molecules,
            candidate_nm: vec![0; n],
            candidate_mapq: vec![60; n],
            candidate_5p_gap: vec![0; n],
            candidate_pool,
            candidate_as: vec![900, 900, 900],
            transcript_names: vec!["tx0".to_string(), "tx1".to_string(), "tx2".to_string()],
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        }
    }

    #[test]
    fn test_tied_posterior_resolves_to_the_first_candidate_in_pool_order() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("tiebreak_emission.parquet");
        let profile = scratch("tiebreak_profile.parquet");
        write_test_emission(
            &emission,
            &[("tx0", "gene0", 7.0, 0.5), ("tx1", "gene1", 7.0, 0.5)],
        );

        let graph = FlatEvidenceGraph {
            molecules: vec![MoleculeEvidence {
                read_id: "TIED".to_string(),
                log_pal,
                candidate_offset: 0,
                candidate_count: 2,
            }],
            candidate_nm: vec![0, 0],
            candidate_mapq: vec![60, 60],
            candidate_pool: vec![0, 1],
            candidate_as: vec![900, 900],
            candidate_5p_gap: vec![0, 0],
            transcript_names: vec!["tx0".to_string(), "tx1".to_string()],
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        };

        let mut assigns: Vec<ReadAssignment> = Vec::new();
        solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            5,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            Some(&mut assigns),
        )
        .unwrap();

        assert_eq!(assigns.len(), 1, "one read in, one assignment out");
        let a = &assigns[0];
        assert_eq!(a.read_id, "TIED");

        assert!(
            (a.top_weight - 0.5).abs() < 1e-12,
            "the posterior must actually be tied at 0.5/0.5, got top_weight = {} \
             -- if this is 1.0 the fixture collapsed to one candidate and the \
             winner assertion below would be vacuous",
            a.top_weight
        );
        let ln2 = 2.0_f64.ln();
        assert!(
            (a.entropy - ln2).abs() < 1e-9,
            "a 0.5/0.5 posterior has entropy ln 2 = {ln2}, got {} -- same \
             non-vacuity reason as the top_weight assertion",
            a.entropy
        );

        assert_eq!(
            a.top_tx_id.as_deref(),
            Some("tx0"),
            "on an exact tie the FIRST candidate in pool order must win \
             (strict `>` in the argmax fold). Getting tx1 here means the \
             comparison was relaxed to `>=`, which is mutation M9."
        );
    }

    #[test]
    fn test_read_assignments_attach_each_posterior_to_its_own_read() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("readassign_emission.parquet");
        let profile = scratch("readassign_profile.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", 7.0, 0.5),
                ("tx1", "gene1", 7.0, 0.5),
                ("tx2", "gene2", 7.0, 0.5),
            ],
        );

        let mut assigns: Vec<ReadAssignment> = Vec::new();
        solve_and_write_profile(
            mixed_unambiguous_then_ambiguous(log_pal),
            &emission,
            "TEST",
            &profile,
            1e-6,
            5,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            Some(&mut assigns),
        )
        .unwrap();

        let by_id: std::collections::HashMap<&str, &ReadAssignment> =
            assigns.iter().map(|a| (a.read_id.as_str(), a)).collect();

        assert_eq!(
            assigns.len(),
            3,
            "one row per read expected, got {}: {assigns:#?}",
            assigns.len()
        );

        let u = by_id.get("UNAMB").expect("UNAMB must be present");
        assert_eq!(
            u.top_tx_id.as_deref(),
            Some("tx2"),
            "the unambiguous read's only candidate must be its top-1"
        );
        assert_eq!(u.entropy, 0.0, "a one-candidate posterior is deterministic");
        assert!(
            (u.top_weight - 1.0).abs() < 1e-12,
            "a one-candidate posterior has weight 1.0, got {}",
            u.top_weight
        );
        assert_eq!(u.n_candidates_before_cutoff, 1);

        let a0 = by_id.get("AMB_TX0").expect("AMB_TX0 must be present");
        let a1 = by_id.get("AMB_TX1").expect("AMB_TX1 must be present");
        assert_eq!(
            a0.top_tx_id.as_deref(),
            Some("tx0"),
            "AMB_TX0's top-1 must be ITS first candidate, not a neighbour's"
        );
        assert_eq!(
            a1.top_tx_id.as_deref(),
            Some("tx1"),
            "AMB_TX1's top-1 must be ITS first candidate, not a neighbour's"
        );
        assert_ne!(
            a0.top_tx_id, a1.top_tx_id,
            "fixture is broken: the two reads must resolve differently, otherwise \
             a mis-attributed join would still pass"
        );

        for (label, a) in [("AMB_TX0", a0), ("AMB_TX1", a1)] {
            assert!(
                a.top_weight < 1.0,
                "{label}'s posterior must retain the runner-up: top_weight is {}",
                a.top_weight
            );
            assert!(
                a.entropy > 0.0,
                "{label}'s posterior must be spread over two candidates: entropy is {}",
                a.entropy
            );
        }
        assert!(
            (a0.top_weight - a1.top_weight).abs() > 1e-9,
            "the two ambiguous reads must have DIFFERENT top_weight, got {} and {}",
            a0.top_weight,
            a1.top_weight
        );
        assert!(
            (a0.entropy - a1.entropy).abs() > 1e-9,
            "the two ambiguous reads must have DIFFERENT entropy, got {} and {}",
            a0.entropy,
            a1.entropy
        );

        for a in &assigns {
            assert_eq!(a.n_candidates_before_cutoff as u16, {
                match a.read_id.as_str() {
                    "UNAMB" => 1,
                    _ => 2,
                }
            });
        }

        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);
    }

    #[test]
    fn test_read_assignments_are_absent_unless_requested() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("readassign_absent_emission.parquet");
        let with_p = scratch("readassign_absent_with.parquet");
        let without_p = scratch("readassign_absent_without.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", 7.0, 0.5),
                ("tx1", "gene1", 7.0, 0.5),
                ("tx2", "gene2", 7.0, 0.5),
            ],
        );
        let graph = || mixed_unambiguous_then_ambiguous(log_pal);

        let mut assigns: Vec<ReadAssignment> = vec![ReadAssignment {
            read_id: "SENTINEL".to_string(),
            n_candidates_before_cutoff: 999,
            top_tx_id: Some("SENTINEL_TX".to_string()),
            top_weight: -1.0,
            entropy: -1.0,
        }];
        solve_and_write_profile(
            graph(),
            &emission,
            "TEST",
            &with_p,
            1e-6,
            5,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            Some(&mut assigns),
        )
        .unwrap();
        assert!(!assigns.is_empty());
        assert!(
            !assigns.iter().any(|a| a.read_id == "SENTINEL"),
            "the out-param must be REPLACED, not appended to: {assigns:#?}"
        );
        let after_requested: Vec<(String, Option<String>, f64, f64)> = assigns
            .iter()
            .map(|a| {
                (
                    a.read_id.clone(),
                    a.top_tx_id.clone(),
                    a.top_weight,
                    a.entropy,
                )
            })
            .collect();

        solve_and_write_profile(
            graph(),
            &emission,
            "TEST",
            &without_p,
            1e-6,
            5,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            None,
        )
        .unwrap();
        let after_not_requested: Vec<(String, Option<String>, f64, f64)> = assigns
            .iter()
            .map(|a| {
                (
                    a.read_id.clone(),
                    a.top_tx_id.clone(),
                    a.top_weight,
                    a.entropy,
                )
            })
            .collect();
        assert_eq!(
            after_not_requested, after_requested,
            "passing `None` must leave the caller's Vec untouched"
        );

        let a = std::fs::read(&with_p).expect("profile with assignments must exist");
        let b = std::fs::read(&without_p).expect("profile without must exist");
        assert_eq!(
            a, b,
            "requesting per-read assignments must not change the profile by so \
             much as one byte"
        );

        for f in [&emission, &with_p, &without_p] {
            let _ = std::fs::remove_file(f);
        }
    }

    #[test]
    fn test_all_culled_read_is_published_under_its_argmax_and_joins() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("culled_emission.parquet");
        let profile = scratch("culled_profile.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", log_pal as f64, 0.04),
                ("tx1", "gene1", log_pal as f64, 0.04),
                ("tx2", "gene2", log_pal as f64, 0.04),
            ],
        );

        let mut assigns: Vec<ReadAssignment> = Vec::new();
        let stats = solve_and_write_profile(
            plain_and_min_weight_culled(log_pal),
            &emission,
            "TEST",
            &profile,
            1e-6,
            5,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            0.9,
            None,
            None,
            Some(&mut assigns),
        )
        .unwrap();

        let rows = read_profile_by_read(&profile);
        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);

        assert_eq!(
            stats.n_ambiguous, 1,
            "CULLED must have been E-stepped as an ambiguous read"
        );
        assert_eq!(
            rows.len(),
            2,
            "PLAIN clears min_weight = 0.9 on its own, and the 0.5/0.5 CULLED \
             read is published under its forced argmax rather than vanishing: {rows:?}"
        );
        assert!(
            rows.iter().any(|(r, ..)| r == "CULLED"),
            "the all-culled read must appear in the profile, or its mass is lost: {rows:?}"
        );
        assert!(
            stats.ledger.reads_retained_by_argmax == 1,
            "and exactly one row was forced: {ledger:?}",
            ledger = stats.ledger
        );

        for a in &assigns {
            assert!(
                rows.iter().any(|(r, ..)| r == &a.read_id),
                "assignment for {} carries a top_tx_id but has no profile row; \
                 the implication `top_tx_id == Some(..) => profile row` is what \
                 keeps the cross-run join denominator auditable: {rows:?}",
                a.read_id
            );
        }
        if let Some(culled) = assigns.iter().find(|a| a.read_id == "CULLED") {
            let row_tx = rows
                .iter()
                .find(|(r, ..)| r == "CULLED")
                .expect("CULLED has a profile row, asserted above");
            assert_eq!(
                Some(row_tx.1.as_str()),
                culled.top_tx_id.as_deref(),
                "the assignment's top_tx_id and the row it forced must name the \
                 same transcript; they read the same fold, so a disagreement here \
                 means a second argmax was introduced somewhere"
            );
        }
    }

    #[test]
    fn test_ghost_read_is_recorded_as_unassignable_and_absent_from_the_profile() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("ghost_emission.parquet");
        let profile = scratch("ghost_profile.parquet");
        write_test_emission_with_competence(
            &emission,
            &[
                ("tx0", "gene0", log_pal as f64, 0.04, true),
                ("tx1", "gene1", log_pal as f64, 0.04, false),
                ("tx2", "gene2", log_pal as f64, 0.04, false),
            ],
        );

        let mut assigns: Vec<ReadAssignment> = Vec::new();
        let stats = solve_and_write_profile(
            ghost_read_beside_a_live_one(log_pal),
            &emission,
            "TEST",
            &profile,
            1e-6,
            5,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            Some(&mut assigns),
        )
        .unwrap();

        let rows = read_profile_by_read(&profile);
        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);

        assert_eq!(
            rows.len(),
            1,
            "the ghost read has no competent candidate, so it emits no profile \
             row and LIVE is all that is left: {rows:?}"
        );
        assert_eq!(rows[0].0, "LIVE");

        assert_eq!(
            assigns.len(),
            2,
            "the ghost must be recorded, not dropped: {assigns:#?}"
        );
        let ghost = assigns
            .iter()
            .find(|a| a.read_id == "GHOST")
            .expect("GHOST must be present in the assignments table");
        assert_eq!(
            ghost.top_tx_id, None,
            "an argmax over an empty weight vector must not fall back to \
             transcript 0 — a real transcript this read was never assigned to"
        );
        assert_eq!(ghost.top_weight, 0.0);
        assert_eq!(ghost.entropy, 0.0);
        assert_eq!(ghost.n_candidates_before_cutoff, 2);
        assert_eq!(
            stats.n_dropped_no_competent_candidates, 1,
            "GHOST must also be counted as a ghost drop in the stats"
        );

        let live = assigns
            .iter()
            .find(|a| a.read_id == "LIVE")
            .expect("LIVE must be present");
        assert_eq!(live.top_tx_id.as_deref(), Some("tx0"));
        for a in &assigns {
            if a.top_tx_id.is_some() {
                assert!(
                    rows.iter().any(|(r, ..)| r == &a.read_id),
                    "top_tx_id == Some(..) for {} but it has no profile row; the \
                     implication the cross-run join depends on is broken: {rows:?}",
                    a.read_id
                );
            }
        }
    }

    #[test]
    fn test_incompetent_but_present_transcript_is_never_a_reads_top1() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("incompetent_emission.parquet");
        let profile = scratch("incompetent_profile.parquet");
        write_test_emission_with_competence(
            &emission,
            &[
                ("GOOD", "gene0", log_pal as f64 - 1.5, 0.04, true),
                ("BAD", "gene1", log_pal as f64, 0.04, false),
            ],
        );

        let mut assigns: Vec<ReadAssignment> = Vec::new();
        let _ = solve_and_write_profile(
            synth_graph(log_pal, &[("GOOD", 900), ("BAD", 900)]),
            &emission,
            "TEST",
            &profile,
            1e-6,
            5,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            Some(&mut assigns),
        )
        .unwrap();

        let rows = read_profile_by_read(&profile);
        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);

        assert_eq!(
            assigns.len(),
            1,
            "the read has a competent candidate, so it is not a ghost and must be \
             recorded: {assigns:#?}"
        );
        let a = &assigns[0];
        assert_eq!(a.read_id, "r1");
        assert_eq!(
            a.top_tx_id.as_deref(),
            Some("GOOD"),
            "an incompetent transcript must never be a read's top-1; it is present \
             in the graph and its mu is an exact fit, so it wins the argmax the \
             moment the E-step stops pushing -inf for it, and neither the fold nor \
             the `params` lookup re-checks competence"
        );
        assert!(
            rows.iter()
                .any(|(rd, tx, _)| rd == "r1" && tx == a.top_tx_id.as_deref().unwrap()),
            "the transcript named as top-1 must itself have a profile row for \
             this read, not merely have SOME row: {rows:?}"
        );
    }

    #[test]
    fn test_write_read_assignments_parquet_round_trips_a_null_top_tx_id() {
        let path = scratch("readassign_null.parquet");
        let rows = vec![
            ReadAssignment {
                read_id: "NAMED".to_string(),
                n_candidates_before_cutoff: 3,
                top_tx_id: Some("tx7".to_string()),
                top_weight: 0.75,
                entropy: 0.55,
            },
            ReadAssignment {
                read_id: "NULL_TOP".to_string(),
                n_candidates_before_cutoff: 2,
                top_tx_id: None,
                top_weight: 0.5,
                entropy: 0.69,
            },
        ];
        write_read_assignments_parquet(&rows, &path).unwrap();

        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap())
            .unwrap()
            .with_batch_size(1024)
            .build()
            .unwrap();
        let mut seen: Vec<(String, i64, Option<String>, f64, f64)> = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let read = batch
                .column_by_name("read_id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let ncand = batch
                .column_by_name("n_candidates_before_cutoff")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let top = batch
                .column_by_name("top_tx_id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let w = batch
                .column_by_name("top_weight")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let h = batch
                .column_by_name("entropy")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                seen.push((
                    read.value(i).to_string(),
                    ncand.value(i),
                    if top.is_valid(i) {
                        Some(top.value(i).to_string())
                    } else {
                        None
                    },
                    w.value(i),
                    h.value(i),
                ));
            }
        }
        let _ = std::fs::remove_file(&path);

        assert_eq!(seen.len(), 2, "a null must not drop the row: {seen:?}");
        assert_eq!(seen[0].0, "NAMED");
        assert_eq!(seen[0].1, 3);
        assert_eq!(
            seen[0].2.as_deref(),
            Some("tx7"),
            "a present top_tx_id must round-trip verbatim"
        );
        assert_eq!(seen[1].0, "NULL_TOP");
        assert_eq!(seen[1].1, 2);
        assert_eq!(
            seen[1].2, None,
            "`top_tx_id: None` must read back as a genuine NULL — not \"\", not \
             the index-0 transcript, not a dropped row"
        );
        assert!((seen[1].3 - 0.5).abs() < 1e-12, "got {}", seen[1].3);
        assert!((seen[1].4 - 0.69).abs() < 1e-12, "got {}", seen[1].4);
    }

    fn zero_power_probe(
        cands: &[Option<(f64, f64)>],
    ) -> (FlatEvidenceGraph, Vec<Option<TranscriptParam>>) {
        let n = cands.len();
        let graph = FlatEvidenceGraph {
            molecules: vec![MoleculeEvidence {
                read_id: "r1".to_string(),
                log_pal: (100.0_f32).log2(),
                candidate_offset: 0,
                candidate_count: n as u16,
            }],
            candidate_nm: vec![0; n],
            candidate_mapq: vec![60; n],
            candidate_pool: (0..n as u32).collect(),
            candidate_as: vec![900; n],
            candidate_5p_gap: vec![0; n],
            transcript_names: (0..n).map(|i| format!("tx{i}")).collect(),
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        };
        let params = cands
            .iter()
            .enumerate()
            .map(|(i, c)| {
                c.map(|(mu, sigma2)| TranscriptParam {
                    transcript_id: format!("tx{i}"),
                    gene_id: format!("gene{i}"),
                    is_competent: true,
                    mu_cohort: mu,
                    mu,
                    sigma2_cohort: sigma2,
                    sigma2,
                    alpha: 0.5,
                    em_accum_weight: 0.5, // fixture never runs an M-step
                    em_finalized: false,  // ditto: this is the Laplace init
                    is_imputed: false,    // test fixture: every row is anchored
                })
            })
            .collect();
        (graph, params)
    }

    #[test]
    fn test_polya_emission_count_and_identity_cases() {
        const MU: f64 = 7.0;
        const S2: f64 = 0.5;

        {
            let (g, p) = zero_power_probe(&[Some((MU, S2)), Some((MU, S2)), Some((MU, S2))]);
            assert!(
                polya_emission_count_and_identity(&g, &g.molecules[0], &p).1,
                "candidates sharing (mu, sigma2) cancel the poly(A) term exactly"
            );
        }

        {
            let (g, p) = zero_power_probe(&[Some((MU, S2))]);
            assert_eq!(
                polya_emission_count_and_identity(&g, &g.molecules[0], &p),
                (1, true),
                "one emission candidate: flag is vacuously true, count says so"
            );
            assert!(
                1 < ZERO_POWER_MIN_EMISSION_CANDIDATES,
                "if this ever holds, the single-candidate reads are NOT excluded \
                 from zero-power_resolved and the metric stops meaning anything"
            );
        }

        {
            let (g, p) = zero_power_probe(&[Some((MU, S2)), Some((MU + 0.5, S2))]);
            assert!(
                !polya_emission_count_and_identity(&g, &g.molecules[0], &p).1,
                "a mu difference is a non-constant log-ratio — poly(A) has say"
            );
        }

        {
            let (g, p) = zero_power_probe(&[Some((MU, S2)), Some((MU, S2 + 0.4))]);
            assert!(
                !polya_emission_count_and_identity(&g, &g.molecules[0], &p).1,
                "sigma2 alone decides the log-ratio; sharing mu is not enough"
            );
        }

        {
            let (g, p) = zero_power_probe(&[Some((MU, S2)), Some((MU, S2)), Some((MU, S2 + 0.4))]);
            assert_eq!(
                polya_emission_count_and_identity(&g, &g.molecules[0], &p),
                (3, false),
                "the count must cover the WHOLE candidate range even after a \
                 disagreement; 2 here means the early return is back"
            );
        }

        {
            let (g, p) = zero_power_probe(&[]);
            assert_eq!(
                polya_emission_count_and_identity(&g, &g.molecules[0], &p),
                (0, false),
                "an empty candidate set must NOT claim zero discriminative power"
            );
        }

        {
            let (g, p) = zero_power_probe(&[Some((MU, S2)), None, Some((MU, S2))]);
            assert_eq!(
                polya_emission_count_and_identity(&g, &g.molecules[0], &p),
                (2, true),
                "a missing emission row is a skip, not a (mu, sigma2) mismatch — \
                 and it is not counted either"
            );
        }
        {
            let (g, p) = zero_power_probe(&[None, None]);
            assert_eq!(
                polya_emission_count_and_identity(&g, &g.molecules[0], &p),
                (0, false),
                "no candidate with an emission row means nothing was measured"
            );
        }

        {
            let (g, p) = zero_power_probe(&[Some((MU, S2)), Some((MU + 1e-13, S2))]);
            assert_ne!(MU, MU + 1e-13, "1e-13 at mu = 7.0 must be representable");
            assert!(
                !polya_emission_count_and_identity(&g, &g.molecules[0], &p).1,
                "a 1e-13 mu gap is a DIFFERENT fit under the exact comparison"
            );
        }
        {
            let (g, p) = zero_power_probe(&[Some((MU, S2)), Some((MU + 1e-9, S2))]);
            assert!(
                !polya_emission_count_and_identity(&g, &g.molecules[0], &p).1,
                "a 1e-9 mu gap is a DIFFERENT fit under the exact comparison"
            );
        }
        {
            let shared_mu = MU;
            let shared_s2 = S2;
            let (g, p) =
                zero_power_probe(&[Some((shared_mu, shared_s2)), Some((shared_mu, shared_s2))]);
            assert!(
                polya_emission_count_and_identity(&g, &g.molecules[0], &p).1,
                "a bit-identical copy is zero power — the control for case 5"
            );
        }
    }

    #[test]
    fn test_n_polya_zero_power_reaches_stats_with_the_known_mix() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("zero_power_emission.parquet");
        let profile = scratch("zero_power_profile.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", 7.0, 0.5),
                ("tx1", "gene1", 7.0, 0.5),
                ("tx2", "gene2", 7.0, 0.5),
                ("tx3", "gene3", 7.0, 0.5),
                ("tx4", "gene4", 7.5, 0.5),
                ("tx5", "gene5", 7.0, 0.9),
            ],
        );

        let molecules = vec![
            MoleculeEvidence {
                read_id: "UNAMBIGUOUS".into(),
                log_pal,
                candidate_offset: 0,
                candidate_count: 1,
            },
            MoleculeEvidence {
                read_id: "ZERO_2WAY".into(),
                log_pal,
                candidate_offset: 1,
                candidate_count: 2,
            },
            MoleculeEvidence {
                read_id: "ZERO_3WAY".into(),
                log_pal,
                candidate_offset: 3,
                candidate_count: 3,
            },
            MoleculeEvidence {
                read_id: "MU_SPLIT".into(),
                log_pal,
                candidate_offset: 6,
                candidate_count: 2,
            },
            MoleculeEvidence {
                read_id: "SIGMA_SPLIT".into(),
                log_pal,
                candidate_offset: 8,
                candidate_count: 2,
            },
        ];
        let candidate_pool: Vec<u32> = vec![0, 0, 1, 1, 2, 3, 0, 4, 1, 5];
        let n = candidate_pool.len();
        let graph = FlatEvidenceGraph {
            molecules,
            candidate_nm: vec![0; n],
            candidate_mapq: vec![60; n],
            candidate_as: vec![900; n],
            candidate_5p_gap: vec![0; n],
            candidate_pool,
            transcript_names: (0..6).map(|i| format!("tx{i}")).collect(),
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        };

        let stats = solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            1,
            20.0,
            FIVE_PRIME_OFF,
            0.0, // freeze mu/sigma2: see the docstring.
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            None,
        )
        .unwrap();

        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);

        assert_eq!(stats.n_ambiguous, 4, "four of the five reads are contested");
        assert_eq!(
            stats.n_unambiguous, 1,
            "UNAMBIGUOUS must stay a single-candidate read: a one-candidate read \
             has no pairwise odds ratio, so counting it would put this metric \
             near 1.0 and say nothing"
        );

        assert_eq!(
            stats.n_polya_zero_power, 2,
            "exactly ZERO_2WAY and ZERO_3WAY share one (mu, sigma2); MU_SPLIT and \
             SIGMA_SPLIT each disagree in one component and must NOT be counted, \
             and UNAMBIGUOUS is out of scope for the counter entirely"
        );
    }

    #[test]

    fn test_degenerate_tie_split_partitions_the_degenerate_reads() {
        let log_pal = (100.0_f32).log2();

        let call = std::sync::atomic::AtomicUsize::new(0);
        let run = |nobs: u32, tag: &str| {
            let uniq = call.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let emission = local_scratch(&format!("degen_split_{tag}_{uniq}_emission.parquet"));
            let profile = local_scratch(&format!("degen_split_{tag}_{uniq}_profile.parquet"));
            write_test_emission_with_nobs(
                &emission,
                &[
                    ("tx0", "gene0", 7.0, 0.04, true, nobs),
                    ("tx1", "gene0", 7.0, 0.04, true, nobs),
                    ("tx2", "gene0", 7.0, 0.04, true, nobs),
                    ("tx3", "gene0", 7.0, 0.04, true, nobs),
                ],
            );
            let sz = std::fs::metadata(&emission).map(|m| m.len()).unwrap_or(0);
            assert!(
                sz > 8,
                "FIXTURE NOT WRITTEN: {} is {} bytes right after writing",
                emission.display(),
                sz
            );
            let graph = FlatEvidenceGraph {
                molecules: vec![MoleculeEvidence {
                    read_id: "FLAT".into(),
                    log_pal,
                    candidate_offset: 0,
                    candidate_count: 4,
                }],
                candidate_nm: vec![0; 4],
                candidate_mapq: vec![60; 4],
                candidate_as: vec![900; 4],
                candidate_5p_gap: vec![0; 4],
                candidate_pool: vec![0, 1, 2, 3],
                transcript_names: (0..4).map(|i| format!("tx{i}")).collect(),
                poly_a_is_placeholder: false,
                stats: crate::evidence::FilterCounts::default(),
            };
            let stats = solve_and_write_profile(
                graph,
                &emission,
                "TEST",
                &profile,
                1e-6,
                1,
                20.0,
                FIVE_PRIME_OFF,
                POLY_A_ON,
                FIVE_PRIME_WINDOW,
                PRUNE_MAX_DELTA_AS,
                MIN_WEIGHT,
                None,
                None,
                None,
            )
            .unwrap();
            (stats, emission, profile)
        };

        let (all_imputed, e1, p1) = run(0, "imputed");
        assert!(
            all_imputed.n_entropy_degenerate > 0,
            "FIXTURE IS BROKEN: four identical candidates must give H = ln 4 > 0.70, \
             but n_entropy_degenerate == 0"
        );
        assert_eq!(
            all_imputed.n_degenerate_all_imputed + all_imputed.n_degenerate_with_anchored,
            all_imputed.n_entropy_degenerate,
            "the split must PARTITION the degenerate reads, not overlap or leave a remainder"
        );
        assert_eq!(
            all_imputed.n_degenerate_all_imputed, all_imputed.n_entropy_degenerate,
            "every candidate carries n_cohort_flnc == 0, so every degenerate read is \
             padded-padded ambiguity"
        );

        let (all_anchored, e2, p2) = run(10, "anchored");

        for f in [e1, p1, e2, p2] {
            let _ = std::fs::remove_file(f);
        }
        assert_eq!(
            all_anchored.n_degenerate_all_imputed + all_anchored.n_degenerate_with_anchored,
            all_anchored.n_entropy_degenerate,
            "the split must partition here too"
        );
        assert_eq!(
            all_anchored.n_degenerate_with_anchored, all_anchored.n_entropy_degenerate,
            "the SAME fixture with n_cohort_flnc == 10 must move every degenerate read to \
             real competition. If this fails, the classifier is not reading the marker."
        );
    }

    #[test]
    fn test_drop_taxonomy_attributes_every_lost_read_to_exactly_one_cause() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("drop_taxonomy_emission.parquet");
        let profile = scratch("drop_taxonomy_profile.parquet");
        write_test_emission_with_competence(
            &emission,
            &[
                ("tx0", "gene0", 7.0, 0.04, true),
                ("tx1", "gene1", 7.0, 0.04, true),
                ("tx3", "gene3", 7.0, 0.04, false),
                ("tx6", "gene6", 7.0, 0.04, false),
            ],
        );

        let molecules = vec![
            MoleculeEvidence {
                read_id: "UN_MISSING".into(),
                log_pal,
                candidate_offset: 0,
                candidate_count: 1,
            },
            MoleculeEvidence {
                read_id: "UN_INCOMP".into(),
                log_pal,
                candidate_offset: 1,
                candidate_count: 1,
            },
            MoleculeEvidence {
                read_id: "UN_OK".into(),
                log_pal,
                candidate_offset: 2,
                candidate_count: 1,
            },
            MoleculeEvidence {
                read_id: "G_ALL_MISS".into(),
                log_pal,
                candidate_offset: 3,
                candidate_count: 2,
            },
            MoleculeEvidence {
                read_id: "G_ALL_INC".into(),
                log_pal,
                candidate_offset: 5,
                candidate_count: 2,
            },
            MoleculeEvidence {
                read_id: "G_MIXED".into(),
                log_pal,
                candidate_offset: 7,
                candidate_count: 2,
            },
            MoleculeEvidence {
                read_id: "G_LIVE".into(),
                log_pal,
                candidate_offset: 9,
                candidate_count: 2,
            },
        ];
        let candidate_pool: Vec<u32> = vec![4, 3, 0, 4, 5, 3, 6, 4, 3, 0, 1];
        let n = candidate_pool.len();
        let graph = FlatEvidenceGraph {
            molecules,
            candidate_nm: vec![0; n],
            candidate_mapq: vec![60; n],
            candidate_as: vec![900; n],
            candidate_5p_gap: vec![0; n],
            candidate_pool,
            transcript_names: (0..7).map(|i| format!("tx{i}")).collect(),
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        };

        let stats = solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            1,
            20.0,
            FIVE_PRIME_OFF,
            POLY_A_ON,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            None,
        )
        .unwrap();

        let rows = read_profile_by_read(&profile);
        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);

        assert_eq!(stats.n_unambiguous, 3, "UN_MISSING, UN_INCOMP, UN_OK");
        assert_eq!(stats.n_ambiguous, 4, "the four G_* reads are contested");

        assert_eq!(
            stats.n_unambiguous_dropped_missing_emission, 1,
            "only UN_MISSING's candidate is absent from the model"
        );
        assert_eq!(
            stats.n_unambiguous_dropped_incompetent, 1,
            "only UN_INCOMP's candidate carries is_polyA_competent == false"
        );

        assert_eq!(
            stats.n_dropped_no_competent_candidates, 3,
            "G_ALL_MISS, G_ALL_INC and G_MIXED have no live candidate. G_LIVE does \
             NOT: losing candidates is not the same as having no posterior."
        );
        assert_eq!(stats.n_ghost_read_all_missing, 1, "G_ALL_MISS only");
        assert_eq!(stats.n_ghost_read_all_incompetent, 1, "G_ALL_INC only");
        assert_eq!(
            stats.n_ghost_read_all_alpha_zero, 0,
            "see the `max_iter = 1` note: unreachable through the solver"
        );

        let attributed = stats.n_ghost_read_all_missing
            + stats.n_ghost_read_all_incompetent
            + stats.n_ghost_read_all_alpha_zero;
        assert_eq!(
            stats.n_dropped_no_competent_candidates - attributed,
            1,
            "G_MIXED must be counted in the ghost TOTAL and in no cause counter"
        );

        assert_eq!(
            stats.n_dropped_output_row_competence, 0,
            "the output stage's own competence filter is unreachable while the \
             E-step pushes such candidates as -inf. If this is non-zero the \
             read-level counters above no longer account for the whole loss."
        );

        let mut emitted: Vec<&str> = rows.iter().map(|(r, _, _)| r.as_str()).collect();
        emitted.sort_unstable();
        assert_eq!(
            emitted,
            vec!["G_LIVE", "G_LIVE", "UN_OK"],
            "only the reads with a live candidate may appear in the profile: {rows:?}"
        );
    }

    #[test]
    fn test_zero_power_resolved_excludes_reads_with_fewer_than_two_emission_candidates() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("zero_power_split_emission.parquet");
        let profile = scratch("zero_power_split_profile.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", 7.0, 0.5),
                ("tx1", "gene1", 7.0, 0.5),
                ("tx2", "gene2", 7.0, 0.5),
                ("tx3", "gene3", 7.0, 0.5),
                ("tx4", "gene4", 7.5, 0.5),
                ("tx5", "gene5", 7.0, 0.9),
            ],
        );

        let molecules = vec![
            MoleculeEvidence {
                read_id: "UNAMBIGUOUS".into(),
                log_pal,
                candidate_offset: 0,
                candidate_count: 1,
            },
            MoleculeEvidence {
                read_id: "GHOST_ONLY".into(),
                log_pal,
                candidate_offset: 1,
                candidate_count: 1,
            },
            MoleculeEvidence {
                read_id: "ZERO_2WAY".into(),
                log_pal,
                candidate_offset: 2,
                candidate_count: 2,
            },
            MoleculeEvidence {
                read_id: "SINGLE_THEN_GHOST".into(),
                log_pal,
                candidate_offset: 4,
                candidate_count: 2,
            },
            MoleculeEvidence {
                read_id: "MU_SPLIT".into(),
                log_pal,
                candidate_offset: 6,
                candidate_count: 2,
            },
        ];
        let candidate_pool: Vec<u32> = vec![0, 6, 0, 1, 0, 6, 0, 4];
        let n = candidate_pool.len();
        let graph = FlatEvidenceGraph {
            molecules,
            candidate_nm: vec![0; n],
            candidate_mapq: vec![60; n],
            candidate_as: vec![900; n],
            candidate_5p_gap: vec![0; n],
            candidate_pool,
            transcript_names: (0..7).map(|i| format!("tx{i}")).collect(),
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        };

        let stats = solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            1,
            20.0,
            FIVE_PRIME_OFF,
            0.0,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
            None,
            None,
            None,
        )
        .unwrap();

        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);

        assert_eq!(stats.n_unambiguous, 2, "UNAMBIGUOUS and GHOST_ONLY");
        assert_eq!(stats.n_ambiguous, 3, "the other three are contested");

        assert_eq!(
            stats.n_reads_any_emission_candidate, 4,
            "UNAMBIGUOUS, ZERO_2WAY, SINGLE_THEN_GHOST and MU_SPLIT have a row; \
             GHOST_ONLY does not"
        );
        assert_eq!(
            stats.n_reads_single_emission_candidate, 2,
            "UNAMBIGUOUS (one candidate, present) and SINGLE_THEN_GHOST (two \
             candidates, one present). SINGLE_THEN_GHOST is the read on which \
             zero-power_all is vacuous."
        );

        assert_eq!(
            stats.n_reads_multi_emission_candidate, 2,
            "ZERO_2WAY and MU_SPLIT only; SINGLE_THEN_GHOST has ONE present \
             candidate and must be outside this denominator, or \
             zero-power_resolved would silently reproduce the vacuous number"
        );

        assert_eq!(
            stats.n_polya_zero_power, 2,
            "unchanged semantics: ZERO_2WAY and the vacuous SINGLE_THEN_GHOST"
        );
        assert_eq!(
            stats.n_polya_zero_power_resolved, 1,
            "ZERO_2WAY only. If this is 2 the exclusion is not happening; if it \
             is 0 the >= 2 gate rejected a read that had two present candidates."
        );
    }

    #[test]
    fn test_zero_power_telemetry_reads_the_e_step_state_not_the_post_m_step() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("estep_state_emission.parquet");
        write_test_emission(
            &emission,
            &[("tx0", "gene0", 7.0, 0.5), ("tx1", "gene0", 7.0, 0.5)],
        );
        let molecules = vec![
            MoleculeEvidence {
                read_id: "TIED_A".into(),
                log_pal,
                candidate_offset: 0,
                candidate_count: 2,
            },
            MoleculeEvidence {
                read_id: "TIED_B".into(),
                log_pal,
                candidate_offset: 2,
                candidate_count: 2,
            },
        ];
        let candidate_pool: Vec<u32> = vec![0, 1, 0, 1];
        let n = candidate_pool.len();
        let mut candidate_as = vec![900; n];
        candidate_as[1] = 850;
        candidate_as[3] = 850;
        let graph = FlatEvidenceGraph {
            molecules,
            candidate_nm: vec![0; n],
            candidate_mapq: vec![60; n],
            candidate_as,
            candidate_5p_gap: vec![0; n],
            candidate_pool,
            transcript_names: (0..2).map(|i| format!("tx{i}")).collect(),
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        };

        let run = |w: f64, tag: &str| {
            let profile = scratch(&format!("estep_state_profile_{tag}.parquet"));
            let mut out: Vec<Option<TranscriptParam>> = Vec::new();
            let stats = solve_and_write_profile(
                FlatEvidenceGraph {
                    molecules: graph.molecules.clone(),
                    candidate_nm: graph.candidate_nm.clone(),
                    candidate_mapq: graph.candidate_mapq.clone(),
                    candidate_as: graph.candidate_as.clone(),
                    candidate_5p_gap: graph.candidate_5p_gap.clone(),
                    candidate_pool: graph.candidate_pool.clone(),
                    transcript_names: graph.transcript_names.clone(),
                    poly_a_is_placeholder: false,
                    stats: crate::evidence::FilterCounts::default(),
                },
                &emission,
                "TEST",
                &profile,
                1e-6,
                1, // max_iter: EXHAUSTION, the path where the bug lives
                20.0,
                FIVE_PRIME_OFF,
                w,
                FIVE_PRIME_WINDOW,
                PRUNE_MAX_DELTA_AS,
                MIN_WEIGHT,
                None,
                Some(&mut out),
                None,
            )
            .unwrap();
            let _ = std::fs::remove_file(&profile);
            (stats, out)
        };

        let (off, _) = run(0.0, "off");
        let (on, on_params) = run(1.0, "on");

        let _ = std::fs::remove_file(&emission);

        assert_eq!(off.n_ambiguous, 2, "both reads are contested");
        assert_eq!(
            off.n_polya_zero_power, 2,
            "the OFF arm freezes mu/sigma2, so both reads keep the shared loaded \
             fit and both are zero-power"
        );
        assert_eq!(
            on.n_polya_zero_power, off.n_polya_zero_power,
            "the ON arm ran an M-step, but the telemetry must describe the \
             POSTERIOR, which was built BEFORE that M-step from the same loaded \
             fit the OFF arm saw. Pre-fix this reads post-M-step params, where \
             the shrinkage has pulled the tied fits apart, and reports {} — \
             strictly fewer, because a genuine tie got scored as a difference.",
            on.n_polya_zero_power
        );

        let p0 = on_params[0].as_ref().expect("tx0 has an emission row");
        assert_eq!(
            (p0.mu, p0.sigma2),
            (7.0, 0.5),
            "the sidecar must carry the params the LAST E-step CONSUMED, which at \
             max_iter=1 is the loaded fit verbatim. Reverting this half of the \
             fix hands back the post-M-step values instead (~6.83 / ~0.42 here), \
             which is the same stale-mu defect the telemetry half had."
        );
    }
}
