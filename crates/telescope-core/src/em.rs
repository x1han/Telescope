//! TEL-EM solver: the bifurcated Terminal-aware Expectation-Maximization solver.
//!
//! Consumes the flat evidence graph and writes `telescope.profile.parquet`.
//!
//! Bifurcated inference:
//! - Unambiguous reads (`candidate_count == 1`) get `w_rk ≡ 1.0` (deterministic).
//! - Ambiguous reads (`candidate_count > 1`) enter the EM loop, whose
//!   responsibility is `w_rk ∝ π_k · A_rk · F_rk · N(y_r; μ_k, σ_k²)`, with
//!   `ln A_rk = (AS_rk − max_j AS_rj) / λ`, `ln_5p_k = β if g_k ≤ window`, and
//!   a `ΔAS > 50` prune. See `solve_and_write_profile` for the parameters.
//!
//! Main items: `load_emission_parameters`, `solve_and_write_profile`,
//! `write_read_assignments_parquet`, `write_emission_params_parquet`,
//! `CandidateLedger`, `TelEmStats`.

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

/// Shape parameter of the Inverse-Gamma prior on σ².
pub const NU_0: f64 = 1.0;
/// Scale multiplier of the Inverse-Gamma prior on σ².
pub const BETA_0: f64 = 1.0;
/// MAP pseudo-count damping for the μ update (`tau_0`).
///
/// `pub const` so the value is assertable; pinned by
/// `test_m_step_prior_constants_are_pinned`.
pub const TAU_0: f64 = 2.0;
/// Degrees-of-freedom offset `c` in the σ² M-step denominator
/// `(S(mu) + 2*beta_0*sigma2_cohort) / (W + 2*nu_0 + c)`.
///
/// `c` is a regularisation-strength knob, not a model selector: the numerator
/// stays Model A's for every `c`. `c = 2` is Model A; `c != 2` is NEVER Model B,
/// which also needs `tau_0*(mu - mu_cohort)^2` back in the numerator.
pub const SIGMA2_DOF_OFFSET: f64 = 2.0;

/// Per-transcript parameters loaded from `telescope.emission.parquet`.
#[derive(Debug, Clone)]
pub struct TranscriptParam {
    pub transcript_id: String,
    pub gene_id: String,
    pub is_competent: bool,
    /// Shrinkage-prior μ, bound at load from the emission parquet's own
    /// `mu_logPAL` column.
    ///
    /// MISNOMER: "cohort" means the tier the value was resolved at, not a
    /// cohort-wide constant; tier-1/2 rows carry their own fitted `mu_logPAL`.
    pub mu_cohort: f64,
    /// Current μ estimate (initialized to mu_cohort, updated each M-step).
    pub mu: f64,
    /// Shrinkage-prior σ², bound at load from the emission parquet's own
    /// `sigma2_logPAL` column.
    ///
    /// MISNOMER as `mu_cohort` above: only cohort-fit and imputed rows hold the
    /// cohort variance. Fixed across iterations — the M-step rewrites `sigma2`.
    pub sigma2_cohort: f64,
    /// Current EM iteration σ² (initialized to cohort, updated each M-step).
    pub sigma2: f64,
    /// Mixing proportion π_k (initialized at iter 0, updated each M-step).
    pub alpha: f64,
    /// The M-step's W for this transcript: the E-step weight sum, pre-truncation
    /// and pre-renormalisation.
    ///
    /// NOT recoverable from `alpha`: `alpha` is W divided by the total molecule
    /// count, and the divisor is recorded nowhere.
    ///
    /// Stamped after the EM loop from the FINAL E-step's weights, so it is one
    /// iteration behind the M-step's own `sigma2`.
    pub em_accum_weight: f64,
    /// TRUE when at least one M-step completed, so `alpha`, `mu` and `sigma2` are
    /// an M-step's output rather than the Laplace initialisation.
    ///
    /// Convergence breaks BEFORE the M-step, so a run that converges on its
    /// first iteration exports the initialisation, not a fit. `em_accum_weight`
    /// is NOT gated on this flag: W exists on the converge-immediately path too.
    pub em_finalized: bool,
    /// TRUE when this row was IMPUTED to reach the reference universe, i.e.
    /// `n_cohort_flnc == 0` in the emission parquet.
    ///
    /// It carries the cohort fit, so every imputed row of one build shares an
    /// identical `(mu, sigma2)` pair. Not inferable from the likelihood.
    pub is_imputed: bool,
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
        // `n_cohort_flnc == 0` is the ONLY in-schema marker of an imputed row,
        // and it is REQUIRED: a model omitting the column is rejected here
        // rather than silently classifying every row as anchored.
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

            // `sigma2_cohort` is bound from this transcript's own
            // `sigma2_logPAL` and is a VARIANCE: it must arrive finite and
            // strictly positive, or it poisons `p.sigma2` for this transcript
            // and for every transcript that borrows it.
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

/// Run TEL-EM and write `telescope.profile.parquet`.
///
/// `aln_weight_lambda` scales the alignment term `ln A_rk = (AS_rk − max_j
/// AS_rj) / λ`; it must be > 0 and only matters for `candidate_count > 1`.
/// `output_em_trace`, if `Some`, receives `(iter_index, log_likelihood,
/// mean_entropy)` per EM iteration.
///
/// Evidence terms, in the form that must not be re-derived:
///     `g_k` = `s_k − 1`, the 1-BASED start of the read's BEST alignment to
///     candidate k, in candidate k's OWN cDNA coordinates (never a genomic span).
///     `ln_5p_k` = `five_prime_weight` if `g_k <= five_prime_window` else 0.0.
///     prune: drop candidates with `ΔAS_k > prune_max_delta_as`, against the
///     read's best AS.
///     `w_k ∝ π_k · exp(ln_a_k + ln_5p_k)`.
///
/// This is the shipped entry point, at the default `SIGMA2_DOF_OFFSET` (2.0,
/// Model A); the production binary cannot reach any other `c`.
///
/// The argument list is many adjacent positional scalars — do not reorder it.
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

/// `solve_and_write_profile` with the σ² regularisation degrees-of-freedom
/// offset `c` supplied by the caller (same arguments, in the same order, plus
/// this one).
///
/// `c` is the trailing `+ c` of the σ² denominator; the numerator stays Model
/// A's for every `c`, so raising `c` only sharpens the Gaussian. `c = 3` is
/// NOT Model B — read `SIGMA2_DOF_OFFSET`.
///
/// Not reachable from the production `telescope` binary.
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
    // Guard at the boundary: a non-positive or non-finite `c` is absorbed by
    // the `.max(INSTRUMENTAL_PAL_NOISE_FLOOR)` below and would return a
    // plausible sigma2 with no error at all.
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

/// The body. `sigma2_dof_offset` is the trailing constant of the σ²
/// denominator; at `SIGMA2_DOF_OFFSET` the arithmetic is bit-identical to the
/// unparameterised code.
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
    // evidence.
    //
    // At 0.0 the score is still `ln alpha_k + ln A_rk + ln_5p_k`: the abundance
    // prior survives at any weight, so that arm is NOT "alignment-only", and it
    // cannot be neutralised from outside because the M-step re-estimates
    // `alpha`.
    poly_a_weight: f64,
    five_prime_window: u32,
    prune_max_delta_as: i32,
    min_weight: f64,
    mut output_em_trace: Option<&mut Vec<(usize, f64, f64)>>,
    // Optional out-param for the fitted `(mu, sigma2)`; `TelEmStats` carries
    // neither, so without it the M-step's parameters are unobservable.
    mut output_emission_params: Option<&mut Vec<Option<TranscriptParam>>>,
    // Per-READ posterior summary. The profile parquet is per (read, candidate)
    // PAIR and carries no per-read summary, so this is the only way to recover
    // "what did this read end up assigned to, and how sure was it".
    output_read_assignments: Option<&mut Vec<ReadAssignment>>,
    // Trailing constant `c` of the σ² denominator. Callers reach this through
    // `solve_and_write_profile` (which supplies `SIGMA2_DOF_OFFSET`) or
    // `solve_and_write_profile_with_dof_offset` (which validates it first).
    sigma2_dof_offset: f64,
) -> Result<TelEmStats> {
    // API-LEVEL DEFENCE: the CLI forces `poly_a_weight = 0` when the graph has
    // no poly(A), but a library caller never goes through the CLI. Pairing a
    // placeholder graph with a positive weight feeds a constant emission into
    // the E-step and silently distorts every posterior.
    //
    // The check is on the graph's provenance flag, not on `log_pal` values: a
    // real BAM can legitimately contain reads whose `pt:i` is absent.
    if poly_a_weight > 0.0 && graph.poly_a_is_placeholder {
        anyhow::bail!(
            "FATAL: poly_a_weight = {poly_a_weight} on a graph built with \
             --polyA-source none. Every molecule's log_pal is the placeholder, \
             not an observed tail length, so the Gaussian would score a constant \
             emission and distort every posterior. Re-run with \
             --polyA-source bam_tag:<TAG>, or set --poly-a-weight 0."
        );
    }

    // NaN-safe equivalent of the CLI's `!(aln_weight_lambda > 0.0)`: `<= 0.0`
    // alone would let NaN through and emit NaN posterior weights.
    if aln_weight_lambda <= 0.0 || aln_weight_lambda.is_nan() {
        anyhow::bail!(
            "aln_weight_lambda must be > 0 (got {}); it scales ln A_rk = (AS_rk − \
             max_j AS_rj)/λ, so 0 and NaN would emit NaN posterior weights",
            aln_weight_lambda
        );
    }
    // NaN-safe: `(0.0..1.0).contains(&NaN)` is false, so the range test alone
    // rejects NaN; `is_finite` is kept explicit because +inf must fail too.
    // 0.0 is legal and means NO truncation.
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
    // 5'-anchoring weight. 0.0 is a LEGAL value and the documented off switch:
    // it makes `ln_5p ≡ 0`, so the term leaves the softmax EXACTLY. Only a
    // negative weight or NaN are rejected.
    //
    // `is_finite()` rather than `is_nan()`: +inf would poison the log-sum-exp
    // with `inf - inf`, emitting NaN weights.
    if !five_prime_weight.is_finite() || five_prime_weight < 0.0 {
        anyhow::bail!(
            "five_prime_weight must be >= 0 and non-NaN (got {}); it is the log-space \
             bonus ln_5p awarded to a candidate whose observed 5' gap g_k <= \
             five_prime_window, so 0.0 (the off switch) is valid but a negative \
             weight would invert the term",
            five_prime_weight
        );
    }
    // A negative prune threshold would drop every candidate INCLUDING the
    // read's own argmax, emptying `raw` and turning every ambiguous read into
    // a ghost drop. `i32::MAX` is a legal "prune off" value.
    if prune_max_delta_as < 0 {
        anyhow::bail!(
            "prune_max_delta_as must be >= 0 (got {}); it drops candidates whose \
             alignment score is more than this far below the read's best, and a \
             negative threshold would drop the argmax too, emptying every read",
            prune_max_delta_as
        );
    }

    // `max_iter == 0` runs no E-step, so the candidate census classifies
    // EVERY contested candidate as having no emission row — a confident
    // falsehood about a run that never looked at one. Reject the input.
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

    // Index transcripts by position in `graph.transcript_names`.
    let mut params: Vec<Option<TranscriptParam>> = Vec::with_capacity(num_transcripts);
    for name in &graph.transcript_names {
        params.push(param_map.remove(name));
    }
    let n_matched = params.iter().filter(|p| p.is_some()).count();

    // ID-SPACE GUARD. `graph.transcript_names` comes from the BAM @SQ header;
    // the emission parquet is keyed by whatever `build-emission` was handed, and
    // the two can be DIFFERENT SPACES. A zero overlap makes every `params[i]`
    // `None`, so every read takes the ghost-drop path and the run still exits 0.
    //
    // Zero overlap is never a legitimate configuration, so refuse it; partial
    // overlap is legitimate and is NOT checked here.
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
             silently dropped and an EMPTY profile written. The retired AIDRS \
             quant path was the one configuration that produced this: its BAM was \
             keyed `AIDRS_tx_N` while the AIDRS polyA/assessment tables were keyed \
             `ENST`/`ENSG_NovelTr`; it was bridged with `--tx-id-mapping <bam_id<TAB> \
             original_id tsv>`. Universal Mode (`build-anchors`) needs no bridge, \
             because its ids come from the BAM itself."
        );
    }

    // 1. Split molecules by candidate_count.
    let mut unambiguous_indices: Vec<usize> = Vec::new();
    let mut ambiguous_indices: Vec<usize> = Vec::new();
    // Left-hand side of the candidate conservation law. Accumulated HERE, in the
    // one place that already walks every molecule: an arm-scoped sum would
    // silently exclude the other arm's candidates.
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
    use crate::em::{BETA_0, NU_0, TAU_0};
    let nu_0 = NU_0;
    let beta_0 = BETA_0;
    let tau_0 = TAU_0;
    let mut prev_ll = f64::NEG_INFINITY;
    let mut final_weights: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_ambiguous];
    let mut n_em_iter = 0_usize;

    // The `(mu, sigma2)` the LAST E-step actually consumed.
    //
    // "Did poly(A) have any say on this read?" is a property of the evidence
    // THE POSTERIOR WAS BUILT FROM. The M-step rewrites `p.mu` / `p.sigma2` but
    // only under `poly_a_weight > 0.0`, so a poly(A) A/B would read DIFFERENT
    // states at the output stage while having consumed the SAME state.
    //
    // ONE snapshot, TWO consumers — the telemetry and the sidecar — so both
    // describe the same state by construction.
    let mut last_estep_params: Vec<Option<TranscriptParam>> = Vec::new();

    // "Did an M-step actually execute?" -- set where the M-step runs, NOT
    // inferred from `n_em_iter`: `max_iter == 1` runs one E-step, does not
    // converge, runs the M-step and ends the loop with `n_em_iter == 1`.
    let mut m_step_ran = false;

    // The W the LAST M-step actually consumed, captured by moving ownership out
    // of the M-step block (a `Vec` move, not a clone).
    //
    // `p.em_accum_weight` is stamped at the end of the run from a FRESH
    // suffstats pass over the post-loop `final_weights`, while `p.sigma2` in
    // the same row came from the M-step's OWN suffstats. The two are one
    // E-step apart.
    let mut last_mstep_w: Vec<f64> = Vec::new();

    for iter in 0..max_iter {
        // E-step: parallel over ambiguous reads. Compute log-sum-exp normalized weights,
        // per-read log-likelihood, and per-read posterior entropy.
        let e_results: Vec<(f64, Vec<(usize, f64)>, f64)> = ambiguous_indices
            .par_iter()
            .map(|&mol_idx| {
                let mol = &graph.molecules[mol_idx];
                let y = mol.log_pal as f64;

                // Alignment-score reference point for this read. Taken over the
                // FULL `0..candidate_count` range — before competence masking, and
                // over candidates the `if let Some(ref p) = params[tx_id]` arm
                // below skips entirely.
                //
                // NOT the same range as `max_ll` below, which is over `raw`: the
                // offset is shared by every surviving candidate, so it cancels in
                // the log-sum-exp and the emitted posteriors are unaffected.
                let max_as = (0..mol.candidate_count)
                    .map(|c| graph.candidate_as[(mol.candidate_offset + c as u32) as usize])
                    .fold(i32::MIN, i32::max);

                // Compute raw log responsibilities (with alpha + Gaussian +
                // A_rk + the 5' term).
                let mut raw: Vec<(usize, f64)> = Vec::with_capacity(mol.candidate_count as usize);
                for c in 0..mol.candidate_count {
                    let idx = (mol.candidate_offset + c as u32) as usize;
                    let tx_id = graph.candidate_pool[idx] as usize;

                    // ΔAS is a MAGNITUDE: the read's best score minus this
                    // candidate's, so ΔAS >= 0. SIGN MATTERS: the prune
                    // thresholds the magnitude, the log term negates it.
                    //
                    // `saturating_sub`, not `-`: this sits OUTSIDE the
                    // `if let Some(params)`, so it runs for candidates with no
                    // emission row. A plain subtraction WRAPS under `--release`,
                    // and a wrapped negative ΔAS satisfies
                    // `delta_as > prune_max_delta_as` as FALSE.
                    let delta_as = max_as.saturating_sub(graph.candidate_as[idx]);

                    // ΔAS prune: drop candidates whose alignment score is more
                    // than `prune_max_delta_as` below the read's best.
                    //
                    // `-inf` rather than a bare `continue`, matching the
                    // competence mask below: the two are observationally
                    // IDENTICAL here because `weights` is built with
                    // `filter(ll > -inf)`, and would DIVERGE if that filter were
                    // removed.
                    //
                    // Placed OUTSIDE the `if let Some(params)` so the prune is a
                    // property of the alignment evidence alone. It cannot empty
                    // a read: the argmax has `delta_as == 0` and the threshold
                    // is validated `>= 0` above.
                    if delta_as > prune_max_delta_as {
                        raw.push((tx_id, f64::NEG_INFINITY));
                        continue;
                    }

                    // ln A_rk = (AS_rk − max_j AS_rj) / λ = −ΔAS / λ. Added to
                    // `ll` *before* the `max_ll` log-sum-exp shift, so any λ is
                    // numerically safe — no clamping needed.
                    let ln_a = -(delta_as as f64) / aln_weight_lambda;

                    // ln_5p_k = β if g_k ≤ window else 0. `g_k` is the observed
                    // 5' gap of the read's best alignment to THIS candidate, in
                    // this candidate's own cDNA coordinates, used as-is (no strand
                    // correction is applied upstream).
                    //
                    // It is a PRIOR, not an EM parameter: `g_k` is fixed per
                    // (read, candidate) exactly like `AS`, so it belongs in no
                    // accumulator and no M-step.
                    //
                    // Unlike `ln_a`, `ln_5p` is a step function of one
                    // coordinate: it can only reorder candidates, never delete
                    // them — deletion is the prune's job.
                    let ln_5p = if graph.candidate_5p_gap[idx] <= five_prime_window {
                        five_prime_weight
                    } else {
                        0.0
                    };

                    if let Some(ref p) = params[tx_id] {
                        if !p.is_competent || p.alpha <= 0.0 {
                            // Push -inf so normalization kills this candidate.
                            raw.push((tx_id, f64::NEG_INFINITY));
                        } else {
                            // `ln alpha` is the MIXTURE PRIOR, not part of the
                            // poly(A) emission, so it is OUTSIDE the
                            // poly_a_weight scale. At `poly_a_weight = 0` the
                            // E-step would never read alpha, the M-step's
                            // update would affect no posterior, and the "EM"
                            // would degenerate into a one-shot static alignment
                            // scorer with an inert M-step.
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

        // SNAPSHOT THE E-STEP'S VIEW OF THE MODEL — see `last_estep_params`.
        // Taken after the E-step consumed `params` and before the M-step rewrites them.
        last_estep_params = params.clone();

        let current_ll: f64 = e_results.iter().map(|&(ll, _, _)| ll).sum();
        n_em_iter = iter + 1;

        // Record EM trace (per-iter log-L + mean posterior entropy).
        if let Some(ref mut trace) = output_em_trace {
            // Exclude ghost drops (h == 0) from the denominator, so the reported
            // mean_entropy reflects reads that actually participated in the E-step.
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
        // The ABSOLUTE delta has summation noise of order sqrt(N_ambig) * eps *
        // |log L| from parallel (Rayon) summation reordering, which at large N masks
        // the real convergence signal. Normalising by N_ambig gives a per-molecule
        // metric that is numerically well-behaved.
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

        // M-step: single-pass algebraic closed-form.
        //
        // Pre-M-step accumulators collect only (W, S1=Σ w·y, S2=Σ w·y²) per
        // transcript. The σ² deviation Σ w·(y − μ_new)² is reconstructed
        // algebraically AFTER μ is updated — never around a stale μ.
        m_step_ran = true; // see the declaration above: this, not `n_em_iter`,
                           // is what "an M-step ran" means
        let (accum_w, accum_w_y, accum_w_y2) = accumulate_mstep_suffstats(
            &params,
            &graph,
            &unambiguous_indices,
            &ambiguous_indices,
            &final_weights,
            num_transcripts,
        );

        // 3-step update: μ → σ² (around μ_new) → π. The order is locked.
        //
        // STEPS 1 AND 2 ARE SKIPPED ENTIRELY AT `poly_a_weight == 0`. Guarding only
        // the two `accum_w_y*` lines does NOT work: with S1 = 0 that drags μ to 0
        // and σ² onto INSTRUMENTAL_PAL_NOISE_FLOOR.
        //
        // STEP 3 is NOT skipped: alpha is the abundance prior, it does not read
        // poly(A) at all, and skipping it would silently delete the prior.
        for i in 0..num_transcripts {
            if let Some(ref mut p) = params[i] {
                let w_sum = accum_w[i];

                // STEP 0: record W itself, OUTSIDE the poly(A) guard below and
                // beside STEP 3, for the same reason STEP 3 is outside it: at
                // `poly_a_weight == 0` W is still what `alpha` is built from, and
                // a W column that empties itself would make the sidecar
                // unreadable on exactly the configuration that most needs it.
                //
                // This is `p.em_accum_weight`, NOT the `accum_w` local above — the
                // field carries the SUM out of the loop, the local is read-only.
                p.em_accum_weight = w_sum;

                if poly_a_weight > 0.0 {
                    // STEP 1: μ update — weak empirical Bayes shrinkage (τ₀=2).
                    p.mu = (accum_w_y[i] + tau_0 * p.mu_cohort) / (w_sum + tau_0);

                    // STEP 2: σ² update — algebraic closed-form around μ_new.
                    //   Σ w·(y − μ_new)² = S2 − 2·μ_new·S1 + μ_new²·W   (identity).
                    let s1 = accum_w_y[i];
                    let s2 = accum_w_y2[i];
                    let sq_dev_new = (s2 - 2.0 * p.mu * s1 + p.mu * p.mu * w_sum).max(0.0);
                    // MODEL A — fixed-absolute-precision prior.
                    //
                    // Model B (the joint NIG MAP) would put `tau_0*(mu - mu_cohort)^2` back in
                    // the NUMERATOR and `+3` in the denominator. Both changes go together:
                    // keeping one without the other is a different estimator, not a variant.
                    //
                    // A vs B HAS NO ORDER — the quadratic raises the numerator and +3 raises
                    // the denominator, so which is larger is decided by W and by the mu shift.
                    // Do not claim an ordering.
                    //
                    // The trailing `+ 2` in the denominator is `sigma2_dof_offset`, pinned to
                    // `SIGMA2_DOF_OFFSET == 2.0`. At other `c` this derivation no longer holds,
                    // which is why `c` is a knob and not a second estimator.
                    p.sigma2 = (sq_dev_new + 2.0 * beta_0 * p.sigma2_cohort)
                        / (w_sum + 2.0 * nu_0 + sigma2_dof_offset);
                    p.sigma2 = p.sigma2.max(INSTRUMENTAL_PAL_NOISE_FLOOR);
                }

                // STEP 3: π update.
                // Defensive: with `n_total_molecules == 0` every `p.alpha` would become
                // 0/0 = NaN.
                p.alpha = if n_total_molecules > 0.0 {
                    w_sum / n_total_molecules
                } else {
                    0.0
                };
            }
        }

        // Hand the M-step's own suffstats out of the block by MOVE: `accum_w` is dead
        // past the `let w_sum = accum_w[i]` read above, so this is a pointer move —
        // no clone and no second accumulation pass.
        last_mstep_w = accum_w;
    }

    // STAMP THE RUN'S ACTUAL W AND FINALISATION STATE INTO THE SNAPSHOT.
    //
    // The M-step's own assignment is not enough: the snapshot the sidecar is
    // written from is taken BEFORE the M-step runs, and on convergence the loop
    // breaks before the M-step too.
    //
    // `accumulate_mstep_suffstats` is a pure function of (`params` presence, the
    // graph, the indices, `final_weights`), and `final_weights` is exactly what
    // the profile writer consumes — so the sidecar's W describes the SAME
    // weights that produced that profile.
    //
    // The convergence `break` fills `final_weights` before breaking, so W is
    // stamped whether or not an M-step ran; `em_finalized` says which.
    //
    // `alpha` is stamped here too: `alpha == W / n_total_molecules` by
    // definition, so stamping one and not the other leaves a row whose two
    // columns disagree.
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
                // Same accumulator, same generation, same row. The `else` branch is the
                // guard above, kept identical: a BAM with zero mapped molecules must not
                // publish NaN here either.
                p.alpha = if n_total_molecules > 0.0 {
                    final_w[i] / n_total_molecules
                } else {
                    0.0
                };
                p.em_finalized = m_step_ran;
            }
        }

        // How far apart are the W the M-step CONSUMED and the W the sidecar
        // PUBLISHES, for the same transcript? They are one E-step apart by
        // construction; these two fields make the offset a number the CLI can print.
        //
        // The relative form divides by the LARGER of the two arms, so a transcript
        // that gained weight reports its proportional move, and transcripts that are
        // zero in both contribute exactly 0.
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

    // 4. Build output parquet.
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
    // EMISSION COVERAGE. Denominator for the two read-weighted numbers the CLI
    // prints: `n_unambiguous + n_ambiguous`, i.e. reads with at least one
    // candidate. Molecules with `candidate_count == 0` are dropped by the split
    // above and are NOT in any denominator here.
    let mut n_reads_any_emission_candidate = 0_u64;
    let mut n_reads_single_emission_candidate = 0_u64;
    // DENOMINATOR for `zero-power_resolved`. Read-level, not candidate-weighted:
    // a candidate-weighted coverage over this slice comes out near 100% by
    // construction.
    let mut n_reads_multi_emission_candidate = 0_u64;
    let mut n_polya_zero_power_resolved = 0_u64;
    let mut n_entropy_clean = 0_u64;
    let mut n_entropy_partial = 0_u64;
    let mut n_entropy_degenerate = 0_u64;
    let mut n_degenerate_all_imputed = 0_u64;
    let mut n_degenerate_with_anchored = 0_u64;
    let mut n_polya_zero_power = 0_u64;

    // Per-read assignments are filled into a molecule-indexed slot vector and
    // flattened at the end, so the emitted table is in MOLECULE order and carries
    // exactly one row per read. The two loops write disjoint slots, and slotting
    // by molecule index keeps the read_id attached to its OWN posterior: zipping
    // `final_weights` against `graph.molecules` would mis-pair them, because
    // `final_weights` is indexed by AMBIGUOUS RANK.
    //
    // Allocated only when the caller asked for it.
    let mut read_assign: Vec<Option<ReadAssignment>> = if want_read_assignments {
        vec![None; graph.molecules.len()]
    } else {
        Vec::new()
    };

    // Unambiguous rows: 1 per read.
    for &idx in &unambiguous_indices {
        let mol = &graph.molecules[idx];
        let tx_id = graph.candidate_pool[mol.candidate_offset as usize] as usize;
        let pal_i32 = poly_a_length_from_log2(mol.log_pal);
        // Observational columns. Both are per READ properties, derived once here and
        // repeated verbatim on every row of this read — they do NOT vary with the
        // candidate a row describes. `candidate_count` is read off the graph as-is,
        // BEFORE the `ΔAS` prune and the `min_weight` truncation can drop rows.
        let ncand = mol.candidate_count;
        let as_ratio = read_max_as_ratio(&graph, mol);
        // EMISSION COVERAGE, counted over ALL reads. An unambiguous read has exactly
        // one candidate, so its `n_with_emission` is 0 or 1 and is readable straight
        // off the lookup below. These counters are deliberately NOT scoped to the
        // ambiguous branch: the motivating figure was measured over profile rows,
        // which include unambiguous reads.
        let un_with_emission = params[tx_id].is_some();
        n_reads_any_emission_candidate += u64::from(un_with_emission);
        n_reads_single_emission_candidate += u64::from(un_with_emission);
        if let Some(ref p) = params[tx_id] {
            if !p.is_competent {
                n_unambiguous_dropped_incompetent += 1;
                // One candidate, one fate. The read-level counter above and the
                // candidate-level counter here are the same number; they are kept separate so
                // the ledger does not depend on the invariant "an unambiguous read has
                // exactly one candidate" holding somewhere else in the file.
                ledger.unambig_incompetent += 1;
                continue;
            }
            if want_read_assignments {
                // An unambiguous read has one candidate, so its posterior is the
                // deterministic `w ≡ 1.0` and its entropy is exactly 0.0. Emitting these is
                // not optional bookkeeping: they are the bulk of the read set, so a table of
                // ambiguous reads only would answer a much smaller question than the one
                // this table exists for.
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
            // Transcript missing from the emission parquet → skip. This is the case that
            // PADDING removes, and it is independent of the `!p.is_competent` case
            // above: only one of them is an emission-coverage problem.
            n_unambiguous_dropped_missing_emission += 1;
            ledger.unambig_missing += 1;
        }
    }

    // Ambiguous rows: per (read × candidate) with posterior ≥ `min_weight`.
    //
    // `min_weight` is a CALLER-SUPPLIED parameter: at the shipped 0.01 a read whose
    // losing candidate sits below 0.01 loses that row and has its survivors
    // renormalised to 1.0, so the arm that shifts mass onto a candidate emits MORE
    // rows than the arm that does not.
    //
    // Any ablation whose arms emit different row counts must be re-run at
    // `min_weight = 0` before its delta is quoted.
    for (i, weights) in final_weights.into_iter().enumerate() {
        let mol_idx = ambiguous_indices[i];
        let mol = &graph.molecules[mol_idx];
        let pal_i32 = poly_a_length_from_log2(mol.log_pal);
        // Observational columns, per READ — computed once, before the `ΔAS` prune and
        // the `min_weight` truncation below can drop rows, and repeated verbatim on
        // every emitted row of this read. `candidate_count` is the ORIGINAL count off
        // the graph, NOT the number of rows this loop emits.
        let ncand = mol.candidate_count;
        let as_ratio = read_max_as_ratio(&graph, mol);

        // DEGRADATION TELEMETRY — how often poly(A) had no say at all. See
        // `polya_emission_count_and_identity` for the definition.
        //
        // `last_estep_params`, NOT `params`: this must answer "did poly(A) cancel in
        // the posterior we are emitting?", and the posterior came from the last
        // E-step. `params` has since been through one more M-step.
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
        // `zero-power_all` keeps its original meaning: `n_with_emission > 0 &&
        // all_identical`. It is retained so earlier numbers stay comparable.
        if n_with_emission > 0 && all_identical {
            n_polya_zero_power += 1;
        }

        // CANDIDATE CONSERVATION LEDGER. Taken for EVERY contested read, not only for
        // ghosts: a counter incremented inside the ghost branch accumulates over the
        // ghost subpopulation and then reads as though it described the whole run.
        // The census is called ONCE and both consumers read the result.
        let av = candidate_availability(&graph, mol, &last_estep_params, prune_max_delta_as);
        ledger.pruned_delta_as += av.n_pruned_delta_as as u64;
        ledger.missing += av.n_missing as u64;
        ledger.incompetent += av.n_incompetent as u64;
        ledger.alpha_zero += av.n_alpha_zero as u64;
        ledger.live_candidates += av.n_live as u64;

        // Entropy 3-tier + ghost-drop telemetry.
        if weights.is_empty() {
            n_dropped_no_competent_candidates += 1;
            // GHOST DECOMPOSITION. `weights.is_empty()` on its own conflates three
            // unrelated causes — absent emission row, `is_polyA_competent == false`, and
            // `alpha <= 0.0` — and cannot be attributed without a census. The census
            // replays the E-step's own filter over the read's ORIGINAL candidate range,
            // against `last_estep_params` for the same reason the telemetry above uses
            // it: `final_weights` came from the pre-M-step model.
            //
            // `n_dropped_no_competent_candidates` stays the TOTAL, so the three cause
            // counters are a partial decomposition of it.
            match av.uniform_cause(mol.candidate_count as usize) {
                Some(GhostCause::Missing) => n_ghost_read_all_missing += 1,
                Some(GhostCause::Incompetent) => n_ghost_read_all_incompetent += 1,
                Some(GhostCause::AlphaZero) => n_ghost_read_all_alpha_zero += 1,
                None => {}
            }
        } else {
            let h = posterior_entropy(&weights);
            // The 'degenerate' entropy bound is 0.70, so a 2-way ambiguous read with
            // balanced 50/50 weights (H = ln 2 ≈ 0.693) classifies as 'partial'.
            if h < 0.20 {
                n_entropy_clean += 1;
            } else if h <= 0.70 {
                n_entropy_partial += 1;
            } else {
                n_entropy_degenerate += 1;
                // Split the degenerate reads by WHY they are flat.
                //
                // `weights` carries the GLOBAL transcript id of every surviving candidate,
                // and `params` is indexed by that same global id — so the imputed flag is
                // one lookup, with no second pass and no change to the hot path's arithmetic.
                //
                // This deliberately does NOT also split on "ΔAS == 0 among the survivors":
                // `raw` is NOT index-aligned with the candidate list (a candidate with no
                // emission row pushes NOTHING), so the survivors cannot be mapped back to
                // `graph.candidate_as` positions without duplicating the E-step's survival
                // predicate.
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
        // Three shapes, and the middle one is the only silent one: a GHOST read has no
        // profile row (and is recorded with `top_tx_id: None`, because "could not
        // assign it" and "never in this run" are different facts); a `min_weight`-culled
        // read HAS a real argmax and IS published under it at weight 1.0, counted as
        // `reads_retained_by_argmax` so a forced assignment is never mistaken for an
        // earned one; and a read with a surviving candidate is recorded with a real
        // `top_tx_id`.
        //
        // The argmax is computed ONCE, here, and unconditionally: the output stage
        // below needs the same answer, and a second copy of this fold would be a
        // second, independently driftable copy of the tie-breaking rule.
        let (top, top_w) = weights.iter().fold((None, 0.0f64), |acc, &(tx, w)| {
            // Strict `>` (not `>=`) => on an exact tie the FIRST candidate in pool order
            // wins, deterministically. Read the rule as stated: flipping it to `>=`
            // changes which transcript a tied read is assigned to.
            if w > acc.1 {
                (Some(tx), w)
            } else {
                acc
            }
        });

        // No case filter is needed any more, and keeping one would be a bug: the
        // ghost clause must stay, because dropping it from this table would make
        // "the solver looked and could not assign it" indistinguishable from "this
        // read was never in this run".
        if want_read_assignments {
            // `weights` is pre-truncation and pre-renormalisation (it is what the M-step
            // consumed), so `top_weight` here is the RAW E-step weight. The
            // `posterior_weight` column in the profile is `w / retained_sum` after the
            // `min_weight` cutoff. Different quantities on purpose.
            read_assign[mol_idx] = Some(ReadAssignment {
                read_id: mol.read_id.clone(),
                n_candidates_before_cutoff: ncand as usize,
                // `None` only on the ghost path: the fold never ran, so there is no candidate
                // to name.
                //
                // The `params` lookup below cannot fail for a tx that is in `weights` — the
                // E-step only pushes a tx for which `params[tx_id]` is `Some`. So the
                // `get`/`and_then` is defensive plumbing, NOT the reason the
                // `top_tx_id == Some(..) => profile row` implication holds.
                //
                // That implication rests on a separate precondition: the E-step pushes `-inf`
                // for any candidate with `!p.is_competent || p.alpha <= 0.0`, so `weights` can
                // contain competent candidates ONLY. Neither the fold nor this lookup
                // re-checks competence.
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

        // Renormalize the retained weights so each read's emitted posterior weights
        // sum to 1.0. The denominator is taken over the rows actually emitted, i.e.
        // `{w >= min_weight}` — the truncation above discards real probability mass.
        // This is a *pure output-stage* correction: the M-step consumed the
        // untruncated `final_weights` before this point.
        //
        // COUPLING: `{w >= min_weight}` is the right denominator only because the
        // E-step's `if let Some` (above) *skips* candidates with missing params
        // instead of pushing `-inf`. If that ever changes, the later filters here
        // become reachable and this denominator silently breaks.
        //
        // THE COUPLED PREDICATE. This filter and the `w < min_weight` test in the
        // emit loop below are two spellings of ONE decision, and they must agree: if
        // this returned 0 while the loop still emitted the argmax, `w_norm` would be
        // `w / 0.0` and a NaN would be written into `posterior_weight` — and nothing
        // in the suite would go red, because a NaN compares false against every
        // expectation.
        //
        // When `min_weight` culls EVERY live candidate, the read would otherwise
        // vanish from both tables. So the argmax is retained on its own and
        // `retained_sum` collapses to its weight, which makes its `w_norm` exactly
        // 1.0. `reads_retained_by_argmax` counts exactly those reads.
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

        // The census's `n_live` claims this many candidates reached `weights`.
        // Comparing the two is the only check that the census and the E-step
        // still agree; see `CandidateLedger::census_vs_estep_deficit`.
        ledger.live_candidates_entered += weights.len() as u64;

        // True when `min_weight` culled the whole read, so the argmax below is
        // being retained under protest rather than earning its row.
        let argmax_only = weights.iter().all(|&(_, w)| w < min_weight);

        for (tx_id, w) in weights {
            if w < min_weight && !(argmax_only && Some(tx_id) == top) {
                // Provably 0 at `min_weight = 0.0`: every posterior is `exp(ll - log_z) >= 0`,
                // so `w < 0.0` is never true. A test pins that.
                //
                // The argmax is the ONE exception: it is culled from the prune count because
                // it is NOT culled — it is emitted, one line below. Counting a candidate as
                // pruned while also emitting a row for it would double-count it against
                // `terminal_sum()`.
                ledger.pruned_min_weight += 1;
                continue;
            }
            // Safe: this body only runs when at least one weight is retained.
            // `retained_sum > 0` follows from the softmax summing to 1 over the
            // candidates that survived `filter(ll > -inf)`, NOT from `min_weight > 0`.
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
                    // The read is being published under a forced argmax: every live candidate was
                    // below `min_weight`. Counted here, at the point the row genuinely exists,
                    // rather than where `argmax_only` is computed.
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
        // The two columns below are OBSERVATIONAL ONLY — nothing in TEL-EM reads them
        // back, and they do not participate in A_rk, the `min_weight` cutoff, or the
        // renormalization. Both are per READ properties repeated identically on every
        // row of that read.
        //
        // CAVEAT — THE EMITTED DISTRIBUTION IS SURVIVAL-BIASED. Both columns ride on
        // rows that survive TWO independent filters: the E-step's `ΔAS >
        // prune_max_delta_as` prune, and the `min_weight` cutoff. Reads with decisive
        // evidence are therefore over-represented relative to genuinely contested
        // reads. Any analysis of these columns MUST group by `read_id` and report
        // the read-level coverage rate alongside the distribution.
        //
        // The molecule's ORIGINAL candidate count, captured before BOTH filters drop
        // rows. It differs from the number of emitted rows whenever either fired, and
        // it is NOT the count the E-step normalised over (the E-step saw
        // `candidate_count − n_pruned`).
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

    // THE CANDIDATE CONSERVATION TRIPWIRE.
    //
    // Deliberately AFTER the parquet is closed: the profile is the expensive
    // artefact, and deleting it because an instrumentation counter disagreed
    // would destroy the evidence needed to find out why.
    //
    // Deliberately HERE and not in the CLI: `solve_and_write_profile` is a public
    // entry point and every test calls it directly, so an invariant that only the
    // binary checks is one the test suite never exercises.
    //
    // A non-zero value means one of exactly two things: (a) the census and the
    // E-step have drifted apart, so a class of candidate reached no terminal
    // bucket; or (b) a candidate produced a non-finite posterior, so
    // `filter(ll > -inf)` dropped it while the census counted it live.
    //
    // It does NOT mean candidates are being lost from the profile: `min_weight`
    // is a caller-supplied cutoff and a candidate falling below it belongs in
    // `pruned_min_weight`.
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
    // Same condition as the assert above, not an independent one -- see
    // `CandidateLedger::census_vs_estep_deficit`. Kept because when it fires it
    // names WHICH quantity disagreed, and `deficit()` alone reports the symptom.
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
        // `last_estep_params`, NOT `params` — see the declaration. On the
        // `max_iter`-exhaustion path `params` is one M-step past the state that
        // produced `final_weights`.
        //
        // `take`, not `clone`: the telemetry above has already run, so nothing reads
        // the snapshot again.
        *out = std::mem::take(&mut last_estep_params);
    }

    if let Some(out) = output_read_assignments {
        // `flatten()` drops the `None` slots. Every molecule is visited by exactly one
        // of the two loops above, and what survives obeys the weaker, one-directional
        // invariant the cross-run join needs:
        //
        //     every emitted assignment either HAS a profile row, or is a ghost with
        //     `top_tx_id = None`.
        //
        // The `None`s left are exactly the reads with no profile row AND no reason to
        // be counted: transcripts absent from the emission parquet, and
        // `min_weight`-culled reads. Ghosts are deliberately NOT in this list — they
        // are recorded, because "unassignable" and "never seen" must not collapse
        // into the same absence when two runs are compared.
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

/// Round-trip polyA length through log2.
fn poly_a_length_from_log2(log_pal: f32) -> i32 {
    log_pal.exp2().round() as i32
}

/// DEGRADATION TELEMETRY. Returns `(n_with_emission, all_identical)` for the
/// read's candidate set. `all_identical` is true iff every candidate that
/// carries an emission row shares one identical `(mu, sigma2)` pair, so the
/// poly(A) term cancels exactly in every pairwise odds ratio.
///
/// The comparison is EXACT `==`, NOT A TOLERANCE: the shared fits are
/// bit-identical copies by construction, so `==` fires on exactly the rows the
/// fallback produced. Do not loosen it — the metric would then over-report
/// zero discriminative power, the expensive direction for a degradation
/// headline.
///
/// A candidate with NO emission row is skipped rather than treated as a
/// mismatch: a ghost transcript has no `(mu, sigma2)` to disagree with, and it
/// is already counted by `n_dropped_no_competent_candidates`. A read with no
/// candidate that has an emission row at all returns `(0, false)`.
///
/// A read with exactly ONE candidate carrying an emission row returns
/// `(1, true)`: nothing was compared, so the count is what makes that
/// vacuity visible. `zero-power_all` keeps counting those reads;
/// `zero-power_resolved` excludes them.
///
/// The read's FULL candidate range is used, not the post-prune /
/// post-truncation set: the question is what evidence the read HAD.
///
/// The loop never early-returns: an early return would undercount the present
/// candidates on exactly the DISAGREEING reads — the ones where poly(A)
/// actually did something.
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

/// How many candidates must carry an emission row before "they all share one
/// `mu`" is a MEASURED statement rather than a comparison that never happened.
///
/// With one candidate there is no pairwise odds ratio to cancel. This is the
/// threshold that separates `zero-power_all` from `zero-power_resolved`; do
/// not inline the literal.
const ZERO_POWER_MIN_EMISSION_CANDIDATES: usize = 2;

/// Per-read observational statistic: `second_largest_AS / largest_AS` over the
/// read's FULL candidate range, emitted on every row of that read.
///
/// Three defined cases, all yielding `0.0`:
/// - `candidate_count == 1` — there is no second candidate.
/// - `largest_AS <= 0` — every candidate lacks an `AS:i` tag, so the ratio
///   would be `0/0`; `0.0` means "no alignment evidence", not "a tie".
/// - `second_AS < 0` — `AS` is a raw `i32` and a malformed BAM can carry a
///   negative value, which would otherwise yield a negative "proportion".
///
/// Otherwise `second_largest / largest`, in `[0, 1]`; the division is done in
/// `f64` and cast down, because an `f32` division of two `i32`s would lose
/// precision before the narrowing.
///
/// The `0.0` SENTINEL is deliberately overloaded across those three
/// situations, so a query of the form "fraction of reads with
/// `max_as_ratio < 0.5`" silently pools all three.
///
/// NOT USED BY THE SOLVER: no threshold reads it, and it is deliberately NOT
/// fed into `A_rk`. It is emitted so the decision can be made on real data.
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

/// Why a read's candidate set contains nothing that could carry posterior mass.
///
/// The census is taken over the read's ORIGINAL candidate range, and each
/// bucket is defined by the E-step's own decision procedure, in the E-step's
/// own order: a read is a ghost exactly when the E-step's `max_ll` came out
/// `-inf`.
///
/// The two ways a candidate can be absent from the E-step's `raw` vector are
/// NOT the same and must not share a bucket:
/// - **pruned** (`delta_as > prune_max_delta_as`) — the E-step pushes `-inf`,
///   so the entry IS in `raw`;
/// - **no emission row** — the `if let Some(ref p) = params[tx_id]` arm has no
///   `else`, so the entry is never pushed at all.
///
/// The prune is counted in its own bucket for the ledger's benefit and still
/// excluded from cause attribution: the two questions are different — "would
/// padding have saved this read?" and "where did this candidate go?".
struct CandidateAvailability {
    /// No emission row at all — the candidate never reached `raw`.
    n_missing: usize,
    /// Present, but `is_polyA_competent == false` — pushed as `-inf`.
    n_incompetent: usize,
    /// Present and competent, but `alpha <= 0.0` — pushed as `-inf`.
    ///
    /// Distinct from `n_incompetent`: the two are independent facts about the
    /// same transcript, and the M-step can drive `alpha` to exactly 0 without
    /// touching `is_competent`. A combined "incompetent or starved" bucket would
    /// hide which of the two happened.
    n_alpha_zero: usize,
    /// Killed by the E-step's `ΔAS` prune — `delta_as > prune_max_delta_as`.
    ///
    /// This bucket is what makes the candidate ledger closable: without it a pruned
    /// candidate would belong to no bucket at all, so the census would account for
    /// fewer candidates than the graph produced.
    ///
    /// It is deliberately NOT consulted by [`CandidateAvailability::uniform_cause`].
    /// A pruned candidate is dead for a reason no amount of emission padding
    /// addresses, so attributing a read with one to `Missing` would over-claim.
    n_pruned_delta_as: usize,
    /// Carried posterior mass. A ghost read has `n_live == 0`; see
    /// [`CandidateAvailability::is_ghost`].
    ///
    /// NOT a terminal fate: live candidates go on to be split by `min_weight` and by
    /// the output stage's own competence recheck, so this bucket is EXCLUDED from
    /// [`CandidateLedger::deficit`].
    n_live: usize,
}

impl CandidateAvailability {
    /// The single cause shared by EVERY candidate, when exactly one cause accounts
    /// for all of them; `None` when the read's candidates died for a mix of
    /// reasons, or when any candidate was pruned. `candidate_count` is the read's
    /// original candidate count — the denominator — and pruned candidates are IN
    /// it and in NO bucket, so a read with a pruned candidate plus a uniformly
    /// missing remainder is reported as `None`.
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

/// The one cause behind a ghost read, when the read has exactly one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GhostCause {
    Missing,
    Incompetent,
    AlphaZero,
}

/// One walk of a read's candidate range, classifying each candidate by the
/// E-step's own filter. See [`CandidateAvailability`] for why the buckets are
/// defined this way.
///
/// `params` must be `last_estep_params`, NOT `params`: the census explains a
/// `final_weights` produced from the pre-M-step model, and `alpha` is exactly
/// the field the M-step rewrites.
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
    // candidate range, taken before competence masking and including candidates
    // that have no emission row. Reproducing it exactly is what makes the
    // `delta_as` below comparable to the prune the E-step actually applied.
    let max_as = graph.candidate_as[off..off + n]
        .iter()
        .fold(i32::MIN, |a, &b| a.max(b));
    for c in 0..n {
        let idx = off + c;
        let tx_id = graph.candidate_pool[idx] as usize;
        // `saturating_sub`, matching the E-step. A plain `-` would wrap under
        // `--release` and a wrapped NEGATIVE delta_as would pass the prune, putting a
        // candidate the E-step killed into the `n_live` bucket.
        let delta_as = max_as.saturating_sub(graph.candidate_as[idx]);
        if delta_as > prune_max_delta_as {
            // Counted, but not attributed: see the struct doc. The E-step pushes this
            // candidate as `-inf`, so it never reaches `weights` and never emits a row;
            // without this increment it is a candidate the graph produced and no bucket
            // claims.
            av.n_pruned_delta_as += 1;
            continue;
        }
        let Some(p) = params.get(tx_id).and_then(|o| o.as_ref()) else {
            av.n_missing += 1;
            continue;
        };
        // Order matches the E-step's `!p.is_competent || p.alpha <= 0.0`, and it is
        // load-bearing for a candidate that is BOTH: competence is tested first, so a
        // doubly-dead candidate is attributed to incompetence.
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
/// WHAT "TERMINAL FATE" MEANS, and the one trap in this struct. A candidate is
/// counted exactly once, at the point where it stops moving. The E-step census
/// splits the contested read's candidates four ways, and the survivor bucket
/// is then split three ways at the output stage — so the survivors are counted
/// there and NOT here. `live_candidates` is recorded because
/// [`CandidateLedger::census_vs_estep_deficit`] needs it, but
/// [`CandidateLedger::deficit`] deliberately does not sum it: adding an
/// intermediate to the sum of its own descendants double-counts every live
/// candidate.
///
/// SCOPE. `total_candidates` counts every candidate of every molecule with
/// `candidate_count >= 1`, including candidates on reads that emitted nothing
/// at all. That is the point: a read that produced no row contributes to the
/// left and to no row on the right, and the ledger's job is to make it
/// visible.
///
/// The two molecule populations are counted separately because they cannot die
/// the same way. A single-candidate molecule has `max_as` equal to its own
/// `AS`, so `delta_as == 0` and the `ΔAS` prune can never fire on it; the
/// unambiguous arm also applies no `alpha` mask, by design, since its
/// posterior is the deterministic `w ≡ 1.0`. Its three buckets are therefore
/// exhaustive among themselves and there is no `unambig_*_pruned` column
/// because there is nothing to put in it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CandidateLedger {
    /// Σ `MoleculeEvidence::candidate_count` over molecules with
    /// `candidate_count >= 1`. This is the ledger's left-hand side.
    pub total_candidates: u64,

    // --- contested arm, decided by the E-step census ---
    /// Killed by `ΔAS > prune_max_delta_as`.
    pub pruned_delta_as: u64,
    /// No emission row for this transcript.
    pub missing: u64,
    /// `is_polyA_competent == false`.
    pub incompetent: u64,
    /// Competent but `alpha <= 0.0`, i.e. it won no posterior mass anywhere.
    pub alpha_zero: u64,
    /// Candidates that reached `weights`. INTERMEDIATE — see the struct doc.
    pub live_candidates: u64,

    /// Live candidates that were culled by `min_weight` AND not emitted.
    ///
    /// The argmax of an all-culled read is culled-but-emitted, so it is NOT
    /// counted here; see `reads_retained_by_argmax`. Counting both would
    /// double-count that candidate against `terminal_sum()`.
    ///
    /// Not a constant: `min_weight` is a caller-supplied parameter, so this bucket
    /// moves when the flag moves. At `min_weight = 0.0` it is provably exactly 0.
    /// Its MAGNITUDE is not a stable baseline; only a non-zero deficit is a bug
    /// signal.
    pub pruned_min_weight: u64,
    /// Reads whose every live candidate fell below `min_weight`, and whose argmax
    /// was therefore retained and emitted with `posterior_weight = 1.0` anyway.
    ///
    /// THIS COUNTS READS, NOT CANDIDATES, so it must NOT appear in
    /// `terminal_sum()` -- it is a cross-cutting annotation of `rows_emitted`, not
    /// a disjoint fate competing with it.
    ///
    /// It exists so the cost of the retention is measurable: a retained argmax is
    /// a low-confidence assignment published as a confident one.
    pub reads_retained_by_argmax: u64,
    /// Passed `min_weight`, then dropped by the output stage's own
    /// `is_competent` / `params` recheck. Structurally 0 in production — the
    /// E-step already filtered both — which is the point of counting it.
    pub dropped_output_competence: u64,
    /// Candidate rows actually written to the profile parquet, contested arm.
    pub rows_emitted: u64,

    // --- unambiguous arm ---
    pub unambig_incompetent: u64,
    pub unambig_missing: u64,
    pub unambig_emitted: u64,

    /// Σ `weights.len()` over contested reads — the other side of
    /// [`CandidateLedger::census_vs_estep_deficit`].
    pub live_candidates_entered: u64,
}

impl CandidateLedger {
    /// Terminal fates only. See the struct doc for why `live_candidates` is
    /// recorded but not summed.
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
    /// **Must be exactly 0.** A non-zero value is a real defect in the
    /// instrumentation — a candidate class that reached no bucket.
    pub fn deficit(&self) -> i64 {
        self.total_candidates as i64 - self.terminal_sum() as i64
    }

    /// `live_candidates` minus `live_candidates_entered`.
    ///
    /// **This is NOT an independent check on `deficit()`.** It is algebraically
    /// the same condition, and it is kept for a narrower reason than independence:
    /// when it fires, it says *which* quantity disagreed. `deficit()` reports the
    /// symptom; this names the cause. Deleting it would be defensible; calling it
    /// a second identity is not.
    pub fn census_vs_estep_deficit(&self) -> i64 {
        self.live_candidates as i64 - self.live_candidates_entered as i64
    }
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

/// Per-read posterior summary, for callers that need read-level outcomes
/// (top-1 switch rate between two runs, entropy shift) which the per
/// (read, candidate) profile cannot express.
///
/// Three decisions here are load-bearing:
/// - `top_tx_id` is the transcript NAME, not the pool index. The index is only
///   meaningful relative to one run's `transcript_names` order, and this table's
///   whole purpose is being joined against a SECOND run's table.
/// - `n_candidates_before_cutoff` is `MoleculeEvidence::candidate_count`, i.e.
///   the fan-out BEFORE the `ΔAS` prune, the `is_competent` filter and the
///   `min_weight` truncation. It matches the `n_candidates_before_cutoff`
///   column the profile parquet already emits.
/// - `top_tx_id` is `None` for a GHOST read — no competent candidate, empty
///   `weights`. An argmax over an empty vector would either panic or, worse,
///   yield transcript 0. The ghost has no row in the profile either, so this
///   table is deliberately NOT a subset of the profile's reads. It obeys the
///   weaker one-directional invariant the cross-run join needs:
///
///   ```text
///   top_tx_id == Some(..)  IMPLIES  that read has a profile row.
///   ```
///
///   A `None` means "the solver looked and could not assign", which is not
///   the same as "this read was absent from the run".
#[derive(Debug, Clone)]
pub struct ReadAssignment {
    pub read_id: String,
    pub n_candidates_before_cutoff: usize,
    /// `None` for a ghost read (every candidate dropped before the softmax).
    /// This is why the parquet column is nullable: the cross-run join needs
    /// "unassignable" to be a value it can see, not an absence it has to infer.
    pub top_tx_id: Option<String>,
    /// Raw E-step weight of the top candidate: PRE `min_weight` truncation and PRE
    /// renormalisation. It is therefore NOT equal to the `posterior_weight` the
    /// profile parquet reports for the same read, which is `w / retained_sum`.
    pub top_weight: f64,
    /// Posterior entropy in nats, recomputed from the final weights with the same
    /// `posterior_entropy` the output stage uses. 0.0 for an unambiguous read,
    /// consistent with its deterministic `w ≡ 1.0` branch.
    pub entropy: f64,
}

/// Write the per-read posterior summary as Parquet, matching every other
/// output this tool produces (profile, emission, anchors, summary are all
/// Parquet; TSV appears in this codebase only as an INPUT format).
///
/// `top_tx_id` is written as a nullable string rather than an index: the whole
/// purpose of this table is to be joined against another run's table, and a
/// pool index only means something relative to one run's `transcript_names`.
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

    // Unambiguous: w=1.0 always.
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
        // Null, not "" and not index 0: an empty string would be a transcript that
        // does not exist, and index 0 is a real transcript this read was never
        // assigned to.
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
/// The first five columns are named and typed to match the emission parquet, so
/// the result loads back through `load_emission_parameters` unchanged and a
/// converged fit can be re-measured without re-deriving it. `load_emission`
/// `column_by_name`s exactly those five and ignores the rest.
///
/// The remaining columns exist so the file can be audited:
/// - `mu_cohort` — the shrinkage prior, so a reader can see that re-loading
///   this file RE-ANCHORS the prior: the loader sets `mu_cohort = mu_logPAL`
///   AND `sigma2_cohort = sigma2_logPAL`, so BOTH priors move to the fitted
///   values and τ₀=2 / 2β₀ are applied a second time.
/// - `alpha` — the mixture weight the last E-step used.
/// - `state` — which iterate this row holds.
/// - `em_accum_weight` — the E-step weight sum W, in molecule weight units.
/// - `mu_fitted`, `sigma2_fitted` — ALIASES of `mu_logPAL` / `sigma2_logPAL`,
///   same values, same iteration, kept so a reader who wants "the fitted
///   value" does not have to adopt the emission schema's name for it.
/// - `em_finalized` — whether an M-step ran at all.
///
/// `em_accum_weight` and `alpha` are the two sides of `alpha == W /
/// n_total_molecules`, so they agree whenever `em_finalized` is TRUE. When it
/// is FALSE — convergence on the first iteration — `alpha` is the Laplace
/// initialisation instead, so the flag is NOT recoverable from the numeric
/// columns. Filter on `em_finalized` before treating a row as a fit.
pub fn write_emission_params_parquet(
    params: &[Option<TranscriptParam>],
    path: &std::path::Path,
) -> Result<()> {
    // `BooleanBuilder` is NOT at module level (unlike the others), so it is
    // imported here. Verified by compiling, not by reading the import list.
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
    // as in the emission parquet. Writing a placeholder `mu` would invent a
    // parameter for a transcript the model never scored.
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

/// TEL-EM execution stats for logging / telemetry.
#[derive(Debug, Clone)]
pub struct TelEmStats {
    /// Candidate conservation ledger. `deficit()` is asserted to be 0 inside
    /// `solve_and_write_profile` before this is ever returned, so reaching a
    /// caller with a non-zero deficit is not reachable through that entry point;
    /// the field is public because the CLI prints the decomposition.
    pub ledger: CandidateLedger,
    pub n_unambiguous: usize,
    pub n_ambiguous: usize,
    pub n_em_iter: usize,
    /// READ-LEVEL, unambiguous population. Denominator `n_unambiguous`.
    /// An unambiguous read has exactly one candidate, so this counts reads whose
    /// single candidate was present and `is_polyA_competent == false`.
    ///
    /// The contested case is `n_ghost_read_all_incompetent` and the row-level one
    /// is `n_dropped_output_row_competence`; do not mix the three.
    pub n_unambiguous_dropped_incompetent: u64,
    /// READ-LEVEL, unambiguous population. Denominator `n_unambiguous`.
    /// The read's single candidate has no emission row at all.
    ///
    /// There is deliberately NO unambiguous counterpart to
    /// `n_ghost_read_all_alpha_zero`: the unambiguous branch applies no `alpha`
    /// mask, because a single-candidate read's posterior is the deterministic
    /// `w == 1.0` and its abundance prior is never consulted. The two unambiguous
    /// buckets are therefore exhaustive for the drops that can happen there.
    ///
    /// This is the counter padding is expected to drive to zero: a non-zero value
    /// after a padded build is a defect in the padding, not a property of the data.
    pub n_unambiguous_dropped_missing_emission: u64,
    /// READ-LEVEL, ghost population. Denominator
    /// `n_dropped_no_competent_candidates`. Contested reads with no posterior
    /// whose ENTIRE candidate range had no emission row.
    ///
    /// Kept separate from `n_unambiguous_dropped_missing_emission` rather than
    /// summed into it: the two have different denominators, and the whole purpose
    /// of the decomposition is to say what fraction of each population padding
    /// would have rescued.
    pub n_ghost_read_all_missing: u64,
    /// READ-LEVEL, ghost population. Denominator
    /// `n_dropped_no_competent_candidates`. Contested reads with no posterior
    /// whose entire candidate range was present and `is_polyA_competent == false`.
    pub n_ghost_read_all_incompetent: u64,
    /// READ-LEVEL, ghost population. Denominator
    /// `n_dropped_no_competent_candidates`. Contested reads with no posterior
    /// whose entire candidate range was present and competent but had
    /// `alpha <= 0.0` — the abundance prior the M-step drove to zero.
    ///
    /// A genuinely distinct cause from the other two: `is_competent` is
    /// M-step-invariant while `alpha` is exactly the field the M-step rewrites. A
    /// combined "incompetent or starved" bucket would hide which of the two
    /// happened, and they call for opposite remedies.
    pub n_ghost_read_all_alpha_zero: u64,
    /// ROW-LEVEL, and **not** comparable with any of the read-level counters
    /// above. Incremented once per (read × candidate) row by the output stage when
    /// a row's transcript is `!is_competent` or absent from `params`.
    ///
    /// This is 0 in practice, and it is meant to stay 0: the E-step already pushed
    /// such candidates as `-inf` and `filter(ll > -inf)` removed them from
    /// `weights`, so no such row can reach the emit loop. It is a TRIPWIRE: if the
    /// E-step's `-inf` mask is ever removed, this fires, and the read-level
    /// counters would silently stop accounting for the output stage's share of the
    /// loss.
    pub n_dropped_output_row_competence: u64,
    /// READ-LEVEL, ghost population. Contested reads (>= 2 candidates) with NO
    /// posterior at all — `weights.is_empty()`. This is the TOTAL for the
    /// decomposition above.
    ///
    /// The NAME is misleading in two ways: it fires for several causes, not one,
    /// and the reads it counts are reads rather than candidates. It is kept
    /// verbatim because an external validation harness compares its own
    /// Python-side ghost count against this field and fails on a mismatch.
    ///
    /// `n_ghost_read_all_missing + n_ghost_read_all_incompetent +
    /// n_ghost_read_all_alpha_zero` is a PARTIAL decomposition of it and does not
    /// sum to it. The CLI prints the remainder.
    pub n_dropped_no_competent_candidates: u64,
    /// Posterior entropy 3-tier counters (per ambiguous read, in nats).
    /// clean: H < 0.20 nats (near-deterministic assignment).
    pub n_entropy_clean: u64,
    /// partial: 0.20 <= H <= 0.70 nats (two-three-way partial ambiguity).
    pub n_entropy_partial: u64,
    /// degenerate: H > 0.70 nats (effectively uniform over candidates).
    /// The upper bounds are 0.20 / 0.70, NOT 0.20 / 0.60: re-derive these tiers
    /// from the comparison in the E-step, not from this doc comment.
    pub n_entropy_degenerate: u64,
    /// Of the `n_entropy_degenerate` reads, the ones whose ENTIRE surviving
    /// candidate set is IMPUTED (`n_cohort_flnc == 0`, carrying the shared cohort
    /// fit). Unit: READS. Denominator: `n_entropy_degenerate`.
    ///
    /// Those reads are flat for a MECHANISM that is not the same as a genuine
    /// tie: every candidate shares one identical density, so the poly(A) term is
    /// EXACTLY zero in every pairwise odds ratio. A read flat across two real,
    /// differently-fitted transcripts is flat for a different reason, and
    /// conflating the two makes the count unreadable.
    ///
    /// Read this as a DIAGNOSTIC split, not a verdict.
    pub n_degenerate_all_imputed: u64,
    /// `n_degenerate_all_imputed`'s complement inside `n_entropy_degenerate`:
    /// reads that are degenerate with at least one ANCHORED candidate in the set.
    /// Unit: READS. Denominator: `n_entropy_degenerate`.
    ///
    /// `n_degenerate_all_imputed + n_degenerate_with_anchored` must equal
    /// `n_entropy_degenerate` EXACTLY. Three counters that partition a fourth is
    /// only useful if the partition is checked.
    pub n_degenerate_with_anchored: u64,
    /// DEGRADATION TELEMETRY. Contested reads (≥ 2 candidates) whose candidate set
    /// carries ZERO poly(A) discriminative power: every candidate that has an
    /// emission row shares one identical `(mu, sigma2)` pair, so the poly(A) term
    /// cancels EXACTLY in every pairwise odds ratio.
    ///
    /// Read `n_polya_zero_power / n_ambiguous` for the fraction. The denominator is
    /// deliberately `n_ambiguous` and not `n_total`: a single-candidate read has
    /// no pairwise odds ratio at all.
    ///
    /// ⚠️ NOT A VALIDITY METRIC, AND NOT PRINTED. This counter FIRES when a
    /// read's candidates all share one identical `(mu, sigma2)` — it counts the
    /// reads the M-step did NOT personalise. A low value means non-degenerate, not
    /// discriminative. Its only consumers are the zero-power unit tests, which
    /// pin it as a tripwire for parameters collapsing numerically into one
    /// identical float.
    ///
    /// THE COMPARISON IS EXACT `==`, NOT A TOLERANCE, and the consequence is part
    /// of this metric's definition: candidates that disagree by as little as one
    /// ULP are treated as DISCRIMINATING, so the read is not counted here. The
    /// number therefore UNDER-reports zero poly(A) power.
    pub n_polya_zero_power: u64,
    /// DEGRADATION TELEMETRY, second metric. Contested reads carrying at least
    /// `ZERO_POWER_MIN_EMISSION_CANDIDATES` emission rows in which all of them
    /// share one identical `(mu, sigma2)`. Unlike `n_polya_zero_power` this cannot
    /// be satisfied by a read that had nothing to compare.
    ///
    /// Read `n_polya_zero_power_resolved / n_reads_multi_emission_candidate`. Do
    /// NOT divide by `n_ambiguous` — that denominator contains the reads this
    /// metric deliberately excludes.
    ///
    /// Also not printed, for the same reason as `n_polya_zero_power`.
    pub n_polya_zero_power_resolved: u64,
    /// DENOMINATOR for `n_polya_zero_power_resolved`: contested reads with at
    /// least `ZERO_POWER_MIN_EMISSION_CANDIDATES` candidates carrying an
    /// emission row. Can legitimately be 0 on a sparse model. No longer printed
    /// by the CLI, so any ratio built from it must divide by this explicitly
    /// rather than assume a rendered percentage.
    pub n_reads_multi_emission_candidate: u64,
    /// NUMERATOR for emission coverage. Counts over ALL reads with at least one
    /// candidate, not just contested ones: coverage is a property of the model's
    /// completeness, and scoping it to the contested reads would report a
    /// different number than the 31.6% that motivated the metric. Divide by
    /// `n_unambiguous + n_ambiguous` and label the denominator "reads with
    /// >=1 candidate" -- molecules with zero candidates are dropped earlier.
    pub n_reads_any_emission_candidate: u64,
    /// NUMERATOR for the single-candidate share: reads with exactly ONE candidate
    /// carrying an emission row. Same denominator as
    /// `n_reads_any_emission_candidate`.
    ///
    /// ⚠️ THIS IS A STRICT SUPERSET OF THE VACUOUS READS, NOT THE SAME SET: it
    /// includes unambiguous reads with one present candidate, and those never reach
    /// `n_polya_zero_power`. The printed pair does NOT sum to it.
    pub n_reads_single_emission_candidate: u64,
    /// EM trace: per-iter `(iter_index, log_likelihood, mean_entropy)`.
    pub em_trace: Vec<(usize, f64, f64)>,
    /// D2. Largest absolute disagreement, over all transcripts, between the W the
    /// last M-step CONSUMED and the W the sidecar PUBLISHES as
    /// `em_accum_weight`. Unit: molecules. Denominator: none — this is a max, not
    /// a rate. `0.0` when no M-step ran.
    ///
    /// Non-zero means the run's published W is one E-step behind the W that
    /// produced the published `sigma2`. It is a MEASUREMENT of a known
    /// one-iteration phase offset, not a defect.
    pub max_dw_phase_abs: f64,
    /// D2. The same disagreement divided, per transcript, by the larger of the
    /// two arms; `0.0` when no M-step ran or when a transcript is zero in both.
    /// Dimensionless. Denominator: `max(W_final, W_mstep)` per transcript, named
    /// here because a bare ratio against a moving denominator is unreadable.
    pub max_dw_phase_rel: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    use arrow::array::{Array, BooleanBuilder, Int64Array, UInt32Builder};

    use crate::evidence::MoleculeEvidence;

    /// The M-step's three prior constants are pinned BY VALUE.
    ///
    /// Exact equality, not approximate: these are hyper-parameters, not
    /// measurements. A tolerance would let a drift hide inside it.
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
             denominator and 2.0 IS Model A -- Model A drops the quadratic \
             numerator term AND the -0.5 ln v normaliser together, so +2 and \
             +3 are not the same model with a knob turned. The knob exists for \
             the simulation harness; moving it moves the shipped estimator."
        );
    }

    /// The "only naming site" invariant, pinned with a red proof.
    ///
    /// `solve_and_write_profile`'s doc comment claims it is the ONLY place
    /// `SIGMA2_DOF_OFFSET` is named, and the pin test above pins only the
    /// constant's VALUE -- two different claims. Replacing the forwarder's
    /// argument with a literal `2.0` leaves the const declared at 2.0, leaves the
    /// declared-source checker green, and leaves this module green, while the
    /// shipped path stops reading the declared source at all.
    ///
    /// So this counts naming sites in PRODUCTION code and pins the count at 2 --
    /// the declaration, and the one call. Zero means a literal crept into the
    /// shipped path; three or more means a second path can supply its own `c`.
    ///
    /// Two things make the count honest rather than decorative:
    /// - Comments are stripped FIRST. Otherwise the constant's own doc comment
    ///   satisfies the count and the assertion is a tautology.
    /// - The split anchors on the module header, not on the bare attribute: the
    ///   string `#[cfg(test)]` alone occurs more than once in this file, and
    ///   `rfind` on the bare attribute can pick the later one, which lands inside
    ///   this module and makes the slice swallow the pin test above.
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
            // Rust lexer, so a line with `//` inside a string literal would be truncated
            // and the count could fall short of 2.
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

        // The name count has a demonstrated blind spot: a third `pub fn` that calls
        // the impl with a LITERAL never mentions the constant, so the count above
        // stays 2. So pin the call sites too. The definition does NOT match this
        // pattern (`fn solve_and_write_profile_impl<P: AsRef<Path>>(` -- the generic
        // sits between the name and the paren), so the count below is callers only.
        let callers = code.matches("solve_and_write_profile_impl(").count();
        assert_eq!(
            callers, 2,
            "expected exactly 2 callers of solve_and_write_profile_impl -- the \
             shipped forwarder and the parameterised entry point -- but found \
             {callers}. A third caller can supply a literal `c` without ever \
             naming SIGMA2_DOF_OFFSET, which is invisible to the count above."
        );
    }

    /// Output-stage truncation threshold mirrored from `solve_and_write_profile`.
    const MIN_WEIGHT: f64 = 0.01;

    /// Frozen 5'-anchoring weight, mirrored from the CLI default.
    const FIVE_PRIME_WEIGHT: f64 = 1.8;
    /// Frozen 5' window `g_k <= 20`, mirrored from the CLI default.
    const FIVE_PRIME_WINDOW: u32 = 20;
    /// Frozen `ΔAS` prune threshold, mirrored from the CLI default.
    const PRUNE_MAX_DELTA_AS: i32 = 50;
    /// The 5' term switched off. `ln_5p ≡ 0` exactly, so any test passing this
    /// is provably independent of `candidate_5p_gap` — which is what keeps the
    /// pre-existing A_rk and renormalisation tests unperturbed by this change
    /// instead of accidentally also covering the new term.
    const POLY_A_ON: f64 = 1.0;
    const FIVE_PRIME_OFF: f64 = 0.0;
    /// The `ΔAS` prune switched off. Legal (`i32::MAX` is a valid threshold);
    /// used only by the A_rk test, whose `ΔAS = 200` fixture is exactly the
    /// quantity the prune thresholds on, so testing both at once would conflate
    /// them into one number.
    const PRUNE_OFF: i32 = i32::MAX;

    /// A MULTI-molecule `FlatEvidenceGraph`. `synth_graph_with_gaps` hardcodes a
    /// single `MoleculeEvidence`, so no fixture could be built with a known
    /// (W, S1, S2) until this existed.
    ///
    /// Transcript ids are interned in first-seen order, so `params[i]` is the
    /// parameters of the i-th DISTINCT transcript named, not the i-th molecule.
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
    /// NOT A DESCRIPTION OF PRODUCTION: the M-step is Model A. It is retained as
    /// an INDEPENDENT REFERENCE for the estimator production is not using, so the
    /// test can measure the gap in the same units rather than assert that some
    /// golden constant happens to match. If production ever goes back to Model B,
    /// the gap goes to zero and the test goes red.
    ///
    ///     Q = -(W/2) ln(2 pi v) - S(mu)/(2v) - (nu_0+1) ln v - beta_0/v
    ///         - 0.5 ln v - kappa_0 (mu - mu_cohort)^2 / (2v)
    ///
    /// The `-0.5 ln v` is the normalising factor of the conditional prior
    /// `mu | v ~ N(mu_cohort, v/kappa_0)`; it is what puts the `+1` into the
    /// sigma^2 denominator. Written out longhand rather than reusing any helper
    /// from the M-step, so a production change cannot silently follow this
    /// function's mistakes.
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
            // The Inverse-Gamma SCALE is `beta_0 * sigma2_cohort`, not `beta_0`.
            // Writing `beta_0` here makes the argmax come out wrong, which looks like a
            // production defect and is not one. It is the coefficient, not the exponent:
            // InvGamma(a, b) has density v^{-(a+1)} exp(-b/v), and `em.rs` puts
            // `beta_0 * sigma2_cohort` in the exponent so the prior's scale tracks the
            // cohort's own variance.
            - beta_0 * sigma2_cohort / v
            - 0.5 * v.ln()
            - kappa_0 * (mu - mu_cohort).powi(2) / (2.0 * v)
    }

    /// Golden-section maximisation of a unimodal 1-D function. This stalls at
    /// the f64 representation limit (~2.5e-8 absolute on a log-posterior of
    /// magnitude ~3), which is why the caller asserts on the OBJECTIVE VALUE
    /// rather than on the argmax's coordinates: the objective is smooth and
    /// well conditioned where the coordinates are not.
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
    ///
    /// It asserts two things, and the second is the one that carries the dual
    /// implementation:
    ///   (a) production is strictly below Model B's numeric argmax;
    ///   (b) production equals Model A's closed form, recomputed here from the
    ///       accumulators with no shared code.
    ///
    /// WHY A NEW FIXTURE. Three properties are needed at once and no existing
    /// helper provides them:
    /// 1. **Exact, iteration-independent accumulators.** A molecule with
    ///    `candidate_count == 1` is routed to `unambiguous_indices` and
    ///    accumulates `accum_w += 1.0` unconditionally -- the likelihood never
    ///    touches the weight. So (W, S1, S2) is exactly what the fixture says, on
    ///    every iteration, for ever.
    /// 2. **The M-step must actually execute.** `solve_and_write_profile` computes
    ///    `delta_avg_ll` over AMBIGUOUS molecules only, so an all-unambiguous graph
    ///    gives `n_ambig == 0`, `delta_avg_ll == 0.0` and breaks BEFORE the
    ///    M-step. Hence the decoy molecule: it contributes `n_ambig = 1` and
    ///    nothing else.
    /// 3. **The defect must be visible.** The missing term is
    ///    `kappa_0 (mu_MAP - mu_cohort)^2`, and shrinkage drives `mu_MAP` TOWARD
    ///    `mu_cohort` as W grows -- so a large-W, tightly-clustered fixture hides
    ///    it. This fixture puts the defect five orders of magnitude above the
    ///    tolerance.
    ///
    /// The assertion is on the OBJECTIVE VALUE, not on the argmax's coordinates:
    /// golden-section maximisation of a log-posterior of magnitude ~3 stalls at
    /// ~2.5e-8 absolute in f64, so coordinate assertions would need a tolerance
    /// four orders looser than the defect-detection margin.
    #[test]
    fn test_mstep_is_model_a_and_is_not_model_b() {
        // NU_0 / BETA_0 / TAU_0 are deliberately NOT restated here. This test is
        // named for production Model A, so it must read production's own constants --
        // a local transcription is a guarantee about the test, not about the crate.
        const MU_COHORT: f64 = 8.5; // deliberately NOT near the observations
        const SIGMA2_COHORT: f64 = 0.5;
        const FLOOR: f64 = crate::emission_model::INSTRUMENTAL_PAL_NOISE_FLOOR;

        // Single-candidate molecules -> unambiguous -> w = 1.0 each.
        let lens: [f32; 3] = [64.0, 100.0, 256.0];
        let mut mols: Vec<(f32, Vec<(&str, i32, u32)>)> = lens
            .iter()
            .map(|&l| (l.log2(), vec![("tx0", 1000, 0u32)]))
            .collect();
        // The decoy. Never asserted on; it exists only so n_ambig >= 1.
        mols.push((256.0f32.log2(), vec![("tx1", 1000, 0), ("tx2", 1000, 0)]));

        // The accumulators the production code will build, computed here from
        // the SAME f32 -> f64 path it uses (`mol.log_pal as f64`).
        let ys: Vec<f64> = lens.iter().map(|&l| l.log2() as f64).collect();
        let w: f64 = ys.len() as f64;
        let s1: f64 = ys.iter().sum();
        let s2: f64 = ys.iter().map(|&y| y * y).sum();

        // Feasibility, stated so a future fixture edit cannot silently make the
        // variance term negative (which would `.max(0.0)` it away).
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

        // GUARD, not decoration: if the M-step never ran, mu still equals
        // mu_cohort and sigma2 still equals sigma2_cohort. Say so, instead of
        // letting the objective assertion below report a mystery.
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

        // Neither engineering clamp may bind here, or this fixture stops
        // exercising the closed form and the comparison below stops meaning
        // what it says.
        assert!(
            v_prod > FLOOR * 10.0,
            "sigma2 = {v_prod} is near INSTRUMENTAL_PAL_NOISE_FLOOR ({FLOOR}); the \
             floor is clamping and this fixture no longer tests the closed form."
        );

        // Numeric argmax: golden-section on mu, with v optimised inside.
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

        // (a) THE SWITCH TOOK EFFECT. Production is no longer the joint-MAP optimum
        //     of Model B, and the gap is three orders of magnitude above any tolerance
        //     that could be argued about. This assertion is the one that fails if
        //     someone silently restores Model B.
        let q_prod = q(mu_prod, v_prod);
        let q_best = q(mu_star, v_star);
        assert!(
            q_prod < q_best - 1e-6,
            "production is still sitting at Model B's joint-MAP optimum.\n  \
             production (mu, sigma2) = ({mu_prod:.17}, {v_prod:.17})\n  \
             Model B argmax (mu, sigma2) = ({mu_star:.17}, {v_star:.17})\n  \
             Q_B(production) - Q_B(argmax) = {:.3e}, which is not the expected \
             NEGATIVE gap.\n  \
             Since 2026-10-12 the production M-step is Model A: numerator \
             without the kappa_0*(mu-mu_cohort)^2 term, denominator +2 not +3. \
             If this fires, Model B is back.",
            q_prod - q_best
        );

        // (b) WHAT PRODUCTION IS INSTEAD: Model A's closed form, computed here a
        //     SECOND time from the accumulators and compared.
        //
        //     The two derivations share no code: this block recomputes mu from
        //     (S1, W, tau_0, mu_cohort) and sigma2 from (S(mu), W, nu_0, beta_0,
        //     sigma2_cohort) by hand, while production ran the M-step over a synthetic
        //     graph.
        //
        //     NOTE ON WHAT MODEL A IS NOT. Model A is not the argmax of any joint
        //     objective written down here, and this test deliberately does not claim
        //     it is. Dropping the kappa_0 quadratic alone would leave the `-0.5 ln v`
        //     normaliser and give denominator +3, not +2 -- so Model A drops both.
        //     Whether that is the right prior is a MODELLING question, not a unit
        //     test. What this test pins is narrower and checkable: production computes
        //     Model A's closed form, and does not compute Model B's.
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
        // log2(120) ≈ 6.9069, exp2 = 120
        assert_eq!(poly_a_length_from_log2(6.9069), 120);
        assert_eq!(poly_a_length_from_log2(7.0), 128);
    }

    /// Unique scratch path under the system temp dir. Callers remove it.
    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("telescope_em_{}_{}", name, std::process::id()))
    }

    /// Same as `scratch` but on the LOCAL filesystem (`/tmp`) rather than
    /// `$TMPDIR`.
    ///
    /// A parquet written to `$TMPDIR` and then immediately opened for reading is
    /// subject to NFS client attribute caching: under enough concurrent I/O the
    /// reader can observe `Size is 0 but need 8` even though the writer closed the
    /// file successfully moments earlier. That is a real, observed failure of this
    /// suite.
    fn local_scratch(name: &str) -> PathBuf {
        PathBuf::from("/tmp").join(format!("telescope_em_{}_{}", name, std::process::id()))
    }

    /// Write a minimal `telescope.emission.parquet`: one poly(A)-competent row
    /// per `(transcript_id, gene_id, mu_logPAL, sigma2_logPAL)`.
    fn write_test_emission(path: &Path, rows: &[(&str, &str, f64, f64)]) {
        let all_competent: Vec<(&str, &str, f64, f64, bool)> = rows
            .iter()
            .map(|&(a, b, c, d)| (a, b, c, d, true))
            .collect();
        write_test_emission_with_competence(path, &all_competent);
    }

    /// As `write_test_emission`, but each row also carries its
    /// `is_polyA_competent` flag. Needed because the E-step pushes an
    /// INCOMPETENT candidate as `-inf` and the output stage's `retained_sum`
    /// denominator is documented as correct only while that keeps incompetent rows
    /// out of `weights`. Nothing tested that coupling, so a future change to the
    /// mask would silently break normalisation with every test still green.
    ///
    /// `n_cohort_flnc` is written as a NON-ZERO constant for every row, so every
    /// pre-existing test using this helper keeps its original meaning: no row is
    /// imputed. It is written at all, rather than omitted, because the loader now
    /// REQUIRES the column.
    fn write_test_emission_with_competence(path: &Path, rows: &[(&str, &str, f64, f64, bool)]) {
        let with_nobs: Vec<(&str, &str, f64, f64, bool, u32)> = rows
            .iter()
            .map(|&(tx, g, mu, s2, c)| (tx, g, mu, s2, c, 10))
            .collect();
        write_test_emission_with_nobs(path, &with_nobs)
    }

    /// Same as `write_test_emission_with_competence` but with an explicit
    /// `n_cohort_flnc` per row, so a test can mark a row IMPUTED with 0 -- the
    /// only in-schema signal that the row carries the cohort fit rather than
    /// one of its own.
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
    /// `log_pal` and one candidate per `(transcript_id, AS:i)` entry. Every
    /// candidate gets `gap5p = 0`; use `synth_graph_with_gaps` when the 5' term
    /// is under test.
    fn synth_graph(log_pal: f32, candidates: &[(&str, i32)]) -> FlatEvidenceGraph {
        let with_gaps: Vec<(&str, i32, u32)> =
            candidates.iter().map(|&(tx, as_)| (tx, as_, 0)).collect();
        synth_graph_with_gaps(log_pal, &with_gaps)
    }

    /// At `poly_a_weight == 0` the M-step must FREEZE mu and sigma2, not
    /// recompute them from a placeholder `y`.
    ///
    /// The failure it would have caused is specific, not cosmetic. With `accum_w`
    /// (W) still accumulating and S1 forced to 0:
    ///
    ///     mu     = (0 + tau_0 * mu_cohort) / (w_sum + tau_0)  ->  0
    ///     sigma2 -> INSTRUMENTAL_PAL_NOISE_FLOOR
    ///
    /// so every transcript would be dragged to the placeholder's value. The
    /// emission parquet is written with mu = 7.0, which is far from both 0 and
    /// the floor, so "unchanged" and "dragged" are trivially distinguishable.
    #[test]
    fn test_poly_a_weight_zero_freezes_mu_and_sigma2_instead_of_dragging_them() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("mstep_freeze_emission.parquet");
        let profile = scratch("mstep_freeze_profile.parquet");
        write_test_emission(
            &emission,
            &[("tx0", "gene0", 7.0, 0.5), ("tx1", "gene1", 7.0, 0.5)],
        );
        // Built twice rather than cloned: `FlatEvidenceGraph` is deliberately
        // not `Clone` (it is a set of parallel pools, and a Clone would invite
        // the two drifting apart).
        let cands: [(&str, i32, u32); 2] = [("tx0", 1000, 0), ("tx1", 1000, 0)];

        // Arm A: poly(A) ON. mu must move off its initial 7.0 -- this is the
        // control, and without it the test below would pass for the wrong
        // reason (e.g. a solver that never touches mu at all).
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

        // Arm B: poly(A) OFF. mu and sigma2 must be exactly what the parquet
        // said -- the placeholder `y` must not reach the accumulators at all.
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

        // alpha is the abundance prior and does NOT read poly(A), so it must
        // still be updated. If the bypass had swallowed STEP 3 as well, the
        // prior would silently vanish and every candidate would be scored
        // without it.
        let alpha_off = params_off[0].as_ref().unwrap().alpha;
        assert!(
            alpha_off > 0.0,
            "STEP 3 (alpha) must still run at poly_a_weight = 0: it is the \
             abundance prior, not a poly(A) term"
        );

        // The same argument applies to W, the quantity alpha is built from, and this
        // arm is the ONLY place it can be seen. Every sidecar test runs at
        // `poly_a_weight > 0`, where the guard is unconditionally true, so a W write
        // moved inside that guard would pass all of them.
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
    /// form, not a numeric estimate, so any tolerance wider than that is buying
    /// tolerance for a bug.
    ///
    /// `max_iter = 2` AND NOT MORE: `output_emission_params` is fed from
    /// `last_estep_params`, snapshotted after the E-step consumed `params` and
    /// before the M-step rewrites it, so the exported value is one M-step BEHIND
    /// `params`. `max_iter = 2` makes the export M-step#1's output on EVERY exit
    /// path; a larger value would mean re-deriving the whole iteration chain, and
    /// the loop exits on `delta_avg_ll < tol`, so a fixed-point argument holds
    /// only to ~1e-6 and can never support a 1e-12 pin.
    ///
    /// WHY THE EXPECTED NUMBERS LOOK ROUNDED AND ARE NOT. `y` is NOT log2(100):
    /// the fixture computes it in f32 and the solver widens it, so
    ///     y = f64(f32(log2(100))) = 6.6438560485839844
    /// which sits 1.41e-07 BELOW the true logarithm; asserting the f64-path
    /// decimal would fail, and loosening to 1e-6 would destroy the exact
    /// sensitivity this test exists to provide.
    ///
    ///     mu      = (S1 + tau_0*mu_cohort) / (w_sum + tau_0) = 6.928771209716797
    ///     sq_dev  = max(S2 - 2*mu*S1 + mu^2*W, 0)             = 0.040588324521667830
    ///     sigma2  = (sq_dev + 2*beta_0*sigma2_cohort) / (w_sum + 2*nu_0 + 2)
    ///             = (0.040588324521667830 + 1.0) / 4.5        = 0.23124184989370397
    ///
    /// `sq_dev` is a cancelling sum -- terms of order 22 and 48 collapsing to
    /// 0.0406 -- and it still carries ~1e-15 relative accuracy, 4 orders inside
    /// the 1e-12 budget. The result is far above INSTRUMENTAL_PAL_NOISE_FLOOR, so
    /// the `.max()` clamp must NOT fire.
    ///
    /// The shrinkage prior `mu_cohort` is loaded once, from the emission
    /// parquet's `mu_logPAL`, and is the fixed point the M-step pulls `mu` TOWARD:
    /// `p.mu = (accum_w_y[i] + tau_0 * p.mu_cohort) / (w_sum + tau_0)`. If the
    /// M-step also WROTE `p.mu_cohort`, the prior would chase the current iterate
    /// and the shrinkage would collapse toward a fixed point with no data in it.
    ///
    /// The second assertion is what stops this from passing vacuously: it
    /// requires `mu` to have actually MOVED off the prior.
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

            // VACUITY GUARD: the prior must be a real shrinkage target, i.e.
            // the data must have moved `mu` away from it. Equal values would
            // make the immutability assertion below true for free.
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
        //   The fixture is one molecule with two SYMMETRIC candidates, so the
        //   posterior is exactly (0.5, 0.5) and each transcript sees
        //       W = 0.5,  S1 = 0.5*y,  S2 = 0.5*y^2,  y = (100f32).log2() as f64
        //                                   = 6.6438560485839844
        //   => S1 = 3.3219280242919922, S2 = 22.070411597152997
        //
        //   mu = (S1 + tau_0*mu_cohort)/(W + tau_0) = 6.9287712097167971
        //   S(mu) = S2 - 2*mu*S1 + mu^2*W          = 0.040588324521667829
        //
        //   MODEL A: no kappa_0*(mu - mu_cohort)^2 term, and the denominator loses
        //   the `+1` that term's normaliser contributed:
        //   numerator   = S(mu) + 2*beta_0*sigma2_cohort = 1.040588324521667829
        //   denominator = W + 2*nu_0 + 2                  = 4.5
        //   sigma2 = 1.040588324521667829 / 4.5            = 0.23124184989370397
        //
        // READ THIS NUMBER OFF A RUN, NOT OFF PAPER. It was produced by the
        // f32-widened `log2` path this crate actually uses, and recomputing it in f64
        // from the printed operands gives 0.23124184274217108 -- agreement to eight
        // digits, not to seventeen. The f32 widening is why these constants are
        // pinned here at all.
        const EXPECTED_SIGMA2: f64 = 0.231_241_849_893_703_97;
        const TOL: f64 = 1e-12;

        // The observation, and the cohort prior it must be shrunk TOWARD.
        // mu_cohort != y is what gives the `tau_0 * mu_cohort` term non-zero
        // leverage: with mu_cohort == y the closed form collapses to y for any
        // prior weight, and deleting `tau_0` entirely would go unnoticed. Here
        // they differ by 0.285, which is 2.8e11 tolerances.
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

            // The shrinkage is directional: mu must sit strictly BETWEEN the observation
            // and the prior, equal to neither. Written against min/max rather than as
            // `mu > y && mu < prior` so the assertion encodes the invariant instead of a
            // guess about which endpoint is larger.
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

    // ── σ² REGULARISATION DEGREES-OF-FREEDOM OFFSET `c` (`SIGMA2_DOF_OFFSET`) ──

    /// The observation and the cohort prior of the closed-form fixture,
    /// repeated here so these tests do not depend on a `const` declared inside
    /// another test's body.
    const DOF_OFFSET_MU_COHORT: f64 = 7.0;
    /// `sigma2_cohort` the fixture is built with.
    const DOF_OFFSET_SIGMA2_COHORT: f64 = 0.5;
    /// Posterior weight each of the two symmetric candidates receives. It is
    /// exactly 0.5 because the fixture has one molecule and two candidates with
    /// EQUAL `AS` and EQUAL `g_k` — the E-step cannot break the tie, at any `c`.
    const DOF_OFFSET_W: f64 = 0.5;

    /// The closed form, recomputed INDEPENDENTLY from the fixture's documented
    /// operands rather than read off a run.
    ///
    /// `y = (100.0_f32).log2() as f64` = 6.6438560485839844, posterior exactly
    /// (0.5, 0.5), so `W = 0.5`, `S1 = 0.5*y`, `S2 = 0.5*y*y`. `S1`/`S2` are
    /// spelled as multiplies rather than `powf` because that is the reassociation
    /// the E-step's accumulation performs; reassociating them moves the last ULP
    /// and bit-comparability is this function's whole purpose.
    fn dof_offset_closed_form(c: f64) -> f64 {
        let y = (100.0_f32).log2() as f64;
        let mu = (DOF_OFFSET_W * y + TAU_0 * DOF_OFFSET_MU_COHORT) / (DOF_OFFSET_W + TAU_0);
        let s1 = DOF_OFFSET_W * y;
        let s2 = DOF_OFFSET_W * y * y;
        let sq_dev_new = (s2 - 2.0 * mu * s1 + mu * mu * DOF_OFFSET_W).max(0.0);
        let numerator = sq_dev_new + 2.0 * BETA_0 * DOF_OFFSET_SIGMA2_COHORT;
        numerator / (DOF_OFFSET_W + 2.0 * NU_0 + c)
    }

    /// `(stats, emission_params, em_trace, read_assignments)` from one run of the
    /// closed-form fixture.
    ///
    /// `explicit_c == None` takes the SHIPPED DEFAULT; `Some(c)` takes
    /// `solve_and_write_profile_with_dof_offset`. Every observable is requested,
    /// because the question this exists to answer is "did anything at all move",
    /// and an out-param that was not requested cannot answer it.
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

    /// Write the closed-form fixture's emission parquet and hand back its path.
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

    /// The default must be the shipped Model A estimator, and this says it the
    /// only way it can be checked without re-deriving anything: run the SAME
    /// fixture through the parameterised entry point with `c = 2` explicitly, and
    /// through the default entry point, and require the two to agree to the LAST
    /// BIT on every observable.
    ///
    /// EXACT `to_bits()` EQUALITY, NO TOLERANCE -- a `< 1e-15` bound here would
    /// still pass for every `c` within an ULP of 2, which is precisely the drift
    /// this parameter exists to make impossible.
    ///
    /// The comparison is deliberately WIDER than σ² -- `mu`, `alpha`,
    /// `em_accum_weight`, the ids, the flags, the per-iteration EM trace, the
    /// per-read assignment, and the whole `TelEmStats` -- because a `c` that
    /// reached anything but the denominator would show up in one of those first,
    /// and a σ²-only test would report it as a σ² regression.
    #[test]
    fn test_default_sigma2_dof_offset_is_bit_identical_to_the_named_parameter() {
        let emission = dof_offset_emission("default");
        let (stats_a, params_a, trace_a, reads_a) = dof_offset_run(&emission, "default", None);
        let (stats_b, params_b, trace_b, reads_b) =
            dof_offset_run(&emission, "default_explicit", Some(2.0));
        let _ = std::fs::remove_file(&emission);

        // VACUITY GUARDS. Two arms of a comparison agree for free if both are
        // empty, and every out-param here is a `Vec` that starts empty. If an
        // arm produced nothing, every loop below would pass without executing
        // a single comparison.
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

        // `TelEmStats` derives only `Debug`, and every one of its fields is an
        // integer, the `CandidateLedger`, or the internally-recorded trace -- so the
        // rendered form is a TOTAL, exact comparison of the whole struct, and it is
        // total BY NAME.
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
    /// BEFORE `sigma2_dof_offset` became a parameter. They exist so that "the
    /// refactor is bit-identical at the default" is a claim with a receipt behind
    /// it rather than an argument from reading the diff.
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

        // NO TOLERANCE. If this goes red the answer is the ULP distance between the
        // two patterns, REPORTED -- never a widened bound. A tolerance here would let
        // the exact failure this test exists to prevent (a parameterised `2.0` that
        // quietly computes something adjacent to it) pass.
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
    ///   computed in the test from the fixture's documented operands. Comparing
    ///   only against sigma2(2) being smaller would be satisfied by any
    ///   monotone-in-`c` implementation, including one that ignores `c` entirely.
    /// * `c = 3` is NOT Model B, and this asserts that with a number rather than
    ///   with prose: Model B also restores `tau_0 * (mu - mu_cohort)^2` in the
    ///   NUMERATOR, while the `c` knob moves the denominator only. A later reader
    ///   who describes a `c = 3` arm as "Model B" is contradicting this
    ///   assertion.
    ///
    /// FLOOR HEADROOM. The smallest value in the [1, 5] sweep is 0.13874 at
    /// `c = 5`, clear of clamping, but NOT the 10x margin the joint-MAP test
    /// demands -- so this test does not demand 10x and would be wrong to.
    #[test]
    fn test_sigma2_dof_offset_three_is_the_third_estimator_not_model_b() {
        let emission = dof_offset_emission("c3");
        let (_s3, params3, _t3, _r3) = dof_offset_run(&emission, "c3", Some(3.0));
        let (_s2, params2, _t2, _r2) = dof_offset_run(&emission, "c2", Some(2.0));
        let _ = std::fs::remove_file(&emission);
        let s3 = params3[0].as_ref().expect("tx0 at c=3").sigma2;
        let s2 = params2[0].as_ref().expect("tx0 at c=2").sigma2;

        // Bitwise, against the independently derived closed form.
        assert_eq!(
            s3.to_bits(),
            dof_offset_closed_form(3.0).to_bits(),
            "sigma2 at c = 3 must equal the closed form \
             (S(mu) + 2*beta_0*sigma2_cohort) / (W + 2*nu_0 + 3) BIT FOR BIT; \
             got {s3:.17}, closed form {:.17}",
            dof_offset_closed_form(3.0)
        );
        // Pin the value itself, so a change to the fixture or to the closed
        // form cannot quietly walk BOTH sides of the assertion above at once.
        assert_eq!(
            s3.to_bits(),
            0x3fc8_37a2_d37e_235du64,
            "sigma2 at c = 3 moved: got {s3:.17} ({:#018x}), expected 0.18919787718575778",
            s3.to_bits()
        );

        // `c = 3` is NOT Model B. Recomputed here rather than quoted, so the
        // distinction survives an edit to the constant or to a prior.
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
            "the Model B reference moved: got {model_b:.17}. Re-derive it before \
             trusting the comparison below"
        );
        assert_ne!(
            s3.to_bits(),
            model_b.to_bits(),
            "c = 3 has become Model B. The knob moves the DENOMINATOR only; Model B \
             also needs tau_0*(mu - mu_cohort)^2 back in the numerator, and the two \
             land on different values ({s3} vs {model_b}). Either the numerator was \
             parameterised too, or this arm is no longer the estimator it claims."
        );

        // Directional, and the whole reason the knob exists.
        assert!(
            s3 < s2,
            "sigma2 must SHRINK as c grows: sigma2(3) = {s3} is not below \
             sigma2(2) = {s2}"
        );

        // Floor headroom, asserted rather than assumed (see the doc comment).
        assert!(
            s3 > INSTRUMENTAL_PAL_NOISE_FLOOR * 2.0,
            "sigma2 at c = 3 is {s3}, within 2x INSTRUMENTAL_PAL_NOISE_FLOOR \
             ({INSTRUMENTAL_PAL_NOISE_FLOOR}); the floor would be clamping and this \
             arm would be measuring the floor rather than the closed form"
        );
    }

    /// σ² is strictly decreasing in `c` over [1, 5], and every point equals the
    /// closed form bit for bit.
    ///
    /// A NON-MONOTONE SWEEP IS A BUG, NOT A RESULT. The arithmetic says
    /// `sigma2 = N / (W + 2*nu_0 + c)` with `N` positive and `c`-independent,
    /// hence strictly decreasing; any other shape means `c` reached something it
    /// should not have -- the E-step's weights, `mu`, or the floor. `N` is
    /// `c`-independent HERE because the two candidates are symmetric, so the
    /// posterior is pinned at (0.5, 0.5) for every `c`.
    #[test]
    fn test_sigma2_is_strictly_decreasing_over_the_c_sweep() {
        let sweep = [1.0f64, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let emission = dof_offset_emission("sweep");
        let mut measured = Vec::new();
        for (i, c) in sweep.iter().enumerate() {
            let (_stats, params, _trace, _reads) =
                dof_offset_run(&emission, &format!("sweep{i}"), Some(*c));
            let s = params[0].as_ref().expect("tx0").sigma2;
            // EVERY point against the closed form, not just the endpoints.
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

    /// A transcript that wins NO posterior mass still gets an M-step, and the σ²
    /// it lands on is `2*beta_0*sigma2_cohort / (2*nu_0 + c)` -- NOT the prior.
    ///
    /// The M-step loop iterates EVERY graph transcript, so `w_sum == 0` is a state
    /// the estimator genuinely visits rather than a hypothetical.
    ///
    /// WHAT THE FACTOR IS, and what it is not. The ratio to the prior is
    /// `2 / (2*nu_0 + c)`, so it TRACKS `c` -- 0.500x at the production value
    /// c = 2. "With no evidence σ² is half the prior" is a statement about that
    /// production value, NOT a hyperparameter-independent property.
    ///
    /// `c = 0` -- the value that would make the ratio exactly 1.000 and reduce the
    /// M-step to a no-op at W = 0 -- is UNREACHABLE through the parameterised entry
    /// point, whose guard is `is_finite() && > 0.0`. The shrinkage cannot be
    /// switched off.
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
                // INCOMPETENT: identical parameters, so the ONLY reason this
                // transcript wins no mass is the competence mask. Move the flag
                // off tx2 and the W = 0 state disappears -- which is what the
                // `em_accum_weight` assertion below would catch.
                (
                    "tx2",
                    "gene2",
                    DOF_OFFSET_MU_COHORT,
                    DOF_OFFSET_SIGMA2_COHORT,
                    false,
                ),
            ],
        );

        // c = 0 is excluded on purpose: the entry point rejects it
        // (`is_finite() && > 0.0`). 5.0 is the top of Fig 3C's sweep.
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
                // 2, not 1: with max_iter = 1 the loop runs the E-step and exits
                // BEFORE the M-step, so sigma2 and mu are still their INITIAL
                // values and every assertion below would pass while measuring
                // nothing. The other dof-offset fixtures use 2 for this reason.
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

            // Locate by NAME, never by index: `params` is ordered by the graph's
            // transcript list, and an index would silently assert the wrong row.
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

            // VACUITY GUARD. Everything below is about W = 0, so assert it
            // rather than assuming it: if the competence mask stopped firing,
            // tx2 would win half the posterior and this whole test would be
            // measuring the ordinary W > 0 path while appearing to pass.
            assert_eq!(
                w0.em_accum_weight, 0.0,
                "tx2 is pushed to -inf, so its M-step W must be exactly 0.0; \
                 got {}. If this is non-zero the fixture no longer produces a \
                 W = 0 transcript and the sigma2 assertions below are measuring \
                 something else.",
                w0.em_accum_weight
            );
            // CONTROL: the other two DO win mass, so W = 0 is specific to tx2
            // and not a property of this fixture as a whole.
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

            // THE ASYMMETRY. At W = 0 the tau_0 momentum term cancels exactly and mu
            // returns to the prior, while sigma2 does not. The mu half is what makes it an
            // asymmetry rather than a uniform "no evidence means no update".
            assert_eq!(
                w0.mu.to_bits(),
                DOF_OFFSET_MU_COHORT.to_bits(),
                "at W = 0 mu must land EXACTLY on the cohort prior, not near it; \
                 got {:.17}",
                w0.mu
            );

            // Not the floor. Below this the closed form would be unreachable and
            // the bit-exact assertion above would be measuring the clamp.
            assert!(
                w0.sigma2 > INSTRUMENTAL_PAL_NOISE_FLOOR,
                "sigma2 at c = {c} is {}, at or below INSTRUMENTAL_PAL_NOISE_FLOOR \
                 ({INSTRUMENTAL_PAL_NOISE_FLOOR}); the floor is clamping and the \
                 bit-exact comparison is vacuous",
                w0.sigma2
            );

            // The ratio IS the mechanism, stated as a measurement rather than
            // as prose. `sq_dev_new` is 0 at W = 0, so the numerator is the
            // prior term alone and the ratio must be exactly 2/(2*nu_0 + c).
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

        // Pin the production number, which is the one that reached a report:
        // 0.5x at c = 2, from `sigma2_cohort = 0.5`.
        let prod = measured
            .iter()
            .find(|(c, _)| *c == 2.0)
            .expect("the sweep contains the production c = 2");
        assert_eq!(
            prod.1.to_bits(),
            0.5f64.to_bits(),
            "at the production c = 2 the W = 0 sigma2 is exactly HALF the prior; \
             got {:.17}. If this moved, R99 §3.4's table is stale.",
            prod.1
        );
        // Strictly decreasing in c: the factor has no free parameter to break
        // monotonicity, so a flat or rising step means `c` reached the floor or
        // the E-step. Same reading rule as the sweep test above.
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

    /// The boundary guard, and why it has to exist: `f64::max` ABSORBS the
    /// failure. Measured: `f64::max(f64::NAN, 0.04) == 0.04` and
    /// `f64::max(-1.04, 0.04) == 0.04`. So a `c` of `-2.5`, `0.0` or `NaN` makes
    /// `W + 2*nu_0 + c` zero, negative or NaN, and the M-step's `.max(
    /// INSTRUMENTAL_PAL_NOISE_FLOOR)` hands back a perfectly plausible `0.04` with
    /// no error anywhere.
    #[test]
    #[should_panic(expected = "sigma2_dof_offset must be finite and positive")]
    fn test_sigma2_dof_offset_guard_rejects_the_exact_denominator_zero() {
        let emission = dof_offset_emission("guard");
        let _ = dof_offset_run(&emission, "guard", Some(-2.5));
    }

    /// The guard must refuse `NaN` too, and `NaN` is the case every `> 0.0`
    /// comparison silently passes: `NaN > 0.0` is false, `NaN <= 0.0` is false,
    /// and `NaN.is_nan()` is the only test that sees it.
    #[test]
    #[should_panic(expected = "sigma2_dof_offset must be finite and positive")]
    fn test_sigma2_dof_offset_guard_rejects_nan() {
        let emission = dof_offset_emission("guard_nan");
        let _ = dof_offset_run(&emission, "guard_nan", Some(f64::NAN));
    }

    /// And `+inf`, which passes every finiteness test written as a comparison
    /// but drives the quotient to zero rather than to NaN — a silent collapse to
    /// the floor all the same.
    #[test]
    #[should_panic(expected = "sigma2_dof_offset must be finite and positive")]
    fn test_sigma2_dof_offset_guard_rejects_infinity() {
        let emission = dof_offset_emission("guard_inf");
        let _ = dof_offset_run(&emission, "guard_inf", Some(f64::INFINITY));
    }

    /// Re-read a sidecar written by `write_emission_params_parquet`: the column
    /// NAMES in file order, plus one entry per emitted row holding the transcript
    /// id and every OTHER column's value looked up BY NAME.
    ///
    /// A column that is absent, or present but not Float64, reports `None` for
    /// every row rather than panicking: the point of a schema assertion is that
    /// the schema is wrong, and a helper that panicked would report it as a panic
    /// instead of a failure naming the column.
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

    /// One Float64 cell of a `read_sidecar` row, by column name.
    fn sidecar_cell(row: &(String, Vec<(String, Option<f64>)>), col: &str) -> Option<f64> {
        row.1.iter().find(|(n, _)| n == col).and_then(|(_, v)| *v)
    }

    /// The `em_finalized` column as `(transcript_id, value)` pairs.
    ///
    /// `read_sidecar` keeps only Float64 cells, so a Boolean column reads back as
    /// `None` there by design. This reopens the file instead. It also *panics* on
    /// a missing or non-Boolean column rather than reporting `None`: a schema
    /// mistake is exactly what the caller is asserting about.
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

    /// THE `em_accum_weight` COLUMN IS THE E-STEP'S W, IN MOLECULES, NOT A SHARE.
    ///
    /// WHY THE FIXTURE HAS A SECOND MOLECULE. The obvious fixture is the
    /// symmetric single-molecule one, and it is VACUOUS here: with one molecule in
    /// the graph, `alpha` is W divided by 1, so W and the written alpha are the
    /// same float and a column carrying the share instead of the sum cannot be
    /// told apart. Two molecules make the two columns differ by exactly the factor
    /// under test -- alpha 0.25 against W 0.5.
    ///
    /// WHY W IS EXACTLY 0.5, derivable rather than asserted:
    ///
    ///   molecule r0 has two candidates, both present in the emission parquet,
    ///   with identical (mu_cohort, sigma2_cohort), identical alignment evidence
    ///   and identical 5' gaps. The Laplace init (eps = 0.01) gives each of them
    ///   0.01 + 1/2 = 0.51 and the solo transcript 0.01 + 1, so `alpha` comes out
    ///   identical for the pair; equal alpha, equal mu, equal sigma2 and equal
    ///   every other term make the two log-likelihoods bit-identical, hence
    ///   `log_z = ll + ln 2` and each posterior `exp(ll - log_z) = exp(-ln 2)`,
    ///   which is 0.5 exactly in f64.
    ///
    ///   W for each of the pair is therefore 0.5, and for the solo molecule it is
    ///   1.0 (an unambiguous read contributes exactly its weight of 1).
    ///
    /// `max_iter = 2` for the reason the closed-form test uses it: the M-step must
    /// have run for `alpha` to be a fit rather than the Laplace initialisation.
    #[test]
    fn test_sidecar_em_accum_weight_is_the_estep_weight_sum_in_molecules() {
        const TOL: f64 = 1e-12;
        const EXPECTED_W: f64 = 0.5;
        // `alpha` = W / n_total_molecules with n_total_molecules = 2. Asserted
        // so that a column carrying the share instead of the sum cannot pass
        // the W assertion: it would report 0.25 where 0.5 is required.
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

        // Two iterations with a contested molecule means an M-step ran, so the
        // row is a fit and the W/alpha identity is allowed to hold. Asserted
        // here rather than assumed: if `em_finalized` ever reports false for
        // this fixture the EXPECTED_ALPHA assertions below would be measuring
        // the Laplace initialisation while claiming to measure the fit.
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

            // The headline assertion comes FIRST so that a column carrying the
            // abundance share fails on the value itself rather than on the
            // guard below explaining why the value would have been vacuous.
            assert!(
                (w - EXPECTED_W).abs() < TOL,
                "{tx}: em_accum_weight REGRESSION: got {w:.17}, expected \
                 {EXPECTED_W}. em_accum_weight is the E-step weight sum, in \
                 MOLECULE weight units -- not alpha, not the per-read posterior \
                 the profile parquet stores. Derivation is on this test."
            );
            // VACUITY GUARD: W and alpha must be distinguishable numbers on this
            // fixture, or the assertion above is satisfied by the share as well.
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

    /// THE WRITER EMITS THE NEW COLUMNS, AND DROPS NOTHING ELSE.
    ///
    /// The three new columns are appended rather than inserted so existing
    /// name-selecting readers keep working, the `None`-row omission is a
    /// documented contract of this function and not of the solver, and both are
    /// properties of the WRITER that no end-to-end solve can distinguish from the
    /// solver's own behaviour. Driven directly with a hand-built vector so the
    /// expected numbers are the ones written, not re-derived by the code under
    /// test.
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
            // A transcript with no emission row. It must produce NO row here:
            // writing a placeholder would invent a parameter for a transcript
            // the model never scored.
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
        // Back-compat: the five columns `load_emission_parameters` reads, and
        // the two diagnostic columns already in the file, are still there.
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

        // Written values, checked by column name against the literal above. The
        // numbers are pairwise distinct on purpose: a writer that mis-ordered
        // its builders, or aliased the wrong field, cannot pass.
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
                // The frozen first five: `mu_fitted` is an ALIAS, not a
                // replacement, so both names must carry the same fit.
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

        // `em_finalized` is Boolean, so `read_sidecar` cannot see it at all (it keeps
        // only Float64 cells) and asserting it here is the only place the writer's
        // Boolean branch is exercised.
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

    /// THE SIDECAR IS NOT A RE-LOADABLE MODEL.
    ///
    /// The gap is one column. The writer emits five of the six columns
    /// `load_emission_parameters` requires, but never `n_cohort_flnc`. The loader
    /// reads that column as REQUIRED on purpose, which means the round trip dies
    /// there instead of producing a plausible number.
    ///
    /// This test asserts the FAILURE, which is the whole point: the defect was a
    /// *docstring* claiming a capability, and a docstring cannot be made to go
    /// red. If someone ever adds `n_cohort_flnc` to the sidecar this test goes
    /// red, and the message tells them that the prior-reanchoring hazard has to be
    /// re-decided too -- because a successful load would silently re-anchor both
    /// shrinkage priors to the already-fitted values and apply τ₀ and 2β₀ a
    /// second time.
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

    /// WEAK BY CONSTRUCTION, and labelled so. `alpha` is DEFINED as W divided by
    /// the total molecule count, so this identity holds for ANY
    /// proportionally-wrong W -- including W scaled by 1/1000. It is a consistency
    /// check between two columns, NOT a proof that either carries the right value.
    ///
    /// What this one does add is the reading every downstream consumer will
    /// actually perform, and one thing the value test structurally cannot see: it
    /// is written down against a fixture with MORE THAN ONE ambiguous molecule, so
    /// `alpha` and W are not both derived from a single read.
    ///
    /// It cannot catch W landing inside the poly(A) guard: this fixture runs with
    /// the weight ON, where that guard is unconditionally true.
    ///
    /// `max_iter` must be >= 2 for the same snapshot-lag reason throughout: with
    /// one iteration the file carries the Laplace initialisation (alpha summing
    /// to 1, W summing to 0) and this identity is not merely weak but FALSE.
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
        // `n_total_molecules` is every row of the input graph, candidates and
        // all: 3 molecules here, and the graph's own count is not recoverable
        // from the sidecar -- which is half of why W is worth writing down.
        let n_total_molecules = 3.0_f64;

        // Gated on `em_finalized`, which is the whole reason that column
        // exists: the identity is a claim about a row whose `alpha` is an
        // M-step's output. An un-finalised row carries the Laplace
        // initialisation for `alpha` and MUST disagree, so asserting the
        // identity without this guard would be asserting the flag is true.
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
        // ... and the sides must be non-degenerate, or the assertion above is
        // satisfied by a file whose W never ran.
        assert!(
            sum_w > 1.0 && (sum_alpha - 1.0).abs() < TOL,
            "non-degenerate guard: sum(em_accum_weight) = {sum_w}, sum(alpha) = {sum_alpha}"
        );

        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);
        let _ = std::fs::remove_file(&sidecar);
    }

    /// THE FIXTURE THAT MAKES THE C2 STAMPING NON-VACUOUS.
    ///
    /// The identity test above pins `sum(em_accum_weight) == n_total_molecules *
    /// sum(alpha)` on a fixture that is ALREADY AT A FIXED POINT after one
    /// iteration. **At a fixed point the stale value equals the current value**,
    /// so the identity above passes whether `alpha` is stamped from the final
    /// E-step or from the previous one. A test that cannot tell those two apart
    /// is not testing the claim it is written to support.
    ///
    /// So this fixture is built to MOVE. r0's two candidates carry mu 5.0 and 8.0,
    /// straddling the read's own log2(PAL), so neither is the better explanation
    /// in advance: every M-step refit moves the E-step posteriors, and W changes
    /// from one iteration to the next.
    ///
    /// WHICH RUNS, AND WHY 1 IS IN THE LIST. The stamping block runs on every
    /// exit path, so `max_iter = 1` stamps from the FIRST E-step and
    /// `max_iter = 2` stamps from the SECOND -- exactly the pair the staleness
    /// mutation disagrees about.
    ///
    /// The first assertion is the load-bearing one, and it is a property of the
    /// FIXTURE rather than of the stamping: it proves W really moves here.
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
                    // Contested, and AS-asymmetric so the split is not a tie. The gap MUST stay
                    // under PRUNE_MAX_DELTA_AS = 50: a wider gap prunes tx1 outright, which
                    // silently demotes a contested molecule to an unambiguous one whose posterior
                    // is pinned at 1.0, so its W could not move no matter how the fit changed.
                    vec![("tx0", 900, 0), ("tx1", 870, 0)],
                    // Unambiguous: contributes exactly 1.0 to W, which is what
                    // keeps the fixture non-degenerate in the unambiguous arm.
                    vec![("solo", 1000, 0)],
                    // Contested and AS-tied: decided only by the fitted mu.
                    vec![("a", 1000, 0), ("b", 1000, 0)],
                ],
            )
        };

        // `(transcript_id, em_accum_weight, alpha, em_finalized)` read back from the
        // out-param. That vector IS `last_estep_params` -- the same snapshot the
        // sidecar parquet is written from -- so this compares the shipped values
        // rather than a re-derivation that could agree with a broken solver.
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

        // (1) NON-VACUITY. Compared on the RAW W column, never on alpha: alpha
        // is W divided by a constant, so it moves whenever W does and comparing
        // it would hide a fixture that held W still. This is the assertion that
        // makes every assertion below worth making.
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

        // (2) THE IDENTITY, at every stopping point. `alpha` is DEFINED as W
        // over the molecule count, so this asserts the two columns describe ONE
        // quantity rather than two -- which is the whole point of stamping them
        // together.
        for (n, run) in RUNS.iter().zip(&runs) {
            let sum_w: f64 = run.iter().map(|(_, w, _, _)| *w).sum();
            let sum_alpha: f64 = run.iter().map(|(_, _, a, _)| *a).sum();
            assert!(
                (sum_w - N_MOLECULES * sum_alpha).abs() < TOL,
                "max_iter={n}: sum(em_accum_weight) = {sum_w} but \
                 n_total_molecules * sum(alpha) = {}",
                N_MOLECULES * sum_alpha
            );
            // Per transcript as well as on the sum: two columns that disagree
            // in opposite directions on different rows still sum to agreement.
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

        // (3) `em_finalized` turns on exactly when an M-step ran, which is why
        // max_iter = 1 is excluded from this assertion even though the stamping is
        // not: asserting it across all runs would be asserting two things at once.
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

    /// CONSERVATION, WHICH THE IDENTITY ABOVE STRUCTURALLY CANNOT SEE.
    ///
    /// A molecule all of whose candidates are missing from the emission parquet
    /// contributes NOTHING to W -- it has no parameter to be scored against --
    /// while still counting in `n_total_molecules`. So `sum(alpha)` falls below 1
    /// and `sum(em_accum_weight)` falls below the molecule count, and neither
    /// quantity is wrong. This is asserted here because the conservation is now
    /// visible in a FILE, and a reader of that file will check it.
    #[test]
    fn test_sidecar_em_accum_weight_drops_molecules_whose_candidates_are_all_missing() {
        let emission = scratch("accum_w_gap_emission.parquet");
        let profile = scratch("accum_w_gap_profile.parquet");
        let sidecar = scratch("accum_w_gap_sidecar.parquet");

        // r0 is contested and both its candidates are present, so the loop
        // runs an M-step and the sidecar carries a real W. r1's only candidate
        // is absent from the emission parquet: the molecule is counted, scored
        // against nothing, and its weight vanishes.
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
        // The molecule is not silently dropped from the count; it is dropped
        // from the WEIGHT. Were it dropped from both, this file would look
        // normal to a reader comparing against the BAM.
        assert!(
            n_total_molecules > sum_w,
            "the missing-candidate molecule still counts as a molecule"
        );

        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);
        let _ = std::fs::remove_file(&sidecar);
    }

    /// `em_finalized` HAS TO BE FALSE ON SOME RUNS, AND THE RULE FOR WHICH RUNS IS
    /// NOT `n_em_iter >= 2`.
    ///
    /// The convergence `break` fires BEFORE the M-step. With `n_ambig == 0` the
    /// per-molecule delta is defined as 0.0, so `0.0 < tol` on the very first
    /// iteration and the loop breaks having run zero M-steps -- an all-unambiguous
    /// BAM. Those rows carry the Laplace initialisation, and without this column a
    /// reader cannot tell them from a fit.
    ///
    /// `max_iter == 1` is the arm that pins the rule. One E-step runs, `prev_ll`
    /// is `-inf` so the delta is `inf` and no convergence is claimed, the M-step
    /// runs, the loop ends -- and `n_em_iter` is 1. So `finalized = n_em_iter >= 2`
    /// calls a completed M-step an initialisation.
    ///
    /// The two arms are also each other's falsifier: a flag hard-wired to `true`
    /// fails arm A, `false` fails arm B.
    ///
    /// Arm A additionally pins that W is NOT gated on the flag: an unambiguous
    /// molecule's weight is a real 1.0 whether or not an M-step ran.
    #[test]
    fn test_em_finalized_is_false_when_the_run_never_reaches_an_m_step() {
        const TOL: f64 = 1e-12;

        // ---- ARM A: all-unambiguous, converges on iteration 0 ----------------
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

        // ---- ARM B: contested, max_iter == 1, M-step DID run ------------------
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

    /// THE COLUMN DESCRIBES THE **FINAL** ITERATION, NOT THE ONE BEFORE IT.
    ///
    /// This is the only test that can see the staleness the post-loop stamping
    /// block fixes, and it needs an ASYMMETRIC fixture to do it: the two-molecule
    /// fixture the value test uses is symmetric by construction, so W never moves
    /// and a column carrying the PREVIOUS iteration's W is indistinguishable from
    /// one carrying the right one.
    ///
    /// THE CROSS-CHECK, AND WHY IT IS NOT CIRCULAR. The sidecar's W is compared
    /// against the PROFILE parquet's own weights, aggregated per transcript. Two
    /// facts make that a real second data flow rather than a restatement:
    ///   - the profile's `posterior_weight` is `w / retained_sum`, so it equals
    ///     the raw E-step weight ONLY when nothing was truncated. Asserted, not
    ///     assumed: the fixture has exactly two candidates and the test requires
    ///     exactly two profile rows, so both survive.
    ///   - the profile is written from `final_weights`, the same vector the
    ///     recompute consumes, but by a different function, a different file and a
    ///     different builder.
    ///
    /// THE VACUITY GUARD. A fixture whose EM has already converged by iteration 1
    /// would make the previous iteration's W equal the final one and pass whatever
    /// the column carried. So the same fixture is also solved at `max_iter = 1`,
    /// and the two sidecars are required to DISAGREE.
    #[test]
    fn test_em_accum_weight_tracks_the_final_estep_not_the_m_step_before_it() {
        const TOL: f64 = 1e-12;

        // Solve the fixture at `max_iter` and return, per transcript and sorted by
        // name, (a) the sidecar's W and (b) the PROFILE's own per-read weight sum.
        // `mu_cohort` differs between the two candidates, which is what makes the
        // posteriors move from one E-step to the next.
        //
        // (b) is read out of a different parquet, by a different builder, from a
        // different function -- and calls nothing that produces (a).
        let run = |tag: &str, max_iter: usize| -> (Vec<(String, f64)>, Vec<(String, f64)>) {
            let emission = scratch(&format!("stale_{tag}_emission.parquet"));
            let profile = scratch(&format!("stale_{tag}_profile.parquet"));
            let sidecar = scratch(&format!("stale_{tag}_sidecar.parquet"));

            let graph =
                multi_molecule_graph((100.0_f32).log2(), &[vec![("x", 1000, 0), ("y", 1000, 0)]]);
            // The read's log2(100) is ~6.644: r0_x sits above it, r0_y below.
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
                // `min_weight = 0`, NOT `MIN_WEIGHT`. The output stage stores
                // `w / retained_sum`, so `retained_sum` must be 1 for the profile's number to
                // be the raw weight the cross-check needs. At `MIN_WEIGHT` the far candidate
                // falls below the cutoff, is dropped, and `retained_sum` becomes the near
                // candidate alone.
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

        // VACUITY GUARD, FIRST. Without it the headline assertion below can be
        // satisfied by a stale column whenever the EM converges immediately.
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

        // THE CROSS-CHECK. `retained_sum == 1` requires BOTH candidates to be
        // emitted; otherwise `posterior_weight` carries `w / retained_sum` and
        // the comparison below is between two different quantities. Two
        // candidates in, two profile rows out.
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

    /// `TelEmStats` carries the measured size of the one-iteration phase offset
    /// between the W the M-step CONSUMED and the W the sidecar PUBLISHES.
    ///
    /// WHY THE STRUCTURAL CLAIM IS "exactly zero at `max_iter = 1`". With one
    /// iteration the loop runs E-step, then M-step over those same
    /// `final_weights`, then exits; the stamp block then recomputes `final_w`
    /// from that identical vector. Both arms are the same numbers and the metric
    /// MUST be bit-zero. That is the falsifiable half: a `max_dw_*` that is
    /// non-zero here would mean the two vectors are not what the comment says they
    /// are.
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

        // The other direction, so a metric stuck at zero cannot pass: the two
        // candidates have different `mu_cohort`, which is what makes the
        // posteriors MOVE between E-steps. A run that iterates must therefore
        // report a strictly positive absolute gap.
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

    /// `sigma2_cohort` is a MISNOMER: the field carries this transcript's OWN
    /// `sigma2_logPAL`, not a cohort-wide variance. The M-step multiplies it by
    /// `beta_0` in the sigma^2 numerator, so anyone reading it as a single cohort
    /// constant overstates the prior on every tier-1/2 row.
    ///
    /// This is the RED half. The falsifiable claim is not "the field equals
    /// something" -- that is true by construction -- it is that three transcripts
    /// with three DIFFERENT emission variances come back with three DIFFERENT
    /// values rather than one shared number.
    #[test]
    fn test_emission_prior_fields_are_bound_from_the_emission_parquet() {
        let emission = scratch("prior_binding_emission.parquet");
        // Three distinct variances AND three distinct mus, so a mixture of the
        // two columns cannot accidentally satisfy the assertions below.
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

        // `sigma2` is INITIALISED from the same column and the M-step rewrites
        // only that one, so before any solve the two agree exactly.
        assert_eq!(get("tx_a").sigma2.to_bits(), 0.5_f64.to_bits());

        // The property that makes this more than three restatements of the same
        // number: the three values must be pairwise distinct. If any future
        // edit made them share a value, the "own column" wording in the field
        // docs would be false while every assertion above still passed.
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

    /// D1, negative half. `sigma2_cohort` enters the M-step as a VARIANCE
    /// (`2.0 * beta_0 * p.sigma2_cohort`). A non-finite or non-positive value
    /// poisons `p.sigma2` for that transcript and, through the cohort fallback,
    /// for every transcript borrowing it -- silently, because the arithmetic still
    /// completes. So the loader refuses it instead.
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

    /// `solve_and_write_profile` is public, so the CLI's `poly_a_weight = 0`
    /// interlock is not enough: a library caller can pair a placeholder graph with
    /// a positive weight, which feeds a constant emission into the E-step and
    /// distorts every posterior. That combination must be unexpressible at the
    /// API boundary too.
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

        // The SAME graph at weight 0 must still run: the guard is about the
        // pairing, not about the placeholder existing.
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

        // And a REAL graph at a positive weight must be unaffected -- otherwise
        // the guard would fire on ordinary runs.
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

    /// As `synth_graph`, but each candidate also carries its observed 5' gap `g_k`
    /// (bp from the transcript's 5' end).
    ///
    /// `AS` is held under the caller's control and `g_k` is independent of it,
    /// which is what lets a 5'-term test separate `ln_5p` from `ln_a`: hold AS
    /// EQUAL and any posterior shift is attributable to the gap alone.
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
            // synthetic/test graph: `log_pal` is a real measurement
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        }
    }

    /// One-candidate-per-entry graph plus a matching `params` vector, for testing
    /// [`candidate_availability`] directly.
    ///
    /// `spec` is one `(AS, is_competent, alpha)` per candidate, in the order they
    /// should occupy the read's candidate range; `present` says whether that
    /// candidate keeps its emission row. `synth_graph_with_gaps` assigns
    /// `tx_id == position`, so `params[i]` is the parameters of candidate `i`.
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

    /// A graph with one molecule PER ENTRY of `molecules`, so a fixture can mix
    /// the unambiguous and contested populations in a single solve.
    ///
    /// `synth_graph_with_gaps` builds exactly one molecule, so it cannot express
    /// "one read with one candidate and another with three" -- and the ledger is
    /// precisely a statement about BOTH populations at once. Transcript names are
    /// made globally unique by prefixing the read index, because `candidate_pool`
    /// holds indices into ONE shared `transcript_names`.
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

    /// The emission parquet matching [`multi_molecule_graph`]'s naming, for
    /// every transcript named in `wants` and no others.
    fn multi_molecule_emission(path: &Path, wants: &[&str]) {
        let rows: Vec<(&str, &str, f64, f64)> = wants.iter().map(|t| (*t, "g", 7.0, 0.5)).collect();
        write_test_emission(path, &rows);
    }

    /// THE CONSERVATION LAW. Every candidate the graph produced is claimed by
    /// exactly one terminal fate.
    ///
    /// Reverting the prune increment must turn THIS test red -- a ledger test
    /// that cannot detect an unaccounted candidate is the same class of failure as
    /// the bug it was written for.
    ///
    /// The fixture deliberately covers the four contested fates and all three
    /// unambiguous ones in one solve, because the identity is a claim about their
    /// SUM. A fixture exercising one bucket at a time would pass just as happily
    /// with the buckets wired to the wrong branches.
    #[test]
    fn test_candidate_ledger_claims_every_candidate_exactly_once() {
        let emission = local_scratch("ledger_emission.parquet");
        let profile = local_scratch("ledger_profile.parquet");
        // r0: 1 candidate  -> unambig, emitted.
        // r1: 2 equal-AS   -> contested, both live, both emitted at MIN_WEIGHT.
        // r2: 1 live + 1 at AS 500 -> contested, the second is ΔAS-pruned
        //     (delta_as = 500 > 50) BEFORE the emission lookup, which is the
        //     precedence being pinned: r2_bad is deliberately absent from the
        //     emission parquet, and must be reported as pruned, not missing.
        // r3: 1 candidate with no emission row -> unambig, missing.
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

    /// `min_weight` culling is a terminal fate and must be on the books, and an
    /// all-culled read must still be PUBLISHED rather than silently lost.
    ///
    /// Uses `min_weight = 0.6` on a 2-way contested read whose posteriors are
    /// 0.5/0.5, so BOTH fall below the cut. The argmax is then retained on its
    /// own and emitted, so the ledger must show THREE independent facts at once:
    ///
    ///   * `pruned_min_weight == 1` -- the LOSER is culled and not emitted. It is
    ///     1, not 2, because the argmax is culled-but-emitted and counting both
    ///     would double-count it against `terminal_sum()`.
    ///   * `rows_emitted == 1` -- the read is on the books.
    ///   * `reads_retained_by_argmax == 1` -- and that row was forced, so nothing
    ///     downstream mistakes it for an earned one.
    ///
    /// The separation of the first two is the load-bearing assertion: it is what
    /// stops the retention from corrupting the conservation check.
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

    /// Mass conservation, pinned as its own fact. The read above is published with
    /// `posterior_weight = 1.0`, so the profile's weight sum must equal the number
    /// of reads that reached the output stage. If the retained argmax were emitted
    /// with a `retained_sum` of 0 this would be `NaN`, and a NaN compares false
    /// against every expectation -- so the sum is checked for finiteness
    /// explicitly rather than only for equality.
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

    /// At `min_weight = 0.0` the cutoff is provably inert: every posterior is
    /// `exp(ll - log_z) >= 0`, so `w < 0.0` is never true. Pinning the zero is
    /// a free check that `pruned_min_weight` is wired to THAT branch and not to
    /// some other `continue` in the emit loop.
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

    /// `incompetent` is the one death cause the conservation law alone cannot
    /// police, and this pins it end to end.
    ///
    /// The sum is preserved whichever of the `incompetent` and `alpha_zero`
    /// increments is misplaced, so a total-only assertion cannot see it.
    ///
    /// WHY `alpha_zero` IS NOT PINNED HERE. It is **structurally unreachable
    /// through the solver**: a transcript reaches `alpha == 0` only by winning no
    /// posterior mass, and a transcript that wins no mass was already dead by
    /// competence or by having no emission row.
    ///
    /// So `alpha_zero`'s own bucketing is pinned where it IS reachable, by
    /// `test_candidate_availability_keeps_four_death_causes_apart` calling
    /// `candidate_availability` directly, and the AGGREGATION wiring is pinned
    /// from the `incompetent` side: transposing the two increments drives
    /// `incompetent` to 0 and turns THIS test red even though `alpha_zero` is
    /// unreachable either way.
    #[test]
    fn test_candidate_ledger_attributes_an_incompetent_candidate_end_to_end() {
        let emission = local_scratch("ledger_cause_emission.parquet");
        let profile = local_scratch("ledger_cause_profile.parquet");
        // r0_incomp: present, is_competent == false
        // r0_ok / r0_ok2: present, competent, identical fit, so they tie
        // exactly and split the posterior. r0_incomp is pushed as `-inf` by the
        // E-step and must land in `incompetent`.
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

    /// The census must COUNT a pruned candidate even though it must not ATTRIBUTE
    /// one to it. These are different questions and the ledger needs the first
    /// while ghost decomposition needs the second.
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
        // The prune still disqualifies uniform attribution: a read with a
        // pruned candidate is not reported as `Missing`, because padding
        // cannot revive a candidate the aligner already ruled out.
        assert_eq!(
            av.uniform_cause(2),
            None,
            "one live + one pruned is not a uniform cause"
        );
    }

    /// `max_iter = 0` runs no E-step, so `last_estep_params` is empty and the
    /// census would classify every contested candidate as having no emission row
    /// -- a confident falsehood about a run that never looked at one. The input is
    /// rejected instead.
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

    /// The census has to keep four things apart that all end in "this read has no
    /// posterior", and the four are distinguished ONLY by which filter killed the
    /// candidate. Three of the four are reachable from production data and one
    /// (the `ΔAS` prune) is not: a pruned candidate is dead for a reason no amount
    /// of emission padding addresses, so a read that has one is NOT reported as
    /// recoverable.
    #[test]
    fn test_candidate_availability_keeps_four_death_causes_apart() {
        // AS 100 is the read's best; AS 20 gives delta_as = 80 > 50, so
        // candidate 4 is pruned however healthy its emission row is.
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

        // Each uniform cause, one at a time. `alpha == 0` is unreachable through the
        // solver, which is exactly why it is pinned HERE rather than only in the
        // end-to-end test.
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
        // A missing candidate ALONGSIDE a pruned one: padding would restore the
        // missing row, but the pruned candidate stays dead, so the read stays a
        // ghost. Reporting `Missing` would promise a rescue that does not come.
        assert_eq!(
            census_fixture(&[(100, true, 0.5), (20, true, 0.5)], &[false, true]).uniform_cause(2),
            None,
            "one candidate is missing and the other is pruned: the read is NOT \
             rescued by padding, because the pruned candidate dies either way"
        );
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

    /// Regression test for the A_rk term: it must reach the posterior.
    ///
    /// Both candidates carry IDENTICAL emission parameters and identical π at
    /// initialization, so `π_k · N(y; μ_k, σ_k²)` is exactly equal for the two
    /// and A_rk is the only term able to separate them. ΔAS = 200 at λ = 100 is
    /// a 2.0-nat gap → posterior 0.8808 / 0.1192.
    ///
    /// Mutation target: dropping `+ ln_a` from the E-step flattens both candidates
    /// to 0.5 and this test goes red.
    ///
    /// `max_iter = 1` runs exactly one E-step from the symmetric-π initialization,
    /// so the M-step feedback cannot be mistaken for the A_rk effect.
    ///
    /// Both new terms are explicitly disabled: `PRUNE_OFF` because this
    /// fixture's `ΔAS = 200` IS the quantity the `ΔAS > 50` prune thresholds on,
    /// and `FIVE_PRIME_OFF` so the 0.880797 pin below remains a pure A_rk
    /// measurement.
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

    /// Regression test for the output-stage renormalization after the `min_weight`
    /// truncation.
    ///
    /// tx1's emission term is 5 nats worse than tx0's, landing it at
    /// `e⁻⁵/(1+e⁻⁵) = 0.0067` -- below `MIN_WEIGHT`, so it is truncated away and
    /// 0.0067 of probability mass is destroyed. The rows actually emitted must
    /// still sum to exactly 1.0.
    ///
    /// Mutation target: summing the denominator over ALL weights instead of the
    /// retained ones leaves the surviving row at 0.9933 and this test goes red.
    /// Equal AS on both candidates keeps A_rk ≡ 1, so the split is produced by the
    /// emission term alone and this test stays independent of λ.
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

    /// Read the profile as `(read_id, transcript_id, weight)` so a test can
    /// isolate ONE read's posterior instead of the transcript-level aggregate.
    /// The aggregate is useless for this: a graph with many unambiguous reads
    /// is dominated by them, and the ambiguous read's own signal is swamped.
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

    /// A graph whose abundance prior is set entirely by UNAMBIGUOUS reads, so the
    /// prior is knowable without running the EM.
    ///
    /// 30 single-candidate molecules on `tx0`, 3 on `tx1`, then one two-candidate
    /// `PROBE` molecule with EQUAL `AS` to both and `g_k = 0` on both. The probe
    /// therefore has no evidence of its own, and the init prior is
    /// `(0.01 + 30 + 0.5) : (0.01 + 3 + 0.5)` -- the Laplace epsilon and the
    /// probe's half-share are `solve_and_write_profile`'s convention, read here so
    /// the fixture does not hide where the number comes from.
    fn prior_biased_graph() -> FlatEvidenceGraph {
        let log_pal = (100.0_f32).log2();
        // Pools are molecule-major / candidate-minor and all must be the same
        // length, so the count is tracked as we go.
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
            // synthetic/test graph: `log_pal` is a real measurement
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        };
        assert_eq!(graph.candidate_pool.len(), n);
        assert_eq!(graph.candidate_as.len(), n);
        assert_eq!(graph.candidate_5p_gap.len(), n);
        graph
    }

    /// MODEL IDENTITY AXIOM: **when every likelihood evidence term is equal across
    /// candidates, the posterior must reduce to the abundance prior.** This is a
    /// property of the E-step, not of any flag, so it must hold in EVERY arm.
    ///
    /// Why it is a separate test from the `poly_a_weight` regression below: that
    /// one pins one flag at one setting and would still pass if a future edit
    /// swallowed `ln alpha` inside some *other* weight's bracket. The shared
    /// failure mode is not "a parameter is set wrong", it is "the mixture prior
    /// stopped being a mixture prior", and that can happen through any of the
    /// three weights. So this one sweeps all three.
    ///
    /// The sweep is exact rather than approximate because the fixture makes each
    /// evidence term a CONSTANT across the two candidates, and a constant cancels
    /// in the softmax for any scale factor:
    ///   - `AS` equal      -> `ln_a = -ΔAS/λ = 0` for every `λ`
    ///   - `g_k = 0` equal -> `ln_5p = five_prime_weight` for both
    ///   - `mu`, `sigma2` equal -> the Gaussian is one constant times
    ///     `poly_a_weight`
    ///
    /// So the odds `w_0 / w_1` must equal the prior odds in all arms, and must be
    /// invariant ACROSS arms. That invariance is scale-free, so it survives a
    /// legitimate change to the init convention, which a hard-coded value would
    /// not.
    ///
    /// A posterior near 0.5 in any arm means `ln alpha` is not reaching that arm's
    /// softmax at all -- the M-step then re-estimates abundances that no posterior
    /// can see, and the "EM" is inert.
    #[test]
    fn test_e_step_reduces_to_the_abundance_prior_when_all_evidence_is_tied() {
        let log_pal = (100.0_f32).log2();
        // Equal mu and sigma2: the poly(A) Gaussian is a shared constant, so it
        // cancels and poly_a_weight can be swept without breaking the tie.
        let emission = scratch("identity_emission.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", log_pal as f64, 0.04),
                ("tx1", "gene1", log_pal as f64, 0.04),
            ],
        );

        // (poly_a_weight, five_prime_weight, aln_weight_lambda, label)
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

    /// The M-step's `alpha` denominator is `n_total_molecules`, and that is
    /// OBSERVABLE even though nothing downstream can see the scale.
    ///
    /// `alpha` is a mixture weight vector, and `p.alpha = w_sum / n_total_molecules`
    /// is what makes it one: every molecule contributes total weight 1 (an
    /// unambiguous read contributes exactly 1.0; an ambiguous read's posteriors sum
    /// to 1 by construction of the log-sum-exp), so `sum_k w_sum_k ==
    /// n_total_molecules` and the updated `alpha` sums to 1 across transcripts.
    ///
    /// A UNIFORM rescale of the alpha vector is invisible to a softmax -- `ln c` is
    /// common to every candidate -- so no test that only reads posteriors can see
    /// the denominator at all, at any scale. That is a real property of the model,
    /// not a gap in the test, and it is why the assertion reads `alpha` itself.
    /// What duplication adds is that it compares NON-uniform rescalings: `N ->
    /// N-5` rescales every component identically, whereas `N -> 2N` is a genuine
    /// change of normaliser.
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

        // 30 unambiguous molecules on tx0, 3 on tx1, 1 ambiguous PROBE: 34 molecules
        // over 2 transcripts, every one of them with at least one candidate that has
        // emission parameters. That last property is what the sum-to-1 invariant
        // rests on -- a molecule whose every candidate is missing from the emission
        // parquet contributes 0 to `accum_w` while still counting in
        // `n_total_molecules`.
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

        // CONTROL, and it has to come first: `init_counts / total_init` ALSO sums to
        // 1, so the assertion below would pass on a solver whose M-step never ran.
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

        // The numerator is the unambiguous counts plus the PROBE's soft share,
        // so tx0 (30) must stay above tx1 (3) by a wide margin: the PROBE can
        // move at most 1.0 of weight between them. Pins that the M-step read the
        // same `accum_w` its denominator is normalised by.
        assert!(
            alpha[0] > alpha[1] && (alpha[0] - alpha[1]).abs() > 0.5,
            "tx0 carries 30 unambiguous molecules against tx1's 3, so alpha must \
             stay strongly ordered: {alpha:?}"
        );
    }

    /// Two copies of every molecule, laid out so each copy owns its own slice
    /// of the candidate pool. Read ids get a suffix because the profile writer
    /// keys rows by them.
    ///
    /// The candidate slices are re-walked per molecule rather than assumed
    /// contiguous, so this stays correct for any `candidate_offset` layout.
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

    /// The abundance estimate is a PER-MOLECULE quantity: the M-step divides each
    /// transcript's accumulated weight by the TOTAL molecule count, so
    /// `alpha_k = W_k / N`. Both the numerator and the denominator therefore
    /// double when every read is duplicated, and
    ///
    ///     alpha_k(2x) = 2 W_k / 2 N = alpha_k(x)
    ///
    /// The solution must not depend on how many molecules the run happened to
    /// contain. This is a statement about the ESTIMATOR, not about any particular
    /// divisor.
    ///
    /// The tightness, and what limits it. The one-shot quotient is EXACTLY
    /// invariant, but `alpha` is a FIXED POINT of EM, not a one-shot value: `W_k`
    /// is itself a sum of posteriors that depend on `alpha`. And the fixed point
    /// is not exactly duplication-invariant either, because none of the prior terms
    /// scale with the data: `tau_0 * mu_cohort`, `beta_0 * sigma2_cohort` and the
    /// Laplace `0.01` all stay put while the data doubles. Measured on this
    /// fixture: 3.4e-6.
    ///
    /// That is the number the tolerance is set against. A divisor that is not a
    /// common factor of the count moves `alpha` by 8.6e-2 on the same fixture, so
    /// the tolerance below sits ~30x above the clean spread and ~860x below the
    /// defect.
    ///
    /// `mu` and `sigma2` are deliberately NOT asserted here: they are not
    /// duplication-invariant. The prior term `tau_0 * mu_cohort` does not scale
    /// with the data, so the shrinkage moves by order 1e-3 here. Asserting them
    /// would force a tolerance four orders looser than the one this defect needs.
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

        // GUARD. If the duplication were a no-op -- empty pool, a wrong
        // candidate_count, a slice that does not get appended -- the two runs
        // would be identical by construction and the invariance below would be
        // true no matter what the M-step computes.
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
            // The duplication is two consecutive passes over the original, so
            // molecule i of copy c corresponds to original i % n.
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

        // No type annotations: the out-param's element type is inferred, so
        // this reads the field rather than restating a signature that could
        // drift.
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

        // 1e-4: ~30x the measured clean spread (3.4e-6) and ~860x below what a
        // wrong divisor produces (8.6e-2). Both numbers are in the doc comment.
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

    /// The abundance prior `ln alpha` is the MIXTURE prior, not part of the
    /// poly(A) emission, so it must survive `poly_a_weight = 0`.
    ///
    /// The parameter was originally written as `poly_a_weight * (alpha.ln() - ...)`,
    /// which looks like a harmless grouping. It is not: at weight 0 the E-step
    /// never reads alpha, so the M-step's abundance update cannot influence any
    /// posterior, and the "EM" silently degenerates into a one-shot static
    /// alignment scorer. The number it prints is still a plausible profile, which
    /// is exactly why it would have survived review.
    ///
    /// Fixture: unambiguous reads set the prior (tx0 gets 30, tx1 gets 3), then
    /// ONE ambiguous read with EQUAL `AS` to both, 5' off, poly(A) off. With no
    /// evidence of its own, that read must follow the prior. If alpha is scaled
    /// away the two candidates are exactly tied and it splits 50/50.
    ///
    /// The assertion is on the AMBIGUOUS READ'S OWN weight, not the aggregate:
    /// the 33 unambiguous reads would otherwise swamp the signal.
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
        // init alpha is proportional to the unambiguous counts: 30 vs 3, so
        // ~0.90 once the probe's own half-share is added to both.
        assert!(
            probe > 0.7,
            "with poly(A) OFF the ambiguous PROBE read has no evidence of its own, \
             so it must follow the abundance prior (~0.9 onto tx0). It got {probe}. \
             If this is ~0.5, `ln alpha` is being scaled by poly_a_weight and the \
             M-step cannot influence any posterior -- the EM is inert."
        );
    }

    /// `poly_a_weight = 0` must remove the poly(A) term COMPLETELY. Note what
    /// remains is `ln alpha_k + ln_a`, i.e. the abundance prior PLUS alignment --
    /// this arm is not, and cannot be made, an alignment-only baseline.
    ///
    /// This cannot be faked from the outside. The obvious data trick -- pin `mu`
    /// and `sigma2` to the same value for every transcript so the Gaussian term
    /// is a constant that cancels in the softmax -- does NOT work, because the
    /// `ln alpha_k` term is re-estimated by the M-step and would still carry the
    /// abundance prior into every candidate's log-likelihood. Hence a parameter.
    ///
    /// Fixture: two transcripts at EQUAL `AS`, so `ln_a = 0` for both and any
    /// posterior movement is attributable to poly(A) alone; and well-separated
    /// `mu`, so poly(A) is decisive rather than marginal. The 5' term is off.
    ///
    /// The second arm is the neuter: if `poly_a_weight` were being ignored, or
    /// applied to the wrong term, the off-arm would still favour tx0 and the
    /// `0.5` assertion would fail.
    #[test]
    fn test_poly_a_weight_zero_removes_the_poly_a_term_entirely() {
        let log_pal = (100.0_f32).log2();
        let sigma2 = 0.04;
        let emission = scratch("paw_emission.parquet");
        write_test_emission(
            &emission,
            &[
                ("tx0", "gene0", log_pal as f64, sigma2),
                // Same AS, far-away mu: only poly(A) can separate them.
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
                // 0.0 so no candidate is truncated away before the comparison;
                // otherwise a losing candidate vanishing would look like
                // "removed by the parameter".
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

    /// `min_weight = 0` must mean EXACTLY "no truncation", and that has to be a
    /// real code path, not an accident of `w < 0.0` never being true.
    ///
    /// This is load-bearing for ablation work: the arm that shifts mass onto a
    /// candidate is the arm that clears the 0.01 cutoff, so the two arms are not
    /// scored on the same support and their difference is not purely the ablated
    /// term. The remedy is to re-run at `min_weight = 0`, which is only legitimate
    /// if 0 restores the full untruncated posterior.
    ///
    /// Same fixture as the test above, where the `tx1` candidate sits ~5 nats
    /// down and is therefore truncated at the default but must survive at 0.
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

    /// An INCOMPETENT candidate must not break per-read renormalisation.
    ///
    /// The output stage's `retained_sum` denominator sums `{w >= min_weight}` over
    /// `weights`, but the emit loop then drops rows that are `!p.is_competent` or
    /// missing from `params`. That is only sound because the E-step pushes
    /// incompetent candidates as `-inf` and `filter(ll > -inf)` removes them from
    /// `weights` before the denominator is ever taken. Nothing TESTED it, so a
    /// future change to the competence mask would make every read's emitted
    /// weights sum to less than 1.0 with the suite still green.
    ///
    /// The fixture puts the 5-nat handicap on the **COMPETENT** transcript, so the
    /// INCOMPETENT one is the candidate the likelihood likes most. That direction
    /// matters: with the handicap on the incompetent transcript instead, its
    /// weight lands below `min_weight` and is dropped by TRUNCATION, so the test
    /// would pass even with the competence mask fully disabled.
    #[test]
    fn test_incompetent_candidate_is_excluded_from_the_renormalisation_base() {
        let log_pal = (100.0_f32).log2();
        let sigma2 = 0.04;
        let emission = scratch("incomp_emission.parquet");
        let profile = scratch("incomp_profile.parquet");
        write_test_emission_with_competence(
            &emission,
            &[
                // μ offset sized so `(y − μ)² / (2σ²)` = 5.0 nats. On the
                // COMPETENT transcript, so the incompetent one is favoured.
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
        // NOTE on `stats.n_dropped_output_row_competence` being 0 here. That counter
        // is incremented by the OUTPUT stage's `!p.is_competent` check, which is
        // unreachable for an AMBIGUOUS read: the E-step already pushed this candidate
        // as `-inf` and `filter(ll > -inf)` removed it from `weights`, so no
        // incompetent row ever reaches the emit loop. That unreachability IS the
        // invariant this test exists to pin.
        let _ = stats;
    }

    /// The output stage's `retained_sum` denominator, with ALL THREE of the routes
    /// by which a candidate can leave `weights` or the emitted rows present at
    /// once.
    ///
    /// `retained_sum` sums `{w >= min_weight}` over `weights`, and the emit loop
    /// then drops rows that are `!p.is_competent` or missing from `params`. Those
    /// two later filters are currently unreachable -- the E-step has already pushed
    /// incompetent candidates as `-inf` (and `filter(ll > -inf)` removed them) and
    /// `if let Some(ref p) = params[...]` SKIPS a candidate with no emission row
    /// instead of pushing anything. So the denominator is right only because the
    /// E-step and the emit loop agree on the candidate set.
    ///
    /// Why this is not a third copy of the two tests above: each of those isolates
    /// ONE route on a two-candidate read whose survivor absorbs the entire
    /// posterior, so only the row COUNT is load-bearing. Here the surviving mass
    /// is SPLIT, so the denominator is pinned by the per-row values as well as by
    /// the sum, and all three exclusion routes are in the same read:
    ///
    /// | candidate | route out                                    | weight |
    /// |-----------|----------------------------------------------|--------|
    /// | `tx0`     | retained                                    | 0.80   |
    /// | `tx1`     | retained                                    | 0.15   |
    /// | `tx2`     | `w = 0.005 < min_weight` -> TRUNCATED        | 0.005  |
    /// | `tx3`     | `is_polyA_competent = false` -> E-step `-inf`| 1.00   |
    /// | `tx4`     | no emission row -> E-step `if let Some` skip | 1.00   |
    ///
    /// The handicaps are `mu` offsets sized so `(y - mu)^2 / (2σ²)` differs by
    /// exactly `ln(0.80/0.15)` and `ln(0.80/0.005)` nats. All five candidates
    /// carry EQUAL `AS`, so `ln_a = 0` for all and the split is the emission term
    /// alone. `max_iter = 1` freezes the posterior to the E-step that ran on the
    /// initial (uniform) prior, so the expected values below are exact.
    ///
    /// Mutation targets, both confirmed to turn this red:
    ///   * the denominator summing over ALL weights (`*w >= 0.0`) leaves the two
    ///     survivors at 0.8377 / 0.1571 and the read summing to 0.9948;
    ///   * the E-step's competence mask bypassed lets `tx3` -- the FAVOURED
    ///     candidate, by construction -- take ~0.98 of the mass into
    ///     `retained_sum` and into a row, dropping `tx0` to ~0.0067. The sum
    ///     stays 1.0 under that mutation, which is why the per-row values are
    ///     asserted and not just the sum.
    #[test]
    fn test_retained_sum_covers_exactly_the_rows_emitted_when_all_three_exclusions_fire() {
        let log_pal = (100.0_f32).log2();
        let sigma2 = 0.04;
        let emission = scratch("retained_emission.parquet");
        let profile = scratch("retained_profile.parquet");
        // tx4 is deliberately ABSENT: the E-step's `if let Some(ref p) =
        // params[tx_id]` has nothing to score it with and skips it outright.
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
                // No handicap at all: the FAVOURED candidate is the doomed one.
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
            // 1 iteration: the emitted posterior is the E-step on the uniform
            // init prior, so the expected split is exact rather than
            // convergence-dependent.
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

        // The split, not just the sum. These pin the denominator to the
        // RETAINED set: with tx0 and tx1 sharing `0.80 : 0.15` of the
        // untruncated mass, the only denominator that yields them is
        // `0.80 + 0.15 = 0.95`.
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

    /// Disjoint id spaces must be REFUSED, not silently absorbed.
    ///
    /// The failure is what it looks like: every read takes the ghost-drop path
    /// and the run still exits 0 behind a normal-looking telemetry block.
    ///
    /// The `!profile.exists()` assertion is the one that matters: it pins that
    /// the guard fires BEFORE the writer opens the file. A guard placed after the
    /// write would still read as correct from the error message alone.
    #[test]
    fn test_disjoint_id_spaces_are_refused_before_any_profile_is_written() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("idspace_emission.parquet");
        let profile = scratch("idspace_profile.parquet");
        // Emission ids deliberately share no key with the graph's "tx0".
        write_test_emission(
            &emission,
            &[("AIDRS_tx_999", "gene9", log_pal as f64, 0.04)],
        );
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

    /// Regression test for the *non-degenerate* renormalization path.
    ///
    /// The test above only ever retains one row, so its emitted weight is
    /// `w / w_max = 1.0` by construction and the interesting arithmetic --
    /// dividing **several** surviving rows by a `retained_sum < 1` -- is never
    /// exercised. Three candidates split the posterior in the ratio
    /// 0.80 : 0.15 : 0.005. The third is below `MIN_WEIGHT` (0.01) and is
    /// truncated, so 0.005 of probability mass is destroyed and the denominator
    /// drops to the retained 0.80 + 0.15. The emitted rows must therefore be
    /// ≈ 0.8421 / 0.1579 -- not 0.80 / 0.15 -- and sum to 1.0.
    ///
    /// Mutation target: summing the denominator over ALL weights instead of the
    /// retained ones leaves the emitted rows at 0.8377 / 0.1571, summing to
    /// ≈ 0.9948, and this test goes red.
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

    /// Guards the three parameter validations, which run before any I/O and are
    /// therefore callable with a deliberately nonexistent emission path -- a bail
    /// with a parameter message proves the guard fired, and any other error would
    /// prove it did not.
    ///
    /// The CLI's own `anyhow::bail!` calls cannot be reached this way, which is
    /// why the core check is the one that matters: it is the last gate before a
    /// bad parameter becomes NaN posterior weights, and it is NaN-safe, which
    /// `!(x > 0.0)` is not.
    ///
    /// Mutation targets: `five_prime_weight < 0.0` → `<= 0.0` rejects the
    /// documented off switch and makes the 5' ablation unrunnable; dropping
    /// `.is_nan()` lets a NaN weight through every comparison and emits NaN
    /// posteriors; dropping the `prune_max_delta_as < 0` check lets a negative
    /// threshold drop each read's own argmax.
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

        // Off switch and legal extremes must NOT be rejected by validation.
        // (They will still fail on the missing emission file, which is why the
        // assertion is "not a parameter error" rather than success.)
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

        // Rejected values.
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

    /// Run one E-step (`max_iter = 1`) over a two-candidate read and return the
    /// emitted `(tx0, tx1)` posterior weights. Both candidates share identical
    /// emission parameters, so `π_k · N(y; μ_k, σ_k²)` is equal across the pair
    /// and the split is produced by the evidence terms alone.
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
        // Returned raw: several of these tests assert that a candidate was
        // PRUNED or truncated away, so looking a weight up eagerly would panic
        // inside the helper instead of failing the caller's own assertion.
        rows
    }

    /// TEST A of the ablation gate: the 5' term must reach the posterior.
    ///
    /// Two candidates with IDENTICAL `AS` (so `ln_a ≡ 0` and `A_rk` cannot
    /// separate them) and identical emission parameters, differing only in the
    /// observed 5' gap: tx0 starts 5 bp from the transcript 5' end
    /// (`g_k = 5 <= 20`) and tx1 starts 500 bp in (`g_k = 500 > 20`). The
    /// 1.8-nat bonus must give tx0 `e^1.8 / (1 + e^1.8) = 0.858149`.
    ///
    /// This isolates the term: holding AS and the emission fixed is the only way
    /// `ln_5p` can be the explanation for the shift.
    ///
    /// Mutation targets: dropping `+ ln_5p` from the E-step, computing the window
    /// as `g_k < 20` (this fixture would still pass -- see
    /// `test_five_prime_window_is_inclusive_at_20`), reading the gap from the
    /// wrong pool index, or leaving `FIVE_PRIME_WEIGHT` at 0 all flatten the pair
    /// to 0.5 / 0.5 and turn this red.
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
        // e^1.8 = 6.049647… → 6.049647 / 7.049647.
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

    /// TEST B of the ablation gate, at unit scale: REDRAWING each read's 5' start
    /// must RESTORE the no-5'-term posterior.
    ///
    /// This is a NO-LEAK test, not a liveness test: the redraw makes the two
    /// candidates identical, so a flat 0.5/0.5 is the correct answer under ANY
    /// implementation, including one with the term removed entirely. The tests
    /// that actually prove the term is live are
    /// `test_five_prime_term_shifts_posterior_toward_the_anchored_candidate` and
    /// `test_five_prime_window_is_inclusive_at_20`.
    ///
    /// The acceptance criterion for the real ablation is a DIFFERENCE (Test A −
    /// Test B ≥ 20 pp), never an absolute percentage, because the absolute value
    /// is measured against a label that is not ground truth.
    ///
    /// Both redraw flavours must return the read to a flat 0.5 / 0.5:
    ///   (a) redraw to a gap OUTSIDE the window (500 for both) -- the term fires
    ///       for neither candidate, so it is identically 0;
    ///   (b) redraw to a gap INSIDE the window (7 for both) -- the term fires for
    ///       both, so it is a shared additive constant and CANCELS in the
    ///       log-sum-exp. This is the sharper of the two: a term that leaked
    ///       outside the softmax, or was applied after `log_z`, would NOT cancel
    ///       here and this case would go red.
    ///
    /// The flat reference is the posterior the same fixture produces with the
    /// term switched off, asserted explicitly so the two cannot drift.
    #[test]
    fn test_redrawing_the_5p_start_restores_the_flat_posterior() {
        let reference_rows = two_candidate_posteriors(
            "five_prime_off_ref",
            // A gap inside the window that CANNOT be the discriminator: both
            // candidates identical in every respect that matters.
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

        // (a) redraw outside the window → term is identically zero.
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

        // (b) redraw inside the window → term is a shared constant and cancels.
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

    /// The window is `g_k <= 20`, INCLUSIVE. 20 bp inside must earn the full
    /// bonus; 21 bp must earn none. A `< 20` comparison would move the frozen
    /// threshold by one base against the harness the parameter was validated on,
    /// and -- because the coordinate comes from a 1-based `s_k` with the
    /// off-by-one applied upstream -- an off-by-one here is indistinguishable from
    /// a coordinate error unless it is pinned at the edge.
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

    /// Pins the posterior at the shipped λ default of 20.0, which until now no
    /// test exercised at all: every pre-existing test passed λ = 100 explicitly.
    ///
    /// `ΔAS = 40` at λ = 20 is a 2.0-nat gap → 0.880797, the same value the A_rk
    /// test reaches at `ΔAS = 200` / λ = 100. `ΔAS = 40` is under the prune
    /// threshold, so this pin is unaffected by pruning.
    ///
    /// Mutation target: shipping any λ ≠ 20 moves this and turns it red.
    #[test]
    fn test_lambda_20_pins_the_alignment_posterior() {
        let rows = two_candidate_posteriors(
            "lambda20",
            // Equal gaps (both outside the window) so ln_5p ≡ 0 and the pin
            // measures λ alone.
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

    /// The `ΔAS > 50` prune is a hard drop of real posterior mass, so the two
    /// things that can go wrong silently are (1) it fails to fire, and (2) it
    /// fires in a way that leaves the emitted weights summing to less than 1.
    ///
    /// `ΔAS = 200` against a `PRUNE_OFF` run is the same fixture with the
    /// mechanism toggled, so the only difference between the two is the prune
    /// itself.
    ///
    /// Mutation targets: inverting the comparison drops the argmax and empties
    /// the read; writing the prune on the `ln_a` numerator (which is non-positive)
    /// makes it never fire at all, and every assertion below then passes on the
    /// `min_weight` cutoff instead -- see
    /// `test_prune_boundary_is_strictly_greater_than`.
    ///
    /// Note one mutation that deliberately does NOT turn this red: replacing the
    /// `-inf` push with a bare `continue`. The two are equivalent here because
    /// `weights` is built with `filter(ll > -inf)`; see the comment at the prune
    /// for why, and do not read this test as covering it.
    #[test]
    fn test_delta_as_prune_drops_the_candidate_and_keeps_weights_normalised() {
        let candidates: &[(&str, i32, u32)] = &[("tx0", 1000, 900), ("tx1", 800, 900)];

        // Prune ON: ΔAS = 200 is deleted inside the E-step's softmax.
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

        // Prune OFF, SAME fixture: ΔAS = 200 at λ = 20 is −10 nats, so the
        // `min_weight` cutoff deletes the row instead. The emitted ROW COUNT is
        // identical either way, which is precisely why this pair of runs
        // cannot distinguish the two mechanisms and why the interesting case is
        // the next one.
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

        // THE PRUNE'S ACTUAL JOB — a candidate the alignment term and the output
        // cutoff would both KEEP. ΔAS = 60 is −3.0 nats at λ = 20, i.e. w = 0.0474,
        // comfortably above MIN_WEIGHT = 0.01, so with the prune off it survives as a
        // live second hypothesis. The prune deletes it.
        //
        // The region where this matters is `50 < ΔAS < 4.595 × 20 = 91.9`: inside it
        // the prune is strictly stronger than the output cutoff, and it is the whole
        // reason the prune exists now that λ = 20.
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
        // 1/(1 + e^-3) = 0.952574, renormalised over the retained pair.
        assert!(
            (w0 - 0.952_574).abs() < 1e-5,
            "−3.0 nats of A_rk must give 0.9526 on its own, got {w0}"
        );
        assert!(
            (w0 + w1 - 1.0).abs() < 1e-12,
            "emitted weights must sum to 1.0, got {w0} + {w1}"
        );
    }

    /// The prune comparison is strictly greater-than: `ΔAS = 50` survives and
    /// `ΔAS = 51` does not. An off-by-one here silently moves the frozen
    /// threshold against the harness the parameter was validated on, and it
    /// would not be visible in any test that only used a large ΔAS.
    #[test]
    fn test_prune_boundary_is_strictly_greater_than() {
        // ΔAS = 50 exactly: `delta_as > 50` is false, so it survives.
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
        // −2.5 nats at λ = 20 → 1/(1 + e^-2.5) = 0.924142, and both candidates
        // clear MIN_WEIGHT so the pair is renormalised over itself.
        let w_at = weight_of(&rows_at, "tx0");
        assert!(
            (w_at - 0.924_142).abs() < 1e-5,
            "a surviving ΔAS = 50 candidate must still cost 2.5 nats, got {w_at}"
        );
        assert!(
            (w_at + weight_of(&rows_at, "tx1") - 1.0).abs() < 1e-12,
            "the retained pair must sum to 1.0: {rows_at:?}"
        );

        // ΔAS = 51: pruned.
        let rows_51 = two_candidate_posteriors(
            "prune_at_51",
            &[("tx0", 1000, 900), ("tx1", 949, 900)],
            20.0,
            FIVE_PRIME_OFF,
            PRUNE_MAX_DELTA_AS,
        );
        assert_eq!(rows_51.len(), 1, "ΔAS = 51 must be pruned: {rows_51:?}");
    }

    // ── Per-read assignments ──

    /// A graph where the AMBIGUOUS reads are not the leading molecules, so
    /// `ambiguous_indices != identity`. This is the shape that makes the
    /// read_id-to-posterior join testable at all.
    ///
    /// The single-molecule fixtures every other test here uses would align by
    /// ACCIDENT under a `zip(graph.molecules, final_weights)`, which is the bug
    /// this fixture exists to rule out.
    fn mixed_unambiguous_then_ambiguous(log_pal: f32) -> FlatEvidenceGraph {
        // molecule 0: unambiguous, points ONLY at tx2 -> top must be tx2
        // molecule 1: ambiguous, tx0 wins on AS   -> top must be tx0
        // molecule 2: ambiguous, tx1 wins on AS   -> top must be tx1
        //
        // The two ambiguous reads point at DIFFERENT transcripts, so a join that is
        // off by one position is caught by content and not merely by row count.
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
        // AMB_TX0 sees [tx0@900, tx1@870] and AMB_TX1 sees [tx1@900, tx0@860]
        // -- MIRRORED, so each resolves to a DIFFERENT transcript. Giving both reads
        // the same pair in the same order would make both resolve to tx0 and would
        // let an off-by-one join pass.
        let mut candidate_pool = vec![2, 0, 1, 1, 0];
        // The `ΔAS` of each ambiguous read is STRICTLY POSITIVE and STRICTLY BELOW
        // `PRUNE_MAX_DELTA_AS` (50): 30 for AMB_TX0 and 40 for AMB_TX1.
        //
        // Both bounds are load-bearing. If the upper one is violated the prune drops
        // the runner-up before the softmax ever sees it, so every ambiguous read
        // collapses to a ONE-element posterior with `top_weight == 1.0` and
        // `entropy == 0.0` -- and because those two numbers are forced by the
        // arithmetic of a one-element vector rather than measured, mutating either
        // field to a constant left every test green. The two `ΔAS` values are also
        // deliberately UNEQUAL, so the two reads' posteriors differ and a join that
        // swapped them cannot pass by coincidence.
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

    /// A two-molecule graph: `PLAIN` has one candidate, `CULLED` has two at
    /// EQUAL `AS` -- so its posterior is exactly 0.5 / 0.5.
    ///
    /// The tie is exact and stays exact under the M-step, which is what makes this
    /// fixture usable: `CULLED` splits its weight 50/50 between tx1 and tx2, so
    /// the two receive identical emission-parameter updates and identical
    /// abundances, and the only term left to separate them is the one that is
    /// zero for both. Running at `min_weight = 0.9` therefore culls BOTH of
    /// `CULLED`'s candidates -- and the output stage then retains the argmax and
    /// publishes the read under it, at weight 1.0.
    ///
    /// `PLAIN` is in the same graph so "CULLED's row was forced" is not
    /// vacuously true of a profile that is empty for unrelated reasons.
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
            // ALL EQUAL: `PLAIN` on tx0, `CULLED` on tx1 and tx2 at the same
            // score, which is what pins `CULLED`'s posterior at 0.5 / 0.5.
            candidate_as: vec![900, 900, 900],
            transcript_names: vec!["tx0".to_string(), "tx1".to_string(), "tx2".to_string()],
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        }
    }

    /// A two-molecule graph in which `GHOST` has two candidates that are BOTH
    /// poly(A)-incompetent. The E-step pushes `-inf` for each, `max_ll` is
    /// `-inf`, and the read comes back with an EMPTY weight vector: a ghost
    /// drop, counted in `n_dropped_no_competent_candidates`.
    ///
    /// `LIVE` sits beside it so the assertions that `GHOST` is absent from
    /// both tables are not vacuously true of an empty graph.
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

    /// `final_weights` is indexed by AMBIGUOUS RANK, not molecule index, so
    /// pairing it with `graph.molecules` positionally attributes each read's
    /// top-1 to the WRONG read. `ambiguous_indices` here is `[1, 2]`, so a
    /// positional zip would hand molecule 0's posterior to `UNAMB` and shift both
    /// ambiguous reads by one.
    ///
    /// Tie-break determinism. The argmax fold at the output stage uses strict
    /// `>`, so on an EXACT posterior tie the FIRST candidate in pool order wins.
    /// That rule was unpinned, and a refactor could have changed which
    /// transcript a tied read is assigned to without any test objecting.
    ///
    /// The tie here is EXACT, not approximate, and that is the point. All four
    /// evidence terms are identical between the two candidates:
    ///   - `candidate_as` equal   => the alignment term is equal
    ///   - `candidate_5p_gap` equal (both 0, inside the window) => equal
    ///   - identical (mu, sigma2) in the emission table => the Gaussian is equal
    ///   - so the softmax inputs are equal in every component, hence the
    ///     posterior is bitwise equal and the fold must break the tie
    ///
    /// `ΔAS = 0` is also below `PRUNE_MAX_DELTA_AS`, so neither candidate is
    /// pruned: a one-element posterior would make this test pass for the wrong
    /// reason.
    ///
    /// MUTATION: `w > acc.1` -> `w >= acc.1` must turn THIS test red, because
    /// `>=` lets the later equal element overwrite the earlier one and the winner
    /// becomes `tx1`.
    #[test]
    fn test_tied_posterior_resolves_to_the_first_candidate_in_pool_order() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("tiebreak_emission.parquet");
        let profile = scratch("tiebreak_profile.parquet");
        // IDENTICAL (mu, sigma2) => identical poly(A) term. These two rows are
        // the whole reason the posterior ties.
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
            // tx0 first, tx1 second: the ORDER is the thing under test.
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

        // NON-VACUITY. A one-candidate read -- or a read that resolved by AS
        // rather than by a tie -- would give top_weight == 1.0 and entropy
        // == 0.0. Asserting the 0.5/0.5 split first means a later failure of
        // the winner assertion cannot be explained by "the tie never happened".
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

        // The rule under test.
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

        // Every read must be present -- including the unambiguous one, which is
        // ~99.7% of a real read set. A table of ambiguous reads only would
        // silently answer a different question than the one this exists for.
        assert_eq!(
            assigns.len(),
            3,
            "one row per read expected, got {}: {assigns:#?}",
            assigns.len()
        );

        // The unambiguous read: one candidate, so w == 1.0 and H == 0.0, and
        // its top-1 is the single candidate it was built with.
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

        // The two ambiguous reads point at different transcripts: each sees its
        // own winner FIRST, and the winner is decided by the alignment term
        // (ΔAS = 30 for AMB_TX0, 40 for AMB_TX1 -- strictly positive, so never
        // a tie, and strictly inside `prune_max_delta_as = 50`, so the
        // runner-up is never pruned away).
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

        // The posteriors must be genuinely NON-degenerate. These are the two fields
        // the whole cross-run comparison rests on, and on a one-element weight vector
        // both are pinned by arithmetic rather than measured: `posterior_entropy`
        // returns 0.0 for a one-element vector and the lone weight is 1.0.
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
        // Deliberately UNEQUAL ΔAS (30 vs 40) must show up as unequal
        // posteriors. Without this, two identical posteriors would satisfy the
        // assertions above while leaving the join untestable: swapping the two
        // rows would then be invisible.
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

        // `n_candidates_before_cutoff` is the PRE-prune fan-out, matching the
        // profile column of the same name -- not the post-prune weights.len().
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

    /// The out-param is optional. When it is not supplied the solve must still
    /// produce a byte-identical profile -- this path adds no observable state
    /// to the fit, and a change to the E-step that leaked in through the
    /// bookkeeping would show up here as a different profile.
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

        // Pre-seeded with a sentinel so the two things the out-param's name
        // promises are both observable: `Some` REPLACES (the sentinel is gone
        // afterwards) and `None` does not touch the caller's Vec at all (the
        // rows from the `Some` run survive the `None` run intact).
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

        // BYTE-IDENTICAL, not merely equal-length. The previous assertion was
        // `a.len() == b.len()`, which a profile differing only in VALUES also
        // satisfies -- parquet encodes a weight of 0.765 and one of 0.910 in
        // the same eight bytes. The docstring promised byte-identity; this is
        // the assertion that delivers it.
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

    /// `top_tx_id == Some(..)` must IMPLY that the read has a profile row.
    ///
    /// It now holds for BOTH non-ghost paths. Previously the all-culled path
    /// violated it: `read_assign[mol_idx]` was written in the ambiguous branch,
    /// which runs BEFORE the emit loop's `min_weight` filter, so a read whose
    /// every candidate fell below the cutoff carried a real `top_tx_id` while
    /// emitting ZERO profile rows -- a numerator entry with no denominator, in a
    /// table whose whole purpose is a cross-run join. That read is now retained
    /// and emitted under its argmax instead, so the two tables agree.
    ///
    /// The assertion is the cross-run join's own invariant, stated on the data
    /// rather than argued from the code. The converse is still not claimed, and the
    /// ghost shape is the reason -- see
    /// `test_ghost_read_is_recorded_as_unassignable_and_absent_from_the_profile`.
    ///
    /// Mutation target: removing the `&& !(argmax_only && Some(tx_id) == top)`
    /// exemption from the emit loop's `min_weight` test makes this red.
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

        // The read WAS ambiguous and WAS E-stepped — it is not absent from the
        // graph, it was culled at the output stage. Without this the profile
        // assertions below would also hold for a graph that never had it.
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

        // The actual invariant, asserted on the data rather than argued from
        // the code: every read in the assignments table has a profile row.
        for a in &assigns {
            assert!(
                rows.iter().any(|(r, ..)| r == &a.read_id),
                "assignment for {} carries a top_tx_id but has no profile row; \
                 the implication `top_tx_id == Some(..) => profile row` is what \
                 keeps the cross-run join denominator auditable: {rows:?}",
                a.read_id
            );
        }
        // The forced row and the assignment must name the SAME transcript.
        // The argmax is folded once and shared by both paths precisely so
        // these cannot drift; this is the assertion that would catch it if
        // they ever did.
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

    /// A GHOST read -- both candidates poly(A)-incompetent, so the E-step returns
    /// an EMPTY weight vector -- has no competent candidate, so the profile
    /// legitimately has no row for it. It IS still recorded in the assignments
    /// table, with `top_tx_id = None`.
    ///
    /// That is the distinction a cross-run comparison needs: "the solver looked at
    /// this read and could not assign it" versus "this read was not in this run at
    /// all". Dropping the row would make the two identical, and `top_tx_id: None`
    /// is the only place the difference is visible.
    ///
    /// It is the `None` arm of the invariant `top_tx_id == Some(..) IMPLIES that
    /// read has a profile row` -- the converse is deliberately NOT claimed, and
    /// this test pins both halves of that.
    #[test]
    fn test_ghost_read_is_recorded_as_unassignable_and_absent_from_the_profile() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("ghost_emission.parquet");
        let profile = scratch("ghost_profile.parquet");
        write_test_emission_with_competence(
            &emission,
            &[
                ("tx0", "gene0", log_pal as f64, 0.04, true),
                // Both of GHOST's candidates are incompetent: no `ll` is ever
                // pushed for either, so `max_ll == -inf` and `weights` is
                // empty.
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

        // HALF ONE: no profile row. There is no competent candidate, so there
        // is nothing to emit — the ghost is absent, not emitted with weight 0.
        assert_eq!(
            rows.len(),
            1,
            "the ghost read has no competent candidate, so it emits no profile \
             row and LIVE is all that is left: {rows:?}"
        );
        assert_eq!(rows[0].0, "LIVE");

        // HALF TWO: the assignment table still records it, as unassignable.
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
        // Pre-prune fan-out, matching the profile's column of the same name:
        // 2 even though nothing was scored.
        assert_eq!(ghost.n_candidates_before_cutoff, 2);
        assert_eq!(
            stats.n_dropped_no_competent_candidates, 1,
            "GHOST must also be counted as a ghost drop in the stats"
        );

        // The invariant itself, asserted on the data: every `Some` has a
        // profile row. The ghost is the case that makes the converse FALSE.
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

    /// The E-step's competence filter is the ONLY thing keeping an incompetent
    /// transcript out of a read's `top_tx_id`. Neither the argmax fold nor the
    /// `params` lookup that turns the winner into a `transcript_id` re-checks
    /// `is_competent`, so the named transcript's profile row exists only because
    /// the E-step pushed `-inf` instead of a real log-likelihood for it.
    ///
    /// The fixture makes the failure observable rather than hypothetical. `BAD` is
    /// present in the graph and has an emission row, so nothing about it is
    /// missing; it is simply marked not poly(A)-competent. Its `mu` is the read's
    /// EXACT observed length, so if the filter stops firing it wins the argmax
    /// outright. The output stage's own `!p.is_competent` check then keeps `BAD`
    /// out of the profile, so the read ends up naming, as its top-1, a transcript
    /// it has no profile row for.
    ///
    /// This is the per-TRANSCRIPT half of the invariant the per-read checks above
    /// cover only from the outside: those ask "does the named read have some
    /// profile row", which stays satisfied here because `GOOD` supplies one.
    #[test]
    fn test_incompetent_but_present_transcript_is_never_a_reads_top1() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("incompetent_emission.parquet");
        let profile = scratch("incompetent_profile.parquet");
        write_test_emission_with_competence(
            &emission,
            &[
                // Competent, but its mu is a 1.5-bit miss: `GOOD` wins this
                // fixture ONLY because `BAD` is filtered out.
                ("GOOD", "gene0", log_pal as f64 - 1.5, 0.04, true),
                // Present in the graph, has a row, marked not competent, and an
                // exact fit for the read.
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

    /// Round-trip a `NULL` `top_tx_id`. The output stage reaches that NULL on
    /// every ghost read, so this is not a corner case -- but the writer itself was
    /// never called by a test, so a change to `append_option` or to the column's
    /// nullability would have gone unnoticed.
    ///
    /// The three things a NULL can be confused with, all checked: an empty string
    /// (a transcript that does not exist), the index-0 transcript (a real
    /// transcript the read was never assigned to), and a dropped row.
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
        // The null is in the `top_tx_id` column ONLY. A writer that lost the
        // null mask would still round-trip these two floats, so they are what
        // distinguishes "wrote a null" from "dropped the row and shifted the
        // rest up".
        assert!((seen[1].3 - 0.5).abs() < 1e-12, "got {}", seen[1].3);
        assert!((seen[1].4 - 0.69).abs() < 1e-12, "got {}", seen[1].4);
    }

    /// One read `r1` whose candidate set is exactly `cands`, paired with a
    /// `params` vector where candidate `i` carries `cands[i]`.
    ///
    /// Each candidate gets its OWN transcript slot (`tx0`, `tx1`, ...) so that
    /// `candidate_pool[i] == i` and a test cannot accidentally give two
    /// candidates the same fit by aliasing one entry. `None` means "this
    /// candidate has no emission row", which is how the ghost case is expressed.
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
            // All EQUAL: the helper reads no alignment evidence, so leaving AS
            // constant keeps the fixture about (mu, sigma2) alone.
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

    /// Direct unit test of the poly(A)-degradation predicate. The number it feeds,
    /// `n_polya_zero_power / n_ambiguous`, was computed by this function with
    /// nothing testing it: replacing the body with `false` left the whole suite
    /// green. It is no longer printed by the CLI because it was being read as a
    /// validity metric, but the predicate is still the tripwire that catches
    /// parameters collapsing to one identical float.
    ///
    /// The cases that matter are the ones a plausible re-implementation gets
    /// wrong. A "do the mus match?" comparison alone would pass the headline
    /// fixture -- the fallback shares BOTH components -- while firing `true` on
    /// every read whose transcripts have a common `mu` and different widths, which
    /// is a *false* zero-power report on a read poly(A) does discriminate.
    /// `sigma2` is asserted separately below for exactly that reason.
    #[test]
    fn test_polya_emission_count_and_identity_cases() {
        const MU: f64 = 7.0;
        const S2: f64 = 0.5;

        // 1. Every candidate shares one identical (mu, sigma2) — the fallback
        //    cascade handing the same fit to several transcripts. TRUE.
        {
            let (g, p) = zero_power_probe(&[Some((MU, S2)), Some((MU, S2)), Some((MU, S2))]);
            assert!(
                polya_emission_count_and_identity(&g, &g.molecules[0], &p).1,
                "candidates sharing (mu, sigma2) cancel the poly(A) term exactly"
            );
        }

        // 1b. A SINGLE candidate is trivially "all candidates agree" and is also
        //     `true` here. This is the VACUOUS case the whole change exists to make
        //     visible: nothing was compared, so the flag is true for want of a
        //     counterexample. The COUNT is what distinguishes it.
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

        // 2. Candidates differ in mu. FALSE.
        {
            let (g, p) = zero_power_probe(&[Some((MU, S2)), Some((MU + 0.5, S2))]);
            assert!(
                !polya_emission_count_and_identity(&g, &g.molecules[0], &p).1,
                "a mu difference is a non-constant log-ratio — poly(A) has say"
            );
        }

        // 3. Candidates share mu but differ in sigma2. FALSE. This is the case
        //    a mu-only comparison gets backwards: `sigma2` enters the Gaussian
        //    log-density just as directly as `mu` does, so identical mus with
        //    different widths still give a non-constant odds ratio.
        {
            let (g, p) = zero_power_probe(&[Some((MU, S2)), Some((MU, S2 + 0.4))]);
            assert!(
                !polya_emission_count_and_identity(&g, &g.molecules[0], &p).1,
                "sigma2 alone decides the log-ratio; sharing mu is not enough"
            );
        }

        // 3b. A disagreement in the THIRD candidate must not be masked by the first
        //     two agreeing -- the comparison is against the first candidate, not
        //     against the immediately preceding one.
        //
        //     THE COUNT IS 3, NOT 2, AND THAT IS THE POINT. An earlier version of
        //     this function returned `false` on the first disagreement, so a loop that
        //     counted as it went would report 2 here. That undercount lands only on
        //     DISAGREEING reads -- the ones where poly(A) did something -- which
        //     would bias `emission coverage` downward while leaving the zero-power
        //     headline untouched.
        {
            let (g, p) = zero_power_probe(&[Some((MU, S2)), Some((MU, S2)), Some((MU, S2 + 0.4))]);
            assert_eq!(
                polya_emission_count_and_identity(&g, &g.molecules[0], &p),
                (3, false),
                "the count must cover the WHOLE candidate range even after a \
                 disagreement; 2 here means the early return is back"
            );
        }

        // 4. Empty candidate list. There is no candidate to compare against, so the
        //    code returns `false`: claiming zero power vacuously would inflate the
        //    headline fraction on reads the model never even looked at.
        {
            let (g, p) = zero_power_probe(&[]);
            assert_eq!(
                polya_emission_count_and_identity(&g, &g.molecules[0], &p),
                (0, false),
                "an empty candidate set must NOT claim zero discriminative power"
            );
        }

        // 4b. A candidate with NO emission row is SKIPPED, not counted as a
        //     mismatch: a ghost transcript has no (mu, sigma2) to disagree with. The
        //     remaining real candidate is what the read is judged on, and it is the
        //     same ghost the read is already charged for in
        //     `n_dropped_no_competent_candidates`.
        {
            let (g, p) = zero_power_probe(&[Some((MU, S2)), None, Some((MU, S2))]);
            assert_eq!(
                polya_emission_count_and_identity(&g, &g.molecules[0], &p),
                (2, true),
                "a missing emission row is a skip, not a (mu, sigma2) mismatch — \
                 and it is not counted either"
            );
        }
        // ...and a read whose ONLY candidate is a ghost is the "nothing to
        // compare" case of clause 4, reached through a non-empty candidate
        // range: false, not true, and a count of ZERO.
        {
            let (g, p) = zero_power_probe(&[None, None]);
            assert_eq!(
                polya_emission_count_and_identity(&g, &g.molecules[0], &p),
                (0, false),
                "no candidate with an emission row means nothing was measured"
            );
        }

        // 5. FLOATING-POINT EQUALITY. The comparison is EXACT `==`, and these two
        //    cases pin that: the fallback hands out bit-identical COPIES of one fit,
        //    so anything looser than `==` can only add false positives. Both
        //    sub-cases below therefore return FALSE -- one ULP is already
        //    "different" -- and that is asserted rather than assumed.
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
        // Control for 5: a value copied from a single binding is bit-identical
        // and IS zero power, so 5 is pinning the tolerance and not a blanket
        // "any second candidate means false".
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

    /// End-to-end: the per-read predicate's verdict must actually reach
    /// `TelEmStats.n_polya_zero_power`, on a graph with a known mix.
    ///
    /// The unit test above pins the predicate; this pins the four links around it
    /// -- the counter is incremented from the AMBIGUOUS loop, it reads the LAST
    /// E-STEP's `last_estep_params`, it lands in the struct, and unambiguous reads
    /// are excluded from it.
    ///
    /// It previously read the POST-EM `params`. That was the stale-μ defect: on
    /// the `max_iter`-exhaustion path those are one M-step past the state that
    /// produced the posterior. This fixture still passes either way -- `poly_a_weight
    /// = 0` freezes `mu`/`sigma2`, and the predicate reads ONLY `mu`/`sigma2`, so
    /// pre- and post-M-step are identical for the fields it looks at. That is
    /// exactly why this test could never have caught the defect, and why the
    /// positive arm in
    /// `test_zero_power_telemetry_reads_the_e_step_state_not_the_post_m_step`
    /// exists: it runs at `poly_a_weight > 0`, where the M-step DOES move the
    /// fields the predicate reads.
    #[test]
    fn test_n_polya_zero_power_reaches_stats_with_the_known_mix() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("zero_power_emission.parquet");
        let profile = scratch("zero_power_profile.parquet");
        // tx0..tx3 all carry the SHARED fit (the L2 / L3 fallback shape);
        // tx4 splits on mu, tx5 splits on sigma2. Gene ids are irrelevant here
        // (the M-step is frozen) and are kept distinct so a reader does not
        // infer anything from them.
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

        // Molecule-major candidate pool, as the graph layout invariant requires.
        //   UNAMBIGUOUS  [0]      -> tx0            (1 candidate: never counted)
        //   ZERO_2WAY    [1..3)   -> tx0, tx1       (identical -> counted)
        //   ZERO_3WAY    [3..6)   -> tx1, tx2, tx3  (identical -> counted)
        //   MU_SPLIT     [6..8)   -> tx0, tx4       (mu differs -> not counted)
        //   SIGMA_SPLIT  [8..10)  -> tx1, tx5       (sigma2 differs -> not counted)
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
            // ALL EQUAL: keeps the ΔAS prune from deleting a candidate, so
            // "not counted" can only be the zero-power verdict and never a
            // read that fell off some other path.
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

        // Denominator first: the fixture must be a MIX, or the count below
        // would be satisfied by a graph in which every read agrees.
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

    /// THE VACUITY SPLIT, end to end, on a model that is NOT 100% covered.
    ///
    /// The unit test above pins the predicate and the test above it pins that the
    /// counter reaches `TelEmStats`. Neither can see the difference the whole
    /// change is about, because both fixtures give every candidate an emission
    /// row. This one does not: `tx6` is deliberately absent from the emission
    /// parquet, so reads can have fewer present candidates than they have
    /// candidates -- which is the condition that makes `zero-power_all` report a
    /// cancellation that never happened.
    ///
    /// The fixture's whole job is one number pair:
    ///
    /// ```text
    ///   n_polya_zero_power          == 2   (ZERO_2WAY and SINGLE_THEN_GHOST)
    ///   n_polya_zero_power_resolved == 1   (ZERO_2WAY only)
    /// ```
    ///
    /// `SINGLE_THEN_GHOST` is the read the split exists for: two candidates, one
    /// of them has no emission row, so nothing was ever compared, and
    /// `zero-power_all` counts it anyway because that is what it has always done.
    ///
    /// What this pins that the census unit test cannot: that the CALL SITE routes
    /// each cause to the field the CLI reads, and that a read which loses only
    /// SOME of its candidates increments NOTHING. The second half is the
    /// load-bearing half -- a solver that counted every read with any dead
    /// candidate would produce plausible-looking numbers for every cause and
    /// still be wrong, because the number the padding experiment turns on is
    /// "reads that padding would rescue", not "reads with something wrong".
    ///
    /// `max_iter = 1` on purpose. `alpha` only reaches 0 through the M-step, and
    /// a transcript can only win no mass if it was already dead, so
    /// `n_ghost_read_all_alpha_zero` is NOT reachable through the solver at any
    /// iteration count.
    ///
    /// The `padded-padded ambiguity` split, and the partition it claims.
    ///
    /// DESIGN NOTE, because it is the whole point: this does NOT try to hand-craft
    /// a read that is degenerate in one bucket and not the other. Doing that would
    /// mean tuning alphas until one posterior crossed H = 0.70, and a fixture
    /// tuned that finely is a fixture whose assertion can be made to pass for the
    /// wrong reason.
    ///
    /// Instead it runs the SAME fixture twice, changing ONE column -- the
    /// `n_cohort_flnc` marker -- and requires the split to follow it:
    ///   run A: every row has n_cohort_flnc == 0  => every degenerate read is
    ///          padded-padded
    ///   run B: every row has n_cohort_flnc == 10 => every degenerate read is
    ///          real competition
    ///
    /// Each run also asserts the partition holds, which is the invariant the CLI
    /// comment relies on. A classifier that ignored `is_imputed`, or that put
    /// everything in one bucket, fails one of the two runs. A classifier that
    /// reported 0 for both would pass the partition check alone -- the `> 0`
    /// assertions are what stop that.
    #[test]

    fn test_degenerate_tie_split_partitions_the_degenerate_reads() {
        let log_pal = (100.0_f32).log2();

        // Four candidates, identical mu/sigma2 and identical AS, so the posterior is
        // exactly uniform: H = ln 4 = 1.386 nats, comfortably above the 0.70
        // degenerate bound. Four rather than three so the fixture is not sitting on
        // the boundary of its own threshold.
        //
        // A per-call counter in every fixture path. `scratch()` already keys on the
        // process id, which is constant for all tests in one binary -- so two tests
        // that pick the same fixture name would share a file and truncate each
        // other's parquet while running in parallel.
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

        // Cleanup AFTER both runs, not inside the closure: a solve that failed would
        // otherwise have already deleted the fixture it was reading.
        //
        // The guard is not optional: the fixture must actually produce a degenerate
        // read, or every assertion below is a statement about zero.
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
        // tx4 and tx5 are INTENTIONALLY ABSENT: transcripts the emission model
        // never fitted, the shape of the 37,081 missing rows.
        write_test_emission_with_competence(
            &emission,
            &[
                ("tx0", "gene0", 7.0, 0.04, true),
                ("tx1", "gene1", 7.0, 0.04, true),
                ("tx3", "gene3", 7.0, 0.04, false),
                ("tx6", "gene6", 7.0, 0.04, false),
            ],
        );

        //   UN_MISSING   [0]     -> tx4            absent from the model
        //   UN_INCOMP    [1]     -> tx3            present, incompetent
        //   UN_OK        [2]     -> tx0            present, competent
        //   G_ALL_MISS   [3..5)  -> tx4, tx5       both absent
        //   G_ALL_INC    [5..7)  -> tx3, tx6       both incompetent
        //   G_MIXED      [7..9)  -> tx4, tx3       one absent, one incompetent
        //   G_LIVE       [9..11) -> tx0, tx1       both fine
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
            // Equal AS everywhere, so `delta_as == 0` and the prune cannot fire.
            // The census's prune bucket is exercised directly in
            // `test_candidate_availability_keeps_four_death_causes_apart`; here
            // it would be a fifth cause with nothing to attribute it to.
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

        // Unambiguous population, denominator `n_unambiguous`.
        assert_eq!(
            stats.n_unambiguous_dropped_missing_emission, 1,
            "only UN_MISSING's candidate is absent from the model"
        );
        assert_eq!(
            stats.n_unambiguous_dropped_incompetent, 1,
            "only UN_INCOMP's candidate carries is_polyA_competent == false"
        );

        // Ghost population, denominator `n_dropped_no_competent_candidates`.
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

        // The three cause counters are a PARTIAL decomposition, and the CLI
        // prints the remainder. G_MIXED is the remainder: one candidate absent
        // and one incompetent, so padding alone does not revive it.
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

        // The same fact from the OUTPUT side: a read that loses some but not all
        // of its candidates still gets its survivors' rows, and an all-dead read
        // gets none.
        // Compared as a multiset, not a sequence: the profile emits the
        // unambiguous rows before the contested ones and that ordering is not
        // what this test is about.
        let mut emitted: Vec<&str> = rows.iter().map(|(r, _, _)| r.as_str()).collect();
        emitted.sort_unstable();
        assert_eq!(
            emitted,
            vec!["G_LIVE", "G_LIVE", "UN_OK"],
            "only the reads with a live candidate may appear in the profile: {rows:?}"
        );
    }

    /// not happening.
    #[test]
    fn test_zero_power_resolved_excludes_reads_with_fewer_than_two_emission_candidates() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("zero_power_split_emission.parquet");
        let profile = scratch("zero_power_split_profile.parquet");
        // tx6 is INTENTIONALLY ABSENT. It is a transcript in the graph that the
        // emission model never fitted -- the shape of the 37,081 rows missing
        // from the HC2 model, reduced to one transcript.
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

        //   UNAMBIGUOUS      [0]     -> tx0            1 present
        //   GHOST_ONLY       [1]     -> tx6            0 present
        //   ZERO_2WAY        [2..4)  -> tx0, tx1       2 present, IDENTICAL
        //   SINGLE_THEN_GHOST[4..6)  -> tx0, tx6       1 present  <- vacuous
        //   MU_SPLIT         [6..8)  -> tx0, tx4       2 present, mu differs
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

        // Emission coverage, over ALL reads with >=1 candidate. Four of the
        // five have at least one present candidate; GHOST_ONLY is the only one
        // whose every candidate is missing from the model.
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

    /// The stale-μ defect, as a test.
    ///
    /// Two arms, ONE shared emission parquet, ONE shared graph, differing ONLY in
    /// `poly_a_weight`. Both arms' E-steps therefore consume the IDENTICAL
    /// `(mu, sigma2)` -- the loaded values, before any M-step has run. So the
    /// zero-power verdict is a property of the data, and the two arms MUST report
    /// the same count. The M-step cannot change that: it runs after the E-step
    /// that produced the posterior, and the telemetry is about the posterior.
    ///
    /// Pre-fix this fails, and with a SIGN rather than merely "not equal": reading
    /// post-M-step `params` lets the M-step's τ₀=2 shrinkage pull the tied fits
    /// apart, so the ON arm counts STRICTLY FEWER than the OFF arm. An assertion
    /// of the form `assert_ne!` would also have passed on a fixture that broke
    /// for an unrelated reason.
    ///
    /// `max_iter = 1` is REQUIRED: it is the `max_iter`-exhaustion exit path, the
    /// one where the M-step runs last and leaves `params` a full M-step ahead of
    /// the posterior. On a converged run the loop `break`s BEFORE the M-step and
    /// this defect is invisible.
    #[test]
    fn test_zero_power_telemetry_reads_the_e_step_state_not_the_post_m_step() {
        let log_pal = (100.0_f32).log2();
        let emission = scratch("estep_state_emission.parquet");
        // tx0 and tx1 share ONE loaded fit — the L3-fallback shape, where two
        // underpowered transcripts in a gene both take the same cohort value.
        // That shared fit is what makes the reads below zero-power.
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
        // ASYMMETRIC on purpose. The alignment term `ln A_rk = -ΔAS/λ` gives tx0
        // strictly more posterior mass than tx1 on every read, so their accumulated S1
        // DIFFER and one M-step pulls their mu apart. A fixture with equal AS gives
        // both candidates identical accumulators, the shrinkage moves them to the
        // identical place, and the M-step cannot separate the tie.
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

        // The SECOND consumer of the snapshot, pinned separately.
        //
        // The assertion above went red when the TELEMETRY read `params`, but reverting
        // the `output_emission_params` half of the same fix turned ZERO tests red --
        // the pre-existing tests on that out-param all straddle the difference, so
        // the suite could not tell the two states apart. That left the "one snapshot,
        // two consumers ... by construction" comment an unbacked claim. This is the
        // mutation that makes it true.
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
