//! Module 3: TEL-EM Solver
//!
//! Executes the bifurcated Terminal-aware Expectation Maximization (TEL-EM)
//! algorithm on the flat evidence graph produced by Module 2.
//!
//! Bifurcated inference (spec v0.1.7 F2.1):
//! - Unambiguous reads (candidate_count == 1) get w_rk ≡ 1.0 (deterministic).
//! - Ambiguous reads (candidate_count > 1) enter the EM loop, which carries the
//!   joint `w_rk ∝ π_k · A_rk · F_rk · N(y_r; μ_k, σ_k²)` responsibility of
//!   spec v0.2.7 F-EM-4 plus the 5'-anchoring term; `ln A_rk = (AS_rk − max_j
//!   AS_rj) / λ` and `ln_5p_k = β if g_k ≤ window` come from Module 2's `AS:i`
//!   and 5'-gap pools respectively, and the `ΔAS > 50` prune drops the
//!   hopeless candidates. See `solve_and_write_profile` for the parameter
//!   provenance, including the rejected alternatives.
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
///
/// The 5'-anchoring term and the `ΔAS` prune are `FROZEN SPEC`, validated
/// empirically in the Python prototype against HC3. Provenance, in the form
/// that must not be re-derived:
///
///   `g_k`  = `s_k − 1`, `s_k` the 1-BASED start of the read's BEST alignment
///            to candidate k, in candidate k's OWN cDNA coordinates. Never a
///            genomic span; the BAM reference IS the cDNA.
///   `ln_5p_k` = `five_prime_weight` if `g_k <= five_prime_window` else 0.0.
///   prune: drop candidates with `ΔAS_k > prune_max_delta_as`, where `ΔAS` is
///            measured against the read's best AS.
///   `w_k ∝ π_k · exp(ln_a_k + ln_5p_k)`.
///
/// REJECTED — `o_k = T_k − L_k` (transcript minus read length). Its
///   directionality toward the lower-frequency isoform measured 46.52%, i.e.
///   chance, against 58.34% for `g_k = s_k − 1`. Reason: `o_k` is a length
///   comparison, symmetric across candidates, so it CANCELS in the likelihood
///   ratio. Any future "simplification" to a length difference is a known-wrong
///   quantity, not a different bug.
/// REJECTED — `five_prime_weight = 7.9`. With `prune_max_delta_as = 50` and
///   `aln_weight_lambda = 20` the alignment term spans only `[−2.5, 0]` nats, so
///   7.9 is an unconditional hard override that no read can argue with. 1.8
///   sits inside that range, so the 5' term COMPETES with the alignment term
///   rather than replacing it.
/// REJECTED — `aln_weight_lambda = 1000`. It made every evidence term ≤ 0.1
///   nat, i.e. it silently disabled the evidence the EM is supposed to weigh.
///   The default is now 20.0. This function has NO default of its own: λ, the
///   5' weight, the window and the prune threshold are all arguments, so no
///   test here reads a CLI default and no library embedder can pick one up by
///   accident.
// `aln_weight_lambda` pushed the signature to 8 arguments; collapsing the EM
// hyperparameters into a config struct is a separate refactor, out of scope.
// The 5' term and the prune bring it to 11 adjacent positional scalars, so the
// argument-shuffling hazard that note warned about is now realised and the
// config-struct refactor is overdue rather than merely optional.
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
    five_prime_window: u32,
    prune_max_delta_as: i32,
    min_weight: f64,
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
    // NaN-safe: `(0.0..1.0).contains(&NaN)` is false, so the range test alone
    // rejects NaN; `is_finite` is kept explicit because +inf would also fail
    // the range test but for a different reason, and the two failure modes
    // deserve different error text. 0.0 is legal and means NO truncation.
    if !min_weight.is_finite() || !(0.0..1.0).contains(&min_weight) {
        anyhow::bail!(
            "min_weight must be finite and in [0, 1) (got {}); it is the output-stage \
             posterior truncation threshold, and 1.0 would retain only a read's \
             argmax, silently turning the profile into a hard-assignment table",
            min_weight
        );
    }
    // 5'-anchoring weight. 0.0 is a LEGAL value and is the documented off
    // switch (`--five-prime-weight 0`): it makes `ln_5p ≡ 0`, i.e. `exp(0) = 1`,
    // so the term leaves the softmax EXACTLY rather than approximately. Only a
    // negative weight or NaN are rejected. A negative weight would reward
    // reads that start far from the transcript's 5' end, which is not a
    // configuration the frozen spec admits; NaN would propagate into `ll` and
    // out as NaN posterior weights.
    // `is_finite()` rather than `is_nan()`: +inf would pass a NaN check and
    // then poison the log-sum-exp with `inf - inf`, emitting NaN weights --
    // the exact outcome this guard exists to prevent.
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
    // read's own argmax (whose ΔAS against itself is 0), emptying `raw` and
    // turning every ambiguous read into a ghost drop. `i32::MAX` is a legal
    // "prune off" value.
    if prune_max_delta_as < 0 {
        anyhow::bail!(
            "prune_max_delta_as must be >= 0 (got {}); it drops candidates whose \
             alignment score is more than this far below the read's best, and a \
             negative threshold would drop the argmax too, emptying every read",
            prune_max_delta_as
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
    // the emission parquet is keyed by whatever `build-emission` was handed. In
    // AIDRS Mode those are DIFFERENT SPACES and nothing in the pipeline
    // reconciles them on its own: the BAM uses `AIDRS_tx_N`, the AIDRS
    // polyA/assessment tables use `ENST`/`ENSG_NovelTr`. Measured on HC3:
    // 70,511 BAM references against 70,301 emission rows, intersection ZERO.
    //
    // The failure is silent and total. Every `params[i]` above is `None`, so
    // every read takes the ghost-drop path, and the run still exits 0 behind a
    // normal-looking telemetry block — measured on the GNG11 subset: 1637
    // molecules extracted, EM converged in 2 iterations, 1637 dropped
    // (502 "incompetent" + 1135 "no_competent"), 0 rows emitted, exit 0. The
    // only tells are the drop counters equalling the input totals and an
    // all-zero entropy histogram, neither of which a reader checks.
    //
    // A zero overlap is never a legitimate configuration, so refuse it instead
    // of writing an empty profile. Partial overlap is NOT checked here and is
    // legitimate: transcripts absent from the emission (novel, or polyA-less) are
    // dropped one by one by design.
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
             silently dropped and an EMPTY profile written. In AIDRS Mode the BAM is \
             keyed `AIDRS_tx_N` and the AIDRS polyA/assessment tables are keyed \
             `ENST`/`ENSG_NovelTr`; pass `--tx-id-mapping <bam_id<TAB>original_id tsv>` \
             to bridge them, or use Universal Mode (`build-anchors`), whose ids come \
             from the BAM itself."
        );
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

                // Compute raw log responsibilities (with alpha + Gaussian +
                // A_rk + the 5' term).
                let mut raw: Vec<(usize, f64)> = Vec::with_capacity(mol.candidate_count as usize);
                for c in 0..mol.candidate_count {
                    let idx = (mol.candidate_offset + c as u32) as usize;
                    let tx_id = graph.candidate_pool[idx] as usize;

                    // ΔAS is a MAGNITUDE: the read's best score minus this
                    // candidate's, so ΔAS >= 0. SIGN MATTERS, and the two uses
                    // below disagree on purpose: the prune thresholds the
                    // magnitude, the log term negates it. A prune written on
                    // the `ln_a` numerator compares a non-positive number
                    // against 50 and NEVER fires — which is exactly the bug
                    // `test_prune_boundary_is_strictly_greater_than` exists to
                    // catch, since a large-ΔAS fixture is then silently passing
                    // on the `min_weight` cutoff instead.
                    //
                    // `saturating_sub`, not `-`: both operands are `i32` and
                    // this sits OUTSIDE the `if let Some(params)`, so it runs
                    // for candidates that have no emission row at all. A plain
                    // subtraction panics under `cargo test` (debug overflow
                    // checks) and WRAPS under `--release`, and the wrap is the
                    // dangerous one: a wrapped negative ΔAS satisfies
                    // `delta_as > prune_max_delta_as` as FALSE, so the one
                    // candidate that should be pruned hardest is the one that
                    // survives. Saturation gives `i32::MAX`, which prunes —
                    // the direction that is safe under both profiles.
                    let delta_as = max_as.saturating_sub(graph.candidate_as[idx]);

                    // ΔAS prune (frozen spec): drop candidates whose alignment
                    // score is more than `prune_max_delta_as` below the read's
                    // best.
                    //
                    // `-inf` rather than a bare `continue`, matching the
                    // competence mask below. Note that in THIS E-step the two
                    // are observationally IDENTICAL, and it is worth being
                    // precise about why rather than asserting a safety
                    // property that does not hold: `weights` is built with
                    // `filter(ll > -inf)`, so a `-inf` entry leaves the
                    // population exactly as an absent one would, and
                    // `sum_exp` contributes `exp(-inf) = 0` either way. The
                    // log-sum-exp, `log_z`, the emitted weights and the output
                    // stage's `retained_sum` denominator are all bit-identical
                    // under both spellings. (They would DIVERGE if that filter
                    // were ever removed — a `-inf` entry would then be summed
                    // as 0 by `sum_exp` but never reach `weights`, while an
                    // absent one would not be counted at all; the direction of
                    // the difference is the reason the filter is load-bearing,
                    // not the choice of `-inf` over `continue`.)
                    //
                    // The `-inf` form is kept for the weaker but real reasons:
                    // it keeps one `raw` entry per candidate, so `raw` stays a
                    // faithful per-candidate record and its `with_capacity`
                    // holds, and it routes the prune through the same code path
                    // as every other "this candidate is dead" decision.
                    //
                    // This is placed OUTSIDE the `if let Some(params)` so the
                    // prune is a property of the alignment evidence alone, and
                    // holds for a candidate that has no emission row at all.
                    // It cannot empty a read: the argmax candidate has
                    // `delta_as == 0` against itself and `prune_max_delta_as`
                    // is validated `>= 0` above, so at least one entry always
                    // escapes `-inf`.
                    if delta_as > prune_max_delta_as {
                        raw.push((tx_id, f64::NEG_INFINITY));
                        continue;
                    }

                    // ln A_rk = (AS_rk − max_j AS_rj) / λ (spec v0.2.7 F-EM-4)
                    // = −ΔAS / λ. Added to `ll` *before* the `max_ll`
                    // log-sum-exp shift, so any λ is numerically safe — no
                    // clamping needed.
                    let ln_a = -(delta_as as f64) / aln_weight_lambda;

                    // ln_5p_k = β if g_k ≤ window else 0 (frozen spec). `g_k` is
                    // the observed 5' gap of the read's best alignment to THIS
                    // candidate, in this candidate's own cDNA coordinates, and
                    // is already strand-corrected by Module 2 — there is no
                    // strand flag to consult here and none may be inferred.
                    //
                    // It is a PRIOR, not an EM parameter: `g_k` is fixed per
                    // (read, candidate) exactly like `AS`, so it belongs in no
                    // accumulator and no M-step. Adding it here, before the
                    // `max_ll` shift, is what keeps the log-sum-exp safe; a
                    // bounded [0, 1.8] bump on an un-shifted `exp` would not
                    // overflow, but applying it to the weight AFTER `log_z` is
                    // formed would break the invariant the λ-range safety
                    // argument rests on.
                    //
                    // NOTE the deliberate asymmetry with `ln_a`: `ln_a` is
                    // continuous in the evidence and is the term that ORDERS
                    // candidates, while `ln_5p` is a step function of a single
                    // coordinate. It can only ever reorder, never delete —
                    // deletion is the prune's job.
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
                            let ll = p.alpha.ln()
                                - (p.sigma2.sqrt()).ln()
                                - ((y - p.mu).powi(2) / (2.0 * p.sigma2))
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
        // E-step's `ΔAS` prune and BEFORE the `min_weight` truncation in the
        // ambiguous branch below can drop rows.
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

    // Ambiguous rows: per (read × candidate) with posterior ≥ `min_weight`.
    // `min_weight` is a CALLER-SUPPLIED parameter, not a constant, because it
    // silently changes what an A/B comparison is measuring. At the shipped 0.01
    // a read whose losing candidate sits below 0.01 loses that row and has its
    // survivors renormalised to 1.0 — so an arm that shifts mass onto a
    // candidate emits MORE rows than the arm that does not, and the two arms
    // are no longer scored on the same support. Measured on GNG11: 2726 rows
    // with the 5' term on vs 1816 with it off, same reads, same weight_sum.
    // Any ablation whose arms emit different row counts must be re-run at
    // `min_weight = 0` (where `retained_sum` is the full 1.0 and the term is
    // exactly off) before its delta is quoted.
    for (i, weights) in final_weights.into_iter().enumerate() {
        let mol_idx = ambiguous_indices[i];
        let mol = &graph.molecules[mol_idx];
        let pal_i32 = poly_a_length_from_log2(mol.log_pal);
        // Observational columns, per READ — computed once, before the E-step's
        // `ΔAS` prune and before the `min_weight` truncation below can drop
        // rows, and repeated verbatim on every emitted row of this read.
        // `candidate_count` is the ORIGINAL count off the graph (what Module 2
        // produced), NOT the number of rows this loop emits — and not the
        // number the E-step normalised over, which excludes the pruned.
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
        //
        // The `ΔAS` prune does NOT break this coupling, and it is worth being
        // explicit about why: a pruned candidate is pushed as `-inf`, and
        // `weights` is built with `filter(ll > -inf)`, so it is already gone
        // from the population this denominator sums over — its mass was
        // renormalised away by the log-sum-exp, not left stranded here.
        // (Verified: replacing the `-inf` push with a bare `continue` leaves
        // every emitted weight bit-identical, because the same filter removes
        // it. What the filter protects is the case of a candidate pushed as
        // `-inf` that is then NOT filtered — not the choice of `-inf`.)
        let retained_sum: f64 = weights
            .iter()
            .filter(|&(_, w)| *w >= min_weight)
            .map(|&(_, w)| w)
            .sum();

        for (tx_id, w) in weights {
            if w < min_weight {
                continue;
            }
            // Safe: this body only runs when at least one weight is retained.
            // `retained_sum > 0` follows from the softmax summing to 1 over the
            // candidates that survived `filter(ll > -inf)`, NOT from
            // `min_weight > 0` — the latter is no longer guaranteed now that 0
            // is a legal value, and the argument is what makes `min_weight = 0`
            // (no truncation) well-defined rather than a divide-by-zero.
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
        // ride on rows that survive TWO filters, and the two are independent:
        //   (1) the E-step's `ΔAS > prune_max_delta_as` prune, which removes
        //       the whole candidate from the softmax, and
        //   (2) the `min_weight` cutoff below, which truncates a row that
        //       survived (1).
        // So a read whose top candidate clears (1) and (2) while every
        // competitor fails (1) emits exactly one row, and a read all of whose
        // candidates fall below (2) emits none at all. Reads with decisive
        // evidence are therefore over-represented relative to genuinely
        // contested reads — which is the opposite of the population a
        // score-ratio gate would be judged on. Any analysis of these columns
        // MUST group by `read_id` and report the read-level coverage rate
        // alongside the distribution; do not read the row-level histogram as an
        // unbiased per-read statistic.
        //
        // The molecule's ORIGINAL candidate count (`MoleculeEvidence::candidate_count`
        // as it stands on the graph), captured before BOTH the `ΔAS` prune and
        // the `min_weight` truncation drop rows. It differs from the number of
        // emitted rows whenever EITHER fired. The ΔAS prune is why this column
        // is NOT the count the E-step actually normalised over — the E-step
        // saw `candidate_count − n_pruned` candidates. Downstream readers
        // condition on `n_candidates_before_cutoff >= 2`; that condition now
        // admits reads that were ambiguous on the graph but effectively
        // unambiguous after pruning (one survivor), and the historical series
        // is therefore not comparable across this change without a re-baseline.
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
    const FIVE_PRIME_OFF: f64 = 0.0;
    /// The `ΔAS` prune switched off. Legal (`i32::MAX` is a valid threshold);
    /// used only by the A_rk test, whose `ΔAS = 200` fixture is exactly the
    /// quantity the prune thresholds on, so testing both at once would conflate
    /// them into one number.
    const PRUNE_OFF: i32 = i32::MAX;

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
        let all_competent: Vec<(&str, &str, f64, f64, bool)> = rows
            .iter()
            .map(|&(a, b, c, d)| (a, b, c, d, true))
            .collect();
        write_test_emission_with_competence(path, &all_competent);
    }

    /// As `write_test_emission`, but each row also carries its
    /// `is_polyA_competent` flag. Needed because the E-step pushes an
    /// INCOMPETENT candidate as `-inf` (see the competence mask) and the
    /// output stage's `retained_sum` denominator is documented as correct only
    /// while that keeps incompetent rows out of `weights`. Nothing tested that
    /// coupling, so a future change to the mask would silently break
    /// normalisation with every test still green.
    fn write_test_emission_with_competence(path: &Path, rows: &[(&str, &str, f64, f64, bool)]) {
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
        for &(tx, gene, mu, sigma2, is_competent) in rows {
            tx_b.append_value(tx);
            gene_b.append_value(gene);
            comp_b.append_value(is_competent);
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
    /// `log_pal` and one candidate per `(transcript_id, AS:i)` entry. Every
    /// candidate gets `gap5p = 0`; use `synth_graph_with_gaps` when the 5' term
    /// is under test.
    fn synth_graph(log_pal: f32, candidates: &[(&str, i32)]) -> FlatEvidenceGraph {
        let with_gaps: Vec<(&str, i32, u32)> =
            candidates.iter().map(|&(tx, as_)| (tx, as_, 0)).collect();
        synth_graph_with_gaps(log_pal, &with_gaps)
    }

    /// As `synth_graph`, but each candidate also carries its observed 5' gap
    /// `g_k` (bp from the transcript's 5' end, strand-corrected by Module 2).
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
    ///
    /// Both new terms are explicitly disabled: `PRUNE_OFF` because this
    /// fixture's `ΔAS = 200` IS the quantity the `ΔAS > 50` prune thresholds
    /// on, so leaving it on would delete the very candidate the test measures
    /// and conflate two mechanisms into one number; `FIVE_PRIME_OFF` so the
    /// 0.880797 pin below remains a pure A_rk measurement. The prune has its
    /// own test.
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
            FIVE_PRIME_WINDOW,
            PRUNE_OFF,
            MIN_WEIGHT,
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

        solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            1,
            100.0,
            FIVE_PRIME_OFF,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
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

    /// `min_weight = 0` must mean EXACTLY "no truncation", and that has to be a
    /// real code path, not an accident of `w < 0.0` never being true.
    ///
    /// This is load-bearing for ablation work, not a nicety. The GNG11 gate
    /// measured 2726 emitted rows with the 5' term on against 1816 with it off,
    /// on the same reads and the same total weight: the arm that shifts mass
    /// onto a candidate is the arm that clears the 0.01 cutoff, so the two arms
    /// are not scored on the same support and their difference is not purely the
    /// ablated term. The remedy is to re-run at `min_weight = 0`, which is only
    /// legitimate if 0 restores the full untruncated posterior.
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
                FIVE_PRIME_WINDOW,
                PRUNE_MAX_DELTA_AS,
                min_weight,
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
    /// The output stage's `retained_sum` denominator sums `{w >= min_weight}`
    /// over `weights`, but the emit loop then drops rows that are
    /// `!p.is_competent` or missing from `params`. That is only sound because
    /// the E-step pushes incompetent candidates as `-inf` and
    /// `filter(ll > -inf)` removes them from `weights` before the denominator is
    /// ever taken. The code says so in a comment; nothing TESTED it, so a future
    /// change to the competence mask would make every read's emitted weights sum
    /// to less than 1.0 with the suite still green.
    ///
    /// The fixture puts the 5-nat handicap on the **COMPETENT** transcript, so
    /// the INCOMPETENT one is the candidate the likelihood likes most. That
    /// direction matters and the first draft of this test got it backwards: with
    /// the handicap on the incompetent transcript instead, its weight lands
    /// below `min_weight` and is dropped by TRUNCATION, so the test passed even
    /// with the competence mask fully disabled. Only when the doomed candidate
    /// is the favoured one does it carry a large weight into `retained_sum` —
    /// which is exactly the state in which a bypassed mask strands mass and
    /// leaves the survivor at 0.9933.
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
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
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
        // NOTE on `stats.n_dropped_incompetent` being 0 here, which surprised
        // the first draft of this test. That counter is incremented by the
        // OUTPUT stage's `!p.is_competent` check, which is unreachable for an
        // AMBIGUOUS read: the E-step already pushed this candidate as `-inf`
        // and `filter(ll > -inf)` removed it from `weights`, so no incompetent
        // row ever reaches the emit loop. That unreachability is not a bug --
        // it IS the invariant this test exists to pin -- but it does mean the
        // E-step's competence mask drops candidates with NO telemetry at all:
        // `n_dropped_no_competent_candidates` fires only when `weights` is
        // entirely empty, so a read with 1 competent + 1 incompetent candidate
        // loses the incompetent one silently. Worth a counter; out of scope
        // for a test-only change, flagged to the reviewer instead.
        let _ = stats;
    }

    /// Disjoint id spaces must be REFUSED, not silently absorbed.
    ///
    /// This is the shipped AIDRS Mode default, measured on HC3: the BAM is keyed
    /// `AIDRS_tx_N`, `build-emission` is keyed `ENST`/`ENSG_NovelTr`, and the
    /// intersection is 0. Before this guard the run behaved like a success —
    /// 1637 molecules extracted, EM converged in 2 iterations, 1637 dropped
    /// (502 "incompetent" + 1135 "no_competent"), zero rows emitted, exit 0.
    ///
    /// The `!profile.exists()` assertion is the one that matters: it pins that
    /// the guard fires BEFORE the writer opens the file. A guard placed after
    /// the write would still read as correct from the error message alone.
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
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
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

        solve_and_write_profile(
            graph,
            &emission,
            "TEST",
            &profile,
            1e-6,
            1,
            100.0,
            FIVE_PRIME_OFF,
            FIVE_PRIME_WINDOW,
            PRUNE_MAX_DELTA_AS,
            MIN_WEIGHT,
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

    /// Guards the three parameter validations, which run before any I/O and
    /// are therefore callable with a deliberately nonexistent emission path —
    /// a bail with a parameter message proves the guard fired, and any other
    /// error would prove it did not.
    ///
    /// The CLI's own `anyhow::bail!` calls cannot be reached this way (they sit
    /// inside `main()`'s match arm), which is why the core check is the one
    /// that matters: it is the last gate before a bad parameter becomes NaN
    /// posterior weights, and it is NaN-safe, which `!(x > 0.0)` is not.
    ///
    /// Mutation targets: `five_prime_weight < 0.0` → `<= 0.0` rejects the
    /// documented off switch and makes the Test A / Test B ablation unrunnable;
    /// dropping `.is_nan()` lets a NaN weight through every comparison and
    /// emits NaN posteriors; dropping the `prune_max_delta_as < 0` check lets
    /// a negative threshold drop each read's own argmax.
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
                window,
                prune,
                MIN_WEIGHT,
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
            FIVE_PRIME_WINDOW,
            prune_max_delta_as,
            MIN_WEIGHT,
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
    /// This is the unit-test analogue of the Python harness's Test A. It
    /// isolates the term: holding AS and the emission fixed is the only way
    /// `ln_5p` can be the explanation for the shift.
    ///
    /// Mutation targets: dropping `+ ln_5p` from the E-step, computing the
    /// window as `g_k < 20` (this fixture would still pass — see
    /// `test_five_prime_window_is_inclusive_at_20`), reading the gap from the
    /// wrong pool index, or leaving `FIVE_PRIME_WEIGHT` at 0 all flatten the
    /// pair to 0.5 / 0.5 and turn this red.
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
    /// implementation, including one with the term removed entirely — it passes
    /// with the 5' term disabled. The tests that actually prove the term is live
    /// are `test_five_prime_term_shifts_posterior_toward_the_anchored_candidate`
    /// and `test_five_prime_window_is_inclusive_at_20`; both go red if the term
    /// is neutered. The acceptance criterion for the real ablation is a
    /// DIFFERENCE (Test A −
    /// Test B ≥ 20 pp), never an absolute percentage, because the absolute value
    /// is measured against a label that is not ground truth. The same property
    /// is what makes the term legitimate here: `ln_5p` is a PRIOR over a
    /// coordinate, and a prior that stops mattering when the coordinate is
    /// decorrelated from the truth is doing real work rather than adding a
    /// constant. Both redraw flavours must return the read to a flat 0.5 / 0.5:
    ///
    ///   (a) redraw to a gap OUTSIDE the window (500 for both) — the term
    ///       fires for neither candidate, so it is identically 0;
    ///   (b) redraw to a gap INSIDE the window (7 for both) — the term fires
    ///       for both, so it is a shared additive constant and CANCELS in the
    ///       log-sum-exp. This is the sharper of the two: a term that leaked
    ///       outside the softmax, or was applied after `log_z`, would NOT
    ///       cancel here and this case would go red.
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
    /// threshold by one base against the harness the parameter was validated
    /// on, and — because the coordinate comes from a 1-based `s_k` with the
    /// off-by-one applied in Module 2 — an off-by-one here is indistinguishable
    /// from a coordinate error unless it is pinned at the edge.
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

    /// Pins the posterior at the NEW λ default of 20.0, which until now no test
    /// exercised at all: every pre-existing test passed λ = 100 explicitly, so
    /// the shipped default — at 1000, and now at 20 — has never been pinned.
    ///
    /// `ΔAS = 40` at λ = 20 is a 2.0-nat gap → 0.880797, the same value the
    /// A_rk test reaches at `ΔAS = 200` / λ = 100. That is the point: the same
    /// posterior from a 5× smaller score gap is exactly what dropping λ from
    /// 1000 to 20 means, and at the old default this pair would sit at
    /// 1/(1 + e^-0.04) = 0.99005 instead. `ΔAS = 40` is under the prune
    /// threshold, so this pin is unaffected by pruning.
    ///
    /// Mutation target: shipping λ = 1000 (or any λ ≠ 20) moves this to 0.99005
    /// and turns it red.
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
    /// the read; writing the prune on the `ln_a` numerator (which is
    /// non-positive) makes it never fire at all, and every assertion below
    /// then passes on the `min_weight` cutoff instead — see
    /// `test_prune_boundary_is_strictly_greater_than`.
    ///
    /// Note one mutation that deliberately does NOT turn this red: replacing
    /// the `-inf` push with a bare `continue`. The two are equivalent here
    /// because `weights` is built with `filter(ll > -inf)`; see the comment at
    /// the prune for why, and do not read this test as covering it.
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

        // THE PRUNE'S ACTUAL JOB — a candidate the alignment term and the
        // output cutoff would both KEEP. ΔAS = 60 is −3.0 nats at λ = 20, i.e.
        // w = 0.0474, comfortably above MIN_WEIGHT = 0.01, so with the prune off
        // it survives as a live second hypothesis. The prune deletes it.
        //
        // The region where this matters is `50 < ΔAS < 4.595 × 20 = 91.9`:
        // inside it the prune is strictly stronger than the output cutoff, and
        // it is the whole reason the prune exists now that λ = 20. Under the
        // old λ = 1000 the cutoff alone did this job, at 4.595 × 1000 = 4595.
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
}
