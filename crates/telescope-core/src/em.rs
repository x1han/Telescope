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

use crate::emission_model::INSTRUMENTAL_PAL_NOISE_FLOOR;
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
    // Scale on the poly(A) mixture term
    // `ln alpha_k - ln sigma2_k - (y - mu_k)^2 / (2 sigma2_k)`.
    // 1.0 is the shipped model; 0.0 removes the poly(A) evidence entirely.
    // It does NOT leave an alignment-only likelihood: `ln alpha_k` is the
    // mixture prior and survives at any weight, so at 0.0 the score is
    // `ln alpha_k + ln A_rk + ln_5p_k` -- abundance prior PLUS alignment.
    // Calling that arm "alignment-only" is wrong. It CANNOT be neutralised from the
    // outside: `alpha` is re-estimated by the M-step, so pinning mu and sigma2
    // in the emission parquet still leaks the abundance prior into every
    // candidate's log-likelihood. Hence a parameter, not a data trick.
    poly_a_weight: f64,
    five_prime_window: u32,
    prune_max_delta_as: i32,
    min_weight: f64,
    mut output_em_trace: Option<&mut Vec<(usize, f64, f64)>>,
    // Same idiom as `output_em_trace`: an optional out-param for state a caller
    // may need to inspect. mu/sigma2 were previously unobservable --
    // `TelEmStats` carries neither -- which is exactly why the
    // `poly_a_weight == 0` M-step bypass was untestable when it was added, and
    // why mutation showed removing it changed nothing. Same gap class as the
    // unobservable alpha scale recorded in the Phase A closeout.
    mut output_emission_params: Option<&mut Vec<Option<TranscriptParam>>>,
    // Per-READ posterior summary. The profile parquet is per (read, candidate)
    // PAIR and carries no per-read summary, so "what did this read end up
    // assigned to, and how sure was it" was not recoverable from any output --
    // the entropy existed only as a mean inside `output_em_trace`. This is the
    // `output_emission_params` idiom (filled once, after the solve), not the
    // `output_em_trace` one (pushed per iteration): the state being reported is
    // the FINAL iterate, not the trajectory.
    output_read_assignments: Option<&mut Vec<ReadAssignment>>,
) -> Result<TelEmStats> {
    // API-LEVEL DEFENCE, per the 2026-10-02 ruling.
    //
    // The CLI forces `poly_a_weight = 0` when `--polyA-source none`, but this
    // is a public function and a library caller never goes through the CLI.
    // Pairing a placeholder graph with a positive weight feeds a constant
    // emission into the E-step and silently distorts every posterior — the
    // exact failure the `None` variant's docstring promises cannot be
    // expressed. At the library boundary it CAN be, so it is refused here.
    //
    // The check is on the graph's provenance flag, not on `log_pal` values:
    // a real BAM can legitimately contain reads whose `pt:i` is absent, and
    // those reads are dropped at extraction rather than filled in, so
    // `log_pal == 0.0` in a real graph is not evidence of a placeholder.
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
    // NaN-safe on purpose: `!(w >= 0.0)` is false for NaN, and a NaN weight
    // would poison every posterior weight in the graph. 0.0 is the documented
    // off switch (leaving the abundance prior plus alignment), not an error.
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

    // The `(mu, sigma2)` the LAST E-step actually consumed.
    //
    // WHY THIS EXISTS. "Did poly(A) have any say on this read?" is a property of
    // the evidence THE POSTERIOR WAS BUILT FROM, not of wherever the parameters
    // happen to sit when the output stage runs. The M-step rewrites `p.mu` at
    // :690 and `p.sigma2` at :698, but only under `poly_a_weight > 0.0`, so the
    // two arms of a `--poly-a-weight` A/B read DIFFERENT states at the output
    // stage while having consumed the SAME state in their E-step. Reading
    // `params` there made one shared emission parquet report two different
    // zero-power rates for the same reads.
    //
    // The exit path decides whether that was visible:
    //   - CONVERGENCE (`break` at the `delta_avg_ll < tol` check) happens BEFORE
    //     the M-step, so `params` still equals the snapshot — correct by luck.
    //   - `max_iter` EXHAUSTION runs the M-step last, so `params` is one
    //     M-step NEWER than the posterior. Wrong, and always so at
    //     `--max-iter 1`.
    // So the bug was invisible on converged runs and maximal on the
    // `--max-iter 1` ablations. A measurement that is only wrong on some exit
    // paths is worse than one that is always wrong: the converged number looked
    // fine and was quoted for several rounds.
    //
    // ONE snapshot, TWO consumers — the telemetry below and the
    // `output_emission_params` sidecar — so both describe the same state by
    // construction rather than by two independent correct-looking reads.
    let mut last_estep_params: Vec<Option<TranscriptParam>> = Vec::new();

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
                            // `ln alpha` is the MIXTURE PRIOR, not part of the
                            // poly(A) emission, so it is OUTSIDE the
                            // poly_a_weight scale. Putting it inside is not a
                            // stylistic choice: at `poly_a_weight = 0` the
                            // E-step would never read alpha, the M-step's
                            // update would have no effect on any posterior, and
                            // the "EM" would degenerate into a one-shot static
                            // alignment scorer with an inert M-step.
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
        // Taken HERE, after the E-step consumed `params` and before the M-step
        // rewrites them, so this is exactly the state that produced
        // `final_weights` below.
        last_estep_params = params.clone();

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
        //
        // STEPS 1 AND 2 ARE SKIPPED ENTIRELY AT `poly_a_weight == 0`.
        //
        // The obvious implementation — guard the two `accum_w_y*` lines and
        // leave the rest — does NOT work, and silently produces the very
        // corruption it is meant to prevent. `accum_w` (W) is what STEP 3 needs
        // and must keep accumulating, so with S1 = 0 and W large:
        //
        //     mu     = (0 + tau_0 * mu_cohort) / (w_sum + tau_0)  ->  0
        //     sigma2 = (0 + 2 * beta_0 * sigma2_cohort) / (...)    ->  floor
        //
        // i.e. mu is dragged to 0 and sigma2 onto
        // INSTRUMENTAL_PAL_NOISE_FLOOR, which is precisely the "none"-mode
        // placeholder pollution this bypass exists to stop. Freezing the two
        // STEPS is what "keep mu and sigma2 at their prior/initial values"
        // actually requires.
        //
        // STEP 3 is NOT skipped: alpha is the abundance prior, it does not read
        // poly(A) at all, and skipping it would silently delete the prior.
        for i in 0..num_transcripts {
            if let Some(ref mut p) = params[i] {
                let w_sum = accum_w[i];

                if poly_a_weight > 0.0 {
                    // STEP 1: μ update — weak empirical Bayes shrinkage (τ₀=2).
                    p.mu = (accum_w_y[i] + tau_0 * p.mu_cohort) / (w_sum + tau_0);

                    // STEP 2: σ² update — algebraic closed-form around μ_new.
                    //   Σ w·(y − μ_new)² = S2 − 2·μ_new·S1 + μ_new²·W   (identity).
                    let s1 = accum_w_y[i];
                    let s2 = accum_w_y2[i];
                    let sq_dev_new = (s2 - 2.0 * p.mu * s1 + p.mu * p.mu * w_sum).max(0.0);
                    p.sigma2 =
                        (sq_dev_new + 2.0 * beta_0 * p.sigma2_cohort) / (w_sum + 2.0 * nu_0 + 3.0);
                    p.sigma2 = p.sigma2.max(INSTRUMENTAL_PAL_NOISE_FLOOR);
                }

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
    // above and are NOT in any denominator here -- the CLI labels the
    // denominator "reads with >=1 candidate" rather than "all reads" so the
    // two populations are never conflated.
    let mut n_reads_any_emission_candidate = 0_u64;
    let mut n_reads_single_emission_candidate = 0_u64;
    // DENOMINATOR for `zero-power_resolved`, and the numerator for nothing.
    // Read-level, not candidate-weighted: a candidate-weighted "coverage" over
    // this slice comes out near 100% by construction, because a read is only
    // in the slice once it has at least one present candidate. The read-level
    // ratio is the one that can disagree with the expert-facing number.
    let mut n_reads_multi_emission_candidate = 0_u64;
    let mut n_polya_zero_power_resolved = 0_u64;
    let mut n_entropy_clean = 0_u64;
    let mut n_entropy_partial = 0_u64;
    let mut n_entropy_degenerate = 0_u64;
    let mut n_polya_zero_power = 0_u64;

    // Per-read assignments are filled into a molecule-indexed slot vector and
    // flattened at the end, so the emitted table is in MOLECULE order and
    // carries exactly one row per read. The two loops below write disjoint
    // slots (`unambiguous_indices` vs `ambiguous_indices`), so the two passes
    // cannot clobber each other -- and slotting by molecule index is what
    // keeps the read_id attached to its OWN posterior. An earlier draft zipped
    // the E-step's `final_weights` against `graph.molecules` directly, which is
    // wrong twice over: `final_weights` is indexed by AMBIGUOUS RANK (see
    // `ambiguous_indices[i]` in the M-step and in the loop below), and `zip`
    // truncates to the shorter side, so ~99.7% of reads would have been
    // missing and the survivors mislabelled. Both output loops below index
    // explicitly, copying the pattern already used at the M-step.
    //
    // Allocated only when the caller asked for it: `quant` without
    // `--read-assignments` pays nothing for this.
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
        // Observational columns. Both are per READ properties, so they are
        // derived once here and repeated verbatim on every row of this read
        // (below) — they do NOT vary with the candidate a row describes.
        // `candidate_count` is read off the graph as-is, i.e. BEFORE the
        // E-step's `ΔAS` prune and BEFORE the `min_weight` truncation in the
        // ambiguous branch below can drop rows.
        let ncand = mol.candidate_count;
        let as_ratio = read_max_as_ratio(&graph, mol);
        // EMISSION COVERAGE, counted over ALL reads. An unambiguous read has
        // exactly one candidate, so its `n_with_emission` is 0 or 1 and is
        // readable straight off the lookup below -- no loop needed. These two
        // counters are deliberately NOT scoped to the ambiguous branch: the
        // motivating 31.6% was measured over profile rows, which include
        // unambiguous reads, and scoping to the contested reads would silently
        // report a different number than the one that motivated the change.
        let un_with_emission = params[tx_id].is_some();
        n_reads_any_emission_candidate += u64::from(un_with_emission);
        n_reads_single_emission_candidate += u64::from(un_with_emission);
        if let Some(ref p) = params[tx_id] {
            if !p.is_competent {
                n_unambiguous_dropped_incompetent += 1;
                continue;
            }
            if want_read_assignments {
                // An unambiguous read has one candidate, so its posterior is the
                // deterministic `w ≡ 1.0` and its entropy is exactly 0.0 --
                // which is what `posterior_entropy` returns for a one-element
                // vector. Emitting these is not optional bookkeeping: they are
                // ~99.7% of the read set (see the module header), so a table of
                // ambiguous reads only would silently answer "top-1 switch rate
                // among CONTESTED reads", a much smaller question than the one
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
        } else {
            // transcript missing from emission parquet → skip.
            // This is the case that PADDING removes, and it was conflated with
            // the `!p.is_competent` case above until the 2026-10-05 split; the
            // two are independent facts about the transcript and only one of
            // them is an emission-coverage problem.
            n_unambiguous_dropped_missing_emission += 1;
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

        // DEGRADATION TELEMETRY — how often poly(A) had no say at all.
        // One per-read pass over the candidate slice; no allocation and no new
        // per-read structure. See `polya_emission_count_and_identity` for the
        // definition and for why the comparison is exact rather than
        // tolerance-based.
        //
        // `last_estep_params`, NOT `params`: this must answer "did poly(A)
        // cancel in the posterior we are emitting?", and the posterior came from
        // the last E-step. `params` has since been through one more M-step on
        // the `max_iter`-exhaustion path, which is the whole defect.
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
        // `zero-power_all` keeps its original meaning, which is the pair
        // `n_with_emission > 0 && all_identical` -- i.e. the pre-change
        // `bool`. It is retained so HC2's 16.3% and every earlier number stay
        // comparable, and it is RENAMED rather than deleted because a run that
        // stops printing it cannot be compared with a run that never did.
        if n_with_emission > 0 && all_identical {
            n_polya_zero_power += 1;
        }

        // Phase A entropy 3-tier + ghost-drop telemetry.
        if weights.is_empty() {
            n_dropped_no_competent_candidates += 1;
            // GHOST DECOMPOSITION. `weights.is_empty()` on its own is a
            // conflation of three unrelated causes — absent emission row,
            // `is_polyA_competent == false`, and `alpha <= 0.0` — and the one
            // number that motivated this change (603,936 reads on HC2) cannot
            // be attributed to any of them without a census. The census
            // replays the E-step's own filter over the read's ORIGINAL
            // candidate range, against `last_estep_params` for the same reason
            // `polya_emission_count_and_identity` above uses it: `final_weights`
            // came from the pre-M-step model, and `alpha` is precisely the field
            // the M-step rewrites.
            //
            // `n_dropped_no_competent_candidates` is UNCHANGED and remains the
            // TOTAL, incremented above, so the three cause counters below are a
            // partial decomposition of it rather than a replacement for it. A
            // read whose candidates died for a mix of reasons, or with a pruned
            // candidate among them, increments no cause counter; the CLI prints
            // the remainder explicitly instead of letting the three look like
            // they sum to the total.
            let av = candidate_availability(&graph, mol, &last_estep_params, prune_max_delta_as);
            match av.uniform_cause(mol.candidate_count as usize) {
                Some(GhostCause::Missing) => n_ghost_read_all_missing += 1,
                Some(GhostCause::Incompetent) => n_ghost_read_all_incompetent += 1,
                Some(GhostCause::AlphaZero) => n_ghost_read_all_alpha_zero += 1,
                None => {}
            }
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

        // THE INVARIANT this table has to keep is NOT "assignments is a subset
        // of the profile's reads". It is the one-directional form:
        //
        //     top_tx_id == Some(..)  IMPLIES  that read has a profile row.
        //
        // The subset claim was the wrong invariant and cost a real bug: a read
        // whose every candidate fell below `min_weight` was assigned anyway
        // while emitting zero profile rows, putting a read in the cross-run
        // join's numerator that no profile row could put in its denominator.
        // Three cases, and the middle one is the only silent one:
        //
        //   (1) GHOST -- `weights` is empty (every candidate pushed `-inf`:
        //       pruned, incompetent, or `alpha <= 0`). No profile row, and
        //       that is legitimate: there is nothing to emit. It IS recorded
        //       here with `top_tx_id: None`, because "the solver looked at this
        //       read and could not assign it" and "this read was never in this
        //       run" are different facts and a cross-run comparison needs to
        //       tell them apart. An argmax over the empty vector would either
        //       panic or, worse, yield transcript 0.
        //   (2) `min_weight` culled everything. The read HAS a real argmax, and
        //       it is NOT recorded: the implication above would be violated and
        //       the join denominator would not be auditable. Absent from BOTH
        //       tables. A 0.5/0.5 pair under `--min-weight 0.9` is reachable
        //       directly; at the shipped 0.01 it needs a read spread over
        //       enough survivors that all of them sit under the cutoff.
        //   (3) At least one candidate survives -- recorded, with a real
        //       `top_tx_id`.
        if want_read_assignments
            && (weights.is_empty() || weights.iter().any(|&(_, w)| w >= min_weight))
        {
            // `weights` is pre-truncation and pre-renormalisation (it is what
            // the M-step consumed), so `top_weight` here is the RAW E-step
            // weight. The `posterior_weight` column in the profile is
            // `w / retained_sum` after the `min_weight` cutoff. Different
            // quantities on purpose; do not treat them as interchangeable.
            let (top, top_w) = weights.iter().fold((None, 0.0f64), |acc, &(tx, w)| {
                // Strict `>` (not `>=`) => on an exact tie the FIRST candidate
                // in pool order wins, deterministically. Several fixtures in
                // this file feed the E-step an all-equal `candidate_as`
                // (`zero_power_probe`, the ghost fixture, ...), so ΔAS == 0
                // ties are certainly REACHABLE; what no test pins is that the
                // resulting posterior weights actually come out equal. Flipping
                // this to `>=` leaves the suite green, which is how we know.
                // Read the rule as stated, not as verified.
                if w > acc.1 {
                    (Some(tx), w)
                } else {
                    acc
                }
            });
            read_assign[mol_idx] = Some(ReadAssignment {
                read_id: mol.read_id.clone(),
                n_candidates_before_cutoff: ncand as usize,
                // `None` only on the ghost path: the fold never ran, so there
                // is no candidate to name.
                //
                // The `params` lookup below cannot fail for a tx that is in
                // `weights` -- the E-step only ever pushes a tx for which
                // `params[tx_id]` is `Some`, and nothing after that step adds
                // to or removes from `weights`. So the `get`/`and_then` is
                // defensive plumbing against a panic, NOT the reason the
                // `top_tx_id == Some(..) => profile row` implication holds.
                // That implication rests on a different and separate
                // precondition: the E-step pushes `-inf` for any candidate with
                // `!p.is_competent || p.alpha <= 0.0`, so `weights` can contain
                // competent candidates ONLY, and the output stage filters the
                // same way. Neither the fold nor this lookup re-checks
                // competence; if the E-step's filter is ever neutralised, a
                // transcript can be named here with no profile row behind it
                // rather than degrading to `None`. That filter is pinned by
                // `test_incompetent_but_present_transcript_is_never_a_reads_top1`.
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
                    n_dropped_output_row_competence += 1;
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
                n_dropped_output_row_competence += 1;
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

    if let Some(out) = output_emission_params.as_deref_mut() {
        // `last_estep_params`, NOT `params` — see the declaration. On the
        // `max_iter`-exhaustion path `params` is one M-step past the state that
        // produced `final_weights`, so exporting it would hand a downstream
        // analysis the same stale μ this snapshot was introduced to eliminate.
        //
        // `take`, not `clone`: the telemetry above has already run, so nothing
        // reads the snapshot again. Cloning a 400k-element
        // `Vec<Option<TranscriptParam>>` here would be ~800k String
        // allocations for a value that is about to be dropped anyway.
        *out = std::mem::take(&mut last_estep_params);
    }

    if let Some(out) = output_read_assignments {
        // `flatten()` drops the `None` slots. Every molecule is visited by
        // exactly one of the two loops above, and what survives is NOT a
        // subset of the profile's reads -- it obeys the weaker, one-directional
        // invariant the cross-run join actually needs:
        //
        //     every emitted assignment either HAS a profile row, or is a ghost
        //     with `top_tx_id = None`.
        //
        // The `None`s left are exactly the reads with no profile row AND no
        // reason to be counted: transcripts absent from the emission parquet,
        // and `min_weight`-culled reads. The second group is the one that used
        // to slip through, carrying a real `top_tx_id` into a table whose
        // denominator could not account for it. Ghosts are deliberately NOT in
        // this list -- they are recorded, because "unassignable" and "never
        // seen" must not collapse into the same absence when two runs are
        // compared.
        *out = read_assign.into_iter().flatten().collect();
    }

    Ok(TelEmStats {
        n_unambiguous,
        n_ambiguous,
        n_em_iter,
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
        n_polya_zero_power,
        n_polya_zero_power_resolved,
        n_reads_multi_emission_candidate,
        n_reads_any_emission_candidate,
        n_reads_single_emission_candidate,
        em_trace: output_em_trace.map(|t| t.clone()).unwrap_or_default(),
    })
}

/// Round-trip polyA length through log2.
fn poly_a_length_from_log2(log_pal: f32) -> i32 {
    log_pal.exp2().round() as i32
}

/// DEGRADATION TELEMETRY. Returns `(n_with_emission, all_identical)` for the
/// read's candidate set. `all_identical` is true iff every candidate that
/// carries an emission row shares one identical `(mu, sigma2)` pair, so
///
/// ```text
///   ln [ N(y; mu_a, s2_a) / N(y; mu_b, s2_b) ] = 0   for every a, b
/// ```
///
/// and the read is decided entirely by the alignment (`A_rk`), 5'-anchoring
/// and abundance terms. The COUNT is returned with it because `all_identical`
/// alone cannot distinguish "compared and found equal" from "had nothing to
/// compare against" — see [`ZERO_POWER_MIN_EMISSION_CANDIDATES`].
///
/// WHY THE COMPARISON IS EXACT (`==`), NOT A TOLERANCE. The fits are shared
/// by CONSTRUCTION when Module 1's fallback cascade fires: every
/// underpowered transcript in one gene is handed the *same* `LogNormalFit`
/// value, and every transcript with no usable gene pool is handed the same
/// cohort fit. Those are bit-identical copies, so `==` fires on exactly the
/// rows the fallback actually produced. An approximate test would also fire
/// on two independently-fitted transcripts that merely happen to agree to a
/// few ULPs, which is not the property being measured — and, because the
/// metric is a degradation headline, a false positive there is the
/// expensive direction.
///
/// WHAT THAT MEANS IN NUMBERS, since it is the part a reader is most likely
/// to get wrong. The comparison is bitwise, so the threshold for "different"
/// is ONE ULP and not some tunable epsilon:
///
/// - a `mu` (or `sigma2`) gap of `1e-13` — the smallest difference
///   representable at `mu = 7.0` is about `9e-16`, so `1e-13` is a real,
///   distinct `f64` — already makes the candidates DISAGREE;
/// - a gap of `1e-9` also makes them disagree, by a wide margin.
///
/// A read whose candidates differ by even a vanishing amount in one component
/// is therefore NOT counted as zero-power. That is the CONSERVATIVE
/// direction: the metric UNDER-reports zero discriminative power rather than
/// over-reporting it, and it will never claim poly(A) did nothing on a read
/// it was not provably identical on. Do not "fix" this by loosening the
/// comparison to a tolerance — that would move the metric toward
/// over-reporting, which is the expensive direction for a degradation
/// headline. Pinned by `test_polya_emission_count_and_identity_cases` (cases 5
/// and 5-control): swapping `==` for a `1e-12` tolerance turns it red.
///
/// A candidate with NO emission row is skipped rather than treated as a
/// mismatch: a ghost transcript has no `(mu, sigma2)` to disagree with, and
/// it is already counted by `n_dropped_no_competent_candidates`. A read
/// with no candidate that has an emission row at all returns `(0, false)` —
/// there is nothing to compare, so claiming "zero discriminative power"
/// would be vacuous rather than measured.
///
/// ⚠️ THAT LAST CLAUSE IS ABOUT THE `bool`, AND IT WAS NOT ENOUGH. A read
/// with exactly ONE candidate carrying an emission row returns `(1, true)`:
/// the loop finds no disagreement because there is nothing to disagree with
/// the first candidate. That is the vacuity this change exists to measure, and
/// it is why the count travels with the flag. `zero-power_all` keeps counting
/// those reads (it did before, and continuity is the point of keeping it);
/// `zero-power_resolved` excludes them.
///
/// The read's FULL candidate range is used, not the post-prune /
/// post-truncation set: the question is what evidence the read HAD, not
/// what survived.
///
/// # Returns `(n_with_emission, all_identical)`
///
/// - `n_with_emission` — how many of the read's candidates carry an emission
///   row. **`0` is possible and is not the same as `1`.**
/// - `all_identical` — whether every one of those candidates shares the first
///   one's exact `(mu, sigma2)`. Vacuously `true` when `n_with_emission < 2`,
///   which is precisely why the count is returned alongside it: a caller that
///   reports "poly(A) had no say" without also reporting the count is reporting
///   a coincidence of an empty comparison.
///
/// THE LOOP NEVER EARLY-RETURNS. An earlier version of this function returned
/// `false` the moment two candidates disagreed, which is correct for the
/// `bool` it used to answer and WRONG for the count: a read whose candidates
/// are `[same, same, different]` would report `2` present instead of `3`. That
/// undercount lands only on DISAGREEING reads — the ones where poly(A) actually
/// did something — so it silently biases `emission coverage` downward while
/// leaving the zero-power headline untouched. `disagrees` is a flag for that
/// reason; see case 3b in `test_polya_emission_count_and_identity_cases`, which
/// fails if anyone restores the early return.
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
/// With one candidate there is no pairwise odds ratio to cancel, so
/// `zero-power_all` counts such reads even though poly(A) was never given a
/// chance to disagree with anything. This is the threshold that separates
/// `zero-power_all` (kept, for continuity) from `zero-power_resolved`.
///
/// Mutation M10 changes the `2` to a `3`; the resolved count must change with
/// it. Do not inline this literal.
const ZERO_POWER_MIN_EMISSION_CANDIDATES: usize = 2;

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

/// Why a read's candidate set contains nothing that could carry posterior mass.
///
/// The census is taken over the read's ORIGINAL candidate range, and each
/// bucket is defined by the E-step's own decision procedure, in the E-step's
/// own order. That is the whole point: a read is a ghost exactly when the
/// E-step's `max_ll` came out `-inf`, so any census that does not replay the
/// E-step's filter will disagree with the fact it is being invoked to explain.
///
/// Concretely, the two ways a candidate can be absent from the E-step's `raw`
/// vector are NOT the same and must not share a bucket:
///
/// - **pruned** (`delta_as > prune_max_delta_as`) — the E-step pushes `-inf`
///   and `continue`s, so the entry IS in `raw`;
/// - **no emission row** — the `if let Some(ref p) = params[tx_id]` arm has no
///   `else`, so the entry is never pushed at all.
///
/// A census that lumped these together would report a cause for reads whose
/// real cause was the other, which is the failure `test_ghost_decomposition_*`
/// exists to catch. The prune is likewise excluded from every bucket rather
/// than counted as one of the three causes, because a pruned candidate is dead
/// for a reason no amount of emission padding addresses.
struct CandidateAvailability {
    /// No emission row at all — the candidate never reached `raw`.
    n_missing: usize,
    /// Present, but `is_polyA_competent == false` — pushed as `-inf`.
    n_incompetent: usize,
    /// Present and competent, but `alpha <= 0.0` — pushed as `-inf`.
    ///
    /// Distinct from `n_incompetent` because the two are independent facts
    /// about the same transcript, and the M-step can drive `alpha` to exactly 0
    /// — its update divides the posterior mass a transcript won by the total
    /// molecule count, so a transcript that won none lands on 0 — without
    /// touching `is_competent`. A single combined "incompetent or starved"
    /// bucket would hide which of the two happened.
    n_alpha_zero: usize,
    /// Carried posterior mass. A ghost read has `n_live == 0`; see
    /// [`CandidateAvailability::is_ghost`].
    n_live: usize,
}

impl CandidateAvailability {
    /// The single cause shared by EVERY candidate, when exactly one cause
    /// accounts for all of them; `None` when the read's candidates died for a
    /// mix of reasons, or when any candidate was pruned. `candidate_count` is
    /// the read's original candidate count — the denominator — and pruned
    /// candidates are IN it and in NO bucket, so a read with a pruned candidate
    /// plus a uniformly-missing remainder is reported as `None` rather than as
    /// `Missing`: padding cannot revive a pruned candidate, so calling that
    /// read recoverable would over-claim.
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
/// defined this way and why pruning is excluded from all of them.
///
/// `params` must be `last_estep_params`, NOT `params`: the census explains a
/// `final_weights` that was produced from the pre-M-step model, and `alpha` is
/// exactly the field the M-step rewrites. This is the same trap
/// `polya_emission_count_and_identity` already documents.
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
        // `--release` and a wrapped NEGATIVE delta_as would pass the prune,
        // putting a candidate the E-step killed into the `n_live` bucket.
        let delta_as = max_as.saturating_sub(graph.candidate_as[idx]);
        if delta_as > prune_max_delta_as {
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
/// Three decisions here are load-bearing and each has bitten before:
///
/// - `top_tx_id` is the transcript NAME, not the pool index. The index is only
///   meaningful relative to one run's `transcript_names` order, and this table's
///   whole purpose is being joined against a SECOND run's table.
/// - `n_candidates_before_cutoff` is `MoleculeEvidence::candidate_count`, i.e.
///   the fan-out BEFORE the `ΔAS` prune, the `is_competent` filter and the
///   `min_weight` truncation. Using the post-prune `weights.len()` instead
///   would silently redefine "candidate count" and, at the shipped
///   `prune_max_delta_as = 50`, would select a different and much smaller
///   population for any "fan-out >= N" question. It matches the
///   `n_candidates_before_cutoff` column the profile parquet already emits.
/// - `top_tx_id` is `None` for a GHOST read -- no competent candidate, empty
///   `weights`. An argmax over an empty vector would either panic or, worse,
///   yield transcript 0: a real transcript the read was never assigned to.
///   The ghost has no row in the profile either, so this table is deliberately
///   NOT a subset of the profile's reads. It obeys the weaker one-directional
///   invariant the cross-run join needs, and `None` is what makes the two
///   distinguishable:
///
///   ```text
///   top_tx_id == Some(..)  IMPLIES  that read has a profile row.
///   ```
///
///   So a `Some` always has a denominator, and a `None` means "the solver
///   looked and could not assign", which is not the same as "this read was
///   absent from the run". A read whose candidates ALL fell below
///   `min_weight` has a real argmax but no profile row, so it is recorded in
///   NEITHER table -- that is the one shape that would break the implication.
#[derive(Debug, Clone)]
pub struct ReadAssignment {
    pub read_id: String,
    pub n_candidates_before_cutoff: usize,
    /// `None` for a ghost read (every candidate dropped before the softmax).
    /// This is why the parquet column is nullable: the cross-run join needs
    /// "unassignable" to be a value it can see, not an absence it has to
    /// infer.
    pub top_tx_id: Option<String>,
    /// Raw E-step weight of the top candidate: PRE `min_weight` truncation and
    /// PRE renormalisation. It is therefore NOT equal to the `posterior_weight`
    /// the profile parquet reports for the same read, which is `w /
    /// retained_sum`. Both are defensible; they are different quantities.
    pub top_weight: f64,
    /// Posterior entropy in nats, recomputed from the final weights with the
    /// same `posterior_entropy` the output stage uses. 0.0 for an unambiguous
    /// read, which is what that function returns for a one-element vector and
    /// is consistent with its deterministic `w ≡ 1.0` branch.
    pub entropy: f64,
}

/// Write the per-read posterior summary as Parquet, matching every other
/// output this tool produces (profile, emission, anchors, summary are all
/// Parquet; TSV appears in this codebase only as an INPUT format).
///
/// `top_tx_id` is written as a nullable string rather than an index: the whole
/// purpose of this table is to be joined against another run's table, and a
/// pool index only means something relative to one run's `transcript_names`.
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
        // Null, not "" and not index 0: an empty string would be a transcript
        // that does not exist, and index 0 is a real transcript this read was
        // never assigned to.
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
/// The remaining columns exist because the file's whole purpose is to be
/// auditable against the stale-μ defect it was added to fix:
///
/// - `mu_cohort` — the shrinkage prior, so a reader can see that re-loading
///   this file RE-ANCHORS the prior. The loader sets `mu_cohort = mu_logPAL`
///   *and* `sigma2_cohort = sigma2_logPAL`, so BOTH priors move to the fitted
///   values and τ₀=2 / 2β₀ are applied a second time — the σ² half is as real
///   as the μ half and is the one a reader is likelier to miss. Without this
///   column neither is visible.
/// - `alpha` — the mixture weight the last E-step used.
/// - `state` — which iterate this row holds, `e_step_final`. A file that cannot
///   say which state it carries cannot be checked against the defect.
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
    pub n_unambiguous: usize,
    pub n_ambiguous: usize,
    pub n_em_iter: usize,
    /// READ-LEVEL, unambiguous population. Denominator `n_unambiguous`.
    /// An unambiguous read has exactly one candidate, so this counts reads
    /// whose single candidate was present and `is_polyA_competent == false`.
    ///
    /// Until the 2026-10-05 split this field ALSO counted contested reads
    /// (every candidate incompetent) and, once per ROW, the output stage's own
    /// filter — three different things under one name, and two different
    /// denominators. The contested case is now `n_ghost_read_all_incompetent`
    /// and the row-level one is `n_dropped_output_row_competence`.
    pub n_unambiguous_dropped_incompetent: u64,
    /// READ-LEVEL, unambiguous population. Denominator `n_unambiguous`.
    /// The read's single candidate has no emission row at all.
    ///
    /// This is the counter padding is expected to drive to zero:
    /// `build_emission_from_anchors` iterates only the anchor pool, so a
    /// transcript with no anchor is absent from the table and every read whose
    /// only candidate is such a transcript lands here. A non-zero value after a
    /// padded build is a defect in the padding, not a property of the data.
    pub n_unambiguous_dropped_missing_emission: u64,
    /// READ-LEVEL, ghost population. Denominator
    /// `n_dropped_no_competent_candidates`. Contested reads with no posterior
    /// whose ENTIRE candidate range had no emission row.
    ///
    /// Kept separate from `n_unambiguous_dropped_missing_emission` rather than
    /// summed into it because the two have different denominators: a single
    /// combined "missing emission row" total could only be reported as a raw
    /// count against no meaningful fraction, and the whole purpose of the
    /// decomposition is to be able to say what fraction of each population
    /// padding would have rescued.
    pub n_ghost_read_all_missing: u64,
    /// READ-LEVEL, ghost population. Denominator
    /// `n_dropped_no_competent_candidates`. Contested reads with no posterior
    /// whose entire candidate range was present and `is_polyA_competent == false`.
    pub n_ghost_read_all_incompetent: u64,
    /// READ-LEVEL, ghost population. Denominator
    /// `n_dropped_no_competent_candidates`. Contested reads with no posterior
    /// whose entire candidate range was present and competent but had
    /// `alpha <= 0.0` — the abundance prior the M-step drove to zero because
    /// the transcript won no posterior mass.
    ///
    /// A genuinely distinct cause from the other two: `is_competent` is
    /// M-step-invariant while `alpha` is exactly the field the M-step rewrites,
    /// and it lands on 0 for a transcript that won no posterior mass. A single
    /// combined "incompetent or starved" bucket would hide which of the two
    /// happened, and they call for opposite remedies.
    pub n_ghost_read_all_alpha_zero: u64,
    /// ROW-LEVEL, and **not** comparable with any of the five read-level
    /// counters above. Incremented once per (read × candidate) row by the
    /// output stage when a row's transcript is `!is_competent` or absent from
    /// `params`.
    ///
    /// This is 0 in practice, and it is meant to stay 0: the E-step already
    /// pushed such candidates as `-inf` and `filter(ll > -inf)` removed them
    /// from `weights`, so no such row can reach the emit loop. `is_competent`
    /// is M-step-invariant (the M-step writes only `mu`, `sigma2`, `alpha`),
    /// and a candidate with no emission row never enters `raw` at all, so
    /// neither can appear here. It is a TRIPWIRE: if the E-step's `-inf` mask
    /// is ever removed, this fires, and the read-level counters would silently
    /// stop accounting for the output stage's share of the loss.
    ///
    /// `test_incompetent_candidate_is_excluded_from_the_renormalisation_base`
    /// carries the NOTE recording that this counter reads 0 there and why.
    pub n_dropped_output_row_competence: u64,
    /// READ-LEVEL, ghost population. Contested reads (>= 2 candidates) with NO
    /// posterior at all — `weights.is_empty()`. This is the TOTAL for the
    /// decomposition above.
    ///
    /// The NAME predates the 2026-10-05 decomposition and is misleading in two
    /// ways: it fires for several causes, not one, and the reads it counts are
    /// reads rather than candidates. It is kept verbatim because
    /// `workspace/validation/sim/telescope_sim_benchmark.py` compares its own
    /// Python-side ghost count against this field and `die()`s on a mismatch,
    /// so renaming it would break a gated external validation for no
    /// informational gain. The CLI labels it "ghost reads, no posterior".
    ///
    /// `n_ghost_read_all_missing + n_ghost_read_all_incompetent +
    /// n_ghost_read_all_alpha_zero` is a PARTIAL decomposition of it and does
    /// not sum to it. The remainder is reads whose candidates died for a mix of
    /// reasons, or that had a pruned candidate among them; the CLI prints that
    /// remainder rather than letting the three look exhaustive.
    pub n_dropped_no_competent_candidates: u64,
    /// Posterior entropy 3-tier counters (per ambiguous read, in nats).
    /// clean: H < 0.20 nats (near-deterministic assignment).
    pub n_entropy_clean: u64,
    /// partial: 0.20 <= H <= 0.70 nats (two-three-way partial ambiguity).
    pub n_entropy_partial: u64,
    /// degenerate: H > 0.70 nats (effectively uniform over candidates).
    /// The upper bounds are 0.20 / 0.70, NOT 0.20 / 0.60: the degenerate bound
    /// was raised to 0.70 and this comment was left behind. Anything that
    /// re-derives these tiers from the doc comment rather than from the
    /// comparison in the E-step will bin the wrong reads.
    pub n_entropy_degenerate: u64,
    /// DEGRADATION TELEMETRY. Contested reads (≥ 2 candidates) whose
    /// candidate set carries ZERO poly(A) discriminative power: every
    /// candidate that has an emission row shares one identical
    /// `(mu, sigma2)` pair, so the poly(A) term cancels EXACTLY in every
    /// pairwise odds ratio and the read is decided entirely by the
    /// alignment and abundance terms.
    ///
    /// Read `n_polya_zero_power / n_ambiguous` for the fraction. The
    /// denominator is deliberately `n_ambiguous` and not `n_total`: a
    /// single-candidate read has no pairwise odds ratio at all, so counting
    /// it would make this number ≈ 0.997 and say nothing.
    ///
    /// The mechanism is the L1 → L2 → L3 fallback: two underpowered
    /// transcripts in one gene both take the SAME shared gene fit, and two
    /// transcripts with no usable gene pool both take the SAME cohort fit.
    /// Identical fits are the fallback working as designed, not a fault —
    /// but the fraction of contested reads on which poly(A) contributes
    /// literally zero is the honest headline for how much of the
    /// assignment the tail-length evidence is actually carrying.
    ///
    /// THE COMPARISON IS EXACT `==`, NOT A TOLERANCE, and the consequence is
    /// part of this metric's definition rather than an implementation detail:
    /// candidates that disagree by as little as one ULP — a `mu` gap of
    /// `1e-13` is already a distinct `f64` — are treated as DISCRIMINATING, so
    /// the read is not counted here. The number therefore UNDER-reports zero
    /// poly(A) power and never over-reports it, which is the conservative
    /// direction for a degradation headline. See
    /// `polya_emission_count_and_identity` for why a tolerance would move it
    /// the wrong way.
    pub n_polya_zero_power: u64,
    /// DEGRADATION TELEMETRY, second metric. Contested reads carrying at least
    /// `ZERO_POWER_MIN_EMISSION_CANDIDATES` emission rows in which all of them
    /// share one identical `(mu, sigma2)`. Unlike `n_polya_zero_power` this
    /// cannot be satisfied by a read that had nothing to compare, which is the
    /// whole point: at less-than-full model coverage, most of `n_polya_zero_power`
    /// is that.
    ///
    /// Read `n_polya_zero_power_resolved / n_reads_multi_emission_candidate`.
    /// Do NOT divide by `n_ambiguous` -- that denominator contains the reads
    /// this metric deliberately excludes, and the ratio would silently
    /// reproduce the vacuous number.
    pub n_polya_zero_power_resolved: u64,
    /// DENOMINATOR for `n_polya_zero_power_resolved`: contested reads with at
    /// least `ZERO_POWER_MIN_EMISSION_CANDIDATES` candidates carrying an
    /// emission row. Can legitimately be 0 on a sparse model; the CLI guards.
    pub n_reads_multi_emission_candidate: u64,
    /// NUMERATOR for emission coverage. Counts over ALL reads with at least one
    /// candidate, not just contested ones: coverage is a property of the model's
    /// completeness, and scoping it to the contested reads would report a
    /// different number than the 31.6% that motivated the metric. Divide by
    /// `n_unambiguous + n_ambiguous` and label the denominator "reads with
    /// >=1 candidate" -- molecules with zero candidates are dropped earlier.
    pub n_reads_any_emission_candidate: u64,
    /// NUMERATOR for the single-candidate share: reads with exactly ONE
    /// candidate carrying an emission row. Same denominator as
    /// `n_reads_any_emission_candidate`.
    ///
    /// ⚠️ THIS IS A STRICT SUPERSET OF THE VACUOUS READS, NOT THE SAME SET.
    /// It is incremented in BOTH branches, so it includes unambiguous reads
    /// that have one present candidate -- and those never reach
    /// `n_polya_zero_power` at all, because that counter only runs on the
    /// contested branch. Decomposing `zero-power_all` into "these are the reads
    /// that made it vacuous" needs `n_reads_multi_emission_candidate` and the
    /// per-read counts; the printed pair does NOT sum to it. In the fixture
    /// `test_zero_power_resolved_excludes_reads_with_fewer_than_two_emission_candidates`
    /// this counter is 2 and only ONE of those two is a vacuous zero-power
    /// read (`SINGLE_THEN_GHOST`; the other is `UNAMBIGUOUS`).
    pub n_reads_single_emission_candidate: u64,
    /// EM trace: per-iter `(iter_index, log_likelihood, mean_entropy)`.
    pub em_trace: Vec<(usize, f64, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    use arrow::array::{Array, BooleanBuilder, Int64Array};

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
    const POLY_A_ON: f64 = 1.0;
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

    /// At `poly_a_weight == 0` the M-step must FREEZE mu and sigma2, not
    /// recompute them from a placeholder `y`.
    ///
    /// This is the test the bypass did not have when it was first written.
    /// Removing the bypass (`if poly_a_weight > 0.0` -> always) left all 53
    /// tests green, because mu/sigma2 were not observable from outside --
    /// `TelEmStats` carries neither, and the profile's `polyA_length` column is
    /// the READ's own length, not the fitted mu. The change was untested and
    /// the suite said nothing about it.
    ///
    /// The failure it would have caused is specific, not cosmetic. With
    /// `accum_w` (W) still accumulating and S1 forced to 0:
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

        let _ = std::fs::remove_file(&emission);
        let _ = std::fs::remove_file(&profile);
    }

    /// `solve_and_write_profile` is public, so the CLI's `poly_a_weight = 0`
    /// interlock is not enough: a library caller can pair a placeholder graph
    /// with a positive weight, which feeds a constant emission into the E-step
    /// and distorts every posterior. That combination must be unexpressible at
    /// the API boundary too.
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
            // synthetic/test graph: `log_pal` is a real measurement
            poly_a_is_placeholder: false,
        }
    }

    /// One-candidate-per-entry graph plus a matching `params` vector, for
    /// testing [`candidate_availability`] directly.
    ///
    /// `spec` is one `(AS, is_competent, alpha)` per candidate, in the order
    /// they should occupy the read's candidate range; `present` says whether
    /// that candidate keeps its emission row. `synth_graph_with_gaps` assigns
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
                })
            })
            .collect();
        candidate_availability(&graph, &graph.molecules[0], &params, PRUNE_MAX_DELTA_AS)
    }

    /// The census has to keep four things apart that all end in "this read has
    /// no posterior", and the four are distinguished ONLY by which filter killed
    /// the candidate. Three of the four are reachable from production data and
    /// one (the `ΔAS` prune) is not: a pruned candidate is dead for a reason no
    /// amount of emission padding addresses, so a read that has one is NOT
    /// reported as recoverable.
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
            ],
            &[true, true, true, true, true, false],
        );
        assert_eq!(av.n_live, 2, "0 and 1 only: candidate 4 is pruned");
        assert_eq!(av.n_missing, 1, "candidate 5 has no emission row");
        assert_eq!(
            av.n_incompetent, 1,
            "candidate 2 is is_polyA_competent=false"
        );
        assert_eq!(
            av.n_alpha_zero, 1,
            "candidate 3 is competent but alpha == 0"
        );
        assert_eq!(
            av.uniform_cause(6),
            None,
            "four different causes are present, so no single one accounts for the \
             read. Reporting any of them here would over-claim what padding fixes."
        );

        // Each uniform cause, one at a time. `alpha == 0` is unreachable through
        // the solver (a transcript only reaches it by winning no mass, and a
        // transcript that wins no mass was already dead), which is exactly why
        // it is pinned HERE rather than only in the end-to-end test.
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

    /// A graph whose abundance prior is set entirely by UNAMBIGUOUS reads, so
    /// the prior is knowable without running the EM.
    ///
    /// 30 single-candidate molecules on `tx0`, 3 on `tx1`, then one two-candidate
    /// `PROBE` molecule with EQUAL `AS` to both and `g_k = 0` on both. The probe
    /// therefore has no evidence of its own, and the init prior is
    /// `(0.01 + 30 + 0.5) : (0.01 + 3 + 0.5)` — the Laplace epsilon and the
    /// probe's half-share are `solve_and_write_profile`'s convention, read here
    /// so the fixture does not hide where the number comes from.
    ///
    /// Shared by the two tests below; they differ in what they sweep, not in
    /// what they measure.
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
        };
        assert_eq!(graph.candidate_pool.len(), n);
        assert_eq!(graph.candidate_as.len(), n);
        assert_eq!(graph.candidate_5p_gap.len(), n);
        graph
    }

    /// MODEL IDENTITY AXIOM, permanent P0 guard: **when every likelihood
    /// evidence term is equal across candidates, the posterior must reduce to
    /// the abundance prior.** This is a property of the E-step, not of any
    /// flag, so it must hold in EVERY arm.
    ///
    /// Why it is a separate test from the `poly_a_weight` regression below:
    /// that one pins one flag at one setting and would still pass if a future
    /// edit swallowed `ln alpha` inside some *other* weight's bracket — say
    /// `five_prime_weight * (ln_5p + alpha.ln())`, which is inert at
    /// `five_prime_weight = 0` and would survive a poly(A)-only test. The
    /// shared failure mode is not "a parameter is set wrong", it is "the
    /// mixture prior stopped being a mixture prior", and that can happen
    /// through any of the three weights. So this one sweeps all three.
    ///
    /// The sweep is exact rather than approximate because the fixture makes
    /// each evidence term a CONSTANT across the two candidates, and a constant
    /// cancels in the softmax for any scale factor:
    ///   - `AS` equal      -> `ln_a = -ΔAS/λ = 0` for every `λ`
    ///   - `g_k = 0` equal -> `ln_5p = five_prime_weight` for both, so it is
    ///     the same constant added to both `ll`s
    ///   - `mu`, `sigma2` equal -> the Gaussian is one constant times
    ///     `poly_a_weight`
    /// So the odds `w_0 / w_1` must equal the prior odds in all six arms, and
    /// in particular must be invariant ACROSS arms. That invariance is the
    /// half of the check that pins the weight coefficients: it is scale-free,
    /// so it survives a legitimate change to the init convention, which a
    /// hard-coded 8.69 would not.
    ///
    /// A posterior near 0.5 in any arm means `ln alpha` is not reaching that
    /// arm's softmax at all — the M-step then re-estimates abundances that no
    /// posterior can see, and the "EM" is inert.
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
    /// `alpha` is a mixture weight vector, and `p.alpha = w_sum /
    /// n_total_molecules` is what makes it one: every molecule contributes
    /// total weight 1 (an unambiguous read contributes exactly 1.0; an
    /// ambiguous read's posteriors sum to 1 by construction of the
    /// log-sum-exp), so `sum_k w_sum_k == n_total_molecules` and the updated
    /// `alpha` sums to 1 across transcripts.
    ///
    /// This is the gap the mutation battery walked into. Its M5 mutation
    /// changed the divisor from the total molecule count to that count minus
    /// 5.0, and left all four `toy_analytic_em` tests green, so the battery
    /// concluded "the DENOMINATOR is not [load-bearing] (alpha enters only
    /// through the E-step softmax, so any positive rescaling of it is
    /// invisible to every posterior)". Both halves of that are right and
    /// neither is sufficient: `N / (N - 5)` IS a uniform rescale, so the
    /// softmax genuinely cannot see it, and `alpha` is written to no output
    /// parquet, so no emitted number can see it. But the rescale is uniform
    /// only BECAUSE the divisor is the total molecule count. The same rescale
    /// leaves the vector summing to `N / (N - 5)` = 1.1724 on this fixture
    /// instead of 1.
    ///
    /// (The mutation's exact source text is deliberately NOT quoted here: the
    /// battery's patcher requires its search string to match exactly once, and
    /// a doc comment restating it would make M5 abort with "PATCH DID NOT
    /// APPLY, count = 2". See `mutation_battery.sh`.)
    ///
    /// `output_emission_params` is what makes the vector readable at all; the
    /// comment on that parameter records the same gap class ("mu/sigma2 were
    /// previously unobservable ... which is exactly why the `poly_a_weight ==
    /// 0` M-step bypass was untestable when it was added"). Reaching for a
    /// public out-param rather than a golden posterior is deliberate: the
    /// posteriors cannot pin this at any `alpha` scale, so a test that asserts
    /// on them would be the fourth green test the battery already found.
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

        // 30 unambiguous molecules on tx0, 3 on tx1, 1 ambiguous PROBE: 34
        // molecules over 2 transcripts, every one of them with at least one
        // candidate that has emission parameters. That last property is what
        // the sum-to-1 invariant rests on -- a molecule whose every candidate
        // is missing from the emission parquet contributes 0 to `accum_w` while
        // still counting in `n_total_molecules`, and would legitimately pull
        // the sum below 1.
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

        // CONTROL, and it has to come first: `init_counts / total_init` ALSO
        // sums to 1, so the assertion below would pass on a solver whose
        // M-step never ran. The init prior here is Laplace eps=0.01 plus the
        // unambiguous counts plus the PROBE's 1/2 share, over both transcripts.
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

    /// The abundance prior `ln alpha` is the MIXTURE prior, not part of the
    /// poly(A) emission, so it must survive `poly_a_weight = 0`.
    ///
    /// This is the regression test for a real P0. The parameter was originally
    /// written as `poly_a_weight * (alpha.ln() - ...)`, which looks like a
    /// harmless grouping. It is not: at weight 0 the E-step never reads alpha,
    /// so the M-step's abundance update cannot influence any posterior, and the
    /// "EM" silently degenerates into a one-shot static alignment scorer. The
    /// number it prints is still a plausible profile, which is exactly why it
    /// would have survived review.
    ///
    /// Fixture: unambiguous reads set the prior (tx0 gets 30, tx1 gets 3), then
    /// ONE ambiguous read with EQUAL `AS` to both, 5' off, poly(A) off. With no
    /// evidence of its own, that read must follow the prior. If alpha is scaled
    /// away the two candidates are exactly tied and it splits 50/50.
    ///
    /// The assertion is on the AMBIGUOUS READ'S OWN weight, not the aggregate:
    /// the 33 unambiguous reads would otherwise swamp the signal and the two
    /// outcomes would differ by well under a percentage point.
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
    /// `ln alpha_k` term is re-estimated by the M-step and would still carry
    /// the abundance prior into every candidate's log-likelihood. Hence a
    /// parameter.
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

    /// The output stage's `retained_sum` denominator, with ALL THREE of the
    /// routes by which a candidate can leave `weights` or the emitted rows
    /// present at once.
    ///
    /// `retained_sum` sums `{w >= min_weight}` over `weights`, and the emit
    /// loop then drops rows that are `!p.is_competent` or missing from
    /// `params`. Those two later filters are currently unreachable -- the
    /// E-step has already pushed incompetent candidates as `-inf` (and
    /// `filter(ll > -inf)` removed them) and `if let Some(ref p) = params[…]`
    /// SKIPS a candidate with no emission row instead of pushing anything. So
    /// the denominator is right only because the E-step and the emit loop
    /// agree on the candidate set, and that agreement is what this fixture
    /// exercises end to end.
    ///
    /// Why this is not a third copy of the two tests above. Each of those
    /// isolates ONE route on a two-candidate read whose survivor absorbs the
    /// entire posterior, so the per-row value is 1.0 either way and only the
    /// row COUNT is load-bearing. Here the surviving mass is SPLIT, so the
    /// denominator is pinned by the per-row values as well as by the sum, and
    /// all three exclusion routes are in the same read:
    ///
    /// | candidate | route out                                    | weight |
    /// |-----------|----------------------------------------------|--------|
    /// | `tx0`     | retained                                    | 0.80   |
    /// | `tx1`     | retained                                    | 0.15   |
    /// | `tx2`     | `w = 0.005 < min_weight` -> TRUNCATED        | 0.005  |
    /// | `tx3`     | `is_polyA_competent = false` -> E-step `-inf`| 1.00   |
    /// | `tx4`     | no emission row -> E-step `if let Some` skip | 1.00   |
    ///
    /// The handicaps are `mu` offsets sized so `(y − mu)² / (2σ²)` differs by
    /// exactly `ln(0.80/0.15)` and `ln(0.80/0.005)` nats. All five candidates
    /// carry EQUAL `AS`, so `ln_a = 0` for all and the split is the emission
    /// term alone. `max_iter = 1` freezes the posterior to the E-step that ran
    /// on the initial (uniform) prior, so the expected values below are exact
    /// rather than a property of where the EM happened to converge.
    ///
    /// Mutation targets, both confirmed to turn this red:
    ///   * the denominator summing over ALL weights (`*w >= 0.0`) leaves the
    ///     two survivors at 0.8377 / 0.1571 and the read summing to 0.9948;
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
    /// This WAS the shipped AIDRS Mode default, measured on HC3: the BAM was keyed
    /// `AIDRS_tx_N`, `build-emission` was keyed `ENST`/`ENSG_NovelTr`, and the
    /// intersection was 0. AIDRS Mode is no longer a CLI-reachable path (the
    /// initializer it depended on is deferred — see `emission_model`'s module
    /// docs), but the guard it motivated is generic and still needed: any caller
    /// can hand `solve_and_write_profile` an emission parquet built on a different
    /// ID space, and the failure below is what it looks like. Before this guard
    /// the run behaved like a success —
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

    // ---------------------------------------------------------------------
    // Per-read assignments
    // ---------------------------------------------------------------------

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
        // The two ambiguous reads point at DIFFERENT transcripts, so a join
        // that is off by one position is caught by content and not merely by
        // row count.
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
        // -- MIRRORED, so each resolves to a DIFFERENT transcript. An earlier
        // draft gave both reads the same pair in the same order, which made
        // both resolve to tx0 and would have let an off-by-one join pass.
        let mut candidate_pool = vec![2, 0, 1, 1, 0];
        // The `ΔAS` of each ambiguous read is STRICTLY POSITIVE and STRICTLY
        // BELOW `PRUNE_MAX_DELTA_AS` (50): 30 for AMB_TX0 and 40 for AMB_TX1.
        //
        // Both bounds are load-bearing, and an earlier draft violated the
        // upper one at ΔAS = 100. The prune then dropped the runner-up before
        // the softmax ever saw it, so every ambiguous read collapsed to a
        // ONE-element posterior with `top_weight == 1.0` and `entropy == 0.0`
        // -- and because those two numbers are forced by the arithmetic of a
        // one-element vector rather than measured, mutating either field to a
        // constant left every test green. The two `ΔAS` values are also
        // deliberately UNEQUAL, so the two reads' posteriors differ and a
        // join that swapped them cannot pass by coincidence.
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
        }
    }

    /// A two-molecule graph: `PLAIN` has one candidate, `CULLED` has two at
    /// EQUAL `AS` — so its posterior is exactly 0.5 / 0.5.
    ///
    /// The tie is exact and stays exact under the M-step, which is what makes
    /// this fixture usable: `CULLED` splits its weight 50/50 between tx1 and
    /// tx2, so the two receive identical emission-parameter updates and
    /// identical abundances, and the only term left to separate them is the
    /// one that is zero for both. Re-running at `min_weight = 0.9` then culls
    /// the read outright: both weights are below the cutoff, the emit loop
    /// drops both, and the read produces no profile row.
    ///
    /// `PLAIN` is in the same graph so "CULLED emitted no row" is not
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
        }
    }

    /// The P0 this whole output path could have shipped: `final_weights` is
    /// indexed by AMBIGUOUS RANK, not molecule index, so pairing it with
    /// `graph.molecules` positionally attributes each read's top-1 to the
    /// WRONG read. `ambiguous_indices` here is `[1, 2]`, so a positional zip
    /// would hand molecule 0's posterior to `UNAMB` and shift both ambiguous
    /// reads by one.
    ///
    /// M9 -- tie-break determinism. The argmax fold at the output stage uses
    /// strict `>`, so on an EXACT posterior tie the FIRST candidate in pool
    /// order wins. That rule was asserted in a comment which also claimed the
    /// mutation "leaves the suite green" -- a mutation nobody had run. It did
    /// leave the suite green: 70 passed, 0 failed. So the rule was unpinned,
    /// and a refactor could have changed which transcript a tied read is
    /// assigned to without any test objecting.
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
    /// reason, which is exactly how an earlier draft of a nearby fixture
    /// (`mixed_unambiguous_then_ambiguous`) was caught being vacuous.
    ///
    /// MUTATION: `w > acc.1` -> `w >= acc.1` must turn THIS test red, because
    /// `>=` lets the later equal element overwrite the earlier one and the
    /// winner becomes `tx1`. If it does not, the tie is unreachable through
    /// this path and the assertion below is measuring something else.
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

        // The posteriors must be genuinely NON-degenerate. These are the two
        // fields the whole cross-run comparison rests on, and on a
        // one-element weight vector both are pinned by arithmetic rather than
        // measured: `posterior_entropy` returns 0.0 for a one-element vector
        // and the lone weight is 1.0. Asserting only the top-1 above left
        // `top_weight: top_w -> 0.5` and `entropy: ... -> 0.0` both GREEN.
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

    /// `top_tx_id == Some(..)` must IMPLY that the read has a profile row, and
    /// nothing pinned that before: `read_assign[mol_idx]` was written in the
    /// ambiguous branch, which runs BEFORE the emit loop's `if w < min_weight {
    /// continue }`. A read whose every candidate fell below the cutoff
    /// therefore carried a real `top_tx_id` while emitting ZERO profile rows —
    /// a numerator entry with no denominator, in a table whose whole purpose is
    /// a cross-run join.
    ///
    /// This is the ONE shape that is absent from both tables. A ghost read
    /// (empty `weights`) is also absent from the profile but is deliberately
    /// recorded with `top_tx_id = None` — see
    /// `test_ghost_read_is_recorded_as_unassignable_and_absent_from_the_profile`.
    /// So the claim is one-directional and the converse is not claimed.
    ///
    /// Reachable with `--min-weight 0.9` on the 0.5 / 0.5 read below. At the
    /// shipped 0.01 it needs a read spread over enough survivors that all of
    /// them sit under the cutoff, which a high-fan-out repeat region supplies
    /// (`candidate_count` is `u16`).
    ///
    /// Mutation target: dropping the `any(w >= min_weight)` half of the
    /// assignment's `if` makes this test red.
    #[test]
    fn test_read_assignment_is_omitted_when_min_weight_drops_every_candidate() {
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
            1,
            "only PLAIN (w = 1.0) clears min_weight = 0.9; the 0.5/0.5 read must \
             emit no profile row at all: {rows:?}"
        );
        assert_eq!(rows[0].0, "PLAIN", "the surviving row must be PLAIN's");

        // The actual invariant, asserted on the data rather than argued from
        // the code: every read in the assignments table has a profile row.
        assert!(
            !assigns.iter().any(|a| a.read_id == "CULLED"),
            "a read with no profile row must not appear in the read-assignments \
             table — the join numerator would exceed its denominator: {assigns:#?}"
        );
        assert_eq!(
            assigns.len(),
            1,
            "only PLAIN belongs in the table: {assigns:#?}"
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
    }

    /// A GHOST read -- both candidates poly(A)-incompetent, so the E-step
    /// returns an EMPTY weight vector -- has no competent candidate, so the
    /// profile legitimately has no row for it. It IS still recorded in the
    /// assignments table, with `top_tx_id = None`.
    ///
    /// That is the distinction a cross-run comparison needs: "the solver
    /// looked at this read and could not assign it" versus "this read was not
    /// in this run at all". Dropping the row would make the two identical, and
    /// `top_tx_id: None` is the only place the difference is visible.
    ///
    /// Nothing covered this before. It is the `None` arm of the invariant
    /// `top_tx_id == Some(..) IMPLIES that read has a profile row` -- the
    /// converse is deliberately NOT claimed, and this test pins both halves of
    /// that: `GHOST` is present-and-`None` in the assignments table and absent
    /// from the profile.
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
    /// `is_competent`, so the named transcript's profile row exists only
    /// because the E-step pushed `-inf` instead of a real log-likelihood for
    /// it. That is one predicate -- `if !p.is_competent || p.alpha <= 0.0` in
    /// the E-step -- and nothing pinned it.
    ///
    /// The fixture makes the failure observable rather than hypothetical. `BAD`
    /// is present in the graph and has an emission row, so nothing about it is
    /// missing; it is simply marked not poly(A)-competent. Its `mu` is the
    /// read's EXACT observed length, so if the filter stops firing it wins the
    /// argmax outright -- `GOOD` is a poor fit and loses on the numbers alone.
    /// The output stage's own `!p.is_competent` check then keeps `BAD` out of
    /// the profile, so the read ends up naming, as its top-1, a transcript it
    /// has no profile row for.
    ///
    /// This is the per-TRANSCRIPT half of the invariant the per-read checks
    /// above cover only from the outside: those ask "does the named read have
    /// some profile row", which stays satisfied here because `GOOD` supplies
    /// one. The claim that actually matters for the cross-run join is that the
    /// named transcript is the one the read was assigned to, so that is what is
    /// asserted.
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

    /// behaviour is `top_b.append_option(...)`. The output stage reaches that
    /// NULL on every ghost read, so this is not a corner case — but the writer
    /// itself was never called by a test, so a change to `append_option` or to
    /// the column's nullability would have gone unnoticed.
    ///
    /// The three things a NULL can be confused with, all checked: an empty
    /// string (a transcript that does not exist), the index-0 transcript (a
    /// real transcript the read was never assigned to), and a dropped row.
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
    /// candidate has no emission row", which is how the ghost case below is
    /// expressed.
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
                })
            })
            .collect();
        (graph, params)
    }

    /// Direct unit test of the poly(A)-degradation predicate. The headline
    /// number the CLI prints is `n_polya_zero_power / n_ambiguous`, and it was
    /// computed by this function with nothing testing it: replacing the body
    /// with `false` left the whole suite green.
    ///
    /// The cases that matter are the ones a plausible re-implementation gets
    /// wrong. A "do the mus match?" comparison alone would pass the headline
    /// fixture — the fallback shares BOTH components — while firing `true` on
    /// every read whose transcripts have a common `mu` and different widths,
    /// which is the L2 vs L1 case and is a *false* zero-power report on a read
    /// poly(A) does discriminate. `sigma2` is asserted separately below for
    /// exactly that reason.
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

        // 1b. A SINGLE candidate is trivially "all candidates agree" and is
        //     also `true` here. This is the VACUOUS case the whole change
        //     exists to make visible: nothing was compared, so the flag is
        //     true for want of a counterexample. The COUNT is what
        //     distinguishes it, and it is asserted here so no future caller can
        //     read `true` as "poly(A) was measured and cancelled".
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

        // 3b. A disagreement in the THIRD candidate must not be masked by the
        //     first two agreeing — the comparison is against the first
        //     candidate, not against the immediately preceding one.
        //
        //     THE COUNT IS 3, NOT 2, AND THAT IS THE POINT. An earlier version
        //     of this function returned `false` on the first disagreement, so a
        //     loop that counted as it went would report 2 here. That
        //     undercount lands only on DISAGREEING reads — the ones where
        //     poly(A) did something — which would bias `emission coverage`
        //     downward while leaving the zero-power headline untouched. Anyone
        //     who restores the early return fails this line and nothing else.
        {
            let (g, p) = zero_power_probe(&[Some((MU, S2)), Some((MU, S2)), Some((MU, S2 + 0.4))]);
            assert_eq!(
                polya_emission_count_and_identity(&g, &g.molecules[0], &p),
                (3, false),
                "the count must cover the WHOLE candidate range even after a \
                 disagreement; 2 here means the early return is back"
            );
        }

        // 4. Empty candidate list. There is no candidate to compare against, so
        //    the code returns `false`: the docstring says a read with nothing to
        //    compare returns false rather than claiming zero power vacuously.
        //    `false` is the SAFE answer — the alternative would inflate the
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
        //     mismatch: a ghost transcript has no (mu, sigma2) to disagree
        //     with. The remaining real candidate is what the read is judged on,
        //     and it is the same ghost the read is already charged for in
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

        // 5. FLOATING-POINT EQUALITY. The comparison is EXACT `==`, and these
        //    two cases pin that: the fallback hands out bit-identical COPIES of
        //    one fit, so anything looser than `==` can only add false
        //    positives. Both sub-cases below therefore return FALSE — one ULP
        //    is already "different" — and that is asserted rather than assumed,
        //    so a future tolerance change turns this test red instead of
        //    silently changing the headline number.
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
    /// The unit test above pins the predicate; this pins the four links around
    /// it — the counter is incremented from the AMBIGUOUS loop, it reads the
    /// LAST E-STEP's `last_estep_params`, it lands in the struct, and
    /// unambiguous reads are excluded from it.
    ///
    /// It previously read the POST-EM `params`, and said so here. That was the
    /// stale-μ defect: on the `max_iter`-exhaustion path those are one M-step
    /// past the state that produced the posterior. This fixture still passes
    /// either way — `poly_a_weight = 0` freezes `mu`/`sigma2` (see
    /// `test_poly_a_weight_zero_freezes_mu_and_sigma2_instead_of_dragging_them`),
    /// and the predicate reads ONLY `mu`/`sigma2`, so pre- and post-M-step are
    /// identical for the fields it looks at. That is exactly why this test
    /// could never have caught the defect, and why the positive arm in
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
    /// The unit test above pins the predicate and the test above it pins that
    /// the counter reaches `TelEmStats`. Neither can see the difference the
    /// whole change is about, because both fixtures give every candidate an
    /// emission row. This one does not: `tx6` is deliberately absent from the
    /// emission parquet, so reads can have fewer present candidates than they
    /// have candidates — which is the condition that makes `zero-power_all`
    /// report a cancellation that never happened.
    ///
    /// The fixture's whole job is one number pair:
    ///
    /// ```text
    ///   n_polya_zero_power          == 2   (ZERO_2WAY and SINGLE_THEN_GHOST)
    ///   n_polya_zero_power_resolved == 1   (ZERO_2WAY only)
    /// ```
    ///
    /// `SINGLE_THEN_GHOST` is the read the split exists for: two candidates,
    /// one of them has no emission row, so nothing was ever compared, and
    /// `zero-power_all` counts it anyway because that is what it has always
    /// done. If the two counters ever agree on this fixture, the exclusion is
    /// STEP 0, end to end. Every dropped read lands in exactly one bucket, and
    /// the buckets are the ones the CLI prints.
    ///
    /// What this pins that the census unit test cannot: that the CALL SITE
    /// routes each cause to the field the CLI reads, and that a read which
    /// loses only SOME of its candidates increments NOTHING. The second half is
    /// the load-bearing half — a solver that counted every read with any dead
    /// candidate would produce plausible-looking numbers for every cause and
    /// still be wrong, because the number the padding experiment turns on is
    /// "reads that padding would rescue", not "reads with something wrong".
    ///
    /// `max_iter = 1` on purpose. `alpha` only reaches 0 through the M-step, and
    /// a transcript can only win no mass if it was already dead, so
    /// `n_ghost_read_all_alpha_zero` is NOT reachable through the solver at any
    /// iteration count. It is pinned in
    /// `test_candidate_availability_keeps_four_death_causes_apart` instead, and
    /// asserting 0 here is what documents that the gap is known rather than
    /// accidental.
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
    /// Two arms, ONE shared emission parquet, ONE shared graph, differing ONLY
    /// in `poly_a_weight`. Both arms' E-steps therefore consume the IDENTICAL
    /// `(mu, sigma2)` — the loaded values, before any M-step has run. So the
    /// zero-power verdict is a property of the data, and the two arms MUST
    /// report the same count. The M-step cannot change that: it runs after the
    /// E-step that produced the posterior, and the telemetry is about the
    /// posterior.
    ///
    /// Pre-fix this fails, and with a SIGN rather than merely "not equal":
    /// reading post-M-step `params` lets the M-step's τ₀=2 shrinkage pull the
    /// tied fits apart (they share a loaded `mu_cohort`, but their accumulated
    /// `S1` differ), so the ON arm counts STRICTLY FEWER than the OFF arm. An
    /// assertion of the form `assert_ne!` would also have passed on a fixture
    /// that broke for an unrelated reason; `assert_eq!` on a known value pins
    /// the direction, so a future change that makes the arms differ the other
    /// way is also caught.
    ///
    /// `max_iter = 1` is REQUIRED, and is the whole point: it is the
    /// `max_iter`-exhaustion exit path, the one where the M-step runs last and
    /// leaves `params` a full M-step ahead of the posterior. On a converged run
    /// the loop `break`s BEFORE the M-step and this defect is invisible — which
    /// is why it survived several rounds of converged-run reporting.
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
        // ASYMMETRIC on purpose. The alignment term `ln A_rk = -ΔAS/λ` gives
        // tx0 strictly more posterior mass than tx1 on every read, so their
        // accumulated S1 DIFFER and one M-step pulls their mu apart. A fixture
        // with equal AS gives both candidates identical accumulators, the
        // shrinkage moves them to the identical place, and the M-step cannot
        // separate the tie — which is why the first draft of this test passed
        // against the unfixed code.
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
        // The assertion above went red when the TELEMETRY read `params`, but
        // reverting the `output_emission_params` half of the same fix turned
        // ZERO tests red — the pre-existing tests on that out-param all straddle
        // the difference, so the suite could not tell the two states apart. That
        // left the "one snapshot, two consumers ... by construction" comment an
        // unbacked claim, which is the failure mode this repo has been bitten by
        // repeatedly. This is the mutation that makes it true.
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
