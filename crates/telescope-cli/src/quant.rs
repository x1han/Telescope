//! The `quant` subcommand: Module 2 (evidence) -> Module 3 (TEL-EM).
//!
//! `QuantArgs` is a clap `Args` struct rather than a set of positional
//! parameters so that BOTH binaries can declare the subcommand from ONE
//! definition. A research binary that retyped the 18 flags would compile, run,
//! and silently be a different model the moment a default drifted; the frozen
//! defaults below are the same constants the shipped binary uses, and there is
//! now only one copy of them.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use telescope_core::{em, evidence};

use crate::{
    maybe_write_filter_ledger, pct, print_filter_ledger, ALN_WEIGHT_LAMBDA_DEFAULT,
    FIVE_PRIME_WEIGHT_DEFAULT, FIVE_PRIME_WINDOW_DEFAULT, POLY_A_SOURCE_HELP,
    POLY_A_WEIGHT_DEFAULT, PRUNE_MAX_DELTA_AS_DEFAULT,
};

#[derive(Args, Debug)]
pub struct QuantArgs {
    /// Path to AIDRS-aligned transcriptome BAM.
    #[arg(long)]
    bam: PathBuf,

    /// Path to `telescope.emission.parquet` (from `build-emission`).
    #[arg(long)]
    emission: PathBuf,

    /// Sample identifier (e.g. `E10.5_Brain`). Used as `sample_id` column.
    #[arg(long)]
    sample_id: String,

    /// Output `telescope.profile.parquet`.
    #[arg(long)]
    output: PathBuf,

    /// 3' TES boundary slack in bp (default 35).
    #[arg(long, default_value_t = 35)]
    tes_slack: usize,

    // The ruling specifies this spelling, and the FATAL message in
    // `evidence.rs` quotes it -- the flag and the error that names it have
    // to agree, or the error tells the user to type something that does not
    // exist. The lowercase form is accepted as an alias for the same
    // reason the flag itself is spelled in both cases: a quant command
    // invoked from a shell script should not fail on letter case alone.
    #[arg(
        long = "polyA-source",
        alias = "polya-source",
        default_value = "bam_tag:pt",
        help = POLY_A_SOURCE_HELP
    )]
    poly_a_source: String,

    /// Write the record-level filter ledger here as JSON
    /// (`telescope.filter_ledger/v1`). See `extract-evidence --ledger`.
    ///
    /// `quant` runs the same Module-2 filter chain as `extract-evidence`,
    /// so its ledger describes the same population; the profile it writes
    /// is built from those records, and without this the only trace of the
    /// drop is a stderr block that no later script can read back.
    #[arg(long)]
    ledger: Option<PathBuf>,

    /// EM log-likelihood convergence tolerance (per-molecule average LL).
    #[arg(long, default_value_t = 1e-6)]
    tol: f64,

    /// Maximum EM iterations (safety cap).
    #[arg(long, default_value_t = 1000)]
    max_iter: usize,

    /// Scale λ of the F-EM-4 alignment term `ln A_rk = (AS_rk − max_j AS_rj) / λ`
    /// (default: 20.0). Must be > 0. Larger λ = weaker alignment prior.
    ///
    /// PROVENANCE — this default was 1000.0 until the 5'-anchoring work, and
    /// that value is superseded. 1000 came from a ONE-gene (PGAM1) grid on
    /// the COPD29 pilot BAM whose oracle was minimap2's own primary
    /// alignment, i.e. the term was being judged against the aligner that
    /// produced it. More decisively, at λ = 1000 every evidence term
    /// compresses to ≤ 0.1 nat, so the EM was barely using the alignment
    /// evidence it exists to weigh. At λ = 20 with the `ΔAS > 50` prune the
    /// term spans `[−2.5, 0]` nats, which is the range the 5' weight of 1.8
    /// was chosen to compete inside.
    ///
    /// No value of λ is validated library-wide. Full curve, the superseded
    /// single-gene calibration, and the frozen-parameter table:
    /// `workspace/notes/EXPERIMENT.md`.
    #[arg(long, default_value_t = ALN_WEIGHT_LAMBDA_DEFAULT)]
    aln_weight_lambda: f64,

    /// Weight β of the 5'-anchoring term `ln_5p_k = β` when the read's
    /// observed 5' gap `g_k` is within `--five-prime-window` of the
    /// candidate transcript's 5' end, else 0 (default: 1.8). Must be >= 0.
    ///
    /// `g_k` is the start of the read's BEST alignment to candidate k, in
    /// candidate k's OWN cDNA coordinates — not a genomic span.
    ///
    /// STRAND IS A KNOWN GAP, not a handled case. The frozen spec was
    /// validated on a forward-only population and the reverse fraction is
    /// 0.0000 across 200,000 mapped records in each of the AIDRS / Bambu /
    /// IsoQuant BAMs, so `g_k = s_k − 1` is applied unconditionally here.        /// If the Bambu/IsoQuant general path (Phase F1/F2) ever carries
    /// antisense alignments this term would silently read the wrong end of
    /// the molecule; that needs a separately validated definition, not an
    /// inline guess. See `test_reverse_alignment_uses_the_forward_formula_as_a_documented_gap`.
    ///
    /// **Set to 0 to switch the term off.** That makes `ln_5p ≡ 0` exactly,
    /// so it drops out of the softmax exactly; it is the correct way to run
    /// the Test A / Test B ablation. Note that redrawing each read's 5' start
    /// (Test B) is a different intervention and is not the same as turning
    /// the weight to 0 — a redraw leaves the term active but decorrelated.
    ///
    /// PROVENANCE — 7.9 was tried and rejected: with the `ΔAS > 50` prune
    /// and λ = 20 the alignment term spans only `[−2.5, 0]` nats, so a
    /// 7.9-nat bonus is an unconditional override. 1.8 competes instead.
    #[arg(long, default_value_t = FIVE_PRIME_WEIGHT_DEFAULT)]
    five_prime_weight: f64,

    /// 5' window in bp for `--five-prime-weight`: the bonus fires iff
    /// `g_k <=` this value (default: 20). INCLUSIVE. The comparison is on
    /// the 1-based start `s_k` minus one, which is why 20 and 21 are
    /// different answers and the boundary is tested.
    #[arg(long, default_value_t = FIVE_PRIME_WINDOW_DEFAULT)]
    five_prime_window: u32,

    /// Scale on the poly(A) mixture term. 1.0 is the shipped model.
    /// **0.0 removes the poly(A) evidence entirely.**
    ///
    /// It does NOT leave an alignment-only model. `ln w_rk = ln alpha_k +
    /// poly_a_weight * ln P(polyA|k) + ln a_rk + ln 5p_k`, so the mixture
    /// prior `ln alpha_k` survives at any weight, including 0. With both
    /// this and `--five-prime-weight` at 0 the likelihood is
    /// `ln alpha_k + ln a_rk` — abundance prior PLUS alignment. Calling that
    /// arm "alignment-only" is wrong, and the prior cannot be dropped from
    /// the outside either: pinning `mu`/`sigma2` in the emission parquet
    /// does nothing, because `alpha` is re-estimated by the M-step and
    /// would still leak into every candidate's score. A genuinely
    /// prior-free baseline is a different estimator (e.g. argmax-`AS`
    /// assignment), not an ablation of this one.
    #[arg(long, default_value_t = POLY_A_WEIGHT_DEFAULT)]
    poly_a_weight: f64,

    /// Drop candidates whose `AS:i` is more than this far below the read's
    /// best (default: 50). Must be >= 0; the comparison is strictly
    /// greater-than, so `ΔAS = 50` survives. The read's own best candidate
    /// has `ΔAS = 0` against itself and can never be pruned.
    ///
    /// At λ = 20 the alignment term is too weak to delete candidates on its
    /// own — reaching the `min_weight` cutoff costs `ΔAS > 4.595 × 20 = 91.9`,
    /// past this threshold — so the deletion responsibility sits here. Use
    /// a very large value to switch the prune off.
    #[arg(long, default_value_t = PRUNE_MAX_DELTA_AS_DEFAULT)]
    prune_max_delta_as: i32,

    /// Output-stage posterior truncation threshold (default: 0.01). A
    /// (read × candidate) row is emitted only if its posterior is >= this,
    /// and each read's surviving rows are renormalised to 1.0.
    ///
    /// THIS IS NOT COSMETIC FOR A/B WORK. Two arms of an ablation can emit
    /// different row counts on the same reads — measured on GNG11, 2726 rows
    /// with the 5' term on vs 1816 with it off — because the arm that shifts
    /// mass onto a candidate is the arm that clears the cutoff. The arms are
    /// then not scored on the same support. Any ablation whose arms differ in
    /// row count must be re-run at `--min-weight 0` (no truncation at all)
    /// before its delta is quoted. 0.0 is legal; values >= 1.0 are rejected.
    #[arg(long, default_value_t = 0.01)]
    min_weight: f64,

    /// Optional TSV with header `bam_id\toriginal_id` for ID translation
    /// (replaces offline Python workaround for AIDRS re-keying).
    #[arg(long)]
    tx_id_mapping: Option<PathBuf>,

    /// Print EM per-iter trace (log-L + mean entropy) to stderr.
    #[arg(long)]
    debug: bool,

    /// Also write a per-READ posterior summary
    /// (`read_assignments.parquet`: read_id, n_candidates_before_cutoff,
    /// top_tx_id, top_weight, entropy).
    ///
    /// The profile parquet is per (read, candidate) PAIR and carries no
    /// per-read summary, so "which transcript did this read end up on, and
    /// how sure was it" is not recoverable from it. This exists for
    /// comparing two runs read-by-read (e.g. an alignment re-run at a
    /// different `-N`), which needs a per-read key.
    ///
    /// `top_tx_id` is the transcript NAME, not a pool index, so the table
    /// can be joined against another run's table. It is null for a read
    /// with no nameable transcript. `top_weight` is the raw E-step weight
    /// and is deliberately NOT the profile's `posterior_weight` (that one
    /// is post-truncation and renormalised).
    #[arg(long)]
    read_assignments: Option<PathBuf>,

    /// Also write the `(mu, sigma2)` the FINAL E-step consumed
    /// (`--dump-emission-params <path>.parquet`).
    ///
    /// WHY. There are two different μ in this program and nothing on disk
    /// distinguished them: the one LOADED from `--emission`, and the one the
    /// M-step FITTED. The posterior is computed from whichever the last
    /// E-step saw, so an analysis that bins reads by Δμ must bin by that
    /// one. Before this flag the only μ on disk was the loaded one, which is
    /// the right value only when the last E-step is the FIRST one — i.e. at
    /// `--max-iter 1`, or on a graph with no contested reads. On any longer
    /// run the M-step has moved μ and the loaded value is wrong, whether or
    /// not the run converged.
    ///
    /// This is a DIAGNOSTIC / AUDIT SNAPSHOT. It is NOT a re-loadable
    /// model, and the previous version of this help text said it was --
    /// wrongly. Do not pass it back as `--emission`.
    ///
    /// Why it cannot be re-loaded: the writer emits five of the six columns
    /// `load_emission_parameters` requires (`transcript_id`, `gene_id`,
    /// `is_polyA_competent`, `mu_logPAL`, `sigma2_logPAL`) plus its own
    /// audit columns, but it does NOT emit `n_cohort_flnc`. The loader reads
    /// that column as REQUIRED (`.context("Missing 'n_cohort_flnc'")?`, on
    /// purpose, so a hand-built model cannot silently classify every row as
    /// anchored), so the round trip dies there rather than producing a
    /// number. Pinned by
    /// `test_the_sidecar_cannot_be_reloaded_as_an_emission_model`, which
    /// fails if `n_cohort_flnc` is ever added -- at which point this text
    /// and the prior-reanchoring hazard below have to be re-decided.
    ///
    /// And re-loading would be wrong even if it parsed. `load_emission_
    /// parameters` sets `mu_cohort = mu_logPAL` and `sigma2_cohort =
    /// sigma2_logPAL`, so a second pass would re-anchor BOTH shrinkage
    /// priors to the already-fitted values and apply τ₀ and 2β₀ a second
    /// time. The columns are present so an analyst can SEE that; do not
    /// feed them back in.
    ///
    /// The published profile and read-assignments schemas are unchanged.
    #[arg(long)]
    dump_emission_params: Option<PathBuf>,
}

/// Run the `quant` subcommand.
///
/// `sigma2_dof_offset`: `None` means "the shipped model's offset", i.e.
/// `SIGMA2_DOF_OFFSET`. `Some(c)` is reachable ONLY from the research binary;
/// the production `telescope` never passes it, which is what keeps the frozen
/// `Model A c = 2` guarantee a compile-time property of the shipped artifact
/// rather than a runtime promise.
pub fn run_quant(args: QuantArgs, sigma2_dof_offset: Option<f64>) -> Result<()> {
    let QuantArgs {
        bam,
        emission,
        sample_id,
        output,
        tes_slack,
        tol,
        max_iter,
        aln_weight_lambda,
        five_prime_weight,
        poly_a_weight,
        five_prime_window,
        prune_max_delta_as,
        min_weight,
        tx_id_mapping,
        poly_a_source,
        debug,
        read_assignments,
        dump_emission_params,
        ledger,
    } = args;

    let poly_a = evidence::parse_poly_a_source(&poly_a_source)?;
    // `--polyA-source none` fills `log_pal` with a placeholder, not a
    // measurement. The ruling requires the inconsistent pairing to be
    // UNEXPRESSIBLE rather than merely warned about, so the weight is
    // forced here instead of erroring on a value the user may have set
    // for a different reason (a global default, a wrapper script).
    let poly_a_weight = if matches!(poly_a, evidence::PolyASource::None) {
        if poly_a_weight != 0.0 {
            eprintln!(
                "[telescope] WARNING: --polyA-source none forces --poly-a-weight 0 \
                 (was {poly_a_weight}); the placeholder is not a measurement and \
                 would otherwise enter the E-step as a constant emission."
            );
        }
        0.0
    } else {
        poly_a_weight
    };
    if !(aln_weight_lambda > 0.0) {
        anyhow::bail!(
            "--aln-weight-lambda must be > 0 (got {}); it scales \
             ln A_rk = (AS_rk − max_j AS_rj)/λ, so 0 and NaN would emit \
             NaN posterior weights",
            aln_weight_lambda
        );
    }
    // NaN-safe: `!(w >= 0.0)` would let NaN through, and a NaN weight
    // propagates into every posterior weight. 0.0 is legal and is the
    // documented off switch for the 5' term.
    //
    // clippy's `neg_cmp_op_on_partial_ord` wants this rewritten with
    // `partial_cmp`, which would be a REGRESSION: the point of the
    // negated form is that it is false for NaN. Do not "fix" this.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(five_prime_weight >= 0.0) {
        anyhow::bail!(
            "--five-prime-weight must be >= 0 and non-NaN (got {}); it is the \
             log-space bonus ln_5p awarded inside the 5' window, so 0.0 is the \
             valid off switch but a negative weight inverts the term and NaN \
             would emit NaN posterior weights",
            five_prime_weight
        );
    }
    if prune_max_delta_as < 0 {
        anyhow::bail!(
            "--prune-max-delta-as must be >= 0 (got {}); a negative threshold \
             would also drop the read's own best candidate, emptying every \
             ambiguous read",
            prune_max_delta_as
        );
    }
    eprintln!("[telescope] Quant (M2 → M3)");
    eprintln!("  bam:        {}", bam.display());
    eprintln!("  emission:   {}", emission.display());
    eprintln!("  sample_id:  {}", sample_id);
    eprintln!("  output:     {}", output.display());
    eprintln!("  tes_slack:  {} bp", tes_slack);
    eprintln!("  tol:        {:.1e}", tol);
    eprintln!("  max_iter:   {}", max_iter);
    eprintln!("  aln_weight_lambda: {}", aln_weight_lambda);
    eprintln!("  five_prime_weight: {}", five_prime_weight);
    eprintln!("  five_prime_window: {} bp", five_prime_window);
    eprintln!("  prune_max_delta_as: {}", prune_max_delta_as);
    if aln_weight_lambda == ALN_WEIGHT_LAMBDA_DEFAULT {
        eprintln!(
            "  WARNING: --aln-weight-lambda is at its FROZEN default (20.0).\n\
             \x20            It was set by the 5'-anchoring ablation, not by the \
             earlier one-gene (PGAM1) calibration that produced λ=1000 — that \
             value is superseded and compressed every evidence term to \
             <= 0.1 nat.\n\
             \x20            No value of λ is validated library-wide."
        );
    }
    if let Some(ref p) = tx_id_mapping {
        eprintln!("  tx_id_mapping: {}", p.display());
    }
    eprintln!("  debug:      {}", debug);

    // Parse optional `bam_id\toriginal_id` TSV (header row skipped).
    // This replaces the previous offline Python re-keying workaround.
    let id_mapping: Option<std::collections::HashMap<String, String>> = if let Some(ref p) =
        tx_id_mapping
    {
        let f = std::fs::File::open(p)
            .with_context(|| format!("Failed to open --tx-id-mapping {:?}", p))?;
        let reader = std::io::BufReader::new(f);
        use std::io::BufRead;
        let mut map = std::collections::HashMap::new();
        for (i, line) in reader.lines().enumerate() {
            let l = line.with_context(|| format!("Failed to read line {} from {:?}", i + 1, p))?;
            if i == 0 && l.starts_with("bam_id") {
                continue; // skip header
            }
            let parts: Vec<&str> = l.split('\t').collect();
            if parts.len() >= 2 {
                let bam_id = parts[0].to_string();
                let orig = parts[1].to_string();
                if !bam_id.is_empty() && !orig.is_empty() {
                    map.insert(bam_id, orig);
                }
            }
        }
        eprintln!("[telescope]    tx_id_mapping entries: {}", map.len());
        Some(map)
    } else {
        None
    };

    eprintln!("[telescope]  M2: extract evidence…");
    eprintln!("[telescope]  polyA-src:  {}", poly_a_source);
    let mut graph = evidence::extract_evidence_with(&bam, tes_slack, &poly_a)
        .context("Module 2 extract_evidence failed")?;
    // Apply optional id mapping to transcript names (BAM @SQ → emission parquet keys).
    if let Some(ref map) = id_mapping {
        for name in &mut graph.transcript_names {
            if let Some(mapped) = map.get(name) {
                *name = mapped.clone();
            }
        }
    }
    eprintln!("[telescope]    molecules:      {}", graph.molecules.len());
    print_filter_ledger(&graph.stats, "[telescope]    ");
    maybe_write_filter_ledger(ledger.as_ref(), &graph.stats)?;
    eprintln!(
        "[telescope]    transcripts:    {}",
        graph.transcript_names.len()
    );

    // Phase A: candidate_count histogram (telemetry).
    let cand_hist = graph.candidate_count_histogram();
    let n_total_reads = graph.molecules.len() as f64;
    eprintln!("[telescope]    candidate_count histogram:");
    if n_total_reads == 0.0 {
        eprintln!("[telescope]      (no molecules)");
    } else {
        for (count, n_reads) in &cand_hist {
            eprintln!(
                "[telescope]      candidate={:>3}:  {:>10}  ({:5.1}%)",
                count,
                n_reads,
                *n_reads as f64 / n_total_reads * 100.0
            );
        }
    }

    eprintln!("[telescope]  M3: TEL-EM…");
    eprintln!(
        "[telescope]    A_rk ACTIVE on ambiguous reads: ln A_rk = (AS_rk − max_j AS_rj) / λ, λ = {}",
        aln_weight_lambda
    );
    if five_prime_weight > 0.0 {
        eprintln!(
            "[telescope]    5' term ACTIVE on ambiguous reads: ln_5p = {} if g_k <= {} bp, else 0",
            five_prime_weight, five_prime_window
        );
    } else {
        eprintln!("[telescope]    5' term OFF (--five-prime-weight 0).");
    }
    if prune_max_delta_as < i32::MAX {
        eprintln!(
            "[telescope]    ΔAS prune ACTIVE: drop candidates with ΔAS > {} vs the read's best AS",
            prune_max_delta_as
        );
    } else {
        eprintln!("[telescope]    ΔAS prune OFF (--prune-max-delta-as at i32::MAX).");
    }
    // The calibration claim below belongs to the shipped default only;
    // printing it next to a user-supplied λ would attribute someone
    // else's value to a calibration that did not produce it.
    if aln_weight_lambda == ALN_WEIGHT_LAMBDA_DEFAULT {
        eprintln!(
            "[telescope]    λ default {} comes from the 5'-anchoring ablation, which put the \
             alignment term in the same [−2.5, 0] nat range the 5' bonus competes in. \
             The earlier one-gene (PGAM1) λ=1000 calibration is SUPERSEDED. No value of \
             λ is validated library-wide.",
            ALN_WEIGHT_LAMBDA_DEFAULT
        );
    } else {
        eprintln!(
            "[telescope]    λ = {} is user-supplied; the frozen default (λ = {}) does not \
             apply. No value of λ is validated library-wide.",
            aln_weight_lambda, ALN_WEIGHT_LAMBDA_DEFAULT
        );
    }
    let mut em_trace: Vec<(usize, f64, f64)> = Vec::new();
    let trace_sink: Option<&mut Vec<(usize, f64, f64)>> =
        if debug { Some(&mut em_trace) } else { None };
    // Collected only when the caller asked for the per-read table. The
    // Vec is always declared (it is empty otherwise and is never
    // filled), which keeps the borrow a plain `&mut` for the duration
    // of the call rather than a borrow of a temporary.
    let mut read_sink: Vec<em::ReadAssignment> = Vec::new();
    // Same idiom as `read_sink`: declared unconditionally so the borrow
    // is a plain `&mut` for the call rather than a borrow of a
    // temporary, and filled only when the flag asked for it.
    let mut params_sink: Vec<Option<em::TranscriptParam>> = Vec::new();
    let stats = em::solve_and_write_profile_with_dof_offset(
        graph,
        &emission,
        &sample_id,
        &output,
        tol,
        max_iter,
        aln_weight_lambda,
        five_prime_weight,
        poly_a_weight,
        five_prime_window,
        prune_max_delta_as,
        min_weight,
        trace_sink,
        if dump_emission_params.is_some() {
            Some(&mut params_sink)
        } else {
            None
        },
        if read_assignments.is_some() {
            Some(&mut read_sink)
        } else {
            None
        },
        sigma2_dof_offset.unwrap_or(em::SIGMA2_DOF_OFFSET),
    )
    .context("Module 3 solve_and_write_profile failed")?;
    if let Some(path) = dump_emission_params {
        let rows = std::mem::take(&mut params_sink);
        anyhow::ensure!(
            !rows.is_empty(),
            "--dump-emission-params was set but the solve produced no params"
        );
        em::write_emission_params_parquet(&rows, &path)?;
        eprintln!(
            "[telescope]    E-step emission params:  {} rows -> {}",
            rows.iter().filter(|r| r.is_some()).count(),
            path.display()
        );
    }
    if let Some(path) = read_assignments {
        let rows = std::mem::take(&mut read_sink);
        anyhow::ensure!(
            !rows.is_empty(),
            "--read-assignments was set but the solve produced no rows"
        );
        em::write_read_assignments_parquet(&rows, &path)?;
        eprintln!(
            "[telescope]    read assignments:       {} rows -> {}",
            rows.len(),
            path.display()
        );
    }
    eprintln!(
        "[telescope]    n_unambig / sample:      {}",
        stats.n_unambiguous
    );
    eprintln!(
        "[telescope]    n_ambig / EM:           {}",
        stats.n_ambiguous
    );
    eprintln!("[telescope]    EM iterations:          {}", stats.n_em_iter);
    // Drop taxonomy. Every number in the two groups below is a count
    // of READS, and each group is printed under the denominator that
    // population actually has: a single shared denominator would be
    // meaningless, because an unambiguous read and a contested read
    // cannot be dropped for the same reason at the same site. The
    // tripwire after them is a ROW count and is deliberately outside
    // both groups.
    let n_ghost = stats.n_dropped_no_competent_candidates;
    eprintln!("[telescope]    reads dropped, no profile row:");
    eprintln!(
        "[telescope]      unambiguous, denominator {}:",
        stats.n_unambiguous
    );
    eprintln!(
        "[telescope]        no emission row:              {}",
        stats.n_unambiguous_dropped_missing_emission
    );
    eprintln!(
        "[telescope]        is_polyA_competent == false: {}",
        stats.n_unambiguous_dropped_incompetent
    );
    eprintln!(
        "[telescope]      contested with no posterior at all (\"ghost reads\"), denominator {}:",
        n_ghost
    );
    eprintln!(
        "[telescope]        every candidate had no emission row:  {}",
        stats.n_ghost_read_all_missing
    );
    eprintln!(
        "[telescope]        every candidate incompetent:          {}",
        stats.n_ghost_read_all_incompetent
    );
    eprintln!(
        "[telescope]        every candidate alpha <= 0:           {}",
        stats.n_ghost_read_all_alpha_zero
    );
    eprintln!(
        "[telescope]        mixed causes or a pruned candidate:   {}",
        n_ghost.saturating_sub(
            stats.n_ghost_read_all_missing
                + stats.n_ghost_read_all_incompetent
                + stats.n_ghost_read_all_alpha_zero
        )
    );
    eprintln!(
        "[telescope]    output-stage row filter -- a ROW count, not reads; a \
         tripwire that must stay 0: {}",
        stats.n_dropped_output_row_competence
    );
    // CANDIDATE CONSERVATION LEDGER. A count of CANDIDATE ROWS, on its
    // own denominator, for the same reason the two groups above are not
    // pooled: a candidate cannot be lost for the same reason a read
    // cannot, and reading the two ladders against each other is how the
    // original 20.10% deficit went unexamined for a round.
    let led = &stats.ledger;
    eprintln!("[telescope]    candidate ledger (every candidate, one terminal fate):");
    eprintln!(
        "[telescope]      candidates on the graph (LHS)     : {}",
        led.total_candidates
    );
    // The census tests these four in this order and `continue`s, so a
    // candidate that is BOTH pruned and emission-less is counted ONLY
    // as pruned. The `no emission row` line is not a count of
    // candidates lacking an emission row; it is a count of candidates
    // that got past the prune and then had none.
    eprintln!("[telescope]      -- contested, decided by the E-step (tested in this order):");
    eprintln!(
        "[telescope]         pruned by delta-AS            : {}",
        led.pruned_delta_as
    );
    eprintln!(
        "[telescope]         no emission row                : {}",
        led.missing
    );
    eprintln!(
        "[telescope]         is_polyA_competent == false   : {}",
        led.incompetent
    );
    eprintln!(
        "[telescope]         alpha <= 0                    : {}",
        led.alpha_zero
    );
    eprintln!("[telescope]      -- contested, decided at the output stage:");
    eprintln!(
        "[telescope]         below min_weight              : {}   (moves with --min-weight; 0 by construction at 0)",
        led.pruned_min_weight
    );
    eprintln!(
        "[telescope]         recheck competence / params    : {}   (tripwire, must stay 0)",
        led.dropped_output_competence
    );
    eprintln!(
        "[telescope]         rows emitted                  : {}",
        led.rows_emitted
    );
    eprintln!("[telescope]      -- unambiguous reads (1 candidate each):");
    eprintln!(
        "[telescope]         is_polyA_competent == false   : {}",
        led.unambig_incompetent
    );
    eprintln!(
        "[telescope]         no emission row                : {}",
        led.unambig_missing
    );
    eprintln!(
        "[telescope]         rows emitted                  : {}",
        led.unambig_emitted
    );
    // Both of these can only ever print 0: solve_and_write_profile
    // asserts them and returns Err before the CLI is reached. They are
    // a CONFIRMATION that the assert ran, not independent evidence of
    // closure -- do not cite the printed 0 as the measurement.
    eprintln!(
        "[telescope]      UNACCOUNTED DEFICIT             : {}   (asserted == 0 in \
         solve_and_write_profile; this line only confirms the assert ran)",
        led.deficit()
    );
    eprintln!(
        "[telescope]      census vs E-step disagreement   : {}   (same condition as the \
         deficit, not an independent one; kept because it names the cause)",
        led.census_vs_estep_deficit()
    );
    let n_ambig = stats.n_ambiguous.max(1) as f64;
    eprintln!("[telescope]    posterior entropy histogram:");
    eprintln!(
        "[telescope]      < 0.20 nats (cleanly resolved): {} ({:.1}%)",
        stats.n_entropy_clean,
        stats.n_entropy_clean as f64 / n_ambig * 100.0
    );
    eprintln!(
        "[telescope]      0.20–0.70 nats (partial):     {} ({:.1}%)",
        stats.n_entropy_partial,
        stats.n_entropy_partial as f64 / n_ambig * 100.0
    );
    eprintln!(
        "[telescope]      > 0.70 nats (degenerate tie):  {} ({:.1}%)",
        stats.n_entropy_degenerate,
        stats.n_entropy_degenerate as f64 / n_ambig * 100.0
    );
    // The degenerate reads split by MECHANISM, because the two are
    // flat for different reasons and a single number cannot be read.
    // Both are printed against the SAME denominator -- the degenerate
    // total, not n_ambig -- because the question is "of the degenerate
    // reads, how many are this kind", and dividing by n_ambig here would
    // silently answer a different question. The CLI prints the
    // residual `n_ambig - degenerate` nowhere, so this is the only
    // place the composition is visible.
    eprintln!(
        "[telescope]        of which padded-padded ambiguity: {} ({:.1}% of degenerate)",
        stats.n_degenerate_all_imputed,
        stats.n_degenerate_all_imputed as f64 / stats.n_entropy_degenerate.max(1) as f64 * 100.0
    );
    eprintln!(
        "[telescope]        of which real candidate competition: {} ({:.1}% of degenerate)",
        stats.n_degenerate_with_anchored,
        stats.n_degenerate_with_anchored as f64 / stats.n_entropy_degenerate.max(1) as f64 * 100.0
    );
    // (The two zero-power counters that used to be reported here are gone; see
    // the comment further down. The paragraph below is retained because
    // it still governs the two lines that remain.)
    //
    // A denominator of ZERO PRINTS `n/a`, NOT `0.0%`. `0.0 / 0.0` in
    // Rust is NaN, so an unguarded line prints "NaN%" -- and a
    // `.max(1)` guard alone fixes the NaN while making "no read
    // qualified" render as a confident 0.0%. That is the same class of
    // defect as the metric this change is about: a number that looks
    // measured and is not.
    let n_with_cand = stats.n_unambiguous + stats.n_ambiguous;
    eprintln!("[telescope]    emission model coverage (reads with >=1 candidate):");
    eprintln!(
        "[telescope]      reads with >=1 emission candidate : {} of {} ({})",
        stats.n_reads_any_emission_candidate,
        n_with_cand,
        pct(stats.n_reads_any_emission_candidate, n_with_cand as u64)
    );
    eprintln!(
        "[telescope]      single-candidate reads           : {} of {} ({})",
        stats.n_reads_single_emission_candidate,
        n_with_cand,
        pct(stats.n_reads_single_emission_candidate, n_with_cand as u64)
    );
    // The two zero-power counters used to be printed here, under the
    // header `poly(A) discriminative power:`. They are deliberately NOT
    // printed any more, and this comment exists so the removal is not
    // "tidied up" by a later reader who never saw the reason.
    //
    // WHAT WAS WRONG WITH THEM. The counter FIRES on a contested read whose
    // candidates all share ONE identical `(mu, sigma2)` -- i.e. it fires
    // exactly when the M-step did NOT personalise them, and the poly(A)
    // term cancels in every pairwise odds ratio. So a LOW value means
    // non-degenerate, and it says nothing about whether the poly(A) term
    // can actually discriminate. On HC2 it printed
    // `zero-power_all: 886 of 4,380,726 ambiguous (0.02%)`.
    //
    // THE TRAP WAS THE COMPLEMENT. 0.02% is one number away from
    // "poly(A) discriminates on 99.98% of contested reads", and that
    // inference is false: measured on HC2 the per-read poly(A) residual
    // sd is 0.7521 log2 against a median candidate separation of 0.1919
    // -- SNR 0.255 -- and a controlled experiment at that real
    // separation scored 54.86%, BELOW the 70.0% majority-class
    // baseline. An earlier round of notes made exactly that
    // complement error and wrote the 99.98% claim up as a result.
    //
    // The header was the worst of it: leaving `zero-power_resolved`
    // printed under "poly(A) discriminative power:" would have
    // preserved the same defect, which is why both lines went and not
    // just the one named in the ruling.
    //
    // THE FIELDS STAY. `n_polya_zero_power` and
    // `n_polya_zero_power_resolved` remain on `TelEmStats`. Nothing at
    // runtime reads them any more -- the only consumer is `em.rs`'s
    // unit tests, which assert on the fields directly and cannot see
    // this file -- so they are a tripwire for parameters collapsing
    // numerically into one identical float, and they must never be cited
    // as evidence of poly(A) validity or discriminative power.
    // `crates/telescope-cli/tests/quant_does_not_report_polya_validity.rs`
    // fails if either string reappears on stderr.
    if debug && !stats.em_trace.is_empty() {
        eprintln!("[telescope]    EM trace:");
        for (iter, ll, ent) in &stats.em_trace {
            eprintln!(
                "[telescope]      iter={:>4}  logL={:>14.6e}  mean_entropy={:.4e}",
                iter, ll, ent
            );
        }
    }
    eprintln!("[telescope] done.");
    Ok(())
}
