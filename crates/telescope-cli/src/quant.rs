use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use telescope_core::{em, emission_model, evidence};

use crate::{
    maybe_write_filter_ledger, pct, print_filter_ledger, run_build_anchors, run_build_emission,
    AnchorRequest, EmissionRequest, ScopedTempDir, ALN_WEIGHT_LAMBDA_DEFAULT,
    ANCHOR_GAP_ABS_DEFAULT, ANCHOR_GAP_REL_DEFAULT, ANCHOR_MAX_READS_PER_SAMPLE_DEFAULT,
    ANCHOR_MIN_COVERAGE_DEFAULT, ANCHOR_TES_SLACK_DEFAULT, EMISSION_COHORT_MIN_READS_DEFAULT,
    EMISSION_GENE_MIN_READS_DEFAULT, EMISSION_TX_MIN_READS_DEFAULT, FIVE_PRIME_WEIGHT_DEFAULT,
    FIVE_PRIME_WINDOW_DEFAULT, POLY_A_WEIGHT_DEFAULT, PRUNE_MAX_DELTA_AS_DEFAULT,
    THREE_PRIME_WEIGHT_DEFAULT, THREE_PRIME_WINDOW_DEFAULT,
};

/// Arguments for the `quant` subcommand.
#[derive(Args, Debug)]
pub struct QuantArgs {
    /// Transcriptome-aligned BAM.
    #[arg(long, required = true, help_heading = "INPUTS")]
    bam: PathBuf,

    /// Pre-built emission model parquet (`telescope.emission.parquet`).
    #[arg(long, hide = true, help_heading = "INPUTS")]
    emission: Option<PathBuf>,

    /// Sample identifier written to the `sample_id` column.
    #[arg(long, required = true, help_heading = "INPUTS")]
    sample_id: String,

    /// Output transcript profile parquet (`*.profile.parquet`). The prior is fitted in a temporary directory that is removed when the run ends.
    #[arg(long, required = true, help_heading = "OUTPUTS")]
    output: PathBuf,

    /// Annotation GTF supplying each transcript's gene ID.
    #[arg(short = 'g', long = "gtf", help_heading = "REFERENCE")]
    gtf: Option<PathBuf>,

    /// Maximum 3'-end offset (bp) from the annotated transcript TES.
    #[arg(long, default_value_t = 35, help_heading = "FILTERS")]
    tes_slack: usize,

    /// Source of poly(A) tail length (`bam_tag:<TAG>` or `none`).
    #[arg(
        long = "poly-a-source",
        alias = "polya-source",
        alias = "polyA-source",
        default_value = "bam_tag:pt",
        help_heading = "INPUTS"
    )]
    poly_a_source: String,

    /// Record-level filter ledger, written as JSON.
    #[arg(long, hide = true, help_heading = "OUTPUTS")]
    ledger: Option<PathBuf>,

    /// Convergence tolerance of the EM log-likelihood per molecule.
    #[arg(long = "em-tol", default_value_t = 1e-6, help_heading = "MODEL")]
    tol: f64,

    /// Maximum number of EM iterations.
    #[arg(long = "em-max-iter", default_value_t = 1000, help_heading = "MODEL")]
    max_iter: usize,

    /// Scale λ of the alignment score term `ln A_rk = (AS_rk - max_j AS_rj) / λ`.
    #[arg(long, default_value_t = ALN_WEIGHT_LAMBDA_DEFAULT, help_heading = "MODEL")]
    aln_weight_lambda: f64,

    /// Weight β of the 5'-anchoring bonus within `--five-prime-window`.
    #[arg(long, default_value_t = FIVE_PRIME_WEIGHT_DEFAULT, help_heading = "MODEL")]
    five_prime_weight: f64,

    /// 5' gap window (bp, inclusive) for `--five-prime-weight`.
    #[arg(long, default_value_t = FIVE_PRIME_WINDOW_DEFAULT, help_heading = "MODEL")]
    five_prime_window: u32,

    /// Weight γ of the 3'-end geometry bonus within `--three-prime-window`.
    #[arg(long, default_value_t = THREE_PRIME_WEIGHT_DEFAULT, help_heading = "MODEL")]
    three_prime_weight: f64,

    /// 3' gap window (bp, inclusive) for `--three-prime-weight`.
    #[arg(long, default_value_t = THREE_PRIME_WINDOW_DEFAULT, help_heading = "MODEL")]
    three_prime_window: u32,

    /// Weight on the poly(A) tail-length emission term.
    #[arg(long, default_value_t = POLY_A_WEIGHT_DEFAULT, help_heading = "MODEL")]
    poly_a_weight: f64,

    /// Drop candidate transcripts whose alignment score is more than this below the read's best.
    #[arg(long, default_value_t = PRUNE_MAX_DELTA_AS_DEFAULT, help_heading = "FILTERS")]
    prune_max_delta_as: i32,

    /// Minimum posterior probability required to emit a read-transcript assignment row.
    #[arg(
        long = "min-posterior",
        value_name = "MIN_POSTERIOR",
        default_value_t = 0.01,
        help_heading = "FILTERS"
    )]
    min_weight: f64,

    /// Print the EM per-iteration trace (log-likelihood and mean entropy) to stderr.
    #[arg(long, hide = true, help_heading = "OUTPUTS")]
    debug: bool,

    /// Per-read posterior summary (`read_id`, `n_candidates_before_cutoff`, `top_tx_id`, `top_weight`, `entropy`).
    #[arg(long, hide = true, help_heading = "OUTPUTS")]
    read_assignments: Option<PathBuf>,

    /// Diagnostic `(mu, sigma2)` table consumed by the final E-step.
    #[arg(long, hide = true, help_heading = "OUTPUTS")]
    dump_emission_params: Option<PathBuf>,
}

/// Run the `quant` subcommand.
pub fn run_quant(args: QuantArgs, sigma2_dof_offset: Option<f64>) -> Result<()> {
    let QuantArgs {
        bam,
        emission,
        sample_id,
        output,
        gtf,
        tes_slack,
        tol,
        max_iter,
        aln_weight_lambda,
        five_prime_weight,
        poly_a_weight,
        five_prime_window,
        three_prime_weight,
        three_prime_window,
        prune_max_delta_as,
        min_weight,
        poly_a_source,
        debug,
        read_assignments,
        dump_emission_params,
        ledger,
    } = args;

    if emission.is_some() && gtf.is_some() {
        anyhow::bail!(
            "--emission and --gtf are mutually exclusive. --emission supplies a \
             pre-built prior; --gtf is used when fitting the prior automatically."
        );
    }
    let em_flags = EmFlags {
        tes_slack,
        tol,
        max_iter,
        aln_weight_lambda,
        five_prime_weight,
        poly_a_weight,
        five_prime_window,
        three_prime_weight,
        three_prime_window,
        prune_max_delta_as,
        min_weight,
        poly_a_source,
        debug,
    };
    match emission {
        Some(emission) => run_one_sample(
            OneSample::new(
                bam,
                emission,
                sample_id,
                output,
                &em_flags,
                ledger,
                read_assignments,
                dump_emission_params,
            ),
            sigma2_dof_offset,
        ),
        None => {
            let gtf = gtf.ok_or_else(|| {
                anyhow::anyhow!(
                    "--gtf is required: the anchor and emission stages need the \
                     transcript -> gene map."
                )
            })?;
            run_auto_path(
                &bam,
                &sample_id,
                &gtf,
                &output,
                &em_flags,
                ledger,
                read_assignments,
                dump_emission_params,
                sigma2_dof_offset,
            )
        }
    }
}

/// The EM-stage flags of `quant`.
#[derive(Clone)]
struct EmFlags {
    tes_slack: usize,
    tol: f64,
    max_iter: usize,
    aln_weight_lambda: f64,
    five_prime_weight: f64,
    poly_a_weight: f64,
    five_prime_window: u32,
    three_prime_weight: f64,
    three_prime_window: u32,
    prune_max_delta_as: i32,
    min_weight: f64,
    poly_a_source: String,
    debug: bool,
}

/// Parameters of one sample's TEL-EM run.
pub struct OneSample {
    bam: PathBuf,
    emission: PathBuf,
    sample_id: String,
    output: PathBuf,
    tes_slack: usize,
    tol: f64,
    max_iter: usize,
    aln_weight_lambda: f64,
    five_prime_weight: f64,
    poly_a_weight: f64,
    five_prime_window: u32,
    three_prime_weight: f64,
    three_prime_window: u32,
    prune_max_delta_as: i32,
    min_weight: f64,
    poly_a_source: String,
    debug: bool,
    read_assignments: Option<PathBuf>,
    dump_emission_params: Option<PathBuf>,
    ledger: Option<PathBuf>,
}

impl OneSample {
    #[allow(clippy::too_many_arguments)]
    fn new(
        bam: PathBuf,
        emission: PathBuf,
        sample_id: String,
        output: PathBuf,
        f: &EmFlags,
        ledger: Option<PathBuf>,
        read_assignments: Option<PathBuf>,
        dump_emission_params: Option<PathBuf>,
    ) -> OneSample {
        OneSample {
            bam,
            emission,
            sample_id,
            output,
            tes_slack: f.tes_slack,
            tol: f.tol,
            max_iter: f.max_iter,
            aln_weight_lambda: f.aln_weight_lambda,
            five_prime_weight: f.five_prime_weight,
            poly_a_weight: f.poly_a_weight,
            five_prime_window: f.five_prime_window,
            three_prime_weight: f.three_prime_weight,
            three_prime_window: f.three_prime_window,
            prune_max_delta_as: f.prune_max_delta_as,
            min_weight: f.min_weight,
            poly_a_source: f.poly_a_source.clone(),
            debug: f.debug,
            read_assignments,
            dump_emission_params,
            ledger,
        }
    }
}

/// Run ONE sample's TEL-EM.
pub fn run_one_sample(s: OneSample, sigma2_dof_offset: Option<f64>) -> Result<()> {
    let OneSample {
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
        three_prime_weight,
        three_prime_window,
        prune_max_delta_as,
        min_weight,
        poly_a_source,
        debug,
        read_assignments,
        dump_emission_params,
        ledger,
    } = s;

    let poly_a = evidence::parse_poly_a_source(&poly_a_source)?;
    let poly_a_weight = if matches!(poly_a, evidence::PolyASource::None) {
        if poly_a_weight != 0.0 {
            eprintln!(
                "[telescope] WARNING: --poly-a-source none forces --poly-a-weight 0 \
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
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(three_prime_weight >= 0.0) {
        anyhow::bail!(
            "--three-prime-weight must be >= 0 and non-NaN (got {}); it is the \
             log-space bonus ln_3p awarded inside the 3' window, so 0.0 is the \
             valid off switch but a negative weight inverts the term and NaN \
             would emit NaN posterior weights",
            three_prime_weight
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
    eprintln!("[telescope] Quant");
    eprintln!("  bam:        {}", bam.display());
    eprintln!("  emission:   {}", emission.display());
    eprintln!("  sample_id:  {}", sample_id);
    eprintln!("  output:     {}", output.display());
    eprintln!("  tes_slack:  {} bp", tes_slack);
    eprintln!("  tol:        {:.1e}", tol);
    eprintln!("  max_iter:   {}", max_iter);
    eprintln!("  five_prime_weight: {}", five_prime_weight);
    eprintln!("  five_prime_window: {} bp", five_prime_window);
    eprintln!("  three_prime_weight: {}", three_prime_weight);
    eprintln!("  three_prime_window: {} bp", three_prime_window);
    eprintln!("  prune_max_delta_as: {}", prune_max_delta_as);
    eprintln!("  debug:      {}", debug);

    em::validate_emission_file(&emission)?;

    eprintln!("[telescope]  extract evidence…");
    eprintln!("[telescope]  polyA-src:  {}", poly_a_source);
    let graph = evidence::extract_evidence_with(&bam, tes_slack, &poly_a)
        .context("extract_evidence failed")?;
    eprintln!("[telescope]    molecules:      {}", graph.molecules.len());
    print_filter_ledger(&graph.stats, "[telescope]    ");
    maybe_write_filter_ledger(ledger.as_ref(), &graph.stats)?;
    eprintln!(
        "[telescope]    transcripts:    {}",
        graph.transcript_names.len()
    );

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

    eprintln!("[telescope]  TEL-EM…");
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
    if three_prime_weight > 0.0 {
        eprintln!(
            "[telescope]    3' term ACTIVE on ambiguous reads: ln_3p = {} if d_k <= {} bp, else 0",
            three_prime_weight, three_prime_window
        );
    } else {
        eprintln!("[telescope]    3' term OFF (--three-prime-weight 0).");
    }
    if prune_max_delta_as < i32::MAX {
        eprintln!(
            "[telescope]    ΔAS prune ACTIVE: drop candidates with ΔAS > {} vs the read's best AS",
            prune_max_delta_as
        );
    } else {
        eprintln!("[telescope]    ΔAS prune OFF (--prune-max-delta-as at i32::MAX).");
    }
    eprintln!("[telescope]    aln_weight_lambda: {:?}", aln_weight_lambda);
    let mut em_trace: Vec<(usize, f64, f64)> = Vec::new();
    let trace_sink: Option<&mut Vec<(usize, f64, f64)>> =
        if debug { Some(&mut em_trace) } else { None };
    let mut read_sink: Vec<em::ReadAssignment> = Vec::new();
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
        three_prime_weight,
        three_prime_window,
    )
    .context("solve_and_write_profile failed")?;
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
        "[telescope]    output-stage row filter (rows): {}",
        stats.n_dropped_output_row_competence
    );
    let led = &stats.ledger;
    eprintln!("[telescope]    candidate ledger (every candidate, one terminal fate):");
    eprintln!(
        "[telescope]      candidates on the graph (LHS)     : {}",
        led.total_candidates
    );
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
        "[telescope]         below min_posterior           : {}   (moves with --min-posterior; 0 by construction at 0)",
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
    eprintln!(
        "[telescope]      UNACCOUNTED DEFICIT             : {}",
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

/// Run the automatic single-sample workflow: anchors -> prior -> TEL-EM.
#[allow(clippy::too_many_arguments)]
fn run_auto_path(
    bam: &Path,
    sample_id: &str,
    gtf: &Path,
    output: &Path,
    em_flags: &EmFlags,
    ledger: Option<PathBuf>,
    read_assignments: Option<PathBuf>,
    dump_emission_params: Option<PathBuf>,
    sigma2_dof_offset: Option<f64>,
) -> Result<()> {
    let tmp = ScopedTempDir::new("telescope-cli-auto")?;
    eprintln!(
        "[telescope]  temporary intermediates: {}",
        tmp.path().display()
    );
    let pool = tmp.path().join("anchor_pool.parquet");
    run_build_anchors(AnchorRequest {
        bams: vec![bam.to_path_buf()],
        sample_ids: vec![sample_id.to_string()],
        output: pool.clone(),
        gtf: Some(gtf.to_path_buf()),
        tes_slack: ANCHOR_TES_SLACK_DEFAULT,
        min_coverage: ANCHOR_MIN_COVERAGE_DEFAULT,
        max_reads_per_sample: ANCHOR_MAX_READS_PER_SAMPLE_DEFAULT,
        gap_abs: ANCHOR_GAP_ABS_DEFAULT,
        gap_rel: ANCHOR_GAP_REL_DEFAULT,
        ledger: None,
    })?;
    let emission_dir = tmp.path().join("emission");
    std::fs::create_dir_all(&emission_dir)
        .with_context(|| format!("Failed to create {:?}", emission_dir))?;
    let emission = emission_dir.join("telescope.emission.parquet");
    let universe = emission_model::read_reference_universe(bam).with_context(|| {
        format!(
            "Failed to read the reference universe from the BAM header of {:?}",
            bam
        )
    })?;
    eprintln!(
        "[telescope]  reference universe: {} transcripts (the @SQ header of the input BAM)",
        universe.len()
    );
    run_build_emission(EmissionRequest {
        anchor_pool: pool,
        transcript_evidence: None,
        reference_bam: None,
        reference_universe: Some(universe),
        gtf: Some(gtf.to_path_buf()),
        provenance_out: None,
        output: emission.clone(),
        tx_min_reads: EMISSION_TX_MIN_READS_DEFAULT,
        gene_min_reads: EMISSION_GENE_MIN_READS_DEFAULT,
        cohort_min_reads: EMISSION_COHORT_MIN_READS_DEFAULT,
        prior_scope: None,
    })?;
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create {:?}", parent))?;
        }
    }
    run_one_sample(
        OneSample::new(
            bam.to_path_buf(),
            emission,
            sample_id.to_string(),
            output.to_path_buf(),
            em_flags,
            ledger,
            read_assignments,
            dump_emission_params,
        ),
        sigma2_dof_offset,
    )?;
    eprintln!("[telescope]  profile: {}", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Args as ClapArgs;

    fn quant_command() -> clap::Command {
        QuantArgs::augment_args(clap::Command::new("quant"))
    }

    #[test]
    fn test_the_three_prime_flag_defaults_come_from_the_constants() {
        let m = quant_command()
            .try_get_matches_from([
                "quant",
                "--bam",
                "in.bam",
                "--emission",
                "e.parquet",
                "--sample-id",
                "S1",
                "--output",
                "o.parquet",
            ])
            .expect("required-only invocation parses");
        assert_eq!(
            *m.get_one::<f64>("three_prime_weight").unwrap(),
            THREE_PRIME_WEIGHT_DEFAULT,
            "--three-prime-weight must default to the constant, not a literal"
        );
        assert_eq!(
            *m.get_one::<u32>("three_prime_window").unwrap(),
            THREE_PRIME_WINDOW_DEFAULT,
            "--three-prime-window must default to the constant, not a literal"
        );
    }

    #[test]
    fn test_the_three_prime_defaults_are_off_and_a_15_bp_window() {
        assert_eq!(THREE_PRIME_WEIGHT_DEFAULT, 0.0);
        assert_eq!(THREE_PRIME_WINDOW_DEFAULT, 15);
    }

    #[test]
    fn test_three_prime_weight_zero_parses_as_the_off_switch() {
        let cmd = quant_command()
            .try_get_matches_from([
                "quant",
                "--bam",
                "in.bam",
                "--emission",
                "e.parquet",
                "--sample-id",
                "S1",
                "--output",
                "o.parquet",
                "--three-prime-weight",
                "0",
            ])
            .expect("weight 0 must be accepted");
        assert_eq!(
            *cmd.get_one::<f64>("three_prime_weight").unwrap(),
            0.0,
            "the off switch must survive parsing exactly"
        );
    }

    #[test]
    fn removed_and_hidden_flags_have_expected_quant_help_visibility() {
        let mut cmd = quant_command();
        let help = cmd.render_long_help().to_string();
        for forbidden in [
            "--output-dir",
            "--prior-scope",
            "--reference-bam",
            "--tx-id-mapping",
            "--emission",
            "--ledger",
            "--debug",
            "--read-assignments",
            "--dump-emission-params",
        ] {
            assert!(
                !help.contains(forbidden),
                "`{forbidden}` must not appear in `quant --help`:\n{help}"
            );
        }
        for visible in ["--three-prime-weight", "--three-prime-window"] {
            assert!(
                help.contains(visible),
                "`{visible}` must remain visible in `quant --help`:\n{help}"
            );
        }
        for removed in [
            "--output-dir",
            "--prior-scope",
            "--reference-bam",
            "--tx-id-mapping",
        ] {
            let err = quant_command()
                .try_get_matches_from([
                    "quant",
                    "--bam",
                    "in.bam",
                    "--sample-id",
                    "S1",
                    "--gtf",
                    "a.gtf",
                    "--output",
                    "o.parquet",
                    removed,
                    "val",
                ])
                .expect_err("removed flag must be rejected");
            let msg = err.to_string();
            assert!(
                msg.contains("unexpected argument"),
                "expected unexpected argument for {removed}, got: {msg}"
            );
        }
    }
}
