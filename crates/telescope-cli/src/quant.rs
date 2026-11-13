use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use telescope_core::{em, emission_model, evidence};

use crate::{
    maybe_write_filter_ledger, pct, print_filter_ledger, run_build_anchors, run_build_emission,
    AnchorRequest, EmissionRequest, PriorScope, ScopedTempDir, ALN_WEIGHT_LAMBDA_DEFAULT,
    ANCHOR_GAP_ABS_DEFAULT, ANCHOR_GAP_REL_DEFAULT, ANCHOR_MAX_READS_PER_SAMPLE_DEFAULT,
    ANCHOR_MIN_COVERAGE_DEFAULT, ANCHOR_TES_SLACK_DEFAULT, EMISSION_COHORT_MIN_READS_DEFAULT,
    EMISSION_GENE_MIN_READS_DEFAULT, EMISSION_TX_MIN_READS_DEFAULT, FIVE_PRIME_WEIGHT_DEFAULT,
    FIVE_PRIME_WINDOW_DEFAULT, POLY_A_SOURCE_HELP, POLY_A_WEIGHT_DEFAULT,
    PRUNE_MAX_DELTA_AS_DEFAULT, THREE_PRIME_WEIGHT_DEFAULT, THREE_PRIME_WINDOW_DEFAULT,
};

/// Which samples the poly(A) prior is fitted on.
#[derive(Copy, Clone, PartialEq, Eq, Debug, clap::ValueEnum)]
pub enum PriorScopeArg {
    /// One BAM -> a prior of that one sample.
    Auto,
    /// Every sample gets its own prior, however many BAMs were given.
    Sample,
    /// One shared prior, however many BAMs were given.
    Cohort,
}

impl std::fmt::Display for PriorScopeArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            PriorScopeArg::Auto => "auto",
            PriorScopeArg::Sample => "sample",
            PriorScopeArg::Cohort => "cohort",
        };
        f.write_str(s)
    }
}

/// Arguments for the `quant` subcommand, shared by both binaries.
#[derive(Args, Debug)]
pub struct QuantArgs {
    /// Transcriptome-aligned BAM.
    #[arg(long, required = true, help_heading = "INPUTS")]
    bam: Vec<PathBuf>,

    /// Emission model parquet (`telescope.emission.parquet`), written by `build-emission`.
    #[arg(long, hide = true, help_heading = "INPUTS")]
    emission: Option<PathBuf>,

    /// Sample identifier, written as the `sample_id` column.
    #[arg(long, required = true, help_heading = "INPUTS")]
    sample_id: Vec<String>,

    /// Profile parquet to write. The prior is fitted in a temporary directory that is removed when the run ends.
    #[arg(long, help_heading = "OUTPUTS")]
    output: Option<PathBuf>,

    #[arg(hide = true, long, help_heading = "OUTPUTS")]
    output_dir: Option<PathBuf>,

    /// Annotation GTF.
    #[arg(short = 'g', long = "gtf", help_heading = "REFERENCE")]
    gtf: Option<PathBuf>,

    #[arg(hide = true, long, help_heading = "REFERENCE")]
    reference_bam: Option<PathBuf>,

    #[arg(
        hide = true,
        long,
        value_enum,
        default_value_t = PriorScopeArg::Auto,
        help_heading = "MODEL"
    )]
    prior_scope: PriorScopeArg,

    /// Slack in bp allowed between a read's 3' end and the transcript's TES boundary.
    #[arg(long, default_value_t = 35, help_heading = "FILTERS")]
    tes_slack: usize,

    #[arg(
        long = "poly-a-source",
        alias = "polya-source",
        alias = "polyA-source",
        default_value = "bam_tag:pt",
        help = POLY_A_SOURCE_HELP,
        help_heading = "INPUTS"
    )]
    poly_a_source: String,

    /// Record-level filter ledger, written as JSON.
    #[arg(long, help_heading = "OUTPUTS")]
    ledger: Option<PathBuf>,

    /// Convergence tolerance of the EM log-likelihood, per molecule.
    #[arg(long = "em-tol", default_value_t = 1e-6, help_heading = "MODEL")]
    tol: f64,

    /// Maximum number of EM iterations.
    #[arg(long = "em-max-iter", default_value_t = 1000, help_heading = "MODEL")]
    max_iter: usize,

    /// Scale λ of the alignment term `ln A_rk = (AS_rk − max_j AS_rj) / λ`.
    #[arg(long, default_value_t = ALN_WEIGHT_LAMBDA_DEFAULT, help_heading = "MODEL")]
    aln_weight_lambda: f64,

    /// Weight β of the 5'-anchoring term, applied when the read's 5' gap in candidate k's own cDNA coordinates is within `--five-prime-window`.
    #[arg(long, default_value_t = FIVE_PRIME_WEIGHT_DEFAULT, help_heading = "MODEL")]
    five_prime_weight: f64,

    /// 5' window in bp for `--five-prime-weight`: the bonus fires when the read's 5' gap is at most this value (inclusive).
    #[arg(long, default_value_t = FIVE_PRIME_WINDOW_DEFAULT, help_heading = "MODEL")]
    five_prime_window: u32,

    /// Weight γ of the 3'-end geometry term: the log-space bonus `ln_3p` awarded to a candidate whose observed 3' gap `d_k` -- the number of transcript bases 3' of the read's 3' END, in the candidate's own cDNA coordinates -- is within `--three-prime-window`.
    #[arg(long, default_value_t = THREE_PRIME_WEIGHT_DEFAULT, help_heading = "MODEL")]
    three_prime_weight: f64,

    /// 3' window in bp for `--three-prime-weight`: the bonus fires when the read's 3' gap is at most this value (inclusive).
    #[arg(long, default_value_t = THREE_PRIME_WINDOW_DEFAULT, help_heading = "MODEL")]
    three_prime_window: u32,

    /// Weight on the poly(A) mixture term.
    #[arg(long, default_value_t = POLY_A_WEIGHT_DEFAULT, help_heading = "MODEL")]
    poly_a_weight: f64,

    /// Candidates whose alignment score is more than this far below the read's best are dropped before the softmax.
    #[arg(long, default_value_t = PRUNE_MAX_DELTA_AS_DEFAULT, help_heading = "FILTERS")]
    prune_max_delta_as: i32,

    /// Output-stage posterior truncation threshold.
    #[arg(
        long = "min-posterior",
        value_name = "MIN_POSTERIOR",
        default_value_t = 0.01,
        help_heading = "FILTERS"
    )]
    min_weight: f64,

    /// Optional TSV with header `bam_id<TAB>original_id`, translating the BAM's transcript names into the emission model's ids.
    #[arg(long, help_heading = "INPUTS")]
    tx_id_mapping: Option<PathBuf>,

    /// Print the EM per-iteration trace (log-likelihood and mean entropy) to stderr.
    #[arg(long, help_heading = "OUTPUTS")]
    debug: bool,

    /// Per-read posterior summary (`read_id`, `n_candidates_before_cutoff`, `top_tx_id`, `top_weight`, `entropy`).
    #[arg(long, help_heading = "OUTPUTS")]
    read_assignments: Option<PathBuf>,

    /// The `(mu, sigma2)` the final E-step consumed.
    #[arg(long, help_heading = "OUTPUTS")]
    dump_emission_params: Option<PathBuf>,
}

/// Run the `quant` subcommand.
pub fn run_quant(args: QuantArgs, sigma2_dof_offset: Option<f64>) -> Result<()> {
    let QuantArgs {
        bam,
        emission,
        sample_id,
        output,
        output_dir,
        gtf,
        reference_bam,
        prior_scope,
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
        tx_id_mapping,
        poly_a_source,
        debug,
        read_assignments,
        dump_emission_params,
        ledger,
    } = args;

    if bam.len() != sample_id.len() {
        anyhow::bail!(
            "--bam ({} files) and --sample-id ({} ids) must have equal length",
            bam.len(),
            sample_id.len()
        );
    }
    if emission.is_some() && gtf.is_some() {
        anyhow::bail!(
            "--emission and --gtf are mutually exclusive. --emission says the \
             prior is ALREADY built, and the --emission mode does not read an \
             annotation; --gtf feeds the anchor and prior stages, which only \
             the automatic mode runs. Give one or the other."
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
        tx_id_mapping,
        poly_a_source,
        debug,
    };
    match emission {
        Some(emission) => {
            if output.is_none() {
                anyhow::bail!("--output <FILE> is required");
            }
            if output_dir.is_some() {
                anyhow::bail!(
                    "--output-dir belongs to the automatic mode, which builds \
                     the prior itself. The --emission mode writes a single profile to \
                     --output."
                );
            }
            if bam.len() != 1 {
                anyhow::bail!(
                    "--emission mode runs ONE sample per invocation \
                     and was given {} --bam arguments. Use the automatic mode \
                     (no --emission, with --output-dir and --gtf) for several \
                     samples.",
                    bam.len()
                );
            }
            run_one_sample(
                OneSample::new(
                    bam[0].clone(),
                    emission,
                    sample_id[0].clone(),
                    output.expect("checked above"),
                    &em_flags,
                    ledger,
                    read_assignments,
                    dump_emission_params,
                ),
                sigma2_dof_offset,
            )
        }

        None => {
            match (output_dir.as_ref(), output.as_ref()) {
                (None, None) => {
                    anyhow::bail!(if bam.len() == 1 {
                        "--output <FILE> is required"
                    } else {
                        "--output-dir <DIR> is required for multiple BAM files"
                    })
                }
                (Some(_), Some(_)) => {
                    anyhow::bail!(
                        "--output and --output-dir are mutually exclusive: \
                         --output writes ONE profile and nothing else, \
                         --output-dir writes every product. Give one or the other."
                    )
                }
                (Some(dir), None) => {
                    let gtf = gtf.ok_or_else(|| {
                        anyhow::anyhow!(
                            "--gtf is required by the automatic mode: the anchor stage \
                             and the prior stage both need the transcript -> gene map."
                        )
                    })?;
                    run_auto_path(
                        &bam,
                        &sample_id,
                        &gtf,
                        dir,
                        reference_bam.as_ref(),
                        prior_scope,
                        &em_flags,
                        ledger.as_ref(),
                        read_assignments.as_ref(),
                        dump_emission_params.as_ref(),
                        sigma2_dof_offset,
                    )?;
                }
                (None, Some(out)) => {
                    if bam.len() != 1 {
                        anyhow::bail!(
                            "--output writes ONE profile, and this run was given {} \
                             --bam arguments. Use --output-dir for several samples.",
                            bam.len()
                        );
                    }
                    let gtf = gtf.ok_or_else(|| {
                        anyhow::anyhow!(
                            "--gtf is required by the automatic mode: the anchor stage \
                             and the prior stage both need the transcript -> gene map."
                        )
                    })?;

                    let tmp_out = ScopedTempDir::new("telescope-cli-quant-out")?;
                    eprintln!(
                        "[telescope]  temporary intermediates: {}",
                        tmp_out.path().display()
                    );
                    run_auto_path(
                        &bam,
                        &sample_id,
                        &gtf,
                        tmp_out.path(),
                        reference_bam.as_ref(),
                        prior_scope,
                        &em_flags,
                        ledger.as_ref(),
                        read_assignments.as_ref(),
                        dump_emission_params.as_ref(),
                        sigma2_dof_offset,
                    )?;

                    if let Some(parent) = out.parent() {
                        if !parent.as_os_str().is_empty() {
                            std::fs::create_dir_all(parent)
                                .with_context(|| format!("Failed to create {:?}", parent))?;
                        }
                    }
                    let produced = tmp_out
                        .path()
                        .join(format!("{}.profile.parquet", sample_id[0]));
                    std::fs::copy(&produced, out).with_context(|| {
                        format!(
                            "Failed to copy the profile from {:?} to {:?}",
                            produced, out
                        )
                    })?;
                    eprintln!("[telescope]  profile: {}", out.display());
                }
            }
            Ok(())
        }
    }
}

/// The EM-stage flags of `quant`: everything that is neither a path of an input nor a sample id.
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
    tx_id_mapping: Option<PathBuf>,
    poly_a_source: String,
    debug: bool,
}

/// The parameters of ONE sample's TEL-EM run.
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
    tx_id_mapping: Option<PathBuf>,
    poly_a_source: String,
    debug: bool,
    read_assignments: Option<PathBuf>,
    dump_emission_params: Option<PathBuf>,
    ledger: Option<PathBuf>,
}

impl OneSample {
    /// Assemble one sample's run from the EM flags and the sample's own paths.
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
            tx_id_mapping: f.tx_id_mapping.clone(),
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
        tx_id_mapping,
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
    if let Some(ref p) = tx_id_mapping {
        eprintln!("  tx_id_mapping: {}", p.display());
    }
    eprintln!("  debug:      {}", debug);

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
                continue;
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

    em::validate_emission_file(&emission)?;

    eprintln!("[telescope]  extract evidence…");
    eprintln!("[telescope]  polyA-src:  {}", poly_a_source);
    let mut graph = evidence::extract_evidence_with(&bam, tes_slack, &poly_a)
        .context("extract_evidence failed")?;

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

/// Run the AUTOMATIC path: the anchors, the prior, then one TEL-EM per sample.
#[allow(clippy::too_many_arguments)]
fn run_auto_path(
    bams: &[PathBuf],
    sample_ids: &[String],
    gtf: &Path,
    output_dir: &Path,
    reference_bam: Option<&PathBuf>,
    prior_scope: PriorScopeArg,
    em_flags: &EmFlags,
    ledger: Option<&PathBuf>,
    read_assignments: Option<&PathBuf>,
    dump_emission_params: Option<&PathBuf>,
    sigma2_dof_offset: Option<f64>,
) -> Result<()> {
    let scope = match prior_scope {
        PriorScopeArg::Auto => {
            if bams.len() == 1 {
                PriorScope::SingleSample
            } else {
                PriorScope::Cohort
            }
        }
        PriorScopeArg::Sample => PriorScope::SingleSample,
        PriorScopeArg::Cohort => PriorScope::Cohort,
    };
    eprintln!(
        "[telescope] automatic path: prior scope = {} ({} sample(s) on the command line)",
        scope.as_str(),
        bams.len()
    );
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create the output directory {:?}", output_dir))?;

    let tmp = ScopedTempDir::new("telescope-cli-auto")?;
    eprintln!(
        "[telescope]  temporary intermediates: {}",
        tmp.path().display()
    );

    match scope {
        PriorScope::Cohort => {
            let pool = tmp.path().join("anchor_pool.parquet");
            run_build_anchors(AnchorRequest {
                bams: bams.to_vec(),
                sample_ids: sample_ids.to_vec(),
                output: pool.clone(),
                gtf: Some(gtf.to_path_buf()),
                tes_slack: ANCHOR_TES_SLACK_DEFAULT,
                min_coverage: ANCHOR_MIN_COVERAGE_DEFAULT,
                max_reads_per_sample: ANCHOR_MAX_READS_PER_SAMPLE_DEFAULT,
                gap_abs: ANCHOR_GAP_ABS_DEFAULT,
                gap_rel: ANCHOR_GAP_REL_DEFAULT,
                ledger: None,
            })?;
            let emission_dir = output_dir.join("emission");
            std::fs::create_dir_all(&emission_dir)
                .with_context(|| format!("Failed to create {:?}", emission_dir))?;
            let emission = emission_dir.join("telescope.emission.parquet");
            let universe = universe_for(bams, reference_bam)?;
            run_build_emission(EmissionRequest {
                anchor_pool: pool,

                transcript_evidence: None,
                reference_bam: reference_bam.cloned(),
                reference_universe: universe.clone(),
                gtf: Some(gtf.to_path_buf()),
                provenance_out: None,
                output: emission.clone(),
                tx_min_reads: EMISSION_TX_MIN_READS_DEFAULT,
                gene_min_reads: EMISSION_GENE_MIN_READS_DEFAULT,
                cohort_min_reads: EMISSION_COHORT_MIN_READS_DEFAULT,
                prior_scope: Some(scope),
            })?;
            for (bam, sid) in bams.iter().zip(sample_ids.iter()) {
                let profile = output_dir.join(format!("{sid}.profile.parquet"));
                run_one_sample(
                    OneSample::new(
                        bam.clone(),
                        emission.clone(),
                        sid.clone(),
                        profile,
                        em_flags,
                        with_sample_suffix(ledger, sid),
                        with_sample_suffix(read_assignments, sid),
                        with_sample_suffix(dump_emission_params, sid),
                    ),
                    sigma2_dof_offset,
                )?;
            }
        }
        PriorScope::SingleSample => {
            for (bam, sid) in bams.iter().zip(sample_ids.iter()) {
                let pool = tmp.path().join(format!("anchor_pool.{sid}.parquet"));
                run_build_anchors(AnchorRequest {
                    bams: vec![bam.clone()],
                    sample_ids: vec![sid.clone()],
                    output: pool.clone(),
                    gtf: Some(gtf.to_path_buf()),
                    tes_slack: ANCHOR_TES_SLACK_DEFAULT,
                    min_coverage: ANCHOR_MIN_COVERAGE_DEFAULT,
                    max_reads_per_sample: ANCHOR_MAX_READS_PER_SAMPLE_DEFAULT,
                    gap_abs: ANCHOR_GAP_ABS_DEFAULT,
                    gap_rel: ANCHOR_GAP_REL_DEFAULT,
                    ledger: None,
                })?;

                let dir = if bams.len() > 1 {
                    output_dir.join("emission").join(sid)
                } else {
                    output_dir.join("emission")
                };
                std::fs::create_dir_all(&dir)
                    .with_context(|| format!("Failed to create {:?}", dir))?;
                let emission = dir.join("telescope.emission.parquet");

                let universe = universe_for(std::slice::from_ref(bam), reference_bam)?;
                run_build_emission(EmissionRequest {
                    anchor_pool: pool,
                    transcript_evidence: None,
                    reference_bam: reference_bam.cloned(),
                    reference_universe: universe.clone(),
                    gtf: Some(gtf.to_path_buf()),
                    provenance_out: None,
                    output: emission.clone(),
                    tx_min_reads: EMISSION_TX_MIN_READS_DEFAULT,
                    gene_min_reads: EMISSION_GENE_MIN_READS_DEFAULT,
                    cohort_min_reads: EMISSION_COHORT_MIN_READS_DEFAULT,
                    prior_scope: Some(scope),
                })?;
                let profile = output_dir.join(format!("{sid}.profile.parquet"));
                run_one_sample(
                    OneSample::new(
                        bam.clone(),
                        emission.clone(),
                        sid.clone(),
                        profile,
                        em_flags,
                        with_sample_suffix(ledger, sid),
                        with_sample_suffix(read_assignments, sid),
                        with_sample_suffix(dump_emission_params, sid),
                    ),
                    sigma2_dof_offset,
                )?;
            }
        }
    }
    Ok(())
}

/// The reference universe of the run.
fn universe_for(bams: &[PathBuf], reference_bam: Option<&PathBuf>) -> Result<Option<Vec<String>>> {
    match reference_bam {
        Some(_) => Ok(None),
        None => {
            let u = reference_universe_from_bams(bams)?;
            eprintln!(
                "[telescope]  reference universe: {} transcripts (the @SQ headers \
                 of the {} input BAM(s))",
                u.len(),
                bams.len()
            );
            Ok(Some(u))
        }
    }
}

/// The reference universe of a set of alignment BAMs: the ORDERED UNION of their `@SQ` headers.
fn reference_universe_from_bams(bams: &[PathBuf]) -> Result<Vec<String>> {
    let mut universe: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for bam in bams {
        let u = emission_model::read_reference_universe(bam).with_context(|| {
            format!(
                "Failed to read the reference universe from the BAM header of {:?}",
                bam
            )
        })?;
        for t in u {
            if seen.insert(t.clone()) {
                universe.push(t);
            }
        }
    }
    Ok(universe)
}

/// `<stem>.<sample_id><ext>`: the name a per-sample output gets in the automatic mode.
fn with_sample_suffix(path: Option<&PathBuf>, sample_id: &str) -> Option<PathBuf> {
    let p = path?;
    let stem = p
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let ext = p.extension().map(|s| s.to_string_lossy().into_owned());
    let mut name = format!("{stem}.{sample_id}");
    if let Some(e) = ext {
        name.push('.');
        name.push_str(&e);
    }
    Some(p.with_file_name(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    use clap::Args as ClapArgs;

    fn quant_command() -> clap::Command {
        QuantArgs::augment_args(clap::Command::new("quant"))
    }

    /// The 3' term's DEFAULTS must come from the named constants, not from literals typed into the `default_value_t` attributes.
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

    /// The shipped 3' defaults are 0.0 and 15 bp, and 0.0 is the OFF switch: the E-step must stay bit-identical to a solver without the 3' term.
    #[test]
    fn test_the_three_prime_defaults_are_off_and_a_15_bp_window() {
        assert_eq!(
            THREE_PRIME_WEIGHT_DEFAULT, 0.0,
            "the shipped 3' weight must be 0.0: a non-zero value would change \
             every posterior, and no non-zero value has been validated"
        );
        assert_eq!(
            THREE_PRIME_WINDOW_DEFAULT, 15,
            "the shipped 3' window is 15 bp"
        );
    }

    /// `--three-prime-weight 0` is the documented off switch and must be ACCEPTED, not rejected by the `>= 0` validation.
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
}
