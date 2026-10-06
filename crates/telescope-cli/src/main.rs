//! Telescope CLI entry point (Phase C scaffold).
//!
//! Currently exposes three subcommands:
//! - `build-emission`: drives Module 1 (Emission Model Builder).
//! - `extract-evidence`: drives Module 2 (Transcriptome Evidence Extractor).
//! - `quant`: drives Modules 2 → 3 (BAM → evidence → TEL-EM profile).
//!
//! Module 4 R to follow per `specs/telescope-v0.1.7.md`.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use telescope_core::emission_model::FallbackConfig;
use telescope_core::{anchor_builder, em, emission_model, evidence, resolver, summary};

/// The `--polyA-source` argument, shared verbatim by `extract-evidence` and
/// `quant` so the two subcommands cannot drift apart.
const POLY_A_SOURCE_HELP: &str = "Where per-read poly(A) length comes from: \
     `bam_tag:<TAG>` (a two-character SAM tag; `bam_tag:pt` is Dorado native and \
     the default), or `none` (declared-absent; forces \
     `--poly-a-weight 0`). A BAM carrying no poly(A) at all is a hard error, \
     not an empty graph.";

/// Print the Module-2 record filter ledger.
///
/// One function, three call sites, on purpose: the three Module-2 entry points
/// (`extract`, `quant` on one BAM, and `pool` over a cohort) must report the
/// same buckets in the same order, or two of the three numbers are not
/// comparable and a reader has to know which path produced which. The pooled
/// site's counts are the field-wise SUM over the cohort BAMs, so its
/// `records_entered` covers the union of the inputs rather than any one of them.
///
/// The bucket order matches `FilterCounts::dropped_total` so a reader can add
/// the column by eye and check the identity. `deficit` is printed even though
/// the library asserts it is 0: if that assert is ever removed or compiled out,
/// this line is what would show the ledger failing rather than a plausible
/// number quietly omitting a drop site.
fn print_filter_ledger(s: &telescope_core::evidence::FilterCounts, indent: &str) {
    eprintln!("{indent}record filter ledger (RECORDS, not reads, not molecules)");
    eprintln!(
        "{indent}  records entered        : {:>12}",
        s.records_entered
    );
    eprintln!(
        "{indent}    - unmapped            : {:>12}",
        s.dropped_unmapped
    );
    eprintln!(
        "{indent}    - supplementary       : {:>12}",
        s.dropped_supplementary
    );
    eprintln!(
        "{indent}    - poly(A) tag missing : {:>12}",
        s.dropped_no_poly_a_tag
    );
    eprintln!(
        "{indent}    - poly(A) value bad   : {:>12}",
        s.dropped_invalid_poly_a
    );
    eprintln!(
        "{indent}    - no CIGAR span       : {:>12}",
        s.dropped_no_span
    );
    eprintln!(
        "{indent}    - empty span          : {:>12}",
        s.dropped_empty_span
    );
    eprintln!(
        "{indent}    - 3' TES window       : {:>12}",
        s.dropped_tes_window
    );
    eprintln!(
        "{indent}  dropped total          : {:>12}",
        s.dropped_total()
    );
    eprintln!(
        "{indent}  reached candidates     : {:>12}",
        s.records_with_candidates
    );
    eprintln!(
        "{indent}  ledger deficit         : {:>12}  (must be 0)",
        s.ledger_deficit()
    );
}

/// Default λ of the F-EM-4 alignment term `ln A_rk = (AS_rk − max_j AS_rj)/λ`.
/// Referenced by both clap's default and the `quant` telemetry so the shipped
/// default and the claim printed next to it cannot drift apart. Provenance is
/// documented on the `--aln-weight-lambda` argument.
///
/// PROVENANCE — the previous default of 1000.0 was WRONG and is superseded.
/// It was calibrated on ONE gene (PGAM1) and, more importantly, at that
/// temperature every evidence term is compressed to ≤ 0.1 nat, so the EM was
/// effectively ignoring the alignment evidence it exists to weigh. At λ = 20
/// with the `ΔAS > 50` prune the term spans `[−2.5, 0]` nats, which is the
/// range the 5'-anchoring weight of 1.8 was chosen to COMPETE inside — a 7.9
/// weight (also tried, also rejected) would override the alignment evidence
/// outright rather than compete with it. Provenance: the frozen-parameter table
/// in `workspace/notes/EXPERIMENT.md`. Note the λ = 1000 figures cited in the
/// superseded single-gene calibration come from the COPD29 pilot BAM; the
/// active dataset is the 10-sample COPD/HC cohort, so they were never
/// library-wide evidence in the first place.
const ALN_WEIGHT_LAMBDA_DEFAULT: f64 = 20.0;

/// Default weight β of the 5'-anchoring term `ln_5p_k = β if g_k ≤ window`.
///
/// PROVENANCE — 7.9 was TRIED AND REJECTED: with `prune = 50` and `λ = 20` the
/// alignment term spans only `[−2.5, 0]` nats, so a 7.9-nat bonus is an
/// unconditional hard override that no read's own evidence can argue with. 1.8
/// sits inside that range, so the 5' term competes with the alignment term
/// instead of replacing it.
///
/// This is the 5' term's off switch as well as its weight: `--five-prime-weight
/// 0` makes `ln_5p ≡ 0` exactly (`exp(0) = 1`), so the term leaves the softmax
/// exactly rather than approximately. That is the mechanism behind the Test A /
/// Test B ablation, and it is the only correct way to run arm B.
const FIVE_PRIME_WEIGHT_DEFAULT: f64 = 1.8;

/// Default 5' window in bp: the bonus fires iff the read's observed 5' gap
/// `g_k` in candidate k's cDNA coordinates is at most this. INCLUSIVE (`<=`).
const FIVE_PRIME_WINDOW_DEFAULT: u32 = 20;
const POLY_A_WEIGHT_DEFAULT: f64 = 1.0;

/// Default `ΔAS` prune threshold. Candidates whose alignment score is more
/// than this far below the read's best are dropped before the softmax, which is
/// where the deletion responsibility moved to once λ dropped to 20 (at λ = 1000
/// the `min_weight` cutoff used to do this implicitly, via a much weaker
/// penalty that happened to cross 4.595 nats only for high-AS reads).
const PRUNE_MAX_DELTA_AS_DEFAULT: i32 = 50;

#[derive(Parser, Debug)]
#[command(
    name = "telescope",
    version,
    about = "Terminal-resolved native poly(A) inference for ONT DRS"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Module 1: build `telescope.emission.parquet` from an anchor pool
    /// (Universal mode).
    ///
    /// `--anchor-pool` is the sole input; there is no alternative input mode.
    /// An external poly(A) initializer (the former "AIDRS mode") remains in
    /// `telescope-core::emission_model` but is deferred and is NOT reachable
    /// from any CLI subcommand — see that module's docs.
    ///
    /// `--output` is always required. The `--tx-min-reads`,
    /// `--gene-min-reads`, `--cohort-min-reads` flags configure
    /// the Universal-Mode 3-level fallback (Tx → Gene → Cohort).
    /// They used to be conditional on `--anchor-pool`; that is now this
    /// subcommand's only input, so there is nothing left to condition on
    /// and no `requires = "anchor_pool"` is attached to them.
    BuildEmission {
        /// Universal mode: path to `telescope.anchor_pool.parquet`
        /// (produced by `telescope build-anchors`).
        #[arg(long)]
        anchor_pool: PathBuf,

        /// Universal mode (optional): path to
        /// `telescope.transcript_evidence.parquet` (Module 0
        /// output). When provided, reads whose transcript_id has
        /// `em_eligible == false` are dropped before emission
        /// fitting.
        #[arg(long)]
        transcript_evidence: Option<PathBuf>,

        /// Output path for `telescope.emission.parquet`.
        #[arg(long)]
        output: PathBuf,

        /// Universal mode: L1 Tx-level minimum reads (default 10).
        #[arg(long, default_value_t = 10)]
        tx_min_reads: usize,

        /// Universal mode: L2 Gene-level minimum reads (default 30).
        #[arg(long, default_value_t = 30)]
        gene_min_reads: usize,

        /// Universal mode: L3 Cohort-level minimum reads
        /// (informational; L3 is always applied when L1/L2 fail).
        #[arg(long, default_value_t = 50)]
        cohort_min_reads: usize,

        /// REFERENCE UNIVERSE (padding). Path to the alignment BAM whose `@SQ`
        /// header names every transcript the alignment can assign to.
        ///
        /// Without it the model covers only transcripts the anchor pool
        /// observed, and a read whose candidate is outside that set is
        /// dropped with no profile row -- 705,289 of them on HC2, 14.7% of all
        /// reads. With it, every universe transcript missing from the pool is
        /// emitted as a cohort-prior row carrying `n_cohort_flnc == 0`.
        ///
        /// Only the HEADER is read; no alignment record is touched.
        #[arg(long)]
        reference_bam: Option<PathBuf>,

        /// Optional `transcript_id<TAB>gene_id` TSV, used to give IMPUTED
        /// rows a real `gene_id` instead of falling back to the transcript id.
        ///
        /// The fallback is a different convention from the anchored rows in
        /// the same file, which is harmless numerically -- `gene_id` feeds no
        /// likelihood -- but it is a join trap for a downstream reader, so it
        /// is worth passing when the mapping is on hand.
        #[arg(long)]
        tx_to_gene: Option<PathBuf>,

        /// Where to write the provenance sidecar JSON. Defaults to
        /// `<output>.provenance.json`.
        ///
        /// The parquet's 7 columns are frozen, so the record of WHERE the
        /// universe came from and WHICH rows were imputed cannot live in it.
        #[arg(long)]
        provenance_out: Option<PathBuf>,
    },

    /// Module 2: extract single-molecule evidence from AIDRS-aligned BAM.
    ExtractEvidence {
        /// Path to AIDRS-aligned transcriptome BAM (carries Dorado `pt:i`).
        #[arg(long)]
        bam: PathBuf,

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
    },

    /// Modules 2 + 3: BAM → evidence → TEL-EM profile.
    Quant {
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
        /// The file is loader-compatible with `--emission` (`transcript_id`,
        /// `gene_id`, `is_polyA_competent`, `mu_logPAL`, `sigma2_logPAL`), so a
        /// converged fit can be fed back in to measure the converged state
        /// without re-deriving it. `mu_cohort` is carried as its own column so
        /// the shrinkage prior stays auditable: re-loading this file re-anchors
        /// BOTH priors to the fitted values (`load_emission_parameters` sets
        /// `mu_cohort = mu_logPAL` and `sigma2_cohort = sigma2_logPAL`), so a
        /// second pass applies τ₀=2 and 2β₀ a second time. Compare those columns
        /// before trusting a re-loaded fit.
        ///
        /// This is a DIAGNOSTIC sidecar. The published profile and
        /// read-assignments schemas are unchanged.
        #[arg(long)]
        dump_emission_params: Option<PathBuf>,
    },

    /// Profile Summary Engine: aggregate profile.parquet → sample_summary.parquet.
    Summarize {
        /// Path to `telescope.profile.parquet`.
        #[arg(long)]
        profile: PathBuf,

        /// Output `telescope.sample_summary.parquet`.
        #[arg(long)]
        output: PathBuf,

        /// Per-transcript provenance sidecar from `build-emission`
        /// (`<stem>.emission.provenance.parquet`).
        ///
        /// OPTIONAL, and the difference is not cosmetic. Without it the key set
        /// is whatever the profile happens to contain, which is an INNER JOIN
        /// against the reference universe: a transcript no read landed on is
        /// absent from the output entirely. With it, every transcript in the
        /// universe gets a row, zero-filled, carrying its `tier` and
        /// `n_anchor` -- which is what Tier 3 needs, because NanoCount
        /// transcripts that are missing there must be filled with 0 rather than
        /// dropped by a join.
        #[arg(long)]
        provenance: Option<PathBuf>,
    },

    /// D1 Universal Mode: build poly(A) anchor pool from BAMs (no AIDRS needed).
    BuildAnchors {
        /// BAM files (one per sample). Pass multiple `--bam <path>` flags.
        #[arg(long = "bam", required = true)]
        bams: Vec<PathBuf>,

        /// Sample IDs parallel to `--bam` (one per BAM).
        #[arg(long = "sample-id", required = true)]
        sample_ids: Vec<String>,

        /// Output `telescope.anchor_pool.parquet`.
        #[arg(long)]
        output: PathBuf,

        /// Optional TSV with header `transcript_id\tgene_id` (for gene_id resolution).
        #[arg(long)]
        tx_to_gene: Option<PathBuf>,

        /// 3' TES boundary slack in bp (default 35).
        #[arg(long, default_value_t = 35)]
        tes_slack: usize,

        /// Minimum coverage ratio (alignment_length / ref_len) (default 0.80).
        #[arg(long, default_value_t = 0.80)]
        min_coverage: f64,

        /// Per-sample read cap per transcript (default 200).
        #[arg(long, default_value_t = 200)]
        max_reads_per_sample: usize,
    },

    /// Module 0: Resolve cohort-level transcript evidence (Tier A/B/C,
    /// subsumption, EM/PAL eligibility) → `telescope.transcript_evidence.parquet`.
    ResolveEvidence {
        /// One or more AIDRS-aligned transcriptome BAMs (cohort). All BAMs
        /// must share the same reference catalog (`@SQ` lines).
        #[arg(long = "bam", required = true)]
        bams: Vec<PathBuf>,

        /// Sample IDs parallel to `--bam` (one per BAM, used to namespace
        /// read IDs when pooling).
        #[arg(long = "sample-id", required = true)]
        sample_ids: Vec<String>,

        /// Optional TSV with header `transcript_id\tgene_id`. Missing
        /// mappings emit `gene_id="missing_gene_id"` plus a warning.
        #[arg(long)]
        tx_to_gene: Option<PathBuf>,

        /// Cohort size (must equal the number of `--bam` arguments).
        /// Required and explicit per F6.2 (P0 correction).
        #[arg(long)]
        cohort_size: usize,

        /// TES cluster radius in bp (default 50). Excluded from
        /// subsumption scan iff transcript IDs contain `AlterTes`/`_Alter`.
        #[arg(long, default_value_t = 50)]
        tes_cluster_radius: usize,

        /// 3' TES boundary slack in bp passed to Module 2 (default 35).
        #[arg(long, default_value_t = 35)]
        tes_slack: usize,

        /// Output `telescope.transcript_evidence.parquet`.
        #[arg(long)]
        output: PathBuf,
    },
}

/// Render `numerator / denominator` as a percentage string for the telemetry
/// block, or `"n/a"` when the denominator is zero.
///
/// The `n/a` is the whole point. `0.0 / 0.0` is `NaN` in Rust, and the usual
/// `.max(1)` guard fixes the NaN while rendering "no read qualified" as a
/// confident `0.0%` -- a number that looks measured and is not. That is the
/// exact failure mode the `_all` / `_resolved` split was made to remove, so
/// reintroducing it in the print would undo the change at the last step.
///
/// Pinned by `pct_never_renders_a_zero_denominator_as_a_percentage`.
fn pct(numerator: u64, denominator: u64) -> String {
    if denominator == 0 {
        "n/a".to_string()
    } else {
        format!("{:.1}%", numerator as f64 / denominator as f64 * 100.0)
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::BuildEmission {
            anchor_pool,
            transcript_evidence,
            reference_bam,
            tx_to_gene,
            provenance_out,
            output,
            tx_min_reads,
            gene_min_reads,
            cohort_min_reads,
        } => {
            eprintln!("[telescope] Module 1: building emission model");
            // Universal mode is the only mode; `--anchor-pool` is required.
            let fb = FallbackConfig {
                tx_min_reads,
                gene_min_reads,
                cohort_min_reads,
            };
            eprintln!("  mode:               Universal");
            eprintln!("  anchor_pool:        {}", anchor_pool.display());
            if let Some(ref te) = transcript_evidence {
                eprintln!("  transcript_evidence: {}", te.display());
            } else {
                eprintln!("  transcript_evidence: (none; no em_eligible filter)");
            }
            eprintln!("  tx_min_reads:       {}", fb.tx_min_reads);
            eprintln!("  gene_min_reads:     {}", fb.gene_min_reads);
            eprintln!("  cohort_min_reads:   {}", fb.cohort_min_reads);
            eprintln!("  output:             {}", output.display());
            // Reference universe: read HEADER-ONLY from the alignment BAM.
            let universe: Option<Vec<String>> = match reference_bam {
                Some(ref bam) => {
                    eprintln!("  reference_bam:     {}", bam.display());
                    let u = emission_model::read_reference_universe(bam)
                        .context("Failed to read the reference universe from the BAM header")?;
                    eprintln!("  @SQ transcripts:  {}", u.len());
                    Some(u)
                }
                None => {
                    eprintln!(
                        "  reference_bam:     (none; the model will cover only the \
                         anchor pool, and a read whose candidate is outside it will be \
                         dropped with no profile row)"
                    );
                    None
                }
            };
            let tx2g = match tx_to_gene {
                Some(ref p) => {
                    eprintln!("  tx_to_gene:        {}", p.display());
                    Some(
                        anchor_builder::TxToGene::from_tsv(p)
                            .context("Failed to load tx-to-gene TSV")?,
                    )
                }
                None => {
                    eprintln!(
                        "  tx_to_gene:        (none; imputed rows use gene_id := transcript_id)"
                    );
                    None
                }
            };
            // Captured before `output` and `reference_bam` are moved into the
            // call below: the sidecar defaults to sitting beside the parquet,
            // and it records which file supplied the universe.
            let prov_path: PathBuf = provenance_out.clone().unwrap_or_else(|| {
                let mut q = output.clone();
                let stem = q
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "emission".to_string());
                q.set_file_name(format!("{stem}.provenance.json"));
                q
            });
            let reference_source_for_prov = reference_bam.clone();
            let tiers = emission_model::build_emission_from_anchors(
                anchor_pool,
                transcript_evidence,
                output,
                &fb,
                universe.as_deref(),
                tx2g.as_ref(),
            )
            .context("Module 1 build_emission_from_anchors (Universal) failed")?;
            // L1/L2/L3 split, read off the RETURN VALUE rather than scraped
            // from the library's stderr. The denominator is the number of
            // emitted rows, guarded against the zero-transcript case.
            let n_tx_total = (tiers.l1 + tiers.l2 + tiers.l3).max(1) as f64;
            eprintln!(
                "[telescope]  fallback tiers: L1={} ({:.1}%) L2={} ({:.1}%) L3={} ({:.1}%)",
                tiers.l1,
                tiers.l1 as f64 * 100.0 / n_tx_total,
                tiers.l2,
                tiers.l2 as f64 * 100.0 / n_tx_total,
                tiers.l3,
                tiers.l3 as f64 * 100.0 / n_tx_total,
            );

            // The L1/L2/L3 percentages above keep describing the ANCHORED
            // population: imputed rows are counted separately so those
            // percentages stay comparable with every earlier run rather than
            // silently rebasing onto the padded denominator.
            //
            // ---- EMISSION COMPLETENESS -------------------------------------
            // Four counters, none inferable from the others. The point of
            // printing `n_emission_after` on its own line is that a reader can
            // check it against `n_reference` without trusting the summary.
            let n_anchored = tiers.l1 + tiers.l2 + tiers.l3;
            let n_after = n_anchored + tiers.n_imputed;
            eprintln!(
                "[telescope]  emission completeness: {} anchored, {} unanchored L3-padded ({:.1}%), total {}",
                n_anchored,
                tiers.n_imputed,
                if tiers.n_reference == 0 {
                    0.0
                } else {
                    100.0 * tiers.n_imputed as f64 / tiers.n_reference as f64
                },
                n_after,
            );
            eprintln!(
                "[telescope]    n_reference_transcripts: {}",
                tiers.n_reference
            );
            eprintln!("[telescope]    n_emission_before:        {}", n_anchored);
            eprintln!("[telescope]    n_emission_after:         {}", n_after);
            eprintln!(
                "[telescope]    n_imputed_rows:           {}",
                tiers.n_imputed
            );
            if tiers.n_anchored_not_in_reference > 0 {
                eprintln!(
                    "[telescope][WARN] {} anchored transcript(s) are absent from the reference universe, so n_emission_after ({}) is NOT the universe size ({}). The universe and the model disagree about which transcripts exist.",
                    tiers.n_anchored_not_in_reference,
                    n_after,
                    tiers.n_reference,
                );
            }

            // ---- PROVENANCE SIDECAR -----------------------------------------
            // The parquet keeps 7 frozen columns, so "where did the universe
            // come from" and "which rows were imputed" cannot be recorded
            // inside it. They are recorded here, beside it.
            //
            // `reference_checksum` is explicitly null WITH A STATED REASON
            // rather than omitted: "computed and empty" and "not computed" are
            // different claims, and a reader must be able to tell them apart.
            let prov = format!(
                "{{\n  \"reference_source\": {},\n  \"reference_count\": {},\n  \"reference_checksum\": null,\n  \"reference_checksum_note\": \"not computed: the universe is taken verbatim from the BAM @SQ header, which is the authority by ruling. No reference fasta or GTF was hashed, so `reference_source` identifies a FILE, not a universe; `reference_count` is the checkable quantity.\",\n  \"imputation_rule\": \"cohort_prior\",\n  \"imputed_marker\": \"n_cohort_flnc == 0\",\n  \"schema_columns\": 7,\n  \"n_anchored\": {},\n  \"n_imputed\": {},\n  \"n_total\": {},\n  \"n_anchored_not_in_reference\": {}\n}}\n",
                match reference_source_for_prov {
                    Some(ref p) => format!("{:?}", p.display().to_string()),
                    None => "null".to_string(),
                },
                tiers.n_reference,
                n_anchored,
                tiers.n_imputed,
                n_after,
                tiers.n_anchored_not_in_reference,
            );
            std::fs::write(&prov_path, prov)
                .with_context(|| format!("Failed to write provenance sidecar {:?}", prov_path))?;
            eprintln!("[telescope]  provenance JSON : {}", prov_path.display());
            // The PER-TRANSCRIPT sidecar is the one `summarize --provenance`
            // consumes, and it is a different file from the JSON above. Both are
            // printed because they are easy to confuse and the confusion is
            // silent: handing the JSON to `--provenance` fails as a missing
            // parquet column, which reads like a corrupt build rather than
            // like the wrong argument.
            eprintln!(
                "[telescope]  provenance table: {}",
                tiers.provenance_path.display()
            );
            eprintln!(
                "[telescope]    (pass THIS to `summarize --provenance`; the JSON above \
                 is a human-readable run header)"
            );
            eprintln!("[telescope] done.");
            Ok(())
        }
        Cmd::ExtractEvidence {
            bam,
            tes_slack,
            poly_a_source,
        } => {
            let poly_a = evidence::parse_poly_a_source(&poly_a_source)?;
            eprintln!("[telescope] Module 2: extracting single-molecule evidence");
            eprintln!("  bam:        {}", bam.display());
            eprintln!("  tes_slack:  {} bp", tes_slack);
            eprintln!("  polyA-src:  {}", poly_a_source);
            let graph = evidence::extract_evidence_with(&bam, tes_slack, &poly_a)
                .context("Module 2 extract_evidence failed")?;
            eprintln!("[telescope]  molecules:        {}", graph.molecules.len());
            eprintln!(
                "[telescope]  transcripts:     {}",
                graph.transcript_names.len()
            );
            eprintln!(
                "[telescope]  candidate_pool:   {} entries",
                graph.candidate_pool.len()
            );
            let total_cands: u64 = graph
                .molecules
                .iter()
                .map(|m| m.candidate_count as u64)
                .sum();
            print_filter_ledger(&graph.stats, "[telescope]  ");
            let avg = if graph.molecules.is_empty() {
                0.0
            } else {
                total_cands as f64 / graph.molecules.len() as f64
            };
            eprintln!("[telescope]  avg candidates:   {:.2}", avg);
            eprintln!("[telescope] done.");
            Ok(())
        }
        Cmd::Quant {
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
        } => {
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
                    let l = line
                        .with_context(|| format!("Failed to read line {} from {:?}", i + 1, p))?;
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
            let stats = em::solve_and_write_profile(
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
            eprintln!(
                "[telescope]      -- contested, decided by the E-step (tested in this order):"
            );
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
                stats.n_degenerate_all_imputed as f64 / stats.n_entropy_degenerate.max(1) as f64
                    * 100.0
            );
            eprintln!(
                "[telescope]        of which real candidate competition: {} ({:.1}% of degenerate)",
                stats.n_degenerate_with_anchored,
                stats.n_degenerate_with_anchored as f64 / stats.n_entropy_degenerate.max(1) as f64
                    * 100.0
            );
            // Degradation telemetry. TWO zero-power metrics, because at less
            // than full model coverage the old one is largely vacuous -- a read
            // with one emission candidate satisfies "all candidates share one
            // mu" without comparing anything. The old one is kept (renamed
            // `_all`) so runs stay comparable; the new one excludes those reads
            // from BOTH numerator and denominator.
            //
            // A denominator of ZERO PRINTS `n/a`, NOT `0.0%`. `0.0 / 0.0` in
            // Rust is NaN, so an unguarded line prints "NaN%" -- and a
            // `.max(1)` guard alone fixes the NaN while making "no read
            // qualified" render as a confident 0.0%. That is the same class of
            // defect as the metric this change is about: a number that looks
            // measured and is not. `n_reads_multi_emission_candidate` is
            // legitimately 0 on a sparse model, which is exactly the regime
            // this metric exists for.
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
            eprintln!("[telescope]    poly(A) discriminative power:");
            eprintln!(
                "[telescope]      zero-power_all     (identical mu/sigma2 across all \
                 candidates that have an emission row): {} of {} ambiguous ({})",
                stats.n_polya_zero_power,
                stats.n_ambiguous,
                pct(stats.n_polya_zero_power, stats.n_ambiguous as u64)
            );
            eprintln!(
                "[telescope]      zero-power_resolved (same, but only over reads with \
                 >=2 emission candidates; CONVERGED mu, not the loaded model -- \
                 the M-step pulls every weighted transcript's mu toward its own \
                 reads, so this trends to 0 on any run that converges): {} of {} ({})",
                stats.n_polya_zero_power_resolved,
                stats.n_reads_multi_emission_candidate,
                pct(
                    stats.n_polya_zero_power_resolved,
                    stats.n_reads_multi_emission_candidate
                )
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
        Cmd::Summarize {
            profile,
            output,
            provenance,
        } => {
            eprintln!("[telescope] Profile Summary Engine");
            eprintln!("  profile: {}", profile.display());
            eprintln!("  output:  {}", output.display());
            if let Some(p) = &provenance {
                eprintln!("  provenance: {}", p.display());
            }
            summary::summarize_profile_parquet(&profile, &output, provenance.as_deref())
                .context("summary::summarize_profile_parquet failed")?;
            eprintln!("[telescope] done.");
            Ok(())
        }
        Cmd::BuildAnchors {
            bams,
            sample_ids,
            output,
            tx_to_gene,
            tes_slack,
            min_coverage,
            max_reads_per_sample,
        } => {
            if bams.len() != sample_ids.len() {
                anyhow::bail!(
                    "--bam ({} files) and --sample-id ({} ids) must have equal length",
                    bams.len(),
                    sample_ids.len()
                );
            }
            eprintln!("[telescope] D1 Universal Mode: anchor pool builder");
            eprintln!("  samples:    {}", bams.len());
            eprintln!("  tes_slack:  {} bp", tes_slack);
            eprintln!("  min_cov:    {}", min_coverage);
            eprintln!("  cap_per_tx: {}", max_reads_per_sample);
            eprintln!("  output:     {}", output.display());

            let tx_to_g = match tx_to_gene {
                Some(p) => anchor_builder::TxToGene::from_tsv(&p)
                    .with_context(|| format!("Failed to load tx-to-gene TSV: {:?}", p))?,
                None => anchor_builder::TxToGene::new(),
            };
            eprintln!("  tx_to_gene entries: {}", tx_to_g.len());

            let pairs: Vec<(PathBuf, &str)> = bams
                .iter()
                .cloned()
                .zip(sample_ids.iter().map(|s| s.as_str()))
                .collect();
            let stats = anchor_builder::build_anchor_pool(
                &pairs,
                &tx_to_g,
                &output,
                tes_slack,
                min_coverage,
                max_reads_per_sample,
            )
            .context("anchor_builder::build_anchor_pool failed")?;
            eprintln!("[telescope]  {}", stats.report());
            // The `tx_to_gene entries: 0` line above and this one are far apart,
            // and neither of them, alone, says what L2=0 downstream will mean.
            // Say it here, where the pool that caused it was just written.
            if let Some(warning) = anchor_builder::degenerate_gene_pool_warning(
                stats.distinct_genes,
                stats.distinct_transcripts,
            ) {
                eprintln!("{}", warning);
            }
            eprintln!("[telescope] done.");
            Ok(())
        }
        Cmd::ResolveEvidence {
            bams,
            sample_ids,
            tx_to_gene,
            cohort_size,
            tes_cluster_radius,
            tes_slack,
            output,
        } => {
            if bams.len() != sample_ids.len() {
                anyhow::bail!(
                    "--bam ({} files) and --sample-id ({} ids) must have equal length",
                    bams.len(),
                    sample_ids.len()
                );
            }
            if cohort_size != bams.len() {
                anyhow::bail!(
                    "--cohort-size ({}) must equal the number of --bam arguments ({})",
                    cohort_size,
                    bams.len()
                );
            }
            eprintln!("[telescope] Module 0: transcript evidence resolver");
            eprintln!("  cohort_size:         {}", cohort_size);
            eprintln!("  tes_cluster_radius:  {} bp", tes_cluster_radius);
            eprintln!("  tes_slack:           {} bp", tes_slack);
            eprintln!("  output:              {}", output.display());

            // Pool evidence across all cohort BAMs.
            let pairs: Vec<(PathBuf, String)> = bams
                .iter()
                .cloned()
                .zip(sample_ids.iter().cloned())
                .collect();
            let (graph, n_pooled) = resolver::pool_evidence_graphs(&pairs, tes_slack)
                .context("resolver::pool_evidence_graphs failed")?;
            eprintln!(
                "[telescope]  pooled {} BAM(s): molecules={}, transcripts={}, candidate_pool={}",
                n_pooled,
                graph.molecules.len(),
                graph.transcript_names.len(),
                graph.candidate_pool.len(),
            );
            print_filter_ledger(&graph.stats, "[telescope]  ");

            // Load transcript→gene mapping (TSV) → HashMap<String, String>.
            let tx_to_gene_map: std::collections::HashMap<String, String> = match tx_to_gene {
                Some(p) => {
                    let content = std::fs::read_to_string(&p)
                        .with_context(|| format!("Failed to read tx-to-gene TSV: {:?}", p))?;
                    let mut m = std::collections::HashMap::new();
                    for (i, line) in content.lines().enumerate() {
                        if i == 0 && line.starts_with("transcript_id") {
                            continue;
                        }
                        let mut parts = line.splitn(2, '\t');
                        let tx = parts.next().unwrap_or("").to_string();
                        let g = parts.next().unwrap_or("").to_string();
                        if !tx.is_empty() && !g.is_empty() {
                            m.insert(tx, g);
                        }
                    }
                    eprintln!("  tx_to_gene entries:  {}", m.len());
                    m
                }
                None => {
                    eprintln!("  tx_to_gene entries:  0 (no TSV provided; all transcripts will emit missing_gene_id)");
                    std::collections::HashMap::new()
                }
            };

            let config = resolver::ResolverConfig {
                cohort_size,
                tes_cluster_radius,
            };
            let mut warn_set: HashSet<String> = HashSet::new();
            let records = resolver::resolve_transcript_evidence(
                &graph,
                &tx_to_gene_map,
                &config,
                &mut warn_set,
            );

            let n_tier_a = records.iter().filter(|r| r.tier == "TierA").count();
            let n_tier_b = records.iter().filter(|r| r.tier == "TierB").count();
            let n_tier_c = records.iter().filter(|r| r.tier == "TierC").count();
            let n_subsumed = records.iter().filter(|r| r.is_subsumed).count();
            let n_em_eligible = records.iter().filter(|r| r.em_eligible).count();
            let n_pal_candidate = records.iter().filter(|r| r.pal_candidate).count();
            eprintln!(
                "[telescope]  TierA={} TierB={} TierC={} subsumed={} em_eligible={} pal_candidate={}",
                n_tier_a, n_tier_b, n_tier_c, n_subsumed, n_em_eligible, n_pal_candidate
            );

            resolver::write_transcript_evidence_parquet(&records, &output)
                .context("resolver::write_transcript_evidence_parquet failed")?;
            eprintln!("[telescope] done.");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// A zero denominator must render as `n/a`, never as a percentage.
    ///
    /// `n_reads_multi_emission_candidate` is legitimately 0 whenever no read
    /// has two candidates carrying an emission row — which is the normal state
    /// of a badly-covered model, and precisely the state the `_all` / `_resolved`
    /// split was created to characterise. The tempting guard is
    /// `denominator.max(1)`, which removes the `NaN` and simultaneously reports
    /// "no read qualified" as a confident `0.0%`.
    ///
    /// Asserting the STRING rather than the absence of a panic is the point:
    /// a test that only checked "does not produce NaN" would still pass under
    /// the `.max(1)` version, which is the defect.
    #[test]
    fn pct_never_renders_a_zero_denominator_as_a_percentage() {
        assert_eq!(
            pct(0, 0),
            "n/a",
            "0 of 0 is UNDEFINED, not 0.0% -- this is the vacuous-metric defect \
             reappearing in the print"
        );
        // A real zero over a real denominator IS a real zero and must still say so.
        assert_eq!(pct(0, 715_136), "0.0%");
        assert_eq!(pct(1, 4), "25.0%");
        assert_eq!(pct(1, 3), "33.3%");
    }

    /// Pins the shipped defaults.
    ///
    /// This file previously had NO test module, so nothing in the repository
    /// asserted the value of any default: every `em.rs` test passes λ (and now
    /// the 5' weight, window and prune threshold) EXPLICITLY, by the existing
    /// convention of not letting a unit test read a CLI default. The
    /// consequence was that bumping `ALN_WEIGHT_LAMBDA_DEFAULT` back to the
    /// superseded 1000.0 left the entire suite green — verified by mutation.
    ///
    /// The defaults are the frozen parameters, and the whole change is
    /// "ship these numbers", so the numbers themselves need a guard. The
    /// backend tests pin what each value DOES; this pins what is SHIPPED.
    ///
    /// Mutation target: changing any of the four constants, or changing a
    /// `default_value_t` to diverge from its constant, turns this red — which
    /// is the drift the doc comment on `ALN_WEIGHT_LAMBDA_DEFAULT` promises
    /// cannot happen.
    #[test]
    fn test_shipped_defaults_are_the_frozen_parameters() {
        assert_eq!(
            ALN_WEIGHT_LAMBDA_DEFAULT, 20.0,
            "λ = 1000 compressed every evidence term to <= 0.1 nat and is superseded"
        );
        assert_eq!(FIVE_PRIME_WEIGHT_DEFAULT, 1.8, "frozen 5' weight");
        assert_eq!(FIVE_PRIME_WINDOW_DEFAULT, 20, "frozen 5' window, inclusive");
        assert_eq!(PRUNE_MAX_DELTA_AS_DEFAULT, 50, "frozen ΔAS prune threshold");
    }

    /// The `default_value_t` attributes must resolve to the SAME constants.
    ///
    /// clap derives the shipped default from `default_value_t`, and renders it
    /// into `[default: N]` in `--help`. A literal typed into the attribute
    /// instead of the constant would therefore print a help block
    /// contradicting the telemetry and the doc comment — exactly the drift
    /// that happened before this change, when the prose said
    /// `(default: 1000.0)` while the const said something else.
    ///
    /// Asserted by PARSING rather than by matching the rendered string, because
    /// clap formats `f64` its own way (20.0 renders as `20`, not `20.0`) and
    /// that is an implementation detail, not the property under test.
    #[test]
    fn test_quant_flag_defaults_come_from_the_constants() {
        let m = Cli::command()
            .try_get_matches_from([
                "telescope",
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
        let sub = m.subcommand_matches("quant").expect("quant subcommand");
        assert_eq!(
            *sub.get_one::<f64>("aln_weight_lambda").unwrap(),
            ALN_WEIGHT_LAMBDA_DEFAULT,
            "--aln-weight-lambda must default to the constant, not a literal"
        );
        assert_eq!(
            *sub.get_one::<f64>("five_prime_weight").unwrap(),
            FIVE_PRIME_WEIGHT_DEFAULT,
            "--five-prime-weight must default to the constant, not a literal"
        );
        assert_eq!(
            *sub.get_one::<u32>("five_prime_window").unwrap(),
            FIVE_PRIME_WINDOW_DEFAULT,
            "--five-prime-window must default to the constant, not a literal"
        );
        assert_eq!(
            *sub.get_one::<i32>("prune_max_delta_as").unwrap(),
            PRUNE_MAX_DELTA_AS_DEFAULT,
            "--prune-max-delta-as must default to the constant, not a literal"
        );
    }

    /// `--five-prime-weight 0` is the documented off switch and must be
    /// ACCEPTED, not rejected by the `>= 0` validation. A `> 0` guard would
    /// silently make the Test A / Test B ablation unrunnable.
    #[test]
    fn test_five_prime_weight_zero_parses_as_the_off_switch() {
        let cmd = Cli::command()
            .try_get_matches_from([
                "telescope",
                "quant",
                "--bam",
                "in.bam",
                "--emission",
                "e.parquet",
                "--sample-id",
                "S1",
                "--output",
                "o.parquet",
                "--five-prime-weight",
                "0",
            ])
            .expect("weight 0 must be accepted");
        let sub = cmd.subcommand_matches("quant").expect("quant subcommand");
        assert_eq!(
            *sub.get_one::<f64>("five_prime_weight").unwrap(),
            0.0,
            "the off switch must survive parsing exactly"
        );
    }
}
