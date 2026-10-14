//! Shared Telescope CLI logic.
//!
//! WHY THIS EXISTS. `quant` is implemented once, here, and called by two
//! binaries: the shipped `telescope` and the research-only
//! `telescope-research`. The production binary passes `None` for the
//! sigma^2 degrees-of-freedom offset and is therefore fixed at
//! `SIGMA2_DOF_OFFSET`; only the research binary can vary it. Both call the
//! SAME function, so the two cannot drift apart -- which is the failure mode
//! that duplicated code would reintroduce, and that the `NU_0` / `c` /
//! self-invalidating-checker incidents in this project's history were all
//! instances of.
//!
//! `ledger_json` is declared HERE and ONLY here. Two live copies of the ledger
//! writer, one under test and one shipping, would differ invisibly.

pub mod ledger_json;
pub mod quant;

// Moved verbatim from main.rs, which had these at module scope.
use std::path::PathBuf;

use anyhow::{Context, Result};

/// Write the record-level ledger if `--ledger` was given.
///
/// Attached to each Module-2 exit rather than inside `print_filter_ledger`,
/// because that function prints and this one writes, and a stderr ledger is
/// not readable by the scrapers that consume this artifact. `None` is the
/// default, so the common path costs one `Option` test.
pub fn maybe_write_filter_ledger(
    path: Option<&PathBuf>,
    stats: &telescope_core::evidence::FilterCounts,
) -> Result<()> {
    if let Some(p) = path {
        ledger_json::write_filter(p, stats)
            .with_context(|| format!("failed to write --ledger {}", p.display()))?;
        eprintln!("[telescope]  filter ledger: {}", p.display());
    }
    Ok(())
}

/// The `--polyA-source` argument, shared verbatim by `extract-evidence` and
/// `quant` so the two subcommands cannot drift apart.
pub const POLY_A_SOURCE_HELP: &str = "Where per-read poly(A) length comes from: \
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
pub fn print_filter_ledger(s: &telescope_core::evidence::FilterCounts, indent: &str) {
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
pub const ALN_WEIGHT_LAMBDA_DEFAULT: f64 = 20.0;

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
pub const FIVE_PRIME_WEIGHT_DEFAULT: f64 = 1.8;

/// Default 5' window in bp: the bonus fires iff the read's observed 5' gap
/// `g_k` in candidate k's cDNA coordinates is at most this. INCLUSIVE (`<=`).
pub const FIVE_PRIME_WINDOW_DEFAULT: u32 = 20;
pub const POLY_A_WEIGHT_DEFAULT: f64 = 1.0;

/// Default `ΔAS` prune threshold. Candidates whose alignment score is more
/// than this far below the read's best are dropped before the softmax, which is
/// where the deletion responsibility moved to once λ dropped to 20 (at λ = 1000
/// the `min_weight` cutoff used to do this implicitly, via a much weaker
/// penalty that happened to cross 4.595 nats only for high-AS reads).
pub const PRUNE_MAX_DELTA_AS_DEFAULT: i32 = 50;

/// Render `numerator / denominator` as a percentage string for the telemetry
/// block, or `"n/a"` when the denominator is zero.
///
/// The `n/a` is the whole point. `0.0 / 0.0` is `NaN` in Rust, and the usual
/// `.max(1)` guard fixes the NaN while rendering "no read qualified" as a
/// confident `0.0%` -- a number that looks measured and is not. That is the
/// exact failure mode this helper exists to prevent, and it is still live: the
/// `_all` / `_resolved` print lines that motivated it were removed in
/// 2026-10-07,
/// but `pct` is still called on `n_reads_any_emission_candidate` and
/// `n_reads_single_emission_candidate`, both legitimately 0 on a sparse model.
///
/// Pinned by `pct_never_renders_a_zero_denominator_as_a_percentage`.
pub fn pct(numerator: u64, denominator: u64) -> String {
    if denominator == 0 {
        "n/a".to_string()
    } else {
        format!("{:.1}%", numerator as f64 / denominator as f64 * 100.0)
    }
}
