//! Shared Telescope CLI logic.
//!
//! `quant` is implemented once here and called by both binaries, the shipped
//! `telescope` and the research-only `telescope-research`. The production
//! binary passes `None` for the sigma^2 degrees-of-freedom offset and is
//! therefore fixed at `SIGMA2_DOF_OFFSET`; only the research binary can vary
//! it, so the two cannot drift apart.
//!
//! `ledger_json` is declared HERE and ONLY here: two live copies of the ledger
//! writer would differ invisibly.
//!
//! Items: `maybe_write_filter_ledger`, `print_filter_ledger`, `pct`, and the
//! frozen default constants.

pub mod diff;
pub mod ledger_json;
pub mod quant;

use std::path::PathBuf;

use anyhow::{Context, Result};

/// Write the record-level filter ledger to `path` if `--ledger` was given.
///
/// Kept out of `print_filter_ledger` because a stderr ledger is not readable by
/// the scrapers that consume this artifact.
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
/// Traps: the bucket order must match `FilterCounts::dropped_total`; the pooled
/// site's counts are the field-wise SUM over the cohort BAMs, so its
/// `records_entered` covers the union of the inputs; `deficit` is printed even
/// though the library asserts it is 0.
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
/// default and the claim printed next to it cannot drift apart.
///
/// Trap: λ is not a tuning knob — at λ = 1000 every evidence term compresses
/// to ≤ 0.1 nat, so the EM barely uses the alignment evidence it exists to weigh.
pub const ALN_WEIGHT_LAMBDA_DEFAULT: f64 = 20.0;

/// Default weight β of the 5'-anchoring term `ln_5p_k = β if g_k ≤ window`.
///
/// Traps: `--five-prime-weight 0` makes `ln_5p ≡ 0` exactly, so the term
/// leaves the softmax exactly; and β must stay inside the alignment term's nat
/// range, since a bonus far above it is an unconditional override.
pub const FIVE_PRIME_WEIGHT_DEFAULT: f64 = 1.8;

/// Default 5' window in bp: the bonus fires iff the read's observed 5' gap
/// `g_k` in candidate k's cDNA coordinates is at most this. INCLUSIVE (`<=`).
pub const FIVE_PRIME_WINDOW_DEFAULT: u32 = 20;

/// Default weight on the poly(A) mixture term. 1.0 is the shipped model.
pub const POLY_A_WEIGHT_DEFAULT: f64 = 1.0;

/// Default `ΔAS` prune threshold: candidates whose alignment score is more
/// than this far below the read's best are dropped before the softmax.
///
/// Trap: at the shipped λ the alignment term cannot delete candidates on its
/// own, so this threshold is what carries the deletion responsibility.
pub const PRUNE_MAX_DELTA_AS_DEFAULT: i32 = 50;

/// Render `numerator / denominator` as a percentage string for the telemetry
/// block, or `"n/a"` when the denominator is zero.
///
/// Trap: `0.0 / 0.0` is `NaN` in Rust and a `.max(1)` guard would render "no
/// read qualified" as a confident `0.0%`, so a zero denominator must print
/// `n/a`. Pinned by `pct_never_renders_a_zero_denominator_as_a_percentage`.
pub fn pct(numerator: u64, denominator: u64) -> String {
    if denominator == 0 {
        "n/a".to_string()
    } else {
        format!("{:.1}%", numerator as f64 / denominator as f64 * 100.0)
    }
}
