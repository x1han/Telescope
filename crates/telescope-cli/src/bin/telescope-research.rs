//! Research-only entry point. **Not shipped, not part of the release gate.**
//!
//! WHY THIS EXISTS AS A SEPARATE BINARY. `SIGMA2_DOF_OFFSET` is a `pub const`
//! so that the shipped path cannot be moved at runtime; the production
//! `telescope` therefore exposes no flag that reaches it. But the σ²
//! denominator offset `c` is exactly what a sensitivity analysis has to vary,
//! and the only way to vary it is to reach
//! `solve_and_write_profile_with_dof_offset` from a command line.
//!
//! Putting the knob on `telescope quant` would make the experimental surface
//! and the production surface the same surface, which is precisely what the
//! constant was made `const` to prevent. So: one `quant` implementation in
//! `telescope::quant`, called by two binaries that differ only in what they
//! pass.
//!
//! NOT A FORK. `quant` is defined once, in the library, and both binaries
//! parse the same `QuantArgs`. A research binary that retyped the 18 flags
//! would compile, run, and silently be a different model the moment a default
//! drifted -- the same shape as the `NU_0` and `c` incidents this project has
//! already had once each.
//!
//! EVERYTHING ELSE IS DELEGATED, not reimplemented. `build-anchors` and
//! `build-emission` do not read `c`, so they are exec'd from the production
//! binary rather than duplicated: same program, same bytes, same behaviour by
//! construction.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;

use clap::{Parser, Subcommand};

use telescope::quant::QuantArgs;

#[derive(Parser, Debug)]
#[command(
    name = "telescope-research",
    version,
    about = "EXPERIMENTAL: same pipeline as `telescope`, with the sigma^2 dof offset exposed"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

// clippy's `large_enum_variant` fires here: `Quant` is 304 bytes against
// `Passthrough`'s 24. Boxing `QuantArgs` would trade 280 bytes of stack in a
// frame that is built once and matched once for a heap allocation on every
// invocation -- a worse trade for a 304-byte stack frame, not a better one.
// Do not "fix" this by boxing; if `Quant` grows, fix the growth.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand, Debug)]
enum Cmd {
    /// Modules 2 + 3: BAM -> evidence -> TEL-EM profile.
    ///
    /// Identical to `telescope quant` in every respect except that the trailing
    /// constant `c` of the sigma^2 denominator (`W + 2*nu_0 + c`) is settable.
    /// Omitting the flag is exactly `telescope quant`: both entries call the
    /// same implementation with `SIGMA2_DOF_OFFSET`.
    ///
    /// `c = 2` IS Model A. `c = 3` is NOT Model B -- Model B also needs the
    /// `tau_0*(mu-mu_cohort)^2` term back in the numerator. Any other `c` is a
    /// sensitivity perturbation around the production value and carries no
    /// model identity.
    Quant {
        #[command(flatten)]
        args: QuantArgs,

        /// Trailing constant `c` of the sigma^2 denominator. Defaults to the
        /// production Model A value. This flag is the whole reason this binary
        /// exists and it has no counterpart on the production `telescope`.
        #[arg(long)]
        sigma2_dof_offset: Option<f64>,
    },

    /// Anything else is exec'd from the production `telescope` binary.
    ///
    /// `build-anchors` and `build-emission` never read `c`; re-implementing
    /// them here would buy nothing and risk drift.
    #[command(external_subcommand)]
    Passthrough(Vec<OsString>),
}

fn production_binary() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(exe.with_file_name("telescope"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Quant {
            args,
            sigma2_dof_offset,
        } => {
            // Validate before the call, not after: `f64::from_str` accepts
            // "nan", "inf" and "1e999", and a sweep is exactly where someone
            // tries a boundary value. The library asserts too, but from here
            // the failure names the flag instead of surfacing as a backtrace.
            if let Some(c) = sigma2_dof_offset {
                if !(c.is_finite() && c > 0.0) {
                    return Err(format!(
                        "--sigma2-dof-offset must be finite and positive, got {c}"
                    )
                    .into());
                }
            }
            // Self-identify. A figure built from this binary must be able to
            // say which artifact produced it, and the run directory is not a
            // reliable record of what was on PATH.
            let exe = std::env::current_exe()?;
            eprintln!("[telescope-research] binary: {}", exe.display());
            eprintln!(
                "[telescope-research] sigma2_dof_offset: {}",
                sigma2_dof_offset
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "<unset: production Model A>".to_string())
            );
            telescope::quant::run_quant(args, sigma2_dof_offset)?;
            Ok(())
        }

        Cmd::Passthrough(rest) => {
            let prod = production_binary()
                .ok_or("cannot locate the production `telescope` binary next to this one")?;
            let status = Command::new(&prod).args(&rest).status()?;
            std::process::exit(status.code().unwrap_or(1));
        }
    }
}
