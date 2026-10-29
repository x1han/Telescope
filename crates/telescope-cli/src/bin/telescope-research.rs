//! Research-only entry point. **Not shipped, not part of the release gate.**
//!
//! `SIGMA2_DOF_OFFSET` is a `pub const` so the shipped path cannot be moved at
//! runtime, and the production `telescope` exposes no flag that reaches it.
//! The σ² denominator offset `c` is exactly what a sensitivity analysis has to
//! vary, so this binary exposes `--sigma2-dof-offset` and calls the same
//! `telescope::quant::run_quant` the production binary calls.
//!
//! `quant` is defined once, in the library, and both binaries parse the same
//! `QuantArgs`, so a default cannot drift between them.
//!
//! Every other subcommand is exec'd from the production binary rather than
//! duplicated: same program, same bytes, same behaviour by construction.

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

// clippy's `large_enum_variant` fires here: boxing `QuantArgs` would trade
// stack for a heap allocation on every invocation. Do not "fix" this by boxing.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand, Debug)]
enum Cmd {
    /// Modules 2 + 3: BAM -> evidence -> TEL-EM profile.
    ///
    /// Identical to `telescope quant` except that the trailing constant `c` of
    /// the sigma^2 denominator (`W + 2*nu_0 + c`) is settable; omitting the
    /// flag is exactly `telescope quant`.
    ///
    /// Trap: `c = 2` IS Model A and `c = 3` is NOT Model B — Model B also needs
    /// the `tau_0*(mu-mu_cohort)^2` term back in the numerator.
    Quant {
        #[command(flatten)]
        args: QuantArgs,

        /// Trailing constant `c` of the sigma^2 denominator. Defaults to the
        /// production Model A value; it has no counterpart on `telescope`.
        #[arg(long)]
        sigma2_dof_offset: Option<f64>,
    },

    /// Anything else is exec'd from the production `telescope` binary.
    ///
    /// Trap: `build-anchors` and `build-emission` never read `c`, so they are
    /// exec'd rather than duplicated — a reimplementation would risk drift.
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
            // Validate before the call: `f64::from_str` accepts "nan", "inf"
            // and "1e999", and the library's own assert would surface as a
            // backtrace instead of naming the flag.
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
