//! Telescope CLI entry point (Phase C scaffold).
//!
//! Currently exposes three subcommands:
//! - `build-emission`: drives Module 1 (Emission Model Builder).
//! - `extract-evidence`: drives Module 2 (Transcriptome Evidence Extractor).
//! - `quant`: drives Modules 2 → 3 (BAM → evidence → TEL-EM profile).
//!
//! Module 4 R to follow per `specs/telescope-v0.1.7.md`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use telescope_core::{em, emission_model, evidence};

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
    /// Module 1: build `telescope.emission.parquet` from AIDRS outputs.
    BuildEmission {
        /// Path to `aidrs.transcript_polyA_len.parquet`.
        #[arg(long)]
        polya_parquet: PathBuf,

        /// Path to `aidrs.transcript.assessment.tsv`.
        #[arg(long)]
        assessment_tsv: PathBuf,

        /// Output path for `telescope.emission.parquet`.
        #[arg(long)]
        output: PathBuf,
    },

    /// Module 2: extract single-molecule evidence from AIDRS-aligned BAM.
    ExtractEvidence {
        /// Path to AIDRS-aligned transcriptome BAM (carries Dorado `pt:i`).
        #[arg(long)]
        bam: PathBuf,

        /// 3' TES boundary slack in bp (default 35).
        #[arg(long, default_value_t = 35)]
        tes_slack: usize,
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

        /// EM log-likelihood convergence tolerance.
        #[arg(long, default_value_t = 1e-7)]
        tol: f64,

        /// Maximum EM iterations (safety cap).
        #[arg(long, default_value_t = 1000)]
        max_iter: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::BuildEmission {
            polya_parquet,
            assessment_tsv,
            output,
        } => {
            eprintln!("[telescope] Module 1: building emission model");
            eprintln!("  polya_parquet:  {}", polya_parquet.display());
            eprintln!("  assessment_tsv: {}", assessment_tsv.display());
            eprintln!("  output:         {}", output.display());
            emission_model::build_emission_parquet(
                polya_parquet,
                assessment_tsv,
                output,
            )
            .context("Module 1 build_emission_parquet failed")?;
            eprintln!("[telescope] done.");
            Ok(())
        }
        Cmd::ExtractEvidence { bam, tes_slack } => {
            eprintln!("[telescope] Module 2: extracting single-molecule evidence");
            eprintln!("  bam:        {}", bam.display());
            eprintln!("  tes_slack:  {} bp", tes_slack);
            let graph = evidence::extract_evidence(&bam, tes_slack)
                .context("Module 2 extract_evidence failed")?;
            eprintln!(
                "[telescope]  molecules:        {}",
                graph.molecules.len()
            );
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
        } => {
            eprintln!("[telescope] Quant (M2 → M3)");
            eprintln!("  bam:        {}", bam.display());
            eprintln!("  emission:   {}", emission.display());
            eprintln!("  sample_id:  {}", sample_id);
            eprintln!("  output:     {}", output.display());
            eprintln!("  tes_slack:  {} bp", tes_slack);
            eprintln!("  tol:        {:.1e}", tol);
            eprintln!("  max_iter:   {}", max_iter);

            eprintln!("[telescope]  M2: extract evidence…");
            let graph = evidence::extract_evidence(&bam, tes_slack)
                .context("Module 2 extract_evidence failed")?;
            eprintln!(
                "[telescope]    molecules:      {}",
                graph.molecules.len()
            );
            eprintln!(
                "[telescope]    transcripts:    {}",
                graph.transcript_names.len()
            );

            eprintln!("[telescope]  M3: TEL-EM…");
            let stats = em::solve_and_write_profile(
                graph,
                &emission,
                &sample_id,
                &output,
                tol,
                max_iter,
            )
            .context("Module 3 solve_and_write_profile failed")?;
            eprintln!("[telescope]    n_unambig / ample:      {}", stats.n_unambiguous);
            eprintln!("[telescope]    n_ambig / EM:          {}", stats.n_ambiguous);
            eprintln!("[telescope]    EM iterations:        {}", stats.n_em_iter);
            eprintln!(
                "[telescope]    dropped (incompetent): {}",
                stats.n_dropped_incompetent
            );
            eprintln!("[telescope] done.");
            Ok(())
        }
    }
}