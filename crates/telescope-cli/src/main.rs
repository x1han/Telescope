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
use telescope_core::{anchor_builder, em, emission_model, evidence, summary};

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

        /// EM log-likelihood convergence tolerance (per-molecule average LL).
        #[arg(long, default_value_t = 1e-6)]
        tol: f64,

        /// Maximum EM iterations (safety cap).
        #[arg(long, default_value_t = 1000)]
        max_iter: usize,
    },

    /// Profile Summary Engine: aggregate profile.parquet → sample_summary.parquet.
    Summarize {
        /// Path to `telescope.profile.parquet`.
        #[arg(long)]
        profile: PathBuf,

        /// Output `telescope.sample_summary.parquet`.
        #[arg(long)]
        output: PathBuf,
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
        Cmd::Summarize { profile, output } => {
            eprintln!("[telescope] Profile Summary Engine");
            eprintln!("  profile: {}", profile.display());
            eprintln!("  output:  {}", output.display());
            summary::summarize_profile_parquet(&profile, &output)
                .context("summary::summarize_profile_parquet failed")?;
            eprintln!("[telescope] done.");
            Ok(())
        }
        Cmd::BuildAnchors {
            bams, sample_ids, output, tx_to_gene,
            tes_slack, min_coverage, max_reads_per_sample,
        } => {
            if bams.len() != sample_ids.len() {
                anyhow::bail!(
                    "--bam ({} files) and --sample-id ({} ids) must have equal length",
                    bams.len(), sample_ids.len()
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

            let pairs: Vec<(PathBuf, &str)> = bams.iter().cloned().zip(sample_ids.iter().map(|s| s.as_str())).collect();
            let stats = anchor_builder::build_anchor_pool(
                &pairs,
                &tx_to_g,
                &output,
                tes_slack,
                min_coverage,
                max_reads_per_sample,
            ).context("anchor_builder::build_anchor_pool failed")?;
            eprintln!("[telescope]  {}", stats.report());
            eprintln!("[telescope] done.");
            Ok(())
        }
    }
}