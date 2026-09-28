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
use telescope_core::{
    anchor_builder, em, emission_model, evidence, resolver, summary,
};
use telescope_core::emission_model::FallbackConfig;

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
    /// Module 1: build `telescope.emission.parquet` from AIDRS
    /// outputs (AIDRS mode) or anchor pool (Universal mode).
    ///
    /// The two input modes are MUTUALLY EXCLUSIVE:
    /// - AIDRS mode: `--polya-parquet X --assessment-tsv Y`
    /// - Universal mode: `--anchor-pool X [--transcript-evidence Z]`
    ///
    /// `--output` is always required. The `--tx-min-reads`,
    /// `--gene-min-reads`, `--cohort-min-reads` flags configure
    /// the Universal-Mode 3-level fallback (Tx → Gene → Cohort)
    /// and only take effect when `--anchor-pool` is supplied.
    BuildEmission {
        /// AIDRS mode: path to `aidrs.transcript_polyA_len.parquet`.
        #[arg(long, group = "emission_input")]
        polya_parquet: Option<PathBuf>,

        /// AIDRS mode: path to `aidrs.transcript.assessment.tsv`.
        /// Required iff `--polya-parquet` is set.
        #[arg(long, requires = "polya_parquet")]
        assessment_tsv: Option<PathBuf>,

        /// Universal mode: path to `telescope.anchor_pool.parquet`
        /// (produced by `telescope build-anchors`).
        #[arg(long, group = "emission_input")]
        anchor_pool: Option<PathBuf>,

        /// Universal mode (optional): path to
        /// `telescope.transcript_evidence.parquet` (Module 0
        /// output). When provided, reads whose transcript_id has
        /// `em_eligible == false` are dropped before emission
        /// fitting.
        #[arg(long, requires = "anchor_pool")]
        transcript_evidence: Option<PathBuf>,

        /// Output path for `telescope.emission.parquet`.
        #[arg(long)]
        output: PathBuf,

        /// Universal mode: L1 Tx-level minimum reads (default 10).
        #[arg(long, default_value_t = 10, requires = "anchor_pool")]
        tx_min_reads: usize,

        /// Universal mode: L2 Gene-level minimum reads (default 30).
        #[arg(long, default_value_t = 30, requires = "anchor_pool")]
        gene_min_reads: usize,

        /// Universal mode: L3 Cohort-level minimum reads
        /// (informational; L3 is always applied when L1/L2 fail).
        #[arg(long, default_value_t = 50, requires = "anchor_pool")]
        cohort_min_reads: usize,
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

        /// Optional TSV with header `bam_id\toriginal_id` for ID translation
        /// (replaces offline Python workaround for AIDRS re-keying).
        #[arg(long)]
        tx_id_mapping: Option<PathBuf>,

        /// Print EM per-iter trace (log-L + mean entropy) to stderr.
        #[arg(long)]
        debug: bool,
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::BuildEmission {
            polya_parquet,
            assessment_tsv,
            anchor_pool,
            transcript_evidence,
            output,
            tx_min_reads,
            gene_min_reads,
            cohort_min_reads,
        } => {
            eprintln!("[telescope] Module 1: building emission model");
            match (polya_parquet, anchor_pool) {
                (Some(p), None) => {
                    // AIDRS mode (legacy).
                    let assess = assessment_tsv.context(
                        "--assessment-tsv is required when --polya-parquet is set",
                    )?;
                    eprintln!("  mode:           AIDRS");
                    eprintln!("  polya_parquet:  {}", p.display());
                    eprintln!("  assessment_tsv: {}", assess.display());
                    eprintln!("  output:         {}", output.display());
                    emission_model::build_emission_parquet(p, assess, output)
                        .context("Module 1 build_emission_parquet (AIDRS) failed")?;
                }
                (None, Some(a)) => {
                    // Universal mode (Mode B).
                    let fb = FallbackConfig {
                        tx_min_reads,
                        gene_min_reads,
                        cohort_min_reads,
                    };
                    eprintln!("  mode:               Universal (Mode B)");
                    eprintln!("  anchor_pool:        {}", a.display());
                    if let Some(ref te) = transcript_evidence {
                        eprintln!("  transcript_evidence: {}", te.display());
                    } else {
                        eprintln!("  transcript_evidence: (none; no em_eligible filter)");
                    }
                    eprintln!("  tx_min_reads:       {}", fb.tx_min_reads);
                    eprintln!("  gene_min_reads:     {}", fb.gene_min_reads);
                    eprintln!("  cohort_min_reads:   {}", fb.cohort_min_reads);
                    eprintln!("  output:             {}", output.display());
                    emission_model::build_emission_from_anchors(
                        a,
                        transcript_evidence,
                        output,
                        &fb,
                    )
                    .context("Module 1 build_emission_from_anchors (Universal) failed")?;
                }
                (Some(_), Some(_)) => {
                    anyhow::bail!(
                        "--polya-parquet (AIDRS mode) and --anchor-pool (Universal mode) \
                         are mutually exclusive"
                    );
                }
                (None, None) => {
                    anyhow::bail!(
                        "must specify either --polya-parquet (AIDRS mode) or \
                         --anchor-pool (Universal mode)"
                    );
                }
            }
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
            tx_id_mapping,
            debug,
        } => {
            eprintln!("[telescope] Quant (M2 → M3)");
            eprintln!("  bam:        {}", bam.display());
            eprintln!("  emission:   {}", emission.display());
            eprintln!("  sample_id:  {}", sample_id);
            eprintln!("  output:     {}", output.display());
            eprintln!("  tes_slack:  {} bp", tes_slack);
            eprintln!("  tol:        {:.1e}", tol);
            eprintln!("  max_iter:   {}", max_iter);
            if let Some(ref p) = tx_id_mapping {
                eprintln!("  tx_id_mapping: {}", p.display());
            }
            eprintln!("  debug:      {}", debug);

            // Parse optional `bam_id\toriginal_id` TSV (header row skipped).
            // This replaces the previous offline Python re-keying workaround.
            let id_mapping: Option<std::collections::HashMap<String, String>> =
                if let Some(ref p) = tx_id_mapping {
                    let f = std::fs::File::open(p)
                        .with_context(|| format!("Failed to open --tx-id-mapping {:?}", p))?;
                    let reader = std::io::BufReader::new(f);
                    use std::io::BufRead;
                    let mut map = std::collections::HashMap::new();
                    for (i, line) in reader.lines().enumerate() {
                        let l = line.with_context(|| {
                            format!("Failed to read line {} from {:?}", i + 1, p)
                        })?;
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
                    eprintln!(
                        "[telescope]    tx_id_mapping entries: {}",
                        map.len()
                    );
                    Some(map)
                } else {
                    None
                };

            eprintln!("[telescope]  M2: extract evidence…");
            let mut graph = evidence::extract_evidence(&bam, tes_slack)
                .context("Module 2 extract_evidence failed")?;
            // Apply optional id mapping to transcript names (BAM @SQ → emission parquet keys).
            if let Some(ref map) = id_mapping {
                for name in &mut graph.transcript_names {
                    if let Some(mapped) = map.get(name) {
                        *name = mapped.clone();
                    }
                }
            }
            eprintln!(
                "[telescope]    molecules:      {}",
                graph.molecules.len()
            );
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
            let mut em_trace: Vec<(usize, f64, f64)> = Vec::new();
            let trace_sink: Option<&mut Vec<(usize, f64, f64)>> = if debug {
                Some(&mut em_trace)
            } else {
                None
            };
            let stats = em::solve_and_write_profile(
                graph,
                &emission,
                &sample_id,
                &output,
                tol,
                max_iter,
                trace_sink,
            )
            .context("Module 3 solve_and_write_profile failed")?;
            eprintln!("[telescope]    n_unambig / sample:      {}", stats.n_unambiguous);
            eprintln!("[telescope]    n_ambig / EM:           {}", stats.n_ambiguous);
            eprintln!("[telescope]    EM iterations:          {}", stats.n_em_iter);
            eprintln!(
                "[telescope]    dropped (incompetent):  {}",
                stats.n_dropped_incompetent
            );
            eprintln!(
                "[telescope]    dropped (no_competent): {}",
                stats.n_dropped_no_competent_candidates
            );
            let n_ambig = stats.n_ambiguous.max(1) as f64;
            eprintln!("[telescope]    posterior entropy histogram:");
            eprintln!(
                "[telescope]      < 0.20 nats (cleanly resolved): {} ({:.1}%)",
                stats.n_entropy_clean,
                stats.n_entropy_clean as f64 / n_ambig * 100.0
            );
            eprintln!(
                "[telescope]      0.20–0.60 nats (partial):     {} ({:.1}%)",
                stats.n_entropy_partial,
                stats.n_entropy_partial as f64 / n_ambig * 100.0
            );
            eprintln!(
                "[telescope]      > 0.60 nats (degenerate tie):  {} ({:.1}%)",
                stats.n_entropy_degenerate,
                stats.n_entropy_degenerate as f64 / n_ambig * 100.0
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
        Cmd::ResolveEvidence {
            bams, sample_ids, tx_to_gene, cohort_size,
            tes_cluster_radius, tes_slack, output,
        } => {
            if bams.len() != sample_ids.len() {
                anyhow::bail!(
                    "--bam ({} files) and --sample-id ({} ids) must have equal length",
                    bams.len(), sample_ids.len()
                );
            }
            if cohort_size != bams.len() {
                anyhow::bail!(
                    "--cohort-size ({}) must equal the number of --bam arguments ({})",
                    cohort_size, bams.len()
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

            // Load transcript→gene mapping (TSV) → HashMap<String, String>.
            let tx_to_gene_map: std::collections::HashMap<String, String> = match tx_to_gene {
                Some(p) => {
                    let content = std::fs::read_to_string(&p).with_context(|| {
                        format!("Failed to read tx-to-gene TSV: {:?}", p)
                    })?;
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