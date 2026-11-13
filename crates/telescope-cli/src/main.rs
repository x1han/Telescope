use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use telescope::diff::DiffArgs;
use telescope::quant::QuantArgs;
use telescope::POLY_A_SOURCE_HELP;
use telescope::{maybe_write_filter_ledger, print_filter_ledger};
use telescope::{
    run_build_anchors, run_build_emission, AnchorRequest, EmissionRequest, ANCHOR_GAP_ABS_DEFAULT,
    ANCHOR_GAP_REL_DEFAULT, ANCHOR_MAX_READS_PER_SAMPLE_DEFAULT, ANCHOR_MIN_COVERAGE_DEFAULT,
    ANCHOR_TES_SLACK_DEFAULT, EMISSION_COHORT_MIN_READS_DEFAULT, EMISSION_GENE_MIN_READS_DEFAULT,
    EMISSION_TX_MIN_READS_DEFAULT,
};
use telescope_core::gtf::GtfGenes;
use telescope_core::{evidence, resolver, summary};

/// The production `telescope` CLI.
#[derive(Parser, Debug)]
#[command(
    name = "telescope",
    version,
    about = "Telescope: Transcript-specific poly(A) tail length estimation and isoform quantification in direct RNA sequencing"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

/// The subcommands of the production `telescope` binary.
#[derive(Subcommand, Debug)]
enum Cmd {
    /// Fit isoform-specific poly(A) tail-length priors.
    #[command(display_order = 2, hide = true)]
    BuildEmission {
        /// Anchor pool parquet (`telescope.anchor_pool.parquet`), written by `build-anchors`.
        #[arg(long, help_heading = "INPUTS")]
        anchor_pool: PathBuf,

        /// Optional transcript-evidence parquet.
        #[arg(long, help_heading = "INPUTS")]
        transcript_evidence: Option<PathBuf>,

        /// Emission model parquet (`telescope.emission.parquet`).
        #[arg(long, help_heading = "OUTPUTS")]
        output: PathBuf,

        /// Reads a transcript needs before its own fit is used.
        #[arg(long, default_value_t = EMISSION_TX_MIN_READS_DEFAULT, help_heading = "GATES")]
        tx_min_reads: usize,

        /// Reads a gene needs before its transcripts fall back to the gene fit.
        #[arg(long, default_value_t = EMISSION_GENE_MIN_READS_DEFAULT, help_heading = "GATES")]
        gene_min_reads: usize,

        /// Reads a cohort needs before a transcript falls back to the cohort fit.
        #[arg(long, default_value_t = EMISSION_COHORT_MIN_READS_DEFAULT, help_heading = "GATES")]
        cohort_min_reads: usize,

        /// Alignment BAM whose `@SQ` header names the reference universe.
        #[arg(long, help_heading = "REFERENCE")]
        reference_bam: Option<PathBuf>,

        /// Annotation GTF.
        #[arg(short = 'g', long = "gtf", help_heading = "REFERENCE")]
        gtf: Option<PathBuf>,

        /// Provenance sidecar JSON.
        #[arg(long, help_heading = "OUTPUTS")]
        provenance_out: Option<PathBuf>,
    },

    /// Extract single-molecule evidence from a transcriptome-aligned BAM.
    #[command(display_order = 6, hide = true)]
    ExtractEvidence {
        /// Transcriptome-aligned BAM (carries the Dorado `pt:i` tag).
        #[arg(long, help_heading = "INPUTS")]
        bam: PathBuf,

        /// Slack in bp allowed between a read's 3' end and the transcript's TES boundary.
        #[arg(long, default_value_t = 35, help_heading = "FILTERS")]
        tes_slack: usize,

        #[arg(
            long = "poly-a-source",
            alias = "polya-source",
            alias = "polyA-source",
            default_value = "bam_tag:pt",
            help = POLY_A_SOURCE_HELP
        )]
        poly_a_source: String,

        /// Record-level filter ledger, written as JSON.
        #[arg(long, help_heading = "OUTPUTS")]
        ledger: Option<PathBuf>,
    },

    /// Quantify isoform abundance and per-molecule poly(A) tail length.
    #[command(display_order = 3)]
    Quant(QuantArgs),

    /// Run differential analysis across two conditions.
    #[command(display_order = 5)]
    Diff(DiffArgs),

    /// Aggregate a profile into cohort-level abundance and tail-length statistics.
    #[command(display_order = 4, hide = true)]
    Summarize {
        /// Profile parquet (`telescope.profile.parquet`).
        #[arg(long, help_heading = "INPUTS")]
        profile: PathBuf,

        /// Sample summary parquet (`telescope.sample_summary.parquet`).
        #[arg(long, help_heading = "OUTPUTS")]
        output: PathBuf,

        /// Per-transcript provenance sidecar from `build-emission`.
        #[arg(long, help_heading = "INPUTS")]
        provenance: Option<PathBuf>,
    },

    /// Select full-length anchor reads for the poly(A) prior.
    #[command(display_order = 1, hide = true)]
    BuildAnchors {
        /// Transcriptome-aligned BAM files, one per sample.
        #[arg(long = "bam", required = true, help_heading = "INPUTS")]
        bams: Vec<PathBuf>,

        /// Sample ids, one per `--bam`.
        #[arg(long = "sample-id", required = true, help_heading = "INPUTS")]
        sample_ids: Vec<String>,

        /// Anchor pool parquet (`telescope.anchor_pool.parquet`).
        #[arg(long, help_heading = "OUTPUTS")]
        output: PathBuf,

        /// Annotation GTF.
        #[arg(short = 'g', long = "gtf", help_heading = "REFERENCE")]
        gtf: Option<PathBuf>,

        /// Slack in bp allowed between a read's 3' end and the transcript's TES boundary.
        #[arg(long, default_value_t = ANCHOR_TES_SLACK_DEFAULT, help_heading = "FILTERS")]
        tes_slack: usize,

        /// Minimum ratio of alignment length to reference length.
        #[arg(long, default_value_t = ANCHOR_MIN_COVERAGE_DEFAULT, help_heading = "FILTERS")]
        min_coverage: f64,

        /// Maximum number of anchor reads kept per transcript per sample.
        #[arg(long, default_value_t = ANCHOR_MAX_READS_PER_SAMPLE_DEFAULT, help_heading = "GATES")]
        max_reads_per_sample: usize,

        /// Absolute floor on the per-read `delta_AS_raw` gap, where `delta_AS_raw = AS_1 - (AS_2 or 0)` over the read's full candidate set.
        #[arg(long, default_value_t = ANCHOR_GAP_ABS_DEFAULT, help_heading = "GATES")]
        gap_abs: f64,

        /// Floor on the per-read `delta_AS_raw` gap as a fraction of `AS_1`.
        #[arg(long, default_value_t = ANCHOR_GAP_REL_DEFAULT, help_heading = "GATES")]
        gap_rel: f64,

        /// Anchor-level filter ledger, written as JSON.
        #[arg(long, help_heading = "OUTPUTS")]
        ledger: Option<PathBuf>,
    },

    /// Resolve cohort-level transcript evidence into `telescope.transcript_evidence.parquet`.
    #[command(display_order = 7, hide = true)]
    ResolveEvidence {
        /// Transcriptome-aligned BAM files of the cohort.
        #[arg(long = "bam", required = true, help_heading = "INPUTS")]
        bams: Vec<PathBuf>,

        /// Sample ids, one per `--bam`, used to namespace read ids when the cohort is pooled.
        #[arg(long = "sample-id", required = true, help_heading = "INPUTS")]
        sample_ids: Vec<String>,

        /// Annotation GTF.
        #[arg(short = 'g', long = "gtf", help_heading = "REFERENCE")]
        gtf: Option<PathBuf>,

        /// Cohort size; must equal the number of `--bam` arguments.
        #[arg(long, help_heading = "INPUTS")]
        cohort_size: usize,

        /// Radius in bp of the TES clusters compared during the subsumption scan.
        #[arg(long, default_value_t = 50, help_heading = "MODEL")]
        tes_cluster_radius: usize,

        /// Slack in bp allowed between a read's 3' end and the transcript's TES boundary.
        #[arg(long, default_value_t = 35, help_heading = "FILTERS")]
        tes_slack: usize,

        /// Transcript-evidence parquet (`telescope.transcript_evidence.parquet`).
        #[arg(long, help_heading = "OUTPUTS")]
        output: PathBuf,

        /// Record-level filter ledger, written as JSON.
        #[arg(long, help_heading = "OUTPUTS")]
        ledger: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::BuildEmission {
            anchor_pool,
            transcript_evidence,
            reference_bam,
            gtf,
            provenance_out,
            output,
            tx_min_reads,
            gene_min_reads,
            cohort_min_reads,
        } => {
            run_build_emission(EmissionRequest {
                anchor_pool,
                transcript_evidence,
                reference_bam,

                reference_universe: None,
                gtf,
                provenance_out,
                output,
                tx_min_reads,
                gene_min_reads,
                cohort_min_reads,

                prior_scope: None,
            })?;
            Ok(())
        }
        Cmd::ExtractEvidence {
            bam,
            tes_slack,
            poly_a_source,
            ledger,
        } => {
            let poly_a = evidence::parse_poly_a_source(&poly_a_source)?;
            eprintln!("[telescope] extracting single-molecule evidence");
            eprintln!("  bam:        {}", bam.display());
            eprintln!("  tes_slack:  {} bp", tes_slack);
            eprintln!("  polyA-src:  {}", poly_a_source);
            let graph = evidence::extract_evidence_with(&bam, tes_slack, &poly_a)
                .context("extract_evidence failed")?;
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
            maybe_write_filter_ledger(ledger.as_ref(), &graph.stats)?;
            let avg = if graph.molecules.is_empty() {
                0.0
            } else {
                total_cands as f64 / graph.molecules.len() as f64
            };
            eprintln!("[telescope]  avg candidates:   {:.2}", avg);
            eprintln!("[telescope] done.");
            Ok(())
        }
        Cmd::Quant(quant_args) => {
            telescope::quant::run_quant(quant_args, None)?;
            Ok(())
        }
        Cmd::Diff(diff_args) => {
            eprintln!("[telescope] differential statistics");
            telescope::diff::run(&diff_args)?;
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
            gtf,
            tes_slack,
            min_coverage,
            max_reads_per_sample,
            gap_abs,
            gap_rel,
            ledger,
        } => {
            run_build_anchors(AnchorRequest {
                bams,
                sample_ids,
                output,
                gtf,
                tes_slack,
                min_coverage,
                max_reads_per_sample,
                gap_abs,
                gap_rel,
                ledger,
            })?;
            Ok(())
        }
        Cmd::ResolveEvidence {
            bams,
            sample_ids,
            gtf,
            cohort_size,
            tes_cluster_radius,
            tes_slack,
            output,
            ledger,
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
            eprintln!("[telescope] transcript evidence resolver");
            eprintln!("  cohort_size:         {}", cohort_size);
            eprintln!("  tes_cluster_radius:  {} bp", tes_cluster_radius);
            eprintln!("  tes_slack:           {} bp", tes_slack);
            eprintln!("  output:              {}", output.display());

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
            maybe_write_filter_ledger(ledger.as_ref(), &graph.stats)?;

            let tx_to_gene_map: std::collections::HashMap<String, String> = match gtf {
                Some(ref p) => {
                    let genes = GtfGenes::from_gtf(p)
                        .with_context(|| format!("Failed to read the annotation GTF: {:?}", p))?;
                    let m = genes.gene_map();
                    eprintln!("  gtf transcripts:     {}", m.len());
                    m
                }
                None => {
                    eprintln!("  gtf transcripts:     0 (no annotation given; all transcripts will emit missing_gene_id)");
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

    use telescope::{load_gtf, refuse_a_provenance_path_that_clobbers_a_table};
    use telescope::{
        pct, ALN_WEIGHT_LAMBDA_DEFAULT, FIVE_PRIME_WEIGHT_DEFAULT, FIVE_PRIME_WINDOW_DEFAULT,
        PRUNE_MAX_DELTA_AS_DEFAULT,
    };

    fn write_tmp(name: &str, body: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("tel_test_{}_{}", std::process::id(), name));
        std::fs::write(&p, body).unwrap();
        p
    }

    /// A SUPPLIED `--gtf` that yields no transcript must fail, not silently become Universal Mode.
    #[test]
    fn a_supplied_but_empty_gtf_is_an_error_not_universal_mode() {
        let none = load_gtf(&None).expect("no flag must stay legal");
        assert_eq!(none.len(), 0, "no flag means an empty map, on purpose");

        let hdr = write_tmp("hdr", "#a comment\n");
        let e = load_gtf(&Some(hdr.clone()))
            .expect_err("an annotation with no transcript must NOT pass as an empty-but-valid map");
        assert!(
            format!("{:#}", e).contains("no transcript"),
            "the error says what is missing: {:#}",
            e
        );

        let geneonly = write_tmp(
            "geneonly",
            "chr1\tX\tgene\t1\t10\t.\t+\t.\tgene_id \"G1\"; gene_name \"A\";\n",
        );
        let e = load_gtf(&Some(geneonly.clone()))
            .expect_err("an annotation with no transcript must not pass");
        assert!(
            format!("{:#}", e).contains("no transcript"),
            "the error says what is missing: {:#}",
            e
        );

        let ok = write_tmp(
            "ok",
            "chr1\tX\ttranscript\t1\t10\t.\t+\t.\tgene_id \"G1\"; transcript_id \"tx0\"; gene_name \"GA\";\n\
             chr1\tX\ttranscript\t1\t10\t.\t+\t.\tgene_id \"G2\"; transcript_id \"tx1\"; gene_name \"GB\";\n",
        );
        let m = load_gtf(&Some(ok.clone())).expect("a valid GTF must still load");
        assert_eq!(m.len(), 2, "the guard must not reject a genuine annotation");
        assert_eq!(m.gene_id("tx0").map(String::as_str), Some("G1"));
        assert_eq!(m.gene_name("tx1").map(String::as_str), Some("GB"));

        for f in [hdr, geneonly, ok] {
            let _ = std::fs::remove_file(f);
        }
    }

    /// A `--provenance-out` that would destroy a parquet table is refused — both halves of the guard, called directly.
    #[test]
    fn a_collision_with_a_parquet_table_is_refused_whatever_its_name() {
        let dir = std::env::temp_dir();
        let table = dir.join("tel_prov_guard_table.tbl");
        let legal = dir.join("tel_prov_guard_header.json");

        std::fs::write(&table, b"not really parquet, but a file that exists")
            .expect("seed the table file");

        let e = refuse_a_provenance_path_that_clobbers_a_table(
            &dir.join("anything.parquet"),
            &dir.join("some_other_file.json"),
        )
        .expect_err(".parquet must be refused");
        assert!(e.to_string().contains("ends in .parquet"), "got: {e}");

        let e = refuse_a_provenance_path_that_clobbers_a_table(&table, &table)
            .expect_err("the same path must be refused");
        assert!(e.to_string().contains("same file as"), "got: {e}");

        let sub = dir.join("tel_prov_guard_sub");
        std::fs::create_dir_all(&sub).expect("mkdir");
        let indirect = sub.join("..").join("tel_prov_guard_table.tbl");
        assert_ne!(
            indirect.as_os_str(),
            table.as_os_str(),
            "the fixture must spell the two paths differently as strings, or \
             case 3 is just case 2 again"
        );
        let e = refuse_a_provenance_path_that_clobbers_a_table(&indirect, &table)
            .expect_err("a path that canonicalises onto the table must be refused");
        assert!(e.to_string().contains("same file as"), "got: {e}");
        let _ = std::fs::remove_dir(&sub);

        refuse_a_provenance_path_that_clobbers_a_table(&legal, &table)
            .expect("a legal JSON run-header path must be accepted");

        let _ = std::fs::remove_file(&table);
    }

    /// A zero denominator must render as `n/a`, never as a percentage.
    #[test]
    fn pct_never_renders_a_zero_denominator_as_a_percentage() {
        assert_eq!(
            pct(0, 0),
            "n/a",
            "0 of 0 is UNDEFINED, not 0.0% -- this is the metric formatting \
             reappearing in the print"
        );

        assert_eq!(pct(0, 715_136), "0.0%");
        assert_eq!(pct(1, 4), "25.0%");
        assert_eq!(pct(1, 3), "33.3%");
    }

    /// Pins the shipped defaults.
    #[test]
    fn test_the_default_parameters_are_the_ones_in_the_specification() {
        assert_eq!(ALN_WEIGHT_LAMBDA_DEFAULT, 20.0, "alignment weight default");
        assert_eq!(FIVE_PRIME_WEIGHT_DEFAULT, 1.8, "frozen 5' weight");
        assert_eq!(FIVE_PRIME_WINDOW_DEFAULT, 20, "frozen 5' window, inclusive");
        assert_eq!(PRUNE_MAX_DELTA_AS_DEFAULT, 50, "frozen ΔAS prune threshold");
    }

    /// The `default_value_t` attributes must resolve to the SAME constants.
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

    /// `--five-prime-weight 0` is the documented off switch and must be ACCEPTED, not rejected by the `>= 0` validation.
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
