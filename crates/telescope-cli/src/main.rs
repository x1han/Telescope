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
use telescope::quant::QuantArgs;
use telescope::{ledger_json, POLY_A_SOURCE_HELP};
use telescope::{maybe_write_filter_ledger, print_filter_ledger};
use telescope_core::emission_model::FallbackConfig;
use telescope_core::{anchor_builder, emission_model, evidence, resolver, summary};

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

        /// Write the record-level filter ledger here as JSON
        /// (`telescope.filter_ledger/v1`).
        ///
        /// OPT-IN, not a default sidecar: `--provenance-out` on
        /// `build-emission` writes by default, and a second flag with the
        /// opposite default invites a reader to assume both run. The path is
        /// explicit so the record-level and anchor-level ledgers cannot land in
        /// one file — they are different stages with different denominators,
        /// and a single merged document would invite exactly the "summed the
        /// two totals" mistake the schema's two top-level keys exist to
        /// prevent.
        #[arg(long)]
        ledger: Option<PathBuf>,
    },

    /// Modules 2 + 3: BAM -> evidence -> TEL-EM profile.
    Quant(QuantArgs),

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

        /// Absolute floor on the per-read `AS_best - AS_second` gap.
        ///
        /// ⚠ THE DEFAULT 0.0 IS NOT A CALIBRATED PRODUCTION VALUE. It means
        /// "no gap requirement" — this criterion is switched off unless you set
        /// it. Nothing in this repository measures what it should be; the owner
        /// declined to ship an uncalibrated default. Read 0.0 as "off", never as
        /// "zero gap is correct".
        #[arg(long, default_value_t = 0.0)]
        gap_abs: f64,

        /// Floor on the per-read `AS_best - AS_second` gap as a fraction of
        /// `AS_best`. Same caveat as `--gap-abs`: the 0.0 default means "off".
        #[arg(long, default_value_t = 0.0)]
        gap_rel: f64,

        /// Write the anchor-level filter ledger here as JSON
        /// (`telescope.filter_ledger/v1`).
        ///
        /// This is the ONLY subcommand with an anchor-level ledger, because it
        /// is the only one that produces `AnchorStats`. The record-level ledger
        /// belongs to `extract-evidence` / `quant` / `resolve-evidence`; the
        /// two are different populations (records entering Module 2 vs records
        /// entering the anchor scan) and must go to different files.
        #[arg(long)]
        ledger: Option<PathBuf>,
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

        /// Write the record-level filter ledger here as JSON
        /// (`telescope.filter_ledger/v1`). See `extract-evidence --ledger`.
        ///
        /// For a cohort this is the field-wise SUM over the input BAMs, so
        /// `records_entered` is the union of the inputs and not any one
        /// sample's count.
        #[arg(long)]
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
            // The total stays first and stays backwards-compatible; the two
            // detail counts follow in parentheses so the line still parses for
            // anything reading `n_imputed_rows: <int>` and still ends on an
            // integer. They are printed because the two populations are
            // OPPOSITE facts -- `padded` was never observed, `suppressed` was
            // observed and then removed on purpose -- and a bare total cannot
            // tell a reader which one they are looking at.
            eprintln!(
                "[telescope]    n_imputed_rows:           {} (padded: {}, suppressed: {})",
                tiers.n_imputed, tiers.n_imputed_padded, tiers.n_imputed_suppressed
            );
            // CONSERVATION, asserted rather than warned.
            //
            // The emitted key set is `anchored UNION universe`, so the law that
            // always holds is
            //     n_after == n_reference + n_anchored_not_in_reference
            // NOT `n_after == n_reference`: that one is false whenever an
            // anchored transcript is absent from the universe, which is a
            // configuration the code above deliberately WARNS about rather than
            // rejects. Asserting the strict form would turn a tolerated
            // configuration into a hard failure on runs that work today.
            //
            // WHAT THIS ACTUALLY GUARDS AGAINST, corrected 2026-10-15. The
            // comment here used to say this is "fail-closed against the
            // regression it was added for ... (the `em_eligible`-suppressed
            // resurrection bug was exactly that)". THAT WAS FALSE, and it is
            // worth being precise because the two bugs are easy to conflate:
            //
            //   * P0-2, the bug this assertion was added alongside, was a
            //     COMPETENCE-FLAG error. The suppressed transcript was emitted
            //     either way — universe padding had already re-added it — so
            //     the ROW COUNT NEVER CHANGED and a row-count law could not
            //     have seen it. Its coverage is
            //     `test_suppressed_transcripts_emitted_as_incompetent`, at the
            //     library layer, where the flag is written.
            //   * The defect this law DOES cover is padding STOPPING a key it
            //     should emit (the "drop the row" alternative to the fix). Then
            //     `n_after` falls one short per dropped key.
            //
            // It is fail-closed against the second, and the distinction matters:
            // reading the old comment would credit this assertion with covering
            // P0-2, and dropping it on the strength of that credit would remove
            // a check without anyone knowing P0-2 was ever unprotected.
            //
            // MEASURED LIMITATION, 2026-10-15. This assertion currently CANNOT
            // fire from the CLI. Both sides are the cardinality of
            // `anchored UNION universe`, so the law is an identity on any two
            // SETS; the only input that breaks it is a universe naming a
            // transcript twice, and `read_reference_universe` returns noodles'
            // header map, which collapses duplicate `@SQ` names before this
            // code sees them. Measured: `@SQ SN:tx0, SN:tx1, SN:tx1` yields a
            // two-entry universe and this run exits 0.
            //
            // So it is kept because the expert ruling locks it and because a
            // future `noodles` upgrade could make it reachable -- NOT because it
            // is currently load-bearing. The precondition is pinned by
            // `a_duplicated_universe_entry_is_collapsed_by_the_header_reader`
            // in `crates/telescope-cli/tests/`, and the failure mode itself, at
            // the layer where it IS reachable, by
            // `a_duplicated_universe_key_is_rejected_at_the_entry_point` in
            // `emission_model.rs`.
            //
            // UPDATED 2026-10-15. The line above used to end by filing "moving
            // the law into `build_emission_from_anchors`" as a thawing-zone
            // draft. That draft has since been taken, and it closed this rather
            // than opening it: `build_emission_from_anchors` now (a) `bail!`s
            // at its entry point if `reference_universe` contains a duplicate
            // name, so the malformed input is refused before a row is built
            // rather than after; and (b) asserts
            // `n_imputed == n_imputed_padded + n_imputed_suppressed` on the way
            // out, where the emitted counts are known. So the CLI guard's
            // soundness no longer rests on noodles' map semantics ALONE -- the
            // library entry point refuses the input in its own right. This
            // assertion remains an identity on well-formed input, which is
            // still true and is now a second line of defence rather than the
            // only one.
            //
            // Only meaningful when a universe was supplied; with none,
            // `n_reference == 0` and the sum is not the universe size.
            if tiers.n_reference > 0 {
                anyhow::ensure!(
                    n_after == tiers.n_reference + tiers.n_anchored_not_in_reference,
                    "emission conservation violated: emitted {} != reference universe {} + anchored-not-in-reference {}",
                    n_after,
                    tiers.n_reference,
                    tiers.n_anchored_not_in_reference,
                );
            }
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
            ledger,
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
            // The shipped path passes literal `None`, which the library
            // resolves to the frozen Model A constant. This binary cannot
            // reach the parameterised entry's argument: `ci/gates.sh`
            // asserts both that the argument's name is absent here and
            // that this call passes `None`.
            telescope::quant::run_quant(quant_args, None)?;
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
            gap_abs,
            gap_rel,
            ledger,
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
            // The gap floors are printed even at their defaults precisely
            // BECAUSE the defaults are "off" and not calibrated values: a run
            // log that reads `gap_abs=0.0` without this line is indistinguishable
            // from a run where somebody believes zero is the right threshold.
            eprintln!(
                "  gap_abs:    {} ({})",
                gap_abs,
                if gap_abs == 0.0 {
                    "OFF — not a calibrated value"
                } else {
                    "active"
                }
            );
            eprintln!(
                "  gap_rel:    {} ({})",
                gap_rel,
                if gap_rel == 0.0 {
                    "OFF — not a calibrated value"
                } else {
                    "active"
                }
            );

            // RANGE-CHECK THE GAP FLOORS BEFORE THEY REACH THE RULE.
            //
            // `gap < p.gap_abs` is false for every `gap` when `p.gap_abs` is NaN,
            // and false for every non-negative `gap` when `p.gap_abs` is negative.
            // Both therefore switch the criterion OFF while the run banner keeps
            // printing the number the user typed — the ledger would say the gap was
            // 20 while the rule had no gap in it at all, and `rejected_gap` would
            // be 0 for a reason that looks exactly like "the data never failed the
            // gap". `g_rel >= 1.0` is the mirror trap: it can never be satisfied by
            // a positive relative gap, so it is also a silent disable.
            anyhow::ensure!(
                gap_abs.is_finite() && gap_abs >= 0.0,
                "--gap-abs must be a finite, non-negative number of AS points \
                 (got {:?}). NaN compares false against every gap and a negative \
                 value is unsatisfiable by a real score difference; either one \
                 silently switches the criterion off.",
                gap_abs
            );
            anyhow::ensure!(
                gap_rel.is_finite() && (0.0..1.0).contains(&gap_rel),
                "--gap-rel must be a finite fraction in [0.0, 1.0) (got {:?}). \
                 A value >= 1.0 demands a relative gap no positive score \
                 difference can supply, which switches the criterion off.",
                gap_rel
            );

            let tx_to_g = load_tx_to_gene(&tx_to_gene)?;
            eprintln!("  tx_to_gene entries: {}", tx_to_g.len());
            if tx_to_g.len() == 0 {
                eprintln!(
                    "  tx_to_gene:        (NONE — gene_veto_disabled=true; every \
                     multi-candidate read is discarded because the same-gene \
                     question cannot be answered without a map)"
                );
            }

            let params = anchor_builder::AnchorParams {
                tes_slack,
                min_coverage,
                max_reads_per_sample,
                gap_abs,
                gap_rel,
            };
            let pairs: Vec<(PathBuf, &str)> = bams
                .iter()
                .cloned()
                .zip(sample_ids.iter().map(|s| s.as_str()))
                .collect();
            let stats = anchor_builder::build_anchor_pool(&pairs, &tx_to_g, &output, params)
                .context("anchor_builder::build_anchor_pool failed")?;
            eprintln!("[telescope]  {}", stats.report());
            if let Some(p) = ledger.as_ref() {
                ledger_json::write_anchor(p, &stats)
                    .with_context(|| format!("failed to write --ledger {}", p.display()))?;
                eprintln!("[telescope]  anchor ledger: {}", p.display());
            }
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
            maybe_write_filter_ledger(ledger.as_ref(), &graph.stats)?;

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

/// Load the transcript -> gene map, FAILING CLOSED when one was asked for but
/// yielded nothing.
///
/// `TxToGene::from_tsv` SKIPS every line it cannot split into two non-empty tab
/// fields (`anchor_builder.rs`), so a header-only, space-delimited or truncated
/// file returns `Ok` with ZERO entries. Downstream, `resolve_gene` treats a
/// zero-entry map EXACTLY like no map at all and silently runs Universal Mode.
/// A user who asked for a map and got none would then read `gene_veto_disabled`
/// in the ledger and never learn the veto was disabled by their own typo.
///
/// So: no flag -> an empty map on purpose (and the caller says so out loud);
/// a flag that parses to nothing -> an error.
fn load_tx_to_gene(path: &Option<PathBuf>) -> Result<anchor_builder::TxToGene> {
    let Some(p) = path else {
        return Ok(anchor_builder::TxToGene::new());
    };
    let map = anchor_builder::TxToGene::from_tsv(&p)
        .with_context(|| format!("Failed to load tx-to-gene TSV: {:?}", p))?;
    anyhow::ensure!(
        map.len() > 0,
        "--tx-to-gene was given ({:?}) but parsed to ZERO entries. Every line \
         after the header must be `transcript_id<TAB>gene_id`. Check for spaces \
         instead of tabs, or a truncated file. Omit the flag entirely to run \
         deliberately without a gene map.",
        p
    );
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    // moved to the lib by this change; `use super::*` no longer reaches them
    use clap::CommandFactory;
    use telescope::{
        pct, ALN_WEIGHT_LAMBDA_DEFAULT, FIVE_PRIME_WEIGHT_DEFAULT, FIVE_PRIME_WINDOW_DEFAULT,
        PRUNE_MAX_DELTA_AS_DEFAULT,
    };

    fn write_tmp(name: &str, body: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("tel_q3_{}_{}", std::process::id(), name));
        std::fs::write(&p, body).unwrap();
        p
    }

    /// A SUPPLIED `--tx-to-gene` that parses to nothing must fail, not silently
    /// become Universal Mode.
    ///
    /// This is the falsifiable form of the guard in `load_tx_to_gene`. Delete the
    /// `anyhow::ensure!` from that function and `header_only_is_an_error` goes red,
    /// because `from_tsv` happily returns a ZERO-entry map for a header-only file
    /// and nothing downstream can tell it from "no flag was passed".
    #[test]
    fn a_supplied_but_empty_tx_to_gene_is_an_error_not_universal_mode() {
        // Omitting the flag is the DELIBERATE no-map path and must keep working.
        let none = load_tx_to_gene(&None).expect("no flag must stay legal");
        assert_eq!(none.len(), 0, "no flag means an empty map, on purpose");

        // Header only: parses fine, contains nothing.
        let hdr = write_tmp("hdr", "transcript_id\tgene_id\n");
        let e = load_tx_to_gene(&Some(hdr.clone()))
            .expect_err("a header-only TSV must NOT pass as an empty-but-valid map");
        assert!(e.to_string().contains("ZERO entries"), "got: {e}");

        // Spaces instead of tabs: every line is skipped by from_tsv, zero entries.
        let sp = write_tmp("sp", "transcript_id gene_id\ntx0 GA\ntx1 GB\n");
        let e = load_tx_to_gene(&Some(sp.clone()))
            .expect_err("a space-delimited TSV must not parse to zero and pass");
        assert!(e.to_string().contains("ZERO entries"), "got: {e}");

        // A real map still loads, unchanged.
        let ok = write_tmp("ok", "transcript_id\tgene_id\ntx0\tGA\ntx1\tGB\n");
        let m = load_tx_to_gene(&Some(ok.clone())).expect("a valid TSV must still load");
        assert_eq!(m.len(), 2, "the guard must not reject a genuine map");
        assert_eq!(m.get("tx0").map(String::as_str), Some("GA"));

        for f in [hdr, sp, ok] {
            let _ = std::fs::remove_file(f);
        }
    }

    /// A zero denominator must render as `n/a`, never as a percentage.
    ///
    /// `n_reads_multi_emission_candidate` is legitimately 0 whenever no read
    /// has two candidates carrying an emission row — which is the normal state
    /// of a badly-covered model, and precisely the state the `_all` / `_resolved`
    /// split was created to characterise. The tempting guard is
    /// `denominator.max(1)`, which removes the `NaN` and simultaneously reports
    /// "no read qualified" as a confident `0.0%`.
    ///
    /// (The `_all` / `_resolved` lines no longer print — removed 2026-10-07. The
    /// coverage counters `pct` is still called on can also be legitimately 0,
    /// which is why this guard and this test stay.)
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
