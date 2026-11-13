pub mod diff;
pub mod ledger_json;
pub mod quant;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use arrow_array::{Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use telescope_core::emission_model::FallbackConfig;
use telescope_core::gtf::GtfGenes;
use telescope_core::{anchor_builder, emission_model};

pub const ANCHOR_TES_SLACK_DEFAULT: usize = 35;

pub const ANCHOR_MIN_COVERAGE_DEFAULT: f64 = 0.80;

pub const ANCHOR_MAX_READS_PER_SAMPLE_DEFAULT: usize = 200;

pub const ANCHOR_GAP_ABS_DEFAULT: f64 = 12.0;

pub const ANCHOR_GAP_REL_DEFAULT: f64 = 0.0;

pub const EMISSION_TX_MIN_READS_DEFAULT: usize = 10;

pub const EMISSION_GENE_MIN_READS_DEFAULT: usize = 30;

pub const EMISSION_COHORT_MIN_READS_DEFAULT: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorScope {
    SingleSample,

    Cohort,
}

impl PriorScope {
    pub fn as_str(self) -> &'static str {
        match self {
            PriorScope::SingleSample => "single_sample",
            PriorScope::Cohort => "cohort",
        }
    }

    pub fn from_n_samples(n: usize) -> PriorScope {
        if n <= 1 {
            PriorScope::SingleSample
        } else {
            PriorScope::Cohort
        }
    }
}

pub struct AnchorRequest {
    pub bams: Vec<PathBuf>,

    pub sample_ids: Vec<String>,

    pub output: PathBuf,

    pub gtf: Option<PathBuf>,

    pub tes_slack: usize,

    pub min_coverage: f64,

    pub max_reads_per_sample: usize,

    pub gap_abs: f64,

    pub gap_rel: f64,

    pub ledger: Option<PathBuf>,
}

pub fn run_build_anchors(req: AnchorRequest) -> Result<()> {
    let AnchorRequest {
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
    } = req;
    if bams.len() != sample_ids.len() {
        anyhow::bail!(
            "--bam ({} files) and --sample-id ({} ids) must have equal length",
            bams.len(),
            sample_ids.len()
        );
    }
    eprintln!("[telescope] Universal Mode: anchor pool builder");
    eprintln!("  samples:    {}", bams.len());
    eprintln!("  tes_slack:  {} bp", tes_slack);
    eprintln!("  min_cov:    {}", min_coverage);
    eprintln!("  cap_per_tx: {}", max_reads_per_sample);
    eprintln!("  output:     {}", output.display());

    eprintln!(
        "  gap_abs:    {} ({})",
        gap_abs,
        if gap_abs == 0.0 { "disabled" } else { "active" }
    );
    eprintln!(
        "  gap_rel:    {} ({})",
        gap_rel,
        if gap_rel == 0.0 { "disabled" } else { "active" }
    );

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

    let tx_to_g = load_gtf(&gtf)?;
    eprintln!("  gtf transcripts:    {}", tx_to_g.len());
    if tx_to_g.is_empty() {
        eprintln!(
            "  gtf:               (NONE — no veto applies; a gap failure \
             cannot be classified as same- or cross-gene, so it is \
             counted as rejected_gap and gene_veto_disabled records the \
             reads whose relation was unavailable)"
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

    if let Some(warning) = anchor_builder::degenerate_gene_pool_warning(
        stats.distinct_genes,
        stats.distinct_transcripts,
    ) {
        eprintln!("{}", warning);
    }
    eprintln!("[telescope] done.");
    Ok(())
}

pub struct EmissionRequest {
    pub anchor_pool: PathBuf,

    pub transcript_evidence: Option<PathBuf>,

    pub reference_bam: Option<PathBuf>,

    pub reference_universe: Option<Vec<String>>,

    pub gtf: Option<PathBuf>,

    pub provenance_out: Option<PathBuf>,

    pub output: PathBuf,

    pub tx_min_reads: usize,

    pub gene_min_reads: usize,

    pub cohort_min_reads: usize,

    pub prior_scope: Option<PriorScope>,
}

pub fn run_build_emission(req: EmissionRequest) -> Result<()> {
    let EmissionRequest {
        anchor_pool,
        transcript_evidence,
        reference_bam,
        reference_universe,
        gtf,
        provenance_out,
        output,
        tx_min_reads,
        gene_min_reads,
        cohort_min_reads,
        prior_scope,
    } = req;
    eprintln!("[telescope] building emission model");

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

    let universe: Option<Vec<String>> = if let Some(u) = reference_universe {
        eprintln!(
            "  reference universe: {} transcripts (the @SQ headers of the \
             input BAMs)",
            u.len()
        );
        Some(u)
    } else if let Some(ref bam) = reference_bam {
        eprintln!("  reference_bam:     {}", bam.display());
        let u = emission_model::read_reference_universe(bam)
            .context("Failed to read the reference universe from the BAM header")?;
        eprintln!("  @SQ transcripts:  {}", u.len());
        Some(u)
    } else {
        eprintln!(
            "  reference_bam:     (none; the model will cover only the \
             anchor pool, and a read whose candidate is outside it will be \
             dropped with no profile row)"
        );
        None
    };
    let gtf_genes = match gtf {
        Some(ref p) => {
            eprintln!("  gtf:               {}", p.display());
            Some(GtfGenes::from_gtf(p).context("Failed to read the annotation GTF")?)
        }
        None => {
            eprintln!("  gtf:               (none; imputed rows use gene_id := transcript_id)");
            None
        }
    };

    let prov_path: PathBuf = provenance_out.clone().unwrap_or_else(|| {
        let mut q = output.clone();
        let stem = q
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "emission".to_string());
        q.set_file_name(format!("{stem}.provenance.json"));
        q
    });

    refuse_a_provenance_path_that_clobbers_a_table(&prov_path, &output)?;
    let reference_source_for_prov = reference_bam.clone();
    let tiers = emission_model::build_emission_from_anchors(
        anchor_pool.clone(),
        transcript_evidence,
        output,
        &fb,
        universe.as_deref(),
        gtf_genes.as_ref(),
    )
    .context("build_emission_from_anchors failed")?;

    refuse_a_provenance_path_that_clobbers_a_table(&prov_path, &tiers.provenance_path)?;

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
        "[telescope]    n_imputed_rows:           {} (padded: {}, suppressed: {})",
        tiers.n_imputed, tiers.n_imputed_padded, tiers.n_imputed_suppressed
    );

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

    let prior_sample_ids = anchor_pool_sample_ids(&anchor_pool)
        .with_context(|| format!("Failed to read the sample ids of {:?}", anchor_pool))?;
    let receipt = EmissionReceipt {
        reference_source: reference_source_for_prov,
        reference_count: tiers.n_reference,
        n_anchored,
        n_imputed: tiers.n_imputed,
        n_total: n_after,
        n_anchored_not_in_reference: tiers.n_anchored_not_in_reference,

        prior_scope: prior_scope
            .unwrap_or_else(|| PriorScope::from_n_samples(prior_sample_ids.len())),
        n_prior_samples: prior_sample_ids.len(),
        prior_sample_ids,
    };
    write_emission_provenance_json(&prov_path, &receipt)
        .with_context(|| format!("Failed to write provenance sidecar {:?}", prov_path))?;
    eprintln!("[telescope]  provenance JSON : {}", prov_path.display());

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

pub struct EmissionReceipt {
    pub reference_source: Option<PathBuf>,

    pub reference_count: usize,

    pub n_anchored: usize,

    pub n_imputed: usize,

    pub n_total: usize,

    pub n_anchored_not_in_reference: usize,

    pub prior_scope: PriorScope,

    pub n_prior_samples: usize,

    pub prior_sample_ids: Vec<String>,
}

pub fn emission_provenance_json(receipt: &EmissionReceipt) -> String {
    format!(
        "{{\n  \"reference_source\": {},\n  \"reference_count\": {},\n  \"reference_checksum\": null,\n  \"reference_checksum_note\": \"not computed: the universe is taken verbatim from the BAM @SQ header, which defines the reference transcript universe. No reference fasta or GTF was hashed, so `reference_source` identifies a FILE, not a universe; `reference_count` is the checkable quantity.\",\n  \"imputation_rule\": \"cohort_prior\",\n  \"imputed_marker\": \"n_cohort_flnc == 0\",\n  \"schema_columns\": 7,\n  \"n_anchored\": {},\n  \"n_imputed\": {},\n  \"n_total\": {},\n  \"n_anchored_not_in_reference\": {},\n  \"prior_scope\": \"{}\",\n  \"n_prior_samples\": {},\n  \"prior_sample_ids\": {:?}\n}}\n",
        match receipt.reference_source {
            Some(ref p) => format!("{:?}", p.display().to_string()),
            None => "null".to_string(),
        },
        receipt.reference_count,
        receipt.n_anchored,
        receipt.n_imputed,
        receipt.n_total,
        receipt.n_anchored_not_in_reference,
        receipt.prior_scope.as_str(),
        receipt.n_prior_samples,
        receipt.prior_sample_ids,
    )
}

pub fn write_emission_provenance_json(path: &Path, receipt: &EmissionReceipt) -> Result<()> {
    std::fs::write(path, emission_provenance_json(receipt))
        .with_context(|| format!("Failed to write the provenance sidecar {:?}", path))
}

pub fn anchor_pool_sample_ids(path: &Path) -> Result<Vec<String>> {
    let f = std::fs::File::open(path)
        .with_context(|| format!("Failed to open the anchor pool {:?}", path))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(f)
        .with_context(|| format!("Failed to read the anchor pool {:?}", path))?
        .build()
        .with_context(|| format!("Failed to read the anchor pool {:?}", path))?;
    let mut ids: Vec<String> = Vec::new();
    for batch in reader {
        let batch = batch.with_context(|| format!("Failed to read a batch of {:?}", path))?;
        let col = batch
            .column_by_name("sample_id")
            .ok_or_else(|| anyhow::anyhow!("the anchor pool {:?} has no sample_id column", path))?;
        let strings = col.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
            anyhow::anyhow!("the anchor pool sample_id column is not a string column")
        })?;
        for i in 0..strings.len() {
            if strings.is_null(i) {
                continue;
            }
            let s = strings.value(i).to_string();
            if !ids.contains(&s) {
                ids.push(s);
            }
        }
    }
    Ok(ids)
}

pub fn refuse_a_provenance_path_that_clobbers_a_table(
    prov: &std::path::Path,
    table: &std::path::Path,
) -> Result<()> {
    anyhow::ensure!(
        !same_file_or_path(prov, table),
        "--provenance-out {:?} is the same file as {:?}, which this run has \
         already written or is about to write as a PARQUET table. The run \
         header is JSON and truncates it. Choose a different --provenance-out.",
        prov,
        table
    );
    let is_parquet = prov
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("parquet"));
    anyhow::ensure!(
        !is_parquet,
        "--provenance-out {:?} ends in .parquet. This flag writes the JSON run \
         header, and writing it to a parquet path replaces that table with JSON \
         with no error. If you meant the per-transcript tier table, it is \
         written automatically beside --output and its path is printed by this \
         command; do not name it here.",
        prov
    );
    Ok(())
}

fn same_file_or_path(a: &std::path::Path, b: &std::path::Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

pub fn load_gtf(path: &Option<PathBuf>) -> Result<GtfGenes> {
    let Some(p) = path else {
        return Ok(GtfGenes::new());
    };
    GtfGenes::from_gtf(&p).with_context(|| format!("Failed to read the annotation GTF: {:?}", p))
}

pub fn temp_root() -> PathBuf {
    select_temp_root(std::env::var_os("TMPDIR"), Path::new("/var/tmp"))
}

fn select_temp_root(tmpdir: Option<std::ffi::OsString>, preferred: &Path) -> PathBuf {
    match tmpdir {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ if preferred.is_dir() => preferred.to_path_buf(),
        _ => std::env::temp_dir(),
    }
}

pub struct ScopedTempDir {
    path: PathBuf,
}

impl ScopedTempDir {
    pub fn new(prefix: &str) -> Result<Self> {
        let root = temp_root();
        std::fs::create_dir_all(&root)
            .with_context(|| format!("Failed to create the temporary root {:?}", root))?;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = root.join(format!("{prefix}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&path)
            .with_context(|| format!("Failed to create the temporary directory {:?}", path))?;
        Ok(ScopedTempDir { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScopedTempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

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

pub const POLY_A_SOURCE_HELP: &str = "Where each read's poly(A) length comes from: \
     `bam_tag:<TAG>` (a two-character SAM tag; `bam_tag:pt` is the Dorado native tag \
     and the default), or `none`, which declares that the BAM carries no poly(A) and \
     forces `--poly-a-weight 0`. A BAM with no poly(A) at all is an error, not an \
     empty graph.";

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

pub const ALN_WEIGHT_LAMBDA_DEFAULT: f64 = 20.0;

pub const FIVE_PRIME_WEIGHT_DEFAULT: f64 = 1.8;

pub const FIVE_PRIME_WINDOW_DEFAULT: u32 = 20;

pub const THREE_PRIME_WEIGHT_DEFAULT: f64 = 0.0;

pub const THREE_PRIME_WINDOW_DEFAULT: u32 = 15;

pub const POLY_A_WEIGHT_DEFAULT: f64 = 1.0;

pub const PRUNE_MAX_DELTA_AS_DEFAULT: i32 = 50;

pub fn pct(numerator: u64, denominator: u64) -> String {
    if denominator == 0 {
        "n/a".to_string()
    } else {
        format!("{:.1}%", numerator as f64 / denominator as f64 * 100.0)
    }
}

#[cfg(test)]
mod portable_temp_root_tests {
    use super::*;

    #[test]
    fn temp_root_selection_handles_empty_unset_and_missing_preferred_directory() {
        let preferred =
            std::env::temp_dir().join(format!("telescope-root-test-{}", std::process::id()));
        let absent = preferred.join("absent");
        std::fs::create_dir_all(&preferred).unwrap();
        let custom = preferred.join("custom");
        for value in [None, Some(std::ffi::OsString::new())] {
            assert_eq!(select_temp_root(value.clone(), &preferred), preferred);
            assert_eq!(select_temp_root(value, &absent), std::env::temp_dir());
        }
        assert_eq!(
            select_temp_root(Some(custom.clone().into_os_string()), &preferred),
            custom
        );
        assert_eq!(
            select_temp_root(Some(custom.clone().into_os_string()), &absent),
            custom
        );
        let file = preferred.join("file");
        std::fs::write(&file, b"not a directory").unwrap();
        assert_eq!(select_temp_root(None, &file), std::env::temp_dir());
        std::fs::remove_dir_all(&preferred).unwrap();
    }
}
