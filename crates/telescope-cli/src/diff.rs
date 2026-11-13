use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use arrow_array::array::{Float64Array, Int32Array, StringArray};
use arrow_array::{Array, RecordBatch};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use rayon::prelude::*;
use telescope_core::gtf::GtfGenes;

/// Default minimum posterior-weight mass for a (transcript × sample) cell.
pub const MIN_CELL_WEIGHT_DEFAULT: f64 = 10.0;

/// Default minimum Kish effective sample size for a (transcript × sample) cell.
pub const MIN_NEFF_DEFAULT: f64 = 10.0;

/// Default minimum total count for a transcript in the ΔPAU track.
pub const MIN_COUNTS_DEFAULT: f64 = 20.0;

/// Default minimum number of samples per transcript in the ΔPAL track.
pub const MIN_SAMPLES_DEFAULT: usize = 3;

/// The default threshold of the observed usage difference that makes a poly(A) site significant in the DEXSeq arm.
pub const APA_MIN_DELTA_PAU_DEFAULT: f64 = 0.05;

/// LAPA's counting method: how LAPA counts the reads that end at a poly(A) site.
pub const LAPA_COUNTING_METHOD: &str = "end";

/// LAPA's minimum replication rate: the fraction of a dataset's samples in which a cluster must be observed for it to count as replicated.
pub const LAPA_MIN_REPLICATION_RATE: f64 = 0.95;

/// The LAPA parameters the poly(A) calls of this run were made with.
#[derive(Debug, Clone)]
pub struct LapaParams {
    /// LAPA's counting method.
    pub counting_method: String,
    /// LAPA's minimum replication rate.
    pub min_replication_rate: String,
}

impl LapaParams {
    /// The parameters of a LAPA call this tool made.
    pub fn made_here() -> Self {
        LapaParams {
            counting_method: LAPA_COUNTING_METHOD.to_string(),
            min_replication_rate: format!("{}", LAPA_MIN_REPLICATION_RATE),
        }
    }

    /// The parameters of a LAPA call this run did not make.
    pub fn reused() -> Self {
        let note = "unknown (the LAPA run that produced these calls was not this run)";
        LapaParams {
            counting_method: note.to_string(),
            min_replication_rate: note.to_string(),
        }
    }

    /// These parameters with an explicit statement taking precedence over them.
    pub fn with_overrides(
        self,
        counting_method: Option<&str>,
        min_replication_rate: Option<&str>,
    ) -> Self {
        LapaParams {
            counting_method: counting_method
                .map(str::to_string)
                .unwrap_or(self.counting_method),
            min_replication_rate: min_replication_rate
                .map(str::to_string)
                .unwrap_or(self.min_replication_rate),
        }
    }
}

/// The name of the receipt file that sits next to a set of poly(A) cluster calls and names the parameters they were made with.
pub const LAPA_PARAMS_RECEIPT: &str = "lapa_params.tsv";

/// THE PARAMETERS OF A REUSED SET OF POLY(A) CALLS, READ FROM ITS OWN RECEIPT.
fn lapa_params_from_receipt(dir: &Path) -> Option<LapaParams> {
    for candidate in [
        dir.join(LAPA_PARAMS_RECEIPT),
        dir.join("canonical").join(LAPA_PARAMS_RECEIPT),
    ] {
        let text = match std::fs::read_to_string(&candidate) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let mut counting_method = None;
        let mut min_replication_rate = None;
        for line in text.lines() {
            let mut parts = line.splitn(2, '\t');
            let key = parts.next().unwrap_or("").trim();
            let value = parts.next().unwrap_or("").trim();
            if value.is_empty() {
                continue;
            }
            match key {
                "counting_method" => counting_method = Some(value.to_string()),
                "min_replication_rate" => min_replication_rate = Some(value.to_string()),
                _ => {}
            }
        }
        if counting_method.is_some() || min_replication_rate.is_some() {
            let unknown = LapaParams::reused();
            return Some(LapaParams {
                counting_method: counting_method.unwrap_or(unknown.counting_method),
                min_replication_rate: min_replication_rate.unwrap_or(unknown.min_replication_rate),
            });
        }
    }
    None
}

/// WRITE THE LAPA RECEIPT NEXT TO THE CALLS.
fn write_lapa_params_receipt(dir: &Path, params: &LapaParams) -> Result<()> {
    let path = dir.join(LAPA_PARAMS_RECEIPT);
    let mut text = String::from("parameter\tvalue\n");
    text.push_str(&format!("counting_method\t{}\n", params.counting_method));
    text.push_str(&format!(
        "min_replication_rate\t{}\n",
        params.min_replication_rate
    ));
    std::fs::write(&path, text).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// Default name of the metadata column holding the sample id.
pub const DEFAULT_SAMPLE_COLUMN: &str = "sample_id";

/// Default name of the metadata column holding the profile parquet.
pub const DEFAULT_PROFILE_COLUMN: &str = "profile_path";

/// Default name of the metadata column holding the condition label.
pub const DEFAULT_CONDITION_COLUMN: &str = "condition";

/// Default name of the metadata column holding the batch label.
pub const DEFAULT_BATCH_COLUMN: &str = "batch";

/// Default name of the metadata column holding the sample's alignment.
pub const DEFAULT_ALIGNMENT_COLUMN: &str = "alignment_path";

/// Default LAPA driver, looked up on PATH.
pub const DEFAULT_LAPA: &str = "lapa";

/// Arguments of the `diff` subcommand.
#[derive(clap::Args, Debug)]
pub struct DiffArgs {
    /// Metadata TSV: one row per sample.
    #[arg(long, required = true, help_heading = "INPUTS")]
    pub metadata: PathBuf,

    /// Column of the metadata holding the sample id.
    #[arg(long, default_value = DEFAULT_SAMPLE_COLUMN, help_heading = "INPUTS")]
    pub sample_column: String,

    /// Column of the metadata holding the profile parquet.
    #[arg(long, default_value = DEFAULT_PROFILE_COLUMN, help_heading = "INPUTS")]
    pub profile_column: String,

    /// Column of the metadata holding the condition label.
    #[arg(long, default_value = DEFAULT_CONDITION_COLUMN, help_heading = "INPUTS")]
    pub condition_column: String,

    /// Column of the metadata holding the batch label.
    #[arg(long, default_value = DEFAULT_BATCH_COLUMN, help_heading = "INPUTS")]
    pub batch_column: String,

    /// Column of the metadata holding the sample's genomic alignment.
    #[arg(long, default_value = DEFAULT_ALIGNMENT_COLUMN, help_heading = "INPUTS")]
    pub alignment_column: String,

    /// Annotation (GTF).
    #[arg(short = 'g', long = "gtf", help_heading = "REFERENCE")]
    pub gtf: Option<PathBuf>,

    /// Genome FASTA.
    #[arg(long, help_heading = "REFERENCE")]
    pub fasta: Option<PathBuf>,

    /// Chromosome sizes (`<name><TAB><length>` per sequence).
    #[arg(long, help_heading = "REFERENCE")]
    pub chrom_sizes: Option<PathBuf>,

    /// Directory the differential tables are written to.
    #[arg(long, help_heading = "OUTPUTS")]
    pub output_dir: PathBuf,

    /// The LAPA driver (`lapa`), used by the APA track.
    #[arg(long, default_value = DEFAULT_LAPA, help_heading = "TOOLS")]
    pub lapa_path: PathBuf,

    /// An existing LAPA output directory to use instead of running LAPA.
    #[arg(long, value_name = "DIR", help_heading = "INPUTS")]
    pub lapa_dir: Option<PathBuf>,

    /// The LAPA counting method the poly(A) cluster calls were made with.
    #[arg(long, value_name = "METHOD", help_heading = "TOOLS")]
    pub lapa_counting_method: Option<String>,

    /// The LAPA minimum replication rate the poly(A) cluster calls were made with.
    #[arg(long, value_name = "RATE", help_heading = "TOOLS")]
    pub lapa_min_replication_rate: Option<String>,

    /// Which modalities to compute: a comma-separated list of deg, det, diu, apa and dpl.
    #[arg(long, value_name = "MODALITIES", help_heading = "MODEL")]
    pub modalities: Option<String>,

    /// Minimum posterior-weight mass for a (transcript x sample) cell.
    #[arg(long = "min-cell-weight", value_name = "MIN_CELL_WEIGHT",
          default_value_t = MIN_CELL_WEIGHT_DEFAULT, help_heading = "GATES")]
    pub min_weight: f64,

    /// Minimum Kish effective sample size for a (transcript x sample) cell.
    #[arg(long, default_value_t = MIN_NEFF_DEFAULT, help_heading = "GATES")]
    pub min_neff: f64,

    /// Minimum total count for a feature to be tested: a gene in the expression track, a transcript in the abundance and usage tracks, and the gene's own reads in a sample in the usage track's gate.
    #[arg(long, default_value_t = MIN_COUNTS_DEFAULT, help_heading = "GATES")]
    pub min_counts: f64,

    /// Minimum number of samples per group.
    #[arg(long, default_value_t = MIN_SAMPLES_DEFAULT, help_heading = "GATES")]
    pub min_samples: usize,

    /// The absolute observed usage difference a poly(A) site must reach to be called significant in the DEXSeq arm.
    #[arg(long, default_value_t = APA_MIN_DELTA_PAU_DEFAULT, help_heading = "MODEL")]
    pub apa_min_delta_pau: f64,

    /// R interpreter to run the engine with.
    #[arg(long, help_heading = "TOOLS")]
    pub rscript: Option<PathBuf>,

    /// The R engine itself.
    #[arg(long, help_heading = "TOOLS")]
    pub r_script: Option<PathBuf>,

    /// Write only the (transcript x sample) aggregate table and stop, without running the statistical engine.
    #[arg(long, help_heading = "OUTPUTS")]
    pub aggregate_only: bool,

    /// The case condition label.
    #[arg(long, help_heading = "INPUTS")]
    pub case_condition: Option<String>,

    /// The control condition label.
    #[arg(long, help_heading = "INPUTS")]
    pub control_condition: Option<String>,

    /// The contrast as CASE:CONTROL, for example treated:control.
    #[arg(long, help_heading = "INPUTS")]
    pub contrast: Option<String>,
}

/// The five modalities, in the order the tables are named.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Modality {
    /// Gene-level differential expression (DESeq2, in the R engine).
    Deg,
    /// Transcript-level differential abundance (DESeq2, in the R engine).
    Det,
    /// Differential isoform usage (DEXSeq, in the R engine).
    Diu,
    /// Alternative poly(A) site usage (LAPA's own usage column, tested with DEXSeq by the engine).
    Apa,
    /// Differential poly(A) length (the DT-EB limma length model, in the engine).
    Dpl,
}

impl Modality {
    /// The modality's name, as `--modalities` spells it.
    pub fn as_str(&self) -> &'static str {
        match self {
            Modality::Deg => "deg",
            Modality::Det => "det",
            Modality::Diu => "diu",
            Modality::Apa => "apa",
            Modality::Dpl => "dpl",
        }
    }

    /// The table the engine writes for this modality.
    pub fn table(&self) -> &'static str {
        match self {
            Modality::Deg => "deg.tsv",
            Modality::Det => "det.tsv",
            Modality::Diu => "diu.tsv",
            Modality::Apa => "apa.tsv",
            Modality::Dpl => "dpl.tsv",
        }
    }

    /// Every modality, in the order the contract names them.
    pub fn all() -> Vec<Modality> {
        vec![
            Modality::Deg,
            Modality::Det,
            Modality::Diu,
            Modality::Apa,
            Modality::Dpl,
        ]
    }
}

/// A preflight failure: what is wrong, and what to do about it.
#[derive(Debug, Clone)]
pub struct PreflightFailure {
    /// The `ERROR:` line, without the prefix.
    pub what: String,
    /// The `Fix:` line, without the prefix.
    pub fix: String,
}

impl PreflightFailure {
    /// A failure with an explicit repair.
    pub fn new(what: impl Into<String>, fix: impl Into<String>) -> Self {
        PreflightFailure {
            what: what.into(),
            fix: fix.into(),
        }
    }

    /// The three lines the contract fixes, as one string.
    pub fn message(&self) -> String {
        format!(
            "ERROR: {}\nFix: {}\nNo partial result directory was created.",
            self.what, self.fix
        )
    }
}

impl std::fmt::Display for PreflightFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for PreflightFailure {}

/// One sample of a run: what it is, where its profile is, and what it is.
#[derive(Debug, Clone)]
pub struct Sample {
    /// The sample id, as the metadata (or the design) names it.
    pub sample_id: String,
    /// The profile parquet of the sample.
    pub profile: PathBuf,
    /// The condition label of the sample.
    pub condition: Option<String>,
    /// The batch label of the sample, when the metadata declares one.
    pub batch: Option<String>,
    /// The genomic alignment of the sample, when the metadata declares one.
    pub alignment: Option<PathBuf>,
}

/// What the preflight decided.
#[derive(Debug, Clone)]
pub struct Plan {
    /// The samples of the run, in the order the metadata declares them.
    pub samples: Vec<Sample>,
    /// The profile files to aggregate, in order.
    pub profiles: Vec<PathBuf>,
    /// The case condition label.
    pub case_condition: Option<String>,
    /// The control condition label.
    pub control_condition: Option<String>,
    /// The modalities to compute.
    pub modalities: Vec<Modality>,
    /// The batch column, when the metadata declares a usable one.
    pub batch_column: Option<String>,
    /// What the metadata's batch column turned out to be.
    pub batch_state: BatchState,
    /// The R interpreter.
    pub rscript: PathBuf,
    /// The R engine.
    pub r_script: Option<PathBuf>,
    /// The LAPA driver, when the APA track is computed.
    pub lapa: Option<PathBuf>,
    /// An existing LAPA output directory to reuse instead of running LAPA.
    pub lapa_dir: Option<PathBuf>,
    /// The LAPA parameters the poly(A) calls of this run carry.
    pub lapa_params: LapaParams,
    /// The annotation, when the run was given one.
    pub gtf: Option<PathBuf>,
    /// What every gene of the annotation is CALLED.
    pub gene_names: BTreeMap<String, String>,
    /// The genome FASTA, when the APA track is computed.
    pub fasta: Option<PathBuf>,
    /// The chromosome sizes, when the APA track is computed.
    pub chrom_sizes: Option<PathBuf>,
}

/// The batch note the run records when the metadata has no usable batch.
pub const NO_BATCH_NOTE: &str = "no batch column in the metadata";

/// What the metadata's batch column turned out to be.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum BatchState {
    /// No batch column at all.
    #[default]
    NoColumn,
    /// A batch column, but at least one sample leaves it empty, so it cannot be a covariate.
    ColumnWithMissingValues(String),
    /// A batch column and every sample fills it in.
    ColumnPresent(String),
}

impl BatchState {
    /// The note the run records.
    pub fn note(&self) -> String {
        match self {
            BatchState::NoColumn => NO_BATCH_NOTE.to_string(),
            BatchState::ColumnWithMissingValues(c) => format!(
                "the metadata has a batch column ({c}), but at least one sample leaves it \
                 empty, so it was not fitted as a covariate"
            ),
            BatchState::ColumnPresent(c) => format!(
                "a batch column is in the metadata: {c} (whether the batch was fitted as a \
                 covariate is decided by the engine, which reports it in \
                 logs/telescope_diff.R.log and in the model_parameters column of every table)"
            ),
        }
    }
}

/// The two labels of the contrast, in the order the engine needs them.
pub fn resolve_contrast(
    case_condition: Option<&str>,
    control_condition: Option<&str>,
    contrast: Option<&str>,
) -> Result<(String, String)> {
    let mut case_label = case_condition.map(|s| s.to_string());
    let mut control_label = control_condition.map(|s| s.to_string());
    if let Some(spec) = contrast {
        if case_condition.is_some() || control_condition.is_some() {
            bail!("use either --contrast or --case-condition/--control-condition, not both");
        }
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() != 2 || parts.iter().any(|p| p.is_empty()) {
            bail!("--contrast must be CASE:CONTROL, for example treated:control");
        }
        case_label = Some(parts[0].to_string());
        control_label = Some(parts[1].to_string());
    }
    let (case_label, control_label) = match (case_label, control_label) {
        (Some(c), Some(k)) => (c, k),
        _ => bail!(
            "the contrast must be given. Use --case-condition and --control-condition, \
             or --contrast CASE:CONTROL."
        ),
    };
    if case_label == control_label {
        bail!("the case and the control condition are the same label: {case_label}");
    }
    Ok((case_label, control_label))
}

/// The number of samples in each group of the contrast.
fn group_sizes(plan: &Plan) -> Option<(usize, usize)> {
    let case = plan.case_condition.as_deref()?;
    let control = plan.control_condition.as_deref()?;
    let mut n_case = 0usize;
    let mut n_control = 0usize;
    let mut labelled = 0usize;
    for s in &plan.samples {
        match s.condition.as_deref() {
            Some(c) if c == case => {
                n_case += 1;
                labelled += 1;
            }
            Some(c) if c == control => {
                n_control += 1;
                labelled += 1;
            }
            Some(_) => {}
            None => {}
        }
    }
    if labelled == 0 {
        return None;
    }
    Some((n_control, n_case))
}

/// The modalities a run computes.
pub fn resolve_modalities(spec: Option<&str>) -> Result<Vec<Modality>> {
    let Some(spec) = spec else {
        return Ok(Modality::all());
    };
    let mut out: Vec<Modality> = Vec::new();
    for name in spec.split(',') {
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        let m = match name {
            "deg" => Modality::Deg,
            "det" => Modality::Det,
            "diu" => Modality::Diu,
            "apa" => Modality::Apa,
            "dpl" => Modality::Dpl,
            other => {
                bail!("unknown modality '{other}'. The modalities are deg, det, diu, apa and dpl.")
            }
        };
        if !out.contains(&m) {
            out.push(m);
        }
    }
    if out.is_empty() {
        bail!("--modalities names no modality. The modalities are deg, det, diu, apa and dpl.");
    }
    out.sort();
    Ok(out)
}

/// A TSV with a header, read into its header names and its rows.
struct Tsv {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Tsv {
    /// Read a TSV with a header row.
    fn read(path: &Path) -> std::io::Result<Tsv> {
        let text = std::fs::read_to_string(path)?;
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header_line = lines.next().unwrap_or_default();
        let header: Vec<String> = header_line
            .split('\t')
            .map(|h| h.trim().trim_start_matches('\u{feff}').to_string())
            .collect();
        let rows = lines
            .map(|l| l.split('\t').map(|f| f.trim().to_string()).collect())
            .collect();
        Ok(Tsv { header, rows })
    }

    /// The index of a named column.
    fn column(&self, name: &str) -> Option<usize> {
        self.header.iter().position(|h| h == name)
    }

    /// The value of a column in a row, or the empty string.
    fn value(&self, row: &[String], index: usize) -> String {
        row.get(index).cloned().unwrap_or_default()
    }
}

/// Read the metadata table.
fn read_metadata(args: &DiffArgs, path: &Path) -> Result<MetadataTable, PreflightFailure> {
    if !path.exists() {
        return Err(PreflightFailure::new(
            format!("the metadata file does not exist: {}", path.display()),
            "point --metadata at the TSV that declares the samples.",
        ));
    }
    let tsv = Tsv::read(path).map_err(|e| {
        PreflightFailure::new(
            format!("the metadata file cannot be read: {}: {e}", path.display()),
            "check that the file is a readable TSV with a header row.",
        )
    })?;
    if tsv.header.is_empty() || tsv.header[0].is_empty() {
        return Err(PreflightFailure::new(
            format!("the metadata file has no header row: {}", path.display()),
            "the first line must name the columns.",
        ));
    }
    for (column, flag) in [
        (&args.sample_column, "--sample-column"),
        (&args.profile_column, "--profile-column"),
        (&args.condition_column, "--condition-column"),
    ] {
        if tsv.column(column).is_none() {
            return Err(PreflightFailure::new(
                format!(
                    "the metadata has no '{column}' column. Its columns are: {}.",
                    tsv.header.join(", ")
                ),
                format!("name the column with {flag}, or rename it in the metadata."),
            ));
        }
    }
    let sample_col = tsv.column(&args.sample_column).unwrap();
    let profile_col = tsv.column(&args.profile_column).unwrap();
    let condition_col = tsv.column(&args.condition_column).unwrap();
    let batch_col = tsv.column(&args.batch_column);
    let alignment_col = tsv.column(&args.alignment_column);

    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let resolve = |p: &str| -> PathBuf {
        let candidate = PathBuf::from(p);
        if candidate.is_absolute() {
            candidate
        } else {
            base.join(candidate)
        }
    };

    let mut samples: Vec<Sample> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for (i, row) in tsv.rows.iter().enumerate() {
        let sample_id = tsv.value(row, sample_col);
        if sample_id.is_empty() {
            return Err(PreflightFailure::new(
                format!(
                    "row {} of the metadata has an empty '{}'.",
                    i + 2,
                    args.sample_column
                ),
                "every row must name its sample.",
            ));
        }
        if seen.contains(&sample_id) {
            return Err(PreflightFailure::new(
                format!("the metadata declares the sample '{sample_id}' twice."),
                "one row per sample.",
            ));
        }
        seen.push(sample_id.clone());
        let profile = tsv.value(row, profile_col);
        if profile.is_empty() {
            return Err(PreflightFailure::new(
                format!(
                    "the sample '{sample_id}' has an empty '{}'.",
                    args.profile_column
                ),
                "every row must name its profile parquet.",
            ));
        }
        let condition = tsv.value(row, condition_col);
        let batch = batch_col.map(|c| tsv.value(row, c));
        let alignment = alignment_col
            .map(|c| tsv.value(row, c))
            .filter(|a| !a.is_empty())
            .map(|a| resolve(&a));
        samples.push(Sample {
            sample_id,
            profile: resolve(&profile),
            condition: if condition.is_empty() {
                None
            } else {
                Some(condition)
            },
            batch,
            alignment,
        });
    }
    if samples.is_empty() {
        return Err(PreflightFailure::new(
            format!("the metadata declares no sample: {}", path.display()),
            "one row per sample is what the run needs.",
        ));
    }
    Ok(MetadataTable {
        samples,
        has_batch_column: batch_col.is_some(),
        has_alignment_column: alignment_col.is_some(),
    })
}

/// The metadata of a run.
#[derive(Debug)]
struct MetadataTable {
    /// The samples, in the order the metadata declares them.
    samples: Vec<Sample>,
    /// Whether the metadata declares a batch column at all.
    has_batch_column: bool,
    /// Whether the metadata declares an alignment column at all.
    has_alignment_column: bool,
}

/// Write the metadata TSV the engine reads.
fn write_metadata_tsv(path: &Path, samples: &[Sample], with_batch: bool) -> Result<()> {
    let mut out = String::from("sample_id\tcondition\tprofile_path");
    if with_batch {
        out.push_str("\tbatch");
    }
    out.push('\n');
    for s in samples {
        out.push_str(&format!(
            "{}\t{}\t{}",
            s.sample_id,
            s.condition.as_deref().unwrap_or(""),
            s.profile.display()
        ));
        if with_batch {
            out.push_str(&format!("\t{}", s.batch.as_deref().unwrap_or("")));
        }
        out.push('\n');
    }
    std::fs::write(path, out).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// How a transcript's gene was resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TranscriptGene {
    /// A real gene, different from the transcript id.
    Mapped(String),
    /// The "gene" is the transcript id: there is no gene structure here.
    SelfId,
    /// No gene at all.
    Absent,
}

/// The gene of every transcript of the input.
fn resolve_gene_structure(
    gene_map: Option<&HashMap<String, String>>,
    transcript_genes: &HashMap<String, String>,
    gtf_path: Option<&Path>,
) -> Result<(), PreflightFailure> {
    let mut resolved: Vec<(String, TranscriptGene)> = Vec::new();
    for (tx, profile_gene) in transcript_genes {
        let gene = match gene_map {
            Some(map) => match map.get(tx) {
                Some(g) => {
                    let g = g.trim();
                    if g.is_empty() {
                        TranscriptGene::Absent
                    } else if g == tx {
                        TranscriptGene::SelfId
                    } else {
                        TranscriptGene::Mapped(g.to_string())
                    }
                }
                None => {
                    let where_ = gtf_path
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "--gtf".to_string());
                    return Err(PreflightFailure::new(
                        format!("the transcript '{tx}' is not in the annotation ({where_})."),
                        "use an annotation that covers every transcript, or drop --gtf \
                         and use profiles whose gene_id column holds the real genes.",
                    ));
                }
            },
            None => {
                let g = profile_gene.trim();
                if g.is_empty() {
                    TranscriptGene::Absent
                } else if g == tx {
                    TranscriptGene::SelfId
                } else {
                    TranscriptGene::Mapped(g.to_string())
                }
            }
        };
        resolved.push((tx.clone(), gene));
    }
    if resolved.is_empty() {
        return Err(PreflightFailure::new(
            "the profiles declare no transcript.",
            "check the profile files: a profile with no transcript cannot give the \
             APA track a gene structure.",
        ));
    }
    let all_self = resolved.iter().all(|(_, g)| *g == TranscriptGene::SelfId);
    if all_self {
        return Err(PreflightFailure::new(
            "transcript-to-gene structure is absent. APA is not identifiable for this \
             input. Provide --gtf or a profile containing valid gene_id values.",
            "provide --gtf, or use profiles whose gene_id column holds the real \
             genes.",
        ));
    }
    if let Some((tx, _)) = resolved
        .iter()
        .find(|(_, g)| matches!(g, TranscriptGene::Absent | TranscriptGene::SelfId))
    {
        return Err(PreflightFailure::new(
            format!("the transcript '{tx}' has no gene: its gene is its own transcript id."),
            "provide --gtf, or use profiles whose gene_id column holds the real \
             genes.",
        ));
    }
    Ok(())
}

/// The distinct (transcript, gene) pairs a profile parquet declares.
fn profile_transcript_genes(path: &Path) -> Result<HashMap<String, String>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("failed to read {} as parquet", path.display()))?;
    let mask = ProjectionMask::columns(builder.parquet_schema(), ["gene_id", "transcript_id"]);
    let mut reader = builder
        .with_projection(mask)
        .with_batch_size(8192)
        .build()
        .with_context(|| format!("failed to open {} for reading", path.display()))?;
    let mut map: HashMap<String, String> = HashMap::new();
    while let Some(batch) = reader.next() {
        let batch =
            batch.with_context(|| format!("failed to read a batch of {}", path.display()))?;
        let gene = str_col(&batch, "gene_id", path)?;
        let tx = str_col(&batch, "transcript_id", path)?;
        for i in 0..tx.len() {
            if tx.is_null(i) {
                continue;
            }
            let t = tx.value(i);
            let g = if gene.is_null(i) { "" } else { gene.value(i) };
            match map.get(t) {
                None => {
                    map.insert(t.to_string(), g.to_string());
                }
                Some(prev) if prev == g => {}
                Some(prev) => {
                    bail!(
                        "the transcript '{t}' has two genes in {}: '{prev}' and '{g}'.",
                        path.display()
                    );
                }
            }
        }
    }
    Ok(map)
}

/// One aggregated (sample × gene × transcript) cell.
#[derive(Debug, Clone, PartialEq)]
pub struct AggRow {
    pub sample_id: String,
    pub gene_id: String,
    pub transcript_id: String,
    /// Posterior mass assigned to this transcript in this sample.
    pub weight_sum: f64,
    /// Numerator of the posterior-weighted mean log2 tail length.
    pub weighted_log2pal_sum: f64,
    /// Sum of the squared posterior weights of the cell's reads.
    pub weight_sq_sum: f64,
    /// Sum of posterior weight times squared log2 tail length.
    pub weighted_log2pal_sq_sum: f64,
}

/// String interning, so a 15 M-row scan allocates each distinct id once instead of once per row.
struct Interner {
    map: HashMap<String, u32>,
    names: Vec<String>,
}

impl Interner {
    fn new() -> Self {
        Interner {
            map: HashMap::new(),
            names: Vec::new(),
        }
    }

    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.map.get(s) {
            return i;
        }
        let i = self.names.len() as u32;
        self.names.push(s.to_string());
        self.map.insert(s.to_string(), i);
        i
    }

    fn resolve(&self, i: u32) -> &str {
        self.names
            .get(i as usize)
            .map(|s| s.as_str())
            .expect("interned id must exist")
    }
}

/// Add up one profile parquet into per-(sample × gene × transcript) cells.
pub fn aggregate_profile(path: &Path) -> Result<Vec<AggRow>> {
    aggregate_profile_impl(path, None)
}

/// Add up one profile parquet, attributing every row to `sample_id`.
pub fn aggregate_profile_as(path: &Path, sample_id: &str) -> Result<Vec<AggRow>> {
    aggregate_profile_impl(path, Some(sample_id))
}

/// Add up one profile parquet, optionally overriding the sample id.
fn aggregate_profile_impl(path: &Path, sample_id: Option<&str>) -> Result<Vec<AggRow>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("failed to read {} as parquet", path.display()))?;
    let mut reader = builder
        .with_batch_size(8192)
        .build()
        .with_context(|| format!("failed to open {} for reading", path.display()))?;

    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    let mut interner = Interner::new();
    let mut acc: HashMap<(u32, u32, u32), (f64, f64, f64, f64)> = HashMap::new();
    let mut n_rows = 0u64;
    let mut declared_sample: Option<String> = None;

    while let Some(batch) = reader.next() {
        let batch =
            batch.with_context(|| format!("failed to read a batch of {}", path.display()))?;

        let gene = str_col(&batch, "gene_id", path)?;
        let tx = str_col(&batch, "transcript_id", path)?;
        let pal = int_col(&batch, "polyA_length", path)?;
        let pw = f64_col(&batch, "posterior_weight", path)?;

        let sample = match batch.column_by_name("sample_id") {
            None => None,
            Some(c) => {
                let arr = c
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("profile 'sample_id' column is not a string column")?;
                if arr.null_count() == arr.len() {
                    None
                } else {
                    Some(arr)
                }
            }
        };
        let stem_id = interner.intern(&stem);
        let override_id = sample_id.map(|s| interner.intern(s));

        for i in 0..gene.len() {
            if gene.is_null(i) || tx.is_null(i) || pal.is_null(i) || pw.is_null(i) {
                bail!(
                    "null value in {} at row {} of the first batch: the profile schema \
                     declares these columns non-nullable",
                    path.display(),
                    i
                );
            }
            let s = match (override_id, sample) {
                (Some(id), _) => id,
                (None, Some(arr)) if !arr.is_null(i) => interner.intern(arr.value(i)),
                _ => stem_id,
            };

            if let Some(arr) = sample {
                if !arr.is_null(i) && declared_sample.is_none() {
                    declared_sample = Some(arr.value(i).to_string());
                }
            }
            let g = interner.intern(gene.value(i));
            let t = interner.intern(tx.value(i));
            let w = pw.value(i);

            let log2pal = (pal.value(i).max(1) as f64).log2();
            let e = acc.entry((s, g, t)).or_insert((0.0, 0.0, 0.0, 0.0));
            e.0 += w;
            e.1 += w * log2pal;
            e.2 += w * w;
            e.3 += w * log2pal * log2pal;
            n_rows += 1;
        }
    }

    if let (Some(declared), Some(forced)) = (&declared_sample, sample_id) {
        if declared != forced {
            bail!(
                "{} declares the sample '{}' but the metadata calls it '{}'. \
                 Publishing it would file one sample's rows under another's name.",
                path.display(),
                declared,
                forced
            );
        }
    }

    let mut out: Vec<AggRow> = acc
        .into_iter()
        .map(|((s, g, t), (w, l, w2, l2))| AggRow {
            sample_id: interner.resolve(s).to_string(),
            gene_id: interner.resolve(g).to_string(),
            transcript_id: interner.resolve(t).to_string(),
            weight_sum: w,
            weighted_log2pal_sum: l,
            weight_sq_sum: w2,
            weighted_log2pal_sq_sum: l2,
        })
        .collect();
    out.sort_by(|a, b| {
        a.sample_id
            .cmp(&b.sample_id)
            .then(a.gene_id.cmp(&b.gene_id))
            .then(a.transcript_id.cmp(&b.transcript_id))
    });
    eprintln!(
        "[telescope]  {}: {} reads -> {} (transcript x sample) cells",
        path.display(),
        n_rows,
        out.len()
    );
    Ok(out)
}

fn str_col<'a>(batch: &'a RecordBatch, name: &str, path: &Path) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .with_context(|| format!("{} has no '{name}' column", path.display()))?
        .as_any()
        .downcast_ref::<StringArray>()
        .with_context(|| format!("{}: '{name}' is not a string column", path.display()))
}

fn int_col<'a>(batch: &'a RecordBatch, name: &str, path: &Path) -> Result<&'a Int32Array> {
    batch
        .column_by_name(name)
        .with_context(|| format!("{} has no '{name}' column", path.display()))?
        .as_any()
        .downcast_ref::<Int32Array>()
        .with_context(|| format!("{}: '{name}' is not an int32 column", path.display()))
}

fn f64_col<'a>(batch: &'a RecordBatch, name: &str, path: &Path) -> Result<&'a Float64Array> {
    batch
        .column_by_name(name)
        .with_context(|| format!("{} has no '{name}' column", path.display()))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .with_context(|| format!("{}: '{name}' is not a float64 column", path.display()))
}

/// Merge the cells of several aggregations.
fn merge_agg_rows(per_file: Vec<Result<Vec<AggRow>>>) -> Result<Vec<AggRow>> {
    let mut merged: HashMap<(String, String, String), (f64, f64, f64, f64)> = HashMap::new();

    for r in per_file {
        for row in r? {
            let e = merged
                .entry((row.sample_id, row.gene_id, row.transcript_id))
                .or_insert((0.0, 0.0, 0.0, 0.0));
            e.0 += row.weight_sum;
            e.1 += row.weighted_log2pal_sum;
            e.2 += row.weight_sq_sum;
            e.3 += row.weighted_log2pal_sq_sum;
        }
    }
    let mut out: Vec<AggRow> = merged
        .into_iter()
        .map(
            |(
                (sample_id, gene_id, transcript_id),
                (weight_sum, weighted_log2pal_sum, weight_sq_sum, weighted_log2pal_sq_sum),
            )| AggRow {
                sample_id,
                gene_id,
                transcript_id,
                weight_sum,
                weighted_log2pal_sum,
                weight_sq_sum,
                weighted_log2pal_sq_sum,
            },
        )
        .collect();
    out.sort_by(|a, b| {
        a.sample_id
            .cmp(&b.sample_id)
            .then(a.gene_id.cmp(&b.gene_id))
            .then(a.transcript_id.cmp(&b.transcript_id))
    });
    Ok(out)
}

/// Aggregate the samples' profiles, each attributed to its own sample id.
fn aggregate_samples(samples: &[Sample]) -> Result<Vec<AggRow>> {
    let per_file: Vec<Result<Vec<AggRow>>> = samples
        .par_iter()
        .map(|s| aggregate_profile_as(&s.profile, &s.sample_id))
        .collect();
    merge_agg_rows(per_file)
}

/// Write the gene-name map the R engine reads: `gene_id<TAB>gene_name`.
fn write_gene_names_tsv(path: &Path, names: &BTreeMap<String, String>) -> Result<()> {
    let mut f = std::fs::File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    for (gene_id, gene_name) in names {
        writeln!(f, "{gene_id}\t{gene_name}")?;
    }
    Ok(())
}

/// Write the aggregate table the R engine reads.
pub fn write_aggregate_tsv(path: &Path, rows: &[AggRow]) -> Result<()> {
    let mut out = String::with_capacity(rows.len() * 96 + 64);
    out.push_str(
        "sample_id\tgene_id\ttranscript_id\tweight_sum\tweighted_log2pal_sum\tweight_sq_sum\tweighted_log2pal_sq_sum\n",
    );
    for r in rows {
        out.push_str(&format!(
            "{}\t{}\t{}\t{:?}\t{:?}\t{:?}\t{:?}\n",
            r.sample_id,
            r.gene_id,
            r.transcript_id,
            r.weight_sum,
            r.weighted_log2pal_sum,
            r.weight_sq_sum,
            r.weighted_log2pal_sq_sum
        ));
    }
    std::fs::write(path, out).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// Locate the R interpreter: explicit path, then `$TELESCOPE_RSCRIPT`, then `Rscript` on PATH.
pub fn resolve_rscript(explicit: Option<&Path>) -> PathBuf {
    if let Some(p) = explicit {
        return p.to_path_buf();
    }
    if let Ok(v) = std::env::var("TELESCOPE_RSCRIPT") {
        if !v.trim().is_empty() {
            return PathBuf::from(v);
        }
    }
    PathBuf::from("Rscript")
}

/// Locate the R engine: explicit path, then `$TELESCOPE_DIFF_R`, then the conda data directory, then the copy shipped beside the binary.
pub fn resolve_r_script(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return check_engine_path(p);
    }
    if let Ok(v) = std::env::var("TELESCOPE_DIFF_R") {
        if !v.trim().is_empty() {
            return check_engine_path(Path::new(&v));
        }
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        candidates
            .push(PathBuf::from(&prefix).join("share/telescope/workflow/scripts/telescope_diff.R"));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("telescope_diff.R"));
            candidates.push(dir.join("../share/telescope/workflow/scripts/telescope_diff.R"));
            candidates.push(dir.join("../../workflow/scripts/telescope_diff.R"));
        }
    }
    for c in &candidates {
        if c.is_file() {
            return Ok(c.clone());
        }
    }
    bail!(
        "cannot find the R engine. Pass --r-script <path> pointing at \
         workflow/scripts/telescope_diff.R, or set TELESCOPE_DIFF_R to that path. \
         Looked in: {}",
        candidates
            .iter()
            .map(|c| c.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn check_engine_path(p: &Path) -> Result<PathBuf> {
    if !p.is_file() {
        bail!(
            "the R engine does not exist: {}. Point --r-script at the engine script \
             (workflow/scripts/telescope_diff.R in this repository), or set \
             TELESCOPE_DIFF_R.",
            p.display()
        );
    }
    Ok(p.to_path_buf())
}

/// Find an executable: a path with a separator in it must exist and carry the unix executable bit, a bare name is looked up on PATH.
#[cfg(unix)]
fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    match path.metadata() {
        Ok(md) => md.is_file() && md.permissions().mode() & 0o111 != 0,
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn is_executable_file(path: &Path) -> bool {
    path.is_file()
}

fn find_executable(name: &Path) -> Option<PathBuf> {
    if name.components().count() > 1 {
        return if is_executable_file(name) {
            Some(name.to_path_buf())
        } else {
            None
        };
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if is_executable_file(&candidate) {
            return Some(candidate);
        }
    }
    None
}

/// Check every precondition, before anything is computed.
fn preflight(args: &DiffArgs) -> Result<Plan, PreflightFailure> {
    let contrast = if args.aggregate_only {
        None
    } else {
        Some(
            resolve_contrast(
                args.case_condition.as_deref(),
                args.control_condition.as_deref(),
                args.contrast.as_deref(),
            )
            .map_err(|e| {
                PreflightFailure::new(
                    e.to_string(),
                    "give the contrast with --case-condition and --control-condition, or \
                     with --contrast CASE:CONTROL.",
                )
            })?,
        )
    };

    let modalities = if args.aggregate_only {
        Vec::new()
    } else {
        resolve_modalities(args.modalities.as_deref()).map_err(|e| {
            PreflightFailure::new(
                e.to_string(),
                "name the modalities as a comma-separated list of deg, det, diu, apa and dpl.",
            )
        })?
    };

    let table = read_metadata(args, &args.metadata)?;

    let usable_batch = table.has_batch_column
        && table
            .samples
            .iter()
            .all(|s| s.batch.as_deref().map(|b| !b.is_empty()).unwrap_or(false));
    let batch_state = if !table.has_batch_column {
        BatchState::NoColumn
    } else if usable_batch {
        BatchState::ColumnPresent(args.batch_column.clone())
    } else {
        BatchState::ColumnWithMissingValues(args.batch_column.clone())
    };
    let batch_column = if usable_batch {
        Some(args.batch_column.clone())
    } else {
        None
    };
    let samples = table.samples;
    let profiles = samples.iter().map(|s| s.profile.clone()).collect();
    let has_alignment_column = table.has_alignment_column;

    for s in &samples {
        if !s.profile.is_file() {
            return Err(PreflightFailure::new(
                format!(
                    "the profile of the sample '{}' does not exist: {}",
                    s.sample_id,
                    s.profile.display()
                ),
                "correct the profile path (a relative path is resolved against the \
                 metadata file's own directory).",
            ));
        }
    }

    if let Some((case, control)) = &contrast {
        let mut labels: Vec<String> = samples.iter().filter_map(|s| s.condition.clone()).collect();
        labels.sort();
        labels.dedup();
        for (wanted, which) in [(case, "case"), (control, "control")] {
            if !labels.iter().any(|l| l == wanted) {
                return Err(PreflightFailure::new(
                    format!(
                        "the {which} condition '{wanted}' is not in the metadata's \
                         condition column. Its labels are: {}.",
                        labels.join(", ")
                    ),
                    "use one of the labels in the metadata, or fix the metadata.",
                ));
            }
        }

        let unknown: Vec<&str> = labels
            .iter()
            .map(|l| l.as_str())
            .filter(|l| *l != case.as_str() && *l != control.as_str())
            .collect();
        if !unknown.is_empty() {
            return Err(PreflightFailure::new(
                format!(
                    "the metadata's condition column names labels that are neither the \
                     case nor the control: {}.",
                    unknown.join(", ")
                ),
                "remove those samples from the metadata, or run one contrast per pair.",
            ));
        }
    }

    if let Some(empty) = samples.iter().find(|s| {
        s.condition
            .as_deref()
            .map(|c| c.trim().is_empty())
            .unwrap_or(true)
    }) {
        return Err(PreflightFailure::new(
            format!(
                "the sample '{}' has no condition in the metadata's condition column.",
                empty.sample_id
            ),
            "give every sample a condition, or remove the row.",
        ));
    }

    let rscript = resolve_rscript(args.rscript.as_deref());
    let r_script = if args.aggregate_only {
        None
    } else {
        if find_executable(&rscript).is_none() {
            return Err(PreflightFailure::new(
                format!("the R interpreter was not found: {}", rscript.display()),
                "install R, or point --rscript at it, or set TELESCOPE_RSCRIPT.",
            ));
        }
        Some(resolve_r_script(args.r_script.as_deref()).map_err(|e| {
            PreflightFailure::new(
                e.to_string(),
                "point --r-script at workflow/scripts/telescope_diff.R, or set \
                 TELESCOPE_DIFF_R to that path.",
            )
        })?)
    };

    let gtf_genes = match &args.gtf {
        Some(p) => {
            if !p.is_file() {
                return Err(PreflightFailure::new(
                    format!("the annotation GTF does not exist: {}", p.display()),
                    "point --gtf at the annotation GTF.",
                ));
            }
            Some(GtfGenes::from_gtf(p).map_err(|e| {
                PreflightFailure::new(
                    format!("the annotation GTF cannot be read: {e:#}"),
                    "check that it is a GTF whose transcript lines carry \
                     `transcript_id`, `gene_id` and `gene_name`.",
                )
            })?)
        }
        None => None,
    };
    let gene_map = gtf_genes.as_ref().map(|g| g.gene_map());
    let gene_names = gtf_genes
        .as_ref()
        .map(|g| g.gene_name_by_gene())
        .unwrap_or_default();

    let needs_genes = modalities.contains(&Modality::Apa);
    if needs_genes {
        let mut genes: HashMap<String, String> = HashMap::new();
        for s in &samples {
            for (tx, gene) in profile_transcript_genes(&s.profile).map_err(|e| {
                PreflightFailure::new(
                    format!(
                        "the profile of the sample '{}' cannot be read: {e}",
                        s.sample_id
                    ),
                    "check the profile file.",
                )
            })? {
                match genes.get(&tx) {
                    None => {
                        genes.insert(tx, gene);
                    }
                    Some(prev) if *prev == gene => {}
                    Some(prev) => {
                        return Err(PreflightFailure::new(
                            format!(
                                "the transcript '{tx}' has two genes in the profiles: \
                                 '{prev}' and '{gene}'."
                            ),
                            "the profiles must agree on the gene of every transcript.",
                        ));
                    }
                }
            }
        }
        resolve_gene_structure(gene_map.as_ref(), &genes, args.gtf.as_deref())?;
    }

    let lapa = if modalities.contains(&Modality::Apa) && args.lapa_dir.is_some() {
        find_executable(&args.lapa_path).map(Some).unwrap_or(None)
    } else if modalities.contains(&Modality::Apa) {
        let driver = find_executable(&args.lapa_path).ok_or_else(|| {
            PreflightFailure::new(
                "LAPA executable not found.",
                "install LAPA or provide --lapa-path.",
            )
        })?;
        let _ = require_file(&args.gtf, "--gtf", "the APA track")?;
        let _ = require_file(&args.fasta, "--fasta", "the APA track")?;
        let _ = require_file(&args.chrom_sizes, "--chrom-sizes", "the APA track")?;
        check_alignments(&samples, "--alignment-column", has_alignment_column)?;
        Some(driver)
    } else {
        None
    };

    let gtf = args.gtf.clone();

    let plan = Plan {
        samples,
        profiles,
        case_condition: contrast.as_ref().map(|c| c.0.clone()),
        control_condition: contrast.as_ref().map(|c| c.1.clone()),
        modalities,
        batch_column,
        batch_state,
        rscript,
        r_script,
        lapa,
        lapa_dir: args.lapa_dir.clone(),

        lapa_params: {
            let base = match args.lapa_dir.as_deref() {
                Some(dir) => lapa_params_from_receipt(dir).unwrap_or_else(LapaParams::reused),
                None => LapaParams::made_here(),
            };
            base.with_overrides(
                args.lapa_counting_method.as_deref(),
                args.lapa_min_replication_rate.as_deref(),
            )
        },
        gtf,
        gene_names,
        fasta: args.fasta.clone(),
        chrom_sizes: args.chrom_sizes.clone(),
    };

    if let Some((n_control, n_case)) = group_sizes(&plan) {
        if n_case < 3 || n_control < 3 {
            return Err(PreflightFailure::new(
                format!(
                    "each condition needs at least 3 samples (case={n_case}, control={n_control})."
                ),
                "add replicates, or compare two groups that each have at least 3 samples.",
            ));
        }
    }

    Ok(plan)
}

/// A file the run cannot do without.
fn require_file(
    path: &Option<PathBuf>,
    flag: &str,
    what: &str,
) -> Result<PathBuf, PreflightFailure> {
    match path {
        Some(p) if p.is_file() => Ok(p.clone()),
        Some(p) => Err(PreflightFailure::new(
            format!("the file {flag} needs does not exist: {}", p.display()),
            format!("{what} needs {flag}."),
        )),
        None => Err(PreflightFailure::new(
            format!("{flag} is required."),
            format!("{what} needs {flag}."),
        )),
    }
}

/// Every sample must carry an alignment for the tools that read one.
fn check_alignments(
    samples: &[Sample],
    flag: &str,
    has_column: bool,
) -> Result<(), PreflightFailure> {
    if !has_column {
        return Err(PreflightFailure::new(
            format!("the metadata has no '{flag}' column, so no sample has an alignment."),
            format!(
                "add the column, or name it with {flag}: the APA track reads one \
                 alignment per sample."
            ),
        ));
    }
    for s in samples {
        match &s.alignment {
            None => {
                return Err(PreflightFailure::new(
                    format!("the sample '{}' has no alignment.", s.sample_id),
                    format!(
                        "the metadata needs an alignment column ({flag}), and every \
                         sample needs a path in it."
                    ),
                ));
            }
            Some(p) if !p.is_file() => {
                return Err(PreflightFailure::new(
                    format!(
                        "the alignment of the sample '{}' does not exist: {}",
                        s.sample_id,
                        p.display()
                    ),
                    "correct the alignment path (a relative path is resolved against the \
                     metadata file's own directory).",
                ));
            }
            Some(_) => {}
        }
    }
    Ok(())
}

/// THE TWO FEATURE NAMES, AND LAPA'S NAMES FOR THE SAME TWO FEATURES.
const UTR_FEATURE_ALIASES: [(&str, &str); 2] =
    [("UTR5", "five_prime_utr"), ("UTR3", "three_prime_utr")];

/// The span of a GTF line's feature column, when the line has one.
fn gtf_feature_span(line: &[u8]) -> Option<(usize, usize)> {
    if line.iter().filter(|b| **b == b'\t').count() < 8 {
        return None;
    }
    let mut field = 0usize;
    let mut start = 0usize;
    for (i, b) in line.iter().enumerate() {
        if *b == b'\t' {
            if field == 2 {
                return Some((start, i));
            }
            field += 1;
            start = i + 1;
        }
    }
    None
}

/// The distinct feature names an annotation uses.
fn feature_names(gtf: &Path) -> Result<BTreeSet<String>> {
    use std::io::BufRead;
    let file =
        std::fs::File::open(gtf).with_context(|| format!("failed to open {}", gtf.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let mut names = BTreeSet::new();
    let mut line = Vec::new();
    loop {
        line.clear();
        let n = reader
            .read_until(b'\n', &mut line)
            .with_context(|| format!("failed to read {}", gtf.display()))?;
        if n == 0 {
            return Ok(names);
        }
        if let Some((s, e)) = gtf_feature_span(&line) {
            names.insert(String::from_utf8_lossy(&line[s..e]).into_owned());
        }
    }
}

/// Write the annotation LAPA is given: a copy of the annotation with the upstream assembler's UTR feature names renamed to the GENCODE ones.
fn write_lapa_annotation(gtf: &Path, path: &Path, renames: &[(&str, &str)]) -> Result<()> {
    use std::io::BufRead;
    let input =
        std::fs::File::open(gtf).with_context(|| format!("failed to open {}", gtf.display()))?;
    let mut reader = std::io::BufReader::new(input);
    let output = std::fs::File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(output);
    let mut line = Vec::new();
    loop {
        line.clear();
        let n = reader
            .read_until(b'\n', &mut line)
            .with_context(|| format!("failed to read {}", gtf.display()))?;
        if n == 0 {
            break;
        }
        let span = gtf_feature_span(&line);
        let replacement = span.and_then(|(s, e)| {
            let feature = String::from_utf8_lossy(&line[s..e]);
            renames
                .iter()
                .find(|(alias, _)| *alias == feature.as_ref())
                .map(|(_, gencode)| gencode.as_bytes())
        });
        match (span, replacement) {
            (Some((s, e)), Some(name)) => {
                writer.write_all(&line[..s])?;
                writer.write_all(name)?;
                writer.write_all(&line[e..])?;
            }
            _ => writer.write_all(&line)?,
        }
    }
    writer
        .flush()
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// The annotation LAPA reads.
#[derive(Debug, Clone)]
enum LapaAnnotation {
    /// LAPA read the annotation the caller gave, unchanged.
    Original(PathBuf),
    /// LAPA read a copy in the delivery directory in which the upstream assembler's UTR feature names were renamed to the GENCODE ones LAPA asserts on.
    Renamed {
        /// The annotation the copy was made from.
        source: PathBuf,
        /// The copy LAPA read.
        copy: PathBuf,
    },
}

impl LapaAnnotation {
    /// The path LAPA is called with.
    fn path(&self) -> &Path {
        match self {
            LapaAnnotation::Original(p) => p,
            LapaAnnotation::Renamed { copy, .. } => copy,
        }
    }

    /// What the receipt says about it.
    fn receipt(&self) -> String {
        match self {
            LapaAnnotation::Original(p) => {
                format!("{} (LAPA read it unchanged)", p.display())
            }
            LapaAnnotation::Renamed { source, .. } => format!(
                "{} (a copy with the upstream assembler's UTR feature names renamed to the \
                 GENCODE ones LAPA asserts on; the copy is delivered as \
                 lapa/annotation_lapa.gtf)",
                source.display()
            ),
        }
    }
}

/// The annotation LAPA reads.
fn lapa_annotation(gtf: &Path, dir: &Path) -> Result<LapaAnnotation> {
    let names = feature_names(gtf)?;
    let renames: Vec<(&str, &str)> = UTR_FEATURE_ALIASES
        .iter()
        .filter(|(alias, gencode)| names.contains(*alias) && !names.contains(*gencode))
        .copied()
        .collect();
    if renames.is_empty() {
        return Ok(LapaAnnotation::Original(gtf.to_path_buf()));
    }
    let path = dir.join("annotation_lapa.gtf");

    if path == gtf {
        bail!(
            "the LAPA annotation would be written over the annotation it is made \
             from ({}). Give the run an annotation that is not the delivery's \
             lapa/annotation_lapa.gtf.",
            gtf.display()
        );
    }
    write_lapa_annotation(gtf, &path, &renames)?;
    eprintln!(
        "[telescope]  LAPA annotation: {} ({})",
        path.display(),
        renames
            .iter()
            .map(|(alias, gencode)| format!("{alias}->{gencode}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(LapaAnnotation::Renamed {
        source: gtf.to_path_buf(),
        copy: path,
    })
}

/// Run LAPA for the APA track, once for the whole cohort.
fn run_lapa(
    driver: &Path,
    params: &LapaParams,
    fasta: &Path,
    gtf: &Path,
    chrom_sizes: &Path,
    samples: &[Sample],
    dir: &Path,
    log: &Path,
    surviving: &Path,
) -> Result<(PathBuf, LapaAnnotation)> {
    std::fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;

    let csv = dir.join("lapa_alignment.csv");
    let mut text = String::from("sample,dataset,path\n");
    for s in samples {
        let alignment = s
            .alignment
            .as_ref()
            .expect("the preflight checked every alignment");
        text.push_str(&format!(
            "{},{},{}\n",
            s.sample_id,
            s.condition.as_deref().unwrap_or("all"),
            alignment.display()
        ));
    }
    std::fs::write(&csv, text).with_context(|| format!("failed to write {}", csv.display()))?;

    let run_dir = dir.join("lapa_run");

    let annotation = lapa_annotation(gtf, dir)?;
    run_tool(
        driver,
        &[
            "--alignment".to_string(),
            csv.display().to_string(),
            "--fasta".to_string(),
            fasta.display().to_string(),
            "--annotation".to_string(),
            annotation.path().display().to_string(),
            "--chrom_sizes".to_string(),
            chrom_sizes.display().to_string(),
            "--output_dir".to_string(),
            run_dir.display().to_string(),
            "--counting_method".to_string(),
            params.counting_method.clone(),
            "--min_replication_rate".to_string(),
            params.min_replication_rate.clone(),
        ],
        "LAPA",
        log,
        surviving,
    )?;

    let canonical = dir.join("canonical");
    let sample_dir = canonical.join("sample");
    std::fs::create_dir_all(&sample_dir)
        .with_context(|| format!("failed to create {}", sample_dir.display()))?;
    let produced = list_with_suffix(&run_dir.join("sample"), "bed");
    for bed in &produced {
        let name = bed
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        std::fs::rename(bed, sample_dir.join(&name)).with_context(|| {
            format!(
                "failed to move {} into {}",
                bed.display(),
                sample_dir.display()
            )
        })?;
    }
    if produced.is_empty() {
        bail!(
            "LAPA wrote no cluster file (*.bed) into {}",
            run_dir.join("sample").display()
        );
    }
    for s in samples {
        let raw = sample_dir.join(format!("{}.bed", s.sample_id));
        let safe = sample_dir.join(format!("{}.bed", sanitize(&s.sample_id)));
        let bed = if raw.is_file() {
            raw
        } else if safe.is_file() {
            safe
        } else {
            bail!(
                "LAPA wrote no cluster file for the sample '{}' (looked for {}).",
                s.sample_id,
                raw.display()
            );
        };
        let _ = bed;
    }
    Ok((canonical, annotation))
}

/// Reuse an existing LAPA output instead of running LAPA again.
fn import_lapa(src: &Path, dir: &Path, samples: &[Sample]) -> Result<PathBuf> {
    if !src.is_dir() {
        bail!(
            "the LAPA output directory to reuse does not exist: {}",
            src.display()
        );
    }

    let canonical_src = src.join("canonical");
    let source_dir = if canonical_src.join("sample").is_dir() {
        canonical_src.join("sample")
    } else {
        src.join("sample")
    };
    let beds = list_with_suffix(&source_dir, "bed");
    if beds.is_empty() {
        bail!(
            "no LAPA cluster file (*.bed) in {} or {}",
            source_dir.display(),
            canonical_src.join("sample").display()
        );
    }

    let canonical = dir.join("canonical");
    let sample_dir = canonical.join("sample");
    std::fs::create_dir_all(&sample_dir)
        .with_context(|| format!("failed to create {}", sample_dir.display()))?;

    for bed in &beds {
        let name = bed
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let target = sample_dir.join(&name);
        std::fs::copy(bed, &target).with_context(|| {
            format!(
                "failed to copy {} into {}",
                bed.display(),
                sample_dir.display()
            )
        })?;
    }

    for s in samples {
        let raw = sample_dir.join(format!("{}.bed", s.sample_id));
        let safe = sample_dir.join(format!("{}.bed", sanitize(&s.sample_id)));
        if !raw.is_file() && !safe.is_file() {
            bail!(
                "the reused LAPA output has no cluster file for the sample '{}' (looked for {}).",
                s.sample_id,
                raw.display()
            );
        }
    }
    eprintln!(
        "[telescope]  reusing {} LAPA cluster files from {}",
        beds.len(),
        source_dir.display()
    );
    Ok(canonical)
}

/// Make a name safe to use as a file name.
fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// The files in `dir` whose name ends with `suffix`, sorted.
fn list_with_suffix(dir: &Path, suffix: &str) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if name.ends_with(suffix) {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// Run a command, keep its output in the log, and fail when it does not exit zero.
fn run_tool(
    program: &Path,
    args: &[String],
    tool: &str,
    log: &Path,
    surviving: &Path,
) -> Result<()> {
    let mut command = Command::new(program);
    command.args(args);
    let rendered = format!(
        "{} {}",
        program.display(),
        args.iter()
            .map(|a| if a.contains(' ') {
                format!("\"{a}\"")
            } else {
                a.clone()
            })
            .collect::<Vec<_>>()
            .join(" ")
    );
    eprintln!("[telescope]  running {rendered}");
    let out = command.output().with_context(|| {
        format!("failed to run {tool}: {rendered}. Is {tool} on PATH and executable?")
    })?;
    append_log(log, &rendered, &out.stdout, &out.stderr)?;
    if !out.status.success() {
        bail!(
            "{tool} failed ({}). Its output is in {}.",
            out.status,
            surviving.display()
        );
    }
    Ok(())
}

/// Append a command's output to a log file.
fn append_log(log: &Path, header: &str, stdout: &[u8], stderr: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .with_context(|| format!("failed to open the log file {}", log.display()))?;
    writeln!(f, "\n=== {header} ===")?;
    f.write_all(stdout)?;
    if !stderr.is_empty() {
        writeln!(f, "--- stderr ---")?;
        f.write_all(stderr)?;
    }
    Ok(())
}

/// A temporary directory that is removed unless the run moves its contents out of it.
struct TempDir {
    path: PathBuf,
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// The name of the temporary directory, unique per run.
fn temp_name(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!(".{prefix}-{}-{nanos}", std::process::id())
}

/// The DPL model's facts, read back from the table the engine wrote.
#[derive(Debug, Clone)]
struct DplFacts {
    /// The name of the model the engine fitted, e.g.
    model: Option<String>,
    /// Whether the model moderated with limma's `robust = TRUE`.
    robust: Option<bool>,
    /// Whether the batch is completely confounded with the condition.
    confounded: Option<bool>,
    /// The engine's own sentence describing the model.
    description: Option<String>,
}

/// Read the DPL facts out of `dpl.tsv`.
fn read_dpl_facts(dpl_tsv: &Path) -> Option<DplFacts> {
    use std::io::BufRead;
    let file = std::fs::File::open(dpl_tsv).ok()?;
    let mut reader = std::io::BufReader::new(file);
    let mut header = String::new();
    reader.read_line(&mut header).ok()?;
    let mut row = String::new();
    reader.read_line(&mut row).ok()?;
    let cols: Vec<&str> = header.trim_end().split('\t').collect();
    let vals: Vec<&str> = row.trim_end_matches(['\n', '\r']).split('\t').collect();
    let get = |name: &str| -> Option<&str> {
        cols.iter()
            .position(|c| *c == name)
            .and_then(|i| vals.get(i).copied())
    };

    const PREFIX: &str = "model: ";
    let clause = get("model_parameters").and_then(|params| {
        params.find(PREFIX).map(|start| {
            let rest = &params[start + PREFIX.len()..];
            match rest.find("); ") {
                Some(end) => rest[..end + 1].trim(),
                None => {
                    let end = rest.find("; ").unwrap_or(rest.len());
                    rest[..end].trim()
                }
            }
        })
    });
    let model = clause.and_then(|c| c.split_whitespace().next().map(|s| s.to_string()));

    let robust = clause.map(|c| c.contains("robust = TRUE"));
    let confounded = get("confounded_design").map(|v| v.eq_ignore_ascii_case("TRUE"));
    Some(DplFacts {
        model,
        robust,
        confounded,
        description: clause.map(|c| c.to_string()),
    })
}

/// Write `run_manifest.json`.
fn write_manifest(
    path: &Path,
    plan: &Plan,
    args: &DiffArgs,
    sample_table: &Path,
    start: &str,
    end: &str,
    versions: &[(String, String)],
    dpl_facts: Option<&DplFacts>,
    lapa_annotation: Option<&LapaAnnotation>,
) -> Result<()> {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!(
        "  \"command_line\": {},\n",
        json_string(&std::env::args().collect::<Vec<_>>().join(" "))
    ));
    json.push_str(&format!("  \"start_time\": {},\n", json_string(start)));
    json.push_str(&format!("  \"end_time\": {},\n", json_string(end)));
    json.push_str(&format!(
        "  \"modalities\": [{}],\n",
        plan.modalities
            .iter()
            .map(|m| json_string(m.as_str()))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    json.push_str(&format!(
        "  \"case_condition\": {},\n",
        json_opt(plan.case_condition.as_deref())
    ));
    json.push_str(&format!(
        "  \"control_condition\": {},\n",
        json_opt(plan.control_condition.as_deref())
    ));
    json.push_str(&format!(
        "  \"batch_column\": {},\n",
        json_opt(plan.batch_column.as_deref())
    ));
    json.push_str(&format!(
        "  \"batch_note\": {},\n",
        json_string(&batch_note(plan))
    ));
    json.push_str("  \"inputs\": {\n");
    json.push_str(&format!(
        "    \"metadata\": {},\n",
        json_string(&args.metadata.display().to_string())
    ));
    json.push_str(&format!(
        "    \"sample_table\": {},\n",
        json_string(&sample_table.display().to_string())
    ));
    json.push_str(&format!(
        "    \"gtf\": {},\n",
        json_opt(
            plan.gtf
                .as_ref()
                .map(|p| p.display().to_string())
                .as_deref()
        )
    ));

    if let Some(annotation) = lapa_annotation {
        json.push_str(&format!(
            "    \"lapa_annotation\": {},\n",
            json_string(&annotation.receipt())
        ));
    }
    json.push_str(&format!(
        "    \"fasta\": {},\n",
        json_opt(
            plan.fasta
                .as_ref()
                .map(|p| p.display().to_string())
                .as_deref()
        )
    ));
    json.push_str(&format!(
        "    \"chrom_sizes\": {},\n",
        json_opt(
            plan.chrom_sizes
                .as_ref()
                .map(|p| p.display().to_string())
                .as_deref()
        )
    ));

    match plan.lapa_dir.as_ref() {
        Some(src) => {
            json.push_str(&format!(
                "    \"lapa_reused_from\": {},\n",
                json_string(&src.display().to_string())
            ));
            json.push_str("    \"lapa_cluster_files\": [\n");
            let beds = list_with_suffix(&src.join("canonical").join("sample"), "bed");
            let beds = if beds.is_empty() {
                list_with_suffix(&src.join("sample"), "bed")
            } else {
                beds
            };
            for (i, bed) in beds.iter().enumerate() {
                let size = std::fs::metadata(bed).map(|m| m.len()).unwrap_or(0);

                let stem = bed
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let used = plan
                    .samples
                    .iter()
                    .any(|s| s.sample_id == stem || sanitize(&s.sample_id) == stem);
                json.push_str(&format!(
                    "      {{\"file\": {}, \"bytes\": {}, \"sample_of_this_run\": {}}}",
                    json_string(&bed.display().to_string()),
                    size,
                    used
                ));
                if i + 1 < beds.len() {
                    json.push(',');
                }
                json.push('\n');
            }
            json.push_str("    ],\n");
        }
        None => {}
    }
    json.push_str("    \"samples\": [\n");
    for (i, s) in plan.samples.iter().enumerate() {
        json.push_str("      {");
        json.push_str(&format!("\"sample_id\": {}, ", json_string(&s.sample_id)));
        json.push_str(&format!(
            "\"profile\": {}, ",
            json_string(&s.profile.display().to_string())
        ));
        json.push_str(&format!(
            "\"condition\": {}, ",
            json_opt(s.condition.as_deref())
        ));
        json.push_str(&format!("\"batch\": {}, ", json_opt(s.batch.as_deref())));
        json.push_str(&format!(
            "\"alignment\": {}",
            json_opt(
                s.alignment
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .as_deref()
            )
        ));
        json.push('}');
        if i + 1 < plan.samples.len() {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str("    ]\n  },\n");
    json.push_str("  \"software_versions\": [\n");
    for (i, (tool, version)) in versions.iter().enumerate() {
        json.push_str(&format!(
            "    {{\"tool\": {}, \"version\": {}}}",
            json_string(tool),
            json_string(version)
        ));
        if i + 1 < versions.len() {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str("  ],\n");

    match std::env::var("GIT_COMMIT") {
        Ok(c) if !c.trim().is_empty() => {
            json.push_str(&format!("  \"git_commit\": {},\n", json_string(c.trim())));
        }
        _ => {}
    }
    json.push_str("  \"min_weight\": ");
    json.push_str(&format!("{:?}", args.min_weight));
    json.push_str(",\n  \"min_neff\": ");
    json.push_str(&format!("{:?}", args.min_neff));
    json.push_str(",\n  \"min_counts\": ");
    json.push_str(&format!("{:?}", args.min_counts));
    json.push_str(",\n  \"min_samples\": ");
    json.push_str(&format!("{}", args.min_samples));

    json.push_str(",\n  \"admission_gate\": \"and\"");
    json.push_str(",\n  \"usage_track_matrix\": \"narrowed\"");
    json.push_str(",\n  \"apa_model\": \"DEXSeq\"");

    json.push_str(",\n  \"apa_min_delta_pau\": ");
    json.push_str(&format!("{:?}", args.apa_min_delta_pau));
    if let Some((n_control, n_case)) = group_sizes(plan) {
        json.push_str(",\n  \"n_control_samples\": ");
        json.push_str(&format!("{}", n_control));
        json.push_str(",\n  \"n_case_samples\": ");
        json.push_str(&format!("{}", n_case));
    }

    let mut dpl_fields: Vec<String> = Vec::new();
    if let Some(facts) = dpl_facts {
        if let Some(model) = &facts.model {
            dpl_fields.push(format!("\"dpl_model\": {}", json_string(model)));
        }
        if let Some(robust) = facts.robust {
            dpl_fields.push(format!("\"dpl_eb_robust\": {}", robust));
        }
    }
    if dpl_facts.is_some() {
        dpl_fields.push(format!("\"dpl_min_weight\": {:?}", args.min_weight));
    }
    if let Some(facts) = dpl_facts {
        if let Some(confounded) = facts.confounded {
            dpl_fields.push(format!("\"confounded_design\": {}", confounded));
        }
    }
    for field in dpl_fields {
        json.push_str(",\n  ");
        json.push_str(&field);
    }
    json.push_str("\n}\n");
    std::fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// The batch note the manifest and the model parameters record.
fn batch_note(plan: &Plan) -> String {
    plan.batch_state.note()
}

/// Write `software_versions.tsv`.
fn write_software_versions(path: &Path, versions: &[(String, String)]) -> Result<()> {
    let mut text = String::from("tool\tversion\n");
    for (tool, version) in versions {
        text.push_str(&format!("{tool}\t{version}\n"));
    }
    std::fs::write(path, text).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// Write `model_parameters.tsv`.
fn write_model_parameters(
    path: &Path,
    plan: &Plan,
    args: &DiffArgs,
    dpl_facts: Option<&DplFacts>,
    lapa_annotation: Option<&LapaAnnotation>,
) -> Result<()> {
    let mut rows: Vec<(String, String, String)> = Vec::new();
    let note = batch_note(plan);
    for m in &plan.modalities {
        let modality = m.as_str().to_string();
        match m {
            Modality::Deg => {
                rows.push((modality.clone(), "model_name".into(), "DESeq2".into()));
            }
            Modality::Det => {
                rows.push((modality.clone(), "model_name".into(), "DESeq2".into()));
                rows.push((
                    modality.clone(),
                    "modality".into(),
                    "transcript-level abundance".into(),
                ));
            }
            Modality::Diu => {
                rows.push((modality.clone(), "model_name".into(), "DEXSeq".into()));
                rows.push((
                    modality.clone(),
                    "modality".into(),
                    "within-gene isoform usage".into(),
                ));
            }
            Modality::Dpl => {
                rows.push((
                    modality.clone(),
                    "model_name".into(),
                    "see the model_name column of dpl.tsv: DT-EB, or \
                     DT-EB-no-trend when the engine fitted no variance trend"
                        .into(),
                ));
                rows.push((
                    modality.clone(),
                    "model".into(),
                    match dpl_facts.and_then(|f| f.description.clone()) {
                        Some(description) => description,
                        None => "the engine's own description could not be read \
                                 from dpl.tsv"
                            .into(),
                    },
                ));
            }
            Modality::Apa => {
                rows.push((
                    modality.clone(),
                    "model_name".into(),
                    "LAPA+DEXSeq (see the model_name column of apa.tsv: it names \
                     `not_testable` when no model could be fitted)"
                        .into(),
                ));
                rows.push((
                    modality.clone(),
                    "fasta".into(),
                    plan.fasta
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                ));
                rows.push((
                    modality.clone(),
                    "chrom_sizes".into(),
                    plan.chrom_sizes
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                ));
                rows.push((
                    modality.clone(),
                    "gtf".into(),
                    plan.gtf
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                ));

                if let Some(annotation) = lapa_annotation {
                    rows.push((
                        modality.clone(),
                        "lapa_annotation".into(),
                        annotation.receipt(),
                    ));
                }

                if let Some(src) = plan.lapa_dir.as_ref() {
                    rows.push((
                        modality.clone(),
                        "lapa_reused_from".into(),
                        src.display().to_string(),
                    ));
                    rows.push((
                        modality.clone(),
                        "lapa_reuse_note".into(),
                        "the poly(A) cluster calls were NOT produced by this run; they \
                         were copied from the directory above into lapa/canonical/sample/, \
                         so the fasta, gtf and chrom_sizes rows above are empty by design \
                         and the inputs of the original LAPA call are recorded in that \
                         run's own manifest"
                            .into(),
                    ));
                }
            }
        }
        rows.push((
            modality.clone(),
            "min_weight".into(),
            format!("{:?}", args.min_weight),
        ));
        rows.push((
            modality.clone(),
            "min_neff".into(),
            format!("{:?}", args.min_neff),
        ));
        rows.push((
            modality.clone(),
            "min_counts".into(),
            format!("{:?}", args.min_counts),
        ));
        rows.push((
            modality.clone(),
            "min_samples".into(),
            format!("{}", args.min_samples),
        ));
        rows.push((
            modality.clone(),
            "case_condition".into(),
            plan.case_condition.clone().unwrap_or_default(),
        ));
        rows.push((
            modality.clone(),
            "control_condition".into(),
            plan.control_condition.clone().unwrap_or_default(),
        ));

        rows.push((modality.clone(), "batch".into(), note.clone()));
    }
    let mut text = String::from("modality\tparameter\tvalue\n");
    for (m, p, v) in rows {
        text.push_str(&format!("{m}\t{p}\t{v}\n"));
    }
    std::fs::write(path, text).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// A JSON string.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A JSON string, or `null`.
fn json_opt(s: Option<&str>) -> String {
    match s {
        Some(v) => json_string(v),
        None => "null".to_string(),
    }
}

/// The version of a tool, read out of its own `--version`.
fn tool_version(program: &Path, args: &[&str]) -> String {
    let out = Command::new(program).args(args).output();
    match out {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            let line = text
                .lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("")
                .trim()
                .to_string();
            if line.is_empty() {
                "unknown".to_string()
            } else {
                line
            }
        }
        _ => "unknown".to_string(),
    }
}

/// The current UTC time as `YYYY-MM-DDTHH:MM:SSZ`.
fn utc_timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// The calendar date of a day count since the Unix epoch.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Run the differential layer.
pub fn run(args: &DiffArgs) -> Result<()> {
    let plan = match preflight(args) {
        Ok(p) => p,
        Err(failure) => {
            eprintln!("{}", failure.message());
            std::process::exit(1);
        }
    };
    execute(args, &plan)
}

/// Do the work the preflight cleared.
fn execute(args: &DiffArgs, plan: &Plan) -> Result<()> {
    let start = utc_timestamp();
    let output_dir = &args.output_dir;
    let parent = match output_dir.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    std::fs::create_dir_all(&parent)
        .with_context(|| format!("failed to create {}", parent.display()))?;

    let tmp = parent.join(temp_name("telescope-diff"));
    let work = tmp.join("work");
    let result = tmp.join("result");
    std::fs::create_dir_all(&work)
        .with_context(|| format!("failed to create {}", work.display()))?;
    std::fs::create_dir_all(&result)
        .with_context(|| format!("failed to create {}", result.display()))?;

    let _temp = TempDir { path: tmp.clone() };

    let logs = result.join("logs");
    std::fs::create_dir_all(&logs)
        .with_context(|| format!("failed to create {}", logs.display()))?;

    let surviving = failure_log_path(output_dir, &parent);

    let outcome = execute_inner(args, plan, &start, &work, &result, &logs, &surviving);
    if let Err(e) = outcome {
        preserve_logs(&logs, &surviving);
        return Err(e);
    }
    Ok(())
}

/// Where the log of a failed run is kept.
fn failure_log_path(output_dir: &Path, parent: &Path) -> PathBuf {
    let name = output_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "telescope-diff".to_string());
    parent.join(format!("{name}.failed.log"))
}

/// Copy the logs of a failed run next to the output directory.
fn preserve_logs(logs: &Path, dest: &Path) {
    let files = list_with_suffix(logs, "log");
    if files.is_empty() {
        return;
    }
    let mut text = String::new();
    for f in &files {
        let name = f
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        text.push_str(&format!("=== {name} ===\n"));
        match std::fs::read(f) {
            Ok(bytes) => text.push_str(&String::from_utf8_lossy(&bytes)),
            Err(e) => text.push_str(&format!("(the log could not be read: {e})\n")),
        }
        text.push('\n');
    }
    if let Err(e) = std::fs::write(dest, text) {
        eprintln!(
            "[telescope]  the log of the failed run could not be written to {}: {e}",
            dest.display()
        );
    }
}

/// The work itself, with the temporary directory already in place.
fn execute_inner(
    args: &DiffArgs,
    plan: &Plan,
    start: &str,
    work: &Path,
    result: &Path,
    logs: &Path,
    surviving: &Path,
) -> Result<()> {
    let output_dir = &args.output_dir;

    eprintln!(
        "[telescope]  modalities: {}",
        if plan.modalities.is_empty() {
            "none (--aggregate-only)".to_string()
        } else {
            plan.modalities
                .iter()
                .map(|m| m.as_str())
                .collect::<Vec<_>>()
                .join(",")
        }
    );

    let rows = aggregate_samples(&plan.samples)?;
    let agg_path = work.join("telescope_diff_aggregate.tsv");
    write_aggregate_tsv(&agg_path, &rows)?;
    eprintln!(
        "[telescope]  aggregate: {} (transcript x sample) cells -> {}",
        rows.len(),
        agg_path.display()
    );

    let gene_names_path = work.join("telescope_diff_gene_names.tsv");
    let gene_names = if plan.gene_names.is_empty() {
        None
    } else {
        write_gene_names_tsv(&gene_names_path, &plan.gene_names)?;
        eprintln!(
            "[telescope]  gene names: {} -> {}",
            plan.gene_names.len(),
            gene_names_path.display()
        );
        Some(gene_names_path.clone())
    };

    let metadata_path = work.join("telescope_diff_metadata.tsv");
    let with_batch = plan.batch_column.is_some();
    write_metadata_tsv(&metadata_path, &plan.samples, with_batch)?;
    eprintln!("[telescope]  sample table: {}", metadata_path.display());

    let mut lapa_dir: Option<PathBuf> = None;

    let mut lapa_annotation: Option<LapaAnnotation> = None;
    if plan.modalities.contains(&Modality::Apa) {
        let (path, annotation) = match plan.lapa_dir.as_ref() {
            Some(src) => (import_lapa(src, &result.join("lapa"), &plan.samples)?, None),
            None => {
                let driver = plan.lapa.as_ref().expect("the preflight found LAPA");
                let gtf = plan.gtf.as_ref().expect("the preflight found the GTF");
                let fasta = plan.fasta.as_ref().expect("the preflight found the FASTA");
                let chrom_sizes = plan
                    .chrom_sizes
                    .as_ref()
                    .expect("the preflight found the chromosome sizes");
                let (canonical, annotation) = run_lapa(
                    driver,
                    &plan.lapa_params,
                    fasta,
                    gtf,
                    chrom_sizes,
                    &plan.samples,
                    &result.join("lapa"),
                    &logs.join("lapa.log"),
                    surviving,
                )?;
                (canonical, Some(annotation))
            }
        };

        lapa_annotation = annotation;

        let lapa_out = result.join("lapa");
        write_lapa_params_receipt(&lapa_out, &plan.lapa_params)?;
        write_lapa_params_receipt(&lapa_out.join("canonical"), &plan.lapa_params)?;
        eprintln!("[telescope]  LAPA output directory: {}", path.display());
        lapa_dir = Some(path);
    }

    if args.aggregate_only {
        std::fs::copy(&agg_path, result.join("telescope_diff_aggregate.tsv")).with_context(
            || {
                format!(
                    "failed to copy {} into the result directory",
                    agg_path.display()
                )
            },
        )?;
        eprintln!("[telescope]  --aggregate-only: not running the statistical engine");
    } else {
        run_engine(
            args,
            plan,
            &agg_path,
            &metadata_path,
            gene_names.as_deref(),
            result,
            lapa_dir.as_deref(),
            surviving,
        )?;

        for m in &plan.modalities {
            let table = result.join(m.table());
            if !table.is_file() {
                bail!(
                    "the R engine wrote no {} for the requested modality '{}'. Its \
                     output is in {}.",
                    m.table(),
                    m.as_str(),
                    surviving.display()
                );
            }
        }
    }

    let versions = collect_versions(args, plan);

    let dpl_facts = if plan.modalities.contains(&Modality::Dpl) {
        let table = result.join(Modality::Dpl.table());
        match read_dpl_facts(&table) {
            Some(facts) => {
                if facts.model.is_none() {
                    eprintln!(
                        "[telescope]  WARNING: the DPL model's own description could not be \
                         read from {}; the manifest omits the DPL model fields and the \
                         receipt says so.",
                        table.display()
                    );
                }
                Some(facts)
            }
            None => {
                eprintln!(
                    "[telescope]  WARNING: the DPL table {} could not be read; the manifest \
                     omits the DPL model fields and the receipt says so.",
                    table.display()
                );
                None
            }
        }
    } else {
        None
    };
    write_software_versions(&result.join("software_versions.tsv"), &versions)?;
    write_model_parameters(
        &result.join("model_parameters.tsv"),
        plan,
        args,
        dpl_facts.as_ref(),
        lapa_annotation.as_ref(),
    )?;

    let copy = logs.join("metadata.tsv");
    std::fs::copy(&metadata_path, &copy).with_context(|| {
        format!(
            "failed to copy the sample table {} into {}",
            metadata_path.display(),
            copy.display()
        )
    })?;
    let manifest_sample_table = PathBuf::from("logs/metadata.tsv");
    let end = utc_timestamp();
    write_manifest(
        &result.join("run_manifest.json"),
        plan,
        args,
        &manifest_sample_table,
        start,
        &end,
        &versions,
        dpl_facts.as_ref(),
        lapa_annotation.as_ref(),
    )?;

    let session = result.join("sessionInfo.txt");
    if session.is_file() {
        let _ = std::fs::rename(&session, logs.join("sessionInfo.txt"));
    }

    publish(&result, output_dir)?;
    eprintln!(
        "[telescope]  wrote {} into {}",
        if plan.modalities.is_empty() {
            "the aggregate table".to_string()
        } else {
            plan.modalities
                .iter()
                .map(|m| m.table())
                .collect::<Vec<_>>()
                .join(", ")
        },
        output_dir.display()
    );
    Ok(())
}

/// The engine's modality flags: the tables this run asked for, and the ones it did not.
fn engine_modality_flags(modalities: &[Modality]) -> Vec<String> {
    let mut flags = Vec::new();
    for (m, flag) in [
        (Modality::Deg, "--deg"),
        (Modality::Det, "--det"),
        (Modality::Diu, "--diu"),
        (Modality::Dpl, "--dpl"),
    ] {
        flags.push(flag.to_string());
        if !modalities.contains(&m) {
            flags.push("false".to_string());
        }
    }
    flags
}

/// Run the R engine on the aggregate.
fn run_engine(
    args: &DiffArgs,
    plan: &Plan,
    aggregate: &Path,
    metadata: &Path,
    gene_names: Option<&Path>,
    out_dir: &Path,
    lapa_dir: Option<&Path>,
    surviving: &Path,
) -> Result<()> {
    let mut engine_args: Vec<String> = vec![
        plan.r_script
            .as_ref()
            .expect("the preflight resolved the R engine")
            .display()
            .to_string(),
        "--aggregate".to_string(),
        aggregate.display().to_string(),
        "--metadata".to_string(),
        metadata.display().to_string(),
        "--output-dir".to_string(),
        out_dir.display().to_string(),
        "--min-cell-weight".to_string(),
        args.min_weight.to_string(),
        "--min-neff".to_string(),
        args.min_neff.to_string(),
        "--min-counts".to_string(),
        args.min_counts.to_string(),
        "--min-samples".to_string(),
        args.min_samples.to_string(),
        "--apa-min-delta-pau".to_string(),
        args.apa_min_delta_pau.to_string(),
        "--lapa-counting-method".to_string(),
        plan.lapa_params.counting_method.clone(),
        "--lapa-min-replication-rate".to_string(),
        plan.lapa_params.min_replication_rate.clone(),
    ];
    if let (Some(case), Some(control)) = (&plan.case_condition, &plan.control_condition) {
        engine_args.push("--case-condition".to_string());
        engine_args.push(case.clone());
        engine_args.push("--control-condition".to_string());
        engine_args.push(control.clone());
    }

    engine_args.extend(engine_modality_flags(&plan.modalities));
    if let Some(d) = lapa_dir {
        engine_args.push("--lapa-dir".to_string());
        engine_args.push(d.display().to_string());
    }

    if let Some(g) = gene_names {
        engine_args.push("--gene-names".to_string());
        engine_args.push(g.display().to_string());
    }
    let log = out_dir.join("logs").join("telescope_diff.R.log");
    run_tool(&plan.rscript, &engine_args, "the R engine", &log, surviving)?;
    Ok(())
}

/// The versions of the tools that ran.
fn lapa_version(lapa: &Path) -> String {
    let Some(dir) = lapa.parent() else {
        return "unknown (the LAPA path has no parent directory)".to_string();
    };
    for name in ["python", "python3"] {
        let py = dir.join(name);
        if !py.is_file() {
            continue;
        }
        let out = std::process::Command::new(&py)
            .args([
                "-c",
                "import importlib.metadata as m; print(m.version('lapa'))",
            ])
            .output();
        if let Ok(o) = out {
            if o.status.success() {
                let v = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if !v.is_empty() {
                    return v;
                }
            }
        }
    }
    "unknown (the LAPA distribution metadata could not be read)".to_string()
}

fn collect_versions(args: &DiffArgs, plan: &Plan) -> Vec<(String, String)> {
    let mut versions: Vec<(String, String)> = Vec::new();
    versions.push((
        "telescope".to_string(),
        env!("CARGO_PKG_VERSION").to_string(),
    ));
    if plan.r_script.is_some() {
        versions.push(("R".to_string(), tool_version(&plan.rscript, &["--version"])));
    }
    if let Some(d) = &plan.lapa {
        versions.push(("LAPA".to_string(), lapa_version(d)));
    }
    let _ = args;
    versions
}

/// Move the finished result directory into place.
fn publish(result: &Path, output_dir: &Path) -> Result<()> {
    let parent = match output_dir.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    if output_dir.exists() {
        let old = parent.join(temp_name("telescope-diff-old"));
        std::fs::rename(output_dir, &old)
            .with_context(|| format!("failed to move {} aside", output_dir.display()))?;
        match std::fs::rename(result, output_dir) {
            Ok(()) => {
                let _ = std::fs::remove_dir_all(&old);
            }
            Err(e) => {
                let _ = std::fs::rename(&old, output_dir);
                return Err(e).with_context(|| {
                    format!(
                        "failed to move {} into {}",
                        result.display(),
                        output_dir.display()
                    )
                });
            }
        }
    } else {
        std::fs::rename(result, output_dir).with_context(|| {
            format!(
                "failed to move {} into {}",
                result.display(),
                output_dir.display()
            )
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow::array::{Float64Builder, Int32Builder, StringBuilder};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    /// Build a profile parquet with the published schema and the given rows.
    fn write_profile(path: &Path, rows: &[(&str, &str, &str, i32, f64)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("sample_id", DataType::Utf8, false),
            Field::new("gene_id", DataType::Utf8, false),
            Field::new("transcript_id", DataType::Utf8, false),
            Field::new("polyA_length", DataType::Int32, false),
            Field::new("posterior_weight", DataType::Float64, false),
        ]));
        let mut s = StringBuilder::new();
        let mut g = StringBuilder::new();
        let mut t = StringBuilder::new();
        let mut p = Int32Builder::new();
        let mut w = Float64Builder::new();
        for (sample, gene, tx, pal, pw) in rows {
            s.append_value(sample);
            g.append_value(gene);
            t.append_value(tx);
            p.append_value(*pal);
            w.append_value(*pw);
        }
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(s.finish()),
                Arc::new(g.finish()),
                Arc::new(t.finish()),
                Arc::new(p.finish()),
                Arc::new(w.finish()),
            ],
        )
        .expect("batch");
        let file = File::create(path).expect("create");
        let mut w = ArrowWriter::try_new(file, schema, None).expect("writer");
        w.write(&batch).expect("write");
        w.close().expect("close");
    }

    fn tmp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("telescope_diff_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    #[test]
    fn aggregation_sums_the_posterior_mass_and_the_weighted_tail_length() {
        let dir = tmp_dir("agg");
        let path = dir.join("S1.profile.parquet");
        write_profile(
            &path,
            &[
                ("S1", "G1", "T1", 100, 0.5),
                ("S1", "G1", "T1", 200, 0.25),
                ("S1", "G1", "T2", 50, 1.0),
            ],
        );
        let rows = aggregate_profile(&path).expect("aggregate");
        assert_eq!(rows.len(), 2);

        let t1 = rows.iter().find(|r| r.transcript_id == "T1").unwrap();
        assert!((t1.weight_sum - 0.75).abs() < 1e-12);
        let num = 0.5 * 100f64.log2() + 0.25 * 200f64.log2();
        assert!((t1.weighted_log2pal_sum - num).abs() < 1e-12);
        let mean = t1.weighted_log2pal_sum / t1.weight_sum;
        assert!((mean - num / 0.75).abs() < 1e-12);

        let t2 = rows.iter().find(|r| r.transcript_id == "T2").unwrap();
        assert!((t2.weight_sum - 1.0).abs() < 1e-12);
        assert!((t2.weighted_log2pal_sum - 50f64.log2()).abs() < 1e-12);

        let w2 = 0.5f64 * 0.5 + 0.25 * 0.25;
        assert!((t1.weight_sq_sum - w2).abs() < 1e-12);
        let l2 = 0.5 * 100f64.log2().powi(2) + 0.25 * 200f64.log2().powi(2);
        assert!((t1.weighted_log2pal_sq_sum - l2).abs() < 1e-12);
        let n_eff = t1.weight_sum * t1.weight_sum / t1.weight_sq_sum;
        assert!((n_eff - 0.75 * 0.75 / w2).abs() < 1e-12);
        assert!(n_eff > 1.0 && n_eff <= 2.0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_zero_tail_length_is_floored_at_one_rather_than_producing_minus_infinity() {
        let dir = tmp_dir("pal");
        let path = dir.join("S1.profile.parquet");
        write_profile(&path, &[("S1", "G1", "T1", 0, 1.0)]);
        let rows = aggregate_profile(&path).expect("aggregate");
        assert_eq!(rows.len(), 1);

        assert!(rows[0].weighted_log2pal_sum.is_finite());
        assert!((rows[0].weighted_log2pal_sum - 0.0).abs() < 1e-12);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_aggregate_table_round_trips_the_floats_exactly() {
        let dir = tmp_dir("tsv");
        let rows = vec![AggRow {
            sample_id: "S1".into(),
            gene_id: "G1".into(),
            transcript_id: "T1".into(),
            weight_sum: 0.1 + 0.2,
            weighted_log2pal_sum: 1.0 / 3.0,
            weight_sq_sum: 0.05,
            weighted_log2pal_sq_sum: 1.0 / 7.0,
        }];
        let path = dir.join("agg.tsv");
        write_aggregate_tsv(&path, &rows).expect("write");
        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines = text.lines();
        assert_eq!(
            lines.next().unwrap(),
            "sample_id\tgene_id\ttranscript_id\tweight_sum\tweighted_log2pal_sum\tweight_sq_sum\tweighted_log2pal_sq_sum"
        );
        let f: Vec<&str> = lines.next().unwrap().split('\t').collect();
        assert_eq!(&f[..3], &["S1", "G1", "T1"]);

        assert_eq!(f[3].parse::<f64>().unwrap(), 0.1 + 0.2);
        assert_eq!(f[4].parse::<f64>().unwrap(), 1.0 / 3.0);
        assert_eq!(f[5].parse::<f64>().unwrap(), 0.05);
        assert_eq!(f[6].parse::<f64>().unwrap(), 1.0 / 7.0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_contrast_is_taken_from_the_flags_verbatim_and_not_from_the_label_order() {
        let (case, control) = resolve_contrast(Some("CASE"), Some("CTRL"), None).expect("contrast");
        assert_eq!(case, "CASE");
        assert_eq!(control, "CTRL");

        let (case, control) = resolve_contrast(Some("CTRL"), Some("CASE"), None).expect("contrast");
        assert_eq!(case, "CTRL");
        assert_eq!(control, "CASE");
    }

    #[test]
    fn the_contrast_can_be_given_as_case_colon_control() {
        let (case, control) = resolve_contrast(None, None, Some("CASE:CTRL")).expect("contrast");
        assert_eq!(case, "CASE");
        assert_eq!(control, "CTRL");
    }

    #[test]
    fn a_missing_or_ambiguous_contrast_is_refused_before_anything_is_computed() {
        assert!(resolve_contrast(None, None, None).is_err());
        assert!(resolve_contrast(Some("CASE"), None, None).is_err());
        assert!(resolve_contrast(None, Some("CTRL"), None).is_err());
        assert!(resolve_contrast(Some("CASE"), Some("CTRL"), Some("CASE:CTRL")).is_err());
        assert!(resolve_contrast(None, None, Some("CASE")).is_err());
        assert!(resolve_contrast(None, None, Some("CASE:")).is_err());
        assert!(resolve_contrast(Some("CTRL"), Some("CTRL"), None).is_err());
    }

    #[test]
    fn a_named_engine_that_does_not_exist_is_reported_as_missing_not_as_a_failed_run() {
        let err = resolve_r_script(Some(Path::new("/nonexistent/telescope_diff.R")))
            .expect_err("must fail");
        assert!(
            err.to_string().contains("does not exist"),
            "unhelpful error: {err}"
        );
    }

    #[test]
    fn the_engine_is_found_next_to_the_binary_or_in_the_conda_data_directory() {
        if let Ok(p) = resolve_r_script(None) {
            assert!(p.is_file(), "resolved to a missing file: {}", p.display());
        }
    }

    /// The metadata declares the samples, and a relative profile path is resolved against the metadata file's own directory.
    #[test]
    fn the_metadata_declares_the_samples_and_resolves_relative_paths() {
        let dir = tmp_dir("meta");
        std::fs::create_dir_all(dir.join("profiles")).expect("mkdir");
        let meta = dir.join("metadata.tsv");
        std::fs::write(
            &meta,
            "sample_id\tprofile_path\tcondition\tbatch\talignment_path\n\
             S1\tprofiles/S1.parquet\tCTRL\tb1\taln/S1.bam\n\
             S2\tprofiles/S2.parquet\tCASE\tb2\taln/S2.bam\n",
        )
        .expect("write");
        let args = DiffArgs {
            metadata: meta.clone(),
            sample_column: DEFAULT_SAMPLE_COLUMN.into(),
            profile_column: DEFAULT_PROFILE_COLUMN.into(),
            condition_column: DEFAULT_CONDITION_COLUMN.into(),
            batch_column: DEFAULT_BATCH_COLUMN.into(),
            alignment_column: DEFAULT_ALIGNMENT_COLUMN.into(),
            gtf: None,
            fasta: None,
            chrom_sizes: None,
            output_dir: dir.join("out"),
            lapa_path: PathBuf::from(DEFAULT_LAPA),
            lapa_dir: None,
            lapa_counting_method: None,
            lapa_min_replication_rate: None,
            modalities: None,
            min_weight: MIN_CELL_WEIGHT_DEFAULT,
            min_neff: MIN_NEFF_DEFAULT,
            min_counts: MIN_COUNTS_DEFAULT,
            min_samples: MIN_SAMPLES_DEFAULT,
            apa_min_delta_pau: APA_MIN_DELTA_PAU_DEFAULT,
            rscript: None,
            r_script: None,
            aggregate_only: false,
            case_condition: None,
            control_condition: None,
            contrast: None,
        };
        let table = read_metadata(&args, &meta).expect("metadata");
        assert_eq!(table.samples.len(), 2);
        assert_eq!(table.samples[0].sample_id, "S1");
        assert_eq!(
            table.samples[0].profile,
            dir.join("profiles/S1.parquet"),
            "a relative profile path is resolved against the metadata's directory"
        );
        assert_eq!(
            table.samples[1].alignment.as_deref(),
            Some(dir.join("aln/S2.bam").as_path())
        );
        assert!(table.has_batch_column);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A metadata without the named column is refused, and the refusal says which column is missing.
    #[test]
    fn a_metadata_without_the_named_column_is_refused() {
        let dir = tmp_dir("meta_bad");
        let meta = dir.join("metadata.tsv");
        std::fs::write(&meta, "sample_id\tcondition\nS1\tCTRL\n").expect("write");
        let args = DiffArgs {
            metadata: meta.clone(),
            sample_column: DEFAULT_SAMPLE_COLUMN.into(),
            profile_column: DEFAULT_PROFILE_COLUMN.into(),
            condition_column: DEFAULT_CONDITION_COLUMN.into(),
            batch_column: DEFAULT_BATCH_COLUMN.into(),
            alignment_column: DEFAULT_ALIGNMENT_COLUMN.into(),
            gtf: None,
            fasta: None,
            chrom_sizes: None,
            output_dir: dir.join("out"),
            lapa_path: PathBuf::from(DEFAULT_LAPA),
            lapa_dir: None,
            lapa_counting_method: None,
            lapa_min_replication_rate: None,
            modalities: None,
            min_weight: MIN_CELL_WEIGHT_DEFAULT,
            min_neff: MIN_NEFF_DEFAULT,
            min_counts: MIN_COUNTS_DEFAULT,
            min_samples: MIN_SAMPLES_DEFAULT,
            apa_min_delta_pau: APA_MIN_DELTA_PAU_DEFAULT,
            rscript: None,
            r_script: None,
            aggregate_only: false,
            case_condition: None,
            control_condition: None,
            contrast: None,
        };
        let err = read_metadata(&args, &meta).expect_err("must fail");
        assert!(err.what.contains("profile_path"), "got: {}", err.what);
        assert!(err.fix.contains("--profile-column"), "got: {}", err.fix);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The modality list is parsed, and an unknown name is refused.
    #[test]
    fn the_modality_list_is_parsed_and_unknown_names_are_refused() {
        let all = resolve_modalities(None).expect("default");
        assert_eq!(all.len(), 5);
        assert_eq!(all[0], Modality::Deg);
        assert_eq!(all[1], Modality::Det);
        assert_eq!(all[2], Modality::Diu);
        assert_eq!(all[3], Modality::Apa);
        assert_eq!(all[4], Modality::Dpl);
        let some = resolve_modalities(Some("dpl,deg")).expect("named");
        assert_eq!(some, vec![Modality::Deg, Modality::Dpl]);
        assert!(resolve_modalities(Some("deg,nonsense")).is_err());

        assert_eq!(
            resolve_modalities(Some("diu")).expect("named"),
            vec![Modality::Diu]
        );
        assert!(resolve_modalities(Some("")).is_err());
    }

    /// The three messages the contract fixes are reproduced verbatim.
    #[test]
    fn the_fixed_preflight_messages_are_reproduced_verbatim() {
        let lapa = PreflightFailure::new(
            "LAPA executable not found.",
            "install LAPA or provide --lapa-path.",
        );
        assert_eq!(
            lapa.message(),
            "ERROR: LAPA executable not found.\n\
             Fix: install LAPA or provide --lapa-path.\n\
             No partial result directory was created."
        );
        let genes = PreflightFailure::new(
            "transcript-to-gene structure is absent. APA is not identifiable for this \
             input. Provide --gtf or a profile containing valid gene_id values.",
            "provide --gtf, or use profiles whose gene_id column holds the real genes.",
        );
        assert!(genes.message().starts_with(
            "ERROR: transcript-to-gene structure is absent. APA is not identifiable \
             for this input. Provide --gtf or a profile containing valid gene_id \
             values."
        ));
        assert!(genes
            .message()
            .ends_with("No partial result directory was created."));
    }

    /// A transcriptome whose gene is its own transcript id has no gene structure, and the run says so with the contract's message.
    #[test]
    fn a_transcriptome_without_a_gene_structure_is_refused() {
        let mut genes = HashMap::new();
        genes.insert("T1".to_string(), "T1".to_string());
        genes.insert("T2".to_string(), "T2".to_string());
        let err = resolve_gene_structure(None, &genes, None).expect_err("must fail");
        assert!(
            err.what
                .starts_with("transcript-to-gene structure is absent."),
            "got: {}",
            err.what
        );
    }

    /// A transcript the annotation does not cover is named.
    #[test]
    fn a_transcript_missing_from_the_gene_map_is_named() {
        let mut genes = HashMap::new();
        genes.insert("T1".to_string(), "G1".to_string());
        genes.insert("T9".to_string(), "G9".to_string());
        let mut map = HashMap::new();
        map.insert("T1".to_string(), "G1".to_string());
        let err = resolve_gene_structure(Some(&map), &genes, Some(Path::new("a.gtf")))
            .expect_err("must fail");
        assert!(err.what.contains("'T9'"), "got: {}", err.what);
        assert!(err.fix.contains("--gtf"), "got: {}", err.fix);
    }

    /// A transcriptome with real genes passes.
    #[test]
    fn a_transcriptome_with_real_genes_passes_the_check() {
        let mut genes = HashMap::new();
        genes.insert("T1".to_string(), "G1".to_string());
        genes.insert("T2".to_string(), "G1".to_string());
        resolve_gene_structure(None, &genes, None).expect("must pass");
    }

    /// The timestamp is ISO 8601 in UTC.
    #[test]
    fn the_timestamp_is_iso8601_in_utc() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
        let now = utc_timestamp();
        assert_eq!(now.len(), 20, "not an ISO timestamp: {now}");
        assert!(now.ends_with('Z'), "not UTC: {now}");
        assert_eq!(&now[4..5], "-");
        assert_eq!(&now[10..11], "T");
    }

    /// The engine is told exactly which tables to write.
    #[test]
    fn the_engine_is_named_for_exactly_the_tables_the_run_asked_for() {
        assert_eq!(
            engine_modality_flags(&[Modality::Deg, Modality::Dpl]),
            vec!["--deg", "--det", "false", "--diu", "false", "--dpl"]
        );
        assert_eq!(
            engine_modality_flags(&[]),
            vec!["--deg", "false", "--det", "false", "--diu", "false", "--dpl", "false"]
        );
        assert_eq!(
            engine_modality_flags(&[Modality::Deg, Modality::Det, Modality::Diu, Modality::Dpl]),
            vec!["--deg", "--det", "--diu", "--dpl"]
        );
    }

    /// The log of a failed run outlives the run, and it lives next to the output directory, never inside it.
    #[test]
    fn the_log_of_a_failed_run_survives_next_to_the_output_directory() {
        let dir = tmp_dir("failed_log");
        let output_dir = dir.join("results");
        let parent = dir.clone();
        let surviving = failure_log_path(&output_dir, &parent);
        assert_eq!(surviving, dir.join("results.failed.log"));
        assert!(
            !surviving.starts_with(&output_dir),
            "the log must not be inside the output directory: {}",
            surviving.display()
        );

        let logs = dir.join("logs");
        std::fs::create_dir_all(&logs).expect("mkdir");
        std::fs::write(logs.join("telescope_diff.R.log"), "the engine said no\n").expect("write");
        preserve_logs(&logs, &surviving);
        let text = std::fs::read_to_string(&surviving).expect("the log was kept");
        assert!(text.contains("the engine said no"), "{text}");
        assert!(text.contains("telescope_diff.R.log"), "{text}");

        let empty = dir.join("empty");
        std::fs::create_dir_all(&empty).expect("mkdir");
        let dest = dir.join("nothing.failed.log");
        preserve_logs(&empty, &dest);
        assert!(
            !dest.exists(),
            "a run with no log must not leave a file behind"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The aggregate-only mode computes no statistic, so it needs neither an R interpreter nor the engine.
    #[test]
    fn the_aggregate_only_mode_needs_neither_r_nor_the_engine() {
        let dir = tmp_dir("agg_only");
        let meta = dir.join("metadata.tsv");
        std::fs::write(
            &meta,
            "sample_id\tprofile_path\tcondition\n\
             S1\tS1.parquet\tCTRL\n\
             S2\tS2.parquet\tCASE\n",
        )
        .expect("write");
        write_profile(
            &dir.join("S1.parquet"),
            &[("S1", "G1", "T1", 100, 1.0), ("S1", "G1", "T2", 120, 1.0)],
        );
        write_profile(
            &dir.join("S2.parquet"),
            &[("S2", "G1", "T1", 100, 1.0), ("S2", "G1", "T2", 120, 1.0)],
        );
        let args = DiffArgs {
            metadata: meta.clone(),
            sample_column: DEFAULT_SAMPLE_COLUMN.into(),
            profile_column: DEFAULT_PROFILE_COLUMN.into(),
            condition_column: DEFAULT_CONDITION_COLUMN.into(),
            batch_column: DEFAULT_BATCH_COLUMN.into(),
            alignment_column: DEFAULT_ALIGNMENT_COLUMN.into(),
            gtf: None,
            fasta: None,
            chrom_sizes: None,
            output_dir: dir.join("out"),
            lapa_path: PathBuf::from(DEFAULT_LAPA),
            lapa_dir: None,
            lapa_counting_method: None,
            lapa_min_replication_rate: None,
            modalities: None,
            min_weight: MIN_CELL_WEIGHT_DEFAULT,
            min_neff: MIN_NEFF_DEFAULT,
            min_counts: MIN_COUNTS_DEFAULT,
            min_samples: MIN_SAMPLES_DEFAULT,
            apa_min_delta_pau: APA_MIN_DELTA_PAU_DEFAULT,
            rscript: Some(PathBuf::from("/nonexistent/Rscript")),
            r_script: Some(PathBuf::from("/nonexistent/telescope_diff.R")),
            aggregate_only: true,
            case_condition: None,
            control_condition: None,
            contrast: None,
        };
        let plan = preflight(&args).expect("the aggregate-only mode needs no R");
        assert!(plan.r_script.is_none(), "no engine is needed");
        assert!(plan.modalities.is_empty(), "no statistic is computed");

        let mut with_statistic = args;
        with_statistic.aggregate_only = false;
        with_statistic.case_condition = Some("CASE".to_string());
        with_statistic.control_condition = Some("CTRL".to_string());
        let failure = preflight(&with_statistic).expect_err("R is needed here");
        assert!(
            failure
                .message()
                .contains("the R interpreter was not found"),
            "{}",
            failure.message()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A `DiffArgs` for the metadata interface, with every default filled in.
    fn metadata_args(dir: &Path, meta: &Path) -> DiffArgs {
        DiffArgs {
            metadata: meta.to_path_buf(),
            sample_column: DEFAULT_SAMPLE_COLUMN.into(),
            profile_column: DEFAULT_PROFILE_COLUMN.into(),
            condition_column: DEFAULT_CONDITION_COLUMN.into(),
            batch_column: DEFAULT_BATCH_COLUMN.into(),
            alignment_column: DEFAULT_ALIGNMENT_COLUMN.into(),
            gtf: None,
            fasta: None,
            chrom_sizes: None,
            output_dir: dir.join("out"),
            lapa_path: PathBuf::from(DEFAULT_LAPA),
            lapa_dir: None,
            lapa_counting_method: None,
            lapa_min_replication_rate: None,
            modalities: None,
            min_weight: MIN_CELL_WEIGHT_DEFAULT,
            min_neff: MIN_NEFF_DEFAULT,
            min_counts: MIN_COUNTS_DEFAULT,
            min_samples: MIN_SAMPLES_DEFAULT,
            apa_min_delta_pau: APA_MIN_DELTA_PAU_DEFAULT,
            rscript: None,
            r_script: None,
            aggregate_only: false,
            case_condition: None,
            control_condition: None,
            contrast: None,
        }
    }

    /// A metadata table with three samples and the given condition labels, and the three profile files it points at.
    fn write_metadata_with_conditions(dir: &Path, conditions: &[&str]) -> PathBuf {
        let profiles = dir.join("profiles");
        std::fs::create_dir_all(&profiles).expect("mkdir");
        let mut text = String::from("sample_id\tprofile_path\tcondition\n");
        for (i, condition) in conditions.iter().enumerate() {
            let sample = format!("S{}", i + 1);
            std::fs::write(profiles.join(format!("{sample}.parquet")), b"").expect("write");
            text.push_str(&format!(
                "{sample}\tprofiles/{sample}.parquet\t{condition}\n"
            ));
        }
        let meta = dir.join("metadata.tsv");
        std::fs::write(&meta, text).expect("write");
        meta
    }

    /// A profile parquet that declares a sample name, and the metadata that calls the same file something else, must not be published: the rows would be filed under the metadata's name, which is not this sample's.
    #[test]
    fn a_profile_that_declares_another_sample_is_refused() {
        let dir = tmp_dir("sampleid");
        let path = dir.join("S1.profile.parquet");
        write_profile(&path, &[("S2", "G1", "T1", 100, 1.0)]);
        let err = aggregate_profile_impl(&path, Some("S1"))
            .expect_err("a profile that declares another sample must be refused");
        let text = format!("{err:#}");
        assert!(text.contains("declares the sample 'S2'"), "{text}");
        assert!(text.contains("the metadata calls it 'S1'"), "{text}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The same profile with the same name on both sides is published.
    #[test]
    fn a_profile_that_declares_the_same_sample_is_accepted() {
        let dir = tmp_dir("sampleid_ok");
        let path = dir.join("S1.profile.parquet");
        write_profile(&path, &[("S1", "G1", "T1", 100, 1.0)]);
        let rows = aggregate_profile_impl(&path, Some("S1")).expect("aggregate");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].sample_id, "S1");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The batch note has to tell the three states apart: no column at all, a column some sample leaves empty, and a column everyone fills in.
    #[test]
    fn the_batch_note_says_which_state_the_column_is_in() {
        assert_eq!(BatchState::NoColumn.note(), NO_BATCH_NOTE);
        let holes = BatchState::ColumnWithMissingValues("batch".into()).note();
        assert!(
            holes.contains("at least one sample leaves it empty"),
            "a column with holes is not 'no batch column': {holes}"
        );
        assert!(!holes.contains(NO_BATCH_NOTE), "{holes}");
        let present = BatchState::ColumnPresent("batch".into()).note();
        assert!(
            present.contains("a batch column is in the metadata"),
            "{present}"
        );
        assert!(
            present.contains("decided by the engine"),
            "the CLI cannot know whether the engine fitted the batch: {present}"
        );
    }

    /// A third condition label has no place in a two-group contrast, and the run has to say so BEFORE it aggregates anything.
    #[test]
    fn a_third_condition_label_is_refused_before_anything_is_computed() {
        let dir = tmp_dir("preflight_third_label");
        let meta = write_metadata_with_conditions(&dir, &["CTRL", "CASE", "OTHER"]);
        let mut args = metadata_args(&dir, &meta);
        args.case_condition = Some("CASE".into());
        args.control_condition = Some("CTRL".into());
        let failure = preflight(&args).expect_err("a third label must be refused");
        let text = failure.message();
        assert!(text.contains("neither the case nor the control"), "{text}");
        assert!(text.contains("OTHER"), "{text}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An empty condition cell is a hole in the design, not a third group.
    #[test]
    fn a_sample_with_no_condition_is_refused() {
        let dir = tmp_dir("preflight_empty_condition");
        let meta = write_metadata_with_conditions(&dir, &["CTRL", "CASE", ""]);
        let mut args = metadata_args(&dir, &meta);
        args.case_condition = Some("CASE".into());
        args.control_condition = Some("CTRL".into());
        let failure = preflight(&args).expect_err("an empty condition must be refused");
        let text = failure.message();
        assert!(text.contains("has no condition"), "{text}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A file without the unix executable bit is not a tool.
    #[cfg(unix)]
    #[test]
    fn a_file_without_the_executable_bit_is_not_a_tool() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tmp_dir("execbit");
        let path = dir.join("not_a_tool");
        std::fs::write(&path, "#!/bin/sh\n").expect("write");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).expect("chmod");
        assert!(
            find_executable(&path).is_none(),
            "a file without the executable bit must not be taken for a tool"
        );
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        assert!(
            find_executable(&path).is_some(),
            "a file with the executable bit is a tool"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The manifest's DPL fields are read out of the engine's own table, so the reader has to survive the shape the engine actually writes -- and has to refuse a table it does not understand rather than guess.
    #[test]
    fn the_dpl_facts_are_read_out_of_the_engines_own_table() {
        let dir = tmp_dir("dpl_facts");
        let table = dir.join("dpl.tsv");

        let params = "model: \
                      DT-EB (limma eBayes with trend = the mean over a transcript's \
                      cells of the log2 of the summed posterior weight of the cell, and robust \
                      = TRUE; the prior variance is fitted against that read support instead \
                      of the average expression, so s2.prior and df.prior are per transcript: \
                      s2.prior median 0.1, range 0.01..0.5, df.prior median 10.9, \
                      range 1..20); the moderation is fitted on the testable rows only \
                      (5348 transcripts); cell admission gate: min_weight = 5";
        std::fs::write(
            &table,
            format!(
                "transcript_id\tpvalue\tconfounded_design\tmodel_parameters\n\
                 T_A1\t0.01\tTRUE\t{params}\n"
            ),
        )
        .expect("write");
        let facts = read_dpl_facts(&table).expect("the engine's shape must be readable");
        assert_eq!(facts.model.as_deref(), Some("DT-EB"));
        assert_eq!(
            facts.robust,
            Some(true),
            "the engine's sentence says robust = TRUE"
        );
        assert_eq!(facts.confounded, Some(true), "the column says TRUE");
        let description = facts.description.expect("the engine's sentence is there");
        assert!(
            description.starts_with("DT-EB (limma eBayes with trend"),
            "the receipt is the engine's own sentence: {description}"
        );
        assert!(
            description.ends_with(')'),
            "the receipt is a whole sentence with a closed parenthesis: {description}"
        );
        assert!(
            description.contains("df.prior median 10.9"),
            "the receipt is not cut at the semicolon inside the engine's parenthetical: {description}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A table this reader does not understand yields no facts.
    #[test]
    fn an_unreadable_dpl_table_yields_no_facts() {
        let dir = tmp_dir("dpl_facts_missing");
        let table = dir.join("dpl.tsv");
        std::fs::write(&table, "transcript_id\tpvalue\nT_A1\t0.01\n").expect("write");
        let facts = read_dpl_facts(&table).expect("the table itself is readable");
        assert_eq!(
            facts.model, None,
            "a table without the engine's sentence must not produce a model name"
        );
        assert_eq!(facts.robust, None);
        assert_eq!(facts.confounded, None);

        let table2 = dir.join("dpl2.tsv");
        std::fs::write(
            &table2,
            "transcript_id\tconfounded_design\tmodel_parameters\nT_A1\tTRUE\tbase fit: equal_raw_t (unweighted two-sample t); moderated arm: depth_trend_eb (limma eBayes with trend)\n",
        )
        .expect("write");
        let facts2 = read_dpl_facts(&table2).expect("the table itself is readable");
        assert_eq!(
            facts2.model, None,
            "no sentence naming the model, no model name"
        );
        assert_eq!(
            facts2.confounded,
            Some(true),
            "the column is read on its own: one missing fact must not lose the others"
        );

        assert!(
            read_dpl_facts(&dir.join("absent.tsv")).is_none(),
            "a missing table must not produce facts"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The model's name is the model that was fitted.
    #[test]
    fn the_arm_name_is_the_arm_that_was_fitted() {
        let dir = tmp_dir("dpl_facts_arm");
        let table = dir.join("dpl.tsv");
        std::fs::write(
            &table,
            "transcript_id\tconfounded_design\tmodel_parameters\n\
             T_A1\tFALSE\tmodel: none (no model was fitted: no transcript was testable, so the length track has no contrast to report); the moderation is fitted on the testable rows only (0 transcripts)\n",
        )
        .expect("write");
        let facts = read_dpl_facts(&table).expect("the table is readable");
        assert_eq!(facts.model.as_deref(), Some("none"));
        assert_eq!(
            facts.robust,
            Some(false),
            "no arm was fitted, so no robust moderation was fitted either"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// THE RECEIPT NAMES THE ANNOTATION LAPA READ.
    #[test]
    fn the_receipt_names_the_annotation_lapa_read() {
        let dir = tmp_dir("manifest_lapa_annotation");
        let mut plan = plan_with_conditions(&["CTRL", "CTRL", "CTRL", "CASE", "CASE"]);
        plan.modalities = vec![Modality::Deg, Modality::Apa];
        plan.gtf = Some(PathBuf::from("/annotations/transcript_model.gtf"));
        let args = metadata_args(&dir, &dir.join("metadata.tsv"));
        let annotation = LapaAnnotation::Renamed {
            source: PathBuf::from("/annotations/transcript_model.gtf"),
            copy: PathBuf::from("/out/lapa/annotation_lapa.gtf"),
        };

        let manifest_path = dir.join("run_manifest.json");
        write_manifest(
            &manifest_path,
            &plan,
            &args,
            Path::new("logs/metadata.tsv"),
            "2026-11-01T00:00:00Z",
            "2026-11-01T00:01:00Z",
            &[],
            None,
            Some(&annotation),
        )
        .expect("manifest");
        let json = std::fs::read_to_string(&manifest_path).expect("read");
        assert!(
            json.contains("\"lapa_annotation\""),
            "the manifest names the annotation LAPA read: {json}"
        );
        assert!(
            json.contains("annotation_lapa.gtf"),
            "the manifest names the copy LAPA read: {json}"
        );

        let params_path = dir.join("model_parameters.tsv");
        write_model_parameters(&params_path, &plan, &args, None, Some(&annotation)).expect("write");
        let params = std::fs::read_to_string(&params_path).expect("read");
        assert!(
            params.contains("lapa_annotation"),
            "the poly(A) receipt names the annotation LAPA read: {params}"
        );

        let no_lapa = dir.join("run_manifest_no_lapa.json");
        write_manifest(
            &no_lapa,
            &plan,
            &args,
            Path::new("logs/metadata.tsv"),
            "2026-11-01T00:00:00Z",
            "2026-11-01T00:01:00Z",
            &[],
            None,
            None,
        )
        .expect("manifest");
        let json = std::fs::read_to_string(&no_lapa).expect("read");
        assert!(
            !json.contains("\"lapa_annotation\""),
            "a run that made no LAPA call names no annotation: {json}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A plan with one sample per condition label, for the group-size and manifest tests.
    fn plan_with_conditions(conditions: &[&str]) -> Plan {
        Plan {
            samples: conditions
                .iter()
                .enumerate()
                .map(|(i, condition)| Sample {
                    sample_id: format!("S{}", i + 1),
                    profile: PathBuf::from(format!("S{}.parquet", i + 1)),
                    condition: Some((*condition).to_string()),
                    batch: None,
                    alignment: None,
                })
                .collect(),
            profiles: Vec::new(),
            case_condition: Some("CASE".into()),
            control_condition: Some("CTRL".into()),
            modalities: vec![Modality::Deg, Modality::Det, Modality::Diu],
            batch_column: None,
            batch_state: BatchState::NoColumn,
            rscript: PathBuf::from("Rscript"),
            r_script: Some(PathBuf::from("telescope_diff.R")),
            lapa: None,
            lapa_dir: None,
            lapa_params: LapaParams::made_here(),
            gtf: None,
            gene_names: BTreeMap::new(),
            fasta: None,
            chrom_sizes: None,
        }
    }

    /// The group's sizes are read from the plan's samples, and a plan that does not know its conditions says so rather than guessing.
    #[test]
    fn the_group_sizes_are_counted_from_the_plan() {
        let plan = plan_with_conditions(&["CTRL", "CTRL", "CTRL", "CASE", "CASE"]);
        assert_eq!(
            group_sizes(&plan),
            Some((3, 2)),
            "three controls and two cases"
        );
        let mut unknown = plan_with_conditions(&["CTRL", "CTRL"]);
        unknown.case_condition = None;
        unknown.control_condition = None;
        assert_eq!(
            group_sizes(&unknown),
            None,
            "a plan that does not name its conditions cannot report a group size"
        );
    }

    /// The receipt records the ENGINE'S FIXED CHOICES, not a knob the caller could have moved.
    #[test]
    fn the_manifest_records_the_engines_fixed_choices_and_the_group_sizes() {
        let dir = tmp_dir("manifest_fixed");
        let plan = plan_with_conditions(&["CTRL", "CTRL", "CTRL", "CASE", "CASE"]);
        let args = metadata_args(&dir, &dir.join("metadata.tsv"));
        let path = dir.join("run_manifest.json");
        write_manifest(
            &path,
            &plan,
            &args,
            Path::new("logs/metadata.tsv"),
            "2026-11-01T00:00:00Z",
            "2026-11-01T00:01:00Z",
            &[],
            None,
            None,
        )
        .expect("manifest");
        let json = std::fs::read_to_string(&path).expect("read");
        assert!(
            json.contains("\"admission_gate\": \"and\""),
            "the manifest names the admission rule: {json}"
        );
        assert!(
            json.contains("\"usage_track_matrix\": \"narrowed\""),
            "the manifest names the usage track's matrix: {json}"
        );
        assert!(
            json.contains("\"apa_model\": \"DEXSeq\""),
            "the manifest names the poly(A) model: {json}"
        );
        assert!(
            json.contains("\"n_control_samples\": 3"),
            "the manifest names the group sizes: {json}"
        );
        assert!(
            json.contains("\"sample_table\": \"logs/metadata.tsv\""),
            "the manifest names the sample table the engine was given: {json}"
        );

        for gone in [
            "min_feature_samples",
            "feature_min_counts",
            "diu_matrix",
            "profiles_glob",
        ] {
            assert!(
                !json.contains(gone),
                "the manifest must not name the deleted option `{gone}`: {json}"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The model's parameters record the poly(A) model the run fitted, and they record no admission knob at all: there is only one admission rule and one usage-track matrix, so a receipt that listed a parameter would be listing something that decided nothing.
    #[test]
    fn the_model_parameters_name_the_poly_a_model_and_no_admission_knob() {
        let dir = tmp_dir("model_params_fixed");
        let mut plan = plan_with_conditions(&["CTRL", "CTRL", "CTRL", "CASE", "CASE"]);
        plan.modalities = vec![Modality::Deg, Modality::Apa];
        let args = metadata_args(&dir, &dir.join("metadata.tsv"));
        let path = dir.join("model_parameters.tsv");
        write_model_parameters(&path, &plan, &args, None, None).expect("write");
        let params = std::fs::read_to_string(&path).expect("read");
        assert!(
            params.contains("model_name\tLAPA+DEXSeq"),
            "the poly(A) model is named: {params}"
        );
        for gone in [
            "feature_gate",
            "min_feature_samples",
            "feature_min_counts",
            "diu_matrix",
        ] {
            assert!(
                !params.contains(gone),
                "the receipt must not name the deleted knob `{gone}`: {params}"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A REUSED SET OF POLY(A) CALLS NAMES ITS OWN PARAMETERS.
    #[test]
    fn a_reused_lapa_directory_names_the_parameters_the_calls_were_made_with() {
        let dir = tmp_dir("lapareceipt");
        let params = LapaParams::made_here();
        write_lapa_params_receipt(&dir, &params).expect("write");

        let read = lapa_params_from_receipt(&dir).expect("the receipt is read");
        assert_eq!(read.counting_method, LAPA_COUNTING_METHOD);
        assert_eq!(read.min_replication_rate, "0.95");
        assert!(
            !read.counting_method.contains("unknown"),
            "a reused call set with a receipt names its counting method: {}",
            read.counting_method
        );

        let canonical = dir.join("canonical");
        std::fs::create_dir_all(&canonical).expect("mkdir");
        write_lapa_params_receipt(&canonical, &params).expect("write");
        let read = lapa_params_from_receipt(&canonical).expect("the receipt is read");
        assert_eq!(read.counting_method, LAPA_COUNTING_METHOD);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A REUSED SET OF POLY(A) CALLS WITH NO RECEIPT SAYS "unknown".
    #[test]
    fn a_reused_lapa_directory_without_a_receipt_leaves_the_parameters_unknown() {
        let dir = tmp_dir("lapareceiptnone");
        std::fs::create_dir_all(dir.join("canonical").join("sample")).expect("mkdir");
        assert!(
            lapa_params_from_receipt(&dir).is_none(),
            "a directory without a receipt has no parameters to state"
        );
        let params = LapaParams::reused();
        assert!(params.counting_method.contains("unknown"));
        assert!(params.min_replication_rate.contains("unknown"));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// AN EXPLICIT STATEMENT WINS.
    #[test]
    fn an_explicit_statement_overrides_what_a_receipt_says() {
        let dir = tmp_dir("lapareceiptoverride");
        let params = LapaParams::made_here();
        write_lapa_params_receipt(&dir, &params).expect("write");
        let read = lapa_params_from_receipt(&dir).expect("the receipt is read");

        let stated = read.clone().with_overrides(Some("tail"), Some("0.90"));
        assert_eq!(stated.counting_method, "tail");
        assert_eq!(stated.min_replication_rate, "0.90");

        let half = read.clone().with_overrides(Some("tail"), None);
        assert_eq!(half.counting_method, "tail");
        assert_eq!(half.min_replication_rate, "0.95");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// AN ANNOTATION FROM THE UPSTREAM ASSEMBLER IS RENAMED FOR LAPA, AND NOTHING ELSE CHANGES.
    #[test]
    fn lapa_annotation_renames_the_utr_features_and_copies_everything_else() {
        let dir = tmp_dir("lapaanno_utr");
        let gtf = dir.join("transcript_model.gtf");
        std::fs::write(
            &gtf,
            concat!(
                "chr1\tsrc\tUTR5\t100\t200\t.\t+\t.\tgene_id \"G1\"; gene_name \"S1\";\n",
                "chr1\tsrc\tUTR3\t300\t400\t.\t-\t.\tgene_id \"G1\"; gene_name \"S1\";\n",
                "chr1\tsrc\texon\t100\t400\t.\t+\t.\tgene_id \"G1\"; gene_name \"S1\";\n",
            ),
        )
        .expect("write");

        let out = lapa_annotation(&gtf, &dir).expect("annotation");
        assert!(
            matches!(&out, LapaAnnotation::Renamed { .. }),
            "an annotation from the upstream assembler cannot reach LAPA as it is"
        );
        assert_eq!(
            out.path(),
            dir.join("annotation_lapa.gtf").as_path(),
            "the copy is written into the delivery directory"
        );

        let expected = concat!(
            "chr1\tsrc\tfive_prime_utr\t100\t200\t.\t+\t.\tgene_id \"G1\"; gene_name \"S1\";\n",
            "chr1\tsrc\tthree_prime_utr\t300\t400\t.\t-\t.\tgene_id \"G1\"; gene_name \"S1\";\n",
            "chr1\tsrc\texon\t100\t400\t.\t+\t.\tgene_id \"G1\"; gene_name \"S1\";\n",
        );
        assert_eq!(
            std::fs::read_to_string(out.path()).expect("read"),
            expected,
            "only the feature column may change"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// AN ANNOTATION LAPA ALREADY ACCEPTS IS HANDED TO IT UNCHANGED.
    #[test]
    fn lapa_annotation_hands_a_gencode_annotation_to_lapa_unchanged() {
        let dir = tmp_dir("lapaanno_gencode");
        let gtf = dir.join("annotation.gtf");
        std::fs::write(
            &gtf,
            concat!(
                "chr1\tGENCODE\tfive_prime_utr\t100\t200\t.\t+\t.\tgene_id \"G1\"; gene_name \"S1\";\n",
                "chr1\tGENCODE\tthree_prime_utr\t300\t400\t.\t-\t.\tgene_id \"G1\"; gene_name \"S1\";\n",
            ),
        )
        .expect("write");

        let out = lapa_annotation(&gtf, &dir).expect("annotation");
        assert!(
            matches!(&out, LapaAnnotation::Original(_)),
            "an annotation LAPA accepts must reach it unchanged"
        );
        assert_eq!(
            out.path(),
            gtf.as_path(),
            "an annotation LAPA accepts must reach it unchanged"
        );
        assert!(
            !dir.join("annotation_lapa.gtf").exists(),
            "no copy is written when there is nothing to rename"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A NAME IS RENAMED ONLY WHEN LAPA CANNOT FIND IT.
    #[test]
    fn lapa_annotation_renames_only_the_name_lapa_cannot_find() {
        let dir = tmp_dir("lapaanno_hybrid");
        let gtf = dir.join("hybrid.gtf");
        std::fs::write(
            &gtf,
            concat!(
                "chr1\tX\tUTR5\t100\t200\t.\t+\t.\tgene_id \"G1\"; gene_name \"S1\";\n",
                "chr1\tX\tfive_prime_utr\t100\t200\t.\t+\t.\tgene_id \"G1\"; gene_name \"S1\";\n",
                "chr1\tX\tUTR3\t300\t400\t.\t-\t.\tgene_id \"G1\"; gene_name \"S1\";\n",
            ),
        )
        .expect("write");

        let out = lapa_annotation(&gtf, &dir).expect("annotation");
        let text = std::fs::read_to_string(out.path()).expect("read");
        assert!(
            text.contains("\tUTR5\t"),
            "UTR5 is left alone when the annotation already carries five_prime_utr"
        );
        assert!(
            text.contains("\tthree_prime_utr\t"),
            "UTR3 is renamed when three_prime_utr is absent"
        );
        assert!(
            !text.contains("\tUTR3\t"),
            "UTR3 must not survive the rename"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A LINE WITH FEWER THAN NINE FIELDS HAS NO FEATURE COLUMN.
    #[test]
    fn lapa_annotation_leaves_a_line_without_a_feature_column_alone() {
        let dir = tmp_dir("lapaanno_short");
        let gtf = dir.join("short.gtf");
        std::fs::write(
            &gtf,
            concat!(
                "chr1\tsrc\tUTR5\t100\t200\t.\t+\t.\tgene_id \"G1\"; gene_name \"S1\";\n",
                "chr1\tsrc\tUTR5\t1\t2\n",
            ),
        )
        .expect("write");

        let out = lapa_annotation(&gtf, &dir).expect("annotation");
        let text = std::fs::read_to_string(out.path()).expect("read");
        assert!(
            text.ends_with("chr1\tsrc\tUTR5\t1\t2\n"),
            "a line with fewer than nine fields is copied as it is"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// THE COPY IS A BYTE COPY OUTSIDE THE FEATURE COLUMN.
    #[test]
    fn lapa_annotation_preserves_every_byte_outside_the_feature_column() {
        let dir = tmp_dir("lapaanno_bytes");
        let gtf = dir.join("bytes.gtf");
        let mut source: Vec<u8> = Vec::new();
        source.extend_from_slice(b"chr1\tsrc\tUTR5\t100\t200\t.\t+\t.\tgene_id \"G1\";\r\n");
        source.extend_from_slice(b"chr1\tsrc\tUTR3\t300\t400\t.\t-\t.\tgene_id \"G2\"; \xff\n");
        source.extend_from_slice(b"chr1\tsrc\texon\t1\t2\t.\t+\t.\tgene_id \"G3\";");
        std::fs::write(&gtf, &source).expect("write");

        let out = lapa_annotation(&gtf, &dir).expect("annotation");
        let copied = std::fs::read(out.path()).expect("read");

        let split = |bytes: &[u8]| -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
            bytes
                .split(|b| *b == b'\n')
                .map(|line| match gtf_feature_span(line) {
                    Some((s, e)) => (line[..s].to_vec(), line[s..e].to_vec(), line[e..].to_vec()),
                    None => (line.to_vec(), Vec::new(), Vec::new()),
                })
                .collect()
        };
        let before = split(&source);
        let after = split(&copied);
        assert_eq!(before.len(), after.len(), "the copy has the same lines");
        for (i, (b, a)) in before.iter().zip(after.iter()).enumerate() {
            assert_eq!(b.0, a.0, "line {i}: the bytes before the feature column");
            assert_eq!(b.2, a.2, "line {i}: the bytes after the feature column");
        }

        assert_eq!(after[0].1, b"five_prime_utr".to_vec(), "UTR5");
        assert_eq!(after[1].1, b"three_prime_utr".to_vec(), "UTR3");
        assert_eq!(after[2].1, b"exon".to_vec(), "every other feature");

        assert!(
            !copied.ends_with(b"\n"),
            "a last line without a newline must not gain one"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// LAPA IS GIVEN THE RENAMED ANNOTATION.
    #[test]
    fn lapa_is_given_the_renamed_annotation() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tmp_dir("lapawiring");
        let gtf = dir.join("transcript_model.gtf");
        std::fs::write(
            &gtf,
            "chr1\tsrc\tUTR3\t300\t400\t.\t-\t.\tgene_id \"G1\"; gene_name \"S1\";\n",
        )
        .expect("write");

        let args_file = dir.join("lapa_args.txt");
        let driver = dir.join("fake_lapa.sh");
        std::fs::write(
            &driver,
            format!(
                "#!/bin/sh\n\
                 : > '{args}'\n\
                 for a in \"$@\"; do printf '%s\\n' \"$a\" >> '{args}'; done\n\
                 prev=\"\"; out=\"\"\n\
                 for a in \"$@\"; do\n\
                 \x20 if [ \"$prev\" = \"--output_dir\" ]; then out=\"$a\"; fi\n\
                 \x20 prev=\"$a\"\n\
                 done\n\
                 mkdir -p \"$out/sample\"\n\
                 printf 'chr1\\t100\\t101\\n' > \"$out/sample/S1.bed\"\n",
                args = args_file.display()
            ),
        )
        .expect("write");
        std::fs::set_permissions(&driver, std::fs::Permissions::from_mode(0o755)).expect("chmod");

        let out_dir = dir.join("lapa");
        let samples = vec![Sample {
            sample_id: "S1".to_string(),
            profile: dir.join("S1.profile.parquet"),
            condition: Some("KO".to_string()),
            batch: None,
            alignment: Some(dir.join("S1.sorted.bam")),
        }];

        let (canonical, annotation) = run_lapa(
            &driver,
            &LapaParams::made_here(),
            &dir.join("genome.fa"),
            &gtf,
            &dir.join("chrom_sizes.tsv"),
            &samples,
            &out_dir,
            &dir.join("lapa.log"),
            &dir.join("surviving.log"),
        )
        .expect("lapa");
        assert!(
            matches!(&annotation, LapaAnnotation::Renamed { .. }),
            "LAPA is given the renamed copy, not the upstream assembler's annotation"
        );
        assert_eq!(
            canonical,
            out_dir.join("canonical"),
            "the calls are read from the canonical layout"
        );

        let args = std::fs::read_to_string(&args_file).expect("the driver ran");
        let annotation = args
            .lines()
            .skip_while(|l| *l != "--annotation")
            .nth(1)
            .expect("the driver was called with --annotation");
        assert_eq!(
            annotation,
            out_dir.join("annotation_lapa.gtf").display().to_string(),
            "LAPA must be given the renamed annotation"
        );
        assert!(
            std::fs::read_to_string(annotation)
                .expect("read")
                .contains("\tthree_prime_utr\t"),
            "the file LAPA was given carries the GENCODE name"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
