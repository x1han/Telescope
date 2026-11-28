use std::collections::{BTreeMap, HashMap};
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

/// Default half-saturation Kish effective sample size for the ΔPAL track's transcript-row precision weights.
pub const DPL_NEFF_SAT_DEFAULT: f64 = 100.0;

/// Default minimum total count for a transcript in the ΔPAU track.
pub const MIN_COUNTS_DEFAULT: f64 = 20.0;

/// Default minimum number of samples per transcript in the ΔPAL track.
pub const MIN_SAMPLES_DEFAULT: usize = 3;

/// The default threshold of the observed usage difference that makes a poly(A) site significant in the DEXSeq arm.
pub const APA_MIN_DELTA_PAU_DEFAULT: f64 = 0.05;

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

/// Arguments of the `diff` subcommand.
#[derive(clap::Args, Debug)]
pub struct DiffArgs {
    /// Metadata TSV: one row per sample (sample_id, profile_path, condition, optional batch and alignment_path).
    #[arg(long, required = true, help_heading = "INPUTS")]
    pub metadata: PathBuf,

    /// Metadata column holding the sample identifier.
    #[arg(long, default_value = DEFAULT_SAMPLE_COLUMN, help_heading = "INPUTS")]
    pub sample_column: String,

    /// Metadata column holding the profile parquet path.
    #[arg(long, default_value = DEFAULT_PROFILE_COLUMN, help_heading = "INPUTS")]
    pub profile_column: String,

    /// Metadata column holding the condition label.
    #[arg(long, default_value = DEFAULT_CONDITION_COLUMN, help_heading = "INPUTS")]
    pub condition_column: String,

    /// Metadata column holding the optional batch covariate label.
    #[arg(long, default_value = DEFAULT_BATCH_COLUMN, help_heading = "INPUTS")]
    pub batch_column: String,

    /// Metadata column holding each sample's genomic alignment BAM path (required for APA).
    #[arg(long, default_value = DEFAULT_ALIGNMENT_COLUMN, help_heading = "INPUTS")]
    pub alignment_column: String,

    /// Annotation GTF providing gene names and transcript-to-gene structure for APALORD.
    #[arg(short = 'g', long = "gtf", help_heading = "REFERENCE")]
    pub gtf: Option<PathBuf>,

    /// Output directory for differential result tables and execution receipts.
    #[arg(long, required = true, help_heading = "OUTPUTS")]
    pub output_dir: PathBuf,

    /// Number of worker threads used for poly(A) site calling.
    #[arg(long, default_value_t = 4, help_heading = "TOOLS")]
    pub threads: usize,

    /// Comma-separated modalities to run: deg, det, diu, apa, dpl.
    #[arg(long, value_name = "MODALITIES", help_heading = "MODEL")]
    pub modalities: Option<String>,

    /// Minimum posterior-weight mass per (transcript x sample) cell.
    #[arg(long = "min-cell-weight", value_name = "MIN_CELL_WEIGHT",
          default_value_t = MIN_CELL_WEIGHT_DEFAULT, help_heading = "GATES")]
    pub min_weight: f64,

    /// Minimum Kish effective sample size per (transcript x sample) cell.
    #[arg(long, default_value_t = MIN_NEFF_DEFAULT, help_heading = "GATES")]
    pub min_neff: f64,

    /// Half-saturation Kish effective sample size for the DT-EB length precision weights.
    #[arg(long = "dpl-neff-sat", value_name = "K",
          default_value_t = DPL_NEFF_SAT_DEFAULT, help_heading = "MODEL")]
    pub dpl_neff_sat: f64,

    /// Minimum total read count required for a feature to be tested.
    #[arg(long, default_value_t = MIN_COUNTS_DEFAULT, help_heading = "GATES")]
    pub min_counts: f64,

    /// Minimum number of biological replicates with non-zero counts per group.
    #[arg(long, default_value_t = MIN_SAMPLES_DEFAULT, help_heading = "GATES")]
    pub min_samples: usize,

    /// Minimum absolute poly(A) site usage difference (|delta_pau|) required for APA significance.
    #[arg(long, default_value_t = APA_MIN_DELTA_PAU_DEFAULT, help_heading = "MODEL")]
    pub apa_min_delta_pau: f64,

    /// R interpreter executable (defaults to $TELESCOPE_RSCRIPT or `Rscript` on PATH).
    #[arg(long, help_heading = "TOOLS")]
    pub rscript: Option<PathBuf>,

    /// Path to telescope_diff.R (defaults to $TELESCOPE_DIFF_R or the bundled script).
    #[arg(long, help_heading = "TOOLS")]
    pub r_script: Option<PathBuf>,

    /// Case condition label (reported effect is case minus control).
    #[arg(long, help_heading = "INPUTS")]
    pub case_condition: Option<String>,

    /// Control condition label.
    #[arg(long, help_heading = "INPUTS")]
    pub control_condition: Option<String>,

    /// Contrast specification as CASE:CONTROL (alternative to --case-condition / --control-condition).
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
    /// Alternative poly(A) site usage (the in-tree poly(A) caller's counts, tested with DEXSeq by the engine).
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
    /// The R engine script path.
    pub r_script: PathBuf,
    /// The annotation, when the run was given one.
    pub gtf: Option<PathBuf>,
    /// What every gene of the annotation is CALLED.
    pub gene_names: BTreeMap<String, String>,
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
    out.push_str("\talignment_path");
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
        out.push_str(&format!(
            "\t{}",
            s.alignment
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default()
        ));
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
    let reader = builder
        .with_projection(mask)
        .with_batch_size(8192)
        .build()
        .with_context(|| format!("failed to open {} for reading", path.display()))?;
    let mut map: HashMap<String, String> = HashMap::new();
    for batch in reader {
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
    /// Posterior mass assigned to this transcript in this sample, over EVERY read -- the ABUNDANCE track's quantity.
    pub weight_sum: f64,
    /// Numerator of the posterior-weighted mean log2 tail length.
    pub weighted_log2pal_sum: f64,
    /// Sum of the squared posterior weights of the cell's reads.
    pub weight_sq_sum: f64,
    /// Sum of posterior weight times squared log2 tail length.
    pub weighted_log2pal_sq_sum: f64,
    /// The TAIL-BEARING posterior mass: the denominator of the posterior-weighted mean log2 tail length and the numerator of the tail track's Kish effective sample size.
    pub weight_sum_pal: f64,
    /// The sum of the squared posterior weights of the TAIL-BEARING reads: the denominator of the tail track's Kish effective sample size.
    pub weight_sq_sum_pal: f64,
}

/// The interned ids of one aggregated cell: `(sample, gene, transcript)`.
type CellKey = (u32, u32, u32);
/// The SIX accumulators of one cell, in the order `(weight_sum, weight_sq_sum, weight_sum_pal, weight_sq_sum_pal, weighted_log2pal_sum, weighted_log2pal_sq_sum)`.
type CellAcc = (f64, f64, f64, f64, f64, f64);

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
    let reader = builder
        .with_batch_size(8192)
        .build()
        .with_context(|| format!("failed to open {} for reading", path.display()))?;

    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    let mut interner = Interner::new();

    let mut acc: HashMap<CellKey, CellAcc> = HashMap::new();
    let mut n_rows = 0u64;
    let mut declared_sample: Option<String> = None;

    for batch in reader {
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
            if gene.is_null(i) || tx.is_null(i) || pw.is_null(i) {
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
            let e = acc
                .entry((s, g, t))
                .or_insert((0.0, 0.0, 0.0, 0.0, 0.0, 0.0));

            e.0 += w;
            e.1 += w * w;

            if !pal.is_null(i) {
                let log2pal = (pal.value(i).max(1) as f64).log2();
                e.2 += w;
                e.3 += w * w;
                e.4 += w * log2pal;
                e.5 += w * log2pal * log2pal;
            }
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
        .map(|((s, g, t), (w, w2, wp, wp2, l, l2))| AggRow {
            sample_id: interner.resolve(s).to_string(),
            gene_id: interner.resolve(g).to_string(),
            transcript_id: interner.resolve(t).to_string(),
            weight_sum: w,
            weighted_log2pal_sum: l,
            weight_sq_sum: w2,
            weighted_log2pal_sq_sum: l2,
            weight_sum_pal: wp,
            weight_sq_sum_pal: wp2,
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
    let mut merged: HashMap<(String, String, String), CellAcc> = HashMap::new();

    for r in per_file {
        for row in r? {
            let e = merged
                .entry((row.sample_id, row.gene_id, row.transcript_id))
                .or_insert((0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
            e.0 += row.weight_sum;
            e.1 += row.weight_sq_sum;
            e.2 += row.weight_sum_pal;
            e.3 += row.weight_sq_sum_pal;
            e.4 += row.weighted_log2pal_sum;
            e.5 += row.weighted_log2pal_sq_sum;
        }
    }
    let mut out: Vec<AggRow> = merged
        .into_iter()
        .map(
            |(
                (sample_id, gene_id, transcript_id),
                (
                    weight_sum,
                    weight_sq_sum,
                    weight_sum_pal,
                    weight_sq_sum_pal,
                    weighted_log2pal_sum,
                    weighted_log2pal_sq_sum,
                ),
            )| AggRow {
                sample_id,
                gene_id,
                transcript_id,
                weight_sum,
                weighted_log2pal_sum,
                weight_sq_sum,
                weighted_log2pal_sq_sum,
                weight_sum_pal,
                weight_sq_sum_pal,
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
        "sample_id\tgene_id\ttranscript_id\tweight_sum\tweighted_log2pal_sum\tweight_sq_sum\tweighted_log2pal_sq_sum\tweight_sum_pal\tweight_sq_sum_pal\n",
    );
    for r in rows {
        out.push_str(&format!(
            "{}\t{}\t{}\t{:?}\t{:?}\t{:?}\t{:?}\t{:?}\t{:?}\n",
            r.sample_id,
            r.gene_id,
            r.transcript_id,
            r.weight_sum,
            r.weighted_log2pal_sum,
            r.weight_sq_sum,
            r.weighted_log2pal_sq_sum,
            r.weight_sum_pal,
            r.weight_sq_sum_pal
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
    let contrast = Some(
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
    );

    let modalities = resolve_modalities(args.modalities.as_deref()).map_err(|e| {
        PreflightFailure::new(
            e.to_string(),
            "name the modalities as a comma-separated list of deg, det, diu, apa and dpl.",
        )
    })?;

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
    if find_executable(&rscript).is_none() {
        return Err(PreflightFailure::new(
            format!("the R interpreter was not found: {}", rscript.display()),
            "install R, or point --rscript at it, or set TELESCOPE_RSCRIPT.",
        ));
    }
    let r_script = resolve_r_script(args.r_script.as_deref()).map_err(|e| {
        PreflightFailure::new(
            e.to_string(),
            "point --r-script at workflow/scripts/telescope_diff.R, or set \
             TELESCOPE_DIFF_R to that path.",
        )
    })?;

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

    if modalities.contains(&Modality::Apa) {
        let _ = require_file(&args.gtf, "--gtf", "the APA track")?;
        check_alignments(&samples, "--alignment-column", has_alignment_column)?;
    }

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
        gtf,
        gene_names,
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

fn apa_calls_made_by(result: &Path) -> Option<String> {
    let entries = std::fs::read_dir(result.join("apa_engine")).ok()?;
    let mut outcomes = entries.filter_map(|entry| {
        let text = std::fs::read_to_string(entry.ok()?.path().join("receipt.txt")).ok()?;
        text.lines()
            .find_map(|line| line.strip_prefix("calls_made_by\t").map(str::to_owned))
    });
    let outcome = outcomes.next()?;
    if outcomes.next().is_some() {
        return None;
    }
    Some(outcome)
}

/// Write `run_manifest.json`.
fn write_manifest(
    path: &Path,
    plan: &Plan,
    args: &DiffArgs,
    sample_table: &Path,
    times: (&str, &str),
    versions: &[(String, String)],
    dpl_facts: Option<&DplFacts>,
) -> Result<()> {
    let (start, end) = times;
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
    let apa_outcome = plan
        .modalities
        .contains(&Modality::Apa)
        .then(|| apa_calls_made_by(path.parent().unwrap_or_else(|| Path::new("."))))
        .flatten();
    if let Some(outcome) = apa_outcome.as_deref() {
        json.push_str(&format!(
            "    \"apa_calls_made_by\": {},\n",
            json_string(outcome)
        ));
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

        dpl_fields.push(format!("\"dpl_neff_sat\": {:?}", args.dpl_neff_sat));
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

                rows.push((
                    modality.clone(),
                    "neff_sat".into(),
                    format!(
                        "{} (the Kish-saturation half-saturation point of the \
                         row weights; a value at or below 0 fits the \
                         equal-weight model)",
                        args.dpl_neff_sat
                    ),
                ));
            }
            Modality::Apa => {
                rows.push((
                    modality.clone(),
                    "model_name".into(),
                    "the in-tree poly(A) caller + DEXSeq (see the model_name \
                     column of apa.tsv: it names `not_testable` when no model \
                     could be fitted)"
                        .into(),
                ));
                rows.push((
                    modality.clone(),
                    "caller".into(),
                    "the pure-R poly(A) caller that ships with this tool \
                     (workflow/scripts/apa_engine.R); no external caller is run"
                        .into(),
                ));
                rows.push((
                    modality.clone(),
                    "gtf".into(),
                    plan.gtf
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                ));
                let apa_outcome =
                    apa_calls_made_by(path.parent().unwrap_or_else(|| Path::new(".")));
                if let Some(outcome) = apa_outcome.as_ref() {
                    rows.push((
                        modality.clone(),
                        "apa_calls_made_by".into(),
                        outcome.clone(),
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
    let doe = z - era * 146_097;
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

    run_engine(
        args,
        plan,
        &agg_path,
        &metadata_path,
        gene_names.as_deref(),
        result,
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
                surviving.display(),
            );
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
        (start, &end),
        &versions,
        dpl_facts.as_ref(),
    )?;

    let session = result.join("sessionInfo.txt");
    if session.is_file() {
        let _ = std::fs::rename(&session, logs.join("sessionInfo.txt"));
    }

    publish(result, output_dir)?;
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
        (Modality::Apa, "--apa"),
    ] {
        flags.push(flag.to_string());
        if !modalities.contains(&m) {
            flags.push("false".to_string());
        }
    }
    flags
}

/// Run the R engine on the aggregate.
fn dpl_model_engine_args(args: &DiffArgs) -> Vec<String> {
    vec![
        "--min-cell-weight".to_string(),
        args.min_weight.to_string(),
        "--min-neff".to_string(),
        args.min_neff.to_string(),
        "--dpl-neff-sat".to_string(),
        args.dpl_neff_sat.to_string(),
    ]
}

fn apa_engine_args(args: &DiffArgs, plan: &Plan) -> Vec<String> {
    if !plan.modalities.contains(&Modality::Apa) {
        return Vec::new();
    }
    let mut flags = Vec::new();
    if let Some(gtf) = plan.gtf.as_ref() {
        flags.push("--gtf".to_string());
        flags.push(gtf.display().to_string());
    }
    flags.push("--apa-engine-cores".to_string());
    flags.push(args.threads.max(1).to_string());
    flags
}

fn run_engine(
    args: &DiffArgs,
    plan: &Plan,
    aggregate: &Path,
    metadata: &Path,
    gene_names: Option<&Path>,
    out_dir: &Path,
    surviving: &Path,
) -> Result<()> {
    let mut engine_args: Vec<String> = vec![
        plan.r_script.display().to_string(),
        "--aggregate".to_string(),
        aggregate.display().to_string(),
        "--metadata".to_string(),
        metadata.display().to_string(),
        "--output-dir".to_string(),
        out_dir.display().to_string(),
        "--min-counts".to_string(),
        args.min_counts.to_string(),
        "--min-samples".to_string(),
        args.min_samples.to_string(),
        "--apa-min-delta-pau".to_string(),
        args.apa_min_delta_pau.to_string(),
    ];

    engine_args.extend(dpl_model_engine_args(args));
    if let (Some(case), Some(control)) = (&plan.case_condition, &plan.control_condition) {
        engine_args.push("--case-condition".to_string());
        engine_args.push(case.clone());
        engine_args.push("--control-condition".to_string());
        engine_args.push(control.clone());
    }

    engine_args.extend(engine_modality_flags(&plan.modalities));

    engine_args.extend(apa_engine_args(args, plan));

    if let Some(g) = gene_names {
        engine_args.push("--gene-names".to_string());
        engine_args.push(g.display().to_string());
    }
    let log = out_dir.join("logs").join("telescope_diff.R.log");
    run_tool(&plan.rscript, &engine_args, "the R engine", &log, surviving)?;
    Ok(())
}

/// The versions of the tools that ran.
fn collect_versions(args: &DiffArgs, plan: &Plan) -> Vec<(String, String)> {
    let mut versions: Vec<(String, String)> = vec![(
        "telescope".to_string(),
        env!("CARGO_PKG_VERSION").to_string(),
    )];
    versions.push(("R".to_string(), tool_version(&plan.rscript, &["--version"])));
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

    /// Build a profile parquet whose `polyA_length` is NULLABLE, so a tail-less row can be written.
    fn write_profile_nullable(path: &Path, rows: &[(&str, &str, &str, Option<i32>, f64)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("sample_id", DataType::Utf8, false),
            Field::new("gene_id", DataType::Utf8, false),
            Field::new("transcript_id", DataType::Utf8, false),
            Field::new("polyA_length", DataType::Int32, true),
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
            p.append_option(*pal);
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
            weight_sum_pal: 0.3,
            weight_sq_sum_pal: 0.01,
        }];
        let path = dir.join("agg.tsv");
        write_aggregate_tsv(&path, &rows).expect("write");
        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines = text.lines();
        assert_eq!(
            lines.next().unwrap(),
            "sample_id\tgene_id\ttranscript_id\tweight_sum\tweighted_log2pal_sum\tweight_sq_sum\tweighted_log2pal_sq_sum\tweight_sum_pal\tweight_sq_sum_pal"
        );
        let f: Vec<&str> = lines.next().unwrap().split('\t').collect();
        assert_eq!(&f[..3], &["S1", "G1", "T1"]);

        assert_eq!(f[3].parse::<f64>().unwrap(), 0.1 + 0.2);
        assert_eq!(f[4].parse::<f64>().unwrap(), 1.0 / 3.0);
        assert_eq!(f[5].parse::<f64>().unwrap(), 0.05);
        assert_eq!(f[6].parse::<f64>().unwrap(), 1.0 / 7.0);
        assert_eq!(f[7].parse::<f64>().unwrap(), 0.3);
        assert_eq!(f[8].parse::<f64>().unwrap(), 0.01);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// THE TWO TRACKS, IN THE AGGREGATE.
    #[test]
    fn a_null_tail_row_counts_towards_the_abundance_and_not_towards_the_tail() {
        let dir = tmp_dir("nullpal");
        let path = dir.join("S1.profile.parquet");
        write_profile_nullable(
            &path,
            &[
                ("S1", "G1", "T1", Some(100), 0.5),
                ("S1", "G1", "T1", None, 0.25),
                ("S1", "G1", "T2", None, 1.0),
            ],
        );
        let rows = aggregate_profile(&path).expect("aggregate");
        assert_eq!(rows.len(), 2);
        let t1 = rows.iter().find(|r| r.transcript_id == "T1").unwrap();
        let t2 = rows.iter().find(|r| r.transcript_id == "T2").unwrap();

        assert!((t1.weight_sum - 0.75).abs() < 1e-12, "{t1:?}");
        assert!(
            (t1.weight_sq_sum - (0.5f64 * 0.5 + 0.25f64 * 0.25)).abs() < 1e-12,
            "{t1:?}"
        );

        assert!((t1.weight_sum_pal - 0.5).abs() < 1e-12, "{t1:?}");
        assert!((t1.weight_sq_sum_pal - 0.25).abs() < 1e-12, "{t1:?}");
        assert!(
            (t1.weighted_log2pal_sum - 0.5 * (100.0f64).log2()).abs() < 1e-12,
            "{t1:?}"
        );

        assert!((t2.weight_sum - 1.0).abs() < 1e-12, "{t2:?}");
        assert_eq!(t2.weight_sum_pal, 0.0, "{t2:?}");
        assert_eq!(t2.weight_sq_sum_pal, 0.0, "{t2:?}");
        assert_eq!(t2.weighted_log2pal_sum, 0.0, "{t2:?}");
        assert_eq!(t2.weighted_log2pal_sq_sum, 0.0, "{t2:?}");
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
            output_dir: dir.join("out"),
            threads: 4,
            modalities: None,
            min_weight: MIN_CELL_WEIGHT_DEFAULT,
            min_neff: MIN_NEFF_DEFAULT,
            dpl_neff_sat: DPL_NEFF_SAT_DEFAULT,
            min_counts: MIN_COUNTS_DEFAULT,
            min_samples: MIN_SAMPLES_DEFAULT,
            apa_min_delta_pau: APA_MIN_DELTA_PAU_DEFAULT,
            rscript: None,
            r_script: None,
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
            output_dir: dir.join("out"),
            threads: 4,
            modalities: None,
            min_weight: MIN_CELL_WEIGHT_DEFAULT,
            min_neff: MIN_NEFF_DEFAULT,
            dpl_neff_sat: DPL_NEFF_SAT_DEFAULT,
            min_counts: MIN_COUNTS_DEFAULT,
            min_samples: MIN_SAMPLES_DEFAULT,
            apa_min_delta_pau: APA_MIN_DELTA_PAU_DEFAULT,
            rscript: None,
            r_script: None,
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
            vec!["--deg", "--det", "false", "--diu", "false", "--dpl", "--apa", "false"]
        );
        assert_eq!(
            engine_modality_flags(&[]),
            vec![
                "--deg", "false", "--det", "false", "--diu", "false", "--dpl", "false", "--apa",
                "false"
            ]
        );
        assert_eq!(
            engine_modality_flags(&[Modality::Deg, Modality::Det, Modality::Diu, Modality::Dpl]),
            vec!["--deg", "--det", "--diu", "--dpl", "--apa", "false"]
        );

        assert_eq!(
            engine_modality_flags(&[Modality::Apa]),
            vec!["--deg", "false", "--det", "false", "--diu", "false", "--dpl", "false", "--apa"]
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

    #[test]
    fn missing_r_interpreter_fails_preflight() {
        let dir = tmp_dir("missing_r");
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
        let mut args = metadata_args(&dir, &meta);
        args.rscript = Some(PathBuf::from("/nonexistent/Rscript"));
        args.r_script = Some(PathBuf::from("/nonexistent/telescope_diff.R"));
        args.case_condition = Some("CASE".to_string());
        args.control_condition = Some("CTRL".to_string());
        let failure = preflight(&args).expect_err("R is needed here");
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
            output_dir: dir.join("out"),
            threads: 4,
            modalities: None,
            min_weight: MIN_CELL_WEIGHT_DEFAULT,
            min_neff: MIN_NEFF_DEFAULT,
            dpl_neff_sat: DPL_NEFF_SAT_DEFAULT,
            min_counts: MIN_COUNTS_DEFAULT,
            min_samples: MIN_SAMPLES_DEFAULT,
            apa_min_delta_pau: APA_MIN_DELTA_PAU_DEFAULT,
            rscript: None,
            r_script: None,
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
            r_script: PathBuf::from("telescope_diff.R"),
            gtf: None,
            gene_names: BTreeMap::new(),
        }
    }

    #[test]
    fn apa_receipts_report_calls_made_by_this_run() {
        let dir = tmp_dir("apa_actual_calls");
        let mut plan = plan_with_conditions(&["CTRL", "CTRL", "CASE", "CASE"]);
        plan.modalities = vec![Modality::Apa];
        let args = metadata_args(&dir, &dir.join("metadata.tsv"));
        let receipt_dir = dir.join("apa_engine/current_cohort_key");
        std::fs::create_dir_all(&receipt_dir).unwrap();
        std::fs::write(
            receipt_dir.join("receipt.txt"),
            "cohort_key\tcurrent_cohort_key\ncalls_made_by\tthis run\n",
        )
        .unwrap();
        write_manifest(
            &dir.join("run_manifest.json"),
            &plan,
            &args,
            Path::new("logs/metadata.tsv"),
            ("start", "end"),
            &[],
            None,
        )
        .unwrap();
        write_model_parameters(&dir.join("model_parameters.tsv"), &plan, &args, None).unwrap();
        for file in ["run_manifest.json", "model_parameters.tsv"] {
            let text = std::fs::read_to_string(dir.join(file)).unwrap();
            assert!(text.contains("apa_calls_made_by"), "{file}: {text}");
            assert!(text.contains("this run"), "{file}: {text}");
        }
        std::fs::remove_dir_all(&dir).ok();
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
            ("2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"),
            &[],
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
        write_model_parameters(&path, &plan, &args, None).expect("write");
        let params = std::fs::read_to_string(&path).expect("read");
        assert!(
            params.contains("model_name\tthe in-tree poly(A) caller + DEXSeq"),
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

    /// THE SAMPLE TABLE THE ENGINE IS GIVEN CARRIES THE ALIGNMENT OF EVERY SAMPLE.
    #[test]
    fn the_sample_table_carries_the_alignment_of_every_sample() {
        let dir = tmp_dir("metadata_alignment");
        let path = dir.join("metadata.tsv");
        let samples = vec![
            Sample {
                sample_id: "S1".into(),
                profile: PathBuf::from("S1.parquet"),
                condition: Some("CTRL".into()),
                batch: Some("b1".into()),
                alignment: Some(PathBuf::from("aln/S1.bam")),
            },
            Sample {
                sample_id: "S2".into(),
                profile: PathBuf::from("S2.parquet"),
                condition: Some("CASE".into()),
                batch: None,
                alignment: Some(PathBuf::from("aln/S2.bam")),
            },
        ];
        write_metadata_tsv(&path, &samples, true).expect("write");
        let text = std::fs::read_to_string(&path).expect("read");
        let header = text.lines().next().expect("a header");
        assert_eq!(
            header, "sample_id\tcondition\tprofile_path\tbatch\talignment_path",
            "the alignment column is the last one, after the optional batch: {header}"
        );
        assert!(
            text.lines().nth(1).expect("a row").ends_with("aln/S1.bam"),
            "every sample's alignment is in its row: {text}"
        );

        let path = dir.join("metadata_nobatch.tsv");
        write_metadata_tsv(&path, &samples, false).expect("write");
        let text = std::fs::read_to_string(&path).expect("read");
        assert_eq!(
            text.lines().next().expect("a header"),
            "sample_id\tcondition\tprofile_path\talignment_path",
            "the alignment column does not depend on the batch column: {text}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The saturation point of the DPL row weights is a MODEL parameter, and the engine has to be told which one the run used.
    #[test]
    fn dpl_neff_sat_flag_defaults_and_overrides() {
        #[derive(clap::Parser)]
        struct Cli {
            #[command(flatten)]
            args: DiffArgs,
        }
        let def = <Cli as clap::Parser>::try_parse_from([
            "diff",
            "--metadata",
            "meta.tsv",
            "--output-dir",
            "out",
        ])
        .expect("default parse");
        assert_eq!(def.args.dpl_neff_sat, DPL_NEFF_SAT_DEFAULT);
        assert_eq!(DPL_NEFF_SAT_DEFAULT, 100.0);

        let ov = <Cli as clap::Parser>::try_parse_from([
            "diff",
            "--metadata",
            "meta.tsv",
            "--output-dir",
            "out",
            "--dpl-neff-sat",
            "0",
        ])
        .expect("override parse");
        assert_eq!(ov.args.dpl_neff_sat, 0.0);

        let def_args = dpl_model_engine_args(&def.args);
        let i = def_args
            .iter()
            .position(|a| a == "--dpl-neff-sat")
            .expect("the saturation point reaches the engine");
        assert_eq!(def_args[i + 1], "100");
        let ov_args = dpl_model_engine_args(&ov.args);
        let j = ov_args
            .iter()
            .position(|a| a == "--dpl-neff-sat")
            .expect("the saturation point reaches the engine");
        assert_eq!(ov_args[j + 1], "0");
    }

    /// THE SATURATION POINT IS IN THE RECEIPT, NOT ONLY IN THE ARGUMENTS.
    #[test]
    fn the_receipts_record_the_saturation_point_the_run_was_given() {
        let dir = tmp_dir("dpl_neff_sat_receipt");
        let mut plan = plan_with_conditions(&["CTRL", "CTRL", "CTRL", "CASE", "CASE"]);
        plan.modalities = vec![Modality::Dpl];
        let mut args = metadata_args(&dir, &dir.join("metadata.tsv"));
        args.dpl_neff_sat = 250.0;
        let facts = DplFacts {
            model: Some("DT-EB".into()),
            robust: Some(true),
            confounded: Some(false),
            description: Some("DT-EB (limma eBayes, robust = TRUE)".into()),
        };

        let manifest_path = dir.join("run_manifest.json");
        write_manifest(
            &manifest_path,
            &plan,
            &args,
            &dir.join("sample_table.tsv"),
            ("start", "end"),
            &[],
            Some(&facts),
        )
        .expect("write the manifest");
        let manifest = std::fs::read_to_string(&manifest_path).expect("read the manifest");
        assert!(
            manifest.contains("\"dpl_neff_sat\": 250.0"),
            "the manifest names the saturation point the run was given: {manifest}"
        );

        let params_path = dir.join("model_parameters.tsv");
        write_model_parameters(&params_path, &plan, &args, Some(&facts))
            .expect("write the parameter receipt");
        let params = std::fs::read_to_string(&params_path).expect("read the receipt");
        assert!(
            params.contains("neff_sat"),
            "the parameter receipt carries a row for the saturation point: {params}"
        );
        assert!(
            params.contains("250"),
            "the row carries the value the run was given: {params}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The receipt of a WEIGHTED run has to parse the same way the receipt of an unweighted one does.
    #[test]
    fn a_weighted_receipt_still_yields_the_model_and_the_robust_flag() {
        let dir = tmp_dir("dpl_facts_weighted");
        let table = dir.join("dpl.tsv");
        let params = "model: DT-EB (limma eBayes with trend = the mean over a \
                      transcript's cells of the log2 of the summed posterior weight \
                      of the cell, and robust = TRUE; the prior variance is fitted \
                      against that read support instead of the average expression, \
                      so s2.prior and df.prior are per transcript: s2.prior median \
                      0.1, range 0.01..0.5, df.prior median 10.9, range 1..20); the \
                      fit is a two-sample linear model on the transcript x sample \
                      means whose observations carry the row-normalized \
                      Kish-saturation weights (Kish_Neff / (Kish_Neff + 100), the \
                      mean weight of a row's observed cells being 1, so a row whose \
                      cells all carry the same Kish_Neff is fitted with equal \
                      weights); its `case` coefficient is the difference of the two \
                      WEIGHTED group means; the standard error is limma's, whose \
                      df.residual is the number of OBSERVATIONS minus the rank of \
                      the design and not the effective number of observations the \
                      weights carry (per group W_g - sum(w^2)/W_g), so the weighted \
                      t is NARROWER than a strict weighted t: on this fit the median \
                      ratio of limma's standard error to the strict weighted-t \
                      standard error is 0.8864; the moderation is fitted on the \
                      testable rows only (5348 transcripts); cell admission gate: \
                      min_weight = 5";
        std::fs::write(
            &table,
            format!(
                "transcript_id\tpvalue\tconfounded_design\tmodel_parameters\n\
                 T_A1\t0.01\tFALSE\t{params}\n"
            ),
        )
        .expect("write");
        let facts = read_dpl_facts(&table).expect("the engine's shape must be readable");
        assert_eq!(
            facts.model.as_deref(),
            Some("DT-EB"),
            "the weighted receipt still names the model first"
        );
        assert_eq!(
            facts.robust,
            Some(true),
            "the weighted receipt still carries the robust = TRUE claim"
        );
        let description = facts.description.expect("the engine's sentence is there");
        assert!(
            description.ends_with(')'),
            "the receipt is a whole sentence with a closed parenthesis: {description}"
        );
        assert!(
            !description.contains("Kish"),
            "the clause is cut at the closing parenthesis of the model's own \
             parenthetical, so the weighting clause is NOT part of it: {description}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn threads_flag_forwards_to_apa_engine_cores() {
        #[derive(clap::Parser)]
        struct Cli {
            #[command(flatten)]
            args: DiffArgs,
        }
        let dir = tmp_dir("threads_apa_cores");
        let meta = dir.join("meta.tsv");
        let parsed = <Cli as clap::Parser>::try_parse_from([
            "diff",
            "--metadata",
            meta.to_str().unwrap(),
            "--output-dir",
            "out",
            "--threads",
            "8",
        ])
        .expect("parse --threads");
        assert_eq!(parsed.args.threads, 8);

        let mut plan = plan_with_conditions(&["CTRL", "CASE"]);
        plan.modalities = vec![Modality::Deg, Modality::Apa];
        let apa_args = apa_engine_args(&parsed.args, &plan);
        let idx = apa_args
            .iter()
            .position(|a| a == "--apa-engine-cores")
            .expect("--apa-engine-cores forwarded");
        assert_eq!(apa_args[idx + 1], "8");

        plan.modalities = vec![Modality::Deg];
        assert!(apa_engine_args(&parsed.args, &plan).is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn removed_legacy_flags_are_absent_from_diff_cli() {
        #[derive(clap::Parser)]
        struct Cli {
            #[command(flatten)]
            args: DiffArgs,
        }
        let mut cmd = <Cli as clap::CommandFactory>::command();
        let help = cmd.render_long_help().to_string();
        for removed in [
            "lapa",
            "--fasta",
            "--chrom-sizes",
            "--apa-engine-dir",
            "--aggregate-only",
        ] {
            assert!(
                !help.to_ascii_lowercase().contains(removed),
                "removed flag `{removed}` must not appear in `diff --help`:\n{help}"
            );
        }
    }
}
