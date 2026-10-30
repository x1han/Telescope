//! `telescope diff` — the differential-statistics layer.
//!
//! WHAT LIVES HERE AND WHAT DOES NOT.
//!
//! The statistics are NOT written here. ΔPAU (usage) comes from Bioconductor's
//! DEXSeq, ΔPAL variance moderation from limma, gene-level Δexpression from
//! DESeq2, and the alternative-poly(A) track from LAPA; all of them run
//! outside this process, and the R engine in
//! `workflow/scripts/telescope_diff.R` assembles the tables. What IS
//! written here is the one step that cannot be delegated: the per-read →
//! per-(transcript × sample) aggregation.
//!
//! WHY THE AGGREGATION CANNOT BE DELEGATED. The profile parquet is per READ —
//! one row per (read, candidate) pair, 15.1 M rows for a single sample of a
//! ten-sample cohort. R holds that as a data.frame in 3.8 GB, and both tracks
//! need it only to add it up. The CLI streams the parquet, adds it up, and hands
//! R a table with one row per (transcript × sample).
//!
//! THE AGGREGATION IS ALSO THE PSEUDOREPLICATION FIX, not an optimisation.
//! Fitting on per-read rows would inflate the effective n by ~10^6 and make
//! every p-value meaningless, so "one observation per (transcript × sample)"
//! is the definition of the statistic.
//!
//! WHAT THE AGGREGATE CARRIES. Four moments per (sample × transcript) cell:
//! `weight_sum` (posterior mass), `weighted_log2pal_sum` (mass × log2 tail
//! length), `weight_sq_sum` (sum of squared posterior weights) and
//! `weighted_log2pal_sq_sum` (mass × log2 tail length squared).
//!
//! The first two are the numerator and denominator of the posterior-weighted
//! mean log2 tail length. The last two are what turn that mean into a
//! MEASUREMENT with a variance: the Kish effective sample size is
//! `weight_sum^2 / weight_sq_sum`, the within-cell variance follows from the
//! second moment, and the variance of the cell mean is that variance divided
//! by the effective effective sample size. Without them the tail-length track
//! can only weight an observation by how many reads it has, which is a count
//! and not a precision.
//!
//! DETERMINISM. Each profile file is aggregated independently, so the sum
//! order inside a cell is the file's row order and does not depend on how many
//! samples ran concurrently. Two runs over the same inputs produce the same
//! bytes.
//!
//! THE INTERFACE. Samples are declared by a METADATA TSV (`--metadata`), one
//! row per sample, with the sample, profile and condition columns named by
//! `--sample-column`, `--profile-column` and `--condition-column`; the batch
//! and the alignment come from the optional `--batch-column` and
//! `--alignment-column` columns. `--profiles`/`--design` remain as the older
//! way of declaring the same thing.
//!
//! THE PREFLIGHT. Every precondition is checked BEFORE anything is computed
//! and before the result directory exists, and every failure says what is
//! wrong, what to do about it, and that no partial directory was left behind.
//! A run that cannot compute every modality it was asked for fails; it never
//! quietly produces fewer tables.
//!
//! THE RESULT DIRECTORY. The whole directory is built under a temporary name
//! next to `--output-dir` and moved into place in one rename, so a failed run
//! leaves no directory at all and a partial one is never visible.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use arrow_array::array::{Float64Array, Int32Array, StringArray};
use arrow_array::{Array, RecordBatch};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use rayon::prelude::*;

/// Default minimum posterior-weight mass for a (transcript × sample) cell.
/// Same default as the R engine's `--min-weight`.
pub const MIN_WEIGHT_DEFAULT: f64 = 10.0;

/// Default minimum total count for a transcript in the ΔPAU track.
/// Same default as the R engine's `--min-counts`.
pub const MIN_COUNTS_DEFAULT: f64 = 20.0;

/// Default minimum number of samples per transcript in the ΔPAL track.
///
/// Same default as the R engine's `--min-samples`. A group with a single
/// admissible cell has no within-group variance at all, and a group with two
/// has one degree of freedom; three is the smallest group size whose
/// residual variance the test can carry.
pub const MIN_SAMPLES_DEFAULT: usize = 3;

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
    /// Metadata TSV: one row per sample. The sample, profile and condition
    /// columns are named by --sample-column, --profile-column and
    /// --condition-column; the batch and the alignment come from the optional
    /// --batch-column and --alignment-column columns. A relative path inside
    /// the metadata is resolved against the metadata file's own directory.
    #[arg(long, help_heading = "INPUTS")]
    pub metadata: Option<PathBuf>,

    /// Glob matching one profile parquet per sample, e.g. 'data/*.profile.parquet'.
    /// The older way of declaring the samples; --metadata replaces it.
    #[arg(long, help_heading = "INPUTS")]
    pub profiles: Option<String>,

    /// Design TSV with columns sample_id and condition (batch and tissue are
    /// optional nuisance covariates). The older way; --metadata replaces it.
    #[arg(long, help_heading = "INPUTS")]
    pub design: Option<PathBuf>,

    /// Column of the metadata holding the sample id.
    #[arg(long, default_value = DEFAULT_SAMPLE_COLUMN, help_heading = "INPUTS")]
    pub sample_column: String,

    /// Column of the metadata holding the profile parquet.
    #[arg(long, default_value = DEFAULT_PROFILE_COLUMN, help_heading = "INPUTS")]
    pub profile_column: String,

    /// Column of the metadata holding the condition label.
    #[arg(long, default_value = DEFAULT_CONDITION_COLUMN, help_heading = "INPUTS")]
    pub condition_column: String,

    /// Column of the metadata holding the batch label. A metadata without that
    /// column is run without a batch term, and the run says so; a batch is
    /// never guessed.
    #[arg(long, default_value = DEFAULT_BATCH_COLUMN, help_heading = "INPUTS")]
    pub batch_column: String,

    /// Column of the metadata holding the sample's genomic alignment. The APA
    /// track needs one alignment per sample.
    #[arg(long, default_value = DEFAULT_ALIGNMENT_COLUMN, help_heading = "INPUTS")]
    pub alignment_column: String,

    /// TSV of `transcript_id<TAB>gene_id`, with an optional header row. It
    /// gives the transcripts their genes; without it the APA track is not
    /// identifiable unless the profiles carry a real gene_id.
    #[arg(long, help_heading = "INPUTS")]
    pub tx_to_gene: Option<PathBuf>,

    /// Annotation (GTF) the APA tool reads.
    #[arg(long, help_heading = "INPUTS")]
    pub gtf: Option<PathBuf>,

    /// Genome FASTA. Required by the APA tool.
    #[arg(long, help_heading = "INPUTS")]
    pub fasta: Option<PathBuf>,

    /// Chromosome sizes (`<name><TAB><length>` per sequence). Required by the
    /// APA tool.
    #[arg(long, help_heading = "INPUTS")]
    pub chrom_sizes: Option<PathBuf>,

    /// Directory the differential tables are written to. It is created in one
    /// step, when every requested modality has succeeded.
    #[arg(long, help_heading = "OUTPUTS")]
    pub output_dir: PathBuf,

    /// The LAPA driver (`lapa`), used by the APA track.
    #[arg(long, default_value = DEFAULT_LAPA, help_heading = "OPTIONS")]
    pub lapa_path: PathBuf,

    /// Which modalities to compute: a comma-separated list of deg, det, diu, apa
    /// and dpl.
    ///
    /// The default is all five for the --metadata interface. The older
    /// --profiles/--design interface defaults to deg,det,diu,dpl, because it
    /// declares no alignments and no annotation, so the APA track cannot be
    /// computed from it.
    #[arg(long, value_name = "MODALITIES", help_heading = "OPTIONS")]
    pub modalities: Option<String>,

    /// Minimum posterior-weight mass for a (transcript x sample) cell.
    #[arg(long, default_value_t = MIN_WEIGHT_DEFAULT, help_heading = "OPTIONS")]
    pub min_weight: f64,

    /// Minimum total count for a transcript in the usage track, and the
    /// minimum number of poly(A) reads a gene must have in a sample for that
    /// sample's poly(A) site usage to be usable (the APA track divides by
    /// that total, so a small total is a share taken over a handful of reads).
    #[arg(long, default_value_t = MIN_COUNTS_DEFAULT, help_heading = "OPTIONS")]
    pub min_counts: f64,

    /// Minimum number of usable samples per group: for a transcript in the
    /// tail-length track, and for a poly(A) site in the usage track.
    ///
    /// A group with a single usable cell has no within-group variance, so
    /// what it contributes to a two-sample test is not a variance at all.
    #[arg(long, default_value_t = MIN_SAMPLES_DEFAULT, help_heading = "OPTIONS")]
    pub min_samples: usize,

    /// R interpreter to run the engine with. Defaults to $TELESCOPE_RSCRIPT,
    /// then to `Rscript` on PATH.
    #[arg(long, help_heading = "OPTIONS")]
    pub rscript: Option<PathBuf>,

    /// The R engine itself. Defaults to $TELESCOPE_DIFF_R, then to
    /// $CONDA_PREFIX/share/telescope/workflow/scripts/telescope_diff.R, then
    /// to the copy next to this binary.
    #[arg(long, help_heading = "OPTIONS")]
    pub r_script: Option<PathBuf>,

    /// Write only the (transcript x sample) aggregate table and stop, without
    /// running the statistical engine. The table is the input every
    /// differential model takes, so it is useful on its own.
    #[arg(long, help_heading = "OPTIONS")]
    pub aggregate_only: bool,

    /// The case condition label. The reported effect is case minus control.
    #[arg(long, help_heading = "OPTIONS")]
    pub case_condition: Option<String>,

    /// The control condition label.
    #[arg(long, help_heading = "OPTIONS")]
    pub control_condition: Option<String>,

    /// The contrast as CASE:CONTROL, for example COPD:HC. An alternative to
    /// --case-condition and --control-condition.
    #[arg(long, help_heading = "OPTIONS")]
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
    /// Alternative poly(A) site usage (LAPA's own usage column, tested with
    /// limma by the engine).
    Apa,
    /// Differential poly(A) length (the audited limma model, in the engine).
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

    /// The modalities that only need the profiles and the design.
    pub fn profile_based() -> Vec<Modality> {
        vec![Modality::Deg, Modality::Det, Modality::Diu, Modality::Dpl]
    }
}

/// A preflight failure: what is wrong, and what to do about it.
///
/// The three lines are the contract's, verbatim:
///
/// ```text
/// ERROR: <what is wrong>.
/// Fix: <what to do>.
/// No partial result directory was created.
/// ```
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

/// What the preflight decided. Every field of it has been checked.
#[derive(Debug, Clone)]
pub struct Plan {
    /// The samples of the run, in the order the metadata declares them.
    pub samples: Vec<Sample>,
    /// The profile files to aggregate, in order.
    pub profiles: Vec<PathBuf>,
    /// Whether the rows of a profile are attributed to the sample that
    /// declares it. The metadata interface does; the older one does not,
    /// because there the profile names its own sample.
    pub force_sample_ids: bool,
    /// The design TSV handed to the engine.
    pub design: PathBuf,
    /// The design TSV this run wrote itself (the metadata interface), or the
    /// one the caller supplied (the older interface).
    pub design_is_written: bool,
    /// The case condition label.
    pub case_condition: Option<String>,
    /// The control condition label.
    pub control_condition: Option<String>,
    /// The modalities to compute.
    pub modalities: Vec<Modality>,
    /// The batch column, when the metadata declares a usable one.
    pub batch_column: Option<String>,
    /// What the metadata's batch column turned out to be. The column's mere
    /// presence and its usability are two different facts, and the run's
    /// receipt has to tell them apart: "no batch column" and "a batch column
    /// some sample leaves empty" are not the same statement.
    pub batch_state: BatchState,
    /// Where the batch column was declared.
    pub batch_source: BatchSource,
    /// The R interpreter.
    pub rscript: PathBuf,
    /// The R engine. `None` when the run computes no statistic at all (the
    /// aggregate-only mode), because then no engine is ever started.
    pub r_script: Option<PathBuf>,
    /// The LAPA driver, when the APA track is computed.
    pub lapa: Option<PathBuf>,
    /// The annotation, when the APA track is computed.
    pub gtf: Option<PathBuf>,
    /// The genome FASTA, when the APA track is computed.
    pub fasta: Option<PathBuf>,
    /// The chromosome sizes, when the APA track is computed.
    pub chrom_sizes: Option<PathBuf>,
}

/// The batch note the run records when the metadata has no usable batch.
///
/// The contract's wording, kept verbatim because it is what a reader of
/// `run_manifest.json` and of every table's `model_parameters` is told.
pub const NO_BATCH_NOTE: &str = "no batch column in the metadata";

/// What the metadata's batch column turned out to be.
///
/// The column's presence and its usability are different facts. A column
/// that some sample leaves empty cannot be a covariate — the design matrix
/// would be rank-deficient — but saying "no batch column in the metadata"
/// about it is false: the column is there. The receipt has to say which of
/// the two it is.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum BatchState {
    /// No batch column at all.
    #[default]
    NoColumn,
    /// A batch column, but at least one sample leaves it empty, so it cannot
    /// be a covariate.
    ColumnWithMissingValues(String),
    /// A batch column and every sample fills it in.
    ColumnPresent(String),
}

/// Where the batch column was declared: the metadata file or the design file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchSource {
    /// `--metadata` declares the samples and their conditions.
    Metadata,
    /// `--design` declares them (the older interface).
    Design,
}

impl BatchSource {
    fn as_str(&self) -> &'static str {
        match self {
            BatchSource::Metadata => "metadata",
            BatchSource::Design => "design",
        }
    }
}

impl BatchState {
    /// The note the run records.
    pub fn note(&self, source: BatchSource) -> String {
        match self {
            BatchState::NoColumn => match source {
                BatchSource::Metadata => NO_BATCH_NOTE.to_string(),
                BatchSource::Design => "no batch column in the design".to_string(),
            },
            BatchState::ColumnWithMissingValues(c) => format!(
                "the {src} has a batch column ({c}), but at least one sample leaves it \
                 empty, so it was not fitted as a covariate",
                src = source.as_str()
            ),
            BatchState::ColumnPresent(c) => format!(
                "a batch column is in the {src}: {c} (whether the batch was fitted as a \
                 covariate is decided by the engine, which reports it in \
                 logs/telescope_diff.R.log and in the model_parameters column of every table)",
                src = source.as_str()
            ),
        }
    }
}

/// The two labels of the contrast, in the order the engine needs them.
///
/// The contrast is DECLARED, never inferred. A factor sorts its levels
/// alphabetically, so a direction taken from the labels would make every
/// reported effect depend on how the labels happen to spell: with COPD and
/// HC the reference level was COPD and every effect was control minus case.
///
/// Either --case-condition/--control-condition or --contrast CASE:CONTROL
/// must be given, and they must not both be given.
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
            bail!("--contrast must be CASE:CONTROL, for example COPD:HC");
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

/// The modalities a run computes.
///
/// `--modalities` names them; without it the --metadata interface computes
/// all four and the older --profiles/--design interface computes the three
/// that need nothing but the profiles.
pub fn resolve_modalities(spec: Option<&str>, metadata_interface: bool) -> Result<Vec<Modality>> {
    let Some(spec) = spec else {
        return Ok(if metadata_interface {
            Modality::all()
        } else {
            Modality::profile_based()
        });
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

// ---------------------------------------------------------------------------
// The metadata table
// ---------------------------------------------------------------------------

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
///
/// Every failure is a preflight failure, because every failure here happens
/// before anything is computed.
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

    // Relative paths are resolved against the metadata file's own directory,
    // so a metadata table can be moved with the data it describes.
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

/// Write the design TSV the engine reads.
///
/// The columns are `sample_id`, `condition`, and `batch` when the metadata
/// declares a usable one. The batch is omitted when the column exists but
/// some sample carries no batch: a design with a hole in its batch column
/// makes a rank-deficient design matrix, and the failure would surface inside
/// R as a message about a model that cannot be fitted.
fn write_design_tsv(path: &Path, samples: &[Sample], with_batch: bool) -> Result<()> {
    let mut out = String::from("sample_id\tcondition");
    if with_batch {
        out.push_str("\tbatch");
    }
    out.push('\n');
    for s in samples {
        out.push_str(&format!(
            "{}\t{}",
            s.sample_id,
            s.condition.as_deref().unwrap_or("")
        ));
        if with_batch {
            out.push_str(&format!("\t{}", s.batch.as_deref().unwrap_or("")));
        }
        out.push('\n');
    }
    std::fs::write(path, out).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// The transcript-to-gene structure
// ---------------------------------------------------------------------------

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
///
/// The genes come from `--tx-to-gene` when it is given, and from the
/// profile's own `gene_id` column otherwise.
fn resolve_gene_structure(
    tx_to_gene: Option<&HashMap<String, String>>,
    transcript_genes: &HashMap<String, String>,
    tx_to_gene_path: Option<&Path>,
) -> Result<(), PreflightFailure> {
    let mut resolved: Vec<(String, TranscriptGene)> = Vec::new();
    for (tx, profile_gene) in transcript_genes {
        let gene = match tx_to_gene {
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
                    let where_ = tx_to_gene_path
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "--tx-to-gene".to_string());
                    return Err(PreflightFailure::new(
                        format!("the transcript '{tx}' is not in --tx-to-gene ({where_})."),
                        "add the transcript to the map, or drop --tx-to-gene and use \
                         profiles whose gene_id column holds the real genes.",
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
        // The contract fixes this message.
        return Err(PreflightFailure::new(
            "transcript-to-gene structure is absent. APA is not identifiable for this \
             input. Provide --tx-to-gene or a profile containing valid gene_id values.",
            "provide --tx-to-gene, or use profiles whose gene_id column holds the real \
             genes.",
        ));
    }
    if let Some((tx, _)) = resolved
        .iter()
        .find(|(_, g)| matches!(g, TranscriptGene::Absent | TranscriptGene::SelfId))
    {
        return Err(PreflightFailure::new(
            format!("the transcript '{tx}' has no gene: its gene is its own transcript id."),
            "provide --tx-to-gene, or use profiles whose gene_id column holds the real \
             genes.",
        ));
    }
    Ok(())
}

/// The distinct (transcript, gene) pairs a profile parquet declares.
///
/// Only the two columns are read: parquet is columnar, so the projection
/// costs a fraction of a full scan, and the check has to happen before the
/// aggregation does.
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

/// Read a `transcript_id<TAB>gene_id` TSV. The header row is optional.
fn read_tx_to_gene(path: &Path) -> Result<HashMap<String, String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut map = HashMap::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, '\t');
        let tx = parts.next().unwrap_or("").trim();
        let gene = parts.next().unwrap_or("").trim();
        if tx.is_empty() || gene.is_empty() {
            continue;
        }
        if i == 0 && tx.eq_ignore_ascii_case("transcript_id") {
            continue;
        }
        map.insert(tx.to_string(), gene.to_string());
    }
    Ok(map)
}

// ---------------------------------------------------------------------------
// One aggregated (sample × gene × transcript) cell.
// ---------------------------------------------------------------------------

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

/// String interning, so a 15 M-row scan allocates each distinct id once
/// instead of once per row.
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
///
/// `sample_id` overrides the sample the rows are attributed to. The metadata
/// declares which sample a profile belongs to, and the engine's design is
/// built from the metadata, so the override is what keeps the aggregate and
/// the design speaking about the same samples.
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

    // The sample id falls back to the file stem when the column is absent or
    // empty, which is the R engine's rule too: `file_path_sans_ext(basename(f))`.
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

        // `sample_id` is optional; an absent or fully-null column means the
        // sample is named by its file.
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
            // The profile's OWN name for this sample, read whatever the
            // metadata says. The override above decides which name the row is
            // filed under; this records what the file says about itself, so
            // the two can be compared. Reading it only in the no-override
            // branch made the comparison unreachable: the override and the
            // branch's condition are mutually exclusive, so the check below
            // never ran on the shipped path.
            if let Some(arr) = sample {
                if !arr.is_null(i) && declared_sample.is_none() {
                    declared_sample = Some(arr.value(i).to_string());
                }
            }
            let g = interner.intern(gene.value(i));
            let t = interner.intern(tx.value(i));
            let w = pw.value(i);
            // `log2(max(pal, 1))`: a zero or negative tail length must not
            // produce -inf, and 1 is the floor the R engine uses.
            let log2pal = (pal.value(i).max(1) as f64).log2();
            let e = acc.entry((s, g, t)).or_insert((0.0, 0.0, 0.0, 0.0));
            e.0 += w;
            e.1 += w * log2pal;
            e.2 += w * w;
            e.3 += w * log2pal * log2pal;
            n_rows += 1;
        }
    }

    // The metadata and the profile can disagree about the sample's name. That
    // is not a naming preference: one of the two is wrong, and filing the
    // rows under the metadata's name publishes another sample's tail lengths
    // under this sample's name. The run stops here.
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
    // Files are aggregated independently, so the sum order inside a cell does
    // not depend on how many files ran at the same time.
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

/// Aggregate every profile file and merge the cells.
pub fn aggregate_profiles(paths: &[PathBuf]) -> Result<Vec<AggRow>> {
    let per_file: Vec<Result<Vec<AggRow>>> =
        paths.par_iter().map(|p| aggregate_profile(p)).collect();
    merge_agg_rows(per_file)
}

/// Aggregate the samples' profiles, each attributed to its own sample id.
fn aggregate_samples(samples: &[Sample]) -> Result<Vec<AggRow>> {
    let per_file: Vec<Result<Vec<AggRow>>> = samples
        .par_iter()
        .map(|s| aggregate_profile_as(&s.profile, &s.sample_id))
        .collect();
    merge_agg_rows(per_file)
}

/// Write the aggregate table the R engine reads.
///
/// The floats are written with Rust's shortest round-tripping representation,
/// so the R engine reads back exactly the bits Rust summed.
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

/// Expand the `--profiles` glob into a sorted list of files.
pub fn expand_glob(pattern: &str) -> Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = Vec::new();
    for entry in
        glob::glob(pattern).with_context(|| format!("invalid --profiles glob: {pattern}"))?
    {
        let p = entry.with_context(|| format!("failed to expand --profiles glob: {pattern}"))?;
        if p.is_file() {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

/// Locate the R interpreter: explicit path, then `$TELESCOPE_RSCRIPT`, then
/// `Rscript` on PATH.
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

/// Locate the R engine: explicit path, then `$TELESCOPE_DIFF_R`, then the
/// conda data directory, then the copy shipped beside the binary.
///
/// A path the caller NAMED is checked for existence here, so a typo is
/// reported as "that file does not exist" rather than as "the engine ran and
/// failed" — the two need different responses from the operator.
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
        "cannot find the R engine. Pass --r-script <path>, set TELESCOPE_DIFF_R, or \
         install the telescope conda package (which places it at \
         $CONDA_PREFIX/share/telescope/workflow/scripts/telescope_diff.R). Looked in: {}",
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

/// Find an executable: a path with a separator in it must exist and carry the
/// unix executable bit, a bare name is looked up on PATH.
///
/// The executable bit is checked here, in the preflight, because a file that
/// is not executable is a precondition failure and the user can fix it before
/// anything is computed. Discovering it later — when the tool is spawned and
/// the spawn fails with `Permission denied` — costs the whole run.
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

// ---------------------------------------------------------------------------
// The preflight
// ---------------------------------------------------------------------------

/// Check every precondition, before anything is computed.
///
/// The order is the contract's: the metadata, the profiles, the labels, the
/// R interpreter and the engine, the transcript-to-gene structure, and then
/// the tools of the modalities that were asked for.
fn preflight(args: &DiffArgs) -> Result<Plan, PreflightFailure> {
    if args.metadata.is_some() && args.profiles.is_some() {
        return Err(PreflightFailure::new(
            "the samples are declared twice: --metadata and --profiles both name them.",
            "pass --metadata or --profiles, not both.",
        ));
    }
    if args.metadata.is_some() && args.design.is_some() {
        return Err(PreflightFailure::new(
            "the design is declared twice: --metadata carries the condition of every \
             sample, and --design declares it as well.",
            "pass --metadata or --design, not both.",
        ));
    }
    if args.metadata.is_none() && args.profiles.is_none() {
        return Err(PreflightFailure::new(
            "no samples are declared.",
            "pass --metadata <tsv>, or the older --profiles <glob> with --design <tsv>.",
        ));
    }

    // The contrast is DECLARED. It is resolved before anything is computed:
    // a run that aggregates a hundred million reads and only then finds out
    // that the contrast was missing has wasted all of it. The aggregate-only
    // mode computes no statistic, so it needs no contrast.
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

    let metadata_interface = args.metadata.is_some();
    let modalities = if args.aggregate_only {
        Vec::new()
    } else {
        resolve_modalities(args.modalities.as_deref(), metadata_interface).map_err(|e| {
            PreflightFailure::new(
                e.to_string(),
                "name the modalities as a comma-separated list of deg, det, diu, apa and dpl.",
            )
        })?
    };

    // The samples and the design.
    let (
        samples,
        profiles,
        force_sample_ids,
        design_is_written,
        batch_column,
        batch_state,
        batch_source,
        has_alignment_column,
    ) = if let Some(meta) = &args.metadata {
        let table = read_metadata(args, meta)?;
        // A batch column that some sample leaves empty cannot be a covariate:
        // the design matrix would be rank-deficient, and the failure would
        // surface inside R as a message about a model that cannot be fitted.
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
        let batch_source = BatchSource::Metadata;
        let batch_column = if usable_batch {
            Some(args.batch_column.clone())
        } else {
            None
        };
        let profiles = table.samples.iter().map(|s| s.profile.clone()).collect();
        let has_alignment_column = table.has_alignment_column;
        (
            table.samples,
            profiles,
            true,
            true,
            batch_column,
            batch_state,
            batch_source,
            has_alignment_column,
        )
    } else {
        let pattern = args.profiles.as_deref().unwrap_or_default();
        let profiles = expand_glob(pattern)
            .map_err(|e| PreflightFailure::new(format!("{e}"), "check the --profiles glob."))?;
        if profiles.is_empty() {
            return Err(PreflightFailure::new(
                format!("--profiles matched no files: {pattern}"),
                "check the --profiles glob.",
            ));
        }
        let design = args.design.clone().unwrap_or_default();
        if !design.is_file() {
            return Err(PreflightFailure::new(
                format!("the design file does not exist: {}", design.display()),
                "point --design at the TSV with the sample_id and condition columns.",
            ));
        }
        // The design is read here only to learn which samples exist and
        // whether they carry a batch: the engine reads it again.
        let tsv = Tsv::read(&design).map_err(|e| {
            PreflightFailure::new(
                format!("the design file cannot be read: {}: {e}", design.display()),
                "check that the file is a readable TSV with a header row.",
            )
        })?;
        let sample_col = tsv.column("sample_id").ok_or_else(|| {
            PreflightFailure::new(
                format!(
                    "the design has no 'sample_id' column. Its columns are: {}.",
                    tsv.header.join(", ")
                ),
                "the design needs a sample_id column and a condition column.",
            )
        })?;
        let condition_col = tsv.column("condition").ok_or_else(|| {
            PreflightFailure::new(
                format!(
                    "the design has no 'condition' column. Its columns are: {}.",
                    tsv.header.join(", ")
                ),
                "the design needs a sample_id column and a condition column.",
            )
        })?;
        let batch_col = tsv.column("batch");
        let mut samples = Vec::new();
        for row in &tsv.rows {
            let sid = tsv.value(row, sample_col);
            if sid.is_empty() {
                continue;
            }
            let profile = profiles
                .iter()
                .find(|p| profile_names_the_sample(p, &sid))
                .cloned()
                .ok_or_else(|| {
                    PreflightFailure::new(
                        format!("the design names the sample '{sid}' but no profile file does."),
                        "name the profile files after their samples, or use --metadata.",
                    )
                })?;
            samples.push(Sample {
                sample_id: sid,
                profile,
                condition: Some(tsv.value(row, condition_col)),
                batch: batch_col.map(|c| tsv.value(row, c)),
                alignment: None,
            });
        }
        // The design's batch column, if it has one, is usable only when every
        // row fills it in.
        let batch_values: Vec<String> = batch_col
            .map(|c| tsv.rows.iter().map(|r| tsv.value(r, c)).collect())
            .unwrap_or_default();
        let usable_batch = !batch_values.is_empty() && batch_values.iter().all(|v| !v.is_empty());
        let batch_column = if usable_batch {
            Some("batch".to_string())
        } else {
            None
        };
        let batch_state = if batch_values.is_empty() {
            BatchState::NoColumn
        } else if usable_batch {
            BatchState::ColumnPresent("batch".to_string())
        } else {
            BatchState::ColumnWithMissingValues("batch".to_string())
        };
        let batch_source = BatchSource::Design;
        (
            samples,
            profiles,
            false,
            false,
            batch_column,
            batch_state,
            batch_source,
            false,
        )
    };

    // Every profile path exists.
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

    // The case and the control labels are both present, and they differ.
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
        // A sample that carries a third label has no place in a two-group
        // contrast. This is checked HERE, before anything is computed: the
        // engine rejects it later, after the whole aggregation has been done,
        // and the failure then reads as a bug in the engine instead of a
        // mistake in the metadata.
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
    // An empty condition cell is a hole in the design, not a third group.
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

    // The R interpreter and the R engine. A run that only aggregates never
    // starts R, so it must not insist on an R installation: the aggregate
    // table is the input every differential model takes, and it is useful on
    // its own.
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
                "point --r-script at telescope_diff.R, set TELESCOPE_DIFF_R, or install the \
                 telescope conda package.",
            )
        })?)
    };

    // The transcript-to-gene structure, when the APA track is asked for.
    let needs_genes = modalities.contains(&Modality::Apa);
    if needs_genes {
        let map = match &args.tx_to_gene {
            Some(p) => {
                if !p.is_file() {
                    return Err(PreflightFailure::new(
                        format!(
                            "the transcript-to-gene file does not exist: {}",
                            p.display()
                        ),
                        "point --tx-to-gene at the transcript_id<TAB>gene_id TSV.",
                    ));
                }
                read_tx_to_gene(p).map_err(|e| {
                    PreflightFailure::new(
                        format!("the transcript-to-gene file cannot be read: {e}"),
                        "check that every line is `transcript_id<TAB>gene_id`.",
                    )
                })?
            }
            None => HashMap::new(),
        };
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
        resolve_gene_structure(
            if args.tx_to_gene.is_some() {
                Some(&map)
            } else {
                None
            },
            &genes,
            args.tx_to_gene.as_deref(),
        )?;
    }

    // The tools of the modalities that were asked for.
    if modalities.contains(&Modality::Apa) && !metadata_interface {
        return Err(PreflightFailure::new(
            "the APA track reads one alignment per sample, and the older \
             --profiles/--design interface declares none.",
            "use --metadata, whose alignment column names each sample's alignment.",
        ));
    }
    let lapa = if modalities.contains(&Modality::Apa) {
        let driver = find_executable(&args.lapa_path).ok_or_else(|| {
            // The contract fixes this message.
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

    let gtf = args
        .gtf
        .clone()
        .filter(|_| modalities.contains(&Modality::Apa));

    Ok(Plan {
        samples,
        profiles,
        force_sample_ids,
        design: args.design.clone().unwrap_or_default(),
        design_is_written,
        case_condition: contrast.as_ref().map(|c| c.0.clone()),
        control_condition: contrast.as_ref().map(|c| c.1.clone()),
        modalities,
        batch_column,
        batch_state,
        batch_source,
        rscript,
        r_script,
        lapa,
        gtf,
        fasta: args.fasta.clone(),
        chrom_sizes: args.chrom_sizes.clone(),
    })
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

// ---------------------------------------------------------------------------
// The external tools
// ---------------------------------------------------------------------------

/// Run LAPA for the APA track, once for the whole cohort.
///
/// THE INVOCATION, read out of `lapa --help` of the installed LAPA:
///
/// ```text
/// <lapa> --alignment <samples.csv> --fasta <fasta> --annotation <gtf> \
///         --chrom_sizes <chrom_sizes> --output_dir <dir> --counting_method end
/// ```
///
/// `--alignment` takes either a BAM or a CSV of `sample,dataset,path`. The
/// CSV is used here, so that LAPA sees the whole cohort at once and groups
/// the replicates of a condition together: `dataset` is the sample's
/// condition and `path` is that sample's alignment. The sample column is
/// what names the file LAPA writes, so it is the metadata's sample id and
/// the R engine can join the file to the design.
///
/// LAPA writes the clustered poly(A) sites of a sample to
/// `<output_dir>/sample/<sample>.bed`, and also writes the unreplicated
/// clusters to `<output_dir>/raw_sample/` and the dataset-level clusters to
/// the top of the output directory. The R engine reads the canonical
/// `sample/` layout, so the per-sample files are moved into a directory of
/// their own and that directory is what the engine is given.
fn run_lapa(
    driver: &Path,
    fasta: &Path,
    gtf: &Path,
    chrom_sizes: &Path,
    samples: &[Sample],
    dir: &Path,
    log: &Path,
    surviving: &Path,
) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;

    // The CSV LAPA reads: one row per sample, the condition as the dataset
    // so LAPA groups the replicates, and the alignment as the path.
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
    run_tool(
        driver,
        &[
            "--alignment".to_string(),
            csv.display().to_string(),
            "--fasta".to_string(),
            fasta.display().to_string(),
            "--annotation".to_string(),
            gtf.display().to_string(),
            "--chrom_sizes".to_string(),
            chrom_sizes.display().to_string(),
            "--output_dir".to_string(),
            run_dir.display().to_string(),
            "--counting_method".to_string(),
            "end".to_string(),
        ],
        "LAPA",
        log,
        surviving,
    )?;

    // The canonical layout the R engine reads: `sample/<sample>.bed`, and
    // nothing else, so the engine cannot pick up the unreplicated clusters
    // or the dataset-level ones by accident.
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
        // LAPA writes the file under the sample id it was given in the CSV,
        // which is the raw id. The sanitised spelling is accepted as well,
        // because the two differ only for a sample id that carries a
        // character a file name may not hold, and LAPA's own naming has
        // changed across releases.
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
    Ok(canonical)
}

/// Does this profile file name the sample?
///
/// A profile file is named after its sample, but the convention in this
/// repository appends `.profile` to the sample id: `early_Rep1.profile.parquet`
/// is the profile of `early_Rep1`, and its stem is `early_Rep1.profile`. So
/// the bare stem and the stem without a trailing `.profile` both name a
/// sample.
fn profile_names_the_sample(path: &Path, sample_id: &str) -> bool {
    let Some(stem) = path.file_stem().map(|s| s.to_string_lossy().into_owned()) else {
        return false;
    };
    if stem == sample_id {
        return true;
    }
    match stem.strip_suffix(".profile") {
        Some(trimmed) => trimmed == sample_id,
        None => false,
    }
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

/// Run a command, keep its output in the log, and fail when it does not exit
/// zero.
///
/// `log` is where the output goes while the run lasts; `surviving` is where the
/// log is kept when the run fails, because the run's own directory is removed
/// on the way out and a message that points at a deleted file is no message at
/// all.
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

// ---------------------------------------------------------------------------
// The result directory
// ---------------------------------------------------------------------------

/// A temporary directory that is removed unless the run moves its contents
/// out of it.
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

/// Write `run_manifest.json`.
fn write_manifest(
    path: &Path,
    plan: &Plan,
    args: &DiffArgs,
    design: &Path,
    start: &str,
    end: &str,
    versions: &[(String, String)],
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
        json_opt(
            args.metadata
                .as_ref()
                .map(|p| p.display().to_string())
                .as_deref()
        )
    ));
    json.push_str(&format!(
        "    \"profiles_glob\": {},\n",
        json_opt(args.profiles.as_deref())
    ));
    json.push_str(&format!(
        "    \"design\": {},\n",
        json_string(&design.display().to_string())
    ));
    json.push_str(&format!(
        "    \"tx_to_gene\": {},\n",
        json_opt(
            args.tx_to_gene
                .as_ref()
                .map(|p| p.display().to_string())
                .as_deref()
        )
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
    // The commit of the tree the binary was built from. It is read from the
    // environment, and it is OMITTED when the environment does not carry it:
    // a guessed commit is worse than no commit.
    match std::env::var("GIT_COMMIT") {
        Ok(c) if !c.trim().is_empty() => {
            json.push_str(&format!("  \"git_commit\": {},\n", json_string(c.trim())));
        }
        _ => {}
    }
    json.push_str("  \"min_weight\": ");
    json.push_str(&format!("{:?}", args.min_weight));
    json.push_str(",\n  \"min_counts\": ");
    json.push_str(&format!("{:?}", args.min_counts));
    json.push_str(",\n  \"min_samples\": ");
    json.push_str(&format!("{}", args.min_samples));
    json.push_str("\n}\n");
    std::fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// The batch note the manifest and the model parameters record.
///
/// When the metadata carries a batch column, the CLI does not know whether the
/// batch was used: the engine decides that, because only the engine can see
/// whether the batch is confounded with the condition. The note says so
/// instead of implying that the batch was used.
fn batch_note(plan: &Plan) -> String {
    plan.batch_state.note(plan.batch_source)
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
fn write_model_parameters(path: &Path, plan: &Plan, args: &DiffArgs) -> Result<()> {
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
                // The base fit is NOT limma: it is the unweighted two-sample
                // t on the sample means (`equal_raw_t`). limma supplies the
                // moderation arm (`equal_weight_eb`).
                //
                // The model name is NOT written here as a single value: the
                // engine decides it, because only the engine knows whether
                // the batch was fitted (a batch that is confounded with the
                // condition is present but not fitted). The per-row
                // model_name of dpl.tsv carries the name; this receipt points
                // at it rather than guessing it.
                rows.push((
                    modality.clone(),
                    "model_name".into(),
                    "see the model_name column of dpl.tsv: equal_raw_t, or \
                     batch_adjusted_raw_t when the engine fitted a batch"
                        .into(),
                ));
                rows.push((
                    modality.clone(),
                    "moderated_arm".into(),
                    "equal_weight_eb (limma eBayes)".into(),
                ));
            }
            Modality::Apa => {
                rows.push((modality.clone(), "model_name".into(), "LAPA+limma".into()));
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
            }
        }
        rows.push((
            modality.clone(),
            "min_weight".into(),
            format!("{:?}", args.min_weight),
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
        // The batch note is in the parameters of EVERY table, so a reader of
        // one table can tell whether a batch term was fitted.
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
///
/// A tool that does not answer `--version` is recorded as `unknown`: the
/// version is metadata, and a missing version must not fail a run whose
/// result is otherwise complete.
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
///
/// The civil-from-days algorithm of Howard Hinnant: the days are counted
/// from 1970-03-01, so the leap day lands at the end of a 400-year cycle.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

/// Run the differential layer.
pub fn run(args: &DiffArgs) -> Result<()> {
    // THE PREFLIGHT RUNS FIRST, and a failure of it is the contract's
    // three-line message on stderr with a non-zero exit: nothing has been
    // computed and no result directory exists.
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

    // The whole run happens under a temporary name NEXT TO the output
    // directory, so the final move is a rename on one filesystem.
    let tmp = parent.join(temp_name("telescope-diff"));
    let work = tmp.join("work");
    let result = tmp.join("result");
    std::fs::create_dir_all(&work)
        .with_context(|| format!("failed to create {}", work.display()))?;
    std::fs::create_dir_all(&result)
        .with_context(|| format!("failed to create {}", result.display()))?;
    // The temporary directory is removed when it goes out of scope, so a
    // failure leaves nothing behind at all.
    let _temp = TempDir { path: tmp.clone() };

    let logs = result.join("logs");
    std::fs::create_dir_all(&logs)
        .with_context(|| format!("failed to create {}", logs.display()))?;

    // The log of a failed run has to outlive the run. The temporary directory
    // is removed on the way out, so a message that points at a log inside it
    // points at a file that is already gone. The copy lives NEXT TO the
    // output directory, never inside it: after a failed run the output
    // directory must not exist at all.
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
///
/// The output directory itself is left alone: it must not exist after a failed
/// run, and a diagnostic is not a result.
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

    // 1. The aggregate.
    let rows = if plan.force_sample_ids {
        aggregate_samples(&plan.samples)?
    } else {
        aggregate_profiles(&plan.profiles)?
    };
    let agg_path = work.join("telescope_diff_aggregate.tsv");
    write_aggregate_tsv(&agg_path, &rows)?;
    eprintln!(
        "[telescope]  aggregate: {} (transcript x sample) cells -> {}",
        rows.len(),
        agg_path.display()
    );

    // 2. The design.
    let design_path = if plan.design_is_written {
        let path = work.join("telescope_diff_design.tsv");
        let with_batch = plan.batch_column.is_some();
        write_design_tsv(&path, &plan.samples, with_batch)?;
        eprintln!("[telescope]  design: {}", path.display());
        path
    } else {
        plan.design.clone()
    };

    // 3. The tools.
    let mut lapa_dir: Option<PathBuf> = None;
    if plan.modalities.contains(&Modality::Apa) {
        let driver = plan.lapa.as_ref().expect("the preflight found LAPA");
        let gtf = plan.gtf.as_ref().expect("the preflight found the GTF");
        let fasta = plan.fasta.as_ref().expect("the preflight found the FASTA");
        let chrom_sizes = plan
            .chrom_sizes
            .as_ref()
            .expect("the preflight found the chromosome sizes");
        let path = run_lapa(
            driver,
            fasta,
            gtf,
            chrom_sizes,
            &plan.samples,
            &work.join("lapa"),
            &logs.join("lapa.log"),
            surviving,
        )?;
        eprintln!("[telescope]  LAPA output directory: {}", path.display());
        lapa_dir = Some(path);
    }

    // 4. The engine.
    if args.aggregate_only {
        // The aggregate table is the output of this mode, so it goes into the
        // result directory itself.
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
            &design_path,
            result,
            lapa_dir.as_deref(),
            surviving,
        )?;
        // Every requested modality must have produced its table. A table that
        // is missing means the engine did not do what it was asked, and the
        // run fails rather than shipping a directory with a hole in it.
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

    // 5. The provenance files.
    let versions = collect_versions(args, plan);
    write_software_versions(&result.join("software_versions.tsv"), &versions)?;
    write_model_parameters(&result.join("model_parameters.tsv"), plan, args)?;
    // The design the engine used. When the run wrote it itself, the file
    // lives in the work directory, which is removed when the run ends, so a
    // receipt naming it names a path that no longer exists. The design is
    // copied into the result directory and the receipt names that copy --
    // relative to the result directory, which is what the reader of the
    // result directory holds.
    let manifest_design = if plan.design_is_written {
        let copy = logs.join("design.tsv");
        std::fs::copy(&design_path, &copy).with_context(|| {
            format!(
                "failed to copy the design {} into {}",
                design_path.display(),
                copy.display()
            )
        })?;
        PathBuf::from("logs/design.tsv")
    } else {
        design_path.clone()
    };
    let end = utc_timestamp();
    write_manifest(
        &result.join("run_manifest.json"),
        plan,
        args,
        &manifest_design,
        start,
        &end,
        &versions,
    )?;
    // The engine's session info is a log, and the result directory keeps the
    // logs together.
    let session = result.join("sessionInfo.txt");
    if session.is_file() {
        let _ = std::fs::rename(&session, logs.join("sessionInfo.txt"));
    }

    // 6. The move. One rename, so the directory appears complete or not at
    //    all.
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

/// The engine's modality flags: the tables this run asked for, and the ones it
/// did not.
///
/// ALL THREE ARE NAMED, the ones this run did not ask for with the value
/// `false`. The engine's flags are three-valued: absent means "not named", and
/// a run that names none of them gets the engine's historical det+dpl
/// behaviour. A run that asked for APA alone would therefore have got the usage
/// and length tracks as well — work nobody asked for, and a failure in a track
/// that has nothing to do with the request. `optparse`'s store_true flag takes
/// the token that follows it, so `--det false` reads back as FALSE and
/// `--det --dpl` still reads back as TRUE.
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
    design: &Path,
    out_dir: &Path,
    lapa_dir: Option<&Path>,
    surviving: &Path,
) -> Result<()> {
    // THE SCRIPT IS THE FIRST ARGUMENT. `Rscript <script> <options>` is
    // Rscript's own calling convention, and an argument list that starts
    // with `--aggregate` makes Rscript take the option as the script name.
    let mut engine_args: Vec<String> = vec![
        plan.r_script
            .as_ref()
            .expect("the preflight resolved the R engine")
            .display()
            .to_string(),
        "--aggregate".to_string(),
        aggregate.display().to_string(),
        "--design".to_string(),
        design.display().to_string(),
        "--output-dir".to_string(),
        out_dir.display().to_string(),
        "--min-weight".to_string(),
        args.min_weight.to_string(),
        "--min-counts".to_string(),
        args.min_counts.to_string(),
        "--min-samples".to_string(),
        args.min_samples.to_string(),
    ];
    if let (Some(case), Some(control)) = (&plan.case_condition, &plan.control_condition) {
        engine_args.push("--case-condition".to_string());
        engine_args.push(case.clone());
        engine_args.push("--control-condition".to_string());
        engine_args.push(control.clone());
    }
    // The three profile-based tables are named, so the engine writes exactly
    // the ones this run asked for.
    engine_args.extend(engine_modality_flags(&plan.modalities));
    if let Some(d) = lapa_dir {
        engine_args.push("--lapa-dir".to_string());
        engine_args.push(d.display().to_string());
    }
    let log = out_dir.join("logs").join("telescope_diff.R.log");
    run_tool(&plan.rscript, &engine_args, "the R engine", &log, surviving)?;
    Ok(())
}

/// The versions of the tools that ran.
/// The version of the installed LAPA.
///
/// LAPA is a Python package with a click-based command line, and it has no
/// `--version` option: `lapa --version` fails with "No such option". The
/// version is a property of the installed distribution, so it is read from the
/// distribution metadata with the Python that owns LAPA -- the interpreter
/// next to the `lapa` executable, which is the one that can import it.
///
/// When that cannot be done the receipt says so rather than guessing: an
/// invented version number in `software_versions.tsv` is worse than an
/// explicit "unknown".
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
///
/// An existing directory is moved aside first, then removed once the new one
/// is in place. If the second move fails, the old one goes back: the output
/// directory is either the old result or the new one, never neither.
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
                // Put the old directory back: the output directory must not
                // be left missing because the new one could not be moved in.
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

        // T1: mass 0.75. `weighted_log2pal_sum` is the NUMERATOR of the
        // posterior-weighted mean log2 tail length, i.e. sum(w * log2(pal));
        // the mean itself is that divided by the mass.
        let t1 = rows.iter().find(|r| r.transcript_id == "T1").unwrap();
        assert!((t1.weight_sum - 0.75).abs() < 1e-12);
        let num = 0.5 * 100f64.log2() + 0.25 * 200f64.log2();
        assert!((t1.weighted_log2pal_sum - num).abs() < 1e-12);
        let mean = t1.weighted_log2pal_sum / t1.weight_sum;
        assert!((mean - num / 0.75).abs() < 1e-12);

        // T2: mass 1.0, so numerator and mean are both log2(50)
        let t2 = rows.iter().find(|r| r.transcript_id == "T2").unwrap();
        assert!((t2.weight_sum - 1.0).abs() < 1e-12);
        assert!((t2.weighted_log2pal_sum - 50f64.log2()).abs() < 1e-12);

        // The two extra moments are what make the cell mean a measurement with
        // a variance: the Kish effective sample size is W^2 / sum(w^2), and
        // the second moment gives the within-cell spread.
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
        // log2(max(0, 1)) = 0
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
        // The shortest round-tripping form, so R reads back the same bits.
        assert_eq!(f[3].parse::<f64>().unwrap(), 0.1 + 0.2);
        assert_eq!(f[4].parse::<f64>().unwrap(), 1.0 / 3.0);
        assert_eq!(f[5].parse::<f64>().unwrap(), 0.05);
        assert_eq!(f[6].parse::<f64>().unwrap(), 1.0 / 7.0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_contrast_is_taken_from_the_flags_verbatim_and_not_from_the_label_order() {
        let (case, control) = resolve_contrast(Some("COPD"), Some("HC"), None).expect("contrast");
        assert_eq!(case, "COPD");
        assert_eq!(control, "HC");
        // The reverse spelling must not be silently normalised: the caller
        // says which label is the case.
        let (case, control) = resolve_contrast(Some("HC"), Some("COPD"), None).expect("contrast");
        assert_eq!(case, "HC");
        assert_eq!(control, "COPD");
    }

    #[test]
    fn the_contrast_can_be_given_as_case_colon_control() {
        let (case, control) = resolve_contrast(None, None, Some("COPD:HC")).expect("contrast");
        assert_eq!(case, "COPD");
        assert_eq!(control, "HC");
    }

    #[test]
    fn a_missing_or_ambiguous_contrast_is_refused_before_anything_is_computed() {
        assert!(resolve_contrast(None, None, None).is_err());
        assert!(resolve_contrast(Some("COPD"), None, None).is_err());
        assert!(resolve_contrast(None, Some("HC"), None).is_err());
        assert!(resolve_contrast(Some("COPD"), Some("HC"), Some("COPD:HC")).is_err());
        assert!(resolve_contrast(None, None, Some("COPD")).is_err());
        assert!(resolve_contrast(None, None, Some("COPD:")).is_err());
        assert!(resolve_contrast(Some("HC"), Some("HC"), None).is_err());
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
        // The candidates are searched in order and the first existing one wins.
        // Whatever this machine has, the answer must be a file that exists.
        if let Ok(p) = resolve_r_script(None) {
            assert!(p.is_file(), "resolved to a missing file: {}", p.display());
        }
    }

    // -- the new interface --------------------------------------------------

    /// The metadata declares the samples, and a relative profile path is
    /// resolved against the metadata file's own directory.
    #[test]
    fn the_metadata_declares_the_samples_and_resolves_relative_paths() {
        let dir = tmp_dir("meta");
        std::fs::create_dir_all(dir.join("profiles")).expect("mkdir");
        let meta = dir.join("metadata.tsv");
        std::fs::write(
            &meta,
            "sample_id\tprofile_path\tcondition\tbatch\talignment_path\n\
             S1\tprofiles/S1.parquet\tHC\tb1\taln/S1.bam\n\
             S2\tprofiles/S2.parquet\tCOPD\tb2\taln/S2.bam\n",
        )
        .expect("write");
        let args = DiffArgs {
            metadata: Some(meta.clone()),
            profiles: None,
            design: None,
            sample_column: DEFAULT_SAMPLE_COLUMN.into(),
            profile_column: DEFAULT_PROFILE_COLUMN.into(),
            condition_column: DEFAULT_CONDITION_COLUMN.into(),
            batch_column: DEFAULT_BATCH_COLUMN.into(),
            alignment_column: DEFAULT_ALIGNMENT_COLUMN.into(),
            tx_to_gene: None,
            gtf: None,
            fasta: None,
            chrom_sizes: None,
            output_dir: dir.join("out"),
            lapa_path: PathBuf::from(DEFAULT_LAPA),
            modalities: None,
            min_weight: MIN_WEIGHT_DEFAULT,
            min_counts: MIN_COUNTS_DEFAULT,
            min_samples: MIN_SAMPLES_DEFAULT,
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

    /// A metadata without the named column is refused, and the refusal says
    /// which column is missing.
    #[test]
    fn a_metadata_without_the_named_column_is_refused() {
        let dir = tmp_dir("meta_bad");
        let meta = dir.join("metadata.tsv");
        std::fs::write(&meta, "sample_id\tcondition\nS1\tHC\n").expect("write");
        let args = DiffArgs {
            metadata: Some(meta.clone()),
            profiles: None,
            design: None,
            sample_column: DEFAULT_SAMPLE_COLUMN.into(),
            profile_column: DEFAULT_PROFILE_COLUMN.into(),
            condition_column: DEFAULT_CONDITION_COLUMN.into(),
            batch_column: DEFAULT_BATCH_COLUMN.into(),
            alignment_column: DEFAULT_ALIGNMENT_COLUMN.into(),
            tx_to_gene: None,
            gtf: None,
            fasta: None,
            chrom_sizes: None,
            output_dir: dir.join("out"),
            lapa_path: PathBuf::from(DEFAULT_LAPA),
            modalities: None,
            min_weight: MIN_WEIGHT_DEFAULT,
            min_counts: MIN_COUNTS_DEFAULT,
            min_samples: MIN_SAMPLES_DEFAULT,
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
        let all = resolve_modalities(None, true).expect("default");
        assert_eq!(all.len(), 5);
        assert_eq!(all[0], Modality::Deg);
        assert_eq!(all[1], Modality::Det);
        assert_eq!(all[2], Modality::Diu);
        assert_eq!(all[3], Modality::Apa);
        assert_eq!(all[4], Modality::Dpl);
        // The older interface cannot reach the track that needs the
        // alignments, so it computes the other four.
        let legacy = resolve_modalities(None, false).expect("default");
        assert_eq!(
            legacy,
            vec![Modality::Deg, Modality::Det, Modality::Diu, Modality::Dpl]
        );
        let some = resolve_modalities(Some("dpl,deg"), true).expect("named");
        assert_eq!(some, vec![Modality::Deg, Modality::Dpl]);
        assert!(resolve_modalities(Some("deg,nonsense"), true).is_err());
        // The DIU track is the DEXSeq usage test and it is one of the five.
        assert_eq!(
            resolve_modalities(Some("diu"), true).expect("named"),
            vec![Modality::Diu]
        );
        assert!(resolve_modalities(Some(""), true).is_err());
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
             input. Provide --tx-to-gene or a profile containing valid gene_id values.",
            "provide --tx-to-gene, or use profiles whose gene_id column holds the real genes.",
        );
        assert!(genes.message().starts_with(
            "ERROR: transcript-to-gene structure is absent. APA is not identifiable \
             for this input. Provide --tx-to-gene or a profile containing valid gene_id \
             values."
        ));
        assert!(genes
            .message()
            .ends_with("No partial result directory was created."));
    }

    /// A transcriptome whose gene is its own transcript id has no gene
    /// structure, and the run says so with the contract's message.
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

    /// A transcript missing from --tx-to-gene is named.
    #[test]
    fn a_transcript_missing_from_the_gene_map_is_named() {
        let mut genes = HashMap::new();
        genes.insert("T1".to_string(), "G1".to_string());
        genes.insert("T9".to_string(), "G9".to_string());
        let mut map = HashMap::new();
        map.insert("T1".to_string(), "G1".to_string());
        let err = resolve_gene_structure(Some(&map), &genes, Some(Path::new("tx.tsv")))
            .expect_err("must fail");
        assert!(err.what.contains("'T9'"), "got: {}", err.what);
        assert!(err.fix.contains("--tx-to-gene"), "got: {}", err.fix);
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
        // The epoch itself, and a day that is not a leap day.
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
    ///
    /// A run that asked for APA alone must still say "no deg, no det, no diu,
    /// no dpl": the engine's flags are three-valued, and a run that names
    /// none of them gets every table the engine can write.
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

    /// The log of a failed run outlives the run, and it lives next to the
    /// output directory, never inside it.
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

        // A run that failed after a tool wrote its log: the log is kept.
        let logs = dir.join("logs");
        std::fs::create_dir_all(&logs).expect("mkdir");
        std::fs::write(logs.join("telescope_diff.R.log"), "the engine said no\n").expect("write");
        preserve_logs(&logs, &surviving);
        let text = std::fs::read_to_string(&surviving).expect("the log was kept");
        assert!(text.contains("the engine said no"), "{text}");
        assert!(text.contains("telescope_diff.R.log"), "{text}");

        // A run that failed before any tool ran: nothing is written.
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

    /// The aggregate-only mode computes no statistic, so it needs neither an R
    /// interpreter nor the engine.
    #[test]
    fn the_aggregate_only_mode_needs_neither_r_nor_the_engine() {
        let dir = tmp_dir("agg_only");
        let meta = dir.join("metadata.tsv");
        std::fs::write(
            &meta,
            "sample_id\tprofile_path\tcondition\n\
             S1\tS1.parquet\tHC\n\
             S2\tS2.parquet\tCOPD\n",
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
            metadata: Some(meta.clone()),
            profiles: None,
            design: None,
            sample_column: DEFAULT_SAMPLE_COLUMN.into(),
            profile_column: DEFAULT_PROFILE_COLUMN.into(),
            condition_column: DEFAULT_CONDITION_COLUMN.into(),
            batch_column: DEFAULT_BATCH_COLUMN.into(),
            alignment_column: DEFAULT_ALIGNMENT_COLUMN.into(),
            tx_to_gene: None,
            gtf: None,
            fasta: None,
            chrom_sizes: None,
            output_dir: dir.join("out"),
            lapa_path: PathBuf::from(DEFAULT_LAPA),
            modalities: None,
            min_weight: MIN_WEIGHT_DEFAULT,
            min_counts: MIN_COUNTS_DEFAULT,
            min_samples: MIN_SAMPLES_DEFAULT,
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

        // The same run that computes a statistic does need R, and says so.
        let mut with_statistic = args;
        with_statistic.aggregate_only = false;
        with_statistic.case_condition = Some("COPD".to_string());
        with_statistic.control_condition = Some("HC".to_string());
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
            metadata: Some(meta.to_path_buf()),
            profiles: None,
            design: None,
            sample_column: DEFAULT_SAMPLE_COLUMN.into(),
            profile_column: DEFAULT_PROFILE_COLUMN.into(),
            condition_column: DEFAULT_CONDITION_COLUMN.into(),
            batch_column: DEFAULT_BATCH_COLUMN.into(),
            alignment_column: DEFAULT_ALIGNMENT_COLUMN.into(),
            tx_to_gene: None,
            gtf: None,
            fasta: None,
            chrom_sizes: None,
            output_dir: dir.join("out"),
            lapa_path: PathBuf::from(DEFAULT_LAPA),
            modalities: None,
            min_weight: MIN_WEIGHT_DEFAULT,
            min_counts: MIN_COUNTS_DEFAULT,
            min_samples: MIN_SAMPLES_DEFAULT,
            rscript: None,
            r_script: None,
            aggregate_only: false,
            case_condition: None,
            control_condition: None,
            contrast: None,
        }
    }

    /// A metadata table with three samples and the given condition labels,
    /// and the three profile files it points at.
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

    /// A profile parquet that declares a sample name, and the metadata that
    /// calls the same file something else, must not be published: the rows
    /// would be filed under the metadata's name, which is not this sample's.
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

    /// The batch note has to tell the three states apart: no column at all, a
    /// column some sample leaves empty, and a column everyone fills in. The
    /// first two used to share a wording, and the second one is not the
    /// first one.
    #[test]
    fn the_batch_note_says_which_state_the_column_is_in() {
        assert_eq!(
            BatchState::NoColumn.note(BatchSource::Metadata),
            NO_BATCH_NOTE
        );
        let holes = BatchState::ColumnWithMissingValues("batch".into()).note(BatchSource::Metadata);
        assert!(
            holes.contains("at least one sample leaves it empty"),
            "a column with holes is not 'no batch column': {holes}"
        );
        assert!(!holes.contains(NO_BATCH_NOTE), "{holes}");
        let present = BatchState::ColumnPresent("batch".into()).note(BatchSource::Metadata);
        assert!(
            present.contains("a batch column is in the metadata"),
            "{present}"
        );
        assert!(
            present.contains("decided by the engine"),
            "the CLI cannot know whether the engine fitted the batch: {present}"
        );
    }

    /// A third condition label has no place in a two-group contrast, and the
    /// run has to say so BEFORE it aggregates anything.
    #[test]
    fn a_third_condition_label_is_refused_before_anything_is_computed() {
        let dir = tmp_dir("preflight_third_label");
        let meta = write_metadata_with_conditions(&dir, &["HC", "COPD", "OTHER"]);
        let mut args = metadata_args(&dir, &meta);
        args.case_condition = Some("COPD".into());
        args.control_condition = Some("HC".into());
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
        let meta = write_metadata_with_conditions(&dir, &["HC", "COPD", ""]);
        let mut args = metadata_args(&dir, &meta);
        args.case_condition = Some("COPD".into());
        args.control_condition = Some("HC".into());
        let failure = preflight(&args).expect_err("an empty condition must be refused");
        let text = failure.message();
        assert!(text.contains("has no condition"), "{text}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A file without the unix executable bit is not a tool. Finding that
    /// out when the tool is spawned costs the whole run; finding it out in
    /// the preflight costs nothing.
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
}
