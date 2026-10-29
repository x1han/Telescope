//! `telescope diff` — the differential-statistics layer.
//!
//! WHAT LIVES HERE AND WHAT DOES NOT.
//!
//! The statistics are NOT written here. ΔPAU (usage) comes from Bioconductor's
//! DEXSeq and the ΔPAL variance moderation from limma; both run in an R
//! subprocess against the published engine in `workflow/scripts/telescope_diff.R`.
//! What IS written here is the one step that cannot be delegated: the
//! per-read → per-(transcript × sample) aggregation.
//!
//! WHY THE AGGREGATION CANNOT BE DELEGATED. The profile parquet is per READ —
//! one row per (read, candidate) pair, 15.1 M rows for a single sample of the
//! Tongji cohort. R holds that as a data.frame in 3.8 GB, and both tracks need
//! it only to add it up. The CLI streams the parquet, adds it up, and hands R
//! a table with one row per (transcript × sample).
//!
//! THE AGGREGATION IS ALSO THE PSEUDOREPLICATION FIX, not an optimisation.
//! Fitting on per-read rows would inflate the effective n by ~10^6 and make
//! every p-value meaningless, so "one observation per (transcript × sample)"
//! is the definition of the statistic.
//!
//! WHAT THE AGGREGATE CARRIES. `weight_sum` is the posterior mass assigned to
//! the transcript in that sample: ΔPAU uses its rounded value as the count,
//! and ΔPAL uses it as the observation's precision. `weighted_log2pal_sum` is
//! the numerator of the posterior-weighted mean log2 tail length, which the
//! ΔPAL fit divides by `weight_sum`.
//!
//! DETERMINISM. Each profile file is aggregated independently, so the sum
//! order inside a cell is the file's row order and does not depend on how many
//! samples ran concurrently. Two runs over the same inputs produce the same
//! bytes.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use arrow_array::array::{Float64Array, Int32Array, StringArray};
use arrow_array::{Array, RecordBatch};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;

/// Default minimum posterior-weight mass for a (transcript × sample) cell.
/// Same default as the R engine's `--min-weight`.
pub const MIN_WEIGHT_DEFAULT: f64 = 10.0;

/// Default minimum total count for a transcript in the ΔPAU track.
/// Same default as the R engine's `--min-counts`.
pub const MIN_COUNTS_DEFAULT: f64 = 20.0;

/// Arguments of the `diff` subcommand.
#[derive(clap::Args, Debug)]
pub struct DiffArgs {
    /// Glob matching one profile parquet per sample, e.g. 'data/*.profile.parquet'.
    #[arg(long)]
    pub profiles: String,

    /// Design TSV with columns sample_id and condition (batch and tissue are
    /// optional nuisance covariates).
    #[arg(long)]
    pub design: PathBuf,

    /// Directory the differential tables are written to.
    #[arg(long)]
    pub output_dir: PathBuf,

    /// Minimum posterior-weight mass for a (transcript x sample) cell.
    #[arg(long, default_value_t = MIN_WEIGHT_DEFAULT)]
    pub min_weight: f64,

    /// Minimum total count for a transcript in the usage track.
    #[arg(long, default_value_t = MIN_COUNTS_DEFAULT)]
    pub min_counts: f64,

    /// R interpreter to run the engine with. Defaults to $TELESCOPE_RSCRIPT,
    /// then to `Rscript` on PATH.
    #[arg(long)]
    pub rscript: Option<PathBuf>,

    /// The R engine itself. Defaults to $TELESCOPE_DIFF_R, then to
    /// $CONDA_PREFIX/share/telescope/workflow/scripts/telescope_diff.R, then
    /// to the copy next to this binary.
    #[arg(long)]
    pub r_script: Option<PathBuf>,
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
pub fn aggregate_profile(path: &Path) -> Result<Vec<AggRow>> {
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
    let mut acc: HashMap<(u32, u32, u32), (f64, f64)> = HashMap::new();
    let mut n_rows = 0u64;

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

        for i in 0..gene.len() {
            if gene.is_null(i) || tx.is_null(i) || pal.is_null(i) || pw.is_null(i) {
                bail!(
                    "null value in {} at row {} of the first batch: the profile schema \
                     declares these columns non-nullable",
                    path.display(),
                    i
                );
            }
            let s = match sample {
                Some(arr) if !arr.is_null(i) => interner.intern(arr.value(i)),
                _ => stem_id,
            };
            let g = interner.intern(gene.value(i));
            let t = interner.intern(tx.value(i));
            let w = pw.value(i);
            // `log2(max(pal, 1))`: a zero or negative tail length must not
            // produce -inf, and 1 is the floor the R engine uses.
            let log2pal = (pal.value(i).max(1) as f64).log2();
            let e = acc.entry((s, g, t)).or_insert((0.0, 0.0));
            e.0 += w;
            e.1 += w * log2pal;
            n_rows += 1;
        }
    }

    let mut out: Vec<AggRow> = acc
        .into_iter()
        .map(|((s, g, t), (w, l))| AggRow {
            sample_id: interner.resolve(s).to_string(),
            gene_id: interner.resolve(g).to_string(),
            transcript_id: interner.resolve(t).to_string(),
            weight_sum: w,
            weighted_log2pal_sum: l,
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

/// Aggregate every profile file and merge the cells.
pub fn aggregate_profiles(paths: &[PathBuf]) -> Result<Vec<AggRow>> {
    let mut merged: HashMap<(String, String, String), (f64, f64)> = HashMap::new();
    // Files are aggregated independently, so the sum order inside a cell does
    // not depend on how many files ran at the same time.
    let per_file: Vec<Result<Vec<AggRow>>> =
        paths.par_iter().map(|p| aggregate_profile(p)).collect();
    for r in per_file {
        for row in r? {
            let e = merged
                .entry((row.sample_id, row.gene_id, row.transcript_id))
                .or_insert((0.0, 0.0));
            e.0 += row.weight_sum;
            e.1 += row.weighted_log2pal_sum;
        }
    }
    let mut out: Vec<AggRow> = merged
        .into_iter()
        .map(
            |((sample_id, gene_id, transcript_id), (weight_sum, weighted_log2pal_sum))| AggRow {
                sample_id,
                gene_id,
                transcript_id,
                weight_sum,
                weighted_log2pal_sum,
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

/// Write the aggregate table the R engine reads.
///
/// The floats are written with Rust's shortest round-tripping representation,
/// so the R engine reads back exactly the bits Rust summed.
pub fn write_aggregate_tsv(path: &Path, rows: &[AggRow]) -> Result<()> {
    let mut out = String::with_capacity(rows.len() * 96 + 64);
    out.push_str("sample_id\tgene_id\ttranscript_id\tweight_sum\tweighted_log2pal_sum\n");
    for r in rows {
        out.push_str(&format!(
            "{}\t{}\t{}\t{:?}\t{:?}\n",
            r.sample_id, r.gene_id, r.transcript_id, r.weight_sum, r.weighted_log2pal_sum
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
pub fn resolve_r_script(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    if let Ok(v) = std::env::var("TELESCOPE_DIFF_R") {
        if !v.trim().is_empty() {
            return Ok(PathBuf::from(v));
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

/// Run the differential layer.
pub fn run(args: &DiffArgs) -> Result<()> {
    let profiles = expand_glob(&args.profiles)?;
    if profiles.is_empty() {
        bail!("--profiles matched no files: {}", args.profiles);
    }
    if !args.design.is_file() {
        bail!("--design does not exist: {}", args.design.display());
    }
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("failed to create {}", args.output_dir.display()))?;

    let rows = aggregate_profiles(&profiles)?;
    let agg_path = args.output_dir.join("telescope_diff_aggregate.tsv");
    write_aggregate_tsv(&agg_path, &rows)?;
    eprintln!(
        "[telescope]  aggregate: {} (transcript x sample) cells -> {}",
        rows.len(),
        agg_path.display()
    );

    let rscript = resolve_rscript(args.rscript.as_deref());
    let r_script = resolve_r_script(args.r_script.as_deref())?;

    let status = Command::new(&rscript)
        .arg(&r_script)
        .arg("--aggregate")
        .arg(&agg_path)
        .arg("--design")
        .arg(&args.design)
        .arg("--output-dir")
        .arg(&args.output_dir)
        .arg("--min-weight")
        .arg(args.min_weight.to_string())
        .arg("--min-counts")
        .arg(args.min_counts.to_string())
        .status()
        .with_context(|| {
            format!(
                "failed to run the R engine: {} {}. Is R on PATH? A conda environment \
                 with the Bioconductor packages is required; see workflow/environment.yml.",
                rscript.display(),
                r_script.display()
            )
        })?;
    if !status.success() {
        bail!(
            "the R engine failed ({status}); its output above is the error. \
             The aggregate it was given is still at {}",
            agg_path.display()
        );
    }
    eprintln!(
        "[telescope]  wrote telescope_differential_PAU.tsv and \
         telescope_differential_PAL.tsv into {}",
        args.output_dir.display()
    );
    Ok(())
}
