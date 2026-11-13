<p align="center">
  <img src="assets/logo-icon.svg" width="120" alt="Telescope logo">
</p>

<h1 align="center"><strong>Telescope</strong></h1>

<p align="center"><strong>T</strong>ranscript-level <strong>E</strong>xpression and tail-<strong>L</strong>ength <strong>E</strong>stimation via <strong>S</strong>ingle-molecule de<strong>C</strong>onv<strong>O</strong>lution with <strong>P</strong>oly(A) <strong>E</strong>vidence</p>

---

**Telescope** is a unified computational framework for Oxford Nanopore direct RNA sequencing (DRS) that performs joint single-molecule isoform deconvolution, transcript-specific poly(A) tail-length estimation (`TEL-EM`), and five-modality differential analysis (**DEG**, **DET**, **DIU**, **APA**, and **DPL** via `DT-EB`).

## Architecture

Telescope is implemented as a Rust workspace coupled with a Bioconductor statistical engine:

* `crates/telescope-core` — single-molecule evidence extraction, hierarchical empirical Bayes prior construction, and the `TEL-EM` joint MAP-EM solver
* `crates/telescope-cli` — the unified `telescope` command-line interface
* `workflow/scripts/` — the multi-modal differential engine (`telescope_diff.R` and `dpl_models.R`)

The public interface is **two commands**:

| Command | Description |
|---|---|
| `quant` | Build the poly(A) prior and run joint MAP-EM (`TEL-EM`) for isoform quantification and single-molecule poly(A) deconvolution |
| `diff` | Run five-modality differential analysis (DEG, DET, DIU, APA, and DPL) across conditions |

`quant` runs the whole quantification pipeline in one invocation: it selects anchor reads, fits the three-tier poly(A) tail-length prior, and solves `TEL-EM` for every sample you give it. The intermediate stages remain available as hidden subcommands (`build-anchors`, `build-emission`, `summarize`) for pipelines that need to drive them one at a time.

Run `telescope <command> --help` to inspect all parameters.

---

## Installation

Three steps. The conda environment carries both the Bioconductor stack (for `telescope diff`) and a Rust toolchain (for the binary), so a single `conda activate` is all you need.

### 1. Clone

```bash
git clone https://github.com/x1han/Telescope.git
cd Telescope
```

### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate telescope
```

The environment pins every Bioconductor package exactly, because the published differential tables were produced by those builds. `environment.yml` is the single source for those pins.

### 3. Install the binary and the R engine

```bash
cargo install --path crates/telescope-cli --root "$CONDA_PREFIX"
mkdir -p "$CONDA_PREFIX/telescope/workflow/scripts"
cp workflow/scripts/*.R "$CONDA_PREFIX/telescope/workflow/scripts/"
```

Installing with the prefix set to `$CONDA_PREFIX` places `telescope` in `$CONDA_PREFIX/bin`, which is on `PATH` inside the activated environment. The `cp` places the R engine where `telescope diff` looks for it: `$CONDA_PREFIX/telescope/workflow/scripts/telescope_diff.R`.

`telescope diff` resolves the R interpreter via `$TELESCOPE_RSCRIPT` (or `Rscript` on `PATH`) and locates the R engine via `$TELESCOPE_DIFF_R`, `$CONDA_PREFIX/telescope/workflow/scripts/telescope_diff.R`, or relative to the binary path (overridable via `--rscript` and `--r-script`).

---

## Quick Start

### Step 1: Quantify one sample (`quant`)

Give `quant` a transcriptome-aligned BAM, a sample id, an annotation, and a path for the profile. It selects the anchor reads, fits the poly(A) tail-length prior, solves the posterior, and writes the profile:

```bash
telescope quant \
  --bam s1.transcriptome.bam \
  --sample-id S1 \
  --gtf annotation.gtf \
  --output s1.profile.parquet
```

The prior and the anchor pool are built in a temporary directory and removed when the run ends: `--output` leaves you the profile and nothing else.

**A note on the reference universe.** The universe is the set of transcripts the prior covers, and it is the `@SQ` header of the BAM you gave. Transcripts in that universe that the anchor pool never observed are still in the model -- they enter it as cohort-prior rows rather than being silently absent -- so a read whose candidate is one of them is not dropped for want of a parameter.

Key model hyperparameters on `telescope quant`:

* **Alignment & 5'-completeness terms**: `--aln-weight-lambda` (default 20.0), `--five-prime-window` (default 20 bp), and `--five-prime-weight` (default 1.8).
* **3'-end boundary & geometry terms**: `--tes-slack` enforces the hard 3'-end boundary window (default 35 bp). `--three-prime-window` (default 15 bp) and `--three-prime-weight` (default 0.0, disabled) configure an optional soft 3'-end proximity reward in the E-step.
* **Posterior truncation**: `--min-posterior` (default 0.01) retains `(read × candidate)` pairs with posterior probability at least this threshold and renormalises surviving weights per read.

### Step 2: Differential analysis (`diff`)

Compare two conditions across biological replicates ($\min(m, n) \ge 3$ per group) to simultaneously quantify gene expression, transcript abundance, isoform usage, alternative polyadenylation, and differential poly(A) tail length:

```bash
telescope diff \
  --metadata metadata.tsv \
  --profile-column profile_path \
  --condition-column condition \
  --control-condition control \
  --case-condition treated \
  --gtf annotation.gtf \
  --fasta genome.fa \
  --chrom-sizes chrom.sizes \
  --output-dir results/
```

The `--metadata` TSV specifies one row per sample (`batch` is optional):

```text
sample_id    profile_path    condition    batch    alignment_path
S1    results/S1.profile.parquet    control    b1    s1.genome.bam
S2    results/S2.profile.parquet    control    b2    s2.genome.bam
S3    results/S3.profile.parquet    control    b1    s3.genome.bam
S4    results/S4.profile.parquet    treated    b1    s4.genome.bam
S5    results/S5.profile.parquet    treated    b2    s5.genome.bam
S6    results/S6.profile.parquet    treated    b1    s6.genome.bam
```

---

## Differential Analysis Outputs & Statistical Models

`telescope diff` writes an atomic output directory containing five unfiltered result tables and full provenance receipts:

| File | Modality & Statistical Method |
|---|---|
| `deg.tsv` | **Differential Gene Expression** (`DESeq2` Wald test with unshrunk MLE and normal-prior shrunk estimates) |
| `det.tsv` | **Differential Transcript Abundance** (`DESeq2` Wald test on EM-deconvolved transcript counts) |
| `diu.tsv` | **Differential Isoform Usage** (`DEXSeq` per-feature likelihood ratio test within parent genes) |
| `apa.tsv` | **Alternative Polyadenylation** (`DEXSeq` likelihood ratio test on `LAPA` poly(A) cluster counts) |
| `dpl.tsv` | **Differential Poly(A) Tail Length** (`DT-EB`: Depth-Trended Empirical Bayes via `limma`) |
| `lapa/` | Replicated per-sample poly(A) cluster calls (reusable across contrasts via `--lapa-dir`) |
| `run_manifest.json` | Complete execution receipt, resolved input paths, and admission counts |
| `software_versions.tsv` | Exact versions of all executed software and Bioconductor packages |
| `model_parameters.tsv` | Configured statistical thresholds and fitted prior parameters |

### Admission Gates and the `DT-EB` Model

1. **Per-group feature admission (`DEG`, `DET`, `DIU`)**: Each condition must contain at least `--min-samples` (default 3) biological replicates. A feature is tested only when it reaches at least `--min-counts` (default 20) total reads and exhibits non-zero support in at least `--min-samples` replicates of **each** condition; the usage track additionally requires the feature's **gene** to carry at least `--min-counts` reads in at least `--min-samples` replicates of each condition. Excluded features remain in the output table with their exclusion reason recorded in `padj_na_reason`.
2. **Per-site admission (`APA`)**: a poly(A) site is tested only when its own total poly(A) reads reach **10** — a literal, not `--min-counts` — it has at least `--min-samples` (default 3) usable cells in **each** condition, and the gene's poly(A) total in a usable cell reaches the single-sample floor `max(5, floor(--min-counts / 4))` (5 by default). A site LAPA did not call in a sample counts as a zero share when the gene's total for that sample is known and large enough; otherwise that cell is unusable. Excluded sites remain in the output table with their exclusion reason recorded in `padj_na_reason`.
3. **Cell-level tail-length admission & `DT-EB` (`DPL`)**: A `(transcript × sample)` cell enters the poly(A) tail-length linear model only when its posterior weight sum reaches `--min-cell-weight` (default 10), its Kish effective sample size reaches `--min-neff` (default 10), and its within-sample variance is finite and positive. `DT-EB` fits an equal-weight linear model across qualifying replicates and moderates residual variances using `limma::eBayes(trend = cov_depth, robust = TRUE)`, where the trend covariate is the mean log2 read support of each isoform. If the variance trend cannot be fitted, the engine explicitly records the fallback (`DT-EB-no-trend`) in the receipt.
4. **High-speed Rust aggregation**: `telescope diff` streams multi-million-row per-read `.profile.parquet` files in Rust and reduces them to per-`(transcript × sample)` sufficient statistics before invoking R, preventing per-read pseudoreplication while reducing memory usage from tens of gigabytes to hundreds of megabytes.

---

## Running Unit Tests

```bash
cargo test --release
```

## License

Copyright (c) 2026 x1han.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, version 3 only. This program is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the `LICENSE` file for
the full terms.
