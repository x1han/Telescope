<p align="center">
  <img src="assets/logo-icon.svg" width="120" alt="Telescope logo">
</p>

<h1 align="center"><strong>Telescope</strong></h1>

<p align="center"><strong>T</strong>ranscript-level <strong>E</strong>xpression and tail-<strong>L</strong>ength <strong>E</strong>stimation via <strong>S</strong>ingle-molecule de<strong>C</strong>onv<strong>O</strong>lution with <strong>P</strong>oly(A) <strong>E</strong>vidence</p>

---

Telescope is implemented as two Rust crates:

* `crates/telescope-core` — evidence graph, M-step EM, anchor builder
* `crates/telescope-cli` — the `telescope` binary

The M-step prior is implemented in `crates/telescope-core/src/em.rs` and pinned
by the crate's own tests.

## The differential layer

`telescope diff` compares two conditions over a set of sample profiles and
writes five result matrices:

```bash
telescope diff \
  --metadata metadata.tsv \
  --profile-column profile_path \
  --condition-column condition \
  --control-condition HC \
  --case-condition COPD \
  --tx-to-gene tx_to_gene.tsv \
  --gtf annotation.gtf \
  --output-dir results/
```

The metadata is one row per sample:

```
sample_id  profile_path        condition  batch  alignment_path
HC1        hc1.profile.parquet HC         b1     hc1.bam
COPD1      copd1.profile.parquet COPD     b2     copd1.bam
```

`batch` is optional; when the metadata has no batch column the run says so
rather than guessing one. The contrast is declared, never inferred: the
reference level is the control, so every reported effect is case minus
control however the labels happen to spell.

| file | what it is |
|---|---|
| `deg.tsv` | gene-level differential expression (DESeq2) |
| `det.tsv` | transcript-level differential abundance (DESeq2) |
| `diu.tsv` | differential isoform usage within a gene (DEXSeq) |
| `apa.tsv` | alternative polyadenylation (LAPA) |
| `dpl.tsv` | differential poly(A) tail length (limma) |
| `run_manifest.json` | the command, the resolved inputs, the tools that ran |
| `software_versions.tsv` | the version of every tool that ran |
| `model_parameters.tsv` | the parameters of every model that ran |
| `logs/` | the output of each tool |

The result matrices are **not filtered**. Every row of the input universe is
present, a row without a statistic says why, and filtering and plotting are
the user's downstream job.

The directory appears in one step or not at all: the run builds it under a
temporary name and moves it into place when every requested modality has
succeeded. A missing tool fails before any computation and leaves no
directory behind.

### What is computed, and where

**The statistics are Bioconductor's and the tools are published ones, not
ours.** Gene-level and transcript-level expression come from
[DESeq2](https://bioconductor.org/packages/DESeq2), within-gene isoform usage
from [DEXSeq](https://bioconductor.org/packages/DEXSeq), the tail-length
variance moderation from [limma](https://bioconductor.org/packages/limma),
and polyadenylation from [LAPA](https://github.com/MuhammedHasan/lapa).
`workflow/scripts/telescope_diff.R` is the engine that calls them, and no
statistic is re-implemented on top of them.

**The tail-length model was calibrated, not chosen.** The model layer lives in
`workflow/scripts/dpl_models.R`, and the same file is what
`workflow/scripts/audit_dpl_models.R` calibrates: the audit enumerates the
label permutations exactly, checks the p-values against a known null, injects
known effects at four sizes, and reports power and bias. The shipped model is
the ordinary two-sample t on the (transcript × sample) means; the
precision-weighted alternatives were measured and rejected, and the engine
runs the audited code rather than a copy of it.

**What the CLI does itself is the aggregation.** The profile parquet is per
READ — one row per (read, candidate) pair — so a ten-sample cohort is 111 M
rows, which R holds in 31 GB only to add it up. `telescope diff` streams the
parquet and reduces it to one row per (transcript × sample), then hands that
table to the R engine. Measured on that cohort: R reads and aggregates in
177 s and 30.9 GB, the Rust side in 5.6 s and 233 MB, and both produce the
same 336,709 cells.

That reduction is not an optimisation. Fitting on per-read rows would inflate
the effective sample size by ~10⁶ and make every p-value meaningless, so
"one observation per (transcript × sample)" is the definition of the
statistic rather than a speed-up.

### Requirements

The binary is pure Rust: no C library, no system dependency, no package
manager. The R engine needs Bioconductor, so it runs in a conda environment:

```bash
mamba env create -f workflow/environment.yml
mamba activate telescope-ref
```

`telescope diff` finds the interpreter through `$TELESCOPE_RSCRIPT`, then
`Rscript` on `PATH`; the engine through `$TELESCOPE_DIFF_R`, then
`$CONDA_PREFIX/share/telescope/workflow/scripts/telescope_diff.R`. Both are
overridable with `--rscript` and `--r-script`. The polyadenylation modality
additionally needs LAPA on `PATH` (or `--lapa-path`); if it is missing the run
says so and stops before it computes anything.

## Acceptance checks

The acceptance harness — synthetic profiles, a design matrix, an end-to-end
fixture on a tiny synthetic genome, and the expected output for every track —
is internal development material. It lives in the project workspace layer and
is not part of this release, so the scripts that run it are not published
here. What ships is the crate, its own tests, and the R engine.

## License

Dual licensed under Apache-2.0 or BSD-3-Clause. See `LICENSE-APACHE` and
`LICENSE-BSD-3-Clause`.
