<h1 align="center"><span style="font-size:2.2em;font-weight:700">Telescope</span> <span style="font-size:1.05em;font-weight:700">(TELESCOPE)</span></h1>

<p align="center"><strong>Transcript-level Expression and tail-Length Estimation via Single-molecule deConvOlution with Poly(A) Evidence</strong></p>

<p align="center">
  <em>Poly(A) tail-length-aware isoform quantification from direct RNA sequencing</em>
</p>

---

## Why Telescope

In direct RNA sequencing (DRS), reads that are 5′-truncated within a shared 3′ exon produce
**identical alignment scores** across every isoform of the gene they map to. With no
discriminating sequence feature, sequence-only quantifiers cannot tell these molecules apart at
all: every ambiguous read is assigned the same abundance weight, and the isoform-level signal is
lost before it is ever measured.

Telescope recovers it. The basecaller already reports a per-molecule poly(A) tail length; that
measurement is information the aligner never sees. Telescope folds it, together with the
alignment geometry, into a two-stage empirical-Bayes model that assigns each ambiguous read a
**posterior probability** instead of a coin flip.

## What it does

**Stage 1 — anchor selection and tail-length priors.** Telescope keeps the high-confidence,
full-length reads that align uniquely to a single isoform, and fits an isoform-specific
log-normal tail-length prior to them. Reads are then bucketed into five identifiability tiers:

| Tier | Meaning |
|---|---|
| **L1** | Enough high-purity anchors to estimate isoform-specific tail-length mean and variance directly |
| **L2** | Moderate anchor support; tail-length parameters are stabilised by empirical-Bayes shrinkage |
| **L3 (anchored)** | Very few anchors |
| **L3 (imputed)** | No anchors; imputed from gene- or cohort-level priors |
| **L3 (unexpressed)** | Not detected |

**Stage 2 — joint MAP-EM.** A joint maximum-a-posteriori expectation-maximisation solver combines
the transcript abundance prior, the alignment score, the 5′-end completeness geometry, and the
Gaussian log-tail likelihood to compute a per-molecule posterior. It updates molecular counts
(Counts, CPM) while simultaneously refining the isoform tail-length distributions.

## What it buys you

On simulated benchmarks with known truth, and across two human DRS cohorts (10 clinical lung
parenchyma samples and 9 AsPC-1 cell-line libraries):

- Isoform tail-length distributions are recovered accurately from anchors alone
  (Pearson *r* = 0.987 on the two-isoform benchmark, 0.932 on the multi-isoform benchmark).
- Posterior-weighted reconstruction separates isoform tail-length distributions where an
  unweighted pool of ambiguous reads cannot, and where hard assignment truncates them.
- 5′-end geometry and poly(A) tail length contribute **complementary** signal: the geometry term
  is a structure-driven disambiguator that does not depend on tail-length divergence, while the
  tail term is a divergence-dependent resolver that only switches on when the isoforms actually
  differ in tail length. Together they lower quantification error at every level of structural
  ambiguity.

## Installation

```bash
git clone https://github.com/x1han/Telescope.git
cd Telescope
cargo build --release
```

## Usage

```bash
telescope quantify \
  --bam   sample.bam \
  --reference annotation.gtf \
  --read-level-pal sample.pal.tsv \
  --output sample.telescope.tsv
```

## Citation

If you use Telescope in your research, please cite the associated manuscript.

## License

To be specified. Please open an issue if you need a specific license for your use case.

## Acknowledgements

Telescope is developed with the basecaller-provided poly(A) tail-length tags (`pt:i`) from
Oxford Nanopore direct RNA sequencing.
