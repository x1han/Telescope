#!/usr/bin/env Rscript
# telescope_diff.R — Module 4: Differential Statistical Layer
#
# ROLE OF THIS FILE. The shipped Telescope CLI implements Module 4 as the
# `telescope diff` subcommand. This R script is the REFERENCE implementation
# of the same frozen specification: it is the
# independent second implementation that the Rust port is checked against.
#
#   - It is kept here, beside the code it validates, so that the two cannot
#     drift apart silently.
#   - `workflow/test_data/` holds the synthetic profiles, the design matrix,
#     and the expected output for both tracks. That fixture is the acceptance
#     test for the Rust `telescope diff` port.
#   - The R dependencies (DEXSeq, lmerTest, limma) are Bioconductor packages;
#     the Rust implementation deliberately shares no code path with them, which
#     is what makes the comparison meaningful.
#
# Two-track design:
#   Track A (ΔPAU, usage): DEXSeq on (gene × transcript × sample) counts.
#   Track B (ΔPAL, length): WLS / LMM dual-engine on
#                            (transcript × sample) aggregated log2(polyA length).
#
# Pseudoreplication-safe aggregation.
#   Track B MUST aggregate to one observation per (transcript × sample)
#   BEFORE fitting. Per-read fitting inflates effective n by ~1e6× and
#   p-values become meaningless.
#
# WLS / LMM dual engine:
#   - When `#observations > #samples` (technical replicates, batch blocks,
#     or any within-sample structure exist): use lmerTest::lmer with
#     `(1 | sample_id)` random intercept + `weights = weight_sum`.
#   - When `#observations == #samples` (canonical Tongji case): random
#     intercept is saturated; fall back to stats::lm with
#     `weights = weight_sum`. Reported via the `model` column in output.
#   - Both modes use weighted fitting so `weight_sum` (effective read
#     count per sample) acts as precision.
#
# Required pre-filter for Track A:
#   Transcripts with total count across all samples < MIN_COUNTS = 20
#   are dropped before DEXSeq. Low-count universe destabilizes the
#   negative-binomial dispersion fit. On real Tongji tissue data this
#   trims ~80% near-zero transcripts out of ~180 k, leaving ~20 k.
#
# Implementation note (Bioconductor namespace collisions):
#   When DEXSeq is loaded, `AnnotationDbi::select` and `stats::filter`
#   mask `dplyr::select` / `dplyr::filter`. We qualify all dplyr verbs
#   explicitly to avoid silent dispatch to wrong implementations.
#
# Usage:
#   Rscript telescope_diff.R \
#     --profiles 'data/*.profile.parquet' \
#     --design   design.tsv \
#     --output-dir results/ \
#     --min-weight 10 \
#     --min-counts 20
#
# Inputs:
#   --profiles   Glob pointing to one telescope.profile.parquet per sample.
#                Schema (from spec F4):
#                  sample_id, read_id, gene_id, transcript_id,
#                  polyA_length (Int32), posterior_weight (Float64).
#   --design     TSV with header. Required cols:
#                  sample_id, condition
#                Optional cols: batch, tissue (nuisance covariate).
#   --output-dir Where to write:
#                  telescope_differential_PAU.tsv
#                  telescope_differential_PAL.tsv
#                  sessionInfo.txt

suppressPackageStartupMessages({
  library(optparse)
  library(arrow)
  library(dplyr)
  library(tidyr)
  library(DEXSeq)
  library(lmerTest)
  library(emmeans)
  library(limma)        # Empirical Bayes variance moderation (Smyth 2004 squeezeVar)
})

option_list <- list(
  make_option("--profiles", type = "character",
              help = "Glob to *.profile.parquet"),
  make_option("--aggregate", type = "character",
              help = "Pre-aggregated (transcript x sample) TSV; replaces --profiles"),
  make_option("--design", type = "character",
              help = "Design TSV (sample_id, condition)"),
  make_option("--output-dir", type = "character", default = ".",
              help = "Output directory [default %default]"),
  make_option("--min-weight", type = "double", default = 10,
              help = "Min sum(posterior_weight) per (transcript x sample) [default %default]"),
  make_option("--min-counts", type = "double", default = 20,
              help = "Min total counts per transcript for Track A (DEXSeq dispersion stability; default = 20) [default %default]"),
  make_option("--min-samples", type = "integer", default = 2,
              help = "Min number of samples per transcript for Track B [default %default]")
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$design)) {
  stop("ERROR: --design is required.")
}
if (is.null(opt$profiles) && is.null(opt$aggregate)) {
  stop("ERROR: either --profiles or --aggregate is required.")
}
dir.create(opt$`output-dir`, recursive = TRUE, showWarnings = FALSE)

# Two input modes, one set of tables.
#
#   --profiles  the per-read profile parquets. R does the aggregation itself.
#               This is the reference path: it is the only mode that exercises
#               the aggregation inside R.
#   --aggregate a (transcript x sample) table that `telescope diff` produced.
#               The CLI aggregates in Rust because one sample's per-read
#               profile is 15 M rows, which R holds in 3.8 GB only to add it
#               up.
#
# Both modes produce the same two tables downstream: `counts_long` (the usage
# counts) and `agg` (the tail-length observations and their precision).
if (!is.null(opt$aggregate)) {
  cat("[telescope_diff] Reading the pre-aggregated table:\n")
  agg_raw <- read.delim(opt$aggregate, stringsAsFactors = FALSE)
  need_cols <- c("sample_id", "gene_id", "transcript_id", "weight_sum",
                  "weighted_log2pal_sum")
  missing_cols <- setdiff(need_cols, colnames(agg_raw))
  if (length(missing_cols) > 0) {
    stop("ERROR: the aggregate table is missing column(s): ",
         paste(missing_cols, collapse = ", "))
  }
  cat("  rows:", nrow(agg_raw), "\n")
  profiles_df <- NULL
} else {
  cat("[telescope_diff] Reading profile parquets:\n")
  profile_files <- Sys.glob(opt$profiles)
  if (length(profile_files) == 0) {
    stop("ERROR: --profiles glob matched 0 files: ", opt$profiles)
  }
  cat("  found", length(profile_files), "files\n")

  read_profile <- function(f) {
    df <- as.data.frame(read_parquet(f))
    if (!"sample_id" %in% colnames(df) || all(is.na(df$sample_id))) {
      df$sample_id <- tools::file_path_sans_ext(basename(f))
    }
    df
  }
  profiles_df <- dplyr::bind_rows(lapply(profile_files, read_profile))
  cat("  total rows:", nrow(profiles_df), "\n")
}

design <- read.delim(opt$design, stringsAsFactors = FALSE)
stopifnot(all(c("sample_id", "condition") %in% colnames(design)))
design$sample_id  <- as.character(design$sample_id)
design$condition  <- factor(design$condition)
cat("[telescope_diff] Design:\n")
print(design)

# Sanity check: every sample in profiles appears in design, and vice versa.
prof_samples <- if (is.null(profiles_df)) unique(agg_raw$sample_id) else unique(profiles_df$sample_id)
miss_in_design <- setdiff(prof_samples, design$sample_id)
miss_in_prof   <- setdiff(design$sample_id, prof_samples)
if (length(miss_in_design) > 0) {
  stop("ERROR: samples in profiles but not in design: ",
       paste(miss_in_design, collapse = ", "))
}
if (length(miss_in_prof) > 0) {
  warning("WARN: samples in design but not in profiles: ",
          paste(miss_in_prof, collapse = ", "))
}

# Make sure the contrast reference level is well-defined.
contrast_label <- function(cond_col, prefix) {
  paste0(prefix, "_", cond_col)
}

# =====================================================================
# Track A: ΔPAU (usage) via DEXSeq
# =====================================================================
cat("\n[telescope_diff] Track A: ΔPAU via DEXSeq\n")

counts_long <- if (is.null(profiles_df)) {
  # Pre-aggregated input: the count is the rounded posterior mass the
  # aggregator already summed.
  agg_raw %>%
    dplyr::transmute(gene_id, transcript_id, sample_id,
                     count = as.integer(round(weight_sum)))
} else {
  profiles_df %>%
    dplyr::group_by(gene_id, transcript_id, sample_id) %>%
    dplyr::summarize(count = sum(posterior_weight), .groups = "drop") %>%
    # DESeq2 / DEXSeq require integer counts. posterior_weight is continuous
    # (sums to ~1 across candidates per read); the natural integer proxy is
    # the rounded expected count of assigned reads.
    dplyr::mutate(count = as.integer(round(count)))
}

counts_by_tx <- counts_long %>%
  dplyr::group_by(gene_id, transcript_id) %>%
  dplyr::summarize(total = sum(count), .groups = "drop")
keep_tx <- counts_by_tx %>%
  dplyr::filter(total >= opt$`min-counts`) %>%
  dplyr::select(gene_id, transcript_id)
counts_long <- counts_long %>%
  dplyr::semi_join(keep_tx, by = c("gene_id", "transcript_id"))
cat("  transcripts after min-counts filter:",
    dplyr::n_distinct(counts_long$transcript_id), "\n")

# A gene with ONE transcript has no usage to test. DEXSeq's contrast is
# WITHIN a gene, so a single-transcript gene carries no usage variation, and
# its design matrix is rank-deficient -- DESeq2's checkFullRank refuses it and
# the whole run dies. Such genes are dropped, and the count is reported so the
# drop is never silent.
#
# This is not a corner case: in a transcriptome whose gene and transcript ids
# coincide (AIDRS_tx_1 as both), a large share of the genes are singletons.
tx_per_gene <- counts_long %>% dplyr::count(gene_id, name = "n_tx")
genes_keep  <- tx_per_gene %>% dplyr::filter(n_tx >= 2) %>% dplyr::pull(gene_id)
n_singleton_genes <- nrow(tx_per_gene) - length(genes_keep)
if (n_singleton_genes > 0) {
  cat("  dropped", n_singleton_genes,
      "single-transcript genes (no within-gene usage contrast)\n")
  counts_long <- counts_long %>% dplyr::filter(gene_id %in% genes_keep)
  counts_by_tx <- counts_by_tx %>% dplyr::filter(gene_id %in% genes_keep)
}
cat("  transcripts entering DEXSeq:",
    dplyr::n_distinct(counts_long$transcript_id), "in",
    dplyr::n_distinct(counts_long$gene_id), "genes\n")

# Build wide count matrix: rows = (gene, transcript), cols = sample.
counts_wide <- counts_long %>%
  tidyr::pivot_wider(names_from = sample_id, values_from = count,
                    values_fill = 0)
gene_ids    <- counts_wide$gene_id
tx_ids      <- counts_wide$transcript_id
sample_cols <- setdiff(colnames(counts_wide), c("gene_id", "transcript_id"))
count_mat   <- as.matrix(counts_wide[, sample_cols])
rownames(count_mat) <- paste(gene_ids, tx_ids, sep = "|")
count_mat[is.na(count_mat)] <- 0

# Align design with count matrix columns.
design_aligned <- design[match(colnames(count_mat), design$sample_id), ]
if (any(is.na(design_aligned$sample_id))) {
  stop("ERROR: design sample_ids do not match count matrix columns.")
}

# DEXSeqDataSet: each transcript is an "exon" of its parent gene.
dxd <- DEXSeqDataSet(
  countData  = count_mat,
  sampleData = design_aligned,
  design     = ~ sample + exon + condition:exon,
  featureID  = tx_ids,
  groupID    = gene_ids
)

cat("  running DEXSeq size factors + dispersion + test…\n")
system.time({
  dxd <- estimateSizeFactors(dxd)
  # estimateDispersions can fail on tiny synthetic fixtures where
  # gene-wise dispersions don't span enough orders of magnitude for
  # parametric curve fitting. Fall back to per-gene estimates in that
  # case (still valid for moderate n; same code path as DESeq2).
  disp_ok <- tryCatch({
    dxd <- estimateDispersions(dxd, quiet = TRUE)
    TRUE
  }, error = function(e) {
    cat("  WARN: parametric dispersion fit failed (", conditionMessage(e),
        "); falling back to gene-wise estimates.\n")
    dxd <- estimateDispersionsGeneEst(dxd, quiet = TRUE)
    dispersions(dxd) <- mcols(dxd)$dispGeneEst
    FALSE
  })
  dxd <- testForDEU(dxd,
                    reducedModel = ~ sample + exon + condition,
                    fullModel    = ~ sample + exon + condition:exon)
  dxd <- estimateExonFoldChanges(dxd, fitExpToVar = "condition")
})
dxr <- DEXSeqResults(dxd)

# Compute log2FoldChange directly from per-condition mean counts (more robust
# than depending on DEXSeq version-specific fold-change column names).
treated_samples <- design_aligned$sample_id[design_aligned$condition == levels(design_aligned$condition)[2]]
control_samples <- design_aligned$sample_id[design_aligned$condition == levels(design_aligned$condition)[1]]
tx_mean_treated <- rowMeans(count_mat[, treated_samples, drop = FALSE], na.rm = TRUE)
tx_mean_control <- rowMeans(count_mat[, control_samples, drop = FALSE], na.rm = TRUE)
log2fc_tx <- log2((tx_mean_treated + 0.5) / (tx_mean_control + 0.5))
tx_keys_lfc <- data.frame(
  gene_id       = gene_ids,
  transcript_id = tx_ids,
  log2FoldChange = log2fc_tx,
  stringsAsFactors = FALSE
)

pau_out <- data.frame(
  gene_id        = dxr$groupID,
  transcript_id  = dxr$featureID,
  pvalue         = dxr$pvalue,
  padj           = dxr$padj,
  stringsAsFactors = FALSE
) %>%
  dplyr::inner_join(tx_keys_lfc, by = c("gene_id", "transcript_id")) %>%
  dplyr::inner_join(counts_by_tx, by = c("gene_id", "transcript_id")) %>%
  dplyr::mutate(
    n_samples_treated = sum(design$condition == levels(design$condition)[2]),
    n_samples_control = sum(design$condition == levels(design$condition)[1])
  )

write.table(pau_out,
            file = file.path(opt$`output-dir`, "telescope_differential_PAU.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("[telescope_diff]  wrote telescope_differential_PAU.tsv (",
    nrow(pau_out), "transcripts)\n")

# =====================================================================
# Track B: ΔPAL (length) via lmerTest weighted LMM
# =====================================================================
cat("\n[telescope_diff] Track B: ΔPAL via lmerTest weighted LMM\n")
cat("  aggregating to (transcript × sample) BEFORE LMM to avoid pseudoreplication.\n")

# Aggregate per-read rows to one observation per (transcript × sample).
agg <- if (is.null(profiles_df)) {
  # Pre-aggregated input: the weighted mean is the numerator the aggregator
  # already summed, divided by the mass it summed it against.
  agg_raw %>%
    dplyr::transmute(sample_id, gene_id, transcript_id, weight_sum,
                     log2_pal_wmean = weighted_log2pal_sum / weight_sum) %>%
    dplyr::filter(weight_sum > 0) %>%
    dplyr::filter(weight_sum >= opt$`min-weight`) %>%
    dplyr::inner_join(design, by = "sample_id")
} else {
  profiles_df$log2_pal <- log2(pmax(profiles_df$polyA_length, 1))
  profiles_df %>%
    dplyr::group_by(sample_id, gene_id, transcript_id) %>%
    dplyr::summarize(
      weight_sum     = sum(posterior_weight),
      log2_pal_wmean = sum(posterior_weight * log2_pal) / sum(posterior_weight),
      .groups = "drop"
    ) %>%
    dplyr::filter(weight_sum >= opt$`min-weight`) %>%
    dplyr::inner_join(design, by = "sample_id")
}

cat("  (transcript × sample) rows after min-weight filter:", nrow(agg), "\n")

# Per-transcript LM/LMM. Satterthwaite df is the lmerTest default.
#
# Design note: pseudoreplication-safe aggregation gives exactly one row per
# (transcript × sample). The lme4 random intercept `(1 | sample_id)`
# requires `#groups < #obs`. With 1 obs per sample this is saturated, so
# we use a fixed-effects linear model with `sample_id` as a nuisance
# covariate when each sample contributes only one observation per
# transcript. If the input design has multiple rows per sample (e.g.
# technical replicates, batch blocks), the LMM is used instead.
#
# Empirical Bayes variance moderation is limma::squeezeVar (Smyth 2004;
# Bioconductor gold standard) and NOTHING ELSE. It estimates a transcriptome-
# wide prior variance and prior df from the observed distribution of per-
# transcript residual variances, and returns the per-transcript posterior
# (moderated) variance in `var.post`. The formula is not re-implemented here.
#
# ONE FIT PASS, NOT TWO. The prior enters only the REPORTED moderated
# columns -- it does not enter the model -- so the fit does not depend on it.
# This file used to fit every transcript twice: once to collect the raw
# variances, once with the prior in hand. The second pass reproduced the
# first pass's estimates exactly, so it was a second full pass for nothing.
fit_one_tx <- function(tx_df) {
  if (dplyr::n_distinct(tx_df$condition) < 2)    return(NULL)
  if (nrow(tx_df) < opt$`min-samples`)            return(NULL)
  n_samples <- dplyr::n_distinct(tx_df$sample_id)
  use_lmm   <- nrow(tx_df) > n_samples
  # Optional nuisance covariates (e.g. `tissue` for matched designs).
  # Only included when present in every row and have >= 2 levels.
  extra_covs <- character(0)
  if ("tissue" %in% colnames(tx_df) && dplyr::n_distinct(tx_df$tissue) >= 2) {
    extra_covs <- "tissue"
  }
  extra_formula <- if (length(extra_covs) > 0) paste("+", paste(extra_covs, collapse = " + ")) else ""
  if (use_lmm) {
    fit <- tryCatch(
      lmer(as.formula(paste("log2_pal_wmean ~ condition", extra_formula, "+ (1 | sample_id)")),
           weights = weight_sum, data = tx_df, REML = TRUE),
      error = function(e) NULL
    )
    if (is.null(fit)) return(NULL)
    coefs <- summary(fit)$coefficients
    cond_row <- grep("^condition", rownames(coefs), value = TRUE)
    if (length(cond_row) == 0) return(NULL)
    ci <- suppressMessages(confint(fit, parm = cond_row[1], method = "Wald"))
    # Empirical Bayes variance moderation via limma::squeezeVar (Smyth 2004):
    # capture raw residual variance and Satterthwaite df, compute moderated
    # variance and posterior df (d_post = d_raw + d_prior), then derive
    # t_squeezed and p_value_squeezed that use the same posterior variance
    # — so user filtering on p_value_squeezed < alpha gets a coherent subset.
    s2      <- sigma(fit)^2
    df_raw  <- coefs[cond_row[1], "df"]
    t_raw   <- coefs[cond_row[1], "t value"]
    est     <- coefs[cond_row[1], "Estimate"]
    se_raw  <- coefs[cond_row[1], "Std. Error"]
    data.frame(
      gene_id          = tx_df$gene_id[1],
      transcript_id    = tx_df$transcript_id[1],
      term             = cond_row[1],
      estimate         = est,
      std_error        = se_raw,
      df               = df_raw,
      sigma2_raw       = s2,
      t_value          = t_raw,
      p_value          = coefs[cond_row[1], "Pr(>|t|)"],
      ci_lower         = ci[1],
      ci_upper         = ci[2],
      n_obs            = nrow(tx_df),
      model            = "lmm_random_sample_id",
      stringsAsFactors = FALSE
    )
  } else {
    # 1 obs per (transcript × sample): use fixed-effects lm with sample
    # blocking via condition and optional nuisance covariates.
    fit <- tryCatch(
      lm(as.formula(paste("log2_pal_wmean ~ condition", extra_formula)),
         weights = weight_sum, data = tx_df),
      error = function(e) NULL
    )
    if (is.null(fit)) return(NULL)
    coefs <- summary(fit)$coefficients
    cond_row <- grep("^condition", rownames(coefs), value = TRUE)
    if (length(cond_row) == 0) return(NULL)
    ci <- suppressMessages(confint(fit, parm = cond_row[1]))
    # Empirical Bayes variance moderation via limma::squeezeVar (Smyth 2004):
    # compute moderated variance + posterior df, derive t_squeezed and
    # p_value_squeezed that use the same posterior variance (d_post).
    s2      <- sigma(fit)^2
    df_raw  <- fit$df.residual
    t_raw   <- coefs[cond_row[1], "t value"]
    est     <- coefs[cond_row[1], "Estimate"]
    se_raw  <- coefs[cond_row[1], "Std. Error"]
    data.frame(
      gene_id          = tx_df$gene_id[1],
      transcript_id    = tx_df$transcript_id[1],
      term             = cond_row[1],
      estimate         = est,
      std_error        = se_raw,
      df               = df_raw,
      sigma2_raw       = s2,
      t_value          = t_raw,
      p_value          = coefs[cond_row[1], "Pr(>|t|)"],
      ci_lower         = ci[1],
      ci_upper         = ci[2],
      n_obs            = nrow(tx_df),
      model            = "lm_condition_only",
      stringsAsFactors = FALSE
    )
  }
}

tx_keys <- agg %>% dplyr::distinct(gene_id, transcript_id)
cat("  fitting Track B model for", nrow(tx_keys), "transcripts…\n")

# One fit per transcript.
#
# Every slot holds list(ok, res) and NEVER a bare NULL: `out[[i]] <- NULL`
# would DELETE the element and shift every later transcript one slot up,
# and the table's gene/transcript labels would then be off by one for
# every transcript after the first failure. The identity of a transcript
# travels inside its own result (gene_id / transcript_id), so a dropped
# fit cannot move anybody else's label.
fit_all <- function() {
  out <- vector("list", nrow(tx_keys))
  for (i in seq_len(nrow(tx_keys))) {
    if (i %% 500 == 0) cat("    [", i, "/", nrow(tx_keys), "]\n")
    g <- tx_keys$gene_id[i]; t <- tx_keys$transcript_id[i]
    # The model fit AND the extraction of its coefficients are both
    # fallible -- a singular fit returns NULL, and a successful fit whose
    # summary has no `df` column throws on extraction. Both must degrade to
    # a DROPPED TRANSCRIPT, never to an aborted cohort.
    res <- tryCatch(
      fit_one_tx(dplyr::filter(agg, gene_id == g, transcript_id == t)),
      error = function(e) {
        cat("    WARN: transcript dropped (", g, "/", t, "):",
            conditionMessage(e), "\n")
        NULL
      }
    )
    out[[i]] <- list(ok = !is.null(res), res = res)
  }
  out
}

cat("  fitting…\n")
# Reject failed fits (singular / failed / too few samples) cleanly.
pal_results <- Filter(function(x) x$ok, fit_all())
n_ok        <- length(pal_results)
all_sigmas  <- vapply(pal_results, function(r) r$res$sigma2_raw, numeric(1))
all_dfs     <- vapply(pal_results, function(r) r$res$df,         numeric(1))
all_t       <- vapply(pal_results, function(r) r$res$t_value,    numeric(1))

# Empirical Bayes moderation, entirely by limma::squeezeVar. `var.post` IS
# the per-transcript posterior variance; nothing here re-derives it.
#
# A transcript whose raw residual variance or df is not finite and positive
# gets NA in the moderated columns. It is not in the set limma estimated the
# prior from, so there is no published posterior variance for it to carry.
finite <- is.finite(all_sigmas) & is.finite(all_dfs) & all_sigmas > 0 & all_dfs > 0
sv <- NULL
if (sum(finite) < 5) {
  cat("  NOTE: the prior could not be estimated (fewer than 5 usable fits), so the\n",
      "        moderated columns are NA for every transcript. The raw columns are\n",
      "        unaffected.\n", sep = "")
} else {
  sv <- limma::squeezeVar(var = all_sigmas[finite], df = all_dfs[finite])
  cat("  limma::squeezeVar priors: var_prior =", sv$var.prior,
      "  df_prior =", sv$df.prior, "\n")
  if (is.infinite(sv$df.prior)) {
    cat("  NOTE: df.prior = Inf -- the observed spread of residual variances does not\n",
        "        support a finite prior df, so the moderated variance collapses to\n",
        "        var_prior for every transcript and the moderated df is infinite\n",
        "        (the moderated p-value is then the normal-tail limit). This is a\n",
        "        property of the data, not a failure; the raw columns are unaffected.\n",
        sep = "")
  }
}

df_post     <- rep(NA_real_, n_ok)
sigma2_post <- rep(NA_real_, n_ok)
t_sq        <- rep(NA_real_, n_ok)
p_sq        <- rep(NA_real_, n_ok)
if (!is.null(sv)) {
  sigma2_post[finite] <- sv$var.post
  df_post <- all_dfs + sv$df.prior
  t_sq    <- all_t * sqrt(all_sigmas / sigma2_post)
  p_sq    <- 2 * stats::pt(-abs(t_sq), df = df_post)
}

if (n_ok == 0) {
  pal_out <- data.frame()
} else {
  pal_out <- dplyr::bind_rows(lapply(pal_results, function(x) x$res))
  pal_out$df_post          <- df_post
  pal_out$sigma2_post      <- sigma2_post
  pal_out$t_value_squeezed <- t_sq
  pal_out$p_value_squeezed <- p_sq
  pal_out <- pal_out %>%
    dplyr::select(gene_id, transcript_id, term, estimate, std_error, df,
                  df_post, sigma2_raw, sigma2_post, t_value, t_value_squeezed,
                  p_value, p_value_squeezed, ci_lower, ci_upper, n_obs, model)
}

write.table(pal_out,
            file = file.path(opt$`output-dir`, "telescope_differential_PAL.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("[telescope_diff]  wrote telescope_differential_PAL.tsv (",
    nrow(pal_out), "transcripts)\n")

# =====================================================================
# Sanity check on the moderation that was applied.
# =====================================================================
# limma's posterior variance must be finite and positive for every transcript
# the prior was estimated from, and the moderated statistics must be finite
# too. A NaN here would silently empty the moderated columns.
#
# `df_post` is checked for NA and NOT for finiteness: an infinite posterior df
# is a legitimate outcome (df.prior = Inf), and it is the normal-tail limit.
if (!is.null(sv)) {
  stopifnot(all(is.finite(sv$var.post)), all(sv$var.post > 0))
  stopifnot(!any(is.na(df_post[finite])))
  stopifnot(all(is.finite(t_sq[finite])), all(is.finite(p_sq[finite])))
  cat("[telescope_diff] moderation sanity check PASS:",
      length(sv$var.post), "posteriors, df.prior =", sv$df.prior, "\n")
}

# =====================================================================
# Session info for reproducibility.
# =====================================================================
writeLines(capture.output(sessionInfo()),
           file.path(opt$`output-dir`, "sessionInfo.txt"))
cat("\n[telescope_diff] done. Output in:", opt$`output-dir`, "\n")