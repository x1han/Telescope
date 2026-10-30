#!/usr/bin/env Rscript
# audit_dpl_models.R -- calibrate the differential poly(A) LENGTH model.
#
# WHY THIS EXISTS. The tail-length model is a STATISTICAL MODEL, and a
# statistical model that has never been calibrated is a claim, not a result.
# This script decides whether a candidate model may be shipped, and it
# decides on measurements, not on how many hits a model produces.
#
# THE UNIT. One transcript x one biological sample. See dpl_models.R for the
# four moments and the Kish effective sample size.
#
# THE CANDIDATES.
#
#   equal_raw_t                 the unweighted two-sample t on the per-sample
#                               means
#   equal_weight_eb             the same fit with limma's empirical Bayes
#                               variance moderation
#   weight_sum_lm               what the shipped engine does today: weight
#                               each sample by its posterior mass, i.e. by a
#                               READ COUNT. Kept as the reference to change
#                               from, and reported next to the candidates.
#   precision_weighted_lm       weight each sample by 1/measurement_variance
#   empirical_bayes_weighted_lm the same fit with limma's empirical Bayes
#                               variance moderation
#   batch_adjusted_lm           the empirical Bayes fit with a batch term,
#                               added ONLY when the design has a batch that
#                               is not confounded with the condition. Random
#                               effects are never forced.
#
# equal_weight_eb is in the list because the shipped output contract carries
# a moderated arm next to the raw one, so the moderated fit that could ship
# has to be audited like every other candidate.
#
# THE FOUR ARMS.
#
#   REAL    the observed fit on the real cohort.
#   PERM    the null distribution of each transcript's statistic, by EXACT
#           enumeration of every way of relabelling the samples. The samples
#           are the unit of exchangeability: a permutation moves a sample's
#           label, never a read inside a sample.
#   NULL    a bootstrap cohort with NO condition effect, so every transcript
#           is known to be null and the false discovery rate can be measured
#           rather than assumed. Two nulls are built, because they ask
#           different questions: `realistic_scatter` gives every cell of a
#           transcript the transcript's pooled within-group variance, which
#           is what the real data look like, and `model_assumed` gives every
#           cell its OWN measurement variance, which is what the
#           precision-weighted model assumes. A candidate that is calibrated
#           under the second but not the first is not broken; it is
#           misspecified for real data, and the report says which.
#   INJECT  the same bootstrap cohorts with a KNOWN effect delta added to a
#           random subset of transcripts. The truth is known exactly, so
#           power and bias can be measured.
#
# WHY THE BOOTSTRAP AND NOT ONLY THE PERMUTATION. The exact permutation
# gives each transcript its own null distribution, and that is what the PERM
# arm uses. It cannot, on its own, give a BH false discovery rate: the real
# cohort carries real signal, so a rejected transcript is not known to be a
# false rejection, and the bootstrap is what makes the truth known. The two
# arms answer different questions and neither substitutes for the other.
#
# NOTHING HERE SELECTS A MODEL BY ITS HIT COUNT. The verdict is a
# calibration verdict: does the procedure control the false discovery rate
# at the nominal level, and does the estimate reproduce the injected effect.
#
# Usage:
#   Rscript audit_dpl_models.R \
#     --aggregate telescope_diff_aggregate.tsv \
#     --design design.tsv \
#     --case COPD --control HC \
#     --output-dir audit/

suppressPackageStartupMessages({
  library(optparse)
  library(limma)
  library(parallel)
})

# --------------------------------------------------------------------------
# Locate the model file
# --------------------------------------------------------------------------
this_script <- tryCatch({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f) == 1) normalizePath(f) else ""
}, error = function(e) "")
model_file <- Sys.getenv("TELESCOPE_DPL_MODELS")
if (!nzchar(model_file)) {
  cand <- c(
    file.path(dirname(this_script), "dpl_models.R"),
    file.path(getwd(), "dpl_models.R")
  )
  cand <- cand[file.exists(cand)]
  if (length(cand) == 0) stop("ERROR: dpl_models.R not found. Set TELESCOPE_DPL_MODELS.")
  model_file <- cand[1]
}
source(model_file)

option_list <- list(
  make_option("--aggregate", type = "character",
              help = "The (transcript x sample) aggregate written by `telescope diff`"),
  make_option("--units", type = "character",
              help = "A unit table built earlier, instead of --aggregate"),
  make_option("--design", type = "character",
              help = "Design TSV with sample_id and condition"),
  make_option("--case", type = "character",
              help = "The case condition label"),
  make_option("--control", type = "character",
              help = "The control condition label"),
  make_option("--output-dir", type = "character", default = "dpl_audit",
              help = "Where the audit tables are written [default %default]"),
  make_option("--batch-column", type = "character", default = "batch",
              help = "Name of the batch column in the design [default %default]"),
  make_option("--min-weight", type = "double", default = 10,
              help = "Min posterior mass for a cell [default %default]"),
  make_option("--min-neff", type = "double", default = 2,
              help = "Min Kish effective sample size for a cell [default %default]"),
  make_option("--min-samples", type = "integer", default = 3,
              help = paste("Min admissible samples per condition for a transcript",
                           "[default %default]. The default matches the engine's",
                           "--min-samples, so the audited population IS the",
                           "shipped population.")),
  make_option("--max-permutations", type = "integer", default = 2000,
              help = "Cap on the enumerated label permutations [default %default]"),
  make_option("--bootstrap", type = "integer", default = 400,
              help = paste("Bootstrap replicates for the NULL arm. The verdict",
                           "resolves alpha, and resolving it needs this many",
                           "cohorts [default %default]")),
  make_option("--inject-replicates", type = "integer", default = 20,
              help = "Replicates per injected effect size [default %default]"),
  make_option("--inject-fraction", type = "double", default = 0.2,
              help = "Fraction of transcripts that receive the injected effect [default %default]"),
  make_option("--deltas", type = "character", default = "0.10,0.20,0.30,0.50",
              help = "Injected effect sizes in log2 units [default %default]"),
  make_option("--alpha", type = "double", default = 0.05,
              help = "The BH level the audit checks [default %default]"),
  make_option("--seed", type = "integer", default = 20251030,
              help = "Seed for the permutations and the bootstrap [default %default]"),
  make_option("--cores", type = "integer", default = 8,
              help = "Worker processes for the permutation loop [default %default]")
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$design)) stop("ERROR: --design is required.")
if (is.null(opt$case))    stop("ERROR: --case is required.")
if (is.null(opt$control)) stop("ERROR: --control is required.")
if (is.null(opt$aggregate) && is.null(opt$units)) {
  stop("ERROR: either --aggregate or --units is required.")
}
dir.create(opt$`output-dir`, recursive = TRUE, showWarnings = FALSE)
set.seed(opt$seed)

# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
des <- read.delim(opt$design, stringsAsFactors = FALSE)
stopifnot(all(c("sample_id", "condition") %in% colnames(des)))
des$sample_id <- as.character(des$sample_id)
des$condition <- as.character(des$condition)

if (!is.null(opt$units)) {
  units <- read.delim(opt$units, stringsAsFactors = FALSE)
} else {
  agg <- read.delim(opt$aggregate, stringsAsFactors = FALSE)
  units <- dpl_units(agg, des, min_weight = opt$`min-weight`, min_neff = opt$`min-neff`)
}
cat("[audit] unit rows:", nrow(units), "\n")
units <- units[units$passed_quality_gate, ]
cat("[audit] quality-gated cells:", nrow(units), "\n")

# The sample order is fixed here and does not depend on how the condition
# labels sort.
samples <- des$sample_id
case_ind <- des$condition == opt$case
ctrl_ind <- des$condition == opt$control
n_case <- sum(case_ind); n_ctrl <- sum(ctrl_ind)
cat("[audit] case:", opt$case, "(", n_case, "samples) control:", opt$control,
    "(", n_ctrl, "samples)\n")
if (n_case < 2 || n_ctrl < 2) {
  stop("ERROR: the contrast needs at least two samples per condition.")
}

batch_vec <- if (opt$`batch-column` %in% colnames(des)) as.character(des[[opt$`batch-column`]]) else NULL
# Two designs: the plain one, and the one that carries a batch term when the
# batch is not confounded with the condition. A model that asks for the batch
# gets the second one only when it really has a batch term.
dl <- dpl_design(samples, des$condition, opt$case, opt$control, NULL)
dl_batch <- dpl_design(samples, des$condition, opt$case, opt$control, batch_vec)
cat("[audit] design:", paste(colnames(dl$design), collapse = ", "), "|", dl$batch_note, "\n")
cat("[audit] batch  :", paste(colnames(dl_batch$design), collapse = ", "), "|", dl_batch$batch_note, "\n")

# --------------------------------------------------------------------------
# Testable transcripts
# --------------------------------------------------------------------------
nc <- tapply(units$condition == opt$case, units$transcript_id, sum)
nt <- tapply(units$condition == opt$control, units$transcript_id, sum)
keep <- names(nc)[nc >= opt$`min-samples` & nt[names(nc)] >= opt$`min-samples`]
transcripts <- sort(unique(keep))
cat("[audit] testable transcripts:", length(transcripts), "\n")
if (length(transcripts) < 10) stop("ERROR: too few testable transcripts to audit.")

gene_of <- tapply(units$gene_id, units$transcript_id, function(x) x[1])
gene_of <- gene_of[transcripts]

mat_pv  <- dpl_matrices(units, samples, transcripts, "mean_log2_pal", "measurement_variance")
mat_ws  <- dpl_matrices(units, samples, transcripts, "mean_log2_pal", NULL)
# weight_sum is not a variance, so it cannot be used as a precision. It is
# used as the shipped engine uses it: a relative weight.
Wsum <- matrix(0, nrow = length(transcripts), ncol = length(samples),
               dimnames = list(transcripts, samples))
uu <- units[units$transcript_id %in% transcripts & units$sample_id %in% samples, ]
Wsum[cbind(match(uu$transcript_id, transcripts), match(uu$sample_id, samples))] <- uu$W
Y <- mat_pv$y

# --------------------------------------------------------------------------
# The fits
# --------------------------------------------------------------------------
# The model layer lives in dpl_models.R and the audit does not re-implement
# it: a second copy of a standard error formula is a second place for it to
# be wrong.
model_specs <- list(
  equal_raw_t                 = list(ebayes = FALSE, weight = "none"),
  equal_weight_eb             = list(ebayes = TRUE,  weight = "none"),
  weight_sum_lm               = list(ebayes = FALSE, weight = "sum"),
  precision_weighted_lm       = list(ebayes = FALSE, weight = "pv"),
  empirical_bayes_weighted_lm = list(ebayes = TRUE,  weight = "pv"),
  batch_adjusted_lm           = list(ebayes = TRUE,  weight = "pv", batch = TRUE)
)
wmat_for <- function(kind) switch(kind, none = NULL, sum = Wsum, pv = mat_pv$w)

# The design a model is fitted to. The batch term is used ONLY when the
# design carries a batch that is not confounded with the condition.
design_for <- function(model) {
  if (isTRUE(model_specs[[model]]$batch) && isTRUE(dl_batch$batch_used)) {
    dl_batch$design
  } else {
    dl$design
  }
}

# A model that cannot be fitted on this cohort says so in every row it
# produces, instead of quietly reporting another model's numbers.
model_note <- function(model) {
  if (isTRUE(model_specs[[model]]$batch) && !isTRUE(dl_batch$batch_used)) {
    paste0("not applicable: ", dl_batch$batch_note,
           "; this row repeats empirical_bayes_weighted_lm")
  } else {
    ""
  }
}

fits <- list()
for (nm in names(model_specs)) {
  sp <- model_specs[[nm]]
  fits[[nm]] <- dpl_fit_model(Y, design_for(nm), wmat_for(sp$weight), sp$ebayes)
}
cat("[audit] observed fits done\n")

# The statistic each model is judged on: the moderated t when the model
# moderates, the raw t otherwise.
stat_of <- function(model, y = Y, design = NULL, weights = NULL) {
  sp <- model_specs[[model]]
  if (is.null(design)) design <- design_for(model)
  dpl_fit_model(y, design, weights, sp$ebayes)$t
}

# --------------------------------------------------------------------------
# The audit validates its own statistics before it uses them
# --------------------------------------------------------------------------
# The unweighted model has to BE the ordinary two-sample t, and the weighted
# model has to BE a weighted least squares fit, to machine precision. If
# either is off, every number below is off in the same direction and the
# comparison between candidates means nothing. This gate is the reason the
# audit is allowed to produce a verdict at all.
validate_statistics <- function(design, weights) {
  complete <- which(rowSums(!is.na(Y)) == length(samples))
  if (length(complete) == 0) {
    stop("ERROR: no transcript has a complete row; the statistics cannot be validated.")
  }
  i <- complete[1]
  v <- Y[i, ]
  tt <- stats::t.test(v[case_ind], v[ctrl_ind])
  f0 <- limma::lmFit(matrix(v, nrow = 1), design)
  se0 <- f0$stdev.unscaled[1, "case"] * f0$sigma[1]
  t0 <- f0$coefficients[1, "case"] / se0
  if (!isTRUE(all.equal(unname(se0), unname(tt$stderr), tolerance = 1e-10)) ||
      !isTRUE(all.equal(unname(t0), unname(tt$statistic), tolerance = 1e-10))) {
    stop("ERROR: the unweighted model is not the ordinary two-sample t. ",
         "se = ", se0, " against ", tt$stderr, ", t = ", t0, " against ",
         tt$statistic, ". The audit cannot continue on an unverified statistic.")
  }
  if (!is.null(weights)) {
    w <- weights[i, ]
    keep <- is.finite(w) & w > 0
    X <- design[keep, , drop = FALSE]
    yv <- v[keep]
    wv <- w[keep]
    A <- crossprod(X, X * wv)
    beta <- solve(A, crossprod(X, yv * wv))
    r <- as.vector(yv - X %*% beta)
    se_hand <- sqrt(sum(wv * r^2) / (length(yv) - ncol(X)) * solve(A)[2, 2])
    fw <- limma::lmFit(matrix(v, nrow = 1), design,
                        weights = matrix(w, nrow = 1))
    se_lim <- fw$stdev.unscaled[1, "case"] * fw$sigma[1]
    if (!isTRUE(all.equal(unname(se_lim), unname(se_hand), tolerance = 1e-8))) {
      stop("ERROR: the weighted model does not match a hand-computed weighted ",
           "least squares fit. se = ", se_lim, " against ", se_hand,
           ". The audit cannot continue on an unverified statistic.")
    }
  }
  invisible(TRUE)
}
cat("[audit] validating the statistics against an independent computation...\n")
validate_statistics(dl$design, mat_pv$w)
cat("[audit]   the unweighted model reproduces the ordinary two-sample t,",
    "the weighted model a weighted least squares fit\n")

# --------------------------------------------------------------------------
# Arm REAL
# --------------------------------------------------------------------------
real_rows <- list()
for (nm in names(model_specs)) {
  f <- fits[[nm]]
  q <- stats::p.adjust(f$p, method = "BH")
  real_rows[[nm]] <- data.frame(
    model_name = nm,
    transcript_id = transcripts,
    gene_id = unname(gene_of[transcripts]),
    estimate = f$estimate,
    se = f$se,
    df_raw = f$df_raw,
    t_raw = f$t,
    p_raw = f$p,
    q_raw = q,
    df_post = f$df_post,
    model_parameters = paste0(
      # With trend = TRUE the prior is fitted per transcript, so it is
      # SUMMARISED here: a per-row vector would not even fit the column.
      ifelse(all(is.na(f$s2_prior)), "",
             paste0("limma eBayes: s2.prior ", dpl_prior_summary(f$s2_prior),
                    ", df.prior ", dpl_prior_summary(f$df_prior))),
      model_note(nm)),
    stringsAsFactors = FALSE
  )
}
real_tab <- do.call(rbind, real_rows)
write.table(real_tab, file = file.path(opt$`output-dir`, "dpl_model_fit.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

units_out <- units[, c("sample_id", "transcript_id", "gene_id", "condition",
                        "mean_log2_pal", "within_variance", "W", "Kish_Neff",
                        "measurement_variance", "passed_quality_gate")]
write.table(units_out, file = file.path(opt$`output-dir`, "dpl_units.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --------------------------------------------------------------------------
# Arm PERM: the exact enumeration of the sample-label permutations
# --------------------------------------------------------------------------
n_perm_total <- choose(n_case + n_ctrl, n_case)
exact <- n_perm_total <= opt$`max-permutations`
obs_case <- which(case_ind)
cmb <- combn(seq_len(n_case + n_ctrl), n_case)
# The identity labelling (the observed one) is always part of the
# enumeration, so the smallest attainable p-value is 1/(1+N) and the
# observed statistic is inside its own null.
ident <- which(vapply(seq_len(ncol(cmb)),
                      function(j) setequal(cmb[, j], obs_case), logical(1)))[1]
perm_index <- if (exact) {
  list(case = cmb)
} else {
  pick <- sample(setdiff(seq_len(ncol(cmb)), ident), opt$`max-permutations` - 1,
                 replace = FALSE)
  list(case = cbind(cmb[, ident, drop = FALSE], cmb[, pick, drop = FALSE]))
}
n_perm <- ncol(perm_index$case)
cat("[audit] permutations:", n_perm, "of", n_perm_total,
    if (exact) "(EXACT enumeration)" else "(SAMPLED)", "\n")

# The statistic of the observed labelling is one of the permutations: the
# identity. It is included, and the p-value uses the (1 + count) / (1 + N)
# convention, so the smallest attainable p-value is 1 / (1 + N).
perm_stats <- function(model) {
  sp <- model_specs[[model]]
  w <- wmat_for(sp$weight)
  d <- design_for(model)
  t_obs <- stat_of(model, Y, d, w)
  chunks <- parallel::mclapply(seq_len(n_perm), function(j) {
    idx <- perm_index$case[, j]
    d2 <- d
    d2[, "case"] <- 0
    d2[idx, "case"] <- 1
    stat_of(model, Y, d2, w)
  }, mc.cores = opt$cores)
  counts <- numeric(length(transcripts))
  usable <- numeric(length(transcripts))
  for (tp in chunks) {
    # A permutation can put every observation of a transcript on one side of
    # the contrast, which makes the design rank-deficient and the statistic
    # NA. Such a permutation is not part of that transcript's null and must
    # not be counted as "not extreme" either, or it would inflate the
    # p-value.
    ok <- !is.na(tp) & !is.na(t_obs)
    counts <- counts + ifelse(ok & (abs(tp) >= abs(t_obs) - 1e-12), 1, 0)
    usable <- usable + as.numeric(ok)
  }
  p_perm <- (1 + counts) / (1 + usable)
  list(p = p_perm, t_obs = t_obs, usable = usable)
}

perm_res <- list()
if (n_perm <= 2000) {
  for (model in names(model_specs)) {
    cat("[audit] permuting:", model, "\n")
    perm_res[[model]] <- perm_stats(model)
  }
} else {
  cat("[audit] too many permutations for the serial loop; skipping the PERM arm\n")
}

perm_rows <- list()
for (model in names(perm_res)) {
  pp <- perm_res[[model]]$p
  an <- fits[[model]]$p
  ok <- !is.na(pp) & !is.na(an)
  perm_rows[[model]] <- data.frame(
    model_name = model,
    n_permutations = n_perm,
    permutation_space = n_perm_total,
    enumeration = if (exact) "exact" else "sampled",
    min_attainable_p = 1 / (1 + n_perm),
    n_transcripts = length(pp),
    median_permutation_p = median(pp, na.rm = TRUE),
    frac_permutation_p_le_0.05 = mean(pp <= 0.05, na.rm = TRUE),
    median_analytic_p = median(an, na.rm = TRUE),
    median_p_difference = median(an[ok] - pp[ok], na.rm = TRUE),
    frac_analytic_p_below_permutation_p = mean(an[ok] < pp[ok], na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}
if (length(perm_rows) > 0) {
  write.table(do.call(rbind, perm_rows),
              file = file.path(opt$`output-dir`, "dpl_permutation_null.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
}

# --------------------------------------------------------------------------
# A positive control for the permutation arm
# --------------------------------------------------------------------------
# The permutation arm is checked against a case whose answer is known: a
# simulated cohort with no condition effect at all. There the analytic
# p-value has to be uniform (its mean is 0.5, which is a fact about the
# t-distribution, not a tuning choice), and the permutation p-value has to
# be a multiple of 1/(1+N). The fraction of transcripts where the analytic
# p falls below the permutation p is the same statistic as on the real
# cohort, so this control says what that fraction looks like when nothing
# is wrong, and the real cohort's value can be read against it.
perm_self_test <- function(n_case, n_ctrl, n_tx = 2000, seed = opt$seed) {
  set.seed(seed)
  y <- matrix(rnorm(n_tx * (n_case + n_ctrl)), nrow = n_tx)
  des <- data.frame(
    sample_id = as.character(seq_len(n_case + n_ctrl)),
    condition = c(rep("case", n_case), rep("control", n_ctrl)),
    stringsAsFactors = FALSE
  )
  d <- dpl_design(des$sample_id, des$condition, "case", "control", NULL)
  # Weight matrices with the same shape as the simulated data, and with the
  # real cohort's weight dispersion, so the weighted models are exercised
  # here under the heterogeneity they will actually meet.
  draw_from <- function(m, positive_only = TRUE) {
    v <- m[is.finite(m) & (m > 0 | !positive_only)]
    if (length(v) == 0) v <- rep(1, n_tx * (n_case + n_ctrl))
    matrix(sample(v, n_tx * (n_case + n_ctrl), replace = TRUE), nrow = n_tx)
  }
  w_pv <- draw_from(mat_pv$w)
  w_sum <- draw_from(Wsum)
  cmb <- combn(seq_len(n_case + n_ctrl), n_case)
  n_perm <- ncol(cmb)
  out <- list()
  for (nm in names(model_specs)) {
    sp <- model_specs[[nm]]
    ww <- switch(sp$weight, none = NULL, sum = w_sum, pv = w_pv)
    fit <- dpl_fit_model(y, d$design, ww, sp$ebayes)
    cnt <- numeric(n_tx)
    for (j in seq_len(n_perm)) {
      d2 <- d$design
      d2[, "case"] <- 0
      d2[cmb[, j], "case"] <- 1
      tp <- dpl_fit_model(y, d2, ww, sp$ebayes)$t
      ok <- !is.na(tp) & !is.na(fit$t)
      cnt <- cnt + ifelse(ok & (abs(tp) >= abs(fit$t) - 1e-12), 1, 0)
    }
    p_perm <- (1 + cnt) / (1 + n_perm)
    out[[nm]] <- data.frame(
      model_name = nm,
      mean_analytic_p = mean(fit$p, na.rm = TRUE),
      sd_analytic_p = stats::sd(fit$p, na.rm = TRUE),
      mean_permutation_p = mean(p_perm, na.rm = TRUE),
      frac_analytic_p_below_permutation_p = mean(fit$p < p_perm, na.rm = TRUE),
      min_permutation_p = min(p_perm),
      n_transcripts = n_tx,
      n_permutations = n_perm,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, out)
}
cat("[audit] permutation positive control on a simulated null cohort\n")
perm_control <- perm_self_test(n_case, n_ctrl)
write.table(perm_control,
            file = file.path(opt$`output-dir`, "dpl_permutation_control.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
print(perm_control)

# --------------------------------------------------------------------------
# Arms NULL and INJECT: cohorts whose truth is known
# --------------------------------------------------------------------------
# A null cohort carries no condition effect, so every transcript in it is
# known to be null and the false discovery rate can be measured rather than
# assumed. The between-group difference is what a null cohort must not have,
# so it is not carried over.
#
# Two nulls are built, because they ask different questions:
#
#   realistic_scatter  every cell of a transcript scatters with the
#                      transcript's pooled within-group variance, which is
#                      what the real cohort looks like. This is the null a
#                      shipped model has to survive.
#   model_assumed      every cell scatters with its OWN measurement variance,
#                      which is what the precision-weighted model assumes. A
#                      model that is calibrated under this one but not under
#                      the realistic one is not broken; it is misspecified
#                      for real data, and the report says which.
#
# All of this is computed with matrix algebra: a per-transcript loop over
# twenty thousand transcripts cost minutes and bought nothing.
present <- !is.na(Y)
# Every row is the same case/control mask over the samples.
case_mask <- matrix(rep(case_ind, each = length(transcripts)),
                     nrow = length(transcripts), ncol = length(samples))

group_ss <- function(mask) {
  m <- mask & present
  n <- rowSums(m)
  s1 <- rowSums(m * Y, na.rm = TRUE)
  s2 <- rowSums(m * Y^2, na.rm = TRUE)
  ss <- s2 - ifelse(n > 0, s1^2 / n, 0)
  ss[!is.finite(ss)] <- 0
  ss[ss < 0] <- 0
  list(n = n, ss = ss)
}
g_case <- group_ss(case_mask)
g_ctrl <- group_ss(!case_mask)
pooled_var <- (g_case$ss + g_ctrl$ss) /
  pmax(1, g_case$n + g_ctrl$n - 2)
grand_mean <- rowSums(Y, na.rm = TRUE) / pmax(1, rowSums(present))
usable <- is.finite(pooled_var) & pooled_var > 0
cat("[audit] transcripts with a pooled within-group variance:",
    sum(usable), "of", length(transcripts), "\n")

# The per-cell standard deviation of each null. A per-transcript value has
# to be constant along a ROW, so the matrix is filled by column: filling it
# by row gives each group a different transcript's scale, and the pooled
# two-sample t on the cell means stops being exact.
sd_realistic <- matrix(sqrt(pooled_var), nrow = length(transcripts),
                       ncol = length(samples))
if (!all(sd_realistic[1, ] == sd_realistic[1, 1])) {
  stop("ERROR: the null cohort's standard deviations are not constant along ",
       "a transcript; the two groups would not carry the same variance.")
}
sd_realistic[!usable, ] <- NA_real_
sd_assumed <- matrix(NA_real_, nrow = length(transcripts), ncol = length(samples),
                     dimnames = dimnames(Y))
have_w <- mat_pv$w > 0
sd_assumed[have_w] <- 1 / sqrt(mat_pv$w[have_w])

# The model's own scale is the measurement variance, and it differs from
# sample to sample. A cohort whose case samples are deeper than its control
# samples therefore gets two different scales under this null, and a null
# whose groups carry different variances is not a null the pooled two-sample t
# is exact on -- the calibration test would then measure the null's own
# asymmetry instead of the model. The scale is taken per TRANSCRIPT as the
# geometric mean of the two groups' scales: the model's scale, with the null's
# validity restored. The ratio the two groups had before that is reported,
# because it is a fact about the cohort and not about the model.
group_geo <- function(mask) {
  m <- mask & !is.na(sd_assumed)
  n <- rowSums(m)
  out <- rep(NA_real_, length(transcripts))
  ok <- n > 0
  out[ok] <- exp(rowSums(m * log(sd_assumed), na.rm = TRUE)[ok] / n[ok])
  out
}
g_case_sd <- group_geo(case_mask)
g_ctrl_sd <- group_geo(!case_mask)
scale_ratio <- g_case_sd / g_ctrl_sd
med_scale_ratio <- stats::median(scale_ratio[is.finite(scale_ratio)])
cat("[audit] model's assumed group scale ratio (case/control), median:",
    med_scale_ratio, "\n")
combined_sd <- sqrt(g_case_sd * g_ctrl_sd)
sd_assumed <- matrix(combined_sd, nrow = length(transcripts), ncol = length(samples),
                     dimnames = dimnames(Y))
sd_assumed[!usable] <- NA_real_
null_variants <- list(realistic_scatter = sd_realistic,
                      model_assumed = sd_assumed)

# A bootstrap cohort: the same transcripts, the same cells, no condition
# effect.
make_boot <- function(seed, sd) {
  set.seed(seed)
  z <- matrix(rnorm(length(Y)), nrow = nrow(Y), ncol = ncol(Y))
  yb <- grand_mean + z * sd
  yb[!present] <- NA_real_
  yb
}

# Guard: the null cohort's two groups have to carry the same variance. The
# pooled two-sample t on the cell means is exact only when they do, and a
# null cohort that breaks that silently shifts every p-value the audit
# reads off it.
check_null_variance <- function(yb, variant) {
  idx <- which(rowSums(!is.na(yb)) == length(samples))
  # The median of a ratio of two sample variances is 1 exactly, but its
  # sampling error over a few hundred transcripts is wide enough to look
  # like a defect. A few thousand makes the check sharp.
  idx <- idx[seq_len(min(3000, length(idx)))]
  ratio <- vapply(idx, function(i) {
    stats::var(yb[i, case_ind]) / stats::var(yb[i, !case_ind])
  }, numeric(1))
  med <- stats::median(ratio)
  if (!is.finite(med) || med < 0.7 || med > 1.4) {
    stop("ERROR: the null cohort (", variant, ") does not give the two groups ",
         "the same variance; the median variance ratio is ", med, ".")
  }
  invisible(med)
}

# The models, applied to an arbitrary cohort.
fit_models <- function(y) {
  out <- list()
  for (nm in names(model_specs)) {
    sp <- model_specs[[nm]]
    out[[nm]] <- dpl_fit_model(y, design_for(nm), wmat_for(sp$weight), sp$ebayes)
  }
  out
}

# --- NULL arm -------------------------------------------------------------
cat("[audit] NULL arm:", opt$bootstrap, "bootstrap cohorts per null variant\n")
for (vname in names(null_variants)) {
  med <- check_null_variance(make_boot(opt$seed + 1000 + 1L, null_variants[[vname]]),
                              vname)
  cat("[audit]   null variant", vname, "median group variance ratio:", med, "\n")
}
null_tab <- do.call(rbind, lapply(names(null_variants), function(vname) {
  sd <- null_variants[[vname]]
  res <- parallel::mclapply(seq_len(opt$bootstrap), function(rep) {
    yb <- make_boot(opt$seed + 1000 + rep, sd)
    fb <- fit_models(yb)
    do.call(rbind, lapply(names(fb), function(nm) {
      p <- fb[[nm]]$p
      q <- stats::p.adjust(p, method = "BH")
      data.frame(model_name = nm, null_variant = vname, replicate = rep,
                 n_tests = sum(!is.na(p)),
                 n_rejected = sum(q <= opt$alpha, na.rm = TRUE),
                 rejection_rate = sum(q <= opt$alpha, na.rm = TRUE) / sum(!is.na(p)),
                 mean_p = mean(p, na.rm = TRUE),
                 stringsAsFactors = FALSE)
    }))
  }, mc.cores = opt$cores)
  cat("[audit]   null variant", vname, "done\n")
  do.call(rbind, res)
}))
write.table(null_tab, file = file.path(opt$`output-dir`, "dpl_null_calibration.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- INJECT arm -----------------------------------------------------------
deltas <- as.numeric(strsplit(opt$deltas, ",")[[1]])
cat("[audit] INJECT arm: deltas", paste(deltas, collapse = ", "), "\n")
inj_tab <- do.call(rbind, lapply(names(null_variants), function(vname) {
  sd <- null_variants[[vname]]
  jobs <- expand.grid(delta = deltas, replicate = seq_len(opt$`inject-replicates`))
  res <- parallel::mclapply(seq_len(nrow(jobs)), function(j) {
    delta <- jobs$delta[j]
    rep <- jobs$replicate[j]
    set.seed(opt$seed + 50000 + as.integer(delta * 1000) * 100 + rep)
    yb <- make_boot(opt$seed + 2000 + rep, sd)
    # A known subset of transcripts receives the known effect on the case
    # side. The subset is drawn independently of the data.
    n_inj <- floor(opt$`inject-fraction` * length(transcripts))
    inj <- sample(seq_along(transcripts), n_inj)
    # The effect goes on the CASE side only. The control cells keep their
    # null value, so the case-minus-control estimate is expected to be
    # +delta; adding delta to both sides would leave the contrast at zero
    # and measure nothing.
    inj_mask <- case_mask[inj, , drop = FALSE]
    yb[inj, ] <- yb[inj, , drop = FALSE] + delta * as.numeric(inj_mask)
    fb <- fit_models(yb)
    do.call(rbind, lapply(names(fb), function(nm) {
      p <- fb[[nm]]$p
      q <- stats::p.adjust(p, method = "BH")
      rej <- q <= opt$alpha
      rej[is.na(rej)] <- FALSE
      is_inj <- seq_along(transcripts) %in% inj
      est <- fb[[nm]]$estimate
      data.frame(model_name = nm, null_variant = vname, delta = delta,
                 replicate = rep, n_injected = n_inj,
                 power = sum(rej & is_inj) / n_inj,
                 n_false_rejections = sum(rej & !is_inj),
                 n_rejections = sum(rej),
                 fdr = if (sum(rej) > 0) sum(rej & !is_inj) / sum(rej) else NA_real_,
                 bias = mean(est[is_inj] - delta, na.rm = TRUE),
                 bias_sd = stats::sd(est[is_inj] - delta, na.rm = TRUE),
                 bias_null_mean = mean(est[!is_inj], na.rm = TRUE),
                 stringsAsFactors = FALSE)
    }))
  }, mc.cores = opt$cores)
  cat("[audit]   inject variant", vname, "done\n")
  do.call(rbind, res)
}))
write.table(inj_tab, file = file.path(opt$`output-dir`, "dpl_injection_power.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------
# Two criteria, and both tolerances are MEASURED, not chosen.
#
# 1. The permutation arm. The analytic p-value claims to be the probability
#    that the statistic is as extreme as it is under the null; the exact
#    permutation p-value IS that probability, enumerated. When the two agree,
#    the analytic p falls below the permutation p about half the time. The
#    reference model is `equal_raw_t`, whose calibration is a mathematical
#    fact (it is the ordinary two-sample t, checked above), so its measured
#    departure from one half is the yardstick for the others.
#
# 2. The null arm. Every transcript of a null cohort is null, so every
#    rejection in it is a false discovery, and BH's guarantee is that the
#    probability of ANY false discovery is at most alpha. The measurement is
#    therefore the fraction of null cohorts in which the model rejected
#    something at all, and the test is the exact binomial one: calibration
#    is rejected when the observed number of cohorts with a rejection is
#    significantly larger than alpha would produce. That is why the number
#    of null cohorts has to be large enough to resolve alpha, and why the
#    count of cohorts is reported next to the verdict.
#
# The verdict is about calibration. The number of discoveries a model makes
# on the real cohort is reported, and it is never part of the verdict.
REF_MODEL <- "equal_raw_t"
perm_tab <- if (length(perm_rows) > 0) {
  do.call(rbind, perm_rows)
} else {
  # The permutation arm did not run; the verdict then has no permutation
  # criterion rather than a made-up one.
  data.frame(model_name = character(0),
             frac_analytic_p_below_permutation_p = numeric(0))
}
perm_ref_dev <- if (nrow(perm_tab) == 0) NA_real_ else abs(
  perm_tab$frac_analytic_p_below_permutation_p[perm_tab$model_name == REF_MODEL] - 0.5)
perm_granularity <- 1 / (1 + n_perm)
cat("[audit] permutation yardstick: the reference model's departure from",
    "one half is", perm_ref_dev, "\n")

# The positive control's band. On the simulated null cohort the mean
# p-value has to be 0.5, and the reference model says how close the
# machinery gets there; the Monte-Carlo error of the mean over the control's
# transcripts is the other half of the band.
ref_ctrl <- perm_control[perm_control$model_name == REF_MODEL, ]
ctrl_band <- if (nrow(ref_ctrl) == 0) {
  NA_real_
} else {
  max(abs(ref_ctrl$mean_analytic_p - 0.5),
      2 * ref_ctrl$sd_analytic_p / sqrt(ref_ctrl$n_transcripts))
}
cat("[audit] uniformity band on the control cohort:", ctrl_band, "\n")

null_stats <- function(model, variant) {
  r <- null_tab[null_tab$model_name == model & null_tab$null_variant == variant, ]
  if (nrow(r) == 0) {
    return(list(p_any = NA_real_, n_any = NA_integer_, n = 0L, excess_p = NA_real_))
  }
  n_any <- sum(r$n_rejected > 0)
  list(p_any = n_any / nrow(r),
       n_any = n_any,
       n = nrow(r),
       # One-sided exact binomial: how surprising is this many cohorts with
       # a rejection, if the model really rejected nothing more often than
       # alpha?
       excess_p = stats::pbinom(n_any - 1, nrow(r), opt$alpha, lower.tail = FALSE))
}

summ <- list()
for (nm in names(model_specs)) {
  applicable <- !(isTRUE(model_specs[[nm]]$batch) && !isTRUE(dl_batch$batch_used))
  st <- null_stats(nm, "realistic_scatter")
  st_assumed <- null_stats(nm, "model_assumed")
  null_pass <- !is.na(st$excess_p) && st$excess_p > 0.05
  crow <- perm_control[perm_control$model_name == nm, ]
  ctrl_dev <- if (nrow(crow) == 0) NA_real_ else abs(crow$mean_analytic_p - 0.5)
  ctrl_pass <- !is.na(ctrl_dev) && ctrl_dev <= ctrl_band
  prow <- perm_tab[perm_tab$model_name == nm, ]
  perm_dev <- if (nrow(prow) == 0) NA_real_ else
    abs(prow$frac_analytic_p_below_permutation_p - 0.5)
  perm_pass <- !is.na(perm_dev) && perm_dev <= max(perm_ref_dev, perm_granularity)
  it <- inj_tab[inj_tab$model_name == nm &
                  inj_tab$null_variant == "realistic_scatter", ]
  # The false discovery rate of the injected cohorts, pooled over the
  # replicates: a maximum over replicates of a ratio with a handful of
  # rejections in it is noise.
  n_rej <- sum(it$n_rejections)
  pooled_fdr <- if (n_rej > 0) sum(it$n_false_rejections) / n_rej else NA_real_
  power_by_delta <- vapply(deltas, function(d) {
    r <- it[it$delta == d, ]
    if (nrow(r) == 0) return(NA_real_)
    mean(r$power)
  }, numeric(1))
  bias_by_delta <- vapply(deltas, function(d) {
    r <- it[it$delta == d, ]
    if (nrow(r) == 0) return(NA_real_)
    mean(r$bias)
  }, numeric(1))
  rp <- real_tab[real_tab$model_name == nm, ]
  verdict <- if (!applicable) {
    "NOT_APPLICABLE"
  } else if (!ctrl_pass) {
    "P_VALUE_NOT_UNIFORM"
  } else if (!null_pass) {
    "FDR_NOT_CONTROLLED"
  } else if (!perm_pass) {
    "PERMUTATION_MISMATCH"
  } else {
    "CALIBRATED"
  }
  summ[[nm]] <- data.frame(
    model_name = nm,
    verdict = verdict,
    control_mean_analytic_p = if (nrow(crow) == 0) NA_real_ else crow$mean_analytic_p,
    control_uniformity_deviation = ctrl_dev,
    control_uniformity_band = ctrl_band,
    control_frac_analytic_below_perm = if (nrow(crow) == 0) NA_real_ else
      crow$frac_analytic_p_below_permutation_p,
    null_cohorts = st$n,
    null_cohorts_with_a_rejection = st$n_any,
    null_p_any_rejection = st$p_any,
    null_excess_binomial_p = st$excess_p,
    null_p_any_rejection_model_assumed = st_assumed$p_any,
    null_excess_binomial_p_model_assumed = st_assumed$excess_p,
    perm_frac_analytic_below_perm = if (nrow(prow) == 0) NA_real_ else
      prow$frac_analytic_p_below_permutation_p,
    perm_deviation_from_one_half = perm_dev,
    perm_reference_deviation = perm_ref_dev,
    alpha = opt$alpha,
    injection_pooled_fdr = pooled_fdr,
    injection_power = paste(sprintf("d=%.2f:%.4f", deltas, power_by_delta),
                             collapse = " "),
    injection_bias = paste(sprintf("d=%.2f:%+.4f", deltas, bias_by_delta),
                           collapse = " "),
    n_real_discoveries_q_le_alpha = sum(rp$q_raw <= opt$alpha, na.rm = TRUE),
    n_real_tested = sum(!is.na(rp$p_raw)),
    batch_note = dl_batch$batch_note,
    stringsAsFactors = FALSE
  )
}
summ_tab <- do.call(rbind, summ)
write.table(summ_tab, file = file.path(opt$`output-dir`, "dpl_audit_summary.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat("\n[audit] summary\n")
print(summ_tab)
cat("\n[audit] wrote dpl_units.tsv, dpl_model_fit.tsv, dpl_permutation_null.tsv,",
    "dpl_null_calibration.tsv, dpl_injection_power.tsv, dpl_audit_summary.tsv to",
    opt$`output-dir`, "\n")
