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
#   weight_sum_lm               weight each sample by its posterior mass, i.e.
#                               by a READ COUNT. NOT what the shipped engine
#                               does: the shipped path collapses each cell to
#                               its posterior-mass-weighted mean_log2_pal and
#                               then fits UNWEIGHTED -- telescope_diff.R:1056
#                               calls dpl_fit_both(y_mat, design, NULL), which
#                               is equal_raw_t. Measured against the shipped
#                               p_raw: equal_raw_t agrees to 2.11e-12 and
#                               weight_sum_lm differs by up to 2.69 (relative),
#                               so they are different estimators. The
#                               measurement that settles this lives in the
#                               project workspace layer, under
#                               validation/dpl_model_review_round206_5/scripts/
#                               (which_model_is_shipped.py); that harness is
#                               internal material and is not part of this
#                               public tree. The shipped model is the
#                               REF_MODEL / DISC_MODEL below.
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
# THE PERMUTATION ARM HAS A POSITIVE CONTROL, AND THE CONTROL IS
# REPLICATED. The permutation statistic is read against a simulated null
# cohort that carries the REAL cohort's structure, and the control is run
# on several independent null cohorts, so that the baseline the real
# cohort's value is read against and the band it is read against are both
# MEASURED rather than chosen. Neither of the two is ever taken from the
# model under test: a band that contains the model's own measurement is a
# band the model cannot leave, and a criterion that cannot fail is not a
# criterion.
#
# THE DISCOVERY-COUNT INSTRUMENT. The script also enumerates, for every
# labelling it enumerates anyway, how many discoveries that labelling
# makes. It is a measurement and not a criterion, and it answers a
# question the permutation statistic does not: are the hits just the deep
# transcripts, or did the label find something?
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
  make_option("--min-weight", type = "double", default = 5,
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
  make_option("--control-replicates", type = "integer", default = 5,
              help = paste("Independent null cohorts for the permutation positive",
                           "control. The control's measured value is the",
                           "permutation statistic's baseline and its spread",
                           "across these cohorts is the acceptance band",
                           "[default %default]")),
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
# The real cohort's structure
# --------------------------------------------------------------------------
# The quantities below are the real cohort's structure, and they are
# computed ONCE and read by two arms: the permutation positive control
# builds its null cohorts from them, and the NULL arm at the end of this
# script builds its null cohorts from the same ones. A structure computed
# twice is a structure that can be computed two ways.
present <- !is.na(Y)
# Every row is the same case/control mask over the samples.
case_mask <- matrix(rep(case_ind, each = length(transcripts)),
                     nrow = length(transcripts), ncol = length(samples))

# The pooled within-group sum of squares of a transcript's cells.
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
# The transcript's own scale: the scatter of its cell means around the
# two group means. This is the real cohort's BETWEEN-TRANSCRIPT
# dispersion, and it is what a simulated null cohort has to carry.
pooled_var <- (g_case$ss + g_ctrl$ss) /
  pmax(1, g_case$n + g_ctrl$n - 2)
grand_mean <- rowSums(Y, na.rm = TRUE) / pmax(1, rowSums(present))
usable <- is.finite(pooled_var) & pooled_var > 0
cat("[audit] transcripts with a pooled within-group variance:",
    sum(usable), "of", length(transcripts), "\n")
cat("[audit] the real cohort's observed cell-missing rate:",
    mean(!present), "\n")

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
#
# The SAME engine serves the real cohort and the positive control below:
# the control's null cohort is another response matrix, and the control's
# statistic is only comparable with the real cohort's if both are computed
# by the same code on the same enumeration.
perm_stats <- function(model, y = Y, weights = NULL, design = NULL,
                       cmb = perm_index$case) {
  sp <- model_specs[[model]]
  w <- if (is.null(weights)) wmat_for(sp$weight) else weights
  d <- if (is.null(design)) design_for(model) else design
  fit_obs <- dpl_fit_model(y, d, w, isTRUE(sp$ebayes))
  t_obs <- fit_obs$t
  chunks <- parallel::mclapply(seq_len(ncol(cmb)), function(j) {
    idx <- cmb[, j]
    d2 <- d
    d2[, "case"] <- 0
    d2[idx, "case"] <- 1
    stat_of(model, y, d2, w)
  }, mc.cores = opt$cores)
  counts <- numeric(nrow(y))
  usable <- numeric(nrow(y))
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
  list(p = p_perm, t_obs = t_obs, usable = usable,
       p_analytic = fit_obs$p)
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
# The discovery-count instrument
# --------------------------------------------------------------------------
# A second instrument, and it is NOT a criterion: it is a measurement, and
# it does not use the model's own numbers as its yardstick. For EVERY
# labelling the script enumerates it computes the analytic p-values, the
# BH q-values over that labelling's OWN universe, and the number of
# discoveries at q < alpha.
#
# The complement labelling -- case and control swapped -- is one of the
# enumerated labellings and it is REPORTED AS ITS OWN ROW. It is not a
# separate experiment: for a two-sided two-sample t, swapping the two
# groups gives the same |t|, the same p, the same BH q and the same
# number of discoveries. The row is written out anyway, because a reader
# who counts the rows has to be able to see it, and the summary says how
# many labellings were enumerated and how many of them the observed and
# the complement are. The complement is NOT folded into an "other" bucket.
#
# The instrument answers "are the hits just the deep transcripts?". For
# every labelling it also reports the median posterior mass W of that
# labelling's top-K set, with K the number of discoveries the OBSERVED
# labelling makes, so the comparison is like-for-like: if the observed
# labelling's hits are no deeper than the other labellings' hits, then
# the hits are a selection effect and not the label.
#
# The instrument is computed on the base fit, equal_raw_t, and its rows
# name the model they belong to.
DISC_MODEL <- "equal_raw_t"
disc_tab <- NULL
if (length(perm_res) > 0) {
  # The transcript's posterior mass: the sum of its cells' posterior mass
  # over the cohort's samples. It is a property of the transcript and not
  # of the labelling, so the same number is compared across the
  # labellings.
  W_tx <- rowSums(Wsum)
  # The complement labelling: the observed case set's complement, which is
  # the control set, is itself one of the enumerated labellings.
  obs_ctrl <- which(!case_ind)
  compl <- which(vapply(seq_len(ncol(perm_index$case)),
                         function(j) setequal(perm_index$case[, j], obs_ctrl),
                         logical(1)))[1]
  disc_model <- DISC_MODEL
  sp_disc <- model_specs[[disc_model]]
  w_disc <- wmat_for(sp_disc$weight)
  d_disc <- design_for(disc_model)
  cmb_disc <- perm_index$case
  disc_fit <- function(j) {
    d2 <- d_disc
    d2[, "case"] <- 0
    d2[cmb_disc[, j], "case"] <- 1
    dpl_fit_model(Y, d2, w_disc, isTRUE(sp_disc$ebayes))
  }
  # The median W of the top-K set of a labelling, with K taken from the
  # OBSERVED labelling so that every labelling is compared over the same
  # number of transcripts.
  topk_median_W <- function(tv, ok, K) {
    if (is.na(K) || K <= 0) return(NA_real_)
    idx <- which(ok)
    if (length(idx) == 0) return(NA_real_)
    idx <- idx[order(abs(tv[idx]), decreasing = TRUE)]
    idx <- idx[seq_len(min(K, length(idx)))]
    stats::median(W_tx[idx])
  }
  # The observed labelling first: its number of discoveries is the K.
  f_obs <- disc_fit(ident)
  q_obs <- stats::p.adjust(f_obs$p, method = "BH")
  ok_obs <- !is.na(f_obs$p)
  K_obs <- sum(q_obs <= opt$alpha, na.rm = TRUE)
  disc_rows <- parallel::mclapply(seq_len(ncol(cmb_disc)), function(j) {
    f <- disc_fit(j)
    q <- stats::p.adjust(f$p, method = "BH")
    ok <- !is.na(f$p)
    data.frame(
      model_name = disc_model,
      labelling_index = j,
      is_observed = j == ident,
      is_complement_of_observed = j == compl,
      n_discoveries = sum(q <= opt$alpha, na.rm = TRUE),
      n_tested = sum(ok),
      top_K = K_obs,
      median_W_top_K = topk_median_W(f$t, ok, K_obs),
      stringsAsFactors = FALSE
    )
  }, mc.cores = opt$cores)
  disc_tab <- do.call(rbind, disc_rows)
  write.table(disc_tab,
              file = file.path(opt$`output-dir`, "dpl_permutation_discovery_counts.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
  disc_other <- disc_tab[!disc_tab$is_observed & !disc_tab$is_complement_of_observed, ]
  cat("[audit] discovery-count instrument:", ncol(perm_index$case),
      "distinct labellings enumerated,", sum(disc_tab$is_observed | disc_tab$is_complement_of_observed),
      "of them the observed and the complement labelling (the complement is",
      "the same experiment with the groups swapped, not a second one)\n")
  cat("[audit]   observed labelling:", sum(disc_tab$n_discoveries[disc_tab$is_observed]),
      "discoveries at q <=", opt$alpha, "of", disc_tab$n_tested[disc_tab$is_observed],
      "transcripts tested; complement labelling:",
      disc_tab$n_discoveries[disc_tab$is_complement_of_observed],
      "discoveries\n")
  if (nrow(disc_other) > 0) {
    cat("[audit]   the other", nrow(disc_other), "labellings:",
        "discoveries median", stats::median(disc_other$n_discoveries),
        "range", min(disc_other$n_discoveries), "..", max(disc_other$n_discoveries),
        "| median W of the top-", K_obs, "median",
        if (K_obs > 0) stats::median(disc_other$median_W_top_K, na.rm = TRUE) else NA_real_,
        "against the observed labelling's",
        disc_tab$median_W_top_K[disc_tab$is_observed], "\n")
    if (K_obs == 0) {
      cat("[audit]     the observed labelling makes no discovery at q <=",
          opt$alpha, ", so it has no discovery set and the median W of the",
          " top-K set is undefined for every labelling\n")
    }
  }
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
#
# THE CONTROL IS REPLICATED, so the statistic has an EMPIRICAL
# DISTRIBUTION. One null cohort gives one number, and a band built from
# one number is a number the author chose. The control is therefore run
# on several INDEPENDENT null cohorts: their mean is the BASELINE the
# real cohort's value is read against, and their spread is the ACCEPTANCE
# BAND. Neither of the two is ever taken from the model under test.
#
# THE NULL COHORT HAS THE REAL COHORT'S STRUCTURE. A cohort that is
# complete, homoscedastic and iid normal is not a null for a cohort that
# is none of those, and a control measured on it answers a question about
# the complete cohort. The simulated null copies:
#
#   the SCALE, per transcript: the real cohort's pooled within-group
#   standard deviation, one real transcript drawn per simulated
#   transcript, so the control carries the real cohort's BETWEEN-TRANSCRIPT
#   dispersion instead of one common scale.
#   the MISSINGNESS: the real cohort's observed cell-missing rate, so the
#   per-row degrees of freedom vary the way they do on the real data.
#   the WEIGHTS: the real cohort's own weight rows, one real transcript
#   per simulated transcript, so the weighted models meet the weight
#   dispersion they meet on the real cohort and a transcript's scatter
#   comes with that transcript's own assumed precision.
#
# WHAT IT IS NOT. The simulated cells are normal and independent within a
# transcript, so the control carries no condition effect at all -- that is
# what a null cohort is for -- and no biological scatter beyond the
# per-transcript scale. The missingness is drawn at random rather than
# copied cell by cell, so WHICH sample is missing is not the real cohort's
# pattern. And the control's cohort is not the real cohort: its
# transcripts are drawn from the real cohort's pool, and its number of
# transcripts is not the real cohort's.

# The number of transcripts in the control. The control's statistic is a
# FRACTION over transcripts, so its Monte-Carlo error goes like
# sqrt(f * (1 - f) / n): at 1000 transcripts and a fraction near 0.8 that
# is 0.013, against the permutation p's granularity floor of 1/(1+N),
# which is 0.048 on the three-versus-three cohort and 0.004 on the
# five-versus-five one. The control is run at 1000 transcripts: enough for
# the fraction's sampling error to be small against the bands it is used
# to build, and small enough to keep the control's cost -- a permutation
# enumeration per model per replicate -- bounded.
CONTROL_TX <- 1000

# One null cohort for the positive control: the real cohort's structure,
# with no condition effect in it.
control_null_cohort <- function(n_case, n_ctrl, n_tx, seed) {
  set.seed(seed)
  n_s <- n_case + n_ctrl
  # The real transcripts the simulated ones copy: the ones with a pooled
  # within-group variance, one drawn per simulated transcript.
  pool <- which(usable)
  j <- sample(pool, n_tx, replace = TRUE)
  # The scale: the real cohort's pooled within-group standard deviation,
  # so the simulated transcript carries the real cohort's
  # between-transcript dispersion instead of one common scale.
  sd_sim <- sqrt(pooled_var[j])
  # The cells: normal, around the real transcript's own mean, with that
  # transcript's scale.
  y <- matrix(rnorm(n_tx * n_s), nrow = n_tx) * sd_sim + grand_mean[j]
  # The missingness: the real cohort's observed cell-missing rate, drawn
  # per transcript and per group, and never taking a group below two
  # observed cells, because a group of one has no within-group variance
  # and the pooled two-sample t is not defined on it.
  miss_rate <- mean(!present)
  mask <- matrix(FALSE, nrow = n_tx, ncol = n_s)
  for (g in list(which(case_ind), which(!case_ind))) {
    k <- pmin(rbinom(n_tx, length(g), miss_rate), max(0L, length(g) - 2L))
    for (i in seq_len(n_tx)) {
      if (k[i] > 0) mask[i, sample(g, k[i])] <- TRUE
    }
  }
  y[mask] <- NA_real_
  # The weights: the real transcript's own weight row. Where the real
  # transcript has no observation its weight is zero, and a zero weight on
  # an OBSERVED cell would drop that cell from the fit, so the value is
  # resampled from that transcript's own observed weights.
  copy_row <- function(M, i, obs) {
    v <- M[i, ]
    bad <- !is.finite(v) | v <= 0
    if (any(bad & obs)) {
      donor <- M[i, !bad]
      if (length(donor) == 0) donor <- 1
      v[bad & obs] <- sample(donor, sum(bad & obs), replace = TRUE)
    }
    v[!obs] <- 0
    v
  }
  obs <- !mask
  # vapply stacks one CALL RESULT PER COLUMN, so the rows are transposed
  # back into place: one simulated transcript per row.
  w_pv <- t(vapply(seq_len(n_tx),
                   function(i) copy_row(mat_pv$w, j[i], obs[i, ]),
                   numeric(n_s)))
  w_sum <- t(vapply(seq_len(n_tx),
                    function(i) copy_row(Wsum, j[i], obs[i, ]),
                    numeric(n_s)))
  list(y = y, w_pv = w_pv, w_sum = w_sum,
       missing_rate = mean(mask), drawn_from = j)
}

# The positive control, replicated: one independent null cohort per
# replicate, one row per (replicate, model).
perm_self_test <- function(n_case, n_ctrl, n_tx = CONTROL_TX,
                           replicates = opt$`control-replicates`,
                           seed = opt$seed) {
  out <- list()
  for (rep in seq_len(replicates)) {
    # An independent null cohort per replicate. The seeds are derived
    # from the run's seed and are all different, so the cohorts are
    # independent draws and not the same cohort twice.
    coh <- control_null_cohort(n_case, n_ctrl, n_tx, seed + 700000L + rep)
    for (nm in names(model_specs)) {
      sp <- model_specs[[nm]]
      ww <- switch(sp$weight, none = NULL, sum = coh$w_sum, pv = coh$w_pv)
      # The control's design is the real cohort's plain design, without a
      # batch term: the simulated cohort carries no batch structure, so a
      # batch term would be fitted on nothing.
      pr <- perm_stats(nm, y = coh$y, weights = ww, design = dl$design,
                       cmb = perm_index$case)
      an <- pr$p_analytic
      ok <- !is.na(pr$p) & !is.na(an)
      out[[length(out) + 1L]] <- data.frame(
        model_name = nm,
        replicate = rep,
        mean_analytic_p = mean(an, na.rm = TRUE),
        sd_analytic_p = stats::sd(an, na.rm = TRUE),
        mean_permutation_p = mean(pr$p, na.rm = TRUE),
        frac_analytic_p_below_permutation_p = mean(an[ok] < pr$p[ok], na.rm = TRUE),
        min_permutation_p = min(pr$p),
        n_transcripts = sum(ok),
        n_permutations = n_perm,
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, out)
}
cat("[audit] permutation positive control on", opt$`control-replicates`,
    "simulated null cohorts with the real cohort's structure\n")
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
# twenty thousand transcripts cost minutes and bought nothing. The
# quantities it reads -- present, case_mask, pooled_var, grand_mean and
# usable -- are the real cohort's structure, computed once above.

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
# Two criteria, and both tolerances are MEASURED, not chosen. Neither of
# them takes its yardstick from the model under test: a criterion whose
# band contains the model's own measurement is a criterion the model
# cannot fail, and a verdict that cannot fail is not a verdict.
#
# 1. The permutation arm. The analytic p-value claims to be the probability
#    that the statistic is as extreme as it is under the null; the exact
#    permutation p-value IS that probability, enumerated. The fraction of
#    transcripts where the analytic p falls below the permutation p is the
#    statistic, and it is NOT compared against one half: the analytic p is
#    CONTINUOUS and the permutation p is DISCRETE (a multiple of 1/(1+N)),
#    so under a true null the fraction is not one half. The baseline is
#    the POSITIVE CONTROL's measured value FOR THE SAME MODEL -- the null
#    cohort of the same structure, on which the same statistic is
#    computed -- and the acceptance band is the control's own spread
#    across its replicates, with the permutation p's granularity
#    1/(1+N) as a floor on that band, because a departure smaller than
#    the granularity of the permutation p cannot be resolved.
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
# 3. The uniformity of the p-value, read on the positive control. On the
#    null cohort the mean p-value has to be 0.5, and the band is the
#    Monte-Carlo error of that mean over the control's transcripts, WIDENED
#    by the control's own spread of the mean across its replicates. The
#    band does NOT contain the control's own departure from one half: that
#    term made the criterion true by construction for the model that owns
#    it.
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
cat("[audit] the permutation p's granularity floor on the acceptance band:",
    perm_granularity, "\n")
cat("[audit] the reference model's departure from one half is", perm_ref_dev,
    "(reported; it is NOT the yardstick any more)\n")

# The positive control's per-model summary: the measured baseline and the
# measured spread, and nothing else. The baseline is the control's mean
# over its replicates and the spread is the control's own standard
# deviation over the same replicates, so the band is the control's
# EMPIRICAL spread rather than a number the author chose.
control_summary <- function(model) {
  r <- perm_control[perm_control$model_name == model, ]
  if (nrow(r) == 0) {
    return(list(n = 0L, frac_mean = NA_real_, frac_sd = NA_real_,
                p_mean = NA_real_, p_sd = NA_real_,
                sampling_band = NA_real_, replicate_band = NA_real_,
                band = NA_real_))
  }
  frac_mean <- mean(r$frac_analytic_p_below_permutation_p, na.rm = TRUE)
  frac_sd <- stats::sd(r$frac_analytic_p_below_permutation_p, na.rm = TRUE)
  p_mean <- mean(r$mean_analytic_p, na.rm = TRUE)
  p_sd <- stats::sd(r$mean_analytic_p, na.rm = TRUE)
  n_tx_ctrl <- mean(r$n_transcripts, na.rm = TRUE)
  sd_p <- mean(r$sd_analytic_p, na.rm = TRUE)
  sampling <- 2 * sd_p / sqrt(n_tx_ctrl)
  replicate_spread <- 2 * p_sd
  list(n = nrow(r), frac_mean = frac_mean, frac_sd = frac_sd,
       p_mean = p_mean, p_sd = p_sd,
       sampling_band = sampling, replicate_band = replicate_spread,
       band = sampling + replicate_spread)
}

# What the permutation arm on THIS cohort is allowed to mean. The arm
# exchanges the case/control LABELS of the samples, so it is a
# randomization test of the GROUPING, and it is a treatment-effect
# randomization test only when the treatment was assigned at random. On a
# cohort whose batch is completely confounded with the condition the
# contrast is a batch-and-condition difference, and no permutation result
# on that cohort is a treatment-effect randomization test at all.
perm_interpretation <- if (isTRUE(dl_batch$confounded)) {
  paste0("the batch is completely confounded with the condition, so this",
         " contrast is a batch-and-condition difference and NO permutation",
         " result on this cohort is a treatment-effect randomization test;",
         " the arm measures a grouping difference only")
} else {
  paste0("the arm exchanges the case/control LABELS of the samples, so it",
         " is a randomization test of the GROUPING; it is a treatment-effect",
         " randomization test only when the treatment was assigned at",
         " random, and on this cohort no permutation result is one")
}
cat("[audit] permutation arm on this cohort:", perm_interpretation, "\n")

# The discovery-count instrument, read once for the summary.
disc_observed <- if (is.null(disc_tab)) NA_integer_ else
  sum(disc_tab$n_discoveries[disc_tab$is_observed])
disc_complement <- if (is.null(disc_tab)) NA_integer_ else
  sum(disc_tab$n_discoveries[disc_tab$is_complement_of_observed])
disc_other <- if (is.null(disc_tab)) NULL else
  disc_tab[!disc_tab$is_observed & !disc_tab$is_complement_of_observed, ]
disc_null_median <- if (is.null(disc_other) || nrow(disc_other) == 0) NA_real_ else
  stats::median(disc_other$n_discoveries)
disc_null_min <- if (is.null(disc_other) || nrow(disc_other) == 0) NA_real_ else
  min(disc_other$n_discoveries)
disc_null_max <- if (is.null(disc_other) || nrow(disc_other) == 0) NA_real_ else
  max(disc_other$n_discoveries)
disc_W_observed <- if (is.null(disc_tab)) NA_real_ else
  disc_tab$median_W_top_K[disc_tab$is_observed]
disc_W_null_median <- if (is.null(disc_other) || nrow(disc_other) == 0) NA_real_ else
  stats::median(disc_other$median_W_top_K, na.rm = TRUE)
n_distinct_labellings <- if (is.null(disc_tab)) 0L else nrow(disc_tab)
n_obs_complement <- if (is.null(disc_tab)) 0L else
  sum(disc_tab$is_observed | disc_tab$is_complement_of_observed)

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
  cs <- control_summary(nm)
  # THE UNIFORMITY CRITERION. On the null cohort the analytic p-value has
  # to be uniform, so its mean has to be one half. The band is the
  # Monte-Carlo error of that mean over the control's transcripts, plus
  # the control's own spread of the mean across its replicates. It does
  # NOT contain the control's own departure from one half: that term made
  # the criterion true by construction for the model that owns it.
  ctrl_dev <- if (is.na(cs$p_mean)) NA_real_ else abs(cs$p_mean - 0.5)
  ctrl_band <- cs$band
  ctrl_pass <- !is.na(ctrl_dev) && !is.na(ctrl_band) && ctrl_dev <= ctrl_band
  # THE PERMUTATION CRITERION. The real cohort's statistic is read against
  # the CONTROL's measured baseline for the SAME model, and the band is
  # the control's own spread across its replicates, floored by the
  # granularity of the permutation p (1/(1+N)): a departure smaller than
  # that granularity cannot be resolved.
  prow <- perm_tab[perm_tab$model_name == nm, ]
  perm_value <- if (nrow(prow) == 0) NA_real_ else
    prow$frac_analytic_p_below_permutation_p
  perm_dev <- if (is.na(perm_value) || is.na(cs$frac_mean)) NA_real_ else
    abs(perm_value - cs$frac_mean)
  perm_spread <- if (is.finite(cs$frac_sd)) 2 * cs$frac_sd else 0
  perm_band <- max(perm_spread, perm_granularity)
  perm_pass <- !is.na(perm_dev) && perm_dev <= perm_band
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
    control_mean_analytic_p = cs$p_mean,
    control_uniformity_deviation = ctrl_dev,
    control_uniformity_band = ctrl_band,
    control_frac_analytic_below_perm = cs$frac_mean,
    null_cohorts = st$n,
    null_cohorts_with_a_rejection = st$n_any,
    null_p_any_rejection = st$p_any,
    null_excess_binomial_p = st$excess_p,
    null_p_any_rejection_model_assumed = st_assumed$p_any,
    null_excess_binomial_p_model_assumed = st_assumed$excess_p,
    perm_frac_analytic_below_perm = perm_value,
    perm_deviation_from_one_half = if (is.na(perm_value)) NA_real_ else abs(perm_value - 0.5),
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
    # --- the evidence the two criteria above are built on ---
    control_replicates = cs$n,
    control_uniformity_band_sampling = cs$sampling_band,
    control_uniformity_band_replicate_spread = cs$replicate_band,
    control_mean_analytic_p_sd_across_replicates = cs$p_sd,
    perm_control_baseline = cs$frac_mean,
    perm_control_baseline_sd_across_replicates = cs$frac_sd,
    perm_control_band_spread = perm_spread,
    perm_granularity_floor = perm_granularity,
    perm_control_band = perm_band,
    perm_deviation_from_control_baseline = perm_dev,
    perm_criterion_pass = perm_pass,
    uniformity_criterion_pass = ctrl_pass,
    # --- the discovery-count instrument ---
    ref_discovery_model = DISC_MODEL,
    ref_discovery_count_observed = disc_observed,
    ref_discovery_count_complement = disc_complement,
    ref_discovery_count_null_median = disc_null_median,
    ref_discovery_count_null_min = disc_null_min,
    ref_discovery_count_null_max = disc_null_max,
    ref_median_W_observed = disc_W_observed,
    ref_median_W_null_median = disc_W_null_median,
    n_distinct_labellings = n_distinct_labellings,
    n_labellings_observed_or_complement = n_obs_complement,
    permutation_interpretation = perm_interpretation,
    stringsAsFactors = FALSE
  )
}
summ_tab <- do.call(rbind, summ)
write.table(summ_tab, file = file.path(opt$`output-dir`, "dpl_audit_summary.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat("\n[audit] summary\n")
print(summ_tab)
cat("\n[audit] wrote dpl_units.tsv, dpl_model_fit.tsv, dpl_permutation_null.tsv,",
    "dpl_permutation_discovery_counts.tsv, dpl_permutation_control.tsv,",
    "dpl_null_calibration.tsv, dpl_injection_power.tsv, dpl_audit_summary.tsv to",
    opt$`output-dir`, "\n")
