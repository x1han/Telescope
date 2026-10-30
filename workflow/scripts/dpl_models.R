# dpl_models.R -- the differential poly(A) LENGTH model.
#
# THE UNIT. One row per (transcript x biological sample). The reads of a
# sample are NOT replicates: they are averaged into one measurement of that
# transcript's tail length in that sample, together with the variance of that
# measurement. A model that fits per-read rows inflates the effective n by
# ~10^6 and its p-values mean nothing.
#
# THE FOUR MOMENTS THE UNIT IS BUILT FROM (all produced by the Rust
# aggregator, one row per transcript x sample):
#
#   W    = sum(w)                  posterior mass: the effective read count
#   swy  = sum(w * y)              y = log2(max(polyA_length, 1))
#   swy2 = sum(w * y^2)
#   sw2  = sum(w^2)
#
# FROM THEM:
#
#   mean_log2_pal      = swy / W
#   Kish_Neff          = W^2 / sw2        (Kish 1965 effective sample size)
#   within_variance    = (swy2 - swy^2/W) / (Kish_Neff - 1)
#   measurement_variance = within_variance / Kish_Neff
#
# The within_variance denominator is Kish_Neff - 1 rather than W - 1 on
# purpose: with the soft posterior a cell holds one row per (read, candidate
# transcript) pair, so the row count is not the sample size, but Kish_Neff is.
# For equal weights the two conventions coincide.
#
# THE WEIGHT. A precision-weighted fit needs a VARIANCE, not a count. The
# shipped engine weighted each sample by its posterior mass (a read count);
# that is a proxy for precision but it is not one, and it is not comparable
# across transcripts. Here the weight is 1 / measurement_variance.
#
# A CAVEAT THE AUDIT MEASURES RATHER THAN ASSUMES: the measurement variance
# is the WITHIN-sample part. The between-sample biological scatter of a
# transcript's tail length is not in it, so a fit that uses only
# 1/measurement_variance is misspecified if the biological scatter dominates.
# Whether that makes the model anti-conservative is a question for
# audit_dpl_models.R, and the answer decides whether this model ships.

suppressPackageStartupMessages({
  library(limma)
})

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

#' Build the (transcript x sample) analysis table.
#'
#' @param agg data.frame with sample_id, gene_id, transcript_id, weight_sum,
#'   weighted_log2pal_sum, weight_sq_sum, weighted_log2pal_sq_sum.
#' @param design data.frame with sample_id and condition.
#' @param min_weight minimum posterior mass for a cell.
#' @param min_neff minimum Kish effective sample size for a cell.
#' @return data.frame with sample_id, transcript_id, gene_id, condition,
#'   mean_log2_pal, within_variance, W, Kish_Neff, measurement_variance,
#'   passed_quality_gate.
dpl_units <- function(agg, design, min_weight = 10, min_neff = 2) {
  stopifnot(all(c("sample_id", "gene_id", "transcript_id", "weight_sum",
                  "weighted_log2pal_sum", "weight_sq_sum",
                  "weighted_log2pal_sq_sum") %in% names(agg)))
  d <- data.frame(
    sample_id           = as.character(agg$sample_id),
    transcript_id       = as.character(agg$transcript_id),
    gene_id             = as.character(agg$gene_id),
    W                  = as.numeric(agg$weight_sum),
    swy                = as.numeric(agg$weighted_log2pal_sum),
    swy2               = as.numeric(agg$weighted_log2pal_sq_sum),
    sw2                = as.numeric(agg$weight_sq_sum),
    stringsAsFactors = FALSE
  )
  des <- data.frame(sample_id = as.character(design$sample_id),
                    condition  = as.character(design$condition),
                    stringsAsFactors = FALSE)
  d <- merge(d, des, by = "sample_id", all.x = TRUE)

  d$mean_log2_pal <- ifelse(d$W > 0, d$swy / d$W, NA_real_)
  d$Kish_Neff     <- ifelse(d$sw2 > 0, d$W^2 / d$sw2, NA_real_)
  # Weighted scatter around the weighted mean. Clamped at zero: the
  # subtraction is catastrophic cancellation when every read has the same
  # tail length, and the true value is then zero, not a small negative.
  ss_w <- d$swy2 - d$swy^2 / d$W
  ss_w[!is.finite(ss_w)] <- 0
  ss_w[ss_w < 0] <- 0
  d$within_variance <- ifelse(is.na(d$Kish_Neff) | d$Kish_Neff <= 1,
                              NA_real_, ss_w / (d$Kish_Neff - 1))
  d$measurement_variance <- d$within_variance / d$Kish_Neff

  d$passed_quality_gate <- !is.na(d$measurement_variance) &
                            is.finite(d$measurement_variance) &
                            d$measurement_variance > 0 &
                            d$W >= min_weight &
                            d$Kish_Neff >= min_neff
  d <- d[, c("sample_id", "transcript_id", "gene_id", "condition",
             "mean_log2_pal", "within_variance", "W", "Kish_Neff",
             "measurement_variance", "passed_quality_gate")]
  d <- d[order(d$transcript_id, d$sample_id), ]
  rownames(d) <- NULL
  d
}

# ---------------------------------------------------------------------------
# Model matrices
# ---------------------------------------------------------------------------

#' Build the design matrix for an explicit case / control contrast.
#'
#' The reference level is the control, always. The effect coefficient is
#' therefore (case - control) and its sign does not depend on how the
#' condition labels happen to sort.
#'
#' @param samples character vector of sample ids, in column order.
#' @param condition character vector, same length as samples.
#' @param case_condition,control_condition the two labels.
#' @param batch optional character vector of batch labels.
#' @return list(design, contrast, batch_used, batch_note)
dpl_design <- function(samples, condition, case_condition, control_condition,
                       batch = NULL) {
  known <- unique(condition)
  missing <- setdiff(c(case_condition, control_condition), known)
  if (length(missing) > 0) {
    stop("ERROR: condition label(s) not present in the design: ",
         paste(missing, collapse = ", "),
         ". Present labels: ", paste(sort(known), collapse = ", "))
  }
  if (identical(case_condition, control_condition)) {
    stop("ERROR: --case-condition and --control-condition are the same label: ",
         case_condition)
  }
  cond <- factor(condition, levels = c(control_condition, case_condition))
  design <- stats::model.matrix(~ cond)
  colnames(design) <- c("intercept", "case")
  batch_used <- FALSE
  batch_note <- "no batch column in the design"
  if (!is.null(batch)) {
    b <- factor(batch)
    if (nlevels(b) >= 2) {
      d2 <- stats::model.matrix(~ cond + b)
      colnames(d2) <- c("intercept", "case", paste0("batch", levels(b)[-1]))
      if (qr(d2)$rank == ncol(d2)) {
        design <- d2
        batch_used <- TRUE
        batch_note <- paste("batch adjusted:", paste(levels(b), collapse = ","))
      } else {
        batch_note <- paste0(
          "batch present (", paste(levels(b), collapse = ","),
          ") but confounded with condition; NOT adjusted")
      }
    } else {
      batch_note <- "batch column has fewer than two levels; NOT adjusted"
    }
  }
  list(design = design, contrast = c(control_condition, case_condition),
       batch_used = batch_used, batch_note = batch_note)
}

# ---------------------------------------------------------------------------
# Rows that carry no observation
# ---------------------------------------------------------------------------
#
# A row with no observation at all has no mean, and limma's empirical Bayes
# with trend = TRUE uses the mean as its trend covariate: limma's lmFit sets
# Amean <- rowMeans(y, na.rm = TRUE), so an empty row's Amean is NaN. eBayes
# then refuses the WHOLE fit -- "covariate contains NA or infinite values"
# with robust = TRUE, "NA covariate values not allowed" without it. One empty
# transcript would take down the transcriptome, and the shipped real-cohort
# tables have thousands of them: a transcript whose every cell failed the
# quality gate. So the empty rows are left out of the fit and put back
# afterwards, as NA.
dpl_observed_rows <- function(y, weights = NULL) {
  obs <- rowSums(!is.na(y)) > 0
  if (!is.null(weights)) {
    w <- weights
    w[is.na(w)] <- 0
    obs <- obs & (rowSums(w > 0) > 0)
  }
  obs
}

#' Put a per-row quantity back onto the full set of rows.
#'
#' limma's df.residual is PER ROW, not one number for the fit: a row whose
#' cells are partly missing has fewer observations and therefore fewer
#' residual degrees of freedom. So it is expanded like every other per-row
#' quantity.
#'
#' @param v the value for the rows that were fitted.
#' @param keep the indices of the rows that were fitted.
#' @param n the number of rows of the response matrix.
#' @return a numeric vector of length n, NA for the rows that were not fitted.
dpl_expand <- function(v, keep, n) {
  out <- rep(NA_real_, n)
  v <- as.numeric(v)
  if (length(v) == 1L) v <- rep(v, length(keep))
  out[keep] <- v
  out
}

#' Summarise a per-row prior instead of listing it.
#'
#' With trend = TRUE limma fits the prior variance against the average
#' expression, so s2.prior and df.prior hold ONE VALUE PER TRANSCRIPT rather
#' than one value for the fit. Printing them one by one would put thousands
#' of numbers in a log line and a paragraph in every cell of a table.
#'
#' @param v a numeric vector.
#' @return a one-line summary.
dpl_prior_summary <- function(v) {
  v <- as.numeric(v)
  finite <- v[is.finite(v)]
  if (length(finite) == 0) {
    if (length(v) > 0 && all(is.infinite(v))) {
      # limma estimates NO prior degrees of freedom when the prior variance
      # is fitted against a trend: the trend carries the whole prior.
      return("infinite (limma estimates no prior df when the trend is fitted)")
    }
    return("not estimable")
  }
  sprintf("median %.6g, range %.6g..%.6g", stats::median(finite),
          min(finite), max(finite))
}

#' limma's empirical Bayes, with the variance trend when the trend can be
#' fitted.
#'
#' `trend = TRUE` fits the prior variance against the average expression of
#' the features, using a natural spline. The spline needs the covariate to
#' have several distinct values, and when it does not -- a handful of
#' features with the same average, which is what a small cohort's poly(A)
#' table looks like -- limma stops with "Problem with covariate". The fit
#' then goes on WITHOUT the trend, and the note says so: a run that silently
#' used a different fit would be a run whose receipt is wrong.
#'
#' @param fit a limma fit.
#' @return list(fit = the moderated fit, trend = whether the trend was used,
#'   note = the note to put in the receipt, "" when nothing happened).
dpl_ebayes <- function(fit) {
  ok <- tryCatch(
    list(fit = limma::eBayes(fit, trend = TRUE, robust = TRUE),
         trend = TRUE, note = ""),
    error = function(e) {
      fallback <- tryCatch(limma::eBayes(fit, robust = TRUE),
                           error = function(e2) stop(e2))
      list(
        fit = fallback,
        trend = FALSE,
        note = paste0(
          "the variance trend could not be fitted (limma: ",
          conditionMessage(e),
          "), so the empirical Bayes moderation was fitted WITHOUT the trend ",
          "(robust = TRUE only)")
      )
    })
  ok
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# The candidate models. `equal_raw_t` is the unweighted two-sample t on the
# per-sample means; `weight_sum_lm` is what the shipped engine does today
# (weighting each sample by its posterior mass, i.e. by a read count); the
# precision-weighted pair replaces that count with 1/measurement_variance.
DPL_MODEL_NAMES <- c(
  "equal_raw_t",
  "weight_sum_lm",
  "precision_weighted_lm",
  "empirical_bayes_weighted_lm",
  "batch_adjusted_lm"
)

#' Fit one model to a (transcript x sample) matrix.
#'
#' @param y matrix, transcripts x samples, NA where the cell is missing.
#' @param design design matrix with an intercept and a `case` column.
#' @param weights matrix of the same shape, 0 where there is no observation.
#' @param ebayes apply limma's empirical Bayes moderation.
#' @return a list with coefficient, se, df, t, p and (for ebayes) the prior.
dpl_fit_model <- function(y, design, weights = NULL, ebayes = FALSE) {
  n <- nrow(y)
  observed <- dpl_observed_rows(y, weights)
  keep <- which(observed)
  if (length(keep) == 0) {
    stop("ERROR: no row of the response matrix carries an observation.")
  }
  # THE WEIGHTS ARE SUBSETTED WITH THE RESPONSE. lmFit checks that the weight
  # matrix has one row per row of the response it is given, so handing it the
  # FULL weights with a SUBSET response is a dimension mismatch that aborts
  # with "x is not a matrix of the right dimensions". It is latent today
  # because every row that reaches this function carries an observation, and
  # it would fire the first time a row does not.
  w <- weights
  if (!is.null(w)) w <- w[keep, , drop = FALSE]
  fit <- limma::lmFit(y[keep, , drop = FALSE], design, weights = w)
  eb_note <- ""
  eb_trend <- FALSE
  if (ebayes) {
    # trend = TRUE fits the prior variance as a function of the average
    # expression, and robust = TRUE shrinks the influence of the transcripts
    # whose variance is extreme (Phipson et al. 2016). Without them a handful
    # of outliers sets the prior for every transcript.
    eb <- dpl_ebayes(fit)
    fit <- eb$fit
    eb_note <- eb$note
    eb_trend <- eb$trend
    # s2.post is the posterior VARIANCE, so the moderated standard error is
    # stdev.unscaled * sqrt(s2.post). That is exactly the divisor limma's
    # own moderated t uses.
    se <- fit$stdev.unscaled[, "case"] * sqrt(fit$s2.post)
    df_raw <- fit$df.residual
    out <- list(
      estimate   = dpl_expand(fit$coefficients[, "case"], keep, n),
      se         = dpl_expand(se, keep, n),
      df_raw     = dpl_expand(df_raw, keep, n),
      t          = dpl_expand(fit$t[, "case"], keep, n),
      p          = dpl_expand(fit$p.value[, "case"], keep, n),
      df_post    = dpl_expand(fit$df.total, keep, n),
      sigma2_raw = dpl_expand(fit$sigma^2, keep, n),
      sigma2_post = dpl_expand(fit$s2.post, keep, n),
      df_prior   = dpl_expand(fit$df.prior, keep, n),
      s2_prior   = dpl_expand(fit$s2.prior, keep, n)
    )
  } else {
    # fit$sigma is the residual STANDARD DEVIATION, so the standard error is
    # stdev.unscaled * sigma. sqrt(sigma) would be wrong: it treats a
    # standard deviation as a variance and inflates the standard error.
    # With equal weights this reproduces the ordinary two-sample t exactly.
    se <- fit$stdev.unscaled[, "case"] * fit$sigma
    out_t <- fit$coefficients[, "case"] / se
    out <- list(
      estimate   = dpl_expand(fit$coefficients[, "case"], keep, n),
      se         = dpl_expand(se, keep, n),
      df_raw     = dpl_expand(fit$df.residual, keep, n),
      t          = dpl_expand(out_t, keep, n),
      p          = dpl_expand(2 * stats::pt(-abs(out_t), df = fit$df.residual), keep, n),
      df_post    = dpl_expand(fit$df.residual, keep, n),
      sigma2_raw = dpl_expand(fit$sigma^2, keep, n),
      sigma2_post = rep(NA_real_, n),
      df_prior   = rep(NA_real_, n),
      s2_prior   = rep(NA_real_, n)
    )
  }
  out$observed <- observed
  out$trend_used <- eb_trend
  out$ebayes_note <- eb_note
  out
}

#' Fit one model and take BOTH the raw and the moderated statistics from it.
#'
#' `eBayes` leaves the coefficients, the unscaled standard deviations, the
#' residual standard deviation and the residual degrees of freedom untouched:
#' it adds the posterior variance and the moderated t and p on top. So one
#' `lmFit` serves both arms, and the whole-transcriptome fit is not paid for
#' twice.
#'
#' @param y matrix, transcripts x samples, NA where the cell is missing.
#' @param design design matrix with an intercept and a `case` column.
#' @param weights matrix of the same shape, 0 where there is no observation.
#' @return a list with a `raw` and a `moderated` element, each as
#'   `dpl_fit_model` returns them.
dpl_fit_both <- function(y, design, weights = NULL) {
  n <- nrow(y)
  observed <- dpl_observed_rows(y, weights)
  keep <- which(observed)
  if (length(keep) == 0) {
    stop("ERROR: no row of the response matrix carries an observation.")
  }
  # The weights are subsetted with the response: lmFit requires one weight row
  # per row of the response it is given.
  w <- weights
  if (!is.null(w)) w <- w[keep, , drop = FALSE]
  fit <- limma::lmFit(y[keep, , drop = FALSE], design, weights = w)
  ebx <- dpl_ebayes(fit)
  eb <- ebx$fit
  # fit$sigma is the residual STANDARD DEVIATION, so the standard error is
  # stdev.unscaled * sigma; sqrt(sigma) would treat a standard deviation as
  # a variance and inflate the standard error.
  se_raw <- eb$stdev.unscaled[, "case"] * eb$sigma
  t_raw <- eb$coefficients[, "case"] / se_raw
  # s2.post is the posterior VARIANCE, so the moderated standard error is
  # stdev.unscaled * sqrt(s2.post).
  se_mod <- eb$stdev.unscaled[, "case"] * sqrt(eb$s2.post)
  list(
    raw = list(
      estimate   = dpl_expand(eb$coefficients[, "case"], keep, n),
      se         = dpl_expand(se_raw, keep, n),
      df_raw     = dpl_expand(eb$df.residual, keep, n),
      t          = dpl_expand(t_raw, keep, n),
      p          = dpl_expand(2 * stats::pt(-abs(t_raw), df = eb$df.residual), keep, n),
      df_post    = dpl_expand(eb$df.residual, keep, n),
      sigma2_raw = dpl_expand(eb$sigma^2, keep, n),
      sigma2_post = rep(NA_real_, n),
      df_prior   = rep(NA_real_, n),
      s2_prior   = rep(NA_real_, n)
    ),
    moderated = list(
      estimate   = dpl_expand(eb$coefficients[, "case"], keep, n),
      se         = dpl_expand(se_mod, keep, n),
      df_raw     = dpl_expand(eb$df.residual, keep, n),
      t          = dpl_expand(eb$t[, "case"], keep, n),
      p          = dpl_expand(eb$p.value[, "case"], keep, n),
      df_post    = dpl_expand(eb$df.total, keep, n),
      sigma2_raw = dpl_expand(eb$sigma^2, keep, n),
      sigma2_post = dpl_expand(eb$s2.post, keep, n),
      df_prior   = dpl_expand(eb$df.prior, keep, n),
      s2_prior   = dpl_expand(eb$s2.prior, keep, n)
    ),
    observed = observed,
    trend_used = ebx$trend,
    ebayes_note = ebx$note
  )
}

#' Build the wide response and weight matrices for a set of transcripts.
#'
#' @param units the unit table.
#' @param samples character vector, the column order.
#' @param transcripts character vector, the row order.
#' @param value which column of `units` becomes the response.
#' @param weight which column of `units` becomes the precision weight
#'   (NULL means equal weights).
#' @return list(y, w) matrices, NA / 0 where the cell is missing.
dpl_matrices <- function(units, samples, transcripts, value = "mean_log2_pal",
                         weight = NULL) {
  u <- units[units$passed_quality_gate, ]
  y <- matrix(NA_real_, nrow = length(transcripts), ncol = length(samples),
              dimnames = list(transcripts, samples))
  w <- matrix(0, nrow = length(transcripts), ncol = length(samples),
              dimnames = list(transcripts, samples))
  si <- match(u$sample_id, samples)
  ti <- match(u$transcript_id, transcripts)
  ok <- !is.na(si) & !is.na(ti)
  y[cbind(ti[ok], si[ok])] <- u[[value]][ok]
  if (is.null(weight)) {
    w[cbind(ti[ok], si[ok])] <- 1
  } else {
    w[cbind(ti[ok], si[ok])] <- 1 / u[[weight]][ok]
  }
  list(y = y, w = w)
}
