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
  fit <- limma::lmFit(y, design, weights = weights)
  if (ebayes) {
    fit <- limma::eBayes(fit)
    # s2.post is the posterior VARIANCE, so the moderated standard error is
    # stdev.unscaled * sqrt(s2.post). That is exactly the divisor limma's
    # own moderated t uses.
    se <- fit$stdev.unscaled[, "case"] * sqrt(fit$s2.post)
    df_raw <- fit$df.residual
    out <- list(
      estimate   = fit$coefficients[, "case"],
      se         = se,
      df_raw     = df_raw,
      t          = fit$t[, "case"],
      p          = fit$p.value[, "case"],
      df_post    = fit$df.total,
      sigma2_raw = fit$sigma^2,
      sigma2_post = fit$s2.post,
      df_prior   = fit$df.prior,
      s2_prior   = fit$s2.prior
    )
  } else {
    # fit$sigma is the residual STANDARD DEVIATION, so the standard error is
    # stdev.unscaled * sigma. sqrt(sigma) would be wrong: it treats a
    # standard deviation as a variance and inflates the standard error.
    # With equal weights this reproduces the ordinary two-sample t exactly.
    se <- fit$stdev.unscaled[, "case"] * fit$sigma
    out <- list(
      estimate   = fit$coefficients[, "case"],
      se         = se,
      df_raw     = fit$df.residual,
      t          = out_t <- fit$coefficients[, "case"] / se,
      p          = 2 * stats::pt(-abs(out_t), df = fit$df.residual),
      df_post    = fit$df.residual + 0,
      sigma2_raw = fit$sigma^2,
      sigma2_post = NA_real_,
      df_prior   = NA_real_,
      s2_prior   = NA_real_
    )
  }
  out
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
