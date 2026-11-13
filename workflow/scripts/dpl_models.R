                                                        
 
                                                                        
                                                                           
                                                                             
                                                                          
                                      
 
                                                                   
                                               
 
                                                                           
                                                                 
                       
                   
 
            
 
                                
                                                                           
                                                       
                                                      
 
                                                                       
                                                                           
                                                                            
                                                                           
           
 
                                                                         
                                                                          
                                                                         
                                                      
 
                                                                           
                                                                       
                                                                
                                                                             
                                                                  
                                                                      

suppressPackageStartupMessages({
  library(limma)
})

                                                                             
       
                                                                             

                                                  
  
                                                                            
                                                                  
                                                         
                                                       
                                                                 
                                                                        
                                                                        
                         
dpl_units <- function(agg, design, min_weight = 10, min_neff = 10) {
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
                                                                   
                                                                         
                                                                       
  ss_w <- d$swy2 - d$swy^2 / d$W
  ss_w[!is.finite(ss_w)] <- 0
  ss_w[ss_w < 0] <- 0
                                           
   
                                                                            
                                                                        
                                                                    
                                                                           
                                                                        
                                                                            
                                       
   
                                                              
                                                                             
                                                                     
                                                                         
   
                                                                         
                                                                            
                                                                       
                                                                           
                                               
  den_w <- d$W - d$sw2 / d$W
  d$within_variance <- ifelse(is.na(d$Kish_Neff) | !is.finite(den_w) | den_w <= 0,
                              NA_real_, ss_w / den_w)
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
  confounded <- FALSE
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
                                                                     
                                                             
                                                                            
                                                                        
                                                                             
                                                                           
                         
        cc <- as.character(condition)
        bb <- as.character(batch)
        ok <- !is.na(bb) & !is.na(cc)
        tb <- table(bb[ok], cc[ok])
        confounded <- all(apply(tb, 1L, function(r) sum(r > 0) == 1L))
        batch_note <- paste0(
          "batch present (", paste(levels(b), collapse = ","),
          ") but confounded with condition; NOT adjusted")
      }
    } else {
      batch_note <- "batch column has fewer than two levels; NOT adjusted"
    }
  }
  list(design = design, contrast = c(control_condition, case_condition),
       batch_used = batch_used, confounded = confounded,
       batch_note = batch_note)
}

                                                                             
                                
                                                                             
 
                                                                           
                                                                            
                                                                            
                                                                          
                                                                             
                                                                           
                                                                         
                                                                      
                    
dpl_observed_rows <- function(y, weights = NULL) {
  obs <- rowSums(!is.na(y)) > 0
  if (!is.null(weights)) {
    w <- weights
    w[is.na(w)] <- 0
    obs <- obs & (rowSums(w > 0) > 0)
  }
  obs
}

                                                         
  
                                                                          
                                                                      
                                                                          
            
  
                                                    
                                                        
                                                      
                                                                              
dpl_expand <- function(v, keep, n) {
  out <- rep(NA_real_, n)
  v <- as.numeric(v)
  if (length(v) == 1L) v <- rep(v, length(keep))
  out[keep] <- v
  out
}

                                                   
  
                                                                             
                                                                            
                                                                            
                                                     
  
                                                                              
                                                                           
                                                                             
                                                                         
                                                 
  
                                                                      
                                                                        
                                                                         
                                                                           
                                                                           
                                                                                  
                                                                           
                                                                        
                                                                            
                                                                             
                                                                           
                                       
  
                                                 
                              
dpl_prior_summary <- function(v) {
  v <- as.numeric(v)
  v <- v[!is.na(v)]
  finite <- v[is.finite(v)]
  n_total <- length(v)
  n_finite <- length(finite)
  if (n_finite == 0) {
    if (n_total > 0 && all(is.infinite(v))) {
      return("infinite (limma fitted no finite prior df for this fit, so the moderated t uses limma's pooled df)")
    }
    return("not estimable")
  }
  if (n_finite < n_total) {
                                                                             
                                                                               
    return(sprintf("median %.6g, range %.6g..%.6g over %d of %d rows; the other %d row(s) had no finite value",
                   stats::median(finite), min(finite), max(finite),
                   n_finite, n_total, n_total - n_finite))
  }
  sprintf("median %.6g, range %.6g..%.6g", stats::median(finite),
          min(finite), max(finite))
}

                                      
  
                                                                           
                                                                          
                                                                        
                                                                            
                                                                     
                                                                      
                                                                     
                                                                         
                  
  
                                                                            
                                                                          
                                                                             
                                                                           
  
                                    
                                                                           
                                                                            
                                  
dpl_trend_took_effect <- function(fit) {
  s2 <- fit$s2.prior
  if (is.null(s2)) return(NA)
  s2 <- s2[!is.na(s2)]
  if (length(s2) == 0L) return(NA)
  length(unique(s2)) > 1L
}

                                     
  
                                                                             
                                                                            
  
                                             
                                                         
                                                                          
                       
dpl_note_dropped_trend <- function(out, cov_label) {
  if (!isTRUE(out$trend)) return(out)
  if (!identical(dpl_trend_took_effect(out$fit), FALSE)) return(out)
  out$trend <- FALSE
  out$trend_kind <- "none"
  out$note <- paste0(
    "the variance trend had NO effect: limma fitted the prior variance ",
    "WITHOUT the trend covariate (", cov_label, "), because the covariate ",
    "could not carry a spline (fewer than two distinct values, or fewer ",
    "than three usable features). A trend-fitted prior differs from feature ",
    "to feature and this one is the same number for every feature, so the ",
    "model actually used is limma::eBayes(robust = TRUE) with NO variance ",
    "trend")
  out
}

                                                                         
          
  
                                                                           
                                                                          
                                                                     
                                                                         
                                                                          
                                                                         
                                                                            
                                                                            
                          
  
                                                                            
                                                                              
                                                               
                                                                           
                                                                           
                              
  
                          
                                                                         
                              
                                                                          
                                                                             
                       
dpl_ebayes <- function(fit, trend = NULL) {
                                                                      
   
                                                                    
   
                                                                             
                                                                           
                                                                      
                                                                         
                                                                            
                                                                           
                                                                            
                                                                          
                             
  if (is.null(trend)) {
    out <- tryCatch(
      list(fit = limma::eBayes(fit, trend = TRUE, robust = TRUE),
           trend = TRUE, trend_kind = "Amean", note = ""),
      error = function(e) {
        fallback <- tryCatch(limma::eBayes(fit, robust = TRUE),
                             error = function(e2) stop(e2))
        list(
          fit = fallback,
          trend = FALSE,
          trend_kind = "none",
          note = paste0(
            "the variance trend could not be fitted (limma: ",
            conditionMessage(e),
            "), so the empirical Bayes moderation was fitted WITHOUT the trend ",
            "(robust = TRUE only)")
        )
      })
    return(dpl_note_dropped_trend(out, "the average expression (Amean)"))
  }
  cov <- as.numeric(trend)
  if (length(cov) != nrow(fit$coefficients)) {
    stop("ERROR: the variance-trend covariate has ", length(cov),
         " values for a fit of ", nrow(fit$coefficients), " rows.")
  }
  if (any(!is.finite(cov))) {
    stop("ERROR: the variance-trend covariate has ", sum(!is.finite(cov)),
         " non-finite value(s). A limma trend covariate must be finite.")
  }
  out <- tryCatch(
    list(fit = limma::eBayes(fit, trend = cov, robust = TRUE),
         trend = TRUE, trend_kind = "depth_support", note = ""),
    error = function(e) {
      fallback <- tryCatch(limma::eBayes(fit, robust = TRUE),
                           error = function(e2) stop(e2))
      list(
        fit = fallback,
        trend = FALSE,
        trend_kind = "none",
        note = paste0(
          "the variance trend could not be fitted (limma: ",
          conditionMessage(e),
          "), so the empirical Bayes moderation was fitted WITHOUT the trend ",
          "(robust = TRUE only)")
      )
    })
  dpl_note_dropped_trend(out,
    "the mean over a transcript's cells of the log2 of the summed posterior weight")
}

                                                 
  
                                                                              
                                                                              
                                                                         
                                                                        
                                                                         
  
                                                                               
                                          
                                                                              
                                
dpl_depth_trend <- function(support) {
  s <- support
  s[!is.finite(s) | s <= 0] <- NA_real_
  l <- log2(s)
  out <- rowMeans(l, na.rm = TRUE)
  none <- rowSums(is.finite(l)) == 0
  out[none] <- NA_real_
  out
}

                                                                             
        
                                                                             

                                                                        
                                                                        
                                                             
 
                                                                           
                                                                      
                                                                  
                                                                       
                                                                           
                                                                
 
                                                                         
                                                                     
                                                                         
DPL_MODEL_NAME          <- "DT-EB"
DPL_MODEL_NAME_NO_TREND <- "DT-EB-no-trend"

                                                                      
                                                                   
                                                                   
                                                                           
                                                                          
                                                                              

                                                   
  
                                                                        
                                                                     
                                                                            
                                                          
                                                                             
                                   
                                                                           
                                               
                                                                            
                                                
  
                                                                           
                                                                             
                              
  
                                                                            
                                                                         
                                                                             
                                                                           
                                                       
  
                                                      
                                           
dpl_empty_fit <- function(n) {
  list(
    estimate    = rep(NA_real_, n),
    se          = rep(NA_real_, n),
    df_raw      = rep(NA_real_, n),
    t           = rep(NA_real_, n),
    p           = rep(NA_real_, n),
    df_post     = rep(NA_real_, n),
    sigma2_raw  = rep(NA_real_, n),
    sigma2_post = rep(NA_real_, n),
    df_prior    = rep(NA_real_, n),
    s2_prior    = rep(NA_real_, n),
                                                                             
                                                                              
                                                       
    n_fitted    = 0L
  )
}

dpl_fit_model <- function(y, design, weights = NULL, ebayes = FALSE,
                          trend = NULL, testable = NULL) {
  n <- nrow(y)
  observed <- dpl_observed_rows(y, weights)
  keep <- which(observed)
  if (!is.null(testable)) {
                                                             
     
                                                                            
                                                                         
                                                                            
                                                                      
                                                                            
                                                                              
                                                                              
    keep <- keep[testable[keep]]
  }
  if (length(keep) == 0) {
    if (is.null(testable)) {
                                                                          
                                                 
      stop("ERROR: no row of the response matrix carries an observation.")
    }
                                                 
    empty <- dpl_empty_fit(n)
    empty$observed <- observed
    empty$trend_used <- FALSE
    empty$trend_kind <- "none"
    empty$ebayes_note <- ""
    return(empty)
  }
                                                                             
                                                                             
                                                                           
                                                                        
                                                                            
                                                
  w <- weights
  if (!is.null(w)) w <- w[keep, , drop = FALSE]
  fit <- limma::lmFit(y[keep, , drop = FALSE], design, weights = w)
  eb_note <- ""
  eb_trend <- FALSE
  eb_trend_kind <- "none"
  if (ebayes) {
                                                                        
                                                                        
                                                                            
                                                                        
                 
    eb <- dpl_ebayes(fit, trend = if (is.null(trend)) NULL else trend[keep])
    fit <- eb$fit
    eb_note <- eb$note
    eb_trend <- eb$trend
    eb_trend_kind <- eb$trend_kind
                                                                           
                                                                         
                           
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
  out$trend_kind <- eb_trend_kind
  out$ebayes_note <- eb_note
                                                                         
                                                
  out$n_fitted <- length(keep)
  out
}

                                                                            
  
                                                                          
                                                                             
                                                                          
                                                                            
         
  
                                                                        
                                                                     
                                                                            
                                                                           
                                               
                                                                             
                                   
                                                                            
                                                                             
                                                                       
                                                                            
                                                         
                                                                 
                                                                       
                         
dpl_fit_both <- function(y, design, weights = NULL, testable = NULL,
                         trend = NULL, moderate = TRUE) {
  n <- nrow(y)
  observed <- dpl_observed_rows(y, weights)
  keep <- which(observed)
  if (!is.null(testable)) {
                                                                          
                                                              
    keep <- keep[testable[keep]]
  }
  if (length(keep) == 0) {
    if (is.null(testable)) {
                                                                          
                                                 
      stop("ERROR: no row of the response matrix carries an observation.")
    }
                                                                             
                                                                             
                                                                      
                                                                            
                                                                         
                                                                             
                                                
    empty <- dpl_empty_fit(n)
    return(list(
      raw = empty,
      moderated = if (moderate) empty else NULL,
      observed = observed,
      trend_used = FALSE,
      trend_kind = "none",
      ebayes_note = ""
    ))
  }
                                                                              
                                        
  w <- weights
  if (!is.null(w)) w <- w[keep, , drop = FALSE]
  fit <- limma::lmFit(y[keep, , drop = FALSE], design, weights = w)
  ebx <- NULL
  if (moderate) {
    ebx <- dpl_ebayes(fit, trend = if (is.null(trend)) NULL else trend[keep])
    fit <- ebx$fit
  }
                                                                          
                                                                           
                                              
  se_raw <- fit$stdev.unscaled[, "case"] * fit$sigma
  t_raw <- fit$coefficients[, "case"] / se_raw
  raw <- list(
    estimate   = dpl_expand(fit$coefficients[, "case"], keep, n),
    se         = dpl_expand(se_raw, keep, n),
    df_raw     = dpl_expand(fit$df.residual, keep, n),
    t          = dpl_expand(t_raw, keep, n),
    p          = dpl_expand(2 * stats::pt(-abs(t_raw), df = fit$df.residual), keep, n),
    df_post    = dpl_expand(fit$df.residual, keep, n),
    sigma2_raw = dpl_expand(fit$sigma^2, keep, n),
    sigma2_post = rep(NA_real_, n),
    df_prior   = rep(NA_real_, n),
    s2_prior   = rep(NA_real_, n),
    n_fitted   = length(keep)
  )
  moderated <- NULL
  if (moderate) {
                                                                           
                                     
    se_mod <- fit$stdev.unscaled[, "case"] * sqrt(fit$s2.post)
    moderated <- list(
      estimate   = dpl_expand(fit$coefficients[, "case"], keep, n),
      se         = dpl_expand(se_mod, keep, n),
      df_raw     = dpl_expand(fit$df.residual, keep, n),
      t          = dpl_expand(fit$t[, "case"], keep, n),
      p          = dpl_expand(fit$p.value[, "case"], keep, n),
      df_post    = dpl_expand(fit$df.total, keep, n),
      sigma2_raw = dpl_expand(fit$sigma^2, keep, n),
      sigma2_post = dpl_expand(fit$s2.post, keep, n),
      df_prior   = dpl_expand(fit$df.prior, keep, n),
      s2_prior   = dpl_expand(fit$s2.prior, keep, n),
      n_fitted   = length(keep)
    )
  }
  list(
    raw = raw,
    moderated = moderated,
    observed = observed,
    trend_used = if (is.null(ebx)) FALSE else ebx$trend,
    trend_kind = if (is.null(ebx)) "none" else ebx$trend_kind,
    ebayes_note = if (is.null(ebx)) "" else ebx$note
  )
}

                                                                        
  
                               
                                                     
                                                      
                                                             
                                                                     
                                
                                                                 
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

                                                                             
                                                             
                                                                             
 
                                                                             
                                                                          
                                                                            
                                                                            
                                                        
 
                                                                           
 
                                                                    
                                                           
                                                                          
                      
                                                                
                                                                             
 
                                                                         
                                                                          
                                                                             
                                                                         
 
                                                                        
                                                                           
                                                             

                                                        
  
                                                                        
                                                                     
                                    
                                                    
dpl_effect_nt <- function(y, is_case, is_control, conf = 0.95) {
  stopifnot(is.matrix(y), length(is_case) == ncol(y),
            length(is_control) == ncol(y))
  Yc <- y[, is_case, drop = FALSE]
  Yk <- y[, is_control, drop = FALSE]
  nc <- rowSums(is.finite(Yc))
  nk <- rowSums(is.finite(Yk))
  mc <- rowMeans(Yc, na.rm = TRUE)
  mk <- rowMeans(Yk, na.rm = TRUE)
                                                                             
                                                                         
                                                                          
              
  sc2 <- rowSums((Yc - mc)^2, na.rm = TRUE) / pmax(nc - 1, 1)
  sk2 <- rowSums((Yk - mk)^2, na.rm = TRUE) / pmax(nk - 1, 1)
  df <- nc + nk - 2
  s2 <- ((nc - 1) * sc2 + (nk - 1) * sk2) / pmax(df, 1)
  d <- mc - mk
  se <- sqrt(s2 * (1 / nc + 1 / nk))
  ok <- df > 0 & is.finite(d) & is.finite(se)
  tcrit <- rep(NA_real_, length(d))
  tcrit[ok] <- stats::qt(1 - (1 - conf) / 2, df[ok])
  lo <- d - tcrit * se
  hi <- d + tcrit * se
  gm_c <- 2^mc
  gm_k <- 2^mk
  dgm <- gm_c - gm_k
                                                                           
                                                               
  se_dgm <- sqrt((log(2))^2 * (gm_c^2 * s2 / nc + gm_k^2 * s2 / nk))
  data.frame(
    delta_log2_pal                  = d,
    geometric_mean_ratio            = 2^d,
    geometric_mean_control_nt       = gm_k,
    geometric_mean_case_nt          = gm_c,
    delta_geometric_mean_nt         = dgm,
    ci_lo_log2_pal                  = lo,
    ci_hi_log2_pal                  = hi,
    ci_lo_ratio                     = 2^lo,
    ci_hi_ratio                     = 2^hi,
    ci_lo_delta_geometric_mean_nt   = dgm - tcrit * se_dgm,
    ci_hi_delta_geometric_mean_nt   = dgm + tcrit * se_dgm,
    stringsAsFactors = FALSE
  )
}
