#!/usr/bin/env Rscript
# telescope_diff.R — the differential-statistics engine.
#
# ROLE OF THIS FILE. The shipped Telescope CLI implements the differential
# statistics as the `telescope diff` subcommand. This R script is the ENGINE it
# runs: it is what computes the statistics, and the Rust side only aggregates
# and runs LAPA.
#
#   - It is kept here, beside the code that drives it, so that the two cannot
#     drift apart silently.
#   - `workflow/test_data/` holds the synthetic profiles, the design matrix,
#     and the expected output. That fixture is the acceptance test for the
#     engine and, through it, for the CLI.
#   - The R dependencies (DEXSeq, DESeq2, limma) are Bioconductor packages;
#     the Rust side deliberately shares no code path with them, which is what
#     makes the comparison meaningful.
#
# FOUR TABLES.
#   usage   (det.tsv) DEXSeq on (gene x transcript x sample) counts.
#   length  (dpl.tsv) a two-sample t on (transcript x sample) log2(polyA
#                    length), with limma's empirical Bayes moderation of the
#                    same fit reported alongside it.
#   gene    (deg.tsv) DESeq2 on the gene-level counts.
#   poly(A) (apa.tsv) LAPA's per-sample site counts, tested with limma.
#
# THE EXTERNAL TOOL IS NOT RUN HERE. LAPA is run by the CLI, which hands its
# output over (`--lapa-dir`). The engine never shells out to it: a statistic
# must come from the tool that owns it, and a tool the engine cannot find has
# to be a missing tool, never a silently different statistic.
#
# THE CONTRAST IS DECLARED. The caller names the case and the control
# (`--case-condition` / `--control-condition`, or `--contrast CASE:CONTROL`).
# Every effect below is case minus control. A factor sorts its levels
# alphabetically, so inferring the direction from the labels would make every
# reported effect depend on how the labels happen to spell.
#
# Pseudoreplication-safe aggregation.
#   The length track MUST aggregate to one observation per (transcript ×
#   sample) BEFORE fitting. Per-read fitting inflates the effective n by
#   ~1e6 and p-values become meaningless.
#
# THE MODEL IS AUDITED, NOT CHOSEN. The length track fits an ordinary
# two-sample t on the (transcript x sample) means and reports limma's
# empirical Bayes moderation of that same fit next to it. The two audited
# candidates are equal_raw_t (the base fit, the decision statistic q_raw) and
# equal_weight_eb (the moderated arm); both are CALIBRATED on the three
# audited cohorts. Weighting the
# observations by 1/measurement_variance was audited and REJECTED: the
# measurement variance is the within-sample part only, the between-sample
# biological scatter of a transcript's tail length is not in it, and the
# weights are therefore wrong for the data they are fitted to.
# audit_dpl_models.R measures that failure -- the weighted p-values are not
# uniform under a known null -- and confirms that the unweighted fit
# reproduces the ordinary two-sample t to machine precision.
#
# THE AUDITED CODE IS THE SHIPPED CODE. The unit builder, the design matrix
# and the fit live in dpl_models.R, and both this engine and the audit call
# them. A copy of the model inside this file would be a second
# implementation, and the audit's verdict would say nothing about it.
#
# THE RESULT MATRIX IS NOT FILTERED. Independent filtering is off, every
# transcript in the input gets a row, and a row without a statistic says why.
# Filtering and plotting are the user's downstream job.
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
#     --case-condition COPD --control-condition HC \
#     --output-dir results/ \
#     --min-weight 10 \
#     --min-counts 20 \
#     --det --dpl
#
# Inputs:
#   --profiles   Glob pointing to one telescope.profile.parquet per sample.
#                Schema (from spec F4):
#                  sample_id, read_id, gene_id, transcript_id,
#                  polyA_length (Int32), posterior_weight (Float64).
#   --aggregate  Pre-aggregated (transcript x sample) TSV; replaces --profiles.
#   --design     TSV with header. Required cols:
#                  sample_id, condition
#                Optional cols: batch, tissue (nuisance covariate).
#   --case-condition, --control-condition
#                The two condition labels. The effect is case minus control.
#   --contrast   The contrast as CASE:CONTROL, an alternative to the two
#                flags above.
#   --lapa-dir     LAPA's output directory, turned into apa.tsv.
#   --output-dir Where to write:
#                  det.tsv, dpl.tsv, deg.tsv, apa.tsv,
#                  sessionInfo.txt

suppressPackageStartupMessages({
  library(optparse)
  library(arrow)
  library(dplyr)
  library(tidyr)
  library(DEXSeq)
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
              help = "Min total counts per transcript for the usage track (DEXSeq dispersion stability; default = 20) [default %default]"),
  make_option("--min-samples", type = "integer", default = 2,
              help = "Min number of samples per transcript for the length track [default %default]"),
  make_option("--case-condition", type = "character",
              help = "The case condition label. The effect is reported as case minus control."),
  make_option("--control-condition", type = "character",
              help = "The control condition label"),
  make_option("--contrast", type = "character",
              help = "The contrast as CASE:CONTROL. An alternative to the two flags above."),
  make_option("--deg", action = "store_true", default = NA,
              help = "Write deg.tsv (DESeq2 on the gene-level counts)"),
  make_option("--det", action = "store_true", default = NA,
              help = "Write det.tsv (DEXSeq differential transcript usage)"),
  make_option("--dpl", action = "store_true", default = NA,
              help = "Write dpl.tsv (the audited limma length model)"),
  make_option("--lapa-dir", type = "character",
              help = "LAPA's output directory, turned into apa.tsv")
)
opt <- parse_args(OptionParser(option_list = option_list))

# WHICH TABLES TO WRITE.
#
# The three flags are three-valued: absent (NA) means "not named". When the
# caller names at least one of them, exactly those are written — the CLI picks
# the modalities it was asked for. When the caller names none, the engine
# keeps its historical behaviour and writes the usage and the length track,
# which is what an unqualified invocation has always meant.
modality_flag <- function(x, when_absent) {
  if (is.null(x) || length(x) == 0 || is.na(x)) when_absent else isTRUE(x)
}
named_modality <- !is.null(opt$deg) && !is.na(opt$deg)
if (!is.null(opt$det)) named_modality <- named_modality || !is.na(opt$det)
if (!is.null(opt$dpl)) named_modality <- named_modality || !is.na(opt$dpl)
if (named_modality) {
  want_deg <- modality_flag(opt$deg, FALSE)
  want_det <- modality_flag(opt$det, FALSE)
  want_dpl <- modality_flag(opt$dpl, FALSE)
} else {
  want_deg <- FALSE
  want_det <- TRUE
  want_dpl <- TRUE
}
cat("[telescope_diff] tables:",
    paste(c(if (want_deg) "deg", if (want_det) "det", if (want_dpl) "dpl"),
          collapse = ", "), "\n")

# The model layer is a separate file so that the audit and this engine run
# the same code. It is found next to this script, or through
# $TELESCOPE_DPL_MODELS.
.this_script <- NA_character_
.fa <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(.fa) > 0) .this_script <- normalizePath(sub("^--file=", "", .fa[1]))
.model_file <- Sys.getenv("TELESCOPE_DPL_MODELS")
if (!nzchar(.model_file)) {
  .cand <- c(if (!is.na(.this_script)) file.path(dirname(.this_script), "dpl_models.R"),
             file.path(getwd(), "dpl_models.R"))
  .cand <- .cand[file.exists(.cand)]
  if (length(.cand) == 0) {
    stop("ERROR: dpl_models.R not found next to telescope_diff.R. ",
         "Set TELESCOPE_DPL_MODELS to its path.")
  }
  .model_file <- .cand[1]
}
source(.model_file)

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

# THE CONTRAST IS DECLARED, NOT INFERRED. A factor sorts its levels
# alphabetically, so the direction of every effect used to depend on how the
# condition labels happened to spell: with COPD and HC the reference level
# was COPD and every reported effect was control minus case. The caller
# now says which label is the case.
resolve_contrast <- function(case_condition, control_condition, contrast) {
  if (!is.null(contrast)) {
    parts <- strsplit(contrast, ":", fixed = TRUE)[[1]]
    if (length(parts) != 2 || any(nzchar(parts) == FALSE)) {
      stop("ERROR: --contrast must be CASE:CONTROL, for example COPD:HC.")
    }
    if (!is.null(case_condition) || !is.null(control_condition)) {
      stop("ERROR: use either --contrast or --case-condition/--control-condition, not both.")
    }
    case_condition <- parts[1]
    control_condition <- parts[2]
  }
  if (is.null(case_condition) || is.null(control_condition)) {
    stop("ERROR: the contrast must be given. Use --case-condition and --control-condition, or --contrast CASE:CONTROL.")
  }
  if (identical(case_condition, control_condition)) {
    stop("ERROR: the case and the control condition are the same label: ", case_condition)
  }
  list(case = case_condition, control = control_condition)
}

contrast_spec <- resolve_contrast(opt$`case-condition`, opt$`control-condition`,
                                  opt$contrast)
case_condition    <- contrast_spec$case
control_condition <- contrast_spec$control
cat("[telescope_diff] Contrast: ", case_condition, " - ", control_condition, "\n", sep = "")

# The reference level is the control, so every coefficient below is
# case minus control regardless of how the labels sort.
raw_condition <- as.character(design$condition)
unknown <- setdiff(unique(raw_condition), c(control_condition, case_condition))
if (length(unknown) > 0) {
  stop("ERROR: the design contains condition label(s) that are neither the case nor the control: ",
       paste(unknown, collapse = ", "),
       ". Known labels: ", paste(sort(unique(raw_condition)), collapse = ", "))
}
design$condition  <- factor(design$condition,
                             levels = c(control_condition, case_condition))
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

# THE DESIGN, BUILT BY THE AUDITED CODE. The reference level is the control,
# so the coefficient of the `case` column is case minus control whatever the
# labels sort like. A batch term is added only when the batch is not
# confounded with the condition. Every track that fits a model uses this one
# design, so they all report the same contrast and the same batch handling.
batch_vec <- if ("batch" %in% colnames(design)) as.character(design$batch) else NULL
dpl_spec <- dpl_design(design$sample_id, as.character(design$condition),
                       case_condition, control_condition, batch_vec)
dpl_design_matrix <- dpl_spec$design
dpl_batch_note <- dpl_spec$batch_note

# The model parameters of every table record the contrast, the batch handling
# and the tool that produced the statistic. The batch note is the audited
# one, never a guess: "no batch column in the design" when the design has
# none.
contrast_note <- paste0("contrast: ", case_condition, " - ", control_condition,
                         " (case - control)")

# =====================================================================
# The transcript counts, shared by the usage and the gene track.
# =====================================================================
# Only the tracks that need them build them: aggregating the per-read
# profiles is the expensive step of the engine, and the length track does
# not use the counts.
if (want_det || want_deg) {
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

# The gene track sums the transcript counts of every gene, so it needs the
# counts BEFORE the usage track's filters. Those filters are the usage
# track's own thresholds (DEXSeq dispersion stability, single-transcript
# genes) and must not remove transcripts from a gene's total.
counts_for_genes <- counts_long
}

# THE MODEL MATRIX DEXSeq FITS THE DISPERSIONS WITH. DEXSeq reduces the
# matrix to its independent columns before it fits the dispersions (its own
# `rmDepCols`). The design `~ sample + exon + condition:exon` is rank-
# deficient as it stands, and DESeq2's rank check aborts the fit that would
# otherwise work, so the same reduction is done here.
rank_reduced_model_matrix <- function(dxd) {
  mm <- model.matrix(design(dxd), as.data.frame(colData(dxd)))
  q <- qr(mm)
  if (q$rank < ncol(mm)) {
    mm[, -q$pivot[(q$rank + 1):ncol(mm)], drop = FALSE]
  } else {
    mm
  }
}

# =====================================================================
# Usage track: ΔPAU via DEXSeq  ->  det.tsv
# =====================================================================
if (want_det) {
cat("\n[telescope_diff] usage track: ΔPAU via DEXSeq\n")
counts_by_tx <- counts_long %>%
  dplyr::group_by(gene_id, transcript_id) %>%
  dplyr::summarize(total = sum(count), .groups = "drop")
# A copy that no filter touches: the reported matrix covers every transcript
# in the input, not only the testable ones.
counts_by_tx_all <- counts_by_tx
keep_tx <- counts_by_tx %>%
  dplyr::filter(total >= opt$`min-counts`) %>%
  dplyr::select(gene_id, transcript_id)
counts_long <- counts_long %>%
  dplyr::semi_join(keep_tx, by = c("gene_id", "transcript_id"))
cat("  transcripts after min-counts filter:",
    dplyr::n_distinct(counts_long$transcript_id), "\n")

# A gene with ONE transcript has no usage to test. DEXSeq's contrast is
# WITHIN a gene, so a single-transcript gene carries no usage variation, and
# its design matrix is rank-deficient. Such genes are dropped, and the count
# is reported so the drop is never silent.
#
# This is not a corner case: in a transcriptome whose gene and transcript ids
# coincide (tx_1 as both), a large share of the genes are singletons.
#
# COUNT TRANSCRIPTS, NOT ROWS. counts_long holds one row per (gene,
# transcript, sample), so dplyr::count(gene_id) counts ROWS -- one transcript
# seen in ten samples counts as ten. Used that way the filter below dropped
# only the genes with a single (transcript x sample) observation and let every
# real single-transcript gene through. They did not kill the run (DEXSeq
# builds its DESeqDataSet with ignoreRank = TRUE, so a rank-deficient design
# yields NA for that exon instead of aborting); they came back as NA p-value
# and NA padj rows. Measured on the ten-sample cohort: 5,845 single-transcript
# genes, 116 dropped, 5,729 NA rows.
tx_per_gene <- counts_long %>%
  dplyr::distinct(gene_id, transcript_id) %>%
  dplyr::count(gene_id, name = "n_tx")
genes_keep  <- tx_per_gene %>% dplyr::filter(n_tx >= 2) %>% dplyr::pull(gene_id)
n_singleton_genes <- nrow(tx_per_gene) - length(genes_keep)
if (n_singleton_genes > 0) {
  cat("  dropped", n_singleton_genes,
      "single-transcript genes (no within-gene usage contrast)\n")
  counts_long <- counts_long %>% dplyr::filter(gene_id %in% genes_keep)
  counts_by_tx <- counts_by_tx %>% dplyr::filter(gene_id %in% genes_keep)
}
# FAIL CLOSED. Every gene that survived must still carry two transcripts.
# Without this check the filter above can silently become a no-op, and the
# surviving single-transcript genes return from DEXSeq as NA p-values that
# look like a result.
tx_after <- counts_long %>%
  dplyr::distinct(gene_id, transcript_id) %>%
  dplyr::count(gene_id, name = "n_tx")
if (nrow(tx_after) > 0 && min(tx_after$n_tx) < 2) {
  stop("ERROR: ", sum(tx_after$n_tx < 2),
       " gene(s) with a single transcript survived the singleton filter.")
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
  # THE DISPERSION FIT. estimateDispersions can fail on tiny synthetic
  # fixtures where the gene-wise dispersions don't span enough orders of
  # magnitude for parametric curve fitting -- and on a handful of genes in a
  # real cohort. DESeq2's own remedy for that error is to keep the gene-wise
  # estimates as the final estimates and go on with the test, and that is
  # what the fallback below does.
  #
  # THE MODEL MATRIX IS WHAT MAKES THE FALLBACK WORK. The gene-wise step
  # checks the full rank of the model matrix, and the design
  # `~ sample + exon + condition:exon` is rank-deficient as it stands, so
  # the call used to abort with "the model matrix is not full rank" instead
  # of producing gene-wise dispersions. DEXSeq fits the dispersions with the
  # matrix reduced to its independent columns, so the fallback uses the same
  # reduction (see rank_reduced_model_matrix above).
  #
  # THE HANDLER'S RESULT IS RETURNED, not assigned in place: a plain
  # assignment there would only touch the handler's own copy and the
  # fallback would silently do nothing.
  disp_result <- tryCatch({
    list(dxd = estimateDispersions(dxd, quiet = TRUE),
         note = "parametric dispersion fit")
  }, error = function(e) {
    cat("  WARN: the dispersion trend could not be fitted (", conditionMessage(e),
        "); keeping DEXSeq's gene-wise dispersion estimates.\n")
    dxd_fallback <- estimateDispersionsGeneEst(
      dxd, quiet = TRUE,
      modelMatrix = rank_reduced_model_matrix(dxd))
    dispersions(dxd_fallback) <- mcols(dxd_fallback)$dispGeneEst
    list(dxd = dxd_fallback,
         note = "gene-wise dispersion estimates (the dispersion trend could not be fitted)")
  })
  dxd <- disp_result$dxd
  disp_note <- disp_result$note
  cat("  dispersion:", disp_note, "\n")
  dxd <- testForDEU(dxd,
                    reducedModel = ~ sample + exon + condition,
                    fullModel    = ~ sample + exon + condition:exon)
  dxd <- estimateExonFoldChanges(dxd, fitExpToVar = "condition")
})
# THE RESULT MATRIX IS NOT FILTERED.
#
# Independent filtering is switched off, so every tested feature gets a padj
# and no feature is dropped for having a low mean. Filtering and plotting are
# the user's downstream job; the software reports what it computed.
#
# The matrix covers EVERY transcript in the input, not only the testable
# ones: a feature that could not be tested is reported with the reason, so
# the matrix is complete and nothing is silently missing.
dxr <- DEXSeqResults(dxd, independentFiltering = FALSE)
filter_threshold <- attr(dxr, "filterThreshold")
if (is.null(filter_threshold) || length(filter_threshold) == 0) filter_threshold <- NA_real_
n_independent_filtered <- 0L

# DEXSeq's OWN effect estimate. The column name carries the contrast
# direction (`log2fold_<case>_<control>`), so its sign is case minus control
# because the factor's reference level is the control.
lfc_col <- grep("^log2fold_", colnames(dxr), value = TRUE)
if (length(lfc_col) != 1) {
  stop("ERROR: expected exactly one DEXSeq effect column, found: ",
       paste(lfc_col, collapse = ", "))
}
dexseq_lfc <- as.numeric(dxr[[lfc_col]])

# The engine's own unnormalised mean ratio, under a name that says what it
# is. It is NOT a DEXSeq effect estimate: it is the ratio of the two groups'
# mean counts with a 0.5 pseudocount.
case_samples    <- design_aligned$sample_id[design_aligned$condition == case_condition]
control_samples <- design_aligned$sample_id[design_aligned$condition == control_condition]
tx_mean_case    <- rowMeans(count_mat[, case_samples, drop = FALSE], na.rm = TRUE)
tx_mean_control <- rowMeans(count_mat[, control_samples, drop = FALSE], na.rm = TRUE)
log2_mean_ratio_by_tx <- log2((tx_mean_case + 0.5) / (tx_mean_control + 0.5))

pau_out <- counts_by_tx_all
pau_out$case_condition    <- case_condition
pau_out$control_condition <- control_condition
pau_out$pvalue         <- NA_real_
pau_out$padj           <- NA_real_
pau_out$log2FoldChange  <- NA_real_
pau_out$log2_mean_ratio <- NA_real_
idx <- match(paste(pau_out$gene_id, pau_out$transcript_id),
             paste(dxr$groupID, dxr$featureID))
tested <- !is.na(idx)
pau_out$pvalue[tested]         <- dxr$pvalue[idx[tested]]
pau_out$padj[tested]           <- dxr$padj[idx[tested]]
pau_out$log2FoldChange[tested] <- dexseq_lfc[idx[tested]]
pau_out$log2_mean_ratio[tested] <- log2_mean_ratio_by_tx[idx[tested]]
pau_out$n_case    <- sum(design$condition == case_condition)
pau_out$n_control <- sum(design$condition == control_condition)
pau_out$independent_filter_threshold <- filter_threshold
pau_out$n_independent_filtered <- n_independent_filtered
# WHY A ROW HAS NO ADJUSTED P-VALUE. With the filtering off, the only
# reasons left are a missing p-value and a feature that was never tested.
pau_out$padj_na_reason <- ifelse(!tested, "not_tested",
                          ifelse(is.na(pau_out$pvalue), "no_p_value", "none"))
# THE MODEL, RECORDED. The tool that produced the statistic, the contrast and
# the batch handling (the audited note, never a guess). A transcript DEXSeq
# never saw carries no model, the same way the length track's untestable rows
# do.
pau_out$model_name <- ifelse(tested, "dexseq", "not_tested")
pau_out$model_parameters <- paste0(
  "DEXSeq differential transcript usage on (gene x transcript x sample) counts; ",
  "log2FoldChange is DEXSeq's own effect estimate, log2_mean_ratio is the ",
  "engine's unnormalised mean ratio of the two groups' mean counts; ",
  contrast_note, "; ", dpl_batch_note)
pau_out <- pau_out[, c("gene_id", "transcript_id", "case_condition",
                       "control_condition", "pvalue", "padj", "log2FoldChange",
                       "log2_mean_ratio", "total", "n_case", "n_control",
                       "independent_filter_threshold", "n_independent_filtered",
                       "padj_na_reason", "model_name", "model_parameters")]

write.table(pau_out,
            file = file.path(opt$`output-dir`, "det.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("[telescope_diff]  wrote det.tsv (",
    nrow(pau_out), "transcripts)\n")
}

# =====================================================================
# Gene track: differential expression via DESeq2  ->  deg.tsv
# =====================================================================
if (want_deg) {
cat("\n[telescope_diff] gene track: differential expression via DESeq2\n")
if (!requireNamespace("DESeq2", quietly = TRUE)) {
  stop("ERROR: the gene track needs DESeq2, which is not installed in this R ",
       "library. Create the pinned environment: mamba env create -f workflow/environment.yml")
}

# THE GENE-LEVEL COUNTS. The transcript counts of the input, summed per gene.
# The counts are the ones BEFORE the usage track's filters (see
# counts_for_genes): the filters are the usage track's thresholds and have no
# business removing transcripts from a gene's total.
gene_counts_long <- counts_for_genes %>%
  dplyr::group_by(gene_id, sample_id) %>%
  dplyr::summarize(count = sum(count), .groups = "drop")
gene_ids_all <- sort(unique(gene_counts_long$gene_id))
cat("  genes:", length(gene_ids_all), "\n")

# The matrix is over the DESIGN's samples: a sample with no counts at all
# still has a column, with zeros.
gene_mat <- matrix(0, nrow = length(gene_ids_all), ncol = nrow(design),
                   dimnames = list(gene_ids_all, design$sample_id))
rows_i <- match(gene_counts_long$gene_id, gene_ids_all)
cols_j <- match(gene_counts_long$sample_id, design$sample_id)
gene_mat[cbind(rows_i, cols_j)] <- gene_counts_long$count
gene_mat <- matrix(as.integer(gene_mat), nrow = nrow(gene_mat),
                   dimnames = dimnames(gene_mat))
if (any(gene_mat < 0)) {
  stop("ERROR: the gene-level counts contain a negative value.")
}

# THE DESIGN. The same design the length track uses: the reference level is
# the control, and the batch term is present only when the batch is not
# confounded with the condition.
design_genes <- design[match(colnames(gene_mat), design$sample_id), , drop = FALSE]
if (any(is.na(design_genes$sample_id))) {
  stop("ERROR: design sample_ids do not match the gene count matrix columns.")
}
rownames(design_genes) <- colnames(gene_mat)
design_genes$condition <- factor(as.character(design_genes$condition),
                                  levels = c(control_condition, case_condition))
if (dpl_spec$batch_used) {
  design_genes$batch <- factor(as.character(design_genes$batch))
  deg_formula <- ~ condition + batch
} else {
  deg_formula <- ~ condition
}

dds <- DESeq2::DESeqDataSetFromMatrix(countData = gene_mat,
                                       colData = design_genes,
                                       design = deg_formula)
dds <- DESeq2::estimateSizeFactors(dds)

# THE DISPERSION FIT. estimateDispersions can fail when the gene-wise
# dispersions do not span enough orders of magnitude for the curve fitting
# -- the case on a handful of genes. DESeq2's own remedy for that error is
# to keep the gene-wise estimates as the final estimates and go on with the
# Wald test, and that is what the fallback below does. The handler assigns
# with `<<-`: a plain assignment there would only touch the handler's own
# copy and the fallback would silently do nothing.
disp_note <- "parametric dispersion fit"
disp_ok <- tryCatch({
  dds <- DESeq2::estimateDispersions(dds, quiet = TRUE)
  TRUE
}, error = function(e) {
  cat("  WARN: the dispersion trend could not be fitted (", conditionMessage(e),
      "); keeping DESeq2's gene-wise dispersion estimates.\n")
  dds_fallback <- DESeq2::estimateDispersionsGeneEst(dds, quiet = TRUE)
  DESeq2::dispersions(dds_fallback) <- S4Vectors::mcols(dds_fallback)$dispGeneEst
  dds <<- dds_fallback
  disp_note <<- "gene-wise dispersion estimates (the dispersion trend could not be fitted)"
  FALSE
})

# THE TEST. The Wald test is DESeq2's own negative binomial test, and
# results() reads its output. Independent filtering is OFF: the result matrix
# is not filtered, and filtering and plotting are the user's downstream job.
# The contrast is given explicitly, so the reported effect is case minus
# control whatever the condition labels sort like.
dds <- DESeq2::nbinomWaldTest(dds)
dres <- DESeq2::results(dds, independentFiltering = FALSE,
                        contrast = c("condition", case_condition, control_condition))
dres_df <- as.data.frame(dres)

deg_out <- data.frame(
  gene_id                = gene_ids_all,
  case_condition         = rep(case_condition, length(gene_ids_all)),
  control_condition      = rep(control_condition, length(gene_ids_all)),
  baseMean               = NA_real_,
  log2FoldChange         = NA_real_,
  lfcSE                  = NA_real_,
  stat                  = NA_real_,
  pvalue                 = NA_real_,
  padj                   = NA_real_,
  n_case                 = rep(sum(design$condition == case_condition), length(gene_ids_all)),
  n_control              = rep(sum(design$condition == control_condition), length(gene_ids_all)),
  independent_filter_threshold = NA_real_,
  n_independent_filtered = 0L,
  stringsAsFactors = FALSE
)
# EVERY GENE IN THE INPUT GETS A ROW. A gene that DESeq2 could not fit is
# reported with the reason, so the matrix is complete.
deg_idx <- match(deg_out$gene_id, rownames(dres_df))
deg_tested <- !is.na(deg_idx)
deg_out$baseMean[deg_tested]       <- dres_df$baseMean[deg_idx[deg_tested]]
deg_out$log2FoldChange[deg_tested] <- dres_df$log2FoldChange[deg_idx[deg_tested]]
deg_out$lfcSE[deg_tested]          <- dres_df$lfcSE[deg_idx[deg_tested]]
deg_out$stat[deg_tested]           <- dres_df$stat[deg_idx[deg_tested]]
deg_out$pvalue[deg_tested]         <- dres_df$pvalue[deg_idx[deg_tested]]
deg_out$padj[deg_tested]           <- dres_df$padj[deg_idx[deg_tested]]
deg_out$padj_na_reason <- ifelse(!deg_tested, "not_tested",
                          ifelse(is.na(deg_out$pvalue), "no_p_value", "none"))
deg_out$model_name <- "deseq2_wald"
deg_out$model_parameters <- paste0(
  "DESeq2 on the gene-level counts (the transcript counts summed per gene); ",
  "negative binomial Wald test; results(independentFiltering = FALSE); ",
  contrast_note, "; ", dpl_batch_note, "; ", disp_note)

write.table(deg_out,
            file = file.path(opt$`output-dir`, "deg.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("[telescope_diff]  wrote deg.tsv (", nrow(deg_out), "genes,",
    sum(deg_out$padj_na_reason == "none"), "tested)\n")
}

# =====================================================================
# Length track: ΔPAL — one observation per (transcript × sample)  ->  dpl.tsv
# =====================================================================
if (want_dpl) {
cat("\n[telescope_diff] length track: ΔPAL via an equal-weight two-sample t on the sample means\n")
cat("  aggregating to (transcript × sample) BEFORE fitting to avoid pseudoreplication.\n")

# THE UNIT. One row per (transcript x biological sample). The reads of a
# sample are not replicates; they are averaged into one measurement of that
# transcript's tail length in that sample, together with the variance of that
# measurement.
#
# From the four moments of the cell:
#   W    = sum(w)                posterior mass
#   swy  = sum(w * y)            y = log2(max(polyA_length, 1))
#   swy2 = sum(w * y^2)
#   sw2  = sum(w^2)
# the mean is swy/W, the Kish effective sample size is W^2/sw2, the
# within-cell variance is (swy2 - swy^2/W) / (Kish_Neff - 1) and the variance
# of the cell mean is that variance divided by Kish_Neff.
#
# The Kish denominator is Kish_Neff - 1 and not W - 1 on purpose: with the
# soft posterior a cell holds one row per (read, candidate transcript) pair,
# so the row count is not the sample size but Kish_Neff is. For equal
# weights the two conventions coincide.

need_unit_cols <- c("sample_id", "gene_id", "transcript_id", "weight_sum",
                    "weighted_log2pal_sum", "weight_sq_sum",
                    "weighted_log2pal_sq_sum")
if (!is.null(profiles_df)) {
  profiles_df$log2_pal <- log2(pmax(profiles_df$polyA_length, 1))
  unit_agg <- profiles_df %>%
    dplyr::group_by(sample_id, gene_id, transcript_id) %>%
    dplyr::summarize(
      weight_sum             = sum(posterior_weight),
      weighted_log2pal_sum   = sum(posterior_weight * log2_pal),
      weight_sq_sum          = sum(posterior_weight^2),
      weighted_log2pal_sq_sum = sum(posterior_weight * log2_pal^2),
      .groups = "drop"
    )
} else {
  missing_unit_cols <- setdiff(need_unit_cols, colnames(agg_raw))
  if (length(missing_unit_cols) > 0) {
    stop("ERROR: the aggregate table is missing column(s): ",
         paste(missing_unit_cols, collapse = ", "))
  }
  unit_agg <- agg_raw[, need_unit_cols]
}

# THE UNIT, BUILT BY THE AUDITED CODE. dpl_units() is the function the audit
# calibrates: the four moments become a mean, a Kish effective sample size, a
# within-cell variance and a quality gate. The gene of a transcript is taken
# from the aggregate, before the merge with the design reorders the rows.
units <- dpl_units(unit_agg, design[, c("sample_id", "condition")],
                   min_weight = opt$`min-weight`, min_neff = 2)
cat("  (transcript × sample) cells:", nrow(units),
    " passing the quality gate:", sum(units$passed_quality_gate), "\n")

# EVERY transcript in the input is reported, whether or not it was testable.
tx_ids_all <- sort(unique(unit_agg$transcript_id))
gene_of <- tapply(unit_agg$gene_id, unit_agg$transcript_id, function(x) x[1])[tx_ids_all]

cat("  design:", paste(colnames(dpl_design_matrix), collapse = ", "),
    "|", dpl_batch_note, "\n")

# THE FIT. One least squares for the whole transcriptome, then limma's
# empirical Bayes variance moderation of that same fit. The observations are
# the (transcript x sample) means and they carry EQUAL weight: the precision
# weighting was audited and rejected, and the header says why.
sample_order <- design$sample_id
y_mat <- dpl_matrices(units, sample_order, tx_ids_all, "mean_log2_pal", NULL)$y

cat("  fitting", length(tx_ids_all), "transcripts…\n")
fit_raw <- dpl_fit_model(y_mat, dpl_design_matrix, NULL, FALSE)
fit     <- dpl_fit_model(y_mat, dpl_design_matrix, NULL, TRUE)

se_raw  <- fit_raw$se
t_raw   <- fit_raw$t
df_raw  <- fit_raw$df_raw
p_raw   <- fit_raw$p
se_mod  <- fit$se
t_mod   <- fit$t
df_post <- fit$df_post
p_mod   <- fit$p

# How many samples of each condition actually backed each transcript.
n_adm <- units[units$passed_quality_gate, ] %>%
  dplyr::count(transcript_id, condition)
# dplyr::count names the tally column `n`.
n_adm_case <- n_adm$n[match(paste(tx_ids_all, case_condition),
                            paste(n_adm$transcript_id, n_adm$condition))]
n_adm_ctrl <- n_adm$n[match(paste(tx_ids_all, control_condition),
                            paste(n_adm$transcript_id, n_adm$condition))]
n_adm_case[is.na(n_adm_case)] <- 0
n_adm_ctrl[is.na(n_adm_ctrl)] <- 0

# THE AUDITED POPULATION, AND THE ADJUSTMENT OVER IT.
#
# The audit keeps a transcript only when BOTH groups have at least
# `min-samples` admissible cells: a group with a single cell has no
# within-group variance, so what it contributes to the two-sample t is not a
# variance at all. The shipped table applies the audit's own rule, so the
# shipped q-value IS the audited q-value. Without the mask the BH adjustment
# would run over every row that happens to have a p-value, which is a WIDER
# population than the one the audit certified, and the shipped q-value would
# then not be the audited statistic.
#
# The mask is applied BEFORE `p.adjust()`, because the adjustment's
# denominator is the number of p-values it is given: masking afterwards would
# leave the denominator at the wider population and change nothing.
min_samples <- opt$`min-samples`
too_few_cells <- n_adm_case < min_samples | n_adm_ctrl < min_samples
fit_failed <- !is.finite(t_raw)
masked <- fit_failed | too_few_cells
t_raw[masked] <- NA_real_
p_raw[masked] <- NA_real_
t_mod[masked] <- NA_real_
p_mod[masked] <- NA_real_
q_raw <- stats::p.adjust(p_raw, method = "BH")
q_mod <- stats::p.adjust(p_mod, method = "BH")
cat("  testable transcripts:", sum(!masked), "(both groups with >=",
    min_samples, "admissible cells; the audit's population)\n")

# WHY A ROW HAS NO ADJUSTED P-VALUE. The length track has no independent
# filter: it drops no row for having a low mean, so the threshold is NA and
# the count of filtered rows is 0. A row without a statistic says why.
pal_na_reason <- ifelse(
  !is.na(p_raw), "none",
  ifelse(n_adm_case == 0 & n_adm_ctrl == 0,
         "no admissible cells at all",
         "too few admissible cells in one group"))

pal_out <- data.frame(
  gene_id                    = unname(gene_of),
  transcript_id              = tx_ids_all,
  case_condition             = rep(case_condition, length(tx_ids_all)),
  control_condition          = rep(control_condition, length(tx_ids_all)),
  delta_pal_case_minus_control = fit_raw$estimate,
  se_raw                     = se_raw,
  df_raw                     = df_raw,
  t_raw                      = t_raw,
  p_raw                      = p_raw,
  q_raw                      = q_raw,
  # limma's empirical Bayes moderates the VARIANCE, not the coefficient, so
  # the moderated effect estimate is the raw one.
  delta_pal_moderated        = fit$estimate,
  se_moderated               = se_mod,
  df_post                    = df_post,
  t_moderated                = t_mod,
  p_moderated                = p_mod,
  q_moderated                = q_mod,
  n_case                     = rep(sum(design$condition == case_condition), length(tx_ids_all)),
  n_control                  = rep(sum(design$condition == control_condition), length(tx_ids_all)),
  n_admissible_case          = n_adm_case,
  n_admissible_control       = n_adm_ctrl,
  independent_filter_threshold = NA_real_,
  n_independent_filtered     = 0L,
  padj_na_reason             = pal_na_reason,
  model_name                 = "equal_raw_t",
  model_parameters           = paste0(
    "base fit: equal_raw_t (unweighted two-sample t on the transcript x sample ",
    "means); moderated arm: equal_weight_eb (limma eBayes: s2.prior=",
    format(fit$s2_prior, digits = 10), ", df.prior=",
    format(fit$df_prior, digits = 10), "); audit: ",
    "workflow/scripts/audit_dpl_models.R (verdict CALIBRATED); ",
    "independent filtering: not applied (the length track drops no row, so the ",
    "independent filter threshold is NA and nothing was filtered); ",
    contrast_note, "; ", dpl_batch_note),
  stringsAsFactors = FALSE
)
# A transcript whose FIT failed carries NA. A transcript that was fitted but
# has too few admissible cells keeps its effect estimate: the difference of
# the two group means is computed, it is the TEST that is not supported. The
# admissible sample counts and the NA reason say which.
pal_out$delta_pal_case_minus_control[fit_failed] <- NA_real_
pal_out$model_name[is.na(pal_out$t_raw)] <- "not_testable"

write.table(pal_out,
            file = file.path(opt$`output-dir`, "dpl.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("[telescope_diff]  wrote dpl.tsv (",
    nrow(pal_out), "transcripts,", sum(!is.na(pal_out$p_raw)), "tested)\n")
cat("[telescope_diff]  limma eBayes priors: s2.prior =", fit$s2_prior,
    " df.prior =", fit$df_prior, "\n")

# The moderation must not have emptied the moderated columns.
if (sum(is.finite(fit$sigma2_post)) > 0) {
  stopifnot(all(is.finite(fit$sigma2_post[is.finite(fit$sigma2_post)])))
  stopifnot(all(is.finite(fit$sigma2_post[is.finite(fit$sigma2_post)]) &
                fit$sigma2_post[is.finite(fit$sigma2_post)] > 0))
}
}

# =====================================================================
# Alternative poly(A) track: LAPA's per-sample counts  ->  apa.tsv
# =====================================================================
if (!is.null(opt$`lapa-dir`)) {
cat("\n[telescope_diff] alternative poly(A) track: LAPA's per-sample counts\n")

# ---------------------------------------------------------------------------
# LAPA's per-sample output.
# ---------------------------------------------------------------------------
# LAPA writes its poly(A) clusters as a BED-like TSV with NO header, in this
# column order (LAPA's own `lapa/utils/io.py`):
#
#   Chromosome  Start  End  polyA_site  count  Strand  Feature  gene_id
#   tpm  gene_count  usage  fracA  signal  annotated_site
#
# `count` is the number of reads ending at that site in that sample, and
# `polyA_site` is the genomic coordinate of the peak. A site is identified
# the way LAPA identifies it: chromosome, peak and strand, which is LAPA's
# own `Chromosome:polyA_site:Strand` name.
LAPA_CLUSTER_COLUMNS <- c("Chromosome", "Start", "End", "polyA_site", "count",
                          "Strand", "Feature", "gene_id", "tpm", "gene_count",
                          "usage", "fracA", "signal", "annotated_site")

read_lapa_sample <- function(p) {
  df <- utils::read.delim(p, header = FALSE, sep = "\t", quote = "",
                          comment.char = "", na.strings = c("NA", ""),
                          col.names = LAPA_CLUSTER_COLUMNS)
  if (nrow(df) == 0) return(NULL)
  df$site_id <- paste(df$Chromosome, df$polyA_site, df$Strand, sep = ":")
  df$count <- suppressWarnings(as.numeric(df$count))
  df
}

# Where LAPA's per-sample cluster files are, and which of the two is read.
# LAPA writes the same clusters twice: `<dir>/sample/<sample>.bed` holds only
# the sites that were replicated within their dataset, `<dir>/raw_sample/
# <sample>.bed` holds every site LAPA called in that sample. The replicated
# clusters are the ones LAPA's own result API reads by default, and they are
# the right denominator for a usage test -- a site that survived the
# replication filter is a real poly(A) site in that sample, whereas an
# unreplicated peak is mostly noise. So `sample/` is preferred here and
# `raw_sample/` is the fallback.
# A per-sample run of LAPA puts the files one directory deeper, so the
# reader also looks one level down.
lapa_dir <- opt$`lapa-dir`
if (!dir.exists(lapa_dir)) {
  stop("ERROR: the LAPA output directory does not exist: ", lapa_dir)
}
lapa_beds <- c(
  list.files(file.path(lapa_dir, "sample"), pattern = "\\.bed$", full.names = TRUE),
  list.files(file.path(lapa_dir, "raw_sample"), pattern = "\\.bed$", full.names = TRUE),
  unlist(lapply(list.dirs(lapa_dir, recursive = FALSE), function(d) {
    list.files(file.path(d, "sample"), pattern = "\\.bed$", full.names = TRUE)
  })),
  list.files(lapa_dir, pattern = "\\.bed$", full.names = TRUE)
)
lapa_beds <- sort(unique(lapa_beds))
if (length(lapa_beds) == 0) {
  stop("ERROR: no LAPA cluster file (*.bed) under ", lapa_dir)
}
lapa_samples <- sub("\\.bed$", "", basename(lapa_beds))
cat("  LAPA samples:", length(lapa_beds), "\n")

lapa_rows <- lapply(lapa_beds, read_lapa_sample)
names(lapa_rows) <- lapa_samples
lapa_rows <- lapa_rows[!vapply(lapa_rows, is.null, logical(1))]

# The samples the fit uses: the ones LAPA produced AND the ones the design
# knows. A sample the design has but LAPA did not is reported, not silently
# dropped.
lapa_sample_names <- names(lapa_rows)
design_samples <- design$sample_id
missing_lapa <- setdiff(design_samples, lapa_sample_names)
if (length(missing_lapa) > 0) {
  warning("WARN: no LAPA output for design sample(s): ",
          paste(missing_lapa, collapse = ", "))
}
apa_samples <- intersect(design_samples, lapa_sample_names)
if (length(apa_samples) < 2) {
  stop("ERROR: the poly(A) track needs at least two samples with both LAPA ",
       "output and a design row; found ", length(apa_samples), ".")
}

# THE (SITE x SAMPLE) COUNT MATRIX. A site that LAPA did not call in a sample
# has no reads there, so its count for that sample is zero.
apa_site_ids <- sort(unique(unlist(lapply(lapa_rows, function(d) d$site_id))))
apa_mat <- matrix(0, nrow = length(apa_site_ids), ncol = length(apa_samples),
                   dimnames = list(apa_site_ids, apa_samples))
for (s in apa_samples) {
  d <- lapa_rows[[s]]
  d <- d[d$site_id %in% apa_site_ids, ]
  j <- match(d$site_id, apa_site_ids)
  apa_mat[cbind(j, match(s, apa_samples))] <- d$count
}
apa_mat[is.na(apa_mat)] <- 0

# The gene of a site: LAPA's own gene_id column. The first non-missing one
# wins, over the samples in a fixed order.
apa_gene <- rep(NA_character_, length(apa_site_ids))
for (s in apa_samples) {
  d <- lapa_rows[[s]]
  d <- d[match(apa_site_ids, d$site_id), ]
  g <- d$gene_id
  take <- is.na(apa_gene) & !is.na(g) & nzchar(g)
  apa_gene[take] <- g[take]
}

# THE TEST. limma on the (site x sample) count matrix, with the same design
# the length track uses: the reference level is the control, so the effect is
# case minus control.
apa_design_mat <- dpl_design_matrix[match(apa_samples, design$sample_id), , drop = FALSE]
rownames(apa_design_mat) <- apa_samples
if (any(is.na(apa_design_mat[, "case"]))) {
  stop("ERROR: the poly(A) design matrix has no 'case' column.")
}
if (nrow(apa_design_mat) - ncol(apa_design_mat) < 1) {
  stop("ERROR: the poly(A) track needs at least one residual degree of freedom; ",
       nrow(apa_design_mat), " samples and ", ncol(apa_design_mat),
       " design columns leave none.")
}
apa_fit <- limma::lmFit(apa_mat, apa_design_mat)
apa_fit <- limma::eBayes(apa_fit)
# The adjusted p-values are limma's own (BH, over all the sites).
apa_tab <- limma::topTable(apa_fit, coef = "case", number = Inf,
                           adjust.method = "BH")
apa_padj <- apa_tab$adj.P.Val[match(apa_site_ids, rownames(apa_tab))]

apa_case_samples    <- apa_samples[as.character(design$condition[
  match(apa_samples, design$sample_id)]) == case_condition]
apa_control_samples <- apa_samples[as.character(design$condition[
  match(apa_samples, design$sample_id)]) == control_condition]

apa_out <- data.frame(
  gene_id           = apa_gene,
  site_id           = apa_site_ids,
  case_condition    = rep(case_condition, length(apa_site_ids)),
  control_condition = rep(control_condition, length(apa_site_ids)),
  usage_case        = rowMeans(apa_mat[, apa_case_samples, drop = FALSE], na.rm = TRUE),
  usage_control     = rowMeans(apa_mat[, apa_control_samples, drop = FALSE], na.rm = TRUE),
  delta_usage       = apa_fit$coefficients[, "case"],
  pvalue            = apa_fit$p.value[, "case"],
  padj              = apa_padj,
  n_case            = rep(length(apa_case_samples), length(apa_site_ids)),
  n_control         = rep(length(apa_control_samples), length(apa_site_ids)),
  model_name        = "limma_eb",
  stringsAsFactors = FALSE
)
apa_out$model_parameters <- paste0(
  "limma::lmFit + eBayes on the (site x sample) count matrix; the counts are ",
  "LAPA's own per-sample counts, taken from the replicated-cluster files ",
  "(sample/), i.e. only the sites LAPA kept after its replication filter; ",
  contrast_note, "; ", dpl_batch_note)

write.table(apa_out,
            file = file.path(opt$`output-dir`, "apa.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("[telescope_diff]  wrote apa.tsv (", nrow(apa_out), "sites,",
    sum(is.finite(apa_out$pvalue)), "tested)\n")
}

# =====================================================================
# Session info for reproducibility.
# =====================================================================
writeLines(capture.output(sessionInfo()),
           file.path(opt$`output-dir`, "sessionInfo.txt"))
cat("\n[telescope_diff] done. Output in:", opt$`output-dir`, "\n")
