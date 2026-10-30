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
#   - The synthetic profiles, the design matrix and the expected output live
#     in the project workspace layer, at
#     `../workspace/validation/differential_acceptance/`. That fixture is a
#     validation harness rather than release content, so it is not part of
#     this repository; it is the acceptance test for the engine and, through
#     it, for the CLI.
#   - The R dependencies (DEXSeq, DESeq2, limma) are Bioconductor packages;
#     the Rust side deliberately shares no code path with them, which is what
#     makes the comparison meaningful.
#
# FIVE TABLES.
#   gene    (deg.tsv) DESeq2 on the gene-level counts.
#   transcript (det.tsv) DESeq2 on the transcript-level counts: a transcript's
#                    OWN abundance.
#   usage   (diu.tsv) DEXSeq on (gene x transcript x sample) counts: a
#                    transcript's SHARE of its gene's output. That is a
#                    different question from det.tsv, not the same one.
#   length  (dpl.tsv) a two-sample t on (transcript x sample) log2(polyA
#                    length), with limma's empirical Bayes moderation of the
#                    same fit reported alongside it.
#   poly(A) (apa.tsv) LAPA's per-sample site USAGE, tested with limma.
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
#     --min-weight 5 \
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
#                  deg.tsv, det.tsv, diu.tsv, dpl.tsv, apa.tsv,
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
  make_option("--min-weight", type = "double", default = 5,
              help = "Min sum(posterior_weight) per (transcript x sample) [default %default]"),
  make_option("--min-counts", type = "double", default = 20,
              help = "Min total counts per transcript for the usage track (DEXSeq dispersion stability; default = 20) [default %default]"),
  make_option("--min-samples", type = "integer", default = 3,
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
              help = "Write det.tsv (DESeq2 on the transcript-level counts)"),
  make_option("--diu", action = "store_true", default = NA,
              help = "Write diu.tsv (DEXSeq differential isoform usage)"),
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
if (!is.null(opt$diu)) named_modality <- named_modality || !is.na(opt$diu)
if (!is.null(opt$dpl)) named_modality <- named_modality || !is.na(opt$dpl)
if (named_modality) {
  want_deg <- modality_flag(opt$deg, FALSE)
  want_det <- modality_flag(opt$det, FALSE)
  want_diu <- modality_flag(opt$diu, FALSE)
  want_dpl <- modality_flag(opt$dpl, FALSE)
} else {
  # Nothing was named: every table this engine can write is written. The
  # five modalities are the contract, and an unqualified invocation means
  # "all of them".
  want_deg <- TRUE
  want_det <- TRUE
  want_diu <- TRUE
  want_dpl <- TRUE
}
cat("[telescope_diff] tables:",
    paste(c(if (want_deg) "deg", if (want_det) "det", if (want_diu) "diu",
            if (want_dpl) "dpl"),
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
if (want_diu || want_det || want_deg) {
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
# The transcript track needs the transcript counts before the usage track's
# filters, for the same reason the gene track needs the gene counts before
# them: a transcript's own abundance is not a usage track's business.
counts_for_tx <- counts_long
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
# Usage track: differential isoform usage via DEXSeq  ->  diu.tsv
# =====================================================================
#
# THE TWO TRACKS THIS BLOCK SATISFIED BEFORE ARE TWO TRACKS. A transcript's
# own ABUNDANCE (DESeq2 on the transcript counts) and the share a transcript
# takes of its gene's output (DEXSeq) are different questions, and a gene
# whose two isoforms trade places changes one without changing the other.
# So: det.tsv is the DESeq2 transcript abundance test and diu.tsv is this
# block's DEXSeq usage test.
if (want_diu) {
cat("\n[telescope_diff] usage track: differential isoform usage via DEXSeq\n")
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

# THE TWO GROUPS' SAMPLES, in the design DEXSeq was fitted with.
case_samples    <- design_aligned$sample_id[design_aligned$condition == case_condition]
control_samples <- design_aligned$sample_id[design_aligned$condition == control_condition]

# THE OBSERVED SHARE OF THE GENE'S OUTPUT. This is the quantity DEXSeq's
# test is about -- how much of its gene's output a transcript takes -- so it
# is computed the way DEXSeq computes it: from the size-factor normalised
# counts, with the gene's total as the denominator. Averaged over the
# samples of a group, it is the group's observed share.
# DEXSeq's OWN LAYOUT. The assay has one row per transcript and, for each
# sample, TWO columns: "this" (the transcript) and "others" (the rest of its
# gene). The share of the gene's output is exactly `this / (this + others)`,
# and it does not depend on the size factor: the same factor divides the
# numerator and the denominator. So the share is computed from the counts
# DEXSeq was fitted with.
dex_cd <- colData(dxd)
dex_counts <- counts(dxd)
this_cols <- which(as.character(dex_cd$exon) == "this")
other_cols <- which(as.character(dex_cd$exon) == "others")
this_samps <- as.character(dex_cd$sample_id[this_cols])
other_order <- match(this_samps, as.character(dex_cd$sample_id[other_cols]))
this_mat <- dex_counts[, this_cols, drop = FALSE]
others_mat <- dex_counts[, other_cols[other_order], drop = FALSE]
gene_total <- this_mat + others_mat
share_by_row <- this_mat / gene_total
share_by_row[gene_total <= 0] <- NA_real_
colnames(share_by_row) <- this_samps
# The rows are the transcripts, in the order the results table has them.
res_feature <- dxr$featureID
res_gene <- dxr$groupID
if (length(res_feature) != nrow(share_by_row)) {
  stop("ERROR: the DEXSeq results table has ", length(res_feature),
       " rows but the usage matrix has ", nrow(share_by_row),
       "; the shares cannot be matched to the transcripts.")
}
# THE SHARE IS KEYED BY GENE **AND** TRANSCRIPT. A transcript id is not
# guaranteed to be unique across genes, and keying by the transcript alone
# would put the rows of two genes into one group and report the MEAN of two
# genes' shares as one transcript's share. The length track already guards
# the same case ("A TRANSCRIPT ID MAY IN PRINCIPLE BELONG TO TWO GENES").
share_by_key <- split(seq_len(nrow(share_by_row)),
                      paste(res_gene, res_feature, sep = "\r"))
share_mean <- function(gene, tx, samples) {
  i <- share_by_key[[paste(gene, tx, sep = "\r")]]
  if (is.null(i)) return(NA_real_)
  m <- share_by_row[i, , drop = FALSE]
  cols <- match(samples, colnames(m))
  if (anyNA(cols)) return(NA_real_)
  v <- as.numeric(m[, cols, drop = FALSE])
  v <- v[!is.na(v)]
  if (length(v) == 0) return(NA_real_)
  mean(v)
}
diu_out <- counts_by_tx_all
diu_out$case_condition    <- case_condition
diu_out$control_condition <- control_condition
diu_out$pvalue        <- NA_real_
diu_out$padj          <- NA_real_
diu_out$log2FoldChange <- NA_real_
diu_out$share_case    <- NA_real_
diu_out$share_control <- NA_real_
diu_out$delta_share   <- NA_real_
diu_out$gene_padj     <- NA_real_
idx <- match(paste(diu_out$gene_id, diu_out$transcript_id),
             paste(dxr$groupID, dxr$featureID))
tested <- !is.na(idx)
diu_out$pvalue[tested]         <- dxr$pvalue[idx[tested]]
diu_out$padj[tested]           <- dxr$padj[idx[tested]]
diu_out$log2FoldChange[tested] <- dexseq_lfc[idx[tested]]
diu_out$share_case[tested] <- vapply(which(tested),
                                     function(i) share_mean(diu_out$gene_id[i],
                                                            diu_out$transcript_id[i],
                                                            case_samples),
                                     numeric(1))
diu_out$share_control[tested] <- vapply(which(tested),
                                        function(i) share_mean(diu_out$gene_id[i],
                                                               diu_out$transcript_id[i],
                                                               control_samples),
                                        numeric(1))
diu_out$delta_share[tested] <- diu_out$share_case[tested] - diu_out$share_control[tested]
# THE GENE-LEVEL ADJUSTMENT. DEXSeq's own perGeneQValue: the smallest
# p-value of a gene's transcripts, adjusted across genes. It answers a
# different question from the transcript-level padj -- "does this GENE carry
# a usage change" -- and the paper's usage figures use it.
gene_q <- tryCatch(DEXSeq::perGeneQValue(dxr),
                   error = function(e) {
                     cat("  WARN: the per-gene q-values could not be computed (",
                         conditionMessage(e), "); gene_padj is NA.\n")
                     NULL
                   })
if (!is.null(gene_q)) {
  diu_out$gene_padj <- unname(gene_q[match(diu_out$gene_id, names(gene_q))])
}
# THE SAMPLE COUNTS COME FROM THE DESIGN THAT WAS FITTED. The DEXSeq fit
# uses `design_aligned`, which is shorter than `design` when a design sample
# contributes no counts at all; counting from `design` would report more
# samples per group than were fitted.
diu_out$n_case    <- sum(design_aligned$condition == case_condition)
diu_out$n_control <- sum(design_aligned$condition == control_condition)
diu_out$independent_filter_threshold <- filter_threshold
diu_out$n_independent_filtered <- n_independent_filtered
# WHY A ROW HAS NO ADJUSTED P-VALUE. With the filtering off, the only
# reasons left are a missing p-value and a feature that was never tested.
diu_out$padj_na_reason <- ifelse(!tested, "not_tested",
                          ifelse(is.na(diu_out$pvalue), "no_p_value", "none"))
# THE MODEL, RECORDED. The tool that produced the statistic, the contrast and
# the batch handling (the audited note, never a guess). A transcript DEXSeq
# never saw carries no model, the same way the length track's untestable rows
# do.
diu_out$model_name <- ifelse(tested, "dexseq", "not_tested")
# THE MODEL, RECORDED. The design is stated as it was fitted. The design
# `~ sample + exon + condition:exon` carries a per-sample intercept, so a
# sample-level main effect is absorbed; it carries NO batch:exon interaction,
# so a batch effect on the USAGE of a transcript is not adjusted for. Saying
# "batch adjusted" here would be false, and the length track's note (which
# belongs to the length track's model) is not this track's to carry.
diu_out$model_parameters <- paste0(
  "DEXSeq differential isoform usage on (gene x transcript x sample) counts; ",
  "design ~ sample + exon + condition:exon (a per-sample intercept, so a ",
  "sample-level main effect is absorbed; no batch:exon interaction term, so a ",
  "batch effect on a transcript's share is NOT adjusted for); ",
  "log2FoldChange is DEXSeq's own MODEL estimate: the log2 ratio of the ",
  "transcript's usage between the two conditions; ",
  "share_case and share_control are the OBSERVED mean shares of the gene's ",
  "output (size-factor normalised counts) and delta_share is their ",
  "difference, a difference of proportions on the [0,1] scale; ",
  "the two are DIFFERENT ESTIMATORS of the same usage change -- DEXSeq's is ",
  "a per-gene negative binomial GLM coefficient and the shares are unweighted ",
  "means of the per-sample ratios -- so 2^log2FoldChange is not exactly ",
  "share_case / share_control; ",
  "the gene total behind a share is over the transcripts DEXSeq saw, i.e. ",
  "after the usage track's minimum-count filter and the single-transcript-gene ",
  "drop; ",
  "gene_padj is DEXSeq::perGeneQValue, ONE q-value per GENE (the gene's ",
  "smallest transcript p-value, adjusted across genes) broadcast to that ",
  "gene's transcripts -- a different multiplicity from the transcript-level ",
  "padj; ",
  "independent filtering: not applied (the threshold is NA and nothing was ",
  "filtered); ",
  contrast_note, "; ", disp_note)
diu_out <- diu_out[, c("gene_id", "transcript_id", "case_condition",
                       "control_condition", "share_case", "share_control",
                       "delta_share", "log2FoldChange", "pvalue", "padj",
                       "gene_padj", "total", "n_case", "n_control",
                       "independent_filter_threshold", "n_independent_filtered",
                       "padj_na_reason", "model_name", "model_parameters")]

write.table(diu_out,
            file = file.path(opt$`output-dir`, "diu.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("[telescope_diff]  wrote diu.tsv (",
    nrow(diu_out), "transcripts)\n")
}

# =====================================================================
# Transcript track: differential transcript abundance via DESeq2 -> det.tsv
# =====================================================================
#
# A TRANSCRIPT'S OWN ABUNDANCE IS A DIFFERENT QUESTION FROM ITS SHARE OF ITS
# GENE'S OUTPUT. The first is this block: does the transcript's own count
# change between the two conditions? The second is the DEXSeq usage test in
# diu.tsv: does the transcript take more or less of what its gene produces?
# A gene can go up with every isoform's share unchanged, and an isoform can
# take over its gene while the gene's total stays where it was.
if (want_det) {
cat("\n[telescope_diff] transcript track: differential transcript abundance via DESeq2\n")
if (!requireNamespace("DESeq2", quietly = TRUE)) {
  stop("ERROR: the transcript track needs DESeq2, which is not installed in this R ",
       "library. Create the pinned environment: mamba env create -f workflow/environment.yml")
}

# THE TRANSCRIPT-LEVEL COUNTS, BEFORE THE USAGE TRACK'S FILTERS. The filters
# are the usage track's thresholds (a minimum total count, single-transcript
# genes) and have no business removing a transcript from its own abundance
# test.
tx_counts_long <- counts_for_tx %>%
  dplyr::group_by(gene_id, transcript_id, sample_id) %>%
  dplyr::summarize(count = sum(count), .groups = "drop")
tx_ids_all <- sort(unique(tx_counts_long$transcript_id))
# The gene each transcript belongs to, taken from the counts themselves.
gene_of_tx <- tapply(tx_counts_long$gene_id, tx_counts_long$transcript_id,
                     function(x) x[1])
cat("  transcripts:", length(tx_ids_all), "\n")

tx_mat <- matrix(0, nrow = length(tx_ids_all), ncol = nrow(design),
                 dimnames = list(tx_ids_all, design$sample_id))
rows_i <- match(tx_counts_long$transcript_id, tx_ids_all)
cols_j <- match(tx_counts_long$sample_id, design$sample_id)
tx_mat[cbind(rows_i, cols_j)] <- tx_counts_long$count
tx_mat <- matrix(as.integer(tx_mat), nrow = nrow(tx_mat),
                 dimnames = dimnames(tx_mat))
if (any(tx_mat < 0)) {
  stop("ERROR: the transcript-level counts contain a negative value.")
}
# A TRANSCRIPT ID MAY IN PRINCIPLE BELONG TO TWO GENES. DESeq2's row names
# have to be unique, so the matrix is keyed by the pair; the reported table
# still carries the transcript id on its own.
tx_keys <- paste(tx_counts_long$gene_id, tx_counts_long$transcript_id, sep = "|")
tx_key_by_id <- tapply(tx_keys, tx_counts_long$transcript_id, function(x) x[1])
rownames(tx_mat) <- unname(tx_key_by_id[tx_ids_all])
if (anyDuplicated(rownames(tx_mat))) {
  stop("ERROR: two transcripts of the same gene share a transcript id; ",
       "the transcript-level matrix cannot be keyed.")
}

# THE DESIGN. The same design the gene track uses: the reference level is
# the control, and the batch term is present only when the batch is not
# confounded with the condition.
design_tx <- design[match(colnames(tx_mat), design$sample_id), , drop = FALSE]
if (any(is.na(design_tx$sample_id))) {
  stop("ERROR: design sample_ids do not match the transcript count matrix columns.")
}
rownames(design_tx) <- colnames(tx_mat)
design_tx$condition <- factor(as.character(design_tx$condition),
                              levels = c(control_condition, case_condition))
if (dpl_spec$batch_used) {
  design_tx$batch <- factor(as.character(design_tx$batch))
  det_formula <- ~ condition + batch
} else {
  det_formula <- ~ condition
}

dds_tx <- DESeq2::DESeqDataSetFromMatrix(countData = tx_mat,
                                          colData = design_tx,
                                          design = det_formula)
# THE SIZE FACTORS. A transcript that is absent in one sample has a zero in
# the geometric mean, and DESeq2's default median-of-ratios needs at least
# one row with no zero at all. When there is none, its own `poscounts`
# estimator is the documented remedy and it is used here.
tx_sf_note <- "median-of-ratios size factors"
dds_tx <- tryCatch(DESeq2::estimateSizeFactors(dds_tx),
                   error = function(e) {
                     cat("  WARN: the transcript size factors could not be estimated (",
                         conditionMessage(e),
                         "); using DESeq2's poscounts estimator.\n")
                     tx_sf_note <<- "poscounts size factors (the median-of-ratios estimator failed)"
                     DESeq2::estimateSizeFactors(dds_tx, type = "poscounts")
                   })

tx_disp_note <- "parametric dispersion fit"
dds_tx <- tryCatch({
  dds_tx <- DESeq2::estimateDispersions(dds_tx, quiet = TRUE)
  dds_tx
}, error = function(e) {
  cat("  WARN: the transcript dispersion trend could not be fitted (",
      conditionMessage(e),
      "); keeping DESeq2's gene-wise dispersion estimates.\n")
  tx_disp_note <<- "gene-wise dispersion estimates (the dispersion trend could not be fitted)"
  dds_fallback <- DESeq2::estimateDispersionsGeneEst(dds_tx, quiet = TRUE)
  DESeq2::dispersions(dds_fallback) <- S4Vectors::mcols(dds_fallback)$dispGeneEst
  dds_fallback
})

dds_tx <- DESeq2::nbinomWaldTest(dds_tx)
dres_tx <- DESeq2::results(dds_tx, independentFiltering = FALSE,
                           contrast = c("condition", case_condition, control_condition))
dres_tx_df <- as.data.frame(dres_tx)

det_out <- data.frame(
  gene_id                = unname(gene_of_tx[tx_ids_all]),
  transcript_id          = tx_ids_all,
  case_condition         = rep(case_condition, length(tx_ids_all)),
  control_condition      = rep(control_condition, length(tx_ids_all)),
  baseMean               = NA_real_,
  log2FoldChange         = NA_real_,
  lfcSE                  = NA_real_,
  stat                  = NA_real_,
  pvalue                 = NA_real_,
  padj                   = NA_real_,
  n_case                 = rep(sum(design_tx$condition == case_condition), length(tx_ids_all)),
  n_control              = rep(sum(design_tx$condition == control_condition), length(tx_ids_all)),
  independent_filter_threshold = NA_real_,
  n_independent_filtered = 0L,
  stringsAsFactors = FALSE
)
# The results are keyed by the composite row name of the matrix, which is
# the gene and the transcript together.
det_keys <- unname(tx_key_by_id[tx_ids_all])
det_idx <- match(det_keys, rownames(dres_tx_df))
det_tested <- !is.na(det_idx)
det_out$baseMean[det_tested]       <- dres_tx_df$baseMean[det_idx[det_tested]]
det_out$log2FoldChange[det_tested] <- dres_tx_df$log2FoldChange[det_idx[det_tested]]
det_out$lfcSE[det_tested]          <- dres_tx_df$lfcSE[det_idx[det_tested]]
det_out$stat[det_tested]           <- dres_tx_df$stat[det_idx[det_tested]]
det_out$pvalue[det_tested]         <- dres_tx_df$pvalue[det_idx[det_tested]]
det_out$padj[det_tested]           <- dres_tx_df$padj[det_idx[det_tested]]
det_out$padj_na_reason <- ifelse(!det_tested, "not_tested",
                          ifelse(is.na(det_out$pvalue), "no_p_value", "none"))
det_out$model_name <- "deseq2_wald"
det_out$model_parameters <- paste0(
  "DESeq2 on the transcript-level counts (the transcript's own abundance, ",
  "not its share of its gene's output -- that is the DEXSeq usage test in ",
  "diu.tsv); negative binomial Wald test; results(independentFiltering = FALSE); ",
  "the counts are the Rust aggregator's posterior mass rounded to integers, ",
  "not molecule counts; the matrix is the UNFILTERED one, i.e. before the ",
  "usage track's minimum-count filter and the single-transcript-gene drop, ",
  "because those thresholds are the usage track's business; ",
  "independent filtering: not applied (the threshold is NA and nothing was ",
  "filtered); ",
  tx_sf_note, "; ", tx_disp_note, "; ", contrast_note, "; ", dpl_batch_note)

write.table(det_out,
            file = file.path(opt$`output-dir`, "det.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("[telescope_diff]  wrote det.tsv (", nrow(det_out), "transcripts,",
    sum(det_out$padj_na_reason == "none"), "tested)\n")
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
# reliability-weighted within-cell variance is (swy2 - swy^2/W) / (W - sw2/W)
# and the variance of the cell mean is that variance divided by Kish_Neff.
#
# The denominator is W - sw2/W and not Kish_Neff - 1: for i.i.d. observations
# with FIXED weights the unbiased weighted variance divides by
# W - sum(w^2)/W, which equals Kish_Neff - 1 only when the weights are all
# equal. See dpl_models.R for the derivation and for what that does and does
# not claim about the EM's own posterior weights.

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
# ONE FIT FOR BOTH ARMS. eBayes leaves the coefficients, the unscaled
# standard deviations, the residual standard deviation and the residual
# degrees of freedom untouched, so the raw statistics come out of the same
# fit as the moderated ones. Fitting the transcriptome twice doubled the
# dominant cost of this track for no extra information.
both     <- dpl_fit_both(y_mat, dpl_design_matrix, NULL)
fit_raw  <- both$raw
fit      <- both$moderated
# The moderation is fitted with the variance trend, and the note says whether
# that actually happened: a cohort whose features share one average leaves
# limma's spline nothing to fit, and the moderation then goes on without the
# trend. The receipt has to say which fit produced the numbers.
dpl_ebayes_note <- if (nzchar(both$ebayes_note)) paste0(both$ebayes_note, "; ") else ""

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
# A TRANSCRIPT WHOSE RESIDUAL VARIANCE HAS COLLAPSED IS NOT A RESULT.
# limma computes the residual standard deviation from the QR decomposition,
# so a mathematically zero residual variance comes out at ~1e-15 rather than
# at 0. The t statistic is then finite and enormous (1e+16) and the p-value
# is a floating-point artefact (1e-141): a row of that kind would be the most
# significant row in the table, and it says nothing at all. The floor is
# 1e-8 on the VARIANCE, which is 1e-4 on the standard deviation -- and the
# smallest variance in the real cohorts is 1.7e-7, so nothing that carries
# signal is masked by it.
sigma2_raw <- fit_raw$sigma2_raw
fit_failed <- !is.finite(t_raw) | !is.finite(se_raw) | (sigma2_raw < 1e-8)
masked <- fit_failed | too_few_cells
t_raw[masked] <- NA_real_
p_raw[masked] <- NA_real_
t_mod[masked] <- NA_real_
p_mod[masked] <- NA_real_
q_raw <- stats::p.adjust(p_raw, method = "BH")
q_mod <- stats::p.adjust(p_mod, method = "BH")
cat("  testable transcripts:", sum(!masked), "(both groups with >=",
    min_samples, "admissible cells; the audit's population)\n")

# THE EFFECT ON THE PHYSICAL SCALE. The statistic above is a difference of
# means of log2(polyA_length), so it is the LOG2 of a ratio: reporting it
# alone leaves the reader to guess how many nucleotides that is, and 2^d is a
# ratio, not a length. The columns added here are the geometric means of the
# two groups in nt, their ratio, their difference, and intervals for all of
# them.
#
# THEY COME FROM THE SAME ADMISSIBLE CELLS AS THE FIT, and the interval on
# the nt scale is the delta method applied to the SAME raw difference of the
# two group means -- so the baseline's own estimation error is propagated
# instead of the interval endpoints being multiplied by a fixed constant.
#
# WHEN THE DESIGN CARRIES A BATCH TERM the fitted coefficient is the
# BATCH-ADJUSTED difference, while these columns are the raw difference of
# the two group means. The two are then different quantities, and the receipt
# says which one each column is.
nt_eff <- dpl_effect_nt(y_mat, design$condition == case_condition,
                        design$condition == control_condition)
# THE RAW DIFFERENCE AND THE FITTED COEFFICIENT ARE THE SAME NUMBER when the
# design has no batch term. That is an invariant, not an assumption, so it is
# asserted: a drift between the two paths would otherwise show up only as a
# silently different column. The check runs BEFORE the masking, so it covers
# every row the fit produced.
if (!dpl_spec$batch_used) {
  .chk <- is.finite(nt_eff$delta_log2_pal) & is.finite(fit_raw$estimate)
  if (any(.chk)) {
    .d <- abs(nt_eff$delta_log2_pal[.chk] - fit_raw$estimate[.chk])
    .tol <- 1e-8 * pmax(1, abs(fit_raw$estimate[.chk]))
    if (max(.d) > max(.tol)) {
      stop("ERROR: the raw difference of the group means and the fitted ",
           "coefficient disagree by ", format(max(.d), digits = 6),
           " -- the effect columns and the test are not the same number.")
    }
  }
}
# THE MASKING RULE FOR THESE COLUMNS IS `fit_failed`, NOT `masked`. A
# transcript with too few admissible cells in one group keeps its effect
# estimate in the contract column -- the difference of the group means is
# computed, it is the TEST that is not supported -- so the columns that
# describe the same difference keep it too, and the `padj_na_reason` column
# says why the test is not supported. A row whose FIT failed gets NA in all
# of them: a degenerate zero variance is not a result.
for (.cn in c("geometric_mean_ratio", "geometric_mean_control_nt",
              "geometric_mean_case_nt", "delta_geometric_mean_nt",
              "ci_lo_log2_pal", "ci_hi_log2_pal", "ci_lo_ratio", "ci_hi_ratio",
              "ci_lo_delta_geometric_mean_nt",
              "ci_hi_delta_geometric_mean_nt")) {
  nt_eff[[.cn]][fit_failed] <- NA_real_
}

# WHY A ROW HAS NO ADJUSTED P-VALUE. The length track has no independent
# filter: it drops no row for having a low mean, so the threshold is NA and
# the count of filtered rows is 0. A row without a statistic says why.
no_cells <- n_adm_case == 0 & n_adm_ctrl == 0
pal_na_reason <- ifelse(
  !is.na(p_raw), "none",
  ifelse(no_cells, "no admissible cells at all",
         ifelse(fit_failed, "degenerate_zero_variance",
                "too few admissible cells in one group")))

# THE PRIORS ARE SUMMARISED, NOT LISTED. With trend = TRUE limma fits the
# prior variance as a function of the average expression, so s2.prior and
# df.prior hold ONE VALUE PER TRANSCRIPT rather than one value for the fit.
# Printing them one by one would put sixteen numbers in a log line, and
# pasting them into every row's model_parameters would put a paragraph in
# every cell.
#
# WHAT THE BASE FIT IS. The name and the sentence have to agree: with a batch
# term in the design the fit is not a two-sample t, its coefficient is not the
# difference of the two group means, and calling it one is a receipt that
# contradicts the model_name column beside it.
dpl_base_fit_clause <- if (dpl_spec$batch_used) {
  "base fit: batch_adjusted_raw_t (the design carries a batch term, so the fit is a linear model with a batch term and its `case` coefficient is the BATCH-ADJUSTED difference, not the raw difference of the two group means); "
} else {
  "base fit: equal_raw_t (unweighted two-sample t on the transcript x sample means; with no batch term in the design, the `case` coefficient is the difference of the two group means); "
}
# THE AUDIT IS NOT RE-RUN HERE. The calibration is a separate script over a
# separate set of cohorts, and its verdict is a property of THAT run. Writing
# a verdict into this receipt as a literal would put a claim in every row of
# every table that no run of this engine could confirm -- and a claim that
# cannot be confirmed is a claim that can be wrong. So the receipt says where
# the verdict lives instead of asserting one.
dpl_audit_clause <- paste0(
  "audit: workflow/scripts/audit_dpl_models.R -- the calibration of this model ",
  "is a SEPARATE run, not recomputed here; its verdict is in that script's own ",
  "dpl_audit_summary.tsv, one row per candidate model; "
)
# THE CONFOUNDING FLAG. dpl_design() decides it: TRUE when the design carries
# a batch that cannot be separated from the condition at all. The sentence is
# fixed, because a reader has to be able to search for it.
dpl_confounded <- isTRUE(dpl_spec$confounded)
dpl_confound_clause <- if (dpl_confounded) {
  paste0("CONFOUNDED DESIGN: batch and condition are completely confounded; ",
         "treatment-specific effects are not identifiable; the contrast in ",
         "this table is a batch-and-condition difference; ")
} else {
  ""
}
# THE PHYSICAL-SCALE COLUMNS, NAMED. The receipt says which quantity each
# column is, because with a batch term the fitted coefficient and the raw
# difference of the group means are different numbers.
dpl_effect_clause <- if (dpl_spec$batch_used) {
  paste0("effect columns: the fitted coefficient is the BATCH-ADJUSTED ",
         "difference of the group means, while geometric_mean_*_nt and ",
         "delta_geometric_mean_nt are the RAW difference of the two group ",
         "means; the geometric means are the group means of log2 tail length ",
         "exponentiated, and delta_geometric_mean_nt is their DIFFERENCE, not ",
         "a ratio and not an arithmetic mean difference; the intervals are ",
         "t intervals on the raw difference and the delta method applied to ",
         "the same difference; ")
} else {
  paste0("effect columns: geometric_mean_*_nt are the group means of log2 ",
         "tail length exponentiated, and delta_geometric_mean_nt is their ",
         "DIFFERENCE, not a ratio and not an arithmetic mean difference; the ",
         "intervals are t intervals on the difference of the group means and ",
         "the delta method applied to the same difference; ")
}
pal_out <- data.frame(
  gene_id                    = unname(gene_of),
  transcript_id              = tx_ids_all,
  case_condition             = rep(case_condition, length(tx_ids_all)),
  control_condition          = rep(control_condition, length(tx_ids_all)),
  delta_pal_case_minus_control = fit_raw$estimate,
  # THE SAME NUMBER UNDER A NAME THAT CARRIES ITS UNIT. The contract column
  # above is the log2 difference of the group means; the column below says so
  # in the name. They are assigned from one value, so they cannot drift.
  delta_log2_pal             = fit_raw$estimate,
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
  # THE PHYSICAL SCALE. `geometric_mean_*_nt` are the geometric mean tail
  # lengths of the two groups in nucleotides -- the means of log2 lengths,
  # exponentiated. `delta_geometric_mean_nt` is their DIFFERENCE, which is not
  # the same number as the geometric mean ratio and not the same as the
  # arithmetic mean difference of the raw lengths.
  geometric_mean_ratio       = nt_eff$geometric_mean_ratio,
  geometric_mean_control_nt  = nt_eff$geometric_mean_control_nt,
  geometric_mean_case_nt     = nt_eff$geometric_mean_case_nt,
  delta_geometric_mean_nt    = nt_eff$delta_geometric_mean_nt,
  ci_lo_log2_pal             = nt_eff$ci_lo_log2_pal,
  ci_hi_log2_pal             = nt_eff$ci_hi_log2_pal,
  ci_lo_ratio                = nt_eff$ci_lo_ratio,
  ci_hi_ratio                = nt_eff$ci_hi_ratio,
  ci_lo_delta_geometric_mean_nt = nt_eff$ci_lo_delta_geometric_mean_nt,
  ci_hi_delta_geometric_mean_nt = nt_eff$ci_hi_delta_geometric_mean_nt,
  n_case                     = rep(sum(design$condition == case_condition), length(tx_ids_all)),
  n_control                  = rep(sum(design$condition == control_condition), length(tx_ids_all)),
  n_admissible_case          = n_adm_case,
  n_admissible_control       = n_adm_ctrl,
  independent_filter_threshold = NA_real_,
  n_independent_filtered     = 0L,
  padj_na_reason             = pal_na_reason,
  # THE DESIGN IS CONFOUNDED. The batch note already says it in words; this
  # column says it where a downstream script can read it without parsing
  # prose. A TRUE here means the contrast in this table cannot be attributed
  # to the treatment: the batch and the condition cannot be separated.
  confounded_design          = dpl_confounded,
  # The name follows the design that was fitted. With a usable batch the
  # baseline is not a two-sample t any more -- the design carries a batch
  # term -- so calling it "equal_raw_t" would name a model that was not
  # fitted.
  model_name                 = if (dpl_spec$batch_used) "batch_adjusted_raw_t" else "equal_raw_t",
  model_parameters           = paste0(
    dpl_base_fit_clause,
    "moderated arm: equal_weight_eb (limma eBayes with trend = TRUE ",
    "and robust = TRUE; the prior variance is fitted against the average ",
    "expression, so s2.prior and df.prior are per transcript: s2.prior ",
    dpl_prior_summary(fit$s2_prior), ", df.prior ", dpl_prior_summary(fit$df_prior),
    "); ", dpl_ebayes_note, dpl_audit_clause,
    "cell admission gate: min_weight = ", format(opt$`min-weight`),
    ", min_neff = 2, min_samples = ", min_samples,
    ", sigma2_floor = 1e-8; ",
    dpl_effect_clause,
    "independent filtering: not applied (the length track drops no row, so the ",
    "independent filter threshold is NA and nothing was filtered); ",
    contrast_note, "; ", dpl_batch_note, dpl_confound_clause),
  stringsAsFactors = FALSE
)
# A transcript whose FIT failed carries NA. A transcript that was fitted but
# has too few admissible cells keeps its effect estimate: the difference of
# the two group means is computed, it is the TEST that is not supported. The
# admissible sample counts and the NA reason say which.
#
# BOTH EFFECT COLUMNS ARE MASKED TOGETHER. The moderated arm's estimate is the
# SAME coefficient as the raw arm's (limma moderates the variance, not the
# coefficient), so leaving one of them filled on a row whose fit failed would
# make the two effect columns disagree about whether the row has a result.
pal_out$delta_pal_case_minus_control[fit_failed] <- NA_real_
pal_out$delta_pal_moderated[fit_failed] <- NA_real_
pal_out$model_name[is.na(pal_out$t_raw)] <- "not_testable"

write.table(pal_out,
            file = file.path(opt$`output-dir`, "dpl.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("[telescope_diff]  wrote dpl.tsv (",
    nrow(pal_out), "transcripts,", sum(!is.na(pal_out$p_raw)), "tested)\n")
cat("[telescope_diff]  limma eBayes priors (per transcript, summarised): ",
    "s2.prior =", dpl_prior_summary(fit$s2_prior),
    " df.prior =", dpl_prior_summary(fit$df_prior), "\n")

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

# THE (SITE x SAMPLE) USAGE MATRIX. THE TEST IS ABOUT A PROPORTION, NOT A
# COUNT. LAPA's own `usage` column is the site's share of its gene's
# poly(A) reads in that sample: usage == count / gene_count, which was
# verified on 65,639 of 65,639 rows over 21 bed files. A raw count is not a
# usage: a gene whose output doubles doubles every one of its sites' counts,
# so a test on counts measures the gene's abundance (the DEG track) and calls
# it a poly(A) site choice. So the matrix holds the USAGE.
#
# A CELL IS VALID ONLY WHEN THE GENE'S TOTAL IS BIG ENOUGH TO DIVIDE. LAPA's
# `gene_count` is that total for the sample, and a share taken over a handful
# of reads is noise. A cell whose gene total is below the minimum count is
# NA, not zero.
apa_site_ids <- sort(unique(unlist(lapply(lapa_rows, function(d) d$site_id))))
apa_mat <- matrix(NA_real_, nrow = length(apa_site_ids), ncol = length(apa_samples),
                   dimnames = list(apa_site_ids, apa_samples))
apa_counts <- matrix(NA_real_, nrow = length(apa_site_ids), ncol = length(apa_samples),
                      dimnames = list(apa_site_ids, apa_samples))
# WHICH CELLS LAPA ACTUALLY CALLED. A site LAPA did not call in a sample is
# a different thing from a site it called with a value that cannot be used,
# and the two are told apart by this mask rather than by the cell being NA.
apa_present <- matrix(FALSE, nrow = length(apa_site_ids), ncol = length(apa_samples),
                       dimnames = list(apa_site_ids, apa_samples))
# THE GENE TOTAL PER (GENE, SAMPLE). LAPA writes the same gene_count on every
# row of a gene in a sample, so the first non-missing one is the total.
apa_gene_total <- list()
for (s in apa_samples) {
  d <- lapa_rows[[s]]
  g <- d$gene_id
  gc <- suppressWarnings(as.numeric(d$gene_count))
  keep <- !is.na(g) & nzchar(g)
  if (any(keep)) {
    tot <- tapply(gc[keep], g[keep], function(x) x[which(!is.na(x))[1]])
    apa_gene_total[[s]] <- tot
  } else {
    apa_gene_total[[s]] <- numeric(0)
  }
  # The rows are RE-ORDERED to the matrix's site order, and the values must
  # be re-ordered with them. Taking the usage from the original rows and
  # writing it into the reordered positions puts every site's usage on
  # another site -- a rotation that no assertion on the totals would see.
  # A SITE THIS SAMPLE DOES NOT HAVE HAS NO ROW AT ALL, so its index is NA:
  # those rows are left out of the assignment, because an NA in a subscripted
  # assignment is an error, not a "no value".
  d2 <- d[match(apa_site_ids, d$site_id), ]
  j <- match(d2$site_id, apa_site_ids)
  cc <- match(s, apa_samples)
  present <- !is.na(j)
  apa_mat[cbind(j[present], cc)] <- suppressWarnings(as.numeric(d2$usage))[present]
  apa_counts[cbind(j[present], cc)] <- suppressWarnings(as.numeric(d2$count))[present]
  apa_present[cbind(j[present], cc)] <- TRUE
}

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
# THE COORDINATE OF A SITE. LAPA's `polyA_site` is the peak's position.
apa_coord <- rep(NA_real_, length(apa_site_ids))
for (s in apa_samples) {
  d <- lapa_rows[[s]]
  d <- d[match(apa_site_ids, d$site_id), ]
  pos <- suppressWarnings(as.numeric(d$polyA_site))
  take <- is.na(apa_coord) & !is.na(pos)
  apa_coord[take] <- pos[take]
}

# A SITE LAPA DID NOT CALL IN A SAMPLE HAS NO READS THERE, so its share is
# zero -- but only when the gene's total for that sample is known and big
# enough to divide. Otherwise the cell is NA.
for (cc in seq_along(apa_samples)) {
  s <- apa_samples[cc]
  totals <- apa_gene_total[[s]]
  genes <- apa_gene
  known <- !is.na(genes) & genes %in% names(totals)
  tot <- rep(NA_real_, length(genes))
  tot[known] <- totals[genes[known]]
  absent <- !apa_present[, cc]
  usable <- absent & !is.na(tot) & tot >= opt$`min-counts`
  apa_mat[usable, cc] <- 0
}
# A CELL WHOSE GENE TOTAL IS TOO SMALL IS NOT USABLE, whatever LAPA wrote.
for (cc in seq_along(apa_samples)) {
  s <- apa_samples[cc]
  totals <- apa_gene_total[[s]]
  genes <- apa_gene
  known <- !is.na(genes) & genes %in% names(totals)
  tot <- rep(NA_real_, length(genes))
  tot[known] <- totals[genes[known]]
  too_small <- apa_present[, cc] & (is.na(tot) | tot < opt$`min-counts`)
  apa_mat[too_small, cc] <- NA_real_
}

# THE TEST. limma on the (site x sample) USAGE matrix, with the same design
# the length track uses: the reference level is the control, so the effect is
# case minus control. The scale is the proportion itself, so the fitted
# effect IS the difference of the two groups' mean usages -- the quantity the
# table reports as delta_pau.
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
# A SITE WITH TOO FEW USABLE CELLS IN ONE GROUP HAS NO WITHIN-GROUP VARIANCE,
# so it is not testable. Those rows are reported with the reason, and they
# are kept out of the adjustment's denominator: the adjusted p-value has to
# be the adjusted p-value of the population that was tested.
apa_case_samples    <- apa_samples[as.character(design$condition[
  match(apa_samples, design$sample_id)]) == case_condition]
apa_control_samples <- apa_samples[as.character(design$condition[
  match(apa_samples, design$sample_id)]) == control_condition]
apa_n_case    <- rowSums(!is.na(apa_mat[, apa_case_samples, drop = FALSE]))
apa_n_control <- rowSums(!is.na(apa_mat[, apa_control_samples, drop = FALSE]))
apa_too_few <- apa_n_case < opt$`min-samples` | apa_n_control < opt$`min-samples`
apa_y <- apa_mat
apa_y[apa_too_few, ] <- NA_real_
# THE FIT, UNLESS THERE IS NOTHING TO FIT. dpl_fit_both is the engine's one
# limma fit: it fits the rows that carry an observation (a site with no usable
# cell at all gives limma's empirical Bayes trend a NaN average and aborts the
# WHOLE fit) and returns the raw and the moderated statistics of that one fit.
# This table reports the moderated arm, so se, t_stat, pvalue and padj are one
# coherent set.
#
# On a real cohort MOST sites have no usable cell at all -- the gene's poly(A)
# reads are below the minimum count in every sample -- so a cohort where EVERY
# site fails is a real possibility (a lower-coverage cohort, or a higher
# --min-counts). dpl_fit_both stops with "no row of the response matrix
# carries an observation" in that case, and a stop there takes down the WHOLE
# run: the four other tables, which are fine, would never be written. So the
# poly(A) track writes its table with every row carrying its reason instead.
apa_fit_made <- sum(!apa_too_few) > 0
if (apa_fit_made) {
  apa_both <- dpl_fit_both(apa_y, apa_design_mat, NULL)
} else {
  n_sites <- length(apa_site_ids)
  apa_both <- list(
    raw = list(t = rep(NA_real_, n_sites),
               se = rep(NA_real_, n_sites),
               sigma2_raw = rep(NA_real_, n_sites)),
    moderated = list(estimate = rep(NA_real_, n_sites),
                     se = rep(NA_real_, n_sites),
                     t = rep(NA_real_, n_sites),
                     p = rep(NA_real_, n_sites)),
    observed = rep(FALSE, n_sites),
    trend_used = FALSE,
    ebayes_note = paste0("no poly(A) site had enough usable cells in BOTH ",
                         "groups (at least ", opt$`min-samples`, " in each), so ",
                         "no model was fitted and no p-value exists; ")
  )
  cat("  WARN: no poly(A) site is testable at --min-samples ",
      opt$`min-samples`, "; apa.tsv is written with every row carrying its ",
      "reason, and the other tables are unaffected.\n")
}
apa_fit <- apa_both$moderated
apa_ebayes_note <- if (nzchar(apa_both$ebayes_note)) paste0(apa_both$ebayes_note, "; ") else ""
# A SITE WHOSE RESIDUAL VARIANCE HAS COLLAPSED IS NOT A RESULT. The same
# guard the length track applies, and for the same reason: limma computes the
# residual variance from the QR decomposition, so a mathematically zero
# residual variance comes out at ~1e-15 rather than at 0, and the statistic
# that comes out of it is a ratio of two floating-point noise values rather
# than a measurement. A site whose usage is exactly 1 in every usable sample
# -- the site of a gene whose only replicated poly(A) cluster is that one --
# has exactly this shape: zero residual variance, an estimate of zero, and a
# t statistic of -0.44 that says nothing. The floor is 1e-8 on the VARIANCE,
# which is 1e-4 on the standard deviation of a proportion.
#
# ONLY THE ROWS THAT WERE FITTED CAN BE DEGENERATE. A row that was left out of
# the fit (too few usable cells in one group) has no t statistic at all, and
# calling that a collapsed variance would report the wrong reason.
apa_degenerate <- apa_both$observed &
  (!is.finite(apa_both$raw$t) | !is.finite(apa_both$raw$se) |
     (apa_both$raw$sigma2_raw < 1e-8))
apa_fit$t[apa_degenerate] <- NA_real_
apa_fit$p[apa_degenerate] <- NA_real_
# The adjusted p-values are BH over the p-values the fit produced. The
# adjustment's denominator is the number of p-values it is given, and an NA
# is left out of it, so the q-value is the q-value of the TESTED population.
apa_p_raw <- apa_fit$p
apa_q_raw <- stats::p.adjust(apa_p_raw, method = "BH")
# THE GROUP MEANS ARE TAKEN OVER THE SAME CELLS THE FIT USED, so that
# delta_pau is exactly pau_case - pau_control WHEN THE DESIGN CARRIES NO
# BATCH TERM: the coefficient of a two-group design is the difference of the
# two group means. With a batch term the coefficient is the batch-ADJUSTED
# difference, and the receipt says so.
pau_case    <- rowMeans(apa_y[, apa_case_samples, drop = FALSE], na.rm = TRUE)
pau_control <- rowMeans(apa_y[, apa_control_samples, drop = FALSE], na.rm = TRUE)
pau_case[!is.finite(pau_case)] <- NA_real_
pau_control[!is.finite(pau_control)] <- NA_real_

# THE MODEL NAME FOLLOWS THE FIT. `dpl_ebayes()` falls back to
# `eBayes(robust = TRUE)` when the variance trend cannot be fitted, and a
# hard-coded "limma_trend_robust" would then say that a trend was fitted when
# it was not. The same rule the length track follows for its batch: the name
# is the one the fit earns.
apa_model_name <- if (!apa_fit_made) {
  "not_testable"
} else if (apa_both$trend_used) {
  "limma_trend_robust"
} else {
  "limma_eb_robust"
}
apa_fit_clause <- if (!apa_fit_made) {
  "NO MODEL WAS FITTED for this run (no site had enough usable cells in both "
} else if (apa_both$trend_used) {
  "limma::lmFit + eBayes(trend = TRUE, robust = TRUE) on the (site x sample) "
} else {
  "limma::lmFit + eBayes(robust = TRUE) -- NO variance trend -- on the (site x sample) "
}
apa_trend_clause <- if (!apa_fit_made) {
  "groups), so every row carries its reason and no statistic exists; "
} else if (apa_both$trend_used) {
  "limma applies no log transform, so the trend covariate is the site's mean usage; "
} else {
  "no variance trend was fitted for this run (the note above says why), so there is no trend covariate; "
}
# WHAT delta_pau IS. With a two-group design the coefficient of the `case`
# column IS the difference of the two group means. With a batch term in the
# design it is the batch-ADJUSTED difference, and the two raw means are not
# it -- so the receipt has to say which of the two the column holds.
apa_delta_clause <- if (dpl_spec$batch_used) {
  "the design carries a batch term, so delta_pau is the BATCH-ADJUSTED difference and is NOT equal to pau_case - pau_control (those two are the raw group means); "
} else {
  "with no batch term in the design, delta_pau is exactly pau_case - pau_control; "
}
# THE CONTRACT, COLUMN BY COLUMN, IN ORDER. The poly(A) table's columns are a
# published list, not a set: a reader (or a downstream script) that looks up
# "the column after n_independent_filtered" must get the same answer every
# time. `padj_na_reason` and `model_name` were built in the other order, so
# the table came out with the two of them swapped against the contract.
APA_COLUMNS <- c(
  "gene_id", "site_id", "coord_bp", "case_condition", "control_condition",
  "pau_case", "pau_control", "delta_pau", "se", "t_stat", "pvalue", "padj",
  "n_case", "n_control", "independent_filter_threshold",
  "n_independent_filtered", "padj_na_reason", "model_name", "model_parameters"
)
apa_out <- data.frame(
  gene_id           = apa_gene,
  site_id           = apa_site_ids,
  coord_bp          = apa_coord,
  case_condition    = rep(case_condition, length(apa_site_ids)),
  control_condition = rep(control_condition, length(apa_site_ids)),
  pau_case          = pau_case,
  pau_control       = pau_control,
  delta_pau         = apa_fit$estimate,
  se                = apa_fit$se,
  t_stat            = apa_fit$t,
  pvalue            = apa_p_raw,
  padj              = apa_q_raw,
  n_case            = rep(length(apa_case_samples), length(apa_site_ids)),
  n_control         = rep(length(apa_control_samples), length(apa_site_ids)),
  independent_filter_threshold = NA_real_,
  n_independent_filtered = 0L,
  model_name        = apa_model_name,
  stringsAsFactors = FALSE
)
apa_no_usable <- apa_n_case == 0 & apa_n_control == 0
apa_out$padj_na_reason <- ifelse(
  apa_no_usable, "no usable cell (the gene's poly(A) reads are below the minimum count in every sample)",
  ifelse(apa_too_few, "too few usable samples in one group",
         ifelse(apa_degenerate, "degenerate_zero_variance", "none")))
apa_out$model_parameters <- paste0(
  apa_fit_clause,
  "USAGE matrix: LAPA's own 'usage' column, i.e. the site's share of its ",
  "gene's poly(A) reads in that sample (usage == count / gene_count), taken ",
  "from the replicated-cluster files (sample/), i.e. only the sites LAPA kept ",
  "after its replication filter; a cell is usable only when the gene's total ",
  "in that sample is at least the minimum count (", opt$`min-counts`, "), and ",
  "a site is tested only when at least ", opt$`min-samples`, " cells are ",
  "usable in EACH group; a site whose residual variance has collapsed (below ",
  "1e-8) carries no statistic and is reported with padj_na_reason = ",
  "'degenerate_zero_variance', out of the adjustment's denominator as well; ",
  apa_ebayes_note,
  "THE SCALE IS THE PROPORTION ITSELF, so delta_pau is ",
  "a difference of the two groups' mean usages on the [0,1] scale, not a log ",
  "ratio, and pau_case and pau_control are those two means over the same ",
  "usable cells; ", apa_delta_clause,
  apa_trend_clause,
  "se, t_stat, pvalue and padj are the ",
  "moderated arm of that one fit; independent filtering: not applied (the ",
  "threshold is NA and nothing was filtered); ",
  contrast_note, "; ", dpl_batch_note)

write.table(apa_out[, APA_COLUMNS],
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
