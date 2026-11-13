#!/usr/bin/env Rscript
                                                        
 
                                                                          
                                                                               
                                                                             
                
 
                                                                             
                           
                                                                            
                                                                         
                                                                           
                                                                  
                                                                           
                                                                             
                                      
 
              
                                                      
                                                                              
                                   
                                                                      
                                                                       
                                                                       
                                                                        
                                                                             
                                                                        
                                                                                
                                                                      
 
                                                                            
                                                                            
                                                                             
                                                             
 
                                                                     
                                                                             
                                                                     
                                                                             
                                                           
 
                                     
                                                                        
                                                                        
                                         
 
                                                                     
                                                                            
                                                                           
                                                                          
                                                                        
                                                                        
                                                                           
            
 
                                                                              
                                                                            
                                                       
                                                                              
                                                                         
                                                                            
                                                                              
                                                                          
                                                                         
                                                                              
                                                                          
        
 
                                                                           
                                                                           
                                                              
                                                                     
 
                                                                        
                                                                             
                                                       
 
                                                          
                                                                      
                                                                      
                                                                 
 
        
                              
                                
                                                      
                             
                            
                       
                 
 
         
                                                
                                                     
                                                          
                                                                         
                                                                         
                                                                       
                                              
                                                                           
                                                                           
                                                              
                                         
                                                                            
                                                                        
                             
                                                                
                                
                                                               
                                  
 
                                                                           
                                                                         
                                                                              
                                                                             
                                            

suppressPackageStartupMessages({
  library(optparse)
  library(arrow)
  library(dplyr)
  library(tidyr)
  library(DEXSeq)
  library(limma)                                                                     
})

option_list <- list(
  make_option("--metadata", type = "character",
              help = paste("Metadata TSV that declares the samples: sample_id,",
                           "condition and profile_path per row, and an optional",
                           "batch column. A relative profile_path is resolved",
                           "against the metadata file's own directory. The rows",
                           "of a profile belong to the sample the metadata",
                           "declares.")),
  make_option("--aggregate", type = "character",
              help = paste("Pre-aggregated (transcript x sample) TSV. When it is",
                           "given the engine does not read the profiles itself;",
                           "the metadata still names the samples and their",
                           "conditions.")),
  make_option("--output-dir", type = "character", default = ".",
              help = "Output directory [default %default]"),
  make_option("--min-cell-weight", type = "double", default = 10,
              help = "Min sum(posterior_weight) per (transcript x sample) cell [default %default]"),
  make_option("--min-neff", type = "double", default = 10,
              help = "Min Kish effective sample size per (transcript x sample) cell [default %default]"),
  make_option("--min-counts", type = "double", default = 20,
              help = paste("Min total count for a feature to be tested: a gene in the",
                           "expression track, a transcript in the abundance and usage",
                           "tracks, and the gene's own reads in a sample in the usage",
                           "track's gate. The poly(A) track's per-SAMPLE threshold is a",
                           "quarter of this number with a floor of 5 [default %default]")),
  make_option("--min-samples", type = "integer", default = 3,
              help = paste("Min number of samples per group. It is the ADMISSION GATE",
                           "for the expression, transcript-abundance and usage tracks",
                           "(a feature needs a non-zero count in this many samples of",
                           "EACH group; a feature the gate excludes is left out of the",
                           "adjustment's denominator and says so), and the minimum",
                           "number of usable cells per group for the tail-length and",
                           "poly(A) tracks [default %default]")),
  make_option("--apa-min-delta-pau", type = "double", default = 0.05,
              help = paste("The absolute OBSERVED usage difference |delta_pau| a",
                           "poly(A) site must reach to be called significant in the",
                           "DEXSeq model. It does NOT touch pvalue or padj:",
                           "those are pure statistics and are adjusted over the",
                           "tested rows alone [default %default]")),
  make_option("--lapa-counting-method", type = "character",
              default = "unknown (not passed to this engine)",
              help = paste("LAPA's counting_method, for apa.tsv's receipt. The CLI",
                           "passes the value it gave LAPA; the engine does not run",
                           "LAPA, so it can only report what it is told")),
  make_option("--lapa-min-replication-rate", type = "character",
              default = "unknown (not passed to this engine)",
              help = paste("LAPA's min_replication_rate, for apa.tsv's receipt. The",
                           "CLI passes the value it gave LAPA; the engine does not",
                           "run LAPA, so it can only report what it is told")),
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
              help = "Write dpl.tsv (the DT-EB limma length model)"),
  make_option("--lapa-dir", type = "character",
              help = paste("LAPA's output directory, turned into apa.tsv. The",
                           "per-sample cluster files are read from <dir>/sample (the",
                           "replicated calls), or from <dir>/*/sample, or -- only if",
                           "neither exists -- from <dir>/raw_sample (the unreplicated",
                           "calls, which apa.tsv's receipt then names). The",
                           "dataset-level cluster file in <dir] is not a sample and is",
                           "never read.")),
  make_option("--gene-names", type = "character",
              help = paste("TSV of `gene_id<TAB>gene_name`: what every gene of the",
                           "annotation is CALLED. The CLI writes it, because the CLI",
                           "reads the annotation GTF, so the engine parses no",
                           "annotation of its own and the two sides cannot disagree",
                           "about which name belongs to which gene. A gene the map",
                           "does not cover keeps its gene id as its name."))
)
opt <- parse_args(OptionParser(option_list = option_list))

                                                                              
                                                                           
                                                                              
                                          
apa_min_delta_pau <- opt$`apa-min-delta-pau`
if (is.null(apa_min_delta_pau) || is.na(apa_min_delta_pau) ||
    apa_min_delta_pau < 0) {
  stop("ERROR: --apa-min-delta-pau must be a non-negative number, got: ",
       opt$`apa-min-delta-pau`)
}

                        
 
                                                                           
                                                                              
                                                                         
                                                                           
                                                           
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
                                                                        
                                                                         
                  
  want_deg <- TRUE
  want_det <- TRUE
  want_diu <- TRUE
  want_dpl <- TRUE
}
cat("[telescope_diff] tables:",
    paste(c(if (want_deg) "deg", if (want_det) "det", if (want_diu) "diu",
            if (want_dpl) "dpl"),
          collapse = ", "), "\n")

                                                                          
                                                            
                        
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

if (is.null(opt$metadata)) {
  stop("ERROR: --metadata is required.")
}
dir.create(opt$`output-dir`, recursive = TRUE, showWarnings = FALSE)

                                                                            
                                                                            
                                                              
 
                                                                     
                                                                       
            
metadata_path <- opt$metadata
if (!file.exists(metadata_path)) {
  stop("ERROR: the metadata file does not exist: ", metadata_path)
}
metadata <- read.delim(metadata_path, stringsAsFactors = FALSE)
need_meta_cols <- c("sample_id", "condition", "profile_path")
missing_meta_cols <- setdiff(need_meta_cols, colnames(metadata))
if (length(missing_meta_cols) > 0) {
  stop("ERROR: the metadata table is missing column(s): ",
       paste(missing_meta_cols, collapse = ", "),
       ". Its columns are: ", paste(colnames(metadata), collapse = ", "))
}
metadata$sample_id   <- as.character(metadata$sample_id)
metadata$condition   <- as.character(metadata$condition)
metadata$profile_path <- as.character(metadata$profile_path)
if (any(is.na(metadata$sample_id)) || any(metadata$sample_id == "")) {
  stop("ERROR: every row of the metadata must name its sample.")
}
if (any(is.na(metadata$profile_path)) || any(metadata$profile_path == "")) {
  stop("ERROR: every row of the metadata must name its profile parquet.")
}
if (anyDuplicated(metadata$sample_id) > 0) {
  stop("ERROR: the metadata declares a sample twice: ",
       paste(unique(metadata$sample_id[duplicated(metadata$sample_id)]),
             collapse = ", "))
}
metadata_base <- dirname(normalizePath(metadata_path))
resolve_profile <- function(p) {
  if (grepl("^(/|[A-Za-z]:)", p)) p else file.path(metadata_base, p)
}
metadata$profile_path <- vapply(metadata$profile_path, resolve_profile,
                                 character(1))

                                                                             
                                                                            
                                                         
design <- data.frame(
  sample_id = metadata$sample_id,
  condition = metadata$condition,
  stringsAsFactors = FALSE
)
if ("batch" %in% colnames(metadata)) {
  design$batch <- as.character(metadata$batch)
}

                                     
 
                                                                    
                                                                              
                                                             
                                                                             
                                                                        
                                                                            
                   
 
                                                                             
                                                                       
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
  profile_files <- metadata$profile_path
  missing_profiles <- profile_files[!file.exists(profile_files)]
  if (length(missing_profiles) > 0) {
    stop("ERROR: the profile parquet(s) named by the metadata do not exist: ",
         paste(missing_profiles, collapse = ", "))
  }
  cat("  found", length(profile_files), "files\n")

                                                                         
                                                                    
                                                                              
                                        
  read_profile <- function(f, sample_id) {
    df <- as.data.frame(read_parquet(f))
    df$sample_id <- sample_id
    df
  }
  profiles_df <- dplyr::bind_rows(
    Map(read_profile, profile_files, metadata$sample_id)
  )
  cat("  total rows:", nrow(profiles_df), "\n")
}

                 
 
                                                                             
                                                                              
                                                                            
                                         
 
                                                                             
                                                                            
        
gene_name_of <- character(0)
if (!is.null(opt$`gene-names`)) {
  if (!file.exists(opt$`gene-names`)) {
    stop("ERROR: the gene-name table does not exist: ", opt$`gene-names`)
  }
  gn <- utils::read.delim(opt$`gene-names`, header = FALSE, sep = "\t",
                          quote = "", comment.char = "",
                          col.names = c("gene_id", "gene_name"),
                          stringsAsFactors = FALSE)
  gene_name_of <- as.character(gn$gene_name)
  names(gene_name_of) <- as.character(gn$gene_id)
  cat("  gene names:", length(gene_name_of), "read from", opt$`gene-names`, "\n")
}

                                                                           
                                     
attach_gene_name <- function(df) {
  if (!("gene_id" %in% colnames(df))) return(df)
  ids <- as.character(df$gene_id)
  nm <- unname(gene_name_of[ids])
  missing <- is.na(nm) | !nzchar(nm)
  nm[missing] <- ids[missing]
  cols <- colnames(df)
  i <- match("gene_id", cols)
  rest <- if (i < length(cols)) cols[(i + 1L):length(cols)] else character(0)
  df$gene_name <- nm
  df[, c(cols[seq_len(i)], "gene_name", rest), drop = FALSE]
}

                                                                   
                                                                              
                                                                              
                                                                               
                                                              
resolve_contrast <- function(case_condition, control_condition, contrast) {
  if (!is.null(contrast)) {
    parts <- strsplit(contrast, ":", fixed = TRUE)[[1]]
    if (length(parts) != 2 || any(nzchar(parts) == FALSE)) {
      stop("ERROR: --contrast must be CASE:CONTROL, for example treated:control.")
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

                                                                           
                                                                             
                                                                            
                                                                
                                        
n_control_samples <- sum(design$condition == control_condition)
n_case_samples    <- sum(design$condition == case_condition)

                                                                     
                                                                 
                                                                   
                                                                       
                                                                        
if (n_case_samples < 3 || n_control_samples < 3) {
  stop("ERROR: each condition needs at least 3 samples (case=",
       n_case_samples, ", control=", n_control_samples, ").")
}
cat("[telescope_diff] Admission gate: and",
    "; usage-track matrix: narrowed",
    " (control ", n_control_samples, ", case ", n_case_samples, ")",
    "\n", sep = "")

                                      
deg_gate_note <- paste0(
  "admission gate: a gene is tested only if it has at least ", opt$`min-counts`,
  " reads in total AND at least ", opt$`min-samples`,
  " samples with a non-zero count in EACH group; ")
det_gate_note <- paste0(
  "admission gate: a transcript is tested only if it has at least ",
  opt$`min-counts`, " reads in total AND at least ", opt$`min-samples`,
  " samples with a non-zero count in EACH group; ")
diu_gate_note <- paste0(
  "admission gate: a transcript is tested only if it has a non-zero count in ",
  "at least ", opt$`min-samples`, " samples of EACH group AND its gene ",
  "carries at least ", opt$`min-counts`, " reads in at least ", opt$`min-samples`,
  " samples of EACH group; the gene's reads in that condition are summed over ",
  "ALL of the gene's transcripts, not only the ones that passed the ",
  "minimum-count filter -- this is the gate's gene total, which is NOT the ",
  "share's denominator (that one is over the transcripts DEXSeq saw); ")

                                                                           
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

                                                                            
                                                                            
                                                                    
                                                                            
                                                                          
                                                                        
                                                                
batch_vec <- if ("batch" %in% colnames(design)) as.character(design$batch) else NULL
dpl_spec <- dpl_design(design$sample_id, as.character(design$condition),
                       case_condition, control_condition, batch_vec)
dpl_design_matrix <- dpl_spec$design
dpl_batch_note <- dpl_spec$batch_note

                                                                             
                                                                         
                                                                         
       
contrast_note <- paste0("contrast: ", case_condition, " - ", control_condition,
                         " (case - control)")

                                                                       
                                                                
                                                                       
                                                                     
                                                                         
                     
if (want_diu || want_det || want_deg) {
counts_long <- if (is.null(profiles_df)) {
                                                                     
                              
  agg_raw %>%
    dplyr::transmute(gene_id, transcript_id, sample_id,
                     count = as.integer(round(weight_sum)))
} else {
  profiles_df %>%
    dplyr::group_by(gene_id, transcript_id, sample_id) %>%
    dplyr::summarize(count = sum(posterior_weight), .groups = "drop") %>%
                                                                            
                                                                           
                                                   
    dplyr::mutate(count = as.integer(round(count)))
}

                                                                          
                                                                      
                                                                        
                                                             
counts_for_genes <- counts_long
                                                                           
                                                                          
                                                                     
counts_for_tx <- counts_long
}

                                                                       
                                                                           
                                                                      
                                                                           
                                                     
rank_reduced_model_matrix <- function(dxd) {
  mm <- model.matrix(design(dxd), as.data.frame(colData(dxd)))
  q <- qr(mm)
  if (q$rank < ncol(mm)) {
    mm[, -q$pivot[(q$rank + 1):ncol(mm)], drop = FALSE]
  } else {
    mm
  }
}

                                                                       
                                                                 
                                                                       
 
                                                                           
                                                                            
                                                                         
                                                                         
                                                                         
                            
if (want_diu) {
cat("\n[telescope_diff] usage track: differential isoform usage via DEXSeq\n")
counts_by_tx <- counts_long %>%
  dplyr::group_by(gene_id, transcript_id) %>%
  dplyr::summarize(total = sum(count), .groups = "drop")
                                                                            
                                           
counts_by_tx_all <- counts_by_tx
                                                                             
                                                                           
                                                                              
                                                             
 
                                                    
                                                                          
                                                                            
                        
                                                                             
                                                                           
                                                    
                                                                           
          
counts_long_all <- counts_long
keep_tx <- counts_by_tx %>%
  dplyr::filter(total >= opt$`min-counts`) %>%
  dplyr::select(gene_id, transcript_id)
                                                               
counts_long <- counts_long_all
cat("  transcripts in the complete matrix (the per-group rule decides what",
    "is tested):", dplyr::n_distinct(counts_long$transcript_id), "\n")

                                                                            
                                                                            
                                                                           
                                                              
                                                                           
                                                                             
                                                 
 
                                                                   
                                                                    
                                                                         
                                                                          
                                                                    
                                                                          
             
 
                                                                            
                                                                           
                                          
diu_case_samples    <- design$sample_id[as.character(design$condition) == case_condition]
diu_control_samples <- design$sample_id[as.character(design$condition) == control_condition]
diu_tx_gate <- counts_long %>%
  dplyr::mutate(is_case = sample_id %in% diu_case_samples,
                is_ctrl = sample_id %in% diu_control_samples,
                pos = count > 0) %>%
  dplyr::group_by(gene_id, transcript_id) %>%
  dplyr::summarize(n_pos_case = sum(pos & is_case),
                   n_pos_ctrl = sum(pos & is_ctrl),
                   total = sum(count),
                   .groups = "drop")
                                                                       
                                                                        
                                                                         
                                                                             
                                                                              
                                                                            
                                                                        
                                                                         
        
diu_gene_gate <- counts_for_genes %>%
  dplyr::group_by(gene_id, sample_id) %>%
  dplyr::summarize(gene_total = sum(count), .groups = "drop") %>%
  dplyr::mutate(is_case = sample_id %in% diu_case_samples,
                is_ctrl = sample_id %in% diu_control_samples,
                big = gene_total >= opt$`min-counts`) %>%
  dplyr::group_by(gene_id) %>%
  dplyr::summarize(n_big_case = sum(big & is_case),
                   n_big_ctrl = sum(big & is_ctrl), .groups = "drop")
diu_admit <- diu_tx_gate %>%
  dplyr::left_join(diu_gene_gate, by = "gene_id") %>%
  dplyr::mutate(admitted = !is.na(n_big_case) &
                          n_big_case >= opt$`min-samples` &
                          n_big_ctrl >= opt$`min-samples`)
                                                                         
                                                                         
                                                                           
                                                                          
                                                                        
       
 
                                                               
                                                                         
                                                                         
                                                                     
                                                                      
                                                                        
                                                                       
                                                                       
                      
diu_admit <- diu_admit %>%
  dplyr::mutate(admitted = admitted & !is.na(n_pos_case) &
                          n_pos_case >= opt$`min-samples` &
                          n_pos_ctrl >= opt$`min-samples` &
                          !is.na(total) & total >= opt$`min-counts`)
diu_admitted <- diu_admit %>% dplyr::filter(admitted) %>% dplyr::select(gene_id, transcript_id)
                                                                         
                                                                            
                                          
 
                                                                            
                                                                           
                                                                             
                                                                             
                                 
diu_candidate_tx <- keep_tx
diu_gate_excluded <- dplyr::anti_join(diu_candidate_tx, diu_admitted,
                                      by = c("gene_id", "transcript_id"))
                                                                           
                                                                          
                                                                           
                                                                         
                                        
 
                                                                      
                                                                    
                                                                        
                                                                         
                                                                         
                                                                         
                                                                  
counts_long <- counts_long %>%
  dplyr::semi_join(keep_tx, by = c("gene_id", "transcript_id")) %>%
  dplyr::semi_join(diu_admitted, by = c("gene_id", "transcript_id"))
cat("  transcripts after the per-group admission gate:",
    dplyr::n_distinct(counts_long$transcript_id), "of",
    dplyr::n_distinct(counts_by_tx_all$transcript_id), "\n")
                                                                           
                                                                             
                                                                           
                                                                         
                                                                        
                                                                         
                                                                 
diu_fitted_gene_totals <- counts_long %>%
  dplyr::group_by(gene_id, sample_id) %>%
  dplyr::summarize(fitted_total = sum(count), .groups = "drop")
diu_complete_gene_totals <- counts_for_genes %>%
  dplyr::group_by(gene_id, sample_id) %>%
  dplyr::summarize(complete_total = sum(count), .groups = "drop")
diu_conservation <- dplyr::left_join(diu_fitted_gene_totals,
                                      diu_complete_gene_totals,
                                      by = c("gene_id", "sample_id"))
diu_conservation$complete_total[is.na(diu_conservation$complete_total)] <- 0
diu_conservation$delta <- diu_conservation$fitted_total -
  diu_conservation$complete_total
diu_conservation_n_bad <- sum(diu_conservation$delta != 0)
diu_conservation_max <- if (nrow(diu_conservation) == 0) 0 else
  max(abs(diu_conservation$delta))
cat("  count-matrix / parent-total conservation:", diu_conservation_n_bad,
    "of", nrow(diu_conservation),
    "(gene x sample) cells differ from the complete gene total;",
    "largest difference:", diu_conservation_max, "reads\n")
                                                                       
                                                                    
                                                                           
diu_matrix_note <- paste0(
  "the gene total behind a share is over the transcripts DEXSeq saw, i.e. ",
  "after the usage track's minimum-count filter and the single-transcript-gene ",
  "drop; ")

                                                                       
                                                                            
                                                                            
                                          
 
                                                                             
                                                                     
 
                                                                   
                                                                             
                                                                           
                                                                              
                                                                        
                                                                            
                                                                            
                                                                              
                                    
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

                                                                             
                                                                           
                                                                      
                                                                          
                                                                       
                                                                              
                                                                           
                                                                            
                                                                           
             
diu_candidate_tx_key <- paste(diu_candidate_tx$gene_id, diu_candidate_tx$transcript_id)
diu_singleton_tx <- diu_admitted %>%
  dplyr::mutate(singleton = !(gene_id %in% genes_keep)) %>%
  dplyr::filter(singleton)
diu_singleton_key <- paste(diu_singleton_tx$gene_id, diu_singleton_tx$transcript_id)

                                                                    
counts_wide <- counts_long %>%
  tidyr::pivot_wider(names_from = sample_id, values_from = count,
                    values_fill = 0)
gene_ids    <- counts_wide$gene_id
tx_ids      <- counts_wide$transcript_id
sample_cols <- setdiff(colnames(counts_wide), c("gene_id", "transcript_id"))
count_mat   <- as.matrix(counts_wide[, sample_cols])
rownames(count_mat) <- paste(gene_ids, tx_ids, sep = "|")
count_mat[is.na(count_mat)] <- 0
                                                                     
                                                                             
                                              
diu_fit_ok <- nrow(count_mat) > 0
if (!diu_fit_ok) {
  cat("  WARN: no transcript passed the admission gate; no transcript is tested, and diu.tsv carries the reason.\n")
}

                                         
design_aligned <- design[match(colnames(count_mat), design$sample_id), ]
if (any(is.na(design_aligned$sample_id))) {
  stop("ERROR: design sample_ids do not match count matrix columns.")
}

                                                                 
if (diu_fit_ok) {
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
                      reducedModel = ~ sample + exon,
                      fullModel    = ~ sample + exon + condition:exon)
    dxd <- estimateExonFoldChanges(dxd, fitExpToVar = "condition")
  })
                                      
   
                                                                              
                                                                               
                                                                     
   
                                                                          
                                                                            
                                                           
  dxr <- DEXSeqResults(dxd, independentFiltering = FALSE)
  filter_threshold <- attr(dxr, "filterThreshold")
  if (is.null(filter_threshold) || length(filter_threshold) == 0) filter_threshold <- NA_real_
  n_independent_filtered <- 0L

                                                                      
                                                                              
                                                        
  lfc_col <- grep("^log2fold_", colnames(dxr), value = TRUE)
  if (length(lfc_col) != 1) {
    stop("ERROR: expected exactly one DEXSeq effect column, found: ",
         paste(lfc_col, collapse = ", "))
  }
  dexseq_lfc <- as.numeric(dxr[[lfc_col]])

                                                                  
  case_samples    <- design_aligned$sample_id[design_aligned$condition == case_condition]
  control_samples <- design_aligned$sample_id[design_aligned$condition == control_condition]

                                                                          
                                                                              
                                                                           
                                                                       
                                                         
                                                                           
                                                                              
                                                                              
                                                                          
                                                                           
                           
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
                                                                          
  res_feature <- dxr$featureID
  res_gene <- dxr$groupID
                                                                           
                                                                          
                                                  
} else {
  dxr <- data.frame(groupID = character(0), featureID = character(0),
                    pvalue = numeric(0), padj = numeric(0),
                    row.names = character(0), stringsAsFactors = FALSE)
  dexseq_lfc <- numeric(0)
  case_samples <- character(0)
  control_samples <- character(0)
  share_by_row <- matrix(numeric(0), nrow = 0, ncol = 0)
  res_feature <- character(0)
  res_gene <- character(0)
  filter_threshold <- NA_real_
  n_independent_filtered <- 0L
  disp_note <- "no dispersion was fitted, because no transcript was fitted"
}
if (length(res_feature) != nrow(share_by_row)) {
  stop("ERROR: the DEXSeq results table has ", length(res_feature),
       " rows but the usage matrix has ", nrow(share_by_row),
       "; the shares cannot be matched to the transcripts.")
}
                                                                       
                                                                          
                                                                           
                                                                          
                                                                         
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
                                                                     
                                                                           
                                                                       
                                   
diu_admitted_key <- paste(diu_admitted$gene_id, diu_admitted$transcript_id)
diu_out_key <- paste(diu_out$gene_id, diu_out$transcript_id)
fitted <- !is.na(idx)
admitted <- diu_out_key %in% diu_admitted_key
tested <- fitted & admitted
diu_out$pvalue[tested]         <- dxr$pvalue[idx[tested]]
diu_out$log2FoldChange[tested] <- dexseq_lfc[idx[tested]]
                                                                         
                                                                         
                                                  
diu_out$padj[tested]           <- dxr$padj[idx[tested]]
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
                                                                     
                                                                      
                                                                            
                                                          
 
                                                                             
                                         
diu_gene_q_object <- dxr
gene_q <- if (!diu_fit_ok || nrow(diu_gene_q_object) == 0) NULL else
  tryCatch(DEXSeq::perGeneQValue(diu_gene_q_object),
           error = function(e) {
             cat("  WARN: the per-gene q-values could not be computed (",
                 conditionMessage(e), "); gene_padj is NA.\n")
             NULL
           })
if (!is.null(gene_q)) {
  diu_out$gene_padj <- unname(gene_q[match(diu_out$gene_id, names(gene_q))])
                                                                  
                                                                          
                                                                           
                                                                          
                                                                    
                                                                         
                                                                    
                                                                         
                                                                         
  diu_out$gene_padj[!tested] <- NA_real_
                                                                         
                                                                       
                                  
  diu_gp_had <- !is.na(diu_out$gene_padj)
  diu_gp_lost_rows <- sum(diu_gp_had & !tested)
  diu_gp_genes <- unique(diu_out$gene_id[diu_gp_had])
  diu_gp_genes_kept <- unique(diu_out$gene_id[diu_gp_had & tested])
  diu_gp_genes_lost <- length(setdiff(diu_gp_genes, diu_gp_genes_kept))
  cat("  DIU gene-level q-value: rows that never carried it (untested):",
      diu_gp_lost_rows, "; genes carrying a gene-level q-value:",
      length(diu_gp_genes), "; genes LOST by removing it from the untested rows:",
      diu_gp_genes_lost, "(0 = the gene-level call is unchanged)\n")
}
                                                                        
                                                                            
                                                                        
                                     
                                                                         
                                                                           
                                                                         
                                                                             
                                                                           
                                                                       
                                                                       
diu_count_design <- if (diu_fit_ok) design_aligned else design
diu_out$n_case    <- sum(diu_count_design$condition == case_condition)
diu_out$n_control <- sum(diu_count_design$condition == control_condition)
diu_out$independent_filter_threshold <- filter_threshold
diu_out$n_independent_filtered <- n_independent_filtered
                                                                            
                                                                        
                                                                            
                                                        
diu_gate_excluded_key <- paste(diu_gate_excluded$gene_id,
                                diu_gate_excluded$transcript_id)
diu_row_key <- paste(diu_out$gene_id, diu_out$transcript_id)
diu_out$padj_na_reason <- ifelse(
  diu_row_key %in% diu_gate_excluded_key,
  "not_admitted_by_gate",
  ifelse(diu_row_key %in% diu_singleton_key,
         "singleton_gene",
         ifelse(!(diu_row_key %in% diu_candidate_tx_key),
                "below_min_counts",
                ifelse(!tested, "not_tested",
                       ifelse(is.na(diu_out$pvalue), "no_p_value", "none")))))
                                                                             
                                                                           
                                                                             
     
diu_out$model_name <- ifelse(tested, "dexseq", "not_tested")
                                                                        
                                                                         
                                                                             
                                                                            
                                                                          
                                                                    
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
  diu_matrix_note,
  diu_gate_note,
  "the transcripts the gate excluded carry ",
  "padj_na_reason = not_admitted_by_gate and are kept out of the adjustment's ",
  "denominator; ",
  "gene_padj is DEXSeq::perGeneQValue, ONE q-value per GENE (the gene's ",
  "smallest transcript p-value, adjusted across genes); it is given on the ",
  "TESTED rows of that gene and is NA on the rows whose transcript was ",
  "never tested, so a row with no transcript-level test never looks like a ",
  "gene-level hit; the value is a property of the gene, so no gene-level ",
  "call is lost by this; a different multiplicity from the ",
  "transcript-level padj; ",
  "independent filtering: not applied (the threshold is NA and nothing was ",
  "filtered); ",
  contrast_note, "; ", disp_note)
diu_out <- diu_out[, c("gene_id", "transcript_id", "case_condition",
                       "control_condition", "share_case", "share_control",
                       "delta_share", "log2FoldChange", "pvalue", "padj",
                       "gene_padj", "total", "n_case", "n_control",
                       "independent_filter_threshold", "n_independent_filtered",
                       "padj_na_reason", "model_name", "model_parameters")]

diu_out <- attach_gene_name(diu_out)
write.table(diu_out,
            file = file.path(opt$`output-dir`, "diu.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("[telescope_diff]  wrote diu.tsv (",
    nrow(diu_out), "transcripts)\n")
}

                                                                       
                                                                           
                                                                       
 
                                                                            
                                                                         
                                                                           
                                                                           
                                                                           
                                                               
if (want_det) {
cat("\n[telescope_diff] transcript track: differential transcript abundance via DESeq2\n")
if (!requireNamespace("DESeq2", quietly = TRUE)) {
  stop("ERROR: the transcript track needs DESeq2, which is not installed in this R ",
       "library. Create the pinned environment: mamba env create -f environment.yml")
}

                                                                            
                                                                            
                                                                          
       
tx_counts_long <- counts_for_tx %>%
  dplyr::group_by(gene_id, transcript_id, sample_id) %>%
  dplyr::summarize(count = sum(count), .groups = "drop")
tx_ids_all <- sort(unique(tx_counts_long$transcript_id))
                                                                        
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
                                                                          
                                                                           
                                             
tx_keys <- paste(tx_counts_long$gene_id, tx_counts_long$transcript_id, sep = "|")
tx_key_by_id <- tapply(tx_keys, tx_counts_long$transcript_id, function(x) x[1])
rownames(tx_mat) <- unname(tx_key_by_id[tx_ids_all])
if (anyDuplicated(rownames(tx_mat))) {
  stop("ERROR: two transcripts of the same gene share a transcript id; ",
       "the transcript-level matrix cannot be keyed.")
}

                                                                         
                                                                       
                                
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

                                                                           
                                                                            
                                                                           
                                                         
                                                                             
                                      
 
                                                                           
                                                                             
                                                                            
                                                                            
                                                                    
det_admitted <- rowSums(tx_mat) >= opt$`min-counts` &
  rowSums(tx_mat[, design_tx$condition == control_condition, drop = FALSE] > 0) >=
    opt$`min-samples` &
  rowSums(tx_mat[, design_tx$condition == case_condition, drop = FALSE] > 0) >=
    opt$`min-samples`
cat("  transcripts admitted by the per-group gate:", sum(det_admitted),
    "of", length(det_admitted), "\n")
                                                                              
                                                             
det_fit_ok <- sum(det_admitted) > 0
if (!det_fit_ok) {
  cat("  WARN: no transcript passed the admission gate; no transcript is tested, and det.tsv carries the reason.\n")
}

if (det_fit_ok) {
  dds_tx <- DESeq2::DESeqDataSetFromMatrix(countData = tx_mat[det_admitted, , drop = FALSE],
                                            colData = design_tx,
                                            design = det_formula)
                                                                             
                                                                            
                                                                        
                                                           
  tx_sf_note <- paste0("median-of-ratios size factors, estimated on the ADMITTED ",
                      "transcripts only (the matrix the gate left), not on the ",
                      "whole input")
  dds_tx <- tryCatch(DESeq2::estimateSizeFactors(dds_tx),
                     error = function(e) {
                       cat("  WARN: the transcript size factors could not be estimated (",
                           conditionMessage(e),
                           "); using DESeq2's poscounts estimator.\n")
                       tx_sf_note <<- "poscounts size factors (the median-of-ratios estimator failed)"
                       DESeq2::estimateSizeFactors(dds_tx, type = "poscounts")
                     })

  tx_disp_note <- paste0("parametric dispersion fit, fitted on the ADMITTED ",
                        "transcripts only (the matrix the gate left), not on the ",
                        "whole input")
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

                                                                        
                                                                              
                                                                              
                                                                            
                                                                               
                                                                         
                                                                             
                                                                  
  det_shrunk_df <- NULL
  det_shrink_note <- "no shrinkage was applied; log2FoldChange_shrunk and lfcSE_shrunk are NA"
  det_coef <- grep("^condition_", DESeq2::resultsNames(dds_tx), value = TRUE)
  if (length(det_coef) == 1) {
    det_shrunk <- tryCatch(
      DESeq2::lfcShrink(dds_tx, coef = det_coef, type = "normal", quiet = TRUE),
      error = function(e) {
        cat("  WARN: the transcript log2 fold changes could not be shrunk (",
            conditionMessage(e), "); the unshrunk estimates are reported.\n")
        NULL
      })
    if (!is.null(det_shrunk)) {
      det_shrunk_df <- as.data.frame(det_shrunk)
                                                                              
                                                                             
                                                                         
      if (!identical(rownames(det_shrunk_df), rownames(dres_tx_df))) {
        stop("ERROR: the shrunk transcript fold changes and the Wald test's ",
             "results are in a different order; the effect size would be paired ",
             "with the wrong p-value.")
      }
      det_shrink_note <- paste0(
        "log2FoldChange and lfcSE are the unshrunk maximum-likelihood estimate ",
        "and its standard error, so stat = log2FoldChange / lfcSE holds exactly; ",
        "log2FoldChange_shrunk and lfcSE_shrunk are DESeq2's normal-prior shrunk ",
        "estimate and its posterior sd (lfcShrink, type = \"normal\"); stat, ",
        "pvalue and padj are the Wald test's and are bit-identical to results()'")
    }
  }
                                                         
} else {
  dres_tx_df <- data.frame(baseMean = numeric(0), log2FoldChange = numeric(0),
                          lfcSE = numeric(0), stat = numeric(0), pvalue = numeric(0),
                          padj = numeric(0), row.names = character(0))
  det_shrunk_df <- NULL
  det_shrink_note <- "no transcript was fitted, so no effect size is reported"
  tx_sf_note <- "no size factors were estimated, because no transcript was fitted"
  tx_disp_note <- "no dispersion was fitted, because no transcript was fitted"
}

det_out <- data.frame(
  gene_id                = unname(gene_of_tx[tx_ids_all]),
  transcript_id          = tx_ids_all,
  case_condition         = rep(case_condition, length(tx_ids_all)),
  control_condition      = rep(control_condition, length(tx_ids_all)),
  baseMean               = NA_real_,
  log2FoldChange         = NA_real_,
  log2FoldChange_shrunk  = NA_real_,
  lfcSE                  = NA_real_,
  lfcSE_shrunk           = NA_real_,
  stat                  = NA_real_,
  pvalue                 = NA_real_,
  padj                   = NA_real_,
  n_case                 = rep(sum(design_tx$condition == case_condition), length(tx_ids_all)),
  n_control              = rep(sum(design_tx$condition == control_condition), length(tx_ids_all)),
  independent_filter_threshold = NA_real_,
  n_independent_filtered = 0L,
  stringsAsFactors = FALSE
)
                                                                         
                                       
det_keys <- unname(tx_key_by_id[tx_ids_all])
det_idx <- match(det_keys, rownames(dres_tx_df))
det_tested <- !is.na(det_idx)
det_out$baseMean[det_tested]       <- dres_tx_df$baseMean[det_idx[det_tested]]
det_out$log2FoldChange[det_tested] <- dres_tx_df$log2FoldChange[det_idx[det_tested]]
det_out$lfcSE[det_tested]          <- dres_tx_df$lfcSE[det_idx[det_tested]]
det_out$log2FoldChange_shrunk[det_tested] <- if (is.null(det_shrunk_df)) {
  rep(NA_real_, sum(det_tested))
} else {
  det_shrunk_df$log2FoldChange[det_idx[det_tested]]
}
det_out$lfcSE_shrunk[det_tested] <- if (is.null(det_shrunk_df)) {
  rep(NA_real_, sum(det_tested))
} else {
  det_shrunk_df$lfcSE[det_idx[det_tested]]
}
det_out$stat[det_tested]           <- dres_tx_df$stat[det_idx[det_tested]]
det_out$pvalue[det_tested]         <- dres_tx_df$pvalue[det_idx[det_tested]]
det_out$padj[det_tested]           <- dres_tx_df$padj[det_idx[det_tested]]
                                                                            
                                                                              
                    
det_out$padj_na_reason <- ifelse(!det_admitted, "not_admitted_by_gate",
                          ifelse(!det_tested, "not_tested",
                          ifelse(is.na(det_out$pvalue), "no_p_value", "none")))
                                                                         
                                                                            
                                                    
det_out$model_name <- ifelse(det_tested, "deseq2_wald", "not_tested")
det_out$model_parameters <- paste0(
  "DESeq2 on the transcript-level counts (the transcript's own abundance, ",
  "not its share of its gene's output -- that is the DEXSeq usage test in ",
  "diu.tsv); negative binomial Wald test; results(independentFiltering = FALSE); ",
  "the counts are the Rust aggregator's posterior mass rounded to integers, ",
  "not molecule counts; the matrix is the UNFILTERED one, i.e. before the ",
  "usage track's minimum-count filter and the single-transcript-gene drop, ",
  "because those thresholds are the usage track's business; ",
  det_gate_note,
  "the transcripts the gate ",
  "excluded carry padj_na_reason = not_admitted_by_gate and are kept out of ",
  "the adjustment's denominator; ",
  det_shrink_note, "; ",
  "independent filtering: not applied (the threshold is NA and nothing was ",
  "filtered); ",
  tx_sf_note, "; ", tx_disp_note, "; ", contrast_note, "; ", dpl_batch_note)

det_out <- attach_gene_name(det_out)
write.table(det_out,
            file = file.path(opt$`output-dir`, "det.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("[telescope_diff]  wrote det.tsv (", nrow(det_out), "transcripts,",
    sum(det_out$padj_na_reason == "none"), "tested)\n")
}

                                                                       
                                                             
                                                                       
if (want_deg) {
cat("\n[telescope_diff] gene track: differential expression via DESeq2\n")
if (!requireNamespace("DESeq2", quietly = TRUE)) {
  stop("ERROR: the gene track needs DESeq2, which is not installed in this R ",
       "library. Create the pinned environment: mamba env create -f environment.yml")
}

                                                                             
                                                               
                                                                             
                                                    
gene_counts_long <- counts_for_genes %>%
  dplyr::group_by(gene_id, sample_id) %>%
  dplyr::summarize(count = sum(count), .groups = "drop")
gene_ids_all <- sort(unique(gene_counts_long$gene_id))
cat("  genes:", length(gene_ids_all), "\n")

                                                                         
                                 
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

                                                                            
                                                                          
                                                                             
                                                                         
                                                                 
                                                                          
                                      
 
                                                                            
                                                                      
                                                                           
                                                                          
                                                                          
                                                                           
                                                                           
                                                            
deg_admitted <- rowSums(gene_mat) >= opt$`min-counts` &
  rowSums(gene_mat[, design_genes$condition == control_condition, drop = FALSE] > 0) >=
    opt$`min-samples` &
  rowSums(gene_mat[, design_genes$condition == case_condition, drop = FALSE] > 0) >=
    opt$`min-samples`
cat("  genes admitted by the per-group gate:", sum(deg_admitted),
    "of", length(deg_admitted), "\n")
                                                                              
                                                                             
                                                                               
                                                                             
                           
deg_fit_ok <- sum(deg_admitted) > 0
if (!deg_fit_ok) {
  cat("  WARN: no gene passed the admission gate; no gene is tested, and deg.tsv carries the reason.\n")
}

if (deg_fit_ok) {
  dds <- DESeq2::DESeqDataSetFromMatrix(countData = gene_mat[deg_admitted, , drop = FALSE],
                                         colData = design_genes,
                                         design = deg_formula)
                                                                           
                                                                                
                                                                                 
                                                                             
                                                                               
                                                                           
                                    
  deg_sf_note <- paste0("median-of-ratios size factors, estimated on the ADMITTED ",
                      "genes only (the matrix the gate left), not on the whole ",
                      "input")
  dds <- tryCatch(DESeq2::estimateSizeFactors(dds),
                  error = function(e) {
                    cat("  WARN: the gene size factors could not be estimated (",
                        conditionMessage(e),
                        "); using DESeq2's poscounts estimator.\n")
                    deg_sf_note <<- "poscounts size factors (the median-of-ratios estimator failed)"
                    DESeq2::estimateSizeFactors(dds, type = "poscounts")
                  })

                                                                       
                                                                            
                                                                            
                                                                             
                                                                            
                                                                           
                                                    
  disp_note <- paste0("parametric dispersion fit, fitted on the ADMITTED genes ",
                      "only (the matrix the gate left), not on the whole input")
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

                                                                       
                                                                               
                                                                              
                                                                          
                                                    
  dds <- DESeq2::nbinomWaldTest(dds)
  dres <- DESeq2::results(dds, independentFiltering = FALSE,
                          contrast = c("condition", case_condition, control_condition))
  dres_df <- as.data.frame(dres)
                                                                     
                                                    
   
                                                                           
                                                                       
                                                                            
                                                                           
                                                                          
                                               
   
                                                                            
                                                                               
                                                                          
                                                                           
                                                                           
                                                                          
              
   
                                                                          
                                                                             
                                                                        
                                                                           
                                                                              
                                                    
   
                                                                           
                                                                         
   
                                                                             
                                                                          
                                                                        
                                                               
   
                                                                       
                                                                         
                                                                              
                                                                           
                                                                              
  deg_shrunk_df <- NULL
  deg_shrink_note <- "no shrinkage was applied; log2FoldChange_shrunk and lfcSE_shrunk are NA"
  deg_coef <- grep("^condition_", DESeq2::resultsNames(dds), value = TRUE)
  if (length(deg_coef) == 1) {
    deg_shrunk <- tryCatch(
      DESeq2::lfcShrink(dds, coef = deg_coef, type = "normal", quiet = TRUE),
      error = function(e) {
        cat("  WARN: the gene log2 fold changes could not be shrunk (",
            conditionMessage(e), "); the unshrunk estimates are reported.\n")
        NULL
      })
    if (!is.null(deg_shrunk)) {
      deg_shrunk_df <- as.data.frame(deg_shrunk)
                                                                              
                                                                            
                                                              
      if (!identical(rownames(deg_shrunk_df), rownames(dres_df))) {
        stop("ERROR: the shrunk gene fold changes and the Wald test's results are ",
             "in a different order; the effect size would be paired with the ",
             "wrong p-value.")
      }
      deg_shrink_note <- paste0(
        "log2FoldChange and lfcSE are the unshrunk maximum-likelihood estimate ",
        "and its standard error, so stat = log2FoldChange / lfcSE holds exactly; ",
        "log2FoldChange_shrunk and lfcSE_shrunk are DESeq2's normal-prior shrunk ",
        "estimate and its posterior sd (lfcShrink, type = \"normal\"); stat, ",
        "pvalue and padj are the Wald test's and are bit-identical to results()'")
    }
  }
                                                                           
                                                                          
                                                  
} else {
  dres_df <- data.frame(baseMean = numeric(0), log2FoldChange = numeric(0),
                        lfcSE = numeric(0), stat = numeric(0), pvalue = numeric(0),
                        padj = numeric(0), row.names = character(0))
  deg_shrunk_df <- NULL
  deg_shrink_note <- "no gene was fitted, so no effect size is reported"
  deg_sf_note <- "no size factors were estimated, because no gene was fitted"
  disp_note <- "no dispersion was fitted, because no gene was fitted"
}

deg_out <- data.frame(
  gene_id                = gene_ids_all,
  case_condition         = rep(case_condition, length(gene_ids_all)),
  control_condition      = rep(control_condition, length(gene_ids_all)),
  baseMean               = NA_real_,
  log2FoldChange         = NA_real_,
  log2FoldChange_shrunk  = NA_real_,
  lfcSE                  = NA_real_,
  lfcSE_shrunk           = NA_real_,
  stat                  = NA_real_,
  pvalue                 = NA_real_,
  padj                   = NA_real_,
  n_case                 = rep(sum(design$condition == case_condition), length(gene_ids_all)),
  n_control              = rep(sum(design$condition == control_condition), length(gene_ids_all)),
  independent_filter_threshold = NA_real_,
  n_independent_filtered = 0L,
  stringsAsFactors = FALSE
)
                                                                         
                                                      
deg_idx <- match(deg_out$gene_id, rownames(dres_df))
deg_tested <- !is.na(deg_idx)
deg_out$baseMean[deg_tested]       <- dres_df$baseMean[deg_idx[deg_tested]]
deg_out$log2FoldChange[deg_tested] <- dres_df$log2FoldChange[deg_idx[deg_tested]]
deg_out$lfcSE[deg_tested]          <- dres_df$lfcSE[deg_idx[deg_tested]]
deg_out$log2FoldChange_shrunk[deg_tested] <- if (is.null(deg_shrunk_df)) {
  rep(NA_real_, sum(deg_tested))
} else {
  deg_shrunk_df$log2FoldChange[deg_idx[deg_tested]]
}
deg_out$lfcSE_shrunk[deg_tested] <- if (is.null(deg_shrunk_df)) {
  rep(NA_real_, sum(deg_tested))
} else {
  deg_shrunk_df$lfcSE[deg_idx[deg_tested]]
}
deg_out$stat[deg_tested]           <- dres_df$stat[deg_idx[deg_tested]]
deg_out$pvalue[deg_tested]         <- dres_df$pvalue[deg_idx[deg_tested]]
deg_out$padj[deg_tested]           <- dres_df$padj[deg_idx[deg_tested]]
                                                                            
                                                                     
deg_out$padj_na_reason <- ifelse(!deg_admitted, "not_admitted_by_gate",
                          ifelse(!deg_tested, "not_tested",
                          ifelse(is.na(deg_out$pvalue), "no_p_value", "none")))
                                               
deg_out$model_name <- ifelse(deg_tested, "deseq2_wald", "not_tested")
                                                                            
                                                                              
                                                                       
                                                                             
                                                        
check_stat_coherence <- function(label, out, tested) {
  if (!any(tested)) return(invisible(NULL))
  lfc <- out$log2FoldChange[tested]
  se  <- out$lfcSE[tested]
  st  <- out$stat[tested]
  ok  <- is.finite(lfc) & is.finite(se) & is.finite(st) & (se > 0)
  if (!any(ok)) return(invisible(NULL))
  recomputed <- (lfc / se)[ok]
  if (!isTRUE(all.equal(unname(st[ok]), unname(recomputed), tolerance = 1e-6))) {
    stop("ERROR: ", label, ": stat is not log2FoldChange / lfcSE for the ",
         "maximum-likelihood estimates (max relative difference ",
         format(max(abs(st[ok] - recomputed) / pmax(abs(st[ok]), 1e-12)),
                scientific = TRUE),
         "). The reported effect size and the reported statistic would ",
         "describe two different fits.")
  }
  invisible(NULL)
}
check_stat_coherence("gene track", deg_out, deg_tested)
check_stat_coherence("transcript track", det_out, det_tested)

deg_out$model_parameters <- paste0(
  "DESeq2 on the gene-level counts (the transcript counts summed per gene); ",
  "negative binomial Wald test; results(independentFiltering = FALSE); ",
  deg_gate_note,
  "the genes the gate excluded ",
  "carry padj_na_reason = not_admitted_by_gate and are kept out of the ",
  "adjustment's denominator; ",
  deg_shrink_note, "; ",
  deg_sf_note, "; ",
  contrast_note, "; ", dpl_batch_note, "; ", disp_note)

deg_out <- attach_gene_name(deg_out)
write.table(deg_out,
            file = file.path(opt$`output-dir`, "deg.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("[telescope_diff]  wrote deg.tsv (", nrow(deg_out), "genes,",
    sum(deg_out$padj_na_reason == "none"), "tested)\n")
}

                                                                       
                                                                             
                                                                       
if (want_dpl) {
                                                                         
                                                                            
                                                                            
                  
cat("\n[telescope_diff] length track: ΔPAL via an unweighted two-sample linear model on the sample means, with limma's empirical Bayes moderation\n")
cat("  aggregating to (transcript × sample) BEFORE fitting to avoid pseudoreplication.\n")

                                                                        
                                                                           
                                                                             
              
 
                                    
                                               
                                                               
                       
                   
                                                                   
                                                                             
                                                                          
 
                                                                             
                                                              
                                                                          
                                                                            
                                                 

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

                                                                            
                                                                             
                                                                            
                                                                         
units <- dpl_units(unit_agg, design[, c("sample_id", "condition")],
                   min_weight = opt$`min-cell-weight`, min_neff = opt$`min-neff`)
cat("  (transcript × sample) cells:", nrow(units),
    " passing the quality gate:", sum(units$passed_quality_gate), "\n")

                                                                            
tx_ids_all <- sort(unique(unit_agg$transcript_id))
gene_of <- tapply(unit_agg$gene_id, unit_agg$transcript_id, function(x) x[1])[tx_ids_all]

cat("  design:", paste(colnames(dpl_design_matrix), collapse = ", "),
    "|", dpl_batch_note, "\n")

                                                                            
                                                                        
                                                                             
                                                                         
                  
sample_order <- design$sample_id
y_mat <- dpl_matrices(units, sample_order, tx_ids_all, "mean_log2_pal", NULL)$y
                                                                           
                                                                              
                                                                        
                 
W_mat <- dpl_matrices(units, sample_order, tx_ids_all, "W", NULL)$y

                                                                             
                                                                             
                                              
n_adm <- units[units$passed_quality_gate, ] %>%
  dplyr::count(transcript_id, condition)
                                          
n_adm_case <- n_adm$n[match(paste(tx_ids_all, case_condition),
                            paste(n_adm$transcript_id, n_adm$condition))]
n_adm_ctrl <- n_adm$n[match(paste(tx_ids_all, control_condition),
                            paste(n_adm$transcript_id, n_adm$condition))]
n_adm_case[is.na(n_adm_case)] <- 0
n_adm_ctrl[is.na(n_adm_ctrl)] <- 0

                                                     
 
                                                                  
                                                                   
                                                                            
                                                                         
                                                                            
                                                                           
                                                                            
                                    
 
                                                                   
                                                                             
                                                                   
min_samples <- opt$`min-samples`
too_few_cells <- n_adm_case < min_samples | n_adm_ctrl < min_samples

cat("  fitting", length(tx_ids_all), "transcripts…\n")
                                                                           
                                                                           
                                                                     
                                                                           
                                                                          
                                      
both_raw <- dpl_fit_both(y_mat, dpl_design_matrix, NULL, moderate = FALSE)
fit_raw  <- both_raw$raw

se_raw  <- fit_raw$se
t_raw   <- fit_raw$t

                                                                     
                                                                           
                                                                            
                                                                           
                                                                             
                                                                        
                                                                          
                                                                          
                         
sigma2_raw <- fit_raw$sigma2_raw
fit_failed <- !is.finite(t_raw) | !is.finite(se_raw) | (sigma2_raw < 1e-8)
masked <- fit_failed | too_few_cells

                                                      
 
                                                                            
                                                                              
                                                                            
                                                                             
                                                                            
                                                                         
                                 
 
                                                                         
                                                                             
                                                                      
                                                                                       
                                                                           
                                                                      
                                                                         
                                                                            
                                             
testable <- !masked
cov_depth <- dpl_depth_trend(W_mat)
both <- dpl_fit_both(y_mat, dpl_design_matrix, NULL,
                     testable = testable, trend = cov_depth)
fit  <- both$moderated
                                                                             
                                                                          
                                                                            
                                                               
dpl_ebayes_note <- if (nzchar(both$ebayes_note)) paste0(both$ebayes_note, "; ") else ""

                                                                    
 
                                                                           
                                                            
 
                                                                            
                                                                        
                                                                          
                                             
                                                                           
 
                                                                       
                                                                            
                                     
dpl_mod_arm <- DPL_MODEL_NAME
if (is.null(fit) || fit$n_fitted == 0) {
  dpl_mod_arm <- "none"
  dpl_mod_desc <- paste0(
    "no model was fitted: no transcript was testable, so the length track has ",
    "no contrast to report")
} else if (!both$trend_used) {
  dpl_mod_arm <- DPL_MODEL_NAME_NO_TREND
  dpl_mod_desc <- paste0(
    "limma eBayes with robust = TRUE and NO variance trend: the read support ",
    "could not be fitted as a trend covariate, so the moderation went on ",
    "without the trend")
} else {
  dpl_mod_desc <- paste0(
    "limma eBayes with trend = the mean over a transcript's cells of the log2 ",
    "of the summed posterior weight of the cell, and robust = TRUE; the prior ",
    "variance is fitted against that read support instead of the average ",
    "expression, so s2.prior and df.prior are per transcript: s2.prior ",
    dpl_prior_summary(fit$s2_prior), ", df.prior ", dpl_prior_summary(fit$df_prior))
}
                                                                          
                                                                           
                                                                             
                                                                             
                                             
if (dpl_mod_arm == DPL_MODEL_NAME_NO_TREND) {
  cat("  WARN: the DPL moderation fitted NO variance trend. The model actually",
      "used is `DT-EB-no-trend` (limma::eBayes, robust = TRUE, no",
      "trend), NOT `DT-EB`. The reason:", both$ebayes_note, "\n")
}
                                                                      
                                                                        
                                                                         
                             
cat("  DPL model: ", dpl_mod_arm, "\n")

se_mod  <- fit$se
t_mod   <- fit$t
df_post <- fit$df_post
p_mod   <- fit$p

                                                                            
                                                                          
                                                                            
                                                                       
                        
.tb <- which(testable)
.d_t <- 0; .d_p <- 0
if (length(.tb) > 0) {
  .d_t <- max(abs(fit_raw$t[.tb] - both$raw$t[.tb]), na.rm = TRUE)
  .d_p <- max(abs(fit_raw$p[.tb] - both$raw$p[.tb]), na.rm = TRUE)
  if (.d_t > 1e-12 || .d_p > 1e-12) {
    stop("ERROR: the raw statistic of the testable rows differs between the ",
         "whole-transcriptome fit and the testable-subset fit (max |d t| = ",
         format(.d_t, digits = 6), ", max |d p| = ", format(.d_p, digits = 6),
         "). The two arms would come from two different regressions.")
  }
}
cat("  the two fits agree on the raw arm of the testable rows:",
    "max |d t| =", format(.d_t, digits = 3),
    ", max |d p| =", format(.d_p, digits = 3), "\n")
t_mod[masked] <- NA_real_
p_mod[masked] <- NA_real_
q_mod <- stats::p.adjust(p_mod, method = "BH")
cat("  testable transcripts:", sum(!masked), "(both groups with >=",
    min_samples, "admissible cells; testable transcripts)\n")

                                                                          
                                                                         
                                                                             
                                                                            
                                                                           
       
 
                                                                          
                                                                            
                                                                         
                                                                         
 
                                                                    
                                                                          
                                                                             
                                
nt_eff <- dpl_effect_nt(y_mat, design$condition == case_condition,
                        design$condition == control_condition)
                                                                            
                                                                             
                                                                           
                                                                            
                             
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
                                                                     
                                                                        
                                                                         
                                                                       
                                                                           
                                                                           
                                                      
for (.cn in c("geometric_mean_ratio", "geometric_mean_control_nt",
              "geometric_mean_case_nt", "delta_geometric_mean_nt",
              "ci_lo_log2_pal", "ci_hi_log2_pal", "ci_lo_ratio", "ci_hi_ratio",
              "ci_lo_delta_geometric_mean_nt",
              "ci_hi_delta_geometric_mean_nt")) {
  nt_eff[[.cn]][fit_failed] <- NA_real_
}

                                                                        
                                                                           
                                                                      
no_cells <- n_adm_case == 0 & n_adm_ctrl == 0
pal_na_reason <- ifelse(
  !is.na(p_mod), "none",
  ifelse(no_cells, "no admissible cells at all",
         ifelse(fit_failed, "degenerate_zero_variance",
                "too few admissible cells in one group")))

                                                                          
                                                                           
                                                                         
                                                                           
                                                                       
                                                                   
 
                                                                             
                                                                              
                                                                         
                                              
dpl_base_fit_clause <- if (dpl_spec$batch_used) {
  "the fit is a linear model with a batch term, so its `case` coefficient is the BATCH-ADJUSTED difference, not the raw difference of the two group means; "
} else {
  "the fit is an unweighted two-sample linear model on the transcript x sample means, so its `case` coefficient is the difference of the two group means; "
}
                                                                           
                                                                             
                                                                            
                                                                          
                                                                             
                                             
dpl_audit_clause <- ""
                                                                             
                                                                             
                                                          
dpl_confounded <- isTRUE(dpl_spec$confounded)
dpl_confound_clause <- if (dpl_confounded) {
  paste0("CONFOUNDED DESIGN: batch and condition are completely confounded; ",
         "treatment-specific effects are not identifiable; the contrast in ",
         "this table is a batch-and-condition difference; ")
} else {
  ""
}
                                                                         
                                                                         
                                                      
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
                                                                          
                                                                             
                                                                             
                                                                           
                                                                          
                                                 
   
                                                                           
                                                       
  delta_log2_pal             = fit$estimate,
  se                         = se_mod,
  t_stat                     = t_mod,
  df_post                    = df_post,
  pvalue                     = p_mod,
  padj                       = q_mod,
                                                                         
                                                                          
                                                                              
                                                                       
                                                  
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
                                                                           
                                                                        
                                                                            
                                                                      
  confounded_design          = dpl_confounded,
                                                                      
                                                                            
                                                                    
                                     
  model_name                 = dpl_mod_arm,
  model_parameters           = paste0(
    "model: ", dpl_mod_arm, " (", dpl_mod_desc, "); ",
    dpl_base_fit_clause,
    "the moderation is fitted on the testable rows only (",
    sum(!masked), " transcripts), so no untestable row contributes to the prior; ",
    dpl_ebayes_note, dpl_audit_clause,
    "cell admission gate: min_weight = ", format(opt$`min-cell-weight`),
    ", min_neff = ", format(opt$`min-neff`),
    ", min_samples = ", min_samples,
    ", sigma2_floor = 1e-8; ",
    dpl_effect_clause,
    "independent filtering: not applied (the length track drops no row, so the ",
    "independent filter threshold is NA and nothing was filtered); ",
    contrast_note, "; ", dpl_batch_note, dpl_confound_clause),
  stringsAsFactors = FALSE
)
                                                                            
                                                                           
                                                                            
                                                       
 
                                                                             
                                                                    
                                                                      
                
 
                                                                              
                                                                           
                                                                           
                                                                          
                                                                           
                                                                          
                             
.no_mod_est <- is.na(pal_out$delta_log2_pal)
pal_out$delta_log2_pal[.no_mod_est] <- fit_raw$estimate[.no_mod_est]
pal_out$delta_log2_pal[fit_failed] <- NA_real_
pal_out$model_name[is.na(pal_out$t_stat)] <- "not_testable"

pal_out <- attach_gene_name(pal_out)
write.table(pal_out,
            file = file.path(opt$`output-dir`, "dpl.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("[telescope_diff]  wrote dpl.tsv (",
    nrow(pal_out), "transcripts,", sum(!is.na(pal_out$pvalue)), "tested)\n")
cat("[telescope_diff]  limma eBayes priors (per transcript, summarised): ",
    "s2.prior =", dpl_prior_summary(fit$s2_prior),
    " df.prior =", dpl_prior_summary(fit$df_prior), "\n")

                                                             
if (sum(is.finite(fit$sigma2_post)) > 0) {
  stopifnot(all(is.finite(fit$sigma2_post[is.finite(fit$sigma2_post)])))
  stopifnot(all(is.finite(fit$sigma2_post[is.finite(fit$sigma2_post)]) &
                fit$sigma2_post[is.finite(fit$sigma2_post)] > 0))
}
}

                                                                       
                                                                  
                                                                       
if (!is.null(opt$`lapa-dir`)) {
cat("\n[telescope_diff] alternative poly(A) track: LAPA's per-sample counts\n")

                                                                             
                           
                                                                             
                                                                            
                                               
 
                                                                       
                                                         
 
                                                                        
                                                                          
                                                                          
                                          
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

                                                                          
                                                                             
                                                                         
                                                                           
                                                                            
                                                                    
                                                                      
                                    
 
                                                                             
                                                                             
                                                                              
                                                                              
                                                         
                                                                              
                                                                            
                                                                       
                                           
 
                                                                             
                                                      
lapa_dir <- opt$`lapa-dir`
if (!dir.exists(lapa_dir)) {
  stop("ERROR: the LAPA output directory does not exist: ", lapa_dir)
}
lapa_beds <- list.files(file.path(lapa_dir, "sample"),
                        pattern = "\\.bed$", full.names = TRUE)
lapa_source <- paste0(lapa_dir, "/sample")
if (length(lapa_beds) == 0) {
                                                                 
  one_down <- list.dirs(lapa_dir, recursive = FALSE)
  for (d in one_down) {
    cand <- list.files(file.path(d, "sample"), pattern = "\\.bed$", full.names = TRUE)
    if (length(cand) > 0) {
      lapa_beds <- cand
      lapa_source <- paste0(d, "/sample")
      break
    }
  }
}
lapa_replicated <- TRUE
if (length(lapa_beds) == 0) {
                                                                            
                                                                           
                                        
  lapa_beds <- list.files(file.path(lapa_dir, "raw_sample"),
                          pattern = "\\.bed$", full.names = TRUE)
  lapa_source <- paste0(lapa_dir, "/raw_sample")
  lapa_replicated <- FALSE
}
lapa_beds <- sort(unique(lapa_beds))
if (length(lapa_beds) == 0) {
  stop("ERROR: no LAPA cluster file (*.bed) under ", lapa_dir,
       "/sample, */sample or ", lapa_dir, "/raw_sample")
}
lapa_samples <- sub("\\.bed$", "", basename(lapa_beds))
cat("  LAPA samples:", length(lapa_beds), "read from", lapa_source, "\n")

                                                                             
                                                                            
                                                              
                                                                              
                                                                            
                                                                        
                                                                                      
                                                                             
                                                                            
                          
 
                                                                           
                                                                                
                                                                         
                                                          
                                                               
lapa_source_receipt <- lapa_source
apa_out_dir <- opt$`output-dir`
if (!is.null(apa_out_dir) && nzchar(apa_out_dir)) {
  apa_abs_out <- normalizePath(apa_out_dir, mustWork = FALSE)
  apa_abs_src <- normalizePath(lapa_source, mustWork = FALSE)
  apa_prefix <- paste0(.Platform$file.sep, "lapa", .Platform$file.sep)
  if (startsWith(apa_abs_src, paste0(apa_abs_out, apa_prefix))) {
    lapa_source_receipt <- substring(apa_abs_src, nchar(apa_abs_out) + 2L)
  }
}
if (!lapa_replicated) {
  cat("  WARN: no replicated cluster files (sample/); the UNREPLICATED calls ",
      "(raw_sample/) are used, and apa.tsv's receipt says so.\n")
}

lapa_rows <- lapply(lapa_beds, read_lapa_sample)
names(lapa_rows) <- lapa_samples
lapa_rows <- lapa_rows[!vapply(lapa_rows, is.null, logical(1))]

                                                                          
                                                                           
          
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

                                                                         
                                                                    
                                                                      
                                                                           
                                                                            
                                                                             
                                                          
 
                                                                            
                                                                             
                                                                          
               
apa_site_ids <- sort(unique(unlist(lapply(lapa_rows, function(d) d$site_id))))
apa_mat <- matrix(NA_real_, nrow = length(apa_site_ids), ncol = length(apa_samples),
                   dimnames = list(apa_site_ids, apa_samples))
                                                                       
                                                                      
                                                                           
                                                                                 
                                                                           
                                                                             
                                                                           
                                                                          
                                                                 
                                                                    
apa_sample_min <- max(5, floor(opt$`min-counts` / 4))
cat("  poly(A) single-sample gene-read threshold:", apa_sample_min,
    "(min-counts is ", opt$`min-counts`, ")\n")
apa_counts <- matrix(NA_real_, nrow = length(apa_site_ids), ncol = length(apa_samples),
                      dimnames = list(apa_site_ids, apa_samples))
                                                                           
                                                                           
                                                                           
apa_present <- matrix(FALSE, nrow = length(apa_site_ids), ncol = length(apa_samples),
                       dimnames = list(apa_site_ids, apa_samples))
                                                                             
                                                                       
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
                                                                           
                                                                        
                                                                      
                                                                         
                                                                           
                                                                             
                                             
  d2 <- d[match(apa_site_ids, d$site_id), ]
  j <- match(d2$site_id, apa_site_ids)
  cc <- match(s, apa_samples)
  present <- !is.na(j)
  apa_mat[cbind(j[present], cc)] <- suppressWarnings(as.numeric(d2$usage))[present]
  apa_counts[cbind(j[present], cc)] <- suppressWarnings(as.numeric(d2$count))[present]
  apa_present[cbind(j[present], cc)] <- TRUE
}

                                                                          
                                          
apa_gene <- rep(NA_character_, length(apa_site_ids))
for (s in apa_samples) {
  d <- lapa_rows[[s]]
  d <- d[match(apa_site_ids, d$site_id), ]
  g <- d$gene_id
  take <- is.na(apa_gene) & !is.na(g) & nzchar(g)
  apa_gene[take] <- g[take]
}
                                                                       
apa_coord <- rep(NA_real_, length(apa_site_ids))
for (s in apa_samples) {
  d <- lapa_rows[[s]]
  d <- d[match(apa_site_ids, d$site_id), ]
  pos <- suppressWarnings(as.numeric(d$polyA_site))
  take <- is.na(apa_coord) & !is.na(pos)
  apa_coord[take] <- pos[take]
}

                                                                          
                                                                         
                                             
for (cc in seq_along(apa_samples)) {
  s <- apa_samples[cc]
  totals <- apa_gene_total[[s]]
  genes <- apa_gene
  known <- !is.na(genes) & genes %in% names(totals)
  tot <- rep(NA_real_, length(genes))
  tot[known] <- totals[genes[known]]
  absent <- !apa_present[, cc]
  usable <- absent & !is.na(tot) & tot >= apa_sample_min
  apa_mat[usable, cc] <- 0
}
                                                                          
for (cc in seq_along(apa_samples)) {
  s <- apa_samples[cc]
  totals <- apa_gene_total[[s]]
  genes <- apa_gene
  known <- !is.na(genes) & genes %in% names(totals)
  tot <- rep(NA_real_, length(genes))
  tot[known] <- totals[genes[known]]
  too_small <- apa_present[, cc] & (is.na(tot) | tot < apa_sample_min)
  apa_mat[too_small, cc] <- NA_real_
}

                                                                        
                                                            
                                                                        
                                                                         
                                                                          
                                                                        
                                                           
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
                                                                             
                                                                          
                                                                           
                                                            
apa_case_samples    <- apa_samples[as.character(design$condition[
  match(apa_samples, design$sample_id)]) == case_condition]
apa_control_samples <- apa_samples[as.character(design$condition[
  match(apa_samples, design$sample_id)]) == control_condition]
apa_n_case    <- rowSums(!is.na(apa_mat[, apa_case_samples, drop = FALSE]))
apa_n_control <- rowSums(!is.na(apa_mat[, apa_control_samples, drop = FALSE]))
                                                                         
                                                                            
                                                                             
                                                                           
                                                   
apa_site_total <- rowSums(apa_counts, na.rm = TRUE)
apa_site_too_small <- apa_site_total < 10
apa_too_few <- apa_n_case < opt$`min-samples` | apa_n_control < opt$`min-samples` |
               apa_site_too_small
apa_y <- apa_mat
apa_y[apa_too_few, ] <- NA_real_
                                                                          
                                                                    
                                                                         
                                                                     
                                                                          
                                                           
pau_case    <- rowMeans(apa_y[, apa_case_samples, drop = FALSE], na.rm = TRUE)
pau_control <- rowMeans(apa_y[, apa_control_samples, drop = FALSE], na.rm = TRUE)
pau_case[!is.finite(pau_case)] <- NA_real_
pau_control[!is.finite(pau_control)] <- NA_real_
apa_no_usable <- apa_n_case == 0 & apa_n_control == 0

                                                                           
                                                                           
                                                          
 
                                                                      
                                                                           
                                                                         
                                                                       
if (lapa_replicated) {
  apa_n_replicated_clusters <- length(apa_site_ids)
  apa_replicated_clause <- paste0(
    "the replicated cluster files that were read hold ",
    apa_n_replicated_clusters, " distinct poly(A) sites (the count of distinct ",
    "replicated clusters in the files that were read, not a genome-wide ",
    "cluster count); ")
} else {
  apa_n_replicated_clusters <- 0L
  apa_replicated_clause <- paste0(
    "NO replicated cluster file was read: the directory held no replicated ",
    "calls, so the unreplicated ones (raw_sample/) were used and 0 replicated ",
    "clusters were read; ")
}
apa_lapa_params_clause <- paste0(
  "LAPA's parameters: counting_method = ", opt$`lapa-counting-method`,
  "; min_replication_rate = ", opt$`lapa-min-replication-rate`,
  "; ", apa_replicated_clause)

                                                                         
                                                                              
                                                                              
                           
 
                                                                            
                                                                         
                      
                                                                       
                                                              
                                    
                                                                       
 
                                                                      
                                                                         
                                                                          
                                                                      
                                                                       
                                                         
apa_counts_int <- apa_counts
apa_counts_int[is.na(apa_counts_int)] <- 0
apa_counts_int <- round(apa_counts_int)
storage.mode(apa_counts_int) <- "integer"
n_sites <- length(apa_site_ids)

                                                                       
                                                                        
                                                                           
                                                                          
                                                                          
                                                                         
                
apa_dx_has_gene <- !is.na(apa_gene) & nzchar(apa_gene)
apa_dx_admit <- !apa_too_few & (apa_site_total > 0) & apa_dx_has_gene
cat("  poly(A) sites admitted by the site-level gate:", sum(apa_dx_admit),
    "of", n_sites, "\n")

                                                                           
                                                                          
                                                                           
                                                                           
                                                                   
                                                                     
                                                                     
                                                                         
                                                                            
 
                                                                      
                                                                        
                                                                          
         
apa_dx_genes <- sort(unique(apa_gene[apa_dx_admit]))
apa_dx_idx <- which(apa_dx_admit)
if (length(apa_dx_idx) > 0) {
  apa_dx_sum_by_gene <- rowsum(apa_counts_int[apa_dx_idx, , drop = FALSE],
                               group = apa_gene[apa_dx_idx], reorder = TRUE)
} else {
  apa_dx_sum_by_gene <- matrix(0, nrow = 0, ncol = length(apa_samples),
                               dimnames = list(NULL, apa_samples))
}
apa_dx_remainder <- matrix(0, nrow = length(apa_dx_genes),
                            ncol = length(apa_samples),
                            dimnames = list(apa_dx_genes, apa_samples))
for (cc in seq_along(apa_samples)) {
  s <- apa_samples[cc]
  totals <- apa_gene_total[[s]]
  g <- apa_dx_genes
  known <- g %in% names(totals)
  gt <- numeric(length(g))
  gt[known] <- totals[g[known]]
  gt[!is.finite(gt)] <- 0
  rs <- apa_dx_sum_by_gene[g, s]
  rs[is.na(rs)] <- 0
  apa_dx_remainder[, cc] <- pmax(0, gt - rs)
}
apa_dx_bucket_genes <- rownames(apa_dx_remainder)[rowSums(apa_dx_remainder) > 0]
cat("  poly(A) genes given a `__other_pas__` remainder row:",
    length(apa_dx_bucket_genes), "of", length(apa_dx_genes), "\n")

                                                                          
                                                                        
                                                                  
                                                                           
                   
apa_dx_n_admitted <- table(apa_gene[apa_dx_idx])
apa_dx_singleton_genes <- names(apa_dx_n_admitted)[
  apa_dx_n_admitted == 1L &
    !(names(apa_dx_n_admitted) %in% apa_dx_bucket_genes)]
cat("  poly(A) genes with one admitted site and no remainder (no contrast):",
    length(apa_dx_singleton_genes), "\n")

                                                                         
                                        
 
                                                                            
                                                                             
                                                                             
                                                                           
                                                  
apa_dx_bucket_feature_ids <- if (length(apa_dx_bucket_genes) > 0) {
  paste0(apa_dx_bucket_genes, ":__other_pas__")
} else {
  character(0)
}
apa_dx_fit <- rbind(
  data.frame(featureID = apa_site_ids[apa_dx_idx],
             groupID = apa_gene[apa_dx_idx],
             site_index = apa_dx_idx,
             is_bucket = rep(FALSE, length(apa_dx_idx)),
             stringsAsFactors = FALSE),
  data.frame(featureID = apa_dx_bucket_feature_ids,
             groupID = apa_dx_bucket_genes,
             site_index = rep(NA_integer_, length(apa_dx_bucket_genes)),
             is_bucket = rep(TRUE, length(apa_dx_bucket_genes)),
             stringsAsFactors = FALSE)
)
apa_dx_counts <- rbind(
  apa_counts_int[apa_dx_idx, , drop = FALSE],
  apa_dx_remainder[apa_dx_bucket_genes, , drop = FALSE]
)
rownames(apa_dx_counts) <- apa_dx_fit$featureID
                                                                       
                                                                     
apa_dx_singleton_sites <- rep(FALSE, n_sites)
apa_dx_singleton_rows <- which(!is.na(apa_dx_fit$site_index) &
                                 apa_dx_fit$groupID %in% apa_dx_singleton_genes)
if (length(apa_dx_singleton_rows) > 0) {
  apa_dx_singleton_sites[apa_dx_fit$site_index[apa_dx_singleton_rows]] <- TRUE
}
                                                                      
apa_dx_keep <- !(apa_dx_fit$groupID %in% apa_dx_singleton_genes)
apa_dx_fit <- apa_dx_fit[apa_dx_keep, , drop = FALSE]
apa_dx_counts <- apa_dx_counts[apa_dx_keep, , drop = FALSE]

                                                                          
                                                                       
                                                      
apa_dx_sample_data <- data.frame(
  sample_id = apa_samples,
  condition = factor(ifelse(apa_design_mat[, "case"] > 0,
                             case_condition, control_condition),
                     levels = c(control_condition, case_condition)),
  stringsAsFactors = FALSE
)
rownames(apa_dx_sample_data) <- apa_samples

                                                                      
                                                  
                                                                       
                                                                          
                      
apa_dx_key <- gsub(" |:", "", apa_dx_fit$featureID)
if (anyDuplicated(apa_dx_key) > 0) {
  stop("ERROR: two poly(A) features have the same DEXSeq identifier after ",
       "DEXSeq's own normalisation (spaces and colons removed); the results ",
       "could not be matched back to the site universe.")
}

                                                                             
                                                                          
                                                                          
                                                                         
                                                   
apa_dx_fit_made <- nrow(apa_dx_counts) > 0
apa_dx_dexseq_version <- as.character(utils::packageVersion("DEXSeq"))
apa_dx_disp_note <- "no dispersion was fitted, because no poly(A) site was fitted"
apa_dx_fit_type <- NA_character_
if (!apa_dx_fit_made) {
  cat("  WARN: no poly(A) site is testable; apa.tsv is written with every row ",
      "carrying its reason, and the other tables are unaffected.\n")
} else {
  dxd <- DEXSeqDataSet(
    countData  = apa_dx_counts,
    sampleData = apa_dx_sample_data,
    design     = ~ sample + exon + condition:exon,
    featureID  = apa_dx_fit$featureID,
    groupID    = apa_dx_fit$groupID
  )
  cat("  running DEXSeq size factors + dispersion + test…\n")
  dxd <- estimateSizeFactors(dxd)
                                                                            
                                                                           
                                                                     
                                                                        
                                                     
  disp_result <- tryCatch({
    list(dxd = estimateDispersions(dxd, quiet = TRUE),
         note = "parametric dispersion fit")
  }, error = function(e) {
    cat("  WARN: the dispersion trend could not be fitted (",
        conditionMessage(e),
        "); keeping DEXSeq's gene-wise dispersion estimates.\n")
    dxd_fallback <- estimateDispersionsGeneEst(
      dxd, quiet = TRUE, modelMatrix = rank_reduced_model_matrix(dxd))
    dispersions(dxd_fallback) <- mcols(dxd_fallback)$dispGeneEst
    list(dxd = dxd_fallback,
         note = "gene-wise dispersion estimates (the dispersion trend could not be fitted)")
  })
  dxd <- disp_result$dxd
  apa_dx_disp_note <- disp_result$note
                                                                           
                                                                          
                                                                             
  apa_dx_fit_type <- tryCatch(attr(dispersionFunction(dxd), "fitType"),
                              error = function(e) NA_character_)
  if (is.null(apa_dx_fit_type) || !nzchar(as.character(apa_dx_fit_type))) {
    apa_dx_fit_type <- NA_character_
  }
  cat("  dispersion:", apa_dx_disp_note, "; fitType:",
      if (is.na(apa_dx_fit_type)) "not reported by the fitted object" else apa_dx_fit_type,
      "\n")
  dxd <- testForDEU(dxd,
                    reducedModel = ~ sample + exon,
                    fullModel    = ~ sample + exon + condition:exon)
                                                                          
                                               
  dxd <- tryCatch(estimateExonFoldChanges(dxd, fitExpToVar = "condition"),
                  error = function(e) {
                    cat("  WARN: the exon fold changes could not be estimated (",
                        conditionMessage(e),
                        "); dexseq_log2fold is NA for every row.\n")
                    dxd
                  })
}

                                                                          
                                                                   
                                                          
apa_dx_pvalue <- rep(NA_real_, n_sites)
apa_dx_lfc    <- rep(NA_real_, n_sites)
apa_dx_stat   <- rep(NA_real_, n_sites)
if (apa_dx_fit_made) {
  dxr <- DEXSeqResults(dxd, independentFiltering = FALSE)
  dx_key <- gsub(" |:", "", as.character(dxr$featureID))
  m <- match(apa_dx_key, dx_key)
  keep <- which(!apa_dx_fit$is_bucket & !is.na(m))
  if (length(keep) > 0) {
    si <- apa_dx_fit$site_index[keep]
    apa_dx_pvalue[si] <- as.numeric(dxr$pvalue[m[keep]])
    if ("stat" %in% colnames(dxr)) {
      apa_dx_stat[si] <- as.numeric(dxr$stat[m[keep]])
    }
    lfc_col <- grep("^log2fold_", colnames(dxr), value = TRUE)
    if (length(lfc_col) == 1L) {
      apa_dx_lfc[si] <- as.numeric(dxr[[lfc_col]][m[keep]])
    } else {
      cat("  WARN: DEXSeq returned ", length(lfc_col),
          " effect columns (expected exactly one named log2fold_*); ",
          "dexseq_log2fold is NA for every row.\n")
    }
  }
}

                                                            
                                                                        
apa_dx_tested <- !is.na(apa_dx_pvalue)
apa_dx_padj <- rep(NA_real_, n_sites)
if (any(apa_dx_tested)) {
  apa_dx_padj[apa_dx_tested] <- stats::p.adjust(apa_dx_pvalue[apa_dx_tested],
                                                 method = "BH")
}
                                                                  
                                                                   
                                                                      
                                                                         
                 
APA_DEXSEQ_PADJ_CUTOFF <- 0.05
apa_dx_delta_pau <- pau_case - pau_control
apa_dx_is_significant <- !is.na(apa_dx_padj) &
  apa_dx_padj < APA_DEXSEQ_PADJ_CUTOFF &
  !is.na(apa_dx_delta_pau) &
  abs(apa_dx_delta_pau) >= apa_min_delta_pau

                                                                           
                                                                        
                                                                         
                                                                          
                                                                        
               
 
                                                                         
                                                                      
                                                      
apa_dx_check_n <- 0L
apa_dx_check_skipped <- 0L
apa_dx_check_max <- NA_real_
if (apa_dx_fit_made) {
  dex_cd <- colData(dxd)
  dex_counts <- counts(dxd)
  this_cols <- which(as.character(dex_cd$exon) == "this")
  other_cols <- which(as.character(dex_cd$exon) == "others")
  this_samps <- as.character(dex_cd$sample_id)[this_cols]
  other_order <- match(this_samps, as.character(dex_cd$sample_id)[other_cols])
                                                                         
                                                                           
                                                                        
  own_mat <- dex_counts[, this_cols, drop = FALSE]
  others_mat <- dex_counts[, other_cols[other_order], drop = FALSE]
                                              
  n_fit <- nrow(apa_dx_counts)
  gt_mat <- matrix(0, nrow = n_fit, ncol = length(this_samps),
                   dimnames = list(NULL, this_samps))
  known_mat <- matrix(FALSE, nrow = n_fit, ncol = length(this_samps))
  for (cc in seq_along(this_samps)) {
    totals <- apa_gene_total[[this_samps[cc]]]
    g <- apa_dx_fit$groupID
    known <- g %in% names(totals)
    v <- numeric(length(g))
    v[known] <- totals[g[known]]
    v[!is.finite(v)] <- 0
    gt_mat[, cc] <- v
    known_mat[, cc] <- known
  }
  site_rows <- which(!apa_dx_fit$is_bucket)
  if (length(site_rows) > 0) {
    diff_mat <- others_mat[site_rows, , drop = FALSE] -
      (gt_mat[site_rows, , drop = FALSE] - own_mat[site_rows, , drop = FALSE])
    known_site <- known_mat[site_rows, , drop = FALSE]
    apa_dx_check_n <- sum(known_site)
    apa_dx_check_skipped <- sum(!known_site)
    if (apa_dx_check_n > 0) {
      apa_dx_check_max <- max(abs(diff_mat[known_site]))
    }
  }
}
cat("  poly(A) denominator conservation:", apa_dx_check_n,
    " (site x sample) cells checked,", apa_dx_check_skipped,
    " not checkable (the gene's total is unknown in that sample), max ",
    "|difference| =", apa_dx_check_max, "reads\n")
if (!is.na(apa_dx_check_max) && apa_dx_check_max > 0) {
  cat("  WARN: the poly(A) denominator conservation check found a difference of ",
      apa_dx_check_max, " reads. The denominator of a site is NOT the gene's ",
      "real poly(A) total minus the site's own count for every cell.\n")
}

                                                                                
                                             
apa_model_name <- if (!apa_dx_fit_made) {
  "not_testable"
} else {
  "DEXSeq (LAPA counts)"
}
                                                                           
                                                                         
                                                                        
APA_COLUMNS <- c(
  "gene_id", "gene_name", "site_id", "coord_bp", "case_condition",
  "control_condition",
  "pau_case", "pau_control", "delta_pau", "dexseq_log2fold", "dexseq_stat",
  "pvalue", "padj", "is_significant",
  "n_case", "n_control", "independent_filter_threshold",
  "n_independent_filtered", "padj_na_reason", "model_name", "model_parameters"
)
apa_out <- data.frame(
  gene_id           = apa_gene,
  site_id           = apa_site_ids,
  coord_bp          = apa_coord,
  case_condition    = rep(case_condition, n_sites),
  control_condition = rep(control_condition, n_sites),
  pau_case          = pau_case,
  pau_control       = pau_control,
  delta_pau         = apa_dx_delta_pau,
  dexseq_log2fold   = apa_dx_lfc,
  dexseq_stat       = apa_dx_stat,
  pvalue            = apa_dx_pvalue,
  padj              = apa_dx_padj,
  is_significant    = apa_dx_is_significant,
  n_case            = rep(length(apa_case_samples), n_sites),
  n_control         = rep(length(apa_control_samples), n_sites),
  independent_filter_threshold = NA_real_,
  n_independent_filtered = 0L,
  model_name        = apa_model_name,
  stringsAsFactors = FALSE
)
apa_out$padj_na_reason <- ifelse(
  apa_no_usable, paste0("no usable cell (the gene's poly(A) reads are below the ",
                        "single-sample threshold (", apa_sample_min, ") in every sample)"),
  ifelse(apa_site_too_small,
         "the site's own total poly(A) reads are below the minimum (10)",
  ifelse(apa_too_few, "too few usable samples in one group",
  ifelse(!apa_dx_has_gene, "no gene (LAPA named no gene for this site)",
  ifelse(apa_dx_singleton_sites, "singleton_pas_gene",
  ifelse(is.na(apa_dx_pvalue), "not_tested", "none"))))))
                                                                            
                                                                     
                                                                           
                                                                           
                                                                          
                                                                           
                                                                
apa_batch_note <- if (is.null(batch_vec)) {
  dpl_batch_note
} else {
  paste0("no batch term in the poly(A) design: the input design's batch column (",
         paste(sort(unique(batch_vec)), collapse = ","),
         ") is not adjusted for by this arm")
}
apa_out$model_parameters <- paste0(
  if (!apa_dx_fit_made) {
    paste0("NO MODEL WAS FITTED for this run (no poly(A) site passed the admission gate), so every row carries its reason and no statistic exists; ")
  } else {
    paste0("DEXSeq ", apa_dx_dexseq_version,
           " on LAPA's INTEGER per-PAS read COUNTS (the site's own `count` column); ")
  },
  "a poly(A) site is a FEATURE of DEXSeq and its gene is the group, so the contrast is WITHIN the gene, between the poly(A) sites of that gene; ",
  "the full model is ~ sample + exon + condition:exon and the reduced model is ~ sample + exon; ",
  "the test is DEXSeq's PER-FEATURE LIKELIHOOD RATIO TEST (testForDEU), so pvalue is that LRT's p-value and dexseq_stat is its statistic; ",
  "the dispersion fitType read back from the fitted object is ",
  if (is.na(apa_dx_fit_type)) "not reported by the fitted object" else apa_dx_fit_type,
  " (", apa_dx_disp_note, "); ",
  "the normalisation is DEXSeq's median-of-ratios size factor (estimateSizeFactors), estimated on the fitted matrix; ",
  "the `__other_pas__` remainder bucket: every gene with at least one admitted site gets ONE auxiliary row carrying the gene's real poly(A) total minus the admitted sites' own counts, so the denominator of a site is the gene's real poly(A) output and not the sum of the sites that survived the gate; the bucket rows are STRIPPED from the results and never enter the BH denominator (",
  length(apa_dx_bucket_genes), " bucket row(s) in this run); ",
  "the admission gate has TWO layers: layer 1 -- the site passes the site-level gate (at least ",
  opt$`min-samples`, " usable cells in EACH group AND the site's own total poly(A) reads at least 10) AND its own counts are not all zero AND LAPA named a gene for it; layer 2 -- the gene must have at least two rows in the fit (its admitted sites plus, when the remainder is not empty, one `__other_pas__` row), so a gene with one admitted site and an empty remainder has no contrast, carries padj_na_reason = 'singleton_pas_gene', and is not fitted; ",
  "is_significant is TRUE iff padj < ", APA_DEXSEQ_PADJ_CUTOFF,
  " AND |delta_pau| >= ", apa_min_delta_pau, "; ",
  "pvalue and padj are PURE STATISTICS: the --apa-min-delta-pau threshold does NOT change them, and the adjustment is BH over the tested rows alone; ",
  "delta_pau is the OBSERVED usage difference pau_case - pau_control (the two groups' mean usages on the [0,1] scale) and is NOT the DEXSeq coefficient; the DEXSeq coefficient is the within-gene usage contrast on the log2 scale and is reported separately as dexseq_log2fold; ",
  "the USAGE matrix (the one pau_case and pau_control are computed from) is LAPA's own 'usage' column, i.e. the site's share of its gene's poly(A) reads in that sample (usage == count / gene_count); ",
  "the cluster files read were ", lapa_source_receipt, "; ",
  apa_lapa_params_clause,
  "the SITE UNIVERSE is the COHORT's, not the contrast's: it is every site in ",
  "every cluster file that was read, so a second contrast of the same cohort ",
  "sees the same universe and a difference between the two tables is a ",
  "difference of usage and not of universe; a site that only a sample outside ",
  "this contrast supports therefore appears here as a row with no usable ",
  "cells; ",
  "a cell is usable only when the gene's total ",
  "in THAT sample is at least the single-sample threshold (", apa_sample_min,
  "), which is a quarter of the cohort-level minimum count (", opt$`min-counts`,
  ") with a floor of 5 -- a literal floor, independent of the length ",
  "track's cell-weight threshold -- because ",
  "min-counts is a cohort-level notion and a single ",
  "low-coverage sample carries single-digit poly(A) reads per gene; ",
  "independent filtering: not applied (the threshold is NA and nothing was filtered); ",
  contrast_note, "; ", apa_batch_note)
cat("[telescope_diff]  poly(A) DEXSeq arm:", sum(apa_dx_tested),
    "sites tested,", sum(apa_dx_is_significant),
    "significant (padj < ", APA_DEXSEQ_PADJ_CUTOFF, " and |delta_pau| >= ",
    apa_min_delta_pau, ")\n")

apa_out <- attach_gene_name(apa_out)
write.table(apa_out[, APA_COLUMNS],
            file = file.path(opt$`output-dir`, "apa.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("[telescope_diff]  wrote apa.tsv (", nrow(apa_out), "sites,",
    sum(is.finite(apa_out$pvalue)), "tested)\n")
}

                                                                       
                                   
                                                                       
writeLines(capture.output(sessionInfo()),
           file.path(opt$`output-dir`, "sessionInfo.txt"))
cat("\n[telescope_diff] done. Output in:", opt$`output-dir`, "\n")
