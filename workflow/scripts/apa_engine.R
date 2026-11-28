APA_ENGINE_NAME <- "Telescope in-tree poly(A) caller (APALORD-derived)"
APA_ENGINE_VERSION <- "1"

apa_engine_default_params <- function() {
  list(min_reads = 5L, min_percent = 1, bin = 50L, direct_rna = TRUE)
}

apa_engine_attribute <- function(attributes, key) {
  pattern <- paste0(key, " ([^;]+)")
  m <- regexpr(pattern, attributes)
  out <- rep(NA_character_, length(attributes))
  hit <- m != -1L
  if (any(hit)) {
    start <- m[hit] + nchar(key) + 1L
    out[hit] <- substring(attributes[hit], start, m[hit] + attr(m, "match.length")[hit] - 1L)
  }
  out
}

apa_engine_gene_table <- function(gtf_file) {
  gtf <- data.table::fread(gtf_file, header = FALSE, sep = "\t", data.table = TRUE)
  data.table::setnames(gtf, c("V1", "V3", "V4", "V5", "V7", "V9"),
                       c("chrom", "feature", "start", "end", "strand", "attributes"))

  gene_lines <- gtf[feature == "gene"]
  gene_lines[, gene_id := apa_engine_attribute(attributes, "gene_id")]
  gene_lines[, gene_name := apa_engine_attribute(attributes, "gene_name")]
  gene_lines[is.na(gene_name), gene_name := apa_engine_attribute(attributes, "gene_symbol")]
  gene_lines[, gene_biotype := apa_engine_attribute(attributes, "gene_biotype")]
  gene_lines[is.na(gene_biotype), gene_biotype := apa_engine_attribute(attributes, "gene_type")]
  gene_info <- unique(gene_lines[, .(gene_id, gene_name, gene_biotype)])
  data.table::setkey(gene_info, gene_id)

  exon_lines <- gtf[feature == "exon"]
  exon_lines[, gene_id := apa_engine_attribute(attributes, "gene_id")]
  exon_info <- unique(exon_lines[, .(gene_id, chrom = as.character(chrom),
                                    chromStart = as.numeric(start),
                                    chromEnd = as.numeric(end),
                                    strand = as.character(strand))])
  exon_info[, order_in_file := .I]
  data.table::setkey(exon_info, gene_id)

  first_strand <- exon_info[, .(first_strand = strand[1L]), by = gene_id]
  plus <- merge(exon_info, first_strand, by = "gene_id")
  plus <- plus[first_strand == "+"]
  if (nrow(plus) > 0) {
    plus[, keep := chromEnd == max(chromEnd), by = gene_id]
    plus <- plus[keep == TRUE]
    plus[, best := max(chromStart), by = gene_id]
    plus <- plus[chromStart == best]
    plus[, best := min(order_in_file), by = gene_id]
    last_plus <- plus[order_in_file == best]
  } else {
    last_plus <- plus
  }
  minus <- merge(exon_info, first_strand, by = "gene_id")
  minus <- minus[first_strand == "-"]
  if (nrow(minus) > 0) {
    minus[, keep := chromStart == min(chromStart), by = gene_id]
    minus <- minus[keep == TRUE]
    minus[, best := min(chromEnd), by = gene_id]
    minus <- minus[chromEnd == best]
    minus[, best := min(order_in_file), by = gene_id]
    last_minus <- minus[order_in_file == best]
  } else {
    last_minus <- minus
  }
  last_exon <- data.table::rbindlist(list(last_plus, last_minus), fill = TRUE)
  if (nrow(last_exon) > 0) {
    last_exon[, c("keep", "best", "first_strand", "order_in_file") := NULL]
    data.table::setcolorder(last_exon, c("gene_id", "chrom", "chromStart",
                                          "chromEnd", "strand"))
  }

  gene_reference <- merge(last_exon, gene_info, by = "gene_id", all.x = TRUE)
  gene_reference$gene_id <- gsub('"', "", gene_reference$gene_id)
  gene_reference$gene_name <- gsub('"', "", gene_reference$gene_name)
  gene_reference$gene_biotype <- gsub('"', "", gene_reference$gene_biotype)
  data.table::setkey(gene_reference, gene_id)
  list(gene = gene_reference, gtf = gtf)
}

apa_engine_gene_spans <- function(gtf_file) {
  txdb <- GenomicFeatures::makeTxDbFromGFF(gtf_file)
  genes <- GenomicFeatures::genes(txdb)
  rm(txdb)
  invisible(gc())
  genes
}

apa_engine_read_sample <- function(bam, sample_id, gene_spans) {
  reads <- GenomicAlignments::readGAlignments(bam, use.names = TRUE)
  read_ids <- names(reads)
  chrom <- as.character(GenomicRanges::seqnames(reads))
  strand_info <- as.character(GenomicRanges::strand(reads))
  five_prime <- ifelse(strand_info == "+", GenomicRanges::start(reads),
                       GenomicRanges::end(reads))
  three_prime <- ifelse(strand_info == "+", GenomicRanges::end(reads),
                        GenomicRanges::start(reads))
  read_table <- data.table::data.table(
    read_id = read_ids,
    chrom = chrom,
    strand = strand_info,
    chromStart = five_prime,
    chromEnd = three_prime,
    gene_id = "NA",
    sample = sample_id,
    stringsAsFactors = FALSE
  )
  overlaps <- GenomicRanges::findOverlaps(reads, gene_spans)
  read_table$gene_id[S4Vectors::queryHits(overlaps)] <-
    names(gene_spans)[S4Vectors::subjectHits(overlaps)]
  read_table <- unique(read_table, by = "read_id")
  read_table <- read_table[gene_id != "NA"]
  read_table
}

apa_engine_peaks <- function(ends, params) {
  bin <- params$bin
  freq_table <- table(ends)
  density_vals <- as.numeric(prop.table(freq_table))
  threshold <- min(sort(density_vals, decreasing = TRUE)[1:50], na.rm = TRUE)
  top_peaks <- density_vals >= threshold
  peaks <- data.frame(
    Value = as.numeric(names(freq_table))[top_peaks],
    Frequency = as.numeric(freq_table[top_peaks]),
    Density = 100 * density_vals[top_peaks],
    stringsAsFactors = FALSE
  )
  peaks <- peaks[order(peaks$Value), ]
  if (nrow(peaks) <= 1) {
    return(peaks)
  }
  group <- integer(nrow(peaks))
  current_group <- 1
  group[1] <- current_group
  for (i in 2:nrow(peaks)) {
    group_df <- peaks[group == current_group, , drop = FALSE]
    rep_val <- group_df$Value[which.max(group_df$Frequency)]
    if (peaks$Value[i] - rep_val > bin) {
      current_group <- current_group + 1
    }
    group[i] <- current_group
  }
  pieces <- lapply(split(peaks, group), function(group_data) {
    best <- group_data[which.max(group_data$Density), , drop = FALSE]
    best$Frequency <- sum(group_data$Frequency)
    best$Density <- sum(group_data$Density)
    best
  })
  data.table::rbindlist(pieces)
}

apa_engine_nearest <- function(position, sites, bin) {
  distances <- abs(position - sites)
  if (any(distances <= bin)) {
    return(sites[which.min(distances)])
  }
  NA
}

apa_engine_call_gene <- function(gene, reads_dt, gene_table, samples, params) {
  gene_all <- reads_dt[gene]
  gene_row <- gene_table[gene]
  if (nrow(gene_row) == 0) {
    return(NULL)
  }
  gene_strand <- as.character(gene_row$strand[1])
  if (isTRUE(params$direct_rna)) {
    gene_all <- gene_all[strand == gene_strand]
  }
  if (nrow(gene_all) == 0) {
    return(NULL)
  }
  if (gene_strand == "+") {
    ends <- gene_all$chromEnd
  } else {
    ends <- gene_all$chromEnd + 1
  }
  if (length(ends) < params$min_reads) {
    return(NULL)
  }
  depths <- table(factor(gene_all$sample, levels = samples))
  if (!all(depths >= params$min_reads)) {
    return(NULL)
  }
  peaks <- apa_engine_peaks(ends, params)
  if (nrow(peaks) == 0) {
    return(NULL)
  }
  called <- peaks[peaks$Density >= params$min_percent &
                    peaks$Frequency >= params$min_reads, , drop = FALSE]
  if (nrow(called) == 0) {
    return(NULL)
  }
  called <- called[order(called$Value), , drop = FALSE]

  sites <- called$Value
  if (gene_strand == "+") {
    positions <- gene_all$chromEnd
  } else {
    positions <- gene_all$chromEnd + 1
  }
  matched <- vapply(positions, apa_engine_nearest, numeric(1),
                    sites = sites, bin = params$bin)
  keep_sites <- names(table(matched)[table(matched) >= params$min_reads])
  keep_sites <- keep_sites[keep_sites != "NA"]
  kept <- as.character(matched) %in% keep_sites
  matched_kept <- matched[kept]
  if (length(matched_kept) == 0) {
    return(NULL)
  }
  all_sites <- names(table(matched_kept))

  out <- data.table::data.table(
    gene_id = gene,
    chrom = as.character(gene_all$chrom[1]),
    strand = gene_strand,
    gene_name = as.character(gene_row$gene_name[1]),
    site = as.numeric(all_sites),
    stringsAsFactors = FALSE
  )
  for (sample_name in samples) {
    in_sample <- gene_all$sample == sample_name
    out[, paste0(sample_name, ".reads") := as.integer(table(
      factor(matched_kept[in_sample[kept]], levels = all_sites)))]
    out[, paste0(sample_name, ".host_reads") := sum(in_sample)]
  }
  out
}

apa_engine_call_cohort <- function(gtf_file, samples, params, cores) {
  gene_table <- apa_engine_gene_table(gtf_file)$gene
  gene_spans <- apa_engine_gene_spans(gtf_file)

  read_tables <- vector("list", nrow(samples))
  for (i in seq_len(nrow(samples))) {
    read_tables[[i]] <- apa_engine_read_sample(samples$alignment_path[i],
                                               samples$sample_id[i],
                                               gene_spans)
  }
  reads_dt <- data.table::rbindlist(read_tables, fill = TRUE)
  rm(read_tables)
  data.table::setkey(reads_dt, gene_id)

  counts_per_gene <- reads_dt[, .N, by = gene_id]
  genes <- counts_per_gene[N >= params$min_reads]$gene_id
  genes <- as.vector(genes[genes != "."])
  genes <- genes[genes %in% gene_table$gene_id]
  if (length(genes) == 0) {
    return(NULL)
  }

  pieces <- parallel::mclapply(genes, apa_engine_call_gene,
                                reads_dt = reads_dt, gene_table = gene_table,
                                samples = samples$sample_id, params = params,
                                mc.cores = cores)
  called <- sum(!vapply(pieces, is.null, logical(1)))
  cat("  poly(A) caller: ", length(genes), " genes carried enough reads, ",
      called, " of them were called\n", sep = "")
  calls <- data.table::rbindlist(pieces, fill = TRUE)
  if (is.null(calls) || nrow(calls) == 0) {
    return(NULL)
  }
  calls
}

apa_engine_cohort_key <- function(gtf_file, samples, params) {
  lines <- c(
    paste0("engine\t", APA_ENGINE_NAME, "\t", APA_ENGINE_VERSION),
    paste0("gtf\t", normalizePath(gtf_file, mustWork = TRUE), "\t",
           tools::md5sum(gtf_file)[[1]]),
    paste0("params\tmin_reads=", params$min_reads,
           "\tmin_percent=", params$min_percent,
           "\tbin=", params$bin,
           "\tdirect_rna=", params$direct_rna)
  )
  for (i in seq_len(nrow(samples))) {
    bam <- samples$alignment_path[i]
    info <- file.info(bam)
    lines <- c(lines, paste0("sample\t", samples$sample_id[i], "\t",
                             normalizePath(bam, mustWork = TRUE), "\t",
                             info$size, "\t", format(info$mtime, "%Y-%m-%dT%H:%M:%S")))
  }
  text <- paste(lines, collapse = "\n")
  tmp <- tempfile("apa_engine_key")
  writeLines(text, tmp)
  on.exit(unlink(tmp), add = TRUE)
  unname(tools::md5sum(tmp)[[1]])
}

apa_engine_call_dir <- function(root, key) {
  file.path(root, key)
}

apa_engine_write_calls <- function(dir, calls, sample_ids, key, receipt) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  sites <- unique(calls[, .(gene_id, chrom, strand, gene_name, site)])
  data.table::setorder(sites, gene_id, site)
  long <- data.table::melt(
    calls[, c("gene_id", "site", grep("\\.reads$", colnames(calls), value = TRUE)),
           with = FALSE],
    id.vars = c("gene_id", "site"),
    variable.name = "sample_column",
    value.name = "reads")
  long[, sample_id := sub("\\.reads$", "", sample_column)]
  data.table::setorder(long, gene_id, site, sample_id)
  data.table::fwrite(sites, file.path(dir, "sites.tsv"), sep = "\t")
  data.table::fwrite(long, file.path(dir, "counts.tsv"), sep = "\t")
  hosts <- data.table::melt(
    calls[, c("gene_id", grep("\\.host_reads$", colnames(calls), value = TRUE)),
           with = FALSE],
    id.vars = "gene_id",
    variable.name = "sample_column",
    value.name = "host_reads")
  hosts <- unique(hosts)
  hosts[, sample_id := sub("\\.host_reads$", "", sample_column)]
  data.table::setorder(hosts, gene_id, sample_id)
  data.table::fwrite(hosts, file.path(dir, "gene_reads.tsv"), sep = "\t")
  writeLines(receipt, file.path(dir, "receipt.txt"))
  invisible(dir)
}

apa_engine_read_calls <- function(dir, sample_ids) {
  sites <- data.table::fread(file.path(dir, "sites.tsv"), data.table = TRUE)
  counts <- data.table::fread(file.path(dir, "counts.tsv"), data.table = TRUE)
  hosts <- data.table::fread(file.path(dir, "gene_reads.tsv"), data.table = TRUE)
  out <- sites
  for (sample_id in sample_ids) {
    rows <- match(paste(sites$gene_id, sites$site),
                  paste(counts$gene_id[counts$sample_id == sample_id],
                        counts$site[counts$sample_id == sample_id]))
    out[, paste0(sample_id, ".reads") :=
          as.integer(counts$reads[counts$sample_id == sample_id][rows])]
    gene_rows <- match(sites$gene_id, hosts$gene_id[hosts$sample_id == sample_id])
    out[, paste0(sample_id, ".host_reads") :=
          as.integer(hosts$host_reads[hosts$sample_id == sample_id][gene_rows])]
  }
  out[!is.na(site)]
}

apa_engine_run <- function(gtf_file, samples, out_dir, reuse_dir = NULL,
                            params = NULL, cores = 1L) {
  if (is.null(params)) {
    params <- apa_engine_default_params()
  }
  if (is.null(samples) || nrow(samples) == 0) {
    stop("ERROR: the poly(A) engine needs at least one sample.")
  }
  key <- apa_engine_cohort_key(gtf_file, samples, params)

  source_dir <- NULL
  if (!is.null(reuse_dir) && nzchar(reuse_dir) && dir.exists(reuse_dir)) {
    candidate <- apa_engine_call_dir(reuse_dir, key)
    if (dir.exists(candidate) && file.exists(file.path(candidate, "counts.tsv"))) {
      source_dir <- candidate
    }
  }

  root <- apa_engine_call_dir(out_dir, key)
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  receipt <- c(
    paste0("engine\t", APA_ENGINE_NAME),
    paste0("engine_version\t", APA_ENGINE_VERSION),
    paste0("cohort_key\t", key),
    paste0("annotation\t", basename(gtf_file)),
    paste0("samples\t", paste(samples$sample_id, collapse = ",")),
    paste0("min_reads\t", params$min_reads),
    paste0("min_percent\t", params$min_percent),
    paste0("bin\t", params$bin),
    paste0("direct_rna\t", params$direct_rna)
  )

  if (!is.null(source_dir)) {
    calls <- apa_engine_read_calls(source_dir, samples$sample_id)
    receipt <- c(receipt,
                 paste0("calls_made_by\tno (this run reused the calls of the cohort)"),
                 paste0("calls_reused_from\t", basename(dirname(source_dir))))
  } else {
    calls <- apa_engine_call_cohort(gtf_file, samples, params, cores)
    if (is.null(calls)) {
      stop("ERROR: the poly(A) engine called no site in this cohort.")
    }
    receipt <- c(receipt,
                 paste0("calls_made_by\tthis run"),
                 paste0("calls_reused_from\tnone (the calls were made by this run)"))
  }
  apa_engine_write_calls(root, calls, samples$sample_id, key, receipt)
  view <- apa_engine_view(calls, samples$sample_id)
  list(key = key, dir = root, calls = calls, receipt = receipt,
       sites = view$sites, counts = view$counts, hosts = view$hosts,
       reused = !is.null(source_dir))
}

apa_engine_view <- function(calls, sample_ids) {
  sites <- unique(calls[, .(gene_id, chrom, strand, gene_name, site)])
  data.table::setorder(sites, gene_id, site)

  site_key <- paste(sites$chrom, sites$site, sites$strand, sep = ":")
  duplicate <- duplicated(site_key) | duplicated(site_key, fromLast = TRUE)
  site_id <- site_key
  if (any(duplicate)) {
    occurrence <- ave(seq_len(nrow(sites)), site_key, FUN = seq_along)
    site_id[duplicate] <- paste0(site_key[duplicate], "#", occurrence[duplicate])
  }
  sites$site_id <- site_id

  site_ids <- sites$site_id
  count_matrix <- matrix(0L, nrow = length(site_ids), ncol = length(sample_ids),
                          dimnames = list(site_ids, sample_ids))
  for (j in seq_along(sample_ids)) {
    column <- paste0(sample_ids[j], ".reads")
    rows <- match(paste(sites$gene_id, sites$site),
                  paste(calls$gene_id, calls$site))
    count_matrix[, j] <- as.integer(calls[[column]][rows])
  }
  gene_ids <- sort(unique(sites$gene_id))
  host_matrix <- matrix(0L, nrow = length(gene_ids), ncol = length(sample_ids),
                         dimnames = list(gene_ids, sample_ids))
  for (j in seq_along(sample_ids)) {
    column <- paste0(sample_ids[j], ".host_reads")
    rows <- match(gene_ids, calls$gene_id)
    host_matrix[, j] <- as.integer(calls[[column]][rows])
  }
  list(sites = sites, counts = count_matrix, hosts = host_matrix)
}
