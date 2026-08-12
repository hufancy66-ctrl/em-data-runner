#!/usr/bin/env Rscript
options(stringsAsFactors = FALSE)
dir.create("outputs_e4_r", showWarnings = FALSE, recursive = TRUE)
dir.create("raw_e4_r", showWarnings = FALSE, recursive = TRUE)

ua <- "Rilostat/2.1.0 (x86_64-pc-linux-gnu; Linux GitHub-Actions)"
dl <- function(url, dest) {
  download.file(url, dest, method = "libcurl", quiet = FALSE,
                headers = c("User-Agent" = ua))
}

meta <- character()
log_meta <- function(k, v) {
  meta <<- c(meta, paste0(k, "=", paste(v, collapse = " | ")))
}

tryCatch({
  toc_url <- "https://rplumber.ilo.org/files/indicator/table_of_contents_en.rds"
  toc_file <- "raw_e4_r/table_of_contents_en.rds"
  dl(toc_url, toc_file)
  toc <- readRDS(toc_file)
  log_meta("toc_rows", nrow(toc))
  log_meta("toc_columns", names(toc))

  label_col <- "indicator.label"
  freq_col <- "freq"
  if (!(label_col %in% names(toc)) || !(freq_col %in% names(toc))) {
    stop("Current ILOSTAT TOC schema missing indicator.label or freq")
  }

  target_phrase <- "Employment by sex, occupation and economic activity - ISIC level 2"
  keep <- grepl(target_phrase, toc[[label_col]], fixed = TRUE) & toupper(as.character(toc[[freq_col]])) == "A"
  cand <- toc[keep, , drop = FALSE]
  write.csv(cand, "outputs_e4_r/ILOSTAT_toc_matches.csv", row.names = FALSE)
  log_meta("toc_match_count", nrow(cand))
  if (nrow(cand) == 0) stop("Target ILOSTAT table absent from current TOC")

  exact_label <- "Employment by sex, occupation and economic activity - ISIC level 2 (thousands)"
  exact <- cand[trimws(as.character(cand[[label_col]])) == exact_label, , drop = FALSE]
  row <- if (nrow(exact) > 0) exact[1, , drop = FALSE] else cand[1, , drop = FALSE]
  id <- as.character(row$id[[1]])
  label <- as.character(row[[label_col]][[1]])
  log_meta("id", id)
  log_meta("label", label)

  data_url <- paste0("https://rplumber.ilo.org/files/indicator/", id, ".rds")
  data_file <- paste0("raw_e4_r/", id, ".rds")
  dl(data_url, data_file)
  d <- readRDS(data_file)
  log_meta("data_rows", nrow(d))
  log_meta("data_columns", names(d))

  # Identify classification columns from current raw codes.
  cls <- names(d)[grepl("^classif|occupation|economic_activity", names(d), ignore.case = TRUE)]
  if (length(cls) < 2) stop(paste("Too few classification columns:", paste(cls, collapse = ",")))
  occ_score <- sapply(cls, function(cn) sum(grepl("ISCO08_(2|3)$", as.character(d[[cn]]), perl = TRUE), na.rm = TRUE))
  act_score <- sapply(cls, function(cn) sum(grepl("ISIC4_(64|65|66|72)$", as.character(d[[cn]]), perl = TRUE), na.rm = TRUE))
  occ_col <- cls[which.max(occ_score)]
  act_candidates <- cls[order(act_score, decreasing = TRUE)]
  act_col <- act_candidates[act_candidates != occ_col][1]
  log_meta("classification_columns", cls)
  log_meta("occupation_scores", paste(names(occ_score), occ_score, sep = ":"))
  log_meta("activity_scores", paste(names(act_score), act_score, sep = ":"))
  log_meta("occupation_column", occ_col)
  log_meta("activity_column", act_col)

  if (!("ref_area" %in% names(d)) || !("time" %in% names(d)) || !("obs_value" %in% names(d))) {
    stop("Raw ILOSTAT table missing ref_area/time/obs_value")
  }

  # Total sex only where present.
  if ("sex" %in% names(d)) {
    sx <- as.character(d$sex)
    sex_keep <- sx %in% c("SEX_T", "T", "TOTAL", "Total")
    log_meta("sex_total_rows", sum(sex_keep, na.rm = TRUE))
    if (any(sex_keep, na.rm = TRUE)) d <- d[sex_keep, , drop = FALSE]
  }

  occv <- as.character(d[[occ_col]])
  actv <- as.character(d[[act_col]])
  occ_keep <- grepl("ISCO08_(2|3)$", occv, perl = TRUE)
  fin_keep <- grepl("ISIC4_(64|65|66)$", actv, perl = TRUE)
  rd_keep <- grepl("ISIC4_72$", actv, perl = TRUE)
  d <- d[occ_keep & (fin_keep | rd_keep), , drop = FALSE]
  if (nrow(d) == 0) stop("No ISCO-08 groups 2/3 × ISIC Rev.4 divisions 64-66/72 rows found")

  # Recalculate sector after filtering.
  actv <- as.character(d[[act_col]])
  d$sector <- ifelse(grepl("ISIC4_(64|65|66)$", actv, perl = TRUE), "finance", "rd")

  # Follow current Rilostat default: best_source = yes, when flag exists.
  if ("best_source" %in% names(d)) {
    bs <- suppressWarnings(as.numeric(as.character(d$best_source)))
    log_meta("best_source_1_rows", sum(bs == 1, na.rm = TRUE))
    if (any(bs == 1, na.rm = TRUE)) d <- d[bs == 1, , drop = FALSE]
  }

  d$year <- suppressWarnings(as.integer(substr(as.character(d$time), 1, 4)))
  d$value <- suppressWarnings(as.numeric(d$obs_value))
  d <- d[!is.na(d$year) & d$year >= 1996 & d$year <= 2021 & !is.na(d$value), , drop = FALSE]
  log_meta("rows_after_all_filters", nrow(d))
  if (nrow(d) == 0) stop("No target rows remain for 1996-2021")

  agg <- aggregate(value ~ ref_area + year + sector, data = d, FUN = sum, na.rm = TRUE)
  wide <- reshape(agg, idvar = c("ref_area", "year"), timevar = "sector", direction = "wide")
  names(wide) <- sub("^value\\.", "", names(wide))
  if (!("finance" %in% names(wide)) || !("rd" %in% names(wide))) stop("Finance and R&D cells do not overlap in wide table")
  wide <- wide[!is.na(wide$finance) & !is.na(wide$rd), , drop = FALSE]
  wide$pool <- wide$finance + wide$rd
  wide <- wide[wide$pool > 0, , drop = FALSE]
  wide$s_fin <- wide$finance / wide$pool
  log_meta("agg_rows_complete", nrow(wide))
  log_meta("agg_countries", length(unique(wide$ref_area)))
  write.csv(wide, "outputs_e4_r/E4_agg.csv", row.names = FALSE)
  writeLines(meta, "outputs_e4_r/ILOSTAT_metadata.txt")
  cat("E4 extraction success; id=", id, "; rows=", nrow(wide), "; countries=", length(unique(wide$ref_area)), "\n", sep = "")
}, error = function(e) {
  log_meta("error", conditionMessage(e))
  writeLines(meta, "outputs_e4_r/ILOSTAT_metadata.txt")
  stop(e)
})
