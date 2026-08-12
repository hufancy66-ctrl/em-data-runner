#!/usr/bin/env Rscript
options(stringsAsFactors = FALSE)
dir.create("outputs_e4_r2", showWarnings = FALSE, recursive = TRUE)
dir.create("raw_e4_r2", showWarnings = FALSE, recursive = TRUE)
ua <- "Rilostat/2.1.0 (x86_64-pc-linux-gnu; Linux GitHub-Actions)"
dl <- function(url, dest) download.file(url, dest, method="libcurl", quiet=FALSE, headers=c("User-Agent"=ua))
logv <- character(); lg <- function(k,v) logv <<- c(logv,paste0(k,"=",paste(v,collapse=" | ")))
read_remote_rds <- function(url, dest) { dl(url,dest); readRDS(dest) }
tryCatch({
  toc <- read_remote_rds("https://rplumber.ilo.org/files/indicator/table_of_contents_en.rds","raw_e4_r2/toc.rds")
  target <- "Employment by sex, occupation and economic activity - ISIC level 2 (thousands)"
  cand <- toc[trimws(as.character(toc$indicator.label))==target & toupper(as.character(toc$freq))=="A",,drop=FALSE]
  if(nrow(cand)==0) stop("Target table absent from current TOC")
  id <- as.character(cand$id[1]); lg("id",id); lg("toc_match",cand$indicator.label[1]); write.csv(cand,"outputs_e4_r2/ILOSTAT_toc_match.csv",row.names=FALSE)
  d <- read_remote_rds(paste0("https://rplumber.ilo.org/files/indicator/",id,".rds"),paste0("raw_e4_r2/",id,".rds"))
  lg("data_rows",nrow(d)); lg("data_cols",names(d));
  # Official dictionaries map classification codes to labels, avoiding assumptions about current code string format.
  dic1 <- read_remote_rds("https://rplumber.ilo.org/files/dic/classif1_en.rds","raw_e4_r2/classif1_en.rds")
  dic2 <- read_remote_rds("https://rplumber.ilo.org/files/dic/classif2_en.rds","raw_e4_r2/classif2_en.rds")
  lg("dic1_cols",names(dic1)); lg("dic2_cols",names(dic2)); lg("dic1_rows",nrow(dic1)); lg("dic2_rows",nrow(dic2))
  code_col <- function(x) { z<-names(x)[tolower(names(x)) %in% c("code","id")]; if(length(z)) z[1] else names(x)[1] }
  label_col <- function(x) { z<-names(x)[grepl("label|name",tolower(names(x)))]; if(length(z)) z[1] else names(x)[2] }
  c1<-code_col(dic1); l1<-label_col(dic1); c2<-code_col(dic2); l2<-label_col(dic2)
  lg("dic1_code_label_cols",c(c1,l1)); lg("dic2_code_label_cols",c(c2,l2))
  # Save relevant dictionary rows for full audit trail.
  olabel <- tolower(as.character(dic1[[l1]])); alabel <- tolower(as.character(dic2[[l2]]))
  occ_dict <- dic1[grepl("professionals|technicians|associate professionals",olabel),,drop=FALSE]
  act_dict <- dic2[grepl("financial service activities|insurance, reinsurance|activities auxiliary to financial|scientific research and development|financial and insurance activities",alabel),,drop=FALSE]
  write.csv(occ_dict,"outputs_e4_r2/occupation_dictionary_matches.csv",row.names=FALSE)
  write.csv(act_dict,"outputs_e4_r2/activity_dictionary_matches.csv",row.names=FALSE)
  lg("occ_dict_matches",nrow(occ_dict)); lg("act_dict_matches",nrow(act_dict))
  if(nrow(occ_dict)==0 || nrow(act_dict)==0) stop("Dictionary label matching failed")
  # Restrict occupation labels to ISCO-08 major groups 2 and 3, excluding nested subgroups when labels/codes expose them.
  occ_codes <- as.character(occ_dict[[c1]])
  # Major-group codes usually contain ISCO08_2 / ISCO08_3; if exact markers exist, use them. Otherwise keep dictionary matches and audit them.
  exact_occ <- occ_codes[grepl("ISCO08.*(^|[_:-])[23]$|ISCO08_[23]$",occ_codes,perl=TRUE)]
  if(length(exact_occ)>=2) occ_codes <- exact_occ
  # Activity matches: retain level-2 divisions 64,65,66,72. Detect digits from labels first, then code suffixes.
  act_codes_all <- as.character(act_dict[[c2]]); act_labels_all <- as.character(act_dict[[l2]])
  fin_mask <- grepl("(^|[^0-9])(64|65|66)([^0-9]|$)",act_labels_all,perl=TRUE) | grepl("financial service activities|insurance, reinsurance|activities auxiliary to financial",tolower(act_labels_all))
  rd_mask <- grepl("(^|[^0-9])72([^0-9]|$)",act_labels_all,perl=TRUE) | grepl("scientific research and development",tolower(act_labels_all))
  fin_codes <- unique(act_codes_all[fin_mask]); rd_codes <- unique(act_codes_all[rd_mask]);
  lg("occ_codes",occ_codes); lg("finance_codes",fin_codes); lg("rd_codes",rd_codes)
  if(length(fin_codes)==0 || length(rd_codes)==0) stop("Could not map finance/R&D level-2 codes from classif2 dictionary")
  # Total sex and target cells.
  if("sex" %in% names(d)) { sx<-as.character(d$sex); sk<-sx %in% c("SEX_T","T","TOTAL","Total"); lg("sex_total_rows",sum(sk,na.rm=TRUE)); if(any(sk,na.rm=TRUE)) d<-d[sk,,drop=FALSE] }
  d <- d[as.character(d$classif1) %in% occ_codes & as.character(d$classif2) %in% c(fin_codes,rd_codes),,drop=FALSE]
  d$sector <- ifelse(as.character(d$classif2) %in% fin_codes,"finance","rd")
  if("best_source" %in% names(d)) { bs<-suppressWarnings(as.numeric(as.character(d$best_source))); lg("best_source_1_rows",sum(bs==1,na.rm=TRUE)); if(any(bs==1,na.rm=TRUE)) d<-d[bs==1,,drop=FALSE] }
  d$year<-suppressWarnings(as.integer(substr(as.character(d$time),1,4))); d$value<-suppressWarnings(as.numeric(d$obs_value)); d<-d[!is.na(d$year)&d$year>=1996&d$year<=2021&!is.na(d$value),,drop=FALSE]
  lg("target_rows_1996_2021",nrow(d)); lg("target_countries",length(unique(d$ref_area)))
  if(nrow(d)==0) stop("No target rows after dictionary-based filtering")
  agg<-aggregate(value~ref_area+year+sector,data=d,FUN=sum,na.rm=TRUE); wide<-reshape(agg,idvar=c("ref_area","year"),timevar="sector",direction="wide"); names(wide)<-sub("^value\\.","",names(wide))
  if(!all(c("finance","rd") %in% names(wide))) stop("Finance/R&D cells do not overlap")
  wide<-wide[!is.na(wide$finance)&!is.na(wide$rd),,drop=FALSE]; wide$pool<-wide$finance+wide$rd; wide<-wide[wide$pool>0,,drop=FALSE]; wide$s_fin<-wide$finance/wide$pool
  lg("complete_rows",nrow(wide)); lg("complete_countries",length(unique(wide$ref_area))); write.csv(wide,"outputs_e4_r2/E4_agg.csv",row.names=FALSE); writeLines(logv,"outputs_e4_r2/ILOSTAT_metadata.txt")
  cat("SUCCESS id=",id," rows=",nrow(wide)," countries=",length(unique(wide$ref_area)),"\n",sep="")
}, error=function(e){ lg("error",conditionMessage(e)); writeLines(logv,"outputs_e4_r2/ILOSTAT_metadata.txt"); stop(e) })
