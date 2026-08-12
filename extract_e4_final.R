#!/usr/bin/env Rscript
options(stringsAsFactors=FALSE)
dir.create('outputs_e4_final',showWarnings=FALSE,recursive=TRUE); dir.create('raw_e4_final',showWarnings=FALSE,recursive=TRUE)
ua <- 'Rilostat/2.1.0 (x86_64-pc-linux-gnu; Linux GitHub-Actions)'
dl <- function(url,dest) download.file(url,dest,method='libcurl',quiet=FALSE,headers=c('User-Agent'=ua))
rr <- function(url,dest){dl(url,dest); readRDS(dest)}
log <- character(); lg <- function(k,v) log <<- c(log,paste0(k,'=',paste(v,collapse=' | ')))
tryCatch({
 toc<-rr('https://rplumber.ilo.org/files/indicator/table_of_contents_en.rds','raw_e4_final/toc.rds')
 target<-'Employment by sex, occupation and economic activity - ISIC level 2 (thousands)'
 cand<-toc[trimws(as.character(toc$indicator.label))==target & toupper(as.character(toc$freq))=='A',,drop=FALSE]
 if(nrow(cand)==0) stop('Target ILOSTAT table not found'); id<-as.character(cand$id[1]); lg('id',id); write.csv(cand,'outputs_e4_final/ILOSTAT_toc_match.csv',row.names=FALSE)
 d<-rr(paste0('https://rplumber.ilo.org/files/indicator/',id,'.rds'),paste0('raw_e4_final/',id,'.rds')); lg('data_rows',nrow(d)); lg('data_cols',names(d))
 dic1<-rr('https://rplumber.ilo.org/files/dic/classif1_en.rds','raw_e4_final/classif1.rds'); dic2<-rr('https://rplumber.ilo.org/files/dic/classif2_en.rds','raw_e4_final/classif2.rds')
 occ<-c('OCU_ISCO08_2','OCU_ISCO08_3'); fin<-c('EC2_ISIC4_K64','EC2_ISIC4_K65','EC2_ISIC4_K66'); rd<-c('EC2_ISIC4_M72')
 if(!all(occ %in% as.character(dic1$classif1))) stop('Expected ISCO-08 major-group codes absent from current dictionary')
 if(!all(c(fin,rd) %in% as.character(dic2$classif2))) stop('Expected ISIC Rev.4 division codes absent from current dictionary')
 write.csv(dic1[dic1$classif1 %in% occ,,drop=FALSE],'outputs_e4_final/occupation_dictionary_exact.csv',row.names=FALSE)
 write.csv(dic2[dic2$classif2 %in% c(fin,rd),,drop=FALSE],'outputs_e4_final/activity_dictionary_exact.csv',row.names=FALSE)
 lg('occupation_codes',occ); lg('finance_codes',fin); lg('rd_codes',rd)
 if('sex' %in% names(d)){sk<-as.character(d$sex) %in% c('SEX_T','T','TOTAL','Total'); lg('sex_total_rows',sum(sk,na.rm=TRUE)); if(any(sk,na.rm=TRUE)) d<-d[sk,,drop=FALSE]}
 d<-d[as.character(d$classif1) %in% occ & as.character(d$classif2) %in% c(fin,rd),,drop=FALSE]; d$sector<-ifelse(as.character(d$classif2) %in% fin,'finance','rd')
 if('best_source' %in% names(d)){bs<-suppressWarnings(as.numeric(as.character(d$best_source))); lg('best_source_1_rows_before_filter',sum(bs==1,na.rm=TRUE)); if(any(bs==1,na.rm=TRUE)) d<-d[bs==1,,drop=FALSE]}
 d$year<-suppressWarnings(as.integer(substr(as.character(d$time),1,4))); d$value<-suppressWarnings(as.numeric(d$obs_value)); d<-d[!is.na(d$year)&d$year>=1996&d$year<=2021&!is.na(d$value),,drop=FALSE]
 lg('target_rows_1996_2021',nrow(d)); lg('target_countries',length(unique(d$ref_area))); if(nrow(d)==0) stop('No exact target rows')
 agg<-aggregate(value~ref_area+year+sector,data=d,FUN=sum,na.rm=TRUE); wide<-reshape(agg,idvar=c('ref_area','year'),timevar='sector',direction='wide'); names(wide)<-sub('^value\\.','',names(wide))
 if(!all(c('finance','rd') %in% names(wide))) stop('Finance and R&D cells do not overlap')
 wide<-wide[!is.na(wide$finance)&!is.na(wide$rd),,drop=FALSE]; wide$pool<-wide$finance+wide$rd; wide<-wide[wide$pool>0,,drop=FALSE]; wide$s_fin<-wide$finance/wide$pool
 lg('complete_rows',nrow(wide)); lg('complete_countries',length(unique(wide$ref_area))); write.csv(wide,'outputs_e4_final/E4_agg.csv',row.names=FALSE); writeLines(log,'outputs_e4_final/ILOSTAT_metadata.txt'); cat('SUCCESS rows=',nrow(wide),' countries=',length(unique(wide$ref_area)),'\n',sep='')
},error=function(e){lg('error',conditionMessage(e)); writeLines(log,'outputs_e4_final/ILOSTAT_metadata.txt'); stop(e)})
