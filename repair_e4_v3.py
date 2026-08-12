#!/usr/bin/env python3
from __future__ import annotations
import json,re,time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import pyreadr
import statsmodels.formula.api as smf

Y0,Y1=1996,2021
OUT=Path('outputs_e4_v3'); RAW=Path('raw_e4_v3'); OUT.mkdir(exist_ok=True); RAW.mkdir(exist_ok=True)
WB='https://api.worldbank.org/v2'
HEAD={'User-Agent':'Rilostat/2.1.0 (x86_64-pc-linux-gnu; Linux GitHub-Actions)','Accept-Encoding':'gzip, deflate, br'}

def save(o,n): (OUT/n).write_text(json.dumps(o,indent=2,ensure_ascii=False,default=float),encoding='utf-8')
def get(url,params=None,timeout=240,retries=4):
    last=None
    for k in range(retries):
        try:
            r=requests.get(url,params=params,headers=HEAD,timeout=timeout,allow_redirects=True); r.raise_for_status(); return r
        except Exception as e: last=e; time.sleep(1.5*(k+1))
    raise last

def read_rds_url(url,name):
    p=RAW/name; p.write_bytes(get(url,timeout=300).content); obj=pyreadr.read_r(str(p))
    if not obj: raise RuntimeError(f'No dataframe in {url}')
    return next(iter(obj.values()))

def countries():
    rows=get(WB+'/country',{'format':'json','per_page':500}).json()[1]
    return {x['id'] for x in rows if x.get('region',{}).get('id')!='NA'}

def credit_panel():
    url=f'{WB}/country/all/indicator/FD.AST.PRVT.GD.ZS'; pars={'format':'json','date':f'{Y0}:{Y1}','per_page':20000,'page':1}; p=get(url,pars).json(); rows=list(p[1] or [])
    for pg in range(2,int(p[0].get('pages',1))+1): pars['page']=pg; rows.extend(get(url,pars).json()[1] or [])
    cc=countries(); d=pd.DataFrame([{'iso3':x.get('countryiso3code'),'year':int(x['date']),'credit':x.get('value')} for x in rows]); d=d[d.iso3.isin(cc)].dropna(); d['F_bank']=d.credit/100.0; return d[['iso3','year','F_bank']]

def norm(x): return '' if pd.isna(x) else str(x).strip().lower()
def skilled(x):
    s=norm(x)
    return bool(re.search(r'(ocu|isco).*[_:-](2|3)(?:\D|$)',s)) or ('professionals' in s and 'associate' not in s) or ('technicians' in s and 'associate professionals' in s)
def sector(x):
    s=norm(x)
    if re.search(r'(eco|isic).*[_:-](64|65|66)(?:\D|$)',s) or s in {'64','65','66'} or 'financial and insurance' in s: return 'finance'
    if re.search(r'(eco|isic).*[_:-]72(?:\D|$)',s) or s=='72' or 'scientific research and development' in s: return 'rd'
    return None

def main():
    meta={'toc_url':'https://rplumber.ilo.org/files/indicator/table_of_contents_en.rds'}
    try:
        toc=read_rds_url(meta['toc_url'],'table_of_contents_en.rds'); meta['toc_columns']=list(map(str,toc.columns)); meta['toc_rows']=len(toc)
        label_col=next(c for c in toc.columns if str(c).lower()=='indicator.label'); freq_col=next(c for c in toc.columns if str(c).lower()=='freq')
        labels=toc[label_col].astype(str)
        mask=labels.str.contains('Employment by sex, occupation and economic activity - ISIC level 2',case=False,regex=False) & toc[freq_col].astype(str).str.upper().eq('A')
        cand=toc.loc[mask].copy(); cand.to_csv(OUT/'ILOSTAT_toc_matches_v3.csv',index=False); meta['toc_match_count']=len(cand)
        if cand.empty: raise RuntimeError('Target table absent from current RDS TOC')
        exact=cand[cand[label_col].astype(str).str.strip().eq('Employment by sex, occupation and economic activity - ISIC level 2 (thousands)')]
        row=exact.iloc[0] if not exact.empty else cand.iloc[0]; ident=str(row['id']); meta['id']=ident; meta['label']=str(row[label_col])
        url=f'https://rplumber.ilo.org/files/indicator/{ident}.rds'; meta['data_url']=url
        raw=read_rds_url(url,f'{ident}.rds'); meta['data_rows']=len(raw); meta['data_columns']=list(map(str,raw.columns)); raw.head(2000).to_csv(OUT/'ILOSTAT_raw_head_v3.csv',index=False)
        low={str(c).lower():c for c in raw.columns}; ref=next((low[x] for x in ('ref_area','country_code','iso3') if x in low),None); yr=next((low[x] for x in ('time','year') if x in low),None); val=next((low[x] for x in ('obs_value','value') if x in low),None); sex=low.get('sex')
        cls=[c for c in raw.columns if str(c).lower().startswith('classif') or 'occupation' in str(c).lower() or 'economic_activity' in str(c).lower()]
        if not ref or not yr or not val or len(cls)<2: raise RuntimeError(f'Unexpected schema ref={ref},yr={yr},val={val},cls={cls}')
        scores=[]
        for c in cls:
            smp=raw[c].dropna().astype(str).head(50000); scores.append((str(c),sum(skilled(x) for x in smp),sum(sector(x) is not None for x in smp)))
        meta['classification_scores']=scores
        occ=max(scores,key=lambda x:x[1])[0]; act=next(x[0] for x in sorted(scores,key=lambda x:x[2],reverse=True) if x[0]!=occ); meta['occupation_column']=occ; meta['activity_column']=act
        d=raw.copy()
        if sex:
            sm=d[sex].map(lambda x:norm(x) in {'sex_t','t','total','all'} or 'total' in norm(x)); meta['sex_total_matches']=int(sm.sum());
            if sm.any(): d=d[sm].copy()
        d=d[d[occ].map(skilled)].copy(); d['sector']=d[act].map(sector); d=d[d.sector.isin(['finance','rd'])].copy(); meta['rows_after_occ_sector']=len(d)
        d['year']=pd.to_numeric(d[yr],errors='coerce'); d['value']=pd.to_numeric(d[val],errors='coerce'); d['iso3']=d[ref].astype(str).str.strip(); d=d.dropna(subset=['year','value']); d['year']=d.year.astype(int); d=d[d.year.between(Y0,Y1)]
        # If best_source is present, use only flagged best sources to avoid double-counting multiple survey/source series.
        if 'best_source' in d.columns:
            bs=pd.to_numeric(d['best_source'],errors='coerce'); meta['best_source_1_rows']=int((bs==1).sum());
            if (bs==1).any(): d=d[bs==1].copy()
        agg=d.groupby(['iso3','year','sector'],as_index=False).value.sum().pivot(index=['iso3','year'],columns='sector',values='value').reset_index(); meta['agg_rows']=len(agg)
        if not {'finance','rd'}.issubset(agg.columns): raise RuntimeError(f'Missing finance/rd after parse; columns={list(agg.columns)}')
        agg=agg.dropna(subset=['finance','rd']); agg['pool']=agg.finance+agg.rd; agg=agg[agg.pool>0]; agg['s_fin']=agg.finance/agg.pool
        m=agg.merge(credit_panel(),on=['iso3','year'],how='inner').dropna().reset_index(drop=True); meta['estimation_rows']=len(m); meta['estimation_countries']=int(m.iso3.nunique()); m.to_csv(OUT/'E4_estimation_sample_v3.csv',index=False)
        if m.iso3.nunique()<5: raise RuntimeError(f'Too few countries after merge: {m.iso3.nunique()}')
        fit=smf.ols('s_fin ~ F_bank + C(iso3) + C(year)',data=m).fit(cov_type='cluster',cov_kwds={'groups':m.iso3})
        out={'d_target':float(fit.params.F_bank),'d_cluster_se':float(fit.bse.F_bank),'n_obs':int(fit.nobs),'n_countries':int(m.iso3.nunique()),'ilostat_id':ident,'occupation_column_detected':occ,'activity_column_detected':act,'metadata':meta}
        save(out,'E4_d_results_v3.json'); (OUT/'E4_regression_summary_v3.txt').write_text(fit.summary().as_text(),encoding='utf-8'); save(meta,'ILOSTAT_download_v3.json'); print(json.dumps(out,indent=2,ensure_ascii=False))
    except Exception as e:
        meta['error']=repr(e); save(meta,'ILOSTAT_download_v3.json'); out={'d_target':None,'error':repr(e),'metadata':meta}; save(out,'E4_d_results_v3.json'); print(json.dumps(out,indent=2,ensure_ascii=False)); raise

if __name__=='__main__': main()
