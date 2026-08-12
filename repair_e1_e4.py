#!/usr/bin/env python3
from __future__ import annotations
import json, math, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import pyreadr
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar

Y0,Y1=1996,2021
OUT=Path('outputs_repair'); RAW=Path('raw_repair'); OUT.mkdir(exist_ok=True); RAW.mkdir(exist_ok=True)
WB='https://api.worldbank.org/v2'

def get(url,params=None,timeout=180,retries=4):
    last=None
    for k in range(retries):
        try:
            r=requests.get(url,params=params,timeout=timeout,headers={'User-Agent':'EM-repair-ci/2.0'})
            r.raise_for_status(); return r
        except Exception as e:
            last=e; time.sleep(1.5*(k+1))
    raise last

def save(obj,name):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=float),encoding='utf-8')

def countries():
    rows=get(WB+'/country',{'format':'json','per_page':500}).json()[1]
    return {x['id'] for x in rows if x.get('region',{}).get('id')!='NA'}

def wdi(code,name):
    url=f'{WB}/country/all/indicator/{code}'
    pars={'format':'json','date':f'{Y0}:{Y1}','per_page':20000,'page':1}
    payload=get(url,pars).json(); rows=list(payload[1] or [])
    for page in range(2,int(payload[0].get('pages',1))+1):
        pars['page']=page; rows.extend(get(url,pars).json()[1] or [])
    return pd.DataFrame([{'iso3':x.get('countryiso3code'),'country':(x.get('country') or {}).get('value'),'year':int(x['date']),'value':x.get('value')} for x in rows]).rename(columns={'value':name})

def build_e1_panel():
    cc=countries()
    a=wdi('FD.AST.PRVT.GD.ZS','credit'); b=wdi('FR.INR.LNDP','spread')
    a=a[a.iso3.isin(cc)][['iso3','year','credit']]; b=b[b.iso3.isin(cc)][['iso3','year','spread']]
    d=a.merge(b,on=['iso3','year']).dropna(); d['F_bank']=d.credit/100.0
    d=d[d.F_bank>=0].reset_index(drop=True)
    d.to_csv(OUT/'E1_estimation_sample.csv',index=False)
    return d

def e1fit(d,eta,cluster=False):
    z=d.copy(); z['exp_eta_F']=np.exp(-eta*z.F_bank)
    m=smf.ols('spread ~ exp_eta_F + C(iso3) + C(year)',data=z)
    return m.fit(cov_type='cluster',cov_kwds={'groups':z.iso3}) if cluster else m.fit()

def estimate_eta(d, lo=.02, hi=100.0, grid_n=180):
    grid=np.geomspace(lo,hi,grid_n); rows=[]
    for eta in grid:
        f=e1fit(d,eta); rows.append([eta,float(np.sum(np.asarray(f.resid)**2)),float(f.params.exp_eta_F)])
    prof=pd.DataFrame(rows,columns=['eta','ssr','beta_exp']); prof['lr']=len(d)*np.log(prof.ssr/prof.ssr.min()); prof.to_csv(OUT/'E1_eta_profile_expanded.csv',index=False)
    opt=minimize_scalar(lambda x: float(np.sum(np.asarray(e1fit(d,x).resid)**2)),bounds=(lo,hi),method='bounded',options={'xatol':1e-6})
    eta=float(opt.x); fit=e1fit(d,eta,cluster=True); beta=float(fit.params.exp_eta_F)
    boundary=eta<=lo*1.03 or eta>=hi*.97
    ci=prof.loc[prof.lr<=3.841459,'eta']; ci_lo=None if ci.empty else float(ci.min()); ci_hi=None if ci.empty else float(ci.max())
    ci_hits_boundary=bool(ci.empty or ci_lo<=lo*1.03 or ci_hi>=hi*.97)
    return prof,eta,fit,beta,boundary,ci_lo,ci_hi,ci_hits_boundary

def run_e1():
    d=build_e1_panel(); prof,eta,fit,beta,boundary,ci_lo,ci_hi,ci_hits=estimate_eta(d)
    rng=np.random.default_rng(20260812); cs=np.array(sorted(d.iso3.unique())); draws=[]
    for b in range(100):
        parts=[]
        for j,c in enumerate(rng.choice(cs,size=len(cs),replace=True)):
            x=d[d.iso3==c].copy(); x['iso3']=f'{c}__{j}'; parts.append(x)
        bd=pd.concat(parts,ignore_index=True)
        try:
            opt=minimize_scalar(lambda x: float(np.sum(np.asarray(e1fit(bd,x).resid)**2)),bounds=(.02,100),method='bounded',options={'xatol':1e-5})
            eh=float(opt.x); be=float(e1fit(bd,eh).params.exp_eta_F); valid=be>0 and .0206<eh<97.0
            draws.append([b,eh,be,valid])
        except Exception:
            draws.append([b,np.nan,np.nan,False])
    boot=pd.DataFrame(draws,columns=['draw','eta','beta_exp','valid']); boot.to_csv(OUT/'E1_eta_country_bootstrap_expanded.csv',index=False)
    share=float(boot.valid.mean()); valid=boot.loc[boot.valid & boot.eta.notna(),'eta']
    # Strict identification: theory-consistent sign, interior optimum, bounded profile-LR interval,
    # and at least half the country-bootstrap draws produce a theory-consistent interior eta.
    identified=bool(beta>0 and not boundary and not ci_hits and share>=.5)
    out={
      'eta_hat':eta if identified else None,'eta_optimizer_raw':eta,
      'beta_exp_term':beta,'beta_cluster_se':float(fit.bse.exp_eta_F),
      'identified':identified,'at_search_boundary':boundary,
      'profile_lr95_ci_low':ci_lo,'profile_lr95_ci_high':ci_hi,'profile_ci_hits_search_boundary':ci_hits,
      'bootstrap_valid_share':share,
      'bootstrap_eta_median_valid':None if valid.empty else float(valid.median()),
      'bootstrap_eta_p05_valid':None if valid.empty else float(valid.quantile(.05)),
      'bootstrap_eta_p95_valid':None if valid.empty else float(valid.quantile(.95)),
      'n_obs':int(fit.nobs),'n_countries':int(d.iso3.nunique())}
    save(out,'E1_eta_results_repair.json'); (OUT/'E1_regression_summary_repair.txt').write_text(fit.summary().as_text(),encoding='utf-8')
    return out

def norm(s): return '' if pd.isna(s) else str(s).strip().lower()
def skilled(x):
    s=norm(x)
    return bool(re.search(r'(ocu|isco)[_:-]?(isco)?0?8?[_:-]?[23](?:\D|$)',s)) or ('professionals' in s and 'associate' not in s) or ('technicians' in s and 'associate professionals' in s)
def sector(x):
    s=norm(x)
    if re.search(r'(eco|isic).*[_:-](64|65|66)(?:\D|$)',s) or s in {'64','65','66'} or 'financial and insurance' in s: return 'finance'
    if re.search(r'(eco|isic).*[_:-]72(?:\D|$)',s) or s=='72' or 'scientific research and development' in s: return 'rd'
    return None

def fetch_ilostat_table():
    toc_url='https://webapps.ilo.org/ilostat-files/WEB_bulk_download/indicator/table_of_contents_en.csv'
    toc=pd.read_csv(toc_url,low_memory=False)
    label_col=next(c for c in toc.columns if str(c).lower()=='indicator.label')
    freq_col=next(c for c in toc.columns if str(c).lower()=='freq')
    mask=toc[label_col].astype(str).str.contains('Employment by sex, occupation and economic activity - ISIC level 2',case=False,regex=False) & toc[freq_col].astype(str).str.upper().eq('A')
    cand=toc.loc[mask].copy(); cand.to_csv(OUT/'ILOSTAT_toc_matches.csv',index=False)
    if cand.empty: raise RuntimeError('Target ILOSTAT table not found in official TOC')
    # Prefer current employment definition over previous-ICLS variants when exact label is available.
    exact=cand[cand[label_col].astype(str).str.strip().eq('Employment by sex, occupation and economic activity - ISIC level 2 (thousands)')]
    row=(exact.iloc[0] if not exact.empty else cand.iloc[0]); ident=str(row['id'])
    url=f'https://rplumber.ilo.org/files/indicator/{ident}.rds'; path=RAW/f'{ident}.rds'; path.write_bytes(get(url,timeout=300).content)
    obj=pyreadr.read_r(str(path));
    if not obj: raise RuntimeError('RDS returned no data frames')
    df=next(iter(obj.values())); df.to_csv(OUT/'ILOSTAT_raw_target.csv',index=False)
    save({'id':ident,'label':str(row[label_col]),'url':url,'rows':len(df),'columns':list(map(str,df.columns))},'ILOSTAT_download_repair.json')
    return df,ident

def build_F():
    cc=countries(); a=wdi('FD.AST.PRVT.GD.ZS','credit'); a=a[a.iso3.isin(cc)][['iso3','year','credit']].dropna(); a['F_bank']=a.credit/100; return a

def run_e4():
    try:
        raw,ident=fetch_ilostat_table()
        low={str(c).lower():c for c in raw.columns}
        ref=next((low[x] for x in ('ref_area','country_code','iso3') if x in low),None)
        yr=next((low[x] for x in ('time','year') if x in low),None)
        val=next((low[x] for x in ('obs_value','value') if x in low),None)
        sex=next((low[x] for x in ('sex',) if x in low),None)
        cls=[c for c in raw.columns if str(c).lower().startswith('classif') or 'occupation' in str(c).lower() or 'economic_activity' in str(c).lower()]
        if not ref or not yr or not val or len(cls)<2: raise RuntimeError(f'Unexpected ILOSTAT schema: {list(raw.columns)}')
        scores=[]
        for c in cls:
            smp=raw[c].dropna().astype(str).head(20000); scores.append((c,sum(skilled(x) for x in smp),sum(sector(x) is not None for x in smp)))
        occ=max(scores,key=lambda x:x[1])[0]; act=next(x[0] for x in sorted(scores,key=lambda x:x[2],reverse=True) if x[0]!=occ)
        d=raw.copy()
        if sex:
            m=d[sex].map(lambda x: norm(x) in {'sex_t','t','total','all'} or 'total' in norm(x));
            if m.any(): d=d[m].copy()
        d=d[d[occ].map(skilled)].copy(); d['sector']=d[act].map(sector); d=d[d.sector.isin(['finance','rd'])].copy()
        d['year']=pd.to_numeric(d[yr],errors='coerce'); d['value']=pd.to_numeric(d[val],errors='coerce'); d['iso3']=d[ref].astype(str).str.strip(); d=d.dropna(subset=['year','value']); d['year']=d.year.astype(int); d=d[d.year.between(Y0,Y1)]
        agg=d.groupby(['iso3','year','sector'],as_index=False).value.sum().pivot(index=['iso3','year'],columns='sector',values='value').reset_index()
        if not {'finance','rd'}.issubset(agg.columns): raise RuntimeError(f'Finance/R&D sectors not both found; columns={list(agg.columns)}')
        agg=agg.dropna(subset=['finance','rd']); agg['pool']=agg.finance+agg.rd; agg=agg[agg.pool>0]; agg['s_fin']=agg.finance/agg.pool
        m=agg.merge(build_F()[['iso3','year','F_bank']],on=['iso3','year']).dropna().reset_index(drop=True)
        if m.iso3.nunique()<5: raise RuntimeError(f'Too few countries after merge: {m.iso3.nunique()}')
        fit=smf.ols('s_fin ~ F_bank + C(iso3) + C(year)',data=m).fit(cov_type='cluster',cov_kwds={'groups':m.iso3})
        out={'d_target':float(fit.params.F_bank),'d_cluster_se':float(fit.bse.F_bank),'n_obs':int(fit.nobs),'n_countries':int(m.iso3.nunique()),'ilostat_id':ident,'occupation_column_detected':occ,'activity_column_detected':act,'classification_scores':scores}
        save(out,'E4_d_results_repair.json'); m.to_csv(OUT/'E4_estimation_sample_repair.csv',index=False); (OUT/'E4_regression_summary_repair.txt').write_text(fit.summary().as_text(),encoding='utf-8'); return out
    except Exception as e:
        out={'d_target':None,'error':repr(e)}; save(out,'E4_d_results_repair.json'); return out

E1=run_e1(); E4=run_e4(); save({'E1':E1,'E4':E4},'REPAIR_SUMMARY.json'); print(json.dumps({'E1':E1,'E4':E4},indent=2,ensure_ascii=False))
