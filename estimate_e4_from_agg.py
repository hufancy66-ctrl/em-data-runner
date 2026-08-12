#!/usr/bin/env python3
import json,time
from pathlib import Path
import pandas as pd
import requests
import statsmodels.formula.api as smf

OUT=Path('outputs_e4_r'); WB='https://api.worldbank.org/v2'; Y0,Y1=1996,2021

def get(url,params=None,timeout=180,retries=4):
    last=None
    for k in range(retries):
        try:
            r=requests.get(url,params=params,timeout=timeout,headers={'User-Agent':'EM-E4-estimator/1.0'}); r.raise_for_status(); return r
        except Exception as e:
            last=e; time.sleep(1.5*(k+1))
    raise last

def countries():
    rows=get(WB+'/country',{'format':'json','per_page':500}).json()[1]
    return {x['id'] for x in rows if x.get('region',{}).get('id')!='NA'}

def credit_panel():
    pars={'format':'json','date':f'{Y0}:{Y1}','per_page':20000,'page':1}; url=f'{WB}/country/all/indicator/FD.AST.PRVT.GD.ZS'; p=get(url,pars).json(); rows=list(p[1] or [])
    for pg in range(2,int(p[0].get('pages',1))+1):
        pars['page']=pg; rows.extend(get(url,pars).json()[1] or [])
    cc=countries(); d=pd.DataFrame([{'iso3':x.get('countryiso3code'),'year':int(x['date']),'credit':x.get('value')} for x in rows]); d=d[d.iso3.isin(cc)].dropna(); d['F_bank']=d.credit/100.0; return d[['iso3','year','F_bank']]

agg=pd.read_csv(OUT/'E4_agg.csv'); agg=agg.rename(columns={'ref_area':'iso3'}); agg['year']=pd.to_numeric(agg.year,errors='coerce'); agg=agg.dropna(subset=['iso3','year','s_fin']); agg['year']=agg.year.astype(int)
m=agg.merge(credit_panel(),on=['iso3','year'],how='inner').dropna().reset_index(drop=True); m.to_csv(OUT/'E4_estimation_sample_final.csv',index=False)
if m.iso3.nunique()<5: raise RuntimeError(f'Too few countries after WDI merge: {m.iso3.nunique()}')
fit=smf.ols('s_fin ~ F_bank + C(iso3) + C(year)',data=m).fit(cov_type='cluster',cov_kwds={'groups':m.iso3})
out={'d_target':float(fit.params.F_bank),'d_cluster_se':float(fit.bse.F_bank),'d_z':float(fit.params.F_bank/fit.bse.F_bank),'n_obs':int(fit.nobs),'n_countries':int(m.iso3.nunique()),'years_min':int(m.year.min()),'years_max':int(m.year.max())}
(OUT/'E4_d_results_final.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); (OUT/'E4_regression_summary_final.txt').write_text(fit.summary().as_text(),encoding='utf-8'); print(json.dumps(out,indent=2))
