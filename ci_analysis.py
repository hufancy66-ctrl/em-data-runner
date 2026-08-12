#!/usr/bin/env python3
from __future__ import annotations
import io, json, math, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar, brentq

Y0,Y1=1996,2021
OUT=Path('outputs'); RAW=Path('raw'); OUT.mkdir(exist_ok=True); RAW.mkdir(exist_ok=True)
WB='https://api.worldbank.org/v2'
IND={
 'credit_banks_pct_gdp':'FD.AST.PRVT.GD.ZS',
 'credit_private_pct_gdp':'FS.AST.PRVT.GD.ZS',
 'spread_pct_points':'FR.INR.LNDP',
 'rd_pct_gdp':'GB.XPD.RSDV.GD.ZS',
 'researchers_per_million':'SP.POP.SCIE.RD.P6',
 'resident_patents':'IP.PAT.RESD',
 'population':'SP.POP.TOTL'}

def save(x,name):
 (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=float),encoding='utf-8')

def get(url,params=None,timeout=180,retries=4):
 e=None
 for k in range(retries):
  try:
   r=requests.get(url,params=params,timeout=timeout,headers={'User-Agent':'EM-revision-ci/1.0'}); r.raise_for_status(); return r
  except Exception as z: e=z; time.sleep(1.5*(k+1))
 raise e

def countries():
 p=get(WB+'/country',{'format':'json','per_page':500}).json()[1]
 return {x['id'] for x in p if x.get('region',{}).get('id')!='NA'}

def wdi(code,name):
 url=f'{WB}/country/all/indicator/{code}'; pars={'format':'json','date':f'{Y0}:{Y1}','per_page':20000,'page':1}
 p=get(url,pars).json(); rows=list(p[1] or [])
 for pg in range(2,int(p[0].get('pages',1))+1): pars['page']=pg; rows.extend(get(url,pars).json()[1] or [])
 return pd.DataFrame([{'iso3':x.get('countryiso3code'),'country':(x.get('country') or {}).get('value'),'year':int(x['date']),'value':x.get('value')} for x in rows]).rename(columns={'value':name})

def build_panel():
 cc=countries(); p=None
 for name,code in IND.items():
  print('WDI',code,flush=True); d=wdi(code,name); d=d[d.iso3.isin(cc)][['iso3','country','year',name]]
  p=d if p is None else p.merge(d.drop(columns='country'),on=['iso3','year'],how='outer')
 p=p.sort_values(['iso3','year']); p['F_bank']=p.credit_banks_pct_gdp/100; p['spread']=p.spread_pct_points; p['spread_decimal']=p.spread_pct_points/100
 p['patents_pm']=p.resident_patents/p.population*1e6
 p['ln_rd']=np.where(p.rd_pct_gdp>0,np.log(p.rd_pct_gdp),np.nan); p['ln_researchers']=np.where(p.researchers_per_million>0,np.log(p.researchers_per_million),np.nan); p['ln1p_patents_pm']=np.where(p.patents_pm.notna(),np.log1p(p.patents_pm),np.nan)
 for l in (1,2,3):
  p[f'L{l}_ln_rd']=p.groupby('iso3').ln_rd.shift(l); p[f'L{l}_ln_researchers']=p.groupby('iso3').ln_researchers.shift(l); p[f'L{l}_spread_decimal']=p.groupby('iso3').spread_decimal.shift(l)
 p.to_csv(OUT/'clean_panel_1996_2021.csv',index=False)
 cov=[]
 for n in IND:
  x=p.dropna(subset=[n]); cov.append([n,len(x),x.iso3.nunique(),None if x.empty else int(x.year.min()),None if x.empty else int(x.year.max())])
 pd.DataFrame(cov,columns=['variable','n_obs','n_countries','first_year','last_year']).to_csv(OUT/'WDI_variable_coverage.csv',index=False)
 return p

def e1fit(d,eta,cluster=False):
 z=d.copy(); z['e']=np.exp(-eta*z.F_bank); m=smf.ols('spread~e+C(iso3)+C(year)',z)
 return m.fit(cov_type='cluster',cov_kwds={'groups':z.iso3}) if cluster else m.fit()

def run_e1(p):
 d=p[['iso3','year','spread','F_bank']].dropna(); d=d[d.F_bank>=0].reset_index(drop=True); grid=np.geomspace(.02,12,120); rr=[]
 for eta in grid:
  f=e1fit(d,eta); rr.append([eta,float(np.sum(f.resid**2)),float(f.params['e'])])
 prof=pd.DataFrame(rr,columns=['eta','ssr','beta_exp']); prof.to_csv(OUT/'E1_eta_profile.csv',index=False)
 opt=minimize_scalar(lambda x:float(np.sum(e1fit(d,x).resid**2)),bounds=(.02,12),method='bounded'); eta=float(opt.x); fit=e1fit(d,eta,True); beta=float(fit.params.e)
 boundary=eta<=.0206 or eta>=11.64; smin=prof.ssr.min(); weak=min(prof.iloc[12].ssr,prof.iloc[-13].ssr)/smin-1<1e-4; ident=beta>0 and not boundary and not weak
 out={'eta_hat':eta if ident else None,'eta_optimizer_raw':eta,'beta_exp_term':beta,'beta_cluster_se':float(fit.bse.e),'identified':ident,'at_search_boundary':boundary,'weak_profile':weak,'n_obs':int(fit.nobs),'n_countries':int(d.iso3.nunique())}; save(out,'E1_eta_results.json'); (OUT/'E1_regression_summary.txt').write_text(fit.summary().as_text())
 rng=np.random.default_rng(20260812); cs=np.array(sorted(d.iso3.unique())); draws=[]
 for b in range(100):
  parts=[]
  for j,c in enumerate(rng.choice(cs,len(cs),replace=True)):
   x=d[d.iso3==c].copy(); x['iso3']=f'{c}_{j}'; parts.append(x)
  bd=pd.concat(parts)
  try:
   o=minimize_scalar(lambda x:float(np.sum(e1fit(bd,x).resid**2)),bounds=(.02,12),method='bounded'); eh=float(o.x); be=float(e1fit(bd,eh).params.e); draws.append([b,eh,be,be>0 and .0206<eh<11.64])
  except Exception: draws.append([b,np.nan,np.nan,False])
 pd.DataFrame(draws,columns=['draw','eta','beta_exp','valid']).to_csv(OUT/'E1_eta_country_bootstrap.csv',index=False)
 return out

def run_e3(p):
 specs=[]; base=None
 for l in (1,2,3):
  rd=f'L{l}_ln_rd'; rs=f'L{l}_ln_researchers'; d=p[['iso3','year','ln1p_patents_pm',rd,rs]].dropna()
  try:
   f=smf.ols(f'ln1p_patents_pm~{rd}+{rs}+C(iso3)+C(year)',d).fit(cov_type='cluster',cov_kwds={'groups':d.iso3}); phi=float(f.params[rd]); psi=float(f.params[rs]); row={'spec':f'OLS_FE_L{l}','phi':phi,'phi_se':float(f.bse[rd]),'psi':psi,'psi_se':float(f.bse[rs]),'phi_plus_psi':phi+psi,'n_obs':int(f.nobs),'n_countries':int(d.iso3.nunique()),'admissible':0<phi<1 and psi>0 and phi+psi<1}; specs.append(row); (OUT/f'E3_OLS_FE_L{l}_summary.txt').write_text(f.summary().as_text()); base=row if l==1 else base
  except Exception as e: specs.append({'spec':f'OLS_FE_L{l}','error':repr(e),'admissible':False})
  try:
   d2=p[['iso3','year','resident_patents','population',rd,rs]].dropna(); d2=d2[(d2.resident_patents>=0)&(d2.population>0)]
   f=smf.glm(f'resident_patents~{rd}+{rs}+C(iso3)+C(year)',d2,family=sm.families.Poisson(),offset=np.log(d2.population)).fit(cov_type='cluster',cov_kwds={'groups':d2.iso3},maxiter=200); specs.append({'spec':f'PPML_FE_L{l}','phi':float(f.params[rd]),'psi':float(f.params[rs]),'n_obs':int(f.nobs),'n_countries':int(d2.iso3.nunique())})
  except Exception as e: specs.append({'spec':f'PPML_FE_L{l}','error':repr(e)})
 pd.DataFrame(specs).to_csv(OUT/'E3_specification_grid.csv',index=False)
 out={'phi_target':None if base is None else base['phi'],'phi_cluster_se':None if base is None else base['phi_se'],'psi_target':None if base is None else base['psi'],'psi_cluster_se':None if base is None else base['psi_se'],'phi_plus_psi':None if base is None else base['phi_plus_psi'],'n_obs':None if base is None else base['n_obs'],'n_countries':None if base is None else base['n_countries'],'baseline_admissible':False if base is None else base['admissible']}; save(out,'E3_phi_psi_results.json'); return out

def run_e2(p,e3):
 if not e3['baseline_admissible']:
  out={'mapping_conditions_satisfied':False,'m_target':None,'lambda_R_target':None,'reason':'E3 baseline inadmissible'}; save(out,'E2_lambda_m_results.json'); return out
 d=p[['iso3','year','ln_rd','L1_spread_decimal']].dropna(); f=smf.ols('ln_rd~L1_spread_decimal+C(iso3)+C(year)',d).fit(cov_type='cluster',cov_kwds={'groups':d.iso3}); beta=float(f.params.L1_spread_decimal); gamma=-beta; phi=float(e3['phi_target']); mu=float(d.L1_spread_decimal.median()); den=1-gamma*(1-phi)*mu; ok=gamma>0 and den>0; lam=gamma*(1-phi)/den if ok else np.nan
 ref=p[['F_bank','spread_decimal']].dropna(); q10,q90=ref.F_bank.quantile([.1,.9]); low=float(ref.loc[ref.F_bank<=q10,'spread_decimal'].median()); high=float(ref.loc[ref.F_bank>=q90,'spread_decimal'].median()); tau0=high; tau1=max(low-high,0); m=lam*tau1/(1+lam*tau0) if ok else np.nan
 out={'beta_spread_decimal':beta,'beta_cluster_se':float(f.bse.L1_spread_decimal),'gamma_mu':gamma,'phi_used':phi,'mu_bar_decimal':mu,'lambda_R_target':None if not np.isfinite(lam) else float(lam),'mapping_conditions_satisfied':ok,'tau0_reference_high_F_spread_decimal':tau0,'tau1_reference_reducible_spread_decimal':tau1,'m_target':None if not np.isfinite(m) else float(m),'n_obs':int(f.nobs),'n_countries':int(d.iso3.nunique())}; save(out,'E2_lambda_m_results.json'); return out

def discover_wgi_codes():
 p=get(WB+'/indicator',{'format':'json','per_page':50000},timeout=240).json()[1]
 def find(dim):
  cand=[]
  for x in p:
   n=(x.get('name') or '').lower(); code=x.get('id') or ''
   if dim in n and ('absolute' in n or ('score' in n and '0' in n and '100' in n)): cand.append((code,x.get('name')))
  return cand[0] if cand else None
 return find('rule of law'),find('regulatory quality')

def wgi_excel():
 url='https://datacatalogfiles.worldbank.org/ddh-published/0038026/DR0095947/wgidataset_with_sourcedata-2025.xlsx'; path=RAW/'wgi2025.xlsx'; path.write_bytes(get(url,timeout=300).content)
 sheets=pd.read_excel(path,sheet_name=None)
 for sn,d0 in sheets.items():
  for hdr in range(0,6):
   try:
    d=pd.read_excel(path,sheet_name=sn,header=hdr)
    cc=next((c for c in d.columns if 'code' in str(c).lower() or str(c).lower() in {'iso3','economy code'}),None); yc=next((c for c in d.columns if str(c).strip().lower()=='year'),None)
    rl=next((c for c in d.columns if 'rule of law' in str(c).lower() and ('absolute' in str(c).lower() or 'score' in str(c).lower())),None); rq=next((c for c in d.columns if 'regulatory quality' in str(c).lower() and ('absolute' in str(c).lower() or 'score' in str(c).lower())),None)
    if cc and yc and rl and rq:
     w=d[[cc,yc,rl,rq]].copy(); w.columns=['iso3','year','rl','rq']; w.year=pd.to_numeric(w.year,errors='coerce'); w.rl=pd.to_numeric(w.rl,errors='coerce'); w.rq=pd.to_numeric(w.rq,errors='coerce'); w=w.dropna(); w.year=w.year.astype(int); w=w[w.year.between(Y0,Y1)]; w['theta_I']=(w.rl+w.rq)/200; return w[['iso3','year','theta_I']]
   except Exception: pass
 raise RuntimeError('WGI absolute-score columns not detected')

def get_wgi():
 try:
  rl,rq=discover_wgi_codes(); print('WGI codes',rl,rq,flush=True)
  if rl and rq:
   a=wdi(rl[0],'rl'); b=wdi(rq[0],'rq'); w=a[['iso3','year','rl']].merge(b[['iso3','year','rq']],on=['iso3','year']); w['theta_I']=(pd.to_numeric(w.rl)+pd.to_numeric(w.rq))/200; w=w[w.theta_I.between(0,1)]; w[['iso3','year','theta_I']].to_csv(OUT/'prepared_wgi_2025.csv',index=False); return w[['iso3','year','theta_I']],{'source':'WDI API','rl':rl,'rq':rq}
 except Exception as e: print('WGI API discovery failed',repr(e),flush=True)
 w=wgi_excel(); w.to_csv(OUT/'prepared_wgi_2025.csv',index=False); return w,{'source':'WGI 2025 Excel'}

def run_e5(p,e3):
 if not e3['baseline_admissible']:
  out={'zeta_target':None,'reason':'E3 baseline inadmissible'}; save(out,'E5_zeta_results.json'); return out
 try: w,meta=get_wgi()
 except Exception as e:
  out={'zeta_target':None,'error':repr(e)}; save(out,'E5_zeta_results.json'); return out
 d=p.merge(w,on=['iso3','year']); d['F_theta']=d.F_bank*d.theta_I; d=d[['iso3','year','ln1p_patents_pm','F_bank','theta_I','F_theta']].dropna(); f=smf.ols('ln1p_patents_pm~F_bank+theta_I+F_theta+C(iso3)+C(year)',d).fit(cov_type='cluster',cov_kwds={'groups':d.iso3}); beta=float(f.params.F_theta); phi=float(e3['phi_target']); zeta=beta*(1-phi)/phi
 out={'beta_FI':beta,'beta_FI_cluster_se':float(f.bse.F_theta),'zeta_target':zeta,'phi_used':phi,'n_obs':int(f.nobs),'n_countries':int(d.iso3.nunique()),'wgi_meta':meta}; save(out,'E5_zeta_results.json'); (OUT/'E5_regression_summary.txt').write_text(f.summary().as_text()); return out

def fetch_ilo():
 base='https://rplumber.ilo.org/data/indicator/'; tries=[]
 for ident in ('EMP_TEMP_SEX_OCU_ECO_NB_A','EMP_TEMP_SEX_OCU_ECO_NB'):
  for typ in ('label','code'):
   try:
    r=get(base,{'id':ident,'lang':'en','type':typ,'format':'.csv','timefrom':Y0,'timeto':Y1},timeout=300); df=pd.read_csv(io.BytesIO(r.content),low_memory=False); low={str(c).lower() for c in df.columns}; cls=[c for c in df.columns if str(c).lower().startswith('classif') or 'occupation' in str(c).lower() or 'economic_activity' in str(c).lower()]; ok=('time' in low or 'year' in low) and ('obs_value' in low or 'value' in low) and len(cls)>=2; tries.append({'id':ident,'type':typ,'rows':len(df),'cols':list(map(str,df.columns)),'ok':ok})
    if ok: df.to_csv(RAW/'ilostat_e4.csv',index=False); save({'attempts':tries},'ILOSTAT_download.json'); return df
   except Exception as e: tries.append({'id':ident,'type':typ,'error':repr(e)})
 save({'attempts':tries},'ILOSTAT_download.json'); raise RuntimeError('ILOSTAT E4 table download failed')

def txt(x): return '' if pd.isna(x) else str(x).strip().lower()
def skilled(x):
 s=txt(x); return ('professionals' in s and 'associate' not in s) or ('technicians' in s and 'associate professionals' in s) or bool(re.search(r'(isco|ocu).*[_:-][23](?:\D|$)',s))
def sector(x):
 s=txt(x)
 if 'financial and insurance' in s or re.search(r'(isic|eco|ec2).*[_:-](64|65|66)(?:\D|$)',s) or s in {'64','65','66'}: return 'finance'
 if 'scientific research and development' in s or re.search(r'(isic|eco|ec2).*[_:-]72(?:\D|$)',s) or s=='72': return 'rd'
 return None

def run_e4(p):
 try: raw=fetch_ilo()
 except Exception as e: out={'d_target':None,'error':repr(e)}; save(out,'E4_d_results.json'); return out
 low={str(c).lower():c for c in raw.columns}; ref=next((low[x] for x in ('ref_area','country_code','iso3','ref_area.label') if x in low),None); yr=next((low[x] for x in ('time','year') if x in low),None); val=next((low[x] for x in ('obs_value','value') if x in low),None); sx=next((low[x] for x in ('sex','sex.label') if x in low),None); cls=[c for c in raw.columns if str(c).lower().startswith('classif') or 'occupation' in str(c).lower() or 'economic_activity' in str(c).lower()]
 scores=[(c,sum(skilled(x) for x in raw[c].dropna().astype(str).head(10000)),sum(sector(x)!=None for x in raw[c].dropna().astype(str).head(10000))) for c in cls]; occ=max(scores,key=lambda x:x[1])[0]; act=next(x[0] for x in sorted(scores,key=lambda x:x[2],reverse=True) if x[0]!=occ)
 d=raw.copy()
 if sx:
  mask=d[sx].map(lambda x:txt(x) in {'sex_t','t','total','all'} or 'total' in txt(x)); d=d[mask] if mask.any() else d
 d=d[d[occ].map(skilled)].copy(); d['sector']=d[act].map(sector); d=d[d.sector.isin(['finance','rd'])]; d['year']=pd.to_numeric(d[yr],errors='coerce'); d['value']=pd.to_numeric(d[val],errors='coerce'); d['iso3']=d[ref].astype(str).str.strip(); d=d.dropna(subset=['year','value']); d['year']=d.year.astype(int); d=d[d.year.between(Y0,Y1)]
 a=d.groupby(['iso3','year','sector'],as_index=False).value.sum().pivot(index=['iso3','year'],columns='sector',values='value').reset_index(); a=a.dropna(subset=['finance','rd']); a['pool']=a.finance+a.rd; a=a[a.pool>0]; a['s_fin']=a.finance/a.pool; m=a.merge(p[['iso3','year','F_bank']],on=['iso3','year']).dropna(); f=smf.ols('s_fin~F_bank+C(iso3)+C(year)',m).fit(cov_type='cluster',cov_kwds={'groups':m.iso3}); out={'d_target':float(f.params.F_bank),'d_cluster_se':float(f.bse.F_bank),'n_obs':int(f.nobs),'n_countries':int(m.iso3.nunique()),'occupation_column_detected':occ,'activity_column_detected':act}; save(out,'E4_d_results.json'); m.to_csv(OUT/'E4_estimation_sample.csv',index=False); return out

def gates(e1,e2,e3,e4,e5):
 g={'E1_eta_identified':bool(e1.get('identified')),'E3_baseline_admissible':bool(e3.get('baseline_admissible')),'E2_m_mapped':bool(e2.get('mapping_conditions_satisfied') and e2.get('m_target') is not None),'E4_d_available_positive':bool(e4.get('d_target') is not None and e4.get('d_target',0)>0),'E5_zeta_available':bool(e5.get('zeta_target') is not None)}; ready=all(g.values()); out={'gates':g,'all_model_inputs_ready':ready,'C1_status':'READY_FOR_POST_ESTIMATION' if ready else 'BLOCKED','F_star_status':'READY_FOR_POST_ESTIMATION' if ready else 'BLOCKED'}
 if ready:
  eta=float(e1['eta_hat']); m=float(e2['m_target']); phi=float(e3['phi_target']); psi=float(e3['psi_target']); d=float(e4['d_target']); z=float(e5['zeta_target']); theta=.6
  c1=phi*(z*theta+eta*m/(1+m))-psi*d; out['C1_gap_reference_theta_0_6']=c1
  def delta(F): return phi*(z*theta+eta*m*math.exp(-eta*F)/(1+m*math.exp(-eta*F)))-psi*d/(1-d*F)
  if c1>0 and d>0:
   hi=(1/d)*(1-1e-8)
   try: out['F_star_reference_theta_0_6']=float(brentq(delta,0,hi))
   except Exception: out['F_star_reference_theta_0_6']=None
 save(out,'MODEL_GATE_SUMMARY.json'); return out

p=build_panel(); e1=run_e1(p); e3=run_e3(p); e2=run_e2(p,e3); e4=run_e4(p); e5=run_e5(p,e3); print(json.dumps(gates(e1,e2,e3,e4,e5),indent=2,ensure_ascii=False))
