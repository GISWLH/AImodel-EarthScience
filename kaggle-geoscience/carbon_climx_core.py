"""Small, explicit training cores for a three-task research notebook.

These educational kernels do not reproduce the complete best-score pipelines.
Only real historical/labelled training data are accepted. No automatic submissions.
"""
from pathlib import Path
import numpy as np
import pandas as pd

CARBON_AGES=[1,10,20,30,50,70,100,150,200,250,300,350,400,450,500]
CARBON_SCALE=np.array([10.90645,3.39567],dtype=np.float32)

def carbon_forcing_features(x):
    """Actual (site,40years,12months,features) -> reference + seasonal summaries."""
    assert x.ndim==4 and x.shape[1:3]==(40,12) and np.isfinite(x).all()
    annual=x.mean(axis=2,dtype=np.float32);centered_year=np.arange(40,dtype=np.float32)-19.5
    early=annual[:,:5].mean(1);late=annual[:,-5:].mean(1)
    reference=np.concatenate([annual.mean(1),annual.std(1),(annual*centered_year[None,:,None]).sum(1)/(centered_year@centered_year),late,*[annual[:,i:i+10].mean(1) for i in range(0,40,10)],np.clip((late-early)/(np.abs(early)+1e-6),-10,10)],axis=1)
    seasonal=x.mean(1)
    return reference.astype(np.float32),np.concatenate([seasonal.std(1),seasonal.min(1),seasonal.max(1),late-early],axis=1).astype(np.float32)

def carbon_site_split(site_ids,seed=42,validation_fraction=.2):
    ids=np.asarray(site_ids);assert len(np.unique(ids))==len(ids) and len(ids)>=10
    rng=np.random.default_rng(seed);order=rng.permutation(ids);n=max(2,int(len(ids)*validation_fraction))
    train,valid=np.sort(order[n:]),np.sort(order[:n]);assert not np.intersect1d(train,valid).size
    return train,valid

def carbon_training_core(data_root,max_sites=512,seed=42,rounds=200):
    """Fixed LightGBM final-year baseline; all15ages of a site have one split.

    For bounded educational execution use the first max_sites training rows.
    This subset is not representative global validation and must not be used as
    evidence of an improved competitive model. No test inputs are opened.
    """
    from lightgbm import LGBMRegressor
    root=Path(data_root)/'data_global';x=np.load(root/'glob_X_fea.npy',mmap_mode='r')
    n=min(max_sites,len(x));assert n>=10
    reference,seasonal=carbon_forcing_features(np.asarray(x[:n]))
    initial=[];target=[]
    for age in CARBON_AGES:
        y=np.load(root/f'glob_Y{age:03d}.npy',mmap_mode='r');assert y.shape[1]==41
        initial.append(np.asarray(y[:n,0,-1,:],dtype=np.float32));target.append(np.asarray(y[:n,40,-1,:2],dtype=np.float32))
    initial=np.stack(initial,1);target=np.stack(target,1);assert initial.shape==(n,15,7)
    ages=np.asarray(CARBON_AGES,dtype=np.float32)[None,:,None]
    rows=np.concatenate([np.repeat(reference,15,axis=0),initial.reshape(-1,7),(initial/ages).reshape(-1,7),np.tile(CARBON_AGES,n)[:,None]],axis=1).astype(np.float32)
    train,valid=carbon_site_split(np.arange(n),seed)
    train_rows=np.concatenate([np.arange(i*15,(i+1)*15) for i in train]);valid_rows=np.concatenate([np.arange(i*15,(i+1)*15) for i in valid])
    truth=target.reshape(-1,2)/CARBON_SCALE;prediction=np.empty((len(valid_rows),2))
    for k in range(2):
        model=LGBMRegressor(n_estimators=rounds,num_leaves=31,learning_rate=.05,min_child_samples=30,reg_alpha=.5,reg_lambda=1.,random_state=seed,n_jobs=2,verbosity=-1)
        model.fit(rows[train_rows],truth[train_rows,k]);prediction[:,k]=np.maximum(model.predict(rows[valid_rows]),0)
    assert np.isfinite(prediction).all()
    return {'kind':'bounded_real_training_core_execution','representative_validation':False,'official_score':False,'test_inputs_read':False,'training_sites':len(train),'held_sites':len(valid),'all_ages_grouped':True,'seed':seed,'rounds':rounds,'target_year':40,'output_order':['height','agb'],'official_unit_divisors':CARBON_SCALE.tolist(),'normalized_held_MSE':float(np.mean((prediction-truth[valid_rows])**2))}

CLIMX_VARIABLES=['tas','tasmax','tasmin','pr','huss','psl','sfcWind']
CLIMX_PROBS=np.array([.01,.02,.035,.05,.075,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.925,.95,.965,.98,.99])

def climx_month_windows():
    centers=np.round((np.arange(12)+.5)*365/12-.5).astype(int)
    return (centers[:,None]+np.arange(-30,31)[None,:])%365

def climx_fit_quantile_core(design,daily,ridge=None,backend='numpy'):
    """Daily arrays(year,365,cell). Fit each month/quantile/cell on forcing.

    numpy is the default. torch_cuda explicitly uses an already-authorized GPU.
    This core omits the full pipeline's fitted forcing memory/aerosol PCA,
    deseasonalization, calibrated tails, weather ensemble and official indices.
    """
    design=np.asarray(design,dtype=np.float64);daily=np.asarray(daily,dtype=np.float32)
    assert daily.shape[:2]==(len(design),365) and np.isfinite(daily).all()
    windows=climx_month_windows()
    q=np.quantile(daily[:,windows,:],CLIMX_PROBS,axis=2).transpose(1,2,0,3)
    penalties=np.r_[0.,np.ones(design.shape[1]-1)] if ridge is None else np.asarray(ridge)
    a=design.T@design+np.diag(penalties);b=design.T@q.reshape(len(design),-1)
    if backend=='torch_cuda':
        import torch
        assert torch.cuda.is_available()
        with torch.inference_mode():coef=torch.linalg.solve(torch.as_tensor(a,device='cuda'),torch.as_tensor(b,device='cuda')).cpu().numpy()
    else:
        assert backend=='numpy';coef=np.linalg.solve(a,b)
    return np.ascontiguousarray(coef.reshape(design.shape[1],12,len(CLIMX_PROBS),daily.shape[-1]))

def climx_daily_knots(coefficients,x,precipitation=False):
    monthly=np.sort(np.einsum('p,pmkc->mkc',np.asarray(x),coefficients,optimize=False),axis=1)
    if precipitation:monthly=np.maximum(monthly,0)
    position=(np.arange(365)+.5)*12/365-.5;left=np.floor(position).astype(int);w=(position-left)[:,None,None]
    return monthly[left%12]*(1-w)+monthly[(left+1)%12]*w

def climx_map_weather(template,source_knots,target_knots):
    """Same real template days/cells across7variables preserve joint day pairing.

    This bounded reference kernel interpolates on fixed quantile probabilities;
    it clips probabilities to[0,1] and is not the production tail extrapolator.
    """
    output=np.empty_like(template,dtype=np.float32)
    for d in range(len(template)):
        for c in range(template.shape[1]):
            p=np.interp(template[d,c],source_knots[d,:,c],CLIMX_PROBS)
            output[d,c]=np.interp(p,CLIMX_PROBS,target_knots[d,:,c])
    return output

def climx_historical_core(data_root,backend='numpy'):
    """Real4year/4cell mechanics check; year4held from fitting. No rnNSE claim."""
    import xarray as xr
    files=list(Path(data_root).rglob('*hist_train_1850_1931.nc'))
    assert len(files)==1,'Attach competition historical lite NetCDF data'
    with xr.open_dataset(files[0],decode_times=False) as ds:
        assert ds.time.attrs['calendar'] in ('noleap','365_day')
        assert np.array_equal(ds.time.values[:4*365],np.arange(4*365)), 'Expected complete daily years'
        co2=np.asarray(ds.CO2_LBC.isel(forcing_scenario=0).values[:4],dtype=np.float64)
        center=co2[:3].mean();scale=max(float(co2[:3].std()),1e-12);design=np.c_[np.ones(4),(co2-center)/scale]
        raw={}
        for v in CLIMX_VARIABLES:
            arr=np.asarray(ds[v].isel(forcing_scenario=0,time=slice(0,4*365),lat=6,lon=slice(0,4)).values,dtype=np.float32)
            if v=='pr':
                assert ds[v].attrs.get('units')=='kg m-2 s-1';arr*=86400.
            raw[v]=arr.reshape(4,365,4)
    generated={};diagnostics=[]
    for v,arr in raw.items():
        transformed=np.cbrt(np.maximum(arr,0)) if v=='pr' else arr
        coef=climx_fit_quantile_core(design[:3],transformed[:3],backend=backend)
        # The template is year3, always inside the training partition.
        source=climx_daily_knots(coef,design[2],v=='pr');target=climx_daily_knots(coef,design[3],v=='pr')
        z=climx_map_weather(transformed[2],source,target)
        generated[v]=np.maximum(z,0)**3 if v=='pr' else z
        if v in ['pr','huss','sfcWind']:generated[v]=np.maximum(generated[v],0)
    generated['tasmax']=np.maximum(generated['tasmax'],generated['tas']);generated['tasmin']=np.minimum(generated['tasmin'],generated['tas'])
    for v,p in generated.items():
        assert np.isfinite(p).all()
        diagnostics.append({'variable':v,'historical_hold_year':1853,'daily_MAE_in_variable_units':float(np.mean(abs(p-raw[v][3]))),'minimum_generated':float(p.min()),'maximum_generated':float(p.max())})
    return {'kind':'limited_real_historical_mechanics_check','official_rnNSE':None,'representative_validation':False,'test_targets_read':False,'train_years':[1850,1851,1852],'held_year':1853,'cells':4,'backend':backend,'all7daily_fields_generated':True,'diagnostics':diagnostics}
