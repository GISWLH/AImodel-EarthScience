from pathlib import Path
import numpy as np
import pandas as pd
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

def climx_historical_core(data_root,backend='numpy',return_fields=False):
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
    report={'kind':'limited_real_historical_mechanics_check','official_rnNSE':None,'representative_validation':False,'test_targets_read':False,'train_years':[1850,1851,1852],'held_year':1853,'cells':4,'backend':backend,'all7daily_fields_generated':True,'diagnostics':diagnostics}

    if return_fields:return report,generated,{v:a[3] for v,a in raw.items()}
    return report
