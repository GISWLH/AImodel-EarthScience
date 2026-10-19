"""Interpretable climate-history fingerprints for forest ecosystem emulation.

A transparent reference pipeline for site-grouped research.
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

"""Bounded reads of public climate inputs; no future targets or fitted predictions."""
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

def read_npz_rows(path,key,rows=64):
    # Reading a prefix avoids materializing a multi-GB array just for an input plot.
    with zipfile.ZipFile(path) as z:
        with z.open(key+'.npy') as member:
            version=np.lib.format.read_magic(member)
            if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(member)
            elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(member)
            else:raise ValueError('Unsupported NumPy header version')
            assert not fortran and not dtype.hasobject
            n=min(rows,shape[0]);needed=int(n*np.prod(shape[1:])*dtype.itemsize)
            raw=member.read(needed);assert len(raw)==needed
            return np.frombuffer(raw,dtype=dtype).reshape((n,*shape[1:])).copy(),shape

def inspect_public_forcing(root,rows=64):
    paths={s:list(Path(root).rglob('test_'+s+'.npz')) for s in ['ssp126','ssp585']}
    values={};shapes={};records=[]
    for scenario,files in paths.items():
        assert len(files)==1,'Attach CarbonGlobe public inputs or specify CARBON_PUBLIC_ROOT'
        forcing,shape=read_npz_rows(files[0],'test_x',rows);assert forcing.shape[1:3]==(40,12)
        initial,initial_shape=read_npz_rows(files[0],'test_y0',rows)
        assert initial.shape[1:]==(15,7) and np.isfinite(forcing).all()
        reference,seasonal=carbon_forcing_features(forcing)
        values[scenario]={'forcing':forcing,'initial':initial,'reference':reference,'seasonal':seasonal};shapes[scenario]=list(shape)
        records.append({'scenario':scenario,'inspected_sites':len(forcing),'provided_sites':shape[0],'years':shape[1],'months':shape[2],'drivers':shape[3],'reference_features':reference.shape[1],'seasonal_features':seasonal.shape[1]})
    return values,pd.DataFrame(records)
