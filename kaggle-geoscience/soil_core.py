"""Train-only, standalone physical texture baseline and grouped domain audit.
Original educational implementation by giswlh, AI-assisted, 2026-10-19.
"""
import os
from pathlib import Path
import re
import unicodedata
import json
import platform
import hashlib
import time
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import sklearn
from PIL import Image, ImageOps
import PIL

START=time.monotonic()
DIAMETERS=np.array([.002,.0063,.02,.063,.2,.63,2,6.3,20,63,200])
COLUMNS=['0.002','0.0063','0.02','0.063','0.2','0.63','2','6.3','20','63','200']
TARGET_PPM=4.5
ALPHA=1.0

def canonical(text):
    text=str(text).translate(str.maketrans({'ü':'ue','ö':'oe','ä':'ae','ß':'ss'}))
    return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',text).encode('ascii','ignore').decode().lower())

def emd(truth,prediction):
    return np.sum(np.abs(np.asarray(truth)[:,:-1]-np.asarray(prediction)[:,:-1])*np.diff(np.log10(DIAMETERS)),axis=1)

def check_cdf(values):
    a=np.asarray(values)
    assert a.ndim==2 and a.shape[1]==11 and np.isfinite(a).all()
    assert a.min()>=0 and a.max()<=100 and (np.diff(a,axis=1)>=-1e-10).all()
    assert np.all(a[:,-1]==100.)

def extract_features(root,labels):
    ppm=pd.read_csv(root/'ppm_updated.csv').set_index('phone')
    ids=list(labels.index);phones=sorted(ppm.index,key=lambda p:len(canonical(p)),reverse=True);records=[]
    # Only Training directories are opened. No test photographs or hidden labels.
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.jpg','.jpeg'}:continue
        if not any('training' in part.lower() for part in path.relative_to(root).parts):continue
        key=canonical(path.stem);matches=[sid for sid in ids if canonical(sid) in key]
        phone=next((p for p in phones if key.startswith(canonical(p))),None)
        if len(matches)!=1 or phone is None:raise ValueError('Ambiguous training photograph mapping: '+path.name)
        with Image.open(path) as im:
            # Scale from the original file dimensions before EXIF orientation.
            effective_ppm=float(ppm.loc[phone,'ppm'])*max(im.size)/max(ppm.loc[phone,'width'],ppm.loc[phone,'height'])
            image=ImageOps.exif_transpose(im).convert('L')
            scale=TARGET_PPM/effective_ppm
            image=image.resize((max(8,round(image.width*scale)),max(8,round(image.height*scale))),Image.Resampling.LANCZOS)
            gray=np.asarray(image,dtype=np.float32)
        height,width=gray.shape;gray=gray[height//4:3*height//4,width//4:3*width//4]
        contrast=float(gray.std())+1e-6
        smooth=[gaussian_filter(gray,sigma=mm*TARGET_PPM) for mm in [.25,.5,1,2,4,8]]
        features=[float(np.log(np.std(a-b)/contrast+1e-6)) for a,b in zip(smooth[:-1],smooth[1:])]
        records.append({'sample_id':matches[0],'phone':phone,'file':path.name,**{f'dog_{i}':v for i,v in enumerate(features)}})
    frame=pd.DataFrame(records)
    assert set(frame.sample_id)==set(ids),'Missing training photo coverage'
    assert np.isfinite(frame.filter(like='dog_').to_numpy()).all()
    return frame

def proxy_sites(ids):
    # Development grouping heuristic only: these are NOT verified geographic sites.
    group=0;previous=None;result={}
    for sid in sorted(ids,key=lambda s:(s[0],int(s[1:]))):
        if previous is None or sid[0]!=previous[0] or int(sid[1:])-int(previous[1:])>12:group+=1
        result[sid]=group;previous=sid
    return result

def predict(fit,valid,labels):
    assert not set(fit.sample_id)&set(valid.sample_id),'Soil sample leakage'
    cols=[f'dog_{i}' for i in range(5)]
    # All fitted statistics and model weights are fold-local; each soil has total weight1.
    weights=1/fit.groupby('sample_id').sample_id.transform('size').to_numpy()
    scaler=StandardScaler().fit(fit[cols],sample_weight=weights)
    x=scaler.transform(fit[cols]);v=scaler.transform(valid[cols])
    y=labels.loc[fit.sample_id].to_numpy()/100
    mass=np.diff(np.c_[np.zeros(len(y)),y],axis=1)
    assert mass.min()>=-1e-10
    model=Ridge(alpha=ALPHA).fit(x,np.sqrt(np.maximum(mass,0)),sample_weight=weights)
    root_mass=np.maximum(model.predict(v),0);mass=root_mass**2
    zero=mass.sum(axis=1)<=1e-12;mass[zero]=1.
    mass/=mass.sum(axis=1,keepdims=True)
    curve=np.cumsum(mass,axis=1)*100;curve=np.clip(curve,0,100);curve[:,-1]=100
    result=pd.DataFrame(curve,index=valid.sample_id,columns=COLUMNS).groupby(level=0).mean()
    check_cdf(result);return result

def evaluate(photos,labels,sites):
    protocols={
        'leave_one_soil_out':{'groups':{s:s for s in labels.index},'train_phone':None,'valid_phone':None},
        'leave_proxy_site_out':{'groups':sites,'train_phone':None,'valid_phone':None},
        'Motorola_to_Samsung':{'groups':sites,'train_phone':'Motorola Edge','valid_phone':'Samsung A52'},
        'Samsung_to_Motorola':{'groups':sites,'train_phone':'Samsung A52','valid_phone':'Motorola Edge'},
    }
    summaries=[];oof_rows=[]
    for name,cfg in protocols.items():
        available=photos if cfg['valid_phone'] is None else photos[photos.phone==cfg['valid_phone']]
        predicted=[]
        for group in sorted(set(cfg['groups'].values()),key=str):
            held={s for s,g in cfg['groups'].items() if g==group}
            fit=photos[~photos.sample_id.isin(held)];valid=available[available.sample_id.isin(held)]
            if cfg['train_phone'] is not None:fit=fit[fit.phone==cfg['train_phone']]
            if valid.empty:continue
            assert len(set(fit.sample_id))>=2 and not set(fit.sample_id)&held
            p=predict(fit,valid,labels);error=emd(labels.loc[p.index].to_numpy(),p.to_numpy())
            for sid,e,curve in zip(p.index,error,p.to_numpy()):
                predicted.append(float(e));oof_rows.append({'protocol':name,'sample_id':sid,'held_group':group,'emd':float(e),**dict(zip(COLUMNS,curve.tolist()))})
        assert len(predicted)==available.sample_id.nunique()
        summaries.append({'protocol':name,'n_soils':len(predicted),'mean_emd':float(np.mean(predicted)),'worst_soil_emd':float(max(predicted))})
    return summaries,pd.DataFrame(oof_rows)

def run_soil_training_audit():
    override=os.environ.get('SOIL_DATA_ROOT')
    if override:root=Path(override)
    else:
        files=list(Path('/kaggle/input').rglob('Training_labels_updated.csv'))
        if len(files)!=1:raise RuntimeError('Attach soil-grain-size-from-photos competition data; expected one updated training label file.')
        root=files[0].parent
    out=Path(os.environ.get('SOIL_OUTPUT_ROOT','/kaggle/working'));out.mkdir(parents=True,exist_ok=True)
    labels=pd.read_csv(root/'Training_labels_updated.csv',dtype={'sample_id':str}).set_index('sample_id')
    labels.columns=[format(float(c),'g') for c in labels.columns]
    labels=labels[COLUMNS];check_cdf(labels)
    photos=extract_features(root,labels);sites=proxy_sites(labels.index)
    summary,oof=evaluate(photos,labels,sites)
    report={'method':'fixed physical DoG + sqrt-mass Ridge','hyperparameters':{'target_ppm':TARGET_PPM,'alpha':ALPHA,'gaussian_scales_mm':[.25,.5,1,2,4,8]},'n_training_soils':len(labels),'n_training_photos':len(photos),'n_proxy_groups':len(set(sites.values())),'results':summary,'formal_submission':False,'leaderboard_score':None,'test_photos_read':False,'labels_sha256':hashlib.file_digest((root/'Training_labels_updated.csv').open('rb'),'sha256').hexdigest(),'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'Pillow':PIL.__version__},'elapsed_seconds':time.monotonic()-START}
    (out/'validation_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');oof.to_csv(out/'grouped_oof.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False));print(json.dumps({k:report[k] for k in ['n_training_soils','n_training_photos','n_proxy_groups','elapsed_seconds','test_photos_read','formal_submission']},indent=2))
    return report,oof

