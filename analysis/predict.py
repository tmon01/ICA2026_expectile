"""Price new policy rows using frozen fitted models.
python analysis/predict.py policies.csv quotes.csv
Required columns: IDpol, Exposure and the 9 rating features from freMTPL2.
"""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,xgboost as xgb,joblib
FEATURES=['VehPower','VehAge','DrivAge','BonusMalus','VehBrand','VehGas','Area','Density','Region']
CATEGORIES=['VehBrand','VehGas','Area','Region']
def predict(source,destination):
 root=Path(__file__).resolve().parent
 summary=json.loads((root/'summary.json').read_text())
 glm=joblib.load(root/'models/frequency_glm.joblib')
 df=pd.read_csv(source)
 assert df.Exposure.gt(0).all() and df.Density.gt(0).all()
 df['LogDensity']=np.log(df.Density)
 # Preserve train category codes across severity DMatrix construction.
 categories=glm[0].named_transformers_['category'].categories_
 for col,levels in zip(CATEGORIES,categories):
  if not df[col].isin(levels).all():raise ValueError(f'Unseen category in {col}; retraining or an explicit new-category policy is required.')
  df[col]=pd.Categorical(df[col],categories=levels)
 freq=glm.predict(df)
 dm=xgb.DMatrix(df[FEATURES],enable_categorical=True)
 out=df[['IDpol','Exposure']].copy();out['frequency_annual']=freq
 for m,meta in summary['selected'].items():
  b=xgb.Booster();b.load_model(root/'models'/f'{meta["name"]}.json')
  sev=b.predict(dm,iteration_range=(0,meta['best_iteration']+1))
  if not np.all(sev>0):raise ValueError('Nonpositive severity forecast; not silently clipped.')
  out[m+'_severity']=sev
  out[m+'_annual']=freq*sev*meta['multiplier']
  out[m+'_period']=out[m+'_annual']*df.Exposure
 out.to_csv(destination,index=False)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();predict(a.source,a.destination)
