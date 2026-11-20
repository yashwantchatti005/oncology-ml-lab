"""Reproducible exploratory data analysis."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from .common import load_data

def run(out='.'): 
    out=Path(out)/'reports'/'figures';out.mkdir(parents=True,exist_ok=True)
    X,y=load_data()
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    y.map({0:'Benign',1:'Malignant'}).value_counts().plot.bar(ax=axs[0],title='Class balance')
    X.groupby(y)['mean_radius'].plot.hist(alpha=.55,bins=22,ax=axs[1],legend=True)
    axs[1].set_title('Mean radius by diagnosis');axs[1].set_xlabel('Mean radius')
    fig.tight_layout();fig.savefig(out/'eda_overview.png',dpi=180);plt.close(fig)
    lines=[f'Rows: {len(X)}',f'Predictors: {X.shape[1]}',f'Missing cells: {int(X.isna().sum().sum())}',f'Duplicated measurement rows: {int(X.duplicated().sum())}',f'Malignant rate: {y.mean():.2%}']
    (out.parent/'data_audit.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
if __name__=='__main__':run()
