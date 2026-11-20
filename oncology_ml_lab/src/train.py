"""Run: python -m src.train --out .
Strictly separate train, validation and final test.
No patient identifiers or leakage-prone outcome-derived features used.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, recall_score,
    precision_score, f1_score, fbeta_score, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay, brier_score_loss)
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from .common import load_data,validate_input

def bootstrap_auc_interval(y, proba, n=1000, seed=42):
    rng=np.random.default_rng(seed); y=np.asarray(y); proba=np.asarray(proba); values=[]
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y))
        if len(np.unique(y[idx]))==2: values.append(roc_auc_score(y[idx],proba[idx]))
    return [round(float(v),4) for v in np.percentile(values,[2.5,97.5])]

def run(out):
    out=Path(out); (out/'reports'/'figures').mkdir(parents=True,exist_ok=True);(out/'models').mkdir(exist_ok=True);(out/'data').mkdir(exist_ok=True)
    X,y=load_data();validate_input(X)
    pd.concat([X,y],axis=1).to_csv(out/'data'/'wisconsin_breast_cancer.csv',index=False)
    X_trainval,X_test,y_trainval,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    X_train,X_val,y_train,y_val=train_test_split(X_trainval,y_trainval,test_size=.25,random_state=42,stratify=y_trainval)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    models={
        'Logistic regression':(Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LogisticRegression(max_iter=3000,random_state=42))]),{'model__C':[.01,.1,1,10]}),
        'Random forest':(Pipeline([('impute',SimpleImputer(strategy='median')),('model',RandomForestClassifier(random_state=42,n_jobs=-1))]),{'model__n_estimators':[200], 'model__max_depth':[None,5,10], 'model__min_samples_leaf':[1,3]})
    }
    comparisons=[];best=None;winner=None
    for name,(pipe,grid) in models.items():
        search=GridSearchCV(pipe,grid,cv=cv,scoring='roc_auc',n_jobs=-1,refit=True)
        search.fit(X_train,y_train)
        val_auc=roc_auc_score(y_val,search.predict_proba(X_val)[:,1])
        comparisons.append({'model':name,'cv_roc_auc':round(float(search.best_score_),4),'val_roc_auc':round(float(val_auc),4),'best_params':str(search.best_params_)})
        if best is None or search.best_score_>best.best_score_:
            best=search;winner=name
    # Model picked by training CV only; decision threshold optimized on held-out validation only.
    val_proba=best.predict_proba(X_val)[:,1]
    candidates=np.linspace(.05,.95,181)
    scores=[fbeta_score(y_val,(val_proba>=t).astype(int),beta=2,zero_division=0) for t in candidates]
    threshold=float(candidates[int(np.argmax(scores))])
    prob=best.predict_proba(X_test)[:,1]; pred=(prob>=threshold).astype(int)
    metrics={'dataset':'Wisconsin Diagnostic Breast Cancer','target_positive':'malignant',
        'sample_count':len(y),'train_count':len(y_train),'validation_count':len(y_val),'test_count':len(y_test),
        'selected_model':winner,'threshold':round(threshold,3),'test_roc_auc':round(float(roc_auc_score(y_test,prob)),4),
        'test_auc_bootstrap_95pct_ci':bootstrap_auc_interval(y_test,prob),
        'test_average_precision':round(float(average_precision_score(y_test,prob)),4),
        'test_recall_malignant':round(float(recall_score(y_test,pred)),4),
        'test_precision_malignant':round(float(precision_score(y_test,pred,zero_division=0)),4),
        'test_f1':round(float(f1_score(y_test,pred)),4),
        'test_f2':round(float(fbeta_score(y_test,pred,beta=2)),4),
        'test_brier_score':round(float(brier_score_loss(y_test,prob)),4),
        'confusion_matrix_rows_true_0_1':confusion_matrix(y_test,pred).tolist(),
        'comparison':comparisons,'notes':['Not clinically validated','No prospective or external validation','Bootstrap CI is internal, based on small test sample'] }
    (out/'reports'/'metrics.json').write_text(json.dumps(metrics,indent=2))
    joblib.dump({'pipeline':best.best_estimator_,'threshold':threshold,'columns':list(X.columns)},out/'models'/'model.joblib')
    pd.DataFrame({'actual':y_test.to_numpy(),'probability_malignant':prob,'prediction':pred}).to_csv(out/'reports'/'holdout_predictions.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    RocCurveDisplay.from_predictions(y_test,prob,ax=axes[0],name=winner)
    PrecisionRecallDisplay.from_predictions(y_test,prob,ax=axes[1],name=winner)
    ConfusionMatrixDisplay.from_predictions(y_test,pred,display_labels=['Benign','Malignant'],ax=axes[2],cmap='Blues',colorbar=False)
    fig.tight_layout();fig.savefig(out/'reports'/'figures'/'model_evaluation.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); CalibrationDisplay.from_predictions(y_test,prob,n_bins=8,strategy='quantile',ax=ax);fig.tight_layout();fig.savefig(out/'reports'/'figures'/'calibration.png',dpi=170);plt.close(fig)
    result=permutation_importance(best.best_estimator_,X_test,y_test,scoring='roc_auc',n_repeats=12,random_state=42,n_jobs=-1)
    order=np.argsort(result.importances_mean)[-12:]
    fig,ax=plt.subplots(figsize=(8,6));ax.barh(np.array(X.columns)[order],result.importances_mean[order],xerr=result.importances_std[order]);ax.set_xlabel('Decrease in test ROC AUC after feature shuffle');ax.set_title('Permutation importance (holdout; descriptive only)');fig.tight_layout();fig.savefig(out/'reports'/'figures'/'feature_importance.png',dpi=170);plt.close(fig)
    print(json.dumps(metrics,indent=2))
    return metrics

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='.');args=parser.parse_args();run(args.out)
