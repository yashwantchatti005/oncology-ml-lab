import numpy as np
from sklearn.model_selection import train_test_split
from src.common import load_data,validate_input
from src.train import bootstrap_auc_interval

def test_data_shape_and_target():
    X,y=load_data()
    assert X.shape==(569,30)
    assert set(y.unique())=={0,1}
    assert y.sum()==212
    validate_input(X)

def test_no_leakage_columns():
    X,_=load_data()
    assert not any(x in X.columns for x in ['target','diagnosis','malignant','patient_id'])

def test_split_is_disjoint():
    X,y=load_data()
    tr,te,_,_=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    assert set(tr.index).isdisjoint(te.index)

def test_bootstrap_interval():
    ci=bootstrap_auc_interval([0,0,1,1,0,1],[.1,.2,.8,.9,.3,.7],n=100,seed=4)
    assert ci==[1.0,1.0]
