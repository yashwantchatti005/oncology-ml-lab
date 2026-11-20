"""Interactive, non-clinical educational model demonstration."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from src.common import load_data

st.set_page_config(page_title='Oncology ML Lab | Research Demo',page_icon='🔬',layout='wide')
st.title('🔬 Oncology ML Lab')
st.caption('Explainable machine learning on the Wisconsin Diagnostic Breast Cancer research dataset')
st.warning('EDUCATIONAL RESEARCH ONLY — not a medical device, screening tool, diagnosis, or treatment advice. Do not use for real patient decisions.')
path=Path(__file__).parent/'models'/'model.joblib'
if not path.exists():
    st.error('Train the model first: python -m src.train --out .');st.stop()
artifact=joblib.load(path)
X,_=load_data();cols=artifact['columns']
st.subheader('Explore a research sample')
idx=st.slider('Sample row in reference dataset',0,len(X)-1,0)
sample=X.iloc[[idx]][cols].copy()
with st.expander('All 30 measurements used by the model'):
    st.dataframe(sample.T.rename(columns={sample.index[0]:'Value'}),use_container_width=True)
st.info('The slider only selects an existing research sample; it does not collect new patient information.')
p=float(artifact['pipeline'].predict_proba(sample)[0,1]);threshold=artifact['threshold']
c1,c2,c3=st.columns(3)
c1.metric('Model score: malignant class',f'{p:.1%}')
c2.metric('Validated decision threshold',f'{threshold:.2f}')
c3.metric('Research-only classification','Malignant' if p>=threshold else 'Benign')
st.progress(min(100,max(0,int(p*100))))
st.caption('Scores may be miscalibrated and do not equal clinical risk. The dataset is small, historical, and non-representative of current clinical populations.')
st.subheader('Model methodology')
st.markdown('Training-only 5-fold CV for model selection; separate validation set for decision threshold; untouched holdout test for final reporting. Logistic regression vs random forest; permutation feature importance and calibration review.')
if (Path(__file__).parent/'reports'/'figures'/'model_evaluation.png').exists():
    st.image(str(Path(__file__).parent/'reports'/'figures'/'model_evaluation.png'),caption='Held-out test evaluation')
