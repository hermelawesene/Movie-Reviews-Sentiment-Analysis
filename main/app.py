import pandas as pd 
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

import os
import sys
sys.path.append(os.path.join(os.path.abspath('..')))

# Load once
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")


model = pk.load(open(MODEL_PATH,'rb'))
scaler = pk.load(open(SCALER_PATH,'rb'))
review = st.text_input('Enter Movie Review')

if st.button('Predict'):
    review_scale = scaler.transform([review]).toarray()
    result = model.predict(review_scale)
    if result[0] == 0:
        st.write('Negative Review')
    else:
        st.write('Positive Review')