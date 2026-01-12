import streamlit as st
import pandas as pd

from data.preprocess import load_and_aggregate
from algorithms.es import optimise_es

st.set_page_config(page_title="Study Schedule Optimisation (ES)", layout="centered")

st.title("📘 Study Schedule Optimisation Using Evolution Strategies")

# Load Excel data
daily_df = load_and_aggregate()

student = st.selectbox("Select Student", daily_df.index)

original = daily_df.loc[student].values
optimised = optimise_es(original)

days = daily_df.columns.tolist()

st.subheader("Original Daily Study Hours")
st.table(pd.DataFrame(original, index=days, columns=["Hours"]))

st.subheader("Optimised Daily Study Hours (Evolution Strategies)")
st.table(pd.DataFrame(optimised.round(2), index=days, columns=["Hours"]))
