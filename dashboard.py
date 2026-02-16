#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Config
DATA_DIR = Path("synthetic_output")
DMM_FILE = DATA_DIR / "dmm_synthetic_flat.csv"
#ROOT_CAUSE_FILE = DATA_DIR / "failure_root_causes_with_trend_and_events.csv"
ROOT_CAUSE_FILE = DATA_DIR / "failure_root_causes_with_igp_models.csv"
EVENTS_FILE = DATA_DIR / "network_events.csv"
IGP_FILE = DATA_DIR / "lsp_from_dmm.csv"

st.set_page_config(page_title="Network Failure Analysis", layout="wide")

@st.cache_data
def load_data():
    df_dmm = pd.read_csv(DMM_FILE)
    df_root = pd.read_csv(ROOT_CAUSE_FILE) if ROOT_CAUSE_FILE.exists() else pd.DataFrame()
    df_events = pd.read_csv(EVENTS_FILE) if EVENTS_FILE.exists() else pd.DataFrame()
    df_igp = pd.read_csv(IGP_FILE) if IGP_FILE.exists() else pd.DataFrame()
    return df_dmm, df_root, df_events, df_igp

df_dmm, df_root, df_events, df_igp = load_data()

if not df_root.empty and "timeissued" in df_root.columns:
    df_root = df_root.copy()
    df_root["timeissued"] = (
        pd.to_datetime(df_root["timeissued"], errors="coerce")
        .astype("int64")
    )

st.title(" Network Failure and Root Cause Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total DMM Tests", len(df_dmm))
col2.metric("Failed DMM Tests", (df_dmm["resultclassification"].str.upper() == "FAIL").sum())
if not df_root.empty:
    col3.metric("Predicted Root Causes", len(df_root))
else:
    col3.metric("Predicted Root Causes", "N/A")

st.divider()

st.subheader("DMM Delay Distribution")
fig_delay = px.histogram(
    df_dmm,
    x="delay",
    color="resultclassification",
    nbins=50,
    barmode="overlay",
    title="Distribution of DMM Delays (Pass vs Fail)",
)
st.plotly_chart(fig_delay, use_container_width=True)

st.subheader("Data Preview")
tabs = st.tabs(["DMM Data", "IGP Data", "Events Data", "Root Cause Predictions"])

with tabs[0]:
    st.dataframe(df_dmm.head(50))

with tabs[1]:
    if not df_igp.empty:
        st.dataframe(df_igp.head(50))
    else:
        st.warning("No IGP data available.")

with tabs[2]:
    if not df_events.empty:
        st.dataframe(df_events.head(50))
    else:
        st.warning("No events data available.")

with tabs[3]:
    if not df_root.empty:
        st.dataframe(df_root.head(50))
    else:
        st.warning("No root cause prediction data available.")