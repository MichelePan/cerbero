import streamlit as st

from app_expo_fx_v6 import render_expo_fx_v6
from app_fore_forecast_fx import render_fore_forecast_fx
from app_weighing_fx import render_weighing_fx

st.set_page_config(
 page_title="FX Suite Dashboard",
 layout="wide",
 initial_sidebar_state="collapsed"
)

st.title("CERBERO Dashboard")
st.caption("EVERYTHING YOU NEED ABOUT FX")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["EXPO FX V6", "FORE / FORECAST FX", "WEIGHING FX"])

with tab1:
 render_expo_fx_v6()

with tab2:
 render_fore_forecast_fx()

with tab3:
 render_weighing_fx()
