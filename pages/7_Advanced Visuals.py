# pages/7_Advanced Visuals.py
import streamlit as st
from utils.ui_helpers import render_adv_plot_ui
for k, v in st.session_state.items():
    st.session_state[k] = v


advanced_plot_types = ["3D Scatter", "3D Surface", "Animated Scatter", "Parallel Coordinates", "Radar Chart", "Animated Line Plot"]
st.title("🌌 Advanced Visualizations")

df = st.session_state.get("df")

if df is None:
    st.warning("Please upload and load a dataset first.")
    st.stop()

# State keys for storing last selections
PLOT_TYPE_KEY = "adv_plot_type"
SELECT_STATE = "adv_plot_selections"



st.selectbox(
    "Choose a plot type",
    advanced_plot_types,
    key=PLOT_TYPE_KEY
)

plot_type = st.session_state[PLOT_TYPE_KEY]

render_adv_plot_ui(df, plot_type)


