import streamlit as st
import pandas as pd
from utils.ui_helpers import render_plot_ui
from utils.session_helpers import set_session_state

for k, v in st.session_state.items():
    st.session_state[k] = v

st.title("📊 Custom Plotting UI")
df = st.session_state.get("df")
plot_types =    ["Histogram", "Boxplot", "Scatterplot", "Heatmap (Correlation)",
                "Pairplot", "Bar Chart", "Pie Chart", "Violin Plot", "Swarm Plot", "Line Plot"]
# Default value
if "plot_type" not in st.session_state:
    set_session_state("plot_type", plot_types[0])

if df is not None:
    plot_type = st.selectbox(
        "Select plot type", plot_types,
        key="plot_type_selectbox")
    st.session_state["plot_type"] = st.session_state["plot_type_selectbox"]

    plot_container = st.container()
    render_plot_ui(df, plot_type, plot_container)
    print(st.session_state["plot_type"])
else:
    st.info("Please upload a file to use the plotting interface.")
