import streamlit as st
import pandas as pd
import plotly.express as px
from utils.dimensionality_utils import apply_pca, apply_tsne

for k, v in st.session_state.items():
    st.session_state[k] = v

st.title("🔻 Dimensionality Reduction")

df = st.session_state.get("df")
if df is None:
    st.warning("Please upload or load a dataset.")
    st.stop()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if len(numeric_cols) < 2:
    st.warning("You need at least 2 numeric features to apply dimensionality reduction.")
    st.stop()

st.subheader("🔧 Settings")

method = st.selectbox("Select Method", ["PCA", "t-SNE"], key="dr_method")
dimensions = st.radio("Target Dimensions", [2, 3], horizontal=True, key="dr_dimensions")

if method == "t-SNE":
    st.slider("Perplexity", 5, 50, 30, key="dr_perplexity")

if st.button("Apply"):
    data = df[numeric_cols].dropna()
    if method == "PCA":
        reduced_df = apply_pca(data, n_components=dimensions)
    elif method == "t-SNE":
        reduced_df = apply_tsne(data, n_components=dimensions, perplexity=st.session_state["dr_perplexity"])
    else:
        reduced_df = None

    if reduced_df is not None:
        st.session_state["reduced_df"] = reduced_df
        st.success(f"{method} applied. Shape: {reduced_df.shape}")

# Only show plot if reduction was already applied
if "reduced_df" in st.session_state:
    st.subheader("🎨 Visualization")

    color_col = st.selectbox(
        "Color points by (optional)",
        [None] + df.columns.tolist(),
        key="df_color")
    st.session_state["dr_color"] = color_col

    reduced_df = st.session_state["reduced_df"].copy()

    # Handle color alignment
    if color_col and color_col in df.columns:
        color_series = df[color_col].reset_index(drop=True)
        if len(color_series) == len(reduced_df):
            reduced_df["Color"] = color_series
            color_arg = "Color"
        else:
            st.warning("⚠️ Could not align color column. Possibly due to dropped rows.")
            color_arg = None
    else:
        color_arg = None

    # Plotting
    if dimensions == 2:
        fig = px.scatter(reduced_df, x=reduced_df.columns[0], y=reduced_df.columns[1], color=color_arg)
    else:
        fig = px.scatter_3d(reduced_df,
                            x=reduced_df.columns[0],
                            y=reduced_df.columns[1],
                            z=reduced_df.columns[2],
                            color=color_arg)

    st.plotly_chart(fig, use_container_width=True)
