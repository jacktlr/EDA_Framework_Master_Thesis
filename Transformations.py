import streamlit as st
import pandas as pd
import numpy as np
from utils.file_loader import load_file
from utils.ui_helpers import show_data_preview, show_data_summary, select_numeric_column, show_histograms_side_by_side, show_comparison_table, show_normality_test,show_distribution_metrics
from utils.transform_recommender import recommend_transformation
from utils.transformations import apply_transformation
import scipy.stats as stats
from utils.ui_helpers import render_plot_ui
from utils.session_helpers import save_column_to_session

for k, v in st.session_state.items():
    st.session_state[k] = v

HEADLINES = ["a", "b", "c"]

st.set_page_config(page_title="EDA Transformation Tool", layout="wide")

st.title("📊 EDA Transformation Framework")
st.markdown("A lightweight tool for data exploration and transformation.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# CSV separator selection
st.sidebar.subheader("⚙️ File Settings")

# Field separator
separator_map = {
    "Comma ( , )": ",",
    "Semicolon ( ; )": ";",
    "Tab ( \\t )": "\t",
    "Pipe ( | )": "|",
    "Space ( )": " "
}
selected_sep_label = st.sidebar.selectbox("CSV Field Separator", list(separator_map.keys()), index=0)
selected_sep = separator_map[selected_sep_label]

# Decimal separator
decimal_map = {
    "Dot ( . )": ".",
    "Comma ( , )": ","
}
selected_decimal_label = st.sidebar.selectbox("Decimal Separator", list(decimal_map.keys()), index=0)
selected_decimal = decimal_map[selected_decimal_label]

# Available transforms
available_transforms = [
    "No transformation",
    "Z-score Standardization",
    "Min-Max Normalization",
    "Log Transform",
    "Square Root Transform",
    "Box-Cox Transform",
    "Quantile Binning"
]

# Initialize df variable
df = None
if "df" not in st.session_state:
    st.session_state.df = None
else:
    df = st.session_state.get("df")
selected_col = None
apply_button = None
# Read the file if uploaded

if uploaded_file is not None:
    try:
        df = load_file(uploaded_file, sep=selected_sep, decimal=selected_decimal)
        file_name = uploaded_file.name
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["df"] = df
        st.success(f"📄 Uploaded file: `{uploaded_file.name}`")
    except Exception as e:
        st.error(f"🚨 Error loading file: {e}")
        df = None  # set df to None in case of failure
else:
    if df is None:
        st.info("📂 Please upload a file to begin.")
    else:
        st.info("📄 Working with file: " + st.session_state["file_name"])

if df is not None:
    # UI logic now extracted to helper functions
    show_data_preview(df)
    show_data_summary(df)

    with st.expander("🧠 Transformation Recommender", expanded=True):
        rec_container = st.container()
        selected_col = select_numeric_column(df,rec_container)

if selected_col is not None:
    col_data = df[selected_col]
    recommendation = recommend_transformation(col_data)
    rec_container.info(f"**Recommended:** {recommendation}")

    rec_container.markdown("🔍 Column statistics")
    desc = col_data.describe()
    rec_container.write(desc)
    skew_val = stats.skew(col_data.dropna())
    rec_container.write(f"**Skewness:** {skew_val:.3f}")

    rec_container.markdown("#### 🔧 Apply Transformation")
    transformation = rec_container.selectbox("Select a transformation", available_transforms, index=0)
    overwrite = rec_container.checkbox("📝 Overwrite original column?", value=False)
    save_button = rec_container.button("💾 Save")

    if transformation != "No transformation":
        try:
            transformed = apply_transformation(df[selected_col], transformation)
            st.session_state["last_transformed"] = {
                "column": selected_col,
                "method": transformation,
                "data": transformed
            }

            with st.expander("📊 Before vs After", expanded=True):
                comparison_container = st.container()
                show_histograms_side_by_side(df[selected_col], transformed, selected_col, comparison_container)
                show_comparison_table(df[selected_col], transformed, comparison_container)
                comparison_container.markdown("📐 Distribution Metrics")
                show_distribution_metrics(df[selected_col], transformed, comparison_container)
                comparison_container.markdown("🧪 Normality Test")
                show_normality_test(df[selected_col], transformed, comparison_container)

                if save_button:
                    transformed_col_name = selected_col if overwrite else f"{selected_col}_cleaned_{transformation.lower().replace(' ', '_')}"
                    save_column_to_session(
                        df=st.session_state["df"],
                        original_col=selected_col,
                        new_col_name=transformed_col_name,
                        cleaned_series=transformed,
                        method_label=transformation,
                        container=comparison_container
                    )
                    st.rerun()  #ensures data preview updates

        except Exception as e:
            st.error(f"❌ Error applying transformation: {e}")
    elif save_button:
        st.warning("Please select a transformation before saving.")