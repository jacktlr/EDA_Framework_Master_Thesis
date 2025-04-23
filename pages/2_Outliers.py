import streamlit as st
import pandas as pd
from utils.ui_helpers import show_histograms_side_by_side,select_numeric_column
from utils.outliers import (
    remove_outliers_zscore,
    remove_outliers_iqr,
    winsorize_series,
    quantile_clip,
    mad_based_outlier_removal,
    manual_clip
)
from utils.session_helpers import save_column_to_session

for k, v in st.session_state.items():
    st.session_state[k] = v

st.title("📏 Outlier Treatment")
df = st.session_state.get("df")
if df is not None:
    selected_col = select_numeric_column(df, st)
    original = df[selected_col]

    method = st.selectbox("Outlier handling method", [
        "Z-score",
        "IQR",
        "Winsorization",
        "Quantile Clipping",
        "MAD",
        "Manual Range"
    ])

    cleaned = None

    if method == "Z-score":
        z_thresh = st.slider("Z-score threshold", 1.0, 5.0, 3.0)
        cleaned = remove_outliers_zscore(original, z_thresh)

    elif method == "IQR":
        cleaned = remove_outliers_iqr(original)

    elif method == "Winsorization":
        p = st.slider("Winsorize percent (each side)", 0.01, 0.2, 0.05, step=0.001, format="%0.3f")
        cleaned = winsorize_series(original, lower_percentile=p, upper_percentile=1 - p)

    elif method == "Quantile Clipping":
        lower_q = st.slider("Lower quantile", 0.0, 0.2, 0.01, step=0.005)
        upper_q = st.slider("Upper quantile", 0.8, 1.0, 0.99, step=0.005)
        cleaned = quantile_clip(original, lower=lower_q, upper=upper_q)

    elif method == "MAD":
        mad_thresh = st.slider("MAD threshold", 1.0, 10.0, 3.5, step=0.1)
        cleaned = mad_based_outlier_removal(original, threshold=mad_thresh)

    elif method == "Manual Range":
        min_val = float(st.number_input("Minimum value", value=float(original.min())))
        max_val = float(st.number_input("Maximum value", value=float(original.max())))
        cleaned = manual_clip(original, min_val=min_val, max_val=max_val)

    if cleaned is not None:
        st.markdown("### 📊 Comparison")

        if method == "Winsorization":
            num_changed = (original != cleaned).sum()
            st.write(f"🔢 Total values: {len(original)}")
            st.write(f"🔧 Winsorized values: {num_changed}")
        else:
            removed_values = original.dropna().shape[0] - cleaned.dropna().shape[0]
            missing_values = original.shape[0] - original.dropna().shape[0]
            st.write(f"🔢 Original count: {original.dropna().shape[0]}")
            st.write(f"❓ Original missing value count: {missing_values}")
            st.write(f"🧹 Removed values: {removed_values}")
            st.write(f"✨ After cleaning: {cleaned.dropna().shape[0]}")

        show_histograms_side_by_side(original, cleaned, selected_col, st)
        overwrite = st.checkbox("📝 Overwrite original column?", value=False)

        if st.button("💾 Save cleaned column to session"):
            if overwrite:
                cleaned_col_name = selected_col
            else:
                cleaned_col_name = f"{selected_col}_cleaned_{method.lower().replace(' ', '_')}"
            save_column_to_session(
                df=st.session_state["df"],
                original_col=selected_col,
                new_col_name=cleaned_col_name,
                cleaned_series=cleaned,
                method_label=method
            )

else:
    st.info("Please upload a file to use the outliers interface.")

