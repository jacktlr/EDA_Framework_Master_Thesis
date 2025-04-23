import streamlit as st
import pandas as pd
from utils.impute_utils import impute_missing, get_imputation_methods
from utils.session_helpers import save_column_to_session
from utils.ui_helpers import show_imputation_summary

for k, v in st.session_state.items():
    st.session_state[k] = v

st.title("🧩 Missing Value Imputation")

df = st.session_state.get("df")

if df is None:
    st.warning("Please upload and load a dataset first.")
    st.stop()

col = st.selectbox("Select a column with missing values", df.columns[df.isnull().any()], index=None)

if col:
    st.markdown(f"**Missing values in `{col}`:** {df[col].isnull().sum()}")

    # Select method
    available_methods = get_imputation_methods(df[col])
    method = st.selectbox("Imputation Method", available_methods)

    # Extra options
    const_value = None
    window_size = 3
    n_neighbors = 5  # default

    if method == "Constant":
        const_value = st.text_input("Enter a value to fill missing entries")

    if method == "Rolling Mean":
        window_size = st.slider("Rolling Window Size", 2, 30, 3)

    if method == "KNN":
        if not pd.api.types.is_numeric_dtype(df[col]):
            st.error("KNN imputation can only be applied to numeric columns.")
            st.stop()
        n_neighbors = st.slider("Number of Nearest Neighbors (K)", 1, 15, 5)

    # Impute
    try:
        imputed_series = impute_missing(
            series=df[col],
            method=method,
            constant_value=const_value,
            df_context=df if method == "KNN" else None,
            window_size=window_size,
            n_neighbors=n_neighbors
        )
    except Exception as e:
        st.error(f"Error during imputation: {e}")
        st.stop()

    # Save
    st.divider()


    show_imputation_summary(df[col], imputed_series, st, col)
    overwrite = st.checkbox("📝 Overwrite original column?", value=False)
    new_col_name = col if overwrite else f"{col}_imputed_{method.lower().replace(' ', '_')}"
    if st.button("💾 Save Imputed Column"):
        save_column_to_session(
            df=df,
            original_col=col,
            new_col_name=new_col_name,
            cleaned_series=imputed_series,
            method_label=f"Imputed using {method}",
            container=st
        )
        # Show before/after summary with chart

        st.rerun()

else:
    st.info("👈 Select a column from the list to begin.")

