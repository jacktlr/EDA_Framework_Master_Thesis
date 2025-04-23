# pages/4_Feature Engineering.py
import streamlit as st
import pandas as pd
from utils.feature_utils import (
    generate_datetime_column,
    create_ratio_column,
    create_manual_bins,
    create_time_bins
)
from utils.session_helpers import save_column_to_session
from datetime import datetime, timedelta

for k, v in st.session_state.items():
    st.session_state[k] = v

st.title("🧠 Feature Engineering")

df = st.session_state.get("df")
if df is None:
    st.warning("Please upload a dataset first.")
    st.stop()

st.subheader("🕒 Generate DateTime Column")
with st.expander("Generate a DateTime column based on start and frequency"):
    start_date = st.date_input("Start Date")
    start_time = st.time_input("Start Time", value=datetime.strptime("00:00:00", "%H:%M:%S").time())
    freq = st.selectbox("Frequency", ["D", "H", "T", "S"], help="Pandas frequency: D=day, H=hour, T=minute, S=second")
    time_col_name = st.text_input("New column name", "generated_datetime")
    if st.button("Generate DateTime"):
        df[time_col_name] = generate_datetime_column(start_date, start_time, freq, len(df))
        save_column_to_session(df, time_col_name, time_col_name, df[time_col_name], method_label="Generated datetime", container=st)
        st.success(f"Added column `{time_col_name}`.")

st.subheader("➗ Create Ratio Column")
with st.expander("Create a new column from A / B"):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    col_a = st.selectbox("Numerator column", numeric_cols, key="ratio_a")
    col_b = st.selectbox("Denominator column", numeric_cols, key="ratio_b")
    ratio_col_name = st.text_input("New column name", f"{col_a}_div_{col_b}")
    if st.button("Create Ratio Column"):
        df[ratio_col_name] = create_ratio_column(df[col_a], df[col_b])
        save_column_to_session(df, ratio_col_name, ratio_col_name, df[ratio_col_name], method_label="Ratio feature", container=st)
        st.success(f"Added column `{ratio_col_name}`.")

st.subheader("🧊 Binning")
with st.expander("Bin a numeric column"):
    bin_col = st.selectbox("Select column to bin", numeric_cols, key="bin_col")
    n_bins = st.slider("Number of bins", 2, 20, 4)
    bin_col_name = st.text_input("New column name", f"{bin_col}_binned")
    if st.button("Create Binned Column"):
        df[bin_col_name] = create_manual_bins(df[bin_col], n_bins)
        save_column_to_session(df, bin_col_name, bin_col_name, df[bin_col_name], method_label=f"Binned into {n_bins} bins", container=st)
        st.success(f"Added binned column `{bin_col_name}`.")
st.subheader("📆 Time-Based Binning")
with st.expander("Bin datetime column by time unit"):
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    if not datetime_cols:
        st.info("No datetime columns found. Try generating one above first.")
    else:
        time_col = st.selectbox("Select datetime column", datetime_cols, key="time_bin_col")
        time_units = {
            "Minute": "T",
            "Hour": "H",
            "Day": "D",
            "Week": "W",
            "Month": "M"
        }
        selected_unit = st.selectbox("Bin by", list(time_units.keys()))
        bin_col_name = st.text_input("New column name", f"{time_col}_{selected_unit.lower()}")

        if st.button("Create Time Bins"):
            binned_col = create_time_bins(df[time_col], time_units[selected_unit])
            df[bin_col_name] = binned_col
            save_column_to_session(df, bin_col_name, bin_col_name, binned_col, method_label=f"Time-binned by {selected_unit}", container=st)
            st.success(f"Added time-binned column `{bin_col_name}`.")