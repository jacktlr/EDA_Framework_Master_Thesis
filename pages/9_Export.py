import streamlit as st
from utils.export_utils import prepare_export

st.title("📤 Export Current Dataset")

df = st.session_state.get("df")

if df is None:
    st.warning("Please upload a dataset first.")
    st.stop()

st.markdown("You can export the **current version** of your dataset to CSV or Excel.")

# Export settings
file_format = st.selectbox("Select export format", ["csv", "xlsx"])
default_filename = f"export.{file_format}"
file_name = st.text_input("Filename", value=default_filename)

if st.button("📦 Generate Download"):
    try:
        buffer, mime = prepare_export(df, filename=file_name, file_format=file_format)
        st.download_button(
            label="⬇️ Download File",
            data=buffer,
            file_name=file_name,
            mime=mime,
        )
    except Exception as e:
        st.error(f"❌ Export failed: {e}")
