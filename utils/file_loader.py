import pandas as pd

def sanitize_dataframe_for_streamlit(df):
    df_cleaned = df.copy()

    # Convert problematic string[python] dtype to plain object (str)
    for col in df_cleaned.columns:
        if pd.api.types.is_string_dtype(df_cleaned[col]):
            df_cleaned[col] = df_cleaned[col].astype("object")

    return df_cleaned
def load_file(uploaded_file, sep=",", decimal="."):
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, sep=sep, decimal=decimal)
        else:
            df = pd.read_excel(uploaded_file)
        return sanitize_dataframe_for_streamlit(df)
    except Exception as e:
        raise ValueError(f"File loading error: {e}")