import pandas as pd
import io

def prepare_export(df, filename="export.csv", file_format="csv"):
    buffer = io.BytesIO()

    if file_format == "csv":
        df.to_csv(buffer, index=False)
        mime = "text/csv"
    elif file_format == "xlsx":
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        raise ValueError("Unsupported file format.")

    buffer.seek(0)
    return buffer, mime