import streamlit as st

def save_column_to_session(df, original_col, new_col_name, cleaned_series, method_label, container=st):
    df_copy = df.copy()
    df_copy[new_col_name] = cleaned_series

    # Update current df in session
    st.session_state["df"] = df_copy

    # Create history storage if not present
    if "transformation_history" not in st.session_state:
        st.session_state["transformation_history"] = []

    if "df_snapshots" not in st.session_state:
        st.session_state["df_snapshots"] = []

    # Save snapshot before transformation
    st.session_state["df_snapshots"].append(df.copy())

    # Save history metadata
    st.session_state["transformation_history"].append({
        "column": original_col,
        "new_column": new_col_name,
        "method": method_label
    })

    container.success(f"✅ Column saved as `{new_col_name}`.")


def set_session_state(key, data):
    try:
        if key not in st.session_state and data is not None:
            st.session_state[key] = data
        else:
            return
    except Exception as e:
        st.error(f"Error: {e}")

