import streamlit as st

for k, v in st.session_state.items():
    st.session_state[k] = v

st.title("🕘 Transformation History")

history = st.session_state.get("transformation_history", [])
snapshots = st.session_state.get("df_snapshots", [])

if history:
    for i, record in enumerate(history, 1):
        st.markdown(f"**{i}.** Column `{record['column']}` → `{record['new_column']}` using `{record['method']}`")

    st.divider()
    if st.button("↩️ Undo Last Transformation"):
        if snapshots:
            # Revert to previous snapshot
            st.session_state["df"] = snapshots[-1]
            st.session_state["df_snapshots"] = snapshots[:-1]
            st.session_state["transformation_history"] = history[:-1]
            st.success("Last transformation undone.")
            st.rerun()
        else:
            st.warning("No snapshot available to undo.")
else:
    st.info("No transformation history yet.")
