import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_utils as pu
from scipy.stats import shapiro, skew
import numpy as np
import io
import base64
import plotly.express as px
from utils.advanced_visuals_utils import (
    plot_3d_scatter,
    plot_3d_surface,
    plot_animated_scatter,
    plot_parallel_coordinates,
    plot_radar_chart,
    plot_animated_line
)

SELECT_STATE = "plot_selections"  # central session state store

for key, val in st.session_state.items():
    st.session_state[key] = val


def create_column_sparkline(series, mode="both", color='skyblue'):
    fig, ax = plt.subplots(figsize=(2.2, 0.45))

    data = series.dropna()

    if mode == "line_chart":
        ax.plot(data.values, color=color, linewidth=1)
    elif mode == "histogram":
        sns.histplot(data, bins=20, ax=ax, color=color)
    else:
        sns.kdeplot(data, ax=ax, color=color, fill=True, linewidth=1.5)

    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.05, dpi=100)
    plt.close(fig)
    encoded = base64.b64encode(buf.getbuffer()).decode("utf-8")
    return f'<img src="data:image/png;base64,{encoded}"/>'


def show_data_preview(df):
    with st.expander("📄 Data Preview", expanded=True):
        # Save original sort column
        if "sort_col" not in st.session_state:
            st.session_state["sort_col"] = df.columns[0]

        sort_col = st.selectbox(
            "Sort by", df.columns,
            key="sort_col_select"
        )
        st.session_state["sort_col"] = sort_col

        sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)
        num_rows = st.slider("Number of rows to view", 5, 100, 10)

        sorted_df = df.sort_values(by=sort_col, ascending=(sort_order == "Ascending"))

        st.dataframe(sorted_df.head(num_rows))

        if st.button("🔁 Overwrite DataFrame with sorted version"):
            st.session_state["df"] = sorted_df
            st.success("DataFrame updated in session.")


def show_data_summary(df):
    with st.expander("📌 Dataset Summary", expanded=False):

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])

        with col2:
            spark_mode = st.radio("Sparkline Type", ["Histogram", "KDE", "Line Chart"], horizontal=True)
            spark_mode = spark_mode.lower().replace(" ", "_")  # turn into valid function input

            summary_df = pd.DataFrame({
                "Type": df.dtypes,
                "Missing": df.isnull().sum()
            }).reset_index().rename(columns={"index": "Column"})

            numeric_cols = df.select_dtypes(include="number").columns
            sparkline_col = []
            for col in summary_df["Column"]:
                if col in numeric_cols:
                    sparkline_col.append(create_column_sparkline(df[col], mode=spark_mode))
                else:
                    sparkline_col.append("")

            summary_df["📈 Sparkline"] = sparkline_col
            summary_df = summary_df[["Column", "Type", "Missing", "📈 Sparkline"]]

            st.markdown(summary_df.to_html(escape=False, index=False), unsafe_allow_html=True)


def select_numeric_column(df,container):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if not numeric_cols:
        container.warning("⚠️ No numeric columns found in this dataset. Please check the import properties.")
        return None
    selected_col = container.selectbox("🔍 Select a numerical column to analyze", numeric_cols)
    if selected_col:
        container.write(f"**Selected column:** `{selected_col}`")
        return selected_col
    return None


def show_histograms_side_by_side(original, transformed, column_name,container):
    col1, col2 = container.columns(2)

    with col1:
        st.write("**Original Distribution**")
        fig1, ax1 = plt.subplots()
        sns.histplot(original.dropna(), bins=30, kde=True, ax=ax1)
        ax1.set_title("Original")
        ax1.set_xlabel(column_name)
        st.pyplot(fig1)

    with col2:
        st.write("**Transformed Distribution**")
        fig2, ax2 = plt.subplots()
        sns.histplot(transformed.dropna(), bins=30, kde=True, ax=ax2, color='orange')
        ax2.set_title("Transformed")
        ax2.set_xlabel(column_name)
        st.pyplot(fig2)


def show_comparison_table(original, transformed, container,max_rows=20):
    container.write("🔍 Sample of Raw Values (Before vs After)")
    comp_df = pd.DataFrame({
        "Original": original,
        "Transformed": transformed
    })
    container.dataframe(comp_df.head(max_rows), use_container_width=True)


def show_distribution_metrics(original, transformed, container):
    with container:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            st.write(f"Skewness: {skew(original.dropna()):.3f}")
            st.write(f"Std Dev: {np.std(original.dropna()):.3f}")
        with col2:
            st.markdown("**Transformed**")
            st.write(f"Skewness: {skew(transformed.dropna()):.3f}")
            st.write(f"Std Dev: {np.std(transformed.dropna()):.3f}")


def show_normality_test(original, transformed, container):
    with container:
        stat_orig, p_orig = shapiro(original.dropna())
        stat_trans, p_trans = shapiro(transformed.dropna())

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original**")
            st.write(f"Statistic: {stat_orig:.3f}")
            st.write(f"P-value: {p_orig:.3f}")
        with col2:
            st.write("**Transformed**")
            st.write(f"Statistic: {stat_trans:.3f}")
            st.write(f"P-value: {p_trans:.3f}")



def ensure_columns_exist(required_types, type_map, container, plot_type):
    for dtype in required_types:
        if not type_map[dtype]:
            container.error(f"❌ No {dtype} columns available for `{plot_type}`.")
            return False
    return True


def ensure_columns_exist(required_types, type_map, container, plot_type):
    for dtype in required_types:
        if not type_map[dtype]:
            container.error(f"❌ No {dtype} columns available for `{plot_type}`.")
            return False
    return True

def render_plot_ui(df, plot_type, container):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include='datetime').columns.tolist()
    all_cols = numeric_cols + categorical_cols + datetime_cols

    type_map = {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols
    }

    st.session_state.setdefault(SELECT_STATE, {})
    st.session_state[SELECT_STATE].setdefault(plot_type, {})
    selections = st.session_state[SELECT_STATE][plot_type]

    if plot_type == "Histogram":
        if not ensure_columns_exist(["numeric"], type_map, container, plot_type):
            return
        st.session_state.setdefault(f"{plot_type}_column", selections.get("column", numeric_cols[0]))
        st.session_state.setdefault(f"{plot_type}_bins", selections.get("bins", 30))

        column = container.selectbox("Select numeric column", numeric_cols, key=f"{plot_type}_column")
        bins = container.slider("Bins", min_value=5, max_value=100, key=f"{plot_type}_bins")

        st.session_state[SELECT_STATE][plot_type] = {"column": column, "bins": bins}
        pu.plot_histogram(df, column, container, bins)

    elif plot_type == "Boxplot":
        if not ensure_columns_exist(["categorical", "numeric"], type_map, container, plot_type):
            return
        st.session_state.setdefault(f"{plot_type}_x", selections.get("x", categorical_cols[0]))
        st.session_state.setdefault(f"{plot_type}_y", selections.get("y", numeric_cols[0]))

        x = container.selectbox("Category (x-axis)", categorical_cols, key=f"{plot_type}_x")
        y = container.selectbox("Numeric (y-axis)", numeric_cols, key=f"{plot_type}_y")

        st.session_state[SELECT_STATE][plot_type] = {"x": x, "y": y}
        pu.plot_boxplot(df, x, y, container)

    elif plot_type == "Scatterplot":
        if not ensure_columns_exist(["numeric"], type_map, container, plot_type):
            return
        st.session_state.setdefault(f"{plot_type}_x", selections.get("x", numeric_cols[0]))
        y_fallback = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        st.session_state.setdefault(f"{plot_type}_y", selections.get("y", y_fallback))
        st.session_state.setdefault(f"{plot_type}_hue", selections.get("hue", "None"))

        x = container.selectbox("X-axis (numeric)", numeric_cols, key=f"{plot_type}_x")
        y = container.selectbox("Y-axis (numeric)", numeric_cols, key=f"{plot_type}_y")
        hue_options = ["None"] + categorical_cols
        hue = container.selectbox("Hue (optional)", hue_options, key=f"{plot_type}_hue")
        hue_val = None if hue == "None" else hue

        st.session_state[SELECT_STATE][plot_type] = {"x": x, "y": y, "hue": hue}
        pu.plot_scatterplot(df, x, y, container, hue=hue_val)

    elif plot_type == "Heatmap (Correlation)":
        if not ensure_columns_exist(["numeric"], type_map, container, plot_type):
            return
        pu.plot_heatmap(df, container)

    elif plot_type == "Pairplot":
        if not ensure_columns_exist(["numeric"], type_map, container, plot_type):
            return
        st.session_state.setdefault(f"{plot_type}_selected", selections.get("selected", numeric_cols[:3]))
        selected = container.multiselect("Select up to 5 numeric columns", numeric_cols, key=f"{plot_type}_selected")

        st.session_state[SELECT_STATE][plot_type] = {"selected": selected}
        pu.plot_pairplot(df, container, columns=selected)

    elif plot_type == "Bar Chart":
        if not ensure_columns_exist(["categorical"], type_map, container, plot_type):
            return

        st.session_state.setdefault(f"{plot_type}_mode", selections.get("mode", "Count Categories"))
        mode = container.radio("Mode", ["Count Categories", "Aggregate Values"], key=f"{plot_type}_mode")

        if mode == "Aggregate Values":
            if not ensure_columns_exist(["numeric"], type_map, container, plot_type):
                return

            st.session_state.setdefault(f"{plot_type}_x_column", selections.get("x_column", categorical_cols[0]))
            st.session_state.setdefault(f"{plot_type}_y_column", selections.get("y_column", numeric_cols[0]))
            st.session_state.setdefault(f"{plot_type}_agg", selections.get("agg", "sum"))

            x_column = container.selectbox("Select categorical column (X-axis)", categorical_cols, key=f"{plot_type}_x_column")
            y_column = container.selectbox("Select numeric column (Y-axis aggregation)", numeric_cols, key=f"{plot_type}_y_column")
            agg_func = container.selectbox("Aggregation function", ["sum", "mean", "count"], key=f"{plot_type}_agg")

            st.session_state[SELECT_STATE][plot_type] = {
                "mode": mode,
                "x_column": x_column,
                "y_column": y_column,
                "agg": agg_func
            }

            pu.plot_bar_chart(df, x_column, y_column, container, mode, agg_func)

        else:
            st.session_state.setdefault(f"{plot_type}_column", selections.get("column", categorical_cols[0]))
            column = container.selectbox("Select categorical column to count", categorical_cols, key=f"{plot_type}_column")

            st.session_state[SELECT_STATE][plot_type] = {"mode": mode, "column": column}
            pu.plot_bar_chart(df, column, None, container, mode, None)

    elif plot_type == "Pie Chart":
        if not ensure_columns_exist(["categorical"], type_map, container, plot_type):
            return

        st.session_state.setdefault(f"{plot_type}_column", selections.get("column", categorical_cols[0]))
        st.session_state.setdefault(f"{plot_type}_limit", selections.get("limit", "All"))

        column = container.selectbox("Select categorical column", categorical_cols, key=f"{plot_type}_column")
        limit = container.selectbox("Show", ["All", "Top 5", "Top 10"], key=f"{plot_type}_limit")

        st.session_state[SELECT_STATE][plot_type] = {"column": column, "limit": limit}
        pu.plot_pie_chart(df, column, limit, container)

    elif plot_type == "Violin Plot":
        if not ensure_columns_exist(["categorical", "numeric"], type_map, container, plot_type):
            return

        st.session_state.setdefault(f"{plot_type}_x", selections.get("x", categorical_cols[0]))
        st.session_state.setdefault(f"{plot_type}_y", selections.get("y", numeric_cols[0]))

        x = container.selectbox("Category (x-axis)", categorical_cols, key=f"{plot_type}_x")
        y = container.selectbox("Numeric (y-axis)", numeric_cols, key=f"{plot_type}_y")

        st.session_state[SELECT_STATE][plot_type] = {"x": x, "y": y}
        pu.plot_violinplot(df, x, y, container)

    elif plot_type == "Swarm Plot":
        if not ensure_columns_exist(["categorical", "numeric"], type_map, container, plot_type):
            return

        st.session_state.setdefault(f"{plot_type}_x", selections.get("x", categorical_cols[0]))
        st.session_state.setdefault(f"{plot_type}_y", selections.get("y", numeric_cols[0]))

        x = container.selectbox("Category (x-axis)", categorical_cols, key=f"{plot_type}_x")
        y = container.selectbox("Numeric (y-axis)", numeric_cols, key=f"{plot_type}_y")

        st.session_state[SELECT_STATE][plot_type] = {"x": x, "y": y}
        pu.plot_swarmplot(df, x, y, container)

    elif plot_type == "Line Plot":
        if not ensure_columns_exist(["numeric"], type_map, container, plot_type):
            return

        x = container.selectbox("X-axis (typically time or ordered)", all_cols, key=f"{plot_type}_x")
        y_selected = container.multiselect("Y-axis variable(s) (numeric)", numeric_cols, key=f"{plot_type}_y")

        # Show format selector only if x is not numeric
        datetime_format = None
        if df[x].dtype in ["object", "category"]:
            datetime_format = container.selectbox(
                "Select datetime format for X-axis parsing",
                options=["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"],
                index=0,
                key=f"{plot_type}_datetime_format"
            )

        smoother = container.selectbox("Add smoother?", ["None", "LOESS", "Rolling Average"], index=0,
                                       key=f"{plot_type}_smoother")

        st.session_state[SELECT_STATE][plot_type] = {
            "x": x, "y": y_selected, "smoother": smoother, "datetime_format": datetime_format
        }

        pu.plot_lineplot(df, x, y_selected, smoother, container, datetime_format)



    else:
        container.warning("Unsupported plot type.")



def render_adv_plot_ui(df,plot_type):
    if SELECT_STATE not in st.session_state:
        st.session_state[SELECT_STATE] = {}
    selections = st.session_state[SELECT_STATE].get(plot_type, {})

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    if plot_type == "3D Scatter":
        prev = st.session_state.get(SELECT_STATE, {}).get(plot_type, {})

        x_val = prev.get("x", numeric_cols[0])
        x = st.selectbox("X-axis", numeric_cols,
                         key=f"{plot_type}_x")

        y_val = prev.get("y", numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])
        y = st.selectbox("Y-axis", numeric_cols,
                         key=f"{plot_type}_y")

        z_val = prev.get("z", numeric_cols[2] if len(numeric_cols) > 2 else numeric_cols[0])
        z = st.selectbox("Z-axis", numeric_cols,
                         key=f"{plot_type}_z")

        color_val = prev.get("color", None)
        color_options = [None] + all_cols
        color = st.selectbox("Color (optional)", color_options,
                             key=f"{plot_type}_color")

        # Warn if same columns are used for multiple axes
        if len({x, y, z}) < 3:
            st.warning("⚠️ Using the same column for multiple axes may result in a flat or meaningless plot.")

        # Save current selections
        st.session_state.setdefault(SELECT_STATE, {})
        st.session_state[SELECT_STATE][plot_type] = {
            "x": x, "y": y, "z": z, "color": color
        }

        plot_3d_scatter(df, x, y, z, color)

    # -------------------------------------------

    elif plot_type == "3D Surface":
        x_key = f"{plot_type}_x"
        y_key = f"{plot_type}_y"
        z_key = f"{plot_type}_z"

        x_default = selections.get("x", numeric_cols[0])
        st.selectbox("X-axis (must be discrete or grid-like)", numeric_cols, key=x_key)

        x_val = st.session_state[x_key]
        y_opts = numeric_cols  # allow duplicates
        y_default = selections.get("y", numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])
        st.selectbox("Y-axis", y_opts, key=y_key)

        y_val = st.session_state[y_key]
        z_opts = numeric_cols
        z_default = selections.get("z", numeric_cols[2] if len(numeric_cols) > 2 else numeric_cols[0])
        st.selectbox("Z-axis", z_opts, key=z_key)

        z_val = st.session_state[z_key]
        if len({x_val, y_val, z_val}) < 3:
            st.warning(
                "⚠️ Using the same column for multiple axes may reduce the effectiveness of the 3D surface plot.")

        st.session_state[SELECT_STATE][plot_type] = {"x": x_val, "y": y_val, "z": z_val}
        plot_3d_surface(df, x_val, y_val, z_val)

    # -------------------------------------------

    elif plot_type == "Animated Scatter":
        x_key = f"{plot_type}_x"
        y_key = f"{plot_type}_y"
        anim_key = f"{plot_type}_animation"
        color_key = f"{plot_type}_color"

        x_default = selections.get("x", numeric_cols[0])
        st.selectbox("X-axis", numeric_cols, key=x_key)

        x_val = st.session_state[x_key]
        y_default = selections.get("y", numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])
        st.selectbox("Y-axis", numeric_cols, key=y_key)

        y_val = st.session_state[y_key]
        anim_default = selections.get("animation", all_cols[0])
        st.selectbox("Animation Frame (e.g., time, year)", all_cols, key=anim_key)

        color_default = selections.get("color", None)
        st.selectbox("Color (optional)", [None] + all_cols, key=color_key)

        if x_val == y_val:
            st.warning("⚠️ X and Y axes are the same — scatter plot will appear as a line or dot.")

        st.session_state[SELECT_STATE][plot_type] = {
            "x": x_val,
            "y": y_val,
            "animation": st.session_state[anim_key],
            "color": st.session_state[color_key]
        }

        plot_animated_scatter(df, x_val, y_val, st.session_state[anim_key], st.session_state[color_key])

    # -------------------------------------------

    elif plot_type == "Parallel Coordinates":
        color_key = f"{plot_type}_color_col"
        color_default = selections.get("color_col", None)

        st.selectbox(
            "Color by (optional)",
            [None] + numeric_cols,
            key=color_key
        )

        st.session_state[SELECT_STATE][plot_type] = {"color_col": st.session_state[color_key]}
        fig = px.parallel_coordinates(df[numeric_cols], color=st.session_state[color_key]) if st.session_state[
            color_key] else px.parallel_coordinates(df[numeric_cols])
        fig.update_layout(margin=dict(l=150, r=50, b=50, t=50))
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # -------------------------------------------

    elif plot_type == "Radar Chart":
        stat_key = f"{plot_type}_stat_col"
        default_col = selections.get("stat_col", numeric_cols[0])

        st.selectbox(
            "Select a numeric column for stats",
            numeric_cols,
            key=stat_key
        )

        stat_col = st.session_state[stat_key]
        st.session_state[SELECT_STATE][plot_type] = {"stat_col": stat_col}
        stat_series = df[stat_col].dropna()
        stats = {
            "Mean": stat_series.mean(),
            "Median": stat_series.median(),
            "Std Dev": stat_series.std(),
            "Skew": stat_series.skew(),
            "Min": stat_series.min(),
            "Max": stat_series.max()
        }
        plot_radar_chart(stats)

    # -------------------------------------------

    elif plot_type == "Animated Line Plot":
        x_key = f"{plot_type}_x"
        y_key = f"{plot_type}_y"
        anim_key = f"{plot_type}_animation"
        color_key = f"{plot_type}_color"

        datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
        x_options = numeric_cols + datetime_cols
        x_default = selections.get("x", x_options[0])
        st.selectbox("X-axis", x_options, key=x_key)

        x_val = st.session_state[x_key]
        y_default = selections.get("y", numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])
        st.selectbox("Y-axis", numeric_cols, key=y_key)

        y_val = st.session_state[y_key]
        anim_default = selections.get("animation", all_cols[0])
        st.selectbox("Animation Frame (e.g., time, year)", all_cols, key=anim_key)

        color_default = selections.get("color", None)
        st.selectbox("Color (optional)", [None] + all_cols, key=color_key)

        if x_val == y_val:
            st.warning("⚠️ X and Y axes are the same — line plot will be flat or redundant.")

        st.session_state[SELECT_STATE][plot_type] = {
            "x": x_val,
            "y": y_val,
            "animation": st.session_state[anim_key],
            "color": st.session_state[color_key]
        }

        plot_animated_line(df, x_val, y_val, st.session_state[anim_key], st.session_state[color_key])

def show_imputation_summary(original, imputed, container, column_name):
    missing_before = original.isnull().sum()
    missing_after = imputed.isnull().sum()
    total = len(original)
    filled = missing_before - missing_after
    fill_pct = (filled / total) * 100

    container.markdown("### 📊 Imputation Summary")
    container.write(f"**Column:** `{column_name}`")
    container.write(f"❓ Missing before: `{missing_before}`")
    container.write(f"✅ Missing after: `{missing_after}`")
    container.write(f"📈 Filled: `{filled}` values ({fill_pct:.2f}%)")

    if pd.api.types.is_numeric_dtype(original):
        container.markdown("#### 🔍 Distribution Comparison")

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(original, label="Original", kde=True, color="gray", alpha=0.5)
        sns.histplot(imputed, label="Imputed", kde=True, color="blue", alpha=0.5)
        ax.legend()
        ax.set_title(f"{column_name} Distribution: Original vs Imputed")
        container.pyplot(fig)
    else:
        container.markdown("#### 📊 Value Counts Comparison")
        original_counts = original.fillna("MISSING").value_counts().sort_index()
        imputed_counts = imputed.fillna("MISSING").value_counts().sort_index()
        counts_df = pd.DataFrame({"Original": original_counts, "Imputed": imputed_counts}).fillna(0)
        container.bar_chart(counts_df)
