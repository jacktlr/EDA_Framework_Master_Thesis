import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import matplotlib.dates as mdates
# 1. Histogram
def plot_histogram(df, column, container, bins=30):
    with container:
        subcol1, subcol2, _ = st.columns([1, 3, 1])  # control the width here
        with subcol2:
            st.write(f"📊 Histogram for `{column}`")
            fig, ax = plt.subplots()
            sns.histplot(df[column].dropna(), bins=bins, kde=True, ax=ax)
            ax.set_xlabel(column)
            st.pyplot(fig)

# 2. Boxplot
def plot_boxplot(df, x, y, container):
    with container:
        subcol1, subcol2, _ = st.columns([1, 3, 1])  # control the width here
        with subcol2:
            st.write(f"📦 Boxplot: `{y}` by `{x}`")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=x, y=y, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig)

# 3. Scatterplot
def plot_scatterplot(df, x, y, container, hue=None, max_default_range=15):
    with container:
        subcol1, subcol2, _ = st.columns([3, 1, 1])  # control the width here

        container.write(f"🔘 Scatterplot: `{y}` vs `{x}`")

        plot_df = df.copy()
        if hue in df.columns and (
                isinstance(df[hue].dtype, pd.CategoricalDtype) or df[hue].dtype == object
        ):
            if hue is not None and isinstance(df[hue].dtype, pd.CategoricalDtype) or plot_df[hue].dtype == object:
                categories = sorted(plot_df[hue].dropna().unique().tolist())
                if len(categories) > 1:
                    idx_start, idx_end = container.slider(
                        f"Select range of `{hue}` categories to display in legend:",
                        min_value=0,
                        max_value=len(categories) - 1,
                        value=(0, min(max_default_range - 1, len(categories) - 1)),
                        step=1,
                        format="Index: %d"
                    )

                    selected_hue_categories = categories[idx_start:idx_end + 1]
                    plot_df = plot_df[plot_df[hue].isin(selected_hue_categories)]
                    container.caption(f"Showing `{len(selected_hue_categories)}` of `{len(categories)}` `{hue}` categories.")
        with subcol1:
            fig, ax = plt.subplots(figsize=(6, 4))

            sns.scatterplot(data=plot_df, x=x, y=y, hue=hue, ax=ax)

            # Move the legend if present
            if hue is not None and plot_df[hue].nunique() > 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            st.pyplot(fig)

# 4. Heatmap (Correlation)
def plot_heatmap(df, container):
    with container:
        subcol1, subcol2, _ = st.columns([1, 3, 1])  # control the width here
        with subcol2:
            st.write("🔥 Correlation Heatmap")
            numeric_df = df.select_dtypes(include='number')

            # Dynamically scale figure size
            num_cols = len(numeric_df.columns)
            fig_width = max(8, num_cols * 0.7)  # Adjust scaling factor as needed
            fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.75))

            sns.heatmap(
                numeric_df.corr(),
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                annot_kws={"size": 8},
                ax=ax
            )

            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            st.pyplot(fig)

# 5. Pairplot
def plot_pairplot(df, container, columns=None):
    with container:
        subcol1, subcol2, _ = st.columns([1, 2, 1])  # control the width here
        with subcol2:
            st.write("🔗 Pairplot")
            if columns is None:
                columns = df.select_dtypes(include='number').columns.tolist()[:5]  # limit for performance
            try:
                sns_plot = sns.pairplot(df[columns].dropna())
                st.pyplot(sns_plot)
            except Exception as e:
                st.error(f"❌ Error: {e}")

# 6. Bar Chart
def plot_bar_chart(df, x_column, y_column, container, mode, agg_func):
    with container:
        subcol1, subcol2, _ = st.columns([1, 2, 1])
        with subcol2:
            if mode == "Aggregate Values":
                st.write(f"📊 `{agg_func}` of `{y_column}` by `{x_column}`")
                grouped = df.groupby(x_column)[y_column].agg(agg_func).sort_values(ascending=False)
                y_label = f"{agg_func.capitalize()} of {y_column}"
            else:
                st.write(f"📊 Count of categories in `{x_column}`")
                grouped = df[x_column].value_counts()
                y_label = "Frequency"

            fig, ax = plt.subplots()
            grouped.plot(kind='bar', ax=ax)

            ax.set_xlabel(x_column)
            ax.set_ylabel(y_label)
            st.pyplot(fig)


# 7. Violin Plot
def plot_violinplot(df, x, y, container):
    with container:
        subcol1, subcol2, _ = st.columns([1, 2, 1])  # control the width here
        with subcol2:
            st.write(f"🎻 Violin Plot: `{y}` by `{x}`")

            categories = sorted(df[x].dropna().unique().tolist())
            if len(categories) < 2:
                container.warning("Not enough categories to show a range.")
                return

            # Create a slider to select a subrange of categories
            idx_start, idx_end = container.slider(
                "Select category range to display:",
                min_value=0,
                max_value=len(categories) - 1,
                value=(0, min(9, len(categories) - 1)),
                step=1,
                format="Index: %d"
            )

            selected_range = categories[idx_start:idx_end + 1]
            df_filtered = df[df[x].isin(selected_range)]

            st.caption(f"Showing categories: `{selected_range[0]}` to `{selected_range[-1]}`")

            # Dynamic figure size
            fig_width = max(8, len(selected_range) * 0.6)
            fig, ax = plt.subplots(figsize=(fig_width, 6))

            sns.violinplot(data=df_filtered, x=x, y=y, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            st.pyplot(fig)

# 8. Swarm Plot
def plot_swarmplot(df, x, y, container):
    with container:
        subcol1, subcol2, _ = st.columns([1, 2, 1])  # control the width here
        with subcol2:
            st.write(f"🐝 Swarm Plot: `{y}` by `{x}`")

            categories = sorted(df[x].dropna().unique().tolist())
            if len(categories) < 2:
                container.warning("Not enough categories to show a range.")
                return

            # Range slider to select a subset of categories
            idx_start, idx_end = container.slider(
                "Select category range to display:",
                min_value=0,
                max_value=len(categories) - 1,
                value=(0, min(9, len(categories) - 1)),
                step=1,
                format="Index: %d"
            )

            selected_range = categories[idx_start:idx_end + 1]
            df_filtered = df[df[x].isin(selected_range)]

            st.caption(f"Showing categories: `{selected_range[0]}` to `{selected_range[-1]}`")

            # Responsive figure size based on category count
            fig_width = max(8, len(selected_range) * 0.6)
            fig, ax = plt.subplots(figsize=(fig_width, 6))

            sns.swarmplot(data=df_filtered, x=x, y=y, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            st.pyplot(fig)

# 9. Line Plot
def plot_lineplot(df, x, y_vars, smoother, container, datetime_format=None):
    df = df.copy()
    with container:
        subcol1, subcol2, _ = st.columns([1, 2, 1])
        with subcol2:
            st.write(f"📈 Line Plot: `{', '.join(y_vars)}` over `{x}`")

            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[x]) and df[x].dtype in ["object", "category"]:
                try:
                    df[x] = pd.to_datetime(df[x], format=datetime_format, errors="coerce")
                except Exception as e:
                    st.error(f"Failed to convert `{x}` to datetime with format `{datetime_format}`: {e}")
                    return

            df = df.sort_values(by=x)

            fig, ax = plt.subplots()

            for y in y_vars:
                sns.lineplot(data=df, x=x, y=y, ax=ax, label=y)

                if smoother == "Rolling Average":
                    smoothed = df[y].rolling(window=7, min_periods=1).mean()
                    ax.plot(df[x].values, smoothed.values, linestyle="--", label=f"{y} (7-day avg)")

                elif smoother == "LOESS":
                    try:
                        from statsmodels.nonparametric.smoothers_lowess import lowess
                        loess_smoothed = lowess(df[y], df[x], frac=0.1, return_sorted=False)
                        ax.plot(df[x].values, loess_smoothed, linestyle="--", label=f"{y} (LOESS)")
                    except ImportError:
                        st.error("LOESS smoothing requires `statsmodels`. Please install it.")

            if pd.api.types.is_datetime64_any_dtype(df[x]):
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                fig.autofmt_xdate()

            ax.set_xlabel(x)
            ax.set_ylabel("Value")
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
            st.pyplot(fig)




# 10. Pie chart
def plot_pie_chart(df, column, limit, container):
    with container:
        subcol1, subcol2, _ = st.columns([1, 2, 1])
        with subcol2:
            st.write(f"🥧 Pie Chart for `{column}`")

            value_counts = df[column].value_counts()

            if limit == "Top 5":
                value_counts = value_counts.head(5)
            elif limit == "Top 10":
                value_counts = value_counts.head(10)
            # else: All — no filtering

            fig, ax = plt.subplots()
            ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures pie is circular.

            st.pyplot(fig)

