import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def plot_3d_scatter(df, x, y, z, color=None):
    fig = px.scatter_3d(df, x=x, y=y, z=z, color=color)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig, use_container_width=True)


def plot_3d_surface(df, x, y, z):
    pivot_table = df.pivot_table(index=y, columns=x, values=z)
    if pivot_table.isnull().any().any():
        st.warning("Some combinations of X and Y are missing values for Z. Plot may be distorted or incomplete.")
        pivot_table = pivot_table.fillna(method="ffill").fillna(method="bfill")

    fig = go.Figure(data=[go.Surface(z=pivot_table.values,
                                     x=pivot_table.columns,
                                     y=pivot_table.index)])
    fig.update_layout(scene=dict(
        xaxis_title=x,
        yaxis_title=y,
        zaxis_title=z
    ), margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig, use_container_width=True)


def plot_animated_scatter(df, x, y, animation_col, color=None):
    fig = px.scatter(df, x=x, y=y, animation_frame=animation_col, color=color,
                     range_x=[df[x].min(), df[x].max()],
                     range_y=[df[y].min(), df[y].max()],
                     title=f"{y} vs {x} over {animation_col}")
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)


def plot_parallel_coordinates(df, color_col=None):
    fig = px.parallel_coordinates(df, color=color_col) if color_col else px.parallel_coordinates(df)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig, use_container_width=True)


def plot_radar_chart(stat_dict_before, stat_dict_after=None):
    categories = list(stat_dict_before.keys())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(stat_dict_before.values()), theta=categories, fill='toself', name="Before"))
    if stat_dict_after:
        fig.add_trace(go.Scatterpolar(r=list(stat_dict_after.values()), theta=categories, fill='toself', name="After"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def plot_animated_line(df, x, y, animation_col, color=None):
    fig = px.line(df, x=x, y=y, animation_frame=animation_col, color=color)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig, use_container_width=True)