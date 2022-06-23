import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objs as go


@st.cache(allow_output_mutation=True)
def plot_chart(x_axis, y_axis,
               name,
               color, title,
               yaxis_title, chart='bar'):
    fig = go.Figure()
    if chart == 'bar':
        fig.add_trace(go.Bar(x=x_axis, y=y_axis, name=name, marker={'color': color}))

    if chart == 'line':
        fig.add_trace(
            go.Scatter(x=x_axis, y=y_axis, name=name,
                       line=dict(color=color)))
    if chart == 'pie':
        fig.add_trace(go.Pie(labels=x_axis, values=y_axis, textinfo='label+percent', sort=False))

    fig.update_layout(title=title, xaxis_tickfont_size=14,
                      yaxis=dict(title=yaxis_title),
                      legend=dict(
                          yanchor="top",
                          y=0.99,
                          xanchor="left",
                          x=0.01,
                      )
                      )
    return fig


@st.cache
def plot_best_fit_line(x_axis, y_axis, fit_line, title="Best Fit Line",
                       yaxis_title='Confirm Cases'):
    df = pd.DataFrame(np.hstack((x_axis, y_axis, fit_line)), columns=['x', 'y', 'Py'])

    fig1 = px.line(df, x="x", y="Py")
    fig1.update_traces(line=dict(color='black'))

    fig2 = px.scatter(df, x="x", y="y")

    fig3 = go.Figure(data=fig1.data + fig2.data)
    fig3.update_layout(title=title, xaxis_tickfont_size=14,
                       yaxis=dict(title=yaxis_title),
                       legend=dict(
                           yanchor="top",
                           y=0.99,
                           xanchor="left",
                           x=0.01,
                       )
                       )
    return fig3


@st.cache
def get_world_map(df, case, color):
    fig = px.choropleth(data_frame=df,
                        locations="iso_alpha",
                        color=case,  # value in column 'Confirmed' determines color
                        hover_name="Country",
                        color_continuous_scale=color,  # color scale red, yellow green
                        basemap_visible=False
                        )
    fig.update_geos(projection_type="natural earth")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, width=1600, height=800)
    fig['layout']['geo']['subunitcolor'] = 'rgba(0,0,0,0)'
    return fig
