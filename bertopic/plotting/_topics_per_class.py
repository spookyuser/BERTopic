from typing import List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import normalize


def visualize_topics_per_class(
    topic_model,
    topics_per_class: pd.DataFrame,
    denomenators_per_class: pd.DataFrame = None,
    as_percentage: bool = False,
    top_n_topics: int = 10,
    topics: List[int] = None,
    normalize_frequency: bool = False,
    custom_labels: bool = False,
    width: int = 1250,
    height: int = 900,
) -> go.Figure:
    """Visualize topics per class

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics_per_class: The topics you would like to be visualized with the
                          corresponding topic representation
        top_n_topics: To visualize the most frequent topics instead of all
        topics: Select which topics you would like to be visualized
        normalize_frequency: Whether to normalize each topic's frequency individually
        custom_labels: Whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
        width: The width of the figure.
        height: The height of the figure.

    Returns:
        A plotly.graph_objects.Figure including all traces

    Examples:

    To visualize the topics per class, simply run:

    ```python
    topics_per_class = topic_model.topics_per_class(docs, classes)
    topic_model.visualize_topics_per_class(topics_per_class)
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_topics_per_class(topics_per_class)
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/topics_per_class.html"
    style="width:1400px; height: 1000px; border: 0px;""></iframe>
    """
    colors = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#D55E00",
        "#0072B2",
        "#CC79A7",
    ]

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        selected_topics = list(topics)
    elif top_n_topics is not None:
        selected_topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        selected_topics = sorted(freq_df.Topic.to_list())

    # Prepare data
    if topic_model.custom_labels_ is not None and custom_labels:
        topic_names = {
            key: topic_model.custom_labels_[key + topic_model._outliers]
            for key, _ in topic_model.topic_labels_.items()
        }
    else:
        topic_names = {
            key: value[:40] + "..." if len(value) > 40 else value
            for key, value in topic_model.topic_labels_.items()
        }
    topics_per_class["Name"] = topics_per_class.Topic.map(topic_names)

    data = topics_per_class.loc[topics_per_class.Topic.isin(selected_topics), :]

    # Add traces
    fig = make_subplots(
        rows=3,
        cols=1,
        specs=[[{"type": "bar"}], [{"type": "table"}], [{"type": "table"}]],
    )
    tables = pd.DataFrame()
    for _, topic in enumerate(selected_topics):
        if _ == 0:
            visible = True
        else:
            visible = "legendonly"
        trace_data = data.loc[data.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values

        if normalize_frequency:
            x = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
            xaxis_title = "Normalized Frequency"
        if as_percentage:
            x = []
            debug_info = []
            numerators = []
            denomenators = []
            percentages = []

            for _, class_name in enumerate(trace_data.Class):
                numerator = trace_data.loc[
                    trace_data.Class == class_name, "Frequency"
                ].values[0]
                denomenator = (
                    denomenators_per_class.loc[
                        denomenators_per_class.Class == class_name, "Denomenator"
                    ]
                    .values[0]
                    .item()
                    if denomenators_per_class is not None
                    else trace_data.Frequency.sum()
                )  # i think the else is right but idk
                numerators.append(numerator)
                denomenators.append(denomenator)
                x.append(numerator / denomenator)
                percent_str = f"{numerator / denomenator:.2%}"
                percentages.append(percent_str)

                debug_info.append(
                    f"Equation: <em>{numerator} / {denomenator}</em> = <em>{percent_str}</em>"
                )

            debug_table = pd.DataFrame(
                {
                    "Class": trace_data.Class.apply(lambda x: str.strip(x)),
                    "Numerator": numerators,
                    "Denomenator": denomenators,
                    "Percentage": percentages,
                    "Topic": topic_name,
                }
            )
            tables = pd.concat([tables, debug_table])
            xaxis_title = "Frequency (%)"
        else:
            x = trace_data.Frequency
            xaxis_title = "Frequency"
        fig.add_trace(
            go.Bar(
                y=trace_data.Class,
                x=x,
                visible=visible,
                marker_color=colors[_ % 7],
                hoverinfo="text",
                orientation="h",
                name=topic_name,
                # Show the words in the hover text and debug_info
                hovertext=[
                    f"{word}<br>{debug_info}"
                    for word, debug_info in zip(words, debug_info)
                ],
            ),
        )

    fig.append_trace(
        go.Table(
            header=dict(values=tables.columns.tolist(), align="left"),
            name=topic_name,
            cells=dict(values=tables.to_numpy().T.tolist(), align="left"),
            # Make the last column the widest
            columnwidth=[0.5, 0.3, 0.3, 0.3, 2, 0.1],
        ),
        row=2,
        col=1,
    )
    if denomenators_per_class is not None:
        fig.append_trace(
            go.Table(
                header=dict(
                    values=denomenators_per_class.columns.tolist(), align="left"
                ),
                cells=dict(
                    values=denomenators_per_class.to_numpy().T.tolist(), align="left"
                ),
            ),
            row=3,
            col=1,
        )

    # Also append a trace just showing the denomenators

    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        autosize=True,
        xaxis_title=xaxis_title,
        yaxis_title="Class",
        title={
            "text": f"<b>Topics per Class - v2</b>",
            "y": 0.95,
            "x": 0.40,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        height=height,
        width=width,
        template="simple_white",
        xaxis=dict(
            tickformat=".0%" if as_percentage else "",
        ),
        hoverlabel=dict(
            bgcolor="white", font_size=12, font_family="verdana", align="left"
        ),
        legend=dict(
            title="<b>Global Topic Representation",
        ),
    )
    updatemenu = {
        "buttons": [
            {
                "label": c,
                "method": "update",
                "args": [
                    {
                        "cells": {
                            "values": tables.T.values
                            if c == "All"
                            else tables.loc[tables["Topic"].eq(c)].T.values
                        }
                    }
                ],
            }
            for c in ["All"] + tables["Topic"].unique().tolist()
        ],
        "direction": "down",
        "pad": {"r": 10, "t": 10},
        "showactive": True,
        "x": 1.6,
        "y": 0.6,
    }
    fig["layout"].update(updatemenus=[{}, updatemenu, {}])

    return fig
