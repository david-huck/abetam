import plotly.graph_objects as go


def sciencify_plotly_fig(fig: go.Figure, font_family="cm", font_size=18, split_annotations=True) -> go.Figure:
    fig.update_layout(
        plot_bgcolor="white", font_family=font_family, font_size=font_size
    )
    if split_annotations:
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )
    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )
    return fig
