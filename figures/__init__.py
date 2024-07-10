import plotly.graph_objects as go


def sciencify_plotly_fig(fig: go.Figure, font_family="cm", font_size=18) -> go.Figure:
    fig.update_layout(
        plot_bgcolor="white", font_family=font_family, font_size=font_size
    )
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
