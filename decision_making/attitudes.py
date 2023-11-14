import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from functools import partial


def simple_diff(atts, inertia=0.5):
    diff = (atts[0] - atts[1]) * (1 - inertia)

    if atts[0] - diff < -1:
        return -1
    elif atts[0] - diff > 1:
        return 1
    else:
        return atts[0] - diff


def scaled_diff(atts, inertia=0.5):
    diff = (atts.iloc[0] - atts.iloc[1]) * (1 - inertia)
    diff *= 1 - abs(atts.iloc[0])
    return atts.iloc[0] - diff


def sigmoid_diff(atts, inertia=0.5):
    diff = (atts.iloc[0] - atts.iloc[1]) * (1 - inertia)

    return 2 / (1 + np.e ** -(2 * (atts.iloc[0] - diff))) - 1


def show_diff_funcs(inertia=0.5):
    att_1 = np.linspace(-1, 1, 50)
    att_2 = np.linspace(-1, 1, 50)
    df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [["simple_diff", "scaled_diff", "sigmoid_diff"], att_1, att_2]
        ),
        columns=["new_att"],
    )

    df.sort_index(inplace=True)

    df.loc["simple_diff", "new_att"] = (
        df.loc["simple_diff", :]
        .reset_index()[["level_0", "level_1"]]
        .apply(partial(simple_diff, inertia=inertia), axis=1)
        .values
    )
    df.loc["scaled_diff", "new_att"] = (
        df.loc["scaled_diff", :]
        .reset_index()[["level_0", "level_1"]]
        .apply(partial(scaled_diff, inertia=inertia), axis=1)
        .values
    )
    df.loc["sigmoid_diff", "new_att"] = (
        df.loc["sigmoid_diff", :]
        .reset_index()[["level_0", "level_1"]]
        .apply(partial(sigmoid_diff, inertia=inertia), axis=1)
        .values
    )

    im_data = df.reset_index().pivot(
        index=["level_0", "level_1"], columns="level_2", values="new_att"
    )
    functions = im_data.index.get_level_values(0).unique()
    # im_data
    im_data = np.array(
        [im_data.loc[idx, :] for idx in im_data.index.get_level_values(0).unique()]
    )
    fig = px.imshow(
        im_data,
        x=att_1,
        y=att_2,
        facet_col=0,
        color_continuous_scale="balance",
        origin="lower",
    )

    for i, function in enumerate(functions):
        fig.layout.annotations[i]["text"] = f"diff func = {function}"
    fig.update_layout(
        yaxis=dict(title="Attitude of agent i"),
        xaxis2=dict(title="Attitude of agent j"),
        coloraxis_colorbar=dict(
            title="Agent i's attitude<br>after interaction",
        ),
        margin={"t": 0, "b": 0},
        height=300,
    )

    for annot in fig.layout.annotations:
        annot.y *= 0.8
    st.plotly_chart(fig)


if __name__ == "__main__":
    run()
