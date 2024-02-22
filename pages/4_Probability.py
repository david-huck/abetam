import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import components.probability as pb


st.markdown(r"# The $\beta$-Distribution")

pb.beta_with_mode_at


modes = np.linspace(0.05, 0.95, 32).round(3)

N = st.slider("N", 1, 10000, 900, step=100)


@st.cache_data
def load_distributions(N):
    distributions = []
    for mode in modes:
        _distribution = pb.beta_with_mode_at(mode, N, interval=(0, 1))
        dist_df = pd.DataFrame(_distribution, columns=["r"])
        dist_df["mode"] = mode
        distributions.append(dist_df)

    return pd.concat(distributions)


distributions = load_distributions(N)

# st.write(distributions)
st.write()
c_scale = px.colors.sequential.Aggrnyl
colors = px.colors.sample_colorscale(c_scale, modes)
color_map = dict(zip(modes, colors))

fig = px.histogram(
    distributions,
    color="mode",
    barmode="overlay",
    histnorm="probability",
    color_discrete_map=color_map,
    # color_continuous_scale=px.colors.diverging.RdBu,
)
fig.update_layout(margin={"r": 0, "l": 0, "t": 0, "b": 0})
st.plotly_chart(fig)

dists_piv = distributions.pivot(columns=["mode"], values="r")
# st.write(dists_piv)
dist_histograms = []
for mode in modes:
    counts, bins = np.histogram(dists_piv[mode], bins=np.linspace(0, 1, 51))
    # st.write(mode_edges)
    _dist_line = pd.DataFrame()
    _dist_line["bins"] = counts / counts.sum()
    _dist_line["edges"] = np.array(bins[:-1]).round(3)
    _dist_line["mode"] = mode
    dist_histograms.append(_dist_line)

dist_img_long = pd.concat(dist_histograms)
dist_img = dist_img_long.pivot(columns="mode", index="edges", values="bins")
# st.write(dist_img)


dist_img_fig = px.imshow(dist_img, aspect="equal", origin='lower')
st.plotly_chart(dist_img_fig)
z = dist_img.values
sh_0, sh_1 = z.shape
y, x = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
fig = go.Figure(
    data=go.Surface(
       z=z, x=x, y=y,
    )
)
fig.update_scenes(
    xaxis_title_text="modes",
    xaxis_autorange="reversed",
    yaxis_title_text="edges",
    yaxis_autorange="reversed",

)
st.plotly_chart(fig)


