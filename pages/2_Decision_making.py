import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from decision_making.attitudes import show_diff_funcs
from decision_making.mcda import normalize, calc_score
from components.technologies import merge_heating_techs_with_share
from components.model import TechnologyAdoptionModel
from functools import partial

st.set_page_config(page_title="Decision Making")
model = TechnologyAdoptionModel(10, "Canada")
sample_agent = model.schedule.agents[0]


st.markdown(
    r"""# How do agents make decisions?
Decision making processes are very complex. Here, the process of decision 
making refers to the purchase decision of heating technologies and is
drastically simplified, to allow for modelling. In this model, agents have
two ways to adopt technologies based on the theory of planned behaviour:
1) Adopt new heating system if the perceived gain surpasses a certain threshold.
2) Adopt new heating system if the current system breaks."""
)

with st.expander("deprecated"):
    st.markdown(
        r"""## Interaction with peers
Each agent $i$ has an attitude towards the diffent technologies $T \in\ $"""
        f"""{list(model.heating_techs_df.index)}"""
        r""" 
    $a_{i,T} \in (-1,1)$. These can be solicited via a survey, but are random for now. 
    This is an example of an agents potential attitudes toward different technologies.
    """
    )
    tech_attitudes = dict(
        Technologies=list(sample_agent.tech_attitudes.keys()),
        Attitudes=list(sample_agent.tech_attitudes.values()),
    )
    fig = px.bar(tech_attitudes, x="Technologies", y="Attitudes", height=300, width=400)
    st.plotly_chart(fig)

    st.markdown(
        r"""Over the course of the simulation, agents meet each other based on proximity (current
    implementation is subject to change in the future). When two agents interact, they 
    exchange values based on one of the following functions. Each of these functions takes 
    an `inertia` parameter $\in (0,1)$, which defaults to `0.5` and alters the attitude of
    one agent.
    """
    )

    inertia = st.slider("select agents' attitude inertia", 0.0, 1.0, 0.5)
    show_diff_funcs(inertia)

    st.markdown(
        """The y-axis shows the attitude of agent $i$ before the interaction, the x-axis shows 
        the attitude of the other agent and the color shows the attitude after the interaction. 
        While the `sigmoid_diff` and the `simple_diff` allow an agent to change it's attitude by
        $1$, the `scaled_diff` only allows for a total attitude change of $0.5$ in the default
        settings and might thereby better represent, that people with a strong 
        opinion might be more resistant to change.
        """
    )


def mermaid(code: str) -> None:
    components.html(
        f"""
        <pre class="mermaid">
            {code}
        </pre>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """
    )


st.markdown("## The purchase decision")


st.markdown(
    r"""
Each agent $i$ is assumed to consider purchasing a new heating appliance, based on the TPB
after 80% of the lifetime of their existing appliance has passed. The components
of the TPB are utilised as follows:
A utility $U_T$ for each Technology $T$ is computed, using the following criteria
$k \in \{C_{i,T,a}, E_{i,T,a}, a_{i,T}\}$, where
1) $C_{i,T,a}$ is the total annual cost of the technology
2) $E_{i,T,a}$ are the total annual emissions and
3) $att_{i,T}$ is the agents attitude towards that technology.
    
First, these three sets are normalised to the domain of $(0,1)$. Second,
weights (preferences) for each of these indicators $w_k \in (0,1)$ are 
multiplied with each of the indicators
$$
U_T = C_{i,T,a} \cdot w_C + E_{i,T,a}\cdot w_E  + att_{i,T} \cdot w_{att}
$$

The utility of a certain technology is then compared to the utility $U_{T_c}$ of the
current technology $T_c$ defining the gain of adopting the new technolgy $G_T$ as

$$
G_{T,i} = U_{T,i} - U_{T_c,i}
$$

The last term defining the "attractivity" of a technology in the agents eye, is
the `peer_effect`, i.e. the percentage of neighbours that have adopted the 
technology in question. This is calculated as
$$
p_{T,i} = \frac{\sum_n (n_T)}{\sum_n 1}
$$
where $n$ are the neighbours of agent $i$ and $n_T \in \{0,1\}$ is a neighbour that either has
or has not adopted technology $T$.

Ultimately a threshold $TR$ must be surpassed by the weighted sum of the gain
and the peer effect
$$
    TR > G_{T,i} \cdot w_1 + p_{T,i} \cdot w_2
$$

"""
)
mermaid(
    """
    graph LR
        start((start)) --> a
        a("TR > G_{T,i} + p_{T,i}") --yes--> p(purchase<br>tech) --> z
        a --noo--> C("`other tech available?`")
        C --yes--> a
        C --noo--> p2(purchase tech with max U) --> z((end))
        C --noo--> z((end))
    """
)


st.markdown(
    r"""
When the lifetime of an appliance has passed, the agent must buy a new
appliance. In that case ...
"""
)
heat_tech_df = sample_agent.heat_techs_df
heat_tech_df["attitude"] = sample_agent.tech_attitudes
heat_tech_df["attitude"] = normalize(heat_tech_df["attitude"] + 1)
# calculate scores
weights = {
    "emissions[kg_CO2/a]_norm": 0.3,
    "total_cost[EUR/a]_norm": 0.4,
    "attitude": 0.3,
}

# st.write(heat_tech_df.reset_index().melt(id_vars="index"))

absolute_fig = px.bar(
    heat_tech_df.reset_index()
    .melt(id_vars="index")
    .query("variable in ['emissions[kg_CO2/kWh_th]','annual_cost']"),
    x="index",
    y="value",
    facet_row="variable",
)
for annot in absolute_fig.layout.annotations:
    annot.text = annot.text.split("=")[1]
    annot.textangle = 30
absolute_fig.update_layout(margin_r=100)
absolute_fig.update_yaxes(matches=None)
st.plotly_chart(absolute_fig, use_container_width=True)

st.markdown("using the weights of:")
st.write(weights)
st.markdown("it results in the following score")

tech_df_w_scores = sample_agent.calc_scores()

score_relevant = [
    "emissions_norm",
    "cost_norm",
    "attitude",
    "total_score",
]
heat_tech_df_score = (
    tech_df_w_scores.reset_index()
    .melt(id_vars="index")
    .query("variable in @score_relevant")
)

mcda_fig = px.bar(
    heat_tech_df_score,
    x=heat_tech_df_score["index"],
    y="value",
    facet_row="variable",
    facet_row_spacing=0.04,
)
for annot in mcda_fig.layout.annotations:
    annot.text = annot.text.split("=")[1]
    annot.textangle = 30
    if "_norm" in annot.text:
        annot.text = annot.text.replace("_norm", "")
mcda_fig.update_layout(height=500, margin_r=100)
st.plotly_chart(mcda_fig, use_container_width=True)
