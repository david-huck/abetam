import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from decision_making.attitudes import show_diff_funcs
from decision_making.mcda import normalize, calc_score
from components.technologies import merge_heating_techs_with_share
from components.model import TechnologyAdoptionModel

st.set_page_config(page_title="Decision Making")

heat_tech_df = merge_heating_techs_with_share()

model = TechnologyAdoptionModel(10, 3, 3, "Canada", heat_tech_df)
sample_agent = model.schedule.agents[0]


st.markdown(
    """# How do agents make decisions?
Decision making processes are very complex. Here, the process is drastically 
simplified, to allow for modelling. In this model, it consists of three steps:
1) Interact with peers to exchange information/values of heating systems.
2) Adopt new heating system if idealogically fitting (TPB).
3) Adopt new heating system if current system surpassed its lifetime (MCDA).

## Interaction with peers
Each agent $i$ has an attitude towards the diffent technologies $T \in\ $"""
    f"""{list(heat_tech_df.index)}"""
    """ 
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
    """Over the course of the simulation, agents meet each other based on proximity (current
implementation is subject to change in the future). When two agents interact, they 
exchange values based on one of the following functions. Each of these functions takes 
an `inertia` parameter $\in (0,1)$, which defaults to `0.5` and alters the attitude of
one agent.
"""
)

inertia = st.slider("select agents' attitude inertia", 0.0, 1.0, 0.5)
show_diff_funcs(inertia)

with st.expander("How does this work?"):
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


with st.expander("... based on the `TBP`"):
    st.markdown(
        r"""
    Agents are assumed to consider purchasing a new heating appliance, based on the TPB
    after 80% of the lifetime of their existing appliance has passed.
    """
    )
    mermaid(
        """
        graph LR
            start((start)) --> a
            a(tech attitude > 0.7<br>&<br> income > annual cost<br>?) --yes--> B(random < PBC?) 
            B --noo--> C("`other tech available?`")
            a --noo--> C
            B --yes--> p(purchase<br>tech) --> z
            C --yes--> a
            C --noo--> z((end))
        """
    )


with st.expander("... based on `MCDA`"):
    st.markdown(
        r"""
    When the lifetime of an appliance has passed, the agent must buy a new appliance.
    In that case it is done through MCDA, taking into account three indicators  
    $k \in \{C_{i,T,a}, E_{i,T,a}, a_{i,T}\}$, where
    1) $C_{i,T,a}$ is the total annual cost of the technology 
    2) $E_{i,T,a}$ are the total annual emissions and
    3) $a_{i,T}$ is the agents attitude towards that technology.
        
    First, these three sets are normalised to the domain 
    of $(0,1)$. Second weights for each of these indicators $w_k \in (0,1)$ are 
    multiplied with each of the indicators, and the agent adopts the technology 
    with the best resulting score.
    $$
    min(\sum_k k \cdot w_k)
    $$
    """
    )
    heat_tech_df["attitude"] = sample_agent.tech_attitudes
    heat_tech_df["attitude"] = normalize(heat_tech_df["attitude"] + 1)
    # calculate scores
    weights = {
        "emissions[kg_CO2/a]_norm": 0.3,
        "total_cost[EUR/a]_norm": 0.4,
        "attitude": 0.3,
    }

    st.write(heat_tech_df.reset_index().melt(id_vars="index"))

    absolute_fig = px.bar(
        heat_tech_df.reset_index()
        .melt(id_vars="index")
        .query("variable in ['emissions[kg_CO2/a]','total_cost[EUR/a]']"),
        x="index",
        y="value",
        facet_row="variable",
    )
    for annot in absolute_fig.layout.annotations:
        annot.text = annot.text.split("=")[1]
        annot.textangle = 30
    absolute_fig.update_layout(margin_r=100)
    st.plotly_chart(absolute_fig, use_container_width=True)

    st.markdown("using the weights of:")
    st.write(weights)
    st.markdown("it results in the following score")
    heat_tech_df["total_score"] = heat_tech_df[
        ["emissions[kg_CO2/a]_norm", "total_cost[EUR/a]_norm", "attitude"]
    ].apply(
        calc_score,
        axis=1,
        weights=weights,
    )

    score_relevant = [
        "emissions[kg_CO2/a]_norm",
        "total_cost[EUR/a]_norm",
        "attitude",
        "total_score",
    ]
    heat_tech_df_score = (
        heat_tech_df.reset_index()
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
