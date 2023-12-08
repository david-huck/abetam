print(__file__)

import streamlit as st
import streamlit_mermaid as stmd


import numpy as np
import pandas as pd
import plotly.express as px

import config
from components.model import TechnologyAdoptionModel
from components.technologies import merge_heating_techs_with_share

from data.canada import all_provinces, create_geo_fig

if "technology_colors" not in st.session_state:
    st.session_state["technology_colors"] = config.TECHNOLOGY_COLORS
    st.session_state["fuel_colors"] = config.FUEL_COLORS

province = st.selectbox(
    "Select a province (multiple provinces might be implemented in the future):",
    all_provinces,
    index=0,
)

geo_fig = create_geo_fig(province)
st.plotly_chart(geo_fig)
start_year = st.select_slider(
    "Select starting year:",
    range(2000, 2021, 5),
)


heat_techs_df = merge_heating_techs_with_share(start_year, province)

if config.DEBUG:
    pass


num_agents = st.slider("Number of Agents:", 10, 1000, 30)

# amount of steps for moving agents in to similar groups
segregation_steps = st.slider("Number of segregation steps:", 0, 50, 40)


# @st.cache_data
# doesn't work with agent reporter because of tech attitude dict
def run_model(num_agents, num_iters, province, heat_techs_df=heat_techs_df):
    model = TechnologyAdoptionModel(
        num_agents, province, n_segregation_steps=segregation_steps
    )
    if segregation_steps:
        with st.expander("Segregation"):
            # raise ValueError("Segregation now takes place in the models __init__ function")
            tab_schem, tab_data,  = st.columns([2,3])

            with tab_schem:
                st.header("schem")
                path = "schelling.svg.svg"
                st.image(path)

            with tab_data:
                st.header("data")
                income_segregation_dfs = model.segregation_df
                st.markdown(
                    r"""
                            Segregation is used to represent typical grouping of households.
                            If the ratio of `agent.disposable_income` between to agents is `>0.7`, they are considered _similar_. 
                            If an the neighborhood of an agent consists of <50\% similar neighbors, the agent moves to a random location. 
                            Otherwise he stays. 
                            """
                )
                imgs = np.array([df.values for df in income_segregation_dfs])
                # this is probably better for publication
                # visualized_segregation_steps = np.linspace(
                #     0, segregation_steps-1, 3, dtype=int
                # )
                # fig = px.imshow(imgs[visualized_segregation_steps], facet_col=0, facet_col_spacing=0.01)
                # st.plotly_chart(fig)
                fig = px.imshow(imgs, animation_frame=0)
                fig.update_layout(
                    coloraxis_colorbar=dict(
                        title="Average<br>income",
                        thicknessmode="pixels",
                        thickness=8,
                        lenmode="pixels",
                        len=200,
                    ),
                    margin={"t": 0, "r": 0, "l": 0, "b": 0},
                    width=500,
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

    for i in range(num_iters):
        model.step()
    return model


num_iters = st.slider("Number of iterations:", 10, 100, 30)
model = run_model(num_agents, num_iters, province)

agent_vars = model.datacollector.get_agent_vars_dataframe()


def show_wealth_distribution():
    st.markdown("# Wealth distribution at end of simulation")
    agent_wealth = [a.wealth for a in model.schedule.agents]
    fig = px.histogram(agent_wealth).update_layout(
        # height=600,
        width=500,
        xaxis_title="Wealth (Coins)",
        yaxis_title="Number of Agents (-)",
        showlegend=False,
    )
    st.plotly_chart(fig)


# show_wealth_distribution()


# show attitudes over time    

def show_agent_attitudes(individual=True):
    if individual:
        agent_attitudes = agent_vars[["Attitudes"]]
        agent_attitudes.loc[:, "Attitudes"] = agent_attitudes["Attitudes"].apply(
            lambda x: x.items()
        )
        agent_attitudes = agent_attitudes.explode("Attitudes")
        agent_attitudes[["tech", "Attitudes"]] = agent_attitudes["Attitudes"].to_list()

        agent_attitudes = agent_attitudes.reset_index()

        selected_agents = st.multiselect(
            "select agents", agent_attitudes.AgentID.unique(), [1, 2, 3]
        )
        agent_attitudes = agent_attitudes.query(f"AgentID in {selected_agents}")
        # agent_attitudes = agent_attitudes.reset_index().groupby(["Step","tech"]).mean().reset_index()
        att_fig = px.scatter(
            agent_attitudes, x="Step", y="Attitudes", color="tech", facet_col="AgentID"
        )
        st.plotly_chart(att_fig)


show_agent_attitudes()

model_vars = model.datacollector.get_model_vars_dataframe()
adoption_col = model_vars["Technology shares"].to_list()
adoption_df = pd.DataFrame.from_records(adoption_col)
adoption_df.index = model.get_steps_as_years()

appliance_sum = adoption_df.sum(axis=1)
adoption_df = adoption_df.apply(lambda x: x / appliance_sum * 100)


fig = px.line(adoption_df, color_discrete_map=config.TECHNOLOGY_COLORS)
fig.update_layout(yaxis_title="Share of technologies (%)", xaxis_title="Year")
st.plotly_chart(fig)


energy_demand_ts = model_vars["Energy demand time series"].to_list()
energy_demand_df = pd.DataFrame.from_records(energy_demand_ts)


def explode_array_column(row):
    return pd.Series(row["value"])


energy_demand_df_long = energy_demand_df.melt(ignore_index=False)

expanded_cols = energy_demand_df_long.apply(explode_array_column, axis=1)
expanded_cols.columns = [i for i in range(expanded_cols.shape[1])]

energy_demand_df_long = pd.concat([energy_demand_df_long, expanded_cols], axis=1)
energy_demand_df_long.drop("value", axis=1, inplace=True)
energy_demand_df_long.rename({"variable": "carrier"}, axis=1, inplace=True)
energy_demand_df_long = energy_demand_df_long.melt(
    id_vars=["carrier"], ignore_index=False, var_name="t"
)

energy_demand_df_long.reset_index(inplace=True, names=["step"])

# plot 4 exemplary timeseries along the model horizon
steps_to_plot = np.linspace(0, num_iters, 5, dtype=int)


fig = px.line(
    energy_demand_df_long.query("step in @steps_to_plot"),
    x="t",
    y="value",
    color="carrier",
    facet_row="step",
    color_discrete_map=config.FUEL_COLORS,
)
fig.update_layout(
    yaxis1_title="",
    yaxis2_title="Energy demand (kWh/h)",
    yaxis3_title="",
    yaxis4_title="",
)
st.plotly_chart(fig)

energy_demand_df_long["step"] = model.steps_to_years(energy_demand_df_long["step"])

agg_carrier_demand = energy_demand_df_long.groupby(["step", "carrier"]).sum()
fig = px.bar(
    agg_carrier_demand.reset_index(),
    x="step",
    y="value",
    color="carrier",
    color_discrete_map=config.FUEL_COLORS,
)
fig.update_layout(xaxis_title="Year", yaxis_title="Energy demand (kWh/a)")
st.plotly_chart(fig)
