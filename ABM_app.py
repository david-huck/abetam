import streamlit as st

import numpy as np
import pandas as pd
import plotly.express as px

from components.model import TechnologyAdoptionModel
from components.technologies import merge_heating_techs_with_share

from data.canada import simplified_heating_stock, all_provinces, create_geo_fig

debug = False



province = st.selectbox(
    "Select a province (multiple provinces might be implemented in the future):",
    all_provinces,
    index=0
)

geo_fig = create_geo_fig(province)
st.plotly_chart(geo_fig)
start_year = st.select_slider(
    "Select starting year:",
    simplified_heating_stock.reset_index()["REF_DATE"].unique(),
)


heat_techs_df = merge_heating_techs_with_share(start_year, province)

if debug:
    st.write(simplified_heating_stock)
    st.write(simplified_heating_stock.loc[(start_year, province), :])
    st.write(heat_techs_df)


num_agents = st.slider("Number of Agents:", 10, 1000, 30)


# @st.cache_data
# doesn't work with agent reporter because of tech attitude dict
def run_model(num_agents, num_iters, province, heat_techs_df=heat_techs_df):
    model = TechnologyAdoptionModel(num_agents, 10, 10, province, heat_techs_df)
    for i in range(num_iters):
        model.step()
    return model


# agent_counts_before_exectution = pd.DataFrame()
# for cell_content, (x, y) in model.grid.coord_iter():
#     agent_count = len(cell_content)
#     agent_counts_before_exectution.at[x, y] = agent_count


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


show_wealth_distribution()


def show_agent_placement():
    st.markdown("# No. Agents in cells of the grid before and after execution")
    agent_counts = pd.DataFrame()
    for cell_content, (x, y) in model.grid.coord_iter():
        agent_count = len(cell_content)
        agent_counts.at[x, y] = agent_count

    agent_counts_before_after = np.array(
        [agent_counts_before_exectution.values, agent_counts.values]
    )

    fig = px.imshow(
        agent_counts_before_after,
        facet_col=0,
        width=600,
    )

    fig.update_layout(
        xaxis1_title="before",
        xaxis2_title="after",
    )
    st.plotly_chart(
        fig,
    )


# show_agent_placement()


def show_wealth_over_time():
    agent_wealth = agent_vars[["Wealth"]]

    wealth_fig = px.line(
        agent_wealth.reset_index(), x="Step", y="Wealth", color="AgentID"
    )
    st.plotly_chart(wealth_fig)


def show_agent_attitudes():
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
    agent_attitudes = agent_attitudes.query("AgentID in @selected_agents")
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

fig = px.line(adoption_df)
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
fig = px.bar(agg_carrier_demand.reset_index(), x="step", y="value", color="carrier")
fig.update_layout(xaxis_title="Year", yaxis_title="Energy demand (kWh/a)")
st.plotly_chart(fig)
