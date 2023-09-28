import mesa
import streamlit as st

import numpy as np
import pandas as pd
import plotly.express as px

from components.model import TechnologyAdoptionModel
from decision_making.mcda import normalize

from data.canada import simplified_heating_systems, all_provinces

debug=False

province = st.selectbox("Select a province:", all_provinces)
start_year = st.select_slider("Select starting year:",simplified_heating_systems.reset_index()["REF_DATE"].unique() )

if debug:
    st.write(simplified_heating_systems)
    st.write(simplified_heating_systems.loc[(start_year, province),:])



technologies = [
"Gas furnance",
"Oil furnace",
"Wood or wood pellets furnace",
"Electric furnance",
"Heat pump",
]
heating_techs_df = pd.DataFrame(index=technologies)
heating_techs_df.loc[:,"specific_cost"] = [800, 600, 900, 500, 1200]
heating_techs_df.loc[:,"specific_fuel_cost"] = [0.06, 0.10, 0.15, 0.1, 0.1]
heating_techs_df.loc[:,"specific_fuel_emission"] = [0.2, 0.5, 0.15, 0.4, 0.4]
heating_techs_df.loc[:,"efficiency"] = [0.9, 0.9, 0.9, 1, 3]
heating_techs_df.loc[:,"lifetime"] = [20, 30, 20, 20, 15]
heating_techs_df.loc[:,"share"] = (
    simplified_heating_systems.loc[(start_year, province),:]
    /sum(simplified_heating_systems.loc[(start_year, province),:]))
heating_techs_df["cum_share"] = heating_techs_df["share"].cumsum()

if debug:
        st.write(heating_techs_df)

# assuming a discount rate
discount_rate = 0.07

heating_techs_df["annuity_factor"] =  discount_rate/(1-(1+discount_rate)**-heating_techs_df["lifetime"]) 
heating_techs_df["annuity"] = heating_techs_df["annuity_factor"] * heating_techs_df["specific_cost"]

demand = 20000 # kWh    
# assuming peak demand to be a certain fraction # TODO needs improvement
peak_demand = demand/1.5e3

# total costs:
heating_techs_df["invest_cost[EUR/a]"] = peak_demand * heating_techs_df["annuity"]
heating_techs_df["fom_cost[EUR/a]"] = heating_techs_df["invest_cost[EUR/a]"] * 0.02
heating_techs_df["vom_cost[EUR/a]"] = demand / heating_techs_df["efficiency"] * heating_techs_df["specific_fuel_cost"]

heating_techs_df["emissions[kg_CO2/a]"] = demand / heating_techs_df["efficiency"] * heating_techs_df["specific_fuel_emission"]

heating_techs_df["total_cost[EUR/a]"] = heating_techs_df[["invest_cost[EUR/a]","fom_cost[EUR/a]","vom_cost[EUR/a]"]].sum(axis=1)

heating_techs_df.loc[:,["emissions[kg_CO2/a]_norm","total_cost[EUR/a]_norm"]] = heating_techs_df[["emissions[kg_CO2/a]","total_cost[EUR/a]"]].apply(normalize).values



num_agents = st.slider(
    "Number of Agents:",
    10,
    1000,
    30
)
model = TechnologyAdoptionModel(num_agents, 10, 10, province, heating_techs_df)


# agent_counts_before_exectution = pd.DataFrame()
# for cell_content, (x, y) in model.grid.coord_iter():
#     agent_count = len(cell_content)
#     agent_counts_before_exectution.at[x, y] = agent_count


num_iters = st.slider(
    "Number of iterations:",
    10,
    100,
    30
)
for i in range(num_iters):
    model.step()

agent_vars = model.datacollector.get_agent_vars_dataframe()

def show_wealth_distribution():
    
    st.markdown("# Wealth distribution")
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


    agent_counts_before_after = np.array([agent_counts_before_exectution.values, agent_counts.values])


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

    wealth_fig = px.line(agent_wealth.reset_index(),x="Step",y="Wealth",color="AgentID")
    st.plotly_chart(wealth_fig)



def show_agent_attitudes():
    

    agent_attitudes = agent_vars[["Attitudes"]]
    agent_attitudes.loc[:,"Attitudes"] = agent_attitudes["Attitudes"].apply(lambda x:x.items())
    agent_attitudes = agent_attitudes.explode("Attitudes")
    agent_attitudes[["tech","Attitudes"]]=agent_attitudes["Attitudes"].to_list()

    agent_attitudes = agent_attitudes.reset_index()


    selected_agents = st.multiselect("select agents",agent_attitudes.AgentID.unique(), [1,2,3])
    agent_attitudes = agent_attitudes.query("AgentID in @selected_agents")
    # agent_attitudes = agent_attitudes.reset_index().groupby(["Step","tech"]).mean().reset_index()
    att_fig = px.scatter(agent_attitudes, x="Step",y="Attitudes",color="tech", facet_col="AgentID")
    st.plotly_chart(att_fig)

show_agent_attitudes()

model_vars = model.datacollector.get_model_vars_dataframe()
model_vars = model_vars["Technology shares"].to_list()
adoption_df = pd.DataFrame.from_records(model_vars)

fig = px.line(adoption_df)
st.plotly_chart(fig)
