import mesa
import streamlit as st

import numpy as np
import pandas as pd
import plotly.express as px

from components.agent import MoneyAgent
from components.model import MoneyModel


heating_techs_df = pd.DataFrame(index=["gas_boiler","oil_boiler","district_heating","other","heat_pump"])
heating_techs_df.loc[:,"specific_cost"] = [800, 600, 200, 8000, 1200]
heating_techs_df.loc[:,"specific_fuel_cost"] = [0.06, 0.10, 0.15, 0.3, 0.3]
heating_techs_df.loc[:,"specific_fuel_emission"] = [0.2, 0.5, 0.15, 0.4, 0.4]
heating_techs_df.loc[:,"efficiency"] = [0.9, 0.9, 1, 1, 3]
heating_techs_df.loc[:,"lifetime"] = [20, 30, 20, 20, 20]
heating_techs_df.loc[:,"country_share_de"] = [0.495, 0.25, 0.141, 0.088, 0.026]
heating_techs_df["cum_share"] = heating_techs_df["country_share_de"].cumsum()


num_agents = st.slider(
    "Number of Agents:",
    10,
    1000,
)
model = MoneyModel(num_agents, 10, 10, 23500, 2000, heating_techs_df)


agent_counts_before_exectution = pd.DataFrame()
for cell_content, (x, y) in model.grid.coord_iter():
    agent_count = len(cell_content)
    agent_counts_before_exectution.at[x, y] = agent_count


num_iters = st.slider(
    "Number of iterations:",
    10,
    100,
)
for i in range(num_iters):
    model.step()


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
