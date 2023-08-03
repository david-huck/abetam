import mesa
import streamlit as st
import pandas as pd
import plotly.express as px


class MoneyAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model):
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        # Create the agent's attribute and set the initial values.
        self.wealth = 1

    def step(self):

        self.move()
        if self.wealth > 0:
            self.give_money()

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1


class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

    def step(self):
        """Advance the model by one step."""

        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.schedule.step()


num_agents = st.slider("Number of Agents:",10,1000,)
model = MoneyModel(num_agents, 10, 10)


agent_counts_before_exectution = pd.DataFrame()
for cell_content, (x, y) in model.grid.coord_iter():
    agent_count = len(cell_content)
    agent_counts_before_exectution.at[x,y] = agent_count



num_iters = st.slider("Number of iterations:", 10, 100,)
for i in range(num_iters):
    model.step()


st.markdown("# Wealth distribution")
agent_wealth = [a.wealth for a in model.schedule.agents]
fig = px.histogram(agent_wealth).update_layout(
    # height=600,
    width=500,
    xaxis_title="Wealth (Coins)",
    yaxis_title="Number of Agents (-)",
    showlegend=False
)
st.plotly_chart(fig)


st.markdown("# No. Agents in cells of the grid before and after execution")
agent_counts = pd.DataFrame()
for cell_content, (x, y) in model.grid.coord_iter():
    agent_count = len(cell_content)
    agent_counts.at[x,y] = agent_count


# fig = px.imshow(agent_counts, width=600)

# st.plotly_chart(fig, )

import numpy as np
before_after = np.array([
    agent_counts_before_exectution.values,
    agent_counts.values ])


fig = px.imshow(before_after, facet_col=0, width=600,)

fig.update_layout(
    xaxis1_title="before",
    xaxis2_title="after",
    )
st.plotly_chart(fig, )


