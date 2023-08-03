import mesa
import streamlit as st
import plotly.express as px


class MoneyAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model):
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        # Create the agent's attribute and set the initial values.
        self.wealth = 1

    def step(self):
        # Verify agent has some wealth
        if self.wealth > 0:
            other_agent = self.random.choice(self.model.schedule.agents)
            if other_agent is not None:
                other_agent.wealth += 1
                self.wealth -= 1


class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N):
        self.num_agents = N
        # Create scheduler and assign it to the model
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            # Add the agent to the scheduler
            self.schedule.add(a)

    def step(self):
        """Advance the model by one step."""

        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.schedule.step()


num_agents = st.slider("Number of Agents:",10,1000,)
model = MoneyModel(num_agents)

num_iters = st.slider("Number of iterations:", 10, 100,)
for i in range(num_iters):
    model.step()

agent_wealth = [a.wealth for a in model.schedule.agents]
fig = px.histogram(agent_wealth).update_layout(
    # height=600,
    width=500,
    xaxis_title="Wealth (Coins)",
    yaxis_title="Number of Agents (-)",
    showlegend=False
)
st.markdown("# Wealth distribution")
st.plotly_chart(fig)
