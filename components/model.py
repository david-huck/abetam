import numpy as np
import pandas as pd

import mesa
from components.agent import MoneyAgent
from components.technologies import HeatingTechnology





class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(
        self,
        N,
        width,
        height,
        disp_income_mean,
        disp_income_stdev,
        heating_techs_df
    ):
        self.num_agents = N
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)

        # generate agent parameters: wealth, technology distribution, education level
        wealth_distribution = np.random.normal(disp_income_mean, disp_income_stdev, N)

        self.heating_techs_df = heating_techs_df
        # "upper_idx" up to which agents receive certain heating tech
        heating_techs_df["upper_idx"] = (heating_techs_df["cum_share"] * N).astype(int)

        # Create agents
        for i in range(self.num_agents):
            # get the first row, where the i < upper_idx
            heat_tech_row = heating_techs_df.query(f"{i} < upper_idx").iloc[0,:]
            
            heat_tech_i = HeatingTechnology.from_series(heat_tech_row)
            a = MoneyAgent(i, self, wealth_distribution[i], heat_tech_i)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # setup a datacollector for tracking changes over time
        self.datacollector = mesa.DataCollector(
            model_reporters={"Technology shares":self.heating_technology_shares},
            agent_reporters={"Attitudes": "tech_attitudes", 
                             "Wealth":"wealth"},
        )

    def heating_technology_shares(self):
        # print(self.heating_techs_df.index)
        shares = dict(zip(self.heating_techs_df.index,[0]*len(self.heating_techs_df)))
        for a in self.schedule.agents:
            shares[a.heating_tech.name] += 1

        return shares.copy()

    def step(self):
        """Advance the model by one step."""
        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.datacollector.collect(self)
        self.schedule.step()
