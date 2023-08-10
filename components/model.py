from dataclasses import dataclass

import numpy as np
import pandas as pd

import mesa
from components.agent import MoneyAgent

@dataclass
class HeatingTechnology:
    name: str
    specific_cost: float
    specific_fuel_cost: float
    specific_fuel_emission: float
    efficiency: float
    lifetime: int



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


        heating_techs_df["upper_idx"] = (heating_techs_df["cum_share"] * N).astype(int)

        # Create agents
        for i in range(self.num_agents):
            heat_tech_row = heating_techs_df.query(f"{i} < upper_idx").iloc[0,:-3]
            heat_tech_i = HeatingTechnology(heat_tech_row.name, **heat_tech_row.to_dict())
            a = MoneyAgent(i, self, wealth_distribution[i], heat_tech_i)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

    def step(self):
        """Advance the model by one step."""

        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.schedule.step()
