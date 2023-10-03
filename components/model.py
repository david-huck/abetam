import numpy as np
import pandas as pd

import mesa
from components.agent import HouseholdAgent
from components.technologies import HeatingTechnology
from data.canada import (
    get_gamma_distributed_incomes,
    energy_demand_from_income_and_province,
)


class TechnologyAdoptionModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, width, height, province, heating_techs_df, random_seed=42):
        self.random.seed(random_seed)
        np.random.seed(random_seed)
        self.num_agents = N
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)

        # generate agent parameters: income, energy demand, technology distribution
        income_distribution = get_gamma_distributed_incomes(N, seed=random_seed)

        # space heating and hot water make up ~80 % of total final energy demand
        # https://oee.nrcan.gc.ca/corporate/statistics/neud/dpa/showTable.cfm?type=CP&sector=res&juris=ca&year=2020&rn=2&page=0
        heat_demand = (
            energy_demand_from_income_and_province(income_distribution, province) * 0.79
        )

        self.heating_techs_df = heating_techs_df
        # "upper_idx" up to which agents receive certain heating tech
        heating_techs_df["upper_idx"] = (heating_techs_df["cum_share"] * N).astype(int)

        # Create agents
        for i in range(self.num_agents):
            # get the first row, where the i < upper_idx
            try:
                heat_tech_row = heating_techs_df.query(f"{i} <= upper_idx").iloc[0, :]
            except IndexError as e:
                print(i, len(heating_techs_df), heating_techs_df["upper_idx"])
                raise e

            heat_tech_i = HeatingTechnology.from_series(heat_tech_row)
            a = HouseholdAgent(
                i, self, income_distribution[i], heat_tech_i, heat_demand[i]
            )
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # setup a datacollector for tracking changes over time
        self.datacollector = mesa.DataCollector(
            model_reporters={"Technology shares": self.heating_technology_shares},
            agent_reporters={"Attitudes": "tech_attitudes", "Wealth": "wealth"},
        )

    def heating_technology_shares(self):
        # print(self.heating_techs_df.index)
        shares = dict(
            zip(self.heating_techs_df.index, [0] * len(self.heating_techs_df))
        )
        for a in self.schedule.agents:
            shares[a.heating_tech.name] += 1

        return shares.copy()

    def step(self):
        """Advance the model by one step."""
        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.datacollector.collect(self)
        self.schedule.step()
