import numpy as np
import pandas as pd

import mesa
from components.agent import HouseholdAgent
from components.technologies import HeatingTechnology
from data.canada import (
    get_gamma_distributed_incomes,
    energy_demand_from_income_and_province,
    get_fuel_price,
)
from data.canada.timeseries import determine_heat_demand_ts


class TechnologyAdoptionModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(
        self,
        N,
        width,
        height,
        province,
        heating_techs_df,
        start_year=2013,
        years_per_step=1 / 4,
        random_seed=42,
    ):
        self.random.seed(random_seed)
        np.random.seed(random_seed)
        self.num_agents = N
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.start_year = start_year
        self.current_year = start_year
        self.years_per_step = years_per_step
        self.province = province
        self.running = True
        # generate agent parameters: income, energy demand, technology distribution
        income_distribution = get_gamma_distributed_incomes(N, seed=random_seed)

        # space heating and hot water make up ~80 % of total final energy demand
        # https://oee.nrcan.gc.ca/corporate/statistics/neud/dpa/showTable.cfm?type=CP&sector=res&juris=ca&year=2020&rn=2&page=0
        heat_demand = (
            energy_demand_from_income_and_province(income_distribution, province) * 0.79
        )

        self.heating_techs_df = heating_techs_df
        self.heating_techs_df["province"] = province
        # retrieve historical prices for selected province
        # self.update_cost_params()
        prices = []
        for fuel in self.heating_techs_df["fuel"]:
            fuel_price = get_fuel_price(fuel, province, start_year)
            prices.append(fuel_price)
        self.heating_techs_df["specific_fuel_cost"] = prices
        # "upper_idx" up to which agents receive certain heating tech
        self.heating_techs_df["upper_idx"] = (self.heating_techs_df["cum_share"] * N).astype(int)

        # Create agents
        for i in range(self.num_agents):
            # get the first row, where the i < upper_idx
            try:
                heat_tech_row = self.heating_techs_df.query(f"{i} <= upper_idx").iloc[0, :]
            except IndexError as e:
                print(i, len(self.heating_techs_df), self.heating_techs_df["upper_idx"])
                raise e

            heat_tech_i = HeatingTechnology.from_series(heat_tech_row)
            a = HouseholdAgent(
                i,
                self,
                income_distribution[i],
                heat_tech_i,
                heat_demand[i],
                years_per_step=self.years_per_step,
            )
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # setup a datacollector for tracking changes over time
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Technology shares": self.heating_technology_shares,
                "Energy demand time series": self.energy_demand_ts,
            },
            agent_reporters={"Attitudes": "tech_attitudes", "Wealth": "wealth"},
        )

    
    def update_cost_params(self, year):
        """updates the parameters of the heating technology dataframe

        Args:
            year (float): the year to which the cost parameters should adhere
        """
        prices = []
        for fuel in self.heating_techs_df["fuel"]:
            fuel_price = get_fuel_price(fuel, self.province, year)
            prices.append(fuel_price)
        self.heating_techs_df["specific_fuel_cost"] = prices
        

    def heating_technology_shares(self):
        shares = dict(
            zip(self.heating_techs_df.index, [0] * len(self.heating_techs_df))
        )
        for a in self.schedule.agents:
            shares[a.heating_tech.name] += 1

        for tech in self.heating_techs_df.index:
            shares[tech] /= self.num_agents

        return shares.copy()

    def step(self):
        """Advance the model by one step."""
        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        # self.update_cost_params(self.current_year)
        self.datacollector.collect(self)
        self.schedule.step()
        self.current_year += self.years_per_step

    def energy_demand_ts(self):
        energy_carriers = self.heating_techs_df.fuel.unique()

        energy_carrier_demand = dict(zip(energy_carriers, [0] * len(energy_carriers)))

        # retrieve the energy demand from each agent
        for a in self.schedule.agents:
            # get agents energy demand
            final_demand = a.heat_demand
            # get agents' heating appliance efficiency and fuel type
            efficiency = a.heating_tech.efficiency
            fuel = a.heating_tech.fuel

            energy_carrier_demand[fuel] = (
                energy_carrier_demand[fuel] + final_demand / efficiency
            )

        # create a timeseries from it
        for carrier, demand in energy_carrier_demand.items():
            energy_carrier_demand[carrier] = determine_heat_demand_ts(demand)

        return energy_carrier_demand

