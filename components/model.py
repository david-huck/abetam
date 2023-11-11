import numpy as np
import pandas as pd
import plotly.express as px
from typing import Iterable
from pathlib import Path
from datetime import datetime

import mesa
from components.agent import HouseholdAgent
from components.technologies import HeatingTechnology, merge_heating_techs_with_share
from components.probability import beta_with_mode_at

from data.canada import (
    get_beta_distributed_incomes,
    energy_demand_from_income_and_province,
    get_fuel_price,
    tech_capex_df,
)
from decision_making.mcda import normalize
from data.canada.timeseries import determine_heat_demand_ts

def get_income_and_attitude_weights(n, price_weight_mode=None):
    incomes = get_beta_distributed_incomes(n)

    if price_weight_mode is None:
        # Assumption 1: richer people are less price sensitive
        # Shape of this distribution is similar to the income distribution mirrored at 0.5
        price_weights = 1 - normalize(np.array(incomes))
    elif isinstance(price_weight_mode, float):
        price_weights = beta_with_mode_at(price_weight_mode, n, interval=(0, 1))
    else:
        raise ValueError(
            f"Parameter `price_weight_mode` must be float, got {price_weight_mode=}."
        )

    # draw random values for emission weights
    emission_weights = np.random.random(len(price_weights))

    # bring the emission weights into the interval (0, price_weight)
    int_len = 1 - price_weights
    emission_weights = emission_weights * int_len

    attitude_weights = 1 - price_weights - emission_weights

    weights_df = pd.DataFrame(
        [price_weights, emission_weights, attitude_weights],
        index=["cost_norm", "emissions_norm", "attitude"],
    ).T
    return incomes, weights_df



class TechnologyAdoptionModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(
        self,
        N: int,
        province: str,
        grid_side_length: int = None,
        start_year=2000,
        interact=True,
        years_per_step=1 / 4,
        random_seed=42,
        n_segregation_steps=0,
        segregation_track_property="disposable_income",
        tech_attitude_dist_func=None,
        tech_attitude_dist_params=None,
        price_weight_mode=None,
    ):
        self.random.seed(random_seed)
        np.random.seed(random_seed)

        if grid_side_length is None:
            # ensure grid has more capacity than agents
            grid_side_length = int(np.sqrt(N)) + 1


        if n_segregation_steps:
            # ensure grid has more capacity than agents
            assert grid_side_length**2 > N, AssertionError(
                f"""Segregation requires empty cells, which might not occur when 
                    placing {N} agents on a {grid_side_length}x{grid_side_length} grid."""
            )

        self.num_agents = N
        # self.grid = mesa.space.MultiGrid(width, height, True)
        self.grid = mesa.space.MultiGrid(grid_side_length, grid_side_length, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.start_year = start_year
        self.current_year = start_year
        self.years_per_step = years_per_step
        self.province = province
        self.running = True
        self.interact = interact
        # generate agent parameters: income, energy demand, technology distribution
        income_distribution, weights_df = get_income_and_attitude_weights(self.num_agents, price_weight_mode=price_weight_mode)

        # space heating and hot water make up ~80 % of total final energy demand
        # https://oee.nrcan.gc.ca/corporate/statistics/neud/dpa/showTable.cfm?type=CP&sector=res&juris=ca&year=2020&rn=2&page=0
        heat_demand = (
            energy_demand_from_income_and_province(income_distribution, province) * 0.79
        )
        
        self.heating_techs_df = merge_heating_techs_with_share(start_year=start_year, province=province)
        self.heating_techs_df["province"] = province
        

        prices = []
        for fuel in self.heating_techs_df["fuel"]:
            fuel_price = get_fuel_price(fuel, province, start_year)
            prices.append(fuel_price)
        self.heating_techs_df["specific_fuel_cost"] = prices
        # "upper_idx" up to which agents receive certain heating tech
        self.heating_techs_df["upper_idx"] = (
            self.heating_techs_df["cum_share"] * N
        ).astype(int)

        # draw tech attitudes if necessary
        if tech_attitude_dist_params is None and tech_attitude_dist_func is None:
            tech_attitudes = [None] * self.num_agents
        else:
            tech_attitudes = self.draw_attitudes_from_distribution(
                tech_attitude_dist_func, tech_attitude_dist_params
            )
            # transform dataframe to dict, where each keys equal the previous index
            # each element in that dict, is itself a dict with columns as keys
            tech_attitudes = tech_attitudes.to_dict(orient="index")

        # Create agents
        for i in range(self.num_agents):
            # get the first row, where the i < upper_idx
            try:
                heat_tech_row = self.heating_techs_df.query(f"{i}<=upper_idx").iloc[
                    0, :
                ]
            except IndexError as e:
                print(i, len(self.heating_techs_df), self.heating_techs_df["upper_idx"])
                raise e

            heat_tech_i = HeatingTechnology.from_series(heat_tech_row)
            tech_attitudes_i = tech_attitudes[i]
            a = HouseholdAgent(
                i,
                self,
                income_distribution[i],
                heat_tech_i,
                heat_demand[i],
                years_per_step=self.years_per_step,
                tech_attitudes=tech_attitudes_i,
                criteria_weights=weights_df.loc[i,:].to_dict(),
            )
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # perform segregation
        self.segregation_df = self.perform_segregation(n_segregation_steps, capture_attribute=segregation_track_property)

        # setup a datacollector for tracking changes over time
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Technology shares": self.heating_technology_shares,
                "Energy demand time series": self.energy_demand_ts,
            },
            agent_reporters={"Attitudes": "tech_attitudes", "Wealth": "wealth", "Adoption details":"adopted_technologies"},
        )


    def draw_attitudes_from_distribution(
        self, tech_attitude_dist_func, tech_attitude_dist_params
    ):
        """draws a random attitude for each technology

        Args:
            tech_attitude_dist_func (func): a function to calculate the distribution. First parameter of that function needs to be the mode of the distrubition.
            tech_attitude_dist_params (dict): a dictionary where keys specify technologies and values the mode of the distribution.
            e.g. {"Heat pump" : 0.5, , ...}
        # Returns
            a `pd.DataFrame` column names equal to the keys of the `tech_attitude_dist_params`
        """
        # ensure all techs are present
        tech_set = set(self.heating_techs_df.index)
        assert tech_set.intersection(tech_attitude_dist_params.keys()) == tech_set

        df = pd.DataFrame()
        for k, v in tech_attitude_dist_params.items():
            df.loc[:, k] = tech_attitude_dist_func(v, self.num_agents)

        return df


    def perform_segregation(self, n_segregation_steps, capture_attribute: str = ""):
        data = []
        for i in range(n_segregation_steps):
            if capture_attribute:
                attribute_df = self.get_agents_attribute_on_grid(capture_attribute)
                data.append(attribute_df)
            for a in self.schedule.agents:
                a.move_or_stay_check()

        if capture_attribute:
            return data

    def get_agents_attribute_on_grid(self, attribute, func=np.mean, dtype=float):
        attribute_df = pd.DataFrame()
        for cell_content, (x, y) in self.grid.coord_iter():
            if func is not None:
                if len(cell_content) >= 1:
                    attribute_value = func(
                        [getattr(a, attribute) for a in cell_content]
                    )
                else:
                    attribute_value = dtype(0)
            else:
                attribute_value = [getattr(a, attribute) for a in cell_content]
            attribute_df.at[x, y] = attribute_value
        return attribute_df

    def get_steps_as_years(self):
        s_year = np.array(self.start_year)
        return s_year + np.arange(self.schedule.steps) * self.years_per_step

    @staticmethod
    def steps_to_years_static(
        start_year: float, steps: Iterable, years_per_step: float
    ):
        s_year = np.array(start_year)
        return s_year + np.array(steps) * years_per_step

    def steps_to_years(self, steps):
        s_year = np.array(self.start_year)
        return s_year + np.array(steps) * self.years_per_step

    def update_cost_params(self, year):
        """updates the parameters of the heating technology dataframe

        Args:
            year (float): the year to which the cost parameters should adhere
        """

        # only update costs for full years
        if year % 1 > 0:
            return

        prices = []
        for fuel in self.heating_techs_df["fuel"]:
            fuel_price = get_fuel_price(fuel, self.province, year)
            prices.append(fuel_price)
        self.heating_techs_df["specific_fuel_cost"] = prices

        data_years = np.array(tech_capex_df.reset_index()["year"].unique())
        dist_to_years = abs(data_years - year)
        closest_year_idx = np.argmin(dist_to_years)
        closest_year = data_years[closest_year_idx]
        new_params = tech_capex_df.loc[closest_year, :].T
        self.heating_techs_df.loc[
            :, ["specific_cost", "specific_fom_cost"]
        ] = new_params[["specific_cost", "specific_fom_cost"]]

    def heating_technology_shares(self):
        shares = dict(
            zip(self.heating_techs_df.index, [0] * len(self.heating_techs_df))
        )
        for a in self.schedule.agents:
            shares[a.heating_tech.name] += 1

        for tech in self.heating_techs_df.index:
            shares[tech] /= self.num_agents

        return shares.copy()

    def visualize_grid_attribute(
        self, attribute, layout_update_dict: dict = dict(legend_traceorder="reversed")
    ):
        if attribute is str:
            g_param_df = self.get_agents_attribute_on_grid(attribute)
            param_values = np.array([g_param_df.values])
        elif isinstance(attribute, pd.DataFrame):
            param_values = np.array([attribute.values])
        elif isinstance(attribute, list):
            param_values = np.array([p.values for p in attribute])

        fig = px.imshow(
            param_values,
            facet_col=0,
            width=600,
        )

        # fig.update_layout(**layout_update_dict)
        return fig

    def step(self):
        """Advance the model by one step."""
        self.update_cost_params(self.current_year)
        # data collection needs to be before step, otherwise collected data is off in batch runs
        self.datacollector.collect(self)
        self.schedule.step()

        # adoption_details = self.get_adoption_details()
        # self.datacollector.add_table_row("Adoption details", adoption_details.to_dict())
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

    @classmethod
    def get_result_dir(cls, subdir=""):
        now = datetime.now().strftime(r"%Y%m%d_%H-%M")

        result_dir = Path("results").joinpath(subdir).joinpath(now)
        if not result_dir.exists():
            result_dir.mkdir(exist_ok=True, parents=True)
        return result_dir

    def get_adoption_details(self, post_run=False) -> pd.DataFrame:
        """get adoption details for each agent

        Returns:
            agents_adoption_df (pd.DataFrame): A dataframe containing the adoption details for each agent. Looks like this:

            ```
            +----------+-----+-----------+--------+-------+ 
            | agent_id | year| tech      | reason | value |
            +----------+-----+-----------+--------+-------+
            | 0        | 2000| Gas boiler| mcda   |  1    |  
            | 1        | 2008| Heat pump | mcda   |  1    |
            +----------+-----+-----------+--------+-------+
            ```
        """
        
        if self.schedule.steps < 1:
            print(f"Warning: {self.schedule.steps=}. There may be no adoption details.")

        agents_adoption_df = self.datacollector.get_table_dataframe("Adoption details")

        if not post_run:
            # this is the case for datacollection
            return agents_adoption_df.query(f"year == {self.current_year}")   
        else:
            return agents_adoption_df

    def get_heating_techs_age(self):
        techs = []
        for a in model.schedule.agents:
            techs.append((a.heating_tech.name, a.heating_tech.age))

        df = pd.DataFrame.from_records(techs, columns=["tech", "age"])
        return df

if __name__ == "__main__":
    province = "Canada"
    
    heating_techs_df = merge_heating_techs_with_share( province=province)
    model = TechnologyAdoptionModel(
        200, province, start_year=2000, n_segregation_steps=40
    )

    # model.perform_segregation(30)

    for _ in range(80):
        model.step()

    # model_vars = model.datacollector.get_model_vars_dataframe()
    # adoption_col = model_vars["Technology shares"].to_list()
    # adoption_df = pd.DataFrame.from_records(adoption_col)
    # adoption_df.index = model.get_steps_as_years()

    # adoption_detail = model_vars[["Step","RunId","Adoption details","AgentID"]]
    # adoption_detail.loc[:,["tech","reason"]] = pd.DataFrame.from_records(adoption_detail["Adoption details"].values)
    # adoption_detail = adoption_detail.drop("Adoption details", axis=1)
    # adoption_detail["amount"] = 1
    # drop_rows = adoption_detail["tech"].apply(lambda x: x is None)
    # adoption_detail = adoption_detail.loc[~drop_rows,:]

    # adoption_detail = adoption_detail.groupby(["Step","RunId","tech","reason"]).sum().reset_index()

    # # get cumulative sum
    # adoption_detail["cumulative_amount"] = adoption_detail.groupby(["RunId","tech","reason"]).cumsum()["amount"]

    # # fig = px.bar(adoption_detail, x="Step", y="amount", color="tech", facet_col="RunId", facet_row="reason", template="plotly") 
    # fig = px.area(adoption_detail, x="Step", y="cumulative_amount", color="tech", facet_col="RunId", facet_row="reason", template="plotly") 
    # fig.update_yaxes(matches=None)
    # fig.show()


    # results_dir = TechnologyAdoptionModel.get_result_dir()
    # adoption_df.plot().get_figure().savefig(results_dir.joinpath("adoption.png"))

    
