import numpy as np
import pandas as pd
import plotly.express as px
from typing import Iterable
from pathlib import Path
from datetime import datetime

import mesa
from components.agent import HouseholdAgent
from components.technologies import (
    HeatingTechnology,
    Technologies,
    merge_heating_techs_with_share,
    Fuels,
    tech_fuel_map,
)
from components.probability import beta_with_mode_at

from data.canada import (
    get_beta_distributed_incomes,
    uncertain_demand_from_income_and_province,
    get_fuel_price,
    tech_capex_df,
    get_end_use_agg_heating_share,
    nrcan_end_use_df,
)
from decision_making.mcda import normalize


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
    emission_weights = np.float32(np.random.random(len(price_weights)))

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
        interact=False,
        years_per_step=1 / 4,
        random_seed=42,
        n_segregation_steps=0,
        segregation_track_property="",  # "disposable_income"
        tech_att_mode_table=None,
        tech_attitude_dist_func=None,
        tech_attitude_dist_params=None,
        price_weight_mode=None,
        global_util_thresh=0.1,
        ts_step_length="H",
        refurbishment_rate=0.0,
        hp_subsidy=0.0,
        fossil_ban_year=None,
    ):
        super().__init__()
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

        self.fossil_ban_year = fossil_ban_year
        self.hp_subsidy = hp_subsidy
        self.refurbishment_rate = refurbishment_rate
        self.refurbished_agents = []
        self.num_agents = N
        self.grid = mesa.space.MultiGrid(grid_side_length, grid_side_length, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.start_year = start_year
        self.current_year = start_year
        self.years_per_step = years_per_step
        self.province = province
        self.running = True
        self.interact = interact
        # generate agent parameters: income, energy demand, tech distribution
        income_distribution, weights_df = get_income_and_attitude_weights(
            self.num_agents, price_weight_mode=price_weight_mode
        )
        self.att_mode_table = tech_att_mode_table
        self.available_techs = list(Technologies)
        # space heating and hot water make up ~80 % of final energy demand
        # https://oee.nrcan.gc.ca/corporate/statistics/neud/dpa/showTable.cfm?type=CP&sector=res&juris=ca&year=2020&rn=2&page=0
        total_energy_demand = uncertain_demand_from_income_and_province(
            income_distribution, province
        )
        # scale total energy demand from "per_household" to "per_income_group"
        total_energy_demand = (
            total_energy_demand
            / total_energy_demand.sum()
            * nrcan_end_use_df.loc[(province, "Total Energy Use (PJ)"), start_year]
            * 1
            / 3600  # J -> Wh
            * 10 ** (4 * 3)  # P(10^15) -> k(10^3)
        )

        province_heat_share = get_end_use_agg_heating_share(province, start_year)
        heat_demand = total_energy_demand * province_heat_share
        self.heating_techs_df = merge_heating_techs_with_share(
            start_year=start_year, province=province
        )
        self.heating_techs_df["province"] = province
        self.global_util_thresh = global_util_thresh
        self.update_fuel_prices(self.province, self.current_year)
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
                criteria_weights=weights_df.loc[i, :].to_dict(),
                ts_step_length=ts_step_length,
                hp_subsidy=hp_subsidy,
                fossil_ban_year=fossil_ban_year,
            )
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # perform segregation
        self.segregation_df = self.perform_segregation(
            n_segregation_steps, capture_attribute=segregation_track_property
        )

        # setup a datacollector for tracking changes over time
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Technology shares": self.heating_technology_shares,
                "Energy demand time series": self.energy_demand_ts,
            },
            agent_reporters={
                "Attitudes": "tech_attitudes",
                "Adoption details": "adopted_technologies",
                "Appliance age": "heating_tech.age",
                "Appliance name": "heating_tech_name",
                "Technology annual_cost": "annual_costs",
                "Heat pump specific_cost": "specific_hp_cost",
                "Refurbished": "is_refurbished",
                "Required heating size": "req_heating_cap",
                "Heat demand": "heat_demand",
                "LCOH": "lcoh",
                "Cost components": "current_cost_components"
            },
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

    def update_attitudes(self, year):
        if year % 1:
            # it's not a full year, do nothing
            return

        if year not in self.att_mode_table.index:
            print(f"{year} not in att_mode_table.")
            return

        # use predefined att_mode_table to draw attitudes
        tech_att_modes = self.att_mode_table.loc[year, :].to_dict()
        tech_att_df = self.draw_attitudes_from_distribution(
            beta_with_mode_at, tech_att_modes
        )
        for i, a in enumerate(self.schedule.agents):
            new_atts = tech_att_df.loc[i, :].to_dict()
            a.tech_attitudes = new_atts

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

    def update_fuel_prices(self, province, year, debug=False):
        for tech, fuel in tech_fuel_map.items():
            fuel_price = get_fuel_price(fuel, self.province, year)
            self.heating_techs_df.at[tech, "specific_fuel_cost"] = fuel_price
        if debug:
            print(year, self.heating_techs_df["specific_fuel_cost"])

    def update_cost_params(self, year, discount_rate=0.07):
        """updates the parameters of the heating technology dataframe

        Args:
            year (float): the year to which the cost parameters should adhere
        """

        self.update_fuel_prices(self.province, year)

        data_years = np.array(tech_capex_df.reset_index()["year"].unique())
        dist_to_years = abs(data_years - year)
        closest_year_idx = np.argmin(dist_to_years)
        closest_year = data_years[closest_year_idx]
        new_params = tech_capex_df.loc[closest_year, :].T
        self.heating_techs_df.loc[:, ["specific_cost", "specific_fom_cost"]] = (
            new_params[["specific_cost", "specific_fom_cost"]]
        )
        if "annuity_factor" in tech_capex_df.index:
            self.heating_techs_df["annuity_factor"] = tech_capex_df.loc[
                (closest_year, "annuity_factor"), :
            ]
        else:
            self.heating_techs_df["annuity_factor"] = discount_rate / (
                1
                - (1 + discount_rate)
                ** -tech_capex_df.loc[(closest_year, "lifetime"), :].astype(float)
            )

    def heating_technology_shares(self):
        shares = dict(zip(list(Technologies), [0] * len(Technologies)))
        for a in self.schedule.agents:
            shares[a.heating_tech.name] += 1

        for tech in shares.keys():
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
        # only update for full years
        # print()
        if self.current_year % 1 == 0:
            self.update_cost_params(self.current_year)
            self.apply_refurbishments(self.refurbishment_rate)
        if isinstance(self.att_mode_table, pd.DataFrame):
            self.update_attitudes(self.current_year)

        if self.fossil_ban_year:
            if self.current_year >= self.fossil_ban_year:
                if Technologies.GAS_FURNACE in self.available_techs:
                    self.available_techs.remove(Technologies.GAS_FURNACE)
                if Technologies.OIL_FURNACE in self.available_techs:
                    self.available_techs.remove(Technologies.OIL_FURNACE)

        # data collection needs to be before step, otherwise collected data is off in batch runs
        self.datacollector.collect(self)
        self.schedule.step()
        self.current_year += self.years_per_step

    def apply_refurbishments(self, rate):
        # handle edge cases
        if rate == 0.0:
            return
        elif rate > 1:
            raise ValueError(f"Refurbishment rates must be < 1. Received:{rate}")

        agents = self.schedule.agents

        unrefurbed_agents = list(set(agents).difference(self.refurbished_agents))
        no_refurb_agents = int(np.ceil(len(unrefurbed_agents) * rate))

        if not unrefurbed_agents:
            # there are no more agents to refurbish
            return
        agents_2b_refurbed: HouseholdAgent = np.random.choice(
            unrefurbed_agents, no_refurb_agents, replace=False
        )
        dem_red = np.random.normal(0.4875, 0.125, no_refurb_agents)
        # ensure (0,1) boundaries
        dem_red[dem_red < 0] = -dem_red[dem_red < 0]
        dem_red[dem_red > 1] = 1 - (dem_red[dem_red > 1] - 1)
        for agent, reduction in zip(agents_2b_refurbed, dem_red):
            agent.refurbish(reduction)
            self.refurbished_agents.append(agent)

        pass

    def energy_demand_ts(self):
        energy_carrier_demand = dict(zip(Fuels, [0] * len(Fuels)))

        zero_demand_fuels = list(Fuels)
        # retrieve the energy demand from each agent
        for a in self.schedule.agents:
            # get fueltype of agent
            fuel = a.heating_tech.fuel
            if fuel in zero_demand_fuels:
                zero_demand_fuels.remove(fuel)
            energy_carrier_demand[fuel] += a.current_fuel_demand

        any_demand_fuel = set(Fuels).difference(zero_demand_fuels).pop()
        for fuel in zero_demand_fuels:
            energy_carrier_demand[fuel] = energy_carrier_demand[any_demand_fuel] * 0

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
    from data.canada import nrcan_tech_shares_df

    historic_tech_shares = nrcan_tech_shares_df.copy()
    historic_tech_shares.index = historic_tech_shares.index.swaplevel()

    province = "Ontario"
    h_tech_shares = historic_tech_shares.loc[province, :] / 100
    att_mode_table = h_tech_shares.copy()

    model = TechnologyAdoptionModel(
        90,
        province,
        start_year=2020,
        n_segregation_steps=40,
        tech_att_mode_table=att_mode_table,
        refurbishment_rate=0.03,
        hp_subsidy=0.3,
        fossil_ban_year=2029,
        ts_step_length="w",
    )

    # model.perform_segregation(30)

    for _ in range(120):
        model.step()

    # results_dir = TechnologyAdoptionModel.get_result_dir()
    # adoption_df.plot().get_figure().savefig(results_dir.joinpath("adoption.png"))
