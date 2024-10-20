from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import ClassVar, Dict, Iterable
import numpy as np
import pandas as pd
from enum import Enum
from data.canada import tech_capex_df, nrcan_tech_shares_df
from data.canada.timeseries import necessary_heating_capacity_for_province, cop_df
from decision_making.mcda import normalize
from functools import partial
from components.probability import beta_with_mode_at


class Fuels(str, Enum):
    NATURAL_GAS = "Natural gas"
    HEATING_OIL = "Heating oil"
    WOOD_OR_WOOD_PELLETS = "Wood or wood pellets"
    ELECTRICITY = "Electricity"


class Technologies(str, Enum):
    GAS_FURNACE = "Gas furnace"
    OIL_FURNACE = "Oil furnace"
    WOOD_PELLETS_FURNACE = "Wood or wood pellets furnace"
    ELECTRIC_FURNACE = "Electric furnace"
    HEAT_PUMP = "Heat pump"

    def __repr__(self) -> str:
        return f"Technologies({self.value})"

    def __str__(self) -> str:
        return self.value


tech_fuel_map = dict(zip(Technologies, Fuels))
tech_fuel_map.update({Technologies.HEAT_PUMP: Fuels.ELECTRICITY})


@dataclass
class HeatingTechnology:
    name: Technologies
    specific_cost: float
    specific_fuel_cost: float
    specific_fuel_emission: float
    efficiency: float
    lifetime: int
    province: str
    age: int = 0
    possible_fuels: ClassVar[Fuels] = list(Fuels)
    tech_fuel_map: ClassVar[Dict[Technologies, Fuels]] = dict(zip(Technologies, Fuels))
    fuel: ClassVar[Fuels] = Field(init=False)

    def __post_init__(self):
        # mapping in tech_fuel_map is based on order, but since techs is 1
        # longer there is no fuel mapped to the heatpump
        self.tech_fuel_map.update({Technologies.HEAT_PUMP: Fuels.ELECTRICITY})
        self.fuel = self.tech_fuel_map[self.name]

    @classmethod
    def from_series(cls, series, existing=True):
        params = list(cls.__match_args__)[1:]
        if existing:
            # age = np.random.choice(int(series.loc["lifetime"]))
            max_age = series.loc["lifetime"]
            age = beta_with_mode_at(0.3, 1, (0, max_age))
            age = int(age)
        else:
            age = 0
        name = series.name
        series = pd.concat([series, pd.Series({"age": age})])
        series.name = name
        values_row = series[params]
        assert series.name is not None

        # assert values_row["fuel"] in cls.possible_fuels
        return HeatingTechnology(series.name, **values_row.to_dict())

    def total_cost_per_year(self, heating_demand, discount_rate=0.07, province=None):
        fuel_cost = heating_demand / self.efficiency * self.specific_fuel_cost
        annuity_factor = discount_rate / (1 - (1 + discount_rate) ** -self.lifetime)

        # TODO: this needs to be precomputed as this introduced a drop in performance
        size = necessary_heating_capacity_for_province(heating_demand, province=province)
        annuity_payment = size * annuity_factor
        fom_cost = annuity_payment * 0.02
        return annuity_payment + fuel_cost + fom_cost

    @classmethod
    def annual_cost_with_fuel_demands(cls, heating_demand, fuel_demands, tech_df, province):
        size = necessary_heating_capacity_for_province(
            sum(heating_demand), province=province
        )
        annuity_payment = size * tech_df["annuity_factor"] * tech_df["specific_cost"]
        fom_cost = size * tech_df["specific_fom_cost"]

        specific_fuel_cost = tech_df["specific_fuel_cost"]
        fuel_cost = (fuel_demands * specific_fuel_cost).sum()
        return annuity_payment + fuel_cost + fom_cost

    @classmethod
    def annual_cost_from_df(
        cls, heating_demand, tech_df, discount_rate=0.07, province=None
    ):
        if "annuity_factor" in tech_df.columns:
            costs, fuel_demands = cls.annual_cost_from_df_fast(
                heating_demand, tech_df, province=province
            )
            return costs, fuel_demands
        else:
            fuel_demands = heating_demand / tech_df["efficiency"]
            fuel_cost = fuel_demands * tech_df["specific_fuel_cost"]
            annuity_factor = discount_rate / (
                1 - (1 + discount_rate) ** -tech_df["lifetime"]
            )

            size = necessary_heating_capacity_for_province(
                heating_demand, province=province
            )
            annuity_payment = size * annuity_factor * tech_df["specific_cost"]
            fom_cost = size * tech_df["specific_fom_cost"]
            return annuity_payment + fuel_cost + fom_cost, fuel_demands

    @classmethod
    def annual_cost_from_df_fast(cls, heating_demand, tech_df, province=None):
        efficiencies = tech_df["efficiency"].values
        specific_fuel_cost = tech_df["specific_fuel_cost"].values
        if not isinstance(heating_demand, Iterable):
            fuel_cost, fuel_demands = cls.annual_fuel_cost(
                heating_demand, efficiencies, specific_fuel_cost
            )
        else:
            if province is None:
                raise NotImplementedError(
                    f"""When passing heat demand as timeseries, need to pass a 
                    province, too. Received {province=}."""
                )
            fuel_cost, fuel_demands = cls.annual_fuel_cost_from_ts(
                heating_demand, province, tech_df
            )
            fuel_cost = fuel_cost.sum()
            heating_demand = heating_demand.sum()
        if province is not None:
            size = necessary_heating_capacity_for_province(
                heating_demand, province=province
            )
        else:
            size = necessary_heating_capacity_for_province(heating_demand)

        annuity = tech_df["annuity_factor"].values * tech_df["specific_cost"].values
        annuity_and_fom = size * np.array([annuity, tech_df["specific_fom_cost"]])
        return fuel_cost + annuity_and_fom.sum(axis=0), fuel_demands

    def annual_fuel_cost(heat_demand, efficiencies, specific_fuel_cost):
        fuel_demands = (heat_demand / efficiencies).astype("float32")
        return fuel_demands * specific_fuel_cost.astype("float32"), fuel_demands

    @classmethod
    def annual_fuel_cost_from_ts(cls, heat_demand_ts, province, tech_df):
        fuel_demand_ts = cls.fuel_demand_ts(heat_demand_ts, province, tech_df)
        fuel_cost_ts = fuel_demand_ts * tech_df["specific_fuel_cost"].values
        fuel_cost = fuel_cost_ts.sum()
        return fuel_cost, fuel_demand_ts

    @staticmethod
    def fuel_demand_ts(heat_demand_ts, province, tech_df):
        # get cop time series for HP and assume constant efficiencies for other techs
        fuel_demand_dict = dict(zip(Technologies, [0] * len(Technologies)))
        for tech, eff in tech_df["efficiency"].to_dict().items():
            if tech == Technologies.HEAT_PUMP:
                fuel_demand_dict[tech] = heat_demand_ts.values / cop_df[province].values
            else:
                fuel_demand_dict[tech] = heat_demand_ts.values / eff

        return pd.DataFrame(fuel_demand_dict, index=heat_demand_ts.index).astype("float32")


def is_num(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def cop_from_temp(T_set, T_amb):
    """
    COP = c + a_1*dT + a_2*dT^2
    proposed in 10.1016/j.enbuild.2016.07.008 for HP COP by fitting to
    manufacturers data enhanced in 10.1038/s41597-019-0199-y with a 0.85
    correction factor to account for imperfections and wear. Reported COPs are
    well in line with findings from 10.1016/j.buildenv.2021.108594
    and 10.3390/su15031880 and the "Performance Assessment of Heat Pump Systems
    TECHNICAL BRIEF" from sustainabletechnologies.ca.
    """
    dT = T_set - T_amb
    COP = 0.85 * (6.08 - 0.09 * dT + 0.0005 * dT**2)
    return COP


def merge_heating_techs_with_share(
    start_year=2013, province="Canada", discount_rate=0.07
):
    data_years = np.array(tech_capex_df.reset_index()["year"].unique())
    dist_to_years = abs(data_years - start_year)
    closest_year_idx = np.argmin(dist_to_years)
    closest_year = data_years[closest_year_idx]
    heat_techs_df = tech_capex_df.loc[closest_year, :].T
    if min(dist_to_years) > 5:
        print(
            "Warning: using data from",
            closest_year,
            " for cost parameters, which is the closest in the data to selected year:",
            start_year,
        )

    data_years = np.array(nrcan_tech_shares_df.reset_index()["year"].unique())
    dist_to_years = abs(data_years - start_year)
    closest_year_idx = np.argmin(dist_to_years)
    closest_year_heating_stock = data_years[closest_year_idx]
    if min(dist_to_years) > 5:
        print(
            "Warning: using data from",
            closest_year_heating_stock,
            " for the heating stock, which is the closest in the data to selected year:",
            start_year,
        )

    heat_techs_df.loc[:, "share"] = nrcan_tech_shares_df.loc[
        (closest_year_heating_stock, province), :
    ] / sum(nrcan_tech_shares_df.loc[(closest_year_heating_stock, province), :])
    heat_techs_df["cum_share"] = heat_techs_df["share"].cumsum()

    heat_techs_df["emissions[kg_CO2/kWh_th]"] = (
        heat_techs_df["specific_fuel_emission"] / heat_techs_df["efficiency"]
    )

    heat_techs_df["annuity_factor"] = discount_rate / (
        1 - (1 + discount_rate) ** -tech_capex_df.loc[(closest_year, "lifetime"), :]
    )

    p_normalize = partial(normalize, direction=-1)
    heat_techs_df.loc[:, "emissions_norm"] = (
        heat_techs_df[["emissions[kg_CO2/kWh_th]"]].apply(p_normalize).values
    )
    num_cols = heat_techs_df.iloc[0, :].apply(is_num)
    heat_techs_df[heat_techs_df.columns[num_cols]] = heat_techs_df[
        heat_techs_df.columns[num_cols]
    ]
    return heat_techs_df
