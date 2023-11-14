from pydantic.dataclasses import dataclass
from typing import ClassVar, Dict
import numpy as np
import pandas as pd
from enum import Enum
from data.canada import tech_capex_df, nrcan_tech_shares_df
from data.canada.timeseries import necessary_heating_capacity_for_province
from decision_making.mcda import normalize
from functools import partial


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


@dataclass
class HeatingTechnology:
    name: Technologies
    specific_cost: float
    specific_fuel_cost: float
    specific_fuel_emission: float
    efficiency: float
    lifetime: int
    fuel: Fuels
    province: str
    age: int = 0
    possible_fuels: ClassVar[Fuels] = list(Fuels)
    tech_fuel_map: ClassVar[Dict[Technologies, Fuels]] = dict(zip(Technologies, Fuels))

    def __post_init__(self):
        self.tech_fuel_map.update({Technologies.HEAT_PUMP: Fuels.ELECTRICITY})

    @classmethod
    def from_series(cls, series, existing=True):
        params = list(cls.__match_args__)[1:]
        if existing:
            age = np.random.choice(int(series.loc["lifetime"]))
        else:
            age = 0
        name = series.name
        series = pd.concat([series, pd.Series({"age": age})])
        series.name = name
        values_row = series[params]
        assert series.name is not None

        assert values_row["fuel"] in cls.possible_fuels
        return HeatingTechnology(series.name, **values_row.to_dict())

    def total_cost_per_year(self, heating_demand, discount_rate=0.07):
        fuel_cost = heating_demand / self.efficiency * self.specific_fuel_cost
        annuity_factor = discount_rate / (1 - (1 + discount_rate) ** -self.lifetime)

        # TODO: this needs to be precomputed as this introduced a drop in performance
        size = necessary_heating_capacity_for_province(heating_demand)
        annuity_payment = size * annuity_factor
        fom_cost = annuity_payment * 0.02
        return annuity_payment + fuel_cost + fom_cost

    @classmethod
    def annual_cost_from_df(cls, heating_demand, tech_df, discount_rate=0.07):
        if "annuity_factor" in tech_df.columns:
            costs = cls.annual_cost_from_df_fast(heating_demand, tech_df)
            return costs
        else:
            fuel_cost = (
                heating_demand / tech_df["efficiency"] * tech_df["specific_fuel_cost"]
            )
            annuity_factor = discount_rate / (
                1 - (1 + discount_rate) ** -tech_df["lifetime"]
            )

            size = necessary_heating_capacity_for_province(heating_demand)
            annuity_payment = size * annuity_factor * tech_df["specific_cost"]
            fom_cost = size * tech_df["specific_fom_cost"].astype(float)
            return annuity_payment + fuel_cost + fom_cost

    @staticmethod
    def annual_cost_from_df_fast(heating_demand, tech_df):
        efficiencies = tech_df["efficiency"].values
        specific_fuel_cost = tech_df["specific_fuel_cost"].values
        fuel_cost = heating_demand / efficiencies * specific_fuel_cost

        size = necessary_heating_capacity_for_province(heating_demand)
        annuity = (
            tech_df["annuity_factor"].values
            * tech_df["specific_cost"].astype(float).values
        )
        annuity_and_fom = size * np.array(
            [annuity, tech_df["specific_fom_cost"].astype(float)]
        )
        return fuel_cost + annuity_and_fom.sum(axis=0)


def is_num(x):
    try:
        float(x)
        return True
    except Exception:
        return False


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

    heat_techs_df["emissions[kg_CO2/kWh_th]"] = heat_techs_df[
        "specific_fuel_emission"
    ].astype(float) / heat_techs_df["efficiency"].astype(float)

    heat_techs_df["annuity_factor"] = discount_rate / (
        1
        - (1 + discount_rate)
        ** -tech_capex_df.loc[(closest_year, "lifetime"), :].astype(float)
    )

    p_normalize = partial(normalize, direction=-1)
    heat_techs_df.loc[:, "emissions_norm"] = (
        heat_techs_df[["emissions[kg_CO2/kWh_th]"]].apply(p_normalize).values
    )
    num_cols = heat_techs_df.iloc[0, :].apply(is_num)
    heat_techs_df[heat_techs_df.columns[num_cols]] = heat_techs_df[
        heat_techs_df.columns[num_cols]
    ].astype(float)
    return heat_techs_df
