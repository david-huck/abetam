from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import ClassVar, Dict
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
    BIOMASS = "Biomass"
    ELECTRICITY = "Electricity"


class Technologies(str, Enum):
    GAS_FURNACE = "Gas furnace"
    OIL_FURNACE = "Oil furnace"
    BIOMASS_FURNACE = "Biomass furnace"
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
    age: float = 0.0
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

        return HeatingTechnology(series.name, **values_row.to_dict())

    @classmethod
    def annual_cost_with_fuel_demands(
        cls, heating_demand, fuel_demands, tech_df, province, size=None, hp_subsidy=0.0
    ):
        if size is None:
            size = necessary_heating_capacity_for_province(
                sum(heating_demand), province=province
            )
        subs = pd.Series(dict(zip(Technologies, [0.0] * len(Technologies))))
        subs["Heat pump"] = hp_subsidy

        annuity_payment = (
            size * tech_df["annuity_factor"] * tech_df["specific_cost"] * (1 - subs)
        )
        fom_cost = size * tech_df["specific_fom_cost"]

        specific_fuel_cost = tech_df["specific_fuel_cost"]
        fuel_cost = (fuel_demands * specific_fuel_cost).sum()
        return pd.concat(
            [
                annuity_payment.rename("annuity_cost"),
                fuel_cost.rename("fuel_cost"),
                fom_cost.rename("fom_cost"),
            ],
            axis=1,
        )

    @classmethod
    def annual_cost_from_df_fast(
        cls,
        heating_demand,
        tech_df,
        province=None,
        ts_step_length="h",
        hp_eff_incr=0,
        hp_subsidy=0,
        size=None,
    ):
        if province is None:
            raise NotImplementedError(
                f"""When passing heat demand as timeseries, need to pass a 
                province, too. Received {province=}."""
            )
        fuel_demands = cls.fuel_demand_ts(
            heating_demand,
            province,
            tech_df,
            ts_step_length=ts_step_length,
            hp_eff_incr=hp_eff_incr,
        )
        cost_components = cls.annual_cost_with_fuel_demands(
            heating_demand,
            fuel_demands,
            tech_df,
            province=province,
            size=size,
            hp_subsidy=hp_subsidy,
        )

        return cost_components, fuel_demands

    @staticmethod
    def fuel_demand_ts(
        heat_demand_ts, province, tech_df, ts_step_length="h", hp_eff_incr=0
    ):
        # get cop time series for HP and assume constant efficiencies for other techs
        fuel_demand_dict = dict(zip(Technologies, [0] * len(Technologies)))
        for tech, eff in tech_df["efficiency"].to_dict().items():
            if tech == Technologies.HEAT_PUMP:
                fuel_demand_dict[tech] = heat_demand_ts.values / (
                    cop_df[province].resample(ts_step_length).mean().values
                    * (1 + hp_eff_incr)
                )
            else:
                fuel_demand_dict[tech] = heat_demand_ts.values / eff

        return pd.DataFrame(fuel_demand_dict, index=heat_demand_ts.index).astype(
            "float32"
        )


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
