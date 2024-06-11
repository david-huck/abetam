import pandas as pd
import numpy as np


# results of parameter fit 07.04.2024
MODES_2020 = {
    "Electric furnace": 0.677991,
    "Gas furnace": 0.076923,
    "Heat pump": 0.534513,
    "Oil furnace": 0.050000,
    "Wood or wood pellets furnace": 0.109409,
}

FAST_TRANSITION_MODES_AND_YEARS = {
    "Electric furnace": {"end_att": 0.2, "at_year": 2040},
    "Gas furnace": {"end_att": 0.05, "at_year": 2030},
    "Heat pump": {"end_att": 0.95, "at_year": 2030},
    "Oil furnace": {"end_att": 0.05, "at_year": 2030},
    "Wood or wood pellets furnace": {"end_att": 0.2, "at_year": 2030},
}
FAST_TRANSITION_HP_LR = 11.1
CER_TRANSITION_HP_LR = 7.5
SLOW_TRANSITION_HP_LR = 5.5

# MODES_2020, unchanged
SLOW_TRANSITION_MODES_AND_YEARS = {
    "Electric furnace": {"end_att": 0.677991, "at_year": 2040},
    "Gas furnace": {"end_att": 0.076923, "at_year": 2030},
    "Heat pump": {"end_att": 0.534513, "at_year": 2030},
    "Oil furnace": {"end_att": 0.050000, "at_year": 2030},
    "Wood or wood pellets furnace": {"end_att": 0.109409, "at_year": 2030},
}


def generate_scenario_attitudes(
    starting_modes: dict[str, float],
    end_modes_and_years: dict[str, dict],
):
    """generates parameters for the scenario define in the parameters

    Args:
        starting_modes (dict[str, float]): Mapping of Technologies an their 2020 attitude mode.
        end_modes_and_years (dict[str, dict]): Mapping of Technologies to their end attitude
            `end_att` and the year in which it is present `at_year`.
    """
    mode_df = pd.DataFrame(
        starting_modes.values(), index=starting_modes.keys(), columns=[2020]
    ).T

    for tech, params in end_modes_and_years.items():
        mode_df.at[params["at_year"], tech] = params["end_att"]
        mode_df.at[2050, tech] = params["end_att"]

    # concat with empty frame for interpolation over missing years
    scenario_years = range(2020, 2051)
    missing_years = [y for y in scenario_years if y not in mode_df.index]
    empty_df = pd.DataFrame(columns=mode_df.columns, index=missing_years)
    scenario_df = pd.concat([mode_df, empty_df]).sort_index().interpolate()
    return scenario_df


def price_reduction(x, lr, p0=1):
    b = -np.log((1 - lr / 100)) / np.log(2)
    return p0 * x**-b


def lr_based_cost_factors(x, lr, p0=1):
    # sets first (2020) data point to 1
    price_factors = price_reduction(x, lr, p0)
    return price_factors / list(price_factors)[0]


def generate_hp_cost_projections(learning_rate=11.1, write_csv=False):
    # qantity according to IEA
    heat_pump_installations = np.linspace(180, 1800, 31)

    # price at 2020
    hp_price_0 = 770.751
    future_prices_np = (
        np.array([lr_based_cost_factors(heat_pump_installations, learning_rate)]).T
        * hp_price_0
    )
    future_prices = pd.DataFrame(
        future_prices_np,
        index=heat_pump_installations,
        columns=[f"{learning_rate:.1f}%"],
    )
    future_prices["year"] = np.arange(2020, 2051)
    future_prices = future_prices.set_index("year")

    from data.canada import repo_root

    costs = pd.read_csv(
        f"{repo_root}/data/canada/heat_tech_params.csv", index_col=list(range(2))
    )
    for i, row in future_prices.iterrows():
        costs.at[(i, "specific_cost"), "Heat pump"] = row[f"{learning_rate:.1f}%"]
        if i > 2020:
            for other_tech in [
                "Electric furnace",
                "Gas furnace",
                "Oil furnace",
                "Wood or wood pellets furnace",
            ]:
                # take last available cost param if not present...
                if (i, "specific_cost") not in costs.index:
                    costs.at[(i, "specific_cost"), other_tech] = costs.loc[
                        (i - 1, "specific_cost"), other_tech
                    ]
                else:
                    # or if it is empty. This happens often, only the Heat pump column was written previously
                    if pd.isna(costs.at[(i, "specific_cost"), other_tech]):
                        costs.at[(i, "specific_cost"), other_tech] = costs.loc[
                            (i - 1, "specific_cost"), other_tech
                        ]

    # add fom costs, where missing:
    years_w_fom = costs.loc[:, "specific_fom_cost", :].index

    years_no_fom = list(set(range(2000, 2051)).difference(years_w_fom))
    for year in years_no_fom:
        for tech in costs.columns:
            costs.at[(year, "specific_fom_cost"), tech] = (
                costs.loc[(year, "specific_cost"), tech] * 0.02
            )

    # interpolate remaining data
    avail_years = costs.reset_index()["year"].unique()
    all_years = range(min(avail_years), max(avail_years))
    params = costs.reset_index()["variable"].unique()

    empty_frame = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            (all_years, params), names=["year", "variable"]
        ),
        columns=costs.columns,
    ).astype(float)
    keep_rows = [i for i, idx in enumerate(empty_frame.index) if idx not in costs.index]
    empty_frame = empty_frame.iloc[keep_rows, :]
    costs = pd.concat([empty_frame, costs]).sort_index(level=(1, 0)).interpolate()

    if write_csv:
        costs.to_csv(
            f"{repo_root}/data/canada/heat_tech_params.csv",
            index_label=["year", "variable"],
        )
    return costs


def interpolate_missing_years(series: pd.Series):
    """
    Interpolates missing years in a pandas Series.

    This function takes a pandas Series object with years as the index and fills in any missing years by interpolation.
    The interpolation method used is linear interpolation.

    Parameters:
    series (pd.Series): The input pandas Series. The index of the series should be the years.

    Returns:
    pd.Series: A pandas Series with missing years filled by interpolation.

    Example:
    >>> s = pd.Series([1, 3], index=[2000, 2002])
    >>> interpolate_missing_years(s)
    2000    1.0
    2001    2.0
    2002    3.0
    dtype: float64
    """
    start = int(series.index.min())
    end = int(series.index.max())
    miss_years = set(range(start, end + 1)).difference(series.index)
    na_series = pd.Series(index=miss_years)
    return pd.concat([series, na_series]).astype(float).sort_index().interpolate()


CT = interpolate_missing_years(
    pd.Series({2020: 0, 2021: 0, 2022: 0, 2023: 65, 2030: 170, 2050: 170}) / 1000
)

def update_price_w_new_CT(tup, new_CT=None):
    if new_CT is None:
        raise ValueError("need to change value of `new_CT` with functools.partial")
    price = tup["Price (ct/kWh)"]
    year = tup["Year"]
    fuel_type = tup["Type of fuel"]
    fuel_emissions = {"Natural gas": 0.2, "Heating oil": 0.5}  # kg/kWh
    known_fuel = fuel_type in fuel_emissions.keys()
    year_applicable = year in CT.index
    if not known_fuel or not year_applicable:
        return price

    spec_em = fuel_emissions[fuel_type]

    sCTp_y = spec_em * CT[year] / (price / 100)
    if sCTp_y > 1: 
        print(f"Share of carbon tax > 1 in {tup['GEO']}. Check {fuel_type=},{spec_em=},{CT[year]=},{year=},{price=}")
    new_price = (price / 100 * (1 - sCTp_y) + new_CT[year] * spec_em) * 100
    return new_price