import pandas as pd
import numpy as np

# results of parameter fit 14.03.2023
MODES_2020 = {
    "Electric furnace": 0.64695,
    "Gas furnace": 0.05,
    "Heat pump": 0.636919,
    "Oil furnace": 0.05,
    "Wood or wood pellets furnace": 0.116165,
}

FAST_TRANSITION_MODES_AND_YEARS = {
    "Electric furnace": {"end_att": 0.2, "at_year": 2040},
    "Gas furnace": {"end_att": 0.05, "at_year": 2030},
    "Heat pump": {"end_att": 0.95, "at_year": 2030},
    "Oil furnace": {"end_att": 0.05, "at_year": 2030},
    "Wood or wood pellets furnace": {"end_att": 0.2, "at_year": 2030},
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


def price_reduction(p0, x, lr):
    b = -np.log((1 - lr / 100)) / np.log(2)
    return p0 * x**-b


def generate_cost_projections(learning_rate=11.1, write_csv=False):
    # qantity according to IEA
    heat_pump_installations = np.linspace(1, 1620, 30)

    # price at 2020
    hp_price_0 = 770.751
    future_prices_np = np.array(
        [price_reduction(hp_price_0, heat_pump_installations, learning_rate)]
    ).T
    future_prices = pd.DataFrame(
        future_prices_np,
        index=heat_pump_installations + 180,
        columns=[f"{learning_rate:.1f}%"],
    )
    future_prices["year"] = np.arange(2020, 2050)
    future_prices = future_prices.set_index("year")
    costs = pd.read_csv("data/canada/heat_tech_params.csv", index_col=list(range(2)))
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
        index=pd.MultiIndex.from_product((all_years, params), names=["year","variable"]), columns=costs.columns
    )
    keep_rows = [i for i, idx in enumerate(empty_frame.index) if idx not in costs.index]
    empty_frame = empty_frame.iloc[keep_rows, :]
    costs = pd.concat([empty_frame, costs]).sort_index(level=(1, 0)).interpolate()

    if write_csv:
        costs.to_csv(
            "data/canada/heat_tech_params.csv", index_label=["year", "variable"]
        )
    return costs
