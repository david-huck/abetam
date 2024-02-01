from mesa.batchrunner import batch_run
from components.model import TechnologyAdoptionModel
from components.probability import beta_with_mode_at
from components.technologies import merge_heating_techs_with_share
from data.canada import nrcan_tech_shares_df

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from batch import transform_dict_column
import seaborn as sns
from datetime import datetime

province = "Ontario"
start_year = 2000

heat_techs_df = merge_heating_techs_with_share(start_year=start_year, province=province)
historic_tech_shares = nrcan_tech_shares_df.copy()
historic_tech_shares.index = historic_tech_shares.index.swaplevel()
h_tech_shares = historic_tech_shares.loc[province, :] / 100

techs = heat_techs_df.index.to_list()
tech_mode_map = {
    "Electric furnace": 0.36274795785719943,
    "Gas furnace": 0.47626887794633843,
    "Heat pump": 0.60884054526341,
    "Oil furnace": 0.1559770459529957,
    "Wood or wood pellets furnace": 0.3777387473412798,
}
# dict(zip(techs, [0.5] * len(techs)))

batch_parameters = {
    "N": [300, 500],
    "province": [province],  # , "Alberta", "Ontario"],
    "random_seed": range(20, 28),
    "start_year": start_year,
    "tech_attitude_dist_func": [beta_with_mode_at],
    "tech_attitude_dist_params": [tech_mode_map],
    "n_segregation_steps": [60],
    "interact": [False],
}

# fit the mcda weights
adoption_dfs = []
for p_mode in np.linspace(0.6,0.85,10):
    batch_parameters["price_weight_mode"] = p_mode

    results = batch_run(
        TechnologyAdoptionModel,
        batch_parameters,
        number_processes=None,
        max_steps=80,
        data_collection_period=1,
    )
    df = pd.DataFrame(results)
    df_no_dict, columns = transform_dict_column(df, dict_col_name="Technology shares")
    df2plot = df_no_dict[["RunId", "Step", *columns]].drop_duplicates()
    df2plot = df2plot.melt(id_vars=["RunId", "Step"]).pivot(
        columns=["variable", "RunId"], index="Step", values="value"
    )

    mean_df = pd.DataFrame()
    for col in df2plot.columns.get_level_values(0).unique():
        mean_df.loc[:, col] = df2plot[col].mean(axis=1)

    mean_df.index = TechnologyAdoptionModel.steps_to_years_static(
        start_year, range(81), 1 / 4
    )
    mean_df["p_mode"] = p_mode
    adoption_dfs.append(mean_df)
    diff_sum = (h_tech_shares - mean_df).sum()

    total_abs_diff = diff_sum.abs().sum()
    print(p_mode, total_abs_diff)
    # print(f"finished iteration {i}")

l_hist_shares = (
    historic_tech_shares.loc[province, :].melt(ignore_index=False).reset_index()
)
l_hist_shares["iteration"] = "historic"
l_hist_shares["value"] *= 0.01


def parameter_fit_results(dfs: list[pd.DataFrame], second_id_var="iteration"):
    results = pd.concat(dfs)
    results.reset_index(names=["year"], inplace=True)
    long_results = results.melt(id_vars=["year", second_id_var])
    return long_results


mcda_fit_results = pd.concat(adoption_dfs)


def update_trace_opacity(trace: go.Trace):
    # TODO: add this variable in loop above
    n_fit_iterations = 10
    iteration = trace.name.split(",")[-1]
    if iteration == " historic":
        opacity = 1
        trace.line.width = 3
        trace.line.dash = "solid"

    else:
        try:
            opacity = int(iteration) * 1 / n_fit_iterations
        except:
            opacity = float(iteration.strip())
    trace.opacity = opacity


pfit_res = parameter_fit_results(adoption_dfs, second_id_var="p_mode")
pfit_res_historic = pd.concat([pfit_res, l_hist_shares])
pfit_res_historic["p_mode"][pfit_res_historic["p_mode"].isna()] = "historic"
# pfit_res_historic.sort_values(by="p_mode", inplace=True)
fig = px.line(
    pfit_res_historic,
    x="year",
    y="value",
    color="variable",
    line_dash="p_mode",
    template="plotly",
)

fig.for_each_trace(lambda t: update_trace_opacity(t))

fig.update_layout(width=900)
fig.write_html(f"param_weight_fit_{datetime.now():%Y%m%d-%H-%M}.html")
