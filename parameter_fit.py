from components.technologies import merge_heating_techs_with_share
from data.canada import nrcan_tech_shares_df

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from batch import BatchResult
import seaborn as sns
from datetime import datetime
from multiprocessing.pool import ThreadPool

print("starting parameter fit")

province = "Ontario"
start_year = 2000

heat_techs_df = merge_heating_techs_with_share(start_year=start_year, province=province)
historic_tech_shares = nrcan_tech_shares_df.copy()
historic_tech_shares.index = historic_tech_shares.index.swaplevel()
h_tech_shares = historic_tech_shares.loc[province, :] / 100


n_steps = 80


def parameter_fit_results(dfs: list[pd.DataFrame], second_id_var="iteration"):
    results = pd.concat(dfs)
    # results.reset_index(names=["year"], inplace=True)
    long_results = results.melt(
        id_vars=[second_id_var], ignore_index=False
    ).reset_index()
    return long_results


def update_trace_opacity(trace: go.Trace):
    iteration = trace.name.split(",")[-1]
    if iteration == " historic":
        opacity = 1
        trace.width = 3

    else:
        try:
            opacity = int(iteration) * 1 / n_fit_iterations
        except:
            pass
            opacity = float(iteration.strip())

    trace.opacity = opacity


def comparison_plot(mean_df):
    historic_tech_shares = nrcan_tech_shares_df.copy()
    historic_tech_shares.index = historic_tech_shares.index.swaplevel()
    h_tech_shares = historic_tech_shares.loc[province, :] / 100

    h_tech_shares_long = h_tech_shares.melt(ignore_index=False)
    h_tech_shares_long["comparison"] = "historic"

    mean_df.long = mean_df.melt(ignore_index=False)
    mean_df.long["comparison"] = "modelled"

    comp_df = pd.concat([h_tech_shares_long, mean_df.long])
    ax = sns.lineplot(
        comp_df.reset_index(), x="index", hue="variable", y="value", style="comparison"
    )
    ax.legend(loc=(1, 0.25))
    return ax.get_figure()


def get_adoption_details_from_batch_results(model_vars_df):
    adoption_detail = model_vars_df[["Step", "RunId", "Adoption details", "AgentID"]]
    adoption_detail.loc[:, ["tech", "reason"]] = pd.DataFrame.from_records(
        adoption_detail["Adoption details"].values
    )
    adoption_detail = adoption_detail.drop("Adoption details", axis=1)
    adoption_detail["amount"] = 1
    drop_rows = adoption_detail["tech"].apply(lambda x: x is None)
    adoption_detail = adoption_detail.loc[~drop_rows, :]

    adoption_detail = (
        adoption_detail.groupby(["Step", "RunId", "tech", "reason"]).sum().reset_index()
    )

    # get cumulative sum
    adoption_detail["cumulative_amount"] = adoption_detail.groupby(
        ["RunId", "tech", "reason"]
    ).cumsum()["amount"]
    return adoption_detail


def fit_attitudes(gut, p_mode, province, att_mode_table: pd.DataFrame, n_fit_iterations=12):
    batch_parameters = {
        "N": [700],
        "province": [province],
        "random_seed": range(20, 26),
        "start_year": 2000,
        "tech_att_mode_table": [h_tech_shares.copy()],
        "n_segregation_steps": [60],
        "interact": [False],
        "global_util_thresh": [gut],
        "price_weight_mode": [p_mode],
    }
    adoption_share_dfs = []
    scale = 2.5
    best_abs_diff = 1e12
    att_mode_tables = []
    best_modes = att_mode_table.copy()
    full_years = range(2000, 2021)
    for i in range(n_fit_iterations):
        b_result = BatchResult.from_parameters(batch_parameters, display_progress=False)
        model_shares = (
            b_result.tech_shares_df.groupby(["province", "year"])
            .mean()
            .drop("RunId", axis=1)
        )
        del b_result
        diff = (h_tech_shares - model_shares.loc[(province, full_years), :]).loc[
            province, :
        ]

        tech_share_abs_diff = diff.abs().sum()
        current_abs_diff = tech_share_abs_diff.sum()
        print(i, current_abs_diff)

        # if current is not smallest diff
        if best_abs_diff <= current_abs_diff:
            scale *= 0.7
            print(gut, p_mode, i, f"Performance degradation. Scaled down {scale=}")
        else:
            # current iteration is the best. store values
            best_abs_diff = current_abs_diff
            best_modes = att_mode_table.copy()

        att_update = diff * scale
        att_mode_table = best_modes + att_update

        # adjust modes to where distributions are sensible
        att_mode_table[att_mode_table < 0.05] = 0.05
        att_mode_table[att_mode_table > 0.95] = 0.95

        protocol_table = att_mode_table.copy()
        protocol_table["iteration"] = i
        protocol_table["p_mode"] = p_mode
        protocol_table["gut"] = gut
        att_mode_tables.append(protocol_table)

        model_shares["iteration"] = i
        adoption_share_dfs.append(model_shares)

        print(gut, p_mode, i, diff.abs().sum())
        batch_parameters["tech_att_mode_table"] = [att_mode_table]

    fitted_tech_shares = parameter_fit_results(adoption_share_dfs)
    fitted_tech_shares["gut"] = gut
    fitted_tech_shares["p_mode"] = p_mode
    fitted_tech_shares["province"] = province

    best_modes["best_abs_diff"] = best_abs_diff
    best_modes["province"] = province
    best_modes["gut"] = gut
    best_modes["p_mode"] = p_mode

    # run the model for the future
    batch_parameters["start_year"] = 2020
    bResult = BatchResult.from_parameters(
        batch_parameters, max_steps=(2050 - 2020) * 4, force_rerun=True
    )
    shares_df = bResult.tech_shares_df
    shares_df["gut"] = gut
    shares_df["p_mode"] = p_mode
    shares_df["province"] = province

    all_att_modes = pd.concat(att_mode_tables)
    all_att_modes["province"] = province
    return shares_df, fitted_tech_shares, all_att_modes, best_modes


future_tech_shares = []
historic_tech_shares = []
fitting_att_mode_tables = []
best_modes = []

techs = heat_techs_df.index.to_list()

att_mode_table = h_tech_shares.copy()

now = f"{datetime.now():%Y.%m.%d-%H.%M}"
results_dir = Path(f"results/fitting/{now}")
results_dir.mkdir(exist_ok=True, parents=True)

# remove projections from input data
tech_params = pd.read_csv("data/canada/heat_tech_params.csv").query("year < 2023").set_index(["variable","year"])
tech_params.loc["specific_cost","Heat pump"] = (tech_params.loc["specific_cost","Heat pump"]*(1-0.2)).values
tech_params.swaplevel().reset_index().to_csv("data/canada/heat_tech_params.csv", index=False)


with ThreadPool(6) as pool:
    jobs = []
    for province in ["Ontario"]:#,"Alberta", "British Columbia"]:
        for gut in np.arange(0.2,0.8, 0.05):
            for p_mode in np.arange(0.2,0.8, 0.05):  # , 0.5, 0.6, 0.7]:
                print("appending job for", province, gut, p_mode)
                jobs.append(
                    pool.apply_async(fit_attitudes, (gut, p_mode, province, h_tech_shares.copy()))
                )
    for job in jobs:
        result = job.get()
        future_tech_shares.append(result[0])
        historic_tech_shares.append(result[1])
        fitting_att_mode_tables.append(result[2])
        best_modes.append(result[3])

all_future_tech_shares = pd.concat(future_tech_shares)
all_future_tech_shares.to_csv(f"{results_dir}/all_future_tech_shares_{datetime.now():%Y%m%d-%H-%M}.csv")
all_historic_tech_shares = pd.concat(historic_tech_shares)
all_historic_tech_shares.to_csv(f"{results_dir}/all_historic_tech_shares_{datetime.now():%Y%m%d-%H-%M}.csv")
all_best_modes = pd.concat(best_modes)
all_best_modes.to_csv(f"{results_dir}/all_best_modes_{datetime.now():%Y%m%d-%H-%M}.csv")


all_attitude_modes = pd.concat(fitting_att_mode_tables)
all_attitude_modes = all_attitude_modes.melt(
    id_vars=["iteration", "gut", "p_mode"], ignore_index=False
).reset_index()
all_attitude_modes.to_csv(
    f"{results_dir}/all_attitude_modes_{datetime.now():%Y%m%d-%H-%M}.csv"
)
