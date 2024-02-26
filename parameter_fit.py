from mesa.batchrunner import batch_run
from components.model import TechnologyAdoptionModel
from components.probability import beta_with_mode_at
from components.technologies import merge_heating_techs_with_share
from data.canada import nrcan_tech_shares_df

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from batch import transform_dict_column, BatchResult
import seaborn as sns
from datetime import datetime

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
    long_results = results.melt(id_vars=[second_id_var], ignore_index=False).reset_index()
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


techs = heat_techs_df.index.to_list()

att_mode_table = h_tech_shares.copy()
att_mode_tables = []
full_years = range(2000, 2021)
n_fit_iterations = 12

now = f"{datetime.now():%Y.%m.%d-%H.%M}"
results_dir = Path(f"results/fitting/{now}")
results_dir.mkdir(exist_ok=True, parents=True)

for gut in [0.2, 0.25, 0.3, 0.35, 0.4]:#, 0.6, 0.7, 0.8]:
    for p_mode in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:#, 0.5, 0.6, 0.7]:
        batch_parameters = {
            "N": [700],
            "province": [province],  # , "Alberta", "Ontario"],
            "random_seed": range(20, 26),
            "start_year": 2000,
            "tech_att_mode_table": [h_tech_shares.copy()],
            "n_segregation_steps": [60],
            "interact": [False],
            "global_util_thresh": [gut],
            "price_weight_mode": [p_mode]
        }
        print("price weight mode:", p_mode)
        adoption_share_dfs = []
        adoption_detail_dfs = []
        scale = 2.5
        best_abs_diff = 1e12
        greatest_diff_sum = None
        best_modes = att_mode_table.copy()
        for i in range(n_fit_iterations):
            b_result = BatchResult.from_parameters(batch_parameters)
            model_shares = (
                b_result.tech_shares_df.groupby(["province", "year"])
                .mean()
                .drop("RunId", axis=1)
            )
            diff = (h_tech_shares - model_shares.loc[(province, full_years), :]).loc[
                province, :
            ]

            tech_share_abs_diff = diff.abs().sum()
            current_abs_diff = tech_share_abs_diff.sum()
            print(i, current_abs_diff)

            # if current is not smallest diff
            if best_abs_diff <= current_abs_diff:
                scale *= 0.7
                print(f"Performance degradation. Scaled down {scale=}")
            else:
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

            print(i, diff.abs().sum())
            batch_parameters["tech_att_mode_table"] = [att_mode_table]
        # print(best_modes)

        l_hist_shares = (
            historic_tech_shares.loc[province, :].melt(ignore_index=False).reset_index()
        )
        l_hist_shares["iteration"] = "historic"
        l_hist_shares["value"] *= 0.01

        pfit_res = parameter_fit_results(adoption_share_dfs)
        pfit_res_historic = pd.concat([pfit_res, l_hist_shares])

        fig = px.line(
            pfit_res,
            x="year",
            y="value",
            color="variable",
            line_dash="iteration",
            template="plotly",
        )

        fig.for_each_trace(lambda t: update_trace_opacity(t))

        for i, tech in enumerate(historic_tech_shares.loc[province, :].columns):
            fig.add_trace(
                go.Scatter(
                    x=historic_tech_shares.loc[province, tech].index,
                    y=historic_tech_shares.loc[province, tech].values / 100,
                    mode="lines",
                    name=f"{tech}, historic",
                    line=dict(
                        dash="solid", width=3, color=px.colors.qualitative.Plotly[i]
                    ),
                )
            )

        fig.update_layout(width=900, title=f"Price mode: {p_mode}, GUT: {gut}")
        fig.write_html(
            f"{results_dir}/ntpb_par_fit_{datetime.now():%Y%m%d-%H-%M}_pmode_{p_mode}_gut_{gut}.html"
        )

        print("written", f"{results_dir}/par_fit_{datetime.now():%Y%m%d-%H-%M}_pmode_{p_mode}.html")

        # best_modes = best_modes.to_dict()
        best_modes_dict = best_modes.to_dict()
        best_modes_dict["best_abs_diff"] = best_abs_diff
        best_modes_dict["province"] = [province]
        json.dump(
            best_modes_dict,
            open(
                f"{results_dir}/ntpb_best_modes_{datetime.now():%Y%m%d-%H-%M}_pmode_{p_mode}_gut_{gut}.json",
                "w",
            ),
        )

        batch_parameters["start_year"] = 2020
        
        bResult = BatchResult.from_parameters(
            batch_parameters, max_steps=(2050 - 2020) * 4, force_rerun=True
        )
        shares_fig = bResult.tech_shares_fig()
        shares_fig.figure.savefig(
            f"{results_dir}/n_tbp_future_tech_shares_{datetime.now():%Y%m%d-%H-%M}_pmode_{p_mode}_gut_{gut}.png"
        )


all_attitude_modes = pd.concat(att_mode_tables)
all_attitude_modes = all_attitude_modes.melt(
    id_vars=["iteration", "gut", "p_mode"], ignore_index=False
).reset_index()
all_attitude_modes.to_csv(f"{results_dir}/all_attitude_modes_{datetime.now():%Y%m%d-%H-%M}.csv")
