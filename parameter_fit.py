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


province = "Ontario"
start_year = 2000

heat_techs_df = merge_heating_techs_with_share(start_year=start_year, province=province)
historic_tech_shares = nrcan_tech_shares_df.copy()
historic_tech_shares.index = historic_tech_shares.index.swaplevel()
h_tech_shares = historic_tech_shares.loc[province, :] / 100


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


def fit_attitudes(
    p_mode,
    peer_eff,
    att_mode_table: pd.DataFrame,
    province="Ontario",
    N=500,
    n_fit_iterations=40,
    ts_step_length="W",
):
    batch_parameters = {
        "N": [N],
        "province": [province],
        "random_seed": range(20, 25),
        "start_year": 2000,
        "tech_att_mode_table": [att_mode_table],
        "n_segregation_steps": [40],
        "interact": [False],
        "price_weight_mode": [p_mode],
        "ts_step_length": ["W"],
        "peer_effect_weight": [peer_eff],
    }
    adoption_share_dfs = []
    starting_scale = pd.DataFrame(
        np.ones((21, 5)) * 0.5, index=range(2000, 2021), columns=att_mode_table.columns
    )
    scale = starting_scale.copy()
    growing_scale = scale.copy()
    shrinking_scale = scale.copy()
    shrink = True # decrease scale if true, grow scale otherwise
    rescaled = False
    iterations_wo_impro = 0
    best_abs_diff_sum = 1e12
    att_mode_tables = []
    best_modes = att_mode_table.copy()
    last_diff = None
    best_abs_diff_ts = pd.DataFrame(
        np.ones((21, 5)) * 2, index=range(2000, 2021), columns=att_mode_table.columns
    )
    for i in range(n_fit_iterations):
        b_result = BatchResult.from_parameters(batch_parameters, display_progress=False)
        model_shares = (
            b_result.tech_shares_df.groupby(["province", "year"])
            .mean()
            .drop("RunId", axis=1)
        )
        diff = (h_tech_shares - model_shares.loc[(province, range(2000, 2021)), :]).loc[
            province, :
        ]

        current_abs_diff_ts = diff.abs()
        current_abs_diff_sum = diff.abs().sum().sum()

        # attitude modes are updated as follows:
        # 1. Annual modes where an improvement was registered are stored in best_modes
        # 2. based on the difference btwn historic and modelled tech shares,
        #    an update to the modes is calculated
        improved_rows = current_abs_diff_ts.sum(axis=1) < best_abs_diff_ts.sum(axis=1)

        print(
            f"{p_mode=:.2f}, {peer_eff=:.2f}, {i=:02}, "
            f"{current_abs_diff_sum=:.3f}, {best_abs_diff_sum=:.3f}, "
            # f"{improved_rows.sum()=}, "
            f"{shrink=}. scale {scale.mean().mean()}, "
            f"worsening iterations:{iterations_wo_impro}, "
            f"diff=\n{diff.abs().sum()}"
        )
        best_abs_diff_ts[improved_rows] = current_abs_diff_ts[improved_rows]
        
        
        if current_abs_diff_sum >= best_abs_diff_sum:
            # no improvement, change scale
            iterations_wo_impro +=1
            if iterations_wo_impro > 3:
                if shrink:
                    shrinking_scale *= 0.7
                    scale = shrinking_scale
                    shrink = False
                else:
                    growing_scale /= 0.7
                    scale = growing_scale
                    shrink = True

            if iterations_wo_impro > 10 and not rescaled:
                print("no improvement for too long. Restarting scaling.")
                shrinking_scale = starting_scale.copy()
                growing_scale = starting_scale.copy()
                scale = np.random.random(starting_scale.shape)
                rescaled = True

                # print(f"\tPerformance degradation {shrink=}. New {scale.mean().mean()=}")
            else:
                # less than 3 iterations without improvement
                # do nothing
                pass
            # diff = last_diff
        else:
            # current iteration is the best. store values
            iterations_wo_impro = 0
            rescaled = False
            shrinking_scale = starting_scale.copy()
            growing_scale = starting_scale.copy()
            
            best_modes = att_mode_table.copy()
            best_abs_diff_sum = current_abs_diff_sum
            # last_diff = diff

        att_mode_table += diff * (scale * diff.abs().sum()/diff.abs().sum().max())
        assert best_modes.isna().sum().sum() == 0, AssertionError(
            f"{best_modes.isna()=}"
        )
        assert att_mode_table.isna().sum().sum() == 0

        # adjust modes to where distributions are sensible
        att_mode_table[att_mode_table < 0.05] = 0.05
        att_mode_table[att_mode_table > 0.95] = 0.95

        protocol_table = att_mode_table.copy()
        protocol_table["iteration"] = i
        protocol_table["p_mode"] = p_mode
        protocol_table["peer_eff"] = peer_eff
        att_mode_tables.append(protocol_table)

        model_shares["iteration"] = i
        adoption_share_dfs.append(model_shares)

        batch_parameters["tech_att_mode_table"] = [att_mode_table]

    print(f"{datetime.now():%Y.%m.%d-%H.%M}")
    fitted_tech_shares = parameter_fit_results(adoption_share_dfs)
    fitted_tech_shares["peer_eff"] = peer_eff
    fitted_tech_shares["p_mode"] = p_mode
    fitted_tech_shares["province"] = province

    best_modes["best_abs_diff_sum"] = best_abs_diff_sum
    best_modes["province"] = province
    best_modes["peer_eff"] = peer_eff
    best_modes["p_mode"] = p_mode

    # run the model for the future
    batch_parameters["start_year"] = 2020
    bResult = BatchResult.from_parameters(
        batch_parameters, max_steps=(2050 - 2020) * 4, force_rerun=True
    )
    shares_df = bResult.tech_shares_df
    shares_df["peer_eff"] = peer_eff
    shares_df["p_mode"] = p_mode
    shares_df["province"] = province

    all_att_modes = pd.concat(att_mode_tables)
    all_att_modes["province"] = province
    return shares_df, fitted_tech_shares, all_att_modes, best_modes


if __name__ == "__main__":
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
    tech_params = (
        pd.read_csv("data/canada/heat_tech_params.csv")
        .query("year < 2023")
        .set_index(["variable", "year"])
    )
    # account for likely subsidies in the period
    tech_params.loc["specific_cost", "Heat pump"] = (
        tech_params.loc["specific_cost", "Heat pump"] * (1 - 0.2)
    ).values
    tech_params.swaplevel().reset_index().to_csv(
        "data/canada/heat_tech_params.csv", index=False
    )

    start_fit_atts = pd.read_csv("results/fitting/start_fit_atts.csv",index_col=0)

    with ThreadPool(5) as pool:
        jobs = []
        for province in ["Ontario"]:
            for p_mode in np.arange(0.6, 0.8, 0.05):
                for peer_eff in [0.15, 0.2, 0.25, 0.3, 0.35]:
                    print("appending job for", province, f"{p_mode=}", f"{peer_eff=}")
                    jobs.append(
                        pool.apply_async(
                            fit_attitudes, (p_mode, peer_eff, start_fit_atts)
                        )
                    )
        for job in jobs:
            result = job.get()
            future_tech_shares.append(result[0])
            historic_tech_shares.append(result[1])
            fitting_att_mode_tables.append(result[2])
            best_modes.append(result[3])

        all_future_tech_shares = pd.concat(future_tech_shares)
        all_future_tech_shares.to_csv(
            f"{results_dir}/all_future_tech_shares_{datetime.now():%Y%m%d-%H-%M}.csv"
        )
        all_historic_tech_shares = pd.concat(historic_tech_shares)
        all_historic_tech_shares.to_csv(
            f"{results_dir}/all_historic_tech_shares_{datetime.now():%Y%m%d-%H-%M}.csv"
        )
        all_best_modes = pd.concat(best_modes)
        all_best_modes.to_csv(
            f"{results_dir}/all_best_modes_{datetime.now():%Y%m%d-%H-%M}.csv"
        )

    all_attitude_modes = pd.concat(fitting_att_mode_tables)
    all_attitude_modes = all_attitude_modes.melt(
        id_vars=["iteration", "p_mode", "peer_eff"], ignore_index=False
    ).reset_index()
    all_attitude_modes.to_csv(
        f"{results_dir}/all_attitude_modes_{datetime.now():%Y%m%d-%H-%M}.csv"
    )
