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
from batch import transform_dict_column
import seaborn as sns
from datetime import datetime

province = "Ontario"
start_year = 2000

heat_techs_df = merge_heating_techs_with_share(start_year=start_year, province=province)
historic_tech_shares = nrcan_tech_shares_df.copy()
historic_tech_shares.index = historic_tech_shares.index.swaplevel()
h_tech_shares = historic_tech_shares.loc[province, :] / 100


n_steps = 80


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


# if Path("best_tech_modes.json").exists():
#     tech_mode_map = json.load(open("best_tech_modes.json","r"))
#     print("read", tech_mode_map)
# else:
techs = heat_techs_df.index.to_list()
tech_mode_map = dict(zip(techs, [0.5] * len(techs)))

batch_parameters = {
    "N": [300,500],
    "province": [province],  # , "Alberta", "Ontario"],
    "random_seed": range(20, 28),
    "start_year": start_year,
    "tech_attitude_dist_func": [beta_with_mode_at],
    "tech_attitude_dist_params": [tech_mode_map],
    "n_segregation_steps": [60],
    "interact": [False],
}

adoption_share_dfs = []
adoption_detail_dfs = []

mode_shift = 0.15
best_abs_diff = 1e12
greatest_diff_sum = None
best_modes = pd.Series(tech_mode_map)
att_update = pd.Series(0,index=best_modes.index)

n_fit_iterations = 12

for i in range(n_fit_iterations):
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

    adoption_detail_dfs.append(get_adoption_details_from_batch_results(df))

    mean_df = pd.DataFrame()
    for col in df2plot.columns.get_level_values(0).unique():
        mean_df.loc[:, col] = df2plot[col].mean(axis=1)

    mean_df.index = TechnologyAdoptionModel.steps_to_years_static(
        batch_parameters["start_year"], range(81), 1 / 4
    )
    diff_sum = (h_tech_shares - mean_df).sum()

    # previously, low tech shares have been mildly adapted, because their
    # difference was small
    rel_diff = (h_tech_shares - mean_df).dropna(how="all") / h_tech_shares
    diff_sum = rel_diff.sum() * 0.5 + diff_sum * 0.5

    current_abs_diff = diff_sum.abs().sum()
    print(i, current_abs_diff)

    if greatest_diff_sum is not None:
        if diff_sum.abs().sum() > greatest_diff_sum.abs().sum():
            greatest_diff_sum = diff_sum.copy()
    else:
        greatest_diff_sum = diff_sum.copy()

    if best_abs_diff < current_abs_diff:
        print("Performance degradation. Scaling down mode_shift")
        mode_shift = mode_shift / 2
        att_update = att_update / 2
        tech_mode_map = best_modes.to_dict()
    else:
        best_abs_diff = current_abs_diff
        if "new_modes" in locals():
            best_modes = new_modes.copy()
        att_update = diff_sum / greatest_diff_sum.abs().max() * mode_shift
    new_modes = pd.Series(tech_mode_map) + att_update

    mean_df["iteration"] = i
    adoption_share_dfs.append(mean_df)

    new_modes[new_modes <= 0] = 0.05
    new_modes[new_modes >= 1] = 0.95
    debug_info = pd.concat(
        [diff_sum.rename("diff_sum"), new_modes.rename("new_modes")],
        axis=1,
    )
    print(i, debug_info)
    tech_mode_map = new_modes.to_dict()
    batch_parameters["tech_attitude_dist_params"] = [tech_mode_map]
print(best_modes)




l_hist_shares = historic_tech_shares.loc[province,:].melt(ignore_index=False).reset_index()
l_hist_shares["iteration"] = "historic"
l_hist_shares["value"] *= 0.01

def parameter_fit_results(dfs: list[pd.DataFrame], second_id_var="iteration"):
    results = pd.concat(dfs)
    results.reset_index(names=["year"], inplace=True)
    long_results = results.melt(id_vars=["year",second_id_var])
    return long_results


def update_trace_opacity(trace: go.Trace):
    iteration = trace.name.split(",")[-1]
    if iteration == " historic":
        opacity = 1
        trace.width = 3
        
    else:
        try:
            opacity = int(iteration) * 1/n_fit_iterations
        except:
            pass
            opacity = float(iteration.strip())


    trace.opacity = opacity

pfit_res = parameter_fit_results(adoption_share_dfs)
pfit_res_historic = pd.concat( [pfit_res, l_hist_shares])

fig = px.line(pfit_res, x="year", y="value", color="variable", line_dash="iteration", template="plotly", )

fig.for_each_trace(lambda t: update_trace_opacity(t))

for i,tech in enumerate(historic_tech_shares.loc[province,:].columns):
    fig.add_trace(
        go.Scatter(
            x=historic_tech_shares.loc[province,tech].index,
            y=historic_tech_shares.loc[province,tech].values/100,
            mode="lines",
            name=f"{tech}, historic",
            line=dict(dash="solid", width=3, color=px.colors.qualitative.Plotly[i]),
        )
    )

fig.update_layout(width=900)
fig.write_html(f"param_fit_{datetime.now():%Y%m%d-%H-%m}.html")


best_modes = best_modes.to_dict()
best_modes["best_abs_diff"] = best_abs_diff
best_modes["province"] = province
json.dump(best_modes, open(f"best_modes_{datetime.now():%Y%m%d-%H-%m}.json","w"))