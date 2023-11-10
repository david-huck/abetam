from mesa.batchrunner import batch_run
from components.model import TechnologyAdoptionModel
from components.technologies import merge_heating_techs_with_share
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path


def transform_dict_column(df, dict_col_name="Technology shares", return_cols=True):
    if dict_col_name not in df.columns:
        return df, None
    adoption_col = df[dict_col_name].to_list()
    adoption_df = pd.DataFrame.from_records(adoption_col)
    df.loc[:, adoption_df.columns] = adoption_df
    if return_cols:
        return df.drop(dict_col_name, axis=1), adoption_df.columns
    else:
        return df.drop(dict_col_name, axis=1)


def transform_dataframe_for_plotly(df, columns, boundary_method="ci", ci=0.95):
    """Transforms the dataframe `df` from the batch run. It will be reshaped, to
      calculate uncertainty boundaries for each column across all `RunId`s.

    Args:
        df (pd.DataFrame): DataFrame from the batch run
        columns (list): columns for which the transformation should take place.
            Usually the adoption of technologies is looked at.
        boundary_method (str, optional): The method used to derive an upper and a lower
            boundary per column. Defaults to "ci".
        ci (float, optional): The confidence interval to be determined if
            `boundary_method` is "ci". Defaults to 0.95.

    Returns:
        df: Dataframe with MultiIndex. Access like df.loc[:,(`col`,`line`)] where `col`
            must be in `columns` and `line` must be in ("lower","upper", "mean")
    """

    df2plot = df[["RunId", "Step", *columns]].drop_duplicates()
    df2plot = df2plot.melt(id_vars=["RunId", "Step"]).pivot(
        columns=["variable", "RunId"], index="Step", values="value"
    )

    plotly_df = pd.DataFrame(
        columns=pd.MultiIndex.from_product([columns, ["lower", "upper", "mean"]])
    )

    for col in columns:
        if boundary_method == "minmax":
            plotly_df.loc[:, (col, "lower")] = df2plot.loc[:, col].min(axis=1)
            plotly_df.loc[:, (col, "upper")] = df2plot.loc[:, col].max(axis=1)
        elif boundary_method == "ci":
            # to have e.g. 95% of values to fall within the range, take the
            # 2.5%ile and the 97.5%ile
            plotly_df.loc[:, (col, "lower")] = df2plot.loc[:, col].quantile(
                (1 - ci) / 2, axis=1
            )
            plotly_df.loc[:, (col, "upper")] = df2plot.loc[:, col].quantile(
                ci + (1 - ci) / 2, axis=1
            )

        plotly_df.loc[:, (col, "mean")] = df2plot.loc[:, col].mean(axis=1)
    return plotly_df


def plotly_lines_with_error(plotly_df, columns, colors: dict = None):
    if colors is None:
        colors = px.colors.qualitative.Pastel1
        colors = dict(zip(columns, colors))
    elif isinstance(colors, list):
        colors = dict(zip(columns, colors))

    idx = plotly_df.index.to_list()
    traces = []
    for col in columns:
        color = colors[col]
        rgb = px.colors.unlabel_rgb(color)
        rgba = f"rgba{(*rgb,0.2)}"

        mean_trace = go.Scatter(
            x=idx,
            y=plotly_df.loc[:, (col, "mean")],
            mode="lines",
            line=dict(color=color),
            name=col,
        )

        fill_x = idx + idx[::-1]
        fill_y = (
            plotly_df.loc[:, (col, "lower")].to_list()
            + plotly_df.loc[:, (col, "upper")][::-1].to_list()
        )
        fill_trace = go.Scatter(
            x=fill_x,
            y=fill_y,
            line=dict(width=0, color=rgba),
            hoverinfo="skip",
            # showlegend=False,
            name=col + " uncertainty",
            fill="toself",
        )

        traces.extend([mean_trace, fill_trace])
    return traces


def adoption_plot_with_quantiles(
    adoption_df, intervals=[0.95], mid_line="median", colors=None
):
    columns = adoption_df.columns.get_level_values(0).unique()

    if colors is None:
        colors = px.colors.qualitative.Pastel1
        colors = dict(zip(columns, colors))
    elif isinstance(colors, list):
        colors = dict(zip(columns, colors))

    idx = adoption_df.index.to_list()
    plotly_df = pd.DataFrame(
        columns=pd.MultiIndex.from_product([columns, ["lower", "upper"], intervals])
    )
    traces = []
    for col in columns:
        color = colors[col]
        rgb = px.colors.unlabel_rgb(color)
        median_trace = go.Scatter(
            x=idx,
            y=adoption_df.loc[:, col].median(axis=1),
            mode="lines",
            line=dict(color=color),
            name=col + "_median",
        )
        mean_trace = go.Scatter(
            x=idx,
            y=adoption_df.loc[:, col].mean(axis=1),
            mode="lines",
            line=dict(
                color=color,
            ),
            name=col + "_mean",
        )
        if mid_line == "mean":
            traces.append(mean_trace)
        elif mid_line == "median":
            traces.append(median_trace)
        elif mid_line == "both":
            mean_trace.line.dash = "dash"
            traces.extend([mean_trace, median_trace])

        for val in intervals:
            plotly_df.loc[:, (col, "lower", val)] = adoption_df.loc[:, col].quantile(
                (1 - val) / 2, axis=1
            )
            plotly_df.loc[:, (col, "upper", val)] = adoption_df.loc[:, col].quantile(
                val + (1 - val) / 2, axis=1
            )
            # greater quantile -> greater transparency
            rgba = f"rgba{(*rgb,1-val)}"

            fill_x = idx + idx[::-1]
            fill_y = (
                plotly_df.loc[:, (col, "lower", val)].to_list()
                + plotly_df.loc[:, (col, "upper", val)][::-1].to_list()
            )
            fill_trace = go.Scatter(
                x=fill_x,
                y=fill_y,
                line=dict(width=0, color=rgba),
                hoverinfo="skip",
                # showlegend=False,
                name=f"{col} {val} interval",
                fill="toself",
            )

            traces.append(fill_trace)

    fig = go.Figure()
    fig.add_traces(traces)
    return fig


def save_batch_parameters(batch_parameters, results_dir):
    results_dir = Path(results_dir)

    if not results_dir.exists():
        results_dir.mkdir(exist_ok=True)

    tech_param_path = None
    for k, v in batch_parameters.items():
        if isinstance(v[0], pd.DataFrame):
            tech_param_path = results_dir.joinpath(k + ".csv")
            break

    tech_df = batch_parameters.pop(k)[0]
    tech_df.to_csv(tech_param_path)

    batch_parameters["heating_techs_df"] = tech_param_path.as_posix()
    with open(results_dir.joinpath("batch_parameters.json"), "w") as fo:
        json.dump(batch_parameters, fo)


def read_batch_parameters(batch_parameter_path):
    with open(batch_parameter_path, "r") as fi:
        batch_parameters = json.load(fi)

    df = None
    for k, v in batch_parameters.items():
        if isinstance(v, str):
            if v.endswith(".csv"):
                df = pd.read_csv(v)
                break
    assert df is not None, AssertionError(
        f"No dataframe was read for {batch_parameter_path}"
    )
    batch_parameters[k] = df
    return batch_parameters



if __name__ == "__main__":
    heat_techs_df = merge_heating_techs_with_share()
    batch_parameters = {
        "N": [200],
        "grid_side_length": [15],
        "heating_techs_df": [heat_techs_df],
        "province": ["Ontario"],  # , "Alberta", "Ontario"],
        "random_seed": list(range(3)),
    }

    # tam = partial(TechnologyAdoptionModel, heat_techs_df)
    results = batch_run(
        TechnologyAdoptionModel,
        batch_parameters,
        number_processes=None,
        max_steps=80,
        data_collection_period=1,
    )

    df = pd.DataFrame(results)
    df_no_dict, columns = transform_dict_column(df, dict_col_name="Technology shares")
    plotly_df = transform_dataframe_for_plotly(df_no_dict, columns)

    result_dir = TechnologyAdoptionModel.get_result_dir("batch")
    save_batch_parameters(batch_parameters, result_dir)
    fig = go.Figure()
    fig.add_traces(plotly_lines_with_error(plotly_df, columns))
    fig.write_html(result_dir.joinpath("adoption_uncertainty.html"))


    # analysis with seaborn is rather straight forward, but takes rather long
    # print(r"creating figure with seaborn (95% ci)")
    # df_4_plot = (
    #     df[["RunId", "Step", *columns]]
    #     .drop_duplicates()
    #     .melt(id_vars=["RunId", "Step"])
    # )
    # ax = sns.lineplot(df_4_plot, x="Step", y="value", hue="variable")
    # ax.get_figure().savefig(result_dir.joinpath("adoption_uncertainty.png"))
