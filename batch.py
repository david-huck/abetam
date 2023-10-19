from mesa.batchrunner import batch_run
from components.model import TechnologyAdoptionModel
from components.technologies import merge_heating_techs_with_share
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def transform_dict_column(df, dict_col_name="Technology shares"):
    if dict_col_name not in df.columns:
        return df, None
    adoption_col = df[dict_col_name].to_list()
    adoption_df = pd.DataFrame.from_records(adoption_col)
    df.loc[:, adoption_df.columns] = adoption_df
    return df.drop(dict_col_name, axis=1), adoption_df.columns


def transform_dataframe_for_plotly(df, columns):
    df2plot = df[["RunId", "Step", *columns]].drop_duplicates()
    df2plot = df2plot.melt(id_vars=["RunId", "Step"]).pivot(
        columns=["variable", "RunId"], index="Step", values="value"
    )

    plotly_df = pd.DataFrame(
        columns=pd.MultiIndex.from_product([columns, ["min", "max", "mean"]])
    )
    for col in columns:
        plotly_df.loc[:, (col, "min")] = df2plot.loc[:, col].min(axis=1)
        plotly_df.loc[:, (col, "max")] = df2plot.loc[:, col].max(axis=1)
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
            plotly_df.loc[:, (col, "min")].to_list()
            + plotly_df.loc[:, (col, "max")][::-1].to_list()
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


if __name__ == "__main__":
    heat_techs_df = merge_heating_techs_with_share()
    batch_parameters = {
        "N": [200],
        "width": [20],
        "height": [20],
        "heating_techs_df": [heat_techs_df],
        "province": ["British Columbia"],  # , "Alberta", "Ontario"],
        "random_seed": range(5),
    }

    # tam = partial(TechnologyAdoptionModel, heat_techs_df)
    results = batch_run(
        TechnologyAdoptionModel, batch_parameters, number_processes=None, max_steps=500
    )

    # analysis with seaborn is rather straight forward
    # import seaborn as sns
    # df_4_plot = df[["RunId","Step",*adoption_df.columns]].drop_duplicates().melt(id_vars=["RunId","Step"])
    # sns.lineplot(df_4_plot, x="Step",y="value", hue="variable")

    df = pd.DataFrame(results)
    df_no_dict, columns = transform_dict_column(df, dict_col_name="Technology shares")
    plotly_df = transform_dataframe_for_plotly(df_no_dict, columns)

    fig = go.Figure()

    fig.add_traces(plotly_lines_with_error(plotly_df, columns))
