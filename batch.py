import seaborn as sns
from components.model import TechnologyAdoptionModel
from config import START_YEAR, STEPS_PER_YEAR
from mesa.batchrunner import batch_run
from components.technologies import merge_heating_techs_with_share, Technologies
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable


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

    for k, v in batch_parameters.items():
        if isinstance(v, range):
            batch_parameters[k] = list(v)

    if not results_dir.exists():
        results_dir.mkdir(exist_ok=True)

    with open(results_dir.joinpath("batch_parameters.json"), "w") as fo:
        json.dump(batch_parameters, fo)


def read_batch_parameters(batch_parameter_path):
    with open(batch_parameter_path, "r") as fi:
        batch_parameters = json.load(fi)
    return batch_parameters


@dataclass
class BatchResult:
    def __init__(self, path, batch_parameters, data=None, results_df=None):
        self.path = path
        self.batch_params = batch_parameters
        
        if data is not None:
            self.data = data
            self.results_df = pd.DataFrame(self.data)
        elif results_df is not None:
            self.results_df = results_df

        def convert_steps_to_years(steps):
            years = TechnologyAdoptionModel.steps_to_years_static(
                START_YEAR, steps, STEPS_PER_YEAR
            )
            return years

        self.results_df["year"] = convert_steps_to_years(self.results_df["Step"])

    @classmethod
    def from_directory(cls, directory):
        path = Path(directory)
        batch_parameters = read_batch_parameters(path.joinpath("batch_parameters.json"))
        
        result_data = pd.read_pickle(path.joinpath("batch_results.pkl"))

        result = BatchResult(path, batch_parameters, results_df=result_data)

        for csv in path.glob("*.csv"):
            df = pd.read_csv(csv)
            setattr(result, "_"+csv.stem, df.copy())

        return result

    def save(self):

        for df_name in ["tech_shares_df","adoption_details_df","attitudes_df"]:
            df = getattr(self, df_name)
            df.to_csv(self.path.joinpath(df_name+".csv"), index=False)

        # ensure no iterables in columns are saved, and drop columns that hold no information
        iterable_cols = []
        redundant_cols = []
        for col in self.results_df.columns:
            sample_val = self.results_df[col][0]
            if isinstance(sample_val, Iterable) and sample_val is not str:
                iterable_cols.append(col)
            elif len(self.results_df[col].unique()) < 2:
                redundant_cols.append(col)
        drop_cols = iterable_cols + redundant_cols

        self.results_df.drop(drop_cols, axis=1).to_pickle(self.path.joinpath("batch_results.pkl"))
        save_batch_parameters(self.batch_params, self.path)
        return True

    @property
    def tech_shares_df(self) -> pd.DataFrame:
        if hasattr(self,"_tech_shares_df"):
            return self._tech_shares_df
        
        shares = self.results_df[
            ["RunId", "province", "year", "Technology shares"]
        ]
        data = pd.DataFrame.from_records(shares["Technology shares"].values)
        shares.loc[:, data.columns] = data
        shares = shares.drop(columns=["Technology shares"])
        shares = shares.drop_duplicates()
        self.results_df.drop(columns=["Technology shares"], axis=1)
        self._tech_shares_df = shares.copy()
        return shares


    def tech_shares_fig(self, show_legend=True):
        shares = self.tech_shares_df

        shares_long = shares.melt(id_vars=["RunId", "year", "province"])
        shares_long.head()
        shares_long["value"] *= 100
        ax = sns.relplot(
            shares_long,
            kind="line",
            x="year",
            y="value",
            hue="variable",
            col="province",
        )
        ax.set_ylabels("Tech share (%)")
        ax.set_xticklabels(rotation=30)
        if not show_legend:
            ax.legend.remove()
        return ax

    @property
    def adoption_details_df(self):
        if hasattr(self, "_adoption_details_df"):
            return self._adoption_details_df


        adoption_detail = self.results_df[
            ["RunId", "year", "AgentID", "Adoption details"]
        ]
        adoption_detail.loc[:, ["tech", "reason"]] = pd.DataFrame.from_records(
            adoption_detail["Adoption details"].values
        )
        adoption_detail = adoption_detail.drop("Adoption details", axis=1)
        adoption_detail["amount"] = 1
        drop_rows = adoption_detail["tech"].apply(lambda x: x is None)
        adoption_detail = adoption_detail.loc[~drop_rows, :].reset_index(drop=True)
        if isinstance(adoption_detail["tech"][0], Technologies):
            adoption_detail["tech"] = adoption_detail["tech"].apply(lambda x: x.value)


        adoption_detail = (
            adoption_detail.groupby(["year", "RunId", "tech", "reason"])
            .sum()
            .reset_index()
        )

        # get cumulative sum
        adoption_detail["cumulative_amount"] = adoption_detail.groupby(
            ["RunId", "tech", "reason"]
        ).cumsum()["amount"]
        
        self.results_df.drop("Adoption details", axis=1, inplace=True)
        self._adoption_details_df = adoption_detail.copy()
        return adoption_detail
    
    
    def adoption_details_fig(self):
        adoption_detail = self.adoption_details_df

        ax = sns.relplot(
            adoption_detail,
            kind="line",
            x="year",
            y="cumulative_amount",
            hue="tech",
            col="reason",
        )
        ax.set_xticklabels(rotation=45)
        return ax

    @property
    def attitudes_df(self) -> pd.DataFrame:
        if hasattr(self, "_attitudes_df"):
            return self._attitudes_df
        
        atts_df = self.results_df[["RunId", "year", "Attitudes"]]

        data = atts_df["Attitudes"].to_list()
        data = pd.DataFrame(data)
        atts_df.loc[:, data.columns] = data

        atts_df = atts_df.drop("Attitudes", axis=1)
        self.results_df.drop("Attitudes",axis=1, inplace=True)
        self._attitudes_df = atts_df.copy()
        return atts_df


    def attitudes_fig(self):
        atts_df = self.attitudes_df
        
        atts_long = atts_df.melt(id_vars=("RunId", "year"))
        ax = sns.relplot(atts_long, kind="line", x="year", y="value", hue="variable")
        ax.set_ylabels("Attitude towards technologies (-)")
        ax.set_xticklabels(rotation=45)
        # ax.set_title("Technology attitues over time.")
        return ax


if __name__ == "__main__":
    heat_techs_df = merge_heating_techs_with_share()
    batch_parameters = {
        "N": [200],
        "province": ["Ontario"],  # , "Alberta", "Ontario"],
        "random_seed": list(range(3)),
    }

    path = Path("results/UNIQUE_MODEL_NAME")

    result = batch_run(
        TechnologyAdoptionModel,
        batch_parameters,
        number_processes=None,
        data_collection_period=1,
        max_steps=80,
    )
    b_result = BatchResult(path, result, batch_parameters)
    # ax = b_result.viz_adoption
    ax = b_result.attitudes_fig
    ax
