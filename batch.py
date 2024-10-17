import seaborn as sns
from components.model import TechnologyAdoptionModel
from config import START_YEAR, YEARS_PER_STEP, TECHNOLOGY_COLORS
from mesa.batchrunner import batch_run
from components.technologies import Technologies, tech_fuel_map
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from dataclasses import dataclass
import git
import hashlib
import dill
import logging
repo = git.Repo(".", search_parent_directories=True)
current_commit_hash = str(repo.references[0].commit)[:6]

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="myapp.log", level=logging.INFO, format=current_commit_hash+" -%(asctime)s %(message)s"
)


def transform_dict_column(df, dict_col_name="Technology shares", return_cols=True):
    if dict_col_name not in df.columns:
        return df, None
    adoption_df = pd.DataFrame.from_records(df[dict_col_name])
    df.loc[:, adoption_df.columns] = adoption_df
    if return_cols:
        return df.drop(dict_col_name, axis=1), adoption_df.columns
    else:
        return df.drop(dict_col_name, axis=1)


def dict_hash(dictionary: dict) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()

    encoded = dill.dumps(dictionary)
    dhash.update(encoded)
    return dhash.hexdigest()


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

    with open(results_dir.joinpath("batch_parameters.dillson"), "wb") as fo:
        dill.dump(batch_parameters, fo)


def read_batch_parameters(batch_parameter_path):
    with open(batch_parameter_path, "rb") as fi:
        batch_parameters = dill.load(fi)
    return batch_parameters


@dataclass
class BatchResult:
    def __init__(self, batch_parameters, data=None, results_df=None, force_rerun=False):
        self.path = self.get_results_dir(batch_parameters, force_rerun=force_rerun)
        # self.force_rerun = force_rerun
        self.batch_params = batch_parameters
        logger.log(logging.INFO, f"Initialized {self}.")
        if data is not None:
            self.data = data
            self.results_df = pd.DataFrame(self.data)
        elif results_df is not None:
            self.results_df = results_df
        else:
            raise ValueError("Either `data` or `results_df` must be passed.")

        def convert_steps_to_years(steps):
            if "start_year" in batch_parameters.keys():
                start_year = batch_parameters["start_year"]
            else:
                start_year = START_YEAR
            if "years_per_step" in batch_parameters.keys():
                years_per_step = batch_parameters["years_per_step"]
            else: 
                years_per_step = YEARS_PER_STEP
            years = TechnologyAdoptionModel.steps_to_years_static(
                start_year, steps, years_per_step
            )
            return years

        self.results_df["year"] = convert_steps_to_years(self.results_df["Step"])

    @classmethod
    def run_batch(
        cls, batch_parameters, max_steps=80, force_rerun=False, display_progress=True
    ):
        if "max_steps" in batch_parameters.keys():
            raise ValueError("`max_steps` in batch_parameters not allowed!")
        path = cls.get_results_dir(batch_parameters)

        if path.exists() and not force_rerun:
            raise ValueError(
                f"""A result for this parameter set exists at {path}.
                              Consider calling 'BatchResult.from_directory({path})' or 
                              `BatchResult.from_parameters({batch_parameters})` instead.
                              If you want to force a re-run run this method with ``force_rerun=True``"""
            )

        results = batch_run(
            TechnologyAdoptionModel,
            batch_parameters,
            number_processes=None,
            max_steps=max_steps,
            data_collection_period=1,
            display_progress=display_progress,
        )
        return pd.DataFrame(results)

    @staticmethod
    def get_repo_root():
        repo = git.Repo(".", search_parent_directories=True)
        repo_root = repo.working_dir
        branch_dir_name = ""
        __current_file_path = Path(__file__).absolute().as_posix()

        # determine, if the repo is a submodule or not
        for smod in repo.submodules:
            submodule_path = Path(smod.path).absolute().as_posix()
            if submodule_path in __current_file_path:
                repo_root = submodule_path
                branch_dir_name = smod.branch_name

        if branch_dir_name == "":
            branch_dir_name = str(repo.head.ref).replace("/", "_")

        return repo_root, branch_dir_name

    @staticmethod
    def get_results_dir(batch_parameters, force_rerun=False):
        repo_root, branch_dir_name = BatchResult.get_repo_root()

        batch_param_hash = dict_hash(batch_parameters)

        results_path = Path(f"{repo_root}/results/{branch_dir_name}").joinpath(
            str(batch_param_hash)
        )

        if force_rerun:
            # in this case, create a folder suffixed by the run number
            r_dir_rerun = results_path.name + "_{i}"
            i = 0
            while results_path.with_name(r_dir_rerun.format(i=i)).exists():
                i += 1
            results_path = results_path.with_name(r_dir_rerun.format(i=i))
        return results_path

    def init_from_directory(self, directory):
        result = self.from_directory(directory)
        for member_name in dir(result):
            if (
                "__" not in member_name
                and member_name[0] == "_"
                and member_name[-2:] == "df"
            ):
                other_member_value = getattr(result, member_name)
                setattr(self, member_name, other_member_value)
        self.results_df = result.results_df

    @classmethod
    def from_parameters(
        cls, batch_parameters, max_steps=80, force_rerun=False, display_progress=True
    ):
        results_dir = cls.get_results_dir(batch_parameters, force_rerun=force_rerun)
        if results_dir.exists():
            logger.log(logging.INFO, f"{results_dir=} exists, loading results")
            return cls.from_directory(results_dir)
        else:
            logger.log(logging.INFO, f"{results_dir=} does not exist. Running model.")
            return cls(
                batch_parameters,
                results_df=cls.run_batch(
                    batch_parameters,
                    max_steps=max_steps,
                    force_rerun=force_rerun,
                    display_progress=display_progress,
                ),
                force_rerun=force_rerun,
            )

    @classmethod
    def from_directory(cls, directory):
        path = Path(directory)
        batch_parameters = read_batch_parameters(
            path.joinpath("batch_parameters.dillson")
        )

        with path.joinpath("batch_results.dill").open("rb") as fi:
            result_data = dill.loads(fi.read())

        result = BatchResult(batch_parameters, results_df=result_data)
        mean_carrier_demand = pd.read_pickle(path.joinpath("mean_carrier_demand.pkl"))
        setattr(result, "_mean_carrier_demand_df", mean_carrier_demand)

        for csv in path.glob("*.csv"):
            df = pd.read_csv(csv)
            setattr(result, "_" + csv.stem, df.copy())

        return result

    def save(self, custom_path=None):
        if custom_path is not None:
            result_path = Path(custom_path)
        else:
            result_path = self.path
        if not result_path.exists():
            result_path.mkdir(parents=True)

        # save small results as .csv
        for df_name in [
            "tech_shares_df",
            "adoption_details_df",
            "attitudes_df",
            "subsidies_df",
        ]:
            df = getattr(self, df_name)
            df.to_csv(result_path.joinpath(df_name + ".csv"), index=False)

        # carrier demand is larger, so save as pkl
        self.mean_carrier_demand_df.to_pickle(
            result_path.joinpath("mean_carrier_demand.pkl")
        )

        # serialize results
        with result_path.joinpath("batch_results.dill").open("wb") as fo:
            fo.write(dill.dumps(self.results_df))

        save_batch_parameters(self.batch_params, result_path)
        return result_path

    @property
    def tech_shares_df(self) -> pd.DataFrame:
        if hasattr(self, "_tech_shares_df"):
            return self._tech_shares_df

        shares = self.results_df[["RunId", "province", "year", "Technology shares"]]
        data = pd.DataFrame.from_records(shares["Technology shares"].values)
        shares.loc[:, data.columns] = data
        shares = shares.drop(columns=["Technology shares"])
        shares = shares.drop_duplicates()
        self.results_df.drop(columns=["Technology shares"], axis=1)
        self._tech_shares_df = shares.copy()
        return shares

    def tech_shares_fig(self, show_legend=True, colors: dict = None):
        if colors is None:
            colors = TECHNOLOGY_COLORS

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
            palette=colors,
        )
        ax.set_ylabels("Technology share (%)")
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
        detail_df = pd.DataFrame.from_records(
            adoption_detail["Adoption details"].values
        )
        adoption_detail.loc[:, detail_df.columns] = detail_df
        adoption_detail = adoption_detail.drop("Adoption details", axis=1)
        adoption_detail["amount"] = 1
        drop_rows = adoption_detail["tech"].apply(lambda x: x is None)
        adoption_detail = adoption_detail.loc[~drop_rows, :].reset_index(drop=True)
        if isinstance(adoption_detail["tech"][0], Technologies):
            adoption_detail["tech"] = adoption_detail["tech"].apply(lambda x: x.value)
        adoption_detail = (
            adoption_detail.groupby(["year", "RunId", "AgentID", "tech"])
            .sum()
            .reset_index()
        )

        # get cumulative sum
        adoption_detail["cumulative_amount"] = adoption_detail.groupby(
            ["RunId", "tech"]
        )["amount"].cumsum()

        self.results_df.drop("Adoption details", axis=1, inplace=True)
        self._adoption_details_df = adoption_detail.copy()
        return adoption_detail

    @property
    def subsidies_df(self):
        hp_df = self.adoption_details_df.query("tech=='Heat pump'").set_index(
            ["year", "RunId", "AgentID"]
        )
        self.results_df["hp specific cost"] = self.results_df[
            "Heat pump specific_cost"
        ].apply(lambda x: x["Heat pump"])
        hp_df[["Required heating size", "hp specific cost", "hp_subsidy"]] = (
            self.results_df[
                [
                    "year",
                    "RunId",
                    "AgentID",
                    "Required heating size",
                    "hp specific cost",
                    "hp_subsidy",
                ]
            ]
            .set_index(["year", "RunId", "AgentID"])
            .loc[hp_df.index, :]
        )
        hp_df["subsidy amount"] = (
            hp_df["Required heating size"]
            * hp_df["hp specific cost"]
            * hp_df["hp_subsidy"]
            * hp_df["amount"]
        )
        hp_df["Cumulative subsidy amount (CAD)"] = hp_df.groupby(["RunId"])[
            "subsidy amount"
        ].cumsum()
        return hp_df

    def emissions(self):
        """Calculates the resulting emissions per year and carrier.

        Returns:
            emission_df: Emission DataFrame (Mt)
        """
        annual_demands = self.mean_carrier_demand_df.groupby("year").sum()
        annual_demands.index = annual_demands.index.astype("int")

        root, _ = self.get_repo_root()
        emissions = (
            pd.read_csv(f"{root}/data/canada/heat_tech_params.csv")
            .set_index(["variable", "year"])
            .loc["specific_fuel_emission", :]
            .iloc[:, 1:]
        )

        emissions.columns = [tech_fuel_map[col] for col in emissions.columns]
        scenario_years = range(2020, 2051)
        missing_years = [y for y in scenario_years if y not in emissions.index]
        empty_df = pd.DataFrame(columns=emissions.columns, index=missing_years).astype(
            float
        )
        all_emission_years = pd.concat([emissions, empty_df]).sort_index().interpolate()
        # kg                    kg/kWh     *  kWh
        emission_df = (
            (all_emission_years * annual_demands).astype(float).interpolate().dropna()
        )
        # kg -> Mt
        emission_df /= 1e9
        return emission_df

    def subsidies_fig(self):
        hp_df = self.subsidies_df
        return sns.lineplot(
            hp_df.reset_index(), x="year", y="Cumulative subsidy amount (CAD)"
        )

    def adoption_details_fig(self, colors: dict = None):
        if colors is None:
            colors = TECHNOLOGY_COLORS

        adoption_detail = self.adoption_details_df

        ax = sns.relplot(
            adoption_detail,
            kind="line",
            x="year",
            y="cumulative_amount",
            hue="tech",
            col="reason",
            palette=colors,
        )
        ax.set_xticklabels(rotation=45)
        return ax

    def adoption_details_fig_facet(self, n_facet_cols=None, colors: dict = None):
        if colors is None:
            colors = TECHNOLOGY_COLORS
        adoption_detail = self.adoption_details_df

        if n_facet_cols is not None:
            display_ids = np.linspace(
                0, adoption_detail["RunId"].max(), n_facet_cols, dtype=int
            )
            adoption_detail = adoption_detail.query(f"RunId in {list(display_ids)}")

        fig = px.bar(
            adoption_detail,
            x="year",
            y="amount",
            color="tech",
            facet_col="RunId",
            facet_row="reason",
            template="plotly",
            color_discrete_map=colors,
        )
        fig.update_yaxes(matches=None)
        return fig

    @property
    def attitudes_df(self) -> pd.DataFrame:
        if hasattr(self, "_attitudes_df"):
            return self._attitudes_df

        atts_df = self.results_df[["RunId", "year", "Attitudes"]]

        data = atts_df["Attitudes"].to_list()
        data = pd.DataFrame(data)
        atts_df.loc[:, data.columns] = data

        atts_df = atts_df.drop("Attitudes", axis=1)
        self.results_df.drop("Attitudes", axis=1, inplace=True)
        self._attitudes_df = atts_df.copy()
        return atts_df

    def attitudes_fig(self, colors: dict = None):
        if colors is None:
            colors = TECHNOLOGY_COLORS
        atts_df = self.attitudes_df

        atts_long = atts_df.melt(id_vars=("RunId", "year"))
        ax = sns.relplot(
            atts_long, kind="line", x="year", y="value", hue="variable", palette=colors
        )
        ax.set_ylabels("Attitude towards technologies (-)")
        ax.set_xticklabels(rotation=45)
        return ax

    @property
    def mean_carrier_demand_df(self):
        if hasattr(self, "_mean_carrier_demand_df"):
            return self._mean_carrier_demand_df

        demand_df = self.results_df[
            ["RunId", "year", "province", "Energy demand time series"]
        ]
        energy_demand_ts = demand_df["Energy demand time series"].to_list()
        energy_demand_df = pd.DataFrame.from_records(energy_demand_ts)
        energy_demand_df["year"] = demand_df["year"]
        energy_demand_df["RunId"] = demand_df["RunId"]
        energy_demand_df["province"] = demand_df["province"]
        keep_rows = ~energy_demand_df[["RunId", "year"]].duplicated()
        keep_years = energy_demand_df["year"] % 5 == 0
        energy_demand_df = energy_demand_df.loc[keep_rows & keep_years, :]
        energy_demand_df = energy_demand_df.set_index(
            ["province", "year", "RunId"]
        ).sort_index()

        len_ts_demand = len(energy_demand_df.iloc[0, 0])

        selected_years = energy_demand_df.reset_index()["year"].unique()
        provinces = demand_df["province"].unique()
        new_idx = pd.MultiIndex.from_product(
            [provinces, selected_years, range(len_ts_demand)],
            names=["province", "year", "hour"],
        )
        mean_carrier_demand = pd.DataFrame(
            index=new_idx, columns=energy_demand_df.columns
        ).sort_index()
        for province in provinces:
            for year in selected_years:
                years_df = energy_demand_df.loc[(province, year), :]
                for carrier in years_df.columns:
                    carrier_vals = years_df[carrier].to_list()
                    sum_array = np.zeros(len(carrier_vals[0]))
                    for vals in carrier_vals:
                        sum_array += vals

                    mean_demand = sum_array / len(carrier_vals)
                    mean_carrier_demand.loc[(province, year), carrier] = mean_demand

        self._mean_carrier_demand_df = mean_carrier_demand
        return mean_carrier_demand

    def appliance_age_fig(self, colors=None):
        if colors is None:
            colors = TECHNOLOGY_COLORS

        ax = sns.lineplot(
            self.results_df[["year", "Appliance age", "Appliance name"]],
            x="year",
            y="Appliance age",
            hue="Appliance name",
            palette=colors,
        )
        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels, rotation=45)
        ax.legend(loc=(1.01, 0.3))
        return ax


if __name__ == "__main__":
    batch_parameters = {
        "N": [100],
        "province": ["Ontario"],
        "random_seed": range(20, 25),
        "start_year": 2000,
        "n_segregation_steps": [40],
        "interact": [False],
        "ts_step_length": ["W"],
        # "hp_subsidy": [0.3]
    }
    b_result = BatchResult.from_parameters(batch_parameters, display_progress=True)

    b_result.save()
