import pandas as pd
import re
import numpy as np
import scipy.stats as scistat
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
import streamlit as st

import config


def drop_redundant_cols(df):
    cols_to_drop = []
    redundant_cols = [
        "UOM_ID",
        "SCALAR_ID",
        "COORDINATE",
        "UOM",
        "VECTOR",
        "STATUS",
        "DECIMALS",
    ]
    for col in df.columns:
        if col in redundant_cols:
            cols_to_drop.append(col)
            continue
        if len(df[col].unique()) < 2:
            cols_to_drop.append(col)

    df.drop(cols_to_drop, axis=1, inplace=True)


def mean_income(hh_income: str):
    """calculates the mean income of the range given as a string
    in the form of `Under $5,000` or '$5,000 to $9,999'
    """
    matches = re.findall(r"[0-9,]{5,}", hh_income)
    matches = [int(m.replace(",", "")) for m in matches]
    if len(matches) < 1:
        return 0
    elif len(matches) > 2:
        raise ValueError(f"Expected max. 2 matches but found: {matches}")
    return np.mean(matches)


def create_geo_fig(province):
    # from https://github.com/codeforgermany/click_that_hood/blob/main/public/data/canada.geojson
    country_shape_df = gpd.read_file("data/canada/canada.geojson")
    country_shape_df.set_index("name", inplace=True)

    if province == "Canada":
        values = 1
    else:
        values = [int(prov == province) for prov in country_shape_df.index]

    country_shape_df["value"] = values

    geo_fig = px.choropleth(
        country_shape_df,
        geojson=country_shape_df.geometry,
        locations=country_shape_df.index,
        color="value",
    )
    geo_fig.update_geos(
        fitbounds="locations",
    )  # visible=False)
    geo_fig.update_layout(coloraxis_showscale=False)
    return geo_fig


# data from nrcan:
nrcan_tech_shares_df = pd.read_csv("data/canada/nrcan_tech_shares.csv").set_index(
    ["year", "province"]
)

# might add table 3610058701 to use savings rate
household_expenditures = pd.read_csv("data/canada/1110022401_databaseLoadingData.csv")

energy_consumption = pd.read_csv("data/canada/2510006201_databaseLoadingData.csv")

all_provinces = sorted(list(energy_consumption["GEO"].unique()))

energy_consumption["Household income"] = energy_consumption[
    "Household income"
].str.removesuffix("(includes income loss)")

_irrelevant_cols = [
    "Total - Total income of households",
    "Median total income of household ($)",
    "$100,000 and over",
]
income_df = pd.read_csv("data/canada/9810005501_databaseLoadingData.csv")
income_df = income_df.query(
    "`Household total income groups (22)` not in @_irrelevant_cols"
)
income_df["Mean income"] = income_df["Household total income groups (22)"].apply(
    mean_income
)


# pre-compute parameters of linear function
_total_en_p_household = energy_consumption.query(
    "`Energy consumption` == 'Gigajoules per household' "
    "and `Energy type`=='Total, all energy types'"
).fillna(0)


_total_en_p_household.loc[:, "Mean household income"] = _total_en_p_household[
    "Household income"
].apply(mean_income)

# dict to hold parameters for regression
_province_demand_regression = {}
for prov in all_provinces:
    x = _total_en_p_household.query(f"GEO=='{prov}'")["Mean household income"].values
    y = _total_en_p_household.query(f"GEO=='{prov}'")["VALUE"].values
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    _province_demand_regression[prov] = (m, c)


def energy_demand_from_income_and_province(income, province, kWh=True):
    """determines energy demand by using a linear fit from canadian input data.
    # Returns
    Energy per household in kWh as `float`
    """
    params = _province_demand_regression[province]
    if params[0] < 0:
        print("Warning: Energy demand decreasing with increasing income for", province)

    if kWh:
        return params[0] * income + params[1] * 1000 / 3.6
    else:
        return params[0] * income + params[1]



def get_beta_distributed_incomes(n, a=1.58595876, b=7.94630802):
    # a & b are the result of the fit to the canadian income distribution

    incomes = np.random.beta(a, b, size=n)
    # norm
    incomes /= incomes.max()
    # scale interval end-point to suit the existing data
    # the result matches https://www.statista.com/statistics/484838/income-distribution-in-canada-by-income-level/ quite nicely
    incomes *= 250000
    return incomes


def gamma(x, a, b):
    return scistat.gamma.pdf(x, a, scale=1 / b)


def get_gamma_distributed_incomes(n):
    # these parameters for a & b of `gamma` are the result of the fit to the
    # canadian income distribution
    p = [2.30603102, 0.38960872]
    # income_dist = pm.Gamma.dist(*p)
    # incomes = pm.draw(income_dist, draws=n, random_seed=seed)
    incomes = np.random.gamma(shape=p[0], scale=p[1], size=n)
    incomes = incomes * 10000 + 10000
    return incomes


heating_systems = pd.read_csv("data/canada/3810028601_databaseLoadingData.csv")

electricity_prices = pd.read_csv("data/canada/ca_electricity_prices.csv", header=14)
electricity_prices.set_index("REF_DATE", inplace=True)
# might add table 9810043901 that relates income to education level in the future

fuel_prices = pd.read_csv("data/canada/1810000101_databaseLoadingData.csv")
# st.write(pd.to_datetime(fuel_prices["REF_DATE"]))
fuel_prices.loc[:, ["Year", "Month"]] = (
    fuel_prices["REF_DATE"].str.split("-", expand=True).astype(float).values
)


def get_province(x):
    prov = x.split(",")[-1]
    if "/" in prov:
        prov = prov.split("/")[-1]
    return prov.strip()


energy_contents_per_l_in_kWh = {
    "diesel": 38.29 / 3.6,
    "gasoline": 33.52 / 3.6,
    "heating oil": 38.2 / 3.6,
}
energy_contents_per_m3_in_kWh = {"natural_gas": 38.64 / 3.6}


def get_energy_per_litre(x):
    for k, v in energy_contents_per_l_in_kWh.items():
        if k in x.lower():
            return v
    raise ValueError(f"{x} not found in {energy_contents_per_l_in_kWh}")


def get_simple_fuel(x):
    if "diesel" in x.lower():
        return "Diesel"
    elif "gasoline" in x.lower():
        return "Gasoline"
    else:
        return x


fuel_prices["GEO"] = fuel_prices["GEO"].apply(get_province)
fuel_prices["Type of fuel"] = fuel_prices["Type of fuel"].apply(get_simple_fuel)
fuel_prices = (
    fuel_prices.groupby(["Year", "GEO", "Type of fuel"])
    .mean(numeric_only=True)
    .reset_index()
)
fuel_prices = fuel_prices.query("GEO != 'Canada'")
fuel_prices["energy_density(kWh/l)"] = fuel_prices["Type of fuel"].apply(
    get_energy_per_litre
)
fuel_prices["Price (ct/kWh)"] = (
    fuel_prices["VALUE"] / fuel_prices["energy_density(kWh/l)"]
)
canada_prices = (
    fuel_prices.groupby(["Year", "Type of fuel"]).mean(numeric_only=True).reset_index()
)
canada_prices["GEO"] = "Canada"
fuel_prices = pd.concat([fuel_prices, canada_prices])


gas_prices = pd.read_csv("data/canada/2510003301_databaseLoadingData.csv")
gas_prices.loc[:, ["Year", "Month"]] = (
    gas_prices["REF_DATE"].str.split("-", expand=True).astype(float).values
)
gas_prices = gas_prices.groupby(["Year", "GEO"]).mean(numeric_only=True).reset_index()
gas_prices["energy_density(kWh/m3)"] = energy_contents_per_m3_in_kWh["natural_gas"]
gas_prices["Price (ct/kWh)"] = (
    gas_prices["VALUE"] / gas_prices["energy_density(kWh/m3)"]
)


biomass_prices = pd.read_csv("data/canada/biomass_prices.csv", header=6)

for df in [
    household_expenditures,
    energy_consumption,
    heating_systems,
    income_df,
    fuel_prices,
    gas_prices,
]:
    drop_redundant_cols(df)

all_fuels = [
    "Electricity",
    "Natural gas",
    "Oil",
    "Wood or wood pellets",
    "Propane",
    "Other fuel",
]
all_techs = [
    "Forced air furnace",
    "Electric baseboard heaters",
    "Heating stove",
    "Boiler with hot water or steam radiators",
    "Electric radiant heating",
    "Heat pump",
    "Other type of heating system",
]
_fuel_df = heating_systems.query(
    "`Primary heating system and type of energy` in @all_fuels"
)
_fuel_df = _fuel_df.pivot(
    index=["REF_DATE", "GEO"],
    columns=["Primary heating system and type of energy"],
    values="VALUE",
).fillna(0)

_appliances_df = heating_systems.query(
    "`Primary heating system and type of energy` in @all_techs"
)
_appliances_df = _appliances_df.pivot(
    index=["REF_DATE", "GEO"],
    columns=["Primary heating system and type of energy"],
    values="VALUE",
).fillna(0)

simplified_heating_stock = _fuel_df.copy()

simplified_heating_stock["Electric furnace"] = (
    simplified_heating_stock["Electricity"] - _appliances_df["Heat pump"]
)
simplified_heating_stock["Heat pump"] = _appliances_df["Heat pump"]
simplified_heating_stock["Gas furnace"] = (
    simplified_heating_stock["Natural gas"] + simplified_heating_stock["Propane"]
)
simplified_heating_stock.drop(
    ["Electricity", "Natural gas", "Propane"], axis=1, inplace=True
)
simplified_heating_stock.columns = [
    col + " furnace" if ("furnace" not in col and "pump" not in col) else col
    for col in simplified_heating_stock.columns
]
simplified_heating_stock.drop("Other fuel furnace", axis=1, inplace=True)

el_prices_long = electricity_prices.melt(
    value_name="Price (ct/kWh)", var_name="GEO", ignore_index=False
)
el_prices_long["Type of fuel"] = "Electricity"
el_prices_long.reset_index(names=["Year"], inplace=True)
# st.write(el_prices_long)
gas_prices["Type of fuel"] = "Natural gas"
biomass_prices["GEO"] = "Canada"
biomass_prices["Type of fuel"] = "Wood or wood pellets"
all_fuel_prices = pd.concat([el_prices_long, fuel_prices, gas_prices, biomass_prices])
all_fuel_prices.set_index(
    [
        "Type of fuel",
        "Year",
    ],
    inplace=True,
)
tech_capex_df = pd.read_csv("data/canada/heat_tech_params.csv").set_index(
    ["year", "variable"]
)


def update_facet_plot_annotation(fig, annot_func=None, textangle=-30, xanchor="left"):
    """updates the annotations of a figure

    ## Args:
        fig (plotly Figure): The figure to be manipulated
        annot_func (func, optional): A custom function to be applied to the annotation
                                    text. Defaults to None.
        textangle (int, optional): Angle of the annotation. Defaults to -30.
        xanchor (str, optional): Where to anchor the annotation. Defaults to "left".

    ## Returns:
        modified figure: _description_
    """
    for annot in fig.layout.annotations:
        if annot_func is None:
            new_text = annot["text"].split("=")[1]
        else:
            new_text = annot_func(annot["text"])
        annot["text"] = new_text
        annot["textangle"] = textangle
        annot["xanchor"] = "left"
    return fig


def get_fuel_price(fuel, province, year, fall_back_province="Canada"):
    # print(all_fuel_prices)
    fuel_prices = all_fuel_prices.loc[fuel, :]

    local_fuel_prices = fuel_prices.query(f"GEO == '{province}'")
    local_fuel_prices = local_fuel_prices[["Price (ct/kWh)"]].dropna()
    if len(local_fuel_prices) == 0:
        # Data is not available for all provinces
        print(
            "Warning: No data for",
            (fuel, province, year),
            ". Using prices from",
            fall_back_province,
            "instead.",
        )
        local_fuel_prices = fuel_prices.query(f"GEO == '{fall_back_province}'")
    local_fuel_prices.reset_index(inplace=True)
    local_fuel_prices.loc[:, "Year"] = local_fuel_prices["Year"].astype(int)
    if year not in local_fuel_prices["Year"].unique():
        # deterine closest year
        time_dist = (local_fuel_prices.loc[:, "Year"] - year).abs()
        minimum_distance_idx = time_dist.idxmin()
        year = local_fuel_prices.loc[minimum_distance_idx, "Year"]

    timely_fuel_prices = local_fuel_prices.query(f"Year == {year}")
    timely_fuel_prices.reset_index(inplace=True)
    price = timely_fuel_prices["Price (ct/kWh)"][0]
    assert price > 0 and not pd.isna(price), AssertionError(
        f"Sorry, no price available for {province}."
    )
    return price / 100  # convert ct/kWh to CAD/kWh


def run():
    st.set_page_config(page_title="Canadian Inputs")

    if "technology_colors" not in st.session_state:
        st.session_state["technology_colors"] = config.TECHNOLOGY_COLORS
        st.session_state["fuel_colors"] = config.FUEL_COLORS

    st.markdown("# Financials")
    with st.expander("currently unused"):
        st.markdown("## Household expeditures")

        fig = px.scatter(
            household_expenditures,
            x="REF_DATE",
            y="VALUE",
            color="Household expenditures, summary-level categories",
            facet_col="Household type",
        )

        fig = update_facet_plot_annotation(fig)

        fig.update_layout(
            yaxis_title="Annual Expenses (CAD)",
            margin={"t": 150, "r": 0},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Household income")
    provinces = st.multiselect("select provinces", all_provinces, all_provinces[:3])
    income = income_df
    income["bin_no"] = income["Mean income"] // 10000
    income["Mean income"] = income["bin_no"] * 10000
    # income.rename({"bin_no":"Mean income"}, axis=1, inplace=True)
    income = income.query("`Mean income` < 100001")
    agg_df = (
        income.groupby(
            [
                "GEO",
                "Year (2)",
                "Mean income",
            ]
        )
        .sum(numeric_only=True)
        .reset_index()
    )
    fig = px.bar(
        agg_df.query(f"`Year (2)`==2015 and GEO in {provinces}"),
        x="Mean income",
        y="VALUE",
        facet_col="GEO",
    )
    fig.update_layout(yaxis_title="Frequency")
    fig = update_facet_plot_annotation(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
        This data (from [statcan](https://www150.statcan.gc.ca/n1/en/type/data?MM=1)) was used to fit a `beta` probability distribution to it. 
                Incomes $> 100.000\ CAD $ were excluded due to uneven bin size.
                See the following figure for the fit vs. the data regarding Canada.
        """
    )

    canada_income = agg_df.query("GEO=='Canada' and `Year (2)`==2015")
    normed_income_freq = canada_income["VALUE"] / canada_income["VALUE"].sum()
    normed_income_bins = (
        canada_income["Mean income"] / canada_income["Mean income"].sum()
    )

    def fit_beta(a, b):
        x = np.linspace(0, 1, 100)
        y = scistat.beta.pdf(x, a, b)
        y = y / y.max() * normed_income_freq.max()
        ax = plt.pyplot.plot(x, y, label="beta fit")
        plt.pyplot.plot(
            canada_income["Mean income"] / canada_income["Mean income"].max(),
            normed_income_freq,
        )
        return ax

    def scaled_beta(x, a, b):
        y = scistat.beta.pdf(x, a, b)
        y = y / y.max() * normed_income_freq.max()
        return y

    # p,v = curve_fit(scaled_beta, normed_income_bins, normed_income_freq, p0=(2, 2))
    y1 = scaled_beta(normed_income_bins, *[1.58595876, 7.9463080])

    x = canada_income["Mean income"] // 10000
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x * 10000, normed_income_freq, label="Canadian income PDF")
    ax.plot(
        agg_df.query("`Year (2)`==2015 and GEO=='Canada'")["Mean income"],
        y1,
        label="beta fit",
    )
    ax.set_xlabel("Income")
    ax.set_ylabel("Probability")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.markdown("## Fuel prices")
    st.write(all_fuel_prices.head())
    all_fuels = all_fuel_prices.index.get_level_values(0).unique().to_list()
    fuel_types = st.multiselect("Select fuels", all_fuels, all_fuels)
    fuel_prices_fig = px.line(
        all_fuel_prices.reset_index().query(
            f"GEO in {provinces} and `Type of fuel` in {fuel_types}"
        ),
        x="Year",
        y="Price (ct/kWh)",
        color="GEO",
        facet_row="Type of fuel",
        height=600,
    )
    fuel_prices_fig = update_facet_plot_annotation(fuel_prices_fig, textangle=-90)
    st.plotly_chart(fuel_prices_fig)

    st.markdown("# Energy consumption")
    with st.expander("Province level consumption"):
        fig = px.scatter(
            energy_consumption.query(
                "`Energy consumption`=='Gigajoules' and GEO in @provinces"
            ),
            x="Household income",
            y="VALUE",
            color="Energy type",
            facet_col="GEO",
            symbol="REF_DATE",
        )
        fig = update_facet_plot_annotation(fig)
        fig.update_layout(
            margin={"t": 100}, yaxis=dict(title="Energy consumption (GJ)")
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        energy_consumption.query(
            "`Energy consumption`=='Gigajoules per household' and GEO in @provinces"
        ),
        x="Household income",
        y="VALUE",
        color="Energy type",
        facet_col="GEO",
        symbol="REF_DATE",
        title="Statistical data",
    )
    fig = update_facet_plot_annotation(fig)

    st.markdown(
        """
        The per household energy consumption was used to derive linear fits to represent
        that more affluent households consume more energy.
        """
    )
    fig.update_layout(
        margin={"t": 100},
        yaxis=dict(title="Energy consumption (GJ/household)"),
    )
    st.plotly_chart(fig, use_container_width=True)
    income_levels = np.linspace(0, 150000, 100)
    energy_demands = [
        energy_demand_from_income_and_province(income_levels, p, kWh=False)
        for p in provinces
    ]

    fit_df = pd.DataFrame(
        energy_demands, columns=income_levels, index=provinces
    ).T.melt(ignore_index=False, var_name="Province")
    fit_df.reset_index(inplace=True)
    fit_df.rename({"index": "Household income"}, axis=1, inplace=True)
    fit_fig = px.line(
        fit_df,
        x="Household income",
        y="value",
        facet_col="Province",
        title="Linear fit",
    )
    fit_fig.update_layout(yaxis_title="GJ/Household", showlegend=False)
    st.plotly_chart(fit_fig, use_container_width=True)

    st.markdown("# Heating technology distribution")
    st.markdown(
        """The `Residential Sector`-data from [nrcan](https://oee.nrcan.gc.ca/corporate/statistics/neud/dpa/menus/trends/comprehensive_tables/list.cfm) 
        have been combined on a province level.
        """
    )
    nrcan_tech_shares_df_long = nrcan_tech_shares_df.reset_index().melt(
        id_vars=["province", "year"], var_name="technology"
    )
    st.write(nrcan_tech_shares_df.head())
    fig = px.area(
        nrcan_tech_shares_df_long.query(f"province in {provinces}"),
        x="year",
        y="value",
        color="technology",
        facet_col="province",
        color_discrete_map=config.TECHNOLOGY_COLORS,
    )
    fig = update_facet_plot_annotation(fig)
    fig.update_layout(legend_traceorder="reversed")
    st.plotly_chart(fig)
    with st.expander("previously used tech shares"):
        st.markdown("## Technologies")
        st.markdown(
            """Initially, technology shares were to be derived from this data. Howver, most
            of these technologies like the `Forced air furnace` can have multiple fuels 
            and further analysis of the statistical data revealed that many data points are 
            not present at the more granular level. Nevertheless, the share of `Heat pumps` 
            from this table was used.
            """
        )

        fig = px.area(
            heating_systems.query(
                "`Primary heating system and type of energy` in @all_techs "
                f"and GEO in {provinces}"
            ),
            x="REF_DATE",
            y="VALUE",
            color="Primary heating system and type of energy",
            facet_col="GEO",
        )
        fig = update_facet_plot_annotation(fig)

        all_fuels_statcan = [
            "Electricity",
            "Natural gas",
            "Oil",
            "Wood or wood pellets",
            "Propane",
            "Other fuel",
        ]
        fig.update_layout(width=900, margin_t=100, yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("## Fuels for heating equipment")
        fig = px.area(
            heating_systems.sort_values(by="GEO").query(
                f"`Primary heating system and type of energy` in {all_fuels_statcan}"
                " and GEO in @provinces"
            ),
            x="REF_DATE",
            y="VALUE",
            color="Primary heating system and type of energy",
            facet_col="GEO",
            color_discrete_map=config.FUEL_COLORS,
        )
        fig = update_facet_plot_annotation(fig)

        fig.update_layout(
            width=900, margin_t=100, yaxis_title="%", legend_traceorder="reversed"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            ### Derive 'simplified' heating technologies
            Since the more granular data (i.e. '<FUEL_NAME> forced air furnace') 
            are often not available, technology shares have been derived from the
            fuel shares. `Propane` and `Natural gas` are grouped as a `Gas furnace`
            , `Wood or wood pellets` becomes a `Biomass furnace` and `Oil` becomes 
            an `Oil furnace`. For these technologies the difference between it 
            being a `Forced air furnace` or a `Boiler` is negligible in terms of 
            efficiency.

            The picture is different however, when regarding electricity. While 
            `Heat pumps` have an efficiency of $\eta>2$ for most of the year, all 
            other heating technologies have an efficiency of $<1$. Hence, the heat 
            pump share from above is used, and subtracted from the electricity share 
            to represent other electricity powered appliances.
            """
        )

        simplified_heating_stock_long = simplified_heating_stock.melt(
            ignore_index=False
        ).reset_index()

        fig = px.area(
            simplified_heating_stock_long.query("GEO in @provinces"),
            x="REF_DATE",
            y="value",
            color="variable",
            facet_col="GEO",
            color_discrete_map=config.TECHNOLOGY_COLORS,
        )
        fig = update_facet_plot_annotation(fig)

        fig.update_layout(
            width=900, margin_t=100, yaxis_title="%", legend_traceorder="reversed"
        )
        st.plotly_chart(fig, use_container_width=True)

    # this code is to show that more fine grained analysis results in less complete data
    # appliances_group_map = {"Forced air furnace": "Forced air furnace",
    # 'Electric forced air furnace': "Forced air furnace",
    # 'Natural gas forced air furnace':  "Forced air furnace",
    # 'Oil forced air furnace': "Forced air furnace",
    # 'Wood or wood pellets forced air furnace': "Forced air furnace",
    # 'Propane forced air furnace': "Forced air furnace",
    # 'Other fuel forced air furnace': "Forced air furnace",
    # 'Electric baseboard heaters': 'Electric baseboard heaters',
    # 'Heating stove': 'Heating stove',
    # 'Electric heating stove': "Heating stove",
    # 'Natural gas heating stove':"Heating stove",
    # 'Oil heating stove':"Heating stove",
    # 'Wood heating stove':"Heating stove",
    # 'Propane heating stove':"Heating stove",
    # 'Other fuel heating stove':"Heating stove",
    # 'Boiler with hot water or steam radiators': 'Boiler with hot water or steam radiators',
    # 'Electric boiler with hot water or steam radiators':'Boiler with hot water or steam radiators',
    # 'Natural gas boiler with hot water or steam radiators':'Boiler with hot water or steam radiators',
    # 'Oil boiler with hot water or steam radiators':'Boiler with hot water or steam radiators',
    # 'Wood boiler with hot water or steam radiators':'Boiler with hot water or steam radiators',
    # 'Propane boiler with hot water or steam radiators':'Boiler with hot water or steam radiators',
    # 'Other fuel boiler with hot water or steam radiators':'Boiler with hot water or steam radiators',
    # 'Electric radiant heating': 'Electric radiant heating',
    # 'Heat pump':'Heat pump',
    # 'Other type of heating system':'Other type of heating system'}

    # def get_appliance_group(appliance_name):
    #     return appliances_group_map.get(appliance_name, appliance_name)

    # df["Appliance Group"] = df["Primary heating system and type of energy"].apply(get_appliance_group)

    # tech_shares_wide = df.pivot(index=["REF_DATE","GEO"], columns=["Appliance Group","Primary heating system and type of energy",], values="VALUE").fillna(0)
    # # tech_shares_wide.head()
    # for l0, l1 in tech_shares_wide.columns:
    #     if l0 == l1:
    #         continue
    #     else:
    #         appliance_part_share = tech_shares_wide.loc[:,(l0,l0)] * tech_shares_wide.loc[:,(l0,l1)]/100
    #         tech_shares_wide.loc[:,(l0,l1)] = appliance_part_share
    # tech_shares_wide.head()

    # # fuels = ["All primary heating systems","Electricity","Natural gas","Oil","Wood or wood pellets","Propane","Other fuel"]

    # for l0, l1 in tech_shares_wide.columns:
    #     if l0 == l1:
    #         tech_shares_wide.drop(columns=(l0,l1), axis=1, inplace=True)
    # # tech_shares_wide.head()

    # tech_shares_long = tech_shares_wide.melt(ignore_index=False).reset_index()
    # px.area(tech_shares_long, x="REF_DATE", y="value", color="Primary heating system and type of energy", facet_col="GEO")


if __name__ == "__main__":
    run()
