import pandas as pd
import re 
import numpy as np
import pymc as pm
import scipy.stats as scistat
import matplotlib.pyplot as plt


def drop_redundant_cols(df):
    cols_to_drop = []
    for col in df.columns:
        if len(df[col].unique()) < 2:
            cols_to_drop.append(col)
    df.drop(cols_to_drop, axis=1, inplace=True)


def mean_income(hh_income: str):
    """ calculates the mean income of the range given as a string
    in the form of `Under $5,000` or '$5,000 to $9,999'
    """
    matches = re.findall(r"[0-9,]{5,}", hh_income)
    matches = [int(m.replace(",","")) for m in matches]
    if len(matches) < 1:
        return 0
    elif len(matches) > 2:
        raise ValueError(f"Expected max. 2 matches but found: {matches}")
    return np.mean(matches)


household_expenditures = pd.read_csv("data/canada/1110022401_databaseLoadingData.csv")

energy_consumption = pd.read_csv("data/canada/2510006201_databaseLoadingData.csv")

all_provinces = list(energy_consumption["GEO"].unique())

energy_consumption["Household income"] = energy_consumption[
    "Household income"
].str.removesuffix("(includes income loss)")

income = pd.read_csv("data/canada/9810005501_databaseLoadingData.csv")
income = income.query("`Household total income groups (22)` not in ['Total - Total income of households','Median total income of household ($)','$100,000 and over']")
income["Mean bin income"] = income["Household total income groups (22)"].apply(mean_income)


# pre-compute parameters of linear function
_total_en_p_household = (energy_consumption
                         .query("`Energy consumption` == 'Gigajoules per household' and `Energy type`=='Total, all energy types'")
                         .fillna(0))
_total_en_p_household.drop(["Energy consumption","UOM","UOM_ID","VECTOR","COORDINATE","STATUS","DECIMALS"], axis=1, inplace=True)

_total_en_p_household.loc[:,"Mean household income"] = _total_en_p_household["Household income"].apply(mean_income)

# dict to hold parameters for regression
_province_demand_regression = {}
for prov in all_provinces:
    x = _total_en_p_household.query(f"GEO=='{prov}'")["Mean household income"].values
    y = _total_en_p_household.query(f"GEO=='{prov}'")["VALUE"].values
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    _province_demand_regression[prov] = (m,c)


def energy_demand_from_income_and_province(income, province):
    """ determines energy demand by using a linear fit from canadian input data.
        # Returns
        Energy per household in kWh as `float`
    """
    params = _province_demand_regression[province]
    if params[0] < 0:
        print("Warning: Energy demand decreasing with increasing income for",province)
    return params[0] * income + params[1] * 1000/3.6
    

def get_gamma_distributed_incomes(n, seed=42):
    p = [2.30603102, 0.38960872]
    income_dist = pm.Gamma.dist(*p)
    incomes = pm.draw(income_dist, draws=n, random_seed=seed)
    incomes = incomes*10000 + 10000
    return incomes


def get_half_normal_canadian_incomes(n):
    # pre compute parameters for drawing "random" income level
    # income["Mean bin income"] = income["Household total income groups (22)"].apply(mean_income)
    # income["bin_no"] = income["Mean bin income"] // 50000 
    # agg_df = income.groupby(["GEO","Year (2)", "bin_no",]).sum(numeric_only=True).reset_index()

    # def half_norm(x, p2):
    #     return scistat.halfnorm.pdf(x, scale=p2)
    # 
    # x = agg_df.query("`Year (2)`==2015 and GEO=='Canada'")["bin_no"].values
    # y = agg_df.query("`Year (2)`==2015 and GEO=='Canada'")["VALUE"] / agg_df.query("`Year (2)`==2015 and GEO=='Canada'")["VALUE"].sum()
    # 
    # p,v = curve_fit(half_norm, x, y, )
    p = 2.40247809
    income_dist = pm.HalfNormal.dist(sigma=p)
    incomes = pm.draw(income_dist, draws=n, random_seed=1)
    incomes = incomes*50000 + 50000
    return incomes


heating_systems = pd.read_csv("data/canada/3810028601_databaseLoadingData.csv")

# might add table 9810043901 that relates income to education level in the future

for df in [household_expenditures, energy_consumption, heating_systems, income]:
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
_fuel_df = heating_systems.query("`Primary heating system and type of energy` in @all_fuels")
_fuel_df = _fuel_df.pivot(index=["REF_DATE","GEO"],columns=["Primary heating system and type of energy"], values="VALUE").fillna(0)

_appliances_df = heating_systems.query("`Primary heating system and type of energy` in @all_techs")
_appliances_df = _appliances_df.pivot(index=["REF_DATE","GEO"],columns=["Primary heating system and type of energy"], values="VALUE").fillna(0)

simplified_heating_systems = _fuel_df.copy()

simplified_heating_systems["Electric furnance"] =  simplified_heating_systems["Electricity"] - _appliances_df["Heat pump"]
simplified_heating_systems["Heat pump"] = _appliances_df["Heat pump"]
simplified_heating_systems["Gas furnance"] = simplified_heating_systems["Natural gas"] + simplified_heating_systems["Propane"]
simplified_heating_systems.drop(["Electricity","Natural gas","Propane"], axis=1, inplace=True)
simplified_heating_systems.columns = [col + " furnace" if ("furnance" not in col and "pump" not in col) else col 
                            for col in simplified_heating_systems.columns ]




if __name__ == "__main__":
    import plotly.express as px
    import streamlit as st
    st.set_page_config(page_title="Canadian Inputs")

    st.title("Input data from statcan")
    st.markdown("The data displayed here has been downloaded from [statcan](https://www150.statcan.gc.ca/n1/en/type/data?MM=1).")
    st.markdown("# Financials")
    st.markdown("""## Household expeditures
                
                currently unused...
                """)

    fig = px.scatter(
        household_expenditures,
        x="REF_DATE",
        y="VALUE",
        color="Household expenditures, summary-level categories",
        facet_col="Household type",
    )

    for i, hh_type in enumerate(household_expenditures["Household type"].unique()):
        fig.layout.annotations[i]["text"] = hh_type.lower().replace("households", "")
        fig.layout.annotations[i]["textangle"] = -30
        fig.layout.annotations[i]["xanchor"] = "left"

    fig.update_layout(yaxis_title="Annual Expenses (CAD)", margin_t=150, width=1000)
    st.plotly_chart(fig)

    st.markdown("## Household income")
    income["Mean bin income"] = income["Household total income groups (22)"].apply(mean_income)
    income["bin_no"] = income["Mean bin income"] // 10000 
    income["bin_no"] = income["bin_no"] * 10000
    income = income.query("`Mean bin income` < 100001")
    agg_df = income.groupby(["GEO","Year (2)", "bin_no",]).sum(numeric_only=True).reset_index()
    fig = px.bar(agg_df.query("`Year (2)`==2015"), x="bin_no", y="VALUE", facet_col="GEO")
    for annot in fig.layout.annotations:
        new_text = annot["text"].split("=")[1]
        annot["text"] = new_text
        annot["textangle"] = -30
        annot["xanchor"] = "left"
    st.plotly_chart(fig)
    st.markdown("""
        This data was used to fit a `gamma` probability distribution to it. 
                Incomes $> 100.000\ CAD $ were excluded due to uneven bin size.
                See the following figure for the fit vs. the data regarding Canada.
        """)
    
    def gamma(x, a, b):
        return scistat.gamma.pdf(x, a, scale=1/b)

    x = agg_df.query("`Year (2)`==2015 and GEO=='Canada'")["bin_no"].values // 10000
    y = agg_df.query("`Year (2)`==2015 and GEO=='Canada'")["VALUE"] / agg_df.query("`Year (2)`==2015 and GEO=='Canada'")["VALUE"].sum()

    x1 = np.linspace(min(x), max(x),100)
    fig, ax = plt.subplots()

    ax.plot(x*10000,y, label="Canadian income PDF")
    ax.plot(x1*10000,gamma(x1,2.30603102, 0.38960872), label="gamma fit")
    ax.set_xlabel("Income")
    ax.set_ylabel("Probability")
    ax.legend()
    st.pyplot(fig)

    st.markdown("# Energy consumption")

    
    provinces = st.multiselect("select provinces", all_provinces, all_provinces[:2])
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
    for i, geo in enumerate(provinces):
        fig.layout.annotations[i]["text"] = geo
        fig.layout.annotations[i]["textangle"] = -30
        fig.layout.annotations[i]["xanchor"] = "left"

    fig.update_layout(
        margin={"t": 100}, width=900, yaxis=dict(title="Energy consumption (GJ)")
    )
    st.plotly_chart(fig)
    fig = px.scatter(
        energy_consumption.query(
            "`Energy consumption`=='Gigajoules per household' and GEO in @provinces"
        ),
        x="Household income",
        y="VALUE",
        color="Energy type",
        facet_col="GEO",
        symbol="REF_DATE",
    )
    for i, geo in enumerate(provinces):
        fig.layout.annotations[i]["text"] = geo
        fig.layout.annotations[i]["textangle"] = -30
        fig.layout.annotations[i]["xanchor"] = "left"

    fig.update_layout(
        margin={"t": 100},
        width=900,
        yaxis=dict(title="Energy consumption (GJ/household)", matches=None),
    )
    st.plotly_chart(fig)

    st.markdown("# Heating technology distribution")

    all_entailing_groups_faf = [
        "Electric forced air furnace",
        "Natural gas forced air furnace",
        "Oil forced air furnace",
        "Wood or wood pellets forced air furnace",
        "Propane forced air furnace",
        "Other fuel forced air furnace",
    ]
    all_entailing_groups_heating_stoves = [
        "Electric heating stove",
        "Natural gas heating stove",
        "Oil heating stove",
        "Wood heating stove",
        "Propane heating stove",
        "Other fuel heating stove",
    ]
    all_entailing_groups_boilers = [
        "Electric boiler with hot water or steam radiators",
        "Natural gas boiler with hot water or steam radiators",
        "Oil boiler with hot water or steam radiators",
        "Wood boiler with hot water or steam radiators",
        "Propane boiler with hot water or steam radiators",
        "Other fuel boiler with hot water or steam radiators",
    ]

    st.markdown("## Technologies")

    fig = px.area(
        heating_systems.query(
            "`Primary heating system and type of energy` in @all_techs"
        ),
        x="REF_DATE",
        y="VALUE",
        color="Primary heating system and type of energy",
        facet_col="GEO",
    )
    for i, geo in enumerate(heating_systems["GEO"].unique()):
        fig.layout.annotations[i]["text"] = geo
        fig.layout.annotations[i]["textangle"] = -30
        fig.layout.annotations[i]["xanchor"] = "left"

    fig.update_layout(width=900, margin_t=100, yaxis_title="%")
    st.plotly_chart(fig)
    st.markdown("## Fuels")
    fig = px.area(
        heating_systems.query(
            "`Primary heating system and type of energy` in @all_fuels"
        ),
        x="REF_DATE",
        y="VALUE",
        color="Primary heating system and type of energy",
        facet_col="GEO",
    )
    for i, geo in enumerate(heating_systems["GEO"].unique()):
        fig.layout.annotations[i]["text"] = geo
        fig.layout.annotations[i]["textangle"] = -30
        fig.layout.annotations[i]["xanchor"] = "left"

    fig.update_layout(width=900, margin_t=100, yaxis_title="%")
    st.plotly_chart(fig)

    st.markdown(
        """
        ### Derive 'simplified' heating technologies
        Since the more granular data (i.e. '<FUEL_NAME> forced air furnance') 
        are often not available, technology shares have been derived from the
        fuel shares. `Propane` and `Natural gas` are grouped as a `Gas furnance`
        , `Wood or wood pellets` becomes a `Biomass urnance` and `Oil` becomes 
        an `Oil furnance`. For these technologies the difference between it 
        being a `Forced air furnance` or a `Boiler` is negligible in terms of 
        efficiency.

        The picture is different however, when regarding electricity. While 
        `Heat pumps` have an efficiency of $\eta>2$ for most of the year, all 
        other heating technologies have an efficiency of $<1$. Hence, the heat 
        pump share from above is used, and subtracted from the electricity share 
        to represent other electricity powered appliances.
        """
    )
    
    simplified_heating_systems_long = simplified_heating_systems.melt(ignore_index=False).reset_index()

    fig = px.area(simplified_heating_systems_long, x="REF_DATE", y="value", color="variable", facet_col="GEO")

    for i, geo in enumerate(simplified_heating_systems_long["GEO"].unique()):
        fig.layout.annotations[i]['text'] = geo
        fig.layout.annotations[i]['textangle'] = -30
        fig.layout.annotations[i]['xanchor'] = "left"


    fig.update_layout(width=900, margin_t=100, yaxis_title="%")
    st.plotly_chart(fig)

    # this code is to show that more fine grained analysis results in less complete data
    # appliances_group_map = {"Forced air furnance": "Forced air furnance", 
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
