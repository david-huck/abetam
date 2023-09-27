import pandas as pd


def drop_redundant_cols(df):
    cols_to_drop = []
    for col in df.columns:
        if len(df[col].unique()) < 2:
            cols_to_drop.append(col)
    df.drop(cols_to_drop, axis=1, inplace=True)


household_expenditures = pd.read_csv("data/canada/1110022401_databaseLoadingData.csv")

energy_consumption = pd.read_csv("data/canada/2510006201_databaseLoadingData.csv")
energy_consumption["Household income"] = energy_consumption[
    "Household income"
].str.removesuffix("(includes income loss)")

heating_systems = pd.read_csv("data/canada/3810028601_databaseLoadingData.csv")

for df in [household_expenditures, energy_consumption, heating_systems]:
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

simple_appliance_df = _fuel_df.copy()

simple_appliance_df["Electric furnance"] =  simple_appliance_df["Electricity"] - _appliances_df["Heat pump"]
simple_appliance_df["Heat pump"] = _appliances_df["Heat pump"]
simple_appliance_df["Gas furnance"] = simple_appliance_df["Natural gas"] + simple_appliance_df["Propane"]
simple_appliance_df.drop(["Electricity","Natural gas","Propane"], axis=1, inplace=True)
simple_appliance_df.columns = [col + " furnace" if ("furnance" not in col and "pump" not in col) else col 
                            for col in simple_appliance_df.columns ]

if __name__ == "__main__":
    import plotly.express as px
    import streamlit as st

    st.set_page_config(page_title="Canadian Inputs")
    st.title("Input data from statcan")
    st.markdown("The data displayed here has been downloaded from [statcan](https://www150.statcan.gc.ca/n1/en/type/data?MM=1).")
    st.markdown("# household expenditures")
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

    st.markdown("# Energy consumption")

    all_provinces = list(energy_consumption["GEO"].unique())
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
    
    simple_appliance_df_long = simple_appliance_df.melt(ignore_index=False).reset_index()

    fig = px.area(simple_appliance_df_long, x="REF_DATE", y="value", color="variable", facet_col="GEO")

    for i, geo in enumerate(simple_appliance_df_long["GEO"].unique()):
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
