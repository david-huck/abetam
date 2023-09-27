import pandas as pd

def drop_redundant_cols(df):
    cols_to_drop = []
    for col in df.columns:
        if len(df[col].unique()) < 2:
            cols_to_drop.append(col)
    df.drop(cols_to_drop, axis=1, inplace=True)


household_expenditures = pd.read_csv("data/canada/1110022401_databaseLoadingData.csv")

energy_consumption = pd.read_csv("data/canada/2510006201_databaseLoadingData.csv")
energy_consumption["Household income"] = energy_consumption["Household income"].str.removesuffix("(includes income loss)")


for df in [household_expenditures, energy_consumption]:
    drop_redundant_cols(df)


if __name__ == "__main__":
    import plotly.express as px
    # household expenditures
    fig = px.scatter(household_expenditures, x="REF_DATE", y="VALUE", color="Household expenditures, summary-level categories", facet_col="Household type")

    for i, hh_type in enumerate(household_expenditures["Household type"].unique()):
        fig.layout.annotations[i]['text'] = hh_type.lower().replace("households","")
        fig.layout.annotations[i]['textangle'] = -30
        fig.layout.annotations[i]['xanchor'] = "left"
        print(fig.layout.annotations[i])

    fig.update_layout(yaxis_title="Annual Expenses (CAD)", width=1300)
    fig.show()

    # Energy consumption
    fig = px.scatter(energy_consumption.query("`Energy consumption`=='Gigajoules'"), x="Household income", y="VALUE", color="Energy type", facet_col="GEO", symbol="REF_DATE")
    for i, geo in enumerate(energy_consumption["GEO"].unique()):
        fig.layout.annotations[i]['text'] = geo
        fig.layout.annotations[i]['textangle'] = -30
        fig.layout.annotations[i]['xanchor'] = "left"

    fig.update_layout(
        margin={
            "t": 100
        },
        width=1300,
        yaxis=dict(title="Energy consumption (GJ)")
    )

    fig = px.scatter(energy_consumption.query("`Energy consumption`=='Gigajoules per household'"), x="Household income", y="VALUE", color="Energy type", facet_col="GEO", symbol="REF_DATE")
    for i, geo in enumerate(energy_consumption["GEO"].unique()):
        fig.layout.annotations[i]['text'] = geo
        fig.layout.annotations[i]['textangle'] = -30
        fig.layout.annotations[i]['xanchor'] = "left"

    fig.update_layout(
        margin={
            "t": 100
        },
        width=1300,
        yaxis=dict(title="Energy consumption (GJ/household)",matches=None)
    )