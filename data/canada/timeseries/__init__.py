import pandas as pd
import streamlit as st

# this is exemplary for the location of vancouver

# download data from jrc
# _jrc_url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat=49.220&lon=-123.055&raddatabase=PVGIS-NSRDB&browser=1&outputformat=csv&userhorizon=&usehorizon=1&angle=&aspect=&startyear=2013&endyear=2013&mountingplace=free&optimalinclination=0&optimalangles=1&js=1&select_database_hourly=PVGIS-NSRDB&hstartyear=2013&hendyear=2013&trackingtype=0&hourlyoptimalangles=1&pvcalculation=1&pvtechchoice=crystSi&peakpower=1&loss=14&components=1"
_rad_df = pd.read_csv(
    "data/canada/Timeseries_49.220_-123.055_NS_1kWp_crystSi_14_36deg_10deg_2013_2013.csv",
    header=8,
)

# remove meta_information
_rad_df = _rad_df.iloc[:-9, :]

# ensure temperature is in Kelvin and convert timestamp
_rad_df["T2m"] += 273.15
_rad_df["time"] = pd.to_datetime(_rad_df["time"].values, format="%Y%m%d:%H%M")


# timeshift according to location
def shift_dataframe(df, delta_t):
    assert "time" in df.columns
    _start = _rad_df["time"][0]
    _end = _rad_df["time"].max()

    _future_tail = _rad_df.iloc[:delta_t, :]
    _new_df = _rad_df.iloc[delta_t:, :]
    _new_df = pd.concat([_new_df, _future_tail], ignore_index=True)
    _new_df["time"] = pd.date_range(_start, _end, freq="h")
    return _new_df


_new_df = shift_dataframe(_rad_df, delta_t=8)

@st.cache_data
def normalize_temperatures(t_set = 20, t_outside: pd.Series = _new_df["T2m"]):
    delta_t = t_set - t_outside
    delta_t[delta_t < 0] = 0
    return delta_t/delta_t.sum()

@st.cache_data
def determine_heat_demand_ts(
    annual_heat_demand: float, t_set: int = 20, t_outside: pd.Series = _new_df["T2m"]
):
    # transform to Kelvin if not already in Kelvin
    if t_set < 100:
        t_set += 273

    normalied_T2m = normalize_temperatures(t_set, t_outside)

    heat_demand_ts = normalied_T2m * annual_heat_demand 

    # temperatures change faster than actual heat demand
    heat_demand_ts = heat_demand_ts.rolling(6, min_periods=1, center=True).mean()
    return heat_demand_ts

def determine_heating_capacity(annual_heat_demand, security_factor=1.2, t_set: int = 20, t_outside: pd.Series = _new_df["T2m"]):
    demand_ts = determine_heat_demand_ts(annual_heat_demand, t_set, t_outside)
    # allow for some error
    return demand_ts.max() * security_factor


def run():
    st.markdown("# Time series data")
    st.markdown(
        """Data from the jrc.eu is downloaded per location. This entails the 
                outdoor temperature, which will be converted to the heat demand by 
                calculating the difference to a set temperature $T_{set}$.
                """
    )

    set_temperature = st.slider("Set temperature", 16, 24, 20)

    _new_df["T2m_C"] = _new_df["T2m"] - 273.15
    _new_df["T_diff"] = set_temperature - _new_df["T2m_C"]

    ax = _new_df[["T2m_C", "T_diff"]].plot()
    ax.plot(_new_df.index, [set_temperature] * len(_new_df), label="T_set")
    ax.legend()
    st.plotly_chart(ax.get_figure())

    st.markdown(
        """Heat demand is set to 0 for all negative temperature differences. 
        Changing the total heat demand only changes the magnitude of this curve, whereas 
        the shape is the same as the temperature difference."""
    )
    final_heat_demand = st.slider("Final heat demand", 18000, 40000, 20000)

    _new_df["heat_demand"] = determine_heat_demand_ts(
        final_heat_demand, t_set=set_temperature
    )
    # _new_df.set_index("time", inplace=True)
    ax = _new_df[["heat_demand"]].plot()
    ax.set_ylabel("Heat demand (kWh)")
    st.pyplot(ax.get_figure())

if __name__ == "__main__":
    run()