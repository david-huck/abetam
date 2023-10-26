import pandas as pd
import streamlit as st
import plotly.express as px

# this is exemplary for the location of vancouver

# data from jrc for each centroid of province
# _jrc_url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat=49.220&lon=-123.055&raddatabase=PVGIS-NSRDB&browser=1&outputformat=csv&userhorizon=&usehorizon=1&angle=&aspect=&startyear=2013&endyear=2013&mountingplace=free&optimalinclination=0&optimalangles=1&js=1&select_database_hourly=PVGIS-NSRDB&hstartyear=2013&hendyear=2013&trackingtype=0&hourlyoptimalangles=1&pvcalculation=1&pvtechchoice=crystSi&peakpower=1&loss=14&components=1"
province_temperatures = pd.read_csv("data/canada/CA_provinces_temperatures.csv")
# ensure temperature is in Kelvin and convert timestamp
province_temperatures["time"] = pd.to_datetime(
    province_temperatures["time"].values, format="%Y%m%d:%H%M"
)
province_temperatures.set_index("time", inplace=True)

_max_norm_T = pd.read_csv("data/canada/CA_provinces_max_norm_T.csv", index_col=0)


# timeshift according to location
def shift_dataframe(df, delta_t):
    assert "time" in df.columns
    start = province_temperatures["time"][0]
    end = province_temperatures["time"].max()

    future_tail = province_temperatures.iloc[:delta_t, :]
    new_df = province_temperatures.iloc[delta_t:, :]
    new_df = pd.concat([new_df, future_tail], ignore_index=True)
    new_df["time"] = pd.date_range(start, end, freq="h")
    return new_df


# @st.cache_data
def normalize_temperature_diff(t_outside: pd.Series, T_set_K=293.15):
    delta_t = T_set_K - t_outside
    delta_t[delta_t < 0] = 0
    nomalized_delta_t = delta_t / delta_t.sum()
    return nomalized_delta_t


# @st.cache_data
def determine_heat_demand_ts(
    annual_heat_demand: float, T_set: int = 20, province="Canada"
):
    t_outside = province_temperatures[province]

    # transform to Kelvin if not already in Kelvin
    if T_set < 100:
        T_set += 273.15

    if any(t_outside < 0):
        t_outside = t_outside.copy()
        t_outside += 273.15

    normalised_T2m = normalize_temperature_diff(t_outside, T_set)

    heat_demand_ts = normalised_T2m * annual_heat_demand

    # temperatures change faster than actual heat demand
    heat_demand_ts = heat_demand_ts.rolling(6, min_periods=1, center=True).mean()
    return heat_demand_ts


def necessary_heating_capacity(
    annual_heat_demand,
    province="Canada",
    security_factor=1.2,
    T_set: int = 20,
):
    demand_ts = determine_heat_demand_ts(annual_heat_demand, T_set, province)
    # allow for some error, by multiplying by a security factor > 1
    return demand_ts.max() * security_factor


def necessary_heating_capacity_for_province(
    annual_heat_demand, T_set=20, province="Canada", security_factor=1.2
):
    if province not in _max_norm_T.columns:
        raise NotImplementedError(
            f"This currently only works for {_max_norm_T.columns}, not for {province}.")
    
    if T_set + 273.15 in _max_norm_T.index:
        max_norm_T = _max_norm_T.loc[T_set+ 273.15, province]
        return annual_heat_demand * max_norm_T * security_factor

    return necessary_heating_capacity(
        annual_heat_demand, T_set=T_set, province=province, security_factor=security_factor
    )


def run():
    st.markdown("# Time series data")
    st.markdown(
        """Data from the jrc.eu is downloaded per location. This entails the 
                outdoor temperature, which will be converted to the heat demand by 
                calculating the difference between one of the temperature time series 
                from below and a set temperature $T_{set}$.
                """
    )
    T_fig = px.line(province_temperatures)
    T_fig.update_layout(yaxis_title="Outdoor temperature (°C)")
    st.plotly_chart(T_fig)

    provinces = province_temperatures.columns

    province = st.selectbox(
        "Select a province",
        provinces,
    )

    set_temperature = st.slider("Select a set temperature", 16, 24, 20)
    province_temperatures_display = province_temperatures.copy()
    province_temperatures_display[province + "_K"] = province_temperatures_display[province] + 273.15
    province_temperatures_display["T_diff"] = set_temperature - province_temperatures_display[province]
    province_temperatures_display["T_set"] = set_temperature
    ax = province_temperatures_display[[province, "T_diff", "T_set"]].plot()
    ax.legend()
    ax.set_ylabel("Temperature (°C)")
    st.pyplot(ax.get_figure())

    st.markdown(
        r"""Since heat demand is the main interest here, the temperature difference is set to 0 for all negative temperature differences. 
        Changing the total heat demand only changes the magnitude of this curve, whereas 
        the shape is the same as the temperature difference.
        
$$
T_\text{diff,t}=\left\{
\begin{array}{ll}
T_\text{set} - T_\text{O,t}  &\text{if }T_\text{set} - T_\text{O,t} > 0 \\ 
0 &\text{otherwise}.
\end{array} 
\right.
$$

$Q_{D,t} = T_\text{diff,t} \cdot \frac{Q_{D,a}}{\sum_t {T_\text{diff,t}}}$

        """
    )
    final_heat_demand = st.slider("Final heat demand", 18000, 40000, 20000)

    province_temperatures_display["heat_demand"] = determine_heat_demand_ts(
        final_heat_demand, T_set=set_temperature, province=province
    )
    # time step responsible for appliance sizing:
    t_cap_size = province_temperatures_display["heat_demand"].idxmax()
    appliance_size = necessary_heating_capacity_for_province(
        final_heat_demand, set_temperature, province=province
    )
    ax = province_temperatures_display[["heat_demand"]].plot()
    
    ax.scatter(t_cap_size, appliance_size, label="Built heating Capacity", color="red")
    ax.set_ylabel("Heat demand (kWh)")
    ax.legend()
    st.pyplot(ax.get_figure())
    st.markdown(r"""
                The `Built heating capacity` is derived as $C_{T} = max(Q_{D,t})\cdot 1.2$, where `1.2` is a 20\% safety margin.
                The resulting fuel demand is $F_{D,t} = \frac{Q_{D,t}}{\eta_T}$
                """)


if __name__ == "__main__":
    run()
