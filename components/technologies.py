from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class HeatingTechnology:
    name: str
    specific_cost: float
    specific_fuel_cost: float
    specific_fuel_emission: float
    efficiency: float
    lifetime: int
    age: int = 0

    @classmethod
    def from_series(cls, series, existing=True):
        params = list(cls.__match_args__)[1:]
        if existing:
            age = np.random.choice(int(series.loc["lifetime"]))
        else:
            age = 0
        name = series.name
        series = pd.concat([series, pd.Series({"age":age})])
        series.name = name
        values_row = series[params]
        assert series.name is not None
        return HeatingTechnology(series.name, **values_row.to_dict())
    
    def total_cost_per_year(self, heating_demand, discount_rate=0.07):
        fuel_cost = heating_demand / self.efficiency * self.specific_fuel_cost
        annuity_factor = discount_rate/(1-(1+discount_rate)**-self.lifetime)

        # TODO: calculate size appropriately
        size = 20 # kW
        annuity_payment = size * annuity_factor
        fom_cost = annuity_payment * 0.02
        return  annuity_payment + fuel_cost + fom_cost
