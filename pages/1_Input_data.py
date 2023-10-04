from data.canada import run as run_canada_inputs
from data.canada import income
from data.canada.timeseries import run as run_canada_inputs_ts


run_canada_inputs(income)
run_canada_inputs_ts()