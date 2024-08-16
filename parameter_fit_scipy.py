from scipy.optimize import minimize
import pandas as pd
import pickle as pkl
from data.canada import nrcan_tech_shares_df

from batch import BatchResult



def diff_btwn_model_historic(att_modes_array, N=100, province="Ontario", p_mode=0.5, peer_eff=0.2):
    att_modes = flat_array_2_table(att_modes_array)
    batch_parameters = {
        "N": [N],
        "province": [province],
        "random_seed": range(20, 25),
        "start_year": 2000,
        "tech_att_mode_table": [att_modes],
        "n_segregation_steps": [40],
        "interact": [False],
        "price_weight_mode": [p_mode],
        "ts_step_length":["W"],
        "peer_effect_weight": [peer_eff]
    }
    b_result = BatchResult.from_parameters(batch_parameters, display_progress=False)
    model_shares = (
        b_result.tech_shares_df.groupby(["province", "year"])
        .mean()
        .drop("RunId", axis=1)
    )
    full_years = range(2000, 2021)
    diff = (h_tech_shares - model_shares.loc[(province, full_years), :]).loc[
        province, :
    ]
    abs_diff = diff.abs().sum().sum()
    print(f"{abs_diff=:.3f}")
    return abs_diff

def flatten_table(table):
    return table.values.flatten()

def flat_array_2_table(array):
    cols = ['Electric furnace', 'Gas furnace', 'Heat pump', 'Oil furnace',
       'Wood or wood pellets furnace']
    return pd.DataFrame(array.reshape((21,5)), columns=cols)


if __name__ == "__main__":

    historic_tech_shares = nrcan_tech_shares_df.copy()
    historic_tech_shares.index = historic_tech_shares.index.swaplevel()

    province = "Ontario"

    h_tech_shares = historic_tech_shares.loc[province, :] / 100
    att_mode_table = h_tech_shares.copy()
    
    
    x0=flatten_table(att_mode_table)
    bounds = [*[(0.05, 0.95)]*len(x0)]
    
    methods = [
        "L-BFGS-B",
        "Powell"
        "trust-constr",
        "COBYLA", 
        "COBYQA",
        "SLSQP",
        "Nelder-Mead",
    ]
    for meth in methods:
        print("method", meth)
        res = minimize(diff_btwn_model_historic, x0=x0, bounds=bounds, options={"maxiter":20}, method=meth)
        pkl.dump(res, open(meth+".res.pkl", "wb"))