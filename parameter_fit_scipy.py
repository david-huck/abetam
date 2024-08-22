from scipy.optimize import minimize
import pandas as pd
import numpy as np
import pickle as pkl
from data.canada import nrcan_tech_shares_df
from skopt import Optimizer
from skopt.space import Real
from joblib import Parallel, delayed
from batch import BatchResult
import warnings
from functools import partial
import os

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"


def diff_btwn_model_historic(
    att_modes_array, N=500, province="Ontario", p_mode=0.6, peer_eff=0.2
):
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
        "ts_step_length": ["W"],
        "peer_effect_weight": [peer_eff],
    }
    b_result = BatchResult.from_parameters(batch_parameters, display_progress=False)
    model_shares = (
        b_result.tech_shares_df.groupby(["province", "year"])
        .mean()
        .drop("RunId", axis=1)
    )

    diff = (h_tech_shares - model_shares.loc[(province, list(range(2000,2021))), :]).loc[
        province, :
    ]
    abs_diff = diff.abs().sum().sum()
    # print(f"{abs_diff=:.3f}")
    return abs_diff


def flatten_table(table):
    return table.values.flatten()


def flat_array_2_table(array):
    cols = [
        "Electric furnace",
        "Gas furnace",
        "Heat pump",
        "Oil furnace",
        "Wood or wood pellets furnace",
    ]
    return pd.DataFrame(array.reshape((21, 5)), columns=cols, index=range(2000,2021))


def run_optimisation(
    p_mode,
    peer_eff,
):
    fit_func = partial(diff_btwn_model_historic, p_mode=p_mode, peer_eff=peer_eff)
    optimizer = Optimizer(
        dimensions=bounds,
        random_state=1,
        base_estimator="gp",
    )
    i = 0
    results = list(range(40, 45))
    delta = 1
    no_impr_steps = 0
    best_result = 1e3
    while delta > 1e-2 and i < 40 and no_impr_steps < 10:
        x = optimizer.ask(n_points=4)
        y = Parallel(n_jobs=4)(delayed(fit_func)(np.array(v)) for v in x)
        optimizer.tell(x, y)
        results.pop(0)
        results.append(min(y))
        delta = np.diff(sorted(results)[:3]).mean()
        if min(optimizer.yi) < best_result:
            best_result = min(optimizer.yi)
            no_impr_steps = 0
        else:
            no_impr_steps += 1
        print(f"{i=:02}", f"{p_mode=:.2f}, {peer_eff=:.2f}, best={best_result:.2f}, {delta=:.2f}")
        i += 1
    print(f"{p_mode=:.2f}, {peer_eff=:.2f} terminated.\n\t {(delta > 1e-2)=} and {(i < 40)=} and {(no_impr_steps < 10)=}")
    pkl.dump(
        optimizer.yi, open(f"optim.yi_pm_{p_mode:.2f}_pe_{peer_eff:.2f}.pkl", "wb")
    )
    pkl.dump(
        optimizer.Xi, open(f"optim.Xi_pm_{p_mode:.2f}_pe_{peer_eff:.2f}.pkl", "wb")
    )


if __name__ == "__main__":
    print("Starting ",__file__)
    historic_tech_shares = nrcan_tech_shares_df.copy()
    historic_tech_shares.index = historic_tech_shares.index.swaplevel()

    province = "Ontario"

    h_tech_shares = historic_tech_shares.loc[province, :] / 100
    att_mode_table = h_tech_shares.copy()

    full_years = range(2000, 2021)
    x0 = flatten_table(att_mode_table)
    bounds = [*[Real(0.05, 0.95)] * len(x0)]

    Parallel(n_jobs=5)(
        delayed(run_optimisation)(p_mode, peer_eff)
        for p_mode in np.arange(0.6, 0.8, 0.05)
        for peer_eff in [0.15, 0.2, 0.25, 0.3, 0.35]
    )
