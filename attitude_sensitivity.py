import pandas as pd
from pathlib import Path
import git
import sys
from datetime import datetime
from functools import partial

root_dir = git.Repo().working_dir
abetam_dir = Path(root_dir) / "abetam"
copper_dir = Path(root_dir) / "copper"
sys.path.append(abetam_dir.as_posix())
# ruff: noqa: E402

from scenarios import (
    generate_scenario_attitudes,
    MODES_2020,
    FAST_TRANSITION_MODES_AND_YEARS,
    SLOW_TRANSITION_MODES_AND_YEARS,
    SLOW_mod_TRANSITION_MODES_AND_YEARS,
    MODERATE_MODES_AND_YEARS,
    NO_TRANSITION_MODES_AND_YEARS,
    update_price_w_new_CT,
    CT,
)
from data.canada import end_use_prices, nrcan_tech_shares_df
from batch import BatchResult


hp_subsidies = {
    "BAU": 0.0,
    "CER": 0.15,
    "CER_plus": 0.15,
    "Rapid": 0.30,
    "Rapid_plus": 0.30,
}

refurbishment_rate = {
    "BAU": 0.01,
    "CER": 0.02,
    "CER_plus": 0.02,
    "Rapid": 0.03,
    "Rapid_plus": 0.03,
}

carbon_tax_mod = {"BAU": 1, "CER": 1, "CER_plus": 1, "Rapid": 2, "Rapid_plus": 2}

emission_limit = {
    "BAU": False,
    "CER": False,
    "CER_plus": False,
    "Rapid": True,
    "Rapid_plus": True,
}
tech_share_as_att = nrcan_tech_shares_df.loc[2020,"Ontario"]/100
tech_share_as_att
SHARE_TRANSITION_MODES_AND_YEARS = {
    "Electric furnace": {"end_att": tech_share_as_att["Electric furnace"], "at_year": 2021},
    "Gas furnace": {"end_att": tech_share_as_att["Gas furnace"], "at_year": 2021},
    "Heat pump": {"end_att": tech_share_as_att["Heat pump"], "at_year": 2021},
    "Oil furnace": {"end_att": tech_share_as_att["Oil furnace"], "at_year": 2021},
    "Wood or wood pellets furnace": {"end_att": tech_share_as_att["Wood or wood pellets furnace"], "at_year": 2021},
}


ATTITUDE_SCENARIOS = {
    "no": NO_TRANSITION_MODES_AND_YEARS,
    "slow": SLOW_TRANSITION_MODES_AND_YEARS,
    "slow_m": SLOW_mod_TRANSITION_MODES_AND_YEARS,
    "moderate": MODERATE_MODES_AND_YEARS,
    "share": SHARE_TRANSITION_MODES_AND_YEARS,
    "fast": FAST_TRANSITION_MODES_AND_YEARS,
}


att_modes = {
    "BAU": MODERATE_MODES_AND_YEARS,  # SLOW_TRANSITION_MODES_AND_YEARS,
    "CER": MODERATE_MODES_AND_YEARS,  # SLOW_TRANSITION_MODES_AND_YEARS,
    "CER_plus": MODERATE_MODES_AND_YEARS,  # SLOW_TRANSITION_MODES_AND_YEARS,
    "Rapid": MODERATE_MODES_AND_YEARS,  # MODERATE_MODES_AND_YEARS,
    "Rapid_plus": MODERATE_MODES_AND_YEARS,  # MODERATE_MODES_AND_YEARS,
}

fossil_ban_years = {
    "BAU": None,
    "CER": None,
    "CER_plus": 2030,
    "Rapid": None,
    "Rapid_plus": 2026,
}

if __name__ == "__main__":
    print("Mode distributions:", att_modes)
    if len(sys.argv) > 1:
        scen_name = sys.argv[1]
    else:
        scen_name = "BAU"  # "BAU", "CER", "Rapid"
    print(f"=== Scenario: {scen_name} ===")
    results_dir = f"./results/att_sens/{scen_name}_" + datetime.now().strftime(
        r"%Y%m%d_%H%M"
    )
    
    scenario = (
        f"{scen_name}_scenario"
        if "Rapid" not in scen_name and scen_name != "CER_plus"
        else "CER_scenario"
    )

    # SCENARIO parameters for ABETAM
    p_mode = 0.7  # result of fit
    province = "Ontario"
    batch_parameters = {
        "N": [500],
        "province": [province],
        "random_seed": list(range(42, 48)),
        "n_segregation_steps": [40],
        "price_weight_mode": [p_mode],
        "ts_step_length": ["w"],
        "start_year": 2020,
        "refurbishment_rate": refurbishment_rate[scen_name],
        "hp_subsidy": hp_subsidies[scen_name],
        "fossil_ban_year": fossil_ban_years[scen_name],
    }

    if carbon_tax_mod[scen_name] != 1:
        new_CT = CT * carbon_tax_mod[scen_name]
        update_prices = partial(update_price_w_new_CT, new_CT=new_CT)
        end_use_prices["Price (ct/kWh)"] = end_use_prices[
            ["Year", "Price (ct/kWh)", "Type of fuel", "GEO"]
        ].apply(update_prices, axis=1)
        end_use_prices.to_csv(
            "data/canada/residential_GNZ_end-use-prices-2023_ct_per_kWh.csv",
            index=False,
        )

    # ensure electricity prices are reset before execution
    el_price_path = "data/canada/ca_electricity_prices.csv"
    el_prices_df = pd.read_csv(el_price_path).set_index("REF_DATE")
    el_prices_df = el_prices_df.loc[:2022, :]

    for att_desc, att_vals in ATTITUDE_SCENARIOS.items():
        print(f"Scenario: '{scen_name}', Attitudes: '{att_desc}', running ABM...")
        start_modes = MODES_2020 if att_desc != "test" else tech_share_as_att.to_dict()
        tech_attitude_scenario = generate_scenario_attitudes(
            start_modes, att_vals
        )
        batch_parameters["tech_att_mode_table"] = [tech_attitude_scenario]
        batch_result = BatchResult.from_parameters(
            batch_parameters, max_steps=(2050 - 2020) * 4, force_rerun=True
        )
        batch_result.save(custom_path=results_dir + f"_{att_desc}")

    pass
