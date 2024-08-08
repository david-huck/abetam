import mesa
import numpy as np
import pandas as pd
from functools import partial
from numba import jit
from numba.typed import List
from copy import copy

from components.technologies import HeatingTechnology, Technologies
from components.probability import beta_with_mode_at

from decision_making.mcda import calc_score, normalize
from decision_making.attitudes import simple_diff

from data.canada.timeseries import (
    determine_heat_demand_ts,
    necessary_heating_capacity_for_province,
)


class HouseholdAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(
        self,
        unique_id,
        model,
        disposable_income,
        installed_heating_tech: HeatingTechnology,
        annual_heating_demand,
        installed_pv_cap=0,
        years_per_step=1 / 4,
        tech_attitudes=None,
        criteria_weights=None,
        ts_step_length="h",
        hp_subsidy=0.0,
        fossil_ban_year=None,
        utility_threshhold=0.2,
    ):
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        # Create the agent's attribute and set the initial values.
        self.wealth = 0
        self.years_per_step = years_per_step
        self.ts_step_length = ts_step_length
        self.disposable_income = disposable_income * years_per_step
        self.heat_demand = annual_heating_demand
        self.heating_tech = installed_heating_tech
        self.hp_subsidy = hp_subsidy
        self.fossil_ban_year = fossil_ban_year
        self.heat_demand_ts = determine_heat_demand_ts(
            annual_heating_demand,
            province=model.province,
            ts_step_length=ts_step_length,
        )
        available_techs = self.model.heating_techs_df.index
        self.adopted_technologies = {
            "tech": None,
            "annual_costs": None,
            "purchase_price": None,
        }.copy()
        if tech_attitudes is None:
            tech_attitudes = dict(
                zip(available_techs, np.random.random(len(available_techs)))
            )
        self.tech_attitudes = tech_attitudes
        if criteria_weights is None:
            criteria_weights = {
                "emissions_norm": 0.3,
                "cost_norm": 0.4,
                "attitude_norm": 0.3,
            }
        self.utility_threshhold = utility_threshhold
        self.criteria_weights = criteria_weights
        self.att_inertia = self.random.random()
        self.heat_techs_df = self.model.heating_techs_df.copy()
        self.hp_eff_boost = 0
        self.is_refurbished = False
        self.was_refurbished = False
        self.current_cost_components = None
        self.update_demands(annual_heating_demand)
        self.annual_costs = self.heat_techs_df["annual_cost"].to_dict()
        self.specific_hp_cost = (
            self.model.heating_techs_df["specific_cost"].to_dict().copy()
        )
    
    @property
    def heating_tech_name(self):
        return str(self.heating_tech.name)

    def refurbish(self, demand_reduction):
        if self.is_refurbished:
            raise RuntimeError(
                f"{self.model.current_year}: {self.unique_id} is already refurbished."
            )
        self.is_refurbished = True
        refurbed_demand_frac = 1 - demand_reduction
        new_demand = self.heat_demand * refurbed_demand_frac
        hp_eff_boost = 183.773 * (refurbed_demand_frac - 1) ** 2 + 0
        hp_eff_boost /= 100
        self.hp_eff_boost = hp_eff_boost
        self.update_demands(new_demand, hp_eff_incr=hp_eff_boost)

    def update_demands(self, new_annual_demand, hp_eff_incr=0):
        self.heat_demand = new_annual_demand
        self.heat_demand_ts = (
            new_annual_demand * self.heat_demand_ts / self.heat_demand_ts.sum()
        )
        size = necessary_heating_capacity_for_province(
            self.heat_demand, province=self.model.province
        )
        self.req_heating_cap = size
        cost_components, fuel_demands = (
            HeatingTechnology.annual_cost_from_df_fast(
                self.heat_demand_ts,
                self.model.heating_techs_df,
                province=self.model.province,
                ts_step_length=self.ts_step_length,
                hp_eff_incr=hp_eff_incr,
                size=size,
                hp_subsidy=self.hp_subsidy,
            )
        )
        self.cost_components = cost_components

        self.lcoh = (self.cost_components.sum(axis=1)/self.heat_demand).to_dict()
        self.heat_techs_df["annual_cost"] = cost_components.sum(axis=1)
        self.annual_costs = cost_components.sum(axis=1).to_dict().copy()
        self.potential_fuel_demands = fuel_demands
        self.current_fuel_demand = fuel_demands[self.heating_tech.name]
        self.current_cost_components = cost_components.loc[self.heating_tech.name,:].to_dict()

    def step(self):
        """called each `step´ of the model.
        This is how the model progresses through time"""
        self.heating_tech.age += self.years_per_step

        # if appliance surpasses lifetime, stop annuity payments
        if self.heating_tech.age >= self.heating_tech.lifetime:
            self.current_cost_components["annuity_cost"] = 0

        adopted_tech, annual_costs, purchase_price = self.check_adoption_decision()
        self.adopted_technologies = {
            "tech": adopted_tech,
            "annual_costs": annual_costs,
            "purchase_price": purchase_price,
        }.copy()

        if self.heating_tech is None:
            raise RuntimeError(
                f"{self.model.current_year}: Agent {self.unique_id} has no heating technology."
            )

    def update_annual_costs(self):
        self.cost_components = (
            HeatingTechnology.annual_cost_with_fuel_demands(
                self.heat_demand_ts,
                self.potential_fuel_demands,
                self.model.heating_techs_df,
                province=self.model.province,
            )
        )
        self.heat_techs_df["annual_cost"] = self.cost_components.sum(axis=1)
        self.annual_costs = self.heat_techs_df["annual_cost"].to_dict().copy()
        self.lcoh = (self.cost_components.sum(axis=1)/self.heat_demand).to_dict()

        self.specific_hp_cost = self.model.heating_techs_df["specific_cost"].to_dict().copy()
        self.current_fuel_demand = self.potential_fuel_demands[self.heating_tech.name]
        self.current_cost_components = self.cost_components.loc[self.heating_tech.name,:].to_dict()

    def check_adoption_decision(self):
        """Check if agent should adopt a new heating technology based on
        the Theory of Planned Behavior (TPB)
        
        Returns:
            (reason, adopted_tech): Tuple with reason for adoption and name of adopted tech
        """
        adopted_tech = None
        annual_costs = 0
        purchase_price = 0

        prob_failure = 1 / self.heating_tech.lifetime * self.years_per_step
        adoption_was_necessary = prob_failure > self.random.random()

        if adoption_was_necessary:
            self.update_annual_costs()
            self.purchase_heating_tpb_based(necessary=adoption_was_necessary)
            
            adopted_tech = self.heating_tech.name
            purchase_price = self.heat_techs_df.loc[adopted_tech, "specific_cost"]
            annual_costs = self.heat_techs_df.loc[adopted_tech, "annual_cost"]
            self.current_fuel_demand = self.potential_fuel_demands[self.heating_tech.name]
            self.current_cost_components = self.cost_components.loc[self.heating_tech.name,:].to_dict()

        elif self.heating_tech.lifetime - self.heating_tech.age < 5:
            self.update_annual_costs()
            self.purchase_heating_tpb_based(necessary=adoption_was_necessary)
            
            adopted_tech = self.heating_tech.name
            purchase_price = self.heat_techs_df.loc[adopted_tech, "specific_cost"]
            annual_costs = self.heat_techs_df.loc[adopted_tech, "annual_cost"]
            self.current_fuel_demand = self.potential_fuel_demands[self.heating_tech.name]
            self.current_cost_components = self.cost_components.loc[self.heating_tech.name,:].to_dict()

        return adopted_tech, annual_costs, purchase_price, adoption_was_necessary

    def calc_scores(
        self,
    ):
        techs_df = self.heat_techs_df.loc[self.model.available_techs, :]
        techs_df["attitude"] = self.tech_attitudes
        techs_df["attitude_norm"] = normalize(techs_df["attitude"])

        # calculate scores
        p_normalize = partial(normalize, direction=-1)
        techs_df.loc[:, ["cost_norm"]] = (
            techs_df[["annual_cost"]].apply(p_normalize).values
        )
        techs_df["total_score"] = techs_df[
            ["emissions_norm", "cost_norm", "attitude_norm"]
        ].apply(
            calc_score,
            axis=1,
            weights=self.criteria_weights,
        )
        return techs_df

    def purchase_heating_tpb_based(self, necessary=False):
        # get utilities (scores) of techs
        scores = self.calc_scores()

        # Removing fossil appliances post self.fossil_ban_year
        # calc_scores() yields `nan` entries for it
        # doesn't actually remove them, but the adoption likelyhood is 0
        scores = scores.fillna(0)
        peer_tech_shares = self.peer_tech_shares()

        peer_tech_shares_sr = pd.DataFrame(peer_tech_shares, index=[0]).T[0]
        total_scores = scores["total_score"]
        utilities = 0.8*total_scores + 0.2 * peer_tech_shares_sr

        
        # keep techs with utilities above threshold
        above_t_utilities = utilities[utilities>self.utility_threshhold]

        if above_t_utilities.empty:
            if necessary:
                # take best technology anyway
                chosen_tech = utilities.index[utilities.argmax()]
            else:
                # if no techs above threshold and it is not necessary (tech still working), do not adopt. return False
                return False
        else:
            # choose randomly from techs with close utility
            utility_indifference = 0.03
            utility_difference = above_t_utilities.max() - above_t_utilities        
            util_techs = utility_difference[utility_difference< utility_indifference]
            chosen_tech = np.random.choice(util_techs.index, 1)[0]
        try:
            self.heating_tech = HeatingTechnology.from_series(
                    self.heat_techs_df.loc[chosen_tech, :], existing=False
                )
        except Exception as e:
            le_problem = f"\n{utility_difference=}"+ f"\n{util_techs=}" + f"\n{chosen_tech=}"
            e.args += (le_problem,)
            raise e
        return True


    def peer_tech_shares(self):
        """Calculates percentage of peers that own `tech_name`.

        Args:
            tech_name (str): name of the technology

        Returns:
            share (float): percentage of peers with `tech_name`
        """
        tech_shares = dict(zip(Technologies, [0] * len(Technologies)))
        for n in self.peers:
            tech_shares[n.heating_tech.name] += 1 / len(self.peers)
        return tech_shares

    def move_or_stay_check(self, radius=6):
        """Used in self.model.perform_segregation to move similar agents
        close to each other other on the grid.

        Args:
            radius (int, optional): radius for neighbor determination. Defaults to 6.
        """

        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, radius=radius)

        atts = [list(a.tech_attitudes.values()) for a in neighbors]
        incs = np.array([a.disposable_income for a in neighbors])

        should_move = move_or_stay_decision(
            List(self.tech_attitudes.values()),
            np.array(atts, dtype=np.float32),
            self.disposable_income,
            incs,
        )

        # move if count of similar_neighbors is smaller than desired number of similar neighbors
        if should_move:
            self.model.grid.move_to_empty(self)


@jit(nopython=True)
def income_similarity(self_income, other_income):
    larger = max(self_income, other_income)
    smaller = min(self_income, other_income)

    income_ratio = smaller / larger
    return income_ratio


@jit(nopython=True)
def attitude_similarity(att0, att1):
    assert len(att0) == len(att1)

    ratio_sum = 0.0
    for i in range(len(att0)):
        r = (att1[i] + 1) / (att0[i] + 1)
        if r > 1:
            r = 1 / r
        ratio_sum += r

    return ratio_sum / len(att0)


@jit(nopython=True)
def move_or_stay_decision(
    self_attitude: list,
    others_attitudes: np.array,
    self_income: float,
    others_income: list[float],
):
    similar_neighbors = 0

    for i in range(others_attitudes.shape[0]):
        inc_similarity = income_similarity(self_income, others_income[i])
        n_att = others_attitudes[i]
        att_similarity = attitude_similarity(self_attitude, n_att)
        if (inc_similarity * 0.8 + att_similarity * 0.2) > 0.7:
            # if inc_similarity > 0.7:
            similar_neighbors += 1

    # 50% of neighbors should have a similarity_index > 0.7
    desired_num_similar_neighbors = len(others_income) * 1 / 3
    should_move = similar_neighbors < desired_num_similar_neighbors
    return should_move
