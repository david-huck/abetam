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

from data.canada.timeseries import determine_heat_demand_ts


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
        interactions_per_step=1,
        years_per_step=1 / 4,
        tech_attitudes=None,
        criteria_weights=None,
        pbc_mode=0.7,
    ):
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        # Create the agent's attribute and set the initial values.
        self.wealth = 0
        self.interactions_this_step = 0
        self.interactions_per_step = interactions_per_step
        self.years_per_step = years_per_step
        self.disposable_income = disposable_income * years_per_step
        self.heat_demand = annual_heating_demand
        self.heating_tech = installed_heating_tech
        self.heat_demand_ts = determine_heat_demand_ts(
            annual_heating_demand, province=model.province
        )
        # self.fuel_demand_ts =
        available_techs = self.model.heating_techs_df.index
        self.adopted_technologies = {"tech": None, "reason": None}.copy()
        if tech_attitudes is None:
            tech_attitudes = dict(
                zip(available_techs, 2 * np.random.random(len(available_techs)) - 1)
            )
        self.tech_attitudes = tech_attitudes
        if criteria_weights is None:
            criteria_weights = {
                "emissions_norm": 0.3,
                "cost_norm": 0.4,
                "attitude": 0.3,
            }
        self.criteria_weights = criteria_weights
        self.att_inertia = self.random.random()
        self.tech_scores = None
        self.pbc = (
            self.random.random()
        )  # beta_with_mode_at(pbc_mode, 1, interval=(0, 1))
        self.heat_techs_df = self.model.heating_techs_df.copy()

        self.heat_techs_df["annual_cost"], fuel_demands = HeatingTechnology.annual_cost_from_df(
            self.heat_demand_ts,
            self.model.heating_techs_df,
            province=self.model.province,
        )
        self.potential_fuel_demands = fuel_demands
        self.current_fuel_demand = fuel_demands[self.heating_tech.name]

    def step(self):
        """called each `step´ of the model.
        This is how the model progresses through time"""
        self.heating_tech.age += self.years_per_step
        if self.model.interact:
            self.interactions_this_step = 0
            self.interact()

        reason, adopted_tech, tech_scores = self.check_adoption_decision()
        self.adopted_technologies = {"tech": adopted_tech, "reason": reason}.copy()
        if adopted_tech is not None:
            self.current_fuel_demand = self.potential_fuel_demands[self.heating_tech.name]

        self.tech_scores = copy(tech_scores)

    def update_annual_costs(self):
        # TODO: this only really needs to be called right before an agent
        # makes a decision. which might reduce runtime
        if self.model.current_year % 1 > 0:
            return
        self.heat_techs_df["annual_cost"] = HeatingTechnology.annual_cost_with_fuel_demands(
            self.heat_demand_ts,
            self.potential_fuel_demands,
            self.model.heating_techs_df,
            province=self.model.province,
        )

    def peer_effect(self):
        neighbours = self.model.grid.get_neighbors(self.pos, moore=True, radius=2)
        # calculate mean tech attitude among neighbors
        n_atts = dict(zip(self.tech_attitudes.keys(), [0] * len(self.tech_attitudes)))
        for n in neighbours:
            for tech, att in n.tech_attitudes.items():
                n_atts[tech] += att

        for k, v in n_atts.items():
            # apply fraction of that attitude to own attitude
            n_atts[k] = v / len(neighbours)
            new_att = simple_diff([self.tech_attitudes[k], n_atts[k]], inertia=0.9)
            self.tech_attitudes[k] = new_att

        pass

    def interact(self):
        """interaction with other agents.
        The interaction should have induce a change in the agents attitude towards
        technologies.
        """
        neighbours = self.model.grid.get_neighbors(self.pos, moore=True, radius=2)
        if len(neighbours) > 1:
            other = self.random.choice(neighbours)
            if any(
                x.interactions_this_step >= x.interactions_per_step
                for x in [self, other]
            ):
                return

            for tech in self.tech_attitudes.keys():
                att_self = self.tech_attitudes[tech]
                att_other = other.tech_attitudes[tech]
                new_att = simple_diff([att_self, att_other], inertia=self.att_inertia)

                self.tech_attitudes[tech] = new_att
                self.tech_attitudes = self.tech_attitudes.copy()
            self.interactions_this_step += 1
            other.interactions_this_step += 1
        else:
            # no neighbours to interact with
            return

    def check_adoption_decision(self):
        """Check if agent should adopt a new heating technology based on two ideas:

        1. Theory of Planned Behavior (TPB) adoption:
        If the current heating system is more than 3/4 through its lifetime,
        see if the agent *might* purchase a new system based on TPB.

        2. Multi-Criteria Decision Analysis (MCDA) adoption:
        If the current heating system has surpassed its lifetime, the agent
        must purchase a new system based on MCDA.

        The function returns a tuple with the reason for adoption ("tbp" or "mcda")
        and the name of the adopted technology.

        Returns:
            (reason, adopted_tech): Tuple with reason for adoption and name of adopted tech
        """
        reason = None
        adopted_tech = None
        tech_scores = None

        purchased_tbp = False

        prob_failure = 1 / self.heating_tech.lifetime * self.years_per_step
        if prob_failure > self.random.random():
            self.update_annual_costs()
            purchased_tbp = self.purchase_heating_tpb_based(necessary=True)
            # tech_scores = self.purchase_new_heating()
            reason = "tpb"
            adopted_tech = self.heating_tech.name

        return reason, adopted_tech, tech_scores

    def calc_scores(
        self,
    ):
        techs_df = self.heat_techs_df
        techs_df["attitude"] = self.tech_attitudes
        techs_df["attitude"] = normalize(techs_df["attitude"] + 1)
        
        # calculate scores
        p_normalize = partial(normalize, direction=-1)
        techs_df.loc[:, ["cost_norm"]] = (
            techs_df[["annual_cost"]].apply(p_normalize).values
        )
        techs_df["total_score"] = techs_df[
            ["emissions_norm", "cost_norm", "attitude"]
        ].apply(
            calc_score,
            axis=1,
            weights=self.criteria_weights,
        )
        return techs_df

    def purchase_new_heating(self):
        techs_df_w_score = self.calc_scores()
        best_tech_idx = techs_df_w_score["total_score"].argmax()
        new_tech = techs_df_w_score.iloc[best_tech_idx, :]

        # TODO: implement affordability constraint

        self.heating_tech = HeatingTechnology.from_series(new_tech, existing=False)
        return techs_df_w_score

    def purchase_heating_tpb_based_old(self):
        # order attitude dict by value, descending
        sorted_atts = sorted(
            self.tech_attitudes.items(), key=lambda it: it[1], reverse=True
        )
        for tech_name, tech_att in sorted_atts:
            # TODO: at least a sensitivity analysis for arbitrary value
            if tech_att > 0.5:
                # TODO: this might lead to the situation in which the lifetime of
                # an appliance has expired, but due to lacking pbc, no new appliance
                # is being bought
                # if self.random.random() < self.pbc:
                if self.neighbor_tech_shares()[tech_name] < self.pbc:
                    self.heating_tech = HeatingTechnology.from_series(
                        self.heat_techs_df.loc[tech_name, :], existing=False
                    )
                    return True

        # if loop ended, no adoption took place
        return False

    def purchase_heating_tpb_based(self, necessary=False):
        # order attitude dict by value, descending
        sorted_atts = sorted(
            self.tech_attitudes.items(), key=lambda it: it[1], reverse=True
        )
        # get utilities (scores) of techs
        scores = self.calc_scores()
        # calculate utility gains over current tech
        gains = (
            scores.loc[:, "total_score"] - scores.loc[self.heating_tech.name, "total_score"]
        ).to_dict()
        neighbor_tech_shares = self.neighbor_tech_shares()
        
        best_tech_score = -1
        best_tech_name = ""
        for tech_name, tech_att in sorted_atts:
            # if gain > threshold, buy tech
            tech_gain = gains[tech_name]
            peer_pressure = neighbor_tech_shares[tech_name]
            if peer_pressure + tech_gain > best_tech_score:
                best_tech_score = peer_pressure + tech_gain
                best_tech_name = tech_name
            # if self.unique_id % 50 == 0:
            #     print(self.unique_id,f"\t{self.pbc=},{tech_name}: {tech_gain=:.2f}, {peer_pressure=:.2f}")
            if self.model.global_util_thresh < tech_gain*.8 + peer_pressure*.2:
                self.heating_tech = HeatingTechnology.from_series(
                    self.heat_techs_df.loc[tech_name, :], existing=False
                )
                return True
        # if self.unique_id % 50 == 0:
        #     print(f"{best_tech_name=}: {best_tech_score=}")
        if necessary:
            self.heating_tech = HeatingTechnology.from_series(
                self.heat_techs_df.loc[best_tech_name, :], existing=False
            )
            return True

        # if loop ended, no adoption took place
        return False
    
    def neighbor_tech_shares(self, radius=4):
        """Calculates percentage of neighbors that own `tech_name`.

        Args:
            tech_name (str): name of the technology
            radius (int, optional): neigbor radius. Defaults to 4.

        Returns:
            share (float): percentage of neighbors with `tech_name`
        """
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, radius=radius)
        tech_shares = dict(zip(Technologies, [0]*len(Technologies)))
        for n in neighbors:
            tech_shares[n.heating_tech.name] += 1/len(neighbors)
        return tech_shares


    def move_or_stay_check(self, radius=6):
        """Used in self.model.perform_segregation to move similar agents
        close to each other other on the grid.

        Args:
            radius (int, optional): radius for neighbor determination. Defaults to 4.
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
