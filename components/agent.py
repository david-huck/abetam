import mesa
import numpy as np
import pandas as pd
from functools import partial
from numba import jit
from numba.typed import List

from components.technologies import HeatingTechnology
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
        installed_heating_tech,
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
        self.heat_demand_ts = determine_heat_demand_ts(annual_heating_demand, province=model.province)
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
        self.pbc = (
            self.random.random()
        )  # beta_with_mode_at(pbc_mode, 1, interval=(0, 1))
        self.heat_techs_df = self.model.heating_techs_df.copy()

        self.heat_techs_df["annual_cost"] = HeatingTechnology.annual_cost_from_df(
            self.heat_demand_ts, self.model.heating_techs_df, province=self.model.province
        )

    def step(self):
        """called each `step´ of the model.
        This is how the model progresses through time"""
        self.update_annual_costs()
        self.heating_tech.age += self.years_per_step
        if self.model.interact:
            self.interactions_this_step = 0
            self.interact()
        # self.peer_effect()
        self.wealth += (
            self.disposable_income
            - self.heating_tech.total_cost_per_year(self.heat_demand)
        ) * self.years_per_step

        reason, adopted_tech = self.check_adoption_decision()
        self.adopted_technologies = {"tech": adopted_tech, "reason": reason}.copy()

    def update_annual_costs(self):
        # TODO: this only really needs to be called right before an agent
        # makes a decision. which might reduce runtime
        if self.model.current_year % 1 > 0:
            return
        self.heat_techs_df["annual_cost"] = HeatingTechnology.annual_cost_from_df(
            self.heat_demand_ts, self.model.heating_techs_df, province=self.model.province
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

        purchased_tbp = False
        if self.heating_tech.age > self.heating_tech.lifetime * 3 / 4:
            purchased_tbp = self.purchase_heating_tpb_based()
            # Attidude change due to pre-/post purchase good expectation/experience
            # if self.tech_attitudes[self.heating_tech.name] + 0.1 < 1:
            #     self.tech_attitudes[self.heating_tech.name] += 0.1

        if purchased_tbp:
            reason = "tbp"
            adopted_tech = self.heating_tech.name
        else:
            # Failure probability = inverse of lifetime (appliance/year * years_per_step(1/4))
            prob_failure = 1 / self.heating_tech.lifetime * self.years_per_step
            if prob_failure > self.random.random():
                # Attidude change due to pre-/post failure bad expectation/experience
                # if self.tech_attitudes[self.heating_tech.name] - 0.1 > -1:
                #     self.tech_attitudes[self.heating_tech.name] -= 0.1

                self.purchase_new_heating()
                reason = "mcda"
                adopted_tech = self.heating_tech.name

        return reason, adopted_tech

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
        pass

    def purchase_heating_tpb_based(self):
        # order attitude dict by value, descending
        sorted_atts = sorted(
            self.tech_attitudes.items(), key=lambda it: it[1], reverse=True
        )
        for tech_name, tech_att in sorted_atts:
            # TODO: at least a sensitivity analysis for arbitrary value
            if tech_att > 0.5:
                if self.random.random() < self.pbc:
                    self.heating_tech = HeatingTechnology.from_series(
                        self.heat_techs_df.loc[tech_name, :], existing=False
                    )
                    return True
            else:
                # tech_att will not be > threshold for following items
                # since dict is sorted
                return False

        # if loop ended, no adoption took place
        return False

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
        if (inc_similarity*0.8 + att_similarity*0.2) > 0.7:
        # if inc_similarity > 0.7:
            similar_neighbors += 1

    # 50% of neighbors should have a similarity_index > 0.7
    desired_num_similar_neighbors = len(others_income) * 1/3
    should_move = similar_neighbors < desired_num_similar_neighbors
    return should_move
