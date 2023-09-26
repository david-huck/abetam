import mesa
import numpy as np
from components.technologies import HeatingTechnology

from decision_making.mcda import calc_score, normalize
from decision_making.attitudes import simple_diff

class HouseholdAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(
        self,
        unique_id,
        model,
        disposable_income,
        installed_heating_tech,
        installed_pv_cap=0,
        interactions_per_step=1,
        step_length_in_years=1 / 4,
    ):
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        # Create the agent's attribute and set the initial values.
        self.wealth = 0
        self.interactions_this_step = 0
        self.interactions_per_step = interactions_per_step
        self.step_length_in_years = step_length_in_years
        self.disposable_income = disposable_income * step_length_in_years
        self.heating_tech = installed_heating_tech
        available_techs = self.model.heating_techs_df.index
        self.tech_attitudes = dict(
            zip(available_techs, 2 * np.random.random(len(available_techs)) - 1)
        )
        self.att_inertia = np.random.random()
        self.pbc = self.random.random()

    def step(self):
        """called each `step´ of the model.
        This is how the model progresses through time"""
        self.heating_tech.age += self.step_length_in_years
        self.interactions_this_step = 0
        self.interact()
        self.wealth += (
            self.disposable_income - self.heating_tech.total_cost_per_year(20_000)
        ) * self.step_length_in_years

        # idealistic adoption happening here
        # this might not lead to adoption if 
        if self.heating_tech.age > self.heating_tech.lifetime * 0.5:
            self.purchase_heating_tbp_based()
        
        # necessary adoption happening here
        if self.heating_tech.age > self.heating_tech.lifetime:
            self.purchase_new_heating()
            

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

    def purchase_new_heating(self):
        techs_df = self.model.heating_techs_df
        techs_df["attitude"] = self.tech_attitudes
        techs_df["attitude"] = normalize(techs_df["attitude"] + 1)
        # calculate scores
        techs_df["total_score"] = techs_df[
            ["emissions[kg_CO2/a]_norm", "total_cost[EUR/a]_norm", "attitude"]
        ].apply(
            calc_score,
            axis=1,
            weights={
                "emissions[kg_CO2/a]_norm": 0.3,
                "total_cost[EUR/a]_norm": 0.5,
                "attitude": 0.2,
            },
        )
        best_tech_idx = techs_df["total_score"].argmin()
        new_tech = techs_df.iloc[best_tech_idx, :]

        # TODO: implement affordability constraint

        self.heating_tech = HeatingTechnology.from_series(new_tech)
        pass

    def purchase_heating_tbp_based(self):
        techs_df = self.model.heating_techs_df
        for tech_name, tech_att in self.tech_attitudes.items():
            # TODO: at least a sensitivity analysis for arbitrary value
            if tech_att > 0.7:
                annual_cost = techs_df.loc[tech_name, "total_cost[EUR/a]"]
                if self.disposable_income > annual_cost:
                    random_pbc = self.random.random()
                    # TODO: this might lead to the situation in which the lifetime of 
                    # an appliance has expired, but due to lacking pbc, no new appliance
                    # is being bought
                    if random_pbc < self.pbc:
                        self.heating_tech = HeatingTechnology.from_series(
                            techs_df.loc[tech_name, :]
                        )
                        return

