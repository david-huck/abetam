import mesa
import numpy as np
from components.technologies import HeatingTechnology

from decision_making.mcda import calc_score, normalize

class MoneyAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(
        self,
        unique_id,
        model,
        disposable_income,
        installed_heating_tech,
        installed_pv_cap=0,
        interactions_per_step=1,
        step_length_in_years=1/4
    ):
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        # Create the agent's attribute and set the initial values.
        self.wealth = 0
        self.interactions_this_step = 0
        self.interactions_per_step = interactions_per_step
        self.step_length_in_years = step_length_in_years
        self.disposable_income = disposable_income*step_length_in_years
        self.heating_tech = installed_heating_tech
        available_techs = self.model.heating_techs_df.index
        self.tech_attitudes = dict(
            zip(available_techs, np.random.random(len(available_techs)))
        )

    def step(self):
        """called each `step´ of the model.
        This is how the model progresses through time"""
        self.heating_tech.age += self.step_length_in_years
        self.interactions_this_step = 0
        self.interact()
        self.wealth += self.disposable_income - self.heating_tech.total_cost_per_year(20_000)

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
            if any(x.interactions_this_step >= x.interactions_per_step for x in [self, other]):
                return

            for tech in self.tech_attitudes.keys():
                att_diff = self.tech_attitudes[tech] - other.tech_attitudes[tech]
                
                att_diff *= 1 - abs(self.tech_attitudes[tech])

                self.tech_attitudes[tech] += att_diff * 0.01
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
        techs_df["total_score"] = techs_df[["emissions[kg_CO2/a]_norm","total_cost[EUR/a]_norm", "attitude"]]\
                .apply(
                    calc_score, 
                    axis=1, 
                    weights={"emissions[kg_CO2/a]_norm":0.3,"total_cost[EUR/a]_norm":0.5, "attitude":0.2}
                    )
        best_tech_idx = techs_df["total_score"].argmin()
        new_tech = techs_df.iloc[best_tech_idx,:]
        
        # TODO: implement affordability constraint
        
        self.heating_tech = HeatingTechnology.from_series(new_tech)
        
        pass

    def choose_new_heating_sys(self):
        """should be called as a function of remaining tech lifetime?"""
        raise NotImplementedError()
