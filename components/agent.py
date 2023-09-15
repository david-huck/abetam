import mesa
import numpy as np


class MoneyAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(
        self,
        unique_id,
        model,
        disposable_income,
        installed_heating_tech,
        installed_pv_cap=0,
    ):
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        # Create the agent's attribute and set the initial values.
        self.wealth = 0
        self.disposable_income = disposable_income/12
        self.heating_tech = installed_heating_tech
        available_techs = self.model.heating_techs_df.index
        self.tech_attitudes = dict(
            zip(available_techs, np.random.random(len(available_techs)))
        )

    def step(self):
        """called each `step´ of the model.
        This is how the model progresses through time"""
        # self.model.datacollector.collect(self)
        self.interact()
        self.wealth += self.disposable_income

    def interact(self):
        """interaction with other agents.
        The interaction should have induce a change in the agents attitude towards
        technologies.
        """
        neighbours = self.model.grid.get_neighbors(self.pos, moore=True, radius=2)
        if len(neighbours) > 1:
            other = self.random.choice(neighbours)

            for tech in self.tech_attitudes.keys():
            # tech = np.random.choice(list(self.tech_attitudes.keys()))
                # print(tech)
                if self.unique_id == 3 and tech=="gas_boiler":
                    print(f"b4 interaction: {self.tech_attitudes[tech]=:.2f}")
                att_diff = self.tech_attitudes[tech] - other.tech_attitudes[tech]
                
                att_diff *= 1 - abs(self.tech_attitudes[tech])

                self.tech_attitudes[tech] += att_diff * 0.5
        else:
            # no neighbours to interact with
            return
        
    def get_attitudes(self):
        return self.tech_attitudes.copy()
    # def __getattribute__(self,name):
    #     if name=='tech_attitudes':
    #         return getattr(self, name).copy()
    #     else:
    #         return object.__getattribute__(self, name)


    def choose_new_heating_sys(self):
        """should be called as a function of remaining tech lifetime?"""
        raise NotImplementedError()
