import mesa


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
        self.disposable_income = disposable_income
        self.heating_tech = installed_heating_tech

    def step(self):
        """called each `step´ of the model. 
        This is how the model progresses through time"""
        self.interact()
        self.wealth += self.disposable_income

    def interact(self):
        """interaction with other agents.
        Currently only neighbors, soon maybe also social network contacts"""
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):
        """should be removed soon"""
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1

    def choose_new_heating_sys(self):
        """should be called as a function of remaining tech lifetime?"""
        raise NotImplementedError()
