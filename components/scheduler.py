from mesa.time import BaseScheduler
from joblib import Parallel, delayed


class ParallelActivation(BaseScheduler):
    """A scheduler which activates each agent once per step, in random order,
    with the order reshuffled every step.

    This is equivalent to the NetLogo 'ask agents...' and is generally the
    default behavior for an ABM.

    Assumes that all agents have a step(model) method.
    """

    def step(self) -> None:
        """Executes the step of all agents, one at a time, in
        random order.

        """
        self.do_each("step", shuffle=True)
        self.steps += 1
        self.time += 1

    def do_each(self, method, agent_keys=None, shuffle=False):
        if agent_keys is None:
            agent_keys = self.get_agent_keys()
        if shuffle:
            self.model.random.shuffle(agent_keys)

        Parallel(n_jobs=20, prefer="threads")(
            delayed(getattr(self._agents[agent_key], method))()
            for agent_key in agent_keys
            # if agent_key in self._agents
        )
