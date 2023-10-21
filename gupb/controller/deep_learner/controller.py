import os

import jax

from gupb.controller import Controller
from gupb.controller.deep_learner.dqn import dqn
from gupb.controller.deep_learner.utils import to_agent_state, init_env_state, int_to_action
from gupb.model.arenas import ArenaDescription
from gupb.model.characters import ChampionKnowledge, Action, Tabard


class DeepLearnerController(Controller):
    def __init__(self, first_name: str) -> None:
        self.first_name = first_name
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.pkl.lz4')

        self.key = jax.random.PRNGKey(42)
        self.dqn = None
        self.state = None

        self.age = None
        self.env_state = None

    def decide(self, knowledge: ChampionKnowledge) -> Action:
        self.env_state, self.age = to_agent_state(knowledge, self.env_state, self.age)
        self.key, act_key = jax.random.split(self.key)
        action = self.dqn.act(self.state, act_key, self.env_state)
        return int_to_action(action)

    def praise(self, score: int) -> None:
        pass

    def reset(self, arena_description: ArenaDescription) -> None:
        self.env_state, self.age = init_env_state(arena_description)

        if not self.dqn:
            self.dqn = dqn(self.env_state.shape)
            self.state = self.dqn.load(self.path)

    @property
    def name(self) -> str:
        return f'DeepLearnerController{self.first_name}'

    @property
    def preferred_tabard(self) -> Tabard:
        return Tabard.ORANGE
