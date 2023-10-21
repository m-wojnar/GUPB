import jax
from tensorboardX import SummaryWriter

from gupb.controller import Controller
from gupb.controller.deep_learner.dqn import dqn
from gupb.controller.deep_learner.utils import init_env_state, to_agent_state, int_to_action
from gupb.model.arenas import ArenaDescription
from gupb.model.characters import Action, ChampionKnowledge, Tabard
from gupb.model.effects import Mist


class DeepLearnerControllerTrain(Controller):
    LOAD_PATH = None

    def __init__(self, first_name: str) -> None:
        self.first_name = first_name

        self.key = jax.random.PRNGKey(42)
        self.dqn = None
        self.state = None

        self.action = None
        self.age = None
        self.env_state = None

        self.writer = SummaryWriter(f'runs/{self.first_name}')

        self.episode = 0
        self.step = 0
        self.total_step = 0
        self.cumulative_reward = 0.
        self.knowledge = None

    def intermediate_reward(self, knowledge: ChampionKnowledge) -> float:
        if not self.knowledge:
            self.knowledge = knowledge
            return 0.

        old_description = self.knowledge.visible_tiles[self.knowledge.position].character
        new_description = knowledge.visible_tiles[knowledge.position].character

        reward = 0.

        if self.knowledge.position != knowledge.position or old_description.facing != new_description.facing:
            reward = 0.1

        if self.knowledge.no_of_champions_alive > knowledge.no_of_champions_alive:
            reward = 10.

        if old_description.weapon != new_description.weapon:
            reward = 3.

        if any(isinstance(x, Mist) for x in knowledge.visible_tiles[knowledge.position].effects):
            reward = -1.

        if old_description.health < new_description.health:
            reward = 1.

        self.knowledge = knowledge

        return reward

    def decide(self, knowledge: ChampionKnowledge) -> Action:
        new_env_state, self.age = to_agent_state(knowledge, self.env_state, self.age)
        self.key, update_key, act_key = jax.random.split(self.key, 3)

        reward = self.intermediate_reward(knowledge)
        self.state, loss = self.dqn.update(self.state, update_key, self.env_state, self.action, reward, False, new_env_state)
        self.action = self.dqn.act(self.state, act_key, new_env_state)

        self.writer.add_scalar('reward', reward, self.total_step)
        self.writer.add_scalar('cumulative_reward', self.cumulative_reward + reward, self.total_step)
        self.writer.add_scalar('loss', loss, self.total_step)

        self.cumulative_reward += reward
        self.env_state = new_env_state
        self.step += 1
        self.total_step += 1

        return int_to_action(self.action)

    def praise(self, score: int) -> None:
        reward = -10 if score < 5 else score
        self.key, update_key = jax.random.split(self.key)

        self.state = self.dqn.update(self.state, update_key, self.env_state, self.action, reward, True, self.env_state)

        self.writer.add_scalar('reward', reward, self.total_step)
        self.writer.add_scalar('cumulative_reward', self.cumulative_reward + reward, self.total_step)
        self.writer.add_scalar('episode_len', self.step, self.episode)
        self.writer.add_scalar('score', score, self.episode)

        self.cumulative_reward += reward
        self.step = 0
        self.total_step += 1
        self.episode += 1

        if self.episode % 50 == 0:
            self.dqn.save(self.state, f'{self.first_name}_{self.episode}.pkl.lz4')

    def reset(self, arena_description: ArenaDescription) -> None:
        self.env_state, self.age = init_env_state(arena_description)

        if not self.dqn:
            self.dqn = dqn(self.env_state.shape)
            self.state = self.dqn.load(self.LOAD_PATH) if self.LOAD_PATH else self.dqn.init(self.key)

    @property
    def name(self) -> str:
        return f'DeepLearnerControllerTrain{self.first_name}'

    @property
    def preferred_tabard(self) -> Tabard:
        return Tabard.ORANGE
