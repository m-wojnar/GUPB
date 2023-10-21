import os
from typing import Tuple

import jax
import jax.numpy as jnp
from chex import Array

from gupb.model.arenas import ArenaDescription
from gupb.model.characters import Action, ChampionKnowledge
from gupb.model.effects import Mist


ACTIONS = [
    Action.TURN_LEFT,
    Action.TURN_RIGHT,
    Action.STEP_FORWARD,
    Action.ATTACK,
    Action.DO_NOTHING,
]

FACING = {
    'UP': 0,
    'RIGHT': 1,
    'DOWN': 2,
    'LEFT': 3
}

WEAPONS = {
    'knife': 0,
    'sword': 1,
    'axe': 2,
    'bow': 3,
    'bow_loaded': 3,
    'bow_unloaded': 3,
    'amulet': 4
}

MAX_AGE = 10

"""
Dimensions of the environment state:
0  - info: my position, my facing (one-hot), my health, my weapon (one-hot), number of champions alive
1  - arena
2  - mist
3  - potion
4  - weapon position - knife
5  - weapon position - sword
6  - weapon position - axe
7  - weapon position - bow
8  - weapon position - amulet
9  - weapon ownership - knife
10 - weapon ownership - sword
11 - weapon ownership - axe
12 - weapon ownership - bow
13 - weapon ownership - amulet
14 - champion position
15 - champion facing - up
16 - champion facing - right
17 - champion facing - down
18 - champion facing - left
19 - health
20 - my position
"""

N_DIMS = 21
PROTECTED_DIMS = [0, 1, 2]
N_INFO = 13


def init_env_state(arena_description: ArenaDescription) -> Tuple[Array, Array]:
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
        'resources', 'arenas', f'{arena_description.name}.gupb'
    )

    with open(path, 'r') as f:
        arena = f.readlines()

    arena = jax.tree_map(lambda x: list(x.strip()), arena)
    arena = jax.tree_map(lambda x: x not in '#=', arena)
    arena = jnp.array(arena, dtype=jnp.float32)

    info = jnp.zeros_like(arena)
    other = jnp.zeros(arena.shape + (N_DIMS - 2,))

    env_state = jnp.concatenate([info[..., None], arena[..., None], other], axis=-1)
    age = jnp.zeros_like(env_state, dtype=jnp.int32)

    return env_state, age


def to_agent_state(knowledge: ChampionKnowledge, env_state: Array, age: Array) -> Tuple[Array, Array]:
    age = age.at[..., PROTECTED_DIMS].set(0)
    env_state = jnp.where(age < MAX_AGE, env_state, 0.)
    age = age + 1

    last_pos_x, last_pos_y = env_state[0, 0, :2].astype(jnp.int32)
    env_state = env_state.at[last_pos_x, last_pos_y, 9:].set(0.)
    env_state = env_state.at[knowledge.position.x, knowledge.position.y, 20].set(1.)

    description = knowledge.visible_tiles[knowledge.position].character
    pos_x, pos_y = knowledge.position.x, knowledge.position.y
    facing = jax.nn.one_hot(FACING[description.facing.name], 4)
    life_points = description.health
    weapon = jax.nn.one_hot(WEAPONS[description.weapon.name], 5)
    n_champions_alive = knowledge.no_of_champions_alive
    info = jnp.hstack([pos_x, pos_y, facing, life_points, weapon, n_champions_alive])

    env_state = env_state.at[0, 0, :N_INFO].set(info)

    for (x, y), tile in knowledge.visible_tiles.items():
        env_state = env_state.at[x, y, 1].set(tile.type in ['land', 'menhir'])

        if any(isinstance(effect, Mist) for effect in tile.effects):
            env_state = env_state.at[x, y, 2].set(1.)

        if tile.consumable:
            env_state = env_state.at[x, y, 3].set(1.)

        if tile.loot:
            env_state = env_state.at[x, y, 4 + WEAPONS[tile.loot.name]].set(1.)

        if tile.character:
            env_state = env_state.at[x, y, 9 + WEAPONS[tile.character.weapon.name]].set(1.)
            env_state = env_state.at[x, y, 14].set(1.)
            env_state = env_state.at[x, y, 15 + FACING[tile.character.facing.name]].set(1.)
            env_state = env_state.at[x, y, 19].set(tile.character.health)

    return env_state, age


def int_to_action(action: int) -> Action:
    return ACTIONS[action]
