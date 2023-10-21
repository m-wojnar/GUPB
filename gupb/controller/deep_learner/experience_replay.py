from typing import Callable

import jax
import jax.numpy as jnp
from chex import dataclass, Array, Numeric, Scalar, PRNGKey, Shape


@dataclass
class ExperienceReplay:
    init: Callable
    append: Callable
    sample: Callable
    is_ready: Callable


@dataclass
class ReplayBuffer:
    states: Array
    actions: Array
    rewards: Array
    terminals: Array
    next_states: Array
    size: jnp.int32
    ptr: jnp.int32


def experience_replay(
        buffer_size: jnp.int32,
        batch_size: jnp.int32,
        obs_space_shape: Shape,
        act_space_shape: Shape
) -> ExperienceReplay:
    def init() -> ReplayBuffer:
        return ReplayBuffer(
            states=jnp.empty((buffer_size, *obs_space_shape)),
            actions=jnp.empty((buffer_size, *act_space_shape)),
            rewards=jnp.empty((buffer_size, 1)),
            terminals=jnp.empty((buffer_size, 1), dtype=jnp.bool_),
            next_states=jnp.empty((buffer_size, *obs_space_shape)),
            size=0,
            ptr=0
        )

    def append(
            buffer: ReplayBuffer,
            state: Numeric,
            action: Numeric,
            reward: Scalar,
            terminal: jnp.bool_,
            next_state: Numeric
    ) -> ReplayBuffer:
        return ReplayBuffer(
            states=buffer.states.at[buffer.ptr].set(state),
            actions=buffer.actions.at[buffer.ptr].set(action),
            rewards=buffer.rewards.at[buffer.ptr].set(reward),
            terminals=buffer.terminals.at[buffer.ptr].set(terminal),
            next_states=buffer.next_states.at[buffer.ptr].set(next_state),
            size=jax.lax.min(buffer.size + 1, buffer_size),
            ptr=(buffer.ptr + 1) % buffer_size
        )

    def sample(buffer: ReplayBuffer, key: PRNGKey) -> tuple:
        idxs = jax.random.uniform(key, shape=(batch_size,), minval=0, maxval=buffer.size).astype(jnp.int32)

        states = buffer.states[idxs]
        actions = buffer.actions[idxs]
        rewards = buffer.rewards[idxs]
        terminals = buffer.terminals[idxs]
        next_states = buffer.next_states[idxs]

        return states, actions, rewards, terminals, next_states

    def is_ready(buffer: ReplayBuffer) -> jnp.bool_:
        return buffer.size >= batch_size

    return ExperienceReplay(
        init=init,
        append=append,
        sample=sample,
        is_ready=is_ready
    )
