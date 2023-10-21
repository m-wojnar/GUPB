import pickle
from copy import deepcopy
from typing import Callable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import lz4.frame
import optax
from chex import dataclass, Array, PRNGKey, Scalar, Shape

from gupb.controller.deep_learner.experience_replay import experience_replay, ReplayBuffer
from gupb.controller.deep_learner.utils import N_INFO


@dataclass
class DQNState:
    params: hk.Params
    params_target: hk.Params
    opt_state: optax.OptState
    replay_buffer: ReplayBuffer
    epsilon: Scalar


@dataclass
class DQN:
    init: Callable
    update: Callable
    act: Callable
    save: Callable
    save_final: Callable
    load: Callable


@hk.transform
def q_network(x: Array) -> Array:
    x = x[None, ...] if len(x.shape) == 3 else x
    x1, x2 = x[..., 1:], x[..., 0, :N_INFO, 0]

    x = hk.Conv2D(50, kernel_shape=1, padding='SAME')(x1)
    x = jax.nn.relu(x)
    x = hk.Conv2D(20, kernel_shape=3, padding='SAME')(x)
    x = jax.nn.relu(x)
    x = jnp.mean(x, axis=(1, 2))
    x = jnp.concatenate([x, x2], axis=-1)
    x = hk.nets.MLP([50, 20, 20, 5])(x)

    return x


def dqn(obs_space_shape: Shape) -> DQN:

    act_space_size = 5
    epsilon_start = 1.
    epsilon_decay = 0.999
    epsilon_min = 0.001
    tau = 0.01
    discount = 0.99
    experience_replay_steps = 5

    optimizer = optax.adam(3e-4)

    er = experience_replay(
        buffer_size=10000,
        batch_size=64,
        obs_space_shape=obs_space_shape,
        act_space_shape=(1,)
    )

    def init(key: PRNGKey) -> DQNState:
        x_dummy = jnp.empty(obs_space_shape)
        params = q_network.init(key, x_dummy)

        opt_state = optimizer.init(params)
        replay_buffer = er.init()

        return DQNState(
            params=params,
            params_target=deepcopy(params),
            opt_state=opt_state,
            replay_buffer=replay_buffer,
            epsilon=epsilon_start
        )

    def loss_fn(
            params: hk.Params,
            key: PRNGKey,
            params_target: hk.Params,
            batch: tuple,
            non_zero_loss: jnp.bool_
    ) -> Scalar:
        states, actions, rewards, terminals, next_states = batch
        q_key, q_target_key = jax.random.split(key)

        q_values = q_network.apply(params, q_key, states)
        q_values = jnp.take_along_axis(q_values, actions.astype(jnp.int32), axis=-1)

        q_values_target = q_network.apply(params_target, q_target_key, next_states)
        target = rewards + (1 - terminals) * discount * jnp.max(q_values_target, axis=-1, keepdims=True)

        target = jax.lax.stop_gradient(target)
        loss = optax.l2_loss(q_values, target).mean()

        return loss * non_zero_loss

    def update(
            state: DQNState,
            key: PRNGKey,
            prev_env_state: Array,
            action: Array,
            reward: Scalar,
            terminal: jnp.bool_,
            env_state: Array
    ) -> Tuple[DQNState, Scalar]:
        params, params_target, opt_state = state.params, state.params_target, state.opt_state
        replay_buffer = er.append(state.replay_buffer, prev_env_state, action, reward, terminal, env_state)

        non_zero_loss = er.is_ready(replay_buffer)
        loss = 0.

        for _ in range(experience_replay_steps):
            batch_key, loss_key, key = jax.random.split(key, 3)
            batch = er.sample(replay_buffer, batch_key)

            loss_val, grads = jax.value_and_grad(loss_fn)(params, loss_key, params_target, batch, non_zero_loss)
            updates, opt_state = optimizer.update(grads, opt_state)
            loss += loss_val

            params = optax.apply_updates(params, updates)
            params_target = optax.incremental_update(params, params_target, tau)

        return DQNState(
            params=params,
            params_target=params_target,
            opt_state=opt_state,
            replay_buffer=replay_buffer,
            epsilon=jax.lax.max(state.epsilon * epsilon_decay, epsilon_min)
        ), loss / experience_replay_steps

    def act(state: DQNState, key: PRNGKey, env_state: Array) -> jnp.int32:
        network_key, epsilon_key, action_key = jax.random.split(key, 3)

        return jax.lax.cond(
            jax.random.uniform(epsilon_key) < state.epsilon,
            lambda: jax.random.choice(action_key, act_space_size),
            lambda: jnp.argmax(q_network.apply(state.params, network_key, env_state)[0])
        )

    def save(state: DQNState, path: str) -> None:
        with lz4.frame.open(path, 'wb') as f:
            f.write(pickle.dumps(state))

    def save_final(state: DQNState, path: str) -> None:
        final_state = DQNState(
            params=state.params,
            params_target=None,
            opt_state=None,
            replay_buffer=None,
            epsilon=0.
        )

        save(final_state, path)

    def load(path: str) -> DQNState:
        with lz4.frame.open(path, 'rb') as f:
            return pickle.loads(f.read())

    return DQN(
        init=init,
        update=jax.jit(update),
        act=jax.jit(act),
        save=save,
        save_final=save_final,
        load=load
    )
