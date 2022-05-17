from typing import Any, Callable, Optional, Union, NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from optax._src import base
from optax._src import combine
from optax._src import transform
from optax._src import numerics


ScalarOrSchedule = Union[float, base.Schedule]
MaskOrFn = Optional[Union[Any, Callable[[base.Params], Any]]]


def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
    m = -1 if flip_sign else 1
    if callable(learning_rate):
        return transform.scale_by_schedule(lambda count: m * learning_rate(count))
    return transform.scale(m * learning_rate)


class ScaleByNovogradState(NamedTuple):
    count: chex.Array
    exp_avg: chex.Array
    exp_avg_sq: chex.Array


def scale_by_novograd(
    b1: float = 0.9,
    b2: float = 0.5,
    eps: float = 1e-8,
) -> base.GradientTransformation:
    def init_fn(params):
        exp_avg = jax.tree_map(jnp.zeros_like, params)
        exp_avg_sq = jax.tree_map(lambda t: jnp.zeros_like(t, shape=[]), params)
        return ScaleByNovogradState(
            count=jnp.zeros([], jnp.int32), exp_avg=exp_avg, exp_avg_sq=exp_avg_sq
        )

    def update_fn(updates, state, params=None):
        del params
        exp_avg, exp_avg_sq = state.exp_avg, state.exp_avg_sq
        count_inc = numerics.safe_int32_increment(state.count)
        norm = jax.tree_map(lambda u: jnp.sum(jnp.square(u)), updates)
        exp_avg_sq = jax.tree_map(
            lambda e, n: jnp.where(jnp.equal(count_inc, 1), n, e * b2 + n * (1 - b2)),
            exp_avg_sq,
            norm,
        )
        denom = jax.tree_map(lambda e: jnp.sqrt(e) + eps, exp_avg_sq)
        updates = jax.tree_map(lambda u, d: u / d, updates, denom)
        exp_avg = jax.tree_map(lambda e, g: e * b1 + g, exp_avg, updates)
        return exp_avg, ScaleByNovogradState(
            count=count_inc, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq
        )

    return base.GradientTransformation(init_fn, update_fn)


def novograd_optimizer(
    learning_rate: Union[float, Callable[[int], float]],
    b1: float = 0.9,
    b2: float = 0.5,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[MaskOrFn] = None,
    gradient_norm_clip: Optional[float] = None,
) -> optax.GradientTransformation:
    if gradient_norm_clip:
        chain = [optax.clip_by_global_norm(gradient_norm_clip)]
    else:
        chain = []
    chain += [
        scale_by_novograd(b1=b1, b2=b2, eps=eps),
        transform.add_decayed_weights(weight_decay, weight_decay_mask),
        _scale_by_learning_rate(learning_rate),
    ]
    return combine.chain(*chain)
