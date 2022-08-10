from typing import Any, Callable

import optax

import jax
import jax.numpy as jnp

from flax.training import train_state
from flax.optim import dynamic_scale as dynamic_scale_lib

from ..models import initialize_model


class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale


def create_lr_schedule(
    base_lr: float, steps_per_epch: int, warmup_epochs: int, num_epochs: int
):
    warmup_optimization = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_epochs * steps_per_epch,
    )
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_lr_schedule = optax.cosine_decay_schedule(
        init_value=base_lr, decay_steps=cosine_epochs * steps_per_epch
    )
    lr_schedule = optax.join_schedules(
        schedules=[warmup_optimization, cosine_lr_schedule],
        boundaries=[warmup_epochs * steps_per_epch],
    )
    return lr_schedule


def create_train_state(
    state_key: jnp.ndarray,
    model,
    image_size: int,
    lr_schedule: Callable,
    momentum: float,
    apply_half_precision: bool,
):
    platform = jax.local_devices()[0].platform
    dynamic_scale = (
        dynamic_scale_lib.DynamicScale()
        if apply_half_precision and platform == "gpu"
        else None
    )
    params, batch_stats = initialize_model(
        model_key=state_key, image_size=image_size, model=model
    )
    tx = optax.sgd(learning_rate=lr_schedule, momentum=momentum, nesterov=True)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        dynamic_scale=dynamic_scale,
    )
