from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from ..models import initialize_model


class TrainState(train_state.TrainState):
    batch_stats: Any


def create_lr_schedule(
    base_lr: float, steps_per_epoch: int, warmup_epochs: int, num_epochs: int
):
    # From https://github.com/google/flax/blob/main/examples/imagenet/train.py#L88
    warmup_optimization = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_epochs * steps_per_epoch,
    )
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_lr_schedule = optax.cosine_decay_schedule(
        init_value=base_lr, decay_steps=cosine_epochs * steps_per_epoch
    )
    lr_schedule = optax.join_schedules(
        schedules=[warmup_optimization, cosine_lr_schedule],
        boundaries=[warmup_epochs * steps_per_epoch],
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

    params, batch_stats = initialize_model(
        model_key=state_key, image_size=image_size, model=model
    )
    # From https://github.com/google/flax/blob/main/examples/imagenet/train.py#L233
    tx = optax.sgd(learning_rate=lr_schedule, momentum=momentum, nesterov=True)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )
