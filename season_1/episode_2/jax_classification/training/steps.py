from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

from .train_state import TrainState
from .utils import compute_accuracy, compute_decayed_weights, cross_entropy_loss


def train_step(
    state: TrainState,
    batch,
    lr_schedule: Callable,
    weight_decay: float,
    num_classes: int,
):
    def loss_fn(params):
        logits, new_model_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            batch["image"],
            mutable=["batch_stats"],
        )
        weight_penalty = compute_decayed_weights(params, weight_decay)
        loss = cross_entropy_loss(logits, batch["label"], num_classes) + weight_penalty
        return loss, (new_model_state, logits)

    # Dynamic Scaling is not being supported at the moment !
    step = state.step
    lr = lr_schedule(step)

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_model_state, logits)), gradients = gradient_fn(state.params)
    gradients = lax.pmean(gradients, axis_name="batch")

    loss = lax.pmean(
        cross_entropy_loss(logits, batch["label"], num_classes), axis_name="batch"
    )
    accuracy = lax.pmean(compute_accuracy(logits, batch["label"]), axis_name="batch")

    updated_state = state.apply_gradients(
        grads=gradients, batch_stats=new_model_state["batch_stats"]
    )

    return updated_state, {
        "loss": loss,
        "accuracy": accuracy,
        "learning_rate": lr,
    }


def validation_step(state: TrainState, batch, num_classes: int):
    logits = state.apply_fn(
        {"params": params, "batch_stats": state.batch_stats},
        batch["image"],
        train=False,
        mutable=False,
    )
    loss = cross_entropy_loss(logits, batch["label"], num_classes)
    accuracy = compute_accuracy(logits, batch["label"])
    return {"loss": loss, "accuracy": accuracy}
