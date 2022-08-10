import jax
from jax import lax
from flax.training import common_utils

from .train_state import TrainState


def cross_entropy_loss(logits, labels, num_classes: int):
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
    cross_entropy_loss = optax.softmax_cross_entropy(
        logits=logits, labels=one_hot_labels
    )
    return jnp.mean(cross_entropy_loss)


def compute_decayed_weights(params, weight_decay: float):
    list_of_weights = jax.tree_util.tree_leaves(params)
    regularized_weights = sum(
        jnp.sum(x**2) for weight in list_of_weights if weight.ndim > 1
    )
    return weight_decay * 0.5 * regularized_weights


def train_step(
    state: TrainState, batch, lr_schedule, weight_decay: float, num_classes: int
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

    step, dynamic_scale = state.step, state.dynamic_scale
    lr = lr_schedule(step)
    if dynamic_scale:
        gradient_fn = dynamic_scale.value_and_grad(
            loss_fn, has_aux=True, axis_name="batch"
        )
        (
            dynamic_scale,
            is_fin,
            (loss, (new_model_state, logits)),
            gradients,
        ) = gradient_fn(state.params)
    else:
        gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (new_model_state, logits)), gradients = gradient_fn(state.params)
        gradients = lax.pmean(gradients, axis_name="batch")
    loss = lax.pmean(
        cross_entropy_loss(logits, batch["label"], num_classes), axis_name="batch"
    )
    accuracy = lax.pmean(jnp.mean(jnp.argmax(logits, -1) == labels), axis_name="batch")
    updated_state = state.apply_gradients(
        grads=gradients, batch_stats=new_model_state["batch_stats"]
    )
    updated_state = updated_state.replace(
        opt_state=jax.tree_util.tree_map(
            functools.partial(jnp.where, is_fin),
            updated_state.opt_state,
            state.opt_state,
        ),
        params=jax.tree_util.tree_map(
            functools.partial(jnp.where, is_fin), updated_state.params, state.params
        ),
        dynamic_scale=dynamic_scale,
    )
    return updated_state, {
        "loss": loss,
        "accuracy": accuracy,
        "learning_rate": lr,
        "scale": dynamic_scale.scale,
    }
