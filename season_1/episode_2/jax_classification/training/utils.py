import jax
import jax.numpy as jnp
import optax
from flax.training import common_utils


def cross_entropy_loss(logits, labels, num_classes: int) -> jnp.ndarray:
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
    cross_entropy_loss = optax.softmax_cross_entropy(
        logits=logits, labels=one_hot_labels
    )
    return jnp.mean(cross_entropy_loss)


def compute_decayed_weights(params, weight_decay: float):
    regularized_weights = sum(
        jnp.sum(x**2)
        for weight in jax.tree_util.tree_leaves(params)
        if weight.ndim > 1
    )
    return weight_decay * 0.5 * regularized_weights


def compute_accuracy(logits, labels):
    accuracy = jnp.argmax(logits, axis=-1) == labels
    return jnp.mean(accuracy)
