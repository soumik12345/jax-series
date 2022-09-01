import jax
import jax.numpy as jnp


def create_model(model_class, apply_half_precision: bool, num_classes: int, **kwargs):
    platform = jax.local_devices()[0].platform
    dtype = jnp.flot32
    if apply_half_precision:
        dtype = jnp.bfloat16 if platform == "tpu" else jnp.float16
    model = model_class(num_classes=num_classes, dtype=dtype, **kwargs)
    return model


def initialize_model(model_key: jnp.ndarray, image_size: int, model):
    # variables = jax.jit(model.init)(
    #     {"params": model_key}, jnp.ones((1, image_size, image_size, 3), model.dtype)
    # )
    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init(
        {"params": model_key}, jnp.ones((1, image_size, image_size, 3), model.dtype)
    )
    return variables["params"], variables["batch_stats"]
