import jax

import tensorflow as tf

from ..dataloader import DataLoaderFromBuilder


# Validation is not yet hapenning...
def train_and_validate(
    seed: int,
    image_size: int,
    local_batch_size: int,
    apply_half_precision: bool,
    epochs: int,
):
    global_batch_size = local_batch_size * jax.device_count()
    platform = jax.local_devices()[0].platform
    input_dtype = (
        tf.bfloat16
        if platform == "tpu"
        else tf.float16
        if apply_half_precision
        else tf.float32
    )
    data_loader = DataLoaderFromBuilder(image_size=image_size)
    train_dataset, train_iterator, train_steps_per_epoch = data_loader.create_split(
        split_name="train", batch_size=local_batch_size, num_prefetch_examples=10
    )
