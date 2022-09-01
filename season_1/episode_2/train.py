import jax
import ml_collections
import tensorflow as tf
from jax_classification.dataloader import DataLoaderFromBuilder
from jax_classification.training.steps import train_step
from jax_classification.training.train_state import (
    TrainState,
    create_lr_schedule,
    create_train_state,
)
from tqdm.autonotebook import tqdm


# Validation is not yet hapenning...
def train_and_validate(
    seed: int,
    image_size: int,
    local_batch_size: int,
    apply_half_precision: bool,
    num_prefetch_examples: int,
    dataloader_config: ml_collections.ConfigDict,
    lr_config: ml_collections.ConfigDict,
    state: TrainState,
    epochs: int,
    weight_decay: float,
    num_classes: int,
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
    data_loader = DataLoaderFromBuilder(**dataloader_config.to_dict())
    train_dataset, train_iterator, train_steps_per_epoch = data_loader.create_split(
        split_name="train",
        batch_size=local_batch_size,
        num_prefetch_examples=num_prefetch_examples,
    )

    lr_schedule = create_lr_schedule(
        steps_per_epoch=train_steps_per_epoch, num_epochs=epochs, **lr_config.to_dict()
    )

    for epoch in range(1, epochs + 1):
        for step_idx in tqdm(range(train_steps_per_epoch)):
            batch = next(train_iterator)
            state, metrics = train_step(
                state, batch, lr_schedule, weight_decay, num_classes
            )
            print(metrics["accuracy"])
