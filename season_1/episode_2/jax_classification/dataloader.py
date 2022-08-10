import jax

import tensorflow as tf
import tensorflow_datasets as tfds

from typing import List
from functools import partial


class DataLoaderFromBuilder:
    def __init__(
        self,
        dataset_alias: str = "imagenette",
        image_size: int = 224,
        crop_padding: int = 32,
        rgb_mean: List[float] = [0.485, 0.456, 0.406],
        rbg_stddev: List[float] = [0.229, 0.224, 0.225],
        cache: bool = False,
        private_threadpool_size: int = 48,
        try_gcs: bool = False,
        bool=False,
    ):
        self.image_size = image_size
        self.crop_padding = crop_padding
        self.rgb_mean = [m * 255.0 for m in rgb_mean]
        self.rbg_stddev = [m * 255.0 for m in rbg_stddev]
        self.cache = cache

        self.dataset_builder = tfds.builder(dataset_alias, try_gcs=try_gcs)
        self.dataset_builder.download_and_prepare()

        self.options = tf.data.Options()
        self.options.threading.private_threadpool_size = private_threadpool_size

    def normalize_image(self, image):
        image -= tf.constant(self.rgb_mean, shape=[1, 1, 3], dtype=image.dtype)
        image /= tf.constant(self.rbg_stddev, shape=[1, 1, 3], dtype=image.dtype)
        return image

    def preprocess_image(self, image_bytes, is_train: bool):
        image = tf.io.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(
            image,
            [self.image_size, self.image_size],
            method=tf.image.ResizeMethod.BICUBIC,
        )
        image = tf.reshape(image, [self.image_size, self.image_size, 3])
        image = tf.image.random_flip_left_right(image) if is_train else image
        image = self.normalize_image(image)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image

    def decode_example(self, example, is_train: bool):
        return {
            "image": self.preprocess_image(example["image"], is_train=is_train),
            "label": example["label"],
        }

    def create_split(
        self, split_name: str, batch_size: int, num_prefetch_examples: int
    ):
        data_examples = self.dataset_builder.info.splits[split_name].num_examples
        split_size = data_examples // jax.process_count()
        start = jax.process_index() * split_size
        split = f"{split_name}[{start}:{start + split_size}]"
        dataset = self.dataset_builder.as_dataset(
            split=split,
            decoders={
                "image": tfds.decode.SkipDecoding(),
            },
        )
        dataset = dataset.with_options(self.options)
        dataset = dataset.cache() if self.cache else dataset
        dataset = (
            dataset.repeat().shuffle(16 * batch_size, seed=0)
            if split_name == "train"
            else dataset
        )
        dataset = dataset.map(
            partial(self.decode_example, is_train=split_name == "train"),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat() if split_name != "train" else dataset
        dataset = dataset.prefetch(num_prefetch_examples)
        return dataset
