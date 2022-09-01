from typing import List

import ml_collections


def get_dataloader_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.dataset_alias: str = "imagenette"
    config.image_size: int = 224
    config.crop_padding: int = 32
    config.rgb_mean: List[float] = [0.485, 0.456, 0.406]
    config.rbg_stddev: List[float] = [0.229, 0.224, 0.225]
    config.cache: bool = False
    config.private_threadpool_size: int = 48
    config.try_gcs: bool = False

    return config


def get_lr_config() -> ml_collections.ConfigDict:

    config = ml_collections.ConfigDict()

    config.base_lr = 0.1
    config.warmup_epochs = 5.0

    return config
