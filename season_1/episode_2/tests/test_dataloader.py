import unittest
from jax_classification.dataloader import DataLoaderFromBuilder


class DataLoaderTester(unittest.TestCase):

    def test_dataloader(self):
        data_loader = DataLoaderFromBuilder()
        
        train_dataset = data_loader.create_split(
            split_name="train", batch_size=8, num_prefetch_examples=10
        )
        train_batch = next(iter(train_dataset))
        self.assertEquals(train_batch["image"].shape, (8, 224, 224, 3))
        self.assertEquals(train_batch["label"].shape, (8))
        
        val_dataset = data_loader.create_split(
            split_name="validation", batch_size=8, num_prefetch_examples=10
        )
        val_batch = next(iter(val_dataset))
        self.assertEquals(val_batch["image"].shape, (8, 224, 224, 3))
        self.assertEquals(val_batch["label"].shape, (8))
