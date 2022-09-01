import unittest

import jax
import jax.numpy as jnp
from jax_classification.models import ResNet18, ResNet50


class ResNetTester(unittest.TestCase):
    def test_resnet_50(self):
        rng = jax.random.PRNGKey(0)
        model = ResNet50(num_classes=10, dtype=jnp.float32)
        variables = model.init(rng, jnp.ones((1, 224, 224, 3), jnp.float32))
        self.assertEqual(len(variables), 2)
        self.assertEqual(len(variables["params"]), 19)

    def test_resnet_18(self):
        rng = jax.random.PRNGKey(0)
        model = ResNet18(num_classes=10, dtype=jnp.float32)
        variables = model.init(rng, jnp.ones((1, 64, 64, 3), jnp.float32))
        self.assertEqual(len(variables), 2)
        self.assertEqual(len(variables["params"]), 11)
