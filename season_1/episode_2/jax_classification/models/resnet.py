"""Reference: https://github.com/google/flax/blob/main/examples/imagenet/models.py"""

import jax
import jax.numpy as jnp

import flax
import flax.linen as nn

from functools import partial
from typing import Any, Callable, Sequence, Tuple


class ResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    convolution: Any
    normalization: Any
    activation: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.convolution(self.filters, (3, 3), self.strides)(x)
        y = self.normalization()(y)
        y = self.activation(y)
        y = self.convolution(self.filters, (3, 3))(y)
        y = self.normalization(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.convolution(
                self.filters, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.normalization(name="norm_proj")(residual)

        return self.activation(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    convolution: Any
    normalization: Any
    activation: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.convolution(self.filters, (1, 1))(x)
        y = self.normalization()(y)
        y = self.activation(y)
        y = self.convolution(self.filters, (3, 3), self.strides)(y)
        y = self.normalization()(y)
        y = self.activation(y)
        y = self.convolution(self.filters * 4, (1, 1))(y)
        y = self.normalization(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.convolution(
                self.filters * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.normalization(name="norm_proj")(residual)

        return self.activation(residual + y)


class ResNet(nn.Module):
    """ResNetV1."""

    stage_sizes: Sequence[int]
    block_cls: Any
    num_classes: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    activation: Callable = nn.relu
    convolution: Any = nn.Conv

    @nn.compact
    def __call__(self, x, train: bool = True):
        convolution = partial(self.convolution, use_bias=False, dtype=self.dtype)
        normalization = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        x = convolution(
            self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name="conv_init"
        )(x)
        x = normalization(name="bn_init")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=strides,
                    convolution=convolution,
                    normalization=normalization,
                    activation=self.activation,
                )(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)
ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock)
ResNet152 = partial(ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock)
ResNet200 = partial(ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock)


ResNet18Local = partial(
    ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock, conv=nn.ConvLocal
)
