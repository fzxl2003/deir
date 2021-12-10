import gym
import numpy as np
import torch as th

from gym.spaces import Dict
from torch import nn, Tensor
from typing import Type

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.utils.common_func import get_norm_layer_2d, get_norm_layer_1d
from src.utils.enum_types import NormType


class CustomCnnFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 features_dim: int,
                 activation_fn: Type[nn.Module],
                 model_type: int,
                 cnn_fc_layers: int,
                 n_input_channels: int,
    ):
        if isinstance(observation_space, Dict):
            observation_space = list(observation_space.values())[0]
        super(CustomCnnFeaturesExtractor, self).\
            __init__(observation_space, features_dim)
        if n_input_channels == 0:
            n_input_channels = observation_space.shape[0]
        self.n_input_channels = n_input_channels
        self.n_input_size = observation_space.shape[1]
        self.activation_fn = activation_fn()
        self.cnn_fc_layers = cnn_fc_layers
        self.model_type = model_type

        n_flatten = None
        if model_type == 0:
            if self.n_input_size > 3:
                self.cnn = nn.Sequential(
                    get_norm_layer_2d(self.norm_type, n_input_channels, self.n_input_size),

                    nn.Conv2d(n_input_channels, 32, (2, 2)),
                    get_norm_layer_2d(self.norm_type, 32, self.n_input_size - 1),
                    activation_fn(),

                    nn.Conv2d(32, 64, (2, 2)),
                    get_norm_layer_2d(self.norm_type, 64, self.n_input_size - 2),
                    activation_fn(),

                    nn.Conv2d(64, 64, (2, 2)),
                    get_norm_layer_2d(self.norm_type, 64, self.n_input_size - 3),
                    activation_fn(),

                    nn.Flatten(),
                )
            else:
                self.cnn = nn.Sequential(
                    get_norm_layer_2d(self.norm_type, n_input_channels, self.n_input_size),

                    nn.Conv2d(n_input_channels, 32, (2, 2), stride=1, padding=1),
                    get_norm_layer_2d(self.norm_type, 32, self.n_input_size + 1),
                    activation_fn(),

                    nn.Conv2d(32, 64, (2, 2), stride=1, padding=0),
                    get_norm_layer_2d(self.norm_type, 64, self.n_input_size),
                    activation_fn(),

                    nn.Conv2d(64, 64, (2, 2), stride=1, padding=0),
                    get_norm_layer_2d(self.norm_type, 64, self.n_input_size - 1),
                    activation_fn(),

                    nn.Flatten(),
                )

        elif model_type == 1:
            if self.n_input_size == 84:
                image_sizes = [20, 9, 6]
                kernel_sizes = [8, 4, 4]
                strides = [4, 2, 1]
            elif self.n_input_size == 64:
                image_sizes = [15, 6, 3]
                kernel_sizes = [8, 4, 4]
                strides = [4, 2, 1]
            elif self.n_input_size == 48:
                image_sizes = [11, 4, 1]
                kernel_sizes = [8, 4, 4]
                strides = [4, 2, 1]
            elif self.n_input_size == 42:
                image_sizes = [19, 9, 7]
                kernel_sizes = [5, 3, 3]
                strides = [2, 2, 1]
            elif self.n_input_size == 36:
                image_sizes = [11, 4, 1]
                kernel_sizes = [6, 4, 4]
                strides = [3, 2, 1]
            elif self.n_input_size == 20:
                image_sizes = [9, 7, 5]
                kernel_sizes = [4, 3, 3]
                strides = [2, 1, 1]
            else: raise NotImplementedError

            # Smaller CNN for ProcGen games
            self.cnn = nn.Sequential(
                get_norm_layer_2d(self.norm_type, n_input_channels, self.n_input_size),

                nn.Conv2d(in_channels=n_input_channels, out_channels=32, kernel_size=kernel_sizes[0], stride=strides[0], padding=0),
                get_norm_layer_2d(self.norm_type, 32, image_sizes[0]),
                activation_fn(),

                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_sizes[1], stride=strides[1], padding=0),
                get_norm_layer_2d(self.norm_type, 64, image_sizes[1]),
                activation_fn(),

                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_sizes[2], stride=strides[2], padding=0),
                get_norm_layer_2d(self.norm_type, 64, image_sizes[2]),
                activation_fn(),

                nn.Flatten(),
            )

        elif model_type == 2:
            # IMPALA CNNs
            layer_scale = 1
            self.build_impala_cnn(
                depths=[
                    16 * layer_scale,
                    32 * layer_scale,
                    32 * layer_scale
                ]
            )

            with th.no_grad():
                samples = th.as_tensor(observation_space.sample()[None]).float()
                if self.n_input_channels == 1:
                    samples = samples[:, -1:, :, :]
                for block in self.impala_conv_sequences:
                    samples = block(samples)
                samples = th.flatten(samples, start_dim=1)
                n_flatten = samples.shape[1]

            self.impala_flattern = nn.Sequential(
                nn.Flatten(),
                get_norm_layer_1d(self.norm_type, n_flatten),
                activation_fn(),
            )
        else:
            raise NotImplementedError

        if n_flatten is None:
            with th.no_grad():
                samples = th.as_tensor(observation_space.sample()[None]).float()
                if self.n_input_channels == 1:
                    samples = samples[:, -1:, :, :]
                n_flatten = self.cnn(samples).shape[1]

        if self.cnn_fc_layers == 1:
            self.linear_layer = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                get_norm_layer_1d(self.norm_type, features_dim),
                activation_fn(),
            )
        elif self.cnn_fc_layers == 2:
            self.linear_layer = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                get_norm_layer_1d(self.norm_type, features_dim),
                activation_fn(),

                nn.Linear(features_dim, features_dim),
                get_norm_layer_1d(self.norm_type, features_dim),
                activation_fn(),
            )
        elif self.cnn_fc_layers == 3:
            self.linear_layer = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                get_norm_layer_1d(self.norm_type, features_dim),
                activation_fn(),

                nn.Linear(features_dim, features_dim),
                get_norm_layer_1d(self.norm_type, features_dim),
                activation_fn(),

                nn.Linear(features_dim, features_dim),
                get_norm_layer_1d(self.norm_type, features_dim),
                activation_fn(),
            )
        else:
            raise ValueError

    def build_impala_cnn(self, depths=None):
        """
        Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
        Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561

        This function is created based on
        https://github.com/openai/baselines/blob/master/baselines/common/models.py
        """
        if depths is None:
            depths = [16, 32, 32]
        if self.n_input_size == 7:
            # MiniGrid
            image_sizes = [
                [7, 4],
                [4, 2],
                [2, 1],
            ]
        elif self.n_input_size == 64:
            # ProcGen
            image_sizes = [
                [64, 32],
                [32, 16],
                [16, 8],
            ]
        elif self.n_input_size == 84:
            # Atari
            image_sizes = [
                [84, 42],
                [42, 21],
                [21, 11],
            ]
        else:
            raise NotImplementedError

        self.impala_conv_sequences = []
        for depth_i in range(len(depths)):
            d_in = self.n_input_channels if depth_i == 0 else depths[depth_i - 1]
            d_out = depths[depth_i]

            module = nn.Sequential(
                get_norm_layer_2d(self.norm_type, d_in, image_sizes[depth_i][0]),
                nn.Conv2d(in_channels=d_in, out_channels=d_out, kernel_size=(3, 3), stride=1, padding=1),
                nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            )

            mod_name = f'impala_blk_{depth_i}_prep'
            setattr(self, mod_name, module)
            self.impala_conv_sequences.append(getattr(self, mod_name))

            for res_block_i in range(2):
                res_blk_module = nn.Sequential(
                    get_norm_layer_2d(self.norm_type, d_out, image_sizes[depth_i][1]),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=d_out, out_channels=d_out, kernel_size=(3, 3), stride=1, padding=1),

                    get_norm_layer_2d(self.norm_type, d_out, image_sizes[depth_i][1]),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=d_out, out_channels=d_out, kernel_size=(3, 3), stride=1, padding=1),
                )

                res_blk_name = f'impala_blk_{depth_i}_res_{res_block_i}'
                setattr(self, res_blk_name, res_blk_module)
                self.impala_conv_sequences.append(getattr(self, res_blk_name))

    def forward(self, observations: Tensor) -> Tensor:
        if self.n_input_channels == 1:
            observations = observations[:, -1:, :, :]
        if self.model_type == 2:
            # IMPALA CNNs
            outputs = observations.float()
            for block in self.impala_conv_sequences:
                outputs = block(outputs)
            outputs = self.impala_flattern(outputs)
        else:
            outputs = self.cnn(observations)
        return self.linear_layer(outputs)


class CnnFeaturesExtractor(CustomCnnFeaturesExtractor):

    def __init__(self, observation_space,
                 features_dim: int = 256,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 model_type: int = 0,
                 cnn_fc_layers: int = 1,
                 n_input_channels: int = 0):
        # Specifying normalization type
        self.norm_type = NormType.NoNorm
        super().__init__(observation_space, features_dim,
                         activation_fn, model_type, cnn_fc_layers, n_input_channels)


class BatchNormCnnFeaturesExtractor(CustomCnnFeaturesExtractor):

    def __init__(self, observation_space,
                 features_dim: int = 256,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 model_type: int = 0,
                 cnn_fc_layers: int = 1,
                 n_input_channels: int = 0):
        # Specifying normalization type
        self.norm_type = NormType.BatchNorm
        super().__init__(observation_space, features_dim,
                         activation_fn, model_type, cnn_fc_layers, n_input_channels)


class LayerNormCnnFeaturesExtractor(CustomCnnFeaturesExtractor):

    def __init__(self, observation_space,
                 features_dim: int = 256,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 model_type: int = 0,
                 cnn_fc_layers: int = 1,
                 n_input_channels: int = 0):
        # Specifying normalization type
        self.norm_type = NormType.LayerNorm
        super().__init__(observation_space, features_dim,
                         activation_fn, model_type, cnn_fc_layers, n_input_channels)

