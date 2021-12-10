import numpy as np
import torch as th

from torch import nn
from typing import Optional, Callable
from src.utils.enum_types import NormType, EnvSrc, ModelType


def bkdr_hash(inputs, seed=131, mask=0x7fffffff):
    if isinstance(inputs, np.ndarray):
        data = inputs.reshape(-1)
    else:
        data = inputs
    res = 0
    is_str = isinstance(data, str)
    get_val = lambda x: ord(val) if is_str else val
    for val in data:
        int_val = get_val(val)
        res = (res * seed) & mask
        res = (res + int_val) & mask
    return res


def get_enum_model_type(model_type):
    if isinstance(model_type, ModelType):
        return model_type
    if isinstance(model_type, str):
        model_type = model_type.strip().lower()
        if model_type == "icm":
            return ModelType.ICM
        elif model_type == "rnd":
            return ModelType.RND
        elif model_type == "ngu":
            return ModelType.NGU
        elif model_type == "noveld":
            return ModelType.NovelD
        elif model_type == "deir":
            return ModelType.DEIR
        elif model_type == "plainforward":
            return ModelType.PlainForward
        elif model_type == "plaininverse":
            return ModelType.PlainInverse
        elif model_type == "plaindiscriminator":
            return ModelType.PlainDiscriminator
        else:
            return ModelType.NoModel
    raise ValueError


def get_enum_env_src(env_src):
    if isinstance(env_src, EnvSrc):
        return env_src
    if isinstance(env_src, str):
        env_src = env_src.strip().lower()
        if env_src == 'minigrid':
            return EnvSrc.MiniGrid
        if env_src == 'procgen':
            return EnvSrc.ProcGen
        if env_src == 'atari':
            return EnvSrc.Atari
    raise ValueError


def get_enum_norm_type(norm_type):
    if isinstance(norm_type, NormType):
        return norm_type
    if isinstance(norm_type, str):
        norm_type = norm_type.strip().lower()
        if norm_type == 'batchnorm':
            return NormType.BatchNorm
        if norm_type == 'layernorm':
            return NormType.LayerNorm
        if norm_type == 'nonorm':
            return NormType.NoNorm
    raise ValueError


def get_norm_layer_1d(norm_type, fetures_dim, momentum=0.1):
    norm_type = get_enum_norm_type(norm_type)
    if norm_type == NormType.BatchNorm:
        return nn.BatchNorm1d(fetures_dim, momentum=momentum)
    if norm_type == NormType.LayerNorm:
        return nn.LayerNorm(fetures_dim)
    if norm_type == NormType.NoNorm:
        return nn.Identity()
    raise NotImplementedError


def get_norm_layer_2d(norm_type, n_channels, n_size, momentum=0.1):
    norm_type = get_enum_norm_type(norm_type)
    if norm_type == NormType.BatchNorm:
        return nn.BatchNorm2d(n_channels, momentum=momentum)
    if norm_type == NormType.LayerNorm:
        return nn.LayerNorm([n_channels, n_size, n_size])
    if norm_type == NormType.NoNorm:
        return nn.Identity()
    raise NotImplementedError


def normalize_rewards(norm_type, rewards, mean, std, eps=1e-5):
    if norm_type <= 0:
        return rewards

    if norm_type == 1:
        # Standardization
        return (rewards - mean) / (std + eps)

    if norm_type == 2:
        # Min-max normalization
        min_int_rew = np.min(rewards)
        max_int_rew = np.max(rewards)
        mean_int_rew = (max_int_rew + min_int_rew) / 2
        return (rewards - mean_int_rew) / (max_int_rew - min_int_rew + eps)

    if norm_type == 3:
        # Standardization without subtracting the mean
        return rewards / (std + eps)


def set_random_seed(self, seed: Optional[int] = None) -> None:
    """
    From Stable Baslines 3
    Set the seed of the pseudo-random generators
    (python, numpy, pytorch, gym, action_space)

    :param seed:
    """
    if seed is None:
        return
    set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)
    self.action_space.seed(seed)

    if self.env is not None:
        seed_method = getattr(self.env, "seed_method", None)
        if seed_method is not None:
            self.env.seed(seed)
    if self.eval_env is not None:
        seed_method = getattr(self.eval_env, "seed_method", None)
        if seed_method is not None:
            self.eval_env.seed(seed)


# Module Initialization
def init_module_with_name(n: str, m: nn.Module, fn: Callable[['Module'], None] = None) -> nn.Module:

    def _reset_parameters(module: nn.Module) -> None:
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()

    has_child = False
    for name, module in m.named_children():
        has_child = True
        init_module_with_name(n + '.' + name, module, fn)
    if not has_child:
        run_id = th.initial_seed()
        hash_val = bkdr_hash(n)
        hash_val = bkdr_hash([hash_val, run_id])
        th.manual_seed(hash_val)
        if fn is None:
            _reset_parameters(m)
        else:
            fn(m)
        th.manual_seed(run_id)
    return m


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'