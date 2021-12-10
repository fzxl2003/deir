from enum import Enum


class ModelType(Enum):
    NoModel = 0
    ICM = 1  # Forward + Inverse
    RND = 2
    NGU = 3
    NovelD = 4  # Inverse
    DEIR = 5
    PlainForward = 6
    PlainInverse = 7
    PlainDiscriminator = 8


class NormType(Enum):
    NoNorm = 0
    BatchNorm = 1
    LayerNorm = 2


class EnvSrc(Enum):
    MiniGrid = 0
    ProcGen = 1
    Atari = 2
