from enum import Enum

class FruitType(Enum):
    AVOCADO = "Avocado"
    KIWI = "Kiwi"
    MANGO = "Mango"
    KAKI = "Kaki"
    PAPAYA = "Papaya"

class CameraType(Enum):
    FX10 = "VIS"
    REDEYE = "NIR"
    MICRO_HSI = "VIS_COR"

class RipenessLabel(Enum):
    UNRIPE = 0
    RIPE = 1
    OVERRIPE = 2

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"