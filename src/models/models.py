from src.models.deit_model import DeiTModel
from src.models.vit_model import VitModel


def get_model(model_type: str, pretrained:bool, num_classes:int, in_channels:int=3):
    return DeiTModel()
