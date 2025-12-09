from src.models.deit_model import DeiTModel
from src.models.vit_model import VitModel
from src.models.attention_cnn_model import AttentionCNNSelection, AttentionCNNModelUnselected
from src.models.lit_spectral_transformer import LitSpectralTransformer
from src.models.fruiths_net import FruitHSNet 

def get_model(config: dict):
    num_classes = config.get("num_classes", 3)
    in_channels = config.get("bands", 224)
    model_type = config.get("model_type", "attention_cnn")
    reduction_strategy = config.get("band_reduction", "all")

    # Check for Fruit-HSNet
    if model_type == "fruiths_net":
        # Note: use_3x3_center=True is recommended because we use RandomResizedCrop in dataset.py
        return FruitHSNet(num_bands=in_channels, num_classes=num_classes, use_3x3_center=True)
    
    elif model_type == "lit_spectral_transformer":
        return LitSpectralTransformer(bands=in_channels, num_classes=num_classes)

    elif model_type in ["deit", "deit_undist"]:
        return DeiTModel(pretrained=True, use_distillation=(model_type=="deit"), num_classes=num_classes, input_bands=in_channels)
    
    elif model_type == "vit":
        return VitModel(pretrained=True, num_classes=num_classes, in_channels=in_channels)
    
    elif model_type == "attention_cnn":
        if reduction_strategy == "all" and in_channels > 30:
            return AttentionCNNModelUnselected(num_bands=in_channels, num_classes=num_classes)
        elif reduction_strategy in ["uniform", "average", "gaussian_average"] and in_channels in [30, 10]: 
            return AttentionCNNSelection(num_bands=in_channels, num_classes=num_classes)
    
    raise ValueError(f"Unsupported model type: {model_type} or incompatible band reduction strategy: {reduction_strategy} for in_channels: {in_channels}")