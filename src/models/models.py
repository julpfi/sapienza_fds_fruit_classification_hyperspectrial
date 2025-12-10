from src.models.deit_model import DeiTModel
from src.models.hybrid_model import HybridModel
from src.models.lit_spectral_transformer import LitSpectralTransformer
from src.models.fruiths_net import FruitHSNet 
from src.models.swin_model import SwinModel

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

    elif model_type == "deit": 
        return DeiTModel(pretrained=True, num_classes=num_classes, in_channels=in_channels)
    
    elif model_type == "hybrid": # Two version: first cnn layer as band reducer or already reduced bands as input 
        return HybridModel(in_channels=in_channels, num_classes=num_classes, reduce_bands=(reduction_strategy == "all"))
    
    elif model_type == "swin":
        return SwinModel(num_classes=num_classes, in_channels=in_channels, pretrained=True)
    
    raise ValueError(f"Unsupported model type: {model_type} or incompatible band reduction strategy: {reduction_strategy} for in_channels: {in_channels}")