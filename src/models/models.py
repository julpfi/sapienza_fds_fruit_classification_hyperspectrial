from src.models.deit_model import DeiTModel
from src.models.hybrid_model import HybridModel
from src.models.lit_spectral_transformer import LitSpectralTransformer
from src.models.fruiths_net import FruitHSNet 
from src.models.swin_model import SwinModel
from src.models.spec_spat_ft_model import SpecSpatFTModel
from src.models.deephs_net import DeephsNet 

def get_model(config: dict):
    num_classes = config.get("num_classes", 3)
    in_channels = config.get("bands", 224)
    model_type = config.get("model_type", "attention_cnn")
    reduction_strategy = config.get("band_reduction", "all")
    image_size = config.get("img_size", (224, 224))

    # BASELINE PAPER 1
    if model_type == "deephs": 
        return DeephsNet(bands=in_channels)
    
    # BASELINE PAPER 2
    elif model_type == "fruiths_net":
        # Note: use_3x3_center=True is recommended because we use RandomResizedCrop in dataset.py
        return FruitHSNet(num_bands=in_channels, num_classes=num_classes, use_3x3_center=True)
    
    elif model_type == "lit":
        is_complex = (reduction_strategy == "dft_complex")
        actual_channels = in_channels * 2 if is_complex else in_channels
        return LitSpectralTransformer(bands=actual_channels, num_classes=num_classes
                                      , is_complex_dft_reduce=is_complex) # Variation dft with real and imag. part -> 1d conv takes 2 channels as input

    elif model_type == "deit" and image_size == (224, 224): 
        return DeiTModel(pretrained=True, num_classes=num_classes, in_channels=in_channels)
    
    elif model_type == "hybrid": # Two version: first cnn layer as band reducer or already reduced bands as input 
        return HybridModel(in_channels=in_channels, num_classes=num_classes
                           , reduce_bands=(reduction_strategy == "all")) # Variation: All bands (no band reduction) -> one more cnn layer for reduction
    
    elif model_type == "swin" and image_size == (224, 224):
        return SwinModel(in_channels=in_channels, num_classes=num_classes,pretrained=True)
    
    elif model_type == "spec_spat_ft" and reduction_strategy == "dft_complex": 
        crop_size = 16 if image_size == (64, 64) else 48
        actual_channels = in_channels * 2
        return SpecSpatFTModel(num_classes=num_classes, in_channels=actual_channels,crop_size=crop_size)
    
    raise ValueError(f"Unsupported model type: {model_type} or incompatible band reduction strategy: {reduction_strategy} for in_channels: {in_channels}")