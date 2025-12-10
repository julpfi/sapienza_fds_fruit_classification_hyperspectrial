from src.data_loader.utils.enums import FruitType, CameraType

defined_models = {0:"fruiths_net", 1: "hybrid", 2: "swin", 3:"deit", 4:"lit_spectral_transformer" ,5:""}
defined_band_reduction_strategies = {0:"all", 1:"uniform", 2:"average", 3:"gaussian_average", 4:"dft"}


CONFIG = {
    
    # Names for wandd 
    "project_name": "sapienza_fds_fruit_ripeness",
    "run_name": "papaya_lit_spectral_transformer_allbands_224px_30epochs", 
    
    #TODO NEW RUN: Updated run_name and update selection of model and data
    # Model and data 
    "fruit": FruitType.PAPAYA,
    "camera": CameraType.FX10,
    "num_classes": 3,
    "model_type": defined_models[2],
    "bands": [224, 30, 10, 3][1],
    "band_selection": [None, (700, 1100)][0],
    "band_reduction": defined_band_reduction_strategies[3],
    "img_size": [(224, 224), (64, 64)][0],
    
    # Hyperparameters
    "batch_size": 16,
    "epochs": 30,
    "lr": 1e-4,
    "num_workers": 2,
    
    # Paths (mounted drive)
    "data_root": "/content/drive/MyDrive/sapienza_fds_fruit_classification/data",
    "json_root": "/content/drive/MyDrive/sapienza_fds_fruit_classification/data/dataset",
    "save_dir": "/content/drive/MyDrive/sapienza_fds_fruit_classification/checkpoints"
}
