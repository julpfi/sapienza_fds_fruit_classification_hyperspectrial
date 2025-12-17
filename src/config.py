from src.data_loader.utils.enums import FruitType, CameraType

defined_models = {0:"deephs", 1: "fruiths_net", 2:"hybrid", 3: "swin", 4:"deit",
                   5:"lit", 6: "spec_spat_ft"}

defined_band_reduction_strategies = {0:"all", 1:"uniform", 2:"average", 
                                     3:"gaussian_average", 4:"dft", 5:"dft_complex"}


CONFIG = {
    
    # Names for wandd 
    "project_name": "sapienza_fds_fruit_ripeness",
    "run_name": "avocado_30bands_average_fruiths", 
    
    #TODO NEW RUN: Updated run_name and update selection of model and data
    # Model and data 
    "fruit": FruitType.AVOCADO,
    "camera": CameraType.FX10,  #We will only use the FX10
    "num_classes": 3,
    "model_type": defined_models[1],
    "bands": [224, 30, 10, 3][1],
    "band_selection": [None, (700, 1100)][0],
    "band_reduction": defined_band_reduction_strategies[2],
    "img_size": [(224, 224), (64, 64)][0],   # deit and swin take only 224x224
    
    # Hyperparameters
    "batch_size": 16,
    "epochs": 25,
    "lr": 1e-4,
    "num_workers": 2,
    
    # Paths (mounted drive)
    "data_root": "/content/drive/MyDrive/sapienza_fds_fruit_classification/data",
    "json_root": "/content/drive/MyDrive/sapienza_fds_fruit_classification/data/dataset",
    "save_dir": "/content/drive/MyDrive/sapienza_fds_fruit_classification/checkpoints"
}
