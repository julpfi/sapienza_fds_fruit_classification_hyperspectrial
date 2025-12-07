import os
from src.data_loader.data_loader import get_data_loader
from src.data_loader.utils.enums import FruitType, CameraType, DatasetSplit
from src.data_loader.dataset import HyperspectralFruitDataset
from src.data_loader.utils.enums import DatasetSplit
from src.data_loader.utils.config import SPLIT_FILES


CONFIG = {
    "project_name": "sapienza_fds_fruit_ripeness",

    "run_name": "TEST_01", 
    "fruit": FruitType.KIWI,
    "camera": CameraType.FX10,

    "img_size": (224, 224),
    "band_strategy" : 'uniform',
    "bands": 30,
    
    # Hyperparameters
    "batch_size": 16,
    "epochs": 15,
    "lr": 1e-4,
    "num_workers": 2,
    
    # Paths
    "data_root": "/content/drive/MyDrive/sapienza_fds_fruit_classification/data",
    "json_root": "/content/drive/MyDrive/sapienza_fds_fruit_classification/data/dataset" 
}

def main():

    # Resolve JSON path
    json_filename = SPLIT_FILES[DatasetSplit.TEST]
    json_path = os.path.join(CONFIG['json_root'], json_filename)
    
    print(f"Loading {DatasetSplit.TEST.name} data from: {json_filename}")

    dataset_test = HyperspectralFruitDataset(
        json_path=json_path,
        data_root=CONFIG['data_root'],
        split=DatasetSplit.TEST,
        fruit_type=CONFIG['fruit'],
        camera_type=CONFIG['camera'],
        band_strategy=CONFIG['band_strategy'],
        target_bands=CONFIG['bands'],
        img_size=CONFIG['img_size']
    )

    json_filename = SPLIT_FILES[DatasetSplit.TRAIN]
    json_path = os.path.join(CONFIG['json_root'], json_filename)
    
    print(f"Loading {DatasetSplit.TRAIN.name} data from: {json_filename}")

    dataset_train = HyperspectralFruitDataset(
        json_path=json_path,
        data_root=CONFIG['data_root'],
        split=DatasetSplit.TRAIN,
        fruit_type=CONFIG['fruit'],
        camera_type=CONFIG['camera'],
        band_strategy=CONFIG['band_strategy'],
        target_bands=CONFIG['bands'],
        img_size=CONFIG['img_size']
    )

    json_filename = SPLIT_FILES[DatasetSplit.VAL]
    json_path = os.path.join(CONFIG['json_root'], json_filename)
    
    print(f"Loading {DatasetSplit.VAL.name} data from: {json_filename}")

    dataset_val = HyperspectralFruitDataset(
        json_path=json_path,
        data_root=CONFIG['data_root'],
        split=DatasetSplit.VAL,
        fruit_type=CONFIG['fruit'],
        camera_type=CONFIG['camera'],
        band_strategy=CONFIG['band_strategy'],
        target_bands=CONFIG['bands'],
        img_size=CONFIG['img_size']
    )

    print("Train:", len(dataset_train))
    print("Test:",len(dataset_test))
    print("Val:",len(dataset_val))

    


if __name__ == "__main__":
    main()