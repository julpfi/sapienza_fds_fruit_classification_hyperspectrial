import os
from torch.utils.data import DataLoader

from src.data_loader.dataset import HyperspectralFruitDataset
from src.data_loader.utils.enums import DatasetSplit 
from src.data_loader.utils.config import SPLIT_FILES


def get_data_loader(CONFIG:dict, split: DatasetSplit, shuffle=False):
    """
    Wrapper: Creates a DataLoader for a specific Dataset and assocaited split (train, val, test)
    Automatically selects the correct JSON file
    """

    # Resolve JSON path
    json_filename = SPLIT_FILES[split][CONFIG['fruit']]
    json_path = os.path.join(CONFIG['json_root'], json_filename)
    
    print(f"Loading {split.name} data from: {json_filename}")

    dataset = HyperspectralFruitDataset(
        json_path=json_path,
        data_root=CONFIG['data_root'],
        split=split,
        fruit_type=CONFIG['fruit'],
        camera_type=CONFIG['camera'],
        band_reduction=CONFIG['band_reduction'],
        band_selection=CONFIG['band_selection'],
        target_bands=CONFIG['bands'],
        img_size=CONFIG['img_size']
    )
    
    return DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=shuffle, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )