import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from src.data_loader.enums import FruitType, CameraType, DatasetSplit
from src.data_loader.config import LABEL_MAP
from src.data_loader.band_reducer import BandReducer
from src.data_loader.utils import load_and_preprocess


class HyperspectralFruitDataset(Dataset):
    def __init__(self, 
                 json_path, 
                 data_root, 
                 split: DatasetSplit, 
                 fruit_type: FruitType,
                 camera_type: CameraType = None,
                 band_strategy='uniform', 
                 target_bands=30, 
                 img_size=(224, 224)):
        """
        Args:
            json_path: path to json file holding dataset annotations
            data_root: path to root folder holding hyperspectrial data (.hdr and .bin files)
            fruit_type: enum see py file (e.g. FruitType.AVOCADO)
            camera_type: enum see py file (e.g. CameraType.FX10)
        """

        self.data_root = data_root
        self.target_fruit = fruit_type
        self.target_camera = camera_type
        self.img_size = img_size

        print(json_path) # TODO REMOVE DEBUGGING 
        print(data_root) # TODO REMOVE DEBUGGING 
        self.samples = self._parse_json(json_path)        
        self.band_reducer = BandReducer(strategy=band_strategy, target_bands=target_bands)
        self.transform = self._get_transforms(is_train=(split==DatasetSplit.TRAIN)) 

    
    def __len__(self): return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and preprccess data: calibrate, mask background, and borders
        img_data = load_and_preprocess(sample['hdr'], sample['bin'])
        
        # TODO Monitor if this check is need => Gets random sample ensure batch size is maintained
        # Safety check: Loading and preprocessing worked 
        if img_data is None:
            print(f"Failed to load index {idx}. Skipping to random sample...")
            new_idx = np.random.randint(0, len(self))
            return self.__getitem__(new_idx)

        # Reduce bands according to strategy 
        img_data = self.band_reducer(img_data)
        
        # Transform according to defined compose pipeline 
        tensor = self.transform(img_data)
        
        return tensor, torch.tensor(sample['label'], dtype=torch.long)


    def _parse_json(self, json_path):
        """
        Parses the dataset annoations json and filters samples based on fruit and camera type
        Returns a list of valid samples with file paths and labels
        """ 
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Get all records matching fruit and camera type
        id_to_record = {}
        for rec in data['records']:
            # Filter 1: Fruit Type
            if rec.get('fruit') != self.target_fruit.value:
                continue
            
            # Filter 2: Camera Type
            if self.target_camera and rec.get('camera_type') != self.target_camera.value:
                continue

            id_to_record[rec['id']] = rec

        # Match annotations to records and build sample list
        samples = []
        for a in data['annotations']:
            record_id = a['record_id']
            
            if record_id in id_to_record:
                rec = id_to_record[record_id]
                label_str = a.get('ripeness_state', '').lower()

                if label_str in LABEL_MAP:
                    full_hdr = os.path.join(self.data_root, rec['files']['header_file'])
                    full_bin = os.path.join(self.data_root, rec['files']['data_file'])
                    print(full_bin) # TODO REMOVE DEBUGGING 
                    if os.path.exists(full_hdr) and os.path.exists(full_bin):
                        samples.append({
                            'hdr': full_hdr,
                            'bin': full_bin,
                            'label': LABEL_MAP[label_str], # Converts ripness label from annoation into int encoding 
                            'fruit': rec['fruit'],
                            'id': rec['id']
                        })
        
        print(f"Loaded {len(samples)} samples for {self.target_fruit.value}")
        return samples

    
    def _get_transforms(self, is_train=False):
        """
        Returns the appropriate transform pipeline 
            Train: ... <define pipeline>
            Eval: ... <define pipeline>
        """
        # TODO: Possible to add noise here (noise injection)
        # TODO: Look into masking possibilites
        # TODO: Or maybe expereient with construction more sample by rotating or flipping the images? 
        if is_train:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.img_size, antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.img_size, antialias=True)
            ])
    
