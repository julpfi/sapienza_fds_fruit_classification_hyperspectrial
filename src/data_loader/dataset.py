import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from src.data_loader.utils.enums import FruitType, CameraType, DatasetSplit
from src.data_loader.utils.config import LABEL_MAP
from src.data_loader.utils.band_reducer import BandReducer
from src.data_loader.utils.band_selector import BandSelector 
from src.data_loader.utils.utils import load_and_preprocess

class HyperspectralFruitDataset(Dataset):
    def __init__(self, 
                 json_path, 
                 data_root, 
                 split: DatasetSplit, 
                 fruit_type: FruitType,
                 camera_type: CameraType = None,
                 band_selection=None,
                 band_reduction='uniform', 
                 target_bands=30, 
                 img_size=(224, 224)):

        """
        Args:
            json_path: path to json file holding dataset annotations
            data_root: path to root folder holding hyperspectrial data (.hdr and .bin files)
            fruit_type: enum see py file (e.g. FruitType.AVOCADO)
            camera_type: enum see py file (e.g. CameraType.FX10)
            band_reduction: strategy for band reduction 
            target_bands: number of bands after reduction 
            selection_bounds: (min_nm, max_nm) to select specific wavelength range
            img_size: desired image size (Height, Width)
        """
        self.data_root = data_root
        self.target_fruit = fruit_type
        self.target_camera = camera_type
        self.img_size = img_size
        self.band_selection = band_selection

        self.wavelengths = [] 
        self.samples = self._parse_json(json_path)        

        self.band_selector = BandSelector(self.wavelengths, band_selection)
        self.band_reducer = BandReducer(strategy=band_reduction, target_bands=target_bands)        
        self.transform = self._get_transforms(is_train=(split==DatasetSplit.TRAIN)) 


    def __len__(self): return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and preprocess data: mask background and borders
        img_data = load_and_preprocess(sample['hdr'], sample['bin'])
        
        # TODO Monitor if this check is need => Gets random sample ensure batch size is maintained
        # Safety check: Loading and preprocessing worked
        if img_data is None:
            print(f"Failed to load index {idx}. Skipping to random sample...")
            new_idx = np.random.randint(0, len(self))
            return self.__getitem__(new_idx)
        
    
        img_data = self.band_selector(img_data)
        img_data = self.band_reducer(img_data)

        if isinstance(img_data, np.ndarray):
            img_data = img_data.astype(np.float32)
        tensor = self.transform(img_data)
        tensor = tensor.float()
        return tensor, torch.tensor(sample['label'], dtype=torch.long)
    

    def _parse_json(self, json_path):
        """
        Parses the dataset annoations json and filters samples based on fruit and camera type
        Returns a list of valid samples with file paths and labels
        """ 
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract wavelengths from camera info
        if self.target_camera is not None and 'cameras' in data:
            for cam in data['cameras']:
                if cam.get('id') == self.target_camera.value:
                    self.wavelengths = cam.get('wavelengths', [])
                    break

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
        '''
         Retruns torchvision transforms for data augmentatiion and preprocessing 
        '''
        if is_train:
            return transforms.Compose([
                transforms.ToTensor(), # (H, W, C) -> (C, H, W)
                transforms.Resize(self.img_size, antialias=True),

                # TODO Assess diff of resize adn ranrezisecrop
                #transforms.RandomResizedCrop(
                #    size=self.img_size, 
                #    scale=(0.9, 1.0), 
                #    ratio=(0.95, 1.05),
                #    antialias=True
                #),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),   
            ])
            
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.img_size[0], antialias=True),
                transforms.CenterCrop(self.img_size)
            ])