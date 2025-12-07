from src.data_loader.enums import CameraType, RipenessLabel, DatasetSplit

CAMERA_SPECS = {
    CameraType.FX10: {
        "description": "Specim FX10 (400-1000 nm)",
        "total_bands": 224,
        "rgb_indices": [30, 60, 100], 
    },
    CameraType.REDEYE: {
        "description": "Innospec Redeye (950-1700 nm)",
        "total_bands": 252,
        # TODO FIX NIR has no RGB 
        "rgb_indices":  None # [50, 120, 200],
    },
    CameraType.MICRO_HSI: {
        "description": "Corning MicroHSI",
        "total_bands": 150, # TODO CHECK EXACT NUMBER
        "rgb_indices": [20, 70, 120],
    }
}

LABEL_MAP = {
    # .value so it maps to int directly -> can be used in cross entropy loss
    "unripe": RipenessLabel.UNRIPE.value,
    "perfect": RipenessLabel.RIPE.value,
    "ripe": RipenessLabel.RIPE.value,
    "overripe": RipenessLabel.OVERRIPE.value,
    "near_overripe": RipenessLabel.OVERRIPE.value 
}

SPLIT_FILES = {
    DatasetSplit.TRAIN: "train_only_labeled_v2.json", # We only use labeled training data
    DatasetSplit.VAL: "val_v2.json",
    DatasetSplit.TEST: "test_v2.json"
}