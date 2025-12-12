from src.data_loader.utils.enums import CameraType, RipenessLabel, DatasetSplit

CAMERA_SPECS = {
    CameraType.FX10: {
        "description": "Specim FX10 (400-1000 nm)",
        "total_bands": 224,
    },
    CameraType.REDEYE: {
        "description": "Innospec Redeye (950-1700 nm)",
        "total_bands": 252,
    },
    CameraType.MICRO_HSI: {
        "description": "Corning MicroHSI",
        "total_bands": 150, # TODO CHECK EXACT NUMBER
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
    DatasetSplit.TRAIN: "train_avocado_kiwi_vis.json", # We only use labeled training data
    DatasetSplit.VAL: "val_avocado__kiwi_vis.json",
    DatasetSplit.TEST: "test_avocado_kiwi_vis.json"
}