from src.data_loader.utils.enums import CameraType, RipenessLabel, DatasetSplit, FruitType

CAMERA_SPECS = {
    CameraType.FX10: {
        "description": "Specim FX10 (400-1000 nm)",
        "total_bands": 224,
    },
    CameraType.REDEYE: {
        "description": "Innospec Redeye (950-1700 nm)",
        "total_bands": 249,
    },
    CameraType.MICRO_HSI: {
        "description": "Corning MicroHSI",
        "total_bands": 252,
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

# We only use labeled training data
# Insights to strucutre: 
# We have identified issue with the from the author given split, duplicate records, and non matching number of records according to the papers
# We have fixed these with another script random creating splitting into train/test/val while ensuring same fruits are in the same set
# Somehow we have issues with merging the newly created files together and looked into the issue 
# As a simple workaround (as we effectivly focus in FX10 adn Acocado/Kiwi) we create this dict of dicts to route to the new paths. 
'''
SPLIT_FILES = {
    DatasetSplit.TRAIN: 
        {FruitType.AVOCADO : "train_only_labeled.json",
         FruitType.KIWI: "train_kiwi_grouped.json"}, 
    DatasetSplit.VAL: 
        {FruitType.AVOCADO : "val.json",
         FruitType.KIWI: "val_kiwi_grouped.json"}, 
    DatasetSplit.TEST: 
        {FruitType.AVOCADO : "test.json",
         FruitType.KIWI: "test_kiwi_grouped.json"}
}
'''

SPLIT_FILES = {
    DatasetSplit.TRAIN: 
        {FruitType.AVOCADO : "train_avocado_grouped.json",
         FruitType.KIWI: "train_kiwi_grouped.json"}, 
    DatasetSplit.VAL: 
        {FruitType.AVOCADO : "val_avocado_grouped.json",
         FruitType.KIWI: "val_kiwi_grouped.json"}, 
    DatasetSplit.TEST: 
        {FruitType.AVOCADO : "test_avocado_grouped.json",
         FruitType.KIWI: "test_kiwi_grouped.json"}
}