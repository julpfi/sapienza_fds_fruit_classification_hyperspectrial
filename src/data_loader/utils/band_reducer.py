import numpy as np

class BandReducer:
    def __init__(self, strategy='all', target_bands:int|list[int]=None):
        self.strategy = strategy
        self.target_bands = target_bands

    def __call__(self, img_data):
        # TODO Think about and add strategies -> aligned with models (e.g. also PCA again)
        # TODO Connect this selectio to the model factory/selection 
        # img_data shape is (H, W, Bands)
        total_channels = img_data.shape[2]

        if self.strategy == 'indices' and isinstance(self.target_bands, list):
            valid_indices = [b for b in self.target_bands if b < total_channels]
            return img_data[:, :, valid_indices]

        elif self.strategy == 'uniform':
            # Select k bands evenly spaced across the spectrum
            indices = np.linspace(0, total_channels - 1, int(self.target_bands), dtype=int)
            return img_data[:, :, indices]

        return img_data