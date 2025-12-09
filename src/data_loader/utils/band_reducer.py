import numpy as np
from scipy.stats import norm

class BandReducer:
    def __init__(self, strategy='all', target_bands:int|list[int]=None):
        self.strategy = strategy
        self.target_bands = target_bands

    def __call__(self, img_data):
        # TODO Think about and add strategies -> aligned with models (e.g. also PCA again)
        # TODO Connect this selectio to the model factory/selection 
        # img_data shape is (H, W, Bands)
        h, w, c = img_data.shape

        if self.strategy == 'uniform':
            # Select k bands evenly spaced across the spectrum
            indices = np.linspace(0, c - 1, int(self.target_bands), dtype=int)
            return img_data[:, :, indices]

        elif self.strategy == 'average':
            return self._average(img_data)
            
        elif self.strategy == 'gaussian_average':
            return self._gaussian_average(img_data, c)
    
        else: 
         return img_data
       

    def _average(self, img_data):
        all_indices = np.arange(img_data.shape[2])
        splits = np.array_split(all_indices, self.target_bands)
        # Initialize reduced channel-averaged image
        reduced_img = np.zeros((img_data.shape[0], img_data.shape[1], self.target_bands), dtype=img_data.dtype)
        
        for i, split_index in enumerate(splits):
            # Mean the bands in one split 
            reduced_img[:, :, i] = np.mean(img_data[:, :, split_index], axis=2)
            
        return reduced_img


    def _gaussian_average(self, img_data, total_channels):
        """
        This captures the idea of a new camera sensor with overlapping spectral sensitivity 
        For that we defined the centers of the new sensors, fit Gaussian curves over them and then used these curves to calculate a weighted sum of the original bands
        """
        # Define Gaussian centres 
        centers = np.linspace(0, total_channels - 1, self.target_bands)
        
        # Define std of Gaussian center
        # Heuristic: Define the std s.t. both Gaussian together cover their inbetween area together roughly fully
        step = total_channels / self.target_bands
        sigma = step / 2.3 
        
        # Init weight matrix
        x = np.arange(total_channels)
        weights = np.zeros((total_channels, self.target_bands))
        
        for i, center in enumerate(centers):
            # Create a gaussian curve over all original bands for each new centre 
            w = norm.pdf(x, loc=center, scale=sigma)
            weights[:, i] = w
            
        # Normalize weights so they sum to 1
        weights /= weights.sum(axis=0, keepdims=True)
        
        # Flatten image to apply weigthed transformation
        flat = img_data.reshape(-1, total_channels)
     
        reduced_flat = np.dot(flat, weights)    
        return reduced_flat.reshape(img_data.shape[0], img_data.shape[1], self.target_bands)
 