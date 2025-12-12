import os
import numpy as np
import spectral.io.envi as envi
import logging

logger = logging.getLogger(__name__)


def load_hsi_pair(hdr_path, bin_path):
    """
    Loads .hdr and .bin files using the spectral library
    Returns the spectral.SpyFile object (None if unable to load)
    """
    if not os.path.exists(hdr_path) or not os.path.exists(bin_path):
        return None
    try:
        # We use our binary path explicitly (instead of relying on hdr internal path) 
        return envi.open(hdr_path, bin_path)
    except Exception as e:
        logger.error(f"Error opening ENVI pair {hdr_path}: {e}")
        return None


def mask_background(img_data, threshold=0.1):
    """
    Sets background pixels to 0.
    Assumes img_data is already normalized to [0, 1].
    """
    # Calculate mean intensity across spectral bands (axis 2)
    mean_intensity = np.mean(img_data, axis=2)
     
    # Create Mask (True = Background)
    mask = mean_intensity < threshold
    
    # Apply Mask
    img_data[mask] = 0.0
    return img_data


def crop_border_pixels(img_data, border_x=10, border_y=10):
    """
    Removes the borders pixels of height border_y and width border_x of spatioal dimension if images 
    This was done in the original paper to remove the conveyor belt remains 
    """
    h, w, c = img_data.shape
    
    # Safety check: Don't crop if image is too small
    start_x = min(border_x, w // 4)
    end_x = max(w - border_x, w * 3 // 4)
    
    return img_data[:, start_x:end_x, :]


def load_and_preprocess(header_path, bin_path):
    """
    Runs preprocessing pipeline: Load -> mask background -> crop borders
    """
    # Load Raw Data
    raw_obj = load_hsi_pair(header_path, bin_path)
    if raw_obj is None:
        return None
    
    # Load fully into memory 
    img_data = raw_obj.load().copy()

    # Clean
    img_data = mask_background(img_data)
    # img_data = crop_border_pixels(img_data)

    return img_data

