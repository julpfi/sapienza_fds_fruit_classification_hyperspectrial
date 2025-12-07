import torch
import numpy as np
from typing import Optional

class BandSelector:
    def __init__(self, wavelengths: list[float], selection_bounds: Optional[tuple[float, float]] = None):
        """
        Args:
            wavelengths (list[float]): List of wavelengths available in the dataset
            selection_bounds (optional: tuple[float, float]): min and max wavelength used for selection 
        """
        self.wavelengths = np.array(wavelengths)
        self.indices = None
        self.active_range = "All Bands"

        if selection_bounds is not None:
            min_nm, max_nm = selection_bounds
            # Creates mask for wave lengths within the bounds 
            selection_mask = np.where((self.wavelengths >= min_nm) & (self.wavelengths <= max_nm))[0]
            
            if len(selection_mask) == 0:
                raise ValueError(f"BandSelector found no bands in the range from {min_nm} to {max_nm}nm.")
                
            self.indices = torch.from_numpy(selection_mask).long()
            self.active_range = f"{min_nm}-{max_nm}nm"
            print(f"BandSelector Initialized: Keeping {len(self.indices)} bands ({self.active_range}).")
        else:
            print("BandSelector Initialized: No bounds specified. Keeping all bands.")


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """ Applies band selection to the input tensor """
        # If no indices were selected, return original tensor
        if self.indices is None:
            return x

        # Slice the channel dimension of tensor (dim 0)
        return x[self.indices, ...]