import torch
import torch.nn as nn
import torch.fft

class SpecSpatFTModel(nn.Module):
    def __init__(self, in_channels=30, num_classes=3, crop_size=16):
        super().__init__()
        self.crop = crop_size

        # Layer acting as a trainable filter that learns which frequencies are important (given the spectrum and spatial dft low pass filter)
        # Init with 1 instead of random to not have a random initla bias for some frequencies; Tryout
        self.freq_weight = nn.Parameter(torch.ones(1, in_channels, crop_size, crop_size))
        
        flat_dim = in_channels * crop_size * crop_size
      
        # Classification head    
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.LayerNorm(256),  # Normalize magnitude variance
            nn.ReLU(),
            nn.Dropout(0.6),    # Heavy dropout for small data (Increased to 0.6)
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x = (Batch, Channels, Height, Width) 
        
        # Spatial (2d) discrete fourier transformation 
        fft_x = torch.fft.fft2(x)          # by default on last positions of tensor 
        
        # We shift the low frequencies to the centre of tensor 
        # Prepares the crop of the center frequencies = low frequencies  
        fft_x = torch.fft.fftshift(fft_x, dim=(-2, -1))
        
        # Low pass filter as a dynamic crop of the centre (high frequencies where shifted to the edges)
        B, C, H, W = fft_x.shape
        start_h = H // 2 - self.crop // 2
        start_w = W // 2 - self.crop // 2
        
        # Safety check for very small inputs
        if start_h < 0 or start_w < 0:
            raise ValueError(f"Input image size ({H}x{W}) is smaller than crop_size ({self.crop})")

        # Crop the center frequencies
        x_cropped = fft_x[:, :, start_h : start_h+self.crop, start_w : start_w+self.crop]
        
        # Calculate log magnitute of low-filtered waves
        # Abs. removes phase (=position)
        x_mag = torch.log(torch.abs(x_cropped) + 1e-6)
        
        # Weighting the frequencies before classification
        x_weighted = x_mag * self.freq_weight
        
        return self.classifier(x_weighted)