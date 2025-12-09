import torch
import torch.nn as nn
import torch.fft

class FruitHSNet(nn.Module):
    def __init__(self, num_bands=30, num_classes=3, use_3x3_center=True):
        """
        Fruit-HSNet implementation based on the paper.
        Args:
            num_bands (int): Number of spectral bands (channels).
            num_classes (int): Number of target classes (e.g., 3 for unripe, ripe, overripe).
            use_3x3_center (bool): If True, averages a 3x3 central area instead of a single pixel
                                   to increase robustness against noise/augmentation.
        """
        super().__init__()
        self.use_3x3_center = use_3x3_center
        
        # --- Learnable Feature Fusion Parameters ---
        # Learnable weights w1 and w2.
        # Initialized with a normal distribution.
        self.w1 = nn.Parameter(torch.randn(num_bands))
        self.w2 = nn.Parameter(torch.randn(num_bands))
        
        # --- Classification Network (MLP) ---
        # Structure: [2*Bands -> 512 -> 256 -> Classes]
        # Input dimension is 2 * num_bands because spectral and spatial features are concatenated.
        input_dim = 2 * num_bands
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4), 
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Bands, Height, Width)
        """
        batch_size, bands, height, width = x.shape
        
        # --- Branch 1: Spectral Feature Extraction (Fourier Transform) ---
        # Apply FFT on spatial dimensions.
        fft_x = torch.fft.fft2(x, dim=(-2, -1))
        
        # Compute Magnitude and Spatial Average
        fft_mag = torch.abs(fft_x)
        spectral_features = fft_mag.mean(dim=(2, 3)) # Shape: (Batch, Bands)
        
        # --- Branch 2: Spatial Feature Extraction (Central Pixel) ---
        # Extracts the spectral signature of the central region.
        center_h, center_w = height // 2, width // 2
        
        if self.use_3x3_center:
            # Robustness: Average of the 3x3 central area
            # This helps mitigate small shifts caused by RandomResizedCrop
            spatial_features = x[:, :, center_h-1:center_h+2, center_w-1:center_w+2].mean(dim=(2,3))
        else:
            # Strict implementation: Single central pixel
            spatial_features = x[:, :, center_h, center_w] # Shape: (Batch, Bands)

        # --- Feature Fusion ---
        # Apply learnable weights.
        weighted_spectral = spectral_features * self.w1
        weighted_spatial = spatial_features * self.w2
        
        # Concatenate weighted features
        fused_features = torch.cat([weighted_spectral, weighted_spatial], dim=1) # Shape: (Batch, 2*Bands)
        
        # --- Classification ---
        out = self.classifier(fused_features)
        
        return out