import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridModel(nn.Module):
    def __init__(self, in_channels=30, num_classes=3, reduce_bands=False, cnn_channels=32, transformer_grid_size=4):
        
        super().__init__()
        self.reduce_bands = reduce_bands

        # If we use non-reduced images we first learn how to reduce it to 32
        # If we use pre-reduced data we skip this very first layer
        
        if self.reduce_bands:
            # Learnable reduction: 224 -> 32
            self.spectral_reduce = nn.Conv2d(in_channels=in_channels, out_channels=cnn_channels, kernel_size=1)
            self.bn_spectral = nn.BatchNorm2d(cnn_channels)
            
            # Adapt next layer input channels
            current_in_channels = cnn_channels
        else:
            # No reduction layer. The next layer receives the input directly.
            self.spectral_reduce = nn.Identity()
            self.bn_spectral = nn.Identity()
            current_in_channels = in_channels

        #  Spatial CNN layers 
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=current_in_channels, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(cnn_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(cnn_channels*2)
        # Optional: Add pool2 if image is large, or skip to keep tokens
        # self.pool2 = nn.MaxPool2d(2, 2) 


        # Transformer encoder 
        transformer_dim = cnn_channels * 2
        # Tokenizer - creates patches for transformer 
        self.tokenizer = nn.AdaptiveAvgPool2d((transformer_grid_size, transformer_grid_size))

        # Positional Embedding
        num_tokens = transformer_grid_size * transformer_grid_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, transformer_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=4, 
            dim_feedforward=256, 
            batch_first=True,
            activation="gelu",
            dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Classiciation head
        self.head_norm = nn.LayerNorm(transformer_dim)
        self.head_drop = nn.Dropout(0.5)
        self.head_fc = nn.Linear(transformer_dim, num_classes)

    def forward(self, x):

        if self.reduce_bands:
            x = self.spectral_reduce(x)
            x = self.bn_spectral(x)
            x = F.relu(x)
        
        # First CNN layer 
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second CNN layer 
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        
        # Transformer encoder 
        x = self.tokenizer(x)                # (Batch, 128, 4, 4)
        x = x.flatten(2).permute(0, 2, 1)    # (Batch, 16, 128)
        
        x = x + self.pos_embedding
        x = self.transformer(x)
        
        # Classification head
        x = x.mean(dim=1)
        x = self.head_norm(x)
        x = self.head_drop(x)
        out = self.head_fc(x)
        
        return out