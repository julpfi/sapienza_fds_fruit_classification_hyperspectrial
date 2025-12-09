import torch
import torch.nn as nn
import torch.nn.functional as F



#--------------- Attention CNN Model with one more CNN layer for selecting bands ------------------
# = 3 CNN layers (one for band selection) + Transformer Encoder + classification head


class AttentionCNNModelUnselected(nn.Module):
    def __init__(self, num_bands=224, num_classes=3, token_size=64, transformer_grid_size=4):
        super().__init__()
        
        # Spectral reduction layer to learn band importance 
        self.spectral_reduce = nn.Conv2d(in_channels=num_bands, out_channels=64, kernel_size=1)
        self.bn_spectral = nn.BatchNorm2d(64)

        # CNN basis for feature extration with inductive bias 
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=token_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(token_size)
        
        
        # Layer 4 - Transformer Block wrapped sinde encoder object
        self.tokenizer = nn.AdaptiveAvgPool2d((transformer_grid_size, transformer_grid_size))

        # Adds positional embedding
        num_tokens = transformer_grid_size * transformer_grid_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, token_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_size, 
            nhead=4, 
            dim_feedforward=128, 
            batch_first=True,
            activation="gelu",
            dropout=0.2
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Classification head
        self.head_norm = nn.LayerNorm(token_size)       # Normalize before classification 
        self.dropout_head = nn.Dropout(p=0.5)           #TODO Regulirization: Test with and without 
        self.head = nn.Linear(token_size, num_classes)

    def forward(self, x):
        # shape of input is (Batch, num_bands, H, W)

        # Spectral reduction
        x = self.spectral_reduce(x)
        x = self.bn_spectral(x)
        x = F.relu(x)
    
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x) 
        
        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Layer 4 - Transformer encoding 
        x = self.tokenizer(x)                       # (Batch, token_size, 4, 4)
        
        b, c, h, w = x.shape 
        x = x.view(b, c, h * w).permute(0, 2, 1)    # (Batch, 16, token_size)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        
        # Layer 5 - Classification head

        x = x.mean(dim=1)         # Avg pooling: Tokens -> feature vector 
        x = self.head_norm(x)
        x = self.dropout_head(x)
        out = self.head(x)
    
        return out



#--------------- Attention CNN Model without extra cnn layer ------------------
# = Two CNN layers + Transformer Encoder + classification head


class AttentionCNNSelected(nn.Module):
    def __init__(self, num_bands=30, num_classes=3, cnn_channels=64, transformer_grid_size=4):
        super().__init__()
        
        # CNN feature extractor 
        # Layer 1
        self.conv1 = nn.Conv2d(num_bands, cnn_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(cnn_channels)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2
        self.conv2 = nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(cnn_channels)

        # Layer 3
        # Tokenizer - reduces spatial dimension into few patches/tokens 
        self.tokenizer = nn.AdaptiveAvgPool2d((transformer_grid_size, transformer_grid_size))

        # Adds positional embedding
        num_tokens = transformer_grid_size * transformer_grid_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, cnn_channels))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_channels, 
            nhead=4, 
            dim_feedforward=128, 
            batch_first=True,
            activation="gelu",
            dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Classification head
        self.head_norm = nn.LayerNorm(cnn_channels)         # normalize before clasification 
        self.head_drop = nn.Dropout(0.5)                    # dropout for regularization   
        self.head_fc = nn.Linear(cnn_channels, num_classes) 

    def forward(self, x):
        # Input: (Batch, num_bands, H, W)
        b, c, h, w = x.shape
        
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Layer 3 - Transformer 
        
        x = self.tokenizer(x)               #(Batch, 64, 4, 4)
        x = x.flatten(2).permute(0, 2, 1)   #flatten:   (Batch, 64, 16) -> (Batch, 16, 64)
        x = x + self.pos_embedding
        x = self.transformer(x)
        
        # Layer 4 - Classification head
        x = x.mean(dim=1)      # Avg pooling: Tokens -> feature vector
        x = self.head_norm(x)
        x = self.head_drop(x)
        out = self.head_fc(x)
        
        return out