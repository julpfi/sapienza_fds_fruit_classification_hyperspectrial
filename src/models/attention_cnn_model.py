import torch
import torch.nn as nn
import torch.nn.functional as F


#--------------- Attention CNN Model with one more CNN layer for selecting bands ------------------
# = 3 CNN layers (one for band selection) + Transformer Encoder + classification head

class AttentionCNNModelUnselected(nn.Module):
    def __init__(self, num_bands=224, num_classes=3, token_size=64):
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
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 4 - Transformer Block wrapped sinde encoder object 
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
        self.dropout_head = nn.Dropout(p=0.5) #TODO Test with and without 
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
        x = self.pool3(x)
        # Shape after three cnn layers (Batch, token_size, H/8, W/8)
        
        # Layer 4 - Transformer encoding 
        b, c, h, w = x.shape 
        # Flatten spatial dimensions into 1d with bands at the end
        x = x.view(b, c, h * w).permute(0, 2, 1) 

        x = self.transformer_encoder(x)
        
        # Global average pooling over tokens 
        x = x.mean(dim=1) 
        
        x = self.dropout_head(x)
        out = self.head(x)
    
        return out



#--------------- Attention CNN Model without extra cnn layer ------------------
# = Two CNN layers + Transformer Encoder + classification head


class AttentionCNNSelection(nn.Module):
    def __init__(self, num_bands=30, num_classes=3, token_size=64):
        super().__init__()

        # CNN basis for feature extraction
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=num_bands, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 224 -> 112

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=token_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(token_size)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 112 -> 56
        
        # Layer 3 - Transformer Block wrapped inside encoder object 
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
        self.dropout_head = nn.Dropout(p=0.5) 
        self.head = nn.Linear(token_size, num_classes)

    def forward(self, x):
        # shape of input is (Batch, 30, H, W)

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
        # Shape after two cnn layers (Batch, token_size, H/4, W/4)
        # Spatial Size: 56x56
        
        # Layer 3 - Transformer encoding 
        b, c, h, w = x.shape 
        # Flatten spatial dimensions into 1d with bands at the end
        x = x.view(b, c, h * w).permute(0, 2, 1) 

        x = self.transformer_encoder(x)
        
        # Global average pooling over tokens 
        x = x.mean(dim=1) 
        
        x = self.dropout_head(x)
        out = self.head(x)
    
        return out