import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionCNNModel(nn.Module):
    def __init__(self, num_bands=30, num_classes=3, token_size=64):
        super().__init__()
        
        # CNN basis for feature extration with inductive bias 
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=num_bands, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=token_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(token_size)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 3 - Transformer Block wrapped sinde encoder object 
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