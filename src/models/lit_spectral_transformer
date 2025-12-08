import torch
import torch.nn as nn

class LitSpectralTransformer(nn.Module):
    def __init__(self, bands, num_classes=3, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        
        # 1. Feature Extractor (1D CNN)
        # Processes the spectral bands sequence as a time series
        # Input shape: (Batch, 1, Num_Bands)
        self.embedding = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Dynamic Dimension Calculation
        # Dynamically compute the size of the flattened features after the CNN layers
        # based on the specific number of input bands provided by the dataset
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, bands) 
            dummy_out = self.embedding(dummy_input)
            self.flatten_dim = dummy_out.shape[1] * dummy_out.shape[2]

        # 4. Classification Head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input x: (Batch, Bands, Height, Width)
        
        # Step 1: Spatial Reduction (Global Average Pooling)
        # Collapse spatial dimensions (H, W) to focus on spectral signature
        # Output: (Batch, Bands)
        x = x.mean(dim=(2, 3)) 
        
        # Step 2: Channel Dimension
        # Add channel dim for Conv1d: (Batch, 1, Bands)
        x = x.unsqueeze(1)
        
        # Step 3: Feature Extraction
        x = self.embedding(x)
        
        # Step 4: Sequence Preparation
        # Permute for Transformer: (Batch, Seq_Len, Features)
        x = x.permute(0, 2, 1)
        
        # Step 5: Transformer Encoding & Classification
        x = self.transformer(x)
        x = self.fc(x)
        
        return x