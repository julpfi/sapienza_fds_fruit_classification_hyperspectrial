import torch
import torch.nn as nn

class LitSpectralTransformer(nn.Module):
    def __init__(self, bands, num_classes=3, d_model=64, nhead=4, num_layers=2, is_complex_dft_reduce=False):
        """
        Args:
            bands: Total number of input channels (e.g., 60 if using 30 Real + 30 Imag)
            is_complex_dft_reduce: If True, reshapes input to (Batch, 2, bands//2)
                             If False, reshapes input to (Batch, 1, bands)
        """
        super().__init__()
        self.is_complex_dft_reduce = is_complex_dft_reduce
        in_channels = 2 if self.is_complex_dft_reduce else 1
        
        # Feature extractor 1d CNN
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # Transformer encoder 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Determine cnn pooling dims dynamically
        with torch.no_grad():
            # Create dummy input based on mode
            if self.is_complex_dft_reduce:
                # Shape: (1, 2, Bands/2)
                dummy_seq_len = bands // 2
                dummy_input = torch.zeros(1, 2, dummy_seq_len)
            else:
                # Shape: (1, 1, Bands)
                dummy_input = torch.zeros(1, 1, bands)
                
            dummy_out = self.embedding(dummy_input)
            # Flatten dim = Sequence_Length * Features
            self.flatten_dim = dummy_out.shape[1] * dummy_out.shape[2]

        # Classification head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input x: (Batch, Total_Bands, Height, Width)
        
        # patial reduction (Global Average Pooling)
        x = x.mean(dim=(2, 3)) # -> (Batch, Total_Bands)
    
        if self.is_complex_dft_reduce:
            # Input (Batch, 60) -> Output (Batch, 2, 30)
            batch_size, total_channels = x.shape
            seq_len = total_channels // 2
            x = x.view(batch_size, 2, seq_len)
        else:
            # Input (Batch, 30) -> Output (Batch, 1, 30)
            x = x.unsqueeze(1)
        
        # CNN feature extraction 
        x = self.embedding(x)
        
        # Transformer + permutation needed 
        x = x.permute(0, 2, 1) # (Batch, Seq_Len, Features)
        x = self.transformer(x)
        # Head
        x = self.fc(x)
        
        return x