import torch
import torch.nn as nn
import timm


class SwinModel(nn.Module):
    def __init__(self, num_classes=3, in_channels=30, pretrained=True, 
                 drop_rate=0.0, drop_path_rate=0.1):
        super().__init__()
        
        # load pretrained swin transformer model 
        # drop_rate = dropout before class. head
        # drop_path_rate = dropout of transformer layers  

        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=pretrained, 
            num_classes=0,            # Remove head
            drop_rate=drop_rate,      # Head dropout
            drop_path_rate=drop_path_rate # Transformer layer dropout
        )
        
        # Adapter from in_channels to swin embedding dimension
        embed_dim = self.swin.embed_dim
        self.spectral_proj = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=embed_dim, 
            kernel_size=4, 
            stride=4
        )
        
        # Initialize the new projection layer
        nn.init.kaiming_normal_(self.spectral_proj.weight, mode='fan_out', nonlinearity='relu')
        if self.spectral_proj.bias is not None:
            nn.init.constant_(self.spectral_proj.bias, 0)
            
        # Swap the projection layer
        self.swin.patch_embed.proj = self.spectral_proj
        self.swin.patch_embed.img_size = (224, 224)
        self.swin.patch_embed.patch_size = (4, 4)
        
        # Classification head 
        self.head_drop = nn.Dropout(p=0.3)      # Dropout before classification
        self.head = nn.Linear(self.swin.num_features, num_classes)

        self._freeze_layers()

    def _freeze_layers(self):
        # Freeze all
        for param in self.swin.parameters():
            param.requires_grad = False
            
        # Unfreeze embedding prohection 
        for param in self.swin.patch_embed.proj.parameters():
            param.requires_grad = True
            
        # Unfreeze normalization layers
        for name, param in self.swin.named_parameters():
            if "norm" in name:
                param.requires_grad = True
                
        # Unfreeze classification head
        for param in self.head.parameters():
            param.requires_grad = True


    def forward(self, x):

        features = self.swin.forward_features(x)
    
        if features.ndim == 3:
            features = features.mean(dim=1)
        # Fix? 
        elif features.ndim == 4:
            features = features.mean(dim=(2, 3))
            
        x = self.head_drop(features)
        return self.head(x)