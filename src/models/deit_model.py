import torch
import torch.nn as nn
import timm

class DeiTModel(nn.Module):
    def __init__(self, num_classes=3, in_channels=30, pretrained=True):
        super().__init__()

        # Bade model 
        self.base_model = timm.create_model(
            'deit_tiny_distilled_patch16_224',
            pretrained=pretrained,
            num_classes=0 
        )
        
        # We replace old projection with new 
        old_proj = self.base_model.patch_embed.proj
        
        new_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding
        )
        
        # Initilization of additional channels of new projection adapter 
        with torch.no_grad():
            # We keep teh original rgb-generated weights
            new_proj.weight[:, :3] = old_proj.weight
            
            # We init. the other 27 weights based on random selected rgb weight plus random noise 
            rgb_weights = old_proj.weight

            random_indices = torch.randint(0, 3, in_channels - 3)
            selected_weights = rgb_weights[:, random_indices, :, :]
  
            noise = torch.randn_like(selected_weights) * (0.1 * selected_weights.std())
            new_proj.weight[:, 3:] = selected_weights + noise
            
            # Copy bias if it exists
            if old_proj.bias is not None:
                new_proj.bias = old_proj.bias

        self.base_model.patch_embed.proj = new_proj

        # Classification head
        self.head = nn.Linear(self.base_model.embed_dim, num_classes)
        
        # Freezing base partly 
        self._freeze_base_model()


    def _freeze_base_model(self):
        # Freeze all at first
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze patch embedding
        for param in self.base_model.patch_embed.proj.parameters():
            param.requires_grad = True
            
        # Unfreeze norm layers
        for name, param in self.base_model.named_parameters():
            if 'norm' in name:
                param.requires_grad = True
                
        # Unfreeze head
        for param in self.head.parameters():
            param.requires_grad = True


    def forward(self, x):
        # x shape: (Batch, 30, 224, 224)
        features = self.base_model.forward_features(x)
        
        # features shape: (Batch, 198, 192) -> [CLS, DIST, Patch_0, ... Patch_195]
        # Robust inference: average the two expert tokens
        x_cls = (features[:, 0] + features[:, 1]) / 2
     
        return self.head(x_cls)