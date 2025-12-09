import torch
import torch.nn as nn
import timm


class DeiTModel(nn.Module):
    def __init__(self, num_classes=3, input_bands=30, use_distillation=True, pretrained=True):
        super().__init__()
        self.use_distillation = use_distillation

        # Adapter to convert input_bands into 3 channels for DeiT
        self.adapter_conv = nn.Conv2d(in_channels=input_bands, out_channels=3, kernel_size=1, bias=False)
        self.adapter_bn = nn.BatchNorm2d(3)
        self.adapter_act = nn.GELU() 

        # Pretrained DeiT Model
        self.base_model = timm.create_model(
            'deit_tiny_distilled_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Remove the classification head from the base model structure 
        self.base_model.head = nn.Identity()
        self.base_model.head_dist = nn.Identity()

        # Add new head
        self.head = nn.Linear(self.base_model.embed_dim, num_classes)
        
        # Initialize the weights and freeze
        self._init_adapter_weights()
        self._freeze_base_model()

    def _init_adapter_weights(self):
        # Initialize the adapter to be roughly a "mean" operation at start
        # This prevents the model from starting with total garbage inputs
        nn.init.constant_(self.adapter_conv.weight, 1.0 / self.adapter_conv.in_channels)

    def _freeze_base_model(self):
        # Freeze -> Do not train base DeiT model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze -> train adapter
        for param in self.adapter_conv.parameters():
            param.requires_grad = True
        for param in self.adapter_bn.parameters():
            param.requires_grad = True
            
        # Unfreeze -> head 
        for param in self.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.adapter_conv(x)
        x = self.adapter_bn(x)
        x = self.adapter_act(x)


        features = self.base_model.forward_features(x)
        
       
        if self.use_distillation:
            # Distillation in DeiT: Average the CLS token (ground truth expert) and DIST token (teacher expert)
            # Togther they give the transformer the global context with the CNN's local awareness (from teacher)
            x_cls = (features[:, 0] + features[:, 1]) / 2
        else:
            x_cls = features[:, 0]
            
        return self.head(x_cls)
    

# Old version with direct modification of first layer of DeiT base model
"""
class DeiTModel(nn.Module):
    def __init__(self, num_classes=3, input_bands=10, use_distillation=True, pretrained=True):
        super().__init__()
        self.use_distillation = use_distillation

        # Load pretrained base_model = DeiT-Tiny Distilled  
        # Input: Standard RGB [Batch, 3, 224, 224] => images are 224x224
        # Output: Feature Map [Batch, 198, 192]
        self.base_model = timm.create_model('deit_tiny_distilled_patch16_224'
                                            ,pretrained=pretrained
                                            #,img_size=64 #WOuld not recude param as attention weights are resued anyway across all patches (same for 224 vs 64)
                                            )
        
        # Adapt the first projection layer to accept 30 channels instead of 3
        original_first_layer = self.base_model.patch_embed.proj
        new_first_layer = nn.Conv2d(
            in_channels=input_bands,
            # settings as original one: 
            out_channels=original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding
        )
        
        # Weight intialization 
        with torch.no_grad():
            new_first_layer.weight[:, :3] = original_first_layer.weight   # copyied bands for first three channels 
            nn.init.kaiming_normal_(new_first_layer.weight[:, 3:]) # random init for added channels
            
        self.base_model.patch_embed.proj = new_first_layer

        # Replacement of head - from original 192 features mapping to 3 (classes)
        self.head = nn.Linear(self.base_model.embed_dim, num_classes)
        
        # Cleanup - remove original heads 
        self.base_model.head = nn.Identity()
        self.base_model.head_dist = nn.Identity()

        self._freeze_base_model()


    def _freeze_base_model(self):
        # Freeze base deit model 
        for param in self.base_model.parameters():
            param.requires_grad = False
        # Unfreeze -> train first layer   
        for param in self.base_model.patch_embed.proj.parameters():
            param.requires_grad = True
        # Unfreeze -> train head
        for param in self.head.parameters():
            param.requires_grad = True


    def forward(self, x):   
        # Returns 198 features = 196 patches + 1 CLS token + 1 DIST token
        features = self.base_model.forward_features(x)
        
        if self.use_distillation:
            # Distillation in DeiT: Average the CLS token (ground truth expert) and DIST token (teacher expert)
            # Togther they give the transformer the global context with the CNN's local awareness (from teacher)
            x = (features[:, 0] + features[:, 1]) / 2
        else:
            #
            x = features[:, 0]
            
        return self.head(x)

"""