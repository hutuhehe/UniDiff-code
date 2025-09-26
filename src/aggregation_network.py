import numpy as np
import torch
from torch import nn
from src.resnet import ResNet, BottleneckBlock
from guided_diffusion.guided_diffusion.nn import (
    zero_module,
    conv_nd,
    normalization,
)

import ipdb

class AggregationNetwork(nn.Module):
    """
    Module for aggregating feature maps across time and space.
    Design inspired by the Feature Extractor from ODISE (Xu et. al., CVPR 2023).
    https://github.com/NVlabs/ODISE/blob/5836c0adfcd8d7fd1f8016ff5604d4a31dd3b145/odise/modeling/backbone/feature_extractor.py
    """
    def __init__(
            self, 
            number_class,
            feature_dims, 
            steps,
            blocks,
            num_norm_groups=32,
            num_res_blocks_bottleneck=1, 
            num_timesteps=None,
            timestep_weight_sharing=False,
            projection_dim= 384, 
            **kwargs
            
        ):
     
        super().__init__()
        self.bottleneck_layers = nn.ModuleList()
        self.feature_dims = feature_dims    
        # For CLIP symmetric cross entropy loss during training
        #self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
        self.save_timestep = steps

        self.mixing_weights_names = []
        for t in steps:
            for l, feature_dim in zip(blocks,self.feature_dims):
                if t == steps[0]:  
                    bottleneck_layer = nn.Sequential(
                    *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=num_res_blocks_bottleneck,
                    in_channels=feature_dim,
                    bottleneck_channels=projection_dim // 4,
                    out_channels=projection_dim,
                    norm="GN",
                    num_norm_groups=num_norm_groups
                    ) )
                    # Append the created layer to bottleneck_layers only once per feature_dim
                    self.bottleneck_layers.append(bottleneck_layer)
        
                # Append mixing_weights_names for each combination of timestep `t` and layer `l`
                self.mixing_weights_names.append(f"timestep-{t}_layer-{l}")

        mixing_weights = torch.ones(len(self.bottleneck_layers) * len(steps))
        self.mixing_weights = nn.Parameter(mixing_weights)


        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_channels=projection_dim, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=number_class, kernel_size=1),
            nn.BatchNorm2d(number_class)  
                )
        """
        if number_class > 15:
            # 3-layer classification head
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                
                nn.Linear(feature_dims[0], 256),  # First fully connected layer
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                
                nn.Linear(256, 128),              # Second fully connected layer
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                
                nn.Linear(128, 64),               # Additional third fully connected layer
                nn.ReLU(),
                nn.BatchNorm1d(num_features=64),
                
                nn.Linear(64, number_class)       # Output layer
            )
        else:
            # 2-layer classification head
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                
                nn.Linear(feature_dims[0], 256),  # First fully connected layer
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                
                nn.Linear(256, number_class)      # Output layer
             )
 
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
                
            nn.Linear(feature_dims[0], number_class) # First fully connected layer
            )
            """
        # if pool = adaptive 
        self.out = nn.Sequential(
                normalization(self.feature_dims),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(2, self.feature_dims,number_class, 1)),
                nn.Flatten()
            )
    
    def forward(self, batch):
        """
        Assumes batch is shape (B, C, H, W) where C is the concatentation of all layer features.
        """
        """
        output_feature = None
        start = 0
        mixing_weights = torch.nn.functional.softmax(self.mixing_weights,dim = 0)
        for i in range(len(mixing_weights)):
            
            # Share bottleneck layers across timesteps22
            bottleneck_layer = self.bottleneck_layers[i % len(self.feature_dims)]
            # Chunk the batch according the layer
            # Account for looping if there are multiple timesteps
            end = start + self.feature_dims[i % len(self.feature_dims)]
   
            feats = batch[:, start:end, :, :]
            start = end
            # Downsample the number of channels and weight the layer
            bottlenecked_feature = bottleneck_layer(feats)
            bottlenecked_feature = mixing_weights[i] * bottlenecked_feature
            if output_feature is None:
                output_feature = bottlenecked_feature
            else:
                output_feature += bottlenecked_feature
        #ipdb.set_trace()
        """
        #out = self.segmentation_head(output_feature) 
        #out = self.segmentation_head(batch) 
        # ipdb.set_trace()
        # out = self.classification_head(batch)
        pdb.set_trace()
        out = self.out(batch)
       
        return out