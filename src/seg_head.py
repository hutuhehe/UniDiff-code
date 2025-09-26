import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb

class Segformer_head(nn.Module):
    def __init__(
        self,
        feature_dims = [32, 64, 160, 256],
        scale_factor = [1,2,4,4],
        proj_dim = 256,
        number_class = 13,
        image_size= 128,
        **kwargs
        
    ):
        super().__init__()    
        self.img_size = image_size

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, proj_dim, 1),
            nn.Upsample(scale_factor = scale )
        ) for scale, dim in zip(scale_factor,feature_dims)])


        """
        self.to_segmentation = nn.Sequential(
            nn.Conv2d(len(feature_dims)* proj_dim, proj_dim, 1),
            nn.Gelu()
            nn.Conv2d(proj_dim, number_class, 1),
        )

        """

        self.to_segmentation = nn.Sequential(
            #nn.Conv2d(len(feature_dims) * proj_dim, proj_dim, kernel_size=3, padding=1, bias=False),

            nn.Conv2d(len(feature_dims) * proj_dim, proj_dim, kernel_size=1),
           #nn.BatchNorm2d(proj_dim),  # Optional normalization
            nn.ReLU(inplace=True),           
            nn.Conv2d(proj_dim ,number_class, kernel_size=1)
        )


    def forward(self, layer_outputs):
        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
        fused = self.to_segmentation(fused)
        fused = F.interpolate(fused, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        return fused