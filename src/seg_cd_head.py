# Change detection head
# adopted from https://github.com/wgcban/ddpm-cd/blob/master/model/cd_modules/cd_head_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from src.se import ChannelSpatialSELayer
import pdb




class AttentionBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)
        )

    def forward(self, x):
        return self.block(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim*len(time_steps), dim, 1)
            if len(time_steps)>1
            else nn.Identity(),
            nn.ReLU()
            if len(time_steps)>1
            else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)



class seg_cd_head(nn.Module):
    '''
    Change detection head (version 2).
    '''

    def __init__(
        self, 
        feature_dims = [32, 64, 160], # from multiscale
        scale_factor = [1,2,4], #  scale factor is times 8by 8 resoulution  
        number_class =2, 
        img_size=64,
        steps=[0, 50, 100],**kwargs):
        super(seg_cd_head, self).__init__()

        self.feat_dims = feature_dims
        self.scale_factor    = scale_factor
        self.img_size       = img_size
        self.time_steps     = steps if isinstance(steps, list) else list(range(steps))

        # Convolutional layers before parsing to difference head
        self.decoder = nn.ModuleList()
        for i in range(0, len(self.feat_dims)):
            dim = self.feat_dims[i]
            self.decoder.append(
                Block(dim=dim, dim_out=dim, time_steps=self.time_steps)
            )

            if i != len(self.feat_dims) - 1:
                dim_out = self.feat_dims[i+1]
                self.decoder.append(
                    AttentionBlock(dim=dim, dim_out=dim_out)
                )

        # Final classification head
        clfr_emb_dim = 64
        self.clfr_stg1 = nn.Conv2d(dim_out, clfr_emb_dim, kernel_size=3, padding=1)
        self.clfr_stg2 = nn.Conv2d(clfr_emb_dim, number_class , kernel_size=3, padding=1)
        self.relu = nn.ReLU()

  

    def forward(self, feats_A):
        """
        Args:
            feats_A, feats_B: Flattened lists where each element corresponds to a timestep & scale.
                Expected structure: [t0-H/8, t0-H/4, t0-H/2, t1-H/8, t1-H/4, t1-H/2, ...]
        
        Returns:
            cm: Change detection map (B, number_class, image_size, image_size)
        """
        scale_idx = 0
        x = None  # Store previous scale output for fusion
        
        for layer in self.decoder:
            if isinstance(layer, Block):


                # Collect all timesteps for current scale using direct indexing
                f_A = torch.cat([feats_A[i * len( self.feat_dims) + scale_idx] for i in range(len(self.time_steps))], dim=1)
              
                #pdb.set_trace()
                # f_A by pass blocks layer
                f_A = layer(f_A)

                # Add residual connection from previous scale
                if scale_idx != 0:
                    f_A = f_A + x

                x = f_A  # Store for next level fusion
                scale_idx += 1

            else:  # Attention Block for feature enhancement
                x = layer(x)
                
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Classifier
        cm = self.clfr_stg2(self.relu(self.clfr_stg1(x)))

        return cm



    