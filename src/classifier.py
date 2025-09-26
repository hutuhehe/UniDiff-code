import os
import numpy as np


from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from aggregation_network import AggregationNetwork
import pdb





# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L68
# pixel_classfier is f_network for spatial feature classfication
class aggregation_classifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super(pixel_classifier, self).__init__()
        self.aggregation_network = AggregationNetwork(
            projection_dim=config["projection_dim"],
            feature_dims=dims,
            device=device,
            save_timestep=config["save_timestep"],
            num_timesteps=config["num_timesteps"]
    )

        if numpy_class < 30:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class)
            )

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        x = self.aggregation_network(x)
        return self.layers(x)














def load_ensemble(args, device='cpu'):
    #pdb.set_trace()
    models = []
    for i in range(args['model_num']):
        model_path = os.path.join(args['exp_dir'], f'model_{i}.pth')
        state_dict = torch.load(model_path)['model_state_dict']

        #model = DualPathNetwork(spatial_dim = args['dim'][-1],spectral_dim = args['bands_num'], num_class = args['number_class'])  # using naive feature conca
        #model = DualPathNetwork(num_class =args['number_class']) # using cross attention 
        #model = DualPathNetwork_cross(num_class =args['number_class']) # using cross attention ,for berlin dataset
        model = pixel_classifier(numpy_class= args['number_class'], dim= args['dim'][-1]) # if using pixel classfier
        #model = Conv1D_Classfier(num_classes= args['number_class'],bands_num = args['bands_num']) # if using Conv1d classfier
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        models.append(model.eval())
    return models
