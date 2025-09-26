import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
#from sklearn.utils import class_weight 
#from utils.lovasz_losses import lovasz_softmax
import pdb

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def get_weights(target):

    # [0.0334, 0.5435, 0.0715, 0.0289, 0.4525, 0.3135, 1.0204, 0.8130, 1.6393, 0.4292, 0.1314, 0.6803]

    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    #cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(12)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()
    



class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss
"""
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()
"""



class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

        
        # Use reduction='none' instead of deprecated reduce=False
        self.CE_loss = nn.CrossEntropyLoss(weight=alpha, ignore_index=ignore_index, reduction='none')

    def forward(self, output, target):

        # Compute the standard cross-entropy loss
        ce_loss = self.CE_loss(output, target)
        
        # Convert CE loss to probabilities
        pt = torch.exp(-ce_loss)
        
        # Apply the focal loss formula
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Return mean or sum based on size_average flag
        return focal_loss.mean() if self.size_average else focal_loss.sum()

class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss
"""
class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
    
    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss
"""

def entropy_loss(predictions):
    # Normalize logits to get the probability distribution (using softmax)
    prob = F.softmax(predictions, dim=1)
    
    # Compute entropy loss
    entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)  # small epsilon to avoid log(0)
    return entropy.mean()  # average across the batch


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, ignore_index=255, size_average= True, scale_factor = 20):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Weighting for class imbalance
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # No reduction, so we can modify it
        self.scale_factor = scale_factor
   

    def forward(self, output, target):
        """
        output: logits (before sigmoid) with shape [batch, H, W]
        target: binary labels {0, 1} with shape [batch, H, W]
        """
        # Ignore padding pixels
        mask = (target != self.ignore_index)
        target = target.float()  # Ensure float type for BCE loss
        
        # Compute standard BCE loss
        bce_loss = self.bce_loss(output, target)

        # Compute probability p = sigmoid(output)
        pt = torch.sigmoid(output)
        pt = pt * target + (1 - pt) * (1 - target)  # pt is p_t in focal loss formula

        # Apply focal loss scaling factor (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        # Apply alpha weighting (if provided)
        if self.alpha is not None:
            alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss *= alpha_weight

        # Apply mask to ignore padding pixels
        loss =  loss * mask *self.scale_factor

        # Reduce loss
        return loss.mean() if self.size_average else loss.sum()
