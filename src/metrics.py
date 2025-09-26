import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)

def batch_pix_accuracy(predict, target):

    correct = (predict == target).sum().item()  # Correct predictions
    total = target.numel()  # Total labeled pixels

    return correct,total

def batch_intersection_union(predict, target, num_class):

    intersection = predict[predict == target]

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class-1, min=0)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class-1, min=0)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class-1, min=0)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()

def eval_metrics(output, target, num_class):

    _, predict = torch.max(output.data, 1)
    
    #predict = predict 
    #target = target 

    labeled = (target >= 0) & (target < num_class)

    masked_predict = predict[labeled]
    masked_target = target[labeled]
    #labeled = (target > 0) * (target <= num_class)
    correct, num_labeled = batch_pix_accuracy(masked_predict, masked_target)
    inter, union = batch_intersection_union(masked_predict, masked_target, num_class)
    """
    
    correct, num_labeled = batch_pix_accuracy(predict, target, target)
    inter, union = batch_intersection_union(predict, target, num_class, target)
    """
    return [np.round(correct, 5), np.round(num_labeled, 5), np.round(inter, 5), np.round(union, 5)]
