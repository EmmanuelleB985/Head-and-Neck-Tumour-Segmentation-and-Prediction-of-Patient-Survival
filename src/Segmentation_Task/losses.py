import numpy as np
import SimpleITK as sitk
import torch
from torch.nn import functional as F
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth_term = 1

    def forward(self, pred, gt):
        axes = tuple(range(1, pred.dim()))
        inter = (pred * gt).sum(dim=axes)
        union = torch.pow(pred, 2).sum(dim=axes) + torch.pow(gt, 2).sum(dim=axes)
        loss = 1 - (2 * inter + self.smooth_term) / (union + self.smooth_term)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-2

    def forward(self, pred, gt):
        pred = pred.clamp(self.eps, 1 - self.eps)
        loss = - (gt * torch.pow((1 - pred), self.gamma) * torch.log(pred) +(1 - gt) * torch.pow(pred, self.gamma) * torch.log(1 - pred))
        return loss.mean()


class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, pred, gt):
        loss = self.dice_loss(pred, gt) + self.focal_loss(pred, gt)
        return loss

#---------Tversky Loss-----------#


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, pred, gt, smooth=1, alpha=0.3, beta=0.7, gamma=1):
              
        
        inputs = pred.view(-1)
        targets = gt.view(-1)
        
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        loss = (1 - Tversky)**gamma
                       
        return loss


#---------Tversky and Focal Loss-----------#

class Tversky_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Tversky_and_FocalLoss, self).__init__()
        self.tversky_loss = TverskyLoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, pred, gt):
        loss = self.tversky_loss(pred, gt) + self.focal_loss(pred, gt)
        return loss



#--------Combo Loss--------#

ALPHA = 0.3 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.3 #weighted contribution of modified CE loss compared to Dice loss

class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, pred, gt, smooth=1, alpha=ALPHA, eps=1e-9):
        
        #flatten label and prediction tensors
        inputs = pred.view(-1)
        targets = gt.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        inputs = torch.clamp(inputs, eps, 1.0 - eps)       
        out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
        
        return combo


#---------Binary Cross Entropy (BCE) Loss-------------#


class BCE_loss(nn.Module):
    def __init__(self):
        super(BCE_loss, self).__init__()

    def forward(self, pred, gt):

        bce_loss = nn.BCELoss(size_average=True)
        bce_out = bce_loss(pred, gt)
        return bce_out


class Log_Cosh_Dice_Loss(nn.Module):
    """
    Implemented from Jadon, Shruti. (2020). A survey of loss functions for semantic segmentation. 
    """
    def __init__(self):
        super(Log_Cosh_Dice_Loss, self).__init__()
        self.dice_loss = DiceLoss()

    def forward(self, pred, gt):
        x = self.dice_loss(pred, gt)
        loss = torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)
        return loss


#------------LogCoshDice and Focal Loss-------------------#

class Log_Cosh_Dice_FL_Loss(nn.Module):
    """
    This loss function is the one used in the presented work.
    Input:
        - pred: the output from model
                shape (N,C,D, H, W)
        - gt: ground truth map
                shape (N,D, H, W)
    Return:
        -  Averaged LogCoshDice_and_Focal loss
        
    """
    def __init__(self,gamma=2):
        super(Log_Cosh_Dice_FL_Loss, self).__init__()
        self.log_cosh_dice_loss = Log_Cosh_Dice_Loss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, pred, gt):
        loss = self.log_cosh_dice_loss(pred, gt) + self.focal_loss(pred, gt)
        return loss



