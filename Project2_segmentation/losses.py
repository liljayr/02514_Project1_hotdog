import torch

def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))

def dice_loss(y_real, y_pred):
    y_pred = torch.sigmoid(y_pred)
    return 1 - (torch.mean(2*y_pred*y_real + 1) / (torch.mean(y_pred + y_real) + 1))

def focal_loss(y_real, y_pred):
    y_pred_s = torch.sigmoid(y_pred)
    return -torch.sum((((1-y_pred_s)**2) * y_real * torch.log(y_pred_s)) + ((1 - y_real) * torch.log(1-y_pred_s)))

def bce_total_variation(y_real, y_pred):
    regularization = torch.sum(torch.sigmoid(torch.roll(y_pred, 1, 2)) - torch.sigmoid(y_pred)) + torch.sum(torch.sigmoid(torch.roll(y_pred, 1, 3)) - torch.sigmoid(y_pred))
    return bce_loss(y_real, y_pred) + 0.1 * regularization