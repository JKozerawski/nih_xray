import torch
import torch.nn as nn
from torchvision import models

class XrayNet(nn.Module):

  def __init__ (self, pretrained=False):
    super(XrayNet, self).__init__()
    self.out_dim = 14
    self.densenet121 = models.densenet121(pretrained=pretrained)
    num_ftrs = self.densenet121.classifier.in_features
    self.densenet121.classifier = nn.Sequential(nn.Linear(num_ftrs, self.out_dim))

  def forward(self, x):
    x = self.densenet121(x)
    x = torch.sigmoid(x)
    return x