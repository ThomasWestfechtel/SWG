import numpy as np
import clip
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
        #return 0 * grad.clone()
    return fun1

class par_bn(nn.Module):
  def __init__(self, s_bn, t_bn):
    super(par_bn, self).__init__()
    self.state = 0
    self.s_bn = s_bn
    self.t_bn = t_bn

  def forward(self, x):
    if(self.state == 0):
      x = self.s_bn(x)
    elif(self.state == 1):
      x = self.t_bn(x)
    else:
      print("Unexpected state")
    return x

class get_stats(nn.Module):
  def __init__(self):
    super(get_stats, self).__init__()
    self.samples = 0.0
    self.mean = []
    self.var = []
    # self.f_mean = []
    # self.f_var = []
    # self.mean_mis = []
    # self.var_mis = []

  def stat_out(self):
    return self.mean, self.var

  def reset(self):
    self.samples = 0.0
    self.mean = []
    self.var = []

  def forward(self, x):
    if(self.samples == 0):
        self.mean = torch.mean(x, dim=[0,2,3])
        self.var = torch.var(x, dim=[0,2,3])
        # self.f_mean = torch.mean(x, dim=[0,2,3])
        # self.f_var = torch.var(x, dim=[0,2,3])
        self.samples = x.shape[0]
    else:
        c_mean = torch.mean(x, dim=[0,2,3])
        c_var = torch.var(x, dim=[0,2,3])
        c_samples = x.shape[0]

        all_samples = self.samples + c_samples

        self.var = (self.samples / all_samples) * (self.var ** 2) + (c_samples / all_samples) * (c_var ** 2) + (self.samples * c_samples) / (all_samples ** 2) * ((self.mean - c_mean) ** 2)
        self.var = torch.sqrt(self.var)
        self.mean = (self.samples / all_samples) * self.mean + (c_samples / all_samples) * c_mean
        self.samples = all_samples

        # self.mean_mis = self.f_mean - self.mean
        # self.var_mis = self.f_var - self.var
    return x


class basenet(nn.Module):
  def __init__(self, name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
    super(basenet, self).__init__()
    model_clip, _  = clip.load(name)
    model_resnet = model_clip.visual.type(torch.float32)
    conv1 = model_resnet.conv1
    bn1 = model_resnet.bn1
    relu1 = model_resnet.relu1
    conv2 = model_resnet.conv2
    bn2 = model_resnet.bn2
    relu2 = model_resnet.relu2
    conv3 = model_resnet.conv3
    bn3 = model_resnet.bn3
    relu3 = model_resnet.relu3
    avgpool = model_resnet.avgpool
    layer1 = model_resnet.layer1
    layer2 = model_resnet.layer2
    layer3 = model_resnet.layer3
    layer4 = model_resnet.layer4
    attnpool = model_resnet.attnpool
    self.feature_layers = nn.Sequential(conv1, bn1, relu1, \
                        conv2, bn2, relu2, \
                        conv3, bn3, relu3, avgpool, \
                         layer1, layer2, layer3, layer4, attnpool)
    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls

    #new_param = torch.load('class_layers.pth.tar')
    layers = []
    if(name == "RN50"):
        bnl = nn.Linear(1024, bottleneck_dim)
    else:
        bnl = nn.Linear(512, bottleneck_dim)
    bnl.apply(init_weights)
    layers.append(bnl)
    self.bottleneck = nn.Sequential(*layers)
    self.fc = nn.Linear(bottleneck_dim, class_num)
    self.fc.apply(init_weights)
    self.__in_features = bottleneck_dim
    self.lr_back = 10

  def change_par_bn_state(self, s):
    for name, module in self.named_modules():
      if isinstance(module, par_bn):
        module.state = s

  def forward(self, x):
    x = self.feature_layers(x)
    x = self.bottleneck(x)
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':1}, \
                    {"params":self.bottleneck.parameters(), "lr_mult":self.lr_back, 'decay_mult':1}, \
                    {"params":self.fc.parameters(), "lr_mult":self.lr_back, 'decay_mult':1}]
    # parameter_list = [{"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
    #                 {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
    return parameter_list


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)

    self.lr_back = 10

  def forward(self, x):
    coeff = 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":self.lr_back, 'decay_mult':1}]
