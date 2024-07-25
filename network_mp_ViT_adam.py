import numpy as np
import clip
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=20000.0):
    # return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
    return 1.0

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

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=512):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

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

class res_layer(nn.Module):
  def __init__(self, size):
    super(res_layer, self).__init__()
    self.l_1 = nn.Linear(size, size)
    self.l_1.apply(init_weights)
    self.relu = nn.ReLU(inplace=True)
    self.l_2 = nn.Linear(size, size)
    self.l_2.apply(init_weights)

  def forward(self, x):
    out = self.l_1(x)
    out = self.relu(out)
    out = self.l_2(out)
    out += x
    out = self.relu(out)
    return out

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

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = nn.functional.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, _ = x.size()
        qkv = self.qkv_proj(x)
        seq_length=1

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.5):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.ReLU(inplace=True)
        self.norm2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        # x = x + attn_out
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + linear_out
        x = self.norm2(x)
        return x

class ResNetFc(nn.Module):
  def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000, num_lay=1, net='res2'):
    super(ResNetFc, self).__init__()
    model_clip, _  = clip.load("ViT-B/16")
    model_ViT = model_clip.visual.type(torch.float32)
    self.feature_layers = model_ViT
    # self.feature_layers = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4)

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    # cw = np.load("TextFeats.npy")
    # class_weights = torch.from_numpy(cw).cuda()
    # class_weights = class_weights / class_weights.norm(dim=1, keepdim=True)
    # logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp().cuda()
    # class_weights = class_weights * logit_scale

    #new_param = torch.load('class_layers.pth.tar')
    layers = []
    if(net=='lin'):
        print("ClassNet: lin")
        for i in range(num_lay):
            c_lay = nn.Linear(512, 512)
            c_lay.apply(init_weights)
            layers.append(c_lay)
            layers.append(nn.ReLU(inplace=True))
        bnl = nn.Linear(512, bottleneck_dim)
        bnl.apply(init_weights)
        layers.append(bnl)
    if(net=='lin2'):
        print("ClassNet: lin2")
        for i in range(num_lay):
            if i == 0:
                c_lay = nn.Linear(512, 1024)
            else:
                c_lay = nn.Linear(1024, 1024)
            c_lay.apply(init_weights)
            layers.append(c_lay)
            layers.append(nn.ReLU(inplace=True))
        bnl = nn.Linear(1024, bottleneck_dim)
        bnl.apply(init_weights)
        layers.append(bnl)
    if(net=='res'):
        print("ClassNet: re")
        for i in range(num_lay):
            c_lay = res_layer(512)
            layers.append(c_lay)
        bnl = nn.Linear(512, bottleneck_dim)
        bnl.apply(init_weights)
        layers.append(bnl)
    if(net=='res2'):
        print("ClassNet: res2")
        upl = nn.Linear(512, 1024)
        upl.apply(init_weights)
        layers.append(upl)
        layers.append(nn.ReLU(inplace=True))
        for i in range(num_lay):
            c_lay = res_layer(1024)
            layers.append(c_lay)
        bnl = nn.Linear(1024, bottleneck_dim)
        bnl.apply(init_weights)
        layers.append(bnl)
    if(net=='att'):
        print("ClassNet: att")
        for i in range(num_lay):
            nl = MultiheadAttention(512, 512, 2)
            layers.append(nl)
            layers.append(nn.ReLU(inplace=True))
        bnl = nn.Linear(512, bottleneck_dim)
        bnl.apply(init_weights)
        layers.append(bnl)
    if(net=='enc'):
        print("ClassNet: enc")
        for i in range(num_lay):
            nl = EncoderBlock(512, 2, 2*512)
            layers.append(nl)
        bnl = nn.Linear(512, bottleneck_dim)
        bnl.apply(init_weights)
        layers.append(bnl)
    self.bottleneck = nn.Sequential(*layers)
    self.fc = nn.Linear(bottleneck_dim, class_num)
    self.fc.apply(init_weights)
    self.__in_features = bottleneck_dim

    self.gradients = []
    self.lr_back = 10

  def change_par_bn_state(self, s):
    for name, module in self.named_modules():
      if isinstance(module, par_bn):
        module.state = s

  def save_gradient(self, grad):
    self.gradients.append(grad)

  def cn_forward(self, x):
    x = self.bottleneck(x)
    x = self.fc(x)
    return(x)

  def del_gradient(self):
    self.gradients = []

  def forward(self, x):
    x = self.feature_layers(x)
    x = self.bottleneck(x)
    y = self.fc(x)
    return x, y

  def bp(self, x):
    x = self.feature_layers(x)
    # x = x * 10.0
    x = self.bottleneck(x)
    x.register_hook(self.save_gradient)
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
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 20000.0
    self.lr_back = 10

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
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
