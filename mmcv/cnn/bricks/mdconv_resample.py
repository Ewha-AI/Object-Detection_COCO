import torch
from torch import nn
from .conv_ws import ConvAWS2d
from .registry import CONV_LAYERS
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable

#class Flatten(nn.Module):
#    def forward(self, x):
#        return x.view(x.size(0), -1)
#    
@CONV_LAYERS.register_module('MDConv_re')
class MDconv_re(ConvAWS2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 use_deform=False):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.weight_diff = torch.nn.Parameter(torch.Tensor(self.weight.size()))
        self.weight_diff.data.zero_()
        self.switch_1 = torch.nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
#            padding = 1,
            stride=stride,
            bias=True)
        self.switch_1.weight.data.fill_(0)
        self.switch_1.bias.data.fill_(1)
        self.switch_2 = torch.nn.Conv2d(
            self.in_channels,
            4,
            kernel_size=1,
            stride=stride,
            bias=True)
        self.switch_2.weight.data.fill_(0)
        self.switch_2.bias.data.fill_(1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

     

    def forward(self, x):


        ###case Global average pooling####
   #     print('in_channels: ',self.in_channels)
        avg_x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
##        ###case avgerage pooling ###
###        print('x:',x.shape)
#        avg_x = torch.nn.functional.pad(x, pad=(2,2,2,2), mode="reflect")
#        avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=self.stride, padding=0)
####        print('avg_x_pool: ',avg_x.shape)
        avg_x = self.switch_1(avg_x)
#        avg_x = self.BN(avg_x)
####        print('avg_x_switch1: ',avg_x.shape)
####        avg_x = self.normlayer(avg_x)
        avg_x = nn.ReLU(inplace=True)(avg_x)
####        print('avg_x_ReLU: ',avg_x.shape)
#        avg_x = self.switch_3(avg_x)
#        avg_x = self.BN2(avg_x)
#        avg_x = nn.ReLU(inplace=True)(avg_x)
##       
        avg_x = self.switch_2(avg_x)
###       avg_x = self.BN2(avg_x)
#####        print('avg_x_switch2: ',avg_x.shape)
#####        switch_weight = nn.Sigmoid()(avg_x)
#####        avg_x = self.normlayer(avg_x)
#####        avg_x = torch.nn.functional.adaptive_avg_pool2d(avg_x, output_size=1)
        switch_weight = nn.Softmax(dim=1)(avg_x)
#        switch_weight = self.switch_3(switch_weight)
        
#        switch_weight = nn.ReLU(inplace=True)(avg_x)
#        switch_weight = self.switch_3(switch_weight)
#        print('avg_x_softmax: ',switch_weight.shape)
       
#        ##### multi-rate dilated conv #####
        weight = self._get_weight(self.weight)
#        print('weight: ',weight.shape)
        out_o = F.conv2d(x, weight,padding=(1,1),dilation=(1,1),stride=self.stride,groups=self.groups)
        weight = weight + self.weight_diff
        out_s = F.conv2d(x, weight,padding=(3,3),dilation=(3,3),stride=self.stride,groups=self.groups)
        weight = torch.nn.functional.pad(weight, pad=(1,1,1,1), mode="reflect")
        self.weight_diff = torch.nn.Parameter(torch.Tensor(weight.size())).cuda()
        print('self.weight_diff : ',self.weight_diff.shape)
##        print('out_o: ',out_o.shape)
        weight = weight + self.weight_diff
        out_l = F.conv2d(x, weight,padding=(2,2),dilation=(1,1),stride=self.stride,groups=self.groups)
        weight = weight + self.weight_diff
        out_m = F.conv2d(x, weight,padding=(6,6),dilation=(3,3),stride=self.stride,groups=self.groups)
        
#        probs = torch.Tensor([[0.1, 0.1, 0.7,0.1], [0.1, 0.6, 0.1,0.2]])
#        probs = torch.Tensor([[[[0.1]],[[0.1]], [[0.7]],[[0.1]]], [[[0.1]], [[0.6]], [[0.1]],[[0.2]]]])
#        new_probs = (switch_weight.squeeze(-1)).squeeze(-1)
#        print('switch_weight: ',new_probs)
#        m = Categorical(new_probs)#.squeeze(-1)).squeeze(-1))
#        print('m: ',m)
#        action = m.sample()
#        print('action: ',action)
#        out_o = nn.Conv2d(self.in_channels,self.out_channels, kernel_size=3,padding=(1,1),dilation=(1,1),stride=self.stride,groups=self.groups)(x)
##        weight = weight + self.weight_diff
#        out_s = nn.Conv2d(self.in_channels,self.out_channels, kernel_size=5,padding=(2,2),dilation=(1,1),stride=self.stride,groups=self.groups)(x)
#        out_l = nn.Conv2d(self.in_channels,self.out_channels, kernel_size=7,padding=(3,3),dilation=(1,1),stride=self.stride,groups=self.groups)(x)
#        out_m = nn.Conv2d(self.in_channels,self.out_channels, kernel_size=9,padding=(4,4),dilation=(1,1),stride=self.stride,groups=self.groups)(x)
        #soft switch
        _,C,H,W = switch_weight.shape
#        print(switch_weight)
#        B,C = switch_weight.shape
        for i in range(C):
            weight_s = switch_weight[:,i,:,:].unsqueeze(1)
#            print('weight: ',weight.shape)
            if i == 0:
                out_o = out_o*weight_s
            elif i == 1:
                out_s = out_s*weight_s
            elif i == 2:
                out_l = out_l*weight_s
            elif i == 3:
                out_m = out_m*weight_s
#                
        out = out_o + out_s + out_l + out_m

#        out = out_o
        return out
