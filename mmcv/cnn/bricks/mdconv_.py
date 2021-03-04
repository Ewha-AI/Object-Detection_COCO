import torch
from torch import nn
from .conv_ws import ConvAWS2d
from .registry import CONV_LAYERS
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
@CONV_LAYERS.register_module('MDConv')
class MDconv(ConvAWS2d):

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
#        self.switch_1 = torch.nn.Conv2d(
#            self.in_channels,
#            self.in_channels,
#            kernel_size=1,
##            padding = 1,
#            stride=1,
#            bias=True)
#        self.switch_1.weight.data.fill_(0)
#        self.switch_1.bias.data.fill_(1)
#        self.switch_2 = nn.Sequential(
#            Flatten(),
#            nn.Linear(self.in_channels, 4)
#            )
        self.switch_2 = torch.nn.Conv2d(
            int(self.in_channels),
            1,
            kernel_size=1,
            stride=stride,
            bias=True)
        self.switch_2.weight.data.fill_(0)
        self.switch_2.bias.data.fill_(1)
#        self.switch_3 = torch.nn.Conv2d(
#            2,
#            2,
#            kernel_size=1,
#            stride=1,
#            bias=True)
#        self.switch_3.weight.data.fill_(0)
#        self.switch_3.bias.data.fill_(1)
#        self.BN = torch.nn.BatchNorm2d(self.in_channels)
#        self.BN2 = torch.nn.BatchNorm2d(2)
#        self.normlayer = nn.LayerNorm([self.in_channels,1,1])
#        self.conv1 = torch.nn.Conv2d(
#            self.in_channels*4,
#            self.out_channels,
#            kernel_size=1,
#            bias=True)
#        self.in_channels = in_channels
        self.stride = stride
#        self.switch = nn.Sequential(
#                nn.avg_pool2d
#                )
#        self.weight_diff = torch.nn.Parameter(torch.Tensor(self.weight.size()))
#        self.weight_diff.data.zero_()
        self.pre_context = torch.nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
            bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)
        self.post_context = torch.nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=1,
            bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)
     

    def forward(self, x):
###################### Method1 #########################
        # pre-context
        avg_x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x
        ###case Global average pooling####
#        print('in_channels: ',self.in_channels)
#        avg_x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
        ###case avgerage pooling ###
#        print('x:',x.shape)
#        avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2), mode="reflect")
#        avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=self.stride, padding=0)
##        print('avg_x_pool: ',avg_x.shape)
#        avg_x = self.switch_1(avg_x)
#        avg_x = self.BN(avg_x)
##        print('avg_x_switch1: ',avg_x.shape)
##        avg_x = self.normlayer(avg_x)
#        avg_x = nn.ReLU(inplace=True)(avg_x)
##        print('avg_x_ReLU: ',avg_x.shape)
#        avg_x = self.switch_2(avg_x)
##        avg_x = self.BN2(avg_x)
##        print('avg_x_switch2: ',avg_x.shape)
##        switch_weight = nn.Sigmoid()(avg_x)
##        avg_x = self.normlayer(avg_x)
##        avg_x = torch.nn.functional.adaptive_avg_pool2d(avg_x, output_size=1)
#        switch_weight = nn.Softmax(dim=1)(avg_x)
#        switch_weight = self.switch_3(switch_weight)
#        print('avg_x_softmax: ',switch_weight.shape)
         # switch
        avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch_2(avg_x)
            
#        ##### multi-rate dilated conv #####
#        weight = self._get_weight(self.weight)
#        out_o = F.conv2d(x, weight,padding=(1,1),dilation=(1,1),stride=self.stride)
#        weight = weight + self.weight_diff
#        out_s = F.conv2d(x, weight,padding=(3,3),dilation=(3,3),stride=self.stride)
#        print('out_s: ',out_s.shape),
#        weight = weight + self.weight_diff
#        out_l = F.conv2d(x, weight,padding=(4,4),dilation=(4,4),stride=self.stride)
#        weight = weight + self.weight_diff
#        out_m = F.conv2d(x, weight,padding=(8,8),dilation=(8,8),stride=self.stride)
        
#        print('out_l: ',out_o.shape)
        #multiply weight switch
#        switch_weight =  switch_weight.unsqueeze(-1)
#        switch_weight =  switch_weight.unsqueeze(-1)
#        _,C,H,W = switch_weight.shape
##        B,C = switch_weight.shape
#        for i in range(C):
#            weight = switch_weight[:,i,:,:].unsqueeze(1)
##            print('weight: ',weight.shape)
#            if i == 0:
#                out_o = out_o*weight
#            elif i == 1:
#                out_s = out_s*weight
#            elif i == 2:
#                out_l = out_l*weight
#            elif i == 3:
#                out_m = out_m*weight
         # sac
        weight = self._get_weight(self.weight)

        out_s = F.conv2d(x, weight)
        ori_p = self.padding
        ori_d = self.dilation
        self.padding = tuple(3 * p for p in self.padding)
        self.dilation = tuple(3 * d for d in self.dilation)
        weight = weight + self.weight_diff
        out_l = F.conv2d(x, weight)
        out = switch * out_s + (1 - switch) * out_l
        self.padding = ori_p
        self.dilation = ori_d
                
#        out = out_o + out_s #+ out_l #+ out_m
        # post-context
        avg_x = torch.nn.functional.adaptive_avg_pool2d(out, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x
###################### Method2 #########################
#        weight = self._get_weight(self.weight)
#        out_o = F.conv2d(x, weight,padding=(1,1),dilation=(1,1),stride=self.stride)
#        weight = weight + self.weight_diff
#        out_s = F.conv2d(x, weight,padding=(2,2),dilation=(2,2),stride=self.stride)
##        print('out_s: ',out_s.shape),
#        weight = weight + self.weight_diff
#        out_l = F.conv2d(x, weight,padding=(4,4),dilation=(4,4),stride=self.stride)
#        weight = weight + self.weight_diff
#        out_m = F.conv2d(x, weight,padding=(8,8),dilation=(8,8),stride=self.stride)
#        out_c = torch.cat((out_o,out_s,out_l,out_m),1)
#        avg_x = torch.nn.functional.adaptive_avg_pool2d(out_c, output_size=1)
#        avg_x = self.switch_1(avg_x)
#        avg_x = nn.ReLU(inplace=True)(avg_x)
#        avg_x = self.switch_2(avg_x)
#        switch_weight = nn.Softmax(dim=1)(avg_x)
#        out = out_c*switch_weight
#        out = self.conv1(out)
        
        return out
