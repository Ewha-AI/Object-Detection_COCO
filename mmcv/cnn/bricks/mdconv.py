import torch
from torch import nn
from .conv_ws import ConvAWS2d
from .registry import CONV_LAYERS
import torch.nn.functional as F

#class Flatten(nn.Module):
#    def forward(self, x):
#        return x.view(x.size(0), -1)
#    
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
#        self.switch_1 = nn.Sequential(
#            Flatten(),
#            nn.Linear(self.in_channels, self.in_channels)
#            )
        self.switch_1 = torch.nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
#            padding = 1,
            stride=stride,
            bias=True)
        self.switch_1.weight.data.fill_(0)
        self.switch_1.bias.data.fill_(1)
#        self.switch_2 = nn.Sequential(
#            Flatten(),
#            nn.Linear(self.in_channels,3)
#            )
#        self.switch_3 = torch.nn.Conv2d(
#            self.in_channels,
#            self.in_channels,
#            kernel_size=1,
##            padding = 1,
#            stride=1,
#            bias=True)
#        self.switch_3.weight.data.fill_(0)
#        self.switch_3.bias.data.fill_(1)
        self.switch_2 = torch.nn.Conv2d(
            self.in_channels,
            3,
            kernel_size=1,
            stride=stride,
            bias=True)
        self.switch_2.weight.data.fill_(0)
        self.switch_2.bias.data.fill_(1)

#        self.switch_3 = torch.nn.Conv2d(
#            8,
#            4,
#            kernel_size=1,
#            stride=1,
#            bias=True)
#        self.switch_3.weight.data.fill_(0)
#        self.switch_3.bias.data.fill_(1)
#        self.BN = torch.nn.BatchNorm2d(self.out_channels)
#        self.BN2 = torch.nn.BatchNorm2d(3)
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
        
       ################# method 2###################
#        avg_x0 = torch.nn.functional.pad(x, pad=(2,2,2,2), mode="reflect")
#        avg_x0 = torch.nn.functional.avg_pool2d(avg_x0, kernel_size=5, stride=self.stride, padding=0)
#        avg_x0 = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
#        avg_x1 = self.switch_1(avg_x0)
##        avg_x1 = self.BN(avg_x1)
##        avg_x1 = nn.ReLU(inplace=True)(avg_x1)
##        avg_x1 = self.switch_3(avg_x1)
##        avg_x1 = self.BN3(avg_x1)   
#        avg_x2 = torch.nn.functional.adaptive_max_pool2d(x, output_size=1)
##        avg_x2 = torch.nn.functional.pad(x, pad=(2,2,2,2), mode="reflect")
##        avg_x2 = torch.nn.functional.max_pool2d(x, kernel_size=5, stride=self.stride, padding=0)
#        avg_x2 = self.switch_2(avg_x2)
##        avg_x2 = self.BN2(avg_x2)  
##        avg_x = torch.cat((avg_x1,avg_x2),1)
#        avg_x = avg_x1+avg_x2
##        avg_x = self.switch_3(avg_x)
#        switch_weight = nn.Softmax(dim=1)(avg_x)
       
       ############### method 3 #################
#        avg_x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
#        avg_x = self.BN(avg_x)
#        avg_x = self.switch_1(avg_x)
#        avg_x = nn.ReLU(inplace=True)(avg_x)
#        avg_x = self.switch_2(avg_x)
#        switch_weight = nn.Softmax(dim=1)(avg_x)
       
#        ##### multi-rate dilated conv #####
        weight = self._get_weight(self.weight)
        out_o = F.conv2d(x, weight,padding=(1,1),dilation=(1,1),stride=self.stride,groups=self.groups)
        weight = weight + self.weight_diff
        out_s = F.conv2d(x, weight,padding=(2,2),dilation=(2,2),stride=self.stride,groups=self.groups)
##        print('out_s: ',out_s.shape)
##        print('out_o: ',out_o.shape)
#        weight = weight + self.weight_diff
#        out_l = F.conv2d(x, weight,padding=(3,3),dilation=(3,3),stride=self.stride,groups=self.groups)
        weight = weight + self.weight_diff
        out_m = F.conv2d(x, weight,padding=(4,4),dilation=(4,4),stride=self.stride,groups=self.groups)
        
##        print('out_l: ',out_o.shape)
        
        
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
#                out_l = out_l*weight_s
#            elif i == 3:
                out_m = out_m*weight_s
##                
        out = out_o + out_s + out_m #+ out_l
#        out = self.BN(out)
#        out = out_o
        return out
