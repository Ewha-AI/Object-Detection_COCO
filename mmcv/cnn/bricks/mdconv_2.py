import torch
from torch import nn
from .conv_ws import ConvAWS2d
from .registry import CONV_LAYERS
import torch.nn.functional as F

@CONV_LAYERS.register_module('MDConv2')
class MDconv2(ConvAWS2d):

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
            self.in_channels*3,
            int(self.in_channels),
            kernel_size=1,
            stride=1,
            bias=True)
        self.switch_1.weight.data.fill_(0)
        self.switch_1.bias.data.fill_(1)
        self.switch_2 = torch.nn.Conv2d(
            int(self.in_channels),
            out_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        self.switch_2.weight.data.fill_(0)
        self.switch_2.bias.data.fill_(1)
#        self.conv1 = torch.nn.Conv2d(
#            self.in_channels,
#            self.out_channels,
#            kernel_size=1,
#            bias=True)
        self.stride = stride
#        self.switch = nn.Sequential(
#                nn.avg_pool2d
#                )
#        self.weight_diff = torch.nn.Parameter(torch.Tensor(self.weight.size()))
#        self.weight_diff.data.zero_()
#        

    def forward(self, x):
        
        ##### multi-rate dilated conv #####
        weight = self._get_weight(self.weight)
        out_s = F.conv2d(x, weight,padding=(1,1),dilation=(1,1),stride=self.stride)
#        print('out_s: ',out_s.shape)
        weight = weight + self.weight_diff
        out_l = F.conv2d(x, weight,padding=(3,3),dilation=(3,3),stride=self.stride)
        weight = weight + self.weight_diff
        out_m = F.conv2d(x, weight,padding=(5,5),dilation=(5,5),stride=self.stride)
#        print('out_s: ', out_s.shape)
        #concat out
        out_a = torch.cat((out_s,out_l,out_m),1)
#        out_a = torch.nn.functional.adaptive_avg_pool2d(out_a, output_size=1)
        out_a = torch.nn.functional.pad(out_a, pad=(2, 2, 2, 2), mode="reflect")
        out_a = torch.nn.functional.avg_pool2d(out_a, kernel_size=5, stride=1, padding=0)
#        print('out_a: ',out_a.shape)
        out_a = self.switch_1(out_a)
#        print('out_a_switch1: ',out_a.shape)
        out_a = nn.ReLU(inplace=True)(out_a)
#        print('avg_x_ReLU: ',avg_x.shape)
        out = self.switch_2(out_a)
#        print('out: ',out.shape)

        return out
