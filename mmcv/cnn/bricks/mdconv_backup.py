import torch

from .conv_ws import ConvAWS2d
from .registry import CONV_LAYERS
import torch.nn.functional as F

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
        self.switch = torch.nn.Conv2d(
            self.in_channels * 3,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=True)
#        self.switch.weight.data.fill_(0)
#        self.switch.bias.data.fill_(1)
#        self.conv3 = torch.nn.Conv2d(
#            self.in_channels,
#            out_channels,
#            kernel_size=3,
#            padding = 1,
#            stride=stride,
#            bias=True)
#        self.conv5 = torch.nn.Conv2d(
#            self.in_channels,
#            out_channels,
#            kernel_size=5,
#            padding = 2,
#            stride=stride,
#            bias=True)
#        self.conv7 = torch.nn.Conv2d(
#            self.in_channels,
#            out_channels,
#            kernel_size=7,
#            padding = 3,
#            stride=stride,
#            bias=True)

    def forward(self, x):
        
#        out_s = self.conv3(x)
#        out_l = self.conv5(x)
#        out_t = self.conv7(x)
##        print('out_t:',out_t.shape)
##        print('out_l:',out_l.shape)
##        print('out_s:',out_s.shape)
#        out_c = torch.cat((out_l,out_s,out_t),1)
#        print('out_c:',out_c.shape)
#        out_x = self.switch(out_c)
##        print('out_x:',out_x.shape)
        weight = self._get_weight(self.weight)
        out_s = F.conv2d(x, weight,padding=(1,1),dilation=(1,1))
#        print('out_s',out_s.shape)
#        ori_p = self.padding
#        ori_d = self.dilation
#        self.padding = tuple(3 * p for p in self.padding)
#        self.dilation = tuple(3 * d for d in self.dilation)
#        print('pad: ',self.padding)
#        print('silated: ',self.dilation)
        weight = weight + self.weight_diff
        out_l = F.conv2d(x, weight,padding=(5,5),dilation=(5,5))
#        print('out_l',out_l.shape)
        weight = weight + self.weight_diff
        out_t = F.conv2d(x, weight,padding=(3,3),dilation=(3,3))
        out_c = torch.cat((out_l,out_s,out_t),1)
        out_x = self.switch(out_c)
#        self.padding = ori_p
#        self.dilation = ori_d

        return out_x
