import torch
import torch.nn as nn
from .conv_ws import CONV_LAYERS,ConvAWS2d
#from mmcv.cnn.utils import constant_init
#from .registry import CONV_LAYERS
import torch.nn.functional as F
#from mmcv.utils import TORCH_VERSION  
#from torch.autograd import Variable

class Attention(nn.Module):
    
    def __init__(self,in_channels,out_channels,stride):
        super(Attention,self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,in_channels, kernel_size=1, stride=1, bias=True)
        self.stride = stride
        self.conv2 = nn.Conv2d(
            in_channels,out_channels, kernel_size=1, stride=1, bias=True)
        
    def forward(self,x):
#        avg_x = F.pad(x, pad=(1, 1, 1, 1), mode='reflect')
#        avg_x = F.avg_pool2d(avg_x, kernel_size=3, stride=1, padding=0)
        gap_x = F.adaptive_avg_pool2d(x,output_size = 1)
        gap_x = self.conv1(gap_x)
        gap_x = nn.ReLU()(gap_x)
        gap_x = self.conv2(gap_x)
#        print(gap_x.shape)
        out = nn.Softmax(dim=1)(gap_x)
        return out
        
@CONV_LAYERS.register_module('NASConv')
class NAS_Conv(ConvAWS2d):

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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
#        self.num_ops = 3
        self.attention1 = Attention(self.in_channels,self.out_channels,self.stride)
#        self.attention2 = Attention(self.in_channels,self.out_channels,self.stride)
#        x,y,k,_ = self.weight.size()        
        self.op51 = nn.Conv2d(self.in_channels,self.out_channels,kernel_size = 5,padding=2,dilation=1,stride=self.stride,groups=self.groups).cuda()
#        self.op11 = nn.Conv2d(self.in_channels,self.out_channels,kernel_size = 1,dilation=1,stride=self.stride,groups=self.groups).cuda()
#        self.weight_diff = torch.nn.Parameter(torch.Tensor(self.weight.size()))
#        self.weight_diff.data.zero_()
#        self.weight51 = torch.nn.Parameter(torch.randn(x,y,5,5).cuda(),requires_grad=True)
#        self.alpha = torch.nn.Parameter(1e-3*torch.randn(1,self.num_ops,1,1).cuda(),requires_grad=True)

    def forward(self,x):

        weight_k31 = self._get_weight(self.weight)
#        weight_k51 = self._get_weight(self.weight51)
#        weight = nn.Softmax(dim=1)(self.alpha)
#        out_11 =self.op11(x)
        out_31 = F.conv2d(x, weight_k31,padding=(1,1),dilation=(1,1),stride=self.stride,groups=self.groups)
#        weight_k31 = weight_k31 + self.weight_diff
#        out_33 = F.conv2d(x, weight_k31,padding=(3,3),dilation=(3,3),stride=self.stride,groups=self.groups)
        out_51 = self.op51(x)
#        out_sum2 = out_11 + out_31
#        mask_sum2 = self.attention2(out_sum2)
        
        out_sum1 = out_31 + out_51
        mask_sum1 = self.attention1(out_sum1)
#        print(mask_sum1.shape)
#        print('out_51',out_51.shape)
        out1 = (mask_sum1*out_31) + ((1-mask_sum1)*out_51)
#        out2 = (mask_sum2*out_11) + ((1-mask_sum2)*out_51)

                       
        return out1# + out2