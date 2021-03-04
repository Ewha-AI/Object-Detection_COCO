### reference code from https://github.com/quark0/darts/tree/master/cnn --- Differentiable Architecture Search
import torch
import torch.nn as nn
from .conv_ws import ConvAWS2d
from .registry import CONV_LAYERS
import torch.nn.functional as F
#from torch.autograd import Variable
        
class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


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
        self.num_ops = 4
#        x,y,k,_ = self.weight.size()
        
#        self.weight51 = torch.nn.Parameter(torch.randn(x,y,5,5).cuda(),requires_grad=True)
#        self.weight32 = torch.nn.Parameter(torch.randn(self.weight.size()))
        self.op51 = nn.Conv2d(self.in_channels,self.out_channels,kernel_size = 5,padding=4,dilation=2,stride=self.stride,groups=self.groups).cuda()
#        self.op71 = nn.Conv2d(self.in_channels,self.out_channels,kernel_size = 7,padding=3,dilation=1,stride=self.stride,groups=self.groups).cuda()
        self.op53 = nn.Conv2d(self.in_channels,self.out_channels,kernel_size = 5,padding=8,dilation=4,stride=self.stride,groups=self.groups).cuda()
#        self.register_parameter('alpha',None)
#        self._initialize_alphas()
#        self.alpha = torch.nn.Parameter(1e-3*torch.randn(1,self.num_ops,self.out_channels,1,1).cuda(),requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.Tensor(1,self.num_ops))
#        torch.nn.init.xavier_normal_(self.alpha)
#        self.alpha[0,:,:] = 1
#        self.alpha[1,:,:] = 0
        self.weight_diff = torch.nn.Parameter(torch.Tensor(self.weight.size()))
        self.weight_diff.data.zero_()
#        self.weight_diff_ = torch.nn.Parameter(torch.Tensor(5,5))
#        self.weight_diff_.data.zero_()
#    def _initialize_alphas(self):
#        k = sum(1 for i in range(1) for n in range(2+i))
#        self.alpha = torch.nn.Parameter(1e-3*torch.randn(2,self.num_ops).cuda(),requires_grad=True)
        
    def forward(self,x):
        
        
        weight_k31 = self._get_weight(self.weight)
#        weight_k51 = self._get_weight(self.weight51)
        weight = nn.Softmax(dim=-1)(self.alpha)

#        print('weight0: ',weight)
#        print('weight1: ',weight[:,1])
#        print('weight_k31: ',self.weight)
#        print('weight2: ',weight[:,2])
#        weight_k31 = (weight[:,0])*weight_k31
#        weight_k51 = (weight[:,1])*weight_k51
#        weight_k1_ = self._get_weight(self.weight_k1)
#        weight_diff_ = torch.nn.Parameter(torch.Tensor(weight_k5_.size())).cuda()
#        weight_diff_.data.zero_()
#        print('weight_k3: ',self.weight.shape)
       # operation1
#        if self.stride == 1:
#            out_skip = Identity()(x)
#        else:
#            o1 = nn.Conv2d(self.in_channels, self.out_channels // 2, 1, stride=2, padding=0, bias=False).cuda()(x)
#            o2 = nn.Conv2d(self.in_channels, self.out_channels // 2, 1, stride=2, padding=0, bias=False).cuda()(x[:,:,1:,1:])
#            out_skip = torch.cat([o1, o2], dim=1)
#        out_52 = self.op52(x)
        #dil_conv_3x3_1
        out_31 = F.conv2d(x, weight_k31,padding=(1,1),dilation=(1,1),stride=self.stride,groups=self.groups)
        #dil_conv_3x3_2
#        weight_k3 = weight_k3 + self.weight_diff
#        out_32 = F.conv2d(x, weight_k51,padding=(2,2),dilation=(1,1),stride=self.stride,groups=self.groups)
        #dil_conv_3x3_3
        weight_k31 = weight_k31 + self.weight_diff
        out_33 = F.conv2d(x, weight_k31,padding=(2,2),dilation=(2,2),stride=self.stride,groups=self.groups)
#        weight_k3 = weight_k3 + self.weight_diff
#        out_34 = F.conv2d(x, weight_k3,padding=(4,4),dilation=(4,4),stride=self.stride,groups=self.groups)
#        dil_conv_5x5_1
#        weight_k3 = weight_k3 + self.weight_diff
        out_51 = self.op51(x)
#        out_71 = self.op71(x)
        out_53 = self.op53(x)
#        out_51 = F.conv2d(x, self.weight_k5,padding=(2,2),dilation=(1,1),stride=self.stride,groups=self.groups)
        #dil_conv_5x5_2
#        weight_k5_ = weight_k5_ + self.weight_diff_
#        out_52 = F.conv2d(x, weight_k1_,padding=(0,0),dilation=(1,1),stride=self.stride,groups=self.groups)
        #dil_conv_5x5_3
#        weight_k5_ = weight_k5_ + weight_diff_
#        out_53 = F.conv2d(x, weight_k5_,padding=(6,6),dilation=(3,3),stride=self.stride,groups=self.groups)

#        if self.alpha is None:
#        weight = nn.Softmax(dim=-1)(self.alpha)
#        weight = nn.LogSoftmax(dim=1)(self.alpha)
##        print('alpha: ',self.alpha)
##        print('weight: ',weight)
###        print('softmax(alpha): ',weight.shape)
#        for i in range(self.num_ops):
#            #new alpha size
#            w = weight[:,i,:,:]#.unsqueeze(1)
#            #old alpha size
#            w = weight[:,i,:,:]#.unsqueeze(1)
#            if i == 0:
#                out_1 = w*out_skip
###               ## print('out1: ',out_1.shape)
#            if i == 0:
#        print('weight0: ',weight[:,0])
#        print('weight1: ',weight[:,1])
#        print('weight2: ',weight[:,2])
        out_2 = (weight[:,0])*out_31
###                print('out2: ',out_2.shape)
#            elif i == 1:
        out_3 = (weight[:,1])*out_33
        out_4 = (weight[:,2])*out_51
        out_5 = (weight[:,3])*out_53
#        out_4 = (weight[:,2])*out_33
#                print('out3: ',out_3.shape)
##            elif i == 3:
##                out_4 = w*out_33
###                print('out4: ',out_4.shape)
##            elif i == 2:
##                out_5 = w*out_51
###                print('out5: ',out_5.shape)
##            elif i == 3:
##                out_6 = w*out_53
##            elif i == 3:
##                out_7 = w*out_53
##        out = out_31 + out_32
        out = out_2 + out_3 + out_4 +out_5# +out_33#+ out_5 + out_6 #+ out_6
##        print('out_shape: ',out.shape)
#        
                       
        return out