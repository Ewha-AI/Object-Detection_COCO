### reference code from https://github.com/quark0/darts/tree/master/cnn --- Differentiable Architecture Search
import torch
import torch.nn as nn
from .conv_ws import ConvAWS2d
from .registry import CONV_LAYERS
import torch.nn.functional as F

OPS = {
       'skip_connect': lambda C, stride,affine,groups: Identity() if stride == 1 else nn.Conv2d(C, C, 1, stride=2, padding=0, bias=False),
       'dil_conv_3x3_1': lambda C, stride,affine,groups: nn.Conv2d(C,C,3,stride,padding=1,dilation=1,groups=groups),
       'dil_conv_3x3_2': lambda C, stride,affine,groups: nn.Conv2d(C,C,3,stride,padding=2,dilation=2,groups=groups),
       'dil_conv_3x3_3': lambda C, stride,affine,groups: nn.Conv2d(C,C,3,stride,padding=3,dilation=3,groups=groups),
       'dil_conv_5x5_1': lambda C, stride,affine,groups: nn.Conv2d(C,C,5,stride,padding=2,dilation=1,groups=groups),
       'dil_conv_5x5_2': lambda C, stride,affine,groups: nn.Conv2d(C,C,5,stride,padding=4,dilation=2,groups=groups),
       'dil_conv_5x5_3': lambda C, stride,affine,groups: nn.Conv2d(C,C,5,stride,padding=6,dilation=3,groups=groups),
       }
PRIMITIVES = [
        'skip_connect',
        'dil_conv_3x3_1',
        'dil_conv_3x3_2',
        'dil_conv_3x3_3',
        'dil_conv_5x5_1',
        'dil_conv_5x5_2',
        'dil_conv_5x5_3'
        ]
        
class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

#class FactorizedReduce(nn.Module):
#
#  def __init__(self, C_in, C_out, affine=True):
#    super(FactorizedReduce, self).__init__()
#    assert C_out % 2 == 0
#    self.conv_1 = nn.Conv2d(C_in, C_out, 1, stride=2, padding=0, bias=False)
#    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
#
#  def forward(self, x):
#      x = x.cpu()
#      out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
#      return out.cuda()

class MixedOp(nn.Module):
    def __init__(self, C, stride,groups):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride,False,groups)
            self._ops.append(op)
        self._ops = self._ops.cuda()
   
    def forward(self, x, weights):
        i = 0
        for op in self._ops:
#            print('_ops: ',op)
            w_ = weights[:,i,:,:].unsqueeze(1)
#            print('w_: ',w_.shape)
            if i == 0:
                out = w_*op(x)
            else: 
                out = out + (w_*op(x))
            i += 1
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
        self.stride = stride
        self.groups = groups
        self._initialize_alphas()
        
    def forward(self,x):

        weight = F.softmax(self.alpha,dim=-1)
#        print('softmax(alpha): ',weight.shape)
        out = MixedOp(self.in_channels,self.stride,self.groups)(x,weight)
        
        return out
    
    def _initialize_alphas(self):
        k = sum(1 for i in range(1) for n in range(2+i))
        num_ops = len(PRIMITIVES)
        self.alpha = torch.nn.Parameter(1e-3*torch.randn(k,num_ops,1,1).cuda(),requires_grad=True)