import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


"""Reference from https://github.com/jchen42703/CapsNetsLASeg/blob/master/capsnets_laseg/models/capsnets.py#L165"""    

class Length(nn.Module):
    """Computes the length of output vector:
        Input_shape  : [None,h,w,num_capsules,num_dims]
        output_shape : [None,h,w,1]
    """
    def __init__(self,last_in_ch):
        super(Length,self).__init__()
        
        self.last_in_ch = last_in_ch
        
    def forward(self,inputs):
        _,_,_,num_cap,num_dim = inputs.shape
        
        norm_dim = torch.norm(inputs,dim=-1)   
        for i in range(num_cap):
            norm_ex = norm_dim[:,:,:,i]
            if i == 0:
                norm = norm_ex
            else:
                norm = norm*norm_ex
        norm = norm.unsqueeze(-1)
#        norm = norm.repeat(1,1,1,self.last_in_ch)
        norm = norm.permute(0,3,1,2)
        return norm        
        

class ConvCapsuleLayer(nn.Module):
    def __init__(self,kernel_size,num_capsule,num_atom,strides=1,padding=2,routings=3,update_routing=True):
        super(ConvCapsuleLayer,self).__init__()
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atom = num_atom
        self.strides = strides
        self.padding = padding
        self.routings = routings
        self.update_routing = update_routing
        
    def weight(self,x):
        input_shape = x.size()
#        print('input_shape: ',input_shape)
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atom = input_shape[4]
        
        #Transform matrix
        self.W = nn.Parameter(torch.Tensor(self.num_capsule*self.num_atom,self.input_num_atom,
                                          self.kernel_size,self.kernel_size))
        self.b = Variable(torch.Tensor(1,1,self.num_capsule,self.num_atom))
        self.W.data.fill_(0)
        self.b.fill_(1)
        
    def forward(self,x):
        self.weight(x)
        input_transposed = x.permute(3,0,1,2,4)
        input_shape = input_transposed.shape
        input_reshape = input_transposed.reshape(input_shape[0]*input_shape[1],self.input_num_atom,
                                         self.input_height,self.input_width)
        conv = F.conv2d(input_reshape.data.cpu(),self.W,stride=self.strides,padding=self.padding,dilation=1)
        conv = conv.view(conv.size(0),conv.size(2),conv.size(3),-1)
        if update_routing:
            votes_shape = conv.shape
            _,_,conv_height,conv_width = conv.size()
            votes = conv.view(input_shape[1],input_shape[0],votes_shape[1],votes_shape[2],
                          self.num_capsule,self.num_atom)
            logit_shape = torch.zeros(input_shape[1],input_shape[0],votes_shape[1],votes_shape[2],
                                   self.num_capsule)
            biases_replicated = self.b.repeat(votes_shape[1],votes_shape[2],1,1)
            activations = update_routing(
                    votes=votes,
                    biases=biases_replicated,
                    logit_shape=logit_shape,
                    num_dims=6,
                    input_dim=self.input_num_capsule,
                    output_dim=self.num_capsule,
                    num_routing=self.routings
                    )
        else:
            activations = conv
#        print(activations.shape)
        return activations
        
def update_routing(votes,biases,logit_shape,num_dims,input_dim,
                   output_dim,num_routing):
    if num_dims == 6:
        votes_t_shape = [5,0,1,2,3,4]
        r_t_shape = [1,2,3,4,5,0]
    elif num_dims == 4:
        votes_t_shape = [3,0,1,2]
        r_t_shape = [1,2,3,0]
    else:
        raise NotImplementedError('Not implemented')
    votes_trans = votes.permute(votes_t_shape)
    _,_,_,height,width,caps = votes_trans.shape
    
    def squash(input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor 
    
    def _body(i,logits,activations):
        "Routing While loop"
        route = F.softmax(logits,dim=-1)
        preactivate_unrolled = route.data.cpu()*votes_trans.data.cpu()
        preact_trans = preactivate_unrolled.permute(r_t_shape)
        preactivate = torch.sum(preact_trans,dim=1) + biases
        activation = squash(preactivate)
        activations_ = activation
        act_3d = activation.unsqueeze(1)
        tile_shape = np.ones(num_dims,dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = act_3d.repeat(tile_shape)
        distances = torch.sum(votes.data.cpu()*act_replicated,dim=-1)
        logits += distances
        return (i+1,logits,activations_)
    
    activations = torch.empty(num_routing,dtype=torch.float32)
#    print('activations: ',activations)
    logits = logit_shape
    
    for i in range(num_routing):
        _,logits,activations = _body(i,logits,activations)
        
    return activations.to(torch.float32)
       

class CapsNet(nn.Module):
    def __init__(self,num_pri_cap = 8,num_att_cap = 1,last_in_ch = 2048,update_routing_pri=False,num_att_routing=3):
        super(CapsNet, self).__init__()

        self.primary_capsules = ConvCapsuleLayer(kernel_size=3,num_capsule=num_pri_cap,num_atom=52,strides=1
                                                 ,padding=1,routings=1,update_routing=update_routing_pri)
        self.attention_capsules = ConvCapsuleLayer(kernel_size=1,num_capsule=num_att_cap,num_atom=52,strides=1,
                                                   padding=0,routings=num_att_routing,update_routing=True)
        self.Length = Length(last_in_ch)

    def forward(self, x):
        output = self.primary_capsules(x)
        output = self.attention_capsules(output)
        norm = self.Length(output)
#        print('norm: ',norm.shape)
        return norm
