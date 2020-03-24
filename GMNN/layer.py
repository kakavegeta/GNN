import torch
import math
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class GConv(nn.Module):
    def __init__(self, opt, adj):
        super(GConv, self).__init__()
        self.opt = opt
        self.in_size = opt['in']
        self.out_size = opt['out']
        self.adj = adj
        self.weight = Parameter(torch.Tensor(self.in_size, self.out_size)) 
    
    def forward(self, x):
        m = torch.mm(x, self.weight)
        m = torch.sparse.mm(m, self.adj)
        return m

    def reset(self):
        stdv = 1.0 / math.sqrt(self.out_size)
        self.weight.data.uniform_(-stdv, stdv)
