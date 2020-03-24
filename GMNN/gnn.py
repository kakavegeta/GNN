import torch
import math
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

class GNNq(nn.Module):
    def __init__(self, opt, adj):
        super(GNNq, self).__init__()
        self.opt = opt
        self.adj = adj

    def forward(self, x):
        pass

    def reset(self):
        pass

class GNNp(nn.Module):
    def __init__(self, opt, adj):
        super(GNNp, opt, adj).__init__()
        self.opt = opt
        self.adj = adj
    
    def forward(self, x):
        pass

    def reset(self):
        pass