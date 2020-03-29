import torch
import math
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from layer import GConv

class GNNq(nn.Module):
    def __init__(self, opt, adj):
        super(GNNq, self).__init__()
        self.opt = opt
        self.adj = adj

        opt1 = {'in': opt['feature_num'], 'out': opt['hidden_dim']}
        opt2 = {'in': opt['hidden_dim'], 'out': opt['hidden_dim']}

        self.m1 = GConv(opt1, adj)
        self.m2 = GConv(opt2, adj)
        self.m3 = nn.Linear(opt['hidden_dim'], opt['class_num'])

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()


    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training = self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['droupout'], training=self.training)
        x = self.m2(x)
        x = F.relu(x)
        x = self.m3(x)
        return x
    
class GNNp(nn.Module):
    def __init__(self, opt, adj):
        super(GNNp, opt, adj).__init__()
        self.opt = opt
        self.adj = adj

        opt1 = {'in': opt['class_num'], 'out': opt['hidden_dim']}
        opt2 = {'in': opt['hidden_dim'], 'out': opt['hidden_dim']}

        self.m1 = GConv(opt1, adj)
        self.m2 = GConv(opt2, adj)
        self.m3 = nn.Linear(opt['hidden_dim'], opt['class_num'])

        if opt['cuda']:
            self.cuda()
    
    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training = self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['droupout'], training=self.training)
        x = self.m2(x)
        x = F.relu(x)
        x = self.m3(x)
        return x


    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()