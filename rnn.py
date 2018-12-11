import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from gpact import GPact

class FastGPLSTM(nn.Module):

    def __init__(self, lstm, hidden_size):
        super(FastGPLSTM, self).__init__()
        self.lstm = lstm
        self.act_set = {'sigmoid', 'tanh', 'relu', 'gsk'}
        self.theta_mean = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.theta_lgstd = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.init_parameters()
    
    def init_parameters(self):
        stdv = 1. / math.sqrt(self.theta_mean.size(1))
        self.theta_mean.data.uniform_(-stdv, stdv)
        self.theta_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))
    
    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return
    
    def kl_divergence(self):
        return torch.sum(self.theta_mean**2.-self.theta_lgstd*2.
                         +torch.exp(self.theta_lgstd*2))/2.

    def forward(self, input, hidden):
        hx, cx = hidden
        if(self.training):
            theta_std = torch.exp(self.theta_lgstd)
            epsilon = theta_std.new_zeros(*theta_std.size()).normal_()
            theta = self.theta_mean+epsilon*theta_std
        else:
            theta = self.theta_mean
        hx_basis = hx.matmul(theta)
        concat_basis = []
        for act in self.act_set:
            if(act == 'gsk'):
                concat_basis.append(torch.cat([torch.sum(torch.stack(basis.cos().chunk(2, -1)), 0), 
                                               torch.sum(torch.stack(basis.sin().chunk(2, -1)), 0)], -1))
            elif(act == 'sin' or act == 'cos'):
                concat_basis.append(getattr(torch, act)(hx_basis))
            else:
                concat_basis.append(getattr(F, act)(hx_basis))
        hx_basis = torch.sum(torch.stack(concat_basis), 0)/math.sqrt(self.theta_mean.size(1)*len(self.act_set))
        return self.lstm.forward(input, (hx_basis, cx))

class GPLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, uncertain_position=1):
        super(GPLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.act_set = {'sigmoid', 'tanh', 'relu', 'gsk'}
        self.uncertain_position = uncertain_position
        self.input_map = nn.Linear(input_size, 4*hidden_size)
        self.hidden_map = nn.Linear(hidden_size, 4*hidden_size)
        self.ingate_act = nn.Sigmoid()
        self.forgetgate_act = nn.Sigmoid()
        self.cellgate_act = nn.Tanh()
        self.outgate_act = nn.Sigmoid()
        self.cell_act = nn.Tanh()
        if(uncertain_position == 0):
            self.theta_mean = nn.Parameter(torch.rand(input_size, input_size))
            self.theta_lgstd = nn.Parameter(torch.rand(input_size, input_size))
        elif(uncertain_position == 1):
            self.theta_mean = nn.Parameter(torch.rand(hidden_size, hidden_size))
            self.theta_lgstd = nn.Parameter(torch.rand(hidden_size, hidden_size))
        elif(uncertain_position >= 2):
            self.theta_mean = nn.Parameter(torch.rand(hidden_size, hidden_size))
            self.theta_lgstd = nn.Parameter(torch.rand(hidden_size, hidden_size))
            self.basis_map = nn.Linear(hidden_size, hidden_size)
        self.init_parameters()
        
    def init_parameters(self):
        if(0 <= self.uncertain_position <= 6):
            stdv = 1. / math.sqrt(self.theta_mean.size(1))
            self.theta_mean.data.uniform_(-stdv, stdv)
            self.theta_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))
    
    def kl_divergence(self):
        # if(0 <= self.uncertain_position <= 6):
        #     return torch.sum(self.theta_mean**2.-self.theta_lgstd*2.
        #                      +torch.exp(self.theta_lgstd*2))/2.
        return 0
    
    def basis_linear(self, x):
        if(self.training):
            theta_std = torch.exp(self.theta_lgstd)
            epsilon = theta_std.new_zeros(*theta_std.size()).normal_()
            theta = self.theta_mean+epsilon*theta_std
        else:
            theta = self.theta_mean
        basis = x.matmul(theta)
        concat_basis = []
        for act in self.act_set:
            if(act == 'gsk'):
                concat_basis.append(torch.cat([torch.sum(torch.stack(basis.cos().chunk(2, -1)), 0), 
                                               torch.sum(torch.stack(basis.sin().chunk(2, -1)), 0)], -1))
            elif(act == 'sin' or act == 'cos'):
                concat_basis.append(getattr(torch, act)(basis))
            else:
                concat_basis.append(getattr(F, act)(basis))
        return torch.sum(torch.stack(concat_basis), 0)/math.sqrt(self.theta_mean.size(1)*len(self.act_set))
        
    def forward(self, emb, hidden):
        hx, cx = hidden
        outputs = []
        for i, x in enumerate(emb):
            if(self.uncertain_position == 0):
                x = self.basis_linear(x)
            elif(self.uncertain_position == 1):
                hx = self.basis_linear(hx)
            gates = self.input_map(x)+self.hidden_map(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, -1)
            if(self.uncertain_position == 2):
                ingate = self.basis_linear(ingate)
            else:
                ingate = self.ingate_act(ingate)
            if(self.uncertain_position == 3):
                forgetgate = self.basis_linear(forgetgate)
            else:
                forgetgate = self.forgetgate_act(forgetgate)
            if(self.uncertain_position == 4):
                cellgate = self.basis_linear(cellgate)
            else:
                cellgate = self.cellgate_act(cellgate)
            if(self.uncertain_position == 5):
                outgate = self.basis_linear(outgate)
            else:
                outgate = self.outgate_act(outgate)
            cy = (forgetgate * cx) + (ingate * cellgate)
            if(self.uncertain_position == 6):
                hy = outgate * self.basis_linear(cy)
            else:
                hy = outgate * self.cell_act(cy)
            outputs.append(hy)
            hx, cx = hy, cy
        outputs = torch.cat(outputs, 0)
        return outputs, (hx, cx)
        

class BayesLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, uncertain_position=1):
        super(BayesLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.uncertain_position = uncertain_position
        self.ingate_act = nn.Sequential(*[nn.Linear(input_size+hidden_size, hidden_size), nn.Sigmoid()])
        self.forgetgate_act = nn.Sequential(*[nn.Linear(input_size+hidden_size, hidden_size), nn.Sigmoid()])
        self.cellgate_act = nn.Sequential(*[nn.Linear(input_size+hidden_size, hidden_size), nn.Tanh()])
        self.outgate_act = nn.Sequential(*[nn.Linear(input_size+hidden_size, hidden_size), nn.Sigmoid()])
        self.cell_act = nn.Tanh()
        self.theta_lgstd = nn.Parameter(torch.rand(input_size+hidden_size, hidden_size))
        self.init_parameters()
        
    def init_parameters(self):
        stdv = 1. / math.sqrt(self.theta_lgstd.size(1))
        self.theta_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))
    
    def kl_divergence(self):
        if(1 <= self.uncertain_position <= 4):
            if(self.uncertain_position == 1):
                theta_mean = self.ingate_act[0].weight.t()
            elif(self.uncertain_position == 2):
                theta_mean = self.forgetgate_act[0].weight.t()
            elif(self.uncertain_position == 3):
                theta_mean = self.cellgate_act[0].weight.t()
            elif(self.uncertain_position == 4):
                theta_mean = self.outgate_act[0].weight.t()
            return torch.sum(theta_mean**2.-self.theta_lgstd*2.
                             +torch.exp(self.theta_lgstd*2))/2.
        return 0
    
    def bayes_linear(self, act, x):
        if(self.training):
            theta_std = torch.exp(self.theta_lgstd)
            epsilon = theta_std.new_zeros(*theta_std.size()).normal_()
            theta = act[0].weight.t()+epsilon*theta_std
            return act[1](F.linear(x, theta.t(), act[0].bias))
        return act(x)
        
    def forward(self, emb, hidden):
        hx, cx = hidden
        outputs = []
        for i, x in enumerate(emb):
            x_hx = torch.cat([x, hx[0]], -1)
            if(self.uncertain_position == 1):
                ingate = self.bayes_linear(self.ingate_act, x_hx)
            else:
                ingate = self.ingate_act(x_hx)
            if(self.uncertain_position == 2):
                forgetgate = self.bayes_linear(self.forgetgate_act, x_hx)
            else:
                forgetgate = self.forgetgate_act(x_hx)
            if(self.uncertain_position == 3):
                cellgate = self.bayes_linear(self.cellgate_act, x_hx)
            else:
                cellgate = self.cellgate_act(x_hx)
            if(self.uncertain_position == 4):
                outgate = self.bayes_linear(self.outgate_act, x_hx)
            else:
                outgate = self.outgate_act(x_hx)
            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * self.cell_act(cy)
            outputs.append(hy)
            hx, cx = hy, cy
        outputs = torch.stack(outputs, 0)
        return outputs, (hx, cx)