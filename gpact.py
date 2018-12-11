import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GPact(nn.Module):

    """ Gaussian Process Activation """
    
    def __init__(self, input_dim, output_dim, n_features=150,
                 uncertain_theta=True, uncertain_lambda=True):
        super(GPact, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_features = n_features
        self.act_set = {'sigmoid', 'tanh', 'relu', 'cos', 'sin'}
        self.uncertain_theta = uncertain_theta
        self.uncertain_lambda = uncertain_lambda
        self.theta_mean = nn.Parameter(torch.rand(input_dim, n_features))
        if(self.uncertain_theta):
            self.theta_lgstd = nn.Parameter(torch.rand(input_dim, n_features))
        self.lambda_mean = nn.Parameter(torch.rand(n_features, output_dim))
        if(self.uncertain_lambda):
            self.lambda_lgstd = nn.Parameter(torch.rand(n_features, output_dim))
        self.init_parameters()
        
    def init_parameters(self):
        stdv = 1. / math.sqrt(self.n_features)
        self.theta_mean.data.uniform_(-stdv, stdv)
        if(self.uncertain_theta):
            self.theta_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))
        stdv = 1. / math.sqrt(self.output_dim)
        self.lambda_mean.data.uniform_(-stdv, stdv)
        if(self.uncertain_lambda):
            self.lambda_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))

    def forward(self, input):
        if(self.training and self.uncertain_theta):
            theta_std = torch.exp(self.theta_lgstd)
            epsilon = theta_std.new_zeros(*theta_std.size()).normal_()
            theta = self.theta_mean+epsilon*theta_std
        else:
            theta = self.theta_mean
        basis = input.matmul(theta)
        concat_basis = [basis]
        for act in self.act_set:
            if(act == 'sin' or act == 'cos'):
                concat_basis.append(getattr(torch, act)(basis))
            else:
                concat_basis.append(getattr(F, act)(basis))
        basis = torch.sum(torch.stack(concat_basis), 0)/math.sqrt(self.n_features)
        if(self.training and self.uncertain_lambda):
            lambda_std = torch.exp(self.lambda_lgstd)
            epsilon = lambda_std.new_zeros(*lambda_std.size()).normal_()
            lambd = self.lambda_mean+epsilon*lambda_std
        else:
            lambd = self.lambda_mean
        return basis.matmul(lambd)
        
    def kl_divergence(self):
        device = next(self.parameters()).device
        self.frequency_mean_prior = self.frequency_mean_prior.to(device)
        self.frequency_lgstd_prior = self.frequency_lgstd_prior.to(device)
        frequency_var = torch.exp(2*self.frequency_lgstd)
        frequency_var_prior = torch.exp(2*self.frequency_lgstd_prior)
        mean_square = (self.frequency_mean-self.frequency_mean_prior)**2./frequency_var_prior
        std_square = frequency_var/frequency_var_prior
        log_std_square = 2*(self.frequency_lgstd_prior-self.frequency_lgstd)/self.frequency_mean.size(1)
        return torch.mean(mean_square+std_square-log_std_square-1)/2.
    
    def reset_prior(self):
        self.frequency_mean_prior =\
            self.frequency_mean.new_zeros([self.input_dim, self.n_features])
        self.frequency_lgstd_prior =\
            self.frequency_lgstd.new_zeros([self.input_dim, self.n_features])
        if(self.update_prior):
            self.frequency_mean_prior.data = self.frequency_mean.data.clone()
            self.frequency_lgstd_prior.data = self.frequency_lgstd.data.clone()

    def __repr__(self):
        return self.__class__.__name__\
            + '(act_set=' + '+'.join(self.act_set)\
            + ', n_features=' + str(self.n_features)\
            + ', uncertain_theta=' + str(self.uncertain_theta)\
            + ', uncertain_lambda=' + str(self.uncertain_lambda) + ')'