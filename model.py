import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import math
from rnn import BayesLSTM, GPLSTM, FastGPLSTM
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop


class RNNLM(nn.Module):
    
    def __init__ (self, model, vocsize, embsize, hiddensize, n_layers,
                  dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0,
                  tie_weights=True, ldropout=0.5, n_experts=5, uncertain='gp', position=1):
        super(RNNLM, self).__init__()
        self.model = model.lower()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(vocsize, embsize)
        self.rnns = []
        for l in range(n_layers):
            if l == 0:
                if uncertain == 'gp':
                    self.rnns.append(GPLSTM(embsize, hiddensize if l != n_layers-1 else embsize, position))
                elif uncertain == 'bayes':
                    self.rnns.append(BayesLSTM(embsize, hiddensize if l != n_layers-1 else embsize, position))
                else:
                    self.rnns.append(torch.nn.LSTM(embsize, hiddensize if l != n_layers-1 else embsize, 1, dropout=0))
            else:
                self.rnns.append(torch.nn.LSTM(hiddensize,
                                               hiddensize if l != n_layers-1 else embsize, 1, dropout=0))
        if wdrop:
            self.rnns = [WeightDrop(rnn, hiddensize if l != n_layers-1 else embsize, ['weight_hh_l0'],
                                    dropout=wdrop) for l, rnn in enumerate(self.rnns) if rnn.__class__.__name__ != "GPLSTM"]
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.prior = nn.Linear(embsize, n_experts, bias=False)
        self.latent = nn.Sequential(nn.Linear(embsize, n_experts*embsize), nn.Tanh())
        self.decoder_bias = nn.Parameter(torch.empty(vocsize))
        if tie_weights:
            # Optionally tie weights as in:
            # "Using the Output Embedding to Improve Language Models"
            # (Press & Wolf 2016)
            # https://arxiv.org/abs/1608.05859
            # and
            # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling"
            # (Inan et al. 2016)
            # https://arxiv.org/abs/1611.01462
            self.decoder_weight = self.encoder.weight
        else:
            self.decoder_weight = nn.Parameter(torch.empty(vocsize, embsize))
        
        self.vocsize = vocsize
        self.embsize = embsize
        self.hiddensize = hiddensize
        self.n_layers = n_layers
        self.tie_weights = tie_weights
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.ldropout = ldropout
        self.dropoutl = ldropout
        self.n_experts = n_experts
        
        self.init_parameters()

    def init_parameters(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder_bias.data.fill_(0)
        if not self.tie_weights:
            self.decoder_weight.data.uniform_(-initrange, initrange)
    
    def embed(self, words, dropout, scale=None):
        masked_embed_weight = self.encoder.weight
        if dropout:
            mask = self.encoder.weight.data.new().resize_((self.encoder.weight.size(0), 1))\
                .bernoulli_(1-dropout).expand_as(self.encoder.weight)/(1-dropout)
            masked_embed_weight = mask*masked_embed_weight
        if scale:
            masked_embed_weight = scale.expand_as(masked_embed_weight)*masked_embed_weight

        padding_idx = self.encoder.padding_idx
        if padding_idx is None:
            padding_idx = -1
        return F.embedding(words, masked_embed_weight,
            padding_idx, self.encoder.max_norm, self.encoder.norm_type,
            self.encoder.scale_grad_by_freq, self.encoder.sparse
        )
    
    def forward(self, input, hidden, return_h=False, return_prob=False):
        batchsize = input.size(1)
        raw_output = self.embed(input, dropout=self.dropoute if self.training else 0)

        raw_output = self.lockdrop(raw_output, self.dropouti)
        
        new_hidden, raw_outputs, outputs = [], [], []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        if self.n_experts > 0:
            latent = self.latent(output)
            latent = self.lockdrop(latent, self.dropoutl)

            logit = F.linear(latent.view(-1, self.embsize), self.decoder_weight, self.decoder_bias)

            prior_logit = self.prior(output).contiguous().view(-1, self.n_experts)
            prior = nn.functional.softmax(prior_logit, -1)

            prob = nn.functional.softmax(logit.view(-1, self.vocsize), -1).view(-1, self.n_experts, self.vocsize)
            prob = (prob*prior.unsqueeze(2).expand_as(prob)).sum(1)
        else:
            logit = F.linear(output.view(-1, self.embsize), self.decoder_weight, self.decoder_bias)
            

        if return_prob:
            model_output = prob
        else:
            log_prob = torch.log(prob.add_(1e-8))
            model_output = log_prob

        model_output = model_output.view(-1, batchsize, self.vocsize)

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(weight.new(1, bsz, self.hiddensize if l != self.n_layers-1 else self.embsize).zero_(),
                 weight.new(1, bsz, self.hiddensize if l != self.n_layers-1 else self.embsize).zero_())
                for l in range(self.n_layers)]