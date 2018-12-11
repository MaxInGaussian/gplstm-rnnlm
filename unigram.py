import torch
import numpy as np
from utils import device

class unigramsampler(object):
    def __init__(self, probs):
        cpu_probs = probs.cpu()
        self.vocsize = len(probs)
        K = len(probs)
        self.prob = [0] * K
        self.J = [0] * K

        smaller = []
        larger  = []
        for idx, prob in enumerate(cpu_probs):
            self.prob[idx] = K*prob
            if self.prob[idx] < 1.0:
                smaller.append(idx)
            else:
                larger.append(idx)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.J[small] = large
            self.prob[large] = self.prob[small] + self.prob[large] - 1.0 

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        self.prob = probs.new(self.prob)
        self.J    = probs.new(self.J).long()

    def draw(self, *size):
        max_value = self.J.size(0)
        kk = self.J.new(*size).random_(0, max_value).long().view(-1)
        prob = self.prob[kk]
        j    = self.J[kk]

        b = torch.bernoulli(prob).long()
        oq = kk.mul(b)
        oj = j.mul(1-b)
        return (oq+oj).view(size)

    def draw_uniform(self, seq_len, minibatch, ncesample):
        '''generate nce sample, and extract the '''
        noise = torch.randint(0, self.vocsize, (seq_len, 1, ncesample), dtype = torch.long)
        noise = noise.expand(seq_len, minibatch, ncesample)
        return noise.to(device).to(device)
