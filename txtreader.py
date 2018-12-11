import torch
import os
import torch.utils.data as Data
import numpy as np
from utils import device
from scipy import sparse
import re
import math



'''
traditional approach to read data. Treat the whole corpus as a long sequence and then chop them into minibatch splits
Not USED ANYMORE
'''
class datareader(object):
    def __init__ (self, txtfile, voc, minibatch, chunksize):
        self.minibatch = minibatch
        self.chunksize = chunksize
        ids = self.tokenize(txtfile, voc)
        raw_data = np.array(ids, dtype=np.int32)
        nwords = len(ids)
        self.nbatch = nwords // minibatch
        self.nchunk = (self.nbatch-1) // chunksize

        self.data = np.zeros([minibatch, self.nbatch], dtype=np.int32)
        for i in range(minibatch):
            self.data[i] = raw_data[self.nbatch*i : self.nbatch*(i+1)]

    def tokenize (self, txtfile, voc):
        words = open(txtfile, 'r').read().replace("\n", "<eos>").split()
        words = ['<eos>'] + words
        ids = [voc.word2id(word) for word in words]
        return ids
    
    def dataiter (self):
        for i in range (self.nchunk):
            x = self.data[:, i*self.chunksize:(i+1)*self.chunksize]
            y = self.data[:, i*self.chunksize+1 : (i+1)*self.chunksize+1]
            input = torch.from_numpy(x.astype(np.int64)).transpose(0,1).contiguous().to(device)
            target = torch.from_numpy(y.astype(np.int64)).transpose(0,1).contiguous().to(device)
            yield (input, target)

