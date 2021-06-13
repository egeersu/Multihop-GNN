# -*- coding: utf-8 -*-
# @Time    : 2021/3/29 14:45
# @Author  : Anda Zhou
# @FileName: model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperpara import *
import math

class Net(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, use_cuda=True):
        super(Net, self).__init__()
        self.n_input = n_input
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.use_cuda = use_cuda
        self.pos_dim = 512
        self.build()

    def build(self):
        self.hidden1 = nn.Linear(self.n_input, self.n_hidden1)
        self.hidden2 = nn.Linear(self.n_hidden1, self.n_hidden2)
        self.predict = nn.Linear(self.n_hidden2, self.n_output)

    def forward(self, node_num, node_position_start):

        batch_size = node_num.shape[0]

        if args.use_gpu:
            self.node_mask = torch.arange(max_nodes, dtype=torch.int).unsqueeze(0).cuda()
        else:
            self.node_mask = torch.arange(max_nodes, dtype=torch.int).unsqueeze(0)   # 1, 500
        # Make node_mask to mask out the padding nodes
        node_num = node_num.unsqueeze(-1)  # (bs, 1)
        self.node_mask = self.node_mask.repeat((batch_size, 1)) < node_num  # bs, 500

        # Position
        pe = torch.zeros(batch_size, max_nodes, self.pos_dim)  # bs, 500, 512
        div_term = torch.exp(torch.arange(0, self.pos_dim, 2).float() * (-math.log(10000.0) / self.pos_dim))  # 256
        if args.use_gpu:
            div_term = div_term.cuda()
        position = node_position_start.unsqueeze(-1)  # node_position_start shape = bs, node_num, 1
        position = position * self.node_mask.unsqueeze(-1)  # bs 500 1
        pe[:, :, 0::2] = torch.sin(position * div_term)  # bs, 500, 512
        pe[:, :, 1::2] = torch.cos(position * div_term)  # bs, 500, 512

        if args.use_gpu:
            pe = pe.cuda()

        out = self.hidden1(pe)
        out = F.sigmoid(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out = self.predict(out)
        return out
