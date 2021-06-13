import torch
import torch.nn as nn
from dgl.nn.pytorch import RelGraphConv
from allennlp.commands.elmo import ElmoEmbedder

from hyperpara import *

import torch.nn.functional as F
import dgl
from functools import partial

import dgl

class RGCN(nn.Module):
    def __init__(self, num_nodes, gnn_h_dim, out_dim, num_rels,
                 num_gcn_hidden_layers, dropout=0,
                 use_self_loop=False, use_cuda=True):
        super(RGCN, self).__init__()
        self.num_nodes = num_nodes
        self.gnn_h_dim = gnn_h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_gcn_hidden_layers = num_gcn_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

    def build_model(self):
        
        # for query
        self.elmo_embedder = self.build_elmo_embedder()
        
        self.gcn_layers = nn.ModuleList()
        
        # 1-layer FFN 256 dim for ElMo embedding
        self.elmo_fnn = nn.Linear(1024 * 3, 256)
        
        # 2layer bi-LSTM for query embedding
        
        # convatenation of q and node embeddings 256+256 = 512
        
        # h2h 512 to 512
        self.shared_gcn = RGCNLayer(self.gnn_h_dim, self.gnn_h_dim, self.num_rels)
        
        # concatenation h_G with q: 512+128 = 768
        
        # 3 layers FF [256, 128, 1]
        
        # Global representation layer
        self.output_fnn = nn.Linear(self.gnn_h_dim, self.out_dim)

    def build_elmo_embedder(self):
        # Setting for Elmo Embedder - CHANGE THE PATH
        options_file = args.project_address+'mlp_project/src/elmo_2x4096_512_2048cnn_2xhighway_options.json'
        weight_file = args.project_address+'mlp_project/src/elmo_2x4096_512_2048cnn_2xhighway_weights'
        
        return ElmoEmbedder(
                  options_file=options_file,
                  weight_file=weight_file)



    def forward(self, h, b_norm_adj):
        
        # embedding the query
        # self.elmo_embedder
        
        # 1-layer FFN 256 dim for ElMo embedding
        h = self.elmo_fnn(h)
        h = F.relu(h, inplace=True)
        
        # 2layer bi-LSTM for query embedding
        # Pending
        
        # convatenation of q and node embeddings 256+256 = 512
        if self.use_cuda:
            q = torch.zeros(h.shape).cuda()
        else:
            q = torch.zeros(h.shape)
        h = torch.cat((h, q), dim=-1)
#         print("h.shape:", h.shape) # 512
        
#         print("h before gcn shape is :", h.shape) # bsx500x512
#         print("Shared gcn check:", self.num_gcn_hidden_layers)
#         print("Before reshape to batch:", len(g), h.shape, r.shape, norm.shape)
        
        bs = h.shape[0]
        
        # Reshape to batch
#         h = h.view(-1, self.gnn_h_dim)
        
        for i in range(self.num_gcn_hidden_layers):
            h = self.shared_gcn(h, b_norm_adj)
        
        
#         h = h.view(bs, -1, self.gnn_h_dim)
#         print("Reshape back to batch:", h.shape)
        
        
        # Average globall pool
#         print("h after gcn shape is :", h.shape) # bsx500x256
        h = torch.mean(h, dim = 1)
#         print(h.shape)
#         print(self.output_fnn)
        out = self.output_fnn(h)
        out = F.softmax(out, dim=-1)
        return out
    

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer
        
        self.rel_weights = nn.ModuleList()
        for _ in range(self.num_rels):
            rel_W = nn.Linear(self.in_feat, self.out_feat)
            self.rel_weights.append(rel_W)

        
        self.weight_hid = nn.Linear(self.in_feat, self.out_feat)
        self.weight_gate = nn.Linear(2 * self.out_feat, self.out_feat)
        
    def forward(self, h, norm_adj):
        
        # 1. message aggregation
        # norm_adj: [batch_size, num_realtion, num_nodes, num_nodes]
        # h: [batch_size, num_nodes, in_dim]
        
        # copy along relational dimension
        h_i = torch.stack([W_r(h) for W_r in self.rel_weights], dim=1) # (bs, rels, num_nodes, hdim)
#         print(h_i.shape)
        # msg: [batch_size * num_relation, num_nodes, in_dim]
        
#         print("before matmul:",norm_adj.shape, h_i.shape)
        msg = torch.matmul(norm_adj, h_i).sum(1) # bs, num_nodes, out_dim
        
#         print("After and reduce sum at dim 1:",msg.shape)
        
        update = msg + self.weight_hid(h) # bs, num_nodes, out_dim
        
        gate = self.weight_gate(torch.cat((update, h), -1))
        gate = F.sigmoid(gate)
        
        h = gate * F.tanh(update) + (1 - gate) * h
        
        return h