import torch
import torch.nn as nn
from dgl.nn.pytorch import RelGraphConv

from hyperpara import *

import torch.nn.functional as F
import dgl
from functools import partial

import dgl


class RGCN(nn.Module):
    def __init__(self, num_nodes, gnn_h_dim, out_dim, num_rels,
                 num_gcn_hidden_layers, dropout_rate,
                 use_self_loop=False, use_cuda=True):
        super(RGCN, self).__init__()
        self.num_nodes = num_nodes
        self.gnn_h_dim = gnn_h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_gcn_hidden_layers = num_gcn_hidden_layers
        self.dropout_rate = dropout_rate
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda
        self.bias = True
        
        self.query_embed_dim = 1024 * 3
        self.query_lstm_out_dim = 128

        # create rgcn layers
        self.build_model()

    def build_model(self):
        
        # 2layer bi-LSTM for query embedding: 3*1024 -> 128
        self.query_lstm = nn.LSTM(self.query_embed_dim, 128, num_layers=1, bidirectional=True, batch_first=True)
        
        # 1-layer FFN 256 dim for ElMo embedding
        self.elmo_fnn = nn.Linear(1024 * 3, 256)
        
        # 2-layer query-aware mention encoding
        self.q_h_encoding = nn.Linear(512, 512)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # h2h 512 to 512
        self.gcn_layers = nn.ModuleList()
        self.shared_gcn = RGCNLayer(self.gnn_h_dim, self.gnn_h_dim, self.num_rels)
        
        # concatenation h_G with q: 512+256 = 768
        
        
        # 3 layers output FF [256, 128, 1]        
        self.output_fnn1 = nn.Linear(512 + 256, 256)
        self.output_fnn2 = nn.Linear(256, 1)


    def forward(self, h, b_norm_adj, query, node_num, answer_mask):
        
        batch_size = node_num.shape[0]
        
        # initialize 500 nodes index for creating mask (mask out padding nodes)
        if args.use_gpu:
            self.node_mask = torch.arange(max_nodes,dtype=torch.int).unsqueeze(0).cuda()
        else:
            self.node_mask = torch.arange(max_nodes,dtype=torch.int).unsqueeze(0)
        
        # Make node_mask to mask out the padding nodes
        node_num = node_num.unsqueeze(-1) # (bs, 1) 
        self.node_mask = self.node_mask.repeat((batch_size, 1)) < node_num
        
        # flat the embed query 
        query = query.contiguous().view(query.shape[0], query.shape[1], 3 * 1024)
    
        # Query embedidng, 2 layer LSTM
        lstm_q_outputs, (hn, cn) = self.query_lstm(query)
#         lstm_q_outputs, (hn, cn) = self.query_lstm_2(lstm_q_outputs)
        
        # take output state as encoding state
        query_compress = torch.cat((hn[0], hn[-1]), dim=-1)
        query_compress = self.dropout(query_compress)
        
        # 1-layer FFN for nodes ElMo embedding (dimension reduction from 3072 -> 256)
        h = self.elmo_fnn(h)
        h = F.tanh(h)
        h = self.dropout(h)
        
        # prepare for concatenation of q and node embeddings
        query_compress = query_compress.unsqueeze(1).expand(h.shape)
        
        # Concatenation bewtween q and nodes (256+256=512)
        h = torch.cat((h, query_compress), dim=-1) # bs, 500, 512
        h = h * self.node_mask.unsqueeze(-1)
        
        # 2 layer FF [1024 512[]
        h = self.q_h_encoding(h)
        h = F.tanh(h)
        
        for i in range(self.num_gcn_hidden_layers):
            h = self.shared_gcn(h, b_norm_adj, self.node_mask)
            h = self.dropout(h)
        
        
        # Concatenation with query again # bs, 500, 768
        h = torch.cat((h, query_compress), dim=-1) * self.node_mask.unsqueeze(-1)
                
        # Graph-level Attention Layer ()
        
        # 2-layer output layers
        h = self.output_fnn1(h) # bs, 500, 256
        h = F.tanh(h)
        h = self.output_fnn2(h) # bs, 500, 128
        
        # Apply answer mask: mask out the information that not belong to any classes
        h = h.view(batch_size, 1, max_nodes).masked_fill(answer_mask, float("-inf"))
        
        # Max reduce over 500 nodes, see Eqt 1. for detail
        out = h.max(dim = -1).values
        
        out = F.log_softmax(out, dim=-1)
#         print(out)
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
        
    def forward(self, h, norm_adj, node_mask):
        
        # 1. message aggregation
        # copy along relational dimension
        h_i = torch.stack([W_r(h) for W_r in self.rel_weights], dim=1) # (bs, rels, num_nodes, hdim)
        
        # Apply node mask: node_mask(bs, 500), h_i: (bs, num_rels, 500, dim)
        h_i = h_i * node_mask.unsqueeze(-1).unsqueeze(1)
        
        msg = torch.matmul(norm_adj, h_i).sum(1) # bs, num_nodes, out_dim
        
        update = msg + self.weight_hid(h) * node_mask.unsqueeze(-1) # bs, num_nodes, out_dim
        
        # Gate mechanism
        gate = self.weight_gate(torch.cat((update, h), -1))
        gate = F.sigmoid(gate)
        gate = gate * node_mask.unsqueeze(-1)
        
        # 2. Update
        h = gate * F.tanh(update) + (1 - gate) * h
        
        return h