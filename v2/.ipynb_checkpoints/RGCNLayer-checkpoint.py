import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial

class MyRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels):
        super(MyRGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.bias = True
        
        self.rel_weights = []
#         self.rel_weights = nn.ModuleList()
        for _ in range(self.num_rels):
            rel_W = nn.Linear(self.in_feat, self.out_feat)
            self.rel_weights.append(rel_W)
        
        self.weight_hid = nn.Linear(self.in_feat, self.out_feat)
        self.weight_gate = nn.Linear(self.in_feat, self.out_feat)
        
            
    def forward(self, h, norm_adj):
        
        # 1. message aggregation
        # norm_adj: [batch_size, num_realtion, num_nodes, num_nodes]
        # h: [batch_size, num_nodes, in_dim]
        
        batch_size = h.shape[0]
        num_nodes = h.shape[1]
        
        # copy along relational dimension
#         h_i = h.expand(batch_size, self.num_rels, -1, -1)
        h_i = torch.stack([W_r(h) for W_r in self.rel_weights], dim=1) # (bs, rels, num_nodes, hdim)
        print(h_i.shape)
        # msg: [batch_size * num_relation, num_nodes, in_dim]
        msg = torch.matmul(norm_adj, h_i).sum(1)
        
        update = msg + self.weight_hid(h)
        
        gated_val = self.weight_gate(torch.cat((update, h), -1))
        gated_val = F.sigmoid(gated_val)
        
        h = gated_val * msg + (1 - gated_val) * h
        
        
#         print("Output for each relation weight:",[W_r(msg).shape for W_r in self.rel_weights]) # batch, 1, 300, 512
#         msg = torch.stack([W_r(msg) for W_r in self.rel_weights]， dim=1) # batch, num_nodes, out_dim
#         print('msg shape before sum=',msg.shape)
#         msg = msg.sum(1) # sum over relation dimension
        
#         # h: [batch_size, num_relation, num_nodes, in_dim]
#         # W_0: [batch_size, num_relation, in_dim, out_dim]
#         h = self.weight_hid(h)
        
#         # now aggregate H: [batch_size, num_nodes, out_dim]
#         h = h + msg
#         h = F.relu(h)
        
        return h