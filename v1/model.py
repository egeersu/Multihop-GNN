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
    def __init__(self, num_nodes, gnn_in_dim, gnn_h_dim, out_dim, num_rels, num_bases,
                 num_gcn_hidden_layers, dropout=0,
                 use_self_loop=False, use_cuda=True):
        super(RGCN, self).__init__()
        self.num_nodes = num_nodes
        self.gnn_in_dim = 1024*3
        self.gnn_h_dim = gnn_h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
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
        self.shared_gcn = self.build_hidden_layer()
            
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
        


    def build_hidden_layer(self):
#         print("Check hidden layer:", self.gnn_h_dim, self.gnn_h_dim, self.num_rels, "basis", self.num_bases)
        return RelGraphConv(self.gnn_h_dim, self.gnn_h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=False,
                dropout=self.dropout,low_mem=False)


    def forward(self, g, h, r, norm):
        
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
        h = h.view(-1, self.gnn_h_dim)
        r = r.view(-1)
        if norm is not None:
            norm = norm.view(-1, 1)
        
#         print("After reshape to batch:", len(g), h.shape, r.shape, norm.shape)
        
        for i in range(self.num_gcn_hidden_layers):
            h = self.shared_gcn(g, h, r, norm)
        
        
        h = h.view(bs, -1, self.gnn_h_dim)
#         print("Reshape back to batch:", h.shape)
        
        
        # Average globall pool
#         print("h after gcn shape is :", h.shape) # bsx500x256
        h = torch.mean(h, dim = 1)
#         print(h.shape)
#         print(self.output_fnn)
        out = self.output_fnn(h)
        out = F.softmax(out, dim=-1)
        return out

def initializer(emb):
    emb.uniform_(-1.0, 1.0)
    return emb

class RelGraphEmbedLayer(nn.Module):
    r"""Embedding layer for featureless heterograph.
    Parameters
    ----------
    dev_id : int
        Device to run the layer.
    num_nodes : int
        Number of nodes.
    node_tides : tensor
        Storing the node type id for each node starting from 0
    num_of_ntype : int
        Number of node types
    input_size : list of int
        A list of input feature size for each node type. If None, we then
        treat certain input feature as an one-hot encoding feature.
    embed_size : int
        Output embed size
    dgl_sparse : bool, optional
        If true, use dgl.nn.NodeEmbedding otherwise use torch.nn.Embedding
    """
    def __init__(self,
                 dev_id,
                 num_nodes,
                 num_of_ntype,
                 input_size,
                 embed_size,
                 dgl_sparse=False):
        super(RelGraphEmbedLayer, self).__init__()
        self.dev_id = torch.device(dev_id if dev_id >= 0 else 'cpu')
        self.embed_size = embed_size
        self.num_nodes = num_nodes
        self.dgl_sparse = dgl_sparse

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        self.node_embeds = {} if dgl_sparse else nn.ModuleDict()
        self.num_of_ntype = num_of_ntype

        for ntype in range(num_of_ntype):
            if isinstance(input_size[ntype], int):
                if dgl_sparse:
                    self.node_embeds[str(ntype)] = dgl.nn.NodeEmbedding(input_size[ntype], embed_size, name=str(ntype),
                        init_func=initializer)
                else:
                    sparse_emb = torch.nn.Embedding(input_size[ntype], embed_size, sparse=True)
                    nn.init.uniform_(sparse_emb.weight, -1.0, 1.0)
                    self.node_embeds[str(ntype)] = sparse_emb
            else:
                input_emb_size = input_size[ntype].shape[1]
                embed = nn.Parameter(torch.Tensor(input_emb_size, self.embed_size))
                nn.init.xavier_uniform_(embed)
                self.embeds[str(ntype)] = embed

    @property
    def dgl_emb(self):
        """
        """
        if self.dgl_sparse:
            embs = [emb for emb in self.node_embeds.values()]
            return embs
        else:
            return []

    def forward(self, node_ids, type_ids, features):
        """Forward computation
        Parameters
        ----------
        node_ids : tensor
            node ids to generate embedding for.
        features : list of features
            list of initial features for nodes belong to different node type.
            If None, the corresponding features is an one-hot encoding feature,
            else use the features directly as input feature and matmul a
            projection matrix.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        tsd_ids = node_ids.to(self.dev_id)
        embeds = torch.empty(node_ids.shape[0], self.embed_size, device=self.dev_id)

        return embeds
