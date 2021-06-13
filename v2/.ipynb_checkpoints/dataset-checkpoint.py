# coding: utf-8

import scipy
import json
import re
import allennlp
from allennlp.predictors.predictor import Predictor


from spacy.lang.en import English
import numpy as np
# import tensorflow as tf
import os
import sys
import torch 

from hyperpara import *
import dgl
from utils import *
from tqdm import tqdm


nlp = English()

class Dataset(object):
    
    def __init__(self, text_add, graph_add, mode):
        
        print("Start initialize dataset...")
        
        self.text_set = []
        self.graph_adj_set =[]
        self.graph_h_set = []
        self.size = 0
        
        # Two sets that required in model
        self.yy_set = [] # ground truth label
        self.query_set = []
        self.test = False
        
        # Load text dataset
        with open(text_add, 'r') as f:
            text_set = json.load(f)
        
        # Load graph dataset
        graph_set, _ = dgl.load_graphs(graph_add)
        
        # Start parallel checking for training samples

        for i, (d, g) in enumerate(tqdm(zip(text_set, graph_set))):
#             print("Graph g nodes:", len(g), "; edges:", g.num_edges())
            # if nodes exceed max nodes
            d['query'] = [str(w) for w in nlp.tokenizer(d['query'])]
            if (len(g) <= max_nodes) and (len(d['query']) <= max_query_size) and (len(d['candidates']) <= max_candidates):
                self.size+=1
                # Add TEXT Sample
                self.text_set.append(d)
                self.yy_set.append(d['candidates'].index(d["answer"]))
                self.query_set.append(d['query'])
                
                # 1. Flat the node embedding from (num_nodes, 3, 1024) to (num_nodes, 3072)
                g.ndata['n_embed'] = g.ndata['n_embed'].view(g.ndata['n_embed'].shape[0],-1).float()
                
                # 2. Padding the graph and add to graph_h_set
                for j in range(len(g), max_nodes):
                    g.add_nodes(1)
                self.graph_h_set.append(g.ndata['n_embed'].float())
                
                # 3. Add the norm adjacent matrix
                self.graph_adj_set.append(dgl2normAdj(g))
    
        print("Check graph set:",len(self.graph_h_set), self.graph_h_set[0].shape)

        print(mode+" set Loaded! with size at ", self.size)
        
    def get_size(self):
        return self.size
    
    def get_text_set(self):
        return self.text_set[:self.size]
    
    def get_graph_node_embed(self):
        return self.graph_h_set[:self.size]
    
    def get_graph_norm_adj(self):
        return self.graph_adj_set[:self.size]
    
    def get_label(self):
        self.yy_set = torch.tensor(self.yy_set, dtype=torch.long)
        print(self.yy_set)
        if args.use_gpu:
            self.yy_set = self.yy_set.cuda()
        return self.yy_set[:self.size]
    
    def get_query_embed(self):
        
        
        queries = [d['query'] for d in self.text_set]
        
        # # (batch_size, 3, num_timesteps, 1024) -> # (batch_size, num_timesteps, 3, 1024)
        query_elmo, _ = ee.batch_to_embeddings(queries)
        query_elmo = query_elmo.transpose(1,2)
        if args.use_gpu:
            query_elmo = query_elmo.cuda()
        return query_elmo, queries