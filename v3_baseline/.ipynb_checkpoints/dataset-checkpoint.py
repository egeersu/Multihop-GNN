# coding: utf-8

import scipy
import json
import re
import allennlp
from allennlp.predictors.predictor import Predictor
from allennlp.commands.elmo import ElmoEmbedder
from torch.nn.utils.rnn import pad_sequence

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

# Setting for Elmo Embedder - CHANGE THE PATH
options_file = args.project_address+'mlp_project/src/elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file = args.project_address+'mlp_project/src/elmo_2x4096_512_2048cnn_2xhighway_weights'
        
ee = ElmoEmbedder(options_file=options_file,weight_file=weight_file)

class Dataset(object):
    
    def __init__(self, text_add, graph_add, mode):
        
        print("Start initialize dataset...")
        
        self.text_set = []
        self.graph_adj_set =[]
        self.graph_h_set = []
        self.node_num_set = [] # (N,) records how many nodes are in graph, to mask out unrelated nodes
        self.answer_mask_set = []
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
        
        # initializse query embedder
        self.elmo_embedder = ee
        
        # Start parallel checking for training samples

        for i, (d, g) in enumerate(tqdm(zip(text_set, graph_set))):
#             print("Graph g nodes:", len(g), "; edges:", g.num_edges())
            # if nodes exceed max nodes
            d['query'] = [str(w) for w in nlp.tokenizer(d['query'])]
            
            if (len(g) <= max_nodes) and (len(d['query']) <= max_query_size) and (len(d['candidates']) <= max_candidates) and (d['candidates'].index(d["answer"]) in g.ndata['e_id']):
                self.size+=1
                # Add TEXT Sample
                self.text_set.append(d)
                self.yy_set.append(d['candidates'].index(d["answer"]))
                
                self.query_set.append(d['query'])
                
                # 1. Flat the node embedding from (num_nodes, 3, 1024) to (num_nodes, 3072)
                g.ndata['n_embed'] = g.ndata['n_embed'].view(g.ndata['n_embed'].shape[0],-1).float()
                
                # Add g's node number
                self.node_num_set.append(g.number_of_nodes())
                
                # 2. Padding the graph and add to graph_h_set
                for j in range(len(g), max_nodes):
                    g.add_nodes(1,data={'e_id':torch.tensor([-1],dtype=torch.int)})
                self.graph_h_set.append(g.ndata['n_embed'].float())
                
                # 3. Add the norm adjacent matrix
                self.graph_adj_set.append(dgl2normAdj(g))
                
                # 4. Add answer_mask
                nodes_candidates_id = g.ndata['e_id']
                # 1 70 500
                answer_mask = torch.tensor([np.pad(np.array([i == np.array(nodes_candidates_id)
                                for i in range(len(d['candidates']))]),
                               ((0, max_candidates - len(d['candidates'])),
                                (0, max_nodes - g.number_of_nodes())), mode='constant')]).squeeze()
                answer_mask = ~answer_mask
                self.answer_mask_set.append(answer_mask)
            
    
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
    
    def get_node_num(self):
        return torch.tensor(self.node_num_set[:self.size],dtype=torch.int)
    
    def get_label(self):
        self.yy_set = torch.tensor(self.yy_set, dtype=torch.long)
        return self.yy_set[:self.size]
    
    def get_answer_mask(self):
        return torch.stack(self.answer_mask_set[:self.size],dim=0)
    
    def get_query(self):
        
        
        # # (batch_size, 3, num_timesteps, 1024) -> # (batch_size, num_timesteps, 3, 1024)
        query_elmo, _ = self.elmo_embedder.batch_to_embeddings(self.query_set)
        query_elmo = query_elmo.transpose(1,2) # (batch_size, max_num_timesteps, 3, 1024)
        
        # Padding query's second dim to hyperpara: max_num_timesteps
        
        
        return query_elmo[:self.size], self.query_set[:self.size]