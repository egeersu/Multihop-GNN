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


nlp = English()

class Dataset(object):
    
    def __init__(self, text_add, graph_add, mode):
        
        self.text_set = []
        self.graph_set =[]
        self.graph_h_set = []
        self.size = 0
        
        # Two sets that required in model
        self.yy_set = [] # ground truth label
        self.embed_query_set = []
        self.test = False
        
        # Load text dataset
        with open(text_add, 'r') as f:
            text_set = json.load(f)
        
        # Load graph dataset
        graph_set, _ = dgl.load_graphs(graph_add)
        
        # Remove the sample that answer not in graph
        # if d['answer_candidates_id'] in d['nodes_candidates_id']]
        
        # Start parallel checking for training samples
        for i, (d, g) in enumerate(zip(text_set, graph_set)):
            print("Graph g nodes:", len(g), "; edges:", g.num_edges())
            # if nodes exceed max nodes
            d['query'] = [str(w) for w in nlp.tokenizer(d['query'])]
            if (len(g) < max_nodes) and (g.num_edges() < max_edges) and (len(d['query']) < max_query_size) and (len(d['candidates']) < max_candidates):
                # Add TEXT Sample
                self.text_set.append(d)
                
                # Add GRAPH Sample
                g = g.int()
                if args.use_gpu:
                    g = g.to('cuda:0')
                # 1. Flat the node embedding from (num_nodes, 3, 1024) to (num_nodes, 3072)
                g.ndata['n_embed'] = g.ndata['n_embed'].view(g.ndata['n_embed'].shape[0],-1).float()

                # 2. Padding the graph
                for j in range(len(g), max_nodes):
                    g.add_nodes(1)
                # Store the node embedding to other variable and delete it in graph for saving memory
                if args.use_gpu:
                    self.graph_h_set.append(g.ndata['n_embed'].float().cuda())
                else:
                    self.graph_h_set.append(g.ndata['n_embed'].float())
#                 g.ndata.pop('n_embed')
                
                self.graph_set.append(g)
    
        print("Check graphh set:",len(self.graph_h_set), self.graph_h_set[0].shape)
        
        if len(self.text_set) == len(self.graph_set):
            self.size = len(self.text_set)
        else:
            print("ERROR: Graph set size not equal with Text set size!")
            
        print(mode+" set Loaded! with size at ", self.size)
        
        
    def demo_test(self, test_num):
        self.test = True
        self.text_set = self.text_set[:test_num]
        self.graph_set = self.graph_set[:test_num]

                
    def get_text_graph_pair_set(self):
        return self.text_set, self.graph_set
    
    def get_graph_node_embed(self):
        return self.graph_h_set
    
    def get_label(self):
        self.yy_set = torch.tensor([d['candidates'].index(d["answer"]) for d in self.text_set])
        if args.use_gpu:
            self.yy_set = self.yy_set.cuda()
        return self.yy_set
    
    def get_query_embed(self):
        
        
        queries = [d['query'] for d in self.text_set]
        
        # # (batch_size, 3, num_timesteps, 1024) -> # (batch_size, num_timesteps, 3, 1024)
        query_elmo, _ = ee.batch_to_embeddings(queries)
        query_elmo = query_elmo.transpose(1,2)
        if args.use_gpu:
            query_elmo = query_elmo.cuda()
        return query_elmo, queries