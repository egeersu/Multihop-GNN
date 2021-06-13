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



# Setting for Elmo Embedder - CHANGE THE PATH
# options_file = args.project_address+'mlp_project/src/elmo_2x4096_512_2048cnn_2xhighway_options.json'
# weight_file = args.project_address+'mlp_project/src/elmo_2x4096_512_2048cnn_2xhighway_weights'

options_file = '/home/watsonzhouanda/multihop/src/elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file = '/home/watsonzhouanda/multihop/src/elmo_2x4096_512_2048cnn_2xhighway_weights'

ee = ElmoEmbedder(options_file=options_file,weight_file=weight_file)
nlp = English()

class Dataset(object):
    
    def __init__(self, text_add, graph_add, mode):
        
        print("Start initialize dataset...")
        
        self.text_set = []
        self.graph_set = []
        self.node_num_set = []
#         self.graph_adj_set =[]
#         self.graph_h_set = []
#         self.node_num_set = [] # (N,) records how many nodes are in graph, to mask out unrelated nodes
#         self.answer_mask_set = []
        self.size = 0
        
        
        # Two sets that required in model
#         self.yy_set = [] # ground truth label
#         self.query_set = []

        # Load text dataset
        with open(text_add, 'r') as f:
            text_set = json.load(f)
        
#         # specify the list of sample - tmp
#         if mode == "Training":
#             text_set = text_set[20000:29001]
        
        
        # Load graph dataset
        graph_set, _ = dgl.load_graphs(graph_add)
        print("Set size check:", len(graph_set), len(text_set))
        # initializse query embedder
        self.elmo_embedder = ee
        
        print("Sample from", text_set[0]['id'], " to ", text_set[-1]['id'], flush=True)
        
        for i, (d, g) in enumerate(tqdm(zip(text_set, graph_set))):
            d['query'] = [str(w) for w in nlp.tokenizer(d['query'])]
#             print((len(g) > 0))
#             print((len(g) <= max_nodes))
#             print((len(d['query']) <= max_query_size))
#             print((len(d['candidates']) <= max_candidates))
#             print((d['candidates'].index(d["answer"]) in g.ndata['e_id']))
            if (len(g) > 0) and (len(g) <= max_nodes) and (len(d['query']) <= max_query_size) and (len(d['candidates']) <= max_candidates) and (d['candidates'].index(d["answer"]) in g.ndata['e_id']):
                self.size+=1
                # Add TEXT Sample
                self.text_set.append(d)
                
#                 self.yy_set.append(d['candidates'].index(d["answer"]))
#                 self.query_set.append(d['query'])
                
                # 1. Flat the node embedding from (num_nodes, 3, 1024) to (num_nodes, 3072)
                g.ndata['n_embed'] = g.ndata['n_embed'].view(g.ndata['n_embed'].shape[0],-1).float()
                
                # Add g's node number
                self.node_num_set.append(g.number_of_nodes())
                
                # 2. Padding the graph and add to graph_h_set
#                 for j in range(len(g), max_nodes):
#                     g.add_nodes(1, data={'e_id':torch.tensor([-1],dtype=torch.int)})
                
                self.graph_set.append(g)
                
#                 self.graph_h_set.append(g.ndata['n_embed'].float())
                
#                 # 4. Add answer_mask
#                 nodes_candidates_id = g.ndata['e_id']
#                 # 1 70 500
#                 answer_mask = torch.tensor([np.pad(np.array([i == np.array(nodes_candidates_id)
#                                 for i in range(len(d['candidates']))]),
#                                ((0, max_candidates - len(d['candidates'])),
#                                 (0, max_nodes - g.number_of_nodes())), mode='constant')]).squeeze()
#                 answer_mask = ~answer_mask
#                 self.answer_mask_set.append(answer_mask)
            
    
        print("Check set size:",len(self.graph_set), len(self.text_set), flush=True)

        print(mode+" set Loaded! with size at ", self.size, flush=True)
        
    def get_size(self):
        return self.size
    
    def get_text_set(self):
        return self.text_set[:self.size]
    
    def get_graph_set(self):
        return self.graph_set[:self.size]
    
    def get_batch_graph_node_embed(self, graph_batch):
        batch_graph_h = []
        for g in graph_batch:
            batch_graph_h.append(g.ndata['n_embed'].float())
        return torch.stack(batch_graph_h)
    
    def get_batch_graph_norm_adj(self, graph_batch):
        batch_graph_adj_set = []
        for g in graph_batch:
            batch_graph_adj_set.append(dgl2normAdj(g))
        return torch.stack(batch_graph_adj_set)
    
    def get_batch_node_num(self, batch_start, batch_end): 
        return torch.tensor(self.node_num_set[batch_start:batch_end],dtype=torch.int)
    
    def get_batch_label(self, batch_graph, batch_text):
        batch_label = torch.zeros(size=[batch_size, max_nodes], dtype=torch.long)
        i = 0
        for g, d in zip(batch_graph, batch_text):
            answer_id = d['candidates'].index(d["answer"])
            text_label = torch.zeros(len(g.ndata['loc_start']), dtype=torch.long)
            for index, e in enumerate(g.ndata['e_id']):
                if int(e) == answer_id:
                    text_label[index] = 1
            batch_label[i] = text_label
            i += 1
        return batch_label

    def get_dev_batch_label(self, batch_graph, batch_text):
        batch_label = torch.zeros(size=[len(batch_graph), max_nodes], dtype=torch.long)
        # print(batch_label.size())
        i = 0
        for g, d in zip(batch_graph, batch_text):
            answer_id = d['candidates'].index(d["answer"])
            # text_label = torch.zeros(len(g.ndata['loc_start']), dtype=torch.long)
            text_label = torch.zeros(max_nodes, dtype=torch.long)
            for index, e in enumerate(g.ndata['e_id']):
                if index <= max_nodes-1:
                    if int(e) == answer_id:
                        text_label[index] = 1
            # print(text_label.size())
            batch_label[i] = text_label
            i += 1
        return batch_label
    
    def get_batch_answer_mask(self, batch_graph, batch_text):
        batch_answer_mask = []
        for g, d in zip(batch_graph, batch_text):
            # 4. Add answer_mask
            nodes_candidates_id = g.ndata['e_id']
            # 1 70 500
            answer_mask = torch.tensor([np.pad(np.array([i == np.array(nodes_candidates_id)
                            for i in range(len(d['candidates']))]),
                           ((0, max_candidates - len(d['candidates'])),
                            (0, max_nodes - g.number_of_nodes())), mode='constant')]).squeeze()
            answer_mask = ~answer_mask
            batch_answer_mask.append(answer_mask)
        return torch.stack(batch_answer_mask,dim=0)

    def get_batch_position_start(self, batch_graph):
        batch_position = []
        for g in batch_graph:
            batch_position.append(g.ndata['loc_start'].float())
        return torch.stack(batch_position)

    def get_batch_query(self, text_batch):
        batch_query = []
        for d in text_batch:
            batch_query.append(d['query'])
        
        # # (batch_size, 3, num_timesteps, 1024) -> # (batch_size, num_timesteps, 3, 1024)
        batch_query_elmo, _ = self.elmo_embedder.batch_to_embeddings(batch_query)
        batch_query_elmo = batch_query_elmo.transpose(1,2) # (batch_size, max_num_timesteps, 3, 1024)
        
        return batch_query_elmo, batch_query