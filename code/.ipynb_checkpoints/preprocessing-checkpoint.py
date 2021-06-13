# coding: utf-8

import os
import sys
import json
import numpy as np
import torch
import scipy.sparse

from nltk.tokenize import TweetTokenizer
from allennlp.modules.elmo import Elmo, batch_to_ids
# from allennlp.commands.elmo import ElmoEmbedder

from hyperpara import *


# Initialization for Tokenizer and Elmo Embedder
tokenize = TweetTokenizer().tokenize

# Setting for Elmo Embedder - CHANGE THE PATH
options_file = '/afs/inf.ed.ac.uk/user/s20/s2041332/mlp_project/src/elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file = '/afs/inf.ed.ac.uk/user/s20/s2041332/mlp_project/src/elmo_2x4096_512_2048cnn_2xhighway_weights'

ee = Elmo(options_file, weight_file, 3, dropout=0)


def check(s, wi, c):
    return sum([s[wi + j].lower() == c_ for j, c_ in  enumerate(c) if wi + j < len(s)]) == len(c)

def ind(si, wi, ci, c):
    return [[si, wi  + i, ci] for i in range(len(c))]

def next_batch(data_mb):
    
    for d in data_mb:

        d['query'] = tokenize(d['query'])

        d['candidates_orig'] = list(d['candidates'])
        
        d['candidates'] = [tokenize(c) for c in d['candidates']]

        d['supports'] = [tokenize(s) for s in d['supports']]

        mask = [[ind(si, wi, ci, c) for wi, w in enumerate(s) for ci, c in enumerate(d['candidates']) 
                              if check(s, wi, c)] for si, s in enumerate(d['supports'])]

        nodes_id_name = []
        c = 0
        for e in [[[x[-1] for x in c][0] for c in s] for s in mask]:
            u = []
            for f in e:
                u.append((c, f))
                c +=1

            nodes_id_name.append(u)

        d['nodes_candidates_id'] = [[x[-1] for x in f][0] for e in mask for f in e]

        edges_in, edges_out = [], []
        for e0 in nodes_id_name:
            for f0, w0 in e0:
                for f1, w1 in e0:
                    if f0 != f1:
                        edges_in.append((f0, f1))

                for e1 in nodes_id_name:
                    for f1, w1 in e1:
                        if e0 !=e1 and w0 == w1:
                            edges_out.append((f0, f1))

        d['edges_in'] = edges_in
        d['edges_out'] = edges_out

        mask_ = [[x[:-1] for x in f] for e in mask for f in e]
        
        # Note: the output shape of ELMo:
        # AllenNLP 0.9 (original paper): ee.batch_to_embeddings: (batch_size, 3, num_timesteps, 1024)
        # AllenNLP 2.0 (current version): ee(supports_ids)['elmo_representations']: [(batch_size, timesteps, embedding_dim), (batch_size, timesteps, embedding_dim), (batch_size, timesteps, embedding_dim)]
        
#         print(len(np.array(d['supports']))) # num_sentence * len_sentence
        supports_ids = batch_to_ids(d['supports']) # padding operation
#         print(supports_ids.shape) # (8, 147, 50) - (batchsize, max sentence length, max word length)
        candidates = ee(supports_ids)['elmo_representations'] # [(batch_size, timesteps, embedding_dim) * 3]
        candidates = torch.stack(candidates) # (3, batch_size, timesteps, embedding_dim)
        candidates = candidates.data.cpu().numpy().transpose((1,0,2,3)) # align with the 0.9 allenNLP
        
        d['nodes_elmo'] = [(candidates.transpose((0, 2, 1, 3))[np.array(m).T.tolist()]).astype(np.float16) 
                           for m in mask_]

        
        
        query_ids = batch_to_ids(d['query']) # padding operation
        query = ee(query_ids)['elmo_representations']
        query = torch.stack(query)
        query = query.data.cpu().numpy().transpose((1,0,2,3))
        
        d['query_elmo'] = (query.transpose((0, 2, 1, 3))).astype(np.float16)[0]

        
    id_mb = [d['id'] for d in data_mb]
    
    candidates_mb = [d['candidates_orig'] for d in data_mb]

    filt = lambda c: np.array([c[:,0].mean(0), c[-1,1], c[0,2]])

    nodes_mb = np.array([np.pad(np.array([filt(c) for c in d['nodes_elmo']]),
                    ((0, max_nodes - len(d['nodes_candidates_id'])), (0, 0), (0, 0)),
                    mode='constant') 
                 for d in data_mb])

    nodes_length_mb = np.stack([len(d['nodes_candidates_id']) for d in data_mb] , 0)
    
    query_mb = np.stack([np.pad(d['query_elmo'],
                                ((0, max_query_size - d['query_elmo'].shape[0]), (0, 0), (0, 0)),
                                mode='constant') 
                         for d in data_mb], 0)

    query_length_mb = np.stack([d['query_elmo'].shape[0] for d in data_mb], 0)

    adj_mb = []
    for d in data_mb:

        adj_ = []
            
        if len(d['edges_in']) == 0:
            adj_.append(np.zeros((max_nodes, max_nodes)))
        else:
            adj = scipy.sparse.coo_matrix((np.ones(len(d['edges_in'])), np.array(d['edges_in']).T),
                shape=(max_nodes, max_nodes)).toarray()

            adj_.append(adj)

        if len(d['edges_out']) == 0:
            adj_.append(np.zeros((max_nodes, max_nodes)))
        else:
            adj = scipy.sparse.coo_matrix((np.ones(len(d['edges_out'])), np.array(d['edges_out']).T),
                shape=(max_nodes, max_nodes)).toarray()   

            adj_.append(adj)

        adj = np.pad(np.ones((len(d['nodes_candidates_id']), len(d['nodes_candidates_id']))), 
                        ((0, max_nodes - len(d['nodes_candidates_id'])),
                        (0, max_nodes - len(d['nodes_candidates_id']))), mode='constant') \
            - adj_[0] - adj_[1] - np.pad(np.eye(len(d['nodes_candidates_id'])),
                                            ((0, max_nodes - len(d['nodes_candidates_id'])),
                                            (0, max_nodes - len(d['nodes_candidates_id']))), mode='constant')

        adj_.append(adj)

        adj = np.stack(adj_, 0)

        d_ = adj.sum(-1)
        d_[np.nonzero(d_)] **=  -1
        adj = adj * np.expand_dims(d_, -1)

        adj_mb.append(adj)

    adj_mb = np.array(adj_mb)

    bmask_mb = np.array([np.pad(np.array([i == np.array(d['nodes_candidates_id']) 
                                for i in range(len(d['candidates']))]),
                               ((0, max_candidates - len(d['candidates'])),
                                (0, max_nodes - len(d['nodes_candidates_id']))), mode='constant') 
                        for d in data_mb])

    return {'id_mb': id_mb, 'nodes_mb': nodes_mb, 'nodes_length_mb': nodes_length_mb, 
            'query_mb': query_mb, 'query_length_mb': query_length_mb, 'bmask_mb': bmask_mb,
            'adj_mb': adj_mb, 'candidates_mb': candidates_mb}