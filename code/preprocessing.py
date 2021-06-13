# coding: utf-8

import scipy
import json
import re
import allennlp
from allennlp.predictors.predictor import Predictor

from allennlp.commands.elmo import ElmoEmbedder
from spacy.lang.en import English
import numpy as np
# import tensorflow as tf
import os
import sys

from hyperpara import *


# Setting for Elmo Embedder - CHANGE THE PATH
options_file = '/afs/inf.ed.ac.uk/user/s20/s2041332/mlp_project/src/elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file = '/afs/inf.ed.ac.uk/user/s20/s2041332/mlp_project/src/elmo_2x4096_512_2048cnn_2xhighway_weights'

# Initialization for each module
nlp = English()
ee = ElmoEmbedder(
                  options_file=options_file,
                  weight_file=weight_file)
predictor = Predictor.from_path('/afs/inf.ed.ac.uk/user/s20/s2041332/mlp_project/src/coref-model-2018.02.05.tar.gz')
print('Pre-trained modules init', flush=True)


def regex(text):
    text = text.replace(u'\xa0', ' ')
    text = text.translate(str.maketrans({key: ' {0} '.format(key) for key in '"!&()*+,/:;<=>?[]^`{|}~'}))
    text = re.sub('\s{2,}', ' ', text).replace('\n', '')

    return text

def check(s, wi, c):
    return sum([s[wi + j].lower() == c_ for j, c_ in  enumerate(c) if wi + j < len(s)]) == len(c)

def ind(si, wi, ci, c):
    return [[si, wi  + i, ci] for i in range(len(c))]

def compute_coref(s):

    try:
        ret = predictor.predict(s)
        return ret['clusters'], ret['document']
    except RuntimeError:
        return [], [str(w) for w in nlp(s)]

def next_batch(data_mb):
    
    for d in data_mb:

        d['candidates_orig'] = list(d['candidates'])
#         print("Answer is ", d['answer'], " in ", list(d['candidates']), ' and id is', list(d['candidates']).index(d['answer']))
        d['answer_id'] = list(d['candidates']).index(d['answer'])
        
        d['candidates'] = [c for c in d['candidates'] if c not in nlp.Defaults.stop_words]
        d['candidates_orig2'] = list(d['candidates'])
        
        d['candidates'] = [[str(w) for w in c] for c in nlp.pipe(d['candidates'])]

        d['query'] = [str(w) for w in nlp.tokenizer(d['query'])]

        d['supports'] = [regex(s) for s in d['supports']]
        
        tmp = [compute_coref(s) for s in d['supports']]
        d['supports'] = [e for _, e in tmp]
        d['coref'] = [e for e, _ in tmp]
        
        d['coref'] = [[[[f, []] for f in e] for e in s]
                      for s in d['coref']]
        
        mask = [[ind(si, wi, ci, c) for wi, w in enumerate(s) 
             for ci, c in enumerate(d['candidates'] + [d['query'][1:]]) 
                          if check(s, wi, c)] for si, s in enumerate(d['supports'])]
        
        nodes = []
        for sc, sm in zip(d['coref'], mask):
            u = []
            for ni, n in enumerate(sm):
                k = []
                for cli, cl in enumerate(sc):

                    x = [(n[0][1] <= co[0] <= n[-1][1]) or (co[0] <= n[0][1] <= co[1]) 
                         for co, cll in cl]

                    for i, v in filter(lambda y: y[1], enumerate(x)):
                        k.append((cli, i))
                        cl[i][1].append(ni)
                u.append(k)
            nodes.append(u)

        # remove one entity with multiple coref
        for sli, sl in enumerate(nodes):
            for ni, n in enumerate(sl):
                if len(n) > 1:
                    for e0, e1 in n:
                        i = d['coref'][sli][e0][e1][1].index(ni)
                        del d['coref'][sli][e0][e1][1][i]
                    sl[ni] = []

        # remove one coref with multiple entity
        for ms, cs in zip(nodes, d['coref']):
            for cli, cl in enumerate(cs):
                for eli, (el, li) in enumerate(cl):
                    if len(li) > 1:
                        for e in li:
                            i = ms[e].index((cli, eli))
                            del ms[e][i]
                        cl[eli][1] = []

        d['edges_coref'] = []
        for si, (ms, cs) in enumerate(zip(mask, d['coref'])):
            tmp = []
            for cl in cs:
                cand = {ms[n[0]][0][-1] for p, n in cl if n}
                if len(cand) == 1:
                    cl_ = []
                    for (p0, p1), _ in cl:
                        if not _:
                            cl_.append(len(ms))
                            ms.append([[si, i, list(cand)[0]] for i in range(p0, p1 + 1)])
                        else:
                            cl_.append(_[0])
                    tmp.append(cl_)
            d['edges_coref'].append(tmp)

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
                        if e0 != e1 and w0 == w1:
                            edges_out.append((f0, f1))

        edges_coref = []
        for nins, cs in zip (nodes_id_name, d['edges_coref']):
            for cl in cs:
                for e0 in cl:
                    for e1 in cl:
                        if e0 != e1:
                            edges_coref.append((nins[e0][0], nins[e1][0]))

        d['edges_coref'] = edges_coref
        d['edges_in'] = edges_in
        d['edges_out'] = edges_out
        d['edges'] = edges_in + edges_out + edges_coref

        mask_ = [[x[:-1] for x in f] for e in mask for f in e]

        candidates, _ = ee.batch_to_embeddings(d['supports'])
        candidates = candidates.data.cpu().numpy()

        d['nodes_elmo'] = [(candidates.transpose((0, 2, 1, 3))[np.array(m).T.tolist()]).astype(np.float16) 
                           for m in mask_]

        for e in d['nodes_elmo']:
            t0, t1 = e[:,2,512:].copy(), e[:,1,512:].copy()
            e[:,1,512:], e[:,2,512:] = t0, t1

        query, _ = ee.batch_to_embeddings([d['query']])
        query = query.data.cpu().numpy()
        d['query_elmo'] = (query.transpose((0, 2, 1, 3))).astype(np.float16)[0]
    
    
    
        d['nodes_candidates_id'] = d['nodes_candidates_id'][:max_nodes]
        d['nodes_elmo'] = d['nodes_elmo'][:max_nodes]
        d['edges_in'] = [e for e in d['edges_in'] if e[0] < max_nodes and e[1] < max_nodes]
        d['edges_out'] = [e for e in d['edges_out'] if e[0] < max_nodes and e[1] < max_nodes]
        d['edges_coref'] = [e for e in d['edges_coref'] if e[0] < max_nodes and e[1] < max_nodes]
        d['query_elmo'] = d['query_elmo'][:max_query_size]
    
    # Braeak to store data
    
    
    
    id_mb = [d['id'] for d in data_mb]
    
    answer_candidates_id = [d['answer_id'] for d in data_mb] # 
    
    candidates_orig_mb = [d['candidates_orig'] for d in data_mb]
    candidates_orig_mb2 = [d['candidates_orig2'] for d in data_mb]
    candidates_mb = [d['candidates'] for d in data_mb]

    filt = lambda c: np.array([c[:,0].mean(0), c[0,1], c[-1,2]])
    
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

        if len(d['edges_coref']) == 0:
            adj_.append(np.zeros((max_nodes, max_nodes)))
        else:
            adj = scipy.sparse.coo_matrix((np.ones(len(d['edges_coref'])), np.array(d['edges_coref']).T),
                shape=(max_nodes, max_nodes)).toarray()   

            adj_.append(adj)


        adj = np.pad(np.ones((len(d['nodes_candidates_id']), len(d['nodes_candidates_id']))), 
                     ((0, max_nodes - len(d['nodes_candidates_id'])),
                      (0, max_nodes - len(d['nodes_candidates_id']))), mode='constant') \
            - adj_[0] - adj_[1] - adj_[2] - np.pad(np.eye(len(d['nodes_candidates_id'])),
                                         ((0, max_nodes - len(d['nodes_candidates_id'])),
                                          (0, max_nodes - len(d['nodes_candidates_id']))), mode='constant')

        adj_.append(np.clip(adj, 0, 1))

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

    return {'id_mb': id_mb, 
            'nodes_mb': nodes_mb, 
            'nodes_length_mb': nodes_length_mb, 
            'query_mb': query_mb, 
            'query_length_mb': query_length_mb, 
            'bmask_mb': bmask_mb,
            'adj_mb': adj_mb, 
            'candidates_mb': candidates_mb, 
            'candidates_orig_mb': candidates_orig_mb,
            'candidates_orig_mb2': candidates_orig_mb2,
            'answer_candidates_id': answer_candidates_id
           }