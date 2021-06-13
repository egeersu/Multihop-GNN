import scipy
import json
import re
import traceback
import allennlp
from allennlp.predictors.predictor import Predictor

from allennlp.commands.elmo import ElmoEmbedder
from spacy.lang.en import English
import numpy as np
# import tensorflow as tf
import torch 

from hyperpara import *
import dgl

from tqdm import tqdm

import dgl
from hyperpara import *

import random

def dgl2normAdj(g):
    edge_pair = g.edges()
    us,vs = edge_pair
    norm_adj = torch.zeros((num_rels, max_nodes, max_nodes))
    for i,(u,v) in enumerate(zip(us, vs)):
        norm_adj[g.edata['rel_type'][i]][u][v] = g.edata['e_weight'][i]
    return norm_adj


def print_model_size(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums


    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))
    
    
    
def regex(text):
    text = text.replace(u'\xa0', ' ')
    text = text.translate(str.maketrans({key: ' {0} '.format(key) for key in '"!&()*+,/:;<=>?[]^`{|}~'}))
    text = re.sub('\s{2,}', ' ', text).replace('\n', '')
    return text

def check(s, wi, c):
    return sum([s[wi + j].lower() == c_ for j, c_ in  enumerate(c) if wi + j < len(s)]) == len(c)

# c_i, c: entity in entitise set {s} U C_q (entites in canddiate answers and query)
# s_i, s: tokenized support document in supports
# wi, w: word in document s
# Turns (tokenized docu, word_i in original doc, candidate i)
def ind(si, wi, ci, c):
    return [[si, wi  + i, ci] for i in range(len(c))]

# predictor = Predictor.from_path(args.project_address+'mlp_project/src/coref-model-2018.02.05.tar.gz')
predictor = Predictor.from_path('/home/watsonzhouanda/multihop/src/coref-model-2018.02.05.tar.gz')

def compute_coref(s):
    try:
        '''
        {
            "document": [tokenised document text]
            "clusters":
              [
                [
                  [start_index, end_index],
                  [start_index, end_index]
                ],
                [
                  [start_index, end_index],
                  [start_index, end_index],
                  [start_index, end_index],
                ],
                ....
              ]
            }
        '''
        ret = predictor.predict(s)
        return ret['clusters'], ret['document']
    except RuntimeError:
        return [], [str(w) for w in nlp(s)]
    
    
def pad_graph_batch(unpad_batch_graph):
    batch_graph = []
    for g in unpad_batch_graph:
        for j in range(len(g), max_nodes):
            g.add_nodes(1, data={'e_id':torch.tensor([-1],dtype=torch.int)})
        batch_graph.append(g)
    return batch_graph


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True