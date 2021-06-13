# coding: utf-8

import os
import sys
import json
import numpy as np
import scipy.sparse
import pickle


from nltk.tokenize import TweetTokenizer

from preprocessing import *
from hyperpara import *

print("Start generating the Entity Graph.")

MODE = sys.argv[1]
num_data = int(sys.argv[2])
batch_size = int(sys.argv[3])

if MODE == 'dev':
    # one batch when validation
    batch_size = num_data

DATA_ADD = "/afs/inf.ed.ac.uk/user/s20/s2041332/mlp_project/dataset/qangaroo_v1.1/wikihop/"
in_file = DATA_ADD+MODE+".json"
out_file = "/afs/inf.ed.ac.uk/user/s20/s2041332/mlp_project/graph/"+MODE+"_graph.json"


with open(in_file, 'r') as f:
    data = json.load(f)
print('Dataset loaded', flush=True)

saved_graph = {}

idd = 0
for i in range(0, num_data, batch_size):
    if i+batch_size <= num_data:
        batch = next_batch(data[i:i+batch_size])
    else:
        batch = next_batch(data[i:num_data])
    feed_dict = {"nodes": batch['nodes_mb'].tolist(),
             "nodes_length": batch['nodes_length_mb'].tolist(),
             "query": batch['query_mb'].tolist(),
             "query_length": batch['query_length_mb'].tolist(),
             "adj": batch['adj_mb'].tolist(),
             "bmask": batch['bmask_mb'].tolist(),
             "candidates_orig_mb": batch['candidates_orig_mb'],
             "answer_candidates_id": batch['answer_candidates_id']
                }
    saved_graph[idd] = feed_dict
    idd+=1

# Store the feed_dict data to json
with open(out_file,'w') as f:
    json.dump(saved_graph,f)
    
print("Graph generation DONE!")