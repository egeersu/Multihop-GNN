# coding: utf-8

import os
import sys
import json
import numpy as np
import torch
import scipy.sparse



from nltk.tokenize import TweetTokenizer

from preprocessing import *
from hyperpara import *


DATA_ADD = "/afs/inf.ed.ac.uk/user/s20/s2041332/mlp_project/dataset/qangaroo_v1.1/wikihop/"
in_file = DATA_ADD+"train.json"
out_file = "/afs/inf.ed.ac.uk/user/s20/s2041332/mlp_project/graph/train_graph.json"


with open(in_file, 'r') as f:
    data = json.load(f)

saved_graph = {}
for i in range(len(data)):
    print(i)
    try:
        batch = next_batch([data[i]])
        feed_dict = {"nodes": batch['nodes_mb'].tolist(),
                 "nodes_length": batch['nodes_length_mb'].tolist(),
                 "query": batch['query_mb'].tolist(),
                 "query_length": batch['query_length_mb'].tolist(),
                 "adj": batch['adj_mb'].tolist(),
                 "bmask": batch['bmask_mb'].tolist()}
        saved_graph[i] = feed_dict
    except:
        # Continue if some queries length surpass the hyperpara max_query_len 
        continue
        
# Store the feed_dict data to json
with open(out_file,'w') as f:
    json.dump(saved_graph,f)