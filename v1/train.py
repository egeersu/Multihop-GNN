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
import torch 
from tqdm import tqdm 

from hyperpara import *
from dataset import *
from model import *

import dgl

print(torch.cuda.is_available())
torch.cuda.set_device(0)

def validate(model, graph_set, n_feats):
    
    val_loss, acc_count = 0, 0
    batch_size = 1
    
    for batch_start_id in range(0, len(graph_set), batch_size):

        g_batch = dgl.batch(graph_set[batch_start_id:batch_start_id+batch_size])
#         print("Dev graph:",g_batch)
        n_feats_batch = n_feats[batch_start_id:batch_start_id+batch_size]
        n_feats_batch = torch.stack(n_feats_batch)
        
        with torch.no_grad():
            b_outputs = model.forward(g_batch, n_feats_batch, g_batch.edata['rel_type'].long(), g_batch.edata['e_weight'].unsqueeze(1).long())
            b_label = train_yy[batch_start_id:batch_start_id+batch_size]

            acc_count += torch.sum(b_outputs.argmax(dim=1) == b_label).item()
            val_loss += F.cross_entropy(b_outputs, b_label).item() # input: (N,C), Target: (N)        
        
    
    val_acc = acc_count / len(graph_set)
    
    return val_loss, val_acc


# Get Train paired set
DATA_ADD = args.project_address+"mlp_project/dataset/qangaroo_v1.1/wikihop/"
GRAPH_ADD = args.project_address+"mlp_project/graph/"

# Get Graph set
training_set = Dataset(DATA_ADD+"train.json", GRAPH_ADD+'train_graphs_medhop_full.dgl', "Training")

print("Start training on "+GRAPH_ADD)

training_set.demo_test(10) # Comment out if not in test
(train_text_set, train_graphs) = training_set.get_text_graph_pair_set()
# print("Train size check:", len(train_text_set), len(train_graphs))

# Get Dev paired set
inference_set = Dataset(DATA_ADD+"dev.json", GRAPH_ADD+'dev_graphs_demo.dgl', "Inference (Val+Test)")
inference_set.demo_test(5) # Comment out if not in test
(inf_text_set, inf_graphs) = inference_set.get_text_graph_pair_set()
print("Inference size check:", len(inf_text_set), len(inf_graphs))

# Get size for 3 datasets: train_graphs 10, test 5
train_size = len(train_graphs)
inf_size = len(inf_graphs)
dev_size = int(0.5 * len(inf_graphs))
test_size = inf_size - dev_size
print("Dev and test set check:", dev_size, test_size)

# Get y labels
train_yy = training_set.get_label()
print(train_yy)
# Get qeury embedding
# train_embed_query, train_query = training_set.get_query_embed() ##

# print("Query check:", train_query[0], train_embed_query[0])

# print(train_query, train_embed_query)

# Get graphs for inference set (dev and test)
dev_graphs = inf_graphs[:dev_size]
# test_graphs = inf_graphs[dev_size:dev_size+test_size]

# Get y labels for inf
inf_yy = inference_set.get_label()
dev_yy = inf_yy[:dev_size]
# test_yy = inf_yy[dev_size:dev_size+test_size]
# Get qeury embedding
# inf_embed_query, inf_query = inference_set.get_query_embed()#
# print("inf_qeury: ",inf_query)
# print("inf_embed_query: ", inf_embed_query.shape) # (query_id, timesteps, 3 hiddens, 1024 dims)
# dev_embed_query = inf_embed_query[:dev_size]
# dev_query = inf_query[:dev_size]
# test_embed_query = inf_embed_query[dev_size : dev_size+test_size]
# test_query = inf_query[dev_size : dev_size+test_size]

print("dev_yy check:", dev_yy)
# print("Test query check:", dev_embed_query[0], dev_query[0])
# print("test_yy check:", test_yy)
# print("Test query check:", test_embed_query[0], test_query[0])

# print("dev_embed_query: ", dev_embed_query.shape) # (3, timesteps, 3 hiddens, 1024 dims)
# print("test_embed_query: ", test_embed_query.shape) # (2, timesteps, 3 hiddens, 1024 dims)




num_classes = max_candidates
model = RGCN(num_nodes=max_nodes, 
             gnn_in_dim=1024*3, 
             gnn_h_dim=gcn_n_hidden, 
             out_dim=max_candidates,
             num_rels=num_rels,
             num_bases=-1,
             num_gcn_hidden_layers=num_gcn_hidden_layers, 
             dropout=dropout,
             use_self_loop=False, use_cuda=args.use_gpu)


print(model)

if args.use_gpu:
    model.cuda()
print('Built a model with {:d} parameters'.format(sum(p.numel() for p in model.parameters())))
print("start training...")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2norm)



train_n_feats = training_set.get_graph_node_embed()
inf_n_feats = inference_set.get_graph_node_embed()
dev_n_feats = inf_n_feats[:dev_size]

print("Train graphs = ", train_graphs)

for epoch_id in tqdm(range(int(EPOCHS))):
    
    model.train()
    optimizer.zero_grad()
    batch_id = 0
    train_loss = 0
    acc_count = 0
    for batch_start_id in range(0, len(train_graphs), batch_size):
#         print("Start batch ",batch_id)
#         print('\nMemory Tracking: {:.1f} MiB / 12288 MiB (12G)'.format(torch.cuda.max_memory_allocated() / 1000000))
        g_batch = dgl.batch(train_graphs[batch_start_id:batch_start_id+batch_size])
        n_feats_batch = train_n_feats[batch_start_id:batch_start_id+batch_size]
        n_feats_batch = torch.stack(n_feats_batch)
        
#         print("Batch ", batch_id, g_batch)
        
        b_outputs = model.forward(g_batch, n_feats_batch, g_batch.edata['rel_type'].long(), g_batch.edata['e_weight'].unsqueeze(1).long())
        b_label = train_yy[batch_start_id:batch_start_id+batch_size]
        
        acc_count += torch.sum(b_outputs.argmax(dim=1) == b_label).item()
        
        loss = F.cross_entropy(b_outputs, b_label) # input: (N,C), Target: (N)
        train_loss += loss.item()
        loss.backward()
        batch_id+=1
        
    optimizer.step()
    
    train_acc = acc_count / len(train_graphs)
    
    dev_loss, dev_acc = validate(model, dev_graphs, dev_n_feats)
    print('\nMemory Tracking: {:.1f} MiB / 12288 MiB (12G)'.format(torch.cuda.max_memory_allocated() / 1000000))
    print("Epoch {:05d} | ".format(epoch_id) +
          "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
              train_acc, train_loss) +
          "Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(
              dev_acc, dev_loss))
    
    
    
#     acc_count = 0
#     for i, g in enumerate(train_graphs):
#         node_features = train_n_feats[i]
# #         g.ndata.pop('n_embed') 
#         logit = model.forward(g, node_features, g.edata['rel_type'], g.edata['e_weight'].unsqueeze(1)).unsqueeze(0)
#         label = train_yy[i].unsqueeze(0)
# #         print(logit, label)
#         train_loss += F.cross_entropy(logit, label) # output_vec, label(candidates_id)
#         if logit.argmax() == label: acc_count+=1
#     train_acc = acc_count / len(train_graphs)
    
    
#    # Find dev loss and acc:
#     dev_loss = 0
#     acc_count = 0
#     for i, g in enumerate(dev_graphs):
#         node_features = dev_n_feats[i]
# #         g.ndata.pop('n_embed') 
#         logit = model.forward(g, node_features, g.edata['rel_type'], g.edata['e_weight'].unsqueeze(1)).unsqueeze(0)
#         label = dev_yy[i].unsqueeze(0)
# #         print(logit, label)
#         dev_loss += F.cross_entropy(logit, label) # output_vec, label(candidates_id)
#         if logit.argmax() == label: acc_count+=1
#     dev_acc = acc_count / len(dev_graphs)
    
    
    