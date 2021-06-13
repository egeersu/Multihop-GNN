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
from utils import *

import dgl

print(torch.cuda.is_available())
torch.cuda.set_device(0)

def validate(model, norm_adj_batch_set, n_feats, yy_set):
    
    val_loss, acc_count = 0, 0
    
    if args.use_gpu:
        n_feats_batch = torch.stack(n_feats).cuda()
        norm_adj_batch = torch.stack(norm_adj_batch_set).cuda()
    else:
        n_feats_batch = torch.stack(n_feats)
        norm_adj_batch = torch.stack(norm_adj_batch_set)
    
    with torch.no_grad():
        outputs = model.forward(n_feats_batch, norm_adj_batch)
        acc_count += torch.sum(outputs.argmax(dim=1) == yy_set).item()
        val_loss += F.cross_entropy(outputs, yy_set).item() # input: (N,C), Target: (N)        
    
    val_loss = val_loss / len(yy_set)
    val_acc = acc_count / len(yy_set)
    
    return val_loss, val_acc


# Get Train paired set
DATA_ADD = args.project_address+"mlp_project/dataset/qangaroo_v1.1/"+args.dataset+"/"
GRAPH_ADD = args.project_address+"mlp_project/graph/"

print("Start training on "+GRAPH_ADD)



# Get Graph set
training_set = Dataset(DATA_ADD+"train.json", GRAPH_ADD+args.run_train_graphs+'.dgl', "Training")

# prepare node embedding, norm adjacent mat, text set, query, label yy
train_text_set = training_set.get_text_set()
train_yy = training_set.get_label()
print("train_yy:",train_yy)
train_n_feats = training_set.get_graph_node_embed()
train_norm_adj = training_set.get_graph_norm_adj()
# train_query = training_set.get_query_embed()

train_size = len(train_text_set)
print("Check train size:", training_set.get_size(), len(train_text_set)==training_set.get_size())
print("Check train size:", training_set.get_size(), len(train_yy)==training_set.get_size())
print("Check train size:", training_set.get_size(), len(train_n_feats)==training_set.get_size())
print("Check train size:", training_set.get_size(), len(train_norm_adj)==training_set.get_size())

# prepare development set
dev_set = Dataset(DATA_ADD+"dev.json", GRAPH_ADD+args.run_dev_graphs+'.dgl', "Development")

dev_text_set = dev_set.get_text_set()
dev_yy = dev_set.get_label()
print("dev_yy:",dev_yy)
dev_n_feats = dev_set.get_graph_node_embed()
dev_norm_adj = dev_set.get_graph_norm_adj()
# dev_query = dev_set.get_query_embed()

dev_size = len(dev_text_set)
print("Check train size:", dev_set.get_size(), len(dev_text_set)==dev_set.get_size())
print("Check train size:", dev_set.get_size(), len(dev_yy)==dev_set.get_size())
print("Check train size:", dev_set.get_size(), len(dev_n_feats)==dev_set.get_size())
print("Check train size:", dev_set.get_size(), len(dev_norm_adj)==dev_set.get_size())

model = RGCN(num_nodes=max_nodes, 
             gnn_h_dim=gcn_n_hidden, 
             out_dim=max_candidates,
             num_rels=num_rels,
             num_gcn_hidden_layers=num_gcn_hidden_layers, 
             dropout=dropout,
             use_self_loop=False, use_cuda=args.use_gpu)


print(model)

if args.use_gpu:
    model.cuda()
print('Built a model with {:d} parameters'.format(sum(p.numel() for p in model.parameters())))
print("start training...")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2norm)

for epoch_id in tqdm(range(int(EPOCHS))):
    
    model.train()
    optimizer.zero_grad()
    batch_id = 0
    train_loss = 0
    acc_count = 0
    for batch_start_id in tqdm(range(0, train_size, batch_size)):
        
        if args.use_gpu:
            n_feats_batch = torch.stack(train_n_feats[batch_start_id:batch_start_id+batch_size]).cuda()
            norm_adj_batch = torch.stack(train_norm_adj[batch_start_id:batch_start_id+batch_size]).cuda()
        else:
            n_feats_batch = torch.stack(train_n_feats[batch_start_id:batch_start_id+batch_size])
            norm_adj_batch = torch.stack(train_norm_adj[batch_start_id:batch_start_id+batch_size])
        
        b_outputs = model.forward(n_feats_batch, norm_adj_batch)
        b_label = train_yy[batch_start_id:batch_start_id+batch_size]
        
        acc_count += torch.sum(b_outputs.argmax(dim=1) == b_label).item()
        
        loss = F.cross_entropy(b_outputs, b_label) # input: (N,C), Target: (N)
        train_loss += loss.item()
        loss.backward()
        batch_id+=1
        
    optimizer.step()
    
    train_loss = train_loss / train_size
    train_acc = acc_count / train_size
    
    dev_loss, dev_acc = validate(model, dev_norm_adj, dev_n_feats, dev_yy)
    
    print('\nMemory Tracking: {:.1f} MiB / 12288 MiB (12G)'.format(torch.cuda.max_memory_allocated() / 1000000))
    print("Epoch {:05d} | ".format(epoch_id) +
          "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
              train_acc, train_loss) +
          "Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(
              dev_acc, dev_loss))
    
    