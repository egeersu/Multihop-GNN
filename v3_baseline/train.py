# coding: utf-8

import scipy
import json
import re
import allennlp
from allennlp.predictors.predictor import Predictor
import logging
from collections import OrderedDict
import pandas as pd

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

def validate(model, criterion, dev_set, dev_text_set, dev_graph_set):
    model.eval()
    
    dev_graph_set = pad_graph_batch(dev_graph_set)
    if args.use_gpu:
        n_feats_batch = dev_set.get_batch_graph_node_embed(dev_graph_set).cuda()
        node_num = dev_set.get_batch_node_num(0,dev_set.get_size()).cuda()
        norm_adj_batch = dev_set.get_batch_graph_norm_adj(dev_graph_set).cuda()
        answer_mask = dev_set.get_batch_answer_mask(dev_graph_set, dev_text_set).cuda()
        query_embed,_ = dev_set.get_batch_query(dev_text_set)
        query_embed = query_embed.cuda()
        yy_set = dev_set.get_batch_label(dev_text_set).cuda()
    
    
    val_loss, acc_count = 0, 0
    
    with torch.no_grad():
        outputs = model.forward(n_feats_batch, norm_adj_batch, query_embed, node_num, answer_mask)
        acc_count = torch.sum(outputs.argmax(dim=1) == yy_set).item()
        val_loss = criterion(outputs, yy_set).item() # input: (N,C), Target: (N)        
    
    val_loss = val_loss / len(yy_set)
    val_acc = acc_count / len(yy_set)
    
    return val_loss, val_acc


# Get Train paired set
DATA_ADD = args.project_address+"mlp_project/dataset/qangaroo_v1.1/"+args.dataset+"/"
GRAPH_ADD = args.project_address+"mlp_project/graph/"

print("Start training on "+GRAPH_ADD)

# Get Graph set
if args.project_address == "/home/qyifu/":
    training_set = Dataset(DATA_ADD+"train_20000_27999.json", GRAPH_ADD+'wiki_train_20000_27999.dgl', "Training")
else:
    training_set = Dataset(DATA_ADD+"train.json", GRAPH_ADD+args.run_train_graphs+'.dgl', "Training")

# prepare node embedding, norm adjacent mat, text set, query, label yy
train_text_set = training_set.get_text_set()
train_graph_set = training_set.get_graph_set()
train_size = len(train_text_set)

# prepare development set
if args.project_address == "/home/qyifu/":
    dev_set = Dataset(DATA_ADD+"dev_text.json", GRAPH_ADD+'dev_500.dgl', "Development")
else:
    dev_set = Dataset(DATA_ADD+"dev.json", GRAPH_ADD+args.run_dev_graphs+'.dgl', "Development")

dev_text_set = dev_set.get_text_set()
dev_graph_set = dev_set.get_graph_set()
dev_size = len(dev_text_set)

model = RGCN(num_nodes=max_nodes, 
             gnn_h_dim=gcn_n_hidden, 
             out_dim=max_candidates,
             num_rels=num_rels,
             num_gcn_hidden_layers=num_gcn_hidden_layers, 
             dropout_rate=dropout_rate,
             use_self_loop=False, use_cuda=args.use_gpu)

print(model)

if args.use_gpu:
    model.cuda()
print('Built a model with {:d} parameters'.format(sum(p.numel() for p in model.parameters())))
print("start training...")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2norm)
criterion = nn.NLLLoss()

# capture the best model during 10 epoch
best_validate = float('inf')
bad_epoch = 0

record_out = []


for epoch_id in tqdm(range(int(EPOCHS))):
    
    model.train()
    
    batch_id = 0
    train_loss = 0
    acc_count = 0
    
    stats = OrderedDict()
    stats['epoch'] = epoch_id
    stats['train_loss'] = 0
    stats['train_acc'] = 0
    stats['dev_loss'] = 0
    stats['dev_acc'] = 0
    
    # Loop over batches
    for batch_start_id in tqdm(range(0, train_size, batch_size)):
        batch_id+=1
        
        batch_text = train_text_set[batch_start_id : batch_start_id+batch_size]
        batch_graph = train_graph_set[batch_start_id : batch_start_id+batch_size]
        batch_graph = pad_graph_batch(batch_graph)
        optimizer.zero_grad()
        
        if args.use_gpu:
            n_feats_batch = training_set.get_batch_graph_node_embed(batch_graph).cuda()
            node_num_batch = training_set.get_batch_node_num(batch_start_id,batch_start_id+batch_size).cuda()
            norm_adj_batch = training_set.get_batch_graph_norm_adj(batch_graph).cuda()
            answer_mask_batch = training_set.get_batch_answer_mask(batch_graph, batch_text).cuda()
            query_embed_batch,_ = training_set.get_batch_query(batch_text)
            query_embed_batch = query_embed_batch.cuda()
            
            b_label = training_set.get_batch_label(batch_text).cuda()
            
        
        b_outputs = model.forward(n_feats_batch, norm_adj_batch, query_embed_batch, node_num_batch, answer_mask_batch)
        
        
        loss = criterion(b_outputs, b_label) # input: (N,C), Target: (N)
        acc_count += torch.sum(b_outputs.argmax(dim=1) == b_label).item()
        
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    stats['train_loss'] = train_loss / train_size
    stats['train_acc'] = acc_count / train_size
    
    stats['dev_loss'], stats['dev_acc'] = validate(model, criterion, dev_set, dev_text_set, dev_graph_set)
    
    # if find better model, save best validate acc and save best model
    if stats['dev_loss'] < best_validate:
        best_validate = stats['dev_loss']
        bad_epochs = 0
        torch.save({
            'epoch': stats['epoch'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': stats['train_loss'],
            "train_acc": stats['train_acc'],
            "dev_loss": stats['dev_loss'],
            "dev_acc": stats['dev_acc']
        }, 'checkpoints/best_model.pth.tar')
    else:
        bad_epoch+=1
    
    print('\nMemory Tracking: {:.1f} MiB / 12288 MiB (12G)'.format(torch.cuda.max_memory_allocated() / 1000000))
    print('Epoch {:03d}: {}'.format(epoch_id, ' | '.join(key + ' {:.4g}'.format(value) for key, value in stats.items())), '\t')
    
    record_out.append([stats['train_acc'], stats['train_loss'], stats['dev_acc'], stats['dev_loss']])
    
    # save last epoch
    torch.save({
            'epoch': stats['epoch'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': stats['train_loss'],
            "train_acc": stats['train_acc'],
            "dev_loss": stats['dev_loss'],
            "dev_acc": stats['dev_acc']
        }, 'checkpoints/last_epoch_model.pth.tar')
    
    pd.DataFrame(record_out).to_csv("results/baseline_train.csv", index=None, header=['train_acc', 'train_loss','dev_acc', 'dev_loss'])
    
    # If use patience training
#     if bad_epoch >= args.patience:
#         print('No validation set improvements observed for {:d} epochs. Early stop!'.format(args.patience))
#         break