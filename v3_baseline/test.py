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
from collections import Counter
import random

from hyperpara import *
from dataset import *
from model import *
from utils import *

import dgl

print(torch.cuda.is_available())
#torch.cuda.set_device(0)

def load_checkpoint(model, checkpoint_PATH):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['model_state_dict'])
        print('loading checkpoint!')
        print("Best model at epoch {} with train loss={} acc={}, dev loss={} acc={}".format(model_CKPT['epoch'], model_CKPT['train_loss'], model_CKPT['train_acc'], model_CKPT['dev_loss'], model_CKPT['dev_acc']))
    return model


def get_top_k(outputs, text_set, k=5):
    '''
    Given the output tensor, return the top-k predictions for each sample. 
    '''
    predicted_indices = []
    for i in range(outputs.shape[0]):
        top_k_indices = np.argsort(-1*outputs[i,:])[:k]
        predicted_indices.append(top_k_indices)
    return predicted_indices
    
def test_statistics(outputs, text_set):

    # gather dataset statistics for query eligibility 
    query_count = Counter([sample['query'][0] for sample in text_set])
    support_count = {sample['query'][0]:0 for sample in text_set}
    candidate_count = {sample['query'][0]:{} for sample in text_set}
    
    for sample in text_set:
        candidates = sample['candidates']
        query = sample['query'][0]
        num_supports = len(sample['supports'])
        support_count[query] += num_supports
        for candidate in candidates:      
            if candidate in candidate_count[query]:   
                candidate_count[query][candidate] += 1
            else:
                candidate_count[query][candidate] = 0
    
    # keep queries with at least 50 supporting documents and at least 5 unique candidates. (as specified by De Cao)
    eligible_queries = []
    for query in query_count.keys():
        num_support = support_count[query]
        unique_candidates = np.count_nonzero(list(candidate_count[query].values()))
        if num_support >= 50 and unique_candidates >= 5:
            eligible_queries.append(query)
    
    predicted_indices_2 = get_top_k(outputs, text_set, k=2)
    predicted_indices_5 = get_top_k(outputs, text_set, k=5)
    
    query_correct = {query:0 for query in eligible_queries}
    query_correct_2 = {query:0 for query in eligible_queries}
    query_correct_5 = {query:0 for query in eligible_queries}
    
    # collect correct counts for each query
    for i,sample in enumerate(text_set):
        query = sample['query'][0]
        if query in eligible_queries:
            correct_index = sample['candidates'].index(sample['answer'])
            if correct_index == predicted_indices_2[i][0]:
                query_correct[query] += 1
            if correct_index in predicted_indices_2[i]:
                query_correct_2[query] += 1
            if correct_index in predicted_indices_5[i]:
                query_correct_5[query] += 1
        
    # compute top-k
    query_accuracies = {query:query_correct[query]/query_count[query] for query in eligible_queries}
    query_accuracies_2 = {query:query_correct_2[query]/query_count[query] for query in eligible_queries}
    query_accuracies_5 = {query:query_correct_5[query]/query_count[query] for query in eligible_queries}

    sorted_accuracies = sorted(query_accuracies.items(), key=lambda item: item[1], reverse=True)

    
    # BEST 3
    print("3 BEST\n")
    for query in sorted_accuracies[0:3]:
        query = query[0]
        acc = query_accuracies[query]
        p_at_2 = query_accuracies_2[query]
        p_at_5 = query_accuracies_5[query]
        print(query, "\naccuracy: ", acc, "| P@2: ", p_at_2, "| P@5: ", p_at_5, "\n")
    
    # WORST 3
    print("3 WORST\n")
    for query in sorted_accuracies[-3:]:
        query = query[0]
        acc = query_accuracies[query]
        p_at_2 = query_accuracies_2[query]
        p_at_5 = query_accuracies_5[query]
        print(query, "\naccuracy: ", acc, "| P@2: ", p_at_2, "| P@5: ", p_at_5, "\n")
        
    # ENTIRE DATASET
    correct_1 = 0
    correct_2 = 0
    correct_5 = 0
    for i,sample in enumerate(text_set):
        correct_index = sample['candidates'].index(sample['answer'])
        if correct_index == predicted_indices_2[i][0]:
            correct_1 += 1
        if correct_index in predicted_indices_2[i]:
            correct_2 += 1
        if correct_index in predicted_indices_5[i]:
            correct_5 += 1
            
    print("ENTIRE DATASET")
    print("accuracy: ", correct_1/len(text_set), "| P@2: ", correct_2/len(text_set), "| P@5: ", correct_5/len(text_set), "\n")




def validate2(model, criterion, dev_set, dev_text_set, dev_graph_set):
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
    else:
        n_feats_batch = dev_set.get_batch_graph_node_embed(dev_graph_set)
        node_num = dev_set.get_batch_node_num(0,dev_set.get_size())
        norm_adj_batch = dev_set.get_batch_graph_norm_adj(dev_graph_set)
        answer_mask = dev_set.get_batch_answer_mask(dev_graph_set, dev_text_set)
        query_embed,_ = dev_set.get_batch_query(dev_text_set)
        query_embed = query_embed
        yy_set = dev_set.get_batch_label(dev_text_set)    
    
    val_loss, acc_count = 0, 0
    
    with torch.no_grad():
        outputs = model.forward(n_feats_batch, norm_adj_batch, query_embed, node_num, answer_mask)
        acc_count = torch.sum(outputs.argmax(dim=1) == yy_set).item()
        val_loss = criterion(outputs, yy_set).item() # input: (N,C), Target: (N)        
    
    val_loss = val_loss / len(yy_set)
    val_acc = acc_count / len(yy_set)
    
    test_statistics(outputs, dev_text_set)
    
    return val_loss, val_acc

if __name__ == "__main__":

    # CHANGE THESE 
    # Get Train paired set
    DEV_GRAPH_PATH = args.project_address+"mlp_project/graph/dev_500.dgl"    
    DEV_DATA_PATH = args.project_address+"mlp_project/src/dataset/qangaroo_v1.1/wikihop/dev_text.json"
    TEST_GRAPH_PATH = args.project_address+"mlp_project/graph/test_500.dgl"
    TEST_DATA_PATH = args.project_address+"mlp_project/src/dataset/qangaroo_v1.1/wikihop/test_text.json"

    args.gpu = False
    print("args.gpu: ", args.gpu)

    # Import your model here
    model = RGCN(num_nodes=max_nodes, 
                gnn_h_dim=gcn_n_hidden, 
                out_dim=max_candidates,
                num_rels=num_rels,
                num_gcn_hidden_layers=num_gcn_hidden_layers, 
                dropout_rate=dropout_rate,
                use_self_loop=False, use_cuda=False)
  
    #model = load_checkpoint(model, "checkpoints/best_model.pth.tar")

    print(model)
        
    #if args.use_gpu:
    #    model.cuda()

    print('Built a model with {:d} parameters'.format(sum(p.numel() for p in model.parameters())))

    criterion = nn.NLLLoss(reduction='sum')

    # Prepare DEV set
    print("Loading DEV set!")
    dev_set = Dataset(DEV_DATA_PATH, DEV_GRAPH_PATH, "Development")
    dev_text_set = dev_set.get_text_set()
    dev_graph_set = dev_set.get_graph_set()
    
    print("dev_text_set: ", len(dev_text_set))
    dev_loss, dev_acc = validate2(model, criterion, dev_set, dev_text_set, dev_graph_set)
    print("dev_loss: ", dev_loss, "\ndev_acc: ", dev_acc)
    
    # Prepare TEST set
    print("Loading TEST set!")
    test_set = Dataset(TEST_DATA_PATH, TEST_GRAPH_PATH, "Testing")
    test_text_set = test_set.get_text_set()
    test_graph_set = test_set.get_graph_set()

    test_loss, test_acc = validate2(model, criterion, test_set, test_text_set, test_graph_set)
    print("test_loss: ", test_loss, "\ntest_acc: ", test_acc)

