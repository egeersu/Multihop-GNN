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
import traceback

# class Logger(object):
#     def __init__(self, filename='default.log', stream=sys.stdout):
#         self.terminal = stream
#         self.log = open(filename, 'w')
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass
#
# sys.stdout = Logger('search_rm_sample_dev_0_3700.log', sys.stdout)
# sys.stderr = Logger('search_rm_sample_0_3700.log', sys.stderr)

nlp = English()
# Setting for Elmo Embedder - CHANGE THE PATH
# options_file = args.project_address+'mlp_project/src/elmo_2x4096_512_2048cnn_2xhighway_options.json'
# weight_file = args.project_address+'mlp_project/src/elmo_2x4096_512_2048cnn_2xhighway_weights'

options_file = '/home/watsonzhouanda/multihop/src/elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file = '/home/watsonzhouanda/multihop/src/elmo_2x4096_512_2048cnn_2xhighway_weights'

ee = ElmoEmbedder(
                  options_file=options_file,
                  weight_file=weight_file)

text_add = '/home/watsonzhouanda/multihop/dataset/qangaroo_v1.1/wikihop/dev.json'

with open(text_add, 'r') as f:
    text_set = json.load(f)
        
def if_keep_text_sample(d):
    
    # Processing the query and candidate entities, find C_q U {s}
        d['candidates_orig'] = list(d['candidates']) # record the original candidate
        d['candidates'] = [c for c in d['candidates'] if c not in nlp.Defaults.stop_words]
        d['candidates'] = [[str(w) for w in c] for c in nlp.pipe(d['candidates'])]


        d['query'] = [str(w) for w in nlp.tokenizer(d['query'])][1:]

        # discard the sample accroding to De Cao
        if (len(d['query']) > max_query_size) and (len(d['candidates']) > max_candidates):
            print("Discard sample because query length (should not be seen)",i_d)
            return False

        entities_set = d['candidates'] + [d['query']] # C_q U {s}

        # Document level coreference prediction
        # First preprocess the document 
        d['supports'] = [regex(s) for s in d['supports']]
        coref_temp = [compute_coref(support_doc) for support_doc in d['supports']]

        entities_span_in_docs = [e for _, e in coref_temp] # [tokenised document text for each document], entities span S_q
        coref_cluster_in_docs = [e for e, _ in coref_temp] # [corefernt spans for each cluster in each document]

        d['coref'] = [[[[f, []] for f in e] for e in s]
                      for s in coref_cluster_in_docs] #[support_doc_id, cluster_id, span_id]

        # c_i, c: entity in entitise set {s} U C_q (entites in canddiate answers and query)
        # s_i, s: tokenized support document in supports
        # wi, w: word in document s
        # shape: [num_supports, i in entities set, tuple]
        # tuple: (#doc, position in doc, id of c in entities set)
        exact_match_doc2entity_set = [[ind(si, wi, ci, c) for wi, w in enumerate(s) 
             for ci, c in enumerate(entities_set) 
                          if check(s, wi, c)] for si, s in enumerate(entities_span_in_docs)]

        exact_match_entity_spans = [] # [cid, start, end, doc_id]

        for support_doc_id in range(len(exact_match_doc2entity_set)):
            if len(exact_match_doc2entity_set[support_doc_id]) == 0:
                continue
            for c_i, exact_matched_entities in enumerate(exact_match_doc2entity_set[support_doc_id]):
                for loc_i, loc in enumerate(exact_matched_entities):
    #                     print(loc)
                    doc_id = loc[0]
                    doc_ent_loc = loc[1]
                    id_in_entities = loc[2]
    #                     span.append(d['supports'][doc_id][doc_ent_loc])
    #                 entity_in_supdoc_id = torch.Tensor(exact_matched_entities[0][0])
                doc_id = torch.tensor(exact_matched_entities[0][0], dtype=torch.int32).unsqueeze(0)
                entities_id = exact_matched_entities[0][-1]

    #                 print([entities_id, exact_matched_entities[0][1],exact_matched_entities[-1][1],support_doc_id])
                exact_match_entity_spans.append([entities_id, exact_matched_entities[0][1],exact_matched_entities[-1][1],support_doc_id])


    #         Compute coreference
    #     print("--------------------------")
    #     print("NEXT WE START ADDING COREFERENCE NODES!")
    #     print("--------------------------")


        # Find the nodes that entities in entities_set has corefrent in coreference prediction
        coref_nodes = []

        for sc, sm in zip(d['coref'], exact_match_doc2entity_set): # overloop  (entity id, loc, doc_id)
            u = [] # doc
            for ni, n in enumerate(sm): # overloop each match entities (entity id, loc, doc_id)
                k = []
                for cli, cl in enumerate(sc): # overloop coref clusters

                    coref_loc = [[co[0], co[1]] for co, cll in cl]

                    x = [(n[0][1] <= co[0] <= n[-1][1]) or (co[0] <= n[0][1] <= co[1]) 
                         for co, cll in cl]

                    # i: entity id
                    for i, v in filter(lambda y: y[1], enumerate(x)):
                        k.append((cli, i)) # De cao's : cluster - entities - loc start - loc end #
                        cl[i][1].append(ni)
                u.append(k)

            coref_nodes.append(u)


        # remove one entity with multiple coref
        for sli, sl in enumerate(coref_nodes): # loop sup document
            for ni, n in enumerate(sl): # loop entities to coref
                if len(n) > 1:
                    for e0, e1 in n:
                        i = d['coref'][sli][e0][e1][1].index(ni)
                        del d['coref'][sli][e0][e1][1][i]
                    sl[ni] = []

        # remove one coref with multiple entity
        for ms, cs in zip(coref_nodes, d['coref']):
            for cli, cl in enumerate(cs):
                for eli, (el, li) in enumerate(cl):
                    if len(li) > 1:
                        for e in li:
                            i = ms[e].index((cli, eli))
                            del ms[e][i]
                        cl[eli][1] = []

        ## Check here
        d['edges_coref'] = []
        for si, (ms, cs) in enumerate(zip(exact_match_doc2entity_set, d['coref'])):
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

    #     print("coref_nodes:", coref_nodes)

        nodes_id_name = []
        c = 0
        for e in [[[x[-1] for x in c][0] for c in s] for s in exact_match_doc2entity_set]:
            u = []
            for f in e:
                u.append((c, f))
                c +=1
            nodes_id_name.append(u)

        mask_ = [[x[:-1] for x in f] for e in exact_match_doc2entity_set for f in e]

    #     print("len mask",len(mask_))
    #     print(mask_)


        record_of_loc_span = []
        for node_i, node in enumerate(mask_):
            node_span = []
            loc_span = []
            doc_id = -1
            for i, unit in enumerate(node):
                doc_id, loc = unit[0], unit[1]
                node_span.append(entities_span_in_docs[doc_id][loc])
                loc_span.append(loc)
            item = (doc_id, loc_span, node_span)
            record_of_loc_span.append(item)



        candidates, _ = ee.batch_to_embeddings(entities_span_in_docs)
        # select out the words (entities) we want
        d['nodes_elmo'] = [(candidates.transpose(2, 1)[torch.tensor(m,dtype=torch.float).T.tolist()]) for m in mask_]


        # change second and first dimension
        for e in d['nodes_elmo']:
            t0, t1 = e[:,2,512:].clone(), e[:,1,512:].clone()
            e[:,1,512:], e[:,2,512:] = t0, t1

        filt = lambda c: torch.stack([c.mean(0)[0], c[0][1], c[-1][2]])
        nodes_embed = torch.stack([filt(a) for a in d['nodes_elmo']])

        # Now we initalize the node in the graph
        wid = 0

        for doc_id, nodes_in_doc in enumerate(nodes_id_name):
            if nodes_in_doc == []:
                continue
            for node_id, e_id in nodes_in_doc:
                doc_id, loc_span, word_span = record_of_loc_span[wid]
                loc_start = torch.tensor([loc_span[0]], dtype=torch.int)
                loc_end = torch.tensor([loc_span[-1]], dtype=torch.int)
    #             print("Add node now:", doc_id, loc_start, loc_end)

                doc_id = torch.tensor([doc_id], dtype=torch.int32)
                e_id = torch.tensor([e_id], dtype=torch.int32)

    #             embed_entities = torch.tensor([nodes_embed[wid]])
    #             print(nodes_embed[wid].shape)
                embed_entities = nodes_embed[wid].unsqueeze(0)
    #             print(embed_entities.shape)

                wid+=1

        d['nodes_candidates_id'] = [[x[-1] for x in f][0] for e in exact_match_doc2entity_set for f in e]
        
#         print(d['nodes_candidates_id'])
        
        # discard the sample according to De Cao
        if len(d['nodes_candidates_id']) > max_nodes or len(d['nodes_candidates_id']) <= 0:
            print("Discard sample because num of nodes is zero or larger than limid. ID:",i_d)
            return False
    
        return True
        
    
remove_id = []
    
for i, d in enumerate(tqdm(text_set[0:3700])):
    try:
        if if_keep_text_sample(d) == False:
            print("Remove:", d['id'])
            remove_id.append(i)
    except:
        print("Remove sample i {} but because exception.".format(i))
        traceback.print_exc()
        remove_id.append(i)

file = open('removed_samples_id_dev_0_3700.txt','w')
file.write(str(remove_id))
file.close()