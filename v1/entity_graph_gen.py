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
import torch 

from hyperpara import *
import dgl

from tqdm import tqdm


# Setting for Elmo Embedder - CHANGE THE PATH
options_file = args.project_address+'mlp_project/src/elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file = args.project_address+'mlp_project/src/elmo_2x4096_512_2048cnn_2xhighway_weights'

# Initialization for each module
nlp = English()
ee = ElmoEmbedder(
                  options_file=options_file,
                  weight_file=weight_file)
predictor = Predictor.from_path(args.project_address+'mlp_project/src/coref-model-2018.02.05.tar.gz')
print('Pre-trained modules init', flush=True)


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

MODE = args.graph_gen_mode
num_data = args.graph_gen_size
    
DATA_ADD = args.project_address+"mlp_project/dataset/qangaroo_v1.1/"+args.dataset+"/"
in_file = DATA_ADD+MODE+".json"
GRAPH_ADD = args.project_address+"mlp_project/graph/"

with open(in_file, 'r') as f:
    data = json.load(f)
print('Dataset loaded', flush=True)



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

graph_set = []

if num_data == -1:
    num_data = len(data)
    print("Note: Now you are generating the full graph dataset! in "+args.dataset)
else:
    print("Note: Now you are generating tiny graph dataset! in "+args.dataset)
    
for i_d, d in enumerate(tqdm(data[:num_data])):
    
    try:
        
        # Processing the query and candidate entities, find C_q U {s}
        d['candidates_orig'] = list(d['candidates']) # record the original candidate
        d['candidates'] = [c for c in d['candidates'] if c not in nlp.Defaults.stop_words]
        d['candidates'] = [[str(w) for w in c] for c in nlp.pipe(d['candidates'])]

        # Test Check for candidates and query entites Paased.        
    #     print("Candidate entities C_q: ", d['candidates'])

        d['query'] = [str(w) for w in nlp.tokenizer(d['query'])][1:]
    #         print("query_entities {s}: ", d['query'])

        entities_set = d['candidates'] + [d['query']] # C_q U {s}
    #     print("Entities set ({s} U C_q): ", entities_set)

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

    #         print(g)
    #         print(g.ndata['n_embed'].shape)
    #         print(g.ndata['belong_doc'])

    #     print("exact_match_entity_spans: ", exact_match_entity_spans)
        # Now we have add basic nodes into graph
        # Then we add the coreference nodes into graph

    #         Compute coreference
    #     print("--------------------------")
    #     print("NEXT WE START ADDING COREFERENCE NODES!")
    #     print("--------------------------")


        # Find the nodes that entities in entities_set has corefrent in coreference prediction
        coref_nodes = []

        for sc, sm in zip(d['coref'], exact_match_doc2entity_set): # overloop  (entity id, loc, doc_id)
            u = []
            for ni, n in enumerate(sm): # overloop each match entities (entity id, loc, doc_id)
                k = []
                for cli, cl in enumerate(sc): # overloop coref clusters

                    x = [(n[0][1] <= co[0] <= n[-1][1]) or (co[0] <= n[0][1] <= co[1]) 
                         for co, cll in cl]

                    for i, v in filter(lambda y: y[1], enumerate(x)):
    #                         k.append((i, cli)) # ours: entities - cluster # 
                        k.append((cli, i)) # De cao's : cluster - entities #
                        cl[i][1].append(ni)
                u.append(k)
            coref_nodes.append(u)

        # Check the corefrence result
    #     coref_nodes_summary = [] # [i_cluster, span_content, c_start_id, c_end_id, support_doc_id]
    #         print("coref_nodes: ", coref_nodes)
    #     for doc_id, d_coref_nodes in enumerate(coref_nodes):
    #         for entity_id, e_coref_nodes in enumerate(d_coref_nodes):
    #             if len(e_coref_nodes) == 0:
    #                 continue
    #             for eid,cid in e_coref_nodes:
    #                 entities_content = entities_set[eid]
    #                 coref_nodes_summary.append((cid, entities_content, eid, doc_id))
    # #     print("coref_nodes_summary:", coref_nodes_summary)

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



        nodes_id_name = []
        c = 0
        for e in [[[x[-1] for x in c][0] for c in s] for s in exact_match_doc2entity_set]:
            u = []
            for f in e:
                u.append((c, f))
                c +=1

            nodes_id_name.append(u)

        mask_ = [[x[:-1] for x in f] for e in exact_match_doc2entity_set for f in e]
        candidates, _ = ee.batch_to_embeddings(entities_span_in_docs)
        candidates = candidates.data.cpu().numpy()
        d['nodes_elmo'] = [(candidates.transpose((0, 2, 1, 3))[np.array(m).T.tolist()]).astype(np.float16) for m in mask_]
        for e in d['nodes_elmo']:
            t0, t1 = e[:,2,512:].copy(), e[:,1,512:].copy()
            e[:,1,512:], e[:,2,512:] = t0, t1

        filt = lambda c: np.array([c[:,0].mean(0), c[0,1], c[-1,2]])
        nodes_embed = np.array([filt(c) for c in d['nodes_elmo']])

    #     print(nodes_embed)

    #         print("d['nodes_elmo']: ", d['nodes_elmo']) # elmo embedding Check passed

    #     print("nodes_id_name: ", nodes_id_name) # [[(node id, entity id)] for all docu]
        g = dgl.DGLGraph()
        # Now we initalize the node in the graph
        wid = 0
        for doc_id, nodes_in_doc in enumerate(nodes_id_name):
            if nodes_in_doc == []:
                continue
            for node_id, e_id in nodes_in_doc:
                word_span = entities_set[e_id]
    #             print(word_span)
                doc_id = torch.tensor([doc_id], dtype=torch.int32)
                e_id = torch.tensor([e_id], dtype=torch.int32)
                embed_entities, _ = ee.batch_to_embeddings([word_span])
                # Average pool the embedding over tokens in a entities span
                embed_entities = torch.mean(embed_entities, dim=2)
    #                 print(nodes_embed[wid])
                embed_entities = torch.tensor([nodes_embed[wid]])
    #                 print(embed_entities.shape)
                wid+=1
                g.add_nodes(1, {"n_embed": embed_entities, "d_id": doc_id, "e_id": e_id})

        # Check Graph
    #         print(g)
    #         print(g.ndata['d_id'])
    #         print(g.ndata['e_id'])
    #         print(g.ndata['n_embed'].shape)

        d['nodes_candidates_id'] = [[x[-1] for x in f][0] for e in exact_match_doc2entity_set for f in e]

        edges_in, edges_out = [], []
        for e0 in nodes_id_name:
            for f0, w0 in e0:
                for f1, w1 in e0:
                    if f0 != f1:
                        # DOC-BASED
                        edges_in.append((f0, f1))

                for e1 in nodes_id_name:
                    for f1, w1 in e1:
                        # Exact match
                        if e0 != e1 and w0 == w1:
                            edges_out.append((f0, f1))

        edges_coref = []
        for nins, cs in zip (nodes_id_name, d['edges_coref']):
            for cl in cs:
                for e0 in cl:
                    for e1 in cl:
                        if e0 != e1:
                            edges_coref.append((nins[e0][0], nins[e1][0]))



        d['edges_DOC_BASED'] = edges_in
        d['edges_MATCH'] = edges_out
        d['edges_COREF'] = edges_coref
        d['edges_n_COMPLETE'] = d['edges_DOC_BASED'] + d['edges_MATCH'] + d['edges_COREF']
    #         print("existing: ",d['edges_n_COMPLETE'])
        d['edges_COMPLETE'] = []
        nodes_id_list = [i for i in g.nodes().data.cpu().numpy()]
        for i in nodes_id_list:
            for j in nodes_id_list:
                if i == j:
                    # ignore same node, no self-loopo
                    continue
                if (i,j) not in d['edges_n_COMPLETE']:
                    d['edges_COMPLETE'].append((i, j))
    #         print(d['edges_COMPLETE'])

        all_edges = [d['edges_DOC_BASED']] + [d['edges_MATCH']] + [d['edges_COREF']] + [d['edges_COMPLETE']]

        # Calculate probability weight
        edge_prob_record = []
        for graph_i, subgraph_edges in enumerate(all_edges):
            edge_prob_in_graph = {}
            for start_node in nodes_id_list: 
                out_count = len([a for a in subgraph_edges if a[0] == start_node])
                if out_count:
                    edge_prob_in_graph[start_node] = 1/out_count
            edge_prob_record.append(edge_prob_in_graph)
    #         print(edge_prob_record)

        for i, rel_graph in enumerate(all_edges):
            for (src, tgt) in rel_graph:
                edge_type = torch.tensor([i], dtype=torch.int)
                p_weight = edge_prob_record[i][src]
                edge_weight = torch.tensor([p_weight], dtype=torch.float16)
    #                 print(edge_weight )
                g.add_edges(src, tgt, data={'rel_type': edge_type, 
                                            'e_weight': edge_weight})


    #     print(g.edata['rel_type'])


        graph_set.append(g)

        dgl.save_graphs(GRAPH_ADD+MODE+'_'+args.dataset+'_'+str(num_data)+'_'+'graphs.dgl', graph_set)
        
    except:
        continue


