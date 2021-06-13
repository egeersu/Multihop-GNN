# -*- coding: utf-8 -*-
# @Time    : 2021/3/22 1:52
# @Author  : Anda Zhou
# @FileName: Position_Distribution.py

import matplotlib.pyplot as plt
import numpy as np
from hyperpara import *
# from dataset import *
import tqdm
import json
import allennlp
from spacy.lang.en import English
import dgl
import re
import torch
from allennlp.predictors.predictor import Predictor

nlp = English()

# Get Train paired set
DATA_ADD = '/home/watsonzhouanda/multihop/dataset/qangaroo_v1.1/' + args.dataset + "/"
GRAPH_ADD = '/home/watsonzhouanda/multihop/graph/train_5.dgl'

predictor = Predictor.from_path('/home/watsonzhouanda/multihop/src/coref-model-2018.02.05.tar.gz')
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


def regex(text):
    text = text.replace(u'\xa0', ' ')
    text = text.translate(str.maketrans({key: ' {0} '.format(key) for key in '"!&()*+,/:;<=>?[]^`{|}~'}))
    text = re.sub('\s{2,}', ' ', text).replace('\n', '')
    return text


def check(s, wi, c):
    return sum([s[wi + j].lower() == c_ for j, c_ in enumerate(c) if wi + j < len(s)]) == len(c)


def ind(si, wi, ci, c):
    return [[si, wi + i, ci] for i in range(len(c))]


class check_dataset:

    def __init__(self, text_add, graph_add, mode):
        print("Start initialize dataset...")
        # Load text dataset
        with open(text_add, 'r') as f:
            self.text_set = json.load(f)

        # Load graph dataset
        self.graph_set, _ = dgl.load_graphs(graph_add)
        print("Graph set size:", len(self.graph_set))

        for i, (d, g) in enumerate(zip(self.text_set[0:5], self.graph_set)):
            d['query'] = [str(w) for w in nlp.tokenizer(d['query'])]
            if (len(g) <= max_nodes) and (len(d['query']) <= max_query_size) and (
                    len(d['candidates']) <= max_candidates) and (d['candidates'].index(d["answer"]) in g.ndata['e_id']):
                # print(d['candidates'])
                print(g.ndata['loc_start'])
                print(g.ndata['loc_end'])

    def count(self):

        print('Dataset loaded! with size:', len(self.text_set[0:1]), flush=True)

        rm_list = []
        for i_d, d in enumerate(self.text_set[0:1]):
            # Processing the query and candidate entities, find C_q U {s}
            d['candidates_orig'] = list(d['candidates'])  # record the original candidate
            d['candidates'] = [c for c in d['candidates'] if c not in nlp.Defaults.stop_words]
            d['candidates'] = [[str(w) for w in c] for c in nlp.pipe(d['candidates'])]
            d['query'] = [str(w) for w in nlp.tokenizer(d['query'])][1:]

            # discard the sample accroding to De Cao
            if (len(d['query']) > max_query_size) or (len(d['candidates']) > max_candidates):
                rm_list.append((i_d, d['id']))
                print("Discard sample because query or candidates length over limitation, ID:", (i_d, d['id']),
                      flush=True)
                continue

            entities_set = d['candidates'] + [d['query']]  # C_q U {s}

            # Document level coreference prediction
            # First preprocess the document
            d['supports'] = [regex(s) for s in d['supports']]
            coref_temp = [compute_coref(support_doc) for support_doc in d['supports']]

            entities_span_in_docs = [e for _, e in
                                     coref_temp]  # [tokenised document text for each document], entities span S_q
            coref_cluster_in_docs = [e for e, _ in
                                     coref_temp]  # [corefernt spans for each cluster in each document]

            d['coref'] = [[[[f, []] for f in e] for e in s]
                          for s in coref_cluster_in_docs]  # [support_doc_id, cluster_id, span_id]

            # c_i, c: entity in entitise set {s} U C_q (entites in canddiate answers and query)
            # s_i, s: tokenized support document in supports
            # wi, w: word in document s
            # shape: [num_supports, i in entities set, tuple]
            # tuple: (#doc, position in doc, id of c in entities set)
            exact_match_doc2entity_set = [[ind(si, wi, ci, c) for wi, w in enumerate(s)
                                           for ci, c in enumerate(entities_set)
                                           if check(s, wi, c)] for si, s in enumerate(entities_span_in_docs)]

            exact_match_entity_spans = []  # [cid, start, end, doc_id]

            for support_doc_id in range(len(exact_match_doc2entity_set)):
                if len(exact_match_doc2entity_set[support_doc_id]) == 0:
                    continue
                for c_i, exact_matched_entities in enumerate(exact_match_doc2entity_set[support_doc_id]):
                    for loc_i, loc in enumerate(exact_matched_entities):
                        doc_id = loc[0]
                        doc_ent_loc = loc[1]
                        id_in_entities = loc[2]
                    doc_id = torch.tensor(exact_matched_entities[0][0], dtype=torch.int32).unsqueeze(0)
                    entities_id = exact_matched_entities[0][-1]
                    exact_match_entity_spans.append(
                        [entities_id, exact_matched_entities[0][1], exact_matched_entities[-1][1], support_doc_id])

            coref_nodes = []

            for sc, sm in zip(d['coref'], exact_match_doc2entity_set):  # overloop  (entity id, loc, doc_id)
                u = []  # doc
                for ni, n in enumerate(sm):  # overloop each match entities (entity id, loc, doc_id)
                    k = []
                    for cli, cl in enumerate(sc):  # overloop coref clusters
                        coref_loc = [[co[0], co[1]] for co, cll in cl]
                        x = [(n[0][1] <= co[0] <= n[-1][1]) or (co[0] <= n[0][1] <= co[1])
                             for co, cll in cl]
                        # i: entity id
                        for i, v in filter(lambda y: y[1], enumerate(x)):
                            k.append((cli, i))  # De cao's : cluster - entities - loc start - loc end #
                            cl[i][1].append(ni)
                    u.append(k)
                coref_nodes.append(u)

            # remove one entity with multiple coref
            for sli, sl in enumerate(coref_nodes):  # loop sup document
                for ni, n in enumerate(sl):  # loop entities to coref
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
                    c += 1
                nodes_id_name.append(u)

            mask_ = [[x[:-1] for x in f] for e in exact_match_doc2entity_set for f in e]

            record_of_loc_span = []
            for _, node in enumerate(mask_):
                loc_span = []
                for i, unit in enumerate(node):
                    loc = unit[1]
                    loc_span.append(loc)
                item = (loc_span)
                record_of_loc_span.append(item)

            wid = 0
            print("wid = ", wid)
            for _, nodes_in_doc in enumerate(nodes_id_name):
                print("nodes_in_doc = ", nodes_in_doc)
                if nodes_in_doc == []:
                    print("continue")
                    # continue
                print("length = ", len(nodes_in_doc))
                for i in range(len(nodes_in_doc)):
                    print("i = ", i)
                    loc_span = record_of_loc_span[wid]
                    print("loc_span ", loc_span)
                    loc_start = torch.tensor([loc_span[0]], dtype=torch.int)
                    loc_end = torch.tensor([loc_span[-1]], dtype=torch.int)
                    print("loc_start = ", loc_start)
                    print("loc_end = ", loc_end)
                    wid += 1


def main():
    # Get Graph set
    training_set = check_dataset(DATA_ADD + "train.json", GRAPH_ADD, "Training")
    training_set.count()

    pass


if __name__ == '__main__':
    main()
