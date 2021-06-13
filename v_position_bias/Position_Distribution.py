# -*- coding: utf-8 -*-
# @Time    : 2021/3/22 1:52
# @Author  : Anda Zhou
# @FileName: Position_Distribution.py

import matplotlib.pyplot as plt
import numpy as np
from hyperpara import *
from dataset import *
import tqdm
import json
import allennlp
from spacy.lang.en import English
import dgl
import re
import torch
from allennlp.predictors.predictor import Predictor
import pandas as pd
import csv


nlp = English()

# Get Train paired set
DATA_ADD = '/home/watsonzhouanda/multihop/graph/train_20000_27999.json'
# GRAPH_ADD = '/home/watsonzhouanda/multihop/graph/train_5.dgl'
GRAPH_ADD = '/home/watsonzhouanda/multihop/graph/wiki_train_20000_27999.dgl'

predictor = Predictor.from_path('/home/watsonzhouanda/multihop/src/coref-model-2018.02.05.tar.gz')
print('Pre-trained modules init', flush=True)

def print_loc(text_file, graph_file, mode):
    # max_loc = 0
    # max_index = 0
    # dist = np.zeros(1925)
    # dist = {}
    # Graph, _ = dgl.load_graphs(file)
    # for index, g in enumerate(Graph):
    #     if 'loc_start' in g.ndata:
    #         loc = g.ndata['loc_start']
    #         for i in loc:
    #             if int(i) not in dist:
    #                 dist[int(i)] = 1
    #             else:
    #                 dist[int(i)] += 1
    # print(dist)
    # plt.bar(dist.keys(), dist.values())
    # plt.show()

    # print("Index {0} loc {1}".format(index, loc))
    # if max(loc) >= max_loc:
    #     max_index = index
    #     max_loc = max(loc)
    # print(max_loc)     # 1924
    # print(max_index)   # 5624

    training_set = Dataset(text_file, graph_file, mode)
    graph = training_set.get_graph_set()
    max_loc = 0
    max_index = 0
    dist_answer = []
    dist_cand = []
    text = training_set.get_text_set()
    # print(text[0]['answer'])
    for index, g in enumerate(graph):
        d = text[index]
        answer = text[index]['answer']
        answer_id = d['candidates'].index(d["answer"])
        e_id = g.ndata['e_id']
        loc = g.ndata['loc_start']
        for answer_index, entity_id in enumerate(e_id):
            if int(loc[int(answer_index)]) <= 500:
                if int(entity_id) == int(answer_id):
                    # if int(loc[int(answer_index)]) not in dist_answer:
                    #     dist_answer[int(loc[int(answer_index)])] = 1
                    #
                    # else:
                    #     dist_answer[int(loc[int(answer_index)])] += 1
                    dist_answer.append(int(loc[int(answer_index)]))
                # if int(loc[int(answer_index)]) not in dist_cand:
                dist_cand.append(int(loc[int(answer_index)]))
                #     dist_cand[int(loc[int(answer_index)])] = 1
                # else:
                #     dist_cand[int(loc[int(answer_index)])] += 1
    # dist_cand = sorted(dist_cand.items(), key=lambda x: x[0], reverse=False)
    # dist_answer = sorted(dist_answer.items(), key=lambda x: x[0], reverse=False)

    # h_cand = [v[1] for v in dist_cand]
    # h_ans = [v[1] for v in dist_answer]

    plt.hist(dist_cand, bins=501, alpha=0.5, density=True, label='Canditate')
    plt.hist(dist_answer, bins=501, alpha=0.5, density=True, label='Answer')
    plt.legend(fontsize=15)
    plt.xlabel("Position of entity in its support document.", fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    plt.title('Distribution of the entity position', fontsize=20)
    plt.xticks(fontsize=10)
    plt.ylim((0, 0.025))
    plt.xlim((0, 200))
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.savefig('v5_pos+baseline/Distribution.pdf')

    plt.show()
    # kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True)
    # plt.hist()
    # plt.bar(dist_cand.keys(), dist_cand.values(), **kwargs)
    # plt.bar(dist_answer.keys(), dist_answer.values(), **kwargs)
    # plt.show()

    # with open('cand_distribution.csv', "w") as csv_file:
    #     writer = csv.writer(csv_file)
    #     for key, value in dist_cand.items():
    #         key = int(key)
    #         value = int(value)
    #         writer.writerow([key, value])
    #
    # with open('answer_distribution.csv', "w") as csv_file:
    #     writer = csv.writer(csv_file)
    #     for key, value in dist_answer.items():
    #         key = int(key)
    #         value = int(value)
    #         writer.writerow([key, value])

    # with open('my_file.csv', 'w') as f:
        # [f.write('{0},{1}\n'.format(key, value)) for key, value in my_dict.items()]


def main():
    # Get Graph set
    # training_set = check_dataset(DATA_ADD + "train.json", GRAPH_ADD, "Training")
    # training_set.count()
    print_loc(DATA_ADD, GRAPH_ADD, 'training')

    pass


if __name__ == '__main__':
    main()
