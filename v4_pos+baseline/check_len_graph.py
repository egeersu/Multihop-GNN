# -*- coding: utf-8 -*-
# @Time    : 2021/3/21 14:22
# @Author  : Anda Zhou
# @FileName: check_len_graph.py

import dgl
a, _ = dgl.load_graphs("graph/train_wikihop_30000_36350_graphs.dgl")
# b, _ = dgl.load_graphs("graph/dev_wikihop_3701_5129_graphs.dgl")
# c, _ = dgl.load_graphs("graph/dev_wikihop_0_3700_graphs.dgl")
print(len(a))