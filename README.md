

## Paper
[Position-Aware Neural Attentive Graph Networks for Question Answering](https://egeersu.github.io/papers/multihop.pdf)

## Code
[Repository](https://github.com/egeersu/Multihop-GNN)

## Abstract

Recently there has been considerable interest in applying [Graph Neural Networks (GNN)](https://distill.pub/2021/gnn-intro/) to Multi-hop Question Answering (QA) tasks, as graph representations can explicitly express rich dependencies in language. However, graph representations suffer from the loss of sequential information and the difficulty of representing global semantic information. In this work, we propose the **query-attention** mechanism to enhance the GNN-QA system by utilizing both global and local contextual information. We also explore injecting the positional information into the graph to complement the sequential information. Our experiments are conducted on the WikiHop dataset to allow direction comparison with [Entity Relational-Graph Convolutional Networks](https://arxiv.org/pdf/1808.09920.pdf). Our contributions identify the existence of *position bias* in the dataset, and we further conduct ablation studies to confirm that our proposed modules improve the generalization accuracy by 1.43%.

## Environment
- **Python**                  3.6.13
- **pytorch**                   1.7.1
- **cudatookit**                11.0.221
- **scipy**                    1.5.2
- **scikit-learn**              0.24.1
- **allennlp**                  0.9.0
- **SpaCy**                    2.1.9
- **tensorflow**                1.13.1
- **dgl**                         0.6.0

## Datasets
- [WikiHop & MedHop](http://qangaroo.cs.ucl.ac.uk)
- [Hotpot QA](https://hotpotqa.github.io)

## Required Pretrained Models
- [ELMo](https://worksheets.codalab.org/worksheets/0xd2fb12d9f637460db16c110b5d3f2ca5)
- [Coreference System](https://worksheets.codalab.org/worksheets/0x96182529f99041408c22715b4ab846b3)

## How to run
- Step 1. Generation for the graph for train & dev set
  - run `python entity_graph_gen.py --project-address --mode --number-of-data`, e.g., `python entity_graph_gen.py --project-address=/path/to/file --graph-gen-mode=train --graph-gen-size=10 --dataset=medhop` will create 10 entity graphs from training set 1-10 samples in medhop. 
- Step 2. Train the model
  - run `python train.py --project-address`
