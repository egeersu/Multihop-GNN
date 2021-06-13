# Multihop-GNN
Multi-hop Question Answering via Graph Neural Model (MLP Project)

## Environment
- **Python                    3.6.13**
- pytorch                   1.7.1
- cudatookit                11.0.221
- **scipy                     1.5.2**
- scikit-learn              0.24.1
- **allennlp                  0.9.0**
- **SpaCy                     2.1.9**
- **tensorflow                1.13.1**
- **dgl                         0.6.0**

## Dataset
- WikiHop & MedHop (Predict answer): http://qangaroo.cs.ucl.ac.uk/
- Hotpot QA (QA with answer generation): https://hotpotqa.github.io/

## Pretrained Modules
- ELMo: https://worksheets.codalab.org/worksheets/0xd2fb12d9f637460db16c110b5d3f2ca5
- Coreference System: https://worksheets.codalab.org/worksheets/0x96182529f99041408c22715b4ab846b3

## How to run
- Step 1. Generation for the graph for train & dev set
  - run `python entity_graph_gen.py --project-address --mode --number-of-data`, e.g., `python entity_graph_gen.py --project-address=/home/qyifu/ --graph-gen-mode=train --graph-gen-size=10 --dataset=medhop` will create 10 entity graphs from training set 1-10 samples in medhop. 
- Step 2. Train the model
  - run `python train.py --project-address`
