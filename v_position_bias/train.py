# -*- coding: utf-8 -*-
# @Time    : 2021/3/29 14:41
# @Author  : Anda Zhou
# @FileName: train.py

from collections import OrderedDict
import pandas as pd
import torch
from tqdm import tqdm
from hyperpara import *
from dataset import *
from model import *
from utils import *
import dgl


def validate(model, loss_func, dev_set, dev_text_set, dev_graph_set):
    model.eval()

    val_loss = 0
    acc_count = 0
    batch_id = 0
    batch_size = 250
    val_FP_TP, val_FN_TP, val_TP = 0, 0, 0
    cur_f1 = 0
    with torch.no_grad():
        if args.use_gpu:
            yy_set = dev_set.get_dev_batch_label((dev_graph_set, dev_text_set)).cuda()
        else:
            yy_set = dev_set.get_dev_batch_label(dev_graph_set, dev_text_set)
        # print("yy_set size", yy_set.size())   # sample_

        # Loop over batches to prevent cuda out of memory
        for batch_start_id in tqdm(range(0, len(dev_text_set), batch_size)):
            batch_text = dev_text_set[batch_start_id: batch_start_id + batch_size]
            batch_graph = dev_graph_set[batch_start_id : batch_start_id+batch_size]
            batch_graph = pad_graph_batch(batch_graph)
            if args.use_gpu:
                node_num_batch = dev_set.get_batch_node_num(batch_start_id, batch_start_id + batch_size).cuda()
                position_batch = dev_set.get_batch_position_start(batch_graph).cuda()

            else:
                node_num_batch = dev_set.get_batch_node_num(batch_start_id, batch_start_id + batch_size)
                position_batch = dev_set.get_batch_position_start(batch_graph)

            b_outputs = model.forward(node_num_batch, position_batch)   # # bs, 500, 2
            b_outputs = b_outputs.transpose(1, 2)  # bs, 2, 500
            if batch_id == 0:
                predictions = b_outputs
            else:
                predictions = torch.cat((predictions, b_outputs), 0)
            # print("predictions size", predictions.size())
            batch_id += 1

        val_loss = loss_func(predictions, yy_set)  # input: (N,C), Target: (N)
        for i, length in enumerate(node_num_batch):
            acc_count += torch.sum(predictions[i].argmax(dim=0)[:length] == yy_set[i][:length]).item() / length
            # val_FP += torch.sum(predictions[i].argmax(dim=0)[:length])
            # val_FN += torch.sum(yy_set[i][:length])
            # val_TP += abs(torch.sum(yy_set[i][:length]) - torch.sum(predictions[i].argmax(dim=0)[:length]))

            cur_FP_TP = torch.sum(predictions[i].argmax(dim=0)[:length])
            cur_FN_TP = torch.sum(yy_set[i][:length])
            cur_TP_TN = torch.sum(predictions[i].argmax(dim=0)[:length] == yy_set[i][:length]).item()
            cur_TP = ((cur_FP_TP + cur_FN_TP + cur_TP_TN) - length) / 2

            cur_pre = cur_TP / cur_FP_TP
            cur_recall = cur_TP / cur_FN_TP
            cur_f1 += 2 * cur_pre * cur_recall / (cur_pre + cur_recall)
            val_TP += cur_TP
            val_FP_TP += cur_FP_TP
            val_FN_TP += cur_FN_TP
            if i == len(node_num_batch) - 1:
                cur_f1 /= len(node_num_batch)
        # acc_count = torch.sum(predictions.argmax(dim=1) == yy_set).item()
    cur_f1 /=
    print("length of dev:", len(dev_text_set), len(yy_set), flush=True)
    val_loss = val_loss / len(yy_set)
    val_acc = acc_count / len(yy_set)
    val_precision = val_TP / val_FP_TP
    val_recall = val_TP / val_FN_TP
    val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall)

    return val_loss, val_acc, val_precision, val_recall, val_f1

# print(torch.cuda.is_available())
# torch.cuda.set_device(0)

# Set random seed
setup_seed(20)

DATA_ADD = '/home/watsonzhouanda/multihop/dataset/'
GRAPH_ADD = '/home/watsonzhouanda/multihop/graph/'
print("Start training on "+GRAPH_ADD, flush=True)

# prepare node embedding, norm adjacent mat, text set, query, label yy
# training_set = Dataset(DATA_ADD+"train.json", GRAPH_ADD+'wiki_train_new_400.dgl', "Training")
training_set = Dataset(DATA_ADD+"train_20000_27999.json", GRAPH_ADD+'wiki_train_20000_27999.dgl', "Training")
train_text_set = training_set.get_text_set()
train_graph_set = training_set.get_graph_set()
train_size = len(train_text_set)

# dev_set = Dataset(DATA_ADD+"dev.json", GRAPH_ADD+'dev_graphs_200.dgl', "Development")
dev_set = Dataset(DATA_ADD+"dev_text.json", GRAPH_ADD+'dev_500.dgl', "Development")
dev_text_set = dev_set.get_text_set()
dev_graph_set = dev_set.get_graph_set()
dev_size = len(dev_text_set)

# Building the model
n_input = 512
n_hidden1 = 128
n_hidden2 = 64
n_output = 2
model = Net(n_input, n_hidden1, n_hidden2, n_output)
print(model, flush=True)

# capture the best model during 10 epoch
best_validate = float('inf')
bad_epoch = 0
record_out = []

if args.use_gpu:
    model.cuda()
print('Built a model with {:d} parameters'.format(sum(p.numel() for p in model.parameters())), flush=True)
print("start training...", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2norm)
loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
# loss_func = nn.NLLLoss(reduction='sum')

for epoch_id in tqdm(range(int(EPOCHS))):

    model.train()

    batch_id = 0
    train_loss = 0
    acc_count = 0
    TP = 0
    FP_TP = 0
    FN_TP = 0

    stats = OrderedDict()
    stats['epoch'] = epoch_id
    stats['train_loss'] = 0
    stats['train_acc'] = 0
    stats['dev_loss'] = 0
    stats['dev_acc'] = 0
    stats['train_precision'] = 0
    stats['train_recall'] = 0
    stats['train_f1_score'] = 0
    stats['dev_precision'] = 0
    stats['dev_recall'] = 0
    stats['dev_f1_score'] = 0

    # Loop over batches
    for batch_start_id in tqdm(range(0, train_size, batch_size)):
        batch_id += 1
        batch_text = train_text_set[batch_start_id: batch_start_id + batch_size]
        batch_graph = train_graph_set[batch_start_id: batch_start_id + batch_size]
        batch_graph = pad_graph_batch(batch_graph)
        optimizer.zero_grad()
        # print("node_num_set", training_set.node_num_set())
        if args.use_gpu:
            node_num_batch = training_set.get_batch_node_num(batch_start_id, batch_start_id + batch_size).cuda()
            position_batch = training_set.get_batch_position_start(batch_graph).cuda()
            b_label = training_set.get_batch_label(batch_graph, batch_text).cuda()   # bs 500
        else:
            node_num_batch = training_set.get_batch_node_num(batch_start_id, batch_start_id + batch_size)
            position_batch = training_set.get_batch_position_start(batch_graph)
            b_label = training_set.get_batch_label(batch_graph, batch_text)   # bs 500
        # print("b_label size", b_label.size())
        # print("node_num_batch", node_num_batch)
        if batch_id == (training_set.size // batch_size) + 1:
            b_label = b_label[:(training_set.size % batch_size)]
        b_outputs = model.forward(node_num_batch, position_batch)   # bs 500 2
        b_outputs = b_outputs.transpose(1, 2)   # bs 2 500
        # print("b_outputs size", b_outputs.size())
        # print("b_label size", b_label.size())
        # print("b_label = ", b_label)
        # print("b_output = ", b_outputs)
        loss = loss_func(b_outputs, b_label)  # input: (N,C), Target: (N)

        # print("", b_outputs.argmax(dim=1) == b_label)
        # acc_count += torch.sum(b_outputs.argmax(dim=1) == b_label).item()
        for i, length in enumerate(node_num_batch):
            acc_count += torch.sum(b_outputs[i].argmax(dim=0)[:length] == b_label[i][:length]).item() / length
            cur_FP_TP = torch.sum(b_outputs[i].argmax(dim=0)[:length])
            cur_FN_TP = torch.sum(b_label[i][:length])
            cur_TP_TN = torch.sum(b_outputs[i].argmax(dim=0)[:length] == b_label[i][:length]).item()
            cur_TP = ((cur_FP_TP + cur_FN_TP + cur_TP_TN) - length) / 2
            TP += cur_TP
            FP_TP += cur_FP_TP
            FN_TP += cur_FN_TP
            # TP += abs(torch.sum(b_label[i][:length]) - torch.sum(b_outputs[i].argmax(dim=0)[:length]))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    Precison = TP / FP_TP
    Recall = TP / FN_TP
    F1 = 2 * Precison * Recall / (Precison + Recall)
    stats['train_loss'] = train_loss / train_size
    stats['train_acc'] = acc_count / train_size
    stats['train_precision'] = Precison
    stats['train_recall'] =Recall
    stats['train_f1_score'] = F1

    stats['dev_loss'], stats['dev_acc'], stats['dev_precision'], stats['dev_recall'], stats['dev_f1_score'] \
        = validate(model, loss_func, dev_set, dev_text_set, dev_graph_set)

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
            "dev_acc": stats['dev_acc'],
            "train_precision": stats['train_precision'],
            "train_recall": stats['train_recall'],
            "train_f1_score": stats['train_f1_score'],
            "dev_precision": stats['dev_precision'],
            "dev_recall": stats['dev_recall'],
            "dev_f1_score": stats['dev_f1_score']
        }, 'checkpoints/best_model.pth.tar')
    else:
        bad_epoch += 1

    # print('\nMemory Tracking: {:.1f} MiB / 12288 MiB (12G)'.format(torch.cuda.max_memory_allocated() / 1000000))
    print(
        'Epoch {:03d}: {}'.format(epoch_id, ' | '.join(key + ' {:.4g}'.format(value) for key, value in stats.items())),
        '\t', flush=True)

    record_out.append([stats['train_acc'], stats['train_loss'], stats['dev_acc'], stats['dev_loss'],
                       stats['train_precision'], stats['train_recall'], stats['train_f1_score'],
                       stats['dev_precision'], stats['dev_recall'], stats['dev_f1_score']])

    # save last epoch
    torch.save({
        'epoch': stats['epoch'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': stats['train_loss'],
        "train_acc": stats['train_acc'],
        "dev_loss": stats['dev_loss'],
        "dev_acc": stats['dev_acc'],
        "train_precision": stats['train_precision'],
        "train_recall": stats['train_recall'],
        "train_f1_score": stats['train_f1_score'],
        "dev_precision": stats['dev_precision'],
        "dev_recall": stats['dev_recall'],
        "dev_f1_score": stats['dev_f1_score'],
    }, 'checkpoints/last_epoch_model.pth.tar')

    # pd.DataFrame(record_out).to_csv("results/qagn_train.csv", index=None, header=['train_acc', 'train_loss','dev_acc', 'dev_loss'])
    pd.DataFrame(record_out).to_csv("results/pos_train.csv", index=None,
                                    header=['train_acc', 'train_loss', 'dev_acc', 'dev_loss',
                                            'train_precision', 'train_recall', 'train_f1_score',
                                            'dev_precision', 'dev_recall', 'dev_f1_score'])











