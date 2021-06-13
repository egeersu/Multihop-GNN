import scipy
import json
import re
import allennlp
from allennlp.predictors.predictor import Predictor
from allennlp.commands.elmo import ElmoEmbedder
from spacy.lang.en import English
import numpy as np
import tensorflow as tf
import os
import sys
import argparse
import tensorflow

from hyperpara import *
from dataset import *

class Model:
    
    def __init__(self):

        # placeholders
        self.nodes = tf.placeholder(shape=(None, max_nodes, 3, 1024), dtype=tf.float32)
        self.nodes_length = tf.placeholder(shape=(None,), dtype=tf.int32)

        self.query = tf.placeholder(shape=(None, max_query_size, 3, 1024), dtype=tf.float32)
        self.query_length = tf.placeholder(shape=(None,), dtype=tf.int32)

#         self.answer_node_mask = tf.placeholder(shape=(None, max_nodes), dtype=tf.float32)
        self.answer_candidates_id = tf.placeholder(shape=(None, ), dtype=tf.int64)
        
        self.adj = tf.placeholder(shape=(None, 4, max_nodes, max_nodes), dtype=tf.float32)
        self.bmask = tf.placeholder(shape=(None, max_candidates, max_nodes), dtype=tf.float32)
        
        self.training = tf.placeholder_with_default(False, shape=())
        self.dropout_rate = tf.placeholder_with_default(0., shape=())

        # masks
        nodes_mask = tf.tile(tf.expand_dims(tf.range(max_nodes, dtype=tf.int32), 0), 
                                  (tf.shape(self.nodes_length)[0], 1)) < tf.expand_dims(self.nodes_length, -1)
        self.nodes_mask = tf.cast(nodes_mask, tf.float32)

        query_mask = tf.tile(tf.expand_dims(tf.range(max_query_size, dtype=tf.int32), 0), 
                             (tf.shape(self.query_length)[0], 1)) < tf.expand_dims(self.query_length, -1)

        # compress and flatten query
        query_flat = tf.reshape(self.query, (-1, max_query_size, 3 * 1024))

        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=[tf.nn.rnn_cell.LSTMCell(256), tf.nn.rnn_cell.LSTMCell(128)],
            cells_bw=[tf.nn.rnn_cell.LSTMCell(256), tf.nn.rnn_cell.LSTMCell(128)],
            inputs=query_flat,
            dtype=tf.float32,
            sequence_length=self.query_length
        )
        
        self.outputs_ = outputs, output_state_fw, output_state_bw

        query_compress = tf.concat((output_state_fw[-1].h, output_state_bw[-1].h), -1)
        query_compress = tf.layers.dropout(query_compress, self.dropout_rate, training=self.training)

        nodes_flat = tf.reshape(self.nodes, (-1, max_nodes, 3 * 1024))
        nodes_compress = tf.layers.dense(nodes_flat, units=256, activation=tf.nn.tanh)
        nodes_compress = tf.layers.dropout(nodes_compress, self.dropout_rate, training=self.training)

        # create nodes
        # Concate nate node reprsentation and query; while multiply the node_mask
        nodes = tf.concat((nodes_compress,
                           tf.tile(tf.expand_dims(query_compress, -2), (1, max_nodes, 1))), 
                          -1) * tf.expand_dims(self.nodes_mask, -1)

        # FFNN nodes
        nodes = tf.layers.dense(nodes, 1024, activation=tf.nn.tanh)
        nodes = tf.layers.dropout(nodes, self.dropout_rate, training=self.training)
        nodes = tf.layers.dense(nodes, 512, activation=tf.nn.tanh)
        nodes = tf.layers.dropout(nodes, self.dropout_rate, training=self.training)
        
        last_hop = nodes
        for _ in range(3):
            last_hop = self.hop_layer(last_hop, self.nodes_mask)
            last_hop = tf.layers.dropout(last_hop, self.dropout_rate, training=self.training)

        predictions_input = tf.concat((last_hop,
            tf.tile(tf.expand_dims(query_compress, -2), (1, max_nodes, 1))), -1) * tf.expand_dims(self.nodes_mask, -1)

        self.predictions1 = tf.squeeze(tf.layers.dense(tf.layers.dense(
            predictions_input, units=128, activation=tf.nn.tanh), units=1), -1)
        self.predictions1 = tf.layers.dropout(self.predictions1, self.dropout_rate, training=self.training)
        
        self.predictions2 = self.bmask * tf.expand_dims(self.predictions1, 1)
        self.predictions2 = tf.where(tf.equal(self.predictions2, 0), 
             tf.fill(tf.shape(self.predictions2), -np.inf), self.predictions2)
        self.predictions2 = tf.reduce_max(self.predictions2, -1)
        
    def hop_layer(self, hidden_tensor, hidden_mask):
        with tf.variable_scope('hop_layer', reuse=tf.AUTO_REUSE):

            adjacency_tensor = self.adj
            hidden_tensors = tf.stack([tf.layers.dense(inputs=hidden_tensor, units=hidden_tensor.shape[-1]) 
                                       for _ in range(adjacency_tensor.shape[1])], 1) * \
                        tf.expand_dims(tf.expand_dims(hidden_mask, -1), 1)
            
            update = tf.reduce_sum(tf.matmul(adjacency_tensor, hidden_tensors), 1) + tf.layers.dense(
                hidden_tensor, units=hidden_tensor.shape[-1]) * tf.expand_dims(hidden_mask, -1)

            att = tf.layers.dense(tf.concat((update, hidden_tensor), -1), units=hidden_tensor.shape[-1], 
                                  activation=tf.nn.sigmoid) * tf.expand_dims(hidden_mask, -1)

            output = att * tf.nn.tanh(update) + (1 - att) * hidden_tensor
            return output

class Optimizer:

    def __init__(self, model):
        
        self.cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            model.answer_candidates_id,
            model.predictions2, reduction=tf.losses.Reduction.NONE)

        self.accuracy_ = tf.cast(tf.equal(tf.argmax(model.predictions2, -1),
                                           model.answer_candidates_id), tf.float32)

        self.accuracy = tf.reduce_mean(self.accuracy_) * 100

        self.loss = tf.reduce_mean(self.cross_entropy)

        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.train_step = self.opt.minimize(self.loss)

def predict(g, s, m, b):
    
    with g.as_default():

        feed_dict = {m.nodes: b['nodes_mb'],
            m.nodes_length: b['nodes_length_mb'],
            m.query: b['query_mb'],
            m.query_length: b['query_length_mb'],
            m.adj: b['adj_mb'],
            m.bmask: b['bmask_mb']}

        pred = s.run(m.predictions2, feed_dict)
    
        pred = pred - pred[~np.isinf(pred)].min()
        pred = np.exp(pred) / np.exp(pred).sum(-1, keepdims=True)
        
        return pred


# Train
print("Start training.")


tf.reset_default_graph()
model = Model()
optimizer = Optimizer(model)

config = tf.ConfigProto(device_count={'CPU': 2})
session = tf.Session(config=config)

session.run(tf.global_variables_initializer())


dataset = Dataset()
train_set = dataset.get_train_set("/afs/inf.ed.ac.uk/user/s20/s2041332/mlp_project/graph/train_graph.json")

# Note: Inference set include dev and future works like testing
# ['0'] index the whole development set
dev_set = dataset.get_inference_set("/afs/inf.ed.ac.uk/user/s20/s2041332/mlp_project/graph/dev_graph.json")['0']


print("numb of batch = ", len(train_set))
for i in range(EPOCHS):
    # Overloop all batch
    for batch_id in range(len(train_set)):
        batch = train_set[str(batch_id)]
        session.run(optimizer.train_step, feed_dict = {model.nodes: batch['nodes'],
                     model.nodes_length: batch['nodes_length'],
                     model.query: batch['query'],
                     model.query_length: batch['query_length'],
    #                  model.answer_node_mask: batch[0]['answer_node_mask'],
                     model.answer_candidates_id: batch['answer_candidates_id'],
                     model.adj: batch['adj'],
                     model.bmask: batch['bmask']})

    # Train Loss - evaluated on full training set
    train_loss = []
    for batch_id in range(len(train_set)):
        batch = train_set[str(batch_id)]
        train_loss_i_batch = session.run(optimizer.loss, feed_dict = {model.nodes: batch['nodes'],
                     model.nodes_length: batch['nodes_length'],
                     model.query: batch['query'],
                     model.query_length: batch['query_length'],
    #                  model.answer_node_mask: batch[0]['answer_node_mask'],
                     model.answer_candidates_id: batch['answer_candidates_id'],
                     model.adj: batch['adj'],
                     model.bmask: batch['bmask']})
        train_loss.append(train_loss_i_batch)
    train_loss = np.sum(train_loss)
    
    # Development Accuracy - evaluated on full development set
    dev_acc = session.run(optimizer.accuracy, feed_dict = {model.nodes: dev_set['nodes'],
                     model.nodes_length: dev_set['nodes_length'],
                     model.query: dev_set['query'],
                     model.query_length: dev_set['query_length'],
    #                  model.answer_node_mask: dev_set[0]['answer_node_mask'],
                     model.answer_candidates_id: dev_set['answer_candidates_id'],
                     model.adj: dev_set['adj'],
                     model.bmask: dev_set['bmask']})
    print("Epoch: ", i, " with train loss at ", train_loss, "; dev acc = ", dev_acc)

print("Training DONE.")

# output
out_file = '/afs/inf.ed.ac.uk/user/s20/s2041332/mlp_project/output/baseline/answer_output.json'

# answers = dict(answers)

with open(out_file, 'w') as f:
    json.dump(answers, f)
    
print('loaded', flush=True)




# Inference

# # for i in range(len(data)):
# for i in range(num_data):
#     try:
#         data_mb = entity_graph_data[i]

#         pred_ = [predict(v['g'], v['session'], v['model'], data_mb) for v in models]
        
#         pred = np.vstack(pred_).prod(0).argmax()

#         ans = data_mb['candidates_orig_mb2'][0][pred]
        
#     except Exception as e:
#         print(e)
#         ans = data_mb['candidates_orig_mb2'][0][0]

#     answers.append(ans)
#     print(i, '/', len(data), ans[1].encode('utf-8'), flush=True)