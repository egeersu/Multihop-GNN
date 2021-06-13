import logging
import argparse

# Fixed dont change
max_nodes = 500
max_query_size = 25
max_candidates = 70
max_candidates_len = 10

# Training parameter
learning_rate = 1e-4
batch_size = 32
EPOCHS = 10

gcn_n_hidden = 512 # number of hidden units
gcn_out_dim = 512
n_bases = 4 # use number of relations as number of bases
num_gcn_hidden_layers = 3 # use 1 input layer, 1 output layer, no hidden layer

l2norm = 0 # L2 norm coefficient
num_rels = 4 # 4 edge types

dropout_rate = 0.25

def get_args():
    """ Defines training-specific hyper-parameters. """
    parser = argparse.ArgumentParser('RGCN-QA')
    
    # Management
    parser.add_argument('--project-address', default='/afs/inf.ed.ac.uk/user/s20/s2041332/', help='identify the address for mlp_project')
    parser.add_argument('--use-gpu', default=True, help='if use GPU, need cuda.')
    
    # hyper for graph gen
    parser.add_argument('--graph-gen-mode', default=None, help='specify generate graph for which dataset.')
    parser.add_argument('--graph-gen-size', type=int, default=-1, help='specify generate how many graph for dataset.')
    parser.add_argument('--graph-gen-out-file', type=str, default=None, help='specify ouput file.')
    parser.add_argument('--dataset', type=str, default="wikihop", help='specify generate graph for which dataset.')


    # Add checkpoint arguments
#     parser.add_argument('--log-file', default=None, help='path to save logs')
#     parser.add_argument('--save-dir', default='checkpoints', help='path to save checkpoints')
#     parser.add_argument('--restore-file', default='checkpoint_last.pt', help='filename to load checkpoint')
#     parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
#     parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
#     parser.add_argument('--epoch-checkpoints', action='store_true', help='store all epoch checkpoints')

    # Trianing file running arugments
    
    
    # Networks architecture
    parser.add_argument('--run-train-graphs', type=str, default="train_graphs", help='specify graph file name in graph folder to use as training.')
    parser.add_argument('--run-dev-graphs', type=str, default="dev_graphs", help='specify graph file name in graph folder to use as validation.')
    
    # GNN Setting
    parser.add_argument('--max-nodes', type=int, default=500, help='if use GPU, need cuda.')
    parser.add_argument('--max-query-size', type=int, default=25, help='if use GPU, need cuda.')
    parser.add_argument('--max-candidates', type=int, default=70, help='if use GPU, need cuda.')
    parser.add_argument('--max-candidates-length', type=int, default=10, help='if use GPU, need cuda.')
    
    # Training parameters
    parser.add_argument('--batch-size', default=1, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--learning-rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--log-file', default='train_log.txt', type=str, help='log_out file')
    parser.add_argument('--patience', default=3, type=int, help='maximum epoch that validation loss do not decrease.')
    
    
    
    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    model_parser = parser.add_argument_group(argument_default=argparse.SUPPRESS)
    args = parser.parse_args()
    return args

args = get_args()