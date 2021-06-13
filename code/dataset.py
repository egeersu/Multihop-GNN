import json

class Dataset(object):
    
    def __init__(self):
        
        self.train_set = []
        self.counter = 0
        self.train_graph_file_add = ''
        self.num_batch = 0
        
        self.inference_set = []
        self.inference_graph_file_add = ''
        
    def get_train_set(self, train_graph_add):
        # Read offline graph files
        self.train_graph_file_add = train_graph_add
        
        with open(self.train_graph_file_add, 'r') as f:
            # For test purpose
            self.train_set = json.load(f)
        self.num_batch = len(self.train_set)
        return self.train_set
            
    def reset(self):
        self.counter = 0
    
    # Used for validation and testing
    def get_inference_set(self, inference_graph_file_add):
        # Read offline graph files
        self.inference_graph_file_add = inference_graph_file_add
        
        with open(self.inference_graph_file_add, 'r') as f:
            # For test purpose
            self.inference_set = json.load(f)
        return self.inference_set