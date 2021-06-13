# Dependencies: same as the original project
import json
import numpy as np
from collections import Counter
import random
import torch

train_path = '~/new_dataset/train_text.json'
dev_path = '~/new_dataset/dev_text.json'
test_path = '~/new_dataset/test_text.json'
train_text = json.load(open(train_path))
dev_text = json.load(open(dev_path))
test_text = json.load(open(test_path))

def get_top_k(outputs, text_set, k=5):
    '''
    Given the output tensor, return the top-k predictions for each sample. 
    '''
    predicted_indices = []
    for i in range(outputs.shape[0]):
        top_k_indices = np.argsort(-1*outputs[i,:])[:k]
        predicted_indices.append(top_k_indices)
    return predicted_indices
    
def test_statistics(outputs, text_set):

    # gather dataset statistics for query eligibility 
    query_count = Counter([sample['query'].split()[0] for sample in text_set])
    support_count = {sample['query'].split()[0]:0 for sample in text_set}
    candidate_count = {sample['query'].split()[0]:{} for sample in text_set}
    
    for sample in text_set:
        candidates = sample['candidates']
        query = sample['query'].split()[0]
        num_supports = len(sample['supports'])
        support_count[query] += num_supports
        for candidate in candidates:      
            if candidate in candidate_count[query]:   
                candidate_count[query][candidate] += 1
            else:
                candidate_count[query][candidate] = 0
    
    # keep queries with at least 50 supporting documents and at least 5 unique candidates. (as specified by De Cao)
    eligible_queries = []
    for query in query_count.keys():
        num_support = support_count[query]
        unique_candidates = np.count_nonzero(list(candidate_count[query].values()))
        if num_support >= 50 and unique_candidates >= 5:
            eligible_queries.append(query)
    
    predicted_indices_2 = get_top_k(outputs, text_set, k=2)
    predicted_indices_5 = get_top_k(outputs, text_set, k=5)
    
    query_correct = {query:0 for query in eligible_queries}
    query_correct_2 = {query:0 for query in eligible_queries}
    query_correct_5 = {query:0 for query in eligible_queries}
    
    # collect correct counts for each query
    for i,sample in enumerate(text_set):
        query = sample['query'].split()[0]
        if query in eligible_queries:
            correct_index = sample['candidates'].index(sample['answer'])
            if correct_index == predicted_indices_2[i][0]:
                query_correct[query] += 1
            if correct_index in predicted_indices_2[i]:
                query_correct_2[query] += 1
            if correct_index in predicted_indices_5[i]:
                query_correct_5[query] += 1
        
    # compute top-k
    query_accuracies = {query:query_correct[query]/query_count[query] for query in eligible_queries}
    query_accuracies_2 = {query:query_correct_2[query]/query_count[query] for query in eligible_queries}
    query_accuracies_5 = {query:query_correct_5[query]/query_count[query] for query in eligible_queries}

    sorted_accuracies = sorted(query_accuracies.items(), key=lambda item: item[1], reverse=True)

    
    # BEST 3
    print("3 BEST\n")
    for query in sorted_accuracies[0:3]:
        query = query[0]
        acc = query_accuracies[query]
        p_at_2 = query_accuracies_2[query]
        p_at_5 = query_accuracies_5[query]
        print(query, "\naccuracy: ", acc, "| P@2: ", p_at_2, "| P@5: ", p_at_5, "\n")
    
    # WORST 3
    print("3 WORST\n")
    for query in sorted_accuracies[-3:]:
        query = query[0]
        acc = query_accuracies[query]
        p_at_2 = query_accuracies_2[query]
        p_at_5 = query_accuracies_5[query]
        print(query, "\naccuracy: ", acc, "| P@2: ", p_at_2, "| P@5: ", p_at_5, "\n")
        
    # ENTIRE DATASET
    correct_1 = 0
    correct_2 = 0
    correct_5 = 0
    for i,sample in enumerate(text_set):
        correct_index = sample['candidates'].index(sample['answer'])
        if correct_index == predicted_indices_2[i][0]:
            correct_1 += 1
        if correct_index in predicted_indices_2[i]:
            correct_2 += 1
        if correct_index in predicted_indices_5[i]:
            correct_5 += 1
            
    print("ENTIRE DATASET")
    print("accuracy: ", correct_1/len(text_set), "| P@2: ", correct_2/len(text_set), "| P@5: ", correct_5/len(text_set), "\n")

def random_model(dataset):
    outputs = np.zeros((len(dataset), 70))
    num_correct = 0
    for i,sample in enumerate(dataset):
        candidates = sample['candidates']
        random_choice = random.choice(candidates)

        correct_index = sample['candidates'].index(sample['answer'])
        prediction_index = sample['candidates'].index(random_choice)
        
        outputs[i,prediction_index] = 1

        if random_choice == sample['answer']:
            num_correct += 1
                
    accuracy = num_correct/(len(dataset))
    print(accuracy)
    return outputs

random_train = random_model(train_text)
random_dev = random_model(dev_text)
random_test = random_model(test_text)

test_statistics(random_test, test_text)  

def max_mention(dataset):
    outputs = np.zeros((len(dataset), 70))
    num_correct = 0
    predictions = []
    for i,sample in enumerate(dataset):
        supports = sample['supports']
        candidates = sample['candidates']
        mention_count = {k:0 for k in candidates}
        for support in supports:
            for candidate in candidates:
                candidate_count = support.lower().count(candidate)
                mention_count[candidate] += candidate_count        
        
        prediction = max(mention_count)        
        correct_index = sample['candidates'].index(sample['answer'])
        
        prediction_index = sample['candidates'].index(prediction)
        outputs[i,prediction_index] = 1
        
        if prediction == sample['answer']:
            num_correct += 1
        
    accuracy = num_correct/(len(dataset))
    print(accuracy)
    return outputs
        
max_mention_output_train = max_mention(train_text)
max_mention_output_dev = max_mention(dev_text)
max_mention_output_test = max_mention(test_text)

test_statistics(max_mention_output_test, test_text)  

def candidates_per_query(dataset):
    query_memory = {}
    for sample in dataset:
        query = sample['query'].split()[0]
        if query not in query_memory:
            query_memory[query] = {}
        if sample['answer'] not in query_memory[query]:
            query_memory[query][sample['answer']] = 1
        else:
            query_memory[query][sample['answer']] += 1
        
    return query_memory
        
def majority_candidate_per_query(dataset):
    outputs = np.zeros((len(dataset), 70))
    num_correct = 0
    query_memory = candidates_per_query(train_text)
    for i,sample in enumerate(dataset):
        query = sample['query'].split()[0]
        candidates = sample['candidates']
        if query in query_memory: 
            counts = {}
            for candidate in candidates:
                get_count = query_memory[query].get(candidate)
                if get_count != None:
                    counts[candidate] = get_count
            if len(counts) > 0:       
                prediction = max(counts)
            else:
                prediction = random.choice(candidates)
        else:
            prediction = random.choice(sample['candidates'])
        if prediction == sample['answer']:
            num_correct += 1
            
        prediction_index = sample['candidates'].index(prediction)
        outputs[i,prediction_index] = 1
    
    return outputs
        
majority_train = majority_candidate_per_query(train_text), 
majority_dev = majority_candidate_per_query(dev_text)
majority_test = majority_candidate_per_query(test_text)

test_statistics(majority_test, test_text)  