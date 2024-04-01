import pickle
import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import train_test_split

PROT_STATES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
PROT_MAP = {state: i for i, state in enumerate(PROT_STATES)}

SS_STATES = ['H', 'E', 'C']
N_STATES = 3
SS_MAP = {state: i for i, state in enumerate(SS_STATES)}

def seq_file_to_json(file_path: str):
    with open(file_path, 'r') as f:
        segments = f.read().split('>')
    params = {}

    for seg in segments:
        if len(seg) == 0:
            continue
        name, value = seg.split('\n', 1)
        params[name] = value.replace('\n', '').replace(' ', '')

    return params

def load_prot_list():
    with open("hw4/list_of_prots", 'r') as f:
        list_of_prots = f.read().splitlines()
    return list_of_prots

def load_dataset():
    data =  []
    prots_list = load_prot_list()
    for prot in prots_list:
        data.append(seq_file_to_json('hw4/' + prot))  # Modified file path to include 'hw4/'
    return data

def extract_ss(data):
    ss =  [x['ss_dssp'] for x in data]
    ss = [map_ss(x) for x in ss]
    ss = [np.array(x).reshape(-1, 1) for x in ss]
    return ss

def extract_seq(data):
    seq = [x[list(x.keys())[-1]] for x in data]
    seq = [map_seq(x) for x in seq]
    seq = [np.array(x).reshape(-1, 1) for x in seq]
    return seq

def map_ss(ss):
    return [SS_MAP[state] for state in ss]

def map_seq(seq):
    return [PROT_MAP[state] for state in seq]

# Prepare the dataset
dataset = load_dataset()
X_seq = extract_seq(dataset)
Y_ss = extract_ss(dataset)

# Split dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_seq, Y_ss, test_size=0.2, random_state=22)

# Initialize and train the HMM model
model = hmm.GaussianHMM(n_components=N_STATES, n_iter=10)
X_train_concat = np.concatenate(X_train)
lengths = [len(x) for x in X_train]
model.fit(X_train_concat, lengths)
print(lengths)

# Evaluate the model on the test set
predicted_ss = [model.predict(y) for y in X_test]

# Calculate accuracy
accuracy = np.mean([np.mean(pred == actual) for pred, actual in zip(predicted_ss, Y_test)]) * 100
print(f"Accuracy: {accuracy}%")

# Print predicted and actual sequences
for pred, actual in zip(predicted_ss, Y_test):
    print("Predicted sequence:", pred.flatten())
    print("Actual sequence:", actual.flatten())
    print()
    
# Save model to file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the model from file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
