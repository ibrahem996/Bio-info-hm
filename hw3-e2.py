import pickle
import numpy as np
from hmmlearn import hmm


PROT_STATES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
PROT_MAP = {state: i for i, state in enumerate(PROT_STATES)}

SS_STATES = ['H', 'E', 'C']
N_STATES = 3
SS_MAP = {state: i for i, state in enumerate(SS_STATES)}

# parse data from file
def seq_file_to_json(file_path: str):
    f = open('hw4/' + file_path, 'r')
    segments = f.read().split('>')
    f.close()
    params = {}

    for seg in segments:
        if len(seg) == 0:
            continue
        name, value = seg.split('\n', 1)
        params[name] = value.replace('\n', '').replace(' ', '')

    return params

def load_prot_list():
    f = open("hw4/list_of_prots", 'r')
    list_of_prots = f.read().splitlines()
    f.close()    
    return list_of_prots

def load_dataset():
    data =  []
    prots_list = load_prot_list()
    for prot in prots_list:
        data.append(seq_file_to_json(prot))
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

# HMM model
model = hmm.MultinomialHMM(n_components=N_STATES, n_iter=100)

# Prepare the training data
dataset = load_dataset()
# only want to use first 80 for training
X_seq = extract_seq(dataset[:80])

# Tweaked to concatenate all the sequences
X = np.concatenate(X_seq)
lengths = [len(x) for x in X_seq]

print(len(X))

# Shuffle X
np.random.shuffle(X)

for _ in range(5):
    # Model training
    model.fit(X, lengths)

# Model evaluation

prot_list = load_prot_list()

Y_seq = extract_seq(dataset[80:])
Y_ss = extract_ss(dataset[80:])


predicted_ss = []
for y in Y_seq:
    pred = model.predict(y)
    predicted_ss.append(pred)

equality = []
for i in range(len(Y_seq)):
    curr_equality = predicted_ss[i] == Y_ss[i]
    curr_equality = curr_equality.reshape(-1, 1)
    equality.append(curr_equality)
    print(f"Predicted: {predicted_ss[i].reshape(1,- 1)}")
    # print(f"Actual: {Y_ss[i].reshape(1, -1)}")
    print()

equality = np.concatenate(equality)
accuracy = np.average(equality) * 100
print(f"Accuracy: {accuracy}%")

# Save model to file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


# load the model from file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)



