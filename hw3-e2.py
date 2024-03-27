from hmmlearn import hmm
import numpy as np

### CONSTANTS
OBSERVATIONS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

AB_STATES = ['H', 'E', 'C']
N_STATES = 3

f = open("hw4/list_of_prots", 'r')
list_of_prots = f.read().splitlines()
f.close()

# shuffle the dataset
shuffled_prots = np.random.permutation(list_of_prots)
# shuffled_prots = list_of_prots

def load_training_data(lim: int = 80):
    ss_data = []
    prot_data = []
    for i in range(0, lim):
        prot = shuffled_prots[i]
        raw_text = open("hw4/" + prot, 'r').read().split('>')
        ss_str = raw_text[1][7:].replace('\n', '').replace(' ', '')
        prot_seq = raw_text[-1][7:].replace('\n', '').replace(' ', '')
        ss_data.append(ss_str)
        prot_data.append(prot_seq)
    return prot_data, ss_data

def load_testing_data(start:int = 80):
    ss_data = []
    prot_data = []
    for i in range(start, len(shuffled_prots)):
        prot = shuffled_prots[i]
        raw_text = open("hw4/" + prot, 'r').read().split('>')
        ss_str = raw_text[1][7:].replace('\n', '').replace(' ', '')
        prot_seq = raw_text[-1][7:].replace('\n', '').replace(' ', '')
        ss_data.append(ss_str)
        prot_data.append(prot_seq)
    return prot_data, ss_data

# TODO
def encode_data(prot_data, ss_data):
    encoded_ss = np.array([])
    encoded_prot = np.array([])
    for i in range(len(prot_data)):
        prot = prot_data[i]
        ss = ss_data[i]
        curr_encoded_prot = np.array([])
        curr_encoded_ss = np.array([])
        for j in range(len(prot)):
            curr_encoded_prot += [OBSERVATIONS.index(prot[j])]
            curr_encoded_ss += [AB_STATES.index(ss[j])]
        encoded_prot += curr_encoded_prot
        encoded_ss += curr_encoded_ss
    return encoded_prot, encoded_ss


prot_data, ss_data = load_training_data(80)
encoded_prot, encoded_ss = encode_data(prot_data, ss_data)

print(encoded_prot)

X = np.array(encoded_prot)
lengths = [len(seq) for seq in encoded_prot]
y = np.array(encoded_ss)


model = hmm.MultinomialHMM(n_components=N_STATES, n_iter=100)
model.fit(X, lengths)


test_prot_data, test_ss_data = load_testing_data(80)
encoded_test_prot, encoded_test_ss = encode_data(test_prot_data, test_ss_data)

for i in range(len(encoded_test_prot)):
    seq = encoded_test_prot[i]
    log_probability, viterby_hidden_states = model.decode([seq], lengths = len(seq), algorithm ='viterbi' )
    print("log probability: ", log_probability)
    print("Viterby Hidden states: ", viterby_hidden_states)
    print("True Hidden states: ", encoded_test_ss[i])
    print("===========================================")
    print("\n")

