# This code was made by following the tutorial at: https://www.geeksforgeeks.org/hidden-markov-model-in-machine-learning/
# we used the idea of the tutorial to create a simple example of a hidden markov model

# import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from hmmlearn import hmm


# CODE
states = ["helix", 'sheet', 'loop']
ab_states = ['H', 'E', 'C']
n_states = len(states)

# observations are proteins
observations = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
n_observations = len(observations)


# Chosen at random
state_probability = np.array([0.3, 0.2, 0.5])
transition_probability = np.array([[0.5, 0.1, 0.4],
                                    [0.1, 0.49, 0.41],
                                    [0.333, 0.333, 0.334]])


# make the emission probability matrix with random values
emission_probability = np.random.rand(n_states, n_observations)
emission_probability = emission_probability / emission_probability.sum(axis=1)[:, None]


model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = state_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

######################



def save_result(seq, hidden_states, viterby_hidden_states, log_prob, prob_matrix, filename):
    f = open(filename, "w")
    f.write("===========================================\nSequence:\n\n")
    for i in seq:
        f.write(observations[i[0]])
    f.write("\n\n===========================================\n\nHidden states:\n\n")
    for i in hidden_states:
        f.write(ab_states[i])    
    f.write("\n\n===========================================\n\nViterby Hidden states:\n\n")
    f.write("log probability: " + str(log_prob) + "\n")
    for i in viterby_hidden_states:
        f.write(ab_states[i])
    f.write("\n\n===========================================\n\nProbability matrix:\n\n")
    f.write("Helix\tSheet\tLoop\n")
    for i in prob_matrix:
        for j in i:
            # print only 3 digits after the decimal point
            f.write("{:.3f}".format(j) + "\t")
        f.write("\n")
    f.close()



# the HMM generates a sequence of 10000 proteins 
seq, hidden_states = model.sample(10000)

log_probability, viterby_hidden_states = model.decode(seq,
											lengths = len(seq),
											algorithm ='viterbi' )


# hmmlearn docs
_, prob_matrix = model.score_samples(seq)

save_result(seq, hidden_states, viterby_hidden_states, log_probability,prob_matrix, "result.txt")


