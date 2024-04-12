import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Bio.PDB import PDBParser, PPBuilder
from matplotlib import pyplot as plt
import numpy as np
import math
from Bio.Align import substitution_matrices

##############################################################
# Written by:                                                #
# Talal Zoabi - 213603608                                    #
# Ibrahem Hmad - 316572619                                   #
##############################################################


# ChatGPT generic code to load BLOSUM
blosum_matrix = substitution_matrices.load("BLOSUM62")


# Credit to GitHub user Zuricho for providing base code to automate plotting of protein
# Original code by Zuricho (https://github.com/Zuricho/Ramachandran_plot)
def plot_aa_avg_phi_psi(aa_buffer):
    plt.figure(figsize=(8, 8))

    avgs = [aa_buffer[aa]['avg'] for aa in aa_buffer]

    for avg in avgs:
        plt.scatter(avg[0],
                    avg[1], s=4, c="k", marker="o")

    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)

    plt.xlabel("Phi (radians)")
    plt.ylabel("Psi (radians)")
    plt.title("Phi Psi amino acids averages (Radians)")
    plt.savefig('assets/aa-avg-phi-psi.png')


def build_distance_matrix(aa_buffer):
    distance_matrix = np.zeros((20, 20))

    for i, aa in enumerate(aa_buffer):
        for j, bb in enumerate(aa_buffer):
            if aa == bb:
                continue
            distance_matrix[i][j] = math.dist(
                aa_buffer[aa]['avg'], aa_buffer[bb]['avg'])
    return distance_matrix


def compare_matrices(distance, blosum):
    pass


# Edited code originally generate dby Google Bard
def plot_phi_psi_by_aa(aa_buffer):
    for aa in aa_buffer:
        plt.figure(figsize=(8, 8))

        for phi_psi in aa_buffer[aa]['phi_psi']:
            plt.scatter(phi_psi[0],
                        phi_psi[1], s=4, c="k", marker="o")

        plt.xlim(-np.pi, np.pi)
        plt.ylim(-np.pi, np.pi)

        plt.xlabel("Phi (radians)")
        plt.ylabel("Psi (radians)")
        plt.title(f"Phi Psi distribution for {aa}")
        plt.savefig(f'assets/{aa}-dist-fig.png')


def print_distance_matrix(distance_matrix):
    for i in distance_matrix:
        for j in i:
            print(f"{j}", end=' ')
        print()


def save_distance_matrix(distance_matrix, path):
    f = open(path, 'w')
    for i in distance_matrix:
        for j in i:
            f.write(f"{j:.3f} ")
        f.write('\n')
    f.close()


# Heavily edited code originally from Zuricho GitHub repo
def get_phi_psi_by_aa(name, pdb_file):
    # From Zurchino
    parser = PDBParser()
    structure = parser.get_structure(name, pdb_file)

    ppbuilder = PPBuilder()
    polypeptides = ppbuilder.build_peptides(structure)

    # Original code
    aa_angles = {}
    for poly in polypeptides:
        seq = poly.get_sequence()
        phi_psi = poly.get_phi_psi_list()
        for i in range(len(phi_psi)):
            if seq[i] not in aa_angles:
                aa_angles[seq[i]] = {'sum': [0, 0], 'phi_psi': []}
            aa_angles[seq[i]]['phi_psi'].append(phi_psi[i])
            if phi_psi[i][0] is not None and phi_psi[i][1] is not None:
                aa_angles[seq[i]]['sum'][0] += phi_psi[i][0]
                aa_angles[seq[i]]['sum'][1] += phi_psi[i][1]

    for aa in aa_angles:
        aa_angles[aa]['avg'] = [aa_angles[aa]['sum'][0] / len(aa_angles[aa]['phi_psi']),
                                aa_angles[aa]['sum'][1] / len(aa_angles[aa]['phi_psi'])]
        del aa_angles[aa]['sum']

    return aa_angles


aa_buffer = get_phi_psi_by_aa("2e0p", "pdb/2e0p.pdb")
plot_aa_avg_phi_psi(aa_buffer)


matrix = build_distance_matrix(aa_buffer)
save_distance_matrix(matrix, 'matrix.txt')

# ChatGPT generic code
distance_heatmap = sns.heatmap(matrix, annot=True, cmap='viridis')
plt.title('Distance Heatmap Matrix')
plt.savefig('assets/distance-matrix-heatmap.png')

blosum_heatmap = sns.heatmap(blosum_matrix, cmap='viridis', annot=True,
                             cbar=False)
plt.title('BLOSUM Matrix')
plt.savefig('assets/blosum-matrix-heatmap.png')

print_distance_matrix(blosum_matrix)


correlation_coefficient = np.corrcoef(
    distance_heatmap, blosum_heatmap)[0, 1]
mse = np.mean((matrix - blosum_matrix)**2)

print(f'Correlation Coefficient: {correlation_coefficient}')
print(f'Mean Squared Error: {mse}')
