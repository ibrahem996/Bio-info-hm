from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder
import numpy as np


# Original code by Talal Zoabi

# Load sequence from PDB file
def load_pdb_structure(name, pdb_file):
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure(name, pdb_file)
    return structure

# Extract coordinates of each atom from structure
def get_structure_coords(structure):
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_coord())
    return np.array(coords)

# Calculate distance matrix from coordinates
def calculate_distance_matrix(coords):
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return distance_matrix

def add_noise(matrix, noise_level):
    n = matrix.shape[0]
    noise = np.random.normal(0, noise_level, (n, n))
    return matrix + noise

def main():
    matrix = np.array([[1,2,3], [4,5,6], [7,8,9]])
    print(add_noise(matrix, 0.1))

if __name__ == "__main__":
    main()

