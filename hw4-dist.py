import numpy as np
from Bio import PDB
import numpy as np
import time


# Original code by Talal Zoabi
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder
import numpy as np

# Get the coordinates of the structure
def get_structure_coords(structure):
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_coord())
                    print(atom.get_coord())
    return np.array(coords)

def calc_distance_matrix(coords):
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j % 100 == 0:
                print(f"Atom {i} to atom {j}")
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return dist_matrix

start_time = time.time()

pdb_parser = PDBParser(QUIET=True)
structure = pdb_parser.get_structure("2e0p", "HW1/pdb/2e0p.pdb")
coords = get_structure_coords(structure)

print(len(coords))
print({"number of atoms": len(coords)})
print("size of distance matrix: ", len(coords)*len(coords))


dist_matrix = calc_distance_matrix(coords)
rank = np.linalg.matrix_rank(dist_matrix)

print(f"Rank of the distance matrix: {rank}")

end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time} seconds")

np.save("dist_matrix.npy", dist_matrix)

