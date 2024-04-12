import numpy as np
from Bio import PDB
import numpy as np

# Original code by Talal Zoabi
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder
import numpy as np

# Function to apply rotation to the structure
def apply_rotation(structure, rotation_matrix):
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.transform(rotation_matrix, np.array([0, 0, 0]))

# Load the structure from the PDB file
pdb_parser = PDBParser(QUIET=True)
structure = pdb_parser.get_structure("2e0p", "HW1/pdb/2e0p.pdb")

# Calculated in Q2
rotation_matrix = [
    [0.8660254, -0.2236068, 0.4472136],
    [0.2236068, 0.97320508, 0.05358984],
    [-0.4472136, 0.05358984, 0.89282032],
]


# Apply rotation
apply_rotation(structure, rotation_matrix)

# Save the modified structure to a new PDB file
io = PDBIO()
io.set_structure(structure)
io.save("modified_structure.pdb")


# plot_rama("2e0p", "pdb/2e0p.pdb")
# plot_rama("2vp4", "pdb/2vp4.pdb")
# plot_rama("4yt2", "pdb/4yt2.pdb")
