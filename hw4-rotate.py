

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder
import numpy as np

# Original code by Talal Zoabi

# Function to apply rotation to the structure
def apply_rotation(structure, rotation_matrix):
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.transform(rotation_matrix, np.array([0, 0, 0]))


def rotate_pdb(name, pdb_file):
    # Load the structure from the PDB file
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure(name, pdb_file)

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
    io.save(f"rotated-seq/rotated_{name}.pdb")


rotate_pdb("2e0p", "HW1/pdb/2e0p.pdb")
rotate_pdb("2vp4", "HW1/pdb/2vp4.pdb")
rotate_pdb("4yt2", "HW1/pdb/4yt2.pdb")
