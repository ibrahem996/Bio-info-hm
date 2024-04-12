import numpy as np
from Bio.PDB import PDBParser, PPBuilder
from matplotlib import pyplot as plt
import numpy as np

# Credit to GitHub user Zuricho for providing base code to automate plotting of protein
# Original code by Zuricho (https://github.com/Zuricho/Ramachandran_plot)
# Edited by Talal Zoabi (SuperSpawn)


def plot_rama(name, pdb_file):
    parser = PDBParser()
    structure = parser.get_structure(name, pdb_file)

    ppbuilder = PPBuilder()
    polypeptides = ppbuilder.build_peptides(structure)

    phi_psi_angles = []
    for poly in polypeptides:
        phi_psi = poly.get_phi_psi_list()
        phi_psi_angles.extend(phi_psi)

    phi_psi_array = np.array(phi_psi_angles)[1:-1]

    plt.figure(figsize=(8, 8))

    plt.scatter(phi_psi_array[:, 0],
                phi_psi_array[:, 1], s=4, c="k", marker="o")

    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)

    plt.xlabel("Phi (radians)")
    plt.ylabel("Psi (radians)")
    plt.title("Ramachandran Plot (Radians)")
    plt.show()


plot_rama("2e0p", "pdb/2e0p.pdb")
plot_rama("2vp4", "pdb/2vp4.pdb")
plot_rama("4yt2", "pdb/4yt2.pdb")
