from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder
import numpy as np


# Original code by Talal Zoabi


def get_CA_atoms_coordinates(name, pdb_file):
    parser = PDBParser()
    structure = parser.get_structure(name, pdb_file)

    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ' and residue.get_resname() != 'HOH':  # Exclude water molecules
                    ca_atom = residue['CA']
                    ca_atoms.append(ca_atom.get_coord())

    return ca_atoms



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





def extract_coordinates(D):
    n = D.shape[0]
    # Squared distance matrix
    D2 = D**2
    ones = np.ones((n, n))
    J = np.eye(n) - ones / n
    G = -0.5 * J @ D2 @ J
    
    # Perform SVD
    U, sigma, Vt = np.linalg.svd(G)
    # Check for negative eigenvalues and set them to zero
    sigma[sigma < 0] = 0
    # Extract the 3D coordinates
    X = U[:, :3] @ np.diag(np.sqrt(sigma[:3]))
    return X

def optimal_superposition(left, right):
    # Calculate centroids
    left_center = np.mean(left, axis=0)
    right_center = np.mean(right, axis=0)
    
    # Translate points to origin
    left_centered = left - left_center
    right_centered = right - right_center
    
    # Compute the correlation matrix
    H = left_centered.T @ right_centered
    
    # Singular Value Decomposition
    U, Sigma, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix using Kabsch algorithm
    d = np.linalg.det(Vt.T) * np.linalg.det(U)
    D = np.diag([1, 1, d]) 
    R = U @ D @ Vt
    
    # Apply rotation to set1
    left_rotated = left_centered @ R
    
    # Calculate RMSD
    rmsd = np.sqrt(np.mean(np.linalg.norm(left_rotated - right_centered, axis=1)**2))
    
    return left_rotated + right_center, rmsd  # Return the aligned set and the RMSD

def main():
    ca_coords = get_CA_atoms_coordinates("2e0p", "HW1/pdb/2e0p.pdb")
    distance_matrix = calculate_distance_matrix(ca_coords)

    print("CA atoms count: ", len(ca_coords))
    
    noisy_distance_matrix = add_noise(distance_matrix, 0.03)
    noisy_coords = extract_coordinates(noisy_distance_matrix)
    
    # aligned_coords, rmsd = optimal_superposition(ca_coords, noisy_coords)
    aligned_coords, rmsd = optimal_superposition(noisy_coords, ca_coords)

    print("Aligned Coordinates:")
    print(aligned_coords)
    print("RMSD:", rmsd)

    

if __name__ == "__main__":
    main()

