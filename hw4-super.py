import numpy as np


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

# Example atom sets
set1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
set2 = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])

aligned_set, rmsd_value = optimal_superposition(set1, set2)
print("Aligned Set 1:")
print(aligned_set)
print("RMSD:", rmsd_value)



