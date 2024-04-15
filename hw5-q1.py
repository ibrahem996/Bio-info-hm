import torch
from transformers import AutoTokenizer, AutoModel
from Bio.PDB import PDBParser, PPBuilder
from Bio import BiopythonWarning
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

# Suppress specific Biopython warnings about PDB construction
warnings.simplefilter('ignore', BiopythonWarning)

# GLOBAL VARIABLES
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'




# Function to extract sequences from a PDB file
def get_sequence_from_pdb(pdb_path):
    parser = PDBParser(QUIET=True)  # Quiet mode will suppress the warnings
    structure = parser.get_structure('protein', pdb_path)
    for model in structure:
        for chain in model:
            ppb = PPBuilder()
            for pp in ppb.build_peptides(chain):
                return str(pp.get_sequence())


# Function to compute embeddings
def compute_embeddings(sequence, model, tokenizer):
    # Tokenize the sequence and get embeddings from the model
    tokens = tokenizer(text=sequence, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
    # Remove batch dimension and truncate to actual sequence length
    return outputs.last_hidden_state.squeeze(0)[:len(sequence)]



def heatmap(embeddings):
    # Compute distance matrix for each protein and plot heatmap
    for pdb_id, embedding in embeddings.items():
        # Ensure the embedding is 2D
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)

        # Calculate pairwise distances
        distance_matrix = squareform(pdist(embedding.numpy(), 'euclidean'))

        plt.figure(figsize=(10, 8))
        sns.heatmap(distance_matrix, cmap="viridis")
        plt.title(f'Distance Matrix Heatmap for Protein {pdb_id}')
        plt.savefig(f'{pdb_id}_heatmap.png')
        plt.close()

    print("Protein analysis completed and heatmaps saved.")


def embed_aa(model, tokenizer):
    return np.array([compute_embeddings(aa, model, tokenizer)[0] for aa in amino_acids])


def embed_protein(model, tokenizer, pdb_files):
     # Load PDB files and calculate embeddings
    sequences = {}
    embeddings = {}

    # Define the path to the directory containing the PDB files
    pdb_dir = 'HW1/pdb'  # Assuming the script is run from the directory hw5 where 'pdb' is a subdirectory
    pdb_files = {
        '2e0p': os.path.join(pdb_dir, '2e0p.pdb'),
        '2vp4': os.path.join(pdb_dir, '2vp4.pdb'),
        '4yt2': os.path.join(pdb_dir, '4yt2.pdb')
    }

    for pdb_id, pdb_path in pdb_files.items():
        sequence = get_sequence_from_pdb(pdb_path)
        if not sequence:
            print(f"No sequence found for {pdb_id} at {pdb_path}")
            continue
        sequences[pdb_id] = sequence
        embeddings[pdb_id] = compute_embeddings(sequence, model, tokenizer)

    print(embeddings)


def plot_aa_distribution(distance_matrix):
    for i, aa in enumerate(amino_acids):
        distances = np.delete(distance_matrix[i], i)
        
        # Calculate the histogram manually
        counts, bin_edges = np.histogram(distances, bins=20, density=True)
        probabilities = counts / counts.sum()  # Normalize counts to probabilities
        
        # Plotting the probabilities as a histogram
        plt.figure(figsize=(10, 8))
        plt.bar((bin_edges[:-1] + bin_edges[1:]) / 2, probabilities, align='center', width=np.diff(bin_edges), alpha=0.7, color='blue')
        plt.title(f'Probability Distribution of Distances from {aa}')
        plt.xlabel('Distance')
        plt.ylabel('Probability')
        plt.grid(True)
        
        # Save each plot as a PNG
        plt.savefig(f'side/hw5-distribution-aa/{aa}_distance_distribution.png')
        plt.close()


def aa_mean_distances(distance_matrix):
    mean_distances = np.zeros(len(amino_acids))

    # Calculate the mean distance for each amino acid, excluding the distance to itself
    for i in range(len(amino_acids)):
        # Extract distances excluding the distance to itself
        distances = np.delete(distance_matrix[i], i)
        
        # Compute the mean of these distances
        mean_distances[i] = np.mean(distances)

    return mean_distances


def plot_aa_distance_heatmap(distance_matrix):
    # Create a heatmap of the distance matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, annot=True, fmt=".1f", cmap='viridis',
                xticklabels=amino_acids, yticklabels=amino_acids)
    plt.title('Heatmap of Amino Acid Distance Matrix')
    plt.savefig('side/distance_matrix_heatmap.png')
    plt.show()

def plot_aa_heatmap(distance_matrix):
    for i, aa in enumerate(amino_acids):
        # Initialize a new figure
        plt.figure(figsize=(8, 6))

        # Extract the distance array for the specific amino acid
        # Note: We use [np.newaxis] to keep the matrix two-dimensional
        specific_distances = distance_matrix[i, np.newaxis]

        # Create a heatmap for the specific amino acid
        sns.heatmap(specific_distances, annot=True, cmap='viridis', fmt=".1f",
                    cbar_kws={'label': 'Distance'},
                    xticklabels=amino_acids, yticklabels=[aa])

        plt.title(f'Heatmap of Distances from Amino Acid {aa}')
        plt.xlabel('Amino Acids')
        plt.ylabel('Specific Amino Acid')

        # Save the heatmap as a PNG file
        plt.savefig(f'side/hw5-aa-heatmaps/heatmap_distance_from_{aa}.png')
        plt.close()



def main():
    # Load the ESM model
    model_name = 'Rostlab/prot_bert'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Embed amino acids
    aa_embeddings = embed_aa(model, tokenizer)
   
    distance_matrix = squareform(pdist(aa_embeddings, 'euclidean'))
    print(distance_matrix)

    # plot_aa_distribution(distance_matrix)
    # plot_aa_distance_heatmap(distance_matrix)

    plot_aa_heatmap(distance_matrix)




    
if __name__ == "__main__":
    main()