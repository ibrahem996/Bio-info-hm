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

# Function to extract sequences from a PDB file
def get_sequence_from_pdb(pdb_path):
    parser = PDBParser(QUIET=True)  # Quiet mode will suppress the warnings
    structure = parser.get_structure('protein', pdb_path)
    for model in structure:
        for chain in model:
            ppb = PPBuilder()
            for pp in ppb.build_peptides(chain):
                return str(pp.get_sequence())

# Load the ESM model
model_name = 'Rostlab/prot_bert'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to compute embeddings
def compute_embeddings(sequence, model, tokenizer):
    # Tokenize the sequence and get embeddings from the model
    tokens = tokenizer(text=sequence, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
    # Remove batch dimension and truncate to actual sequence length
    return outputs.last_hidden_state.squeeze(0)[:len(sequence)]

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

