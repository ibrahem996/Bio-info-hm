import torch
from transformers import EsmModel, EsmTokenizer
import numpy as np

# Define the standard amino acids
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

# Load the ESM model and tokenizer
model_name = "facebook/esm-1b"  # You can choose different models as per requirement
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmModel.from_pretrained(model_name)

# Function to embed an amino acid
def embed_amino_acid(aa):
    # Tokenize the amino acid
    tokens = tokenizer(aa, return_tensors="pt")
    
    # Get model output
    with torch.no_grad():
        output = model(**tokens, return_dict=True)
    
    # Extract the last hidden states of the sequence's first token (the amino acid)
    last_hidden_states = output.last_hidden_state[:, 1, :]  # Skip the start token ([CLS])
    return last_hidden_states.squeeze().numpy()

# Embed each amino acid and collect the embeddings
embeddings = np.array([embed_amino_acid(aa) for aa in amino_acids])

# Save the embeddings to a file
np.save('amino_acid_embeddings.npy', embeddings)

print("Embeddings saved successfully.")
