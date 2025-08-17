# ---------------------------- External Imports ----------------------------
import numpy as np
import torch

# ---------------------------- Internal Imports ----------------------------
from .base_embedder import BaseEmbedder
from .utils import cosine_similarity

# ---------------------------- ConceptEmbedder Class ----------------------------
class ConceptEmbedder(BaseEmbedder):
    """
    Extends BaseEmbedder to create higher-level concept embeddings by combining
    embeddings of multiple words. Also made callable to return embeddings from token IDs.
    """

    def embed_concept(self, words):
        """
        Creates a concept embedding by averaging the embeddings of constituent words.

        Args:
            words (list of str): Words that make up the concept.

        Returns:
            np.ndarray: Concept embedding vector.
        """
        vectors = []
        for word in words:
            vec = self.get_vector(word)
            if np.any(vec):  # Ignore zero vectors (OOV words)
                vectors.append(vec)
        if not vectors:
            # Return zero vector if no valid words
            return np.zeros(self.embedding_dim)
        # Average vectors to form concept embedding
        concept_vector = np.mean(vectors, axis=0)
        return concept_vector

    def similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1 (np.ndarray): First vector.
            vec2 (np.ndarray): Second vector.

        Returns:
            float: Cosine similarity score.
        """
        return cosine_similarity(vec1, vec2)

    def nearest_concepts(self, concept_vector, concept_dict, top_k=5):
        """
        Find the top_k nearest concepts from a dictionary of concepts to the given concept vector.

        Args:
            concept_vector (np.ndarray): Query concept vector.
            concept_dict (dict): Dictionary with concept names as keys and vectors as values.
            top_k (int): Number of nearest concepts to return.

        Returns:
            list of tuples: List of (concept_name, similarity_score), sorted by score descending.
        """
        similarities = []
        for concept_name, vec in concept_dict.items():
            sim = self.similarity(concept_vector, vec)
            similarities.append((concept_name, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    # ---------------------------- Make Embedder Callable ----------------------------
    def __call__(self, token_input):
        """
        Converts a list or tensor of token IDs into an embedding tensor.

        Args:
            token_input (list or torch.Tensor): Token IDs or strings.

        Returns:
            torch.Tensor: Embedding tensor of shape (1, embedding_dim).
        """
        # Convert tensor to list if needed
        if isinstance(token_input, torch.Tensor):
            token_input = token_input.tolist()[0]

        # Convert integers to string tokens for dummy embeddings
        words = [str(t) for t in token_input] if all(isinstance(t, int) for t in token_input) else token_input

        # Get concept embedding
        emb = self.embed_concept(words)

        # Return as torch tensor
        return torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
