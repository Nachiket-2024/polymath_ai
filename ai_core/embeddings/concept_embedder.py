# ---------------------------- External Imports ----------------------------
import numpy as np

# ---------------------------- Internal Imports ----------------------------
from .base_embedder import BaseEmbedder
from .utils import cosine_similarity

# ---------------------------- ConceptEmbedder Class ----------------------------
class ConceptEmbedder(BaseEmbedder):
    """
    Extends BaseEmbedder to create higher-level concept embeddings by combining
    embeddings of multiple words.
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
