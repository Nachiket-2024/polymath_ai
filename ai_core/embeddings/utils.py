# ---------------------------- External Imports ----------------------------
import numpy as np

# ---------------------------- Utility Functions ----------------------------

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity score between -1 and 1.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        # Handle zero vectors
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to have unit length.

    Args:
        vec (np.ndarray): Input vector.

    Returns:
        np.ndarray: Normalized vector.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm
