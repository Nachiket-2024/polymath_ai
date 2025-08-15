# ---------------------------- External Imports ----------------------------
import numpy as np

# ---------------------------- Memory Class ----------------------------
class MemoryNetwork:
    """
    A simple memory module that stores items and retrieves the most similar
    one based on a similarity function.
    """

    def __init__(self, size, similarity_function=None):
        """
        Initialize memory.

        Args:
            size (int): Maximum number of items to store.
            similarity_function (callable): Function to compute similarity between query and memory item.
        """
        self.size = size
        self.memory = []
        # Default to cosine similarity if none provided
        self.similarity_function = similarity_function or self.cosine_similarity

    def store(self, item):
        """
        Store an item in memory. If size limit exceeded, remove oldest.

        Args:
            item (np.ndarray): The item (e.g., embedding vector) to store.
        """
        self.memory.append(item)
        if len(self.memory) > self.size:
            self.memory.pop(0)

    def retrieve(self, query):
        """
        Retrieve the most similar memory item to the query.

        Args:
            query (np.ndarray): Query vector.

        Returns:
            np.ndarray: Most similar stored item.
        """
        if not self.memory:
            return None

        similarities = [self.similarity_function(query, item) for item in self.memory]
        best_idx = int(np.argmax(similarities))
        return self.memory[best_idx]

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1 (np.ndarray): First vector.
            vec2 (np.ndarray): Second vector.

        Returns:
            float: Cosine similarity score.
        """
        dot = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norm_product == 0:
            return 0.0
        return dot / norm_product
