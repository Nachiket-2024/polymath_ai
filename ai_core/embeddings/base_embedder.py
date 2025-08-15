# ---------------------------- External Imports ----------------------------
import os
import numpy as np

# ---------------------------- BaseEmbedder Class ----------------------------
class BaseEmbedder:
    """
    Loads and manages word embeddings.
    Provides methods to get vector for a word and check word presence.
    Supports loading from a pretrained embedding file (e.g., GloVe format).
    """

    def __init__(self, embedding_path: str):
        """
        Initialize the embedder by loading embeddings from the given file path.

        Args:
            embedding_path (str): Path to the pretrained embeddings file.
        """
        self.embedding_path = embedding_path
        self.embedding_dim = None
        self.word_to_vec = {}
        self._load_embeddings()

    def _load_embeddings(self):
        """
        Loads embeddings from a text file with each line containing a word followed by its vector.
        Assumes a whitespace-separated format.
        """
        if not os.path.isfile(self.embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {self.embedding_path}")

        with open(self.embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                if len(values) < 10:
                    # Skip bad lines or header
                    continue
                word = values[0]
                vector = np.asarray(values[1:], dtype=np.float32)
                if self.embedding_dim is None:
                    self.embedding_dim = len(vector)
                self.word_to_vec[word] = vector

    def get_vector(self, word: str) -> np.ndarray:
        """
        Returns the embedding vector for the given word.
        Returns a zero vector if the word is not in the vocabulary.

        Args:
            word (str): Word to get embedding vector for.

        Returns:
            np.ndarray: Embedding vector.
        """
        if word in self.word_to_vec:
            return self.word_to_vec[word]
        else:
            # Return zero vector for out-of-vocab words
            return np.zeros(self.embedding_dim)

    def has_word(self, word: str) -> bool:
        """
        Checks if the word exists in the embedding vocabulary.

        Args:
            word (str): Word to check.

        Returns:
            bool: True if word exists, False otherwise.
        """
        return word in self.word_to_vec
