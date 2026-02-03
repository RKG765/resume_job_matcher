"""
Similarity Module
Implements various similarity measures for document matching.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from typing import List, Tuple, Union
from scipy.sparse import issparse


class CosineSimilarity:
    """
    Cosine Similarity calculator for document matching.
    """
    
    @staticmethod
    def compute(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector (can be sparse or dense)
            vec2: Second vector (can be sparse or dense)
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Handle sparse matrices
        if issparse(vec1):
            vec1 = vec1.toarray().flatten()
        if issparse(vec2):
            vec2 = vec2.toarray().flatten()
        
        # Ensure 1D arrays
        vec1 = np.asarray(vec1).flatten()
        vec2 = np.asarray(vec2).flatten()
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    @staticmethod
    def compute_batch(query_vec: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between a query and multiple documents.
        
        Args:
            query_vec: Query vector (1, n_features)
            doc_vectors: Document vectors matrix (n_docs, n_features)
            
        Returns:
            Array of similarity scores
        """
        # Reshape query if needed
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        
        # Use sklearn for efficient batch computation
        similarities = sklearn_cosine(query_vec, doc_vectors)
        return similarities.flatten()


class JaccardSimilarity:
    """
    Jaccard Similarity for set-based matching (e.g., skills).
    """
    
    @staticmethod
    def compute(set1: set, set2: set) -> float:
        """
        Compute Jaccard similarity between two sets.
        
        Args:
            set1: First set of items
            set2: Second set of items
            
        Returns:
            Jaccard similarity score between 0 and 1
        """
        if not set1 and not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def compute_overlap(set1: set, set2: set) -> float:
        """
        Compute overlap coefficient (coverage of smaller set).
        
        Args:
            set1: First set (e.g., required skills)
            set2: Second set (e.g., candidate skills)
            
        Returns:
            Overlap coefficient between 0 and 1
        """
        if not set1:
            return 1.0  # No requirements means full match
        
        intersection = len(set1 & set2)
        return intersection / len(set1)


class WeightedSimilarity:
    """
    Weighted combination of different similarity measures.
    """
    
    def __init__(self, content_weight: float = 0.6, skill_weight: float = 0.4):
        """
        Initialize with weights for different components.
        
        Args:
            content_weight: Weight for content-based similarity
            skill_weight: Weight for skill-based similarity
        """
        self.content_weight = content_weight
        self.skill_weight = skill_weight
    
    def compute(self, content_sim: float, skill_sim: float) -> float:
        """
        Compute weighted similarity score.
        
        Args:
            content_sim: Content-based (TF-IDF) similarity
            skill_sim: Skill-based (Jaccard) similarity
            
        Returns:
            Weighted combined score
        """
        return (self.content_weight * content_sim + 
                self.skill_weight * skill_sim)


def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Convenience function for cosine similarity."""
    return CosineSimilarity.compute(vec1, vec2)
