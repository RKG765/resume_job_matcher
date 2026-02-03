"""
TF-IDF Vectorizer Module
Implements TF-IDF document representation for IR system.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidf
from typing import List, Dict, Tuple, Optional
import pickle
import os


class TFIDFVectorizer:
    """
    TF-IDF Vectorizer for document representation.
    Wraps scikit-learn's TfidfVectorizer with additional functionality.
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 1, max_df: float = 0.95):
        """
        Initialize TF-IDF Vectorizer.
        
        Args:
            max_features: Maximum number of features (vocabulary size)
            ngram_range: Range of n-grams to include (default: unigrams and bigrams)
            min_df: Minimum document frequency for a term
            max_df: Maximum document frequency (proportion) for a term
        """
        self.vectorizer = SklearnTfidf(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            sublinear_tf=True  # Use 1 + log(tf) instead of raw tf
        )
        self.is_fitted = False
        self.document_vectors = None
        self.document_ids = []
    
    def fit(self, documents: List[str], document_ids: Optional[List[str]] = None):
        """
        Fit the vectorizer on a corpus of documents.
        
        Args:
            documents: List of preprocessed document texts
            document_ids: Optional list of document identifiers
        """
        self.vectorizer.fit(documents)
        self.is_fitted = True
        
        if document_ids:
            self.document_ids = document_ids
        else:
            self.document_ids = [f"doc_{i}" for i in range(len(documents))]
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to TF-IDF vectors.
        
        Args:
            documents: List of preprocessed document texts
            
        Returns:
            Sparse matrix of TF-IDF vectors
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        return self.vectorizer.transform(documents)
    
    def fit_transform(self, documents: List[str], 
                      document_ids: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            documents: List of preprocessed document texts
            document_ids: Optional list of document identifiers
            
        Returns:
            Sparse matrix of TF-IDF vectors
        """
        self.fit(documents, document_ids)
        self.document_vectors = self.transform(documents)
        return self.document_vectors
    
    def get_feature_names(self) -> List[str]:
        """Get the vocabulary (feature names)."""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_top_terms(self, vector: np.ndarray, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N terms from a TF-IDF vector.
        
        Args:
            vector: TF-IDF vector (sparse or dense)
            n: Number of top terms to return
            
        Returns:
            List of (term, weight) tuples sorted by weight descending
        """
        feature_names = self.get_feature_names()
        
        # Handle sparse matrix
        if hasattr(vector, 'toarray'):
            vector = vector.toarray().flatten()
        else:
            vector = np.asarray(vector).flatten()
        
        # Get indices of top N values
        top_indices = vector.argsort()[-n:][::-1]
        
        return [(feature_names[i], vector[i]) for i in top_indices if vector[i] > 0]
    
    def get_document_vector(self, doc_id: str) -> Optional[np.ndarray]:
        """
        Get the TF-IDF vector for a specific document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            TF-IDF vector or None if not found
        """
        if doc_id in self.document_ids:
            idx = self.document_ids.index(doc_id)
            return self.document_vectors[idx]
        return None
    
    def save(self, filepath: str):
        """Save the vectorizer to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'is_fitted': self.is_fitted,
                'document_vectors': self.document_vectors,
                'document_ids': self.document_ids
            }, f)
    
    def load(self, filepath: str):
        """Load the vectorizer from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.is_fitted = data['is_fitted']
            self.document_vectors = data['document_vectors']
            self.document_ids = data['document_ids']


def compute_tfidf(documents: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function to compute TF-IDF for a list of documents.
    
    Returns:
        Tuple of (tfidf_matrix, feature_names)
    """
    vectorizer = TFIDFVectorizer()
    matrix = vectorizer.fit_transform(documents)
    features = vectorizer.get_feature_names()
    return matrix, features
