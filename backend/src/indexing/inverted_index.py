"""
Inverted Index Module
Implements inverted index for efficient term-based document lookup.
"""

from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
import json
import os


class InvertedIndex:
    """
    Inverted Index for document retrieval.
    Maps terms to the documents containing them.
    """
    
    def __init__(self):
        # term -> {doc_id: term_frequency}
        self.index: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        # doc_id -> document metadata
        self.documents: Dict[str, dict] = {}
        
        # Document frequency for each term
        self.doc_freq: Dict[str, int] = defaultdict(int)
        
        # Total number of documents
        self.num_docs: int = 0
    
    def add_document(self, doc_id: str, tokens: List[str], metadata: Optional[dict] = None):
        """
        Add a document to the index.
        
        Args:
            doc_id: Unique document identifier
            tokens: List of tokens in the document
            metadata: Optional document metadata (title, category, etc.)
        """
        if doc_id in self.documents:
            # Remove old entry first
            self.remove_document(doc_id)
        
        # Count term frequencies
        term_freq = defaultdict(int)
        for token in tokens:
            term_freq[token] += 1
        
        # Update index
        for term, freq in term_freq.items():
            self.index[term][doc_id] = freq
            self.doc_freq[term] += 1
        
        # Store document metadata
        self.documents[doc_id] = metadata or {}
        self.documents[doc_id]['_token_count'] = len(tokens)
        
        self.num_docs += 1
    
    def remove_document(self, doc_id: str):
        """Remove a document from the index."""
        if doc_id not in self.documents:
            return
        
        # Remove from all posting lists
        for term in list(self.index.keys()):
            if doc_id in self.index[term]:
                del self.index[term][doc_id]
                self.doc_freq[term] -= 1
                
                # Clean up empty terms
                if not self.index[term]:
                    del self.index[term]
                    del self.doc_freq[term]
        
        del self.documents[doc_id]
        self.num_docs -= 1
    
    def search(self, query_tokens: List[str]) -> List[Tuple[str, float]]:
        """
        Search for documents containing query terms.
        Returns documents ranked by term overlap.
        
        Args:
            query_tokens: List of query tokens
            
        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        if not query_tokens:
            return []
        
        # Score documents by number of matching terms
        doc_scores: Dict[str, float] = defaultdict(float)
        
        for term in query_tokens:
            if term in self.index:
                for doc_id, freq in self.index[term].items():
                    # Simple scoring: term frequency
                    doc_scores[doc_id] += freq
        
        # Sort by score descending
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked
    
    def get_documents_with_term(self, term: str) -> Set[str]:
        """Get all document IDs containing a term."""
        if term in self.index:
            return set(self.index[term].keys())
        return set()
    
    def get_term_frequency(self, term: str, doc_id: str) -> int:
        """Get the frequency of a term in a document."""
        if term in self.index and doc_id in self.index[term]:
            return self.index[term][doc_id]
        return 0
    
    def get_document_frequency(self, term: str) -> int:
        """Get the number of documents containing a term."""
        return self.doc_freq.get(term, 0)
    
    def get_vocabulary(self) -> List[str]:
        """Get all unique terms in the index."""
        return list(self.index.keys())
    
    def get_document_metadata(self, doc_id: str) -> Optional[dict]:
        """Get metadata for a document."""
        return self.documents.get(doc_id)
    
    def save(self, filepath: str):
        """Save the index to a JSON file."""
        data = {
            'index': {k: dict(v) for k, v in self.index.items()},
            'documents': self.documents,
            'doc_freq': dict(self.doc_freq),
            'num_docs': self.num_docs
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load the index from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.index = defaultdict(dict, {k: dict(v) for k, v in data['index'].items()})
        self.documents = data['documents']
        self.doc_freq = defaultdict(int, data['doc_freq'])
        self.num_docs = data['num_docs']
    
    def __len__(self) -> int:
        """Return number of indexed documents."""
        return self.num_docs
    
    def __contains__(self, doc_id: str) -> bool:
        """Check if a document is indexed."""
        return doc_id in self.documents
