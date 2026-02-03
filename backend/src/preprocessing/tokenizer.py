"""
Tokenizer Module
Handles text tokenization for resumes and job descriptions.
"""

import re
from typing import List

# Simple tokenization without NLTK dependency to avoid punkt issues
def simple_word_tokenize(text):
    """Simple word tokenizer that doesn't require NLTK punkt data."""
    # Split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


class Tokenizer:
    """
    Tokenizer for IR system with support for n-grams and technical terms.
    """
    
    def __init__(self):
        pass
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Preprocessed text string
            
        Returns:
            List of word tokens
        """
        if not text:
            return []
        
        # Use simple word tokenization
        tokens = simple_word_tokenize(text)
        
        # Filter out single characters (except important ones like 'r', 'c')
        tokens = [t for t in tokens if len(t) > 1 or t.lower() in ['r', 'c']]
        
        return tokens
    
    def generate_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """
        Generate n-grams from token list.
        
        Args:
            tokens: List of word tokens
            n: Size of n-gram (default: bigrams)
            
        Returns:
            List of n-gram strings
        """
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = '_'.join(tokens[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def tokenize_with_ngrams(self, text: str, include_unigrams: bool = True, 
                             include_bigrams: bool = True) -> List[str]:
        """
        Tokenize and optionally include n-grams.
        
        Args:
            text: Text to tokenize
            include_unigrams: Whether to include single words
            include_bigrams: Whether to include word pairs
            
        Returns:
            Combined list of tokens and n-grams
        """
        tokens = self.tokenize(text)
        result = []
        
        if include_unigrams:
            result.extend(tokens)
        
        if include_bigrams:
            bigrams = self.generate_ngrams(tokens, n=2)
            result.extend(bigrams)
        
        return result


# Convenience function
def tokenize(text: str) -> List[str]:
    """Quick access to tokenization."""
    tokenizer = Tokenizer()
    return tokenizer.tokenize(text)
