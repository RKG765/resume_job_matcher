"""
Text Cleaner Module
Handles text preprocessing for resumes and job descriptions.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class TextCleaner:
    """
    Text preprocessing pipeline for IR system.
    Handles cleaning, normalization, and lemmatization.
    """
    
    # Technical terms to preserve (won't be lowercased or split)
    TECHNICAL_TERMS = {
        'c++', 'c#', 'node.js', 'vue.js', 'react.js', 'angular.js',
        '.net', 'asp.net', 'vb.net', 'f#', 'objective-c', 'r',
        'aws', 'gcp', 'azure', 'sql', 'nosql', 'mongodb', 'postgresql',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas',
        'numpy', 'matplotlib', 'docker', 'kubernetes', 'jenkins',
        'git', 'github', 'gitlab', 'jira', 'confluence', 'slack',
        'html5', 'css3', 'sass', 'less', 'bootstrap', 'tailwind',
        'api', 'rest', 'graphql', 'json', 'xml', 'yaml',
        'agile', 'scrum', 'kanban', 'devops', 'ci/cd',
        'ml', 'ai', 'nlp', 'cv', 'dl', 'llm', 'gpt',
        'b.tech', 'm.tech', 'b.sc', 'm.sc', 'ph.d', 'mba'
    }
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add domain-specific stopwords
        self.stop_words.update([
            'experience', 'work', 'working', 'worked', 'job',
            'position', 'role', 'responsibilities', 'duties',
            'candidate', 'applicant', 'resume', 'cv'
        ])
    
    def clean(self, text: str) -> str:
        """
        Main cleaning pipeline.
        
        Args:
            text: Raw text from resume or job description
            
        Returns:
            Cleaned and preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Preserve technical terms
        text = self._preserve_technical_terms(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{10,}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\+\#\.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _preserve_technical_terms(self, text: str) -> str:
        """Preserve technical terms by replacing them with placeholders."""
        text_lower = text.lower()
        preserved = text_lower
        
        for term in self.TECHNICAL_TERMS:
            if term in text_lower:
                # Replace with underscore version to preserve during tokenization
                preserved = preserved.replace(term, term.replace('.', '_').replace('+', 'plus').replace('#', 'sharp'))
        
        return preserved
    
    def remove_stopwords(self, tokens: list) -> list:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Filtered list without stopwords
        """
        return [token for token in tokens if token.lower() not in self.stop_words and len(token) > 1]
    
    def lemmatize(self, tokens: list) -> list:
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline: clean -> tokenize -> remove stopwords -> lemmatize -> join.
        
        Args:
            text: Raw text
            
        Returns:
            Fully preprocessed text as a string
        """
        from preprocessing.tokenizer import Tokenizer
        
        tokenizer = Tokenizer()
        
        # Clean text
        cleaned = self.clean(text)
        
        # Tokenize
        tokens = tokenizer.tokenize(cleaned)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize(tokens)
        
        return ' '.join(tokens)


# Convenience function
def clean_text(text: str) -> str:
    """Quick access to text cleaning."""
    cleaner = TextCleaner()
    return cleaner.preprocess(text)
