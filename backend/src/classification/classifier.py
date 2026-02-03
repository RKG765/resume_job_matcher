"""
Classifier Module
Implements fit/partial/reject classification for resume-job matches.
"""

from enum import Enum
from typing import Dict, Tuple
from dataclasses import dataclass


class MatchLevel(Enum):
    """Match classification levels."""
    FIT = "fit"
    PARTIAL = "partial"
    REJECT = "reject"


@dataclass
class ClassificationResult:
    """Result of match classification."""
    level: MatchLevel
    score: float
    label: str
    color: str
    description: str
    recommendations: list


class FitClassifier:
    """
    Classifies resume-job matches into fit/partial/reject categories.
    """
    
    def __init__(self, fit_threshold: float = 0.7, partial_threshold: float = 0.4):
        """
        Initialize classifier with thresholds.
        
        Args:
            fit_threshold: Minimum score for 'fit' classification
            partial_threshold: Minimum score for 'partial' classification
        """
        self.fit_threshold = fit_threshold
        self.partial_threshold = partial_threshold
    
    def classify(self, score: float, matched_skills: list = None, 
                 missing_skills: list = None) -> ClassificationResult:
        """
        Classify a match based on score and skill coverage.
        
        Args:
            score: Overall match score (0-1)
            matched_skills: List of skills the candidate has
            missing_skills: List of skills the candidate lacks
            
        Returns:
            ClassificationResult with level, label, and recommendations
        """
        matched_skills = matched_skills or []
        missing_skills = missing_skills or []
        
        if score >= self.fit_threshold:
            level = MatchLevel.FIT
            label = "Strong Match"
            color = "#22c55e"  # Green
            description = "Excellent candidate with strong alignment to job requirements."
            recommendations = self._generate_fit_recommendations(matched_skills, missing_skills)
        
        elif score >= self.partial_threshold:
            level = MatchLevel.PARTIAL
            label = "Potential Match"
            color = "#f59e0b"  # Amber
            description = "Candidate shows potential but has some skill gaps."
            recommendations = self._generate_partial_recommendations(matched_skills, missing_skills)
        
        else:
            level = MatchLevel.REJECT
            label = "Low Match"
            color = "#ef4444"  # Red
            description = "Candidate does not meet key job requirements."
            recommendations = self._generate_reject_recommendations(matched_skills, missing_skills)
        
        return ClassificationResult(
            level=level,
            score=score,
            label=label,
            color=color,
            description=description,
            recommendations=recommendations
        )
    
    def _generate_fit_recommendations(self, matched: list, missing: list) -> list:
        """Generate recommendations for fit candidates."""
        recs = ["✓ Strong candidate - consider for interview"]
        
        if matched:
            recs.append(f"✓ Key skills matched: {', '.join(matched[:5])}")
        
        if missing:
            recs.append(f"Note: Minor gaps in: {', '.join(missing[:3])}")
        
        return recs
    
    def _generate_partial_recommendations(self, matched: list, missing: list) -> list:
        """Generate recommendations for partial match candidates."""
        recs = ["○ Candidate shows potential with some gaps"]
        
        if matched:
            recs.append(f"✓ Has: {', '.join(matched[:5])}")
        
        if missing:
            recs.append(f"✗ Needs development in: {', '.join(missing[:5])}")
            recs.append("→ Consider if training/upskilling is feasible")
        
        return recs
    
    def _generate_reject_recommendations(self, matched: list, missing: list) -> list:
        """Generate recommendations for rejected candidates."""
        recs = ["✗ Does not meet minimum requirements"]
        
        if missing:
            recs.append(f"Missing critical skills: {', '.join(missing[:5])}")
        
        if matched:
            recs.append(f"Has some relevant skills: {', '.join(matched[:3])}")
            recs.append("→ May be suitable for a different role")
        
        return recs
    
    def get_threshold_info(self) -> Dict:
        """Get current threshold configuration."""
        return {
            'fit': {
                'threshold': self.fit_threshold,
                'label': 'Strong Match',
                'color': '#22c55e'
            },
            'partial': {
                'threshold': self.partial_threshold,
                'label': 'Potential Match',
                'color': '#f59e0b'
            },
            'reject': {
                'threshold': 0,
                'label': 'Low Match',
                'color': '#ef4444'
            }
        }


def classify_match(score: float, matched_skills: list = None, 
                   missing_skills: list = None) -> Dict:
    """
    Convenience function to classify a match.
    
    Returns:
        Dictionary with classification details
    """
    classifier = FitClassifier()
    result = classifier.classify(score, matched_skills, missing_skills)
    
    return {
        'level': result.level.value,
        'label': result.label,
        'color': result.color,
        'description': result.description,
        'recommendations': result.recommendations
    }
