"""
Ranker Module
Implements resume ranking for job descriptions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from indexing.tfidf_vectorizer import TFIDFVectorizer
from ranking.similarity import CosineSimilarity, JaccardSimilarity, WeightedSimilarity


class ResumeRanker:
    """
    Ranks resumes against job descriptions using TF-IDF and similarity measures.
    """
    
    def __init__(self, vectorizer: Optional[TFIDFVectorizer] = None):
        """
        Initialize the ranker.
        
        Args:
            vectorizer: Pre-fitted TF-IDF vectorizer (optional)
        """
        self.vectorizer = vectorizer or TFIDFVectorizer()
        self.job_vectors = {}
        self.resume_vectors = {}
        self.job_data = {}
        self.resume_data = {}
    
    def index_jobs(self, jobs: List[Dict]):
        """
        Index job descriptions.
        
        Args:
            jobs: List of job dictionaries with 'id', 'title', 'description', 'skills'
        """
        job_texts = [job.get('description', '') for job in jobs]
        job_ids = [job.get('id', f'job_{i}') for i, job in enumerate(jobs)]
        
        # Store job data
        for job in jobs:
            job_id = job.get('id', f"job_{len(self.job_data)}")
            self.job_data[job_id] = job
        
        # Fit vectorizer on jobs if not already fitted
        if not self.vectorizer.is_fitted:
            vectors = self.vectorizer.fit_transform(job_texts, job_ids)
        else:
            vectors = self.vectorizer.transform(job_texts)
        
        # Store vectors
        for i, job_id in enumerate(job_ids):
            self.job_vectors[job_id] = vectors[i]
    
    def index_resumes(self, resumes: List[Dict]):
        """
        Index resumes.
        
        Args:
            resumes: List of resume dictionaries with 'id', 'name', 'content', 'skills'
        """
        resume_texts = [r.get('content', '') for r in resumes]
        resume_ids = [r.get('id', f'resume_{i}') for i, r in enumerate(resumes)]
        
        # Store resume data
        for resume in resumes:
            resume_id = resume.get('id', f"resume_{len(self.resume_data)}")
            self.resume_data[resume_id] = resume
        
        # Transform resumes using fitted vectorizer
        if not self.vectorizer.is_fitted:
            raise ValueError("Vectorizer must be fitted on jobs first")
        
        vectors = self.vectorizer.transform(resume_texts)
        
        # Store vectors
        for i, resume_id in enumerate(resume_ids):
            self.resume_vectors[resume_id] = vectors[i]
    
    def rank_resumes_for_job(self, job_id: str, top_k: int = 10, 
                             min_score: float = 0.0) -> List[Dict]:
        """
        Rank all resumes for a specific job.
        
        Args:
            job_id: Job identifier
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of ranked resume results with scores
        """
        if job_id not in self.job_vectors:
            raise ValueError(f"Job {job_id} not found in index")
        
        job_vector = self.job_vectors[job_id]
        job_data = self.job_data.get(job_id, {})
        job_skills = set(s.lower() for s in job_data.get('skills', []))
        
        results = []
        
        for resume_id, resume_vector in self.resume_vectors.items():
            # Compute content similarity
            content_sim = CosineSimilarity.compute(job_vector, resume_vector)
            
            # Compute skill similarity
            resume_data = self.resume_data.get(resume_id, {})
            resume_skills = set(s.lower() for s in resume_data.get('skills', []))
            skill_sim = JaccardSimilarity.compute_overlap(job_skills, resume_skills)
            
            # Combined score
            weighted = WeightedSimilarity()
            final_score = weighted.compute(content_sim, skill_sim)
            
            if final_score >= min_score:
                results.append({
                    'resume_id': resume_id,
                    'name': resume_data.get('name', 'Unknown'),
                    'score': round(final_score, 4),
                    'content_similarity': round(content_sim, 4),
                    'skill_similarity': round(skill_sim, 4),
                    'matched_skills': list(job_skills & resume_skills),
                    'missing_skills': list(job_skills - resume_skills)
                })
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]
    
    def match_resume_to_job(self, resume_text: str, job_text: str,
                            resume_skills: List[str] = None,
                            job_skills: List[str] = None) -> Dict:
        """
        Direct matching between a resume and job description.
        Used for manual matching mode.
        
        Args:
            resume_text: Resume content
            job_text: Job description
            resume_skills: List of resume skills
            job_skills: List of required job skills
            
        Returns:
            Match result with scores
        """
        # Transform both texts
        if not self.vectorizer.is_fitted:
            # Fit on job and transform both
            self.vectorizer.fit([job_text])
        
        job_vector = self.vectorizer.transform([job_text])[0]
        resume_vector = self.vectorizer.transform([resume_text])[0]
        
        # Content similarity
        content_sim = CosineSimilarity.compute(job_vector, resume_vector)
        
        # Skill similarity
        resume_skills = set(s.lower() for s in (resume_skills or []))
        job_skills = set(s.lower() for s in (job_skills or []))
        skill_sim = JaccardSimilarity.compute_overlap(job_skills, resume_skills) if job_skills else content_sim
        
        # Combined score
        weighted = WeightedSimilarity()
        final_score = weighted.compute(content_sim, skill_sim)
        
        return {
            'score': round(final_score, 4),
            'content_similarity': round(content_sim, 4),
            'skill_similarity': round(skill_sim, 4),
            'matched_skills': list(job_skills & resume_skills) if job_skills else [],
            'missing_skills': list(job_skills - resume_skills) if job_skills else []
        }
    
    def batch_match(self, job_text: str, resumes: List[Dict], 
                    job_skills: List[str] = None) -> List[Dict]:
        """
        Match multiple resumes against a single job description.
        Used for manual matching mode.
        
        Args:
            job_text: Job description text
            resumes: List of resume dicts with 'id', 'name', 'content', 'skills'
            job_skills: List of required job skills
            
        Returns:
            Ranked list of match results
        """
        results = []
        
        # Create temporary vectorizer for this batch
        temp_vectorizer = TFIDFVectorizer()
        all_texts = [job_text] + [r.get('content', '') for r in resumes]
        temp_vectorizer.fit_transform(all_texts)
        
        job_vector = temp_vectorizer.transform([job_text])[0]
        job_skills_set = set(s.lower() for s in (job_skills or []))
        
        for resume in resumes:
            resume_vector = temp_vectorizer.transform([resume.get('content', '')])[0]
            
            # Content similarity
            content_sim = CosineSimilarity.compute(job_vector, resume_vector)
            
            # Skill similarity
            resume_skills = set(s.lower() for s in resume.get('skills', []))
            skill_sim = JaccardSimilarity.compute_overlap(job_skills_set, resume_skills) if job_skills_set else content_sim
            
            # Combined score
            weighted = WeightedSimilarity()
            final_score = weighted.compute(content_sim, skill_sim)
            
            results.append({
                'resume_id': resume.get('id', 'unknown'),
                'name': resume.get('name', 'Unknown'),
                'score': round(final_score, 4),
                'content_similarity': round(content_sim, 4),
                'skill_similarity': round(skill_sim, 4),
                'matched_skills': list(job_skills_set & resume_skills) if job_skills_set else [],
                'missing_skills': list(job_skills_set - resume_skills) if job_skills_set else []
            })
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
