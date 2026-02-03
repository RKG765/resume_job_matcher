"""
Job Clusterer Module
Implements K-Means clustering for job domain categorization.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Dict, Tuple, Optional
from indexing.tfidf_vectorizer import TFIDFVectorizer


class JobClusterer:
    """
    Clusters job descriptions into domain categories using K-Means.
    """
    
    # Default cluster labels based on common job domains
    DEFAULT_DOMAINS = [
        "Software Engineering",
        "Data Science & Analytics",
        "DevOps & Cloud",
        "Frontend Development",
        "Backend Development",
        "Machine Learning",
        "Product Management",
        "Quality Assurance",
        "Cybersecurity",
        "Other"
    ]
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        """
        Initialize the clusterer.
        
        Args:
            n_clusters: Number of clusters (job domains)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.vectorizer = TFIDFVectorizer(max_features=1000)
        self.cluster_labels = {}
        self.cluster_top_terms = {}
        self.job_clusters = {}
    
    def fit(self, jobs: List[Dict]) -> Dict:
        """
        Fit the clusterer on job descriptions.
        
        Args:
            jobs: List of job dicts with 'id' and 'description'
            
        Returns:
            Clustering results
        """
        if len(jobs) < self.n_clusters:
            self.n_clusters = max(2, len(jobs) // 2)
        
        # Extract texts and ids
        texts = [job.get('description', '') for job in jobs]
        job_ids = [job.get('id', f'job_{i}') for i, job in enumerate(jobs)]
        
        # Vectorize
        vectors = self.vectorizer.fit_transform(texts, job_ids)
        
        # Run K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        cluster_assignments = self.kmeans.fit_predict(vectors)
        
        # Store job cluster assignments
        for i, job_id in enumerate(job_ids):
            self.job_clusters[job_id] = int(cluster_assignments[i])
        
        # Generate cluster labels from top terms
        self._generate_cluster_labels(vectors)
        
        # Calculate silhouette score if we have enough samples
        silhouette = -1
        if len(jobs) > self.n_clusters:
            try:
                silhouette = silhouette_score(vectors, cluster_assignments)
            except:
                pass
        
        return {
            'n_clusters': self.n_clusters,
            'silhouette_score': round(silhouette, 4) if silhouette > 0 else None,
            'clusters': self._get_cluster_summary(jobs, cluster_assignments)
        }
    
    def _generate_cluster_labels(self, vectors):
        """Generate descriptive labels for each cluster based on top terms."""
        feature_names = self.vectorizer.get_feature_names()
        
        for cluster_id in range(self.n_clusters):
            # Get centroid
            centroid = self.kmeans.cluster_centers_[cluster_id]
            
            # Get top terms
            top_indices = centroid.argsort()[-5:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            
            self.cluster_top_terms[cluster_id] = top_terms
            
            # Generate label from top terms
            label = self._infer_domain_label(top_terms)
            self.cluster_labels[cluster_id] = label
    
    def _infer_domain_label(self, top_terms: List[str]) -> str:
        """Infer a domain label from top cluster terms."""
        terms_lower = set(t.lower() for t in top_terms)
        
        # Heuristic matching to common domains
        if terms_lower & {'python', 'machine', 'learning', 'data', 'model', 'ml'}:
            return "Data Science & ML"
        elif terms_lower & {'react', 'frontend', 'css', 'javascript', 'ui', 'ux'}:
            return "Frontend Development"
        elif terms_lower & {'backend', 'api', 'server', 'database', 'sql'}:
            return "Backend Development"
        elif terms_lower & {'devops', 'docker', 'kubernetes', 'aws', 'cloud', 'infrastructure'}:
            return "DevOps & Cloud"
        elif terms_lower & {'java', 'spring', 'enterprise', 'microservices'}:
            return "Enterprise Software"
        elif terms_lower & {'ios', 'android', 'mobile', 'flutter', 'swift'}:
            return "Mobile Development"
        elif terms_lower & {'security', 'penetration', 'vulnerability', 'cyber'}:
            return "Cybersecurity"
        elif terms_lower & {'product', 'manager', 'agile', 'stakeholder'}:
            return "Product Management"
        elif terms_lower & {'test', 'qa', 'quality', 'automation'}:
            return "Quality Assurance"
        else:
            # Use top terms as label
            return " / ".join(top_terms[:3]).title()
    
    def _get_cluster_summary(self, jobs: List[Dict], assignments: np.ndarray) -> List[Dict]:
        """Get summary information for each cluster."""
        clusters = []
        
        for cluster_id in range(self.n_clusters):
            cluster_jobs = [
                jobs[i] for i in range(len(jobs))
                if assignments[i] == cluster_id
            ]
            
            clusters.append({
                'id': cluster_id,
                'label': self.cluster_labels.get(cluster_id, f"Cluster {cluster_id}"),
                'top_terms': self.cluster_top_terms.get(cluster_id, []),
                'job_count': len(cluster_jobs),
                'jobs': [
                    {'id': j.get('id', ''), 'title': j.get('title', 'Unknown')}
                    for j in cluster_jobs[:5]  # Limit to 5 example jobs
                ]
            })
        
        return clusters
    
    def predict(self, job_text: str) -> Dict:
        """
        Predict the cluster for a new job description.
        
        Args:
            job_text: Job description text
            
        Returns:
            Prediction with cluster ID and label
        """
        if self.kmeans is None:
            raise ValueError("Clusterer must be fitted first")
        
        vector = self.vectorizer.transform([job_text])
        cluster_id = int(self.kmeans.predict(vector)[0])
        
        return {
            'cluster_id': cluster_id,
            'label': self.cluster_labels.get(cluster_id, f"Cluster {cluster_id}"),
            'top_terms': self.cluster_top_terms.get(cluster_id, [])
        }
    
    def find_optimal_k(self, jobs: List[Dict], k_range: Tuple[int, int] = (2, 10)) -> int:
        """
        Find optimal number of clusters using elbow method.
        
        Args:
            jobs: List of job dicts
            k_range: Range of k values to try
            
        Returns:
            Optimal number of clusters
        """
        texts = [job.get('description', '') for job in jobs]
        vectors = self.vectorizer.fit_transform(texts)
        
        inertias = []
        silhouettes = []
        
        for k in range(k_range[0], min(k_range[1], len(jobs))):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(vectors)
            inertias.append(kmeans.inertia_)
            
            if len(jobs) > k:
                try:
                    score = silhouette_score(vectors, kmeans.labels_)
                    silhouettes.append((k, score))
                except:
                    pass
        
        # Return k with best silhouette score
        if silhouettes:
            best_k = max(silhouettes, key=lambda x: x[1])[0]
            return best_k
        
        return min(5, len(jobs) // 2)
    
    def get_cluster_for_job(self, job_id: str) -> Optional[Dict]:
        """Get cluster information for a specific job."""
        if job_id not in self.job_clusters:
            return None
        
        cluster_id = self.job_clusters[job_id]
        return {
            'cluster_id': cluster_id,
            'label': self.cluster_labels.get(cluster_id, f"Cluster {cluster_id}"),
            'top_terms': self.cluster_top_terms.get(cluster_id, [])
        }
