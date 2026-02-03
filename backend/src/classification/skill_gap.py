"""
Skill Gap Analyzer Module
Identifies and analyzes skill gaps between resumes and job requirements.
"""

import re
from typing import List, Dict, Set, Tuple
from collections import Counter
from difflib import SequenceMatcher


class SkillGapAnalyzer:
    """
    Analyzes skill gaps between job requirements and candidate resumes.
    Enhanced with synonym matching and fuzzy spelling correction.
    """
    
    # Comprehensive technical skills for extraction
    SKILL_PATTERNS = {
        # Programming Languages
        'python', 'java', 'javascript', 'typescript', 'csharp', 'c#', 'c++', 'cpp',
        'ruby', 'php', 'swift', 'kotlin', 'go', 'golang', 'rust', 'scala',
        'r', 'matlab', 'perl', 'bash', 'shell', 'powershell', 'lua', 'dart',
        'objective-c', 'groovy', 'haskell', 'clojure', 'elixir', 'erlang',
        
        # Web Technologies
        'html', 'html5', 'css', 'css3', 'sass', 'scss', 'less', 'bootstrap', 'tailwind',
        'react', 'reactjs', 'react.js', 'angular', 'angularjs', 'vue', 'vuejs', 'vue.js',
        'svelte', 'nextjs', 'next.js', 'nuxtjs', 'gatsby', 'webpack', 'vite',
        'nodejs', 'node.js', 'express', 'expressjs', 'nestjs', 'fastify',
        'django', 'flask', 'fastapi', 'spring', 'spring boot', 'laravel', 'rails',
        'asp.net', 'dotnet', '.net', 'jquery', 'ajax', 'websocket', 'graphql',
        
        # Databases
        'sql', 'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'elasticsearch',
        'oracle', 'sqlite', 'dynamodb', 'cassandra', 'firebase', 'firestore',
        'mariadb', 'neo4j', 'couchdb', 'memcached', 'cockroachdb', 'snowflake',
        
        # Cloud & DevOps
        'aws', 'amazon web services', 'azure', 'microsoft azure', 'gcp', 'google cloud',
        'docker', 'kubernetes', 'k8s', 'jenkins', 'terraform', 'ansible', 'chef', 'puppet',
        'ci/cd', 'cicd', 'continuous integration', 'continuous deployment',
        'devops', 'sre', 'site reliability', 'linux', 'unix', 'nginx', 'apache',
        'cloudformation', 'serverless', 'lambda', 'ecs', 'eks', 'fargate',
        'helm', 'prometheus', 'grafana', 'datadog', 'splunk', 'kibana', 'logstash',
        
        # Data & ML
        'machine learning', 'ml', 'deep learning', 'dl', 'artificial intelligence', 'ai',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
        'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
        'nlp', 'natural language processing', 'computer vision', 'cv',
        'data analysis', 'data analytics', 'data science', 'data engineering',
        'big data', 'hadoop', 'spark', 'pyspark', 'hive', 'presto', 'airflow',
        'tableau', 'power bi', 'looker', 'metabase', 'excel', 'statistics',
        'neural networks', 'cnn', 'rnn', 'lstm', 'transformer', 'bert', 'gpt',
        'mlops', 'feature engineering', 'etl', 'data pipeline', 'data warehouse',
        
        # Mobile Development
        'android', 'ios', 'react native', 'flutter', 'xamarin', 'ionic',
        'mobile development', 'mobile app', 'swift ui', 'jetpack compose',
        
        # Other Technical
        'git', 'github', 'gitlab', 'bitbucket', 'svn', 'version control',
        'api', 'rest', 'restful', 'soap', 'grpc', 'microservices', 'monolith',
        'agile', 'scrum', 'kanban', 'jira', 'confluence', 'trello', 'asana',
        'testing', 'unit testing', 'integration testing', 'e2e', 'selenium',
        'jest', 'mocha', 'pytest', 'junit', 'cypress', 'playwright',
        'security', 'cybersecurity', 'encryption', 'oauth', 'jwt', 'ssl', 'https',
        'blockchain', 'web3', 'solidity', 'ethereum', 'smart contracts',
        
        # Job Role Terms (commonly used)
        'full stack', 'fullstack', 'full-stack', 'frontend', 'front-end', 'front end',
        'backend', 'back-end', 'back end', 'software engineer', 'software developer',
        'web developer', 'web development', 'application developer',
        'devops engineer', 'data engineer', 'data analyst', 'data scientist',
        'ml engineer', 'ai engineer', 'cloud engineer', 'systems engineer',
        'qa engineer', 'quality assurance', 'test engineer', 'automation engineer',
        'real estate', 'realstate', 'real-estate', 'realestate',
        
        # Soft Skills
        'communication', 'leadership', 'teamwork', 'team player', 'collaboration',
        'problem solving', 'problem-solving', 'analytical', 'critical thinking',
        'project management', 'time management', 'multitasking', 'adaptability',
        'presentation', 'negotiation', 'client facing', 'stakeholder management',
    }
    
    # Synonym mapping - maps variations to canonical form
    SYNONYMS = {
        # Language synonyms
        'js': 'javascript', 'node': 'nodejs', 'node.js': 'nodejs',
        'py': 'python', 'c#': 'csharp', 'c sharp': 'csharp',
        'golang': 'go', 'ts': 'typescript',
        
        # Framework synonyms
        'react.js': 'react', 'reactjs': 'react', 'vue.js': 'vue', 'vuejs': 'vue',
        'angular.js': 'angular', 'angularjs': 'angular',
        'next.js': 'nextjs', 'nuxt.js': 'nuxtjs',
        'express.js': 'express', 'expressjs': 'express',
        'spring boot': 'spring', 'springboot': 'spring',
        
        # Database synonyms
        'postgres': 'postgresql', 'mongo': 'mongodb',
        'elastic': 'elasticsearch', 'dynamodb': 'dynamodb',
        
        # Cloud synonyms
        'amazon web services': 'aws', 'microsoft azure': 'azure',
        'google cloud': 'gcp', 'google cloud platform': 'gcp',
        
        # DevOps synonyms
        'k8s': 'kubernetes', 'kube': 'kubernetes',
        'ci/cd': 'cicd', 'ci-cd': 'cicd',
        'continuous integration': 'cicd', 'continuous deployment': 'cicd',
        
        # Role synonyms
        'fullstack': 'full stack', 'full-stack': 'full stack',
        'frontend': 'front end', 'front-end': 'front end',
        'backend': 'back end', 'back-end': 'back end',
        'devops': 'devops engineer', 'devsecops': 'devops engineer',
        'real-estate': 'real estate', 'realestate': 'real estate', 'realstate': 'real estate',
        
        # ML/AI synonyms
        'ml': 'machine learning', 'dl': 'deep learning',
        'ai': 'artificial intelligence', 'nlp': 'natural language processing',
        'cv': 'computer vision', 'sklearn': 'scikit-learn',
        
        # Other synonyms
        'rest api': 'rest', 'restful api': 'rest', 'web api': 'api',
        'problem-solving': 'problem solving', 'team-work': 'teamwork',
    }
    
    def __init__(self, fuzzy_threshold: float = 0.85):
        """
        Initialize the analyzer with configurable fuzzy matching threshold.
        Loads skills from dataset file if available.
        
        Args:
            fuzzy_threshold: Minimum similarity ratio for fuzzy matching (0.0 to 1.0)
        """
        self.skill_patterns = set(s.lower() for s in self.SKILL_PATTERNS)
        self.synonyms = {k.lower(): v.lower() for k, v in self.SYNONYMS.items()}
        self.fuzzy_threshold = fuzzy_threshold
        
        # Try to load additional skills from dataset file
        self._load_skills_dataset()
    
    def _load_skills_dataset(self):
        """Load skills from external dataset file if available."""
        import os
        import json
        
        # Try multiple possible paths
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'skills_dataset.json'),
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'skills_dataset.json'),
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend', 'data', 'skills_dataset.json'),
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                try:
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Add all skills from the dataset
                    for category, skills in data.items():
                        for skill in skills:
                            self.skill_patterns.add(skill.lower())
                    
                    print(f"Loaded skills dataset from {abs_path} - Total skills: {len(self.skill_patterns)}")
                    return
                except Exception as e:
                    print(f"Warning: Failed to load skills dataset: {e}")
        
        print(f"Skills dataset not found, using built-in patterns ({len(self.skill_patterns)} skills)")
    
    def _normalize_skill(self, skill: str) -> str:
        """Normalize skill using synonym mapping."""
        skill_lower = skill.lower().strip()
        return self.synonyms.get(skill_lower, skill_lower)
    
    def _fuzzy_match(self, word: str, skills: Set[str]) -> str:
        """
        Find best fuzzy match for a word in the skill set.
        Used for spelling error correction - STRICT matching to avoid false positives.
        
        Args:
            word: Word to match
            skills: Set of known skills
            
        Returns:
            Best matching skill or None if no match above threshold
        """
        word_lower = word.lower()
        # Skip short words to avoid false positives like "web" -> "web3"
        if len(word_lower) < 5:
            return None
        
        # Common words to skip (too generic, cause false positives)
        skip_words = {'development', 'developer', 'engineer', 'software', 'system',
                      'application', 'service', 'manager', 'analyst', 'design',
                      'experience', 'working', 'knowledge', 'skills', 'years',
                      'project', 'team', 'work', 'data', 'cloud', 'platform'}
        if word_lower in skip_words:
            return None
            
        best_match = None
        best_ratio = 0
        
        for skill in skills:
            # Only compare single-word skills for fuzzy matching
            if ' ' in skill:
                continue
            # Require similar length to avoid "web" matching "web3"
            if abs(len(word_lower) - len(skill)) > 2:
                continue
            ratio = SequenceMatcher(None, word_lower, skill).ratio()
            # Higher threshold (0.92) to avoid false positives
            if ratio > best_ratio and ratio >= 0.92:
                best_ratio = ratio
                best_match = skill
        
        return best_match
    
    def extract_skills(self, text: str) -> Set[str]:
        """
        Extract skills from text with synonym resolution and fuzzy matching.
        
        Args:
            text: Document text
            
        Returns:
            Set of identified skills (normalized)
        """
        if not text:
            return set()
        
        text_lower = text.lower()
        found_skills = set()
        
        # 0. Pre-process: normalize compound words (e.g., "webdevelopment" -> "web development")
        compound_mappings = {
            'webdevelopment': 'web development',
            'webdeveloper': 'web developer',
            'machinelearning': 'machine learning',
            'deeplearning': 'deep learning',
            'datascience': 'data science',
            'dataanalysis': 'data analysis',
            'dataengineer': 'data engineer',
            'dataanalytics': 'data analytics',
            'fullstack': 'full stack',
            'frontend': 'front end',
            'backend': 'back end',
            'realestate': 'real estate',
            'devops': 'devops',
            'mlops': 'mlops',
            'aiml': 'machine learning',
            'cicd': 'cicd',
            'nodejs': 'nodejs',
            'reactjs': 'react',
            'vuejs': 'vue',
            'angularjs': 'angular',
        }
        
        for compound, expanded in compound_mappings.items():
            if compound in text_lower:
                if expanded in self.skill_patterns or expanded in self.synonyms.values():
                    found_skills.add(expanded)
        
        # 1. Direct pattern matching
        for skill in self.skill_patterns:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                normalized = self._normalize_skill(skill)
                found_skills.add(normalized)
        
        # 2. Check synonyms explicitly
        for synonym, canonical in self.synonyms.items():
            pattern = r'\b' + re.escape(synonym) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(canonical)
        
        # 3. Fuzzy matching for potential typos (STRICT - only for longer words)
        words = re.findall(r'\b[a-z]{5,15}\b', text_lower)  # Min 5 chars
        for word in set(words):
            if word not in self.skill_patterns and word not in found_skills:
                match = self._fuzzy_match(word, self.skill_patterns)
                if match:
                    normalized = self._normalize_skill(match)
                    found_skills.add(normalized)
        
        # 4. Dynamic pattern detection (capitalized/technical terms)
        # Detect patterns like "AWS", "GCP", "CI/CD" style abbreviations
        abbreviations = re.findall(r'\b[A-Z]{2,6}\b', text)
        for abbr in abbreviations:
            abbr_lower = abbr.lower()
            if abbr_lower in self.skill_patterns:
                found_skills.add(self._normalize_skill(abbr_lower))
        
        return found_skills
    
    def analyze_gap(self, job_text: str, resume_text: str,
                    job_skills: List[str] = None, 
                    resume_skills: List[str] = None) -> Dict:
        """
        Analyze skill gaps between job requirements and resume.
        
        Args:
            job_text: Job description text
            resume_text: Resume content text
            job_skills: Optional pre-extracted job skills
            resume_skills: Optional pre-extracted resume skills
            
        Returns:
            Detailed gap analysis
        """
        # Extract skills if not provided
        if job_skills is None:
            job_skills = self.extract_skills(job_text)
        else:
            job_skills = set(s.lower() for s in job_skills)
        
        if resume_skills is None:
            resume_skills = self.extract_skills(resume_text)
        else:
            resume_skills = set(s.lower() for s in resume_skills)
        
        # Calculate gaps and overlaps
        matched_skills = job_skills & resume_skills
        missing_skills = job_skills - resume_skills
        extra_skills = resume_skills - job_skills  # Skills candidate has but job doesn't require
        
        # Calculate coverage
        coverage = len(matched_skills) / len(job_skills) if job_skills else 1.0
        
        # Categorize missing skills by importance (rough heuristic)
        critical_missing = []
        nice_to_have_missing = []
        
        # Define critical skill patterns
        critical_keywords = {
            'python', 'java', 'javascript', 'sql', 'aws', 'docker',
            'machine learning', 'data analysis', 'react', 'nodejs'
        }
        
        for skill in missing_skills:
            if skill in critical_keywords:
                critical_missing.append(skill)
            else:
                nice_to_have_missing.append(skill)
        
        return {
            'coverage_percentage': round(coverage * 100, 1),
            'total_required': len(job_skills),
            'total_matched': len(matched_skills),
            'total_missing': len(missing_skills),
            'matched_skills': sorted(list(matched_skills)),
            'missing_skills': sorted(list(missing_skills)),
            'critical_missing': critical_missing,
            'nice_to_have_missing': nice_to_have_missing,
            'extra_skills': sorted(list(extra_skills)),
            'recommendations': self._generate_recommendations(
                coverage, critical_missing, nice_to_have_missing, list(extra_skills)
            )
        }
    
    def _generate_recommendations(self, coverage: float, critical: List[str],
                                   nice_to_have: List[str], extra: List[str]) -> List[str]:
        """Generate actionable recommendations based on gap analysis."""
        recommendations = []
        
        if coverage >= 0.8:
            recommendations.append("✓ Strong skill alignment with job requirements")
        elif coverage >= 0.5:
            recommendations.append("○ Moderate skill alignment - some development needed")
        else:
            recommendations.append("✗ Significant skill gaps identified")
        
        if critical:
            recommendations.append(f"🔴 Priority: Develop skills in {', '.join(critical[:3])}")
        
        if nice_to_have:
            recommendations.append(f"🟡 Consider learning: {', '.join(nice_to_have[:3])}")
        
        if extra:
            recommendations.append(f"💡 Highlight transferable skills: {', '.join(extra[:3])}")
        
        return recommendations
    
    def batch_analyze(self, job_text: str, resumes: List[Dict]) -> List[Dict]:
        """
        Analyze skill gaps for multiple resumes against one job.
        
        Args:
            job_text: Job description
            resumes: List of resume dicts with 'content' and optionally 'skills'
            
        Returns:
            List of gap analyses for each resume
        """
        job_skills = self.extract_skills(job_text)
        results = []
        
        for resume in resumes:
            resume_text = resume.get('content', '')
            resume_skills = resume.get('skills', None)
            
            analysis = self.analyze_gap(
                job_text, resume_text,
                job_skills=list(job_skills),
                resume_skills=resume_skills
            )
            
            analysis['resume_id'] = resume.get('id', 'unknown')
            analysis['resume_name'] = resume.get('name', 'Unknown')
            results.append(analysis)
        
        # Sort by coverage
        results.sort(key=lambda x: x['coverage_percentage'], reverse=True)
        
        return results


def analyze_skill_gap(job_text: str, resume_text: str) -> Dict:
    """Convenience function for skill gap analysis."""
    analyzer = SkillGapAnalyzer()
    return analyzer.analyze_gap(job_text, resume_text)
