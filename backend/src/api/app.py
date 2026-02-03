"""
Flask API Application
Backend for Resume-Job Description Matching System.
"""

import os
import sys
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.text_cleaner import TextCleaner
from preprocessing.tokenizer import Tokenizer
from indexing.tfidf_vectorizer import TFIDFVectorizer
from ranking.ranker import ResumeRanker
from classification.classifier import FitClassifier, classify_match
from classification.skill_gap import SkillGapAnalyzer
from clustering.job_clusterer import JobClusterer
from services.llm_service import get_llm_service, configure_llm


def create_app():
    """Create and configure Flask application."""
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    app = Flask(__name__,
                template_folder=os.path.join(project_root, 'templates'),
                static_folder=os.path.join(project_root, 'static'))
    
    CORS(app)
    
    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
    app.config['UPLOAD_FOLDER'] = os.path.join(project_root, 'data', 'uploads')
    
    # Create upload folder if needed
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize components
    text_cleaner = TextCleaner()
    ranker = ResumeRanker()
    classifier = FitClassifier()
    skill_analyzer = SkillGapAnalyzer()
    clusterer = JobClusterer()
    
    # Store for indexed data
    app.jobs = []
    app.resumes = []
    
    # Load sample data on startup
    load_sample_data(app, project_root)
    
    # ==================== FRONTEND ROUTES ====================
    
    @app.route('/')
    def index():
        """Serve main application page."""
        return render_template('index.html')
    
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        """Serve static files."""
        return send_from_directory(app.static_folder, filename)
    
    # ==================== API ROUTES ====================
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'jobs_indexed': len(app.jobs),
            'resumes_indexed': len(app.resumes)
        })
    
    @app.route('/api/jobs', methods=['GET'])
    def get_jobs():
        """Get all indexed job descriptions."""
        return jsonify({
            'count': len(app.jobs),
            'jobs': [{
                'id': j['id'],
                'title': j.get('title', 'Unknown'),
                'company': j.get('company', ''),
                'skills': j.get('skills', [])
            } for j in app.jobs]
        })
    
    @app.route('/api/jobs/<job_id>', methods=['GET'])
    def get_job(job_id):
        """Get a specific job by ID."""
        job = next((j for j in app.jobs if j['id'] == job_id), None)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        return jsonify(job)
    
    @app.route('/api/resumes', methods=['GET'])
    def get_resumes():
        """Get all indexed resumes."""
        return jsonify({
            'count': len(app.resumes),
            'resumes': [{
                'id': r['id'],
                'name': r.get('name', 'Unknown'),
                'skills': r.get('skills', [])
            } for r in app.resumes]
        })
    
    @app.route('/api/rank/<job_id>', methods=['GET'])
    def rank_resumes_for_job(job_id):
        """Rank all resumes for a specific job."""
        job = next((j for j in app.jobs if j['id'] == job_id), None)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Get ranking parameters
        top_k = request.args.get('top_k', 10, type=int)
        min_score = request.args.get('min_score', 0.0, type=float)
        
        # Perform ranking
        job_text = text_cleaner.preprocess(job.get('description', ''))
        job_skills = job.get('skills', [])
        
        results = []
        for resume in app.resumes:
            resume_text = text_cleaner.preprocess(resume.get('content', ''))
            resume_skills = resume.get('skills', [])
            
            # Get match score
            match_result = ranker.match_resume_to_job(
                resume_text, job_text, resume_skills, job_skills
            )
            
            if match_result['score'] >= min_score:
                # Classify match
                classification = classify_match(
                    match_result['score'],
                    match_result['matched_skills'],
                    match_result['missing_skills']
                )
                
                results.append({
                    'resume_id': resume['id'],
                    'name': resume.get('name', 'Unknown'),
                    'score': match_result['score'],
                    'classification': classification,
                    'matched_skills': match_result['matched_skills'],
                    'missing_skills': match_result['missing_skills']
                })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            'job_id': job_id,
            'job_title': job.get('title', 'Unknown'),
            'total_resumes': len(app.resumes),
            'results': results[:top_k]
        })
    
    @app.route('/api/manual-match', methods=['POST'])
    def manual_match():
        """
        Manual matching mode: Match multiple resumes against a job description.
        Accepts JSON with job_description and resumes array.
        """
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        job_description = data.get('job_description', '')
        job_skills = data.get('job_skills', [])
        resumes = data.get('resumes', [])
        
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400
        
        if not resumes:
            return jsonify({'error': 'At least one resume is required'}), 400
        
        try:
            # Preprocess job description
            job_text = text_cleaner.preprocess(job_description)
            
            # Extract skills if not provided
            if not job_skills:
                job_skills = list(skill_analyzer.extract_skills(job_description))
            
            # Prepare all texts for vectorization
            resume_texts = []
            resume_metadata = []
            for resume in resumes:
                resume_content = resume.get('content', '')
                resume_name = resume.get('name', f"Resume {len(resume_metadata) + 1}")
                resume_skills = resume.get('skills', [])
                
                # Preprocess resume
                resume_text = text_cleaner.preprocess(resume_content)
                
                # Extract skills if not provided
                if not resume_skills:
                    resume_skills = list(skill_analyzer.extract_skills(resume_content))
                
                resume_texts.append(resume_text)
                resume_metadata.append({
                    'name': resume_name,
                    'content': resume_content,
                    'skills': resume_skills
                })
            
            # Create a fresh TF-IDF vectorizer and fit on ALL texts together
            from indexing.tfidf_vectorizer import TFIDFVectorizer
            from ranking.similarity import CosineSimilarity, JaccardSimilarity, WeightedSimilarity
            
            temp_vectorizer = TFIDFVectorizer()
            all_texts = [job_text] + resume_texts
            temp_vectorizer.fit_transform(all_texts)
            
            # Get job vector
            job_vector = temp_vectorizer.transform([job_text])[0]
            job_skills_set = set(s.lower() for s in job_skills)
            
            results = []
            for i, (resume_text, meta) in enumerate(zip(resume_texts, resume_metadata)):
                # Get resume vector
                resume_vector = temp_vectorizer.transform([resume_text])[0]
                
                # Content similarity
                content_sim = CosineSimilarity.compute(job_vector, resume_vector)
                
                # Skill similarity
                resume_skills_set = set(s.lower() for s in meta['skills'])
                skill_sim = JaccardSimilarity.compute_overlap(job_skills_set, resume_skills_set) if job_skills_set else content_sim
                
                # Combined score
                weighted = WeightedSimilarity()
                final_score = weighted.compute(content_sim, skill_sim)
                
                # Calculate matched and missing skills
                matched_skills = list(job_skills_set & resume_skills_set) if job_skills_set else []
                missing_skills = list(job_skills_set - resume_skills_set) if job_skills_set else []
                
                # Classify match
                classification = classify_match(final_score, matched_skills, missing_skills)
                
                # Get skill gap analysis
                skill_gap = skill_analyzer.analyze_gap(
                    job_description, meta['content'],
                    job_skills, list(meta['skills'])
                )
                
                results.append({
                    'name': meta['name'],
                    'score': round(final_score, 4),
                    'content_similarity': round(content_sim, 4),
                    'skill_similarity': round(skill_sim, 4),
                    'classification': classification,
                    'matched_skills': matched_skills,
                    'missing_skills': missing_skills,
                    'resume_extract': meta['content'][:2000] if meta['content'] else '',
                    'resume_full': meta['content'] if meta['content'] else '',
                    'skill_gap': {
                        'coverage': skill_gap['coverage_percentage'],
                        'critical_missing': skill_gap['critical_missing'],
                        'recommendations': skill_gap['recommendations']
                    }
                })
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Generate explanations for each result
            llm = get_llm_service()
            for result in results:
                explanation = llm.generate_match_explanation(
                    job_description=job_description,
                    resume_content=result.get('resume_extract', ''),
                    resume_name=result.get('name', 'Unknown'),
                    score=result.get('score', 0),
                    matched_skills=result.get('matched_skills', []),
                    missing_skills=result.get('missing_skills', []),
                    content_similarity=result.get('content_similarity', 0),
                    skill_similarity=result.get('skill_similarity', 0)
                )
                result['explanation'] = explanation
            
            return jsonify({
                'job_skills_detected': job_skills,
                'total_resumes': len(resumes),
                'results': results,
                'llm_enabled': llm.enabled and llm.is_available()
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Matching failed: {str(e)}'}), 500
    
    @app.route('/api/skill-gap', methods=['POST'])
    def analyze_skill_gap():
        """Analyze skill gaps between a resume and job."""
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        job_description = data.get('job_description', '')
        resume_content = data.get('resume_content', '')
        
        if not job_description or not resume_content:
            return jsonify({'error': 'Both job_description and resume_content are required'}), 400
        
        analysis = skill_analyzer.analyze_gap(job_description, resume_content)
        
        return jsonify(analysis)
    
    @app.route('/api/llm/configure', methods=['POST'])
    def configure_llm_service():
        """
        Configure the LLM service for generating explanations.
        
        JSON body:
        - api_url: LM Studio or OpenAI-compatible API URL (default: http://localhost:1234/v1/chat/completions)
        - api_key: API key (default: 'lm-studio')
        - model: Model name
        - enabled: Whether to enable LLM explanations
        """
        data = request.get_json() or {}
        
        llm = get_llm_service()
        
        if 'api_url' in data:
            llm.api_url = data['api_url']
        if 'api_key' in data:
            llm.api_key = data['api_key']
        if 'model' in data:
            llm.model = data['model']
        if 'enabled' in data:
            llm.enabled = data['enabled']
        
        return jsonify({
            'success': True,
            'config': {
                'api_url': llm.api_url,
                'model': llm.model,
                'enabled': llm.enabled,
                'is_available': llm.is_available()
            }
        })
    
    @app.route('/api/llm/status', methods=['GET'])
    def llm_status():
        """Check LLM service status."""
        llm = get_llm_service()
        return jsonify({
            'enabled': llm.enabled,
            'is_available': llm.is_available(),
            'api_url': llm.api_url,
            'model': llm.model
        })
    
    @app.route('/api/clusters', methods=['GET'])
    def get_clusters():
        """Get job domain clusters."""
        if not app.jobs:
            return jsonify({'error': 'No jobs indexed'}), 400
        
        # Cluster jobs
        result = clusterer.fit(app.jobs)
        
        return jsonify(result)
    
    @app.route('/api/extract-skills', methods=['POST'])
    def extract_skills():
        """Extract skills from text."""
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        skills = list(skill_analyzer.extract_skills(text))
        
        return jsonify({
            'skills': sorted(skills),
            'count': len(skills)
        })
    
    @app.route('/api/upload-file', methods=['POST'])
    def upload_file():
        """Upload and parse a file (PDF or TXT) to extract text."""
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        
        try:
            if file_ext == 'pdf':
                # Parse PDF
                import PyPDF2
                import io
                
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                # Clean up the text
                text = text.strip()
                
            elif file_ext in ['txt', 'text']:
                # Read text file
                text = file.read().decode('utf-8', errors='ignore')
                
            else:
                return jsonify({'error': f'Unsupported file type: {file_ext}. Use PDF or TXT files.'}), 400
            
            # Extract skills from the parsed text
            skills = list(skill_analyzer.extract_skills(text))
            
            return jsonify({
                'success': True,
                'filename': filename,
                'text': text,
                'skills': skills,
                'char_count': len(text)
            })
            
        except Exception as e:
            return jsonify({'error': f'Failed to parse file: {str(e)}'}), 500
    
    @app.route('/api/upload-resume', methods=['POST'])
    def upload_resume():
        """Upload a resume file and add it to the matching pool."""
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        candidate_name = request.form.get('name', '')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        
        try:
            if file_ext == 'pdf':
                import PyPDF2
                import io
                
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                text = text.strip()
                
            elif file_ext in ['txt', 'text']:
                text = file.read().decode('utf-8', errors='ignore')
                
            else:
                return jsonify({'error': f'Unsupported file type: {file_ext}'}), 400
            
            # Use filename as candidate name if not provided
            if not candidate_name:
                candidate_name = filename.rsplit('.', 1)[0]
            
            skills = list(skill_analyzer.extract_skills(text))
            
            return jsonify({
                'success': True,
                'name': candidate_name,
                'content': text,
                'skills': skills,
                'filename': filename
            })
            
        except Exception as e:
            return jsonify({'error': f'Failed to parse resume: {str(e)}'}), 500
    
    return app


def load_sample_data(app, project_root):
    """Load sample job descriptions and resumes."""
    sample_dir = os.path.join(project_root, 'data', 'sample')
    
    # Sample job descriptions
    app.jobs = [
        {
            'id': 'job_1',
            'title': 'Senior Python Developer',
            'company': 'Tech Corp',
            'description': '''
                We are looking for a Senior Python Developer to join our team.
                
                Requirements:
                - 5+ years of experience with Python
                - Strong knowledge of Django or Flask frameworks
                - Experience with REST API development
                - Familiarity with SQL databases (PostgreSQL, MySQL)
                - Experience with Docker and Kubernetes
                - Understanding of CI/CD pipelines
                - Good communication skills
                
                Nice to have:
                - Experience with AWS or GCP
                - Knowledge of machine learning
                - Experience with microservices architecture
            ''',
            'skills': ['python', 'django', 'flask', 'sql', 'postgresql', 'docker', 
                      'kubernetes', 'rest', 'api', 'aws', 'machine learning']
        },
        {
            'id': 'job_2',
            'title': 'Data Scientist',
            'company': 'Analytics Inc',
            'description': '''
                Join our data science team to build ML models and derive insights.
                
                Requirements:
                - MS/PhD in Computer Science, Statistics, or related field
                - Strong Python and R programming skills
                - Experience with TensorFlow, PyTorch, or Keras
                - Knowledge of machine learning algorithms
                - Experience with data visualization (Tableau, Power BI)
                - Strong SQL skills
                - Experience with big data tools (Spark, Hadoop)
                
                Responsibilities:
                - Build predictive models
                - Analyze large datasets
                - Present findings to stakeholders
            ''',
            'skills': ['python', 'r', 'machine learning', 'tensorflow', 'pytorch', 
                      'sql', 'tableau', 'spark', 'hadoop', 'statistics']
        },
        {
            'id': 'job_3',
            'title': 'Frontend Developer',
            'company': 'WebStart',
            'description': '''
                Looking for a talented Frontend Developer to create amazing user experiences.
                
                Requirements:
                - 3+ years of experience with React or Vue.js
                - Strong HTML5, CSS3, and JavaScript skills
                - Experience with responsive design
                - Knowledge of state management (Redux, Vuex)
                - Familiarity with testing frameworks (Jest, Cypress)
                - Experience with Git version control
                
                Nice to have:
                - TypeScript experience
                - Node.js knowledge
                - UI/UX design skills
            ''',
            'skills': ['react', 'vue', 'javascript', 'html5', 'css3', 'redux', 
                      'jest', 'git', 'typescript', 'nodejs']
        },
        {
            'id': 'job_4',
            'title': 'DevOps Engineer',
            'company': 'CloudSys',
            'description': '''
                We need a DevOps Engineer to manage our cloud infrastructure.
                
                Requirements:
                - 4+ years of DevOps/SRE experience
                - Strong experience with AWS, Azure, or GCP
                - Expertise in Docker and Kubernetes
                - Experience with Terraform and Ansible
                - Knowledge of CI/CD tools (Jenkins, GitLab CI)
                - Strong Linux administration skills
                - Scripting skills (Python, Bash)
                
                Responsibilities:
                - Manage cloud infrastructure
                - Implement CI/CD pipelines
                - Monitor system performance
            ''',
            'skills': ['aws', 'azure', 'docker', 'kubernetes', 'terraform', 
                      'ansible', 'jenkins', 'linux', 'python', 'bash', 'devops']
        },
        {
            'id': 'job_5',
            'title': 'Full Stack Developer',
            'company': 'StartupXYZ',
            'description': '''
                Join our fast-paced startup as a Full Stack Developer.
                
                Requirements:
                - Experience with React or Angular frontend
                - Backend experience with Node.js or Python
                - Database experience (MongoDB, PostgreSQL)
                - REST API development
                - Understanding of agile methodologies
                - Good problem-solving skills
                
                Tech Stack:
                React, Node.js, Express, MongoDB, Docker, AWS
            ''',
            'skills': ['react', 'angular', 'nodejs', 'python', 'mongodb', 
                      'postgresql', 'rest', 'docker', 'aws', 'agile']
        }
    ]
    
    # Sample resumes
    app.resumes = [
        {
            'id': 'resume_1',
            'name': 'Alex Johnson',
            'content': '''
                ALEX JOHNSON
                Senior Software Engineer
                
                SUMMARY
                Experienced Python developer with 6 years of experience building 
                scalable web applications and APIs.
                
                SKILLS
                Python, Django, Flask, FastAPI, PostgreSQL, MySQL, Docker, 
                Kubernetes, AWS, REST APIs, Git, Agile, CI/CD
                
                EXPERIENCE
                Senior Python Developer at TechCorp (2020-Present)
                - Built microservices using Django and Flask
                - Deployed applications on AWS using Docker and Kubernetes
                - Implemented CI/CD pipelines with Jenkins
                
                Python Developer at WebAgency (2018-2020)
                - Developed REST APIs for mobile applications
                - Worked with PostgreSQL and Redis
                
                EDUCATION
                B.Tech in Computer Science, State University
            ''',
            'skills': ['python', 'django', 'flask', 'fastapi', 'postgresql', 
                      'mysql', 'docker', 'kubernetes', 'aws', 'rest', 'git', 'jenkins']
        },
        {
            'id': 'resume_2',
            'name': 'Sarah Chen',
            'content': '''
                SARAH CHEN
                Data Scientist
                
                SUMMARY
                PhD in Machine Learning with 4 years of industry experience
                in predictive modeling and data analysis.
                
                SKILLS
                Python, R, TensorFlow, PyTorch, Scikit-learn, SQL, Pandas,
                NumPy, Tableau, Spark, Statistics, Deep Learning, NLP
                
                EXPERIENCE
                Data Scientist at AI Solutions (2021-Present)
                - Built recommendation systems using deep learning
                - Analyzed large datasets with Spark
                - Created dashboards in Tableau
                
                ML Engineer at DataCorp (2019-2021)
                - Developed NLP models for text classification
                - Implemented computer vision solutions
                
                EDUCATION
                PhD in Computer Science (ML focus), Tech University
                MS in Statistics, State University
            ''',
            'skills': ['python', 'r', 'tensorflow', 'pytorch', 'scikit-learn', 
                      'sql', 'pandas', 'numpy', 'tableau', 'spark', 'machine learning', 'nlp']
        },
        {
            'id': 'resume_3',
            'name': 'Mike Williams',
            'content': '''
                MIKE WILLIAMS
                Frontend Developer
                
                SUMMARY
                Creative frontend developer with 4 years of experience
                building responsive web applications.
                
                SKILLS
                React, Vue.js, JavaScript, TypeScript, HTML5, CSS3, SASS,
                Redux, Jest, Cypress, Git, Figma
                
                EXPERIENCE
                Frontend Developer at DesignStudio (2020-Present)
                - Built React applications with Redux
                - Implemented responsive designs
                - Wrote unit tests with Jest
                
                Junior Developer at WebAgency (2019-2020)
                - Created Vue.js components
                - Worked with REST APIs
                
                EDUCATION
                B.Sc in Computer Science, City College
            ''',
            'skills': ['react', 'vue', 'javascript', 'typescript', 'html5', 
                      'css3', 'sass', 'redux', 'jest', 'cypress', 'git']
        },
        {
            'id': 'resume_4',
            'name': 'Emily Davis',
            'content': '''
                EMILY DAVIS
                Junior Developer
                
                SUMMARY
                Recent graduate with strong fundamentals in web development
                and eagerness to learn.
                
                SKILLS
                HTML, CSS, JavaScript, Python basics, SQL basics, Git
                
                EXPERIENCE
                Intern at TechStartup (2023)
                - Assisted in building web pages
                - Fixed bugs in JavaScript code
                - Learned Git version control
                
                PROJECTS
                - Personal portfolio website (HTML, CSS, JavaScript)
                - Simple Python scripts for data processing
                
                EDUCATION
                B.Sc in Computer Science, University of Tech (2023)
            ''',
            'skills': ['html', 'css', 'javascript', 'python', 'sql', 'git']
        },
        {
            'id': 'resume_5',
            'name': 'David Kumar',
            'content': '''
                DAVID KUMAR
                DevOps Engineer
                
                SUMMARY
                Experienced DevOps professional with expertise in cloud
                infrastructure and automation.
                
                SKILLS
                AWS, Azure, GCP, Docker, Kubernetes, Terraform, Ansible,
                Jenkins, GitLab CI, Linux, Python, Bash, Prometheus, Grafana
                
                EXPERIENCE
                Senior DevOps Engineer at CloudCorp (2020-Present)
                - Managed AWS infrastructure for 50+ services
                - Implemented Kubernetes clusters
                - Built CI/CD pipelines with Jenkins and GitLab
                
                DevOps Engineer at TechSolutions (2018-2020)
                - Automated deployments with Ansible and Terraform
                - Monitored systems with Prometheus and Grafana
                
                EDUCATION
                B.Tech in Information Technology, Tech Institute
            ''',
            'skills': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 
                      'ansible', 'jenkins', 'linux', 'python', 'bash', 'devops']
        }
    ]


# Run the application
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
