# Resume-Job Description Matching System

An **Information Retrieval system** that matches resumes to job descriptions, ranks candidates, identifies skill gaps, and clusters job domains.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![ReactJs](https://img.shields.io/badge/React.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

---

## 🌟 Features

- **TF-IDF Document Representation** - Vectorize resumes and job descriptions
- **Cosine Similarity Matching** - Rank resumes by relevance
- **Skill Gap Analysis** - Identify missing skills
- **Fit/Partial/Reject Classification** - Categorize candidates
- **K-Means Job Clustering** - Group jobs by domain
- **Manual Matching Mode** - Paste JD + multiple resumes for instant analysis

---

## 📁 Project Structure

```
resume_job_matcher/
├── src/
│   ├── preprocessing/      # Text cleaning, tokenization
│   ├── indexing/          # TF-IDF, inverted index
│   ├── ranking/           # Similarity, ranker
│   ├── classification/    # Classifier, skill gap
│   ├── clustering/        # K-Means clustering
│   └── api/               # Flask REST API
├── frontend/              # Electron app (for .exe)
│   ├── renderer/          # HTML, CSS, JS
│   ├── main.js           # Electron main process
│   └── package.json
├── data/                  # Datasets
├── Dockerfile            # Docker build
├── docker-compose.yml    # Docker orchestration
├── requirements.txt      # Python dependencies
└── main.py               # Entry point
```

---

## 🚀 Quick Start

### Option 1: Local Development (Recommended for Testing)

```bash
# 1. Navigate to project
cd resume_job_matcher

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the backend
python main.py
```

Backend runs at: `http://127.0.0.1:5000`

### Option 2: Run with Electron (Desktop App)

```bash
# 1. Start backend first (in one terminal)
cd resume_job_matcher
.\venv\Scripts\activate
python main.py

# 2. Run Electron (in another terminal)
cd frontend
npm install
npm start
```

### Option 3: Docker (for AWS Deployment)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or just the backend
docker build -t resume-matcher .
docker run -p 5000:5000 resume-matcher
```

---

## ☁️ AWS Deployment

### EC2 Deployment

1. **Launch EC2 Instance** (Ubuntu 22.04, t2.micro or larger)

2. **Install Docker**:
```bash
sudo apt update
sudo apt install docker.io docker-compose -y
sudo usermod -aG docker $USER
```

3. **Clone and Deploy**:
```bash
git clone <your-repo-url>
cd resume_job_matcher
docker-compose up -d
```

4. **Configure Security Group**:
   - Allow inbound HTTP (port 80)
   - Allow inbound on port 5000 (if not using Nginx)

### Elastic Beanstalk Deployment

1. Install EB CLI: `pip install awsebcli`
2. Initialize: `eb init -p docker resume-matcher`
3. Create environment: `eb create production`
4. Deploy: `eb deploy`

### AWS App Runner (Easiest)

1. Push to GitHub/ECR
2. Create App Runner service
3. Select Docker as source
4. Deploy automatically

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/jobs` | GET | List all indexed jobs |
| `/api/jobs/<id>` | GET | Get job details |
| `/api/resumes` | GET | List all resumes |
| `/api/manual-match` | POST | **Main feature**: Match resumes to JD |
| `/api/rank/<job_id>` | GET | Rank resumes for a job |
| `/api/skill-gap` | POST | Analyze skill gaps |
| `/api/clusters` | GET | Get job domain clusters |

### Example: Manual Match

```bash
curl -X POST http://localhost:5000/api/manual-match \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Looking for Python developer with Django...",
    "resumes": [
      {"name": "John Doe", "content": "5 years Python experience..."},
      {"name": "Jane Smith", "content": "React and Node.js developer..."}
    ]
  }'
```

---

## 🎨 Frontend Usage

### Manual Matching Mode (Main Feature)

1. **Paste Job Description** in the left panel
2. **Add Resumes** - paste text or upload files
3. Click **"Run Matching Analysis"**
4. View ranked results with:
   - Match scores (0-100%)
   - Classification (Fit/Partial/Reject)
   - Matched & missing skills
   - Recommendations

---

## 📊 IR Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| **Document Representation** | TF-IDF vectors |
| **Term Weighting** | Sublinear TF-IDF (1 + log(tf)) |
| **Similarity** | Cosine similarity |
| **Ranking** | Score-based ranking |
| **Classification** | Threshold-based (≥0.7 Fit, ≥0.4 Partial) |
| **Clustering** | K-Means on TF-IDF vectors |

---

## 🔧 Building Electron .exe

```bash
cd frontend
npm install
npm run build:win
```

Output: `frontend/dist/Resume Job Matcher Setup.exe`

---

## 📝 License

MIT License - Free for educational use.
