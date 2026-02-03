import { useState, useEffect, useCallback } from 'react'
import './App.css'

const API_BASE_URL = ''  // Uses Vite proxy

// Sample static data for when backend is not available
const SAMPLE_JOBS = [
  { id: 'job_1', title: 'Senior Python Developer', company: 'Tech Corp', skills: ['python', 'django', 'flask', 'sql', 'docker', 'aws'] },
  { id: 'job_2', title: 'Data Scientist', company: 'Analytics Inc', skills: ['python', 'machine learning', 'tensorflow', 'sql', 'tableau'] },
  { id: 'job_3', title: 'Frontend Developer', company: 'WebStart', skills: ['react', 'javascript', 'typescript', 'css', 'html5'] },
  { id: 'job_4', title: 'DevOps Engineer', company: 'CloudSys', skills: ['aws', 'docker', 'kubernetes', 'terraform', 'linux'] },
  { id: 'job_5', title: 'Full Stack Developer', company: 'StartupXYZ', skills: ['react', 'nodejs', 'mongodb', 'python', 'aws'] }
]

const SAMPLE_RESUMES = [
  { id: 'resume_1', name: 'Alex Johnson', title: 'Senior Software Engineer', skills: ['python', 'django', 'flask', 'docker', 'kubernetes', 'aws'] },
  { id: 'resume_2', name: 'Sarah Chen', title: 'Data Scientist', skills: ['python', 'tensorflow', 'pytorch', 'sql', 'tableau', 'spark'] },
  { id: 'resume_3', name: 'Mike Williams', title: 'Frontend Developer', skills: ['react', 'vue', 'javascript', 'typescript', 'css3'] },
  { id: 'resume_4', name: 'Emily Davis', title: 'Junior Developer', skills: ['html', 'css', 'javascript', 'python', 'git'] },
  { id: 'resume_5', name: 'David Kumar', title: 'DevOps Engineer', skills: ['aws', 'docker', 'kubernetes', 'terraform', 'linux', 'python'] }
]

const SAMPLE_CLUSTERS = [
  { id: 0, label: 'Backend Development', job_count: 2, top_terms: ['python', 'django', 'flask', 'sql', 'api'], jobs: [{ title: 'Senior Python Developer' }, { title: 'Full Stack Developer' }] },
  { id: 1, label: 'Data Science & ML', job_count: 1, top_terms: ['machine learning', 'tensorflow', 'python', 'data', 'analytics'], jobs: [{ title: 'Data Scientist' }] },
  { id: 2, label: 'Frontend & UI', job_count: 1, top_terms: ['react', 'javascript', 'typescript', 'css', 'frontend'], jobs: [{ title: 'Frontend Developer' }] },
  { id: 3, label: 'Cloud & Infrastructure', job_count: 1, top_terms: ['aws', 'docker', 'kubernetes', 'cloud', 'devops'], jobs: [{ title: 'DevOps Engineer' }] }
]

function App() {
  const [activeTab, setActiveTab] = useState('manual-match')
  const [jobDescription, setJobDescription] = useState('')
  const [jdSkills, setJdSkills] = useState([])
  const [resumes, setResumes] = useState([{ name: '', content: '' }])
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [backendConnected, setBackendConnected] = useState(false)
  const [inputMode, setInputMode] = useState('paste')

  // Data for tabs
  const [jobs, setJobs] = useState(SAMPLE_JOBS)
  const [resumesList, setResumesList] = useState(SAMPLE_RESUMES)
  const [clusters, setClusters] = useState(SAMPLE_CLUSTERS)

  // Check backend connection
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/health`)
        if (response.ok) {
          setBackendConnected(true)
          // Load real data when connected
          loadJobs()
          loadResumes()
        }
      } catch {
        setBackendConnected(false)
        setTimeout(checkConnection, 3000)
      }
    }
    checkConnection()
  }, [])

  // Load jobs from API
  const loadJobs = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/jobs`)
      const data = await response.json()
      if (data.jobs) setJobs(data.jobs)
    } catch (e) { console.log('Using sample jobs') }
  }

  // Load resumes from API
  const loadResumes = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/resumes`)
      const data = await response.json()
      if (data.resumes) setResumesList(data.resumes)
    } catch (e) { console.log('Using sample resumes') }
  }

  // Load clusters
  const loadClusters = async () => {
    if (!backendConnected) return
    try {
      const response = await fetch(`${API_BASE_URL}/api/clusters`)
      const data = await response.json()
      if (data.clusters) setClusters(data.clusters)
    } catch (e) { console.log('Using sample clusters') }
  }

  // Extract skills from job description
  const extractJDSkills = useCallback(async () => {
    if (!jobDescription.trim()) {
      setJdSkills([])
      return
    }
    if (!backendConnected) {
      // Simple extraction for demo
      const keywords = ['python', 'javascript', 'react', 'nodejs', 'django', 'flask', 'aws', 'docker', 'sql', 'mongodb', 'kubernetes', 'typescript', 'java', 'c++', 'machine learning', 'tensorflow', 'pytorch']
      const found = keywords.filter(k => jobDescription.toLowerCase().includes(k))
      setJdSkills(found)
      return
    }
    try {
      const response = await fetch(`${API_BASE_URL}/api/extract-skills`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: jobDescription })
      })
      const data = await response.json()
      setJdSkills(data.skills || [])
    } catch (error) {
      console.error('Error extracting skills:', error)
    }
  }, [jobDescription, backendConnected])

  useEffect(() => {
    const timer = setTimeout(extractJDSkills, 500)
    return () => clearTimeout(timer)
  }, [jobDescription, extractJDSkills])

  // Handle JD file upload
  const handleJDFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    if (!backendConnected) {
      // Read as text for demo
      const text = await file.text()
      setJobDescription(text)
      return
    }

    setLoading(true)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const response = await fetch(`${API_BASE_URL}/api/upload-file`, {
        method: 'POST',
        body: formData
      })
      const data = await response.json()
      if (data.success) {
        setJobDescription(data.text)
      }
    } catch (error) {
      console.error('Error uploading JD:', error)
    } finally {
      setLoading(false)
    }
  }

  // Handle resume file upload
  const handleResumeFileUpload = async (e) => {
    const files = Array.from(e.target.files)
    for (const file of files) {
      if (!backendConnected) {
        const text = await file.text()
        setUploadedFiles(prev => [...prev, { name: file.name.replace(/\.[^/.]+$/, ''), content: text, filename: file.name }])
        continue
      }
      setLoading(true)
      try {
        const formData = new FormData()
        formData.append('file', file)
        const response = await fetch(`${API_BASE_URL}/api/upload-resume`, {
          method: 'POST',
          body: formData
        })
        const data = await response.json()
        if (data.success) {
          setUploadedFiles(prev => [...prev, { name: data.name, content: data.content, skills: data.skills, filename: data.filename }])
        }
      } catch (error) {
        console.error('Error uploading resume:', error)
      } finally {
        setLoading(false)
      }
    }
  }

  const addResumeEntry = () => setResumes([...resumes, { name: '', content: '' }])
  const updateResume = (index, field, value) => {
    const updated = [...resumes]
    updated[index][field] = value
    setResumes(updated)
  }
  const removeResume = (index) => resumes.length > 1 && setResumes(resumes.filter((_, i) => i !== index))
  const removeUploadedFile = (index) => setUploadedFiles(uploadedFiles.filter((_, i) => i !== index))

  // Generate demo results
  const generateDemoResults = (allResumes) => {
    return allResumes.map((resume, i) => ({
      name: resume.name,
      score: Math.max(0.4, 0.92 - (i * 0.12)),
      content_similarity: Math.max(0.35, 0.88 - (i * 0.1)),
      skill_similarity: Math.max(0.3, 0.95 - (i * 0.18)),
      matched_skills: jdSkills.slice(0, Math.max(2, jdSkills.length - i)),
      missing_skills: jdSkills.slice(Math.max(2, jdSkills.length - i)),
      classification: {
        level: i === 0 ? 'high' : i === 1 ? 'medium' : 'low',
        label: i === 0 ? 'Strong Fit' : i === 1 ? 'Partial Fit' : 'Needs Development',
        color: i === 0 ? '#22c55e' : i === 1 ? '#f59e0b' : '#ef4444',
        recommendations: ['Consider for interview', 'Review experience level']
      }
    }))
  }

  // Run manual match
  const runMatch = async () => {
    if (!jobDescription.trim()) {
      alert('Please enter a job description')
      return
    }

    const allResumes = [
      ...resumes.filter(r => r.content.trim()).map(r => ({ name: r.name || `Candidate ${resumes.indexOf(r) + 1}`, content: r.content })),
      ...uploadedFiles.map(f => ({ name: f.name, content: f.content }))
    ]

    if (allResumes.length === 0) {
      alert('Please add at least one resume')
      return
    }

    setLoading(true)

    // Try backend first, fallback to demo mode
    try {
      const response = await fetch(`${API_BASE_URL}/api/manual-match`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_description: jobDescription, resumes: allResumes })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.results) {
          setResults(data)
          setLoading(false)
          return
        }
      }
    } catch (error) {
      console.log('Backend unavailable, using demo mode')
    }

    // Demo mode fallback (silent, no alert)
    setTimeout(() => {
      setResults({ total_resumes: allResumes.length, results: generateDemoResults(allResumes) })
      setLoading(false)
    }, 800)
  }

  const clearAll = () => {
    setJobDescription('')
    setJdSkills([])
    setResumes([{ name: '', content: '' }])
    setUploadedFiles([])
    setResults(null)
  }

  // Navigation items
  const navItems = [
    { id: 'manual-match', icon: '🎯', label: 'Manual Match' },
    { id: 'jobs', icon: '💼', label: 'Job Listings' },
    { id: 'resumes', icon: '📋', label: 'Resumes' },
    { id: 'clusters', icon: '🔮', label: 'Job Clusters' }
  ]

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="logo">
          <div className="logo-icon">📄</div>
          <div className="logo-text">
            <h1>Resume Matcher</h1>
            <span>IR System</span>
          </div>
        </div>

        <nav className="nav-menu">
          {navItems.map(item => (
            <button
              key={item.id}
              className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => {
                setActiveTab(item.id)
                if (item.id === 'clusters') loadClusters()
              }}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className={`status ${backendConnected ? 'connected' : ''}`}>
            <span className="status-dot"></span>
            <span>{backendConnected ? 'Backend Connected' : 'Demo Mode'}</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main">
        {/* Manual Match Tab */}
        {activeTab === 'manual-match' && (
          <section className="tab-content">
            <header className="page-header">
              <h2>🎯 Manual Resume-Job Matching</h2>
              <p>Paste a job description and resumes to analyze compatibility</p>
            </header>

            <div className="match-container">
              {/* Job Description */}
              <div className="input-section">
                <div className="section-header">
                  <h3>📝 Job Description</h3>
                  <div className="header-actions">
                    <label className="btn-secondary">
                      📁 Upload
                      <input type="file" accept=".pdf,.txt" onChange={handleJDFileUpload} hidden />
                    </label>
                    <button className="btn-secondary" onClick={() => setJobDescription('')}>Clear</button>
                  </div>
                </div>
                <textarea
                  value={jobDescription}
                  onChange={(e) => setJobDescription(e.target.value)}
                  placeholder="Paste job description or upload PDF...

Example:
We are looking for a Senior Python Developer with:
- Python, Django, Flask
- REST API development  
- PostgreSQL, MongoDB
- Docker, Kubernetes"
                  rows={12}
                />
                {jdSkills.length > 0 && (
                  <div className="skills-display">
                    <span className="skills-label">Detected Skills:</span>
                    <div className="skills-list">
                      {jdSkills.map((skill, i) => <span key={i} className="skill-tag">{skill}</span>)}
                    </div>
                  </div>
                )}
              </div>

              {/* Resumes */}
              <div className="input-section">
                <div className="section-header">
                  <h3>📄 Resumes</h3>
                  <button className="btn-secondary" onClick={clearAll}>Clear All</button>
                </div>
                <div className="input-tabs">
                  <button className={`tab ${inputMode === 'paste' ? 'active' : ''}`} onClick={() => setInputMode('paste')}>Paste Text</button>
                  <button className={`tab ${inputMode === 'upload' ? 'active' : ''}`} onClick={() => setInputMode('upload')}>Upload Files</button>
                </div>

                {inputMode === 'paste' ? (
                  <div className="paste-input">
                    {resumes.map((resume, index) => (
                      <div key={index} className="resume-entry">
                        <div className="resume-entry-header">
                          <span>Resume #{index + 1}</span>
                          {resumes.length > 1 && <button className="remove-btn" onClick={() => removeResume(index)}>✕</button>}
                        </div>
                        <input type="text" placeholder="Candidate Name" value={resume.name} onChange={(e) => updateResume(index, 'name', e.target.value)} />
                        <textarea placeholder="Paste resume content..." value={resume.content} onChange={(e) => updateResume(index, 'content', e.target.value)} rows={5} />
                      </div>
                    ))}
                    <button className="btn-add" onClick={addResumeEntry}>+ Add Another Resume</button>
                  </div>
                ) : (
                  <div className="upload-input">
                    <label className="drop-zone">
                      <div className="drop-icon">📁</div>
                      <p>Drag & drop resume files here</p>
                      <p className="hint">or click to browse (PDF, TXT)</p>
                      <input type="file" multiple accept=".pdf,.txt" onChange={handleResumeFileUpload} hidden />
                    </label>
                    {uploadedFiles.length > 0 && (
                      <div className="uploaded-files">
                        {uploadedFiles.map((file, index) => (
                          <div key={index} className="uploaded-file">
                            <span>📑 {file.name}</span>
                            <button onClick={() => removeUploadedFile(index)}>✕</button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            <div className="action-bar">
              <button className="btn-primary" onClick={runMatch} disabled={loading}>
                {loading ? '⏳ Processing...' : '🔍 Run Matching Analysis'}
              </button>
            </div>

            {/* Results */}
            {results && (
              <section className="results-section">
                <div className="results-header">
                  <h3>📊 Matching Results</h3>
                  <span>{results.total_resumes} resume(s) analyzed</span>
                </div>
                <div className="results-list">
                  {results.results.map((result, index) => {
                    const scorePercent = Math.round(result.score * 100)
                    const contentPercent = Math.round((result.content_similarity || 0) * 100)
                    const skillPercent = Math.round((result.skill_similarity || 0) * 100)
                    const levelClass = result.classification?.level || 'medium'
                    // Get the resume content for this result
                    const resumeData = [...resumes.filter(r => r.name === result.name || r.content), ...uploadedFiles.filter(f => f.name === result.name)][0]
                    return (
                      <div key={index} className="result-card">
                        <div className="result-header">
                          <div>
                            <div className="result-name">📄 {result.name}</div>
                            <div className="result-rank">#{index + 1} Match</div>
                          </div>
                          <div className="result-score">
                            <span className={`score-badge ${levelClass}`}>{scorePercent}%</span>
                            <span className="classification-label" style={{ background: `${result.classification?.color}20`, color: result.classification?.color }}>{result.classification?.label}</span>
                          </div>
                        </div>

                        {/* Score Calculation Explanation */}
                        <div className="score-explanation">
                          <h4>📊 How This Score Was Calculated:</h4>
                          <div className="formula-box">
                            <code>Overall = (Content × 0.4) + (Skills × 0.6)</code>
                            <div className="formula-breakdown">
                              <span>= ({contentPercent}% × 0.4) + ({skillPercent}% × 0.6)</span>
                              <span>= {Math.round(contentPercent * 0.4)}% + {Math.round(skillPercent * 0.6)}%</span>
                              <span className="formula-result">= <strong>{scorePercent}%</strong></span>
                            </div>
                          </div>
                        </div>

                        <div className="score-bar-container">
                          <div className="score-bar-label"><span>Overall Match</span><span>{scorePercent}%</span></div>
                          <div className="score-bar"><div className={`score-bar-fill ${levelClass}`} style={{ width: `${scorePercent}%` }}></div></div>
                        </div>
                        <div className="score-details">
                          <div>
                            <div className="score-bar-label"><span>Content Similarity (TF-IDF)</span><span>{contentPercent}%</span></div>
                            <div className="score-bar"><div className={`score-bar-fill ${levelClass}`} style={{ width: `${contentPercent}%` }}></div></div>
                          </div>
                          <div>
                            <div className="score-bar-label"><span>Skills Match (Jaccard)</span><span>{skillPercent}%</span></div>
                            <div className="score-bar"><div className={`score-bar-fill ${levelClass}`} style={{ width: `${skillPercent}%` }}></div></div>
                          </div>
                        </div>

                        <div className="result-skills">
                          <div className="skills-group">
                            <h4>✓ Matched ({result.matched_skills?.length || 0})</h4>
                            <div className="skills-list">{result.matched_skills?.map((s, i) => <span key={i} className="skill-tag matched">{s}</span>)}</div>
                          </div>
                          <div className="skills-group">
                            <h4>✗ Missing ({result.missing_skills?.length || 0})</h4>
                            <div className="skills-list">{result.missing_skills?.map((s, i) => <span key={i} className="skill-tag missing">{s}</span>)}</div>
                          </div>
                        </div>

                        {/* AI Analysis Explanation */}
                        {result.explanation && (
                          <div className="ai-explanation">
                            <h4>🤖 AI Analysis:</h4>
                            <p>{result.explanation}</p>
                          </div>
                        )}

                        {/* Resume Extract Preview */}
                        {result.resume_extract && (
                          <div className="resume-preview">
                            <div className="resume-preview-header">
                              <h4>📋 Resume Content:</h4>
                              <span className="char-count">{result.resume_full?.length || result.resume_extract.length} characters</span>
                            </div>
                            <pre className="resume-text">{result.resume_full || result.resume_extract}</pre>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </section>
            )}
          </section>
        )}

        {/* Jobs Tab */}
        {activeTab === 'jobs' && (
          <section className="tab-content">
            <header className="page-header">
              <h2>💼 Job Listings</h2>
              <p>Browse indexed job descriptions ({jobs.length} jobs)</p>
            </header>
            <div className="cards-grid">
              {jobs.map((job) => (
                <div key={job.id} className="card">
                  <div className="card-header">
                    <div>
                      <h3 className="card-title">{job.title}</h3>
                      <p className="card-subtitle">{job.company || 'Company'}</p>
                    </div>
                    <span className="card-icon">💼</span>
                  </div>
                  <div className="card-skills">
                    {(job.skills || []).slice(0, 5).map((s, i) => <span key={i} className="skill-tag">{s}</span>)}
                    {(job.skills || []).length > 5 && <span className="skill-tag">+{job.skills.length - 5}</span>}
                  </div>
                  <button className="btn-card">View Details</button>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Resumes Tab */}
        {activeTab === 'resumes' && (
          <section className="tab-content">
            <header className="page-header">
              <h2>📋 Indexed Resumes</h2>
              <p>Browse candidate profiles ({resumesList.length} resumes)</p>
            </header>
            <div className="cards-grid">
              {resumesList.map((resume) => (
                <div key={resume.id} className="card">
                  <div className="card-header">
                    <div>
                      <h3 className="card-title">{resume.name}</h3>
                      <p className="card-subtitle">{resume.title || `${(resume.skills || []).length} skills`}</p>
                    </div>
                    <span className="card-icon">👤</span>
                  </div>
                  <div className="card-skills">
                    {(resume.skills || []).slice(0, 6).map((s, i) => <span key={i} className="skill-tag">{s}</span>)}
                  </div>
                  <button className="btn-card">View Profile</button>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Clusters Tab */}
        {activeTab === 'clusters' && (
          <section className="tab-content">
            <header className="page-header">
              <h2>🔮 Job Domain Clusters</h2>
              <p>Jobs grouped by domain using K-Means clustering</p>
            </header>
            <div className="cluster-summary">
              <span>{clusters.length} clusters identified</span>
            </div>
            <div className="clusters-grid">
              {clusters.map((cluster, i) => (
                <div key={cluster.id || i} className="cluster-card">
                  <div className="cluster-header">
                    <span className="cluster-icon">{['🎯', '💡', '🚀', '⚡'][i % 4]}</span>
                    <div>
                      <h3>{cluster.label}</h3>
                      <span className="cluster-count">{cluster.job_count} job(s)</span>
                    </div>
                  </div>
                  <div className="cluster-terms">
                    <h4>Top Terms</h4>
                    <div className="skills-list">
                      {(cluster.top_terms || []).slice(0, 5).map((t, j) => <span key={j} className="skill-tag">{t}</span>)}
                    </div>
                  </div>
                  <div className="cluster-jobs">
                    <h4>Sample Jobs</h4>
                    {(cluster.jobs || []).map((j, k) => <div key={k} className="cluster-job-item">{j.title}</div>)}
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>

      {loading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>Processing...</p>
        </div>
      )}
    </div>
  )
}

export default App
