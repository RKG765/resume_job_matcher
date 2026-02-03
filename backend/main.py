"""
Resume-Job Description Matching System
Main entry point for the backend server.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from api.app import create_app

if __name__ == '__main__':
    app = create_app()
    print("\n" + "="*50)
    print("Resume-Job Matching System Backend")
    print("="*50)
    print(f"Server running at: http://127.0.0.1:5000")
    print("API endpoints:")
    print("  GET  /api/health        - Health check")
    print("  GET  /api/jobs          - List all jobs")
    print("  GET  /api/resumes       - List all resumes")
    print("  POST /api/manual-match  - Manual matching mode")
    print("  GET  /api/rank/<id>     - Rank resumes for job")
    print("  POST /api/skill-gap     - Analyze skill gaps")
    print("  GET  /api/clusters      - Get job clusters")
    print("="*50 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
