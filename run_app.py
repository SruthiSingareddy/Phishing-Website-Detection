#!/usr/bin/env python3
"""
Startup script for the Phishing Detection System
Ensures the app runs from the correct directory
"""

import os
import sys
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Change to the script directory
    os.chdir(script_dir)
    
    print("=" * 60)
    print("ADVANCED PHISHING DETECTION SYSTEM")
    print("=" * 60)
    print("URLNet + Transformer Ensemble Model")
    print("Dynamic SQLite Database Integration")
    print("Real-time Analysis | ML-based Risk Assessment")
    print("=" * 60)
    print(f"Working directory: {script_dir}")
    print("Web Interface: http://localhost:5000")
    print("Admin Dashboard: http://localhost:5000/admin/login")
    print("Admin Credentials: admin / admin123")
    print("=" * 60)
    
    # Import and run the Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"Error importing app: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install flask sqlite3 numpy pandas scikit-learn")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()