#!/usr/bin/env python3
"""
Quick run script for Advanced Phishing Detection System
"""

from app import app

if __name__ == '__main__':
    print("=" * 60)
    print("ADVANCED PHISHING DETECTION SYSTEM")
    print("=" * 60)
    print("Starting Flask web application...")
    print("Access the application at: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)