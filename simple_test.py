#!/usr/bin/env python3
"""
Simple test server to verify database integration
"""

from flask import Flask, jsonify, request
from database_manager import DatabaseManager
import json

app = Flask(__name__)
db_manager = DatabaseManager()

@app.route('/')
def home():
    return '''
    <h1>Phishing Detection Database Test</h1>
    <p><a href="/stats">View Statistics</a></p>
    <p><a href="/recent">View Recent Scans</a></p>
    <p><a href="/add-test">Add Test Data</a></p>
    '''

@app.route('/stats')
def get_stats():
    stats = db_manager.get_scan_statistics()
    return jsonify(stats)

@app.route('/recent')
def get_recent():
    scans = db_manager.get_recent_scans(limit=10)
    return jsonify(scans)

@app.route('/add-test')
def add_test():
    # Add some test data
    test_urls = [
        ("https://secure-banking-login.tk/verify", 0.89, "PHISHING"),
        ("https://www.microsoft.com", 0.01, "SAFE"),
        ("https://paypal-security-check.ml/account", 0.92, "PHISHING"),
        ("https://www.github.com", 0.03, "SAFE"),
    ]
    
    added_ids = []
    for url, risk_score, status in test_urls:
        scan_id = db_manager.add_scan(url, risk_score, status)
        added_ids.append(scan_id)
    
    return jsonify({
        "message": f"Added {len(test_urls)} test scans",
        "scan_ids": added_ids,
        "stats": db_manager.get_scan_statistics()
    })

if __name__ == '__main__':
    print("Starting simple test server...")
    print("Visit: http://localhost:5001")
    app.run(debug=True, port=5001)