import sqlite3
from datetime import datetime
from typing import List, Dict, Any

class DatabaseManager:
    def __init__(self, db_path: str = "phishing_detection.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    url TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    status TEXT NOT NULL
                )
            ''')
            conn.commit()
    
    def add_scan(self, url: str, risk_score: float, status: str) -> int:
        """Add a new scan result to the database"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO scans (timestamp, url, risk_score, status)
                VALUES (?, ?, ?, ?)
            ''', (timestamp, url, risk_score, status))
            conn.commit()
            return cursor.lastrowid
    
    def get_recent_scans(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent scans from the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, url, risk_score, status
                FROM scans
                ORDER BY id DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            return [
                {
                    'timestamp': row[0],
                    'url': row[1],
                    'risk_score': row[2],
                    'status': row[3]
                }
                for row in rows
            ]
    
    def get_scan_statistics(self) -> Dict[str, Any]:
        """Get scan statistics for dashboard"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total scans
            cursor.execute('SELECT COUNT(*) FROM scans')
            total_scans = cursor.fetchone()[0]
            
            # Phishing detected
            cursor.execute('SELECT COUNT(*) FROM scans WHERE status = "PHISHING"')
            phishing_detected = cursor.fetchone()[0]
            
            # Legitimate URLs
            cursor.execute('SELECT COUNT(*) FROM scans WHERE status = "SAFE"')
            legitimate_urls = cursor.fetchone()[0]
            
            # Detection rate
            detection_rate = (phishing_detected / max(total_scans, 1)) * 100
            
            return {
                'total_scans': total_scans,
                'phishing_detected': phishing_detected,
                'legitimate_urls': legitimate_urls,
                'detection_rate': round(detection_rate, 1)
            }