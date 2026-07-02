#!/usr/bin/env python3
"""
Test script to verify database functionality
"""

from database_manager import DatabaseManager
import os

def test_database():
    print("Testing Database Manager...")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Test adding a scan
    print("\n1. Adding test scan...")
    scan_id = db_manager.add_scan(
        url="https://test-phishing-site.tk/login",
        risk_score=0.85,
        status="PHISHING"
    )
    print(f"Added scan with ID: {scan_id}")
    
    # Test adding another scan
    scan_id2 = db_manager.add_scan(
        url="https://www.google.com",
        risk_score=0.02,
        status="SAFE"
    )
    print(f"Added scan with ID: {scan_id2}")
    
    # Test getting statistics
    print("\n2. Getting statistics...")
    stats = db_manager.get_scan_statistics()
    print(f"Total scans: {stats['total_scans']}")
    print(f"Phishing detected: {stats['phishing_detected']}")
    print(f"Legitimate URLs: {stats['legitimate_urls']}")
    print(f"Detection rate: {stats['detection_rate']}%")
    
    # Test getting recent scans
    print("\n3. Getting recent scans...")
    recent_scans = db_manager.get_recent_scans(limit=5)
    for scan in recent_scans:
        print(f"  {scan['timestamp']} - {scan['url'][:50]}... - {scan['status']}")
    
    print("\n✅ Database test completed successfully!")

if __name__ == "__main__":
    test_database()