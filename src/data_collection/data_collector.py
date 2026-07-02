"""
Advanced Data Collection for Phishing Detection
Collects URLs from multiple sources and creates balanced datasets
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import json
import sqlite3
from datetime import datetime, timedelta
import concurrent.futures
from urllib.parse import urlparse
import hashlib
import os
from pathlib import Path


class PhishingDataCollector:
    """Collect phishing and legitimate URLs from multiple sources"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.data_dir / "urls.db"
        self._init_database()
        
        # API endpoints and sources
        self.sources = {
            'phishtank': 'http://data.phishtank.com/data/online-valid.json',
            'openphish': 'https://openphish.com/feed.txt',
            'alexa_top': 'https://www.alexa.com/topsites',  # Alternative sources needed
            'tranco_list': 'https://tranco-list.eu/list/1m/1000000'
        }
        
        # Rate limiting
        self.request_delay = 1.0  # seconds between requests
        self.last_request_time = 0
    
    def _init_database(self):
        """Initialize SQLite database for URL storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                label INTEGER NOT NULL,  -- 0: legitimate, 1: phishing
                source TEXT NOT NULL,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                url_hash TEXT UNIQUE,
                domain TEXT,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_url_hash ON urls(url_hash);
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_label ON urls(label);
        ''')
        
        conn.commit()
        conn.close()
    
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def _get_url_hash(self, url: str) -> str:
        """Generate hash for URL deduplication"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc.lower()
        except:
            return ""
    
    def collect_phishtank_data(self, limit: Optional[int] = None) -> List[Dict]:
        """Collect phishing URLs from PhishTank"""
        print("Collecting data from PhishTank...")
        
        try:
            self._rate_limit()
            response = requests.get(self.sources['phishtank'], timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            urls = []
            for entry in data[:limit] if limit else data:
                if entry.get('online') == 'yes':  # Only active phishing sites
                    url = entry.get('url', '').strip()
                    if url:
                        urls.append({
                            'url': url,
                            'label': 1,  # phishing
                            'source': 'phishtank',
                            'url_hash': self._get_url_hash(url),
                            'domain': self._extract_domain(url)
                        })
            
            print(f"Collected {len(urls)} URLs from PhishTank")
            return urls
            
        except Exception as e:
            print(f"Error collecting PhishTank data: {e}")
            return []
    
    def collect_openphish_data(self, limit: Optional[int] = None) -> List[Dict]:
        """Collect phishing URLs from OpenPhish"""
        print("Collecting data from OpenPhish...")
        
        try:
            self._rate_limit()
            response = requests.get(self.sources['openphish'], timeout=30)
            response.raise_for_status()
            
            urls = []
            lines = response.text.strip().split('\n')
            
            for line in lines[:limit] if limit else lines:
                url = line.strip()
                if url and url.startswith(('http://', 'https://')):
                    urls.append({
                        'url': url,
                        'label': 1,  # phishing
                        'source': 'openphish',
                        'url_hash': self._get_url_hash(url),
                        'domain': self._extract_domain(url)
                    })
            
            print(f"Collected {len(urls)} URLs from OpenPhish")
            return urls
            
        except Exception as e:
            print(f"Error collecting OpenPhish data: {e}")
            return []
    
    def collect_legitimate_urls(self, limit: Optional[int] = None) -> List[Dict]:
        """Collect legitimate URLs from various sources"""
        print("Collecting legitimate URLs...")
        
        urls = []
        
        # Popular legitimate websites (manually curated)
        legitimate_sites = [
            'https://www.google.com', 'https://www.youtube.com', 'https://www.facebook.com',
            'https://www.amazon.com', 'https://www.wikipedia.org', 'https://www.twitter.com',
            'https://www.instagram.com', 'https://www.linkedin.com', 'https://www.reddit.com',
            'https://www.netflix.com', 'https://www.microsoft.com', 'https://www.apple.com',
            'https://www.github.com', 'https://www.stackoverflow.com', 'https://www.medium.com',
            'https://www.cnn.com', 'https://www.bbc.com', 'https://www.nytimes.com',
            'https://www.ebay.com', 'https://www.paypal.com', 'https://www.adobe.com',
            'https://www.salesforce.com', 'https://www.dropbox.com', 'https://www.spotify.com',
            'https://www.zoom.us', 'https://www.slack.com', 'https://www.twitch.tv',
            'https://www.pinterest.com', 'https://www.tumblr.com', 'https://www.quora.com'
        ]
        
        # Add variations and subpages
        for base_url in legitimate_sites:
            # Base URL
            urls.append({
                'url': base_url,
                'label': 0,  # legitimate
                'source': 'curated_legitimate',
                'url_hash': self._get_url_hash(base_url),
                'domain': self._extract_domain(base_url)
            })
            
            # Add some common subpages
            subpages = ['/about', '/contact', '/help', '/support', '/login', '/signup']
            for subpage in subpages:
                full_url = base_url + subpage
                urls.append({
                    'url': full_url,
                    'label': 0,  # legitimate
                    'source': 'curated_legitimate',
                    'url_hash': self._get_url_hash(full_url),
                    'domain': self._extract_domain(full_url)
                })
        
        # Generate additional legitimate URLs from common patterns
        legitimate_domains = [
            'edu', 'gov', 'org', 'com', 'net'
        ]
        
        common_legitimate_patterns = [
            'university', 'college', 'school', 'library', 'museum',
            'hospital', 'clinic', 'bank', 'insurance', 'news',
            'blog', 'shop', 'store', 'market', 'service'
        ]
        
        for pattern in common_legitimate_patterns:
            for tld in legitimate_domains:
                url = f"https://www.{pattern}.{tld}"
                urls.append({
                    'url': url,
                    'label': 0,  # legitimate
                    'source': 'generated_legitimate',
                    'url_hash': self._get_url_hash(url),
                    'domain': self._extract_domain(url)
                })
        
        if limit:
            urls = urls[:limit]
        
        print(f"Collected {len(urls)} legitimate URLs")
        return urls
    
    def save_urls_to_db(self, urls: List[Dict]):
        """Save URLs to database with deduplication"""
        if not urls:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        inserted_count = 0
        duplicate_count = 0
        
        for url_data in urls:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO urls (url, label, source, url_hash, domain)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    url_data['url'],
                    url_data['label'],
                    url_data['source'],
                    url_data['url_hash'],
                    url_data['domain']
                ))
                
                if cursor.rowcount > 0:
                    inserted_count += 1
                else:
                    duplicate_count += 1
                    
            except Exception as e:
                print(f"Error inserting URL {url_data['url']}: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"Inserted {inserted_count} new URLs, {duplicate_count} duplicates skipped")
    
    def get_dataset(self, 
                   balance_classes: bool = True,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get balanced dataset from database"""
        conn = sqlite3.connect(self.db_path)
        
        # Get all URLs
        df = pd.read_sql_query('''
            SELECT url, label, source, domain, collected_at
            FROM urls
            WHERE is_active = 1
            ORDER BY collected_at DESC
        ''', conn)
        
        conn.close()
        
        if df.empty:
            raise ValueError("No URLs found in database. Please collect data first.")
        
        print(f"Total URLs in database: {len(df)}")
        print(f"Phishing URLs: {len(df[df['label'] == 1])}")
        print(f"Legitimate URLs: {len(df[df['label'] == 0])}")
        
        # Balance classes if requested
        if balance_classes:
            phishing_df = df[df['label'] == 1]
            legitimate_df = df[df['label'] == 0]
            
            min_count = min(len(phishing_df), len(legitimate_df))
            
            if min_count == 0:
                raise ValueError("Need both phishing and legitimate URLs for balanced dataset")
            
            # Sample equal amounts from each class
            phishing_sample = phishing_df.sample(n=min_count, random_state=random_state)
            legitimate_sample = legitimate_df.sample(n=min_count, random_state=random_state)
            
            df = pd.concat([phishing_sample, legitimate_sample], ignore_index=True)
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            print(f"Balanced dataset: {len(df)} URLs ({min_count} per class)")
        
        # Split into train and test
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['label']
        )
        
        return train_df, test_df
    
    def collect_all_data(self, 
                        phishing_limit: Optional[int] = 5000,
                        legitimate_limit: Optional[int] = 5000):
        """Collect data from all sources"""
        print("Starting comprehensive data collection...")
        
        all_urls = []
        
        # Collect phishing URLs
        phishtank_urls = self.collect_phishtank_data(limit=phishing_limit//2 if phishing_limit else None)
        all_urls.extend(phishtank_urls)
        
        openphish_urls = self.collect_openphish_data(limit=phishing_limit//2 if phishing_limit else None)
        all_urls.extend(openphish_urls)
        
        # Collect legitimate URLs
        legitimate_urls = self.collect_legitimate_urls(limit=legitimate_limit)
        all_urls.extend(legitimate_urls)
        
        # Save to database
        self.save_urls_to_db(all_urls)
        
        print(f"Data collection complete. Total URLs collected: {len(all_urls)}")
        
        return all_urls
    
    def export_dataset(self, output_path: str, format: str = 'csv'):
        """Export dataset to file"""
        train_df, test_df = self.get_dataset()
        
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        if format.lower() == 'csv':
            train_df.to_csv(output_path / 'train_urls.csv', index=False)
            test_df.to_csv(output_path / 'test_urls.csv', index=False)
        elif format.lower() == 'json':
            train_df.to_json(output_path / 'train_urls.json', orient='records', indent=2)
            test_df.to_json(output_path / 'test_urls.json', orient='records', indent=2)
        
        print(f"Dataset exported to {output_path}")
        print(f"Training set: {len(train_df)} URLs")
        print(f"Test set: {len(test_df)} URLs")


def main():
    """Main function for data collection"""
    collector = PhishingDataCollector()
    
    # Collect data from all sources
    collector.collect_all_data(phishing_limit=10000, legitimate_limit=10000)
    
    # Export dataset
    collector.export_dataset('data/processed')


if __name__ == "__main__":
    main()