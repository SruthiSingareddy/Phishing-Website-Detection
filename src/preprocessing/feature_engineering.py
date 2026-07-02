"""
Advanced Feature Engineering for Phishing URL Detection
Combines traditional lexical features with modern deep learning preprocessing
"""

import re
import tldextract
import whois
import socket
import requests
from urllib.parse import urlparse, parse_qs
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import dns.resolver
from datetime import datetime
import ssl
import concurrent.futures
from dataclasses import dataclass
import hashlib


@dataclass
class URLFeatures:
    """Container for extracted URL features"""
    # Basic URL components
    url: str
    protocol: str
    domain: str
    subdomain: str
    path: str
    query: str
    fragment: str
    
    # Lexical features
    url_length: int
    domain_length: int
    path_length: int
    query_length: int
    
    # Character-based features
    num_dots: int
    num_hyphens: int
    num_underscores: int
    num_slashes: int
    num_question_marks: int
    num_equal_signs: int
    num_at_symbols: int
    num_ampersands: int
    num_exclamation: int
    num_space: int
    num_tilde: int
    num_comma: int
    num_plus: int
    num_asterisk: int
    num_hashtag: int
    num_dollar: int
    num_percent: int
    
    # Advanced lexical features
    entropy: float
    vowel_consonant_ratio: float
    digit_letter_ratio: float
    uppercase_lowercase_ratio: float
    
    # Domain-based features
    is_ip_address: bool
    has_port: bool
    domain_tokens: int
    subdomain_length: int
    tld: str
    tld_length: int
    
    # Suspicious patterns
    has_suspicious_words: bool
    suspicious_word_count: int
    has_url_shortener: bool
    has_redirect: bool
    
    # Network features
    dns_record_exists: bool
    whois_registered: bool
    domain_age_days: Optional[int]
    ssl_certificate_valid: bool
    
    # Behavioral features
    redirect_count: int
    final_url_different: bool
    response_time: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ML processing"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class AdvancedFeatureExtractor:
    """Advanced feature extraction for phishing detection"""
    
    def __init__(self):
        self.suspicious_words = {
            'secure', 'account', 'update', 'confirm', 'verify', 'login', 'signin',
            'banking', 'paypal', 'ebay', 'amazon', 'microsoft', 'apple', 'google',
            'facebook', 'twitter', 'instagram', 'linkedin', 'netflix', 'spotify',
            'suspended', 'limited', 'expired', 'urgent', 'immediate', 'action',
            'click', 'here', 'now', 'today', 'winner', 'congratulations',
            'free', 'bonus', 'gift', 'prize', 'offer', 'deal', 'discount'
        }
        
        self.url_shorteners = {
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
            'buff.ly', 'adf.ly', 'bl.ink', 'lnkd.in', 'short.link', 'tiny.cc',
            'rb.gy', 'cutt.ly', 'shorturl.at', 'v.gd', 'x.co', 'po.st'
        }
        
        self.tld_suspicious = {
            'tk', 'ml', 'ga', 'cf', 'click', 'download', 'link', 'zip',
            'review', 'country', 'kim', 'science', 'work'
        }
    
    def extract_features(self, url: str, timeout: int = 5) -> URLFeatures:
        """Extract comprehensive features from URL"""
        try:
            # Parse URL components
            parsed = urlparse(url)
            extracted = tldextract.extract(url)
            
            # Basic components
            protocol = parsed.scheme or 'http'
            domain = extracted.domain + '.' + extracted.suffix if extracted.suffix else extracted.domain
            subdomain = extracted.subdomain
            path = parsed.path
            query = parsed.query
            fragment = parsed.fragment
            
            # Calculate lexical features
            lexical_features = self._extract_lexical_features(url, domain, path, query)
            
            # Calculate character-based features
            char_features = self._extract_character_features(url)
            
            # Calculate advanced lexical features
            advanced_features = self._extract_advanced_lexical_features(url)
            
            # Calculate domain-based features
            domain_features = self._extract_domain_features(url, domain, subdomain, extracted.suffix)
            
            # Calculate suspicious pattern features
            suspicious_features = self._extract_suspicious_patterns(url, domain)
            
            # Calculate network features (with timeout)
            network_features = self._extract_network_features(domain, timeout)
            
            # Calculate behavioral features
            behavioral_features = self._extract_behavioral_features(url, timeout)
            
            return URLFeatures(
                url=url,
                protocol=protocol,
                domain=domain,
                subdomain=subdomain,
                path=path,
                query=query,
                fragment=fragment,
                **lexical_features,
                **char_features,
                **advanced_features,
                **domain_features,
                **suspicious_features,
                **network_features,
                **behavioral_features
            )
            
        except Exception as e:
            print(f"Error extracting features from {url}: {e}")
            return self._get_default_features(url)
    
    def _extract_lexical_features(self, url: str, domain: str, path: str, query: str) -> Dict[str, int]:
        """Extract basic lexical features"""
        return {
            'url_length': len(url),
            'domain_length': len(domain),
            'path_length': len(path),
            'query_length': len(query)
        }
    
    def _extract_character_features(self, url: str) -> Dict[str, int]:
        """Extract character-based features"""
        return {
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_underscores': url.count('_'),
            'num_slashes': url.count('/'),
            'num_question_marks': url.count('?'),
            'num_equal_signs': url.count('='),
            'num_at_symbols': url.count('@'),
            'num_ampersands': url.count('&'),
            'num_exclamation': url.count('!'),
            'num_space': url.count(' '),
            'num_tilde': url.count('~'),
            'num_comma': url.count(','),
            'num_plus': url.count('+'),
            'num_asterisk': url.count('*'),
            'num_hashtag': url.count('#'),
            'num_dollar': url.count('$'),
            'num_percent': url.count('%')
        }
    
    def _extract_advanced_lexical_features(self, url: str) -> Dict[str, float]:
        """Extract advanced lexical features"""
        # Shannon entropy
        entropy = self._calculate_entropy(url)
        
        # Character ratios
        vowels = sum(1 for c in url.lower() if c in 'aeiou')
        consonants = sum(1 for c in url.lower() if c.isalpha() and c not in 'aeiou')
        vowel_consonant_ratio = vowels / max(consonants, 1)
        
        digits = sum(1 for c in url if c.isdigit())
        letters = sum(1 for c in url if c.isalpha())
        digit_letter_ratio = digits / max(letters, 1)
        
        uppercase = sum(1 for c in url if c.isupper())
        lowercase = sum(1 for c in url if c.islower())
        uppercase_lowercase_ratio = uppercase / max(lowercase, 1)
        
        return {
            'entropy': entropy,
            'vowel_consonant_ratio': vowel_consonant_ratio,
            'digit_letter_ratio': digit_letter_ratio,
            'uppercase_lowercase_ratio': uppercase_lowercase_ratio
        }
    
    def _extract_domain_features(self, url: str, domain: str, subdomain: str, tld: str) -> Dict[str, Any]:
        """Extract domain-based features"""
        # Check if domain is IP address
        is_ip = self._is_ip_address(domain)
        
        # Check for port
        has_port = ':' in urlparse(url).netloc and not is_ip
        
        # Domain tokens (split by dots)
        domain_tokens = len(domain.split('.'))
        
        # Subdomain length
        subdomain_length = len(subdomain) if subdomain else 0
        
        # TLD features
        tld_length = len(tld) if tld else 0
        
        return {
            'is_ip_address': is_ip,
            'has_port': has_port,
            'domain_tokens': domain_tokens,
            'subdomain_length': subdomain_length,
            'tld': tld or '',
            'tld_length': tld_length
        }
    
    def _extract_suspicious_patterns(self, url: str, domain: str) -> Dict[str, Any]:
        """Extract suspicious pattern features"""
        url_lower = url.lower()
        
        # Check for suspicious words
        suspicious_words_found = [word for word in self.suspicious_words if word in url_lower]
        has_suspicious_words = len(suspicious_words_found) > 0
        suspicious_word_count = len(suspicious_words_found)
        
        # Check for URL shorteners
        has_url_shortener = any(shortener in domain.lower() for shortener in self.url_shorteners)
        
        # Check for redirect patterns
        has_redirect = any(pattern in url_lower for pattern in ['redirect', 'redir', 'r.php', 'link.php'])
        
        return {
            'has_suspicious_words': has_suspicious_words,
            'suspicious_word_count': suspicious_word_count,
            'has_url_shortener': has_url_shortener,
            'has_redirect': has_redirect
        }
    
    def _extract_network_features(self, domain: str, timeout: int) -> Dict[str, Any]:
        """Extract network-based features"""
        features = {
            'dns_record_exists': False,
            'whois_registered': False,
            'domain_age_days': None,
            'ssl_certificate_valid': False
        }
        
        try:
            # DNS lookup
            try:
                dns.resolver.resolve(domain, 'A', lifetime=timeout)
                features['dns_record_exists'] = True
            except:
                pass
            
            # WHOIS lookup
            try:
                w = whois.whois(domain)
                if w.creation_date:
                    creation_date = w.creation_date
                    if isinstance(creation_date, list):
                        creation_date = creation_date[0]
                    
                    if creation_date:
                        features['whois_registered'] = True
                        age = (datetime.now() - creation_date).days
                        features['domain_age_days'] = age
            except:
                pass
            
            # SSL certificate check
            try:
                context = ssl.create_default_context()
                with socket.create_connection((domain, 443), timeout=timeout) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        features['ssl_certificate_valid'] = True
            except:
                pass
                
        except Exception as e:
            print(f"Network feature extraction error for {domain}: {e}")
        
        return features
    
    def _extract_behavioral_features(self, url: str, timeout: int) -> Dict[str, Any]:
        """Extract behavioral features through HTTP requests"""
        features = {
            'redirect_count': 0,
            'final_url_different': False,
            'response_time': None
        }
        
        try:
            start_time = datetime.now()
            
            # Follow redirects and measure
            session = requests.Session()
            session.max_redirects = 10
            
            response = session.get(url, timeout=timeout, allow_redirects=True)
            
            end_time = datetime.now()
            features['response_time'] = (end_time - start_time).total_seconds()
            
            # Count redirects
            features['redirect_count'] = len(response.history)
            
            # Check if final URL is different
            final_url = response.url
            features['final_url_different'] = self._normalize_url(url) != self._normalize_url(final_url)
            
        except Exception as e:
            print(f"Behavioral feature extraction error for {url}: {e}")
        
        return features
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        text_length = len(text)
        
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _is_ip_address(self, domain: str) -> bool:
        """Check if domain is an IP address"""
        try:
            socket.inet_aton(domain)
            return True
        except socket.error:
            return False
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison"""
        parsed = urlparse(url.lower())
        return f"{parsed.netloc}{parsed.path}"
    
    def _get_default_features(self, url: str) -> URLFeatures:
        """Return default features when extraction fails"""
        return URLFeatures(
            url=url,
            protocol='',
            domain='',
            subdomain='',
            path='',
            query='',
            fragment='',
            url_length=len(url),
            domain_length=0,
            path_length=0,
            query_length=0,
            num_dots=0,
            num_hyphens=0,
            num_underscores=0,
            num_slashes=0,
            num_question_marks=0,
            num_equal_signs=0,
            num_at_symbols=0,
            num_ampersands=0,
            num_exclamation=0,
            num_space=0,
            num_tilde=0,
            num_comma=0,
            num_plus=0,
            num_asterisk=0,
            num_hashtag=0,
            num_dollar=0,
            num_percent=0,
            entropy=0.0,
            vowel_consonant_ratio=0.0,
            digit_letter_ratio=0.0,
            uppercase_lowercase_ratio=0.0,
            is_ip_address=False,
            has_port=False,
            domain_tokens=0,
            subdomain_length=0,
            tld='',
            tld_length=0,
            has_suspicious_words=False,
            suspicious_word_count=0,
            has_url_shortener=False,
            has_redirect=False,
            dns_record_exists=False,
            whois_registered=False,
            domain_age_days=None,
            ssl_certificate_valid=False,
            redirect_count=0,
            final_url_different=False,
            response_time=None
        )


class BatchFeatureExtractor:
    """Batch processing for feature extraction with parallel processing"""
    
    def __init__(self, max_workers: int = 10, timeout: int = 5):
        self.extractor = AdvancedFeatureExtractor()
        self.max_workers = max_workers
        self.timeout = timeout
    
    def extract_features_batch(self, urls: List[str]) -> pd.DataFrame:
        """Extract features for multiple URLs in parallel"""
        features_list = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(self.extractor.extract_features, url, self.timeout): url 
                for url in urls
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    features = future.result()
                    features_list.append(features.to_dict())
                except Exception as e:
                    print(f"Error processing {url}: {e}")
                    # Add default features for failed URLs
                    default_features = self.extractor._get_default_features(url)
                    features_list.append(default_features.to_dict())
        
        return pd.DataFrame(features_list)


def create_feature_pipeline(urls: List[str], max_workers: int = 10) -> pd.DataFrame:
    """Create complete feature extraction pipeline"""
    extractor = BatchFeatureExtractor(max_workers=max_workers)
    return extractor.extract_features_batch(urls)