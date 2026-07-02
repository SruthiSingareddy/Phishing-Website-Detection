#!/usr/bin/env python3
"""
Advanced Phishing Detection System - Main Entry Point
Comprehensive phishing URL detection using modern deep learning
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import logging
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import modules
from data_collection.data_collector import PhishingDataCollector
from preprocessing.feature_engineering import create_feature_pipeline
from training.train_models import ModelTrainer
from api.main import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return get_default_config()
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        'data': {
            'processed_data_dir': 'data/processed',
            'balance_classes': True,
            'test_size': 0.2,
            'random_state': 42
        },
        'models': {
            'urlnet': {
                'vocab_size': 128,
                'embed_dim': 128,
                'num_filters': 256,
                'filter_sizes': [3, 4, 5, 6],
                'num_heads': 8,
                'hidden_dim': 512,
                'num_classes': 2,
                'max_length': 200,
                'dropout': 0.3
            },
            'transformer': {
                'hidden_size': 768,
                'num_attention_heads': 12,
                'num_hidden_layers': 6,
                'intermediate_size': 3072,
                'max_position_embeddings': 200,
                'num_classes': 2,
                'dropout': 0.1
            }
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'epochs': 50,
            'patience': 7,
            'max_length': 200,
            'model_dir': 'models'
        },
        'collection': {
            'sources': {
                'phishtank': {'limit': 5000},
                'openphish': {'limit': 5000},
                'legitimate': {'limit': 10000}
            }
        }
    }


def collect_data(config: Dict[str, Any]):
    """Collect phishing and legitimate URLs"""
    logger.info("Starting data collection...")
    
    collector = PhishingDataCollector(data_dir="data")
    
    # Get limits from config
    phishing_limit = (
        config['collection']['sources']['phishtank']['limit'] + 
        config['collection']['sources']['openphish']['limit']
    )
    legitimate_limit = config['collection']['sources']['legitimate']['limit']
    
    # Collect data
    collector.collect_all_data(
        phishing_limit=phishing_limit,
        legitimate_limit=legitimate_limit
    )
    
    # Export processed dataset
    collector.export_dataset(
        output_path=config['data']['processed_data_dir'],
        format='csv'
    )
    
    logger.info("Data collection completed successfully")


def extract_features(config: Dict[str, Any]):
    """Extract features from collected URLs"""
    logger.info("Starting feature extraction...")
    
    import pandas as pd
    
    # Load URLs
    data_dir = config['data']['processed_data_dir']
    train_df = pd.read_csv(f"{data_dir}/train_urls.csv")
    test_df = pd.read_csv(f"{data_dir}/test_urls.csv")
    
    # Extract features for a sample (to avoid long processing)
    sample_size = min(1000, len(train_df))
    sample_urls = train_df.sample(n=sample_size, random_state=42)['url'].tolist()
    
    logger.info(f"Extracting features for {len(sample_urls)} URLs...")
    features_df = create_feature_pipeline(sample_urls, max_workers=5)
    
    # Save features
    features_df.to_csv(f"{data_dir}/sample_features.csv", index=False)
    
    logger.info("Feature extraction completed successfully")


def train_models(config: Dict[str, Any]):
    """Train deep learning models"""
    logger.info("Starting model training...")
    
    # Create training configuration
    training_config = {
        'model_dir': config['training']['model_dir'],
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'epochs': config['training']['epochs'],
        'patience': config['training']['patience'],
        'max_length': config['training']['max_length'],
        'urlnet': config['models']['urlnet'],
        'transformer': config['models']['transformer']
    }
    
    # Initialize trainer
    trainer = ModelTrainer(training_config)
    
    # Train models
    results = trainer.train_all_models(config['data']['processed_data_dir'])
    
    logger.info("Model training completed successfully")
    return results


def run_api(config: Dict[str, Any]):
    """Run the FastAPI server"""
    logger.info("Starting API server...")
    
    import uvicorn
    
    # Get API configuration
    api_config = config.get('api', {})
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    workers = api_config.get('workers', 1)
    
    # Run server
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=False
    )


def run_demo():
    """Run a simple demo of the system"""
    logger.info("Running demo...")
    
    # Sample URLs for testing
    test_urls = [
        "https://www.google.com",
        "https://www.github.com",
        "http://suspicious-phishing-site.tk/login.php?redirect=bank",
        "https://paypal-security-update.com/verify-account",
        "https://www.microsoft.com/support",
        "http://bit.ly/suspicious-link"
    ]
    
    print("\n" + "="*60)
    print("ADVANCED PHISHING DETECTION DEMO")
    print("="*60)
    
    try:
        # Try to load models and make predictions
        from preprocessing.feature_engineering import AdvancedFeatureExtractor
        
        extractor = AdvancedFeatureExtractor()
        
        for url in test_urls:
            print(f"\n🔍 Analyzing: {url}")
            
            try:
                features = extractor.extract_features(url, timeout=3)
                
                # Simple heuristic-based classification for demo
                risk_score = 0.0
                
                # Check suspicious indicators
                if features.url_length > 100:
                    risk_score += 0.2
                if features.num_dots > 5:
                    risk_score += 0.2
                if features.has_suspicious_words:
                    risk_score += 0.3
                if features.is_ip_address:
                    risk_score += 0.4
                if features.has_url_shortener:
                    risk_score += 0.3
                if not features.ssl_certificate_valid:
                    risk_score += 0.2
                
                risk_score = min(risk_score, 1.0)
                
                status = "🚨 PHISHING" if risk_score > 0.5 else "✅ LEGITIMATE"
                print(f"   Status: {status}")
                print(f"   Risk Score: {risk_score:.2f}")
                print(f"   URL Length: {features.url_length}")
                print(f"   Suspicious Words: {features.suspicious_word_count}")
                
            except Exception as e:
                print(f"   ❌ Error analyzing URL: {e}")
    
    except ImportError:
        print("⚠️  Models not available. Please run training first.")
        print("   Use: python main.py train")
    
    print("\n" + "="*60)
    print("Demo completed! For full functionality, train models first.")
    print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced Phishing Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py collect              # Collect data from sources
  python main.py extract              # Extract features from URLs
  python main.py train                # Train deep learning models
  python main.py api                  # Run API server
  python main.py demo                 # Run demo
  python main.py all                  # Run complete pipeline
        """
    )
    
    parser.add_argument(
        'command',
        choices=['collect', 'extract', 'train', 'api', 'demo', 'all'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    try:
        if args.command == 'collect':
            collect_data(config)
            
        elif args.command == 'extract':
            extract_features(config)
            
        elif args.command == 'train':
            train_models(config)
            
        elif args.command == 'api':
            run_api(config)
            
        elif args.command == 'demo':
            run_demo()
            
        elif args.command == 'all':
            logger.info("Running complete pipeline...")
            collect_data(config)
            extract_features(config)
            train_models(config)
            logger.info("Pipeline completed! You can now run 'python main.py api' to start the server.")
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()