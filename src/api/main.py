"""
FastAPI Application for Phishing Detection
Real-time URL analysis with advanced deep learning models
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import torch
import pickle
import numpy as np
from typing import Dict, List, Optional, Any
import json
import time
from datetime import datetime
import logging
from pathlib import Path
import asyncio
import aiohttp
from contextlib import asynccontextmanager

# Import our models and utilities
import sys
sys.path.append('..')
from models.urlnet import URLNet, create_urlnet_model
from models.url_transformer import URLTransformer, URLTokenizer
from preprocessing.feature_engineering import AdvancedFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
models = {}
tokenizer = None
feature_extractor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    global models, tokenizer, feature_extractor
    
    logger.info("Loading models...")
    
    try:
        # Load configuration
        config_path = Path("../../models/training_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                'config': {
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
                'vocab_size': 128
            }
        
        # Load tokenizer
        tokenizer_path = Path("../../models/tokenizer.pkl")
        if tokenizer_path.exists():
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
        else:
            tokenizer = URLTokenizer(max_length=200)
        
        # Initialize feature extractor
        feature_extractor = AdvancedFeatureExtractor()
        
        # Load URLNet model
        try:
            urlnet_model = create_urlnet_model(config['config']['urlnet'])
            urlnet_path = Path("../../models/urlnet_best.pth")
            if urlnet_path.exists():
                urlnet_model.load_state_dict(torch.load(urlnet_path, map_location='cpu'))
            urlnet_model.eval()
            models['urlnet'] = urlnet_model
            logger.info("URLNet model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load URLNet model: {e}")
        
        # Load Transformer model
        try:
            transformer_model = URLTransformer(
                vocab_size=config['vocab_size'],
                **config['config']['transformer']
            )
            transformer_path = Path("../../models/transformer_best.pth")
            if transformer_path.exists():
                transformer_model.load_state_dict(torch.load(transformer_path, map_location='cpu'))
            transformer_model.eval()
            models['transformer'] = transformer_model
            logger.info("Transformer model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Transformer model: {e}")
        
        logger.info(f"Loaded {len(models)} models successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
    
    yield
    
    # Cleanup
    models.clear()


# Create FastAPI app
app = FastAPI(
    title="Advanced Phishing Detection API",
    description="Real-time phishing URL detection using deep learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class URLRequest(BaseModel):
    url: str
    include_features: bool = False
    include_explanations: bool = False


class URLBatchRequest(BaseModel):
    urls: List[str]
    include_features: bool = False


class PredictionResponse(BaseModel):
    url: str
    is_phishing: bool
    confidence: float
    risk_score: float
    prediction_time: float
    model_predictions: Dict[str, Dict[str, float]]
    features: Optional[Dict[str, Any]] = None
    explanations: Optional[Dict[str, str]] = None


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    timestamp: datetime


# Utility functions
def preprocess_url_for_urlnet(url: str, max_length: int = 200) -> torch.Tensor:
    """Preprocess URL for URLNet model"""
    char_indices = []
    for char in url[:max_length]:
        char_idx = ord(char) if ord(char) < 128 else 1  # UNK token
        char_indices.append(char_idx)
    
    # Pad or truncate
    if len(char_indices) < max_length:
        char_indices.extend([0] * (max_length - len(char_indices)))  # PAD token
    else:
        char_indices = char_indices[:max_length]
    
    return torch.tensor([char_indices], dtype=torch.long)


def get_risk_explanations(url: str, features: Dict[str, Any]) -> Dict[str, str]:
    """Generate risk explanations based on features"""
    explanations = {}
    
    # URL length
    if features.get('url_length', 0) > 100:
        explanations['long_url'] = "URL is unusually long, which is common in phishing attacks"
    
    # Suspicious characters
    if features.get('num_dots', 0) > 5:
        explanations['many_dots'] = "URL contains many dots, potentially indicating subdomain abuse"
    
    if features.get('num_hyphens', 0) > 3:
        explanations['many_hyphens'] = "URL contains many hyphens, often used to mimic legitimate domains"
    
    # IP address
    if features.get('is_ip_address', False):
        explanations['ip_address'] = "URL uses IP address instead of domain name"
    
    # Suspicious words
    if features.get('has_suspicious_words', False):
        explanations['suspicious_words'] = f"URL contains {features.get('suspicious_word_count', 0)} suspicious words"
    
    # URL shortener
    if features.get('has_url_shortener', False):
        explanations['url_shortener'] = "URL uses a URL shortening service"
    
    # Domain age
    domain_age = features.get('domain_age_days')
    if domain_age is not None and domain_age < 30:
        explanations['new_domain'] = f"Domain is only {domain_age} days old"
    
    # SSL certificate
    if not features.get('ssl_certificate_valid', True):
        explanations['no_ssl'] = "Domain does not have a valid SSL certificate"
    
    return explanations


async def predict_single_url(url: str, include_features: bool = False, include_explanations: bool = False) -> PredictionResponse:
    """Predict phishing probability for a single URL"""
    start_time = time.time()
    
    try:
        # Extract features if requested
        features = None
        if include_features or include_explanations:
            features = feature_extractor.extract_features(url, timeout=3).to_dict()
        
        model_predictions = {}
        all_probabilities = []
        
        # URLNet prediction
        if 'urlnet' in models:
            try:
                char_indices = preprocess_url_for_urlnet(url)
                lengths = torch.tensor([min(len(url), 200)], dtype=torch.long)
                
                with torch.no_grad():
                    outputs = models['urlnet'](char_indices, lengths)
                    probabilities = outputs['probabilities'][0].numpy()
                    
                model_predictions['urlnet'] = {
                    'legitimate_prob': float(probabilities[0]),
                    'phishing_prob': float(probabilities[1])
                }
                all_probabilities.append(probabilities[1])
                
            except Exception as e:
                logger.error(f"URLNet prediction error: {e}")
        
        # Transformer prediction
        if 'transformer' in models:
            try:
                tokenized = tokenizer.tokenize_url(url)
                input_ids = tokenized['input_ids'].unsqueeze(0)
                attention_mask = tokenized['attention_mask'].unsqueeze(0)
                token_type_ids = tokenized['token_type_ids'].unsqueeze(0)
                
                with torch.no_grad():
                    outputs = models['transformer'](input_ids, attention_mask, token_type_ids)
                    probabilities = outputs['probabilities'][0].numpy()
                    
                model_predictions['transformer'] = {
                    'legitimate_prob': float(probabilities[0]),
                    'phishing_prob': float(probabilities[1])
                }
                all_probabilities.append(probabilities[1])
                
            except Exception as e:
                logger.error(f"Transformer prediction error: {e}")
        
        # Ensemble prediction (average of all models)
        if all_probabilities:
            ensemble_prob = np.mean(all_probabilities)
            confidence = float(max(ensemble_prob, 1 - ensemble_prob))
            is_phishing = ensemble_prob > 0.5
            risk_score = float(ensemble_prob)
        else:
            # Fallback if no models available
            ensemble_prob = 0.5
            confidence = 0.5
            is_phishing = False
            risk_score = 0.5
        
        # Add ensemble prediction
        model_predictions['ensemble'] = {
            'legitimate_prob': float(1 - ensemble_prob),
            'phishing_prob': float(ensemble_prob)
        }
        
        # Generate explanations if requested
        explanations = None
        if include_explanations and features:
            explanations = get_risk_explanations(url, features)
        
        prediction_time = time.time() - start_time
        
        return PredictionResponse(
            url=url,
            is_phishing=is_phishing,
            confidence=confidence,
            risk_score=risk_score,
            prediction_time=prediction_time,
            model_predictions=model_predictions,
            features=features if include_features else None,
            explanations=explanations
        )
        
    except Exception as e:
        logger.error(f"Prediction error for URL {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Advanced Phishing Detection API",
        "version": "1.0.0",
        "status": "active"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=list(models.keys()),
        timestamp=datetime.now()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_url(request: URLRequest):
    """Predict phishing probability for a single URL"""
    return await predict_single_url(
        request.url,
        request.include_features,
        request.include_explanations
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_urls_batch(request: URLBatchRequest):
    """Predict phishing probability for multiple URLs"""
    start_time = time.time()
    
    if len(request.urls) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 URLs per batch request")
    
    # Process URLs concurrently
    tasks = [
        predict_single_url(url, request.include_features, False)
        for url in request.urls
    ]
    
    predictions = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and convert to proper responses
    valid_predictions = []
    for i, pred in enumerate(predictions):
        if isinstance(pred, Exception):
            logger.error(f"Error predicting URL {request.urls[i]}: {pred}")
            # Create error response
            valid_predictions.append(PredictionResponse(
                url=request.urls[i],
                is_phishing=False,
                confidence=0.0,
                risk_score=0.5,
                prediction_time=0.0,
                model_predictions={},
                features=None,
                explanations={"error": str(pred)}
            ))
        else:
            valid_predictions.append(pred)
    
    processing_time = time.time() - start_time
    
    return BatchPredictionResponse(
        predictions=valid_predictions,
        total_processed=len(valid_predictions),
        processing_time=processing_time
    )


@app.get("/models", response_model=Dict[str, Any])
async def get_model_info():
    """Get information about loaded models"""
    model_info = {}
    
    for model_name, model in models.items():
        model_info[model_name] = {
            "loaded": True,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    
    return {
        "models": model_info,
        "total_models": len(models),
        "tokenizer_vocab_size": tokenizer.vocab_size if tokenizer else 0
    }


@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_url_detailed(request: URLRequest):
    """Detailed URL analysis with all available information"""
    prediction = await predict_single_url(
        request.url,
        include_features=True,
        include_explanations=True
    )
    
    # Additional analysis
    analysis = {
        "prediction": prediction.dict(),
        "risk_assessment": {
            "level": "HIGH" if prediction.risk_score > 0.7 else "MEDIUM" if prediction.risk_score > 0.3 else "LOW",
            "recommendation": "BLOCK" if prediction.is_phishing else "ALLOW",
            "confidence_level": "HIGH" if prediction.confidence > 0.8 else "MEDIUM" if prediction.confidence > 0.6 else "LOW"
        },
        "model_consensus": {
            "agreement": len([p for p in prediction.model_predictions.values() 
                           if p['phishing_prob'] > 0.5]) == len(prediction.model_predictions),
            "variance": np.var([p['phishing_prob'] for p in prediction.model_predictions.values()]) if prediction.model_predictions else 0
        }
    }
    
    return analysis


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)