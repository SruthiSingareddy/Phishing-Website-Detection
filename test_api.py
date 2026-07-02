#!/usr/bin/env python3
"""
Test script for the Advanced Phishing Detection API
Demonstrates real-time URL analysis capabilities
"""

import sys
from pathlib import Path
import asyncio
import json
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from api.main import predict_single_url

async def test_api():
    """Test the API functionality"""
    print("🚀 Testing Advanced Phishing Detection API")
    print("=" * 60)
    
    # Test URLs
    test_cases = [
        {
            "url": "https://www.google.com",
            "expected": "legitimate",
            "description": "Popular legitimate site"
        },
        {
            "url": "https://github.com/microsoft/vscode",
            "expected": "legitimate", 
            "description": "GitHub repository"
        },
        {
            "url": "http://phishing-example.tk/login.php?redirect=bank",
            "expected": "phishing",
            "description": "Suspicious phishing pattern"
        },
        {
            "url": "https://paypal-security-update.com/verify-account",
            "expected": "phishing",
            "description": "Fake PayPal security update"
        },
        {
            "url": "https://www.amazon.com/products",
            "expected": "legitimate",
            "description": "Amazon products page"
        },
        {
            "url": "http://bit.ly/suspicious-link",
            "expected": "suspicious",
            "description": "URL shortener"
        }
    ]
    
    print(f"🔍 Testing {len(test_cases)} URLs with advanced algorithms...\n")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Testing: {test_case['url']}")
        print(f"   Description: {test_case['description']}")
        
        try:
            # Test with full analysis
            result = await predict_single_url(
                test_case['url'], 
                include_features=True, 
                include_explanations=True
            )
            
            # Display results
            status = "🚨 PHISHING" if result.is_phishing else "✅ LEGITIMATE"
            risk_level = get_risk_level(result.risk_score)
            
            print(f"   Status: {status}")
            print(f"   Risk Score: {result.risk_score:.3f}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Risk Level: {risk_level}")
            print(f"   Processing Time: {result.prediction_time:.3f}s")
            
            # Show model predictions
            if result.model_predictions:
                print(f"   Model Predictions:")
                for model_name, predictions in result.model_predictions.items():
                    phishing_prob = predictions.get('phishing_prob', 0)
                    print(f"     • {model_name}: {phishing_prob:.3f}")
            
            # Show explanations
            if result.explanations:
                print(f"   Risk Indicators: {', '.join(result.explanations.keys())}")
            
            results.append({
                'url': test_case['url'],
                'expected': test_case['expected'],
                'predicted': 'phishing' if result.is_phishing else 'legitimate',
                'risk_score': result.risk_score,
                'confidence': result.confidence,
                'processing_time': result.prediction_time
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'url': test_case['url'],
                'expected': test_case['expected'],
                'predicted': 'error',
                'error': str(e)
            })
        
        print()
    
    # Summary
    print("=" * 60)
    print("🎯 API Test Results Summary")
    print("=" * 60)
    
    correct_predictions = 0
    total_processing_time = 0
    
    for i, result in enumerate(results, 1):
        if 'error' not in result:
            expected = result['expected']
            predicted = result['predicted']
            
            # Simple accuracy check
            if (expected == 'legitimate' and predicted == 'legitimate') or \
               (expected == 'phishing' and predicted == 'phishing') or \
               (expected == 'suspicious' and result['risk_score'] > 0.3):
                correct_predictions += 1
                status = "✅"
            else:
                status = "❌"
            
            total_processing_time += result['processing_time']
            
            print(f"{status} Test {i}: {predicted.upper()} (Risk: {result['risk_score']:.3f}, "
                  f"Time: {result['processing_time']:.3f}s)")
        else:
            print(f"❌ Test {i}: ERROR - {result['error']}")
    
    accuracy = (correct_predictions / len([r for r in results if 'error' not in r])) * 100
    avg_processing_time = total_processing_time / len([r for r in results if 'error' not in r])
    
    print(f"\n📊 Performance Metrics:")
    print(f"   • Accuracy: {accuracy:.1f}%")
    print(f"   • Average Processing Time: {avg_processing_time:.3f}s")
    print(f"   • Total Tests: {len(test_cases)}")
    print(f"   • Successful Predictions: {len([r for r in results if 'error' not in r])}")
    
    print(f"\n🚀 Advanced Features Demonstrated:")
    print(f"   • Real-time URL analysis")
    print(f"   • Multi-model ensemble predictions")
    print(f"   • Advanced feature extraction")
    print(f"   • Risk scoring and explanations")
    print(f"   • FastAPI-ready deployment")
    
    return results


def get_risk_level(risk_score):
    """Convert risk score to human-readable level"""
    if risk_score >= 0.8:
        return "🔴 CRITICAL"
    elif risk_score >= 0.6:
        return "🟠 HIGH"
    elif risk_score >= 0.4:
        return "🟡 MEDIUM"
    elif risk_score >= 0.2:
        return "🟢 LOW"
    else:
        return "✅ MINIMAL"


async def main():
    """Main test function"""
    try:
        results = await test_api()
        
        print("\n" + "=" * 60)
        print("🎉 API Testing Complete!")
        print("=" * 60)
        print("\n🔧 Next Steps:")
        print("   1. Run 'python main.py api' to start the full API server")
        print("   2. Access http://localhost:8000/docs for interactive API documentation")
        print("   3. Use the /predict endpoint for real-time URL analysis")
        print("   4. Deploy to cloud for production use")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())