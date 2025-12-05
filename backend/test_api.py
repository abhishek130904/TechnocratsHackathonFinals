"""
Simple test script for the Counterfeit Detection API.

This script tests the API endpoints to ensure everything is working.

Run this after starting the backend server:
    python backend/test_api.py
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"


def test_health_endpoint():
    """Test the health check endpoint."""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            print("✓ Health check passed!")
            return True
        else:
            print(f"✗ Health check failed: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to backend. Is it running?")
        print("  Start it with: uvicorn app.main:app --reload")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_predict_endpoint():
    """Test the predict endpoint with dummy data."""
    print("\n" + "=" * 60)
    print("Testing Predict Endpoint")
    print("=" * 60)
    
    # Check if we have a test image
    test_image_path = Path("dataset/real/image.png")  # Adjust path if needed
    
    if test_image_path.exists():
        print(f"Using test image: {test_image_path}")
        with open(test_image_path, "rb") as f:
            image_data = f.read()
    else:
        print("No test image found. Using dummy data.")
        # Create a minimal dummy image (1x1 pixel PNG)
        image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'
    
    # Prepare request
    files = {
        'image': ('test_image.png', image_data, 'image/png')
    }
    
    data = {
        'title': 'Nike Air Max 90 Running Shoes',
        'description': 'Authentic Nike Air Max 90 with original box and tags. Brand new condition.',
        'seller_rating': 4.5,
        'num_reviews': 1250
    }
    
    try:
        print(f"Sending request to {BASE_URL}/api/predict...")
        response = requests.post(
            f"{BASE_URL}/api/predict",
            files=files,
            data=data,
            timeout=30  # 30 second timeout
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nResponse:")
            print(json.dumps(result, indent=2))
            print("\n✓ Predict endpoint works!")
            return True
        else:
            print(f"✗ Predict endpoint failed:")
            print(f"  Status: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to backend. Is it running?")
        return False
    except requests.exceptions.Timeout:
        print("✗ Request timed out. This might be normal if models are loading.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("API Test Suite")
    print("=" * 60)
    print("\nMake sure the backend server is running:")
    print("  cd backend")
    print("  uvicorn app.main:app --reload\n")
    
    # Run tests
    health_ok = test_health_endpoint()
    predict_ok = test_predict_endpoint()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Health Endpoint: {'✓ PASS' if health_ok else '✗ FAIL'}")
    print(f"Predict Endpoint: {'✓ PASS' if predict_ok else '✗ FAIL'}")
    
    if health_ok and predict_ok:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")


if __name__ == "__main__":
    main()

