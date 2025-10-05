#!/usr/bin/env python3
"""
Test script to verify server endpoints are available
"""

import requests
import sys

def test_endpoints():
    base_url = "http://localhost:8080"
    
    print("Testing server endpoints...")
    print("=" * 50)
    
    # Test root
    try:
        response = requests.get(f"{base_url}/", timeout=2)
        print(f"✓ Root endpoint: {response.status_code}")
    except Exception as e:
        print(f"✗ Root endpoint failed: {e}")
    
    # Test health
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        print(f"✓ Health endpoint: {response.status_code}")
    except Exception as e:
        print(f"✗ Health endpoint failed: {e}")
    
    # Test profile endpoints
    try:
        response = requests.get(f"{base_url}/api/profile/health", timeout=2)
        print(f"✓ Profile health endpoint: {response.status_code}")
    except Exception as e:
        print(f"✗ Profile health endpoint failed: {e}")
    
    try:
        response = requests.get(f"{base_url}/api/profile/profiles", timeout=2)
        print(f"✓ List profiles endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Available profiles: {list(data.get('profiles', {}).keys())}")
    except Exception as e:
        print(f"✗ List profiles endpoint failed: {e}")
    
    # Test API docs
    try:
        response = requests.get(f"{base_url}/docs", timeout=2)
        print(f"✓ API docs: {response.status_code}")
    except Exception as e:
        print(f"✗ API docs failed: {e}")

if __name__ == "__main__":
    test_endpoints()