import pytest
from app.main import app

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to NearbyNLU API"}

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_query_endpoint(client):
    # Test with just a query
    response = client.post(
        "/query",
        json={"query": "Find me a good restaurant"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert "entities" in data
    assert "confidence" in data
    assert "api_call" in data
    assert "results" in data

    # Test with query and location
    response = client.post(
        "/query",
        json={
            "query": "Find me a good restaurant",
            "location": "San Francisco"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert "entities" in data
    assert "confidence" in data
    assert "api_call" in data
    assert "results" in data

def test_query_endpoint_invalid_input(client):
    # Test with missing query
    response = client.post(
        "/query",
        json={"location": "San Francisco"}
    )
    assert response.status_code == 422  # Validation error
