import pytest
from app.services.google_maps import GoogleMapsService

@pytest.fixture
def maps_service():
    return GoogleMapsService()

def test_search_nearby(maps_service):
    # Test with a known location
    results = maps_service.search_nearby("Times Square, New York", radius=1000)
    assert isinstance(results, list)
    
    # If we got results, check their structure
    if results:
        first_result = results[0]
        assert 'name' in first_result
        assert 'place_id' in first_result
        assert 'geometry' in first_result
        assert 'location' in first_result['geometry']

def test_get_place_details(maps_service):
    # Test with a known place ID (Times Square)
    place_id = "ChIJmQJIxlVYwokRdw8sonh1xYc"
    details = maps_service.get_place_details(place_id)
    
    if details:
        assert 'result' in details
        result = details['result']
        assert 'name' in result
        assert 'formatted_address' in result
        assert 'geometry' in result 