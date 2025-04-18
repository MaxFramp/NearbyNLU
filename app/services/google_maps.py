import googlemaps
from ..config import GOOGLE_MAPS_API_KEY

class GoogleMapsService:
    def __init__(self):
        if not GOOGLE_MAPS_API_KEY:
            raise ValueError("Google Maps API key not found in environment variables")
        self.client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

    def search_nearby(self, location: str, radius: int = 5000, type: str = None):
        """
        Search for places near a given location.
        
        Args:
            location (str): Location to search around
            radius (int): Search radius in meters
            type (str): Type of place to search for
            
        Returns:
            list: List of places found
        """
        try:
            # First, geocode the location to get coordinates
            geocode_result = self.client.geocode(location)
            if not geocode_result:
                return []
                
            location_coords = geocode_result[0]['geometry']['location']
            
            # Search for nearby places
            places_result = self.client.places_nearby(
                location=location_coords,
                radius=radius,
                type=type
            )
            
            return places_result.get('results', [])
            
        except Exception as e:
            print(f"Error searching nearby places: {e}")
            return []

    def get_place_details(self, place_id: str):
        """
        Get detailed information about a specific place.
        
        Args:
            place_id (str): Google Places ID
            
        Returns:
            dict: Place details
        """
        try:
            return self.client.place(place_id)
        except Exception as e:
            print(f"Error getting place details: {e}")
            return None 