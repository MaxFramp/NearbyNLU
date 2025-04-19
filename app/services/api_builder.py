from typing import Dict, Optional
from .google_maps import GoogleMapsService
from ..models.nlu_model import NLUModel

class APIBuilder:
    def __init__(self):
        self.nlu_model = NLUModel()
        self.google_maps = GoogleMapsService()

    def build_api_call(self, user_input: str) -> Dict:
        """
        Build an API call based on the NLU model's predictions.
        
        Args:
            user_input (str): User's natural language input
            
        Returns:
            Dict: API call parameters and results
        """
        # Get prediction from NLU model
        prediction = self.nlu_model.predict(user_input)
        
        # Initialize response structure
        response = {
            "intent": prediction["intent"],
            "entities": prediction["entities"],
            "confidence": prediction["confidence"],
            "api_call": None,
            "results": None
        }
        
        # Build API call based on intent and entities
        if prediction["intent"] == "restaurant":
            # Default parameters
            params = {
                "location": None,
                "radius": 5000,  # 5km default radius
                "type": "restaurant"
            }
            
            # Update parameters based on entities
            if "location_hint" in prediction["entities"]:
                params["location"] = prediction["entities"]["location_hint"]
            
            # Make the API call
            if params["location"]:
                response["api_call"] = params
                response["results"] = self.google_maps.search_nearby(
                    location=params["location"],
                    radius=params["radius"],
                    type=params["type"]
                )
        
        return response 