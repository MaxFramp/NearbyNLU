import tensorflow as tf
import json
import os
import numpy as np
from typing import Dict
import pickle
from sentence_transformers import SentenceTransformer

try:
    from app.config import MODEL_PATH
except ImportError:
    # Fallback for direct script execution
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from app.config import MODEL_PATH

INTENT_CONFIDENCE_THRESHOLD = 0.70

class NLUModel:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.label_encoder = None
        self.load_model()

    @staticmethod
    def extract_entities(text: str) -> Dict[str, str]:
        # Simple example using regex. You can customize this logic.
        entities = {}
        if "near" in text or "nearby" in text:
            entities["location_hint"] = "nearby"
        if any(word in text for word in ["open", "now", "today"]):
            entities["time_hint"] = "open_now"
        return entities

    @staticmethod
    def apply_confidence_fallback(intent, confidence):
        if confidence < INTENT_CONFIDENCE_THRESHOLD:
            if "restaurant" in intent:
                return "restaurant"
            if "park" in intent:
                return "park"       
            if "store" in intent:
                return "store"
        return intent

    def load_model(self):
        """Load the TensorFlow model and sentence transformer encoder."""
        try:
            # Load the transformer encoder
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load the classifier model
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully")
            
            # Load the label encoder
            with open("models/transformer_label_encoder.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict(self, text: str):
        """
        Make a prediction using the loaded model.
        
        Args:
            text (str): Input text to process
            
        Returns:
            dict: Prediction results including intent and entities
        """
        if not self.model or not self.encoder:
            raise ValueError("Model or encoder not loaded")
        
        # Encode the input text using the transformer
        text_embedding = self.encoder.encode([text])
        
        # Predict intent
        pred_probs = self.model.predict(text_embedding)
        intent_index = np.argmax(pred_probs[0])
        confidence = float(pred_probs[0][intent_index])
        intent = self.label_encoder.inverse_transform([intent_index])[0]
        intent = self.apply_confidence_fallback(intent, confidence)

        # Extract entities
        entities = self.extract_entities(text)
            
        return {
            "intent": intent,
            "entities": entities,
            "confidence": confidence
        }

if __name__ == "__main__":
    model = NLUModel()
    
    # Test cases for different restaurant types
    print("\n=== Restaurant Queries ===")
    print(model.predict("I want mexican food near me"))
    print(model.predict("Show me nearby italian restaurants that are open now"))
    print(model.predict("Find me the closest sushi place"))
    print(model.predict("Looking a vegan-friendly place to eat in my neighborhood"))
    print(model.predict("What's the best thai restaurant in this area?"))
    
    # Test cases for different business types
    print("\n=== Business Queries ===")
    print(model.predict("Where can I find a pharmacy?"))
    print(model.predict("Is there a grocery store open now?"))
    print(model.predict("Find me the nearest gas station"))
    print(model.predict("Looking for a hardware store nearby"))
    print(model.predict("What's the best coffee shop around here?"))
    
    # Test cases for entertainment and leisure
    print("\n=== Entertainment & Leisure Queries ===")
    print(model.predict("Show me nearby parks"))
    print(model.predict("Find me the closest movie theater"))
    print(model.predict("Looking for a bowling alley that's open now"))
    print(model.predict("What's the best museum in this area?"))
    print(model.predict("Are there any amusement parks nearby?"))
    
    # Test cases for services
    print("\n=== Service Queries ===")
    print(model.predict("I need a doctor near me"))
    print(model.predict("Find me the closest dentist"))
    print(model.predict("Looking for a hair salon that's open now"))
    print(model.predict("Where can I find a car repair shop?"))
    print(model.predict("I need a mechanic near me"))
    print(model.predict("What's the best gym in this area?"))
    
    # Test cases with time and location hints
    print("\n=== Queries with Time/Location Hints ===")
    print(model.predict("Show me restaurants that are open now"))
    print(model.predict("Find me a coffee shop near me"))
    print(model.predict("Looking for a park that's open today"))
    print(model.predict("What's the best restaurant nearby?"))
    print(model.predict("Find me the closest gas station that's open now"))
    
    # Test cases for transportation
    print("\n=== Transportation Queries ===")
    print(model.predict("Where's the nearest train station?"))
    print(model.predict("Find me the closest bus stop"))
    print(model.predict("Looking for a subway station nearby"))
    print(model.predict("What's the best airport in this area?"))
    print(model.predict("Find me the nearest taxi stand"))

    # Test cases for client profile
    print("\n=== Client Profile Queries ===")
    print(model.predict("Loves vegan food."))
    print(model.predict("Client is a family with young children."))
    print(model.predict("Has a thing for antiques and old bookstores"))
    print(model.predict("Client is a fitness enthusiast."))
    print(model.predict("Client is a movie buff."))
    print(model.predict("Client owns a dog."))
    