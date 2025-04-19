import tensorflow as tf
import json
import os
import numpy as np
from typing import Dict
import pickle
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
        self.tokenizer_config = None
        self.load_model()
        # self.load_tokenizer_config()

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
        """Load the TensorFlow model from the models directory."""
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully")
            print(self.model)
            print(self.model.signatures)
            with open("models/label_encoder.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    # def load_tokenizer_config(self):
    #     """Load tokenizer configuration."""
    #     try:
    #         with open(TOKENIZER_CONFIG_PATH, 'r') as f:
    #             self.tokenizer_config = json.load(f)
    #     except Exception as e:
    #         print(f"Error loading tokenizer config: {e}")
    #         raise


    def predict(self, text: str):
        """
        Make a prediction using the loaded model.
        
        Args:
            text (str): Input text to process
            
        Returns:
            dict: Prediction results including intent and entities
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Predict intent
        pred_probs = self.model.predict([text])
        intent_index = np.argmax(pred_probs[0])
        confidence = float(pred_probs[0][intent_index])
        intent = self.label_encoder.inverse_transform([intent_index])[0]
        intent = self.apply_confidence_fallback(intent, confidence)

        # Extract entities
        entities = self.extract_entities(text)
            
        # TODO: Implement actual prediction logic
        # This is a placeholder that will be implemented based on the specific model
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