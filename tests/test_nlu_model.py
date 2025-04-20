import pytest
import numpy as np
import warnings
from app.models.nlu_model import NLUModel

# Filter out the specific TensorFlow deprecation warning
# warnings.filterwarnings("ignore", category=DeprecationWarning, 
#                        message=".*ml_dtypes.float8_e4m3b11 is deprecated.*")

@pytest.fixture
def nlu_model():
    return NLUModel()

def test_extract_entities():
    # Test with location hint
    entities = NLUModel.extract_entities("Find restaurants near me")
    assert "location_hint" in entities
    assert entities["location_hint"] == "nearby"

    # Test with time hint
    entities = NLUModel.extract_entities("Show me restaurants open now")
    assert "time_hint" in entities
    assert entities["time_hint"] == "open_now"

    # Test with both hints
    entities = NLUModel.extract_entities("Find restaurants near me that are open now")
    assert "location_hint" in entities
    assert "time_hint" in entities
    assert entities["location_hint"] == "nearby"
    assert entities["time_hint"] == "open_now"

    # Test with no hints
    entities = NLUModel.extract_entities("Just a regular sentence")
    assert not entities

def test_apply_confidence_fallback():
    # Test with high confidence
    intent = NLUModel.apply_confidence_fallback("restaurant", 0.8)
    assert intent == "restaurant"

    # Test with low confidence for restaurant
    intent = NLUModel.apply_confidence_fallback("restaurant", 0.2)
    assert intent == "restaurant"  # Should fall back to restaurant

    # Test with low confidence for non-restaurant
    intent = NLUModel.apply_confidence_fallback("hotel", 0.2)
    assert intent == "hotel"  # Should not change

def test_predict_structure(nlu_model):
    # Test the structure of the prediction output
    result = nlu_model.predict("Find me a restaurant")
    
    assert isinstance(result, dict)
    assert "intent" in result
    assert "entities" in result
    assert "confidence" in result
    assert isinstance(result["intent"], str)
    assert isinstance(result["entities"], dict)
    assert isinstance(result["confidence"], float)

def test_predict_with_location(nlu_model):
    # Test prediction with location context
    result = nlu_model.predict("Find me a restaurant near San Francisco")
    
    assert "location_hint" in result["entities"]
    assert result["entities"]["location_hint"] == "nearby"

def test_predict_with_time(nlu_model):
    # Test prediction with time context
    result = nlu_model.predict("Show me restaurants open now")
    
    assert "time_hint" in result["entities"]
    assert result["entities"]["time_hint"] == "open_now"

def test_predict_empty_input(nlu_model):
    # Test prediction with empty input
    result = nlu_model.predict("")
    
    assert isinstance(result, dict)
    assert "intent" in result
    assert "entities" in result
    assert "confidence" in result 