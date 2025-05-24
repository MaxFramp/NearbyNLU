from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import Dict, Optional

# Import our services
from app.services.api_builder import APIBuilder

# Load environment variables
load_dotenv()

app = FastAPI(
    title="NearbyNLU",
    description="Natural Language Understanding for Location-Based Queries",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our services
api_builder = APIBuilder()

class QueryRequest(BaseModel):
    query: str
    location: Optional[str] = None

class QueryResponse(BaseModel):
    intent: str
    entities: Dict
    confidence: float
    api_call: Optional[Dict] = None
    results: Optional[list] = None

@app.get("/")
async def root():
    return {"message": "Welcome to NearbyNLU API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a natural language query and return location-based results.
    
    Args:
        request: QueryRequest containing the user's query and optional location
        
    Returns:
        QueryResponse containing the processed results
    """
    try:
        # Build the full query including location if provided
        full_query = request.query
        if request.location:
            full_query = f"{request.query} near {request.location}"
            
        # Process the query using our API builder
        response = api_builder.build_api_call(full_query)

        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 