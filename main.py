from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import ollama
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Load API keys and credits from environment
def load_api_keys():
    """Load API keys from environment variables"""
    keys = {}
    # Support multiple API keys: API_KEY_1, API_KEY_2, etc.
    for key, value in os.environ.items():
        if key.startswith("API_KEY") and not key.endswith("_CREDITS"):
            credit_key = f"{key}_CREDITS"
            credits = int(os.getenv(credit_key, 100))  # Default 100 credits
            keys[value] = credits
    return keys

API_KEY_CREDITS = load_api_keys()
logger.info(f"Loaded {len(API_KEY_CREDITS)} API key(s)")

app = FastAPI(
    title="LLM API Service",
    description="API for interacting with local LLM models via Ollama",
    version="1.0.0"
)

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = "mistral"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class GenerateResponse(BaseModel):
    response: str
    model: str
    credits_remaining: int

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: Optional[str] = "mistral"

class ChatResponse(BaseModel):
    message: Dict[str, str]
    model: str
    credits_remaining: int

# Dependency for API key verification
def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key and check credits"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    credits = API_KEY_CREDITS.get(x_api_key)
    
    if credits is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if credits <= 0:
        raise HTTPException(status_code=403, detail="No credits remaining")
    
    return x_api_key

# Routes
@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "LLM API",
        "version": "1.0.0"
    }

@app.get("/credits")
def check_credits(x_api_key: str = Depends(verify_api_key)):
    """Check remaining credits for your API key"""
    return {
        "credits_remaining": API_KEY_CREDITS[x_api_key]
    }

@app.get("/models")
def list_models(x_api_key: str = Depends(verify_api_key)):
    """List available Ollama models"""
    try:
        models = ollama.list()
        return {"models": [m["name"] for m in models.get("models", [])]}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models")

@app.post("/generate", response_model=GenerateResponse)
def generate(
    request: GenerateRequest,
    x_api_key: str = Depends(verify_api_key)
):
    """Generate text from LLM based on prompt"""
    try:
        # Deduct credit before processing
        API_KEY_CREDITS[x_api_key] -= 1
        
        # Prepare Ollama options
        options = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens:
            options["num_predict"] = request.max_tokens
        
        # Call Ollama
        response = ollama.chat(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
            options=options if options else None
        )
        
        logger.info(f"Generated response using model: {request.model}")
        
        return {
            "response": response["message"]["content"],
            "model": request.model,
            "credits_remaining": API_KEY_CREDITS[x_api_key]
        }
        
    except ollama.ResponseError as e:
        # Refund credit on error
        API_KEY_CREDITS[x_api_key] += 1
        logger.error(f"Ollama error: {e}")
        raise HTTPException(status_code=400, detail=f"Model error: {str(e)}")
    
    except Exception as e:
        # Refund credit on error
        API_KEY_CREDITS[x_api_key] += 1
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    x_api_key: str = Depends(verify_api_key)
):
    """Multi-turn chat endpoint"""
    try:
        API_KEY_CREDITS[x_api_key] -= 1
        
        response = ollama.chat(
            model=request.model,
            messages=request.messages
        )
        
        return {
            "message": response["message"],
            "model": request.model,
            "credits_remaining": API_KEY_CREDITS[x_api_key]
        }
        
    except Exception as e:
        API_KEY_CREDITS[x_api_key] += 1
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
