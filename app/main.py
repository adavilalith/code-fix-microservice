# app/main.py

from fastapi import FastAPI, HTTPException
from app.models.schemas import CodeFixRequest, CodeFixResponse
from app.core.llm_manager import load_llm, generate_fix, get_llm_pipeline

# Initialize FastAPI application
app = FastAPI(title="AI Code Remediation Microservice (verrdeterra)") # Project name context

# --- Event Handlers (Startup/Shutdown) ---

@app.on_event("startup")
def startup_event():
    """Load the LLM when the application starts."""
    load_llm()

# --- API Endpoints ---

# Health check
@app.get("/")
def health_check():
    model_is_loaded = get_llm_pipeline() is not None 
    return {"status": "ok", "model_loaded": model_is_loaded}

# Mandatory /local_fix endpoint
@app.post("/local_fix", response_model=CodeFixResponse)
async def local_fix(request: CodeFixRequest):
    """
    Analyzes vulnerable code and generates a secure fix using the local LLM.
    """
    LLM_PIPELINE = get_llm_pipeline()
    if LLM_PIPELINE is None:
        raise HTTPException(
            status_code=503, 
            detail="Service Unavailable: LLM model is not loaded."
        )
        
    # Pass the request to the business logic layer
    return generate_fix(request)