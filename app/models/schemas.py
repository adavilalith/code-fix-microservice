# app/models/schemas.py

from pydantic import BaseModel, Field

from typing import Optional, Dict

class CodeFixRequest(BaseModel):
    vulnerable_code: str = Field(..., example="def my_function(user_input):\n  os.system(user_input)")
    contextual_guidelines: Optional[str] = Field(None, example="Injection attacks should be prevented.")

class CodeFixResponse(BaseModel):
    secure_code: str = Field(..., example="def my_function(user_input):\n  # Secure code...")
    diff: str = Field(..., example="--- vulnerable.py\n+++ secure.py\n@@ ...")
    explanation: str = Field(..., example="Used a safe method to prevent injection.")
    latency_ms: float = Field(..., description="Latency of the LLM generation in milliseconds.")
    token_usage: Dict[str, int] = Field(..., example={"prompt_tokens": 100, "completion_tokens": 50})