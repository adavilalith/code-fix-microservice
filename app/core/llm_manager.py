import time
import json
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from app.models.schemas import CodeFixRequest, CodeFixResponse
from app.utils.diff_generator import generate_unified_diff
from typing import Optional, Any, Dict

# Configuration
# Using your specified model: deepseek-ai/deepseek-coder-1.3b-base
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"
LLM_PIPELINE: Optional[Any] = None

# --- Helper Function for Prompt Formatting ---

def format_prompt(system_prompt: str, user_prompt: str) -> str:
    """
    Formats the system and user prompts into the required chat template
    for instruction-tuned models like DeepSeek Coder.
    """
    # DeepSeek Coder uses the 'Inst' format:
    # <|begin_of_text|><|system|>system_prompt<|end_of_system|><|user|>user_prompt<|end_of_user|><|assistant|>
    
    formatted_string = (
        f"<|begin_of_text|><|system|>{system_prompt}<|end_of_system|>"
        f"<|user|>{user_prompt}<|end_of_user|><|assistant|>"
    )
    return formatted_string

# --- Model Loading Function ---


def load_llm():
    """Load the Hugging Face model and tokenizer using 4-bit quantization."""
    global LLM_PIPELINE
    if LLM_PIPELINE:
        print("Model already loaded.")
        return

    print(f"Loading LLM: {MODEL_NAME} in 4-bit mode...")
    try:
        # 💥 CHANGE: 8-bit quantization configuration 💥
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_use_double_quant=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            quantization_config=bnb_config,
            device_map="cpu" # Helps manage device allocation and quantization
        )

        # Create the generation pipeline
        LLM_PIPELINE = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
           
        )
        print("LLM loaded successfully in 4-bit mode.")
    except Exception as e:
        print(f"Error loading model: {e}")
        LLM_PIPELINE = None


def get_llm_pipeline() -> Optional[Any]:
    """Function to retrieve the current, globally-updated LLM_PIPELINE."""
    global LLM_PIPELINE
    return LLM_PIPELINE

# --- Core Inference Function ---

def generate_fix(request: CodeFixRequest) -> CodeFixResponse:
    """
    Orchestrates the LLM call and processes the response into the required format.
    """
    pipeline = get_llm_pipeline()

    if pipeline is None:
        return CodeFixResponse(
            secure_code="Error: LLM not loaded.",
            diff="",
            explanation="The LLM failed to load during startup.",
            latency_ms=0.0,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0}
        )

    # 1. Prompt Engineering
    system_prompt = (
        "You are an expert secure coding assistant. Your task is to analyze the user's "
        "vulnerable code snippet and provide a secure, working version. "
        "Your response **MUST** be a **single, valid, raw JSON object** only. "
        "Do not output any surrounding text, markdown fences (```json, ```), or comments outside the JSON object. "
        "The JSON object must contain exactly the following two keys:\n"
        "1. **secure_code** (string): The complete, fixed Python code.\n"
        "2. **explanation** (string): A detailed explanation of the vulnerability and the fix.\n\n"
        "**EXAMPLE OUTPUT:**\n"
        "{\n"
        "  \"secure_code\": \"import subprocess\\n\\ndef process_file(path):\\n    subprocess.run(['cat', path], check=True)\",\n"
        "  \"explanation\": \"Fixed command injection by replacing os.system with subprocess.run and passing arguments as a list.\"\n"
        "}"
    )
    
    context_part = ""
    if request.contextual_guidelines:
        context_part = f"\n\nContextual Security Guidelines: {request.contextual_guidelines}"
        
    user_prompt = (
        f"Vulnerable Code:\n```python\n{request.vulnerable_code}\n```"
        f"\n\nTask: Provide the secure version and a detailed explanation in the requested JSON format. {context_part}"
    )
    
    # Format the prompt for the model
    formatted_prompt = format_prompt(system_prompt, user_prompt)

    # 2. LLM Inference and Latency Tracking
    start_time = time.time()
    
    try:
        # 2a. Call the pipeline
        response = pipeline(
            formatted_prompt,
            max_new_tokens=512, # Limit the output size
            do_sample=False,    # Use deterministic generation
            return_full_text=False # Only return the generated part
        )
        
        # Extract the raw text output from the model
        raw_output = response[0]['generated_text'].strip()
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # 2b. Parse the JSON output
        # Attempt to clean up and parse the JSON. Models sometimes add markdown fences (```json)
        if raw_output.startswith("```json"):
            raw_output = raw_output.strip().replace("```json", "").replace("```", "")
        
        llm_response_dict: Dict[str, str] = json.loads(raw_output)

        secure_code = llm_response_dict.get("secure_code", "ERROR: Secure code missing from LLM response.")
        explanation = llm_response_dict.get("explanation", "ERROR: Explanation missing from LLM response.")

    except json.JSONDecodeError as e:
        # Handle cases where the LLM output is not valid JSON
        latency_ms = (time.time() - start_time) * 1000
        print(f"JSON Decode Error: {e} - Raw output: {raw_output}")
        return CodeFixResponse(
            secure_code=f"LLM output error. Could not parse JSON. Raw response: {raw_output[:   ]}...",
            diff="",
            explanation=f"The model generated non-JSON output. Please adjust the prompt or model generation parameters. Error: {e}",
            latency_ms=latency_ms,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0}
        )
    except Exception as e:
        # Catch other potential errors during inference
        latency_ms = (time.time() - start_time) * 1000
        print(f"Inference Error: {e}")
        return CodeFixResponse(
            secure_code="Inference failed.",
            diff="",
            explanation=f"An unexpected error occurred during model inference: {e}",
            latency_ms=latency_ms,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0}
        )

    # 3. Post-processing: Generate the diff
    generated_diff = generate_unified_diff(
        original_code=request.vulnerable_code,
        secure_code=secure_code
    )

    # 4. Token Calculation (Approximation for standard HF pipelines)
    # Note: Accurate token calculation often requires direct access to pipeline internals,
    # but we can approximate based on the prompt size.
    prompt_tokens = len(pipeline.tokenizer.encode(formatted_prompt))
    completion_tokens = len(pipeline.tokenizer.encode(raw_output)) if 'raw_output' in locals() else 0

    return CodeFixResponse(
        secure_code=secure_code,
        diff=generated_diff,
        explanation=explanation,
        latency_ms=latency_ms,
        token_usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    )