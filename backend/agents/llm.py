import os
import json
import logging
from typing import Optional, Dict, Any, Type
from pydantic import BaseModel
from dotenv import load_dotenv

# Setup logger
logger = logging.getLogger("drug_checker.llm")

# Load latest environment variables (crucial for hot-reloads)
load_dotenv(override=True)

# Flag to verify if Gemini API is available and configured
_is_gemini_active = False

try:
    import google.generativeai as genai
    
    # Retrieve key
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key and api_key != "YOUR_GEMINI_API_KEY_HERE":
        genai.configure(api_key=api_key)
        _is_gemini_active = True
        logger.info("[+] Gemini API successfully configured.")
    else:
        logger.info("[-] Gemini API Key not set. Using Simulation Mode.")
except ImportError:
    logger.warning("[-] google-generativeai not installed. Falling back to Simulation Mode.")

def is_ai_active() -> bool:
    """Check if the live GenAI integration is active."""
    # Support overriding via APP_MODE
    app_mode = os.getenv("APP_MODE", "auto").lower()
    if app_mode == "simulation":
        return False
    return _is_gemini_active

def call_llm(prompt: str, system_instruction: Optional[str] = None, response_schema: Optional[Type[BaseModel]] = None) -> str:
    """Helper to query Gemini if active, raising an error otherwise to trigger simulation fallback."""
    if not is_ai_active():
        raise RuntimeError("LLM is inactive in Simulation Mode")
        
    try:
        import google.generativeai as genai
        # We use gemini-1.5-flash as it is extremely fast and robust
        generation_config = {}
        if response_schema:
            generation_config["response_mime_type"] = "application/json"
            generation_config["response_schema"] = response_schema
            
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            system_instruction=system_instruction
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config if response_schema else None
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"[-] Gemini API Call failed: {e}. Falling back to simulation.")
        raise e
