import os
import json
import logging
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, Type, Tuple, List
from pydantic import BaseModel
from dotenv import load_dotenv

# Setup logger
logger = logging.getLogger("drug_checker.llm")

# Load latest environment variables
load_dotenv(override=True)

# Provider model constants
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

# Flag to track configured Gemini API
_is_gemini_active = False

try:
    import google.generativeai as genai
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and gemini_key != "YOUR_GEMINI_API_KEY_HERE":
        genai.configure(api_key=gemini_key)
        _is_gemini_active = True
        logger.info("[+] Gemini API successfully configured.")
    else:
        logger.info("[-] Gemini API Key not set.")
except ImportError:
    logger.warning("[-] google-generativeai package not installed.")

def get_available_providers() -> List[str]:
    """Returns list of active LLM providers based on environment configuration."""
    load_dotenv(override=True)
    providers = []
    
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and gemini_key != "YOUR_GEMINI_API_KEY_HERE":
        providers.append("gemini")
        
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key and groq_key != "YOUR_GROQ_API_KEY_HERE":
        providers.append("groq")
        
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if openrouter_key and openrouter_key != "YOUR_OPENROUTER_API_KEY_HERE":
        providers.append("openrouter")
        
    return providers

def is_ai_active() -> bool:
    """Check if any live GenAI provider integration is available and enabled."""
    app_mode = os.getenv("APP_MODE", "auto").lower()
    if app_mode == "simulation":
        return False
    return len(get_available_providers()) > 0

def _call_gemini(prompt: str, system_instruction: Optional[str] = None, response_schema: Optional[Type[BaseModel]] = None) -> str:
    """Invokes Google Gemini model."""
    import google.generativeai as genai
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError("GEMINI_API_KEY not configured")
        
    genai.configure(api_key=gemini_key)
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

def _call_openai_compatible_endpoint(
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    system_instruction: Optional[str] = None,
    response_schema: Optional[Type[BaseModel]] = None,
    timeout: float = 12.0
) -> str:
    """Generic helper for OpenAI-compatible REST endpoints (Groq, OpenRouter)."""
    messages = []
    sys_content = system_instruction or ""
    
    if response_schema:
        schema_json = json.dumps(response_schema.model_json_schema() if hasattr(response_schema, "model_json_schema") else response_schema.schema())
        schema_instruction = f"\nCRITICAL: Respond ONLY with a valid JSON object strictly conforming to this schema:\n{schema_json}"
        sys_content = f"{sys_content}\n{schema_instruction}".strip()
        
    if sys_content:
        messages.append({"role": "system", "content": sys_content})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    if response_schema:
        payload["response_format"] = {"type": "json_object"}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "CliniqueAgent/1.0",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "CliniqueAgent"
        },
        method="POST"
    )
    
    with urllib.request.urlopen(req, timeout=timeout) as response:
        resp_data = json.loads(response.read().decode("utf-8"))
        content = resp_data["choices"][0]["message"]["content"].strip()
        return content

def call_llm(
    prompt: str, 
    system_instruction: Optional[str] = None, 
    response_schema: Optional[Type[BaseModel]] = None
) -> Tuple[str, str]:
    """
    Executes a multi-provider LLM failover chain:
    1. Gemini (primary)
    2. Groq (llama-3.3-70b-versatile)
    3. OpenRouter (free-tier model fallback)
    
    Returns a tuple of (response_text, successful_provider_name).
    Raises RuntimeError if all providers fail or are inactive.
    """
    if not is_ai_active():
        raise RuntimeError("LLM is inactive in Simulation Mode")
        
    errors = []
    
    # 1. Try Gemini
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and gemini_key != "YOUR_GEMINI_API_KEY_HERE":
        try:
            logger.info("[LLM Chain] Attempting primary provider: Gemini...")
            text = _call_gemini(prompt, system_instruction, response_schema)
            return text, "gemini"
        except Exception as e:
            logger.warning(f"[LLM Chain] Gemini provider failed ({e}). Attempting failover to Groq...")
            errors.append(f"Gemini: {e}")
            
    # 2. Try Groq
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key and groq_key != "YOUR_GROQ_API_KEY_HERE":
        try:
            logger.info("[LLM Chain] Attempting secondary provider: Groq (llama-3.3-70b)...")
            text = _call_openai_compatible_endpoint(
                GROQ_API_URL, 
                groq_key, 
                GROQ_MODEL, 
                prompt, 
                system_instruction, 
                response_schema,
                timeout=12.0
            )
            return text, "groq"
        except Exception as e:
            logger.warning(f"[LLM Chain] Groq provider failed ({e}). Attempting failover to OpenRouter...")
            errors.append(f"Groq: {e}")

    # 3. Try OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if openrouter_key and openrouter_key != "YOUR_OPENROUTER_API_KEY_HERE":
        try:
            logger.info("[LLM Chain] Attempting tertiary provider: OpenRouter...")
            text = _call_openai_compatible_endpoint(
                OPENROUTER_API_URL, 
                openrouter_key, 
                OPENROUTER_MODEL, 
                prompt, 
                system_instruction, 
                response_schema,
                timeout=12.0
            )
            return text, "openrouter"
        except Exception as e:
            logger.warning(f"[LLM Chain] OpenRouter provider failed ({e}). All failover providers exhausted.")
            errors.append(f"OpenRouter: {e}")

    # If all configured providers failed
    err_summary = "; ".join(errors) if errors else "No active provider API keys configured."
    logger.error(f"[-] All LLM providers failed: {err_summary}")
    raise RuntimeError(f"All LLM providers failed: {err_summary}")
