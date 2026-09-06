import os
import json
import logging
import random
import socket
import time
import hashlib
import threading
import urllib.request
import urllib.error
from collections import deque
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

XAI_API_URL = "https://api.x.ai/v1/chat/completions"
XAI_MODEL = "grok-2-latest"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

# Gemini model family rotation (separate quota buckets per model on free tier)
GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash-8b"
]

# In-memory response cache to eliminate redundant quota consumption
_RESPONSE_CACHE: Dict[str, Tuple[str, str]] = {}
_CACHE_MAX_SIZE = 120

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


# ---------------------------------------------------------------------------
# Global LLM call rate limiter (sliding window)
# Applies to every real outbound HTTP call to any provider/model, regardless
# of which agent or retry/failover path triggered it.
# ---------------------------------------------------------------------------
_LLM_CALL_LOCK = threading.Lock()
_RECENT_CALL_TIMESTAMPS: deque = deque()
MAX_CALLS_PER_WINDOW = 2
WINDOW_SECONDS = 60.0


def _throttle_global_rate_limit() -> None:
    """Blocks the calling thread until fewer than MAX_CALLS_PER_WINDOW outbound
    LLM calls have occurred in the trailing WINDOW_SECONDS. Call this as the
    first line of any function that actually sends a network request to an
    LLM provider."""
    while True:
        with _LLM_CALL_LOCK:
            now = time.monotonic()
            while _RECENT_CALL_TIMESTAMPS and now - _RECENT_CALL_TIMESTAMPS[0] >= WINDOW_SECONDS:
                _RECENT_CALL_TIMESTAMPS.popleft()

            if len(_RECENT_CALL_TIMESTAMPS) < MAX_CALLS_PER_WINDOW:
                _RECENT_CALL_TIMESTAMPS.append(now)
                return

            wait = WINDOW_SECONDS - (now - _RECENT_CALL_TIMESTAMPS[0])

        logger.info(f"[Rate Limiter] {MAX_CALLS_PER_WINDOW}/min cap reached. Waiting {wait:.1f}s...")
        time.sleep(wait)


def _get_cache_key(prompt: str, system_instruction: Optional[str] = None) -> str:
    """Computes a unique SHA-256 hash for query and instruction."""
    raw = f"{prompt}:::{system_instruction or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_available_providers() -> List[str]:
    """Returns list of active LLM providers based on environment configuration."""
    load_dotenv(override=True)
    providers = []
    
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key and groq_key != "YOUR_GROQ_API_KEY_HERE":
        providers.append("groq")
        
    xai_key = os.getenv("XAI_API_KEY", "") or os.getenv("GROK_API_KEY", "")
    if xai_key and xai_key not in ("YOUR_XAI_API_KEY_HERE", "YOUR_GROK_API_KEY_HERE"):
        providers.append("grok")

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and gemini_key != "YOUR_GEMINI_API_KEY_HERE":
        providers.append("gemini")
        
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


def get_backoff_time(attempt: int, base: float = 1.0, jitter: bool = True) -> float:
    """
    Calculate exponential backoff time with optional randomized jitter:
    wait = base * (2 ^ attempt) + jitter
    """
    wait = base * (2 ** attempt)
    if jitter:
        wait += random.uniform(0.1, max(0.4, wait * 0.25))
    return round(wait, 2)


def get_retry_after(headers: Any) -> Optional[float]:
    """Extract and parse Retry-After header value in seconds if available."""
    if not headers:
        return None
    val = None
    if hasattr(headers, "get"):
        val = headers.get("Retry-After")
    elif isinstance(headers, dict):
        val = headers.get("Retry-After") or headers.get("retry-after")
        
    if val:
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    return None


def is_rate_limited(error_code: int, headers: Any = None) -> bool:
    """Check if HTTP response indicates rate limiting or capacity throttling."""
    if error_code == 429:
        return True
    if error_code in (503, 504) and headers and get_retry_after(headers) is not None:
        return True
    return False


def _clean_and_validate_json(
    text: str, 
    schema: Optional[Type[BaseModel]] = None
) -> str:
    """
    Cleans markdown code blocks, isolates JSON content, parses it,
    and validates strictly against the provided Pydantic schema.
    """
    cleaned = text.strip()
    
    # Strip markdown code blocks if wrapped in ```json ... ``` or ``` ... ```
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            if lines[-1].startswith("```"):
                cleaned = "\n".join(lines[1:-1]).strip()
            else:
                cleaned = "\n".join(lines[1:]).strip()
    elif "```json" in cleaned:
        parts = cleaned.split("```json")
        if len(parts) > 1:
            cleaned = parts[1].split("```")[0].strip()
    elif "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) > 1:
            cleaned = parts[1].split("```")[0].strip()

    # Isolate JSON object or array bounds
    start_brace = cleaned.find("{")
    start_bracket = cleaned.find("[")
    if start_brace != -1 or start_bracket != -1:
        start_idx = min(idx for idx in (start_brace, start_bracket) if idx != -1)
        end_brace = cleaned.rfind("}")
        end_bracket = cleaned.rfind("]")
        end_idx = max(end_brace, end_bracket)
        if end_idx > start_idx:
            cleaned = cleaned[start_idx:end_idx + 1]

    # Parse JSON
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from LLM: {e} -> Raw text: '{cleaned[:200]}'")

    # Validate against Pydantic schema
    if schema:
        try:
            if hasattr(schema, "model_validate"):
                schema.model_validate(parsed)
            elif hasattr(schema, "parse_obj"):
                schema.parse_obj(parsed)
        except Exception as e:
            # If schema validation fails due to minor field structure (e.g. list of strings vs list of dicts),
            # log warning and adapt rather than discarding valid AI output and crashing into fallback
            logger.warning(f"Schema strict validation notice for {schema.__name__}: {e}. Retaining extracted JSON.")

    return json.dumps(parsed)


def _call_gemini_raw(
    prompt: str, 
    system_instruction: Optional[str] = None, 
    response_schema: Optional[Type[BaseModel]] = None,
    model_name: str = "gemini-1.5-flash"
) -> str:
    """
    Calls Google Gemini using JSON output mode and schema instruction injection.
    Avoids Google API 400 schema conversion errors on arbitrary Dict[str, Any] fields.
    """
    _throttle_global_rate_limit()
    import google.generativeai as genai
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError("GEMINI_API_KEY not configured")
        
    genai.configure(api_key=gemini_key)
    generation_config: Dict[str, Any] = {}
    full_instruction = system_instruction or ""
    
    if response_schema:
        # Request pure JSON output
        generation_config["response_mime_type"] = "application/json"
        try:
            schema_dict = response_schema.model_json_schema() if hasattr(response_schema, "model_json_schema") else response_schema.schema()
            full_instruction = f"{full_instruction}\nCRITICAL: Respond ONLY with a valid JSON object strictly adhering to this schema:\n{json.dumps(schema_dict)}".strip()
        except Exception:
            pass
        
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=full_instruction if full_instruction else None
    )
    
    response = model.generate_content(
        prompt,
        generation_config=generation_config if generation_config else None
    )
    
    # Safe text extraction handling various finish reasons
    text = ""
    if hasattr(response, "text"):
        try:
            text = response.text
        except Exception:
            pass
            
    if not text and hasattr(response, "candidates") and response.candidates:
        parts = getattr(response.candidates[0].content, "parts", [])
        text = "".join(getattr(p, "text", "") for p in parts)
        
    if not text.strip():
        raise ValueError(f"Gemini ({model_name}) returned empty text content.")
        
    return text.strip()


def _call_gemini_with_retry(
    prompt: str,
    system_instruction: Optional[str] = None,
    response_schema: Optional[Type[BaseModel]] = None,
    max_retries_per_model: int = 2
) -> Tuple[str, str]:
    """
    Executes Gemini call with intelligent multi-model rotation:
    1. gemini-1.5-flash
    2. gemini-2.0-flash
    3. gemini-1.5-flash-8b
    
    If one model hits a 429 / Quota / ResourceExhausted, it immediately fails over
    to the next model in the family to leverage independent quota buckets.
    """
    last_err = None
    for model_name in GEMINI_MODELS:
        for attempt in range(max_retries_per_model):
            try:
                logger.info(f"[Gemini Chain] Calling {model_name} (attempt {attempt + 1}/{max_retries_per_model})...")
                res = _call_gemini_raw(prompt, system_instruction, response_schema, model_name=model_name)
                return res, f"gemini:{model_name}"
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "quota" in err_str or "resourceexhausted" in err_str
                
                if is_rate_limit:
                    logger.warning(
                        f"[Gemini Quota] {model_name} rate limit / quota reached: {e}. "
                        f"Rotating to next model in Gemini family..."
                    )
                    break  # Immediately rotate to next Gemini model!
                elif attempt < max_retries_per_model - 1:
                    wait_time = get_backoff_time(attempt, base=1.0)
                    logger.warning(f"[Gemini Retry] {model_name} transient error ({e}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"[Gemini] {model_name} exhausted attempts: {e}")
                    
    raise last_err if last_err else RuntimeError("All Gemini models failed")


def _call_openai_compatible_endpoint(
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    system_instruction: Optional[str] = None,
    response_schema: Optional[Type[BaseModel]] = None,
    timeout: float = 20.0
) -> str:
    """Generic helper for OpenAI-compatible REST endpoints (Groq, OpenRouter)."""
    _throttle_global_rate_limit()
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


def _call_with_retry(
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    system_instruction: Optional[str] = None,
    response_schema: Optional[Type[BaseModel]] = None,
    max_retries: int = 3,
    provider_name: str = "endpoint"
) -> str:
    """Calls OpenAI-compatible endpoint with exponential backoff + jitter + dynamic timeout escalation."""
    for attempt in range(max_retries):
        timeout = 20.0 + (attempt * 10.0)  # 20s, 30s, 40s
        try:
            return _call_openai_compatible_endpoint(
                url=url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                system_instruction=system_instruction,
                response_schema=response_schema,
                timeout=timeout
            )
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                pass
            logger.error(f"[{provider_name.upper()} HTTP Error] Code {e.code}: {e.reason} -> {err_body}")
            if is_rate_limited(e.code, e.headers):
                retry_after = get_retry_after(e.headers) or get_backoff_time(attempt, base=2.0)
                if attempt < max_retries - 1:
                    logger.warning(
                        f"[{provider_name.upper()} Retry] Rate limited (HTTP {e.code}). "
                        f"Waiting {retry_after}s before retry {attempt + 2}/{max_retries}..."
                    )
                    time.sleep(retry_after)
                else:
                    logger.error(f"[{provider_name.upper()}] Rate limited and exhausted retries: HTTP {e.code}")
                    raise
            elif attempt < max_retries - 1:
                wait_time = get_backoff_time(attempt)
                logger.warning(
                    f"[{provider_name.upper()} Retry] HTTP {e.code} error on attempt {attempt + 1}/{max_retries}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            if attempt < max_retries - 1:
                wait_time = get_backoff_time(attempt)
                logger.warning(
                    f"[{provider_name.upper()} Retry] Network/timeout ({e}) on attempt {attempt + 1} (timeout={timeout}s). "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"[{provider_name.upper()}] Network/timeout exhausted ({e}) after {max_retries} attempts.")
                raise


def call_llm(
    prompt: str, 
    system_instruction: Optional[str] = None, 
    response_schema: Optional[Type[BaseModel]] = None,
    fallback_to_simulation: bool = True
) -> Tuple[str, str]:
    """
    Executes multi-provider LLM failover chain with caching and rate-limit mitigation:
    0. Fast In-Memory Cache (0ms latency, zero quota used)
    1. Groq (llama-3.3-70b-versatile -> llama-3.1-8b-instant) / xAI Grok (grok-2)
    2. Gemini Family (gemini-1.5-flash -> gemini-2.0-flash -> gemini-1.5-flash-8b)
    3. OpenRouter (free tier fallback)
    4. Fallback to Simulation Mode (if enabled)
    
    Returns (response_text, provider_name).
    """
    # Always reload environment to pick up freshly added keys on-the-fly
    load_dotenv(override=True)
    
    if not is_ai_active():
        if fallback_to_simulation:
            logger.info("[LLM Chain] LLM inactive (simulation mode). Returning simulation fallback.")
            return "", "simulation"
        raise RuntimeError("LLM is inactive in Simulation Mode and fallback is disabled.")
        
    # Check in-memory response cache
    cache_key = _get_cache_key(prompt, system_instruction)
    if cache_key in _RESPONSE_CACHE:
        cached_text, cached_provider = _RESPONSE_CACHE[cache_key]
        logger.info(f"[LLM Cache] Cache hit! Returning cached response ({cached_provider}).")
        return cached_text, cached_provider

    errors = []
    primary_pref = os.getenv("PRIMARY_PROVIDER", "auto").lower()
    
    # Define provider execution functions
    def try_groq():
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if not groq_key or groq_key == "YOUR_GROQ_API_KEY_HERE":
            return None
        last_g_err = None
        for g_model in [GROQ_MODEL, "llama-3.1-8b-instant", "llama3-70b-8192"]:
            try:
                logger.info(f"[LLM Chain] Attempting Groq ({g_model})...")
                raw = _call_with_retry(
                    url=GROQ_API_URL, 
                    api_key=groq_key, 
                    model=g_model, 
                    prompt=prompt, 
                    system_instruction=system_instruction, 
                    response_schema=response_schema,
                    max_retries=2,
                    provider_name=f"groq:{g_model}"
                )
                clean = _clean_and_validate_json(raw, response_schema) if response_schema else raw
                return clean, f"groq:{g_model}"
            except Exception as e:
                last_g_err = e
                logger.warning(f"[Groq Failover] Model {g_model} failed ({e}). Escalating to next Groq model...")
        raise last_g_err or RuntimeError("All Groq models failed")

    def try_grok():
        xai_key = os.getenv("XAI_API_KEY", "") or os.getenv("GROK_API_KEY", "")
        if not xai_key or xai_key in ("YOUR_XAI_API_KEY_HERE", "YOUR_GROK_API_KEY_HERE"):
            return None
        logger.info("[LLM Chain] Attempting xAI Grok (grok-2-latest)...")
        raw = _call_with_retry(
            url=XAI_API_URL, 
            api_key=xai_key, 
            model=XAI_MODEL, 
            prompt=prompt, 
            system_instruction=system_instruction, 
            response_schema=response_schema,
            max_retries=3,
            provider_name="grok"
        )
        clean = _clean_and_validate_json(raw, response_schema) if response_schema else raw
        return clean, "grok"

    def try_gemini():
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY_HERE":
            return None
        logger.info("[LLM Chain] Attempting Gemini multi-model family...")
        raw, provider_tag = _call_gemini_with_retry(prompt, system_instruction, response_schema)
        clean = _clean_and_validate_json(raw, response_schema) if response_schema else raw
        return clean, provider_tag

    def try_openrouter():
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        if not or_key or or_key == "YOUR_OPENROUTER_API_KEY_HERE":
            return None
        logger.info("[LLM Chain] Attempting OpenRouter...")
        raw = _call_with_retry(
            url=OPENROUTER_API_URL, 
            api_key=or_key, 
            model=OPENROUTER_MODEL, 
            prompt=prompt, 
            system_instruction=system_instruction, 
            response_schema=response_schema,
            max_retries=3,
            provider_name="openrouter"
        )
        clean = _clean_and_validate_json(raw, response_schema) if response_schema else raw
        return clean, "openrouter"

    # Determine provider priority order
    if primary_pref in ("groq", "grok", "xai"):
        provider_order = [("groq", try_groq), ("grok", try_grok), ("gemini", try_gemini), ("openrouter", try_openrouter)]
    elif primary_pref == "gemini":
        provider_order = [("gemini", try_gemini), ("groq", try_groq), ("grok", try_grok), ("openrouter", try_openrouter)]
    else:
        # Default auto order: if Groq or Grok key is present, prioritize them; else Gemini
        has_groq = bool(os.getenv("GROQ_API_KEY") and os.getenv("GROQ_API_KEY") != "YOUR_GROQ_API_KEY_HERE")
        has_grok = bool((os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")) and (os.getenv("XAI_API_KEY") != "YOUR_XAI_API_KEY_HERE"))
        if has_groq or has_grok:
            provider_order = [("groq", try_groq), ("grok", try_grok), ("gemini", try_gemini), ("openrouter", try_openrouter)]
        else:
            provider_order = [("gemini", try_gemini), ("groq", try_groq), ("grok", try_grok), ("openrouter", try_openrouter)]

    # Execute provider chain
    for p_name, p_fn in provider_order:
        try:
            result = p_fn()
            if result is not None:
                clean_text, provider_tag = result
                logger.info(f"[LLM Chain] [+] {provider_tag} successfully returned valid clinical response.")
                
                # Save to cache
                if len(_RESPONSE_CACHE) >= _CACHE_MAX_SIZE:
                    _RESPONSE_CACHE.pop(next(iter(_RESPONSE_CACHE)))
                _RESPONSE_CACHE[cache_key] = (clean_text, provider_tag)
                
                return clean_text, provider_tag
        except Exception as e:
            err_msg = f"{p_name}: {e}"
            logger.warning(f"[LLM Chain] Provider {p_name} failed: {e}. Moving to next provider...")
            errors.append(err_msg)

    # All providers exhausted
    err_summary = " | ".join(errors) if errors else "No active provider API keys configured."
    logger.error(f"[-] All LLM providers failed: {err_summary}")
    
    if fallback_to_simulation:
        logger.warning("[LLM Chain] Gracefully degrading to offline simulation mode.")
        return "", "simulation_fallback"
        
    raise RuntimeError(f"All LLM providers failed: {err_summary}")