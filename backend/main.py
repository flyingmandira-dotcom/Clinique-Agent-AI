import os
import time
import logging
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import CheckRequest, CheckResponse, ClinicalReport, AgentStepResult
from backend.agents import llm, input_agent, retrieval_agent, severity_agent, generation_agent, hallucination_guard, output_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("drug_checker.main")

app = FastAPI(
    title="Multi-Agent Drug Interaction Checker",
    description="FastAPI clinical checker powered by a 6-agent cooperative pipeline.",
    version="1.0.0"
)

# CORS Policy configuration (fix CORS wildcard credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status")
async def get_status():
    """Returns whether live GenAI mode is configured and active, along with available providers."""
    ai_active = llm.is_ai_active()
    providers = llm.get_available_providers()
    return {
        "engine_mode": "live_ai" if ai_active else "simulation",
        "providers_available": providers,
        "gemini_api_key_configured": "gemini" in providers,
        "groq_api_key_configured": "groq" in providers,
        "grok_api_key_configured": "grok" in providers,
        "openrouter_api_key_configured": "openrouter" in providers,
        "primary_provider": os.getenv("PRIMARY_PROVIDER", "auto"),
        "app_mode_config": os.getenv("APP_MODE", "auto")
    }

@app.post("/api/check", response_model=CheckResponse)
async def check_interactions(request: CheckRequest):
    """Executes the complete 6-agent clinical checker pipeline."""
    # Fix: Truncate and sanitize query for logging to prevent sensitive PHI leakage
    query_preview = (request.query[:60] + "...") if len(request.query) > 60 else request.query
    logger.info(f"Received interaction check request (length: {len(request.query)}, query_preview='{query_preview}')")
    pipeline_steps: List[AgentStepResult] = []
    
    try:
        # ---- STEP 1: INPUT AGENT ----
        t0 = time.time()
        agent1_res = input_agent.run(
            query=request.query,
            pre_extracted_drugs=request.drugs,
            pre_extracted_conditions=request.conditions,
            target_illness=request.target_illness,
            medical_history=request.medical_history,
            allergies=request.allergies
        )
        agent1_res.logs.append(f"Agent 1 Execution Time: {round(time.time() - t0, 3)}s")
        pipeline_steps.append(agent1_res)
        
        extracted = agent1_res.output_data
        drugs = extracted.get("drugs", [])
        
        # Fix: Merge target_illness and medical_history into conditions list before downstream use
        conditions = list(extracted.get("conditions", []))
        target_illness = extracted.get("target_illness") or request.target_illness
        medical_history = extracted.get("medical_history") or request.medical_history
        allergies = extracted.get("allergies") or request.allergies

        if target_illness and target_illness.strip():
            t_clean = target_illness.strip().lower()
            if t_clean not in [c.lower() for c in conditions]:
                conditions.append(t_clean)
                
        if medical_history and medical_history.strip():
            m_clean = medical_history.strip().lower()
            if m_clean not in [c.lower() for c in conditions]:
                conditions.append(m_clean)
        
        if not drugs:
            # Fix: Return distinct "NO_DRUGS_DETECTED" status instead of false "SAFE"
            empty_report = ClinicalReport(
                severity="NO_DRUGS_DETECTED",
                summary="No active pharmaceutical agents or generic medications were detected in the query. Please input drug names to perform a check.",
                interactions_details=[],
                disease_warnings=[],
                metabolism_interactions=[],
                recommendations=[output_agent.STANDARD_DISCLAIMER],
                suggested_alternatives=[],
                citations=[],
                engine_mode="simulation" if not llm.is_ai_active() else "fallback_static"
            )
            parsed_algs = [a.strip() for a in (allergies or "").split(",") if a.strip() and a.strip().lower() not in ("none", "none reported")]
            return CheckResponse(
                drugs=[],
                conditions=conditions,
                allergies=parsed_algs,
                severity="NO_DRUGS_DETECTED",
                report=empty_report,
                pipeline_steps=pipeline_steps
            )
            
        # ---- STEP 2: RETRIEVAL AGENT ----
        t0 = time.time()
        agent2_res = retrieval_agent.run(
            drugs, 
            conditions,
            target_illness=target_illness,
            medical_history=medical_history,
            allergies=allergies
        )
        agent2_res.logs.append(f"Agent 2 Execution Time: {round(time.time() - t0, 3)}s")
        pipeline_steps.append(agent2_res)
        retrieval_data = agent2_res.output_data
        
        # ---- STEP 3: SEVERITY AGENT ----
        t0 = time.time()
        agent3_res = severity_agent.run(
            retrieval_data,
            drugs=drugs,
            conditions=conditions,
            allergies=allergies
        )
        agent3_res.logs.append(f"Agent 3 Execution Time: {round(time.time() - t0, 3)}s")
        pipeline_steps.append(agent3_res)
        severity_data = agent3_res.output_data
        
        # ---- STEP 4: GENERATION AGENT ----
        t0 = time.time()
        agent4_res = generation_agent.run(
            drugs, 
            conditions, 
            severity_data, 
            retrieval_data,
            target_illness=target_illness,
            medical_history=medical_history,
            allergies=allergies
        )
        agent4_res.logs.append(f"Agent 4 Execution Time: {round(time.time() - t0, 3)}s")
        pipeline_steps.append(agent4_res)
        generated_report = agent4_res.output_data
        
        # ---- STEP 5: HALLUCINATION GUARD ----
        t0 = time.time()
        agent5_res = hallucination_guard.run(generated_report, retrieval_data)
        agent5_res.logs.append(f"Agent 5 Execution Time: {round(time.time() - t0, 3)}s")
        pipeline_steps.append(agent5_res)
        guard_data = agent5_res.output_data
        
        # ---- STEP 6: OUTPUT AGENT ----
        t0 = time.time()
        agent6_res = output_agent.run(guard_data)
        agent6_res.logs.append(f"Agent 6 Execution Time: {round(time.time() - t0, 3)}s")
        pipeline_steps.append(agent6_res)
        
        final_report_data = agent6_res.output_data
        severity = final_report_data.get("severity", "SAFE")
        
        # Parse final report as Pydantic ClinicalReport
        clinical_report = ClinicalReport(**final_report_data)
        
        logger.info(f"Pipeline completed. Overall Severity: {severity}. Steps count: {len(pipeline_steps)}")
        
        parsed_algs = [a.strip() for a in (allergies or "").split(",") if a.strip() and a.strip().lower() not in ("none", "none reported")]
        
        return CheckResponse(
            drugs=drugs,
            conditions=conditions,
            allergies=parsed_algs,
            severity=severity,
            report=clinical_report,
            pipeline_steps=pipeline_steps
        )
        
    except Exception as e:
        logger.error(f"Error executing agent pipeline: {e}", exc_info=True)
        # Fix: Do not leak internal exception details to client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred in the multi-agent clinical pipeline. Please try again later."
        )

# Mount the static files directory at the root to serve the frontend dashboard
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    logger.info(f"[+] Static frontend files mounted from: {frontend_dir}")
else:
    logger.warning(f"[-] Frontend directory not found at: {frontend_dir}. API endpoints are active, but UI will not be served.")
