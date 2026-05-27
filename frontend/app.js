/* ==========================================================================
   CLINIQUEAGENT AI — PIPELINE COORDINATOR LOGIC
   ========================================================================== */

// Configure the backend API URL. If running locally, default to relative paths.
// Otherwise, point to the deployed backend URL on Render.
const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? ""
    : "https://clinique-agent-ai.onrender.com";

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const queryInput = document.getElementById("query-input");
    const checkForm = document.getElementById("check-form");
    const submitBtn = document.getElementById("submit-btn");
    const btnText = submitBtn.querySelector(".btn-text");
    const spinner = submitBtn.querySelector(".spinner");
    const engineStatusBadge = document.getElementById("engine-status-badge");
    const engineStatusText = document.getElementById("engine-status-text");
    const consoleStream = document.getElementById("console-stream");
    const clearConsoleBtn = document.getElementById("clear-console");
    
    // Diagnostic Report Elements
    const reportSection = document.getElementById("report-section");
    const reportDrugs = document.getElementById("report-drugs");
    const reportConditions = document.getElementById("report-conditions");
    const reportSeverity = document.getElementById("report-severity");
    const reportSummaryText = document.getElementById("report-summary-text");
    const drugInteractionsCard = document.getElementById("drug-interactions-card");
    const drugInteractionsContainer = document.getElementById("drug-interactions-container");
    const diseaseWarningsCard = document.getElementById("disease-warnings-card");
    const diseaseWarningsContainer = document.getElementById("disease-warnings-container");
    const metabolismCard = document.getElementById("metabolism-card");
    const metabolismContainer = document.getElementById("metabolism-container");
    const recommendationsList = document.getElementById("recommendations-list");
    const citationsList = document.getElementById("citations-list");
    const printReportBtn = document.getElementById("print-report-btn");

    // Suggested Alternatives Elements
    const alternativesCard = document.getElementById("alternatives-card");
    const alternativesContainer = document.getElementById("alternatives-container");

    // History Elements
    const historyCard = document.getElementById("history-card");
    const historyList = document.getElementById("history-list");
    const clearHistoryBtn = document.getElementById("clear-history-btn");

    // Quick tag triggers
    const tagBtns = document.querySelectorAll(".tag-btn");

    // Modal & Profile DOM Elements
    const profileCard = document.getElementById("profile-card");
    const editProfileBtn = document.getElementById("edit-profile-btn");
    const profileIllness = document.getElementById("profile-illness");
    const profileHistory = document.getElementById("profile-history");
    const profileAllergies = document.getElementById("profile-allergies");
    
    const healthContextModal = document.getElementById("health-context-modal");
    const closeModalBtn = document.getElementById("close-modal-btn");
    const healthContextForm = document.getElementById("health-context-form");
    const illnessInput = document.getElementById("illness-input");
    const historyInput = document.getElementById("history-input");
    const allergiesInput = document.getElementById("allergies-input");

    // App state
    let isChecking = false;
    let diagnosisHistory = [];
    let pendingDiagnosticQuery = null;
    let patientProfile = {
        illness: "",
        history: "",
        allergies: ""
    };

    // Initialize System Status, History & Profile
    checkSystemStatus();
    loadHistory();
    loadPatientProfile();

    // Event Listeners
    checkForm.addEventListener("submit", handleDiagnosticSubmit);
    clearConsoleBtn.addEventListener("click", resetDashboard);
    printReportBtn.addEventListener("click", () => window.print());

    // Edit Profile triggers modal
    editProfileBtn.addEventListener("click", () => {
        illnessInput.value = patientProfile.illness || "";
        historyInput.value = patientProfile.history || "";
        allergiesInput.value = patientProfile.allergies || "";
        healthContextModal.classList.remove("hidden");
    });

    // Close Modal
    closeModalBtn.addEventListener("click", () => {
        healthContextModal.classList.add("hidden");
        pendingDiagnosticQuery = null;
    });

    // Submit Health Context Form
    healthContextForm.addEventListener("submit", (e) => {
        e.preventDefault();
        
        const newIllness = illnessInput.value.trim();
        const newHistory = historyInput.value.trim();
        const newAllergies = allergiesInput.value.trim();
        
        savePatientProfile(newIllness, newHistory, newAllergies);
        healthContextModal.classList.add("hidden");
        
        addConsoleLine("SYS", "Saved Patient Health Profile context successfully.", "success-line");
        
        if (pendingDiagnosticQuery) {
            const queryToRun = pendingDiagnosticQuery;
            pendingDiagnosticQuery = null;
            addConsoleLine("SYS", `Re-triggering diagnostic check for "${queryToRun}" with active profile...`, "system-line");
            queryInput.value = queryToRun;
            handleDiagnosticSubmit();
        }
    });

    /**
     * Loads patient profile from localStorage
     */
    function loadPatientProfile() {
        try {
            const saved = localStorage.getItem("patient_medical_profile");
            if (saved) {
                patientProfile = JSON.parse(saved);
            }
        } catch (err) {
            console.error("Error loading patient profile:", err);
        }
        renderProfileUI();
    }

    /**
     * Saves patient profile to localStorage
     */
    function savePatientProfile(illness, history, allergies) {
        patientProfile = { illness, history, allergies };
        localStorage.setItem("patient_medical_profile", JSON.stringify(patientProfile));
        renderProfileUI();
    }

    /**
     * Renders active patient profile in sidebar
     */
    function renderProfileUI() {
        profileIllness.textContent = patientProfile.illness || "Not specified";
        profileHistory.textContent = patientProfile.history || "None reported";
        profileAllergies.textContent = patientProfile.allergies || "None reported";
    }

    clearHistoryBtn.addEventListener("click", () => {
        diagnosisHistory = [];
        localStorage.removeItem("drug_checker_history");
        renderHistoryUI();
    });

    // Connect Clinical Scenario Tags (including the ones in the new Questions You Might Have section)
    document.addEventListener("click", (e) => {
        const btn = e.target.closest(".tag-btn");
        if (btn && !btn.closest("#history-list")) {
            e.preventDefault();
            if (isChecking) return;
            const query = btn.getAttribute("data-query");
            if (query) {
                queryInput.value = query;
                handleDiagnosticSubmit();
            }
        }
    });

    // Connect Diagnosis History click events
    historyList.addEventListener("click", (e) => {
        const itemBtn = e.target.closest(".history-item-btn");
        if (itemBtn) {
            e.preventDefault();
            if (isChecking) return;
            const query = itemBtn.getAttribute("data-query");
            if (query) {
                queryInput.value = query;
                handleDiagnosticSubmit();
            }
        }
    });

    /**
     * Fetches current engine state (Simulation vs Live GenAI)
     */
    async function checkSystemStatus() {
        try {
            const res = await fetch(`${API_BASE}/api/status`);
            const data = await res.json();
            
            // Format status badge
            engineStatusBadge.className = `engine-badge ${data.engine_mode}`;
            if (data.engine_mode === "live_ai") {
                engineStatusText.textContent = "Live Gemini AI Active";
                addConsoleLine("SYS", "Connected to Google Gemini Engine. Live clinical-grade generation active.", "success-line");
            } else {
                engineStatusText.textContent = "Offline Simulation Mode";
                addConsoleLine("SYS", "Running in offline Simulation mode. Preloaded clinical local databases are active.", "warning-line");
            }
        } catch (err) {
            console.error("Error fetching system status:", err);
            engineStatusBadge.className = "engine-badge simulation";
            engineStatusText.textContent = "Offline Core Engine";
            addConsoleLine("SYS", "Could not query status endpoint. Running on local simulation engine.", "warning-line");
        }
    }



    /**
     * Appends a log line to the scrolling Agent Console
     */
    function addConsoleLine(tag, message, className = "") {
        const line = document.createElement("div");
        line.className = `console-line ${className}`;
        
        // Generate current timestamp [HH:MM:SS]
        const now = new Date();
        const timeStr = now.toTimeString().split(" ")[0];
        
        line.innerHTML = `
            <span class="console-time">[${timeStr}]</span>
            <span class="console-tag ${tag.toLowerCase().replace(/\s+/g, '-')}">${tag}</span>
            <span class="console-msg">${escapeHtml(message)}</span>
        `;
        
        consoleStream.appendChild(line);
        consoleStream.scrollTop = consoleStream.scrollHeight;
    }

    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }

    /**
     * Handles checker submission
     */
    async function handleDiagnosticSubmit(e) {
        if (e) e.preventDefault();
        if (isChecking) return;

        const query = queryInput.value.trim();
        if (!query) return;

        isChecking = true;
        setFormLoadingState(true);
        reportSection.classList.add("hidden");
        
        addConsoleLine("SYS", `Diagnostic query submitted: "${query}"`, "system-line");
        if (patientProfile.illness) {
            addConsoleLine("SYS", `Using active health profile (Treating: ${patientProfile.illness}, History: ${patientProfile.history || 'None'}, Allergies: ${patientProfile.allergies || 'None'})`, "system-line");
        }
        addConsoleLine("SYS", "Initializing cooperative Multi-Agent pipeline orchestration...", "system-line");

        try {
            // Trigger backend check with active patient medical profile details
            const payload = {
                query,
                target_illness: patientProfile.illness || null,
                medical_history: patientProfile.history || null,
                allergies: patientProfile.allergies || null
            };
            
            const response = await fetch(`${API_BASE}/api/check`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`API returned HTTP ${response.status}`);
            }

            const data = await response.json();
            
            // Intercept if severity is high-risk (WARNING or CRITICAL) and we haven't asked for target illness / allergies
            if ((data.severity === "CRITICAL" || data.severity === "WARNING") && !patientProfile.illness) {
                addConsoleLine("SYS", "High clinical risk flagged! safety screening required to tailor alternative recommendations.", "warning-line");
                pendingDiagnosticQuery = query;
                
                // Prefill history with any conditions extracted by Agent 1 if available
                illnessInput.value = "";
                historyInput.value = data.conditions ? data.conditions.join(", ") : "";
                allergiesInput.value = "";
                
                // Open Safety screening Modal
                healthContextModal.classList.remove("hidden");
                setFormLoadingState(false);
                isChecking = false;
                return;
            }
            
            // Save query to history
            saveHistoryEntry(query);
            
            // Execute animated rolling pipeline steps
            await animatePipelineExecution(data);
            
        } catch (err) {
            console.error("Diagnostic error:", err);
            addConsoleLine("SYS", `Orchestration Failure: ${err.message}`, "error-line");
            alert("A critical error occurred while executing the multi-agent diagnostic check. Check backend server console.");
            setFormLoadingState(false);
            isChecking = false;
        }
    }

    /**
     * Highlights agents one by one in sync with console streaming
     */
    async function animatePipelineExecution(data) {
        const steps = data.pipeline_steps;

        for (let i = 0; i < steps.length; i++) {
            const step = steps[i];

            // 1. Announce agent launch in Console
            addConsoleLine(`Agent ${i + 1}`, `Launching: ${step.agent_name} - ${step.description}`, "system-line");
            
            // 2. Stream agent logs with staggered delay to look natural
            for (const log of step.logs) {
                let cssClass = "";
                if (log.includes("[FDA MATCH]") || log.includes("[DISEASE MATCH]") || log.includes("[METABOLISM MATCH]")) {
                    cssClass = "warning-line";
                } else if (log.includes("[Success]") || log.includes("succeeded")) {
                    cssClass = "success-line";
                } else if (log.includes("[WARNING]") || log.includes("Caution")) {
                    cssClass = "warning-line";
                } else if (log.includes("[Error]") || log.includes("failed")) {
                    cssClass = "error-line";
                }
                
                addConsoleLine(step.agent_name, log, cssClass);
                await delay(120); // slight millisecond stagger per log
            }
            
            // Delay before moving to the next agent (simulates active deliberation)
            await delay(1000);
        }

        addConsoleLine("SYS", "Pipeline fully executed. Compiling advisory note into structured view.", "success-line");
        
        // Render report
        renderClinicalReport(data);
        
        setFormLoadingState(false);
        isChecking = false;
    }

    /**
     * Renders Pydantic report details into HTML UI components
     */
    function renderClinicalReport(data) {
        const report = data.report;
        
        // 1. Setup metadata
        reportDrugs.textContent = data.drugs.length > 0 ? data.drugs.join(", ") : "None";
        reportConditions.textContent = data.conditions.length > 0 ? data.conditions.join(", ") : "None";
        
        // 1b. Setup patient clinical profile context display associated with this report
        const reportPatientContextBar = document.getElementById("report-patient-context-bar");
        const reportPatientIllness = document.getElementById("report-patient-illness");
        const reportPatientHistory = document.getElementById("report-patient-history");
        const reportPatientAllergies = document.getElementById("report-patient-allergies");

        if (patientProfile.illness || patientProfile.history || patientProfile.allergies) {
            reportPatientContextBar.classList.remove("hidden");
            reportPatientIllness.innerHTML = `<strong>Treating:</strong> ${patientProfile.illness || 'Not specified'}`;
            reportPatientHistory.innerHTML = `<strong>History:</strong> ${patientProfile.history || 'None reported'}`;
            reportPatientAllergies.innerHTML = `<strong>Allergies:</strong> ${patientProfile.allergies || 'None reported'}`;
        } else {
            reportPatientContextBar.classList.add("hidden");
        }
        
        // 2. Setup severity badge
        const sev = data.severity.toLowerCase();
        reportSeverity.className = `severity-badge-large ${sev}`;
        reportSeverity.textContent = data.severity;
        
        // 3. Setup summary
        reportSummaryText.textContent = report.summary;
        
        // 4. Render Drug-Drug interactions
        drugInteractionsContainer.innerHTML = "";
        if (report.interactions_details.length > 0) {
            drugInteractionsCard.classList.remove("hidden");
            report.interactions_details.forEach(item => {
                const row = document.createElement("div");
                const itemSev = item.severity.toLowerCase();
                row.className = `interaction-item item-${itemSev}`;
                row.innerHTML = `
                    <div class="item-header-row">
                        <span class="item-title">${item.drugs}</span>
                        <span class="item-badge ${itemSev}">${item.severity}</span>
                    </div>
                    <div class="item-row">
                        <span class="item-row-label">Pharmacological Mechanism:</span>
                        <span class="item-row-val">${item.mechanism}</span>
                    </div>
                    <div class="item-row">
                        <span class="item-row-label">Expected Adverse Effects:</span>
                        <span class="item-row-val">${item.clinical_effects}</span>
                    </div>
                    <div class="item-row">
                        <span class="item-row-label">Clinical Action Plan:</span>
                        <span class="item-row-val">${item.management}</span>
                    </div>
                `;
                drugInteractionsContainer.appendChild(row);
            });
        } else {
            drugInteractionsCard.classList.add("hidden");
        }

        // 5. Render Disease contraindications
        diseaseWarningsContainer.innerHTML = "";
        if (report.disease_warnings.length > 0) {
            diseaseWarningsCard.classList.remove("hidden");
            report.disease_warnings.forEach(item => {
                const row = document.createElement("div");
                const itemSev = item.severity.toLowerCase();
                row.className = `interaction-item item-${itemSev}`;
                row.innerHTML = `
                    <div class="item-header-row">
                        <span class="item-title">${item.drug} in patients with ${item.condition}</span>
                        <span class="item-badge ${itemSev}">${item.severity}</span>
                    </div>
                    <div class="item-row">
                        <span class="item-row-label">Contraindication Pathway:</span>
                        <span class="item-row-val">${item.mechanism}</span>
                    </div>
                    <div class="item-row">
                        <span class="item-row-label">Expected Adverse Effects:</span>
                        <span class="item-row-val">${item.clinical_effects}</span>
                    </div>
                    <div class="item-row">
                        <span class="item-row-label">Clinical Action Plan:</span>
                        <span class="item-row-val">${item.management}</span>
                    </div>
                `;
                diseaseWarningsContainer.appendChild(row);
            });
        } else {
            diseaseWarningsCard.classList.add("hidden");
        }

        // 6. Render Metabolism interactions
        metabolismContainer.innerHTML = "";
        if (report.metabolism_interactions.length > 0) {
            metabolismCard.classList.remove("hidden");
            report.metabolism_interactions.forEach(item => {
                const row = document.createElement("div");
                const itemSev = item.severity.toLowerCase();
                row.className = `interaction-item item-${itemSev}`;
                row.innerHTML = `
                    <div class="item-header-row">
                        <span class="item-title">${item.substrate} cleared via ${item.enzyme} (${item.modulator})</span>
                        <span class="item-badge ${itemSev}">${item.severity}</span>
                    </div>
                    <div class="item-row">
                        <span class="item-row-label">Metabolic Channel Conflict:</span>
                        <span class="item-row-val">${item.substrate} is a substrate of ${item.enzyme}, while ${item.modulator} is a ${item.interaction_type}.</span>
                    </div>
                    <div class="item-row">
                        <span class="item-row-label">Pharmacokinetic Effect:</span>
                        <span class="item-row-val">${item.clinical_details}</span>
                    </div>
                `;
                metabolismContainer.appendChild(row);
            });
        } else {
            metabolismCard.classList.add("hidden");
        }

        // 6b. Render Suggested Alternatives
        alternativesContainer.innerHTML = "";
        const alts = report.suggested_alternatives || [];
        if (alts.length > 0) {
            alternativesCard.classList.remove("hidden");
            
            // Add prominent Doctor Disclaimer first!
            const disclaimerBox = document.createElement("div");
            disclaimerBox.className = "alternative-disclaimer-card";
            disclaimerBox.innerHTML = `
                <span class="alternative-disclaimer-icon">⚠️</span>
                <span class="alternative-disclaimer-text"><strong>Medical Disclaimer:</strong> Always consult with a licensed physician or doctor before taking or switching to any of these suggested medications.</span>
            `;
            alternativesContainer.appendChild(disclaimerBox);
            
            alts.forEach(alt => {
                const row = document.createElement("div");
                row.className = "interaction-item item-safe";
                row.innerHTML = `
                    <div class="item-header-row">
                        <span class="item-title">${alt.alternative}</span>
                        <span class="item-badge safe">Safer Alternative</span>
                    </div>
                    <div class="item-row">
                        <span class="item-row-label">Replaces Conflict:</span>
                        <span class="item-row-val">${alt.original_drugs}</span>
                    </div>
                    <div class="item-row">
                        <span class="item-row-label">Clinical Reasoning:</span>
                        <span class="item-row-val">${alt.reasoning}</span>
                    </div>
                `;
                alternativesContainer.appendChild(row);
            });
        } else {
            alternativesCard.classList.add("hidden");
        }

        // 7. Render Recommendations
        recommendationsList.innerHTML = "";
        report.recommendations.forEach(rec => {
            const li = document.createElement("li");
            li.textContent = rec;
            
            // Style disclaimer differently
            if (rec.startsWith("DISCLAIMER:") || rec.startsWith("CAUTION: Some generated")) {
                li.className = "disclaimer-item";
            }
            recommendationsList.appendChild(li);
        });

        // 8. Render Bibliography
        citationsList.innerHTML = "";
        report.citations.forEach(cit => {
            const li = document.createElement("li");
            li.textContent = cit;
            citationsList.appendChild(li);
        });

        // Reveal full card
        reportSection.classList.remove("hidden");
        
        // Scroll slightly down to make report visible
        setTimeout(() => {
            reportSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 150);
    }

    /**
     * Controls diagnostic form loading animations
     */
    function setFormLoadingState(loading) {
        if (loading) {
            btnText.textContent = "Orchestrating Agents...";
            spinner.classList.remove("hidden");
            submitBtn.disabled = true;
        } else {
            btnText.textContent = "Run Diagnostic";
            spinner.classList.add("hidden");
            submitBtn.disabled = false;
        }
    }

    /**
     * Utility delay helper
     */
    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Resets dashboard to initial clean slate
     */
    function resetDashboard() {
        if (isChecking) return;
        
        queryInput.value = "";
        reportSection.classList.add("hidden");
        
        consoleStream.innerHTML = "";
        addConsoleLine("SYS", "System dashboard reset complete.", "system-line");
        checkSystemStatus();
    }

    /**
     * Loads diagnosis history from localStorage
     */
    function loadHistory() {
        try {
            const saved = localStorage.getItem("drug_checker_history");
            if (saved) {
                diagnosisHistory = JSON.parse(saved);
            }
        } catch (e) {
            console.error("Error loading history:", e);
            diagnosisHistory = [];
        }
        renderHistoryUI();
    }

    /**
     * Adds a new entry to the diagnosis history
     */
    function saveHistoryEntry(query) {
        // Avoid duplicates (remove older duplicate and put at the top)
        diagnosisHistory = diagnosisHistory.filter(q => q !== query);
        diagnosisHistory.unshift(query);
        
        // Cap history at 5 items
        if (diagnosisHistory.length > 5) {
            diagnosisHistory.pop();
        }
        
        localStorage.setItem("drug_checker_history", JSON.stringify(diagnosisHistory));
        renderHistoryUI();
    }

    /**
     * Renders history entries in the UI
     */
    function renderHistoryUI() {
        historyList.innerHTML = "";
        if (diagnosisHistory.length > 0) {
            historyCard.classList.remove("hidden");
            diagnosisHistory.forEach(query => {
                const button = document.createElement("button");
                button.className = "history-item-btn";
                button.setAttribute("data-query", query);
                button.innerHTML = `
                    <span class="history-item-text">${escapeHtml(query)}</span>
                    <span class="history-arrow">➔</span>
                `;
                historyList.appendChild(button);
            });
        } else {
            historyCard.classList.add("hidden");
        }
    }
});

// Expose click helper in window so SVG inline attributes aren't blocked
window.sendPrompt = function(promptText) {
    // Inject the selected text to show user details
    const consoleBody = document.getElementById("console-stream");
    const timeStr = new Date().toTimeString().split(" ")[0];
    const line = document.createElement("div");
    line.className = "console-line system-line";
    line.innerHTML = `
        <span class="console-time">[${timeStr}]</span>
        <span class="console-tag system">INFO</span>
        <span class="console-msg">Node Clicked: ${promptText}</span>
    `;
    consoleBody.appendChild(line);
    consoleBody.scrollTop = consoleBody.scrollHeight;
};
