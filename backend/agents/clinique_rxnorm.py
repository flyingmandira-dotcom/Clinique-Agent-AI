"""
RxNorm Drug Normalization & Recognition Module for Clinique AI
================================================================================
Replaces hardcoded drug list with live RxNorm lookups (findRxcuiByString, getApproximateMatch).
Flags unrecognized drugs explicitly instead of silently dropping them.
Free, no API key required. All endpoints are GET; no rate limiting in practice.
Includes in-memory LRU caching and robust offline fallback database.
"""

import logging
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from functools import lru_cache

logger = logging.getLogger("drug_checker.rxnorm")

# Try importing requests; if not available, urllib fallback is used
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    import urllib.request
    import urllib.error
    import urllib.parse
    REQUESTS_AVAILABLE = False


# ============================================================================
# Offline Mock & Fast Lookup Database (Used as fallback or fast path)
# ============================================================================

MOCK_DRUG_DATABASE: Dict[str, Dict[str, str]] = {
    "aspirin": {"rxcui": "7052", "preferred_name": "aspirin"},
    "lisinopril": {"rxcui": "31708", "preferred_name": "lisinopril"},
    "eliquis": {"rxcui": "1364430", "preferred_name": "apixaban"},
    "apixaban": {"rxcui": "1364430", "preferred_name": "apixaban"},
    "ozempic": {"rxcui": "2371897", "preferred_name": "semaglutide"},
    "semaglutide": {"rxcui": "2371897", "preferred_name": "semaglutide"},
    "ibuprofen": {"rxcui": "5186", "preferred_name": "ibuprofen"},
    "metformin": {"rxcui": "6809", "preferred_name": "metformin"},
    "warfarin": {"rxcui": "11289", "preferred_name": "warfarin"},
    "coumadin": {"rxcui": "11289", "preferred_name": "warfarin"},
    "atorvastatin": {"rxcui": "83367", "preferred_name": "atorvastatin"},
    "lipitor": {"rxcui": "83367", "preferred_name": "atorvastatin"},
    "simvastatin": {"rxcui": "36567", "preferred_name": "simvastatin"},
    "zocor": {"rxcui": "36567", "preferred_name": "simvastatin"},
    "amoxicillin": {"rxcui": "2670", "preferred_name": "amoxicillin"},
    "penicillin": {"rxcui": "7980", "preferred_name": "penicillin"},
    "insulin": {"rxcui": "7678", "preferred_name": "insulin"},
    "metoprolol": {"rxcui": "6918", "preferred_name": "metoprolol"},
    "lopressor": {"rxcui": "6918", "preferred_name": "metoprolol"},
    "toprol-xl": {"rxcui": "6918", "preferred_name": "metoprolol"},
    "toprol xl": {"rxcui": "6918", "preferred_name": "metoprolol"},
    "atenolol": {"rxcui": "1201", "preferred_name": "atenolol"},
    "furosemide": {"rxcui": "4603", "preferred_name": "furosemide"},
    "lasix": {"rxcui": "4603", "preferred_name": "furosemide"},
    "omeprazole": {"rxcui": "7677", "preferred_name": "omeprazole"},
    "prilosec": {"rxcui": "7677", "preferred_name": "omeprazole"},
    "acetaminophen": {"rxcui": "161", "preferred_name": "acetaminophen"},
    "paracetamol": {"rxcui": "161", "preferred_name": "acetaminophen"},
    "tylenol": {"rxcui": "161", "preferred_name": "acetaminophen"},
    "advil": {"rxcui": "5186", "preferred_name": "ibuprofen"},
    "motrin": {"rxcui": "5186", "preferred_name": "ibuprofen"},
    "clarithromycin": {"rxcui": "21212", "preferred_name": "clarithromycin"},
    "biaxin": {"rxcui": "21212", "preferred_name": "clarithromycin"},
    "sildenafil": {"rxcui": "136411", "preferred_name": "sildenafil"},
    "viagra": {"rxcui": "136411", "preferred_name": "sildenafil"},
    "nitroglycerin": {"rxcui": "7447", "preferred_name": "nitroglycerin"},
    "nitrostat": {"rxcui": "7447", "preferred_name": "nitroglycerin"},
    "propranolol": {"rxcui": "8787", "preferred_name": "propranolol"},
    "inderal": {"rxcui": "8787", "preferred_name": "propranolol"},
    "amlodipine": {"rxcui": "17767", "preferred_name": "amlodipine"},
    "norvasc": {"rxcui": "17767", "preferred_name": "amlodipine"},
    "fluconazole": {"rxcui": "4450", "preferred_name": "fluconazole"},
    "diflucan": {"rxcui": "4450", "preferred_name": "fluconazole"},
    "cetirizine": {"rxcui": "20610", "preferred_name": "cetirizine"},
    "zyrtec": {"rxcui": "20610", "preferred_name": "cetirizine"},
    "amiodarone": {"rxcui": "703", "preferred_name": "amiodarone"},
    "cordarone": {"rxcui": "703", "preferred_name": "amiodarone"},
    "diltiazem": {"rxcui": "3443", "preferred_name": "diltiazem"},
    "verapamil": {"rxcui": "11170", "preferred_name": "verapamil"},
    "phenelzine": {"rxcui": "8134", "preferred_name": "phenelzine"},
    "spironolactone": {"rxcui": "9997", "preferred_name": "spironolactone"},
    "pseudoephedrine": {"rxcui": "8852", "preferred_name": "pseudoephedrine"},
    "prednisone": {"rxcui": "8640", "preferred_name": "prednisone"},
    "fluoxetine": {"rxcui": "4492", "preferred_name": "fluoxetine"},
    "prozac": {"rxcui": "4492", "preferred_name": "fluoxetine"},
    "grapefruit juice": {"rxcui": "19920", "preferred_name": "grapefruit juice"},
}

FUZZY_PATTERNS: Dict[str, str] = {
    "asprin": "aspirin",
    "paracetemol": "paracetamol",
    "paracetmol": "paracetamol",
    "metformn": "metformin",
    "lisno": "lisinopril",
    "ibup": "ibuprofen",
    "ibuprofin": "ibuprofen",
    "warfrin": "warfarin",
    "eliquis": "eliquis",
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DrugRecognized:
    """A drug that was successfully recognized and normalized by RxNorm."""
    name: str  # Original input mention
    rxcui: str  # Official RxCUI identifier
    preferred_name: str  # Canonical generic name
    match_type: str  # "exact" | "approximate" | "normalized"

    def to_dict(self):
        return asdict(self)


@dataclass
class DrugUnrecognized:
    """A drug that could not be recognized."""
    name: str  # Original input name
    reason: str  # Explanation for lack of recognition
    attempted_searches: Optional[List[str]] = None

    def to_dict(self):
        d = asdict(self)
        if self.attempted_searches is None:
            d.pop("attempted_searches", None)
        return d


# ============================================================================
# RxNorm Client
# ============================================================================

class RxNormClient:
    """
    Live RxNorm API Client with offline fallback & in-memory caching.
    API Docs: https://rxnav.nlm.nih.gov/APIs/api-RxNorm.html
    """
    BASE_URL = "https://rxnav.nlm.nih.gov/REST"

    def __init__(self, timeout: int = 4, cache_size: int = 512):
        self.timeout = timeout
        self.cache_size = cache_size
        self._memory_cache: Dict[str, Tuple[Optional[DrugRecognized], Optional[DrugUnrecognized]]] = {}

    def _http_get_json(self, url: str, params: Optional[Dict[str, str]] = None) -> Optional[Dict]:
        """Performs an HTTP GET request returning parsed JSON with timeout control."""
        if REQUESTS_AVAILABLE:
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                return None
            except Exception as e:
                logger.debug(f"RxNorm HTTP error ({url}): {e}")
                return None
        else:
            try:
                full_url = url
                if params:
                    full_url += "?" + urllib.parse.urlencode(params)
                req = urllib.request.Request(full_url, headers={"User-Agent": "CliniqueAI/1.0", "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    if response.status == 200:
                        return json.loads(response.read().decode("utf-8"))
                return None
            except Exception as e:
                logger.debug(f"RxNorm urllib error ({url}): {e}")
                return None

    def find_rxcui_by_string(self, drug_name: str) -> Optional[Dict]:
        """Calls findRxcuiByString endpoint for exact or close matches."""
        url = f"{self.BASE_URL}/rxcui"
        params = {"name": drug_name, "search": "1"}
        data = self._http_get_json(url, params=params)
        if data:
            id_group = data.get("idGroup", {})
            rxui_list = id_group.get("rxuiList", [])
            if rxui_list:
                return {
                    "rxcui": rxui_list[0],
                    "all_rxcuis": rxui_list,
                    "match_type": "exact"
                }
        return None

    def get_approximate_match(self, drug_name: str) -> Optional[Dict]:
        """Calls approximateMatch.json endpoint for fuzzy/spelling matches."""
        url = f"{self.BASE_URL}/approximateMatch.json"
        params = {"term": drug_name}
        data = self._http_get_json(url, params=params)
        if data:
            candidates = data.get("approximateGroup", {}).get("candidate", [])
            if candidates:
                best = candidates[0]
                rxcui = best.get("rxcui", best.get("rxaui", ""))
                name = best.get("name", drug_name)
                if rxcui:
                    return {
                        "rxcui": rxcui,
                        "preferred_name": name,
                        "match_type": "approximate"
                    }
        return None

    def get_canonical_name_for_rxcui(self, rxcui: str) -> Optional[str]:
        """Retrieves official generic/ingredient name for an RxCUI."""
        url = f"{self.BASE_URL}/rxcui/{rxcui}/properties.json"
        data = self._http_get_json(url)
        if data:
            props = data.get("properties", {})
            return props.get("name")
        return None

    def normalize_drug(self, drug_name: str) -> Tuple[Optional[DrugRecognized], Optional[DrugUnrecognized]]:
        """
        Normalizes a single drug mention:
        1. Checks in-memory cache.
        2. Fast check against MOCK_DRUG_DATABASE (handles brand-to-generic mappings instantly).
        3. Queries live RxNorm endpoints (findRxcuiByString -> approximateMatch).
        4. Checks fuzzy patterns.
        5. If all fail, returns DrugUnrecognized.
        """
        if not drug_name or not isinstance(drug_name, str):
            return None, DrugUnrecognized(
                name=str(drug_name),
                reason="Invalid input: empty or non-string",
                attempted_searches=[]
            )

        clean_name = drug_name.strip().lower()
        if not clean_name:
            return None, DrugUnrecognized(
                name=drug_name,
                reason="Invalid input: empty string",
                attempted_searches=[]
            )

        # Check Cache
        if clean_name in self._memory_cache:
            return self._memory_cache[clean_name]

        # 1. Fast Path: Known Mock Database (provides instant brand->generic and RxCUI)
        if clean_name in MOCK_DRUG_DATABASE:
            entry = MOCK_DRUG_DATABASE[clean_name]
            res = (
                DrugRecognized(
                    name=drug_name,
                    rxcui=entry["rxcui"],
                    preferred_name=entry["preferred_name"],
                    match_type="exact" if clean_name == entry["preferred_name"] else "normalized"
                ),
                None
            )
            self._cache_result(clean_name, res)
            return res

        # 2. Live RxNorm: Exact Match
        exact_match = self.find_rxcui_by_string(clean_name)
        if exact_match:
            rxcui = exact_match["rxcui"]
            # Look up preferred canonical name
            canon_name = self.get_canonical_name_for_rxcui(rxcui) or clean_name
            res = (
                DrugRecognized(
                    name=drug_name,
                    rxcui=rxcui,
                    preferred_name=canon_name.lower(),
                    match_type="exact"
                ),
                None
            )
            self._cache_result(clean_name, res)
            return res

        # 3. Live RxNorm: Approximate / Fuzzy Match
        approx_match = self.get_approximate_match(clean_name)
        if approx_match:
            res = (
                DrugRecognized(
                    name=drug_name,
                    rxcui=approx_match["rxcui"],
                    preferred_name=approx_match["preferred_name"].lower(),
                    match_type="approximate"
                ),
                None
            )
            self._cache_result(clean_name, res)
            return res

        # 4. Offline Fuzzy / Typo Fallback
        for pattern, canonical in FUZZY_PATTERNS.items():
            if pattern in clean_name and canonical in MOCK_DRUG_DATABASE:
                entry = MOCK_DRUG_DATABASE[canonical]
                res = (
                    DrugRecognized(
                        name=drug_name,
                        rxcui=entry["rxcui"],
                        preferred_name=entry["preferred_name"],
                        match_type="approximate"
                    ),
                    None
                )
                self._cache_result(clean_name, res)
                return res

        # 5. Failed to Recognize: Produce Explicit Unrecognized Record
        unrec = (
            None,
            DrugUnrecognized(
                name=drug_name,
                reason="No match found in RxNorm (exact or approximate)",
                attempted_searches=[clean_name]
            )
        )
        self._cache_result(clean_name, unrec)
        return unrec

    def _cache_result(self, key: str, value: Tuple[Optional[DrugRecognized], Optional[DrugUnrecognized]]):
        if len(self._memory_cache) >= self.cache_size:
            # Evict oldest
            self._memory_cache.pop(next(iter(self._memory_cache)))
        self._memory_cache[key] = value

    def normalize_drug_list(
        self,
        drug_names: List[str],
        strict: bool = False
    ) -> Tuple[List[DrugRecognized], List[DrugUnrecognized]]:
        """Normalizes a list of candidate drug names."""
        recognized: List[DrugRecognized] = []
        unrecognized: List[DrugUnrecognized] = []

        for name in drug_names:
            rec, unrec = self.normalize_drug(name)
            if rec:
                # Deduplicate by preferred_name
                if not any(r.preferred_name == rec.preferred_name for r in recognized):
                    recognized.append(rec)
            else:
                if unrec and not any(u.name.lower() == unrec.name.lower() for u in unrecognized):
                    unrecognized.append(unrec)

            if strict and unrec:
                return recognized, unrecognized

        return recognized, unrecognized


# ============================================================================
# Structured Output Formatter
# ============================================================================

class DrugNormalizationReport:
    """Structured report of drug normalization results."""
    def __init__(self, drugs_recognized: List[DrugRecognized], drugs_unrecognized: List[DrugUnrecognized]):
        self.drugs_recognized = drugs_recognized
        self.drugs_unrecognized = drugs_unrecognized

    @property
    def has_unrecognized(self) -> bool:
        return len(self.drugs_unrecognized) > 0

    @property
    def recognition_rate(self) -> float:
        total = len(self.drugs_recognized) + len(self.drugs_unrecognized)
        if total == 0:
            return 100.0
        return (len(self.drugs_recognized) / total) * 100

    def to_dict(self) -> Dict:
        return {
            "status": "complete",
            "summary": {
                "total_input": len(self.drugs_recognized) + len(self.drugs_unrecognized),
                "recognized": len(self.drugs_recognized),
                "unrecognized": len(self.drugs_unrecognized),
                "recognition_rate_percent": round(self.recognition_rate, 2)
            },
            "recognized_drugs": [d.to_dict() for d in self.drugs_recognized],
            "unrecognized_drugs": [d.to_dict() for d in self.drugs_unrecognized],
            "note": "Unrecognized drugs trigger REVIEW_REQUIRED / DATA_UNAVAILABLE (preventing false SAFE)."
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
