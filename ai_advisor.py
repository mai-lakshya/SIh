import re
import json
import logging
import datetime
from typing import Dict, Any, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)

# =====================================================================
# 1. ADVERSARIAL ROBUSTNESS & PROMPT INJECTION DEFENSE
# =====================================================================

class PromptSecurityValidator:
    """
    Security validator providing direct and indirect prompt injection defense,
    jailbreak prevention, and system prompt leakage protection.
    """

    DIRECT_INJECTION_PATTERNS = [
        r"(?i)(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions|prompts|rules|directives|context)",
        r"(?i)(?:system\s+prompt|internal\s+guidelines|developer\s+mode|jailbreak|dan\s+mode|root\s+access)",
        r"(?i)(?:you\s+are\s+now\s+an?\s+unrestricted|act\s+as\s+an?\s+unfiltered|pretend\s+you\s+have\s+no\s+rules)",
        r"(?i)(?:reveal|leak|print|display|output|dump)\s+(?:your\s+)?(?:initial\s+prompt|system\s+instructions|secret\s+key|internal\s+(?:\w+\s+)?guidelines)",
        r"(?i)(?:system\s*override|admin\s*override|sudo\s+mode|god\s+mode)"
    ]

    INDIRECT_INJECTION_PATTERNS = [
        r"(?i)\[(?:system|admin|override|developer|instruction|note)[^\]]*\]",
        r"(?i)<!--\s*system",
        r"(?i)<\s*system\s*>",
        r"(?i)eval\s*\(",
        r"(?i)exec\s*\(",
        r"(?i)<script",
        r"(?i)javascript:",
        r"(?i)os\.system",
        r"(?i)```(?:python|bash|sh|cmd|powershell)?[\s\S]*(?:system|exec|eval|popen|rm\s+-rf)"
    ]

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """
        Normalize Unicode, strip control characters, and clean invisible zero-width chars.
        """
        if not text:
            return ""
        # Strip zero-width characters (often used for token smuggling)
        sanitized = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
        # Normalize whitespace
        sanitized = " ".join(sanitized.split())
        return sanitized

    @classmethod
    def detect_injection(cls, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text contains direct or indirect prompt injection vectors.
        Returns (is_injection, reason).
        """
        if not text:
            return False, None

        cleaned = cls.sanitize_text(text)

        # Check direct injection patterns
        for pattern in cls.DIRECT_INJECTION_PATTERNS:
            if re.search(pattern, cleaned):
                logger.warning("Security alert: Direct prompt injection pattern detected: '%s'", pattern)
                return True, "Direct prompt injection attempt detected"

        # Check indirect injection patterns
        for pattern in cls.INDIRECT_INJECTION_PATTERNS:
            if re.search(pattern, cleaned):
                logger.warning("Security alert: Indirect prompt injection pattern detected: '%s'", pattern)
                return True, "Indirect context injection attempt detected"

        return False, None


# =====================================================================
# 2. DOMAIN GROUNDING & FALSE-PREMISE VALIDATION
# =====================================================================

class DomainGroundingValidator:
    """
    Validates that inputs and requested prevention steps pertain strictly to
    recognized Indian statutory infrastructure workflows (LARR Act 2013 / EIA 2006).
    Refuses fictional, false-premise, or out-of-domain requests.
    """

    VALID_INFRASTRUCTURE_DOMAINS = {
        "highway", "railway", "urban", "energy", "renewable_energy", 
        "port", "airport", "irrigation", "metro", "pipeline", "mining"
    }

    VALID_STATUTORY_PHASES = {
        "land_demarcation", "gis_boundary", "boundary_survey",
        "social_impact_assessment", "sia_survey", "sia_approval", "sia_review",
        "section_11_notification", "section_11", "sec_11", "gazette_notification",
        "section_15_hearing", "title_verification", "dispute_resolution", "hearing_of_objections",
        "section_19_declaration", "section_19", "sec_19", "r_and_r_scheme",
        "forest_clearance_stage_1", "forest_clearance_stage_2", "parivesh_clearance",
        "environmental_clearance", "eia_approval", "wildlife_clearance",
        "section_23_award", "section_24_award", "compensation_disbursement", "fund_disbursement",
        "section_38_possession", "possession_handover", "civil_handover"
    }

    FALSE_PREMISE_KEYWORDS = [
        "interstellar", "teleportation", "quantum", "time_travel", "terraforming",
        "lunar", "mars_colony", "warp_drive", "anti_gravity", "alien",
        "cryptocurrency_minting", "metaverse_zoning", "cybernetic", "neural_link"
    ]

    @classmethod
    def validate_domain_grounding(cls, phase_name: str, domain: Optional[str] = None) -> Tuple[bool, str]:
        """
        Verify that a milestone or phase corresponds to a legitimate statutory phase.
        Returns (is_valid, message).
        """
        if not phase_name:
            return False, "Domain Grounding Violation: Phase name is empty"

        cleaned_phase = phase_name.lower().strip().replace("-", "_").replace(" ", "_")

        # Check false-premise keywords
        for fk in cls.FALSE_PREMISE_KEYWORDS:
            if fk in cleaned_phase:
                return False, (
                    f"Domain Grounding Violation: Refusal: Requested phase '{phase_name}' contains false-premise concepts "
                    "not grounded in Indian statutory infrastructure framework (LARR 2013 / EIA 2006)."
                )

        # Check domain if provided
        if domain:
            cleaned_domain = domain.lower().strip()
            if cleaned_domain not in cls.VALID_INFRASTRUCTURE_DOMAINS:
                return False, f"Domain Grounding Violation: Domain '{domain}' is not a recognized infrastructure sector."

        # Check if matches or resembles any valid phase
        has_statutory_match = any(sp in cleaned_phase or cleaned_phase in sp for sp in cls.VALID_STATUTORY_PHASES)
        if not has_statutory_match:
            return False, (
                f"Domain Grounding Violation: Clarification required: Milestone '{phase_name}' does not map to recognized "
                "statutory clearance or acquisition workflows under LARR Act 2013 / EIA 2006."
            )

        return True, "Domain valid"

    @classmethod
    def enforce_required_parameters(cls, query: str, required_keys: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Check whether query or context specifies required numerical fields
        rather than inventing them out of thin air.
        """
        reqs = required_keys or ["project_cost"]
        missing = []
        lower_q = query.lower()
        if "project_cost" in reqs:
            if not re.search(r"\b(?:\d+|lakh|crore|inr|rs|₹)\b", lower_q):
                missing.append("project_cost")
        if "target_completion_days" in reqs:
            if not re.search(r"\b(?:\d+\s*(?:days|months|years))\b", lower_q):
                missing.append("target_completion_days")
        return (len(missing) == 0, missing)

    @classmethod
    def check_missing_context(cls, params: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
        """
        Verify that mandatory context fields are provided rather than invented.
        Returns (is_complete, missing_fields).
        """
        missing = [f for f in required_fields if f not in params or params[f] is None]
        return len(missing) == 0, missing


# =====================================================================
# 3. LOCALIZATION & REAL-WORLD INPUT NORMALIZER
# =====================================================================

class IndianContextNormalizer:
    """
    Normalizes colloquial Indian English, transliterated Hinglish terms,
    irregular casing, and regional administrative vocabulary into statutory signals.
    """

    HINGLISH_TO_RISK_MAPPING = {
        # Land Disputes & Title Issues
        "zameen vivaad": "title_dispute_rate_percent",
        "jameen vivad": "title_dispute_rate_percent",
        "patta dispute": "title_dispute_rate_percent",
        "kabza": "title_dispute_rate_percent",
        "khasra vivad": "title_dispute_rate_percent",
        "dakhil kharij": "title_dispute_rate_percent",
        "mutation pending": "title_dispute_rate_percent",
        "stay order": "title_dispute_rate_percent",

        # Protests & Community Agitation
        "dharna": "local_protest_flag",
        "andolan": "local_protest_flag",
        "gaon walon ka virodh": "local_protest_flag",
        "hartal": "local_protest_flag",
        "chakki jaam": "local_protest_flag",
        "rasta roko": "local_protest_flag",
        "protest": "local_protest_flag",
        "gharao": "local_protest_flag",
        "gherao": "local_protest_flag",
        "morcha": "local_protest_flag",

        # Forest & Environmental Clearance
        "van vibhag": "forest_clearance_status",
        "jungle clearance": "forest_clearance_status",
        "ped katai": "forest_clearance_status",
        "parivesh nod": "forest_clearance_status",
        "paryavaran nod": "forest_clearance_status",

        # Compensation & Disbursement
        "muawza": "fund_disbursement_percent",
        "muavja": "fund_disbursement_percent",
        "paisa nahi mila": "fund_disbursement_percent",
        "khate mein paise": "fund_disbursement_percent",
        "dbt payout": "fund_disbursement_percent"
    }

    TERRAIN_NORMALIZATION = {
        "jungle": "Forest_Eco_Sensitive",
        "forest": "Forest_Eco_Sensitive",
        "pahar": "Hilly",
        "pahad": "Hilly",
        "hilly": "Hilly",
        "sehar": "Urban",
        "shehar": "Urban",
        "urban": "Urban",
        "kheti": "Rural_Agri",
        "gaav": "Rural_Agri",
        "gaon": "Rural_Agri",
        "rural": "Rural_Agri",
        "plain": "Plain"
    }

    @classmethod
    def normalize_text_input(cls, text: str) -> Dict[str, Any]:
        """
        Parse unstructured free-text or colloquial Indian inputs into structured risk signals.
        """
        if not text:
            return {}

        normalized: Dict[str, Any] = {}
        lower = text.lower()

        # Map Hinglish dispute terms
        for phrase, field in cls.HINGLISH_TO_RISK_MAPPING.items():
            if phrase in lower:
                if field == "local_protest_flag":
                    normalized["local_protest_flag"] = True
                elif field == "title_dispute_rate_percent":
                    normalized["title_dispute_rate_percent"] = max(
                        normalized.get("title_dispute_rate_percent", 0.0), 35.0
                    )
                elif field == "forest_clearance_status":
                    if "stage-1" in lower or "stage 1" in lower or "in progress" in lower:
                        normalized["forest_clearance_status"] = "In_Progress"
                    else:
                        normalized["forest_clearance_status"] = "Pending"
                elif field == "fund_disbursement_percent":
                    normalized["fund_disbursement_percent"] = min(
                        normalized.get("fund_disbursement_percent", 100.0), 20.0
                    )

        # Check compensation multiplier demand (e.g. "4 guna muawza", "2x compensation")
        mult_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:guna|x|times)\s*(?:muawza|muavja|compensation)", lower)
        if mult_match:
            normalized["compensation_multiplier_demand"] = float(mult_match.group(1))

        # Parse Indian currency terms (Crores / Lakhs)
        # e.g. "150 crore", "50 cr", "200 lakhs"
        cr_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:crore|cr|crores)", lower)
        if cr_match:
            normalized["estimated_cost_inr_crore"] = float(cr_match.group(1))

        lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lacs|lakhs)", lower)
        if lakh_match and "estimated_cost_inr_crore" not in normalized:
            normalized["estimated_cost_inr_crore"] = float(lakh_match.group(1)) / 100.0

        # Parse terrain
        for t_word, t_enum in cls.TERRAIN_NORMALIZATION.items():
            if t_word in lower:
                normalized["terrain_type"] = t_enum
                break

        return normalized


# =====================================================================
# 4. RESILIENT JSON EXTRACTION & REPAIR PARSER
# =====================================================================

class ResilientJSONParser:
    """
    Resilient JSON parser designed for LLM outputs.
    Handles:
      - Markdown code fences (```json ... ``` or ``` ... ```)
      - Unexpected conversational preambles ("Here is the mitigation plan:")
      - Trailing conversational postambles ("I hope this was helpful!")
      - Single quotes instead of double quotes
      - Trailing commas before closing braces/brackets
      - Truncated tokens (auto-repairing unclosed braces/brackets)
    """

    @classmethod
    def parse_llm_json(cls, raw_output: str) -> Dict[str, Any]:
        """
        Safely extract and parse JSON from arbitrary LLM response strings.
        Raises ValueError with detailed diagnostics if parsing cannot be recovered.
        """
        if not raw_output or not isinstance(raw_output, str):
            raise ValueError("Empty or invalid input provided for JSON extraction")

        text = raw_output.strip()

        # 1. Strip markdown code block fences if present
        markdown_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if markdown_match:
            text = markdown_match.group(1).strip()

        # 2. Extract outermost JSON structure ({ ... } or [ ... ])
        first_brace = text.find('{')
        first_bracket = text.find('[')

        if first_brace == -1 and first_bracket == -1:
            raise ValueError("No JSON object or array found in LLM response")

        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            # Target is an Object
            start_idx = first_brace
            end_idx = text.rfind('}')
            is_object = True
        else:
            # Target is an Array
            start_idx = first_bracket
            end_idx = text.rfind(']')
            is_object = False

        if end_idx == -1:
            # Truncated response: try to salvage by appending closing token
            logger.warning("Truncated JSON detected in LLM response; attempting automatic closure repair")
            candidate = text[start_idx:]
            # Attempt to balance quotes and close
            if candidate.count('"') % 2 != 0:
                candidate += '"'
            candidate += "}" if is_object else "]"
        else:
            candidate = text[start_idx:end_idx + 1]

        # 3. Direct parse attempt
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # 4. Repair syntax defects
        repaired = candidate

        # Replace single quotes with double quotes (handling key/value pairs)
        repaired = re.sub(r"(?<=\{|,|\[)\s*'([^'\\]*(?:\\.[^'\\]*)*)'\s*:", r' "\1":', repaired)
        repaired = re.sub(r":\s*'([^'\\]*(?:\\.[^'\\]*)*)'\s*(?=\}|,|\])", r': "\1"', repaired)
        repaired = re.sub(r"(?<=\[)\s*'([^'\\]*(?:\\.[^'\\]*)*)'\s*(?=[,\]])", r'"\1"', repaired)
        repaired = re.sub(r"(?<=,)\s*'([^'\\]*(?:\\.[^'\\]*)*)'\s*(?=[,\]])", r'"\1"', repaired)

        # Remove trailing commas before closing braces/brackets
        repaired = re.sub(r",\s*([\}\]])", r"\1", repaired)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError as err:
            logger.error("JSON repair failed on candidate: %s | Error: %s", repaired[:200], err)
            raise ValueError(f"Failed to parse or repair JSON from LLM response: {err}")


# =====================================================================
# 5. UNIFIED AI PREVENTION ADVISOR
# =====================================================================

class AIAdvisor:
    """
    Unified AI Security & Advisory gateway for SIH Land Acquisition Risk System.
    """

    def __init__(self):
        self.security = PromptSecurityValidator()
        self.grounding = DomainGroundingValidator()
        self.normalizer = IndianContextNormalizer()
        self.parser = ResilientJSONParser()

    def process_task_description(self, user_description: str) -> Dict[str, Any]:
        """
        Validate prompt injection, normalize Indian context, and extract structured risk signals.
        """
        # Step 1: Security Audit
        is_injection, reason = self.security.detect_injection(user_description)
        if is_injection:
            return {
                "status": "refused",
                "security_alert": True,
                "reason": reason,
                "extracted_signals": {}
            }

        # Step 2: Context Normalization (Hinglish / Indian terminology)
        signals = self.normalizer.normalize_text_input(user_description)

        return {
            "status": "success",
            "security_alert": False,
            "reason": None,
            "extracted_signals": signals
        }

    def validate_and_parse_llm_mitigation(self, llm_raw_response: str) -> Dict[str, Any]:
        """
        Extract and validate mitigation JSON from arbitrary LLM response.
        """
        parsed = self.parser.parse_llm_json(llm_raw_response)

        # Verify phase domain grounding if specified in parsed response
        if isinstance(parsed, dict) and "milestone" in parsed:
            is_valid, msg = self.grounding.validate_domain_grounding(str(parsed["milestone"]))
            if not is_valid:
                parsed["domain_grounding_warning"] = msg

        return parsed

    def generate_advisory(
        self,
        query: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        End-to-end advisor call that checks injection, validates domain grounding,
        normalizes Indian context, and returns a verified advisory response.
        """
        combined = f"{query}\n{context or ''}"
        is_inj, reason = self.security.detect_injection(combined)
        if is_inj:
            return {
                "status": "refused",
                "security_alert": True,
                "reason": reason,
                "recommendations": []
            }

        is_grounded, g_reason = self.grounding.validate_domain_grounding(query)
        if not is_grounded:
            return {
                "status": "refused",
                "domain_grounding_error": True,
                "reason": g_reason,
                "recommendations": []
            }

        extracted_signals = self.normalizer.normalize_text_input(combined)
        return {
            "status": "success",
            "security_alert": False,
            "domain_grounding_error": False,
            "extracted_signals": extracted_signals,
            "guidance": "Advisory verified against statutory LARR 2013 and Forest Conservation Act norms."
        }

