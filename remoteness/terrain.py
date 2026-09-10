"""
Terrain & Forest / Tribal Regulatory Area Classifier.
Classifies topography into plain, hilly, coastal, or forest_tribal
and determines applicability of the Forest Rights Act (FRA 2006) and PESA compliance.
"""

from typing import Tuple, Optional, Set

# Known coastal districts / coordinates in India
COASTAL_DISTRICTS: Set[str] = {
    "mumbai", "mumbai city", "mumbai suburban", "thane", "palghar", "raigad",
    "ratnagiri", "sindhudurg", "north goa", "south goa", "uttara kannada", "udupi",
    "dakshina kannada", "kasaragod", "kannur", "kozhikode", "malappuram", "thrissur",
    "ernakulam", "alappuzha", "kollam", "thiruvananthapuram", "kanyakumari",
    "tirunelveli", "thoothukudi", "ramanathapuram", "pudukkottai", "thanjavur",
    "tiruvarur", "nagapattinam", "cuddalore", "viluppuram", "chengalpattu", "chennai",
    "tiruvallur", "nellore", "prakasam", "bapatla", "krishna", "west godavari",
    "east godavari", "konaseema", "visakhapatnam", "vizianagaram", "srikakulam",
    "ganjam", "puri", "jagatsinghpur", "kendrapara", "bhadrak", "balasore",
    "purba medinipur", "south 24 parganas", "north 24 parganas"
}

# Known Schedule V / Forest Tribal Districts under Constitution of India & FRA 2006
SCHEDULE_V_TRIBAL_DISTRICTS: Set[str] = {
    "bastar", "dantewada", "bijapur", "sukma", "narayanpur", "kondagaon", "kanker",
    "dhamtari", "gariaband", "rajnandgaon", "kawardha", "bilaspur", "korba", "surguja",
    "surajpur", "balrampur", "jashpur", "koriya", "latehar", "gumla", "khunti", "simdega",
    "west singhbhum", "east singhbhum", "ranchi", "dumka", "pakur", "sahebganj", "jamtara",
    "mayurbhanj", "keonjhar", "sundargarh", "kandhamal", "koraput", "malkangiri",
    "nabarangpur", "rayagada", "gajapati", "gadchiroli", "chandrapur", "nandurbar",
    "dhule", "palghar", "nashik", "banswara", "dungarpur", "pratapgarh", "udaipur",
    "sirohi", "alirajpur", "jhabua", "barwani", "dhar", "khargone", "mandla", "dindori",
    "chhindwara", "seoni", "balaghat", "anuppur", "shahdol", "umaria", "adilabad",
    "komaram bheem asifabad", "bhadradri kothagudem", "alluri sitharama raju", "parvathipuram manyam"
}

# Known Hilly / Himalayan / Western Ghats Districts
HILLY_DISTRICTS: Set[str] = {
    "shimla", "kullu", "mandi", "chamba", "kangra", "kinnaur", "lahaul and spiti",
    "solan", "sirmour", "dehradun", "haridwar", "tehri garhwal", "pauri garhwal",
    "uttarkashi", "chamoli", "rudraprayag", "almora", "pithoragarh", "nainital",
    "bageshwar", "champawat", "leh", "kargil", "srinagar", "anantnag", "baramulla",
    "kupwara", "budgam", "pulwama", "shopian", "kulgam", "poonch", "rajouri",
    "udhampur", "reasi", "doda", "ramban", "kishtwar", "darjeeling", "kalimpong",
    "east sikkim", "west sikkim", "north sikkim", "south sikkim", "nilgiris", "wayanad", "idukki"
}


from .vedas_client import VedasTerrainDetector, VALID_TERRAIN_ENUMS, TERRAIN_LABELS


class TerrainClassifier:
    """
    Classifies terrain and statutory forest/tribal status, integrating ISRO VEDAS satellite telemetry.
    """

    def detect_vedas_telemetry(
        self,
        lat: Optional[float],
        lon: Optional[float],
        state: Optional[str] = None,
        district: Optional[str] = None
    ) -> dict:
        """
        Queries ISRO VEDAS satellite telemetry engine for elevation, slope, and LULC classification.
        """
        return VedasTerrainDetector.detect_terrain(lat=lat, lon=lon, state=state, district=district)

    def classify_terrain(
        self,
        lat: float,
        lon: float,
        district: Optional[str] = None,
        provided_terrain: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Determines the terrain type and whether the site falls under Forest/Tribal (FRA 2006) regulation.

        Returns:
            (terrain_type, is_forest_tribal)
            where terrain_type in {"plain", "hilly", "coastal", "forest_tribal"}
        """
        # 1. Explicitly supplied terrain from project record
        if provided_terrain:
            clean = str(provided_terrain).strip().lower()
            if "forest" in clean or "tribal" in clean or "schedulev" in clean:
                return "forest_tribal", True
            elif "hill" in clean or "mountain" in clean:
                return "hilly", False
            elif "coast" in clean:
                return "coastal", False
            elif "plain" in clean or "rural_agri" in clean or "urban" in clean:
                return "plain", False

        # 2. Query VEDAS satellite telemetry engine
        try:
            v_res = VedasTerrainDetector.detect_terrain(lat=lat, lon=lon, district=district)
            v_type = v_res.get("terrain_type")
            is_ft = bool(v_res.get("telemetry", {}).get("is_forest_tribal", False))
            if v_type in ["Forest_Eco_Sensitive", "Tribal_Schedule_V"]:
                return "forest_tribal", True
            elif v_type == "Hilly":
                return "hilly", False
            elif v_type == "Urban":
                return "plain", False
            elif v_type == "Rural_Agri":
                return "plain", False
        except Exception:
            pass

        # 3. District-based statutory Schedule V / Forest check
        if district:
            d_norm = district.strip().lower()
            if d_norm in SCHEDULE_V_TRIBAL_DISTRICTS:
                return "forest_tribal", True
            if d_norm in HILLY_DISTRICTS:
                return "hilly", False
            if d_norm in COASTAL_DISTRICTS:
                return "coastal", False

        # 4. Coordinate bounds geographic proxy
        # Himalayan / Northern Hill region
        if lat >= 30.0:
            return "hilly", False
        # Western Ghats / Nilgiris high-elevation belt
        if (8.5 <= lat <= 15.5) and (75.0 <= lon <= 77.0):
            return "hilly", False

        return "plain", False

