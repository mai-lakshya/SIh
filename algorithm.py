import numpy as np
import pandas as pd
import math
from typing import Dict, Any, Tuple

class RiskEngine:
    """
    Production-Grade Risk Engine for SIH Problem Statement 26017
    Features:
      - Bounded [0, 100] guarantee with numerical safety
      - Bayesian shrinkage for historical delay priors
      - Feature attribution & statutory LARR Act lapse detection
      - Vectorized SIMD pipeline for high-throughput batch scoring
    """

    TERRAIN_WEIGHTS = {
        "plain": 0.1, "agricultural": 0.2, "plain/agricultural": 0.2,
        "urban": 0.4, "semi-arid": 0.3, "coastal": 0.5, "wetland": 0.6,
        "coastal/wetland": 0.5, "hilly": 0.7, "mountainous": 0.8,
        "hilly/mountainous": 0.8, "dense forest": 1.0, "forest": 0.9,
        "riverbed": 0.7, "default": 0.5
    }

    def __init__(self, f_max: float = 50000.0, d_max: float = 1000.0, 
                 n_base: float = 60.0, n_limit: float = 365.0):
        self.f_max = max(1.0, f_max)
        self.d_max = max(1.0, d_max)
        self.n_base = n_base
        self.n_limit = n_limit
        self.log_fmax_plus1 = math.log10(self.f_max + 1)
        self.historical_stats = {}
        self.global_mean_delay = 0.0

    def fit_historical_priors(self, df: pd.DataFrame, delay_col: str = "Actual_Delay_Days",
                             state_col: str = "State", proj_col: str = "Project_Type",
                             m_smoothing: float = 5.0):
        """Bayesian empirical mean smoothing to prevent cold-start anomalies."""
        if delay_col not in df.columns:
            return
        self.global_mean_delay = float(df[delay_col].mean())
        if df[delay_col].max() > 0:
            self.d_max = max(self.d_max, float(df[delay_col].max()))
            
        grp = df.groupby([state_col, proj_col])[delay_col].agg(['count', 'mean']).reset_index()
        for _, row in grp.iterrows():
            st, pr = str(row[state_col]), str(row[proj_col])
            n, mean_val = row['count'], row['mean']
            smoothed = (n * mean_val + m_smoothing * self.global_mean_delay) / (n + m_smoothing)
            self.historical_stats[(st, pr)] = smoothed

    def analyze_timeline(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Statutory & Empirical Timeline Slippage Analysis.
        Evaluates schedule friction across legal, regulatory, and compensation factors,
        statutory lapse countdowns under RFCTLARR Act 2013, confidence intervals (P10/P50/P90),
        and critical path bottleneck identification.
        """
        eps = 1e-6
        
        # 1. Base empirical prior delay from state and project type
        st, pr = str(record.get("State", "")), str(record.get("Project_Type", ""))
        base_delay = float(self.historical_stats.get((st, pr), self.global_mean_delay if self.global_mean_delay > 0 else 120.0))

        # 2. Statutory clearance schedule delays
        sia_raw = str(record.get("SIA_Approval_Status", "Approved")).strip().lower()
        if sia_raw in ["approved", "granted", "cleared", "exempted", "not applicable", "n/a"]:
            sia_delay = 0.0
        elif "pending <= 6" in sia_raw or sia_raw in ["in progress", "applied", "in_progress"]:
            sia_delay = 60.0
        elif "pending > 6" in sia_raw or sia_raw in ["pending"]:
            sia_delay = 180.0
        elif sia_raw in ["rejected", "denied"]:
            sia_delay = 365.0
        else:
            sia_delay = 90.0

        fc_raw = str(record.get("Forest_Clearance_Status", "Approved")).strip().lower()
        if fc_raw in ["approved", "granted", "cleared", "exempted", "not required", "not_required"]:
            fc_delay = 0.0
        elif "stage 2" in fc_raw or "stage_2" in fc_raw:
            fc_delay = 45.0
        elif "stage 1" in fc_raw or "stage_1" in fc_raw:
            fc_delay = 120.0
        elif "pending" in fc_raw or "in progress" in fc_raw:
            fc_delay = 180.0
        elif fc_raw in ["rejected", "denied"]:
            fc_delay = 365.0
        else:
            fc_delay = 60.0

        # 3. Dispute litigation & compensation friction
        t_rate = max(0.0, min(100.0, float(record.get("Title_Dispute_Rate_Percent", 0.0))))
        dispute_delay = (t_rate / 100.0) * 240.0

        c_offer = max(0.0, float(record.get("C_offer", 0.0)))
        c_demand = max(0.0, float(record.get("C_demand", 0.0)))
        if c_offer > eps and c_demand > c_offer:
            comp_ratio = min(1.0, (c_demand - c_offer) / c_offer)
            comp_delay = comp_ratio * 150.0
        else:
            comp_delay = 0.0

        # 4. Social agitation & protests
        p_raw = record.get("Local_Protest_Flag", False)
        is_protest = str(p_raw).strip().lower() in ["true", "1", "yes", "high"]
        protest_delay = 90.0 if is_protest else 0.0

        # 5. Section 11 Notification Aging & Statutory Lapse Clock
        days_sec11 = max(0.0, float(record.get("Section_11_Notification_Days", 0.0)))
        days_to_lapse = max(0.0, self.n_limit - days_sec11)
        lapse_warning = days_sec11 >= self.n_limit
        sec11_aging_delay = max(0.0, (days_sec11 - self.n_base) * 0.45) if days_sec11 > self.n_base else 0.0

        # 6. Terrain & Weather Friction
        terrain_str = str(record.get("Terrain_Type", "default")).strip().lower()
        t_weight = self.TERRAIN_WEIGHTS.get(terrain_str, self.TERRAIN_WEIGHTS["default"])
        terrain_delay = t_weight * 60.0
        w_idx = max(0.0, min(10.0, float(record.get("Weather_Index", 1.0))))
        weather_delay = (w_idx / 10.0) * 45.0

        # Composite Expected Delay (P50)
        additive_delay = (sia_delay + fc_delay + dispute_delay + comp_delay + 
                          protest_delay + sec11_aging_delay + terrain_delay + weather_delay)
        expected_delay_days = round(max(0.0, (0.30 * base_delay) + (0.70 * additive_delay)), 1)
        expected_delay_months = round(expected_delay_days / 30.0, 1)

        # Confidence intervals (P10, P50, P90)
        p10_delay = round(max(0.0, expected_delay_days * 0.60), 1)
        p90_delay = round(expected_delay_days * 1.50 + (180.0 if lapse_warning else 0.0), 1)

        # Timeline Risk Phase Classification
        if expected_delay_days < 90.0:
            risk_phase = "Immediate (< 90 Days)"
        elif expected_delay_days < 180.0:
            risk_phase = "Short-Term (90-180 Days)"
        else:
            risk_phase = "Long-Term Severe (> 180 Days)"

        # Detailed Milestone Tracking
        milestones = [
            {
                "milestone": "Social Impact Assessment (SIA)",
                "statutory_act": "RFCTLARR Act 2013 Sec 4 & 7",
                "status": record.get("SIA_Approval_Status", "Approved"),
                "estimated_delay_days": round(sia_delay, 1),
                "is_critical_path": sia_delay >= max(fc_delay, dispute_delay, comp_delay, protest_delay, sec11_aging_delay)
            },
            {
                "milestone": "Section 11 Preliminary Notification",
                "statutory_act": "RFCTLARR Act 2013 Sec 11 & 19(7)",
                "days_elapsed": int(days_sec11),
                "statutory_limit_days": int(self.n_limit),
                "days_remaining_to_lapse": int(days_to_lapse),
                "lapse_warning": lapse_warning,
                "estimated_delay_days": round(sec11_aging_delay, 1),
                "is_critical_path": lapse_warning or (sec11_aging_delay >= max(sia_delay, fc_delay, dispute_delay))
            },
            {
                "milestone": "Forest & Environmental Clearances",
                "statutory_act": "Forest Conservation Act 1980",
                "status": record.get("Forest_Clearance_Status", "Approved"),
                "estimated_delay_days": round(fc_delay, 1),
                "is_critical_path": fc_delay >= max(sia_delay, dispute_delay, comp_delay, protest_delay, sec11_aging_delay)
            },
            {
                "milestone": "Land Title Dispute Adjudication",
                "statutory_act": "RFCTLARR Act 2013 Sec 15 / High Court",
                "dispute_rate_pct": round(t_rate, 1),
                "estimated_delay_days": round(dispute_delay, 1),
                "is_critical_path": dispute_delay >= max(sia_delay, fc_delay, comp_delay, protest_delay, sec11_aging_delay)
            },
            {
                "milestone": "Compensation & Rehabilitation Settlement",
                "statutory_act": "RFCTLARR Act 2013 Sec 23 & 25 (Award)",
                "estimated_delay_days": round(comp_delay + protest_delay, 1),
                "is_critical_path": (comp_delay + protest_delay) >= max(sia_delay, fc_delay, dispute_delay, sec11_aging_delay)
            }
        ]

        # Identify Primary Critical Path Bottleneck
        critical_items = [m for m in milestones if m.get("is_critical_path")]
        bottleneck = critical_items[0]["milestone"] if critical_items else "General Procedural Friction"

        return {
            "Expected_Delay_Days": expected_delay_days,
            "Expected_Delay_Months": expected_delay_months,
            "Confidence_Interval": {
                "P10_Optimistic_Days": p10_delay,
                "P50_Expected_Days": expected_delay_days,
                "P90_Pessimistic_Days": p90_delay
            },
            "Timeline_Risk_Phase": risk_phase,
            "Critical_Path_Bottleneck": bottleneck,
            "Section_11_Lapse_Clock": {
                "Days_Elapsed": int(days_sec11),
                "Days_Remaining": int(days_to_lapse),
                "Lapse_Triggered": lapse_warning,
                "Statutory_Window_Days": int(self.n_limit)
            },
            "Milestones_Breakdown": milestones
        }

    def score_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Single-record evaluation with explainability breakdown, statutory warning, and timeline analysis."""
        eps = 1e-6

        # 1. Socio-Political Risk (R_SP)
        p_raw = record.get("Local_Protest_Flag", False)
        p_r = 1.0 if str(p_raw).strip().lower() in ["true", "1", "yes", "high"] else 0.0

        c_offer = max(0.0, float(record.get("C_offer", 0.0)))
        c_demand = max(0.0, float(record.get("C_demand", 0.0)))
        if c_offer <= eps:
            c_r = 1.0 if c_demand > 0 else 0.0
        else:
            c_r = max(0.0, min(1.0, (c_demand - c_offer) / c_offer))

        aff_fam = max(0.0, float(record.get("Affected_Families_Count", 0.0)))
        f_r = min(1.0, math.log10(aff_fam + 1) / self.log_fmax_plus1)
        r_sp = (0.50 * p_r) + (0.30 * c_r) + (0.20 * f_r)

        # 2. Legal & Regulatory Risk (R_LR)
        t_rate = max(0.0, min(100.0, float(record.get("Title_Dispute_Rate_Percent", 0.0))))
        t_r = t_rate / 100.0

        def parse_clearance(val: Any) -> float:
            if val is None or pd.isna(val): return 0.5
            s = str(val).strip().lower()
            if s in ["approved", "granted", "cleared", "exempted", "not applicable", "n/a"]:
                return 0.0
            elif "pending <= 6" in s or s in ["pending", "in review", "applied"]:
                return 0.5
            elif "pending > 6" in s or s in ["rejected", "denied"]:
                return 1.0
            return 0.5

        sia_r = parse_clearance(record.get("SIA_Approval_Status", "Approved"))
        fc_r = parse_clearance(record.get("Forest_Clearance_Status", "Approved"))
        days_elapsed = max(0.0, float(record.get("Section_11_Notification_Days", 0.0)))
        n_r = max(0.0, min(1.0, (days_elapsed - self.n_base) / max(eps, self.n_limit - self.n_base)))
        r_lr = (0.40 * t_r) + (0.20 * sia_r) + (0.20 * fc_r) + (0.20 * n_r)

        # 3. Environmental & Geo Risk (R_EG)
        terrain_str = str(record.get("Terrain_Type", "default")).strip().lower()
        t_m = self.TERRAIN_WEIGHTS.get(terrain_str, self.TERRAIN_WEIGHTS["default"])
        w_idx = max(0.0, min(10.0, float(record.get("Weather_Index", 1.0))))
        w_r = w_idx / 10.0
        r_eg = (0.50 * t_m) + (0.50 * w_r)

        # 4. Financial & Admin Risk (R_FA)
        f_disb = max(0.0, min(100.0, float(record.get("Fund_Disbursement_Percent", 100.0))))
        f_dr = 1.0 - (f_disb / 100.0)
        st, pr = str(record.get("State", "")), str(record.get("Project_Type", ""))
        h_delay = self.historical_stats.get((st, pr), self.global_mean_delay)
        h_r = max(0.0, min(1.0, h_delay / max(eps, self.d_max)))
        r_fa = (0.60 * f_dr) + (0.40 * h_r)

        # 5. Composite Score & Drivers
        crs = round(max(0.0, min(100.0, ((0.35 * r_sp) + (0.30 * r_lr) + (0.20 * r_eg) + (0.15 * r_fa)) * 100.0)), 2)
        
        risk_band = ("Low Risk" if crs < 35 else "Medium Risk" if crs < 65 else 
                     "High Risk" if crs < 85 else "Critical / Severe Risk")

        contributions = {
            "Socio-Political (R_SP)": round(0.35 * r_sp * 100, 2),
            "Legal & Regulatory (R_LR)": round(0.30 * r_lr * 100, 2),
            "Environmental & Geo (R_EG)": round(0.20 * r_eg * 100, 2),
            "Financial & Admin (R_FA)": round(0.15 * r_fa * 100, 2),
        }

        # 6. Timeline Analysis Integration
        timeline_analysis = self.analyze_timeline(record)

        return {
            "CRS": crs,
            "Risk_Band": risk_band,
            "Dominant_Risk_Driver": max(contributions.items(), key=lambda x: x[1])[0],
            "LARR_Section11_Lapse_Warning": days_elapsed >= self.n_limit,
            "Sub_Indices": {"R_SP": round(r_sp, 4), "R_LR": round(r_lr, 4), 
                            "R_EG": round(r_eg, 4), "R_FA": round(r_fa, 4)},
            "Contributions": contributions,
            "Timeline_Analysis": timeline_analysis
        }

    def score_dataframe_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """High-throughput vectorized execution for batch processing with timeline forecasting."""
        n, eps = len(df), 1e-6

        # Vectorized R_SP
        p_series = df["Local_Protest_Flag"] if "Local_Protest_Flag" in df.columns else pd.Series([False]*n, index=df.index)
        p_r = (p_series.astype(float).values if p_series.dtype in [bool, np.bool_] 
               else np.where(p_series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "high"]), 1.0, 0.0))

        c_offer_s = df["C_offer"] if "C_offer" in df.columns else pd.Series([1.0]*n, index=df.index)
        c_offer = pd.to_numeric(c_offer_s, errors="coerce").fillna(1.0).to_numpy(dtype=float)
        
        c_demand_s = df["C_demand"] if "C_demand" in df.columns else pd.Series([1.0]*n, index=df.index)
        c_demand = pd.to_numeric(c_demand_s, errors="coerce").fillna(1.0).to_numpy(dtype=float)
        c_r = np.clip((c_demand - c_offer) / np.where(c_offer <= eps, eps, c_offer), 0.0, 1.0)
        c_r[(c_offer <= eps) & (c_demand <= eps)] = 0.0

        aff_fam_s = df["Affected_Families_Count"] if "Affected_Families_Count" in df.columns else pd.Series([0]*n, index=df.index)
        aff_fam = np.maximum(0.0, pd.to_numeric(aff_fam_s, errors="coerce").fillna(0).to_numpy(dtype=float))
        f_r = np.clip(np.log10(aff_fam + 1.0) / self.log_fmax_plus1, 0.0, 1.0)
        r_sp = (0.50 * p_r) + (0.30 * c_r) + (0.20 * f_r)

        # Vectorized R_LR
        t_r_s = df["Title_Dispute_Rate_Percent"] if "Title_Dispute_Rate_Percent" in df.columns else pd.Series([0]*n, index=df.index)
        t_r = np.clip(pd.to_numeric(t_r_s, errors="coerce").fillna(0).to_numpy(dtype=float) / 100.0, 0.0, 1.0)
        
        def parse_vec_clearance(col_name: str) -> np.ndarray:
            s = (df[col_name] if col_name in df.columns else pd.Series(["Approved"]*n, index=df.index)).astype(str).str.strip().str.lower()
            arr = np.full(len(s), 0.5, dtype=float)
            arr[s.isin(["approved", "granted", "cleared", "exempted", "not applicable", "n/a"])] = 0.0
            arr[s.str.contains("pending > 6|pending_gt_6|rejected|denied", regex=True)] = 1.0
            return arr

        sia_r, fc_r = parse_vec_clearance("SIA_Approval_Status"), parse_vec_clearance("Forest_Clearance_Status")
        days_s = df["Section_11_Notification_Days"] if "Section_11_Notification_Days" in df.columns else pd.Series([0]*n, index=df.index)
        days = np.maximum(0.0, pd.to_numeric(days_s, errors="coerce").fillna(0).to_numpy(dtype=float))
        n_r = np.clip((days - self.n_base) / max(eps, self.n_limit - self.n_base), 0.0, 1.0)
        r_lr = (0.40 * t_r) + (0.20 * sia_r) + (0.20 * fc_r) + (0.20 * n_r)

        # Vectorized R_EG
        terrain_series = (df["Terrain_Type"] if "Terrain_Type" in df.columns else pd.Series(["default"]*n, index=df.index)).astype(str).str.strip().str.lower()
        t_m = terrain_series.map(self.TERRAIN_WEIGHTS).fillna(self.TERRAIN_WEIGHTS["default"]).to_numpy(dtype=float)
        w_s = df["Weather_Index"] if "Weather_Index" in df.columns else pd.Series([1.0]*n, index=df.index)
        w_r = np.clip(pd.to_numeric(w_s, errors="coerce").fillna(1.0).to_numpy(dtype=float) / 10.0, 0.0, 1.0)
        r_eg = (0.50 * t_m) + (0.50 * w_r)

        # Vectorized R_FA
        f_s = df["Fund_Disbursement_Percent"] if "Fund_Disbursement_Percent" in df.columns else pd.Series([100.0]*n, index=df.index)
        f_dr = np.clip(1.0 - (pd.to_numeric(f_s, errors="coerce").fillna(100.0).to_numpy(dtype=float) / 100.0), 0.0, 1.0)
        states = (df["State"] if "State" in df.columns else pd.Series([""]*n, index=df.index)).astype(str).values
        projs = (df["Project_Type"] if "Project_Type" in df.columns else pd.Series([""]*n, index=df.index)).astype(str).values
        h_delays = np.array([self.historical_stats.get((s, p), self.global_mean_delay if self.global_mean_delay > 0 else 120.0) for s, p in zip(states, projs)], dtype=float)
        h_r = np.clip(h_delays / max(eps, self.d_max), 0.0, 1.0)
        r_fa = (0.60 * f_dr) + (0.40 * h_r)

        # Final CRS Calculation
        crs = np.clip(((0.35 * r_sp) + (0.30 * r_lr) + (0.20 * r_eg) + (0.15 * r_fa)) * 100.0, 0.0, 100.0)

        # Vectorized Timeline Delay Estimation
        def parse_vec_delay(col_name: str, mapping: Dict[str, float], default_val: float) -> np.ndarray:
            s = (df[col_name] if col_name in df.columns else pd.Series(["Approved"]*n, index=df.index)).astype(str).str.strip().str.lower()
            arr = np.full(len(s), default_val, dtype=float)
            for k, v in mapping.items():
                arr[s.str.contains(k, regex=True)] = v
            return arr

        sia_delays = parse_vec_delay("SIA_Approval_Status", {
            "approved|granted|cleared|exempted|not applicable|n/a": 0.0,
            "pending <= 6|in progress|applied|in_progress": 60.0,
            "pending > 6|pending_gt_6": 180.0,
            "rejected|denied": 365.0
        }, default_val=60.0)

        fc_delays = parse_vec_delay("Forest_Clearance_Status", {
            "approved|granted|cleared|exempted|not required|not_required": 0.0,
            "stage 2|stage_2": 45.0,
            "stage 1|stage_1": 120.0,
            "pending|in progress": 180.0,
            "rejected|denied": 365.0
        }, default_val=60.0)

        dispute_delays = t_r * 240.0
        comp_delays = c_r * 150.0
        protest_delays = p_r * 90.0
        sec11_delays = np.where(days > self.n_base, (days - self.n_base) * 0.45, 0.0)
        terrain_delays = t_m * 60.0
        weather_delays = w_r * 45.0

        additive_delays = (sia_delays + fc_delays + dispute_delays + comp_delays +
                           protest_delays + sec11_delays + terrain_delays + weather_delays)
        expected_delays = np.maximum(0.0, np.round((0.30 * h_delays) + (0.70 * additive_delays), 1))

        res_df = df.copy()
        res_df["R_SP"], res_df["R_LR"] = np.round(r_sp, 4), np.round(r_lr, 4)
        res_df["R_EG"], res_df["R_FA"] = np.round(r_eg, 4), np.round(r_fa, 4)
        res_df["CRS"] = np.round(crs, 2)
        res_df["Risk_Band"] = np.select([crs < 35.0, crs < 65.0, crs < 85.0], 
                                        ["Low Risk", "Medium Risk", "High Risk"], default="Critical / Severe Risk")
        res_df["LARR_Sec11_Lapse_Warning"] = days >= self.n_limit
        res_df["Predicted_Delay_Days"] = expected_delays
        res_df["Predicted_Delay_Months"] = np.round(expected_delays / 30.0, 1)
        res_df["Timeline_Risk_Phase"] = np.select(
            [expected_delays < 90.0, expected_delays < 180.0],
            ["Immediate (< 90 Days)", "Short-Term (90-180 Days)"],
            default="Long-Term Severe (> 180 Days)"
        )
        res_df["Sec11_Days_Remaining"] = np.maximum(0.0, self.n_limit - days).astype(int)
        return res_df
