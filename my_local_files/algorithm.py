import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

class RiskEngine:
    """
    Production-Grade Comprehensive Risk Engine for Infrastructure Projects
    Compliant with SIH Problem Statement 26017 & Indian Statutory Regulatory Frameworks:
      - Right to Fair Compensation and Transparency in Land Acquisition, Rehabilitation 
        and Resettlement (RFCTLARR) Act, 2013 (Sections 4, 7, 8, 11, 14, 15, 19, 25, 41)
      - Forest (Conservation) Act, 1980 / Van (Sanrakshan Evam Samvardhan) Adhiniyam, 2023
      - Wildlife (Protection) Act, 1972 (NBWL Eco-Sensitive Zones)
      - Environment (Protection) Act, 1986 & EIA Notification 2006 (Categorization A/B)
      - Panchayats (Extension to Scheduled Areas) Act (PESA), 1996 (Schedule V Gram Sabha)
    """

    # Comprehensive Geotechnical & Topographic Multipliers
    TERRAIN_WEIGHTS = {
        # Flat / Low Geotechnical Resistance
        "plain": 0.10,
        "agricultural": 0.18,
        "plain / agricultural": 0.18,
        "plain/agricultural": 0.18,
        "rural_agri": 0.18,
        "semi-arid": 0.28,
        "desert_arid": 0.35,

        # Urban & High-Density Acquisition
        "urban": 0.45,
        "urban_dense": 0.55,
        "industrial": 0.38,

        # Water Bodies & Coastal Regimes
        "coastal": 0.52,
        "coastal / wetland": 0.58,
        "coastal/wetland": 0.58,
        "wetland": 0.65,
        "floodplain_riverine": 0.68,
        "riverbed": 0.70,

        # High Slope & Mountainous Terrain
        "hilly": 0.72,
        "mountainous": 0.82,
        "hilly / mountainous": 0.82,
        "hilly/mountainous": 0.82,
        "steep_ghats_landslide_prone": 0.92,

        # Forest & Eco-Sensitive
        "forest": 0.85,
        "dense forest": 0.95,
        "forest_eco_sensitive": 0.90,
        "tribal_schedule_v": 0.88,
        
        "default": 0.50
    }

    # State-Specific Institutional Fast-Track Prior Modifiers
    STATE_INSTITUTIONAL_EFFICIENCY = {
        "Gujarat": 0.82,        # e-Bhoomi Land pooling & GIDC streamlined acquisition
        "Haryana": 0.85,        # e-Bhoomi transparent portal
        "Tamil Nadu": 0.88,     # State Industrial Highways Act specialized land benches
        "Maharashtra": 0.92,    # Direct purchase model (GR 2015)
        "Andhra Pradesh": 0.94, # Capital city land pooling model
        "Karnataka": 0.95,      # KIADB direct consent acquisition
        "Uttar Pradesh": 1.05,  # High population density & court litigation rate
        "Bihar": 1.15,          # High land fragmentation & title disputes
        "West Bengal": 1.22,    # Multi-crop protection & stringent consent thresholds
        "Kerala": 1.25,         # Extreme density & highest market compensation expectations
        "default": 1.00
    }

    def __init__(self, 
                 f_max: float = 50000.0, 
                 d_max: float = 1000.0, 
                 n_base: float = 60.0, 
                 n_limit: float = 365.0,
                 cost_of_capital_annual: float = 0.105):
        """
        Args:
            f_max: Max scale for affected families log-normalization (default 50,000 PAPs).
            d_max: Max historical delay scale in days (default 1,000 days).
            n_base: Statutory objection buffer window under Section 15 (default 60 days).
            n_limit: Drop-dead statutory preliminary notification lapse under Section 14 (365 days).
            cost_of_capital_annual: Weighted Average Cost of Capital (WACC) for IDC financial burn (10.5%).
        """
        self.f_max = max(1.0, float(f_max))
        self.d_max = max(1.0, float(d_max))
        self.n_base = float(n_base)
        self.n_limit = float(n_limit)
        self.cost_of_capital_annual = float(cost_of_capital_annual)
        self.log_fmax_plus1 = math.log10(self.f_max + 1.0)
        
        # Bayesian prior matrices
        self.historical_stats = {}
        self.state_delay_stats = {}
        self.sector_delay_stats = {}
        self.global_mean_delay = 180.0
        self.global_std_delay = 90.0

    def fit_historical_priors(self, df: pd.DataFrame, 
                              delay_col: str = "Actual_Delay_Days",
                              state_col: str = "State", 
                              proj_col: str = "Project_Type",
                              m_smoothing: float = 5.0):
        """
        Hierarchical Bayesian Empirical Bayes Smoothing.
        Prevents cold-start anomalies across rare state-sector combinations:
          p(delay | state, sector) ~ smoothed with empirical state & sector priors.
        """
        if delay_col not in df.columns:
            return

        valid_df = df.dropna(subset=[delay_col])
        if len(valid_df) == 0:
            return

        self.global_mean_delay = float(valid_df[delay_col].mean())
        self.global_std_delay = float(valid_df[delay_col].std()) if len(valid_df) > 1 else 90.0
        if valid_df[delay_col].max() > 0:
            self.d_max = max(self.d_max, float(valid_df[delay_col].max()))

        # 1. State-level priors
        if state_col in valid_df.columns:
            st_grp = valid_df.groupby(state_col)[delay_col].agg(['count', 'mean']).reset_index()
            for _, r in st_grp.iterrows():
                n_st, mean_st = r['count'], r['mean']
                self.state_delay_stats[str(r[state_col])] = (n_st * mean_st + m_smoothing * self.global_mean_delay) / (n_st + m_smoothing)

        # 2. Sector-level priors
        if proj_col in valid_df.columns:
            pr_grp = valid_df.groupby(proj_col)[delay_col].agg(['count', 'mean']).reset_index()
            for _, r in pr_grp.iterrows():
                n_pr, mean_pr = r['count'], r['mean']
                self.sector_delay_stats[str(r[proj_col])] = (n_pr * mean_pr + m_smoothing * self.global_mean_delay) / (n_pr + m_smoothing)

        # 3. State x Sector Hierarchical priors
        if state_col in valid_df.columns and proj_col in valid_df.columns:
            grp = valid_df.groupby([state_col, proj_col])[delay_col].agg(['count', 'mean']).reset_index()
            for _, row in grp.iterrows():
                st, pr = str(row[state_col]), str(row[proj_col])
                n, mean_val = row['count'], row['mean']
                base_prior = self.state_delay_stats.get(st, self.sector_delay_stats.get(pr, self.global_mean_delay))
                smoothed = (n * mean_val + m_smoothing * base_prior) / (n + m_smoothing)
                self.historical_stats[(st, pr)] = smoothed

    def score_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive Multi-Pillar Risk Evaluation Engine.
        Evaluates 4 Statutory & Operational Risk Pillars:
          1. Pillar I: Socio-Legal & Resettlement Risk (R_SL) - Weight 30%
          2. Pillar II: Statutory & Environmental Clearance Risk (R_EC) - Weight 25%
          3. Pillar III: Financial & Disbursement Risk (R_FD) - Weight 25%
          4. Pillar IV: Administrative Workflow & Timeline Drift (R_AW) - Weight 20%
        """
        eps = 1e-6
        statutory_flags: List[Dict[str, str]] = []
        action_recommendations: List[Dict[str, Any]] = []

        # =====================================================================
        # PILLAR I: SOCIO-LEGAL & RESETTLEMENT RISK (R_SL)
        # =====================================================================
        # 1.1 Local Protest & Agitation Intensity (P_r)
        p_raw = record.get("Local_Protest_Flag", record.get("local_protest_flag", False))
        if str(p_raw).strip().lower() in ["true", "1", "yes", "high", "active"]:
            p_r = 1.0
            statutory_flags.append({
                "code": "SL-01-PROTEST",
                "severity": "CRITICAL",
                "title": "Active Local Civil Agitation",
                "act_reference": "RFCTLARR Act 2013 Section 15 & CrPC Sec 144",
                "description": "Ground agitation or organized resistance detected in project influence zone."
            })
            action_recommendations.append({
                "pillar": "Socio-Legal",
                "priority": "P0 - Immediate",
                "action": "Convene Multi-Stakeholder Conciliation Cell & District Grievance Redressal Committee",
                "expected_timeline_saved_days": 45,
                "statutory_pathway": "Establish formal hearing mechanism under District Collector / LARR Authority"
            })
        elif str(p_raw).strip().lower() in ["moderate", "medium"]:
            p_r = 0.5
        else:
            p_r = 0.0

        # 1.2 Compensation Discrepancy Ratio (C_r)
        c_offer = max(0.0, float(record.get("C_offer", record.get("compensation_offer", 0.0))))
        c_demand = max(0.0, float(record.get("C_demand", record.get("compensation_demand", 0.0))))
        mult_demand = float(record.get("compensation_multiplier_demand", record.get("multiplier", 1.0)))

        if c_offer > eps and c_demand > eps:
            c_r = max(0.0, min(1.0, (c_demand - c_offer) / c_offer))
        elif mult_demand > 1.0:
            c_r = max(0.0, min(1.0, (mult_demand - 1.0) / 2.0))
        else:
            c_r = 0.0

        if c_r > 0.40:
            statutory_flags.append({
                "code": "SL-02-COMP-DISPUTE",
                "severity": "HIGH",
                "title": f"Substantial Compensation Expectation Gap ({(c_r * 100):.1f}%)",
                "act_reference": "RFCTLARR Act 2013 First Schedule (Market Value Multiplier 1x - 2x + 100% Solatium)",
                "description": "Landowners demanding compensation significantly exceeding benchmark circle rate."
            })

        # 1.3 Affected Families / Displacement Magnitude (F_r)
        aff_fam = max(0.0, float(record.get("Affected_Families_Count", record.get("affected_families_count", record.get("affected_families", 0.0)))))
        f_r = min(1.0, math.log10(aff_fam + 1.0) / self.log_fmax_plus1)

        if aff_fam >= 500:
            statutory_flags.append({
                "code": "SL-03-LARGE-DISPLACEMENT",
                "severity": "HIGH",
                "title": f"Large-Scale Human Displacement ({int(aff_fam):,} PAPs)",
                "act_reference": "RFCTLARR Act 2013 Chapter V (Resettlement & Rehabilitation Scheme)",
                "description": "Project triggers mandatory appointment of Administrator for R&R under Section 43."
            })

        # 1.4 Title Dispute Rate (T_r)
        t_rate = max(0.0, min(100.0, float(record.get("Title_Dispute_Rate_Percent", record.get("title_dispute_rate_percent", 0.0)))))
        t_r = t_rate / 100.0
        if t_rate >= 15.0:
            statutory_flags.append({
                "code": "SL-04-TITLE-ENCUMBRANCE",
                "severity": "CRITICAL" if t_rate >= 25.0 else "HIGH",
                "title": f"Elevated Land Title & Succession Dispute Rate ({t_rate:.1f}%)",
                "act_reference": "RFCTLARR Act 2013 Section 64 (Reference to Land Acquisition Authority)",
                "description": "High rate of defective land titles risks revenue court injunctions and disbursement freeze."
            })
            action_recommendations.append({
                "pillar": "Socio-Legal",
                "priority": "P1 - High",
                "action": "Deploy Special Revenue Lok Adalats & Drone Cadastral Resurvey for Fast-Track Mutation",
                "expected_timeline_saved_days": 60,
                "statutory_pathway": "Summary adjudication of heirship disputes under Special Tahsildar powers"
            })

        # Sub-Index 1: R_SL
        r_sl = (0.35 * p_r) + (0.25 * c_r) + (0.20 * t_r) + (0.20 * f_r)

        # =====================================================================
        # PILLAR II: STATUTORY & ENVIRONMENTAL CLEARANCE RISK (R_EC)
        # =====================================================================
        def parse_statutory_clearance(val: Any) -> Tuple[float, str]:
            if val is None or pd.isna(val):
                return 0.50, "Unknown / In Review"
            s = str(val).strip().lower().replace(" ", "_").replace("-", "_")
            if s in ["approved", "granted", "cleared", "stage_2", "exempted", "not_required", "n/a", "stage_2_final"]:
                return 0.0, "Cleared / Not Required"
            elif s in ["stage_1", "stage_1_approved"]:
                return 0.35, "Stage-1 In-Principle Granted"
            elif s in ["in_progress", "in_review", "applied"]:
                return 0.45, "Under Statutory Appraisal"
            elif s in ["pending", "stage_1_pending", "pending_le_6"]:
                return 0.70, "Pending Formal Examination"
            elif s in ["rejected", "denied", "cancelled", "pending_gt_6"]:
                return 1.00, "Statutory Rejection / Severe Block"
            return 0.50, "Intermediate Review"

        sia_raw_val = record.get("SIA_Approval_Status", record.get("sia_approval_status", "Approved"))
        sia_r, sia_label = parse_statutory_clearance(sia_raw_val)
        
        if sia_r >= 0.90:
            statutory_flags.append({
                "code": "EC-01-SIA-REJECTED",
                "severity": "CRITICAL",
                "title": "Social Impact Assessment (SIA) Rejected by Expert Group",
                "act_reference": "RFCTLARR Act 2013 Section 7(4) & Section 8",
                "description": "Expert Committee evaluated that social costs exceed benefits. Acquisition is legally halted."
            })
            action_recommendations.append({
                "pillar": "Environmental & Statutory",
                "priority": "P0 - Immediate",
                "action": "Revise Alignment to Minimize Habitation & File Urgent Cabinet Reconsideration Appeal",
                "expected_timeline_saved_days": 120,
                "statutory_pathway": "State Government recording of special reasons in writing under Section 8(2)"
            })
        elif sia_r >= 0.40:
            statutory_flags.append({
                "code": "EC-02-SIA-IN-PROGRESS",
                "severity": "MEDIUM",
                "title": f"SIA Review Active ({sia_label})",
                "act_reference": "RFCTLARR Act 2013 Section 4(2) (6-Month Statutory Completion Mandate)",
                "description": "SIA study and public hearing proceedings ongoing."
            })

        fc_raw_val = record.get("Forest_Clearance_Status", record.get("forest_clearance_status", "Approved"))
        fc_r, fc_label = parse_statutory_clearance(fc_raw_val)

        if fc_r >= 0.90:
            statutory_flags.append({
                "code": "EC-03-FOREST-REJECTED",
                "severity": "CRITICAL",
                "title": "Forest Divergence Proposal Rejected by MoEFCC / FAC",
                "act_reference": "Forest (Conservation) Act, 1980 / Van Adhiniyam 2023",
                "description": "Forest land diversion denied. Tree felling and physical access strictly barred under criminal penalties."
            })
        elif fc_r >= 0.35:
            statutory_flags.append({
                "code": "EC-04-FOREST-STAGE1",
                "severity": "HIGH" if fc_r >= 0.7 else "MEDIUM",
                "title": f"Forest Clearance Pending Stage-II ({fc_label})",
                "act_reference": "MoEFCC Guidelines on Forest Land Divergence",
                "description": "Requires deposition of Net Present Value (NPV) & transfer of Non-Forest Land for CA."
            })

        # 2.3 Terrain & Geotechnical Difficulty (T_m)
        terrain_str = str(record.get("Terrain_Type", record.get("terrain_type", "default"))).strip().lower()
        t_m = self.TERRAIN_WEIGHTS.get(terrain_str, self.TERRAIN_WEIGHTS["default"])
        
        # 2.4 Weather / Eco-Vulnerability (W_r)
        w_idx = max(0.0, min(10.0, float(record.get("Weather_Index", record.get("weather_index", record.get("W_r", 0.5) * 10.0)))))
        w_r = w_idx / 10.0

        # Sub-Index 2: R_EC
        r_ec = (0.35 * fc_r) + (0.30 * sia_r) + (0.25 * t_m) + (0.10 * w_r)

        # =====================================================================
        # PILLAR III: FINANCIAL & DISBURSEMENT RISK (R_FD)
        # =====================================================================
        f_disb = max(0.0, min(100.0, float(record.get("Fund_Disbursement_Percent", record.get("fund_disbursement_percent", 100.0)))))
        f_dr = 1.0 - (f_disb / 100.0)

        capex_crore = max(1.0, float(record.get("Estimated_Cost_INR_Crore", record.get("estimated_cost_inr_crore", 1000.0))))
        land_area_ha = max(1.0, float(record.get("Land_Area_Hectares", record.get("land_area_hectares", 100.0))))
        
        cost_density_crore_per_ha = capex_crore / land_area_ha
        density_risk = min(1.0, cost_density_crore_per_ha / 15.0)

        if f_disb <= 15.0:
            statutory_flags.append({
                "code": "FD-01-DISBURSEMENT-DEFICIT",
                "severity": "CRITICAL",
                "title": f"Severe Compensation Disbursement Liquidity Deficit ({f_disb:.1f}% disbursed)",
                "act_reference": "RFCTLARR Act 2013 Section 77 & 80 (Compulsory 9% to 15% Annual Interest Penalty)",
                "description": "Delayed award deposit triggers compounding statutory interest liabilities against the project proponent."
            })
            action_recommendations.append({
                "pillar": "Financial",
                "priority": "P0 - Immediate",
                "action": "Establish Escrow Direct Benefit Transfer (DBT) Liquidity Pool with State Treasury",
                "expected_timeline_saved_days": 75,
                "statutory_pathway": "Direct electronic compensation disbursement to verified landowner Aadhaar-linked accounts"
            })

        # Sub-Index 3: R_FD
        r_fd = (0.70 * f_dr) + (0.30 * density_risk)

        # =====================================================================
        # PILLAR IV: ADMINISTRATIVE WORKFLOW & TIMELINE DRIFT (R_AW)
        # =====================================================================
        days_elapsed = max(0.0, float(record.get("Section_11_Notification_Days", record.get("section_11_notification_days", 0.0))))
        
        # Section 11 to Section 19 Lapse Clock Analysis
        n_r = max(0.0, min(1.0, (days_elapsed - self.n_base) / max(eps, self.n_limit - self.n_base)))
        days_to_lapse = max(0.0, self.n_limit - days_elapsed)
        is_lapsed = days_elapsed >= self.n_limit

        if is_lapsed:
            statutory_flags.append({
                "code": "AW-01-SEC11-LAPSED",
                "severity": "CRITICAL",
                "title": "SECTION 11 PRELIMINARY NOTIFICATION STATUTORILY LAPSED",
                "act_reference": "RFCTLARR Act 2013 Section 14 (Mandatory Lapse of Preliminary Notification)",
                "description": f"Notification elapsed {int(days_elapsed)} days (> 365 days). Proceedings are void ab initio; fresh notification mandatory."
            })
            action_recommendations.append({
                "pillar": "Administrative",
                "priority": "P0 - Critical",
                "action": "File Emergency Request for Extension under Section 14 Proviso or Issue Re-Notification",
                "expected_timeline_saved_days": 180,
                "statutory_pathway": "State Government executive notification of extension under special circumstances proviso"
            })
        elif days_elapsed >= 270:
            statutory_flags.append({
                "code": "AW-02-LAPSE-IMMINENT",
                "severity": "HIGH",
                "title": f"Section 11 Lapse Imminent ({int(days_to_lapse)} Days Remaining)",
                "act_reference": "RFCTLARR Act 2013 Section 14 Deadline Approaching",
                "description": "Less than 90 days remaining to issue Section 19 Final Declaration before total procedural lapse."
            })

        # 4.2 Historical Empirical Prior (H_r)
        st = str(record.get("State", record.get("state", "")))
        pr = str(record.get("Project_Type", record.get("project_type", "")))
        h_delay = self.historical_stats.get((st, pr), self.state_delay_stats.get(st, self.global_mean_delay))
        h_r = max(0.0, min(1.0, h_delay / max(eps, self.d_max)))

        st_eff = self.STATE_INSTITUTIONAL_EFFICIENCY.get(st, self.STATE_INSTITUTIONAL_EFFICIENCY["default"])
        h_r_adjusted = min(1.0, h_r * st_eff)

        proj_age = max(0.0, float(record.get("Project_Age_Years", record.get("project_age_years", 1.0))))
        age_r = min(1.0, proj_age / 8.0)

        # Sub-Index 4: R_AW
        r_aw = (0.50 * n_r) + (0.30 * h_r_adjusted) + (0.20 * age_r)

        # =====================================================================
        # FINAL COMPOSITE RISK SCORE (CRS) & METRICS
        # =====================================================================
        crs_raw = ((0.30 * r_sl) + (0.25 * r_ec) + (0.25 * r_fd) + (0.20 * r_aw)) * 100.0
        crs = round(max(0.0, min(100.0, crs_raw)), 2)

        if crs < 30.0:
            risk_band = "Low Risk"
            risk_tier_code = "LOW"
            color_hex = "#10b981"
        elif crs < 55.0:
            risk_band = "Medium Risk"
            risk_tier_code = "MEDIUM"
            color_hex = "#f59e0b"
        elif crs < 75.0:
            risk_band = "High Risk"
            risk_tier_code = "HIGH"
            color_hex = "#f43f5e"
        else:
            risk_band = "Critical / Severe Risk"
            risk_tier_code = "CRITICAL"
            color_hex = "#e11d48"

        predicted_drift_days = int(round(max(15.0, (crs / 100.0) * 750.0 + (h_delay * 0.25) + (is_lapsed * 180.0))))
        predicted_drift_months = round(predicted_drift_days / 30.0, 1)

        daily_capex_burn_crore = (capex_crore * self.cost_of_capital_annual) / 365.0
        total_cost_escalation_crore = round(daily_capex_burn_crore * predicted_drift_days, 2)

        contributions = {
            "Socio-Legal & Resettlement (R_SL)": round(0.30 * r_sl * 100.0, 2),
            "Environmental & Statutory Clearance (R_EC)": round(0.25 * r_ec * 100.0, 2),
            "Financial Liquidity & Disbursement (R_FD)": round(0.25 * r_fd * 100.0, 2),
            "Administrative & Milestone Float (R_AW)": round(0.20 * r_aw * 100.0, 2),
        }

        detailed_sub_factors = {
            "P_r_local_protests": round(p_r, 4),
            "C_r_compensation_demand_ratio": round(c_r, 4),
            "F_r_population_displacement_burden": round(f_r, 4),
            "T_r_title_dispute_rate": round(t_r, 4),
            "SIA_r_social_impact_status": round(sia_r, 4),
            "FC_r_forest_clearance_status": round(fc_r, 4),
            "T_m_terrain_topographic_multiplier": round(t_m, 4),
            "W_r_weather_vulnerability_index": round(w_r, 4),
            "F_dr_fund_disbursement_deficit": round(f_dr, 4),
            "N_r_section11_statutory_lapse_progress": round(n_r, 4),
            "H_r_state_sector_historical_delay": round(h_r_adjusted, 4),
            "A_r_project_age_drift": round(age_r, 4)
        }

        dominant_driver = max(contributions.items(), key=lambda x: x[1])[0]

        return {
            "CRS": crs,
            "Risk_Band": risk_band,
            "Risk_Tier": risk_tier_code,
            "Color_Hex": color_hex,
            "Dominant_Risk_Driver": dominant_driver,
            "Predicted_Schedule_Drift_Days": predicted_drift_days,
            "Predicted_Schedule_Drift_Months": predicted_drift_months,
            "Estimated_Financial_Escalation_Crore": total_cost_escalation_crore,
            "Days_Until_Section_11_Lapse": int(days_to_lapse),
            "LARR_Section11_Lapse_Warning": is_lapsed,
            "LARR_Sec11_Lapse_Warning": is_lapsed,
            "Statutory_Audit_Flags": statutory_flags,
            "Prescriptive_Mitigation_Actions": action_recommendations,
            "Sub_Indices": {
                "R_SL": round(r_sl, 4),
                "R_EC": round(r_ec, 4),
                "R_FD": round(r_fd, 4),
                "R_AW": round(r_aw, 4),
                # Backwards-compatible aliases
                "R_SP": round(r_sl, 4),
                "R_LR": round(r_ec, 4),
                "R_EG": round(r_ec, 4),
                "R_FA": round(r_fd, 4)
            },
            "Contributions": contributions,
            "Detailed_Sub_Factors": detailed_sub_factors
        }

    def score_dataframe_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        High-Throughput Vectorized SIMD Batch Execution.
        Evaluates thousands of infrastructure projects simultaneously with complete
        numerical safety, NaN interpolation, and multi-pillar metric extraction.
        """
        n, eps = len(df), 1e-6
        if n == 0:
            return df.copy()

        # 1. Vectorized Socio-Legal
        if "Local_Protest_Flag" in df.columns:
            p_series = df["Local_Protest_Flag"]
        elif "local_protest_flag" in df.columns:
            p_series = df["local_protest_flag"]
        else:
            p_series = pd.Series([False] * n)
        
        p_r = (p_series.astype(float).values if p_series.dtype in [bool, np.bool_] 
               else np.where(p_series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "high", "active"]), 1.0, 0.0))

        c_offer = pd.to_numeric(df["C_offer"], errors="coerce").fillna(1.0).to_numpy(dtype=float) if "C_offer" in df.columns else (
            pd.to_numeric(df["compensation_offer"], errors="coerce").fillna(1.0).to_numpy(dtype=float) if "compensation_offer" in df.columns else np.ones(n, dtype=float)
        )
        c_demand = pd.to_numeric(df["C_demand"], errors="coerce").fillna(1.0).to_numpy(dtype=float) if "C_demand" in df.columns else (
            pd.to_numeric(df["compensation_demand"], errors="coerce").fillna(1.0).to_numpy(dtype=float) if "compensation_demand" in df.columns else np.ones(n, dtype=float)
        )
        c_r = np.clip((c_demand - c_offer) / np.where(c_offer <= eps, eps, c_offer), 0.0, 1.0)
        c_r[(c_offer <= eps) & (c_demand <= eps)] = 0.0

        aff_fam_s = df["Affected_Families_Count"] if "Affected_Families_Count" in df.columns else (
            df["affected_families_count"] if "affected_families_count" in df.columns else (
                df["affected_families"] if "affected_families" in df.columns else pd.Series([0] * n)
            )
        )
        aff_fam = np.maximum(0.0, pd.to_numeric(aff_fam_s, errors="coerce").fillna(0).to_numpy(dtype=float))
        f_r = np.clip(np.log10(aff_fam + 1.0) / self.log_fmax_plus1, 0.0, 1.0)

        t_rate_s = df["Title_Dispute_Rate_Percent"] if "Title_Dispute_Rate_Percent" in df.columns else (
            df["title_dispute_rate_percent"] if "title_dispute_rate_percent" in df.columns else pd.Series([0] * n)
        )
        t_rate = np.clip(pd.to_numeric(t_rate_s, errors="coerce").fillna(0).to_numpy(dtype=float) / 100.0, 0.0, 1.0)
        r_sl = (0.35 * p_r) + (0.25 * c_r) + (0.20 * t_rate) + (0.20 * f_r)

        # 2. Vectorized Environmental & Statutory
        def parse_vec_clearance(col_candidates: List[str]) -> np.ndarray:
            for col in col_candidates:
                if col in df.columns:
                    s = df[col].astype(str).str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
                    arr = np.full(len(s), 0.5, dtype=float)
                    arr[s.isin(["approved", "granted", "cleared", "stage_2", "exempted", "not_required", "n/a", "stage_2_final"])] = 0.0
                    arr[s.isin(["stage_1", "stage_1_approved"])] = 0.35
                    arr[s.isin(["in_progress", "in_review", "applied"])] = 0.45
                    arr[s.isin(["pending", "stage_1_pending"])] = 0.70
                    arr[s.str.contains("rejected|denied|cancelled|pending_gt_6", regex=True)] = 1.0
                    return arr
            return np.full(n, 0.5, dtype=float)

        sia_r = parse_vec_clearance(["SIA_Approval_Status", "sia_approval_status"])
        fc_r = parse_vec_clearance(["Forest_Clearance_Status", "forest_clearance_status"])

        terrain_s = df["Terrain_Type"] if "Terrain_Type" in df.columns else (
            df["terrain_type"] if "terrain_type" in df.columns else pd.Series(["default"] * n)
        )
        t_m = terrain_s.astype(str).str.strip().str.lower().map(self.TERRAIN_WEIGHTS).fillna(self.TERRAIN_WEIGHTS["default"]).to_numpy(dtype=float)
        
        w_idx_s = df["Weather_Index"] if "Weather_Index" in df.columns else (
            df["weather_index"] if "weather_index" in df.columns else (
                df["W_r"] * 10.0 if "W_r" in df.columns else pd.Series([1.0] * n)
            )
        )
        w_r = np.clip(pd.to_numeric(w_idx_s, errors="coerce").fillna(1.0).to_numpy(dtype=float) / 10.0, 0.0, 1.0)
        r_ec = (0.35 * fc_r) + (0.30 * sia_r) + (0.25 * t_m) + (0.10 * w_r)

        # 3. Vectorized Financial
        f_disb_s = df["Fund_Disbursement_Percent"] if "Fund_Disbursement_Percent" in df.columns else (
            df["fund_disbursement_percent"] if "fund_disbursement_percent" in df.columns else pd.Series([100.0] * n)
        )
        f_disb = pd.to_numeric(f_disb_s, errors="coerce").fillna(100.0).to_numpy(dtype=float)
        f_dr = np.clip(1.0 - (f_disb / 100.0), 0.0, 1.0)

        capex_s = df["Estimated_Cost_INR_Crore"] if "Estimated_Cost_INR_Crore" in df.columns else (
            df["estimated_cost_inr_crore"] if "estimated_cost_inr_crore" in df.columns else pd.Series([1000.0] * n)
        )
        capex = pd.to_numeric(capex_s, errors="coerce").fillna(1000.0).to_numpy(dtype=float)

        land_ha_s = df["Land_Area_Hectares"] if "Land_Area_Hectares" in df.columns else (
            df["land_area_hectares"] if "land_area_hectares" in df.columns else pd.Series([100.0] * n)
        )
        land_ha = np.maximum(1.0, pd.to_numeric(land_ha_s, errors="coerce").fillna(100.0).to_numpy(dtype=float))
        density_r = np.clip((capex / land_ha) / 15.0, 0.0, 1.0)
        r_fd = (0.70 * f_dr) + (0.30 * density_r)

        # 4. Vectorized Administrative
        days_s = df["Section_11_Notification_Days"] if "Section_11_Notification_Days" in df.columns else (
            df["section_11_notification_days"] if "section_11_notification_days" in df.columns else pd.Series([0] * n)
        )
        days = np.maximum(0.0, pd.to_numeric(days_s, errors="coerce").fillna(0).to_numpy(dtype=float))
        n_r = np.clip((days - self.n_base) / max(eps, self.n_limit - self.n_base), 0.0, 1.0)

        states = (df["State"] if "State" in df.columns else (df["state"] if "state" in df.columns else pd.Series([""] * n))).astype(str).values
        projs = (df["Project_Type"] if "Project_Type" in df.columns else (df["project_type"] if "project_type" in df.columns else pd.Series([""] * n))).astype(str).values
        h_delays = np.array([self.historical_stats.get((s, p), self.state_delay_stats.get(s, self.global_mean_delay)) for s, p in zip(states, projs)], dtype=float)
        h_r = np.clip(h_delays / max(eps, self.d_max), 0.0, 1.0)
        st_eff = np.array([self.STATE_INSTITUTIONAL_EFFICIENCY.get(s, self.STATE_INSTITUTIONAL_EFFICIENCY["default"]) for s in states], dtype=float)
        h_r_adj = np.clip(h_r * st_eff, 0.0, 1.0)
        r_aw = (0.60 * n_r) + (0.40 * h_r_adj)

        # Composite Risk Score (CRS)
        crs = np.clip(((0.30 * r_sl) + (0.25 * r_ec) + (0.25 * r_fd) + (0.20 * r_aw)) * 100.0, 0.0, 100.0)

        res_df = df.copy()
        res_df["R_SL"] = np.round(r_sl, 4)
        res_df["R_EC"] = np.round(r_ec, 4)
        res_df["R_FD"] = np.round(r_fd, 4)
        res_df["R_AW"] = np.round(r_aw, 4)
        # Backwards compatible alias columns
        res_df["R_SP"] = res_df["R_SL"]
        res_df["R_LR"] = res_df["R_EC"]
        res_df["R_EG"] = res_df["R_EC"]
        res_df["R_FA"] = res_df["R_FD"]
        
        res_df["CRS"] = np.round(crs, 2)
        res_df["Risk_Band"] = np.select(
            [crs < 30.0, crs < 55.0, crs < 75.0],
            ["Low Risk", "Medium Risk", "High Risk"],
            default="Critical / Severe Risk"
        )
        res_df["Predicted_Delay_Drift_Days"] = np.round(np.maximum(15.0, (crs / 100.0) * 750.0 + (h_delays * 0.25))).astype(int)
        res_df["LARR_Sec11_Lapse_Warning"] = days >= self.n_limit
        res_df["LARR_Section11_Lapse_Warning"] = res_df["LARR_Sec11_Lapse_Warning"]
        return res_df
