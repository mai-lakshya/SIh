import pytest
import pandas as pd
from algorithm import RiskEngine

def test_risk_engine_timeline_analysis():
    engine = RiskEngine()
    
    # 1. Normal project with moderate delay
    record = {
        "State": "Maharashtra",
        "Project_Type": "Highways",
        "Section_11_Notification_Days": 120,
        "SIA_Approval_Status": "Approved",
        "Forest_Clearance_Status": "Stage 1 Approved",
        "Title_Dispute_Rate_Percent": 10.0,
        "Local_Protest_Flag": False,
        "C_offer": 50.0,
        "C_demand": 60.0,
        "Affected_Families_Count": 100,
        "Terrain_Type": "Plain",
        "Weather_Index": 3.0,
        "Fund_Disbursement_Percent": 80.0
    }
    
    res = engine.score_single(record)
    assert "Timeline_Analysis" in res
    tl = res["Timeline_Analysis"]
    
    assert tl["Expected_Delay_Days"] > 0
    assert tl["Expected_Delay_Months"] > 0
    assert tl["Confidence_Interval"]["P10_Optimistic_Days"] < tl["Confidence_Interval"]["P50_Expected_Days"]
    assert tl["Confidence_Interval"]["P50_Expected_Days"] < tl["Confidence_Interval"]["P90_Pessimistic_Days"]
    assert "Timeline_Risk_Phase" in tl
    assert "Critical_Path_Bottleneck" in tl
    assert "Section_11_Lapse_Clock" in tl
    assert len(tl["Milestones_Breakdown"]) >= 5

def test_risk_engine_timeline_sec11_lapse_warning():
    engine = RiskEngine(n_limit=365.0)
    
    # Project past the 365 statutory day lapse cliff
    record = {
        "Section_11_Notification_Days": 380,
        "SIA_Approval_Status": "Rejected",
        "Forest_Clearance_Status": "Pending",
        "Title_Dispute_Rate_Percent": 50.0,
        "Local_Protest_Flag": True,
        "C_offer": 50.0,
        "C_demand": 150.0
    }
    
    tl = engine.analyze_timeline(record)
    assert tl["Section_11_Lapse_Clock"]["Lapse_Triggered"] is True
    assert tl["Section_11_Lapse_Clock"]["Days_Remaining"] == 0
    assert tl["Timeline_Risk_Phase"] == "Long-Term Severe (> 180 Days)"

def test_risk_engine_vectorized_timeline():
    engine = RiskEngine()
    df = pd.DataFrame([
        {"Section_11_Notification_Days": 40, "SIA_Approval_Status": "Approved", "Forest_Clearance_Status": "Approved"},
        {"Section_11_Notification_Days": 200, "SIA_Approval_Status": "Pending > 6 Months", "Forest_Clearance_Status": "Rejected"}
    ])
    
    scored_df = engine.score_dataframe_vectorized(df)
    assert "Predicted_Delay_Days" in scored_df.columns
    assert "Predicted_Delay_Months" in scored_df.columns
    assert "Timeline_Risk_Phase" in scored_df.columns
    assert "Sec11_Days_Remaining" in scored_df.columns
    assert scored_df["Predicted_Delay_Days"].iloc[1] > scored_df["Predicted_Delay_Days"].iloc[0]
