import pandas as pd
import logging
import os
from risk_analysis_system import RiskAnalysisSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- User Inputs ---
user_state = "Telangana" 
user_land_area = 100.0   
user_project_type = "Highway" 
user_terrain = "Urban" 
user_cost = 1500.0  

def run_custom_analysis():
    logging.info('--- Running Interactive Test with RiskAnalysisSystem ---')
    
    if not os.path.exists('pipeline.joblib') or not os.path.exists('ensemble.joblib') or not os.path.exists('timeline.joblib'):
        logging.error("Missing artifact files. Please run evaluate_model.py to train and serialize the pipeline.")
        return
        
    logging.info('Loading RiskAnalysisSystem...')
    try:
        system = RiskAnalysisSystem(
            pipeline_path='pipeline.joblib',
            ensemble_path='ensemble.joblib',
            timeline_path='timeline.joblib'
        )
    except Exception as e:
        logging.error(f"Failed to load system artifacts: {e}")
        return

    # Create dummy dataframe 
    input_data = pd.DataFrame([{
        'state': user_state,
        'land_area_hectares': user_land_area,
        'project_type': user_project_type,
        'terrain_type': user_terrain,
        'estimated_cost_inr_crore': user_cost,
        'affected_families_count': 500,
        'title_dispute_rate_percent': 10.0,
        'local_protest_flag': False,
        'compensation_multiplier_demand': 1.5,
        'sia_approval_status': 'Pending',
        'section_11_notification_days': 30,
        'forest_clearance_status': 'Not_Required',
        'fund_disbursement_percent': 10.0,
        'project_age_years': 1,
        'C_r': 0.5,
        'F_r': 0.5,
        'H_r': 0.5,
        'W_r': 0.5,
        'P_r': 0.5
    }])
    
    logging.info(f"Generating prediction for {user_project_type} in {user_state}...")
    try:
        result = system.predict(input_data)
        
        print("\n=== SYSTEM PREDICTION OUTPUT ===")
        print(f"Delay Probability: {result['predictions']['delay_probability']:.2%}")
        print(f"Composite Risk Score (CRS): {result['predictions']['crs']:.1f} / 100")
        print(f"Predicted Delay: {result['predictions']['predicted_delay_days']:.0f} days")
        print(f"Risk Tier: {result['predictions']['risk_tier']}")
        print(f"\nPhase: {result['timeline']['risk_phase']}")
        print(f"Median Survival: {result['timeline']['median_survival_days']:.0f} days")
        
        print("\n=== TOP RECOMMENDATIONS ===")
        for i, rec in enumerate(result['recommendations'][:3], 1):
            print(f"{i}. [{rec['priority']}] {rec['actions'][0]}")
            
    except Exception as e:
        logging.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    run_custom_analysis()
