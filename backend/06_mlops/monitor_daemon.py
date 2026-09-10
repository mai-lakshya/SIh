import os
import time
import schedule
import pandas as pd
import logging
from risk_analysis_system import RiskAnalysisSystem
from monitor import ModelMonitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_monitoring_cycle():
    logging.info("Starting monitoring cycle...")
    try:
        system = RiskAnalysisSystem(
            pipeline_path='pipeline.joblib',
            ensemble_path='ensemble.joblib',
            timeline_path='timeline.joblib'
        )
        monitor = ModelMonitor()
        
        # In a real environment, this would pull new ground truth data from a DB
        # Here we simulate using a random subset of the historical data and adding noise
        dataset_path = 'indian_infrastructure_projects_dataset.csv'
        if not os.path.exists(dataset_path) and os.path.exists('Revolution-main/indian_infrastructure_projects_dataset.csv'):
            dataset_path = 'Revolution-main/indian_infrastructure_projects_dataset.csv'
        df = pd.read_csv(dataset_path)
        ref_df = df.sample(1000, random_state=42)
        
        # Simulate new data arriving (with slight drift)
        new_df = df.sample(200).copy()
        new_df['estimated_cost_inr_crore'] *= 1.5 # Introduce drift
        
        # Evaluate Drift
        cont_cols = ['estimated_cost_inr_crore', 'land_area_hectares']
        cat_cols = ['state', 'project_type']
        
        logging.info("Checking data drift...")
        monitor.detect_data_drift(new_df, ref_df, cat_cols=cat_cols, cont_cols=cont_cols)
        
        # Evaluate Performance (Mocking actuals for demonstration)
        logging.info("Checking model performance...")
        # We need predictions on the new_df
        # Simulate true labels
        y_cls_true = new_df['delay_binary_label'] if 'delay_binary_label' in new_df else [1]*200
        y_crs_true = new_df['CRS'] if 'CRS' in new_df else [50.0]*200
        y_days_true = new_df['Actual_Delay_Days'] if 'Actual_Delay_Days' in new_df else [100.0]*200
        
        X_proc = system.pipeline.transform(new_df)
        preds = system.hybrid_model.predict(X_proc)
        
        monitor.check_performance(
            y_cls_true=y_cls_true,
            y_prob=preds['delay_probability'],
            y_crs_true=y_crs_true,
            crs_pred=preds['crs'],
            y_days_true=y_days_true,
            days_pred=preds['delay_days']
        )
        
        logging.info("Monitoring cycle completed.")
        
    except Exception as e:
        logging.error(f"Monitoring cycle failed: {e}")

if __name__ == "__main__":
    logging.info("Monitor Daemon Started. Running every 60 minutes...")
    schedule.every(60).minutes.do(run_monitoring_cycle)
    
    # Run once immediately
    run_monitoring_cycle()
    
    while True:
        schedule.run_pending()
        time.sleep(1)
