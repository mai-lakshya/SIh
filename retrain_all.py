import pandas as pd
import joblib
from pipeline import get_preprocessing_pipeline
from hybrid_model import HybridRiskPredictor
from timeline_predictor import NonLinearTimelinePredictor

def main():
    print("Loading data...")
    df = pd.read_csv('indian_infrastructure_projects_dataset.csv')
    X = df.drop(columns=['delay_binary_label', 'Actual_Delay_Days', 'CRS', 'project_index'], errors='ignore')
    y_binary = df['delay_binary_label']
    
    print("Rebuilding and fitting pipeline...")
    pipeline = get_preprocessing_pipeline()
    pipeline.fit(X, y_binary)
    joblib.dump(pipeline, 'pipeline.joblib')
    
    print("Transforming data for models...")
    X_tf = pipeline.transform(X)
    
    print("Training Ensemble...")
    y_train = pd.DataFrame({
        'delay_binary': y_binary,
        'CRS': df.get('CRS', y_binary * 100),
        'delay_days': df.get('Actual_Delay_Days', y_binary * 90)
    })
    
    best_xgb = {'n_estimators': 125, 'max_depth': 8, 'learning_rate': 0.0935}
    best_lgb = {'n_estimators': 126, 'num_leaves': 44, 'learning_rate': 0.1336}
    best_rf = {'n_estimators': 107, 'max_depth': 12}
    best_gb = {'n_estimators': 173, 'max_depth': 3, 'learning_rate': 0.0383}
    
    predictor = HybridRiskPredictor()
    predictor.train_ensemble(X_tf, y_train, xgb_params=best_xgb, lgb_params=best_lgb, rf_params=best_rf, gb_params=best_gb)
    predictor.save('ensemble.joblib')
    
    print("Training Timeline Predictor...")
    # NonLinearTimelinePredictor expects Actual_Delay_Days and delay_binary_label
    y_time = df.get('Actual_Delay_Days', df['delay_binary_label'] * 90).replace(0, 365)
    
    timeline = NonLinearTimelinePredictor()
    timeline.train_survival_model(X_tf, y_time, y_binary)
    joblib.dump(timeline, 'timeline.joblib')
    if hasattr(timeline, 'rsf') and timeline.rsf is not None:
        joblib.dump(timeline.rsf, 'rsf_only.joblib')
    
    print("Done retraining all artifacts!")

if __name__ == '__main__':
    main()
