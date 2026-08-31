import pandas as pd
import joblib
from pipeline import get_preprocessing_pipeline
from timeline_predictor import NonLinearTimelinePredictor

def main():
    print("Loading data...")
    df = pd.read_csv('Revolution-main/indian_infrastructure_projects_dataset.csv')
    X = df.drop(columns=['delay_binary_label', 'Actual_Delay_Days', 'CRS', 'project_index'], errors='ignore')
    
    y_time = df.get('Actual_Delay_Days', df['delay_binary_label'] * 90).replace(0, 365)
    y_event = df['delay_binary_label']
    
    pipeline = joblib.load('pipeline.joblib')
    X_tf = pipeline.transform(X)
    
    print("Training CoxPH Timeline Predictor...")
    timeline = NonLinearTimelinePredictor()
    timeline.train_survival_model(X_tf, y_time, y_event)
    
    print("Saving timeline.joblib...")
    joblib.dump(timeline, 'timeline.joblib')
    print("Done!")

if __name__ == '__main__':
    main()
