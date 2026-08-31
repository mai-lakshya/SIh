import optuna
import sqlite3
import pandas as pd
import os

db_path = "god_mode_study.db"
log_path = "training_12hr.log"

try:
    conn = sqlite3.connect(db_path)
    studies = pd.read_sql("SELECT * FROM studies", conn)
    conn.close()
    
    if studies.empty:
        print("Waiting for first fold to finish initialization...")
    else:
        study_name = studies.iloc[-1]['study_name']
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"Current Fold/Study: {study_name.split('_')[2] if len(study_name.split('_'))>2 else study_name}")
        print(f"Completed Trials: {completed_trials}")
        
        pareto = study.best_trials
        if pareto:
            best_t = pareto[0]
            print(f"Best Recall: {best_t.values[0]:.4f}")
            print(f"Best ECE: {best_t.values[1]:.4f}")
            print(f"Best CRS RMSE: {best_t.values[2]:.2f}")
            print(f"Best C-Index: {best_t.values[3]:.4f}")
            
except Exception as e:
    print(f"Error checking DB: {e}")

try:
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            lines = f.readlines()
        print(f"Log length: {len(lines)} lines")
        if lines:
            print(f"Last log line: {lines[-1].strip()}")
except Exception as e:
    print(f"Error checking logs: {e}")
