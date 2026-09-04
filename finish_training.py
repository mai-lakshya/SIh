import os
import pandas as pd
import optuna
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from pipeline import get_preprocessing_pipeline
from hybrid_model import HybridRiskPredictor
from timeline_predictor import NonLinearTimelinePredictor

if __name__ == '__main__':
    print("EMERGENCY PROTOCOL: Extracting best model from unfinished study...")
    
    # 1. Load the database and get the best trial
    study = optuna.load_study(study_name="god_mode_study", storage="sqlite:///god_mode_study.db")
    best_trial = study.best_trials[0]
    print(f"Best Trial found: {best_trial.number}")
    
    # 2. Reconstruct parameters
    cpu_params = {
        'xgb': {'tree_method': 'hist', 'n_jobs': -1},
        'lgb': {'n_jobs': -1},
        'cat': {'thread_count': -1},
        'et': {'n_jobs': -1}
    }
    
    xgb_params = {
        'n_estimators': best_trial.params['xgb_n_estimators'],
        'learning_rate': best_trial.params['xgb_lr'],
        'max_depth': best_trial.params['xgb_max_depth'],
        'reg_alpha': best_trial.params['xgb_reg_alpha'],
        'reg_lambda': best_trial.params['xgb_reg_lambda'],
        'early_stopping_rounds': 50,
        **cpu_params['xgb']
    }
    lgb_params = {
        'n_estimators': best_trial.params['lgb_n_estimators'],
        'learning_rate': best_trial.params['lgb_lr'],
        'max_depth': best_trial.params['lgb_max_depth'],
        'reg_alpha': best_trial.params['lgb_reg_alpha'],
        'reg_lambda': best_trial.params['lgb_reg_lambda'],
        'early_stopping_rounds': 50,
        'verbose': -1,
        **cpu_params['lgb']
    }
    cat_params = {
        'n_estimators': best_trial.params['cat_n_estimators'],
        'learning_rate': best_trial.params['cat_lr'],
        'depth': best_trial.params['cat_depth'],
        'l2_leaf_reg': best_trial.params['cat_l2'],
        'early_stopping_rounds': 50,
        **cpu_params['cat']
    }
    et_params = {
        'n_estimators': best_trial.params['et_n_estimators'],
        'max_depth': best_trial.params['et_max_depth'],
        'min_samples_split': best_trial.params['et_min_samples_split'],
        **cpu_params['et']
    }
    
    model_params = {
        'xgb': xgb_params,
        'lgb': lgb_params,
        'cat': cat_params,
        'et': et_params
    }
    
    rsf_params = {
        'n_estimators': best_trial.params.get('rsf_n_estimators', 100),
        'min_samples_split': best_trial.params.get('rsf_min_samples_split', 10)
    }
    
    deepsurv_params = {
        'hidden_layers': best_trial.params.get('ds_hidden_layers', 2),
        'dropout_p': best_trial.params.get('ds_dropout_p', 0.1),
        'lr': best_trial.params.get('ds_lr', 1e-3)
    }
    
    # 3. Load Data
    print("Loading data and running full fit...")
    df = pd.read_csv('indian_infrastructure_projects_dataset.csv')
    X = df.drop(columns=['delay_binary_label', 'section_11_notification_days', 'CRS', 'project_index', 'Actual_Delay_Days', 'delay_risk_tier', 'CRS_tier'], errors='ignore')
    y_cls = df['delay_binary_label']
    y_crs = df['CRS']
    y_days = df['section_11_notification_days']
    
    pipeline_final = get_preprocessing_pipeline(use_smote=best_trial.params['use_smote'])
    X_proc = pipeline_final.fit_transform(X, y_cls)
    
    final_hybrid = HybridRiskPredictor(random_state=42, model_params=model_params)
    final_hybrid.fit(X_proc, y_cls, y_crs, y_days)
    
    final_tl = NonLinearTimelinePredictor(random_state=42, rsf_params=rsf_params, deepsurv_params=deepsurv_params)
    final_tl.fit(X_proc, y_cls, y_days)
    
    # 4. Save
    print("Saving models...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_hybrid, "ensemble.joblib")
    joblib.dump(pipeline_final, "pipeline.joblib")
    final_tl.save("timeline.joblib")
    if hasattr(final_tl, 'rsf') and final_tl.rsf is not None:
        joblib.dump(final_tl.rsf, "rsf_only.joblib")
    
    # 5. Extract SHAP
    print("Extracting SHAP values...")
    try:
        xgb_wrapper = dict(final_hybrid.classifier.estimators_)['xgb']
        xgb_model = xgb_wrapper.model
        if not isinstance(X_proc, pd.DataFrame):
            X_proc_df = pd.DataFrame(X_proc)
        else:
            X_proc_df = X_proc
            
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_proc_df)
        
        plt.figure()
        shap.summary_plot(shap_values, X_proc_df, show=False)
        plt.savefig('models/shap_summary_plot.png', bbox_inches='tight')
        plt.close()
        
        vals = np.abs(shap_values).mean(0)
        shap_df = pd.DataFrame(list(zip(X_proc_df.columns, vals)), columns=['Feature', 'SHAP_Importance'])
        shap_df.sort_values(by=['SHAP_Importance'], ascending=False, inplace=True)
        shap_df.to_csv('models/shap_feature_importance.csv', index=False)
        print("Done! SHAP values saved.")
    except Exception as e:
        print(f"Error during SHAP: {e}")
