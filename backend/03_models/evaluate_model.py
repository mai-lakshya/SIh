import os
import time
import argparse
import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import recall_score, log_loss
import joblib
import shap
import matplotlib.pyplot as plt

# Local imports
from pipeline import get_preprocessing_pipeline
from hybrid_model import HybridRiskPredictor
from timeline_predictor import NonLinearTimelinePredictor

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) across n equal-width bins.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if np.sum(in_bin) > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)

def evaluate_nested_cv(X, y_cls, y_crs, y_days, n_outer_splits=5, n_inner_splits=3, n_trials=5):
    """
    Performs nested cross-validation with hyperparameter evaluation.
    """
    outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in outer_cv.split(X, y_cls):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_cls_tr, y_cls_te = y_cls.iloc[train_idx], y_cls.iloc[test_idx]
        y_crs_tr, y_crs_te = y_crs.iloc[train_idx], y_crs.iloc[test_idx]
        y_days_tr, y_days_te = y_days.iloc[train_idx], y_days.iloc[test_idx]
        
        pipeline = get_preprocessing_pipeline(use_smote=False)
        X_tr_proc = pipeline.fit_transform(X_tr, y_cls_tr)
        X_te_proc = pipeline.transform(X_te)
        
        model = HybridRiskPredictor(random_state=42)
        model.fit(X_tr_proc, y_cls_tr, y_crs_tr, y_days_tr)
        preds = model.predict(X_te_proc)
        
        recall = recall_score(y_cls_te, (preds['delay_probability'] >= 0.5).astype(int), zero_division=0)
        ece = expected_calibration_error(y_cls_te, preds['delay_probability'])
        scores.append({'recall': recall, 'ece': ece})
        
    return scores

def objective(trial, X, y_cls, y_crs, y_days, quick_check=False):
    use_smote = trial.suggest_categorical('use_smote', [True, False])
    
    # 1. Tree Ensembles Params
    cpu_params = {
        'xgb': {'tree_method': 'hist', 'n_jobs': -1},
        'lgb': {'n_jobs': -1},
        'cat': {'thread_count': -1},
        'et': {'n_jobs': -1}
    }

    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 500, 2000) if hasattr(trial, 'suggest_int') else 100,
        'learning_rate': trial.suggest_float('xgb_lr', 1e-3, 1e-1, log=True) if hasattr(trial, 'suggest_float') else 0.05,
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10) if hasattr(trial, 'suggest_int') else 4,
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True) if hasattr(trial, 'suggest_float') else 0.1,
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True) if hasattr(trial, 'suggest_float') else 1.0,
        'early_stopping_rounds': 50,
        **cpu_params['xgb']
    }
    
    lgb_params = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 500, 2000) if hasattr(trial, 'suggest_int') else 100,
        'learning_rate': trial.suggest_float('lgb_lr', 1e-3, 1e-1, log=True) if hasattr(trial, 'suggest_float') else 0.05,
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 10) if hasattr(trial, 'suggest_int') else 4,
        'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-8, 10.0, log=True) if hasattr(trial, 'suggest_float') else 0.1,
        'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-8, 10.0, log=True) if hasattr(trial, 'suggest_float') else 1.0,
        'early_stopping_rounds': 50,
        'verbose': -1,
        **cpu_params['lgb']
    }

    cat_params = {
        'n_estimators': trial.suggest_int('cat_n_estimators', 500, 2000) if hasattr(trial, 'suggest_int') else 100,
        'learning_rate': trial.suggest_float('cat_lr', 1e-3, 1e-1, log=True) if hasattr(trial, 'suggest_float') else 0.05,
        'depth': trial.suggest_int('cat_depth', 4, 10) if hasattr(trial, 'suggest_int') else 4,
        'l2_leaf_reg': trial.suggest_float('cat_l2', 1e-8, 10.0, log=True) if hasattr(trial, 'suggest_float') else 1.0,
        'early_stopping_rounds': 50,
        **cpu_params['cat']
    }
    
    et_params = {
        'n_estimators': trial.suggest_int('et_n_estimators', 100, 500) if hasattr(trial, 'suggest_int') else 50,
        'max_depth': trial.suggest_int('et_max_depth', 3, 15) if hasattr(trial, 'suggest_int') else 5,
        'min_samples_split': trial.suggest_int('et_min_samples_split', 2, 20) if hasattr(trial, 'suggest_int') else 2,
        **cpu_params['et']
    }
    
    model_params = {
        'xgb': xgb_params,
        'lgb': lgb_params,
        'cat': cat_params,
        'et': et_params
    }
    
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    recalls = []
    log_losses = []
    eces = []
    maes = []
    
    try:
        for train_idx, val_idx in inner_cv.split(X, y_cls):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_cls_tr, y_cls_val = y_cls.iloc[train_idx], y_cls.iloc[val_idx]
            y_crs_tr, y_crs_val = y_crs.iloc[train_idx], y_crs.iloc[val_idx]
            y_days_tr, y_days_val = y_days.iloc[train_idx], y_days.iloc[val_idx]
            
            # 1. Preprocessing
            pipeline = get_preprocessing_pipeline(use_smote=use_smote)
            X_tr_proc = pipeline.fit_transform(X_tr, y_cls_tr)
            X_val_proc = pipeline.transform(X_val)
            
            # 2. Hybrid Model
            model = HybridRiskPredictor(random_state=42, model_params=model_params)
            model.fit(X_tr_proc, y_cls_tr, y_crs_tr, y_days_tr)
            
            # 3. Predict & Evaluate Hybrid Model
            preds = model.predict(X_val_proc, blend_monotonicity=True)
            y_pred_cls = (preds['delay_probability'] >= 0.5).astype(int)
            y_prob = preds['delay_probability']
            
            recalls.append(recall_score(y_cls_val, y_pred_cls, zero_division=0))
            log_losses.append(log_loss(y_cls_val, y_prob, labels=[0, 1]))
            eces.append(expected_calibration_error(y_cls_val, y_prob))
            maes.append(float(np.mean(np.abs(preds['delay_days'] - y_days_val))))
            
            if quick_check:
                break
            
        return float(np.mean(recalls)), float(np.mean(eces)), float(np.mean(maes))
    except Exception as e:
        print(f"Trial failed: {e}")
        if hasattr(optuna, 'TrialPruned'):
            raise optuna.TrialPruned()
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=40, help='Number of optuna trials')
    parser.add_argument('--quick-check', action='store_true', help='Run minimal folds for testing')
    args = parser.parse_args()
    
    df = pd.read_csv('indian_infrastructure_projects_dataset.csv')
    if args.quick_check:
        df = df.sample(200)
        
    X = df.drop(columns=['delay_binary_label', 'section_11_notification_days', 'CRS', 'project_index', 'Actual_Delay_Days', 'delay_risk_tier', 'CRS_tier'], errors='ignore')
    
    cat_cols = ['state', 'district', 'project_type', 'terrain_type', 'sia_approval_status', 'forest_clearance_status']
    
    pipeline = get_preprocessing_pipeline(use_smote=False)
    X_tf = pipeline.fit_transform(X, df['delay_binary_label'])
    
    y_cls = df['delay_binary_label']
    y_crs = df['CRS']
    y_days = df['section_11_notification_days'] # we are predicting this
    n_trials = args.n_trials
    quick_check = args.quick_check

    mlflow.set_experiment("God_Mode_Evaluation")
    
    # --- PHASE 1: GLOBAL HYPERPARAMETER SEARCH ---
    print(f"Starting Phase 1: Global Optuna Search ({n_trials} trials)")
    sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=10, seed=42)
    
    storage = "sqlite:///god_mode_study.db"
    study_name = "god_mode_study"
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
        study_name=study_name
    )
    
    remaining_trials = max(0, n_trials - len(study.trials))
    print(f"Study currently has {len(study.trials)} trials. Running {remaining_trials} more.")
    if remaining_trials > 0:
        study.optimize(lambda trial: objective(trial, X_tf, y_cls, y_crs, y_days, quick_check), n_trials=remaining_trials, timeout=41400)
    
    # Select best trade-off from Pareto front
    best_trials = study.best_trials
    best_trial = best_trials[0]
    
    print("\nPhase 1 Complete. Best Hyperparameters found.")
    
    # --- PHASE 2: FINAL RETRAINING ON 100% DATA ---
    print("\nStarting Phase 2: Final Model Training on Entire Dataset")
    
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
    
    pipeline_final = get_preprocessing_pipeline(use_smote=best_trial.params['use_smote'])
    X_proc = pipeline_final.fit_transform(X, y_cls)
    
    final_hybrid = HybridRiskPredictor(random_state=42, model_params=model_params)
    final_hybrid.fit(X_proc, y_cls, y_crs, y_days)
    
    final_tl = NonLinearTimelinePredictor(random_state=42, rsf_params=rsf_params, deepsurv_params=deepsurv_params)
    final_tl.fit(X_proc, y_cls, y_days)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_hybrid, "ensemble.joblib")
    joblib.dump(pipeline_final, "pipeline.joblib")
    final_tl.save("timeline.joblib")
    
    print("\nPhase 3: Extracting SHAP values from Best Base Model (LightGBM)")
    # Get the fitted LightGBM model from the hybrid model's classifier
    try:
        lgb_wrapper = final_hybrid.classifier.named_estimators_['lgb']
        lgb_model = lgb_wrapper.model
        
        # Check if X_proc is DataFrame, else convert
        if not isinstance(X_proc, pd.DataFrame):
            try:
                feature_names = pipeline_final.named_steps['preprocessor'].get_feature_names_out()
            except:
                feature_names = [f"Feature_{i}" for i in range(X_proc.shape[1])]
            X_proc_df = pd.DataFrame(X_proc, columns=feature_names)
        else:
            X_proc_df = X_proc
            
        def clean_val(x):
            if isinstance(x, str):
                x = x.replace('[', '').replace(']', '')
                try: return float(x)
                except: return 0.0
            elif isinstance(x, (list, np.ndarray)):
                return float(x[0]) if len(x) > 0 else 0.0
            try: return float(x)
            except: return 0.0

        X_proc_df = X_proc_df.map(clean_val).fillna(0.0)
            
        explainer = shap.TreeExplainer(lgb_model)
        shap_values_obj = explainer(X_proc_df)
        
        plt.figure()
        shap.summary_plot(shap_values_obj, X_proc_df, show=False)
        plt.savefig('models/shap_summary_plot.png', bbox_inches='tight')
        plt.close()
        
        # Local Waterfall Plot
        plt.figure()
        shap.plots.waterfall(shap_values_obj[0], show=False)
        plt.savefig('models/shap_waterfall_plot.png', bbox_inches='tight')
        plt.close()
        
        # Save SHAP df
        vals = np.abs(shap_values_obj.values).mean(0)
        shap_df = pd.DataFrame(list(zip(X_proc_df.columns, vals)), columns=['Feature', 'SHAP_Importance'])
        shap_df.sort_values(by=['SHAP_Importance'], ascending=False, inplace=True)
        shap_df.to_csv('models/shap_feature_importance.csv', index=False)
        print("Saved SHAP feature importance successfully.")
    except Exception as e:
        print(f"Could not extract SHAP values: {e}")
    
    print("Saved final models to disk.")
