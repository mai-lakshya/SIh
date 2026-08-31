import os

with open("evaluate_model.py", "r") as f:
    content = f.read()

# We want to keep everything from start to the end of `def objective(...)` function.
# `def evaluate_model(...)` starts around line 162.
# So we can split at `def evaluate_model`
parts = content.split("def evaluate_model(X, y_cls, y_crs, y_days, n_trials=500, quick_check=False):")

top_part = parts[0]

main_block = """if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=500, help='Number of optuna trials')
    parser.add_argument('--quick-check', action='store_true', help='Run minimal folds for testing')
    args = parser.parse_args()
    
    df = pd.read_csv('indian_infrastructure_projects_dataset.csv').sample(200 if args.quick_check else 1000)
    X = df.drop(columns=['delay_binary_label', 'section_11_notification_days', 'CRS', 'project_index', 'Actual_Delay_Days'], errors='ignore')
    
    cat_cols = ['state', 'district', 'project_type', 'terrain_type', 'sia_approval_status', 'forest_clearance_status']
    
    pipeline = get_preprocessing_pipeline(use_smote=False)
    X_tf = pipeline.fit_transform(X, df['delay_binary_label'])
    
    y_cls = df['delay_binary_label']
    y_crs = df['CRS']
    y_days = df['section_11_notification_days']
    n_trials = args.n_trials
    quick_check = args.quick_check

    mlflow.set_experiment("God_Mode_Evaluation")
    
    # --- PHASE 1: GLOBAL HYPERPARAMETER SEARCH ---
    print(f"Starting Phase 1: Global Optuna Search ({n_trials} trials)")
    sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=10, seed=42)
    
    storage = "sqlite:///god_mode_study.db"
    study_name = "god_mode_study"
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize", "maximize"],
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
        study_name=study_name
    )
    
    study.optimize(lambda trial: objective(trial, X_tf, y_cls, y_crs, y_days, quick_check), n_trials=n_trials, timeout=41400)
    
    # Select best trade-off from Pareto front
    best_trials = study.best_trials
    best_trial = best_trials[0]
    
    print("\\nPhase 1 Complete. Best Hyperparameters found.")
    
    # --- PHASE 2: OUTER GENERALIZATION EVALUATION ---
    print("\\nStarting Phase 2: Outer 10-Fold CV Evaluation")
    
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    outer_results = []
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_tf, y_cls)):
        with mlflow.start_run(run_name=f"Outer_Fold_{fold}", nested=True):
            X_tr, X_te = X_tf.iloc[train_idx], X_tf.iloc[test_idx]
            y_cls_tr, y_cls_te = y_cls.iloc[train_idx], y_cls.iloc[test_idx]
            y_crs_tr, y_crs_te = y_crs.iloc[train_idx], y_crs.iloc[test_idx]
            y_days_tr, y_days_te = y_days.iloc[train_idx], y_days.iloc[test_idx]
            
            mlflow.log_params(best_trial.params)
            
            # Reconstruct optimal params for final model training on entire outer fold train set
            gpu_params = {
                'xgb': {'tree_method': 'hist', 'n_jobs': -1},
                'lgb': {'n_jobs': -1},
                'cat': {}
            }

            xgb_params = {
                'n_estimators': best_trial.params['xgb_n_estimators'],
                'learning_rate': best_trial.params['xgb_lr'],
                'max_depth': best_trial.params['xgb_max_depth'],
                'reg_alpha': best_trial.params['xgb_reg_alpha'],
                'reg_lambda': best_trial.params['xgb_reg_lambda'],
                'early_stopping_rounds': 100,
                **gpu_params['xgb']
            }
            lgb_params = {
                'n_estimators': best_trial.params['lgb_n_estimators'],
                'learning_rate': best_trial.params['lgb_lr'],
                'max_depth': best_trial.params['lgb_max_depth'],
                'reg_alpha': best_trial.params['lgb_reg_alpha'],
                'reg_lambda': best_trial.params['lgb_reg_lambda'],
                'early_stopping_rounds': 100,
                'verbose': -1,
                **gpu_params['lgb']
            }
            cat_params = {
                'n_estimators': best_trial.params['cat_n_estimators'],
                'learning_rate': best_trial.params['cat_lr'],
                'depth': best_trial.params['cat_depth'],
                'l2_leaf_reg': best_trial.params['cat_l2'],
                'early_stopping_rounds': 100,
                **gpu_params['cat']
            }
            tab_params = {
                'n_d': best_trial.params['tab_n_da'],
                'n_a': best_trial.params['tab_n_da'],
                'n_steps': best_trial.params['tab_n_steps'],
                'gamma': best_trial.params['tab_gamma'],
                'batch_size': best_trial.params['dl_batch_size'],
                'max_epochs': 40,
                'patience': 5
            }
            ftt_params = {
                'batch_size': best_trial.params['dl_batch_size'],
                'epochs': 40,
                'patience': 5
            }
            
            model_params = {
                'xgb': xgb_params,
                'lgb': lgb_params,
                'cat': cat_params,
                'tab': tab_params,
                'ftt': ftt_params
            }
            
            rsf_params = {
                'n_estimators': best_trial.params['rsf_n_estimators'],
                'min_samples_split': best_trial.params['rsf_min_samples_split']
            }
            
            deepsurv_params = {
                'hidden_layers': best_trial.params['ds_hidden_layers'],
                'dropout_p': best_trial.params['ds_dropout_p'],
                'lr': best_trial.params['ds_lr']
            }
            
            pipeline_cv = get_preprocessing_pipeline(use_smote=best_trial.params['use_smote'])
            X_tr_proc = pipeline_cv.fit_transform(X_tr, y_cls_tr)
            X_te_proc = pipeline_cv.transform(X_te)
            
            # Train Hybrid Model
            print(f"  Training Meta-Model for Fold {fold}...")
            model = HybridRiskPredictor(random_state=42, model_params=model_params)
            model.fit(X_tr_proc, y_cls_tr, y_crs_tr, y_days_tr)
            
            preds = model.predict(X_te_proc, blend_monotonicity=True)
            y_pred_cls = (preds['delay_probability'] >= 0.5).astype(int)
            
            recall = recall_score(y_cls_te, y_pred_cls, zero_division=0)
            ece = expected_calibration_error(y_cls_te, preds['delay_probability'])
            rmse = np.sqrt(mean_squared_error(y_crs_te, preds['crs']))
            
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_ece", ece)
            mlflow.log_metric("test_crs_rmse", rmse)
            
            # Train Timeline Predictor
            print(f"  Training Survival Model for Fold {fold}...")
            tl_model = NonLinearTimelinePredictor(random_state=42, rsf_params=rsf_params, deepsurv_params=deepsurv_params)
            tl_model.fit(X_tr_proc, y_cls_tr, y_days_tr)
            
            metrics = tl_model.evaluate(X_tr_proc, y_cls_tr, y_days_tr, X_te_proc, y_cls_te, y_days_te)
            c_index = metrics.get('c_index_uno', np.nan)
            
            mlflow.log_metric("test_c_index_uno", c_index)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"models/run_god_mode_{timestamp}/fold_{fold}"
            os.makedirs(save_dir, exist_ok=True)
            
            model.save(f"{save_dir}/hybrid_model.joblib")
            tl_model.save(f"{save_dir}/timeline_predictor.joblib")
            mlflow.log_artifact(f"{save_dir}/hybrid_model.joblib")
            mlflow.log_artifact(f"{save_dir}/timeline_predictor.joblib")
            
            outer_results.append((recall, ece, rmse, c_index))
            
    print("\\nPhase 3: Final Model Training on Entire Dataset")
    pipeline_final = get_preprocessing_pipeline(use_smote=best_trial.params['use_smote'])
    X_proc = pipeline_final.fit_transform(X_tf, y_cls)
    
    final_hybrid = HybridRiskPredictor(random_state=42, model_params=model_params)
    final_hybrid.fit(X_proc, y_cls, y_crs, y_days)
    
    final_tl = NonLinearTimelinePredictor(random_state=42, rsf_params=rsf_params, deepsurv_params=deepsurv_params)
    final_tl.fit(X_proc, y_cls, y_days)
    
    joblib.dump(pipeline_final, "final_pipeline.joblib")
    final_hybrid.save("final_hybrid_model.joblib")
    final_tl.save("final_timeline_predictor.joblib")
    mlflow.log_artifact("final_pipeline.joblib")
    mlflow.log_artifact("final_hybrid_model.joblib")
    mlflow.log_artifact("final_timeline_predictor.joblib")
    print("Saved final models to disk and MLflow.")
"""

with open("evaluate_model.py", "w") as f:
    f.write(top_part + main_block)
