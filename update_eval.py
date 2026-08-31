import re

with open('c:/Users/26beevlsi049/.gemini/antigravity-ide/scratch/sih-personal-2-main/evaluate_model.py', 'r') as f:
    content = f.read()

# Add early stopping to objective params
xgb_orig = """    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 1000, 5000),
        'learning_rate': trial.suggest_float('xgb_lr', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True),
        **gpu_params['xgb']
    }"""
xgb_new = """    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 1000, 5000),
        'learning_rate': trial.suggest_float('xgb_lr', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True),
        'early_stopping_rounds': 100,
        **gpu_params['xgb']
    }"""
content = content.replace(xgb_orig, xgb_new)

lgb_orig = """    lgb_params = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 1000, 5000),
        'learning_rate': trial.suggest_float('lgb_lr', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 12),
        'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-8, 10.0, log=True),
        'verbose': -1,
        **gpu_params['lgb']
    }"""
lgb_new = """    lgb_params = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 1000, 5000),
        'learning_rate': trial.suggest_float('lgb_lr', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 12),
        'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-8, 10.0, log=True),
        'early_stopping_rounds': 100,
        'verbose': -1,
        **gpu_params['lgb']
    }"""
content = content.replace(lgb_orig, lgb_new)

cat_orig = """    cat_params = {
        'n_estimators': trial.suggest_int('cat_n_estimators', 1000, 5000),
        'learning_rate': trial.suggest_float('cat_lr', 1e-4, 1e-1, log=True),
        'depth': trial.suggest_int('cat_depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('cat_l2', 1e-8, 10.0, log=True),
        **gpu_params['cat']
    }"""
cat_new = """    cat_params = {
        'n_estimators': trial.suggest_int('cat_n_estimators', 1000, 5000),
        'learning_rate': trial.suggest_float('cat_lr', 1e-4, 1e-1, log=True),
        'depth': trial.suggest_int('cat_depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('cat_l2', 1e-8, 10.0, log=True),
        'early_stopping_rounds': 100,
        **gpu_params['cat']
    }"""
content = content.replace(cat_orig, cat_new)

# FTTransformer patience in objective
ftt_orig = """    ftt_params = {
        'batch_size': batch_size,
        'epochs': 500,
    }"""
ftt_new = """    ftt_params = {
        'batch_size': batch_size,
        'epochs': 500,
        'patience': 30
    }"""
content = content.replace(ftt_orig, ftt_new)

# Inner CV fold 10
content = content.replace("inner_cv = StratifiedKFold(n_splits=2 if quick_check else 10, shuffle=True, random_state=42)", "inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)")

# Outer CV fold 10
content = content.replace("outer_cv = StratifiedKFold(n_splits=2 if quick_check else 10, shuffle=True, random_state=42)", "outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)")


# evaluate_nested_cv reconstructions
xgb_recon_orig = """            xgb_params = {
                'n_estimators': best_trial.params['xgb_n_estimators'],
                'learning_rate': best_trial.params['xgb_lr'],
                'max_depth': best_trial.params['xgb_max_depth'],
                'reg_alpha': best_trial.params['xgb_reg_alpha'],
                'reg_lambda': best_trial.params['xgb_reg_lambda'],
                **gpu_params['xgb']
            }"""
xgb_recon_new = """            xgb_params = {
                'n_estimators': best_trial.params['xgb_n_estimators'],
                'learning_rate': best_trial.params['xgb_lr'],
                'max_depth': best_trial.params['xgb_max_depth'],
                'reg_alpha': best_trial.params['xgb_reg_alpha'],
                'reg_lambda': best_trial.params['xgb_reg_lambda'],
                'early_stopping_rounds': 100,
                **gpu_params['xgb']
            }"""
content = content.replace(xgb_recon_orig, xgb_recon_new)

lgb_recon_orig = """            lgb_params = {
                'n_estimators': best_trial.params['lgb_n_estimators'],
                'learning_rate': best_trial.params['lgb_lr'],
                'max_depth': best_trial.params['lgb_max_depth'],
                'reg_alpha': best_trial.params['lgb_reg_alpha'],
                'reg_lambda': best_trial.params['lgb_reg_lambda'],
                'verbose': -1,
                **gpu_params['lgb']
            }"""
lgb_recon_new = """            lgb_params = {
                'n_estimators': best_trial.params['lgb_n_estimators'],
                'learning_rate': best_trial.params['lgb_lr'],
                'max_depth': best_trial.params['lgb_max_depth'],
                'reg_alpha': best_trial.params['lgb_reg_alpha'],
                'reg_lambda': best_trial.params['lgb_reg_lambda'],
                'early_stopping_rounds': 100,
                'verbose': -1,
                **gpu_params['lgb']
            }"""
content = content.replace(lgb_recon_orig, lgb_recon_new)

cat_recon_orig = """            cat_params = {
                'n_estimators': best_trial.params['cat_n_estimators'],
                'learning_rate': best_trial.params['cat_lr'],
                'depth': best_trial.params['cat_depth'],
                'l2_leaf_reg': best_trial.params['cat_l2'],
                **gpu_params['cat']
            }"""
cat_recon_new = """            cat_params = {
                'n_estimators': best_trial.params['cat_n_estimators'],
                'learning_rate': best_trial.params['cat_lr'],
                'depth': best_trial.params['cat_depth'],
                'l2_leaf_reg': best_trial.params['cat_l2'],
                'early_stopping_rounds': 100,
                **gpu_params['cat']
            }"""
content = content.replace(cat_recon_orig, cat_recon_new)

ftt_recon_orig = """            ftt_params = {
                'batch_size': best_trial.params['dl_batch_size'],
                'epochs': 500
            }"""
ftt_recon_new = """            ftt_params = {
                'batch_size': best_trial.params['dl_batch_size'],
                'epochs': 500,
                'patience': 30
            }"""
content = content.replace(ftt_recon_orig, ftt_recon_new)

with open('c:/Users/26beevlsi049/.gemini/antigravity-ide/scratch/sih-personal-2-main/evaluate_model.py', 'w') as f:
    f.write(content)
