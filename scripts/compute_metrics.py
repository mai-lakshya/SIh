import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.calibration import calibration_curve
import sys
import timeline_predictor
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from timeline_predictor import create_structured_survival_array
timeline_predictor.DelayTimelinePredictor = timeline_predictor.NonLinearTimelinePredictor
sys.modules['timeline_predictor'] = timeline_predictor
import warnings
warnings.filterwarnings('ignore')

print('Loading dataset...')
df = pd.read_csv('indian_infrastructure_projects_dataset.csv')
drop_cols = ['delay_binary_label', 'section_11_notification_days', 'CRS', 'project_index', 'Actual_Delay_Days', 'delay_risk_tier', 'CRS_tier']
X = df.drop(columns=drop_cols, errors='ignore')
y_binary = df['delay_binary_label']
y_days = df['section_11_notification_days']
y_crs = df.get('CRS', y_binary * 100)

print('Loading models...')
pipeline = joblib.load('pipeline.joblib')
hybrid = joblib.load('ensemble.joblib')
timeline = joblib.load('timeline.joblib')

print('Transforming data...')
X_tf = pipeline.transform(X)

print('Predictions...')
preds = hybrid.predict(X_tf, blend_monotonicity=True)
delay_prob = preds['delay_probability']
pred_crs = preds['crs']
pred_days_hybrid = preds['delay_days']

y_pred = (delay_prob >= 0.5).astype(int)

print('\n--- Section 3: Classification Performance ---')
print('Accuracy:', accuracy_score(y_binary, y_pred))
print('Precision:', precision_score(y_binary, y_pred))
print('Recall:', recall_score(y_binary, y_pred))
print('F1 (Macro):', f1_score(y_binary, y_pred, average='macro'))
print('F1 (Weighted):', f1_score(y_binary, y_pred, average='weighted'))
print('ROC-AUC:', roc_auc_score(y_binary, delay_prob))
print('PR-AUC:', average_precision_score(y_binary, delay_prob))
print('Confusion Matrix:\n', confusion_matrix(y_binary, y_pred))
print('Brier Score:', brier_score_loss(y_binary, delay_prob))

prob_true, prob_pred = calibration_curve(y_binary, delay_prob, n_bins=10)
ece = np.sum(np.abs(prob_pred - prob_true)) / 10
print('ECE (approx):', ece)

print('\n--- Section 4: Survival & Timeline Performance ---')
# Survival c-index
risk_scores = timeline.rsf.predict(X_tf)
y_surv = create_structured_survival_array(y_binary, y_days)

try:
    c_index, _, _, _, _ = concordance_index_ipcw(y_surv, y_surv, risk_scores)
    print('C-index (RSF):', c_index)
except Exception as e:
    print('C-index error:', e)

try:
    horizons = np.array([90, 180, 270, 365])
    max_time = y_surv['time'].max()
    min_time = y_surv['time'].min()
    valid_horizons = horizons[(horizons >= min_time) & (horizons <= max_time - 1e-5)]
    
    if len(valid_horizons) > 0:
        surv_funcs = timeline.rsf.predict_survival_function(X_tf)
        surv_probs = np.vstack([fn(valid_horizons) for fn in surv_funcs])
        ibs = integrated_brier_score(y_surv, y_surv, surv_probs, valid_horizons)
        print('Integrated Brier Score:', ibs)
    else:
        print('Integrated Brier Score: N/A')
except Exception as e:
    print('IBS error:', e)
    
print('MAE on delay_days:', mean_absolute_error(y_days, pred_days_hybrid))
print('RMSE on delay_days:', np.sqrt(mean_squared_error(y_days, pred_days_hybrid)))
print('MAPE on delay_days:', mean_absolute_percentage_error(y_days, pred_days_hybrid))

print('\n--- Subgroup Fairness ---')
for col in ['project_type', 'state']:
    print(f'\nSubgroup: {col}')
    if col in df.columns:
        for val in df[col].unique():
            mask = df[col] == val
            if mask.sum() > 10:
                print(f"  {val} - Acc: {accuracy_score(y_binary[mask], y_pred[mask]):.3f}")
