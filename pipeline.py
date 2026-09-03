import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import KFold
import warnings

class DynamicFeatureTracker(BaseEstimator, TransformerMixin):
    """
    Automatically tracks categorical and continuous features to maintain
    consistent order and identify indices for downstream steps like SMOTE-NC.
    """
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols
        self.cat_cols_ = []
        self.cont_cols_ = []
        self.cat_indices_ = []

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        if self.cat_cols is None:
            # Dynamically identify categorical columns
            self.cat_cols_ = list(X.select_dtypes(include=['object', 'category', 'bool']).columns)
        else:
            self.cat_cols_ = [c for c in self.cat_cols if c in self.feature_names_in_]
            
        self.cont_cols_ = [c for c in self.feature_names_in_ if c not in self.cat_cols_]
        self.cat_indices_ = [self.feature_names_in_.index(c) for c in self.cat_cols_]
        return self

    def transform(self, X, y=None):
        # Ensure column order consistency
        if hasattr(self, 'feature_names_in_'):
            # Some features might be added by FeatureEngineer
            cols = [c for c in self.feature_names_in_ if c in X.columns]
            # Add any new columns that are in X but not in feature_names_in_
            new_cols = [c for c in X.columns if c not in cols]
            return X[cols + new_cols]
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Computes financial density, population density, velocity, and categorical interactions.
    """
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X_out = X.copy()
        
        # 1. Ratios (Density)
        if 'estimated_cost_inr_crore' in X_out.columns and 'land_area_hectares' in X_out.columns:
            X_out['financial_density'] = X_out['estimated_cost_inr_crore'] / np.maximum(X_out['land_area_hectares'], 1e-5)
            
        if 'affected_families_count' in X_out.columns and 'land_area_hectares' in X_out.columns:
            X_out['population_density'] = X_out['affected_families_count'] / np.maximum(X_out['land_area_hectares'], 1e-5)
            
        # 2. Velocity (Fallback: Financial Burn Rate to date since planned_duration is missing)
        # We don't have planned_project_duration_years, initial_proposal_date, or planned_completion_date
        # Therefore, we use the actual project_age_years.
        if 'estimated_cost_inr_crore' in X_out.columns and 'project_age_years' in X_out.columns:
            X_out['financial_burn_rate_to_date'] = X_out['estimated_cost_inr_crore'] / np.maximum(X_out['project_age_years'], 1)
            
        # 3. Interactions
        if 'state' in X_out.columns and 'project_type' in X_out.columns:
            X_out['state_project_type'] = X_out['state'].astype(str) + "_" + X_out['project_type'].astype(str)
            
        return X_out


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies np.log1p to variables with heavy right-skew before scaling.
    """
    def __init__(self, cols=None):
        self.cols = cols if cols else ['project_cost_cr', 'land_area_hectares', 'affected_families_count']
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X_out = X.copy()
        for col in self.cols:
            if col in X_out.columns:
                # Convert to numeric just in case, coerce errors to NaN, fill with 0
                X_out[col] = pd.to_numeric(X_out[col], errors='coerce').fillna(0)
                # Apply log1p safely
                X_out[col] = np.log1p(np.maximum(0, X_out[col]))
        return X_out

class OOFTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed Target Encoder (m-estimate) with Out-Of-Fold regularization
    to prevent target leakage during training.
    """
    def __init__(self, cols=None, m=10.0, cv=5, random_state=42):
        self.cols = cols
        self.m = m
        self.cv = cv
        self.random_state = random_state
        
    def fit(self, X, y):
        y = np.array(y)
        self.global_mean_ = np.mean(y)
        self.cols_to_encode_ = self.cols if self.cols is not None else list(X.select_dtypes(include=['object', 'category']).columns)
        self.mapping_ = {}
        
        # Precompute global mappings for transform() on test data
        for col in self.cols_to_encode_:
            if col in X.columns:
                df_temp = pd.DataFrame({col: X[col], 'target': y})
                agg = df_temp.groupby(col)['target'].agg(['sum', 'count'])
                smoothed = (agg['sum'] + self.m * self.global_mean_) / (agg['count'] + self.m)
                self.mapping_[col] = smoothed.to_dict()
                
        return self
        
    def fit_transform(self, X, y=None):
        if y is None:
            return self.fit(X).transform(X)
            
        y = np.array(y)
        self.fit(X, y)
        X_out = X.copy()
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        for col in self.cols_to_encode_:
            if col not in X.columns:
                continue
                
            out_col = np.zeros(len(X))
            
            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr = y[train_idx]
                
                df_temp = pd.DataFrame({col: X_tr[col], 'target': y_tr})
                agg = df_temp.groupby(col)['target'].agg(['sum', 'count'])
                smoothed = (agg['sum'] + self.m * self.global_mean_) / (agg['count'] + self.m)
                
                out_col[val_idx] = X_val[col].map(smoothed).fillna(self.global_mean_)
                
            # Replace the categorical column with the encoded float
            X_out[col] = out_col.astype(float)
            
        return X_out

    def transform(self, X, y=None):
        X_out = X.copy()
        for col in self.cols_to_encode_:
            if col in X_out.columns:
                # Use global mapping learned during fit
                X_out[col] = X_out[col].map(self.mapping_.get(col, {})).fillna(self.global_mean_).astype(float)
        return X_out

class SMOTENCDynamicWrapper(BaseEstimator):
    """
    Wrapper for SMOTE-NC that dynamically fetches categorical indices.
    If no categorical features exist, it falls back to regular SMOTE.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.sampler_ = None
        
    def fit_resample(self, X, y):
        cat_indices = []
        if isinstance(X, pd.DataFrame):
            cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
            cat_indices = [X.columns.get_loc(c) for c in cat_cols]
            
        if len(cat_indices) > 0:
            self.sampler_ = SMOTENC(categorical_features=cat_indices, random_state=self.random_state)
        else:
            from imblearn.over_sampling import SMOTE
            self.sampler_ = SMOTE(random_state=self.random_state)
            
        # Avoid unnecessary warnings or errors if data is not imbalanced
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) <= 1 or min(counts) == max(counts):
            return X, y
            
        X_res, y_res = self.sampler_.fit_resample(X, y)
        return X_res, y_res
        
    def fit(self, X, y=None):
        self.fit_resample(X, y)
        return self

def get_preprocessing_pipeline(cat_cols=None, log_cols=None, te_cols=None, use_smote=True):
    """
    Constructs the leakage-free preprocessing pipeline.
    """
    steps = [
        ('feature_engineer', FeatureEngineer()),
        ('tracker', DynamicFeatureTracker(cat_cols=cat_cols)),
        ('log_transform', LogTransformer(cols=log_cols)),
    ]
    
    if use_smote:
        steps.append(('smote_nc', SMOTENCDynamicWrapper(random_state=42)))
        
    steps.append(('target_encoder', OOFTargetEncoder(cols=te_cols, m=10.0, cv=5)))
    
    return ImblearnPipeline(steps)
