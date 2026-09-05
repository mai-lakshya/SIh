import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import StackingClassifier, StackingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV, RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import logit
import joblib

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

def safe_logit(X):
    """
    Apply logit transform to probabilities safely, 
    avoiding exactly 0 or 1 which result in -inf or inf.
    """
    X_clipped = np.clip(X, 1e-6, 1.0 - 1e-6)
    return logit(X_clipped)

class TreeWrapperBase(BaseEstimator):
    def __init__(self, model_class, random_state=42, **kwargs):
        self.model_class = model_class
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y):
        X_np = X.values if hasattr(X, 'values') else X
        y_np = np.array(y)
        
        stratify = y_np if isinstance(self, ClassifierMixin) else None
        if len(X_np) < 20:
            X_tr, X_val, y_tr, y_val = X_np, X_np, y_np, y_np
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_np, y_np, test_size=0.1, random_state=self.random_state, stratify=stratify
            )
            
        # extract early_stopping_rounds
        esr = self.kwargs.pop('early_stopping_rounds', None)
        
        # recreate model with remaining kwargs
        self.model = self.model_class(random_state=self.random_state, **self.kwargs)
        
        fit_kwargs = {}
        if esr is not None:
            if 'XGB' in self.model_class.__name__:
                self.model.set_params(early_stopping_rounds=esr)
                fit_kwargs['eval_set'] = [(X_val, y_val)]
                fit_kwargs['verbose'] = False
            elif 'LGBM' in self.model_class.__name__:
                import lightgbm as lgb
                fit_kwargs['eval_set'] = [(X_val, y_val)]
                fit_kwargs['callbacks'] = [lgb.early_stopping(stopping_rounds=esr, verbose=False)]
            elif 'CatBoost' in self.model_class.__name__:
                self.model.set_params(early_stopping_rounds=esr)
                fit_kwargs['eval_set'] = [(X_val, y_val)]
                fit_kwargs['verbose'] = False
                
        self.model.fit(X_tr, y_tr, **fit_kwargs)
        
        if isinstance(self, ClassifierMixin):
            self.classes_ = np.unique(y_np)
        return self

    def predict(self, X):
        X_np = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_np)

class TreeWrapperClassifier(ClassifierMixin, TreeWrapperBase):
    _estimator_type = "classifier"
    def __init__(self, model_class, random_state=42, **kwargs):
        super().__init__(model_class, random_state, **kwargs)
        
    def predict_proba(self, X):
        X_np = X.values if hasattr(X, 'values') else X
        return self.model.predict_proba(X_np)

class TreeWrapperRegressor(RegressorMixin, TreeWrapperBase):
    _estimator_type = "regressor"
    def __init__(self, model_class, random_state=42, **kwargs):
        super().__init__(model_class, random_state, **kwargs)

class HybridRiskPredictor:
    def __init__(self, cat_features=None, random_state=42, model_params=None, calibration_method='sigmoid'):
        self.cat_features = cat_features
        self.random_state = random_state
        self.model_params = model_params or {}
        self.calibration_method = calibration_method
        self.classifier = None
        self.calibrated_classifier = None
        self.regressor_crs = None
        self.regressor_days = None

    def _build_classifiers(self, cv=3):
        # Extract params for each model
        lgb_params = self.model_params.get('lgb', {})
        xgb_params = self.model_params.get('xgb', {})
        cat_params = self.model_params.get('cat', {})
        et_params = self.model_params.get('et', {})
        
        lgb_params = {'verbose': -1, **lgb_params}
        xgb_params = {'verbosity': 0, **xgb_params}
        cat_params = {'verbose': False, **cat_params}
        
        lgb_clf = TreeWrapperClassifier(lgb.LGBMClassifier, random_state=self.random_state, **lgb_params)
        xgb_clf = TreeWrapperClassifier(xgb.XGBClassifier, random_state=self.random_state, **xgb_params)
        cat_clf = TreeWrapperClassifier(CatBoostClassifier, random_state=self.random_state, **cat_params)
        et_clf = TreeWrapperClassifier(ExtraTreesClassifier, random_state=self.random_state, **et_params)
        
        base_classifiers = [
            ('lgb', lgb_clf),
            ('xgb', xgb_clf),
            ('cat', cat_clf),
            ('et', et_clf)
        ]
        
        meta_classifier = Pipeline([
            ('logit', FunctionTransformer(safe_logit)),
            ('lr', LogisticRegressionCV(
                class_weight='balanced',
                max_iter=2000,
                cv=cv,
                random_state=self.random_state,
                # Pinned under scikit-learn 1.9.0: set to (0.0,) representing pure L2 penalty.
                # Replaces deprecated None to eliminate FutureWarning and ensure full forward-compatibility with 1.10+.
                l1_ratios=(0.0,),
                # Pinned under scikit-learn 1.9.0: default is 'accuracy'.
                # The default changes to 'neg_log_loss' in version 1.11.
                scoring='accuracy',
                # Pinned under scikit-learn 1.9.0: default is True.
                # The default changes to False in version 1.10.
                # Preserves current fitted-attribute behavior and attribute shapes.
                # NOTE: Flipping this to False in the future is a deliberate migration decision
                # requiring re-validation of validate_additivity() and the faithfulness benchmarks,
                # not a silent warning-suppression.
                use_legacy_attributes=True
            ))
        ])
        
        return StackingClassifier(
            estimators=base_classifiers,
            final_estimator=meta_classifier,
            cv=cv,
            n_jobs=1,
            passthrough=False
        )

    def _build_regressors(self, cv=3):
        lgb_params = self.model_params.get('lgb', {})
        xgb_params = self.model_params.get('xgb', {})
        cat_params = self.model_params.get('cat', {})
        et_params = self.model_params.get('et', {})
        
        lgb_params = {'verbose': -1, **lgb_params}
        xgb_params = {'verbosity': 0, **xgb_params}
        cat_params = {'verbose': False, **cat_params}
        
        lgb_reg = TreeWrapperRegressor(lgb.LGBMRegressor, random_state=self.random_state, **lgb_params)
        xgb_reg = TreeWrapperRegressor(xgb.XGBRegressor, random_state=self.random_state, **xgb_params)
        cat_reg = TreeWrapperRegressor(CatBoostRegressor, random_state=self.random_state, **cat_params)
        et_reg = TreeWrapperRegressor(ExtraTreesRegressor, random_state=self.random_state, **et_params)
        
        base_regressors = [
            ('lgb', lgb_reg),
            ('xgb', xgb_reg),
            ('cat', cat_reg),
            ('et', et_reg)
        ]
        
        alphas = np.logspace(-3, 4, 30)
        meta_regressor = RidgeCV(alphas=alphas, cv=cv)
        
        return StackingRegressor(
            estimators=base_regressors,
            final_estimator=meta_regressor,
            cv=cv,
            n_jobs=1,
            passthrough=False
        )

    def fit(self, X, y_cls, y_crs, y_days):
        y_cls_arr = np.asarray(y_cls, dtype=int)
        counts = np.bincount(y_cls_arr)
        min_class_count = min(counts) if len(counts) > 1 else 1

        # Adaptive CV splits based on minimum class count
        cv_splits = min(3, max(2, min_class_count // 3)) if min_class_count >= 4 else 2

        self.classifier = self._build_classifiers(cv=cv_splits)
        cal_method = getattr(self, 'calibration_method', 'sigmoid')
        if min_class_count < 15 and cal_method == 'isotonic':
            cal_method = 'sigmoid'
        self.calibrated_classifier = CalibratedClassifierCV(
            estimator=self.classifier,
            method=cal_method,
            cv=cv_splits
        )
        self.calibrated_classifier.fit(X, y_cls)
        
        if hasattr(self.calibrated_classifier, 'calibrated_classifiers_') and len(self.calibrated_classifier.calibrated_classifiers_) > 0:
            self.classifier = self.calibrated_classifier.calibrated_classifiers_[0].estimator

        self.regressor_crs = self._build_regressors(cv=cv_splits)
        self.regressor_crs.fit(X, y_crs)
        
        self.regressor_days = self._build_regressors(cv=cv_splits)
        self.regressor_days.fit(X, y_days)
        
        return self

    def predict(self, X, blend_monotonicity=True):
        delay_prob = self.calibrated_classifier.predict_proba(X)[:, 1]
        pred_crs = self.regressor_crs.predict(X)
        pred_days = self.regressor_days.predict(X)
        
        if blend_monotonicity:
            pred_days = pred_days * (0.5 + delay_prob)
            pred_crs = pred_crs * (0.5 + delay_prob)
            pred_days = np.maximum(0, pred_days)
            pred_crs = np.clip(pred_crs, 0, 100)

        risk_tiers = np.where(pred_crs > 75, "Critical",
                     np.where(pred_crs > 50, "High",
                     np.where(pred_crs > 25, "Medium", "Low")))
            
        return {
            'delay_probability': delay_prob,
            'crs': pred_crs,
            'delay_days': pred_days,
            'predicted_delay_days': pred_days,
            'risk_tier': risk_tiers
        }

    def save(self, filepath):
        joblib.dump(self, filepath, compress=3)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
