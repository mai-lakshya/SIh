import re
import json
import logging
import warnings
import numpy as np
import pandas as pd
from scipy.special import logit

logger = logging.getLogger(__name__)

# Narrowly filter third-party PendingDeprecationWarnings from shap color palette functions
warnings.filterwarnings(
    "ignore",
    message=r"The (set_bad|set_over|set_under) function will be deprecated in a future version\..*",
    category=PendingDeprecationWarning,
    module=r"shap\.plots\.colors\._colors"
)
import shap


class DualParadigmExplainer:
    """
    Explainability Engine combining Meta-Learner-Weighted TreeSHAP across
    heterogeneous tree ensembles (LightGBM, XGBoost, CatBoost, ExtraTrees).
    Attributions are mathematically weighted by the stacking meta-learner's
    fitted coefficients in logit space. Supports honest neural attention
    verification (TabNet), single-instance and batch explanations, robust input
    coercion, and an input-sensitive local perturbation fallback path.
    """

    def __init__(self, hybrid_predictor, feature_names, background_data=None, allow_fallback=False):
        self.hybrid_predictor = hybrid_predictor
        self.feature_names = list(feature_names)
        self.tree_models = {}
        self.meta_coefficients = {}
        self.meta_intercept = 0.0
        self._tree_explainers = {}
        self.tabnet_model = None
        self.background_data = None
        self._cached_global_importance = None
        self.allow_fallback = allow_fallback

        # Extract base estimators and meta-learner coefficients from StackingClassifier
        self._extract_models()

        # Section 2: Explicit honest check for TabNet
        if self.tabnet_model is None:
            logger.info(
                "No TabNet neural attention estimator detected in the ensemble artifact. "
                "DualParadigmExplainer is operating in Meta-Learner-Weighted TreeSHAP mode."
            )

        # If base models specify an expected feature count, align feature_names
        expected_dim = None
        for model in self.tree_models.values():
            if hasattr(model, 'n_features_in_') and model.n_features_in_ > 0:
                expected_dim = model.n_features_in_
                break
            elif hasattr(model, 'n_features_') and model.n_features_ > 0:
                expected_dim = model.n_features_
                break

        if expected_dim is not None and len(self.feature_names) > expected_dim:
            self.feature_names = self.feature_names[:expected_dim]

        # Initialize background dataset if provided
        if background_data is not None:
            self.set_background_data(background_data)

    def _extract_models(self):
        """
        Dynamically extracts all base estimators from the stacking classifier
        and pulls the meta-learner's fitted coefficients directly at runtime.
        Includes ExtraTrees alongside LightGBM, XGBoost, and CatBoost without
        hardcoding names or numeric weights.
        """
        stacker = None
        if hasattr(self.hybrid_predictor, 'calibrated_classifier') and hasattr(
            self.hybrid_predictor.calibrated_classifier, 'calibrated_classifiers_'
        ) and len(self.hybrid_predictor.calibrated_classifier.calibrated_classifiers_) > 0:
            stacker = self.hybrid_predictor.calibrated_classifier.calibrated_classifiers_[0].estimator
        elif hasattr(self.hybrid_predictor, 'classifier'):
            stacker = self.hybrid_predictor.classifier

        if stacker is not None:
            # Extract meta-learner coefficients and intercept from final_estimator_
            coefs = None
            intercept = 0.0
            if hasattr(stacker, 'final_estimator_'):
                final_est = stacker.final_estimator_
                if hasattr(final_est, 'steps'):
                    final_step = final_est.steps[-1][1]
                else:
                    final_step = final_est
                if hasattr(final_step, 'coef_'):
                    coefs = np.asarray(final_step.coef_[0], dtype=np.float64)
                if hasattr(final_step, 'intercept_'):
                    intercept = float(final_step.intercept_[0])
            self.meta_intercept = intercept

            # Extract base estimators dynamically
            names_list = []
            estimators_list = []
            if hasattr(stacker, 'estimators_') and hasattr(stacker, 'estimators'):
                names_list = [name for name, _ in stacker.estimators]
                estimators_list = list(stacker.estimators_)
            elif hasattr(stacker, 'named_estimators_'):
                names_list = list(stacker.named_estimators_.keys())
                estimators_list = list(stacker.named_estimators_.values())

            for idx, (name, estimator) in enumerate(zip(names_list, estimators_list)):
                unwrapped = estimator.model if hasattr(estimator, 'model') else estimator
                if coefs is not None and idx < len(coefs):
                    self.meta_coefficients[name] = float(coefs[idx])
                else:
                    self.meta_coefficients[name] = 1.0 / max(len(names_list), 1)

                if name == 'tab':
                    self.tabnet_model = unwrapped
                else:
                    self.tree_models[name] = unwrapped

    def _normalize(self, arr):
        """Min-max normalize array to [0, 1] range."""
        arr = np.asarray(arr, dtype=np.float64)
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    def _safe_logit(self, p):
        """Applies safe logit transform to probabilities clipped to [1e-6, 1 - 1e-6]."""
        p_clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
        return logit(p_clipped)

    def _coerce_input(self, X):
        """
        Ensures input is transformed into a clean pandas DataFrame conforming
        strictly to self.feature_names with numeric dtypes. Handles dicts, Series,
        ndarrays, stringified numbers, missing levels, and unexpected columns.
        """
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            df = X.to_frame().T
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if len(self.feature_names) == X.shape[1]:
                df = pd.DataFrame(X, columns=self.feature_names)
            else:
                df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(f"Unsupported input type for explanation: {type(X)}")

        # Add missing columns with default 0.0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0

        # Retain and order strictly by self.feature_names
        df = df[self.feature_names].copy()

        # Coerce column values safely to numeric
        bool_map = {
            'true': 1.0, 'false': 0.0, 'yes': 1.0, 'no': 0.0,
            't': 1.0, 'f': 0.0, 'approved': 0.0, 'pending': 1.0, 'rejected': 1.0
        }
        for col in self.feature_names:
            series = df[col]
            if pd.api.types.is_bool_dtype(series):
                df[col] = series.astype(float)
            elif pd.api.types.is_numeric_dtype(series):
                df[col] = pd.to_numeric(series, errors='coerce').fillna(0.0).astype(float)
            else:
                str_mapped = series.astype(str).str.strip().str.lower().map(bool_map)
                numeric_val = pd.to_numeric(series, errors='coerce')
                df[col] = str_mapped.fillna(numeric_val).fillna(0.0).astype(float)

        return df

    def set_background_data(self, background_data):
        """
        Registers a background dataset for global feature importance calculation.
        Subsamples 50-100 rows to maintain responsive SLAs.
        """
        coerced = self._coerce_input(background_data)
        if len(coerced) > 100:
            self.background_data = coerced.sample(n=min(len(coerced), 100), random_state=42).reset_index(drop=True)
        else:
            self.background_data = coerced.reset_index(drop=True)
        self._compute_and_cache_global_importance()

    def _compute_and_cache_global_importance(self):
        """Computes and caches global importance on the reference dataset."""
        if self.background_data is not None and len(self.background_data) > 0:
            unified, _, _ = self.get_global_importance(self.background_data)
            global_importance = []
            for i, feat in enumerate(self.feature_names):
                global_importance.append({
                    "feature": feat,
                    "importance": float(unified[i])
                })
            self._cached_global_importance = sorted(global_importance, key=lambda x: x["importance"], reverse=True)

    def _compute_model_shap(self, name, model, X_df):
        """
        Computes TreeSHAP for a single base estimator and maps outputs into
        additive logit space.
        - For margin-space models (lgb, xgb, cat): TreeSHAP values are directly in logit space.
        - For probability-space models (ExtraTrees / Random Forest): TreeSHAP values are scaled
          to logit space via the exact secant transformation:
            scale = (logit(p) - logit(p_base)) / (p - p_base)
          guaranteeing exact additive decomposition in the ensemble's meta-learner logit space.
        """
        if name not in self._tree_explainers:
            self._tree_explainers[name] = shap.TreeExplainer(model)
        explainer = self._tree_explainers[name]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_vals = explainer.shap_values(X_df)

        ev = explainer.expected_value

        # Standardize multi-class / list outputs to class 1 (delay)
        if isinstance(shap_vals, list) and len(shap_vals) > 1:
            shap_vals = shap_vals[1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]

        if isinstance(ev, (list, np.ndarray)) and len(ev) > 1:
            base_val = float(ev[1])
        elif isinstance(ev, (list, np.ndarray)) and len(ev) == 1:
            base_val = float(ev[0])
        else:
            base_val = float(ev)

        shap_vals = np.asarray(shap_vals, dtype=np.float64)

        # Detect probability space output (e.g. ExtraTreesClassifier)
        is_prob_model = (
            name == 'et' or
            'ExtraTrees' in type(model).__name__ or
            'RandomForest' in type(model).__name__ or
            (0.0 <= base_val <= 1.0 and hasattr(model, 'estimators_'))
        )

        if is_prob_model:
            p_row = base_val + np.sum(shap_vals, axis=1)
            dp = p_row - base_val
            dlogit = self._safe_logit(p_row) - self._safe_logit(base_val)
            scale = np.where(
                np.abs(dp) > 1e-7,
                dlogit / (dp + 1e-12),
                1.0 / max(base_val * (1.0 - base_val), 1e-6)
            )
            shap_vals = shap_vals * scale[:, None]
            base_val = float(self._safe_logit(base_val))

        return shap_vals, base_val

    def validate_additivity(self, X_sample):
        """
        Hard additivity validation check: computes reconstructed logit
        sum_j weighted_shap_j + combined_base_value and compares against
        the ensemble's actual pre-calibration stacked logit for each row.
        Includes TabNet contribution when self.tabnet_model is present.
        Returns error distribution metrics.
        """
        X_df = self._coerce_input(X_sample)
        N = len(X_df)
        X_np = X_df.values.astype(np.float32)

        # Stacker pre-calibration decision function
        stacker = None
        if hasattr(self.hybrid_predictor, 'calibrated_classifier') and hasattr(
            self.hybrid_predictor.calibrated_classifier, 'calibrated_classifiers_'
        ) and len(self.hybrid_predictor.calibrated_classifier.calibrated_classifiers_) > 0:
            stacker = self.hybrid_predictor.calibrated_classifier.calibrated_classifiers_[0].estimator
        elif hasattr(self.hybrid_predictor, 'classifier'):
            stacker = self.hybrid_predictor.classifier

        if stacker is None or not hasattr(stacker, 'final_estimator_'):
            return {"error": "Stacking classifier not found on hybrid_predictor"}

        lr = stacker.final_estimator_.steps[-1][1] if hasattr(stacker.final_estimator_, 'steps') else stacker.final_estimator_
        probas = np.column_stack([est.predict_proba(X_df)[:, 1] for est in stacker.estimators_])
        true_logit = lr.decision_function(self._safe_logit(probas))

        # Reconstructed logit from weighted TreeSHAP
        recon_shap = np.zeros((N, len(self.feature_names)), dtype=np.float64)
        recon_base = self.meta_intercept

        for name, model in self.tree_models.items():
            s_vals, b_val = self._compute_model_shap(name, model, X_df)
            c = self.meta_coefficients.get(name, 1.0)
            recon_shap += c * s_vals
            recon_base += c * b_val

        # Reconstructed contribution from TabNet if present
        if self.tabnet_model is not None:
            c_tab = self.meta_coefficients.get('tab', 1.0)
            p_tab = np.asarray(self.tabnet_model.predict_proba(X_df)[:, 1], dtype=np.float64)
            logit_tab = self._safe_logit(p_tab)

            # Baseline logit for TabNet
            if self.background_data is not None and len(self.background_data) > 0:
                p_bg = float(np.mean(self.tabnet_model.predict_proba(self.background_data)[:, 1]))
                b_tab = float(self._safe_logit(p_bg))
            else:
                p_bg = float(np.mean(p_tab))
                b_tab = float(self._safe_logit(p_bg))

            # Neural attention masks if available
            attn_masks = None
            if hasattr(self.tabnet_model, 'explain'):
                try:
                    attn_masks, _ = self.tabnet_model.explain(X_np)
                except Exception:
                    pass
            elif hasattr(self.tabnet_model, 'model') and hasattr(self.tabnet_model.model, 'explain'):
                try:
                    attn_masks, _ = self.tabnet_model.model.explain(X_np)
                except Exception:
                    pass

            if attn_masks is None or not isinstance(attn_masks, np.ndarray) or attn_masks.shape != (N, len(self.feature_names)):
                attn_masks = np.ones((N, len(self.feature_names)), dtype=np.float64)

            attn_sum = np.sum(attn_masks, axis=1, keepdims=True)
            norm_attn = np.where(attn_sum > 1e-12, attn_masks / attn_sum, 1.0 / len(self.feature_names))
            delta_logit_tab = logit_tab - b_tab
            s_tab = norm_attn * delta_logit_tab[:, None]

            recon_shap += c_tab * s_tab
            recon_base += c_tab * b_tab

        reconstructed_logit = np.sum(recon_shap, axis=1) + recon_base
        errors = np.abs(reconstructed_logit - true_logit)

        return {
            "sample_size": N,
            "max_absolute_error": float(np.max(errors)),
            "mean_absolute_error": float(np.mean(errors)),
            "median_absolute_error": float(np.median(errors)),
            "is_exact": bool(np.max(errors) < 1e-4)
        }

    def get_global_importance(self, X=None, allow_fallback=None):
        """
        Computes global feature importance using meta-learner weighted TreeSHAP
        across all samples in X. Surfaces warnings for failing models and raises
        if all fail unless fallback path is enabled.
        """
        if X is None:
            if self.background_data is not None:
                X = self.background_data
            else:
                raise ValueError("No background dataset provided for global importance calculation.")

        X_df = self._coerce_input(X)
        N = len(X_df)
        X_np = X_df.values.astype(np.float32)

        weighted_shap = np.zeros((N, len(self.feature_names)), dtype=np.float64)
        models_succeeded = 0
        models_failed = []

        for name, model in self.tree_models.items():
            try:
                shap_vals, _ = self._compute_model_shap(name, model, X_df)
                if shap_vals.shape == (N, len(self.feature_names)):
                    coef = self.meta_coefficients.get(name, 1.0)
                    weighted_shap += coef * shap_vals
                    models_succeeded += 1
            except Exception as e:
                warnings.warn(f"Base model '{name}' failed during global importance: {e}", RuntimeWarning)
                models_failed.append(name)

        fallback_enabled = allow_fallback if allow_fallback is not None else self.allow_fallback
        if len(self.tree_models) > 0 and models_succeeded == 0 and not fallback_enabled:
            raise RuntimeError(
                f"All tree models {list(self.tree_models.keys())} failed during global importance: {models_failed}. "
                "Fallback path is not enabled."
            )

        if models_succeeded > 0:
            mean_abs = np.mean(np.abs(weighted_shap), axis=0)
            norm_tree = self._normalize(mean_abs)
        else:
            norm_tree = np.zeros(len(self.feature_names), dtype=np.float64)

        if self.tabnet_model is not None:
            try:
                if hasattr(self.tabnet_model, 'explain'):
                    res_explain, _ = self.tabnet_model.explain(X_np)
                elif hasattr(self.tabnet_model, 'model') and hasattr(self.tabnet_model.model, 'explain'):
                    res_explain, _ = self.tabnet_model.model.explain(X_np)
                else:
                    res_explain = None

                if res_explain is not None:
                    mean_tabnet = np.mean(res_explain, axis=0)
                    norm_tabnet = self._normalize(mean_tabnet)
                    unified_importance = (norm_tree + norm_tabnet) / 2.0
                else:
                    norm_tabnet = np.zeros_like(norm_tree)
                    unified_importance = norm_tree
            except Exception as e:
                warnings.warn(f"TabNet model failed during global importance: {e}", RuntimeWarning)
                models_failed.append('tab')
                norm_tabnet = np.zeros_like(norm_tree)
                unified_importance = norm_tree
        else:
            norm_tabnet = np.zeros_like(norm_tree)
            unified_importance = norm_tree

        unified_importance = self._normalize(unified_importance)
        return unified_importance, norm_tree, norm_tabnet

    # Strict column mapping dictionary to prevent geography / area columns from leaking
    COLUMN_CATEGORY_MAPPING = {
        # Environmental Clearance
        "forest_clearance_status": "environmental_clearance",
        "forest_clearance_status_risk_score": "environmental_clearance",
        "terrain_type": "environmental_clearance",
        "environmental": "environmental_clearance",
        "eco_sensitive": "environmental_clearance",

        # Socio-Legal Disputes
        "affected_families_count": "socio_legal_disputes",
        "title_dispute_rate_percent": "socio_legal_disputes",
        "local_protest_flag": "socio_legal_disputes",
        "compensation_multiplier_demand": "socio_legal_disputes",
        "sia_approval_status": "socio_legal_disputes",
        "sia_approval_status_risk_score": "socio_legal_disputes",
        "population_density": "socio_legal_disputes",

        # Financial Disbursement
        "estimated_cost_inr_crore": "financial_disbursement",
        "fund_disbursement_percent": "financial_disbursement",
        "financial_density": "financial_disbursement",
        "financial_burn_rate_to_date": "financial_disbursement",
        "C_r": "financial_disbursement",
        "F_r": "financial_disbursement",

        # Administrative Workflow & Geography (strictly non-environmental)
        "project_id": "administrative_workflow",
        "project_type": "administrative_workflow",
        "state": "administrative_workflow",
        "district": "administrative_workflow",
        "land_area_hectares": "administrative_workflow",
        "land_area_log": "administrative_workflow",
        "project_start_year": "administrative_workflow",
        "project_age_years": "administrative_workflow",
        "state_project_type": "administrative_workflow",
        "H_r": "administrative_workflow",
        "W_r": "administrative_workflow",
        "P_r": "administrative_workflow",
    }

    def _compute_category_breakdown(self, feature_scores):
        """
        Maps feature impact scores to explicit risk categories using a strict column
        mapping dictionary to ensure geography features never leak into environmental clearance.
        Guarantees that categories sum cleanly to 1.0.
        """
        breakdown = {
            "environmental_clearance": 0.0,
            "socio_legal_disputes": 0.0,
            "financial_disbursement": 0.0,
            "administrative_workflow": 0.0
        }

        for i, feat in enumerate(self.feature_names):
            score = float(feature_scores[i])
            category = self.COLUMN_CATEGORY_MAPPING.get(feat)
            if category is None:
                feat_lower = feat.lower()
                if any(w in feat_lower for w in ['forest', 'clearance', 'environmental', 'terrain']):
                    category = "environmental_clearance"
                elif any(w in feat_lower for w in ['family', 'families', 'protest', 'dispute', 'compensation', 'sia']):
                    category = "socio_legal_disputes"
                elif any(w in feat_lower for w in ['cost', 'fund', 'disbursement', 'financial']):
                    category = "financial_disbursement"
                else:
                    category = "administrative_workflow"

            breakdown[category] += score

        total = sum(breakdown.values())
        if total > 0:
            for k in breakdown:
                breakdown[k] = round(breakdown[k] / total, 4)
            # Ensure exact sum of 1.0
            diff = round(1.0 - sum(breakdown.values()), 4)
            breakdown["administrative_workflow"] = round(breakdown["administrative_workflow"] + diff, 4)
        else:
            breakdown["administrative_workflow"] = 1.0

        return breakdown

    def _compute_local_fallback(self, row_df):
        """
        Per-instance local perturbation fallback (Section 3).
        Replaces static training-time feature importance with live, input-specific
        sensitivity measurements. Perturbs candidate features for this specific row,
        measures the actual change in the ensemble's output, and derives dynamic,
        directionally honest attributions.
        """
        p_base = float(self.hybrid_predictor.predict(row_df)['delay_probability'][0])

        # Underlying stacker for high-resolution perturbation when calibrated probability saturates
        stacker = None
        if hasattr(self.hybrid_predictor, 'calibrated_classifier') and hasattr(
            self.hybrid_predictor.calibrated_classifier, 'calibrated_classifiers_'
        ) and len(self.hybrid_predictor.calibrated_classifier.calibrated_classifiers_) > 0:
            stacker = self.hybrid_predictor.calibrated_classifier.calibrated_classifiers_[0].estimator
        elif hasattr(self.hybrid_predictor, 'classifier'):
            stacker = self.hybrid_predictor.classifier

        p_base_raw = float(stacker.predict_proba(row_df)[0, 1]) if stacker is not None else p_base

        # Reference medians from background data or zero baseline
        neutral_medians = {}
        if self.background_data is not None:
            neutral_medians = self.background_data.median(numeric_only=True).to_dict()

        # Candidate features: top features from cached global importance or column list
        if self._cached_global_importance:
            candidate_features = [item["feature"] for item in self._cached_global_importance[:8]]
        else:
            candidate_features = [f for f in self.feature_names if f in self.COLUMN_CATEGORY_MAPPING][:8]

        row_local_deltas = {}
        for feat in candidate_features:
            if feat not in row_df.columns:
                continue
            orig_val = float(row_df[feat].values[0])
            ref_val = neutral_medians.get(feat, 0.0)

            # Deletion/insertion perturbation
            row_pert = row_df.copy()
            if abs(orig_val - ref_val) > 1e-4:
                row_pert[feat] = ref_val
            else:
                # Value matches reference median; apply local directional step
                step = 0.05 * (abs(orig_val) + 1.0)
                row_pert[feat] = orig_val + step

            pred_pert = self.hybrid_predictor.predict(row_pert)
            p_pert = float(pred_pert['delay_probability'][0])
            delta = p_base - p_pert

            # If calibrated probability saturates into identical value, measure on continuous stacker
            if abs(delta) < 1e-4 and stacker is not None:
                p_pert_raw = float(stacker.predict_proba(row_pert)[0, 1])
                delta = p_base_raw - p_pert_raw

            row_local_deltas[feat] = delta

        # Construct full attribution vector across all features
        fallback_attribution = np.zeros(len(self.feature_names), dtype=np.float64)
        for i, feat in enumerate(self.feature_names):
            if feat in row_local_deltas:
                fallback_attribution[i] = row_local_deltas[feat]

        return fallback_attribution

    def explain(self, X, allow_fallback=None):
        """
        Primary explanation method. Handles single rows and batch inputs.
        Returns a single dictionary payload for single instances, or a list of
        dictionary payloads for batch inputs. Surfaces warnings for failing
        base models, tracks models_failed in returned payloads, and raises
        RuntimeError if all models fail unless fallback is enabled.
        """
        is_single = isinstance(X, dict) or isinstance(X, pd.Series) or (
            isinstance(X, (pd.DataFrame, np.ndarray)) and len(X) == 1
        )

        X_df = self._coerce_input(X)
        N = len(X_df)
        X_np = X_df.values.astype(np.float32)

        # 1. Compute Meta-Learner-Weighted TreeSHAP
        weighted_shap = np.zeros((N, len(self.feature_names)), dtype=np.float64)
        models_succeeded = 0
        models_failed = []

        for name, model in self.tree_models.items():
            try:
                shap_vals, _ = self._compute_model_shap(name, model, X_df)
                if shap_vals.shape == (N, len(self.feature_names)):
                    coef = self.meta_coefficients.get(name, 1.0)
                    weighted_shap += coef * shap_vals
                    models_succeeded += 1
            except Exception as e:
                warnings.warn(f"Base model '{name}' failed during attribution: {e}", RuntimeWarning)
                models_failed.append(name)

        fallback_enabled = allow_fallback if allow_fallback is not None else self.allow_fallback
        if len(self.tree_models) > 0 and models_succeeded == 0 and not fallback_enabled:
            raise RuntimeError(
                f"All tree models {list(self.tree_models.keys())} failed during TreeSHAP attribution: {models_failed}. "
                "Fallback path is not enabled."
            )

        if models_succeeded > 0:
            ensemble_shap_raw = weighted_shap
            used_fallback = False
        else:
            # Fallback path: compute per-instance local perturbation attributions
            ensemble_shap_raw = np.zeros((N, len(self.feature_names)), dtype=np.float64)
            for k in range(N):
                row_k = X_df.iloc[[k]]
                ensemble_shap_raw[k] = self._compute_local_fallback(row_k)
            used_fallback = True

        # 2. Compute TabNet Attention (honest check)
        tabnet_attentions = None
        if self.tabnet_model is not None:
            try:
                if hasattr(self.tabnet_model, 'explain'):
                    res_explain, _ = self.tabnet_model.explain(X_np)
                    tabnet_attentions = res_explain
                elif hasattr(self.tabnet_model, 'model') and hasattr(self.tabnet_model.model, 'explain'):
                    res_explain, _ = self.tabnet_model.model.explain(X_np)
                    tabnet_attentions = res_explain
            except Exception as e:
                warnings.warn(f"TabNet model failed during attribution: {e}", RuntimeWarning)
                models_failed.append('tab')
                tabnet_attentions = None

        if tabnet_attentions is None:
            tabnet_attentions = np.zeros((N, len(self.feature_names)), dtype=np.float64)

        # 3. Global Importance Approx (cached or computed from distribution)
        if self._cached_global_importance is not None:
            global_importance_list = self._cached_global_importance
        elif self.background_data is not None:
            self._compute_and_cache_global_importance()
            global_importance_list = self._cached_global_importance or []
        else:
            if N > 1:
                batch_global, _, _ = self.get_global_importance(X_df, allow_fallback=fallback_enabled)
                global_importance_list = [
                    {"feature": self.feature_names[i], "importance": float(batch_global[i])}
                    for i in range(len(self.feature_names))
                ]
                global_importance_list = sorted(global_importance_list, key=lambda x: x["importance"], reverse=True)
            else:
                global_importance_list = [
                    {"feature": feat, "importance": float(1.0 / len(self.feature_names))}
                    for feat in self.feature_names
                ]

        # 4. Construct payload per sample
        payloads = []
        for k in range(N):
            sample_shap_raw = ensemble_shap_raw[k]
            sample_attn = tabnet_attentions[k]
            sample_values = X_np[k]

            # Proportional attribution preserving cross-row variance and bounded in [0, 1]
            total_abs = np.sum(np.abs(sample_shap_raw)) + 1e-9
            norm_local_shap = np.abs(sample_shap_raw) / total_abs
            norm_local_attn = self._normalize(sample_attn)

            if self.tabnet_model is not None and np.any(sample_attn > 0):
                unified_local_impact = (norm_local_shap + norm_local_attn) / 2.0
            else:
                unified_local_impact = norm_local_shap

            sorted_indices = np.argsort(unified_local_impact)[::-1]

            # Detailed local explanation
            local_explanation = []
            for i in range(len(self.feature_names)):
                local_explanation.append({
                    "feature": self.feature_names[i],
                    "value": float(sample_values[i]),
                    "shap_impact": float(sample_shap_raw[i]),
                    "attention": float(sample_attn[i]),
                    "unified_score": float(unified_local_impact[i])
                })

            # Top risk drivers (up to 5)
            risk_drivers = []
            for idx in sorted_indices[:5]:
                shap_score = norm_local_shap[idx]
                attn_score = norm_local_attn[idx]
                if used_fallback:
                    source = "Fallback_Heuristic"
                elif self.tabnet_model is not None and attn_score > shap_score:
                    source = "TabNet_Attention"
                else:
                    source = "TreeSHAP"

                direction = "increases_delay" if sample_shap_raw[idx] > 0 else "decreases_delay"

                risk_drivers.append({
                    "feature": self.feature_names[idx],
                    "impact_score": float(unified_local_impact[idx]),
                    "direction": direction,
                    "source": source
                })

            # Category Breakdown
            category_breakdown = self._compute_category_breakdown(unified_local_impact)

            payloads.append({
                "risk_drivers": risk_drivers,
                "category_breakdown": category_breakdown,
                "local_explanation_full": local_explanation,
                "global_importance_approx": global_importance_list,
                "models_failed": list(models_failed)
            })

        return payloads[0] if is_single else payloads

    def get_local_explanation(self, X_local):
        """Backwards-compatible alias for single or batch explanation."""
        return self.explain(X_local)

    def generate_json_payload(self, X_local):
        """Generates JSON formatted string of explanation payload."""
        payload = self.explain(X_local)
        return json.dumps(payload, indent=2)
