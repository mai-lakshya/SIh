import re
import json
import warnings
import numpy as np
import pandas as pd
import shap


class DualParadigmExplainer:
    """
    Dual-Paradigm Explainability Engine combining TreeSHAP across heterogeneous
    gradient-boosted tree ensembles with neural attention mechanisms (TabNet).
    Supports single-instance inference and batch explanations with normalized
    attributions, robust data coercion, and graceful fallback paths.
    """

    def __init__(self, hybrid_predictor, feature_names, background_data=None):
        self.hybrid_predictor = hybrid_predictor
        self.feature_names = list(feature_names)
        self.tree_models = {}
        self.tabnet_model = None
        self.background_data = None
        self._cached_global_importance = None

        # Extract base estimators from StackingClassifier / CalibratedClassifierCV
        self._extract_models()

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
        """Extracts base estimators from the hybrid predictor."""
        stacker = None
        if hasattr(self.hybrid_predictor, 'calibrated_classifier') and hasattr(
            self.hybrid_predictor.calibrated_classifier, 'calibrated_classifiers_'
        ) and len(self.hybrid_predictor.calibrated_classifier.calibrated_classifiers_) > 0:
            stacker = self.hybrid_predictor.calibrated_classifier.calibrated_classifiers_[0].estimator
        elif hasattr(self.hybrid_predictor, 'classifier'):
            stacker = self.hybrid_predictor.classifier

        if stacker is not None:
            # Check estimators_ or named_estimators_
            if hasattr(stacker, 'estimators_') and hasattr(stacker, 'estimators'):
                names = [name for name, _ in stacker.estimators]
                for name, estimator in zip(names, stacker.estimators_):
                    if name in ['lgb', 'xgb', 'cat']:
                        self.tree_models[name] = estimator.model if hasattr(estimator, 'model') else estimator
                    elif name == 'tab':
                        self.tabnet_model = estimator.model if hasattr(estimator, 'model') else estimator
            elif hasattr(stacker, 'named_estimators_'):
                for name, estimator in stacker.named_estimators_.items():
                    if name in ['lgb', 'xgb', 'cat']:
                        self.tree_models[name] = estimator.model if hasattr(estimator, 'model') else estimator
                    elif name == 'tab':
                        self.tabnet_model = estimator.model if hasattr(estimator, 'model') else estimator

    def _normalize(self, arr):
        """Min-max normalize array to [0, 1] range."""
        arr = np.asarray(arr, dtype=np.float64)
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    def _l1_normalize_per_sample(self, matrix):
        """
        L1-normalizes attribution vector per sample, preserving sign and
        scaling disparate model attributions to a uniform scale.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim == 1:
            norm = np.sum(np.abs(matrix)) + 1e-9
            return matrix / norm
        norms = np.sum(np.abs(matrix), axis=1, keepdims=True) + 1e-9
        return matrix / norms

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
                # String / object column: try boolean mapping first, then numeric conversion
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

    def get_global_importance(self, X=None):
        """
        Computes mean absolute SHAP values and mean TabNet attention across all samples in X,
        normalized per model before aggregation.
        """
        if X is None:
            if self.background_data is not None:
                X = self.background_data
            else:
                raise ValueError("No background dataset provided for global importance calculation.")

        X_df = self._coerce_input(X)
        X_np = X_df.values.astype(np.float32)

        tree_importances = []
        for name, model in self.tree_models.items():
            try:
                explainer = shap.TreeExplainer(model)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    shap_values = explainer.shap_values(X_df)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                mean_abs = np.mean(np.abs(shap_values), axis=0)
                norm_mean_abs = self._l1_normalize_per_sample(mean_abs)
                if len(norm_mean_abs) == len(self.feature_names):
                    tree_importances.append(norm_mean_abs)
            except Exception as e:
                # Fallback to model feature importances if available
                if hasattr(model, 'feature_importances_'):
                    fi = np.asarray(model.feature_importances_, dtype=np.float64)
                    if len(fi) == len(self.feature_names):
                        tree_importances.append(self._l1_normalize_per_sample(fi))

        if len(tree_importances) > 0:
            avg_tree_importance = np.mean(tree_importances, axis=0)
            norm_tree = self._normalize(avg_tree_importance)
        else:
            norm_tree = np.zeros(len(self.feature_names), dtype=np.float64)

        if self.tabnet_model is not None and hasattr(self.tabnet_model, 'model'):
            try:
                res_explain, _ = self.tabnet_model.model.explain(X_np)
                mean_tabnet = np.mean(res_explain, axis=0)
                norm_tabnet = self._normalize(self._l1_normalize_per_sample(mean_tabnet))
                unified_importance = (norm_tree + norm_tabnet) / 2.0
            except Exception:
                norm_tabnet = np.zeros_like(norm_tree)
                unified_importance = norm_tree
        else:
            norm_tabnet = np.zeros_like(norm_tree)
            unified_importance = norm_tree

        unified_importance = self._normalize(unified_importance)
        return unified_importance, norm_tree, norm_tabnet

    def _compute_category_breakdown(self, feature_scores):
        """
        Maps feature impact scores to explicit risk categories using strict regex matching.
        Guarantees that categories sum cleanly to 1.0.
        """
        env_pattern = re.compile(r'\b(forest|clearance|environmental|terrain|tree|eco)\b', re.IGNORECASE)
        soc_pattern = re.compile(r'\b(family|families|protest|dispute|compensation|sia|population|rehabilitation|resettlement)\b', re.IGNORECASE)
        fin_pattern = re.compile(r'\b(cost|fund|disbursement|deficit|gap|financial|burn|budget)\b', re.IGNORECASE)

        breakdown = {
            "environmental_clearance": 0.0,
            "socio_legal_disputes": 0.0,
            "financial_disbursement": 0.0,
            "administrative_workflow": 0.0
        }

        for i, feat in enumerate(self.feature_names):
            score = float(feature_scores[i])
            feat_lower = feat.lower()

            if env_pattern.search(feat_lower):
                breakdown["environmental_clearance"] += score
            elif soc_pattern.search(feat_lower):
                breakdown["socio_legal_disputes"] += score
            elif fin_pattern.search(feat_lower):
                breakdown["financial_disbursement"] += score
            else:
                breakdown["administrative_workflow"] += score

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

    def explain(self, X):
        """
        Primary explanation method. Handles single rows and batch inputs.
        Returns a single dictionary payload for single instances, or a list of
        dictionary payloads for batch inputs.
        """
        is_single = isinstance(X, dict) or isinstance(X, pd.Series) or (
            isinstance(X, (pd.DataFrame, np.ndarray)) and len(X) == 1
        )

        X_df = self._coerce_input(X)
        N = len(X_df)
        X_np = X_df.values.astype(np.float32)

        # 1. Compute TreeSHAP with cross-model normalization
        tree_shaps_normalized = []
        tree_shaps_raw = []

        for name, model in self.tree_models.items():
            try:
                explainer = shap.TreeExplainer(model)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    shap_vals = explainer.shap_values(X_df)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                shap_vals = np.asarray(shap_vals, dtype=np.float64)
                if shap_vals.ndim == 2 and shap_vals.shape[1] == len(self.feature_names):
                    tree_shaps_raw.append(shap_vals)
                    # L1-normalize each sample's attribution vector to place models on identical scale
                    tree_shaps_normalized.append(self._l1_normalize_per_sample(shap_vals))
            except Exception:
                # TreeSHAP unavailable or failed
                pass

        # If all TreeSHAP attempts failed, provide graceful fallback
        if len(tree_shaps_normalized) > 0:
            ensemble_shap = np.mean(tree_shaps_normalized, axis=0)
            ensemble_shap_raw = np.mean(tree_shaps_raw, axis=0)
            used_fallback = False
        else:
            # Fallback: estimate from feature variance or static uniform proxy
            ensemble_shap = np.zeros((N, len(self.feature_names)), dtype=np.float64)
            ensemble_shap_raw = np.zeros((N, len(self.feature_names)), dtype=np.float64)
            used_fallback = True

        # 2. Compute TabNet Attention
        tabnet_attentions = None
        if self.tabnet_model is not None and hasattr(self.tabnet_model, 'model'):
            try:
                res_explain, _ = self.tabnet_model.model.explain(X_np)
                tabnet_attentions = self._l1_normalize_per_sample(res_explain)
            except Exception:
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
            # Compute across current batch distribution or uniform fallback
            if N > 1:
                batch_global, _, _ = self.get_global_importance(X_df)
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
            sample_shap = ensemble_shap[k]
            sample_shap_raw = ensemble_shap_raw[k]
            sample_attn = tabnet_attentions[k]
            sample_values = X_np[k]

            norm_local_shap = self._normalize(np.abs(sample_shap))
            norm_local_attn = self._normalize(sample_attn)

            if self.tabnet_model is not None and np.any(sample_attn > 0):
                unified_local_impact = self._normalize((norm_local_shap + norm_local_attn) / 2.0)
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
                elif shap_score >= attn_score:
                    source = "TreeSHAP"
                else:
                    source = "TabNet_Attention"

                direction = "increases_delay" if sample_shap[idx] > 0 else "decreases_delay"

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
                "global_importance_approx": global_importance_list
            })

        return payloads[0] if is_single else payloads

    def get_local_explanation(self, X_local):
        """Backwards-compatible alias for single or batch explanation."""
        return self.explain(X_local)

    def generate_json_payload(self, X_local):
        """Generates JSON formatted string of explanation payload."""
        payload = self.explain(X_local)
        return json.dumps(payload, indent=2)
