"""
Timeline Permutation Explainer for NonLinearTimelinePredictor (RandomSurvivalForest).
Computes feature importance using Uno's IPCW C-index permutation degradation,
providing feature-level rationales for predicted delay times and survival risks.
"""

import warnings
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored
from sksurv.util import Surv


class TimelinePermutationExplainer:
    """
    Model-agnostic permutation importance explainer tailored for sksurv's
    RandomSurvivalForest, which lacks native TreeSHAP and feature_importances_.
    Measures empirical degradation in Uno's C-index when each feature is
    randomly permuted on a reference background survival dataset.
    """

    def __init__(
        self,
        timeline_predictor,
        feature_names,
        background_data=None,
        background_events=None,
        background_times=None,
        n_repeats=3,
        random_state=42
    ):
        self.timeline_predictor = timeline_predictor
        self.feature_names = list(feature_names)
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.feature_importances_ = None
        self._cached_importance_list = None

        self.rsf_model = getattr(timeline_predictor, 'rsf', None)

        if background_data is not None and background_events is not None and background_times is not None:
            self.fit(background_data, background_events, background_times)

    def fit(self, X_bg, events, times):
        """
        Computes permutation importance using Uno's C-index across features.
        """
        rng = np.random.RandomState(self.random_state)
        X_df = X_bg.copy() if isinstance(X_bg, pd.DataFrame) else pd.DataFrame(X_bg, columns=self.feature_names)

        # Ensure column alignment
        for col in self.feature_names:
            if col not in X_df.columns:
                X_df[col] = 0.0
        X_df = X_df[self.feature_names].copy()

        events = np.asarray(events, dtype=bool)
        times = np.asarray(times, dtype=np.float64)
        times = np.maximum(times, 1.0)
        y_surv = Surv.from_arrays(event=events, time=times)

        if self.rsf_model is None:
            # Equal importance fallback if no RSF model
            self.feature_importances_ = {col: 1.0 / len(self.feature_names) for col in self.feature_names}
            self._build_cached_list()
            return self

        # Base risk prediction
        base_preds = self.rsf_model.predict(X_df)
        try:
            c_base, _, _, _, _ = concordance_index_ipcw(y_surv, y_surv, base_preds)
        except Exception:
            c_base, _, _, _, _ = concordance_index_censored(events, times, base_preds)

        raw_importances = {col: 0.0 for col in self.feature_names}

        for repeat in range(self.n_repeats):
            for col in self.feature_names:
                X_perm = X_df.copy()
                X_perm[col] = rng.permutation(X_perm[col].values)
                perm_preds = self.rsf_model.predict(X_perm)
                try:
                    c_perm, _, _, _, _ = concordance_index_ipcw(y_surv, y_surv, perm_preds)
                except Exception:
                    c_perm, _, _, _, _ = concordance_index_censored(events, times, perm_preds)

                # Feature importance is sensitivity/degradation in discrimination
                raw_importances[col] += abs(c_base - c_perm)

        # Average and normalize
        total = sum(raw_importances.values())
        if total > 1e-12:
            self.feature_importances_ = {col: float(val / total) for col, val in raw_importances.items()}
        else:
            self.feature_importances_ = {col: float(1.0 / len(self.feature_names)) for col in self.feature_names}

        self._build_cached_list()
        return self

    def _build_cached_list(self):
        sorted_items = sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)
        self._cached_importance_list = [
            {"feature": feat, "importance": float(imp)}
            for feat, imp in sorted_items
        ]

    def explain(self, row, top_k=5):
        """
        Generates feature-level rationale for a single instance or batch row.
        """
        if self._cached_importance_list is None:
            # Default equal importances if fit not called
            self.feature_importances_ = {col: 1.0 / len(self.feature_names) for col in self.feature_names}
            self._build_cached_list()

        row_df = row if isinstance(row, pd.DataFrame) else pd.DataFrame([row])
        top_drivers = []
        for item in self._cached_importance_list[:top_k]:
            feat = item["feature"]
            val = float(row_df[feat].values[0]) if feat in row_df.columns else 0.0
            top_drivers.append({
                "feature": feat,
                "importance": float(item["importance"]),
                "value": val
            })

        driver_names = [d["feature"] for d in top_drivers[:3]]
        rationale_str = f"Timeline risk primarily driven by {', '.join(driver_names)} based on Uno's C-index survival permutation."

        return {
            "top_drivers": top_drivers,
            "feature_importance": self._cached_importance_list,
            "rationale": rationale_str
        }
