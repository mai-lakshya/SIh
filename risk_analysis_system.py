import time
import threading
import joblib
import pandas as pd
import numpy as np

from hybrid_model import HybridRiskPredictor
from timeline_predictor import NonLinearTimelinePredictor
from explainer import DualParadigmExplainer
from timeline_explainer import TimelinePermutationExplainer
from recommendation_engine import RecommendationEngine

class RiskAnalysisSystem:
    """
    Unified Orchestrator Class representing Phase 6 of the Risk Prediction System.
    Provides end-to-end predictions, explanations, timelines, and recommendations.
    """
    def __init__(self, pipeline_path=None, ensemble_path=None, timeline_path=None):
        self.pipeline = None
        self.hybrid_model = None
        self.timeline_predictor = None
        self.explainer = None
        self.timeline_explainer = None
        self.recommendation_engine = RecommendationEngine()
        
        # Thread safety lock for concurrent requests
        self._lock = threading.RLock()
        # In-memory LRU cache for predictions (hash of first column if single row)
        self._cache = {}
        
        if pipeline_path:
            self.pipeline = joblib.load(pipeline_path)
        if ensemble_path:
            self.hybrid_model = HybridRiskPredictor.load(ensemble_path)
        if timeline_path:
            self.timeline_predictor = NonLinearTimelinePredictor.load(timeline_path)
            
    def initialize_explainer(self, feature_names):
        """Initializes DualParadigmExplainer and TimelinePermutationExplainer."""
        if self.hybrid_model and feature_names:
            self.explainer = DualParadigmExplainer(self.hybrid_model, feature_names)
        if self.timeline_predictor and feature_names and self.timeline_explainer is None:
            self.timeline_explainer = TimelinePermutationExplainer(self.timeline_predictor, feature_names)
            # Precompute permutation importance if background dataset is available
            try:
                import os
                if os.path.exists('indian_infrastructure_projects_dataset.csv') and self.pipeline:
                    df = pd.read_csv('indian_infrastructure_projects_dataset.csv', nrows=50)
                    X_raw = df.drop(columns=['delay_binary_label', 'Actual_Delay_Days', 'CRS', 'project_index'], errors='ignore')
                    X_bg = self.pipeline.transform(X_raw)
                    events = df['delay_binary_label'].values.astype(bool)
                    times = df.get('Actual_Delay_Days', df['delay_binary_label'] * 90).replace(0, 365).values.astype(float)
                    self.timeline_explainer.fit(X_bg, events, times)
            except Exception:
                pass

    def predict(self, raw_data: pd.DataFrame, metadata: dict = None) -> dict:
        """
        Accepts a single row or batch dataframe.
        Returns a structured dictionary of results.
        """
        start_time = time.perf_counter()
        
        # Return from cache if single row and seen recently
        # A robust hash converting any nested lists/dicts to immutable tuples
        if len(raw_data) == 1:
            def _to_hashable(val):
                if isinstance(val, (list, tuple)):
                    return tuple(_to_hashable(x) for x in val)
                if isinstance(val, dict):
                    return tuple(sorted((k, _to_hashable(v)) for k, v in val.items()))
                if isinstance(val, np.ndarray):
                    return tuple(val.tolist())
                return val

            try:
                row_hash = hash(tuple(_to_hashable(v) for v in raw_data.iloc[0].values))
            except Exception:
                row_hash = None

            if row_hash is not None:
                with self._lock:
                    if row_hash in self._cache:
                        return self._cache[row_hash]
        else:
            row_hash = None

        # 1. Preprocessing
        try:
            X_proc = self.pipeline.transform(raw_data)
        except Exception as e:
            raise ValueError(f"Error during preprocessing: {e}")
            
        feature_names = list(X_proc.columns)
        if self.explainer is None:
            with self._lock:
                if self.explainer is None:
                    self.initialize_explainer(feature_names)

        # 2. Ensemble Predictions
        try:
            preds = self.hybrid_model.predict(X_proc, blend_monotonicity=True)
            delay_prob = preds['delay_probability'][0]
            crs = preds['crs'][0]
            delay_days = preds['delay_days'][0]
            
            risk_tier = "Critical" if crs > 75 else "High" if crs > 50 else "Medium" if crs > 25 else "Low"
        except Exception as e:
            raise ValueError(f"Error during hybrid model prediction: {e}")

        # 3. Timeline Predictor & Explainer
        try:
            median_times = self.timeline_predictor.get_dynamic_risk_threshold(X_proc)
            median_survival = median_times[0]
            
            # Phase is arbitrary logic based on predicted delay
            if delay_days < 90:
                risk_phase = "Immediate"
            elif delay_days < 180:
                risk_phase = "Short-term"
            else:
                risk_phase = "Long-term"

            # Timeline explanation rationale
            timeline_explanation = {}
            if self.timeline_explainer is not None:
                timeline_explanation = self.timeline_explainer.explain(X_proc.iloc[0:1])
        except Exception as e:
            raise ValueError(f"Error during survival prediction: {e}")

        # 4. Explanation
        try:
            explainer_payload = self.explainer.get_local_explanation(X_proc.iloc[0:1])
        except Exception as e:
            raise ValueError(f"Error during explainer generation: {e}")

        # 5. Recommendations
        try:
            # Map risk_drivers to the tuple format RecommendationEngine expects
            risk_drivers_formatted = [
                (rd['feature'], rd['impact_score']) for rd in explainer_payload['risk_drivers']
            ]
            recommendations = self.recommendation_engine.generate_recommendations(risk_drivers_formatted, metadata or {})
        except Exception as e:
            raise ValueError(f"Error during recommendation generation: {e}")
            
        latency = (time.perf_counter() - start_time) * 1000

        result = {
            "predictions": {
                "delay_probability": float(delay_prob),
                "crs": float(crs),
                "predicted_delay_days": float(delay_days),
                "delay_days": float(delay_days),
                "predicted_delay_rationale": timeline_explanation.get("rationale", ""),
                "risk_tier": risk_tier,
                "calibrated_risk_tier": risk_tier
            },
            "timeline": {
                "median_survival_days": float(median_survival),
                "risk_phase": risk_phase,
                "top_drivers": timeline_explanation.get("top_drivers", []),
                "feature_importance": timeline_explanation.get("feature_importance", []),
                "rationale": timeline_explanation.get("rationale", "")
            },
            "explanation": explainer_payload,
            "recommendations": recommendations,
            "metadata": metadata,
        }
        
        if row_hash:
            # Manage simple cache size under lock
            with self._lock:
                if len(self._cache) > 100:
                    self._cache.clear()
                self._cache[row_hash] = result

        return result

    def predict_batch(self, raw_data: pd.DataFrame, metadata_list: list = None) -> list:
        """
        Loops over rows and aggregates results for batch analysis.
        """
        results = []
        for i in range(len(raw_data)):
            row = raw_data.iloc[i:i+1]
            meta = metadata_list[i] if metadata_list else None
            results.append(self.predict(row, metadata=meta))
        return results

    def save(self, path: str):
        """Bundles the entire system into a single joblib file"""
        payload = {
            'pipeline': self.pipeline,
            'hybrid_model': self.hybrid_model,
            'timeline_predictor': self.timeline_predictor
        }
        joblib.dump(payload, path, compress=3)

    @classmethod
    def load(cls, path: str):
        """Loads a bundled system"""
        payload = joblib.load(path)
        system = cls()
        system.pipeline = payload['pipeline']
        system.hybrid_model = payload['hybrid_model']
        system.timeline_predictor = payload['timeline_predictor']
        return system
