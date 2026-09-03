import time
import joblib
import pandas as pd
import numpy as np

from hybrid_model import HybridRiskPredictor
from timeline_predictor import NonLinearTimelinePredictor
from explainer import DualParadigmExplainer
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
        self.recommendation_engine = RecommendationEngine()
        
        # In-memory LRU cache for predictions (hash of first column if single row)
        self._cache = {}
        
        if pipeline_path:
            self.pipeline = joblib.load(pipeline_path)
        if ensemble_path:
            self.hybrid_model = HybridRiskPredictor.load(ensemble_path)
        if timeline_path:
            self.timeline_predictor = NonLinearTimelinePredictor.load(timeline_path)
            
    def initialize_explainer(self, feature_names):
        """Initializes the DualParadigmExplainer once the model and features are known."""
        if self.hybrid_model and feature_names:
            self.explainer = DualParadigmExplainer(self.hybrid_model, feature_names)

    def predict(self, raw_data: pd.DataFrame, metadata: dict = None) -> dict:
        """
        Accepts a single row or batch dataframe.
        Returns a structured dictionary of results.
        """
        start_time = time.perf_counter()
        
        # Return from cache if single row and seen recently
        # A rudimentary hash using the values of the first row
        if len(raw_data) == 1:
            row_hash = hash(tuple(raw_data.iloc[0].values))
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

        # 3. Timeline Predictor
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
                "calibrated_risk_tier": risk_tier
            },
            "timeline": {
                "median_survival_days": float(median_survival),
                "risk_phase": risk_phase
            },
            "explanation": explainer_payload,
            "recommendations": recommendations,
            "metadata": metadata,
            "latency_ms": latency
        }
        
        if row_hash:
            # Manage simple cache size
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
