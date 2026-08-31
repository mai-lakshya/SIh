import numpy as np
import pandas as pd
import shap
import json

class DualParadigmExplainer:
    def __init__(self, hybrid_predictor, feature_names):
        """
        Initializes the explainer with a fitted HybridRiskPredictor.
        Extracts base models from the stacking classifier.
        """
        self.hybrid_predictor = hybrid_predictor
        self.feature_names = feature_names
        self.tree_models = {}
        self.tabnet_model = None
        
        # Extract base estimators from StackingClassifier
        stacker = self.hybrid_predictor.classifier
        
        for name, estimator in stacker.estimators_:
            if name in ['lgb', 'xgb', 'cat']:
                self.tree_models[name] = estimator
            elif name == 'tab':
                self.tabnet_model = estimator
                
    def _normalize(self, arr):
        # Normalize array to [0, 1] range
        arr = np.array(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    def get_global_importance(self, X):
        """
        Computes mean absolute SHAP values and mean TabNet attention across all samples.
        """
        # 1. TreeSHAP
        tree_importances = []
        for name, model in self.tree_models.items():
            explainer = shap.TreeExplainer(model)
            # TreeExplainer might return a list for multiclass, we take positive class [1] if so
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Mean absolute SHAP across samples
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            tree_importances.append(mean_abs_shap)
            
        avg_tree_importance = np.mean(tree_importances, axis=0)
        
        # 2. TabNet Attention
        X_np = X.values if hasattr(X, 'values') else X
        X_np = X_np.astype(np.float32)
        
        # pytorch_tabnet explain() returns masks and dict of masks
        # masks is an array of size (n_samples, n_features)
        # res_explain, masks_dict = self.tabnet_model.model.explain(X_np)
        # res_explain is the aggregated mask.
        res_explain, masks = self.tabnet_model.model.explain(X_np)
        mean_tabnet_attention = np.mean(res_explain, axis=0)
        
        # Normalize both and combine (simple average)
        norm_tree = self._normalize(avg_tree_importance)
        norm_tabnet = self._normalize(mean_tabnet_attention)
        
        unified_importance = (norm_tree + norm_tabnet) / 2.0
        # Final normalization to ensure [0, 1]
        unified_importance = self._normalize(unified_importance)
        
        return unified_importance, norm_tree, norm_tabnet

    def get_local_explanation(self, X_local):
        """
        Computes SHAP and TabNet attention for a single instance (or batch).
        Returns a dictionary representing the JSON payload.
        """
        is_single = False
        if len(X_local) == 1:
            is_single = True
            
        tree_local_shaps = []
        for name, model in self.tree_models.items():
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_local)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            tree_local_shaps.append(shap_values)
            
        avg_local_shap = np.mean(tree_local_shaps, axis=0)
        
        X_np = X_local.values if hasattr(X_local, 'values') else X_local
        X_np = X_np.astype(np.float32)
        local_attention, _ = self.tabnet_model.model.explain(X_np)
        
        # Build payload for the first instance
        feature_values = X_np[0]
        local_shap_0 = avg_local_shap[0]
        local_attn_0 = local_attention[0]
        
        # Compute local unified impact
        norm_local_shap = self._normalize(np.abs(local_shap_0))
        norm_local_attn = self._normalize(local_attn_0)
        unified_local_impact = self._normalize((norm_local_shap + norm_local_attn) / 2.0)
        
        # Sort by unified impact for risk drivers
        sorted_indices = np.argsort(unified_local_impact)[::-1]
        
        local_explanation = []
        for i in range(len(self.feature_names)):
            local_explanation.append({
                "feature": self.feature_names[i],
                "value": float(feature_values[i]),
                "shap_impact": float(local_shap_0[i]),
                "attention": float(local_attn_0[i]),
                "unified_score": float(unified_local_impact[i])
            })
            
        risk_drivers = []
        for idx in sorted_indices[:5]:
            shap_score = norm_local_shap[idx]
            attn_score = norm_local_attn[idx]
            source = "TreeSHAP" if shap_score > attn_score else "TabNet_Attention"
            direction = "increases_delay" if local_shap_0[idx] > 0 else "decreases_delay"
            
            risk_drivers.append({
                "feature": self.feature_names[idx],
                "impact_score": float(unified_local_impact[idx]),
                "direction": direction,
                "source": source
            })
            
        # Calculate Category Breakdown
        env_cols = ['forest', 'terrain', 'area', 'clearance', 'environmental']
        soc_cols = ['family', 'families', 'protest', 'dispute', 'compensation', 'sia']
        fin_cols = ['cost', 'fund', 'disbursement', 'deficit', 'gap']
        
        breakdown = {
            "environmental_clearance": 0.0,
            "socio_legal_disputes": 0.0,
            "financial_disbursement": 0.0,
            "administrative_workflow": 0.0
        }
        
        for i in range(len(self.feature_names)):
            feat = self.feature_names[i].lower()
            score = float(unified_local_impact[i])
            assigned = False
            for c in env_cols:
                if c in feat:
                    breakdown["environmental_clearance"] += score
                    assigned = True
                    break
            if not assigned:
                for c in soc_cols:
                    if c in feat:
                        breakdown["socio_legal_disputes"] += score
                        assigned = True
                        break
            if not assigned:
                for c in fin_cols:
                    if c in feat:
                        breakdown["financial_disbursement"] += score
                        assigned = True
                        break
            if not assigned:
                breakdown["administrative_workflow"] += score
                
        # Normalize breakdown to sum to 1.0
        total = sum(breakdown.values())
        if total > 0:
            for k in breakdown:
                breakdown[k] = round(breakdown[k] / total, 2)
            
        # Get Global to include in payload
        global_unified, _, _ = self.get_global_importance(X_local) # Typically done on background set, but we approximate here
        
        global_importance = []
        for i in range(len(self.feature_names)):
            global_importance.append({
                "feature": self.feature_names[i],
                "importance": float(global_unified[i])
            })
            
        # Sort global
        global_importance = sorted(global_importance, key=lambda x: x["importance"], reverse=True)
            
        return {
            "risk_drivers": risk_drivers,
            "category_breakdown": breakdown,
            "local_explanation_full": local_explanation,
            "global_importance_approx": global_importance
        }

    def generate_json_payload(self, X_local):
        payload = self.get_local_explanation(X_local)
        return json.dumps(payload, indent=2)
