"""
Global pytest configuration and compatibility patches.
"""
import shap

# Safe compatibility patch for shap <= 0.49.1 with xgboost >= 3.0:
# In shap <= 0.49.1, XGBTreeModelLoader does float(learner_model_param["base_score"]),
# which crashes if XGBoost serializes base_score as a bracketed string like '[4E-1]' or '[0.4]'.
def _apply_shap_xgb_compatibility_patch():
    try:
        import shap.explainers._tree as tree_mod
        if hasattr(tree_mod, 'decode_ubjson_buffer'):
            _orig_decode = tree_mod.decode_ubjson_buffer

            def _clean_base_score_inplace(obj):
                if isinstance(obj, dict):
                    if "base_score" in obj:
                        bs = obj["base_score"]
                        if isinstance(bs, str) and (bs.startswith("[") and bs.endswith("]")):
                            import ast
                            try:
                                parsed = ast.literal_eval(bs)
                                if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                                    obj["base_score"] = str(parsed[0])
                                else:
                                    obj["base_score"] = str(parsed)
                            except Exception:
                                stripped = bs.strip("[] \t\r\n")
                                if stripped:
                                    obj["base_score"] = stripped.split(",")[0].strip()
                    for v in obj.values():
                        if isinstance(v, (dict, list)):
                            _clean_base_score_inplace(v)
                elif isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, (dict, list)):
                            _clean_base_score_inplace(item)

            def _patched_decode_ubjson_buffer(fd):
                jmodel = _orig_decode(fd)
                try:
                    _clean_base_score_inplace(jmodel)
                except Exception:
                    pass
                return jmodel

            tree_mod.decode_ubjson_buffer = _patched_decode_ubjson_buffer

            try:
                import shap.explainers.other._ubjson as ubjson_mod
                ubjson_mod.decode_ubjson_buffer = _patched_decode_ubjson_buffer
            except Exception:
                pass
    except Exception:
        pass


_apply_shap_xgb_compatibility_patch()


# Ensure unpickled LogisticRegression / LogisticRegressionCV models from scikit-learn >= 1.8
# work seamlessly when loaded under scikit-learn <= 1.7
def _patch_sklearn_multi_class():
    try:
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
        for cls in (LogisticRegression, LogisticRegressionCV):
            if not hasattr(cls, 'multi_class'):
                setattr(cls, 'multi_class', 'auto')
    except Exception:
        pass


_patch_sklearn_multi_class()
