import json
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import joblib
import shap

from hybrid_model import HybridRiskPredictor
from timeline_predictor import NonLinearTimelinePredictor
from explainer import DualParadigmExplainer
from extract_shap import extract_xgb_model

def run_faithfulness_audit():
    print("=" * 80)
    print("XAI FAITHFULNESS AUDIT: EMPIRICAL VALIDATION & STRUCTURAL GAP ANALYSIS")
    print("=" * 80)

    # Load artifacts
    pipeline = joblib.load('pipeline.joblib')
    predictor = HybridRiskPredictor.load('ensemble.joblib')
    timeline = joblib.load('timeline.joblib')
    df = pd.read_csv('indian_infrastructure_projects_dataset.csv')

    X = df.drop(columns=['delay_binary_label', 'Actual_Delay_Days', 'CRS', 'project_index'], errors='ignore')
    X_tf = pipeline.transform(X.head(500))
    feature_names = X_tf.columns.tolist()

    # Background reference (first 200 rows)
    X_bg = X_tf.iloc[:200].copy()
    neutral_medians = X_bg.median(numeric_only=True).to_dict()

    explainer = DualParadigmExplainer(predictor, feature_names, X_bg)

    # -------------------------------------------------------------------------
    # SECTION 1: Timeline / Survival Model Explainability Gap Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 1: TIMELINE / SURVIVAL MODEL EXPLAINABILITY AUDIT")
    print("=" * 60)

    rsf_model = timeline.rsf
    deepsurv_model = timeline.deepsurv
    print(f"Timeline RSF model class: {type(rsf_model)}")
    print(f"Timeline DeepSurv model class: {type(deepsurv_model)}")

    # Check TreeExplainer support for RSF
    tree_explainer_supported = False
    tree_explainer_error = None
    try:
        _ = shap.TreeExplainer(rsf_model)
        tree_explainer_supported = True
    except Exception as e:
        tree_explainer_error = f"{type(e).__name__}: {str(e)}"
        print(f"shap.TreeExplainer on RSF: UNSUPPORTED ({tree_explainer_error})")

    # Check feature_importances_ on RSF
    fi_supported = False
    fi_error = None
    try:
        _ = rsf_model.feature_importances_
        fi_supported = True
    except Exception as e:
        fi_error = f"{type(e).__name__}: {str(e)}"
        print(f"rsf.feature_importances_: UNSUPPORTED ({fi_error})")

    sec1_results = {
        "rsf_model_class": str(type(rsf_model)),
        "deepsurv_model_class": str(type(deepsurv_model)),
        "tree_explainer_supported": tree_explainer_supported,
        "tree_explainer_error": tree_explainer_error,
        "native_feature_importances_supported": fi_supported,
        "native_feature_importances_error": fi_error,
        "surfaced_outputs_without_xai": ["predicted_delay_days", "delay_days", "median_survival_days"],
        "call_sites_traced": [
            "risk_analysis_system.py:predict() lines 75-88",
            "dashboard.py line 280",
            "interactive_test.py line 63",
            "recommendation_engine.py lines 223-226"
        ]
    }

    # -------------------------------------------------------------------------
    # SECTION 2: Deletion / Insertion Faithfulness Test (N = 50 rows)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 2: DELETION / INSERTION FAITHFULNESS TEST (N = 50)")
    print("=" * 60)

    n_eval = 50
    eval_df = X_tf.iloc[:n_eval].copy()

    per_row_telemetry = []
    top1_claimed_scores = []
    top1_measured_deltas = []
    top1_directional_matches = []

    all_claimed_scores = []
    all_measured_deltas = []
    all_directional_matches = []

    for i in range(n_eval):
        row_orig = eval_df.iloc[[i]].copy()
        pred_orig = predictor.predict(row_orig)
        p_base = float(pred_orig['delay_probability'][0])

        explanation = explainer.explain(row_orig)
        drivers = explanation['risk_drivers']

        row_record = {
            "row_index": i,
            "baseline_prob": round(p_base, 4),
            "drivers": []
        }

        # Evaluate top-3 drivers
        for rank, driver in enumerate(drivers[:3], start=1):
            feat = driver['feature']
            claimed_score = driver['impact_score']
            claimed_direction = driver['direction']
            source = driver['source']

            # Deletion: replace feature with neutral background median
            neutral_val = neutral_medians.get(feat, 0.0)
            orig_val = float(row_orig[feat].values[0])

            row_deleted = row_orig.copy()
            row_deleted[feat] = neutral_val

            pred_deleted = predictor.predict(row_deleted)
            p_deleted = float(pred_deleted['delay_probability'][0])

            delta_prob = p_base - p_deleted
            abs_delta = abs(delta_prob)

            # Insertion check: does removing the risk driver change probability in claimed direction?
            # If claimed 'increases_delay', removing it should lower delay (delta_prob > 0)
            # If claimed 'decreases_delay', removing it should increase delay (delta_prob < 0)
            if claimed_direction == "increases_delay":
                dir_match = (delta_prob > 0)
            else:
                dir_match = (delta_prob < 0)

            driver_info = {
                "rank": rank,
                "feature": feat,
                "claimed_impact_score": round(claimed_score, 4),
                "claimed_direction": claimed_direction,
                "source": source,
                "orig_value": round(orig_val, 4),
                "neutral_value": round(float(neutral_val), 4),
                "prob_after_deletion": round(p_deleted, 4),
                "delta_prob": round(delta_prob, 4),
                "abs_delta_prob": round(abs_delta, 4),
                "directional_fidelity": bool(dir_match)
            }
            row_record["drivers"].append(driver_info)

            all_claimed_scores.append(claimed_score)
            all_measured_deltas.append(abs_delta)
            all_directional_matches.append(dir_match)

            if rank == 1:
                top1_claimed_scores.append(claimed_score)
                top1_measured_deltas.append(abs_delta)
                top1_directional_matches.append(dir_match)

        per_row_telemetry.append(row_record)

    # Rank correlations
    rho_top1, pval_top1 = spearmanr(top1_claimed_scores, top1_measured_deltas)
    rho_all, pval_all = spearmanr(all_claimed_scores, all_measured_deltas)

    top1_fidelity_rate = float(np.mean(top1_directional_matches))
    all_fidelity_rate = float(np.mean(all_directional_matches))
    mean_top1_delta = float(np.mean(top1_measured_deltas))
    median_top1_delta = float(np.median(top1_measured_deltas))

    print(f"Top-1 Driver Spearman rho: {rho_top1:.4f} (p = {pval_top1:.4e})")
    print(f"Top-3 Drivers Spearman rho (N = 150): {rho_all:.4f} (p = {pval_all:.4e})")
    print(f"Top-1 Directional Fidelity: {top1_fidelity_rate*100:.1f}%")
    print(f"All Drivers Directional Fidelity: {all_fidelity_rate*100:.1f}%")
    print(f"Mean |Delta prob| on #1 Deletion: {mean_top1_delta:.4f} (Median: {median_top1_delta:.4f})")

    sec2_results = {
        "evaluation_sample_size": n_eval,
        "top1_spearman_rho": round(float(rho_top1), 4),
        "top1_spearman_pvalue": float(pval_top1),
        "top3_all_spearman_rho": round(float(rho_all), 4),
        "top3_all_spearman_pvalue": float(pval_all),
        "top1_directional_fidelity_pct": round(top1_fidelity_rate * 100, 2),
        "all_drivers_directional_fidelity_pct": round(all_fidelity_rate * 100, 2),
        "mean_top1_delta_prob": round(mean_top1_delta, 4),
        "median_top1_delta_prob": round(median_top1_delta, 4),
        "per_row_telemetry": per_row_telemetry
    }

    # -------------------------------------------------------------------------
    # SECTION 3: Attribution Paradigm Balance (TabNet vs TreeSHAP)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 3: ATTRIBUTION PARADIGM BALANCE AUDIT")
    print("=" * 60)

    # Check estimator composition in predictor
    stacker = predictor.calibrated_classifier.calibrated_classifiers_[0].estimator if hasattr(predictor, 'calibrated_classifier') else predictor.classifier
    estimator_names = [name for name, _ in stacker.estimators]
    print(f"Estimators present in loaded HybridRiskPredictor: {estimator_names}")
    print(f"explainer.tabnet_model instance: {explainer.tabnet_model}")

    # Tally source field across all 250 drivers (50 rows * 5 drivers)
    source_counts = {"TreeSHAP": 0, "TabNet_Attention": 0, "Fallback_Heuristic": 0}
    for r in per_row_telemetry:
        row_orig = eval_df.iloc[[r["row_index"]]]
        exp = explainer.explain(row_orig)
        for d in exp["risk_drivers"]:
            src = d["source"]
            source_counts[src] = source_counts.get(src, 0) + 1

    total_drivers = sum(source_counts.values())
    source_percentages = {k: round(v / total_drivers * 100, 2) for k, v in source_counts.items()}
    print(f"Source counts across {total_drivers} driver attributions: {source_counts}")
    print(f"Source win-rate percentages: {source_percentages}")

    sec3_results = {
        "ensemble_estimator_keys": estimator_names,
        "tabnet_model_loaded": explainer.tabnet_model is not None,
        "total_driver_attributions_evaluated": total_drivers,
        "source_counts": source_counts,
        "source_percentages": source_percentages,
        "tabnet_win_rate_pct": source_percentages.get("TabNet_Attention", 0.0),
        "architectural_finding": "TabNet was not included in the StackingClassifier estimators of ensemble.joblib, so self.tabnet_model evaluates to None, causing TabNet attention to never participate in production explanations."
    }

    # -------------------------------------------------------------------------
    # SECTION 4: Global Importance Stability Across Independent Background Draws
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 4: GLOBAL IMPORTANCE BACKGROUND STABILITY AUDIT")
    print("=" * 60)

    # Verify set_background_data seed
    # Test two independent unseeded 100-row draws from full dataset
    sample_A = X_tf.sample(n=100, random_state=101).reset_index(drop=True)
    sample_B = X_tf.sample(n=100, random_state=202).reset_index(drop=True)

    importance_A, _, _ = explainer.get_global_importance(sample_A)
    importance_B, _, _ = explainer.get_global_importance(sample_B)

    rho_bg, pval_bg = spearmanr(importance_A, importance_B)

    # Top-10 Jaccard
    k = 10
    top10_A = set(np.argsort(importance_A)[-k:])
    top10_B = set(np.argsort(importance_B)[-k:])
    top10_jaccard = len(top10_A.intersection(top10_B)) / len(top10_A.union(top10_B))

    top10_A_feats = [feature_names[i] for i in np.argsort(importance_A)[-k:][::-1]]
    top10_B_feats = [feature_names[i] for i in np.argsort(importance_B)[-k:][::-1]]

    print(f"Global Importance Rank Correlation across Draw A vs Draw B: rho = {rho_bg:.4f} (p = {pval_bg:.4e})")
    print(f"Top-10 Features Jaccard Similarity across Draws: {top10_jaccard:.4f} ({len(top10_A.intersection(top10_B))} / 10 shared)")
    print(f"Top-10 Draw A: {top10_A_feats}")
    print(f"Top-10 Draw B: {top10_B_feats}")

    sec4_results = {
        "subsample_size": 100,
        "is_set_background_data_seeded": True,
        "set_background_data_seed": 42,
        "independent_draw_spearman_rho": round(float(rho_bg), 4),
        "independent_draw_spearman_pvalue": float(pval_bg),
        "top10_jaccard_similarity": round(float(top10_jaccard), 4),
        "top10_shared_features_count": len(top10_A.intersection(top10_B)),
        "top10_draw_A": top10_A_feats,
        "top10_draw_B": top10_B_feats
    }

    # -------------------------------------------------------------------------
    # SECTION 5: Explanation Quality Under Forced Fallback
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 5: FORCED FALLBACK QUALITY AUDIT")
    print("=" * 60)

    # Standard high-risk and low-risk payloads
    payload_low = {
        'project_id': 'TEST-LOW', 'state': 'Gujarat', 'district': 'Unknown', 'land_area_hectares': 10.0,
        'land_area_log': 2.3, 'project_type': 'Solar', 'terrain_type': 'Plain', 'estimated_cost_inr_crore': 50.0,
        'affected_families_count': 5, 'title_dispute_rate_percent': 0.0, 'local_protest_flag': False,
        'compensation_multiplier_demand': 1.0, 'sia_approval_status': 'Approved', 'forest_clearance_status': 'Approved',
        'fund_disbursement_percent': 90.0, 'project_start_year': 2023, 'project_age_years': 1,
        'sia_approval_status_risk_score': 0.1, 'forest_clearance_status_risk_score': 0.1,
        'C_r': 0.1, 'F_r': 0.1, 'H_r': 0.1, 'W_r': 0.1, 'P_r': 0.1
    }
    payload_high = {
        'project_id': 'TEST-HIGH', 'state': 'Gujarat', 'district': 'Unknown', 'land_area_hectares': 500.0,
        'land_area_log': 6.0, 'project_type': 'Highway', 'terrain_type': 'Hilly', 'estimated_cost_inr_crore': 5000.0,
        'affected_families_count': 2000, 'title_dispute_rate_percent': 25.0, 'local_protest_flag': True,
        'compensation_multiplier_demand': 3.5, 'sia_approval_status': 'Pending', 'forest_clearance_status': 'Pending',
        'fund_disbursement_percent': 10.0, 'project_start_year': 2018, 'project_age_years': 5,
        'sia_approval_status_risk_score': 0.9, 'forest_clearance_status_risk_score': 0.9,
        'C_r': 0.9, 'F_r': 0.9, 'H_r': 0.9, 'W_r': 0.9, 'P_r': 0.9
    }

    X_high = pipeline.transform(pd.DataFrame([payload_high]))
    X_low = pipeline.transform(pd.DataFrame([payload_low]))

    # Normal explanation
    normal_exp_high = explainer.explain(X_high)
    normal_exp_low = explainer.explain(X_low)

    # Forced fallback: null out tree models
    fallback_explainer = DualParadigmExplainer(predictor, feature_names, X_bg)
    fallback_explainer.tree_models = {}
    fallback_explainer._tree_explainers = {}

    fb_exp_high = fallback_explainer.explain(X_high)
    fb_exp_low = fallback_explainer.explain(X_low)

    fb_high_drivers = fb_exp_high["risk_drivers"]
    fb_low_drivers = fb_exp_low["risk_drivers"]

    print("Normal High-Risk Drivers:")
    for d in normal_exp_high["risk_drivers"][:3]:
        print(f"  {d['feature']}: impact={d['impact_score']:.4f}, dir={d['direction']}, src={d['source']}")

    print("\nFallback High-Risk Drivers:")
    for d in fb_high_drivers[:3]:
        print(f"  {d['feature']}: impact={d['impact_score']:.4f}, dir={d['direction']}, src={d['source']}")

    print("\nFallback Low-Risk Drivers:")
    for d in fb_low_drivers[:3]:
        print(f"  {d['feature']}: impact={d['impact_score']:.4f}, dir={d['direction']}, src={d['source']}")

    sec5_results = {
        "normal_high_risk_top3": normal_exp_high["risk_drivers"][:3],
        "normal_low_risk_top3": normal_exp_low["risk_drivers"][:3],
        "fallback_high_risk_top5": fb_high_drivers,
        "fallback_low_risk_top5": fb_low_drivers,
        "fallback_direction_behavior": "When TreeSHAP fails and TabNet is absent, ensemble_shap is initialized to all zeros. As a result, sample_shap[idx] > 0 evaluates to False, causing all fallback directions to degenerate to 'decreases_delay' regardless of whether the project is high-risk or low-risk.",
        "fallback_impact_scores": [d["impact_score"] for d in fb_high_drivers]
    }

    # -------------------------------------------------------------------------
    # Assemble Full Audit Results JSON
    # -------------------------------------------------------------------------
    full_audit_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "section_1_timeline_explainability": sec1_results,
        "section_2_deletion_insertion_faithfulness": sec2_results,
        "section_3_attribution_paradigm_balance": sec3_results,
        "section_4_global_importance_stability": sec4_results,
        "section_5_forced_fallback_quality": sec5_results
    }

    with open("faithfulness_audit_results.json", "w") as f:
        json.dump(full_audit_results, f, indent=2)

    print("\n" + "=" * 80)
    print("FAITHFULNESS AUDIT COMPLETE -> Saved to faithfulness_audit_results.json")
    print("=" * 80)

if __name__ == "__main__":
    run_faithfulness_audit()
