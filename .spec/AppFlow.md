# Application Flow & State Machine
## Project: SYZYGY Decision Dashboard

### 1. User Journey
1. **Scenario Selection:** User selects one of four quick-load presets (Urban Expressway, Transit Metro, Eco Railway, Solar Park).
2. **Parameter Adjustment:** User adjusts Capex, land area, PAP count, title dispute rate, SIA status, forest clearance, or fund disbursement.
3. **Execution:** User triggers "Run Prediction & SHAP Explanation".
4. **Analysis:**
   - Review 4 high-level KPI cards (Delay Probability, Risk Tier, Predicted Delay Days, Median Survival).
   - Inspect Explainability tab for primary escalators/mitigators and signed TreeSHAP waterfall.
   - Analyze Survival Timeline Curve for 90d, 180d, and 365d milestone completion probabilities.
   - Open Policy Simulator tab to test interactive governance levers with real-time counterfactual output.
   - Review Prescriptive Action Plan for prioritized interventions with cost savings and ROI.
5. **Feedback Loop:** Interactive Sonner-style toasts confirm model execution latency and simulation updates.
