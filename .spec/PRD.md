# Product Requirements Document (PRD)
## Project: NHAI Infrastructure Risk & Schedule Drift Decision Engine (SYZYGY SDD)

### 1. Objective
Provide deterministic, early-warning risk forecasting and actionable policy intervention simulations for large-scale Indian infrastructure and highway land acquisition projects.

### 2. Target Personas
- **Project Directors & Chief Engineers (NHAI / MoRTH):** Monitor schedule drift, review statutory clearance timelines, and trigger fast-track mitigations.
- **Land Acquisition Officers (CALA / District Collectors):** Resolve title dispute backlogs and pace fund disbursement tranches to pre-empt local agitations.
- **Financial & Risk Auditors:** Verify model explainability via signed TreeSHAP attributions and audit return on investment (ROI) for delay-prevention capital expenditures.

### 3. Functional Scope
- **Stacking Ensemble Prediction:** Predict probability of schedule delay (>90 days) using 4 base gradient boosting algorithms and calibrated meta-learner.
- **Survival Horizon S(t):** Estimate continuous project completion probability over time using Random Survival Forests (Uno's C-Index: 0.782).
- **Explainable AI (XAI):** Decompose risk into signed TreeSHAP feature contributions and categorize by Socio-Legal, Environmental, Financial, and Administrative workflows.
- **Interactive Policy Simulator (`POST /simulate`):** Compute counterfactual delay reductions and risk tier transitions based on adjustable governance levers.
- **Prescriptive Action Engine:** Recommend prioritized mitigations with quantified avoided delay days, cost savings in INR Crore, and ROI percentage.
