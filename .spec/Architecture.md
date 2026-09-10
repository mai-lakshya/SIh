# System Architecture
## Project: SYZYGY Topology & Flow

```mermaid
graph TD
    Client[Client Browser - SYZYGY Dashboard] -->|POST /predict, POST /simulate| API[FastAPI Gateway :8000]
    API --> Pipeline[Feature Pipeline & Transformers]
    Pipeline --> Stacker[Stacking Classifier Ensemble]
    Pipeline --> RSF[Random Survival Forest - Timeline]
    Stacker --> Meta[Calibrated Meta-Learner - Logit Space]
    Stacker --> SHAP[Meta-Learner-Weighted TreeSHAP]
    RSF --> Survival[Kaplan-Meier & IPCW Survival Curve]
    Meta --> RecEngine[Prescriptive Recommendation Engine]
    SHAP --> RecEngine
    RecEngine --> Response[Structured Decision Response JSON]
    Response --> Client
```

### Subagent Responsibilities
- **Inference Pipeline:** Executes feature scaling, derivation of statutory clearance risk scores, and ensemble inference in <25ms.
- **XAI Decomposition:** Computes signed local SHAP impacts, aligns categorical attributions, and provides plain-English executive summaries.
- **What-If Simulation:** Isolates counterfactual modifications and calculates schedule drift avoided.
