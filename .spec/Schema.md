# Data Models & Schemas
## Project: SYZYGY Protocol Schemas

### ProjectPayload (JSON Schema)
```json
{
  "project_id": "string",
  "project_type": "string",
  "state": "string (required)",
  "terrain_type": "string",
  "estimated_cost_inr_crore": "number (> 0)",
  "land_area_hectares": "number (> 0)",
  "affected_families_count": "integer (>= 0)",
  "title_dispute_rate_percent": "number (0-100)",
  "sia_approval_status": "string (Pending | In_Progress | Approved | Rejected)",
  "forest_clearance_status": "string (Not_Required | Pending | In_Progress | Stage_1 | Stage_2)",
  "fund_disbursement_percent": "number (0-100)",
  "compensation_multiplier_demand": "number (>= 0)",
  "local_protest_flag": "boolean"
}
```

### SimulationPayload (JSON Schema)
```json
{
  "baseline": "ProjectPayload",
  "interventions": "object (key-value pairs of overrides)"
}
```
