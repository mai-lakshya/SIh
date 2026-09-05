import logging
import math
import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)

# =====================================================================
# 1. UTC CALENDAR & LEAP YEAR HELPERS
# =====================================================================

def ensure_utc(dt: Optional[datetime.datetime]) -> Optional[datetime.datetime]:
    """Ensure a datetime is timezone-aware and converted to UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def add_days_utc(base_date: datetime.datetime, days: float) -> datetime.datetime:
    """
    Accurately add days to a UTC datetime, accounting for leap years
    (e.g., Feb 29 in 2024, 2028) and fractional day precision.
    """
    utc_base = ensure_utc(base_date)
    return utc_base + datetime.timedelta(days=float(days))


def date_diff_days_utc(start_dt: datetime.datetime, end_dt: datetime.datetime) -> float:
    """Compute exact fractional days between two datetimes in UTC."""
    s = ensure_utc(start_dt)
    e = ensure_utc(end_dt)
    return (e - s).total_seconds() / 86400.0


# =====================================================================
# 2. TASK & SCHEDULE DATA MODEL FOR DELAY-PREVENTION
# =====================================================================

@dataclass
class TaskNode:
    """
    Representation of a project milestone / task in the dependency network.
    Supports both day-duration offsets and concrete UTC calendar dates.
    """
    task_id: str
    name: str
    duration_days: float
    deadline_days: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    resources: Dict[str, float] = field(default_factory=dict)
    risk_driver: Optional[str] = None
    category: str = "General"
    
    # Concrete calendar dates (UTC)
    start_date: Optional[datetime.datetime] = None
    deadline_date: Optional[datetime.datetime] = None
    
    # Computed schedule metrics
    earliest_start: float = 0.0
    earliest_finish: float = 0.0
    latest_start: float = 0.0
    latest_finish: float = 0.0
    slack: float = 0.0  # Buffer time: LS - ES = LF - EF
    free_slack: float = 0.0
    is_critical: bool = False
    has_negative_buffer: bool = False
    cascading_impact_days: float = 0.0
    resource_saturated: bool = False

    # Computed calendar projections (UTC)
    earliest_start_date: Optional[datetime.datetime] = None
    earliest_finish_date: Optional[datetime.datetime] = None
    latest_start_date: Optional[datetime.datetime] = None
    latest_finish_date: Optional[datetime.datetime] = None
    calendar_buffer_days: Optional[float] = None

    def __post_init__(self):
        self.start_date = ensure_utc(self.start_date)
        self.deadline_date = ensure_utc(self.deadline_date)

    @property
    def total_float_days(self) -> float:
        return self.slack

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "duration_days": self.duration_days,
            "deadline_days": self.deadline_days,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "deadline_date": self.deadline_date.isoformat() if self.deadline_date else None,
            "dependencies": list(self.dependencies),
            "resources": dict(self.resources),
            "risk_driver": self.risk_driver,
            "category": self.category,
            "earliest_start": round(self.earliest_start, 2),
            "earliest_finish": round(self.earliest_finish, 2),
            "latest_start": round(self.latest_start, 2),
            "latest_finish": round(self.latest_finish, 2),
            "slack": round(self.slack, 2),
            "free_slack": round(self.free_slack, 2),
            "is_critical": self.is_critical,
            "has_negative_buffer": self.has_negative_buffer,
            "cascading_impact_days": round(self.cascading_impact_days, 2),
            "resource_saturated": self.resource_saturated,
            "earliest_start_date": self.earliest_start_date.isoformat() if self.earliest_start_date else None,
            "earliest_finish_date": self.earliest_finish_date.isoformat() if self.earliest_finish_date else None,
            "latest_finish_date": self.latest_finish_date.isoformat() if self.latest_finish_date else None,
            "calendar_buffer_days": round(self.calendar_buffer_days, 2) if self.calendar_buffer_days is not None else None
        }


# =====================================================================
# 3. CRITICAL CHAIN & CRITICAL PATH SCHEDULE ENGINE
# =====================================================================

class CriticalChainEngine:
    """
    High-performance Critical Path / Critical Chain Project Delay-Prevention Engine.
    Features:
      - Strict input sanitization and schema validation
      - O(V + E) linear-time topological DAG traversal (Kahn's algorithm)
      - Circular dependency detection with graceful fallback
      - Timezone-aware UTC date calculations across leap years
      - Edge case handling: Zero-delay states, cascading dependencies,
        negative buffer times, and resource saturation.
    """

    def __init__(self, resource_capacities: Optional[Dict[str, float]] = None):
        self.resource_capacities = resource_capacities or {
            "legal_officers": 5.0,
            "survey_teams": 4.0,
            "clearance_officers": 3.0,
            "disbursement_cells": 4.0
        }

    def validate_and_sanitize(self, raw_tasks: List[Any]) -> List[TaskNode]:
        """
        Sanitize and validate raw task objects into TaskNode instances.
        Guarantees non-negative durations, valid deadlines, and clean dependency references.
        """
        sanitized: List[TaskNode] = []
        seen_ids: Set[str] = set()

        for idx, item in enumerate(raw_tasks):
            if isinstance(item, TaskNode):
                t = item
            elif isinstance(item, dict):
                tid = str(item.get("task_id") or f"task_{idx+1}").strip()
                tname = str(item.get("name") or tid).strip()
                
                try:
                    dur = float(item.get("duration_days", 0.0))
                    if dur < 0:
                        logger.warning("Sanitizing negative duration %s for task %s to 0.0", dur, tid)
                        dur = 0.0
                except (ValueError, TypeError):
                    logger.warning("Invalid duration for task %s; defaulting to 0.0", tid)
                    dur = 0.0

                dl = None
                if item.get("deadline_days") is not None:
                    try:
                        dl = float(item["deadline_days"])
                        if dl < 0:
                            logger.warning("Negative deadline for task %s clamped to 0.0", tid)
                            dl = 0.0
                    except (ValueError, TypeError):
                        dl = None

                # Parse optional ISO dates
                s_date = None
                if item.get("start_date"):
                    try:
                        s_date = datetime.datetime.fromisoformat(str(item["start_date"]))
                        s_date = ensure_utc(s_date)
                    except Exception:
                        s_date = None

                d_date = None
                if item.get("deadline_date"):
                    try:
                        d_date = datetime.datetime.fromisoformat(str(item["deadline_date"]))
                        d_date = ensure_utc(d_date)
                    except Exception:
                        d_date = None

                deps = item.get("dependencies") or []
                if isinstance(deps, str):
                    deps = [d.strip() for d in deps.split(",") if d.strip()]
                elif not isinstance(deps, list):
                    deps = []

                res = item.get("resources") or {}
                if not isinstance(res, dict):
                    res = {}

                t = TaskNode(
                    task_id=tid,
                    name=tname,
                    duration_days=dur,
                    deadline_days=dl,
                    start_date=s_date,
                    deadline_date=d_date,
                    dependencies=[str(d) for d in deps],
                    resources={str(k): float(v) for k, v in res.items() if isinstance(v, (int, float))},
                    risk_driver=item.get("risk_driver"),
                    category=str(item.get("category", "General"))
                )
            else:
                continue

            if t.task_id in seen_ids:
                logger.warning("Duplicate task_id %s encountered; appending unique suffix", t.task_id)
                t.task_id = f"{t.task_id}_{idx}"
            seen_ids.add(t.task_id)
            sanitized.append(t)

        # Drop non-existent or self dependencies
        for t in sanitized:
            valid_deps = [d for d in t.dependencies if d in seen_ids and d != t.task_id]
            if len(valid_deps) != len(t.dependencies):
                dropped = set(t.dependencies) - set(valid_deps)
                logger.warning("Task %s had non-existent or self dependencies dropped: %s", t.task_id, dropped)
            t.dependencies = valid_deps

        return sanitized

    def compute_schedule(
        self, 
        tasks: List[TaskNode], 
        target_completion_days: Optional[float] = None,
        base_start_date: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        """
        Computes forward pass, backward pass, float/buffers, and calendar metrics in O(V + E) time.
        """
        if not tasks:
            return {
                "tasks": {},
                "total_duration_days": 0.0,
                "target_completion_days": target_completion_days or 0.0,
                "project_buffer_days": 0.0,
                "critical_path_tasks": [],
                "negative_buffer_tasks": [],
                "cascading_tasks": [],
                "resource_bottlenecks": {},
                "is_zero_delay_state": True
            }

        task_map = {t.task_id: t for t in tasks}
        
        preds: Dict[str, List[str]] = {t.task_id: list(t.dependencies) for t in tasks}
        succs: Dict[str, List[str]] = {t.task_id: [] for t in tasks}
        in_degree: Dict[str, int] = {t.task_id: len(t.dependencies) for t in tasks}

        for t in tasks:
            for parent_id in t.dependencies:
                succs[parent_id].append(t.task_id)

        # 1. Topological Sort via Kahn's Algorithm (O(V + E))
        queue = deque([t.task_id for t in tasks if in_degree[t.task_id] == 0])
        topo_order: List[str] = []

        while queue:
            curr = queue.popleft()
            topo_order.append(curr)
            for child in succs[curr]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # Cycle detection & graceful fallback
        if len(topo_order) < len(tasks):
            logger.warning(
                "Cycle detected in project task graph! %d tasks unvisited. Falling back to linear sequence.",
                len(tasks) - len(topo_order)
            )
            unvisited = [t.task_id for t in tasks if t.task_id not in set(topo_order)]
            topo_order.extend(unvisited)
            for uid in unvisited:
                preds[uid] = []

        # 2. Forward Pass: Earliest Start (ES) and Earliest Finish (EF)
        for tid in topo_order:
            node = task_map[tid]
            if not preds[tid]:
                node.earliest_start = 0.0
            else:
                node.earliest_start = max(task_map[p].earliest_finish for p in preds[tid])
            node.earliest_finish = node.earliest_start + node.duration_days

        project_finish = max((node.earliest_finish for node in tasks), default=0.0)
        effective_target = target_completion_days if target_completion_days is not None else project_finish

        # Project reference date in UTC
        ref_start_utc = ensure_utc(base_start_date) or datetime.datetime.now(datetime.timezone.utc)

        # 3. Backward Pass: Latest Finish (LF) and Latest Start (LS)
        for tid in reversed(topo_order):
            node = task_map[tid]
            if not succs[tid]:
                base_lf = effective_target
                if node.deadline_days is not None:
                    base_lf = min(base_lf, node.deadline_days)
                node.latest_finish = base_lf
            else:
                min_succ_ls = min(task_map[s].latest_start for s in succs[tid])
                if node.deadline_days is not None:
                    min_succ_ls = min(min_succ_ls, node.deadline_days)
                node.latest_finish = min_succ_ls
                
            node.latest_start = node.latest_finish - node.duration_days

        # 4. Buffer / Slack Calculation & UTC Calendar Projections
        critical_path_tasks: List[str] = []
        negative_buffer_tasks: List[str] = []
        cascading_tasks: List[str] = []

        for tid in topo_order:
            node = task_map[tid]
            # Total Slack = LS - ES = LF - EF
            node.slack = node.latest_start - node.earliest_start

            # Free Slack = min(ES of successors) - EF
            if succs[tid]:
                node.free_slack = min(task_map[s].earliest_start for s in succs[tid]) - node.earliest_finish
            else:
                node.free_slack = max(0.0, effective_target - node.earliest_finish)

            # UTC Calendar Projections
            node_base_start = node.start_date or ref_start_utc
            node.earliest_start_date = add_days_utc(node_base_start, node.earliest_start)
            node.earliest_finish_date = add_days_utc(node_base_start, node.earliest_finish)
            node.latest_start_date = add_days_utc(node_base_start, node.latest_start)
            node.latest_finish_date = add_days_utc(node_base_start, node.latest_finish)

            # Check concrete deadline_date calendar buffer if specified
            if node.deadline_date:
                cal_diff = date_diff_days_utc(node.earliest_finish_date, node.deadline_date)
                node.calendar_buffer_days = cal_diff
                if cal_diff < -1e-4:
                    node.has_negative_buffer = True

            # Critical Path: slack <= 0.0 (within numerical epsilon)
            if node.slack <= 1e-4:
                node.is_critical = True
                critical_path_tasks.append(tid)

            # Negative Buffer check
            if node.slack < -1e-4 or (node.calendar_buffer_days is not None and node.calendar_buffer_days < -1e-4):
                node.has_negative_buffer = True
                if tid not in negative_buffer_tasks:
                    negative_buffer_tasks.append(tid)

            # Cascading Dependency Calculation
            ripple = 0.0
            for child_id in succs[tid]:
                downstream_delay = max(0.0, node.duration_days - max(0.0, node.free_slack))
                ripple += downstream_delay
            node.cascading_impact_days = ripple
            if ripple > 0:
                cascading_tasks.append(tid)

        # 5. Resource Saturation Detection
        resource_bottlenecks: Dict[str, Dict[str, Any]] = {}
        if tasks:
            time_checkpoints = sorted(list(set(
                [node.earliest_start for node in tasks] + 
                [node.earliest_finish for node in tasks]
            )))

            for i in range(len(time_checkpoints) - 1):
                t_mid = (time_checkpoints[i] + time_checkpoints[i+1]) / 2.0
                active_nodes = [n for n in tasks if n.earliest_start <= t_mid < n.earliest_finish]
                
                res_usage: Dict[str, float] = {}
                for an in active_nodes:
                    for r_name, r_amt in an.resources.items():
                        res_usage[r_name] = res_usage.get(r_name, 0.0) + r_amt

                for r_name, demand in res_usage.items():
                    cap = self.resource_capacities.get(r_name, 10.0)
                    if demand > cap:
                        if r_name not in resource_bottlenecks:
                            resource_bottlenecks[r_name] = {
                                "capacity": cap,
                                "peak_demand": demand,
                                "saturated_window": (time_checkpoints[i], time_checkpoints[i+1]),
                                "affected_tasks": []
                            }
                        else:
                            resource_bottlenecks[r_name]["peak_demand"] = max(
                                resource_bottlenecks[r_name]["peak_demand"], demand
                            )
                        
                        for an in active_nodes:
                            if r_name in an.resources:
                                an.resource_saturated = True
                                if an.task_id not in resource_bottlenecks[r_name]["affected_tasks"]:
                                    resource_bottlenecks[r_name]["affected_tasks"].append(an.task_id)

        project_buffer_days = round(effective_target - project_finish, 2)
        is_zero_delay_state = (project_buffer_days >= 0.0 and len(negative_buffer_tasks) == 0)

        return {
            "tasks": task_map,
            "total_duration_days": round(project_finish, 2),
            "target_completion_days": round(effective_target, 2),
            "project_buffer_days": project_buffer_days,
            "critical_path_tasks": critical_path_tasks,
            "negative_buffer_tasks": negative_buffer_tasks,
            "cascading_tasks": cascading_tasks,
            "resource_bottlenecks": resource_bottlenecks,
            "is_zero_delay_state": is_zero_delay_state,
            "project_start_date_utc": ref_start_utc.isoformat(),
            "project_finish_date_utc": add_days_utc(ref_start_utc, project_finish).isoformat()
        }


# =====================================================================
# 4. STATUTORY INFRASTRUCTURE MILESTONE NETWORK TEMPLATE
# =====================================================================

def get_default_infrastructure_schedule(
    project_metadata: Optional[Dict[str, Any]] = None,
    risk_drivers: Optional[List[Tuple[str, float]]] = None
) -> List[TaskNode]:
    """
    Constructs the statutory Indian infrastructure milestone network:
      1. Land Demarcation & GIS Boundary
      2. Social Impact Assessment (SIA) Survey
      3. Expert Group SIA Review & State Approval
      4. Section 11 Notification (LARR Act 2013)
      5. Section 15 Hearing of Objections & Title Verification
      6. MoEF&CC Forest/Environmental Stage-1 Parivesh Clearance
      7. Stage-2 Compensatory Afforestation (CA) Demarcation
      8. Section 19 Resettlement & Rehabilitation Declaration
      9. Section 23/24 Compensation Award & Fund Disbursement
      10. Section 38 Possession Handover
    Dynamically adjusts durations and resource demands from risk parameters.
    """
    meta = project_metadata or {}

    d_demarcation = 30.0
    d_sia_survey = 60.0
    d_sia_approval = 45.0
    d_sec11 = 30.0
    d_disputes = 60.0
    d_forest_stg1 = 90.0
    d_forest_stg2 = 60.0
    d_sec19 = 45.0
    d_disbursement = 60.0
    d_possession = 30.0

    sia_stat = str(meta.get("sia_approval_status", "Approved")).strip().lower()
    if sia_stat in ["pending", "in review", "rejected"]:
        d_sia_approval += 45.0

    forest_stat = str(meta.get("forest_clearance_status", "Not_Required")).strip().lower()
    if "pending" in forest_stat or "stage_1" in forest_stat:
        d_forest_stg1 += 60.0
        d_forest_stg2 += 30.0
    elif forest_stat == "not_required":
        d_forest_stg1 = 0.0
        d_forest_stg2 = 0.0

    dispute_pct = float(meta.get("title_dispute_rate_percent", 5.0))
    if dispute_pct > 15.0:
        d_disputes += min(90.0, dispute_pct * 3.0)

    if bool(meta.get("local_protest_flag", False)):
        d_sia_survey += 30.0
        d_disputes += 45.0

    disb_pct = float(meta.get("fund_disbursement_percent", 50.0))
    if disb_pct < 30.0:
        d_disbursement += 45.0

    sec11_days = float(meta.get("section_11_notification_days", 30.0))
    if sec11_days > 180.0:
        d_sec19 = max(15.0, d_sec19 - 15.0)

    tasks: List[TaskNode] = [
        TaskNode(
            task_id="T1_DEMARCATION",
            name="GIS Boundary & Land Demarcation",
            duration_days=d_demarcation,
            dependencies=[],
            resources={"survey_teams": 2.0},
            category="Administrative"
        ),
        TaskNode(
            task_id="T2_SIA_SURVEY",
            name="Social Impact Assessment (SIA) Survey",
            duration_days=d_sia_survey,
            dependencies=["T1_DEMARCATION"],
            resources={"survey_teams": 3.0},
            risk_driver="affected_families_count",
            category="Socio-Legal"
        ),
        TaskNode(
            task_id="T3_SIA_APPROVAL",
            name="Expert Committee SIA Review & State Approval",
            duration_days=d_sia_approval,
            dependencies=["T2_SIA_SURVEY"],
            resources={"clearance_officers": 2.0},
            risk_driver="sia_approval_status",
            category="Administrative"
        ),
        TaskNode(
            task_id="T4_SEC11_NOTIF",
            name="Preliminary Section 11 Notification (LARR Act)",
            duration_days=d_sec11,
            dependencies=["T3_SIA_APPROVAL"],
            resources={"legal_officers": 2.0},
            risk_driver="section_11_notification_days",
            category="Legal"
        ),
        TaskNode(
            task_id="T5_DISPUTES",
            name="Section 15 Objections & Title Verification",
            duration_days=d_disputes,
            dependencies=["T4_SEC11_NOTIF"],
            resources={"legal_officers": 4.0},
            risk_driver="title_dispute_rate_percent",
            category="Socio-Legal"
        ),
        TaskNode(
            task_id="T6_FOREST_STG1",
            name="Parivesh MoEF&CC Forest Stage-1 In-Principle Clearance",
            duration_days=d_forest_stg1,
            dependencies=["T4_SEC11_NOTIF"],
            resources={"clearance_officers": 2.0},
            risk_driver="forest_clearance_status",
            category="Environmental"
        ),
        TaskNode(
            task_id="T7_FOREST_STG2",
            name="Compensatory Afforestation & Forest Stage-2 Clearance",
            duration_days=d_forest_stg2,
            dependencies=["T6_FOREST_STG1"] if d_forest_stg1 > 0 else ["T4_SEC11_NOTIF"],
            resources={"clearance_officers": 2.0},
            risk_driver="forest_clearance_status",
            category="Environmental"
        ),
        TaskNode(
            task_id="T8_SEC19_DECLARATION",
            name="Section 19 Resettlement & Rehabilitation Declaration",
            duration_days=d_sec19,
            dependencies=["T5_DISPUTES"],
            resources={"legal_officers": 2.0},
            risk_driver="affected_families_count",
            category="Legal"
        ),
        TaskNode(
            task_id="T9_AWARD_DISBURSEMENT",
            name="Section 23/24 Compensation Award & Disbursement",
            duration_days=d_disbursement,
            dependencies=["T8_SEC19_DECLARATION", "T7_FOREST_STG2"] if d_forest_stg2 > 0 else ["T8_SEC19_DECLARATION"],
            resources={"disbursement_cells": 3.0, "legal_officers": 2.0},
            risk_driver="fund_disbursement_percent",
            category="Financial"
        ),
        TaskNode(
            task_id="T10_POSSESSION_HANDOVER",
            name="Section 38 Physical Possession & Site Handover",
            duration_days=d_possession,
            dependencies=["T9_AWARD_DISBURSEMENT"],
            resources={"survey_teams": 1.0, "legal_officers": 1.0},
            category="Administrative"
        )
    ]
    return tasks


# =====================================================================
# 5. RECOMMENDATION ENGINE IMPLEMENTATION
# =====================================================================

class RecommendationEngine:
    """
    Generates actionable, prioritized, deterministic recommendations combining:
      1. Statutory risk template interventions based on SHAP feature attributions
      2. Critical Chain / Critical Path delay-prevention schedule analysis
    """

    def __init__(self, resource_capacities: Optional[Dict[str, float]] = None):
        self.recommendation_templates = self._load_templates()
        self.schedule_engine = CriticalChainEngine(resource_capacities=resource_capacities)

    def _load_templates(self) -> Dict[str, Any]:
        """Load baseline recommendation templates."""
        return {
            'high_legal_risk': {
                'actions': [
                    'Establish fast-track dispute resolution cell with revenue officers',
                    'Conduct title regularization camps for defective agricultural deeds',
                    'Empanel senior land acquisition advocates for high-value claims',
                    'Document all statutory notice service proofs to prevent court stays'
                ],
                'priority': 'Critical',
                'timeframe': 'Immediate (0-30 days)',
                'expected_impact': 'Reduce dispute-related delay risk by 25-40%'
            },
            'high_social_risk': {
                'actions': [
                    'Deploy localized community liaison officers at panchayat level',
                    'Organize structured town hall hearings with affected families',
                    'Formalize transparent rehabilitation & resettlement (R&R) packages',
                    'Establish village-level grievance redressal committee'
                ],
                'priority': 'High',
                'timeframe': 'Short-term (0-60 days)',
                'expected_impact': 'Mitigate community resistance and protest delay by 20-35%'
            },
            'clearance_delays': {
                'actions': [
                    'Expedite MoEF&CC Stage-1/Stage-2 file tracking via Parivesh nodal cell',
                    'Pre-demarcate non-forest land parcels for compensatory afforestation',
                    'Initiate joint forest demarcation inspections with state forest department',
                    'Concurrently process wildlife and eco-sensitive zone clearances'
                ],
                'priority': 'High',
                'timeframe': 'Medium-term (30-90 days)',
                'expected_impact': 'Compress statutory environmental clearance cycle by 30-50 days'
            },
            'financial_risk': {
                'actions': [
                    'Accelerate Treasury escrow account release for pending compensation',
                    'Transition from manual bank verification to Aadhaar-linked DBT payouts',
                    'Establish contingency financial reserve for court-ordered award uplifts',
                    'Implement real-time fund disbursement tracking portal'
                ],
                'priority': 'Medium',
                'timeframe': 'Short-term (0-45 days)',
                'expected_impact': 'Eliminate disbursement bottlenecks and cut financial lag by 15-25%'
            },
            'environmental_risk': {
                'actions': [
                    'Complete baseline bio-diversity assessments with accredited QCI-NABET experts',
                    'Deploy real-time environmental monitoring sensors in eco-sensitive zones',
                    'Prepare site-specific muck disposal and catchment area treatment plans',
                    'Submit quarterly compliance reports to MoEF&CC regional directorate'
                ],
                'priority': 'Medium',
                'timeframe': 'Medium-term (30-90 days)',
                'expected_impact': 'Ensure 100% statutory compliance and prevent stop-work notices'
            },
            'administrative_risk': {
                'actions': [
                    'Establish inter-departmental task force with district collectorate',
                    'Implement weekly digital milestone tracking across line departments',
                    'Harmonize survey and revenue record discrepancies across taluks',
                    'Streamline administrative approval chains to avoid bureaucratic idle time'
                ],
                'priority': 'Medium',
                'timeframe': 'Short-term (0-45 days)',
                'expected_impact': 'Optimize administrative decision velocity and minimize process latency'
            }
        }

    # Explicit feature-to-template mapping isolating geography/area from clearance/environmental
    FEATURE_TO_TEMPLATE_KEY: Dict[str, str] = {
        # High Legal Risk / Title Disputes
        "title_dispute_rate_percent": "high_legal_risk",
        "section_11_notification_days": "high_legal_risk",
        "compensation_multiplier_demand": "high_legal_risk",
        "legal": "high_legal_risk",
        "dispute": "high_legal_risk",

        # High Social Risk / R&R / Protests
        "affected_families_count": "high_social_risk",
        "local_protest_flag": "high_social_risk",
        "population_density": "high_social_risk",

        # Clearance Delays
        "forest_clearance_status": "clearance_delays",
        "forest_clearance_status_risk_score": "clearance_delays",
        "sia_approval_status": "clearance_delays",
        "sia_approval_status_risk_score": "clearance_delays",

        # Financial Risk
        "estimated_cost_inr_crore": "financial_risk",
        "fund_disbursement_percent": "financial_risk",
        "financial_density": "financial_risk",
        "financial_burn_rate_to_date": "financial_risk",
        "C_r": "financial_risk",
        "F_r": "financial_risk",

        # Environmental Risk
        "terrain_type": "environmental_risk",
        "environmental": "environmental_risk",
        "eco_sensitive": "environmental_risk",

        # Administrative Workflow & Geography (strictly non-clearance)
        "project_id": "administrative_risk",
        "project_type": "administrative_risk",
        "state": "administrative_risk",
        "district": "administrative_risk",
        "land_area_hectares": "administrative_risk",
        "land_area_log": "administrative_risk",
        "project_start_year": "administrative_risk",
        "project_age_years": "administrative_risk",
        "state_project_type": "administrative_risk",
        "H_r": "administrative_risk",
        "W_r": "administrative_risk",
        "P_r": "administrative_risk",
    }

    def generate_recommendations(
        self, 
        risk_drivers: List[Tuple[str, float]], 
        project_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate deterministic, prioritized recommendations combining risk drivers
        and schedule critical-chain delay prevention.
        """
        meta = project_metadata or {}
        recommendations: List[Dict[str, Any]] = []
        seen_keys: Set[str] = set()

        # 1. Generate Template-driven Risk Driver Recommendations
        for feature, importance in risk_drivers:
            # Check explicit mapping table first (avoids fragile substring false matches like 'p_r' or 'area')
            t_key = self.FEATURE_TO_TEMPLATE_KEY.get(feature)
            if t_key is None:
                f_lower = feature.lower()
                if 'dispute' in f_lower or 'legal' in f_lower or 'section_11' in f_lower:
                    t_key = 'high_legal_risk'
                elif 'protest' in f_lower or 'family' in f_lower or 'families' in f_lower or 'social' in f_lower:
                    t_key = 'high_social_risk'
                elif 'forest' in f_lower or ('clearance' in f_lower and 'area' not in f_lower):
                    t_key = 'clearance_delays'
                elif 'fund' in f_lower or 'cost' in f_lower or 'burn' in f_lower or 'deficit' in f_lower or 'gap' in f_lower:
                    t_key = 'financial_risk'
                elif 'environment' in f_lower or 'eco' in f_lower:
                    t_key = 'environmental_risk'
                else:
                    t_key = 'financial_risk'

            rec_id = f"driver_{t_key}_{feature}"
            if rec_id not in seen_keys:
                rec = self._create_recommendation(t_key, feature, float(importance))
                rec["recommendation_id"] = rec_id
                rec["source"] = "XAI_Attribution"
                recommendations.append(rec)
                seen_keys.add(rec_id)

        # 2. Critical Chain Schedule Engine: Delay-Prevention Mitigations
        try:
            raw_tasks = meta.get("schedule_tasks")
            if raw_tasks and isinstance(raw_tasks, list):
                sanitized_tasks = self.schedule_engine.validate_and_sanitize(raw_tasks)
            else:
                sanitized_tasks = get_default_infrastructure_schedule(meta, risk_drivers)

            target_completion = meta.get("target_completion_days")
            if target_completion is not None:
                try:
                    target_completion = float(target_completion)
                except (ValueError, TypeError):
                    target_completion = None

            base_start = meta.get("project_start_date")
            if base_start and isinstance(base_start, str):
                try:
                    base_start = datetime.datetime.fromisoformat(base_start)
                except Exception:
                    base_start = None

            schedule_result = self.schedule_engine.compute_schedule(
                sanitized_tasks, 
                target_completion_days=target_completion,
                base_start_date=base_start
            )

            schedule_mitigations = self._generate_schedule_mitigations(
                schedule_result, 
                project_metadata=meta
            )

            for sm in schedule_mitigations:
                if sm["recommendation_id"] not in seen_keys:
                    recommendations.append(sm)
                    seen_keys.add(sm["recommendation_id"])

        except Exception as e:
            logger.error("Schedule critical-chain calculation encountered error: %s; using safe fallback", e)

        # 3. Add Project-Specific Heuristic Recommendations
        for ps_rec in self._get_project_specific_recommendations(meta):
            ps_id = f"specific_{ps_rec.get('issue', 'ps')}"
            if ps_id not in seen_keys:
                ps_rec["recommendation_id"] = ps_id
                ps_rec["source"] = "Project_Constraint"
                recommendations.append(ps_rec)
                seen_keys.add(ps_id)

        # 4. Deterministic Stable Sorting:
        # Priority Rank (Critical=0, High=1, Medium-High=2, Medium=3, Low=4)
        # Then by importance / avoided delay descending, then stable tie-breaker by recommendation_id
        priority_map = {'Critical': 0, 'High': 1, 'Medium-High': 2, 'Medium': 3, 'Low': 4}
        recommendations.sort(
            key=lambda x: (
                priority_map.get(x.get('priority', 'Medium'), 5),
                -float(x.get('importance', 0.5)),
                x.get('recommendation_id', '')
            )
        )

        return recommendations

    def _generate_schedule_mitigations(
        self, 
        schedule_result: Dict[str, Any], 
        project_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Produce deterministic delay-prevention interventions for:
          - Zero-delay states (proactive buffer preservation)
          - Negative buffer times (fast-track crashing)
          - Cascading dependencies (upstream bottleneck compression)
          - Resource saturation (capacity expansion)
        """
        mitigations: List[Dict[str, Any]] = []
        tasks: Dict[str, TaskNode] = schedule_result.get("tasks", {})
        is_zero_delay = schedule_result.get("is_zero_delay_state", False)
        project_buffer = schedule_result.get("project_buffer_days", 0.0)

        # EDGE CASE 1: Zero-Delay State
        if is_zero_delay:
            mitigations.append({
                "recommendation_id": "sched_zero_delay_buffer_preservation",
                "issue": f"Project on schedule with positive float (+{project_buffer} Days buffer)",
                "actions": [
                    "Maintain buffer integrity through bi-weekly milestone audit checkpoints",
                    "Pre-screen statutory expiry dates for Section 11 preliminary notifications",
                    "Ensure reserve legal and survey personnel remain allocated on standby",
                    "Conduct proactive stakeholder liaison to prevent sudden dispute escalations"
                ],
                "priority": "Low",
                "timeframe": "Continuous Monitoring",
                "expected_impact": "Preserves current positive buffer without costly crash measures",
                "risk_driver": "schedule_buffer",
                "importance": 0.40,
                "template_key": "zero_delay_buffer_preservation",
                "source": "Critical_Chain_Engine",
                "buffer_status": "Buffer Preserved (+Float)",
                "is_zero_delay_state": True
            })
            return mitigations

        # EDGE CASE 2: Negative Buffer Times (Active Schedule Breach)
        neg_tasks = schedule_result.get("negative_buffer_tasks", [])
        for tid in neg_tasks:
            node = tasks.get(tid)
            if not node:
                continue
            slip_days = abs(round(node.slack, 1))
            mitigations.append({
                "recommendation_id": f"sched_negative_buffer_{tid}",
                "issue": f"Negative buffer on critical milestone '{node.name}' ({slip_days} Days Breach)",
                "actions": [
                    f"Fast-track {node.name} execution through parallel processing teams",
                    f"Authorize emergency overtime and dedicated statutory officers to recover {slip_days} days",
                    "Request priority hearing / fast-track gazette notification from Competent Authority",
                    "Compress review cycles by conducting joint inter-departmental site verifications"
                ],
                "priority": "Critical",
                "timeframe": "Immediate (0-15 days)",
                "expected_impact": f"Recovers up to {slip_days} days of negative buffer to restore critical path",
                "risk_driver": node.risk_driver or "critical_chain_slip",
                "importance": 0.95,
                "template_key": "clearance_delays" if "forest" in tid.lower() or "sia" in tid.lower() else "high_legal_risk",
                "source": "Critical_Chain_Engine",
                "buffer_status": f"Negative Buffer (-{slip_days} Days)"
            })

        # EDGE CASE 3: Cascading Dependencies
        cascading_ids = schedule_result.get("cascading_tasks", [])
        for cid in cascading_ids:
            node = tasks.get(cid)
            if not node or node.has_negative_buffer:
                continue
            if node.cascading_impact_days >= 20.0:
                mitigations.append({
                    "recommendation_id": f"sched_cascading_{cid}",
                    "issue": f"Cascading dependency risk at '{node.name}' (Threatens {round(node.cascading_impact_days)} Downstream Days)",
                    "actions": [
                        f"Decouple successor activities from '{node.name}' to enable concurrent work packages",
                        "Issue conditional provisional clearances to allow downstream surveying in non-disputed segments",
                        "Implement daily progress burn-down tracking to detect early micro-delays"
                    ],
                    "priority": "High",
                    "timeframe": "Short-term (0-30 days)",
                    "expected_impact": f"Prevents up to {round(node.cascading_impact_days)} days of cascading downstream delay",
                    "risk_driver": node.risk_driver or "cascading_dependency",
                    "importance": 0.85,
                    "template_key": "clearance_delays",
                    "source": "Critical_Chain_Engine",
                    "buffer_status": f"Cascading Risk ({round(node.cascading_impact_days)} Days Impact)"
                })

        # EDGE CASE 4: Resource Saturation
        bottlenecks = schedule_result.get("resource_bottlenecks", {})
        for r_name, b_info in bottlenecks.items():
            cap, peak = b_info["capacity"], b_info["peak_demand"]
            affected = b_info["affected_tasks"]
            mitigations.append({
                "recommendation_id": f"sched_resource_saturation_{r_name}",
                "issue": f"Resource saturation in '{r_name.replace('_', ' ').title()}' (Peak Demand: {peak:.1f} vs Cap: {cap:.1f})",
                "actions": [
                    f"Empanel auxiliary external {r_name.replace('_', ' ')} on short-term project contracts",
                    f"Level resource allocation across milestones: {', '.join(affected[:3])}",
                    "Stagger non-critical path activities to eliminate concurrent resource contention",
                    "Reallocate unutilized staff from low-risk districts to bottleneck phases"
                ],
                "priority": "High",
                "timeframe": "Immediate (0-20 days)",
                "expected_impact": "Relieves resource saturation bottleneck, preventing schedule slippage",
                "risk_driver": "resource_contention",
                "importance": 0.80,
                "template_key": "financial_risk",
                "source": "Critical_Chain_Engine",
                "buffer_status": f"Saturated ({peak:.1f}/{cap:.1f})"
            })

        return mitigations

    def _create_recommendation(self, template_key: str, feature: str, importance: float) -> Dict[str, Any]:
        """Create recommendation from standard template."""
        template = self.recommendation_templates.get(template_key, {})
        return {
            'issue': f"High risk from {feature.replace('_', ' ').title()}",
            'actions': list(template.get('actions', ['Investigate issue'])),
            'priority': template.get('priority', 'Medium'),
            'timeframe': template.get('timeframe', 'Short-term'),
            'expected_impact': template.get('expected_impact', 'Varies'),
            'risk_driver': feature,
            'importance': importance,
            'template_key': template_key
        }

    def _get_project_specific_recommendations(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate project-specific heuristic recommendations."""
        recs: List[Dict[str, Any]] = []

        if metadata.get('estimated_cost_inr_crore', 0) > 5000:
            recs.append({
                'issue': 'Mega-project budget requiring executive steering oversight',
                'actions': [
                    'Establish Project Steering Committee chaired by State Principal Secretary',
                    'Implement real-time GIS & financial monitoring dashboard with weekly executive syncs',
                    'Deploy dedicated dispute-mitigation task force to clear land hurdles'
                ],
                'priority': 'High',
                'timeframe': 'Immediate (0-15 days)',
                'expected_impact': 'Accelerates inter-departmental approvals and mitigates cost overruns',
                'importance': 0.75,
                'template_key': 'financial_risk',
                'is_general': True
            })

        if metadata.get('terrain_type') == 'Forest_Eco_Sensitive':
            recs.append({
                'issue': 'Eco-sensitive forest corridor requiring integrated biodiversity compliance',
                'actions': [
                    'Initiate joint boundary surveys with State Forest Department and Wildlife Board',
                    'Prepare wildlife passage and elephant/animal corridor mitigation plans',
                    'Conduct statutory quarterly public environmental audits on site'
                ],
                'priority': 'High',
                'timeframe': 'Pre-construction (0-60 days)',
                'expected_impact': 'Ensures 100% compliance with MoEF&CC Stage-1 guidelines',
                'importance': 0.70,
                'template_key': 'environmental_risk',
                'is_general': True
            })

        return recs

    def format_recommendations_for_display(self, recommendations: List[Dict[str, Any]]) -> str:
        """Format recommendations for human-readable console/report display."""
        output = "\n" + "="*70 + "\n"
        output += f"{'EXECUTIVE PRESCRIPTIVE MITIGATION PLAN':^70}\n"
        output += "="*70 + "\n\n"

        for i, rec in enumerate(recommendations, 1):
            output += f"* Recommendation #{i}: {rec.get('issue', 'Mitigation')}\n"
            output += f"   Priority: {rec.get('priority', 'Medium')}\n"
            output += f"   Timeframe: {rec.get('timeframe', 'Short-term')}\n"
            if rec.get('buffer_status'):
                output += f"   Buffer Status: {rec.get('buffer_status')}\n"
            output += f"   Expected Impact: {rec.get('expected_impact', 'N/A')}\n"
            output += "   Actionable Steps:\n"
            for action in rec.get('actions', []):
                output += f"      - {action}\n"
            output += "\n"

        return output


# =====================================================================
# 6. ROI & FINANCIAL DELAY MITIGATION EVALUATION
# =====================================================================

def get_dynamic_implementation_cost(template_key: str, project_cost: float) -> float:
    """Return dynamic intervention implementation cost in INR."""
    cost_map = {
        'high_legal_risk': 15_00_000.0,
        'high_social_risk': 5_00_000.0,
        'clearance_delays': 25_00_000.0,
        'financial_risk': 10_00_000.0,
        'environmental_risk': 25_00_000.0,
        'administrative_risk': 12_00_000.0,
        'zero_delay_buffer_preservation': 2_50_000.0
    }
    base_cost = cost_map.get(template_key, 10_00_000.0)

    # Size adjustment for mega-projects: larger threshold must be checked first
    if project_cost > 100_00_00_00_000:   # 10,000 Cr — 1.5x
        base_cost *= 1.5
    elif project_cost > 50_00_00_00_000:  # 5,000 Cr — 1.3x
        base_cost *= 1.3

    return base_cost


def calculate_roi_for_recommendation(
    recommendation: Dict[str, Any], 
    project_cost: float, 
    delay_cost_per_day: float, 
    model: Any = None, 
    X_sample: Any = None
) -> Dict[str, Any]:
    """
    Calculate quantified financial ROI and delay savings for an intervention.
    """
    if model is not None and X_sample is not None and 'risk_driver' in recommendation:
        try:
            X_mitigated = X_sample.copy()
            feature = recommendation['risk_driver']
            direction = recommendation.get('direction', 'increases_delay')

            if feature in X_mitigated.columns:
                curr_val = float(X_mitigated[feature].values[0])
                # Verify mitigation direction per feature:
                # 1. Features where HIGHER is better: simulate improvement by increasing value
                if 'fund' in feature.lower() or 'disbursement' in feature.lower():
                    X_mitigated[feature] = min(100.0, curr_val * 1.25)
                elif 'clearance' in feature.lower() and ('status' in feature.lower() and 'score' not in feature.lower()):
                    X_mitigated[feature] = 1.0
                elif 'sia' in feature.lower() and ('status' in feature.lower() and 'score' not in feature.lower()):
                    X_mitigated[feature] = 1.0
                elif direction == "decreases_delay":
                    X_mitigated[feature] = curr_val * 1.2
                else:
                    # 2. Features where LOWER is better (disputes, protest flags, cost overruns, area, affected families)
                    X_mitigated[feature] = curr_val * 0.8

            pred_orig = model.predict(X_sample)
            pred_mit = model.predict(X_mitigated)
            orig_delay = float(pred_orig.get('predicted_delay_days', pred_orig.get('delay_days', [30]))[0])
            new_delay = float(pred_mit.get('predicted_delay_days', pred_mit.get('delay_days', [30]))[0])
            estimated_delay_days_saved = max(0.0, orig_delay - new_delay)
        except Exception as e:
            logger.warning("Data-driven ROI simulation failed (%s); using deterministic heuristic", e)
            importance = float(recommendation.get('importance', 0.5))
            estimated_delay_days_saved = max(0.0, importance * 0.3 * 180.0)
    else:
        if recommendation.get("is_zero_delay_state", False):
            estimated_delay_days_saved = 5.0
        else:
            importance = float(recommendation.get('importance', 0.5))
            estimated_delay_days_saved = max(0.0, importance * 0.3 * 180.0)

    cost_savings = estimated_delay_days_saved * delay_cost_per_day
    impl_cost = get_dynamic_implementation_cost(
        recommendation.get('template_key', 'default'), 
        project_cost
    )

    if impl_cost <= 0:
        impl_cost = 1.0

    roi_percentage = ((cost_savings - impl_cost) / impl_cost) * 100.0

    return {
        'estimated_delay_days_saved': round(estimated_delay_days_saved, 1),
        'cost_savings': round(cost_savings, 2),
        'implementation_cost': round(impl_cost, 2),
        'roi_percentage': round(roi_percentage, 1),
        'payback_period_days': round(impl_cost / (cost_savings / 180.0), 1) if cost_savings > 0 else 0.0
    }


# =====================================================================
# 7. DELAY-PREVENTION ENGINE API FACADES
# =====================================================================

class ScheduleEngine:
    """
    Standardized Schedule Engine facade providing topological sort,
    strict cycle detection, and Critical Path/Buffer calculations.
    """
    def __init__(
        self, 
        tasks: List[TaskNode], 
        target_completion_days: Optional[float] = None, 
        resource_capacities: Optional[Dict[str, float]] = None
    ):
        self.engine = CriticalChainEngine(resource_capacities=resource_capacities)
        self.tasks = self.engine.validate_and_sanitize(tasks)
        self.target_completion_days = target_completion_days

    def topological_sort(self) -> List[TaskNode]:
        task_map = {t.task_id: t for t in self.tasks}
        preds = {t.task_id: list(t.dependencies) for t in self.tasks}
        succs = {t.task_id: [] for t in self.tasks}
        in_degree = {t.task_id: len(t.dependencies) for t in self.tasks}
        for t in self.tasks:
            for parent_id in t.dependencies:
                succs[parent_id].append(t.task_id)

        queue = deque([t.task_id for t in self.tasks if in_degree[t.task_id] == 0])
        topo_order: List[TaskNode] = []
        while queue:
            curr = queue.popleft()
            topo_order.append(task_map[curr])
            for child in succs[curr]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(topo_order) < len(self.tasks):
            raise ValueError(f"Circular dependency detected in tasks! Visited {len(topo_order)} of {len(self.tasks)}")

        return topo_order

    def calculate_schedule(self, base_start_date: Optional[datetime.datetime] = None) -> Dict[str, Any]:
        return self.engine.compute_schedule(
            self.tasks,
            target_completion_days=self.target_completion_days,
            base_start_date=base_start_date
        )


CriticalPathAnalyzer = ScheduleEngine


class MitigationEngine:
    """
    Deterministic mitigation generator and ranker based on schedule analysis.
    """
    @staticmethod
    def generate_mitigation_actions(
        schedule_analysis: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        rec_engine = RecommendationEngine()
        actions = rec_engine._generate_schedule_mitigations(
            schedule_analysis, 
            project_metadata=metadata or {}
        )
        
        # Deduplicate and sort deterministically by severity
        priority_rank = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        actions.sort(
            key=lambda x: (
                priority_rank.get(x.get("priority", "Medium"), 4),
                -float(x.get("importance", 0.5)),
                x.get("issue", "")
            )
        )
        return actions


def generate_delay_mitigation_plan(
    tasks: List[TaskNode], 
    target_days: Optional[float] = None, 
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generates complete delay mitigation plan from task milestones."""
    sched_engine = ScheduleEngine(tasks, target_completion_days=target_days)
    analysis = sched_engine.calculate_schedule()
    mitigations = MitigationEngine.generate_mitigation_actions(analysis, metadata=metadata)
    return {
        "schedule_analysis": analysis,
        "mitigations": mitigations
    }

