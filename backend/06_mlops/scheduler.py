"""
NEXUS-XAI Continuous Learning Automated Scheduler
=================================================
Manages automated drift monitoring and retraining jobs:
- Daily at 00:00 (Midnight): Run feature-by-feature PSI drift check. Auto-trigger retrain if PSI > 0.20.
- Hourly: Periodic background drift check to ensure continuous distribution monitoring.
- Weekly on Sunday at 02:00 AM: Force retrain regardless of drift on all accumulated data.
- Initial Startup Check: Immediate non-blocking drift evaluation on server boot.
"""

import os
import logging
import datetime
import threading
import sys
from typing import Dict, Any, Optional, Callable

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_CURR_DIR)
_WORKSPACE_ROOT = os.path.dirname(_BACKEND_DIR)
for _sub in [
    _WORKSPACE_ROOT,
    _BACKEND_DIR,
    os.path.join(_BACKEND_DIR, "01_intake"),
    os.path.join(_BACKEND_DIR, "02_preprocessing"),
    os.path.join(_BACKEND_DIR, "03_models"),
    os.path.join(_BACKEND_DIR, "04_xai"),
    os.path.join(_BACKEND_DIR, "05_orchestration"),
    os.path.join(_BACKEND_DIR, "06_mlops"),
    os.path.join(_BACKEND_DIR, "07_api"),
    os.path.join(_WORKSPACE_ROOT, "remoteness"),
    os.path.join(_WORKSPACE_ROOT, "data"),
]:
    if os.path.exists(_sub) and _sub not in sys.path:
        sys.path.insert(0, _sub)

from continuous_learning import (
    DriftDetector,
    RetrainingOrchestrator,
    DATA_STORE_PATH,
    get_current_model_health
)

logger = logging.getLogger("ContinuousLearningScheduler")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [Scheduler] [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

_SCHEDULER_INSTANCE: Optional[BackgroundScheduler] = None
_SCHEDULER_LOCK = threading.Lock()
_HOT_RELOAD_CALLBACK: Optional[Callable[[], Any]] = None

_LAST_RUNS = {
    "daily_drift": None,
    "weekly_retrain": None,
    "last_result": None,
    "last_retrain_result": None
}


def set_hot_reload_callback(callback: Callable[[], Any]) -> None:
    """Registers callback to reload in-memory models in API serving when promoted."""
    global _HOT_RELOAD_CALLBACK
    _HOT_RELOAD_CALLBACK = callback


def run_daily_drift_check() -> Dict[str, Any]:
    """
    Scheduled job: Drift calculation.
    Computes PSI for every feature; triggers retraining if PSI > 0.20.
    If promoted, triggers hot-reload callback for zero-downtime model updates.
    """
    logger.info("Executing scheduled feature drift check...")
    try:
        if not DATA_STORE_PATH.exists():
            logger.warning(f"Data store {DATA_STORE_PATH} not found. Skipping drift check.")
            return {"status": "skipped", "reason": "no_data_store"}

        df = pd.read_csv(DATA_STORE_PATH)
        # Take the most recent 15% or up to 500 records as recent incoming distribution vs earlier baseline
        split_point = max(100, int(len(df) * 0.85))
        baseline_df = df.iloc[:split_point]
        recent_df = df.iloc[split_point:]
        if len(recent_df) < 10:
            recent_df = df.tail(100)

        detector = DriftDetector(baseline_df=baseline_df)
        report = detector.evaluate_drift(incoming_df=recent_df)
        
        _LAST_RUNS["daily_drift"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _LAST_RUNS["last_result"] = report

        logger.info(f"Feature drift check complete. Max PSI: {report['max_psi']:.4f} (Threshold: 0.20)")
        if report.get("auto_retrain_triggered", False):
            logger.warning(f"Significant drift detected (Max PSI {report['max_psi']:.4f} > 0.20)! Auto-triggering retraining...")
            orchestrator = RetrainingOrchestrator()
            retrain_res = orchestrator.run_retrain_cycle(trigger_reason="drift_exceeded_threshold")
            _LAST_RUNS["last_retrain_result"] = retrain_res
            if retrain_res.get("promoted", False) and _HOT_RELOAD_CALLBACK:
                try:
                    _HOT_RELOAD_CALLBACK()
                    logger.info("[Scheduler] Hot-reloaded promoted model into active API serving.")
                except Exception as cb_err:
                    logger.warning(f"Hot-reload callback note: {cb_err}")
            return {"drift_report": report, "retrain_result": retrain_res}

        return {"drift_report": report}
    except Exception as e:
        logger.error(f"Error during drift check: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}


def run_weekly_force_retrain() -> Dict[str, Any]:
    """
    Scheduled job: Sunday at 02:00 AM.
    Forces full retraining on all available data regardless of drift.
    """
    logger.info("Executing scheduled weekly Sunday 2 AM retraining cycle...")
    try:
        orchestrator = RetrainingOrchestrator()
        result = orchestrator.run_retrain_cycle(trigger_reason="weekly_scheduled_sunday_2am")
        _LAST_RUNS["weekly_retrain"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _LAST_RUNS["last_retrain_result"] = result
        if result.get("promoted", False) and _HOT_RELOAD_CALLBACK:
            try:
                _HOT_RELOAD_CALLBACK()
                logger.info("[Scheduler] Hot-reloaded promoted weekly model into active API serving.")
            except Exception as cb_err:
                logger.warning(f"Hot-reload callback note: {cb_err}")
        logger.info(f"Weekly retraining cycle finished. Version: {result.get('version')} Promoted: {result.get('promoted')}")
        return result
    except Exception as e:
        logger.error(f"Error during weekly retraining: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}


def start_scheduler() -> BackgroundScheduler:
    """
    Initializes and starts APScheduler in the background.
    """
    global _SCHEDULER_INSTANCE
    with _SCHEDULER_LOCK:
        if _SCHEDULER_INSTANCE is not None and _SCHEDULER_INSTANCE.running:
            return _SCHEDULER_INSTANCE

        scheduler = BackgroundScheduler(daemon=True)

        # Job 1: Daily drift check at midnight (00:00)
        scheduler.add_job(
            run_daily_drift_check,
            trigger=CronTrigger(hour=0, minute=0),
            id="daily_drift_check",
            name="Daily PSI Drift Check (Midnight)",
            replace_existing=True
        )

        # Job 2: Force retrain every Sunday at 02:00 AM
        scheduler.add_job(
            run_weekly_force_retrain,
            trigger=CronTrigger(day_of_week="sun", hour=2, minute=0),
            id="weekly_sunday_retrain",
            name="Weekly Sunday 2 AM Force Retrain",
            replace_existing=True
        )

        # Job 3: Periodic hourly drift monitor to detect live drift continuously
        scheduler.add_job(
            run_daily_drift_check,
            trigger=IntervalTrigger(hours=1),
            id="hourly_drift_check",
            name="Hourly PSI Drift Monitor",
            replace_existing=True
        )

        scheduler.start()
        _SCHEDULER_INSTANCE = scheduler
        logger.info("APScheduler initialized: Daily midnight check, hourly drift monitor, and Sunday 2 AM retraining scheduled.")

        # Immediate non-blocking initial drift check on startup
        def _initial_drift_check():
            try:
                import time
                time.sleep(2)
                run_daily_drift_check()
            except Exception as e:
                logger.warning(f"Initial startup drift check note: {e}")

        threading.Thread(target=_initial_drift_check, daemon=True).start()

        return _SCHEDULER_INSTANCE


def get_scheduler() -> BackgroundScheduler:
    return start_scheduler()


def get_scheduler_status() -> Dict[str, Any]:
    """
    Returns current scheduler status and upcoming scheduled execution times.
    """
    jobs = []
    if _SCHEDULER_INSTANCE and _SCHEDULER_INSTANCE.running:
        for job in _SCHEDULER_INSTANCE.get_jobs():
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": next_run
            })
    return {
        "running": _SCHEDULER_INSTANCE is not None and _SCHEDULER_INSTANCE.running,
        "jobs": jobs,
        "last_runs": _LAST_RUNS
    }


if __name__ == "__main__":
    import time
    sch = start_scheduler()
    print("Scheduler running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        sch.shutdown()

