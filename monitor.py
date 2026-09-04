import pymysql
import os
import time
import datetime
import json
import threading
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error, mean_squared_error
from scipy.stats import ks_2samp, chi2_contingency
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_limits[i]) & (y_prob < bin_limits[i+1])
        if np.sum(mask) > 0:
            prob_pred = np.mean(y_prob[mask])
            prob_true = np.mean(y_true[mask])
            ece += np.abs(prob_pred - prob_true) * np.sum(mask)
    return ece / len(y_true)

class Alert:
    def __init__(self, severity: str, message: str, metrics: dict):
        self.severity = severity
        self.message = message
        self.metrics = metrics
        self.timestamp = datetime.datetime.now().isoformat()

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "severity": self.severity,
            "message": self.message,
            "metrics": self.metrics
        }

_GLOBAL_SQLITE_LOCK = threading.RLock()

class SQLiteConnectionWrapper:
    def __init__(self, db_path='monitoring.db'):
        import sqlite3
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
    def cursor(self):
        return SQLiteCursorWrapper(self.conn.cursor(), _GLOBAL_SQLITE_LOCK)
    def commit(self):
        with _GLOBAL_SQLITE_LOCK:
            self.conn.commit()
    def close(self):
        with _GLOBAL_SQLITE_LOCK:
            self.conn.close()

class SQLiteCursorWrapper:
    def __init__(self, cur, lock):
        self.cur = cur
        self.lock = lock
    def _translate(self, query):
        return query.replace('%s', '?').replace('INT AUTO_INCREMENT PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT').replace('AUTO_INCREMENT PRIMARY KEY', 'PRIMARY KEY AUTOINCREMENT')
    def execute(self, query, params=None):
        q = self._translate(query)
        with self.lock:
            if params is None:
                return self.cur.execute(q)
            return self.cur.execute(q, params)
    def executemany(self, query, seq):
        q = self._translate(query)
        with self.lock:
            return self.cur.executemany(q, seq)
    def fetchone(self):
        with self.lock:
            row = self.cur.fetchone()
            return dict(row) if row else None
    def fetchall(self):
        with self.lock:
            rows = self.cur.fetchall()
            return [dict(r) for r in rows] if rows else []

class ModelMonitor:
    def __init__(self):
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_user = os.getenv('DB_USER', 'root')
        self.db_password = os.getenv('DB_PASSWORD', 'rootpassword')
        self.db_name = os.getenv('DB_NAME', 'monitoring')
        self.use_sqlite = False
        self._init_db()
        
    def _get_connection(self):
        if self.use_sqlite:
            return SQLiteConnectionWrapper('monitoring.db')
        return pymysql.connect(
            host=self.db_host,
            user=self.db_user,
            password=self.db_password,
            database=self.db_name,
            cursorclass=pymysql.cursors.DictCursor
        )

    def _init_db(self):
        try:
            conn = pymysql.connect(host=self.db_host, user=self.db_user, password=self.db_password, connect_timeout=1)
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_name}")
            conn.commit()
            conn.close()
        except Exception:
            # Fall back to local SQLite database
            self.use_sqlite = True

        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp VARCHAR(255),
                roc_auc FLOAT,
                ece FLOAT,
                brier FLOAT,
                mae FLOAT,
                rmse FLOAT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_metrics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp VARCHAR(255),
                feature_name VARCHAR(255),
                psi FLOAT,
                p_value FLOAT,
                is_drifted BOOLEAN
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp VARCHAR(255),
                severity VARCHAR(50),
                message TEXT,
                metrics TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_performance(self, metrics: dict):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO performance_metrics (timestamp, roc_auc, ece, brier, mae, rmse)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (
            datetime.datetime.now().isoformat(),
            metrics.get('roc_auc', 0),
            metrics.get('ece', 0),
            metrics.get('brier', 0),
            metrics.get('mae', 0),
            metrics.get('rmse', 0)
        ))
        conn.commit()
        conn.close()

    def log_drift(self, drift_results: List[dict]):
        conn = self._get_connection()
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        
        for result in drift_results:
            cursor.execute('''
                INSERT INTO drift_metrics (timestamp, feature_name, psi, p_value, is_drifted)
                VALUES (%s, %s, %s, %s, %s)
            ''', (
                timestamp,
                result['feature'],
                result.get('psi', 0.0),
                result.get('p_value', 1.0),
                bool(result.get('is_drifted', False))
            ))
            
        conn.commit()
        conn.close()
        
    def log_alert(self, alert: Alert):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO alerts (timestamp, severity, message, metrics)
            VALUES (%s, %s, %s, %s)
        ''', (
            alert.timestamp,
            alert.severity,
            alert.message,
            json.dumps(alert.metrics)
        ))
        conn.commit()
        conn.close()
        # Trigger external systems here (stdout, email, slack)
        print(f"[ALERT - {alert.severity}] {alert.message} | {alert.metrics}")

    def check_performance(self, y_cls_true, y_prob, y_crs_true, crs_pred, y_days_true, days_pred) -> dict:
        """
        Evaluate metrics against true hold-out distributions.
        Triggers alerts if thresholds are breached.
        """
        roc = roc_auc_score(y_cls_true, y_prob)
        ece = expected_calibration_error(y_cls_true, y_prob)
        brier = brier_score_loss(y_cls_true, y_prob)
        mae = mean_absolute_error(y_days_true, days_pred)
        rmse = np.sqrt(mean_squared_error(y_crs_true, crs_pred))
        
        metrics = {
            'roc_auc': roc,
            'ece': ece,
            'brier': brier,
            'mae': mae,
            'rmse': rmse
        }
        self.log_performance(metrics)
        
        if roc < 0.70:
            self.log_alert(Alert("CRITICAL", "ROC-AUC dropped below 0.70", metrics))
        elif ece > 0.15:
            self.log_alert(Alert("WARNING", "Calibration ECE exceeded 15%", metrics))
            
        return metrics

    def _calculate_psi(self, expected, actual, buckets=10):
        """Calculate Population Stability Index"""
        expected_array = np.array(expected)
        actual_array = np.array(actual)
        
        # Zero-variance guard
        if np.var(expected_array) == 0 or np.var(actual_array) == 0:
            return 0.0
            
        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input
        
        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        
        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)
        
        def sub_psi(e_perc, a_perc):
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001
            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return value
            
        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(len(expected_percents)))
        return psi_value

    def detect_data_drift(self, X_new: pd.DataFrame, X_ref: pd.DataFrame, cat_cols=None, cont_cols=None) -> List[dict]:
        """
        Calculates PSI and statistical tests for drift.
        Triggers alerts if PSI > 0.20 or p-value < 0.05.
        """
        results = []
        
        if cont_cols:
            for col in cont_cols:
                if col in X_new and col in X_ref:
                    # KS Test
                    stat, p_val = ks_2samp(X_new[col].dropna(), X_ref[col].dropna())
                    # PSI
                    psi_val = self._calculate_psi(X_ref[col].dropna(), X_new[col].dropna())
                    
                    is_drifted = (psi_val > 0.2) or (p_val < 0.05)
                    results.append({
                        'feature': col,
                        'type': 'continuous',
                        'psi': float(psi_val),
                        'p_value': float(p_val),
                        'is_drifted': is_drifted
                    })
                    
        if cat_cols:
            for col in cat_cols:
                if col in X_new and col in X_ref:
                    # Chi-Square Test
                    ref_counts = X_ref[col].value_counts().sort_index()
                    new_counts = X_new[col].value_counts().sort_index()
                    
                    # Align indices
                    all_idx = ref_counts.index.union(new_counts.index)
                    ref_counts = ref_counts.reindex(all_idx, fill_value=0)
                    new_counts = new_counts.reindex(all_idx, fill_value=0)
                    
                    # Needs >0 expected freq for chi2
                    if len(all_idx) > 1 and sum(ref_counts) > 0 and sum(new_counts) > 0:
                        obs = np.array([ref_counts.values, new_counts.values])
                        obs = obs + 1 # Smoothing
                        _, p_val, _, _ = chi2_contingency(obs)
                    else:
                        p_val = 1.0
                        
                    is_drifted = p_val < 0.05
                    results.append({
                        'feature': col,
                        'type': 'categorical',
                        'p_value': float(p_val),
                        'is_drifted': is_drifted
                    })

        self.log_drift(results)
        
        drifted_features = [r['feature'] for r in results if r['is_drifted']]
        if drifted_features:
            self.log_alert(Alert("WARNING", f"Data drift detected in {len(drifted_features)} features", {"features": drifted_features}))
            
        return results

    def get_alert_summary(self, limit=10):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT timestamp, severity, message, metrics FROM alerts ORDER BY id DESC LIMIT %s', (limit,))
        alerts = cursor.fetchall()
        conn.close()
        return [{"timestamp": a['timestamp'], "severity": a['severity'], "message": a['message'], "metrics": json.loads(a['metrics'])} for a in alerts]

    def get_latest_performance(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT timestamp, roc_auc, ece, brier, mae, rmse FROM performance_metrics ORDER BY id DESC LIMIT 1')
        res = cursor.fetchone()
        conn.close()
        if res:
            return {"timestamp": res['timestamp'], "roc_auc": res['roc_auc'], "ece": res['ece'], "brier": res['brier'], "mae": res['mae'], "rmse": res['rmse']}
        return {}
