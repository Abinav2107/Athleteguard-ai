"""
Session History and Trend Tracking Store for AthleteGuard AI.
Uses SQLite to persist each video analysis run and provide historical trend data,
supporting sport categorization, risk drivers narrative, and multi-rep consistency tracking.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "athlete_sessions.db")

def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Optional[str] = None):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            athlete_id TEXT NOT NULL,
            sport TEXT DEFAULT 'Long Jump',
            video_name TEXT,
            overall_risk_score REAL,
            risk_level TEXT,
            peak_risk_score REAL,
            peak_phase TEXT,
            landing_risk_score REAL,
            landing_risk_elevated INTEGER,
            phase_scores_json TEXT,
            components_json TEXT,
            risk_drivers TEXT,
            consistency_score REAL,
            foot_strike TEXT DEFAULT 'Undetermined',
            ankle_angle REAL,
            perspective TEXT DEFAULT 'Sagittal',
            fatigue_index REAL
        )
    """)
    
    # Auto-migration for existing database files
    cursor.execute("PRAGMA table_info(sessions)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    if 'sport' not in existing_cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN sport TEXT DEFAULT 'Long Jump'")
    if 'risk_drivers' not in existing_cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN risk_drivers TEXT")
    if 'consistency_score' not in existing_cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN consistency_score REAL")
    if 'foot_strike' not in existing_cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN foot_strike TEXT DEFAULT 'Undetermined'")
    if 'ankle_angle' not in existing_cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN ankle_angle REAL")
    if 'perspective' not in existing_cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN perspective TEXT DEFAULT 'Sagittal'")
    if 'fatigue_index' not in existing_cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN fatigue_index REAL")
        
    conn.commit()
    conn.close()

def save_session(athlete_id: str,
                 video_name: str,
                 overall_risk_score: float,
                 risk_level: str,
                 peak_risk_score: float,
                 peak_phase: str,
                 landing_risk_score: Optional[float],
                 landing_risk_elevated: bool,
                 phase_scores: Dict[str, Any],
                 component_scores: Dict[str, Any],
                 sport: str = "Long Jump",
                 risk_drivers: Optional[str] = None,
                 consistency_score: Optional[float] = None,
                 foot_strike: Optional[str] = None,
                 ankle_angle: Optional[float] = None,
                 perspective: Optional[str] = None,
                 fatigue_index: Optional[float] = None,
                 db_path: Optional[str] = None) -> int:
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    athlete_clean = athlete_id.strip() if athlete_id and athlete_id.strip() else "Unknown Athlete"
    sport_clean = sport.strip() if sport and sport.strip() else "Long Jump"
    
    cursor.execute("""
        INSERT INTO sessions (
            timestamp, athlete_id, sport, video_name, overall_risk_score, risk_level,
            peak_risk_score, peak_phase, landing_risk_score, landing_risk_elevated,
            phase_scores_json, components_json, risk_drivers, consistency_score,
            foot_strike, ankle_angle, perspective, fatigue_index
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        athlete_clean,
        sport_clean,
        video_name or "video",
        round(float(overall_risk_score), 2),
        risk_level,
        round(float(peak_risk_score), 2),
        peak_phase,
        round(float(landing_risk_score), 2) if landing_risk_score is not None else None,
        1 if landing_risk_elevated else 0,
        json.dumps(phase_scores),
        json.dumps(component_scores),
        risk_drivers or "",
        round(float(consistency_score), 2) if consistency_score is not None else None,
        foot_strike or "Undetermined",
        round(float(ankle_angle), 1) if ankle_angle is not None else None,
        perspective or "Sagittal",
        round(float(fatigue_index), 2) if fatigue_index is not None else None
    ))
    
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def get_athlete_history(athlete_id: Optional[str] = None, 
                        sport: Optional[str] = None,
                        db_path: Optional[str] = None) -> pd.DataFrame:
    init_db(db_path)
    conn = get_connection(db_path)
    
    conditions = []
    params = []
    
    if athlete_id and athlete_id.strip() and athlete_id != "All Athletes":
        conditions.append("athlete_id = ?")
        params.append(athlete_id.strip())
        
    if sport and sport.strip() and sport != "All Sports":
        conditions.append("sport = ?")
        params.append(sport.strip())
        
    if conditions:
        query = f"SELECT * FROM sessions WHERE {' AND '.join(conditions)} ORDER BY id ASC"
        df = pd.read_sql_query(query, conn, params=tuple(params))
    else:
        query = "SELECT * FROM sessions ORDER BY id ASC"
        df = pd.read_sql_query(query, conn)
        
    conn.close()
    return df

def get_all_athlete_ids(db_path: Optional[str] = None) -> List[str]:
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT athlete_id FROM sessions ORDER BY athlete_id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows if r[0]]

def get_all_sports(db_path: Optional[str] = None) -> List[str]:
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT sport FROM sessions WHERE sport IS NOT NULL ORDER BY sport ASC")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows if r[0]]

def get_session_by_id(session_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve full details of a specific session by primary key ID."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    try:
        d['phase_scores'] = json.loads(d.get('phase_scores_json', '{}'))
    except Exception:
        d['phase_scores'] = {}
    try:
        d['component_scores'] = json.loads(d.get('components_json', '{}'))
    except Exception:
        d['component_scores'] = {}
    return d

def get_best_session_for_athlete(athlete_id: str, sport: Optional[str] = None, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve the historical session with the lowest overall injury risk score for an athlete.
    Used as the athlete's personal reference benchmark for coach comparison.
    """
    if not athlete_id or not athlete_id.strip():
        return None
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    query = "SELECT * FROM sessions WHERE athlete_id = ?"
    params = [athlete_id.strip()]
    if sport and sport.strip() and sport != "All Sports":
        query += " AND sport = ?"
        params.append(sport.strip())
    query += " ORDER BY overall_risk_score ASC LIMIT 1"
    cursor.execute(query, tuple(params))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    try:
        d['phase_scores'] = json.loads(d.get('phase_scores_json', '{}'))
    except Exception:
        d['phase_scores'] = {}
    try:
        d['component_scores'] = json.loads(d.get('components_json', '{}'))
    except Exception:
        d['component_scores'] = {}
    return d

def get_squad_overview(sport: Optional[str] = None, db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Get the latest screening session for each unique athlete, ranked by risk score.
    Enables coaches and athletic directors to prioritize vulnerable athletes.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    
    query = """
        SELECT s.*
        FROM sessions s
        INNER JOIN (
            SELECT athlete_id, MAX(id) as max_id
            FROM sessions
            GROUP BY athlete_id
        ) latest ON s.id = latest.max_id
    """
    params = []
    if sport and sport.strip() and sport != "All Sports":
        query += " WHERE s.sport = ?"
        params.append(sport.strip())
        
    query += " ORDER BY s.overall_risk_score DESC"
    
    if params:
        df = pd.read_sql_query(query, conn, params=tuple(params))
    else:
        df = pd.read_sql_query(query, conn)
        
    conn.close()
    return df

def clear_all_sessions(db_path: Optional[str] = None):
    """Delete all session records to reset the database to a completely fresh state."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions")
    try:
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='sessions'")
    except Exception:
        pass
    conn.commit()
    conn.close()

