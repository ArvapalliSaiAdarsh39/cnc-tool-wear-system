"""
log_database.py
================
SQLite-backed persistent storage for maintenance work-order logs (Tab 4).

WHAT THIS SOLVES: logs stored only in st.session_state vanish when a browser
tab closes or a session resets. Writing them to a local SQLite file (logs.db)
fixes that -- data survives tab close/reopen and is shared across concurrent
sessions hitting the same running server process.

WHAT THIS DOES NOT SOLVE -- Streamlit Community Cloud has an EPHEMERAL
filesystem. Community Cloud recycles the container's local disk on every
redeploy, code push, and periodic restart for inactivity -- logs.db is wiped
along with it. This is a platform constraint, not a bug in this module: true
persistence *across Community Cloud restarts* requires external storage (a
hosted Postgres/MySQL instance, a cloud object store, Google Sheets via API,
etc.), not a local file, regardless of which library writes it. The
export_backup_bytes()/import_backup() pair below is a practical mitigation
for this specific platform: export a full backup before a known redeploy,
re-import it after. app.py surfaces this tradeoff in the Tab 4 UI rather
than implying "permanent" means something stronger than it does here.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sqlite3
from contextlib import contextmanager
from typing import Any, Optional

DB_PATH = "logs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS maintenance_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_order_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    operator_name TEXT DEFAULT '',
    work_order_status TEXT DEFAULT 'Open',
    operator_notes TEXT DEFAULT '',
    material TEXT, coolant TEXT, coating TEXT,
    spindle_rpm REAL, feed_rate_mm_rev REAL, axial_depth_mm REAL, radial_width_mm REAL,
    workpiece_diameter_mm REAL, num_flutes INTEGER, num_spindles INTEGER, nose_radius_mm REAL,
    runtime_min REAL, cutting_speed_m_min REAL, taylor_predicted_life_min REAL, taylor_ratio REAL,
    mrr_cm3_min REAL, surface_roughness_ra_um REAL, cumulative_energy_kj REAL,
    wear_q10 REAL, wear_q50 REAL, wear_q90 REAL,
    scrap_probability REAL, scrap_tier TEXT, health_status TEXT,
    bulletin_text TEXT, bulletin_source TEXT,
    full_snapshot_json TEXT
);
"""

WORK_ORDER_STATUSES = ["Open", "In Progress", "Resolved", "Closed"]

# Structured columns pulled straight from a Tab 1 snapshot dict; anything not
# listed here still gets preserved in full_snapshot_json for the detail view.
_SNAPSHOT_COLUMNS = [
    "material", "coolant", "coating", "spindle_rpm", "feed_rate_mm_rev", "axial_depth_mm",
    "radial_width_mm", "workpiece_diameter_mm", "num_flutes", "num_spindles", "nose_radius_mm",
    "runtime_min", "cutting_speed_m_min", "taylor_predicted_life_min", "taylor_ratio",
    "mrr_cm3_min", "surface_roughness_ra_um", "cumulative_energy_kj",
    "wear_q10", "wear_q50", "wear_q90", "scrap_probability", "scrap_tier", "health_status",
    "bulletin_text", "bulletin_source",
]


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Idempotent -- safe to call on every app startup/rerun."""
    with _connect() as conn:
        conn.execute(_SCHEMA)


def _to_native(value: Any) -> Any:
    """Coerce numpy/pandas scalar types to native Python types before they
    reach sqlite3's parameter binding. sqlite3 only has built-in adapters
    for native int/float/str/bytes/None -- a numpy.float32 (exactly what
    model.predict() and pandas Series scalars hand back) is NOT one of
    those, and silently lands as a BLOB of pickled bytes instead of a REAL
    column, which then breaks on the very first `round()` or arithmetic a
    reader does on it. json.dumps() has the identical blind spot, so this
    matters for full_snapshot_json too, not just the structured columns."""
    if value is None:
        return None
    if hasattr(value, "item"):  # numpy scalar (float32, int64, bool_, ...)
        return value.item()
    return value


def _json_safe_snapshot(snapshot: dict) -> dict:
    """Drop non-JSON-serializable entries (the MaterialProfile dataclass
    instance) and coerce numpy scalars before persisting the full snapshot
    as a JSON blob."""
    return {k: _to_native(v) for k, v in snapshot.items() if k != "material_obj"}


def insert_log(snapshot: dict, operator_name: str = "", operator_notes: str = "",
                work_order_status: str = "Open") -> int:
    """Persists one Tab 1 snapshot as a new log row. Returns the new row id."""
    now = dt.datetime.now().isoformat(timespec="seconds")
    work_order_id = f"WO-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    row = {col: _to_native(snapshot.get(col)) for col in _SNAPSHOT_COLUMNS}
    row["full_snapshot_json"] = json.dumps(_json_safe_snapshot(snapshot), default=str)

    columns = ["work_order_id", "created_at", "updated_at", "operator_name",
               "work_order_status", "operator_notes"] + list(row.keys())
    values = [work_order_id, now, now, operator_name, work_order_status, operator_notes] + list(row.values())
    placeholders = ", ".join("?" * len(values))

    with _connect() as conn:
        cur = conn.execute(
            f"INSERT INTO maintenance_logs ({', '.join(columns)}) VALUES ({placeholders})", values
        )
        return cur.lastrowid


def get_all_logs() -> list[dict]:
    """All logs, most recent first."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM maintenance_logs ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def get_log(log_id: int) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM maintenance_logs WHERE id = ?", (log_id,)).fetchone()
        return dict(row) if row else None


def update_log(log_id: int, operator_name: Optional[str] = None, operator_notes: Optional[str] = None,
                work_order_status: Optional[str] = None) -> bool:
    """UPDATE-only surface: annotation fields (who/notes/status), not the
    historical telemetry -- retroactively editing logged sensor readings
    would corrupt the audit trail the log exists to provide."""
    fields, values = [], []
    for col, val in [("operator_name", operator_name), ("operator_notes", operator_notes),
                      ("work_order_status", work_order_status)]:
        if val is not None:
            fields.append(f"{col} = ?")
            values.append(val)
    if not fields:
        return False
    fields.append("updated_at = ?")
    values.append(dt.datetime.now().isoformat(timespec="seconds"))
    values.append(log_id)

    with _connect() as conn:
        cur = conn.execute(f"UPDATE maintenance_logs SET {', '.join(fields)} WHERE id = ?", values)
        return cur.rowcount > 0


def delete_log(log_id: int) -> bool:
    """Explicit, single-row delete. No cascade, no bulk delete -- the caller
    (app.py) is responsible for requiring an explicit confirmation step
    before calling this."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM maintenance_logs WHERE id = ?", (log_id,))
        return cur.rowcount > 0


def logs_to_dataframe():
    import pandas as pd
    logs = get_all_logs()
    if not logs:
        return pd.DataFrame()
    df = pd.DataFrame(logs)
    return df.drop(columns=["full_snapshot_json"], errors="ignore")


def export_csv_bytes() -> bytes:
    df = logs_to_dataframe()
    return df.to_csv(index=False).encode("utf-8")


def export_backup_bytes() -> bytes:
    """Raw bytes of the current logs.db file, for the operator to download
    as a manual backup ahead of a known Streamlit Community Cloud redeploy
    (see module docstring -- this is a mitigation, not a substitute for
    real external persistence)."""
    init_db()
    with open(DB_PATH, "rb") as f:
        return f.read()


def import_backup(uploaded_bytes: bytes) -> None:
    """Restores logs.db from a previously exported backup. Overwrites the
    current database file -- callers should confirm with the user first."""
    with open(DB_PATH, "wb") as f:
        f.write(uploaded_bytes)


if __name__ == "__main__":
    import os
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    init_db()

    demo_snapshot = {
        "material": "Titanium Alloy (Ti-6Al-4V)", "coolant": "Flood", "coating": "CBN",
        "spindle_rpm": 320.0, "feed_rate_mm_rev": 0.11, "axial_depth_mm": 0.6, "radial_width_mm": 0.8,
        "workpiece_diameter_mm": 45.0, "num_flutes": 2, "num_spindles": 1, "nose_radius_mm": 0.4,
        "runtime_min": 100.0, "cutting_speed_m_min": 45.2, "taylor_predicted_life_min": 168.4,
        "taylor_ratio": 0.594, "mrr_cm3_min": 0.0028, "surface_roughness_ra_um": 0.95,
        "cumulative_energy_kj": 12.5, "wear_q10": 0.15, "wear_q50": 0.18, "wear_q90": 0.22,
        "scrap_probability": 0.12, "scrap_tier": "Low", "health_status": "OPTIMAL",
        "bulletin_text": "Test bulletin.", "bulletin_source": "fallback_template",
    }
    new_id = insert_log(demo_snapshot, operator_name="J. Rivera", operator_notes="Routine check")
    print("inserted id:", new_id)
    print("all logs:", len(get_all_logs()))
    print("get_log:", get_log(new_id)["work_order_id"])

    assert update_log(new_id, work_order_status="Resolved", operator_notes="Confirmed within spec")
    print("after update, status:", get_log(new_id)["work_order_status"])

    csv_bytes = export_csv_bytes()
    print("csv export bytes:", len(csv_bytes))
    backup_bytes = export_backup_bytes()
    print("backup export bytes:", len(backup_bytes))

    assert delete_log(new_id)
    print("after delete, all logs:", len(get_all_logs()))
    print("ALL OK")
