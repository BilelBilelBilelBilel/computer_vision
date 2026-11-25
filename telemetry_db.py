"""
"Cloud" side storage: SQLite database + KPI aggregation (no web framework here,
so the API, the dashboard and the tests can all reuse the same functions).

Tables
- events     : every JSON event sent by the vision workers
               event_type = zone_count | dwell | stockout | alert
- kpi_hourly : one row per (hour, zone) with the business KPIs, rebuilt from events

In production this would be PostgreSQL (or TimescaleDB for time series): only the
connection changes, the SQL stays almost the same.
"""
import json
import sqlite3
from collections import defaultdict

DB_PATH = "data/events.db"

EVENT_COLUMNS = {                 # column -> SQL type (added to old databases if missing)
    "timestamp": "TEXT",
    "zone": "TEXT",
    "count": "INTEGER",
    "event_type": "TEXT DEFAULT 'zone_count'",
    "track_ids": "TEXT",          # JSON list, e.g. "[3, 5]"
    "track_id": "INTEGER",
    "dwell_seconds": "REAL",
    "run_id": "TEXT",             # one id per worker launch (ByteTrack restarts at #1)
    "details": "TEXT",            # JSON object with extra fields
}


def connect(path=DB_PATH):
    """Open the DB, create the tables, and upgrade a database made by the old version."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT)")
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(events)")}
    for col, sql_type in EVENT_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE events ADD COLUMN {col} {sql_type}")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS kpi_hourly ("
        " hour TEXT, zone TEXT,"
        " samples INTEGER, avg_count REAL, peak_count INTEGER,"
        " unique_visitors INTEGER, visits INTEGER, avg_dwell_s REAL, max_dwell_s REAL,"
        " alerts INTEGER, stockouts INTEGER,"
        " PRIMARY KEY (hour, zone))")
    conn.commit()
    return conn


def insert_event(conn, event):
    """event: a dict produced by api.Event.model_dump()."""
    conn.execute(
        "INSERT INTO events (timestamp, zone, count, event_type, track_ids, track_id,"
        " dwell_seconds, run_id, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (event["timestamp"], event["zone_name"], event.get("object_count"),
         event["event_type"], json.dumps(event.get("track_ids") or []),
         event.get("track_id"), event.get("dwell_seconds"), event.get("run_id"),
         json.dumps(event.get("details") or {})))
    conn.commit()


def row_to_dict(r):
    d = dict(r)
    for key in ("track_ids", "details"):
        try:
            d[key] = json.loads(d[key]) if d.get(key) else ([] if key == "track_ids" else {})
        except (TypeError, ValueError):
            pass
    return d


def last_events(conn, limit=50, event_type=None):
    sql, args = "SELECT * FROM events", []
    if event_type:
        sql, args = sql + " WHERE event_type = ?", [event_type]
    rows = conn.execute(sql + " ORDER BY id DESC LIMIT ?", args + [limit]).fetchall()
    return [row_to_dict(r) for r in rows]


def hour_of(timestamp):
    """'2026-11-24T14:07:51.95+00:00' -> '2026-11-24T14:00' (timestamps are UTC ISO)."""
    return timestamp[:13] + ":00"


def rebuild_hourly_kpis(conn):
    """Aggregate all events into kpi_hourly (structured hourly KPIs) and return the rows."""
    k = defaultdict(lambda: {"samples": 0, "sum": 0, "peak": 0, "ids": set(),
                             "dwells": [], "alerts": 0, "stockouts": 0})
    for r in conn.execute("SELECT * FROM events"):
        if not r["timestamp"] or not r["zone"]:
            continue
        a = k[(hour_of(r["timestamp"]), r["zone"])]
        run = r["run_id"] or "legacy"
        etype = r["event_type"] or "zone_count"
        if etype == "zone_count" and r["count"] is not None:
            a["samples"] += 1
            a["sum"] += r["count"]
            a["peak"] = max(a["peak"], r["count"])
            for tid in json.loads(r["track_ids"] or "[]"):
                a["ids"].add((run, tid))
        elif etype == "dwell":
            a["dwells"].append(r["dwell_seconds"] or 0)
            a["ids"].add((run, r["track_id"]))
        elif etype == "alert":
            a["alerts"] += 1
        elif etype == "stockout":
            a["stockouts"] += 1

    rows = []
    for (hour, zone), a in sorted(k.items()):
        d = a["dwells"]
        rows.append({
            "hour": hour, "zone": zone,
            "samples": a["samples"],
            "avg_count": round(a["sum"] / a["samples"], 2) if a["samples"] else None,
            "peak_count": a["peak"] if a["samples"] else None,
            "unique_visitors": len(a["ids"]),
            "visits": len(d),
            "avg_dwell_s": round(sum(d) / len(d), 2) if d else None,
            "max_dwell_s": round(max(d), 2) if d else None,
            "alerts": a["alerts"],
            "stockouts": a["stockouts"],
        })
    conn.execute("DELETE FROM kpi_hourly")
    conn.executemany(
        "INSERT INTO kpi_hourly VALUES (:hour, :zone, :samples, :avg_count, :peak_count,"
        " :unique_visitors, :visits, :avg_dwell_s, :max_dwell_s, :alerts, :stockouts)", rows)
    conn.commit()
    return rows


def trajectories(conn, limit=100):
    """Path of each tracked person through the zones, built from the dwell events.
    Example: {"track_id": 7, "path": ["entrance zone", "queue zone"], "total_dwell_s": 9.5}"""
    paths = defaultdict(list)
    for r in conn.execute("SELECT * FROM events WHERE event_type = 'dwell' ORDER BY id"):
        details = json.loads(r["details"] or "{}")
        paths[(r["run_id"], r["track_id"])].append(
            (details.get("entered_s", 0), r["zone"], r["dwell_seconds"] or 0))
    out = []
    for (run_id, tid), visits in paths.items():
        visits.sort()
        out.append({
            "run_id": run_id, "track_id": tid,
            "path": [zone for _, zone, _ in visits],
            "first_seen_s": visits[0][0],
            "total_dwell_s": round(sum(s for _, _, s in visits), 2),
        })
    out.sort(key=lambda t: (t["run_id"] or "", t["first_seen_s"]))
    return out[-limit:]
