"""
"Cloud" side: FastAPI telemetry API (receives the JSON events of the vision workers).

Run it alone :  uvicorn api:app --port 8000      (step2 can also start it for you)
Docs         :  http://127.0.0.1:8000/docs

Endpoints
- POST /events      store one event (validated by Pydantic: bad JSON -> HTTP 422, not 500)
- GET  /events      last events (?event_type=dwell&limit=20)
- GET  /kpi/hourly  rebuild + return the hourly KPIs per zone (table kpi_hourly)
- GET  /trajectories  path of each person through the zones
"""
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field, model_validator

import telemetry_db as tdb

EventType = Literal["zone_count", "dwell", "stockout", "alert"]


class Event(BaseModel):
    """One structured telemetry event. Example (zone_count):
    {"event_type": "zone_count", "timestamp": "2026-11-24T10:15:02Z",
     "zone_name": "queue zone", "object_count": 2, "track_ids": [3, 5]}"""
    event_type: EventType = "zone_count"
    timestamp: datetime
    zone_name: str = Field(min_length=1)
    object_count: Optional[int] = Field(default=None, ge=0)
    track_ids: list[int] = Field(default_factory=list)
    track_id: Optional[int] = None
    dwell_seconds: Optional[float] = Field(default=None, ge=0)
    run_id: Optional[str] = None
    details: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_by_type(self):
        if self.event_type in ("zone_count", "stockout", "alert") and self.object_count is None:
            raise ValueError(f"{self.event_type} needs object_count")
        if self.event_type in ("zone_count", "alert") and self.track_ids \
                and self.object_count != len(self.track_ids):
            # the count must be exactly the number of ByteTrack IDs (deterministic count)
            raise ValueError("object_count must equal len(track_ids)")
        if self.event_type == "dwell" and (self.track_id is None or self.dwell_seconds is None):
            raise ValueError("dwell needs track_id and dwell_seconds")
        return self

    def to_row(self):
        d = self.model_dump()
        ts = self.timestamp if self.timestamp.tzinfo else self.timestamp.replace(tzinfo=timezone.utc)
        d["timestamp"] = ts.astimezone(timezone.utc).isoformat()   # always stored in UTC
        return d


app = FastAPI(title="Vision telemetry (Smart Retail)")


@app.post("/events")
def add_event(event: Event):
    conn = tdb.connect()             # one connection per request: simple and thread-safe
    try:
        tdb.insert_event(conn, event.to_row())
    finally:
        conn.close()
    return {"status": "stored", "event_type": event.event_type}


@app.get("/events")
def get_events(limit: int = Query(50, ge=1, le=1000), event_type: Optional[EventType] = None):
    conn = tdb.connect()
    try:
        return tdb.last_events(conn, limit, event_type)
    finally:
        conn.close()


@app.get("/kpi/hourly")
def kpi_hourly():
    conn = tdb.connect()
    try:
        return tdb.rebuild_hourly_kpis(conn)
    finally:
        conn.close()


@app.get("/trajectories")
def get_trajectories(limit: int = Query(100, ge=1, le=1000)):
    conn = tdb.connect()
    try:
        return tdb.trajectories(conn, limit)
    finally:
        conn.close()
