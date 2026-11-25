"""
Step 5 — BI dashboard (Streamlit).

What it does : reads data/events.db (filled by step2 and step4) and shows
               per-zone KPIs (occupancy, queue wait time, unique visitors), the
               HOURLY KPI table, people over time, trajectories through the zones,
               alerts / stockouts, the latest alert images and the heatmap.
Assignment   : BI dashboard of the example app (Streamlit).
How to run   : 1) python step2_telemetry.py --source data/people.mp4   (fills the database)
               2) streamlit run step5_dashboard.py
"""
import glob
import os

import pandas as pd
import streamlit as st

import telemetry_db as tdb

ALERT_GLOB = "output/alert_frame_*.jpg"
STOCKOUT_GLOB = "output/stockout_*.jpg"
HEATMAP_PATH = "output/heatmap.jpg"


@st.cache_data(ttl=5)
def load():
    """Returns (events, hourly KPIs, trajectories) as DataFrames."""
    if not os.path.exists(tdb.DB_PATH):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    conn = tdb.connect()
    try:
        events = pd.read_sql_query("SELECT * FROM events ORDER BY id", conn)
        hourly = pd.DataFrame(tdb.rebuild_hourly_kpis(conn))   # also refreshes kpi_hourly
        paths = pd.DataFrame(tdb.trajectories(conn, limit=1000))
    finally:
        conn.close()
    events["event_type"] = events["event_type"].fillna("zone_count")
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce", format="ISO8601")
    return events.dropna(subset=["timestamp"]), hourly, paths


def average_over_time(df, step):
    """Average people per zone for each time bucket ('60min' = hour, '1min' = minute)."""
    bucket = df["timestamp"].dt.floor(step)
    table = df.groupby([bucket, df["zone"]])["count"].mean().round(1).unstack("zone")
    table.index = table.index.strftime("%d/%m %H:%M")
    return table


def latest_image(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def main():
    st.set_page_config(page_title="Smart Retail - dashboard", layout="wide")
    st.title("Smart Retail : affluence, temps d'attente et ruptures de stock")
    st.caption("Données lues dans data/events.db (événements JSON envoyés par les workers vision).")

    events, hourly, paths = load()
    counts = events[events["event_type"] == "zone_count"] if not events.empty else events
    if counts.empty:
        st.warning("Aucun événement. Lancez d'abord : python step2_telemetry.py --source data/people.mp4")
        st.stop()

    # ---- Sidebar: filters ----
    zones = sorted(counts["zone"].unique())
    chosen = st.sidebar.multiselect("Zones", zones, default=zones)
    unit = st.sidebar.radio("Regrouper par", ["minute", "heure"])
    threshold = st.sidebar.number_input("Seuil d'alerte (personnes)", min_value=1, value=5)
    if st.sidebar.button("Rafraîchir"):
        st.cache_data.clear()
        st.rerun()
    counts = counts[counts["zone"].isin(chosen)]
    if counts.empty:
        st.info("Choisissez au moins une zone.")
        st.stop()
    dwell = events[(events["event_type"] == "dwell") & events["zone"].isin(chosen)]

    # ---- KPIs per zone ----
    st.subheader("Indicateurs par zone")
    for col, zone in zip(st.columns(len(chosen)), chosen):
        z = counts[counts["zone"] == zone]
        d = dwell[dwell["zone"] == zone]["dwell_seconds"]
        h = hourly[hourly["zone"] == zone] if not hourly.empty else hourly
        col.metric(f"{zone} : dernier comptage", int(z["count"].iloc[-1]) if len(z) else 0)
        col.metric("Temps moyen dans la zone", f"{d.mean():.1f} s" if len(d) else "-")
        col.write(f"moyenne **{z['count'].mean():.1f}** · pic **{int(z['count'].max())}** · "
                  f"visiteurs uniques **{int(h['unique_visitors'].sum()) if len(h) else 0}** · "
                  f"seuil atteint **{int((z['count'] >= threshold).sum())}** fois")

    # ---- People over time ----
    st.subheader("Nombre moyen de personnes dans le temps")
    st.bar_chart(average_over_time(counts, "60min" if unit == "heure" else "1min"))

    # ---- Hourly KPIs (structured, stored in table kpi_hourly) ----
    st.subheader("KPI horaires (table kpi_hourly)")
    if not hourly.empty:
        st.dataframe(hourly[hourly["zone"].isin(chosen)].rename(columns={
            "hour": "heure (UTC)", "zone": "zone", "samples": "mesures",
            "avg_count": "occupation moy.", "peak_count": "pic",
            "unique_visitors": "visiteurs uniques", "visits": "passages",
            "avg_dwell_s": "attente moy. (s)", "max_dwell_s": "attente max (s)",
            "alerts": "alertes", "stockouts": "ruptures"}),
            use_container_width=True, hide_index=True)

    # ---- Wait time + trajectories ----
    left, right = st.columns(2)
    with left:
        st.subheader("Temps passé par zone (dwell time)")
        if len(dwell):
            st.bar_chart(dwell.groupby("zone")["dwell_seconds"].mean().round(1))
            st.caption(f"{len(dwell)} passages mesurés · le plus long : "
                       f"{dwell['dwell_seconds'].max():.1f} s")
        else:
            st.info("Pas encore de temps de passage (relancez step2_telemetry.py).")
    with right:
        st.subheader("Trajectoires des clients entre zones")
        if not paths.empty:
            shown = paths.copy()
            shown["parcours"] = shown["path"].apply(lambda p: " → ".join(p))
            st.dataframe(shown[["track_id", "parcours", "total_dwell_s"]].tail(20)
                         .rename(columns={"track_id": "ID", "total_dwell_s": "durée totale (s)"}),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Aucune trajectoire.")

    # ---- Alerts and stockouts ----
    st.subheader("Alertes et ruptures de stock")
    incidents = events[events["event_type"].isin(["alert", "stockout"])]
    if len(incidents):
        shown = incidents.sort_values("id", ascending=False).head(20).copy()
        shown["timestamp"] = shown["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(shown[["timestamp", "event_type", "zone", "count", "track_ids"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("Aucune alerte enregistrée (lancez step4_alerts.py ou step2_telemetry.py --shelf ...).")

    # ---- Evidence images ----
    a, b, c = st.columns(3)
    for col, title, path, hint in (
            (a, "Dernière alerte d'affluence", latest_image(ALERT_GLOB), "step4_alerts.py"),
            (b, "Dernière rupture de stock", latest_image(STOCKOUT_GLOB), "step2_telemetry.py --shelf"),
            (c, "Heatmap d'occupation", HEATMAP_PATH if os.path.exists(HEATMAP_PATH) else None,
             "step3_benchmark.py")):
        with col:
            st.subheader(title)
            if path:
                st.image(path, caption=os.path.basename(path))
            else:
                st.info(f"Pas d'image (lancez {hint}).")


if __name__ == "__main__":
    main()
