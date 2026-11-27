import time
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium
from supabase import create_client

st.set_page_config(page_title="MargRakshak Live Telemetry", layout="wide")


@st.cache_resource
def get_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


@st.cache_data(ttl=5)
def fetch_trips():
    sb = get_supabase()
    res = (
        sb.table("telemetry_events")
        .select("trip_id, recorded_at")
        .order("recorded_at", desc=True)
        .limit(2000)
        .execute()
    )
    if not res.data:
        return []
    seen = []
    for row in res.data:
        tid = row["trip_id"]
        if tid not in seen:
            seen.append(tid)
    return seen


@st.cache_data(ttl=5)
def fetch_trip_telemetry(trip_id: str):
    sb = get_supabase()
    res = (
        sb.table("telemetry_events")
        .select("*")
        .eq("trip_id", trip_id)
        .order("recorded_at", desc=False)
        .limit(5000)
        .execute()
    )
    return pd.DataFrame(res.data)


st.sidebar.header("Live Trip Controls")
live_mode = st.sidebar.toggle("Auto-refresh live stream (5s)", value=True)
if st.sidebar.button("Refresh Now"):
    st.cache_data.clear()

trips = fetch_trips()
if not trips:
    st.title("MargRakshak — Live Road Condition & Pothole Telemetry")
    st.info("Waiting for first telemetry batch from Raspberry Pi...")
    if live_mode:
        time.sleep(5)
        st.rerun()
    st.stop()

selected_trip = st.sidebar.selectbox("Select Trip", trips)
simulate_gps = st.sidebar.checkbox("Simulate GPS track if indoor (NO_FIX)", value=True)

df = fetch_trip_telemetry(selected_trip)
df["recorded_at"] = pd.to_datetime(df["recorded_at"])
df["elapsed_sec"] = (df["recorded_at"] - df["recorded_at"].iloc[0]).dt.total_seconds()

st.title("MargRakshak — Live Road Condition & Pothole Telemetry")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Logged Frames", f"{len(df):,}")
k2.metric("Pothole Detections", int(df["pothole_count"].sum()))
k3.metric("Rough Road Segments", int((df["roughness"] != "Smooth").sum()))
k4.metric("Peak Impact Spike", f"{df['peak_spike_g'].max():.2f} g")
fix_pct = int(round((df["gps_fix"] > 0).mean() * 100))
k5.metric("GPS Satellite Lock", f"{fix_pct}%")

valid_gps = df[(df["gps_fix"] > 0) & df["lat"].notna() & df["lon"].notna()].copy()
if valid_gps.empty and simulate_gps:
    st.info("Indoor test trip detected (gps_fix = 0). Displaying simulated Pimpri-Chinchwad trajectory.")
    n = len(df)
    df["plot_lat"] = np.linspace(18.6298, 18.7550, n)
    df["plot_lon"] = np.linspace(73.7997, 73.9200, n)
    map_df = df
else:
    df["plot_lat"] = df["lat"]
    df["plot_lon"] = df["lon"]
    map_df = valid_gps

left, right = st.columns(2)

with left:
    st.subheader("Geolocated Road Roughness & Pothole Map")
    if not map_df.empty:
        center = [map_df["plot_lat"].mean(), map_df["plot_lon"].mean()]
        m = folium.Map(location=center, zoom_start=12)
        coords = list(zip(map_df["plot_lat"], map_df["plot_lon"]))
        if len(coords) > 1:
            folium.PolyLine(coords, color="#3b82f6", weight=4, opacity=0.85).add_to(m)

        anomalies = map_df[(map_df["pothole_count"] > 0) | (map_df["roughness"] != "Smooth")]
        for _, r in anomalies.iterrows():
            color = "red" if r["pothole_count"] > 0 else "orange"
            folium.CircleMarker(
                location=[r["plot_lat"], r["plot_lon"]],
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.9,
                popup=f"{r['severity']} | Potholes: {r['pothole_count']} | Spike: {r['peak_spike_g']:.2f}g",
            ).add_to(m)
        st_folium(m, width=None, height=380, returned_objects=[])
    else:
        st.warning("No GPS coordinates locked yet and simulation is disabled.")

with right:
    st.subheader("50 Hz IMU Vibration & Pothole Timeline")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["elapsed_sec"],
            y=df["peak_spike_g"],
            mode="lines",
            name="Peak Spike (g)",
            line=dict(color="#60a5fa", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["elapsed_sec"],
            y=df["rms_vibration_g"],
            mode="lines",
            name="RMS Vibration (g)",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
        )
    )
    potholes = df[df["pothole_count"] > 0]
    if not potholes.empty:
        fig.add_trace(
            go.Scatter(
                x=potholes["elapsed_sec"],
                y=potholes["peak_spike_g"],
                mode="markers",
                name="Pothole Detected",
                marker=dict(color="#ef4444", size=9, symbol="diamond"),
            )
        )
    fig.add_hline(y=0.45, line_dash="dash", line_color="#ef4444", annotation_text="Rough Threshold (0.45g)")
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Trip Elapsed Time (seconds)",
        yaxis_title="Acceleration Spike (g)",
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Captured Pothole & Bump Evidence Frames")
evidence = df[df["frame_url"].notna() & (df["frame_url"] != "")].tail(12)
if evidence.empty:
    st.caption("No evidence frames uploaded for this trip yet.")
else:
    cols = st.columns(4)
    for idx, (_, row) in enumerate(evidence.iterrows()):
        with cols[idx % 4]:
            st.image(
                row["frame_url"],
                caption=f"{row['severity']} | Potholes: {row['pothole_count']} ({row['max_pothole_conf']:.2f}) | {row['peak_spike_g']:.2f}g",
                use_container_width=True,
            )

if live_mode:
    time.sleep(5)
    st.rerun()
