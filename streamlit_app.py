import streamlit as st
from fitparse import FitFile
import io
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
import seaborn as sns
import math

st.set_page_config(page_title="DU COACHING RACE Analyzer", layout="wide")
st.title("📊 DU COACHING RACE Analyzer")
st.info("This analyzer is brought to you by coach Davide Ambrosini")

# ----------------------------
# --- FIT FILE UPLOADER ---
# ----------------------------
uploaded_file = st.file_uploader("Upload a .fit file", type=["fit"])

if uploaded_file is not None:
    st.success("FIT file uploaded!")
    try:
        fitfile = FitFile(io.BytesIO(uploaded_file.getvalue()))
        # --- Build consistent DataFrame ---
        records = []
        for rec in fitfile.get_messages("record"):
            row = {}
            for field in rec:
                if field.name in ["timestamp","heart_rate","distance","enhanced_altitude","position_lat","position_long"]:
                    row[field.name] = field.value
            if row:
                records.append(row)
        if not records:
            st.warning("⚠️ No usable records in this FIT file.")
            st.stop()
        df = pd.DataFrame(records)

        # Fill missing columns
        for col in ["heart_rate","distance","enhanced_altitude","position_lat","position_long"]:
            if col not in df.columns:
                df[col] = np.nan

        # Convert units
        df["distance_km"] = df["distance"].apply(lambda x: x/1000 if pd.notna(x) else np.nan)
        df["elevation_m"] = df["enhanced_altitude"]
        df["lat"] = df["position_lat"].apply(lambda s: s*(180/2**31) if pd.notna(s) else np.nan)
        df["lon"] = df["position_long"].apply(lambda s: s*(180/2**31) if pd.notna(s) else np.nan)

        # Elapsed time
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            start_time = df["timestamp"].iloc[0]
            df["elapsed_sec"] = (df["timestamp"] - start_time).dt.total_seconds()
        else:
            df["elapsed_sec"] = np.arange(len(df))
        df["elapsed_hours"] = df["elapsed_sec"]/3600

        # Smooth HR
        if "heart_rate" in df.columns:
            df["hr_smooth"] = df["heart_rate"].rolling(window=3, min_periods=1).mean()
        else:
            df["hr_smooth"] = np.nan

        # Summary metrics
        kilometers = df["distance_km"].max() if "distance_km" in df.columns else 0
        total_elevation_gain = df["elevation_m"].diff().clip(lower=0).sum() if "elevation_m" in df.columns else 0
        start_coords = (df["lat"].iloc[0], df["lon"].iloc[0]) if "lat" in df.columns and "lon" in df.columns else (None,None)
        end_coords = (df["lat"].iloc[-1], df["lon"].iloc[-1]) if "lat" in df.columns and "lon" in df.columns else (None,None)

        st.session_state['kilometers'] = int(kilometers)
        st.session_state['total_elevation_gain'] = int(total_elevation_gain)
        st.markdown(f"**Start Coordinates:** {start_coords}")
        st.markdown(f"**End Coordinates:** {end_coords}")

    except Exception as e:
        st.error(f"❌ Error reading FIT file: {e}")
else:
    st.info("👆 Please upload a .fit file to begin.")

# --- Athlete and race info form ---
with st.form("race_info_form"):
    athlete_name = st.text_input("🏃 Athlete's Name", value=st.session_state.get('athlete_name', ''))
    race_name = st.text_input("🏁 Race to be Analyzed", value=st.session_state.get('race_name', ''))
    race_date = st.date_input("📅 Date of the Race", value=st.session_state.get('race_date'))
    kilometers = st.session_state.get('kilometers', None)
    if kilometers is not None:
        st.markdown(f"📏 **Distance Run:** {kilometers} km")
    else:
        st.markdown("📏 **Distance Run:** _waiting for FIT file upload_")
    total_elevation_gain = st.session_state.get('total_elevation_gain', None)
    if total_elevation_gain is not None:
        st.markdown(f"🏔️ **Elevation gain:** {total_elevation_gain} m")
    else:
        st.markdown("🏔️ **Elevation gain:** _waiting for FIT file upload_")

    info_submitted = st.form_submit_button("Submit Info")

if info_submitted:
    st.session_state['athlete_name'] = athlete_name
    st.session_state['race_name'] = race_name
    st.session_state['race_date'] = race_date
    st.success("✅ Form submitted successfully!")

# --- Heart Rate Zones Form ---
with st.form("hr_zones_form"):
    st.subheader("❤️ Athlete Heart Rate Zones")
    st.caption("Please input the *upper limit (in bpm)* for each training zone:")

    z1 = st.number_input("Zone 1 (Recovery) - up to:", min_value=60, step=1, value=st.session_state.get('z1', 140))
    z2 = st.number_input("Zone 2 (Aerobic) - up to:", min_value=60, step=1, value=st.session_state.get('z2', 160))
    z3 = st.number_input("Zone 3 (Tempo) - up to:", min_value=60, step=1, value=st.session_state.get('z3', 170))
    z4 = st.number_input("Zone 4 (Sub Threshold) - up to:", min_value=60, step=1, value=st.session_state.get('z4', 180))
    z5 = st.number_input("Zone 5 (Super Threshold) - up to:", min_value=60, step=1, value=st.session_state.get('z5', 200))

    zones_submitted = st.form_submit_button("Submit HR Zones")

if zones_submitted:
    if not (z1 < z2 < z3 < z4 < z5):
        st.error("⚠️ There's something wrong in the HR data. Please correct the values.")
    else:
        st.session_state['z1'], st.session_state['z2'], st.session_state['z3'], st.session_state['z4'], st.session_state['z5'] = z1, z2, z3, z4, z5
        st.success("✅ Heart Rate Zones saved successfully!")
        st.write(f"""
        **HR Zones:**
        - 🩵 Zone 1 (Recovery): ≤ {z1} bpm  
        - 💚 Zone 2 (Aerobic): {z1+1} - {z2} bpm  
        - 💛 Zone 3 (Tempo): {z2+1} - {z3} bpm  
        - 🧡 Zone 4 (Sub Threshold): {z3+1} - {z4} bpm  
        - ❤️ Zone 5 (Super Threshold): {z4+1} - {z5} bpm
        """)

# --- Time Segment Input Form (Start → End) ---
with st.form("time_segment_form"):
    st.subheader("⏱️ Time segments for Time in Zone Analysis")
    st.caption("Please choose your time segment (H:MM) for the time-in-zone analysis")

    col1, col2 = st.columns(2)

    with col1:
        segment1_start = st.text_input("Segment 1 Start", value=st.session_state.get('segment1_start', "0:00"))
        segment2_start = st.text_input("Segment 2 Start", value=st.session_state.get('segment2_start', "0:00"))
        segment3_start = st.text_input("Segment 3 Start", value=st.session_state.get('segment3_start', "0:00"))

    with col2:
        segment1_end = st.text_input("Segment 1 End", value=st.session_state.get('segment1_end', "1:00"))
        segment2_end = st.text_input("Segment 2 End", value=st.session_state.get('segment2_end', "2:00"))
        segment3_end = st.text_input("Segment 3 End", value=st.session_state.get('segment3_end', "3:00"))

    segments_submitted = st.form_submit_button("Save Time Segments")

if segments_submitted:
    st.session_state['segment1_start'], st.session_state['segment1_end'] = segment1_start, segment1_end
    st.session_state['segment2_start'], st.session_state['segment2_end'] = segment2_start, segment2_end
    st.session_state['segment3_start'], st.session_state['segment3_end'] = segment3_start, segment3_end
    st.success("✅ Time segments saved successfully!")

# HELPER FOR LAP DETECTION
# --- Helper functions ---
def semicircles_to_degrees(s):
    return s * (180.0 / 2**31)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def seconds_to_hhmm(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h:02d}:{m:02d}"

st.markdown("## 📋 Lap / Climb Analysis")

# -----------------------------------------------------------
# 1️⃣ SELECT ANALYSIS TYPE (LAP / CLIMB / NONE)
# -----------------------------------------------------------
analysis_type = st.radio(
    "Which type of analysis would you like to perform?",
    ("None", "Lap Analysis", "Climb Analysis"),
    index=0,
    key="analysis_type_selector"
)

# Store clean value in session_state
if analysis_type == "Lap Analysis":
    st.session_state["analysis_type"] = "lap"
    st.session_state["do_lap_analysis"] = True
elif analysis_type == "Climb Analysis":
    st.session_state["analysis_type"] = "climb"
    st.session_state["do_lap_analysis"] = True
else:
    st.session_state["analysis_type"] = None
    st.session_state["do_lap_analysis"] = False

# ---------------------------
# LAP ANALYSIS WITH AUTO LOOP DETECTOR
# ---------------------------
if st.session_state.get("do_lap_analysis") and st.session_state.get("analysis_type") == "lap":
    lap_data_insert = st.radio(
        "How do you want to insert lap data?",
        ("Manually", "Automatic Lap detector", "Distance slicer"),
        index=0,
        key="lap_data_insert_selector"
    )

    # Store clean value
    if lap_data_insert == "Manually":
        st.session_state["lap_data_insert"] = "manual"
    elif lap_data_insert == "Automatic Lap detector":
        st.session_state["lap_data_insert"] = "automatic"
    else:
        st.session_state["lap_data_insert"] = "fix_distance"

# ---------------------------
# AUTOMATIC LOOP DETECTOR
# ---------------------------
loops = []  # default empty
if st.session_state.get("lap_data_insert") == "automatic":
    if uploaded_file is not None:
        # --- Parameters ---
        col1, col2, col3 = st.columns(3)
        with col1:
            base_radius = st.number_input("Base radius (meters)", value=20, min_value=1)
        with col2:
            percent_error = st.slider("GPS antenna error (%)", 0.0, 5.0, 1.0)/100
        with col3:
            min_samples_between_crossings = st.slider("Min samples between crossings", 1, 50, 5)

        # --- Read FIT file ---
        fitfile = FitFile(uploaded_file)
        records = []
        for rec in fitfile.get_messages("record"):
            row = {field.name: field.value for field in rec}
            records.append(row)
        df = pd.DataFrame(records)

        if "timestamp" not in df.columns or "position_lat" not in df.columns or "position_long" not in df.columns:
            st.error("Missing required GPS/timestamp data in FIT file.")
            st.stop()

        df = df.dropna(subset=["position_lat", "position_long", "timestamp"]).reset_index(drop=True)
        df["lat"] = df["position_lat"].apply(lambda s: s * (180.0 / 2**31))
        df["lon"] = df["position_long"].apply(lambda s: s * (180.0 / 2**31))

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        start_time = df["timestamp"].iloc[0]
        df["elapsed_sec"] = (df["timestamp"] - start_time).dt.total_seconds()

        # --- Compute distances from start ---
        start_lat = df.loc[0, "lat"]
        start_lon = df.loc[0, "lon"]
        start_time = pd.to_datetime(df.loc[0, "timestamp"])
        df["dist_from_start_m"] = df.apply(lambda r: haversine(start_lat, start_lon, r["lat"], r["lon"]), axis=1)
        threshold = base_radius + percent_error * df["dist_from_start_m"].max()

        # --- Loop detection ---
        crossings = []
        first_inside = df.loc[0, "dist_from_start_m"] <= threshold
        if first_inside:
            crossings.append((0, df.loc[0, "timestamp"]))

        inside = first_inside
        last_crossing_idx = 0 if first_inside else -1

        for idx in range(1, len(df)):
            d = df.loc[idx, "dist_from_start_m"]
            currently_inside = d <= threshold
            if (not inside) and currently_inside:
                if (idx - last_crossing_idx) > min_samples_between_crossings:
                    crossings.append((idx, df.loc[idx, "timestamp"]))
                    last_crossing_idx = idx
            inside = currently_inside

        # --- Build loops safely only if crossings detected ---
        if len(crossings) > 1:
            for i in range(1, len(crossings)):
                prev_idx, prev_time = crossings[i-1]
                idx, time = crossings[i]
                start_delta = pd.to_datetime(prev_time) - start_time
                end_delta = pd.to_datetime(time) - start_time
                duration = end_delta - start_delta
                loops.append({
                    "name": f"Lap {i}",
                    "start_time": seconds_to_hhmm(start_delta.total_seconds()),
                    "end_time": seconds_to_hhmm(end_delta.total_seconds()),
                    "duration": seconds_to_hhmm(duration.total_seconds()),
                    "start_idx": prev_idx,
                    "end_idx": idx,
                    "ngp": ""
                })

        # --- Store detected loops in session_state ---
        st.session_state["lap_data"] = loops
        st.session_state["lap_form_submitted"] = True

        st.subheader("Detected Loops")
        if loops:
            loops_df = pd.DataFrame(loops)
            st.dataframe(loops_df[["name", "start_time", "end_time", "duration"]])
        else:
            st.warning("No loops detected")
    else:
        st.info("👆 Please upload a .fit file to automatically detect laps")

# ---------------------------
# MANUAL / FIX DISTANCE LAP INPUT
# ---------------------------
if (st.session_state.get("do_lap_analysis") and st.session_state.get("analysis_type") == "climb") \
   or st.session_state.get("lap_data_insert") in ["manual", "fix_distance"]:

    num_laps = st.number_input(
        f"How many {st.session_state.get('analysis_type','')} would you like to analyze?",
        min_value=1,
        max_value=20,
        step=1,
        key="num_laps"
    )

    # Dynamic form
    with st.form("laps_form"):
        st.subheader(f"📋 {st.session_state.get('analysis_type','').capitalize()} Details")
        lap_inputs = []

        for i in range(num_laps):
            st.markdown(f"### {st.session_state.get('analysis_type','').capitalize()} {i+1}")
            lap_name = st.text_input(
                f"{st.session_state.get('analysis_type','').capitalize()} {i+1} name",
                value=f"{st.session_state.get('analysis_type','').capitalize()} {i+1}",
                key=f"lap_name_{i}"
            )
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.text_input(
                    f"{st.session_state.get('analysis_type','').capitalize()} {i+1} Start Time [HH:MM]",
                    value="",
                    key=f"start_time_{i}",
                    placeholder="es. 00:30"
                )
            with col2:
                end_time = st.text_input(
                    f"{st.session_state.get('analysis_type','').capitalize()} {i+1} End Time [HH:MM]",
                    value="",
                    key=f"end_time_{i}",
                    placeholder="es. 02:00"
                )
            ngp = st.text_input(
                f"{st.session_state.get('analysis_type','').capitalize()} {i+1} NGP [mm:ss] (optional)",
                value="",
                key=f"ngp_{i}",
                placeholder="es. 04:30"
            )

            lap_inputs.append({
                "name": lap_name,
                "start_time": start_time,
                "end_time": end_time,
                "ngp": ngp
            })

        submitted = st.form_submit_button(f"Submit {st.session_state.get('analysis_type','').capitalize()} Data")
        if submitted:
            st.session_state["lap_form_submitted"] = True
            st.session_state["lap_data"] = lap_inputs
            st.success(f"✅ {st.session_state.get('analysis_type','').capitalize()} data submitted successfully!")

#-------------
# ------- ANALYSIS START
#-------------
if uploaded_file is None:
    st.info("👆 Upload a FIT file to run the analysis")
else:
    # --- Athlete & race info display ---
    if 'athlete_name' not in st.session_state:
        st.warning("⚠️ Please submit the Athlete and Race info in the form above")
    else:
        st.markdown("---")
        st.markdown(f"**Athlete:** {st.session_state['athlete_name']}")
        st.markdown(f"**Race:** {st.session_state['race_name']}")
        formatted_date = st.session_state['race_date'].strftime("%d/%m/%Y")
        st.markdown(f"**Date:** {formatted_date}")
        st.markdown(f"📏 **Distance Run:** {kilometers} km")
        st.markdown(f"🏔️ **Elevation gain:** {total_elevation_gain} m")

        total_seconds = df["elapsed_sec"].iloc[-1]
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        final_time_str = f"{hours}:{minutes:02d}:{seconds:02d}"
        st.markdown(f"**Final Time:** {final_time_str}")
        st.markdown("---")

    # --- Smooth HR ---
    df["hr_smooth"] = df["heart_rate"].rolling(window=3, min_periods=1).mean()

    # --- HR averages ---
    mid_index = len(df) // 2
    first_half_avg = df["heart_rate"][:mid_index].mean()
    second_half_avg = df["heart_rate"][mid_index:].mean()
    overall_avg = df["heart_rate"].mean()
    percent_diff = ((second_half_avg - first_half_avg) / first_half_avg) * 100

    st.markdown(f"❤️ Overall Average HR: **{overall_avg:.0f} bpm**")
    st.markdown(f"🟢 First Half Average: **{first_half_avg:.0f} bpm**")
    st.markdown(f"🔵 Second Half Average: **{second_half_avg:.0f} bpm**")
    if percent_diff >= -10:
        st.success(f"📊 % Difference: **{percent_diff:.1f}%**")
    else:
        st.error(f"📊 % Difference: **{percent_diff:.1f}%**")

    # Time in zone analysis for segments

    # Check conditions

    def format_hmm(hmm):
        """Convert H:MM to display format: 0 for 0:00, or 1h00' for 1:00 etc."""
        try:
            h, m = hmm.split(":")
            h, m = int(h), int(m)
            if h == 0 and m == 0:
                return "0"
            else:
                return f"{h}h{m:02d}'"
        except:
            return hmm

    if all(k in st.session_state for k in ['z1','z2','z3','z4','z5']):
        z1, z2, z3, z4, z5 = st.session_state['z1'], st.session_state['z2'], st.session_state['z3'], st.session_state['z4'], st.session_state['z5']

        def get_hr_zone(hr):
            if hr <= z1:
                return "Zone 1 // Recovery"
            elif hr <= z2:
                return "Zone 2 // Aerobic"
            elif hr <= z3:
                return "Zone 3 // Tempo"
            elif hr <= z4:
                return "Zone 4 // Sub Threshold"
            else:
                return "Zone 5 // Super Threshold"

        df["HR Zone"] = df["heart_rate"].apply(get_hr_zone)
        df["time_diff_sec"] = df["elapsed_sec"].diff().fillna(0)

        zone_order = ["Zone 1 // Recovery","Zone 2 // Aerobic","Zone 3 // Tempo","Zone 4 // Sub Threshold","Zone 5 // Super Threshold"]

        # Total (overall) time-in-zone
        total_summary = df.groupby("HR Zone")["time_diff_sec"].sum().reindex(zone_order).fillna(0)

        # Time segment warning

        segment_keys = [
            'segment1_start','segment1_end',
            'segment2_start','segment2_end',
            'segment3_start','segment3_end'
        ]

        if not all(k in st.session_state for k in segment_keys):
            missing_seg = [k for k in segment_keys if k not in st.session_state]
            st.warning(f"⚠️ Please submit the Time Segments in the form above to enable Time-in-Zone analysis.")

        # Prepare segment inputs with formatted names
        segment_inputs = []    
        for i in range(1, 4):
            start_key = f'segment{i}_start'
            end_key = f'segment{i}_end'
            if all(k in st.session_state for k in [start_key, end_key]):
                seg_name = f"{format_hmm(st.session_state[start_key])} to {format_hmm(st.session_state[end_key])}"
                segment_inputs.append(
                    (st.session_state[start_key], st.session_state[end_key], seg_name)
                )
        
        segment_data = {}

        def h_mm_to_seconds(hmm):
            try:
                h, m = hmm.split(":")
                return int(h)*3600 + int(m)*60
            except:
                return None

        for start_str, end_str, seg_name in segment_inputs:
            start_sec = h_mm_to_seconds(start_str)
            end_sec = h_mm_to_seconds(end_str)
            if start_sec is None or end_sec is None or start_sec >= end_sec:
                segment_data[seg_name] = pd.Series(0, index=zone_order)
                continue

            df_segment = df[(df["elapsed_sec"] >= start_sec) & (df["elapsed_sec"] <= end_sec)].copy()
            df_segment["time_diff_sec"] = df_segment["elapsed_sec"].diff().fillna(0)
            seg_summary = df_segment.groupby("HR Zone")["time_diff_sec"].sum().reindex(zone_order).fillna(0)
            segment_data[seg_name] = seg_summary

        # Combine all segments + total into one DataFrame
        combined_df = pd.DataFrame(segment_data)
        combined_df["Total"] = total_summary

        # Format as H:MM
        combined_df = combined_df.applymap(lambda x: f"{int(x//3600)}:{int((x%3600)//60):02d}")

        st.markdown("### ⏱️ Time-in-Zone Analysis")
        st.dataframe(combined_df)

    else:
        st.warning("⚠️ Please submit the Heart Rate Zones to enable Time-in-Zone analysis.")    

    # -----------------------------
    # LAP / CLIMB ANALYSIS
    # -----------------------------
# -------------------------------
# LAP / CLIMB ANALYSIS
# -------------------------------
if 'df' in locals() and not df.empty:
    if st.session_state.get("do_lap_analysis") and 'lap_data' in st.session_state:
        lap_data = st.session_state.get("lap_data", [])

        if 'HR Zone' in df.columns:
            # Map full HR zone names to short names
            hr_zone_map = {
                "Zone 1 // Recovery": "Z1",
                "Zone 2 // Aerobic": "Z2",
                "Zone 3 // Tempo": "Z3",
                "Zone 4 // Sub Threshold": "Z4",
                "Zone 5 // Super Threshold": "Z5"
            }

            zone_order = ["Z1", "Z2", "Z3", "Z4", "Z5"]
            lap_zone_data = []

            def h_mm_to_seconds(hmm):
                """Convert H:MM to seconds."""
                try:
                    h, m = map(int, hmm.split(":"))
                    return h*3600 + m*60
                except:
                    return 0

            for lap in lap_data:
                start_sec = h_mm_to_seconds(lap.get("start_time", "0:00"))
                end_sec = h_mm_to_seconds(lap.get("end_time", "0:01"))
                duration_sec = max(end_sec - start_sec, 1)
                duration_hm = f"{int(duration_sec//3600)}:{int((duration_sec%3600)//60):02d}"

                df_lap = df[(df["elapsed_sec"] >= start_sec) & (df["elapsed_sec"] <= end_sec)].copy()
                df_lap["time_diff_sec"] = df_lap["elapsed_sec"].diff().fillna(0)
                df_lap["HR Zone Short"] = df_lap["HR Zone"].map(hr_zone_map)

                # HR zone summary
                lap_summary = df_lap.groupby("HR Zone Short")["time_diff_sec"].sum().reindex(zone_order).fillna(0)
                lap_summary_hm = [f"{int(x//3600)}:{int((x%3600)//60):02d}" for x in lap_summary.values]
                pct_zones = [f"{round((x/duration_sec)*100)}%" for x in lap_summary.values]

                avg_fc = int(df_lap["heart_rate"].mean()) if not df_lap.empty else 0
                distance = lap.get("distance", 0)
                elevation = lap.get("elevation", 0)
                ngp = lap.get("ngp", "")

                if st.session_state.get("analysis_type") == "Climb Analysis":
                    avg_grade = round((elevation / distance / 10) if distance > 0 else 0)
                    vam = round(elevation / (duration_sec / 3600) if duration_sec > 0 else 0)
                    lap_zone_data.append([
                        lap["name"], duration_hm, int(distance), int(elevation),
                        avg_fc, avg_grade, vam, ngp
                    ] + lap_summary_hm + pct_zones)
                else:  # Lap Analysis
                    if distance > 0:
                        pace_min_float = duration_sec / 60 / distance
                        pace_min = int(pace_min_float)
                        pace_sec = int(round((pace_min_float - pace_min) * 60))
                        lap_pace = f"{pace_min:02d}:{pace_sec:02d}"
                    else:
                        lap_pace = "00:00"
                    lap_zone_data.append([
                        lap["name"], duration_hm, int(distance), int(elevation),
                        avg_fc, lap_pace, ngp
                    ] + lap_summary_hm + pct_zones)

            # Build DataFrame
            if st.session_state.get("analysis_type") == "Climb Analysis":
                extra_cols = ["Avg FC", "Avg Grade (%)", "VAM (m/h)", "NGP"]
            else:
                extra_cols = ["Avg FC", "Lap Pace (min/km)", "NGP"]

            pct_cols = [f"% {z}" for z in zone_order]
            columns = [f"{st.session_state.get('analysis_type','').replace(' Analysis','')} Name",
                       "Duration", "Distance (km)", "Elevation (m)"] + extra_cols + zone_order + pct_cols

            lap_zone_df = pd.DataFrame(lap_zone_data, columns=columns)
            st.dataframe(lap_zone_df.style.hide(axis="index"))

        else:
            st.warning("⚠️ Please submit the Heart Rate Zones first to enable Lap/Climb analysis.")
    else:
        st.info("👆 No lap/climb data to analyze yet. Fill the form above.")
else:
    st.warning("⚠️ Please upload a FIT file first to perform analysis.")

    # --- DET Index ---
    X = df["elapsed_sec"].values.reshape(-1,1)
    y = df["hr_smooth"].values
    reg = LinearRegression().fit(X,y)
    slope_m = abs(reg.coef_[0])
    det_index = slope_m * 10000
    det_index_str = f"{det_index:.1f}"
    if det_index <4: comment,color="Scarso decadimento","green"
    elif det_index<=10: comment,color="Decadimento medio","cyan"
    else: comment,color="Alto decadimento","lightcoral"

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Il DET index indica il decadimento della FC nel corso del tempo**")
    tooltip_text = ("DI < 4 - SCARSO DECADIMENTO\nDI = 7 - DECADIMENTO MEDIO\nDI > 10 - ALTO DECADIMENTO")
    st.markdown(f"<div title='{tooltip_text}' style='font-size:16px; background-color:{color}; color:black; padding:5px; border-radius:5px; display:inline-block;'>📈 DET INDEX: <b>{det_index_str}</b> ({comment})</div>", unsafe_allow_html=True)

    if 'combined_df' in locals():

        # GROUPED BAR CHART FOR TIME IN ZONE
        
        bar_df = combined_df.copy()
        bar_df.index = [f"Zone {i+1}" for i in range(len(bar_df))]
        bar_df_reset = bar_df.reset_index().rename(columns={'index':'HR Zone'})
        bar_long = bar_df_reset.melt(id_vars="HR Zone", var_name="Segment", value_name="Time [h:mm]")

        # Convert H:MM to hours float
        def h_mm_to_float(hmm_str):
            try:
                h, m = map(int, hmm_str.split(":"))
                return h + m/60
            except:
                return 0

        bar_long["Hours"] = bar_long["Time [h:mm]"].apply(h_mm_to_float)

        fig_bar = px.bar(
            bar_long,
            x="HR Zone",
            y="Hours",
            color="Segment",
            barmode="group",
            hover_data={"Segment": True, "Time [h:mm]": True, "Hours": False, "HR Zone": True},
            title="⏱️ Time-in-Zone per Segment (Bar Chart)"
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)

        # HEATMAP GRAPH IN MINUTES
        # Convert H:MM to minutes
        def h_mm_to_minutes(hmm_str):
            try:
                h, m = map(int, hmm_str.split(":"))
                return h*60 + m
            except:
                return 0

        heatmap_df_minutes = bar_df.applymap(h_mm_to_minutes)
        heatmap_df_minutes.index = [f"Zone {i+1}" for i in range(len(heatmap_df_minutes))]
        heatmap_df_minutes_reset = heatmap_df_minutes.reset_index().rename(columns={'index':'HR Zone'})
        heatmap_long = heatmap_df_minutes_reset.melt(id_vars="HR Zone", var_name="Segment", value_name="Minutes")

        fig_heat = px.density_heatmap(
            heatmap_long,
            x="Segment",
            y="HR Zone",
            z="Minutes",
            text_auto=True,
            color_continuous_scale="YlOrRd",
            hover_data={"Segment": True, "HR Zone": True, "Minutes": True},
            title="🌡️ Time-in-Zone (minutes) Heatmap"
        )
        fig_heat.update_layout(
            yaxis=dict(autorange='reversed'),
            coloraxis_colorbar=dict(title="Time (minutes)")
        )

        st.plotly_chart(fig_heat, use_container_width=True)
    # --- Plotly chart for % time in HR zones per Lap/Climb ---
        #LAP CARDIAC PERCENTAGE TIME IN ZONES BAR CHART
        if st.session_state["do_lap_analysis"]:
            # Select only % columns and lap names
            pct_cols = [f"% {z}" for z in ["Z1","Z2","Z3","Z4","Z5"]]
            df_plot = lap_zone_df[["{} name".format(analysis_type[:-8])] + pct_cols].copy()

            # Convert percentage strings to numeric
            for col in pct_cols:
                df_plot[col] = df_plot[col].str.rstrip('%').astype(float)

            # Melt the DataFrame for plotly
            df_melted = df_plot.melt(id_vars=["{} name".format(analysis_type[:-8])],
                                    value_vars=pct_cols,
                                    var_name="HR Zone",
                                    value_name="Percentage")

            # Plot
            fig = px.bar(df_melted, 
                        x="HR Zone", 
                        y="Percentage", 
                        color="{} name".format(analysis_type[:-8]),
                        barmode="group",
                        title=f"{analysis_type} - % Time in HR Zones")

            st.plotly_chart(fig)

    # --- Plotly chart for HR vs TIME ---
    df["trend_line"] = reg.predict(X)
    df["Race Time [h:mm] "] = df.apply(lambda row: f"{int(row['elapsed_sec']//3600)}:{int((row['elapsed_sec']%3600)//60):02d} | HR: {int(row['hr_smooth'])} bpm", axis=1)

    fig = px.line(df, x="elapsed_hours", y="hr_smooth",
                labels={"elapsed_hours":"Elapsed Time (hours)", "hr_smooth":"Heart Rate (bpm)"},
                title="Heart Rate Over Time",
                hover_data={"Race Time [h:mm] ":True,"elapsed_hours":False,"hr_smooth":False})
    fig.add_scatter(x=df["elapsed_hours"], y=df["trend_line"], mode='lines', line=dict(color='red', dash='dash'), name='Trend Line')
    fig.update_traces(hovertemplate='%{customdata[0]}', selector=dict(name='hr_smooth'))
    fig.update_yaxes(tickformat='d')
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------
# Modern PDF Class
# --------------------------------------------------------------------
class ModernPDF(FPDF):
    def header(self):
        self.set_fill_color(30, 30, 30)
        self.rect(0, 0, 210, 30, 'F')
        self.set_xy(10, 6)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "DU COACHING - Race Analyzer Report", ln=True)
        self.set_xy(10, 18)  # move cursor below main title
        self.set_font("Helvetica", "I", 11)  # italic or lighter font
        self.set_text_color(200, 200, 200)   # lighter gray
        self.cell(0, 6, "This analyzer is brought to you by Coach Ambro", ln=True)

        self.ln(5)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 30)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, ln=True, fill=True)
        self.ln(4)

    def body_text(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(55, 55, 55)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def add_spacer(self, h=4):
        self.ln(h)

# --------------------------------------------------------------------
# PDF Generation
# --------------------------------------------------------------------
if 'athlete_name' not in st.session_state:
    st.warning("⚠️ Please submit the Athlete and Race info in the form above")
segment_keys = [
    'segment1_start','segment1_end',
    'segment2_start','segment2_end',
    'segment3_start','segment3_end'
    ]

if not all(k in st.session_state for k in segment_keys):
    missing_seg = [k for k in segment_keys if k not in st.session_state]
    st.warning(f"⚠️ Please submit the Time Segments in the form above to enable Time-in-Zone analysis.")

if st.button("📄 Generate PDF Report"):

    pdf = ModernPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ------------------ Athlete Info ------------------
    pdf.section_title("Athlete & Race Information")
    pdf.body_text(f"Athlete: {athlete_name}")
    pdf.body_text(f"Race: {race_name}")
    pdf.body_text(f"Date: {formatted_date}")
    pdf.body_text(f"Distance: {kilometers} km")
    pdf.body_text(f"Elevation: {total_elevation_gain} m")
    pdf.body_text(f"Final Time: {final_time_str}")
    pdf.add_spacer()
    pdf.body_text(f"Overall Average HR: {overall_avg:.0f} bpm")
    pdf.body_text(f"First Half Avg HR: {first_half_avg:.0f} bpm")
    pdf.body_text(f"Second Half Avg HR: {second_half_avg:.0f} bpm")
    pdf.body_text(f"% Difference: {percent_diff:.1f}%")
    pdf.body_text(f"DET Index: {det_index_str} ({comment})")
    pdf.add_spacer(6)

    # ------------------ Time-in-Zone Table ------------------
    if "combined_df" in locals():
        pdf.section_title("Time-in-Zone Table [hh:mm]")

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(245, 245, 245)

        page_w = pdf.w - pdf.l_margin - pdf.r_margin
        n_cols = 1 + len(combined_df.columns)
        col_width = page_w / n_cols
        row_height = 8

        # Header
        pdf.set_text_color(20, 20, 20)
        pdf.cell(col_width, row_height, "HR Zone", border=1, fill=True)
        for col in combined_df.columns:
            pdf.cell(col_width, row_height, str(col), border=1, fill=True)
        pdf.ln()

        # Rows
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        for i, zone in enumerate(combined_df.index, 1):
            pdf.cell(col_width, row_height, f"Zone {i}", border=1)
            for seg in combined_df.columns:
                pdf.cell(col_width, row_height, str(combined_df.loc[zone, seg]), border=1)
            pdf.ln()

        pdf.add_spacer(6)

        # ------------------ Table 1: Lap/Climb Info ------------------
    if st.session_state["do_lap_analysis"]:
        pdf.section_title(f"{analysis_type} - Info Table")

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(245, 245, 245)

        # Select only the columns without zones
        info_cols = [col for col in lap_zone_df.columns if not col.startswith("Z") and not col.startswith("%")]
        n_cols = len(info_cols)
        page_w = pdf.w - pdf.l_margin - pdf.r_margin
        col_width = page_w / n_cols
        row_height = 6

        # Header
        for col in info_cols:
            pdf.cell(col_width, row_height, str(col)[:10], border=1, fill=True, align='C')
        pdf.ln(row_height)

        # Rows
        pdf.set_font("Helvetica", "", 10)
        for _, row in lap_zone_df.iterrows():
            for col in info_cols:
                pdf.cell(col_width, row_height, str(row[col]), border=1, align='C')
            pdf.ln(row_height)

        pdf.add_spacer(6)

        # ------------------ Table 2: LAP/CLIMB CARDIAC ANALYSIS ------------------
    if st.session_state["do_lap_analysis"]:
        pdf.section_title(f"{analysis_type} - HR DATA Analysis")

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(245, 245, 245)

        zone_pct_cols = [f"{analysis_type[:-8]} name"] + [f"Z{i}" for i in range(1,6)] + [f"% Z{i}" for i in range(1,6)]
        n_cols = len(zone_pct_cols)
        col_width = page_w / n_cols
        row_height = 6

        # Header
        for col in zone_pct_cols:
            pdf.cell(col_width, row_height, col, border=1, fill=True, align='C')
        pdf.ln(row_height)

        # Rows
        pdf.set_font("Helvetica", "", 10)
        for _, row in lap_zone_df.iterrows():
            for col in zone_pct_cols:
                pdf.cell(col_width, row_height, str(row[col]), border=1, align='C')
            pdf.ln(row_height)

        pdf.add_spacer(6)
        pdf.add_page()

    # PDF CHARTS
    # ------------------ PDF Chart Helper ------------------
    def add_chart_to_pdf(fig, title=None):
        if title:
            pdf.section_title(title)
        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", dpi=200, bbox_inches="tight")
        buf.seek(0)
        pdf.image(buf, x=10, w=190)
        plt.close(fig)

    # ------------------ Bar Chart ------------------
    if "bar_df" in locals():
        fig = plt.figure(figsize=(10, 4))
        zones = np.arange(len(bar_df.index))
        width = 0.2
        for i, seg in enumerate(bar_df.columns):
            vals = bar_df[seg].apply(lambda t: int(t.split(':')[0]) + int(t.split(':')[1])/60)
            plt.bar(zones + i * width, vals, width=width, label=seg)
        plt.xticks(zones + width * (len(bar_df.columns)-1)/2, bar_df.index)
        plt.ylabel("Hours")
        plt.title("Time-in-Zone per Segment")
        plt.tight_layout()
        add_chart_to_pdf(fig, title="Time-in-Zone - Bar Chart")

    # ------------------ Heatmap ------------------
    if "heatmap_df_minutes" in locals():
        fig, ax = plt.subplots(figsize=(10, 3 + max(0, len(heatmap_df_minutes)/4)))
        sns.heatmap(
            heatmap_df_minutes,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'label':'Minutes'},
            ax=ax
        )
        ax.set_title("Time-in-Zone Heatmap (Minutes)")
        ax.set_ylabel("HR Zone")
        ax.set_xlabel("Segment")
        plt.tight_layout()
        add_chart_to_pdf(fig, title="Time-in-Zone - Heatmap")
    pdf.add_page()

    # ------------------ Lap/Climb % Time-in-Zone Chart ------------------
    if st.session_state["do_lap_analysis"]:

        fig, ax = plt.subplots(figsize=(10, 4))
        zones = ["Z1","Z2","Z3","Z4","Z5"]
        x = np.arange(len(zones))
        width = 0.8 / len(lap_zone_df)  # bar width depending on number of laps

        for i, (_, row) in enumerate(lap_zone_df.iterrows()):
            pct_values = [float(str(row[f"% {z}"]).rstrip('%')) for z in zones]
            ax.bar(x + i*width, pct_values, width=width, label=row[f"{analysis_type[:-8]} name"])

        ax.set_xticks(x + width*(len(lap_zone_df)-1)/2)
        ax.set_xticklabels(zones)
        ax.set_ylabel("Percentage (%)")
        ax.set_title(f"{analysis_type} - % Time in HR Zones")
        ax.legend()
        plt.tight_layout()

        add_chart_to_pdf(fig, title=f"{analysis_type} - % Time in HR Zones")

    # ------------------ HR Trend ------------------
    fig = plt.figure(figsize=(10, 4))
    plt.plot(df["elapsed_hours"], df["hr_smooth"], label="HR Smooth")
    try:
        plt.plot(df["elapsed_hours"], reg.predict(X), label="Trend Line", linestyle="--")
    except Exception:
        pass
    plt.xlabel("Elapsed Time (hours)")
    plt.ylabel("Heart Rate (bpm)")
    plt.title("Heart Rate Over Time")
    plt.legend()
    plt.tight_layout()
    add_chart_to_pdf(fig, title="Heart Rate - Trend Analysis")

    # ------------------ Output PDF ------------------
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)

    st.download_button(
        label="⬇️ Download PDF",
        data=pdf_buffer,
        mime="application/pdf",
        file_name=f"{athlete_name}_{race_name}_report.pdf"
    )