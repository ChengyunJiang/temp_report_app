import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pydeck as pdk
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.postgresql import insert


DATABASE_URL = st.secrets["DATABASE_URL"]
engine = create_engine(DATABASE_URL)


col_candidates = {
    "time": ["定位时间", "采集时间"],
    "temp": ["温度2", "温度 (°C)", "温度 (°C)"],
    "container": ["箱号", "设备号"],
    "location": ["国家", "国家/地区", "国家/地区"],
    "lon": ["经度"],
    "lat": ["纬度"]
}

def dynamic_rename(df):
    rename_map = {}
    for std_col, candidates in col_candidates.items():
        for c in candidates:
            if c in df.columns:
                rename_map[c] = std_col
                break
    return df.rename(columns=rename_map)

def compute_multi_segment_duration_hm(group, threshold=1.0, gap_multiplier=2):
    min_temp = group["temp"].min()
    near_min = group[np.abs(group["temp"] - min_temp) <= threshold].sort_values("time")

    if near_min.empty:
        return "1:00"

    near_min = near_min.reset_index(drop=True)

    time_diffs = near_min["time"].diff().dropna()
    if time_diffs.empty:
        return "0:00"

    typical_gap = time_diffs.median()
    max_allowed_gap = typical_gap * gap_multiplier 

    total_seconds = 0
    segment_start = near_min.loc[0, "time"]

    for i in range(1, len(near_min)):
        current_time = near_min.loc[i, "time"]
        previous_time = near_min.loc[i - 1, "time"]

        if (current_time - previous_time) > max_allowed_gap:
            total_seconds += (previous_time - segment_start).total_seconds()
            segment_start = current_time

    total_seconds += (near_min.iloc[-1]["time"] - segment_start).total_seconds()

    total_minutes = int(total_seconds // 60)
    hours = total_minutes // 60
    minutes = total_minutes % 60

    return f"{hours}:{minutes:02d}"

def temp_to_color(temp, min_t, max_t):
    norm = (temp - min_t) / (max_t - min_t + 1e-6)
    r = int(255 * norm)
    g = int(255 * (1 - norm))
    b = int(255 * (1 - norm))
    return [r, g, b]

def insert_unique_data(df, engine):
    with engine.begin() as conn:
        rows = df.to_dict(orient="records")
        stmt = insert(all_data_table).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=['container', 'time', 'temp', 'location'])
        conn.execute(stmt)


st.markdown("""
    <style>
    .refresh-button {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 100;
    }
    </style>
    <div class="refresh-button">
        <form action="" method="get">
            <button type="submit"> Refresh</button>
        </form>
    </div>
""", unsafe_allow_html=True)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .css-1dp5vir {display: none;} 
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

##--------------------------Layout of MainPage------------------------------------------
st.title("🔒 Internal Upload Portal")

uploaded_files = st.file_uploader("📂 Upload files (.xlsx or .csv)", accept_multiple_files=True, type=["xlsx","csv"])
with st.expander("Delete Container Data"):
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT container FROM temperature_data"))
            containers = [row[0] for row in result.fetchall()]
    except Exception as e:
        st.error(f"❌ Failed to fetch containers. Error: {e}")
        containers = []

    if containers:
        selected_containers = st.multiselect("Select a container to delete", options = ["All"] + containers)
        if selected_containers:
            confirm = st.checkbox(f"✅ I confirm to delete all data for container `{selected_containers}`")
            if st.button("🗑 Delete the Data"):
                if confirm:
                    try:
                        status_placeholder = st.empty() 
                        if "All" in selected_containers:
                            with engine.begin() as conn:
                                status_placeholder.info(f"Deleting all...")
                                conn.execute(text("DELETE FROM temperature_data"))
                            status_placeholder.success("✅ All data deleted successfully.")
                        else :
                            with engine.begin() as conn:
                                conn.execute(text("DELETE FROM temperature_data WHERE container = ANY(:c)"), 
                                             {"c": selected_containers})
                                status_placeholder.info(f"Deleting...")
                        status_placeholder.success(f"✅ Data has been deleted.")
                    except Exception as e:
                        st.error(f"❌ Failed to delete data. Error: {e}")
                else:
                    st.warning("Please check the confirmation box before deleting.")
        else:
            st.info("Please select at least one container.")
    else:
        st.info("No containers found in the database.")

##--------------------------Data Processing------------------------------------------
if uploaded_files:
    status_placeholder = st.empty() 
    status_placeholder = st.info("Data Processing...") 
    dfs = []
    for uploaded_file in uploaded_files:
        # 获取文件扩展名（小写）
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(uploaded_file)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"❌ Unsupported file type: {uploaded_file.name}")
            continue
        df = dynamic_rename(df)
        df = df[df["location"].notna() & (df["location"] != "-")]
        df = df.dropna(subset=["temp"])

        # Parse datetime
        df["time"] = pd.to_datetime(df["time"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
        df = df.dropna(subset=["time"])  # Remove bad rows 
        df["month"] = df["time"].dt.strftime("%b")
        df["year"] = df["time"].dt.year.astype(str)
        dfs.append(df)
    
    all_data = pd.concat(dfs, ignore_index=True)
    all_data["container"] = all_data["container"].astype(str)
    all_data["location"] = all_data["location"].astype(str)

    
    if all_data.empty:
        st.error("⚠️ No valid data after processing! Please check file formats and required columns.")
        st.stop()

    # Statistics by container
    summary_stats = []
    grouped = (all_data.copy().sort_values("time").groupby("container"))
    records = []

    for container_id, group in grouped:
        group = group.copy()
        group["departure_date"] = group["time"].min().normalize()
        group["arrival_date"] = group["time"].max().normalize()
        records.append(group)

    all_data_with_dates = pd.concat(records)
    all_data_with_dates["group_id"] = (
        all_data_with_dates["container"] + " | " +
        all_data_with_dates["departure_date"].dt.strftime("%Y-%m-%d") + " → " +
        all_data_with_dates["arrival_date"].dt.strftime("%Y-%m-%d")
    )
    
    for group_id, group in all_data_with_dates.groupby("group_id"):
        container_id = group["container"].iloc[0]
        departure_date = group["departure_date"].iloc[0].strftime("%Y-%m-%d")
        arrival_date = group["arrival_date"].iloc[0].strftime("%Y-%m-%d")

        avg_temp = group["temp"].mean()
        min_temp = group["temp"].min()
        max_temp = group["temp"].max()
        hours_below_0 = np.sum(group["temp"] < 0) * 2
        duration_hours = compute_multi_segment_duration_hm(group)
        out_of_range = (group["temp"] < 0) | (group["temp"] > 30)
        percent_out = 100 * out_of_range.sum() / len(group)

        summary_stats.append({
            "container": container_id,
            "departure_date": departure_date,
            "arrival_date": arrival_date,
            "avg_temp": f"{avg_temp:.1f}°C",
            "min_temp": f"{min_temp:.1f}°C" + (" ❄️" if min_temp < 0 else ""),
            "max_temp": f"{max_temp:.1f}°C" + (" 🔥" if max_temp > 30 else ""),
            "hours_below_0": f"{hours_below_0}h" + ("⚠️" if hours_below_0 >= 8 else ""),
            "out_of_range(0-30)": f"{percent_out:.1f}%",
            "duration_of_mintemp(h:m)": duration_hours
        })
    summary_stats_df = pd.DataFrame(summary_stats)
    summary_stats_df = summary_stats_df.sort_values(by=["departure_date", "container"])

    # Statistics by month
    all_data["year_month"] = all_data["time"].dt.to_period("M").astype(str)

    valid_data = all_data.dropna(subset=["temp"])
    max_idx = valid_data.groupby("year_month")["temp"].idxmax().dropna()
    min_idx = valid_data.groupby("year_month")["temp"].idxmin().dropna()

    high_temp = (valid_data.loc[valid_data.groupby("year_month")["temp"].idxmax()]
        [["year_month", "temp", "location"]].rename(columns={"temp": "High_Temp", "location": "High_Location"}))
    low_temp = (valid_data.loc[valid_data.groupby("year_month")["temp"].idxmin()]
        [["year_month", "temp", "location"]].rename(columns={"temp": "Low_Temp", "location": "Low_Location"}))

    summary_tbl = pd.merge(high_temp, low_temp, on="year_month", how="outer")
    summary_tbl = summary_tbl.dropna(subset=["year_month"])
    summary_tbl = summary_tbl.sort_values("year_month")
    
    summary_tbl["High_Location"] = summary_tbl["High_Location"].replace("中国", "China")
    summary_tbl["Low_Location"] = summary_tbl["Low_Location"].replace("中国", "China")

    status_placeholder = st.success("✅ Upload successful. Data preview:")
    tab1, tab2, tab3 = st.tabs(["Summary Table", "Temperature Plot", "Map"])
##--------------------------Tab1: Summary Table------------------------------------------
    with tab1:
        st.subheader("Statistics by Month")
        st.dataframe(summary_tbl)

        st.subheader("Statistics by Container")
        #st.dataframe(styled_df, use_container_width=True)
        st.dataframe(summary_stats_df)

##--------------------------Tab2: Temperature Plot------------------------------------------
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {
            "High_Temp": "#F28C8C",   # Coral Pink
            "Low_Temp": "#84C2F2"     # Sky Blue
        }
        plot_data = pd.melt(summary_tbl, id_vars=["year_month"], value_vars=["High_Temp", "Low_Temp"], 
                            var_name="Type", value_name="Temp")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, grp in plot_data.groupby("Type"):
            ax.plot(grp["year_month"], grp["Temp"], marker='o', label="High" if "High" in key else "Low", color=colors[key])
            for x, y in zip(grp["year_month"], grp["Temp"]):
                ax.text(x, y + 0.3, f"{y:.1f}°C", ha='center', fontsize=8, color=colors[key])

        ax.set_title("Monthly High and Low Temperatures")
        ax.set_xlabel("Year-Month")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        
        ax.grid(False)
        plt.xticks(rotation=45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)

##--------------------------Tab3: Map with Extreme Points------------------------------------------
    with tab3:
        st.subheader("Monthly Highs and Lows per Container")
        valid_data = all_data.dropna(subset=["year_month", "container", "temp"])
        high_idx = valid_data.groupby(["year_month", "container"])["temp"].idxmax().dropna()
        low_idx = valid_data.groupby(["year_month", "container"])["temp"].idxmin().dropna()

        high_points = valid_data.loc[high_idx].copy()
        high_points["type"] = "High"

        low_points = valid_data.loc[low_idx].copy()
        low_points["type"] = "Low"

        high_points = high_points.copy()
        high_points["type"] = "High"

        low_points = low_points.copy()
        low_points["type"] = "Low"

        map_points = pd.concat([high_points, low_points])
        map_points = map_points.dropna(subset=["lat", "lon"])

        map_points["text"] = (
            map_points["type"] + " | " +
            map_points["container"] + " | " +
            map_points["year_month"] + " | " +
            map_points["temp"].round(1).astype(str) + "°C"
        )

        min_temp = map_points["temp"].min()
        max_temp = map_points["temp"].max()
        map_points["color"] = map_points["temp"].apply(lambda t: temp_to_color(t, min_temp, max_temp))
        
        layer = pdk.Layer("ScatterplotLayer", data=map_points, get_position='[lon, lat]', 
                            get_color='color', get_radius=20000, pickable=True, tooltip=True)

        view_state = pdk.ViewState(latitude=map_points["lat"].mean(), 
                                longitude=map_points["lon"].mean(),
                                zoom=3, pitch=0)

        r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{text}"})

        st.pydeck_chart(r)

        with st.expander("📋 View raw data"):
            st.dataframe(map_points[["month", "container", "type", "temp", "lat", "lon"]])

##--------------------------Upload the Data------------------------------------------
    meta = MetaData()
    meta.reflect(bind=engine)

    all_data_table = meta.tables["temperature_data"]
    columns_to_keep = ["container", "temp", "lon", "lat", "location", "time", "month", "year_month"]
    cleaned_data = all_data[columns_to_keep]
    
    status_placeholder = st.empty() 
    status_placeholder.info(f"📊 Inserting {len(cleaned_data)} rows...")

    insert_unique_data(cleaned_data, engine)
    status_placeholder.success("📂 Data saved to remote database")
