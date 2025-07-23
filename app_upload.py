import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sqlalchemy import create_engine
import sqlalchemy
import psycopg2

#DATABASE_URL = st.secrets["DATABASE_URL"]
DATABASE_URL = "postgresql://postgres.idtiedxkkknpmeuaklql:xyspog-wixto5-geMnaf@aws-0-eu-central-1.pooler.supabase.com:5432/postgres"
engine = create_engine(DATABASE_URL)


col_candidates = {
    "time": ["定位时间", "采集时间"],
    "temp": ["温度2", "温度 (°C)"],
    "container": ["箱号", "设备号"],
    "location": ["国家", "国家/地区"]
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
        return "near_min empty0:00"

    near_min = near_min.reset_index(drop=True)

    # 自动推断采样间隔（中位数间隔）
    time_diffs = near_min["time"].diff().dropna()
    if time_diffs.empty:
        return "time_diffs 0:00"

    typical_gap = time_diffs.median()
    max_allowed_gap = typical_gap * gap_multiplier  # 允许的最大间隔

    total_seconds = 0
    segment_start = near_min.loc[0, "time"]

    print("正在处理 container:", group["container"].iloc[0])
    print("最小温度:", min_temp)
    print("near_min 条数:", len(near_min))
    print("time span:", near_min["time"].min(), "→", near_min["time"].max())

    for i in range(1, len(near_min)):
        current_time = near_min.loc[i, "time"]
        previous_time = near_min.loc[i - 1, "time"]
        #print(f"⏱ Gap between {previous_time} and {current_time}: {(current_time - previous_time)}")

        if (current_time - previous_time) > max_allowed_gap:
            # 超过合理间隔，断段
            total_seconds += (previous_time - segment_start).total_seconds()
            segment_start = current_time

    # 最后一段
    total_seconds += (near_min.iloc[-1]["time"] - segment_start).total_seconds()

    # 转换为小时:分钟
    total_minutes = int(total_seconds // 60)
    hours = total_minutes // 60
    minutes = total_minutes % 60

    return f"{hours}:{minutes:02d}"



st.title("🔒 Internal Upload Portal")

uploaded_files = st.file_uploader("Upload Excel files", accept_multiple_files=True, type=["xlsx"])

if uploaded_files:
    dfs = []
    for uploaded_file in uploaded_files:
        df = pd.read_excel(uploaded_file)
        df = dynamic_rename(df)
        df = df[df["location"].notna() & (df["location"] != "")]

        # Parse datetime
        df["time"] = pd.to_datetime(df["time"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
        df = df.dropna(subset=["time"])  # Remove bad rows

        df["month"] = df["time"].dt.strftime("%b")  # Month as abbreviated name (Jan, Feb, ...)
        dfs.append(df)
    
    all_data = pd.concat(dfs, ignore_index=True)
    all_data = all_data.dropna(subset=["month"])
    all_data["container"] = all_data["container"].astype(str)
    all_data["location"] = all_data["location"].astype(str)


    if all_data.empty:
        st.error("⚠️ No valid data after processing! Please check file formats and required columns.")
        st.stop()

    st.write("✅ Upload successful. Data preview:")
    # Statistics by container
    summary_stats = []

    for container_id, group in all_data.groupby("container"):
        avg_temp = group["temp"].mean()
        min_temp = group["temp"].min()
        max_temp = group["temp"].max()
        hours_below_0 = np.sum(group["temp"] < 0) * 2

        # Calculate the duration of min_temp
        duration_hours = compute_multi_segment_duration_hm(group)

        summary_stats.append({
            "container": container_id,
            "avg_temp": avg_temp,
            "min_temp": min_temp,
            "max_temp": max_temp,
            "hours_below_0": hours_below_0,
            "duration_of_min": duration_hours
        })

    summary_stats_df = pd.DataFrame(summary_stats)

    def highlight_negative_min_temp(val):
        if isinstance(val, (int, float)) and val < 0:
            return "background-color: lightcoral; color: white"
        return ""

    styled_df = summary_stats_df.style.applymap(highlight_negative_min_temp, subset=["min_temp"])




    # Statistics by month: Highs
    high_temp = (
        all_data
        .loc[all_data.groupby("month")["temp"].idxmax()]
        [["month", "temp", "location"]]
        .rename(columns={"temp": "High_Temp", "location": "High_Location"})
    )

    # Statistics by month: Lows
    low_temp = (
        all_data
        .loc[all_data.groupby("month")["temp"].idxmin()]
        [["month", "temp", "location"]]
        .rename(columns={"temp": "Low_Temp", "location": "Low_Location"})
    )


    # Merge summary
    summary_tbl = pd.merge(high_temp, low_temp, on="month", how="outer")
    summary_tbl = summary_tbl.dropna(subset=["month"])
    summary_tbl = summary_tbl.sort_values(by="month", key=lambda x: pd.to_datetime(x, format="%b"))

    summary_tbl["High_Location"] = summary_tbl["High_Location"].replace("中国", "China")
    summary_tbl["Low_Location"] = summary_tbl["Low_Location"].replace("中国", "China")

    # Layout tabs
    tab1, tab2 = st.tabs(["Summary Table", "Temperature Plot"])

    with tab1:
        st.subheader("Statistics by Month")
        st.dataframe(summary_tbl)

        st.subheader("Statistics by Container")
        #st.dataframe(summary_stats_df)
        st.dataframe(styled_df, use_container_width=True)

    with tab2:
        # Prepare plot data
        plot_data = pd.melt(
            summary_tbl,
            id_vars=["month"],
            value_vars=["High_Temp", "Low_Temp"],
            var_name="Type",
            value_name="Temp"
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        for key, grp in plot_data.groupby("Type"):
            ax.plot(grp["month"], grp["Temp"], marker='o', label=key)

        ax.set_title("Monthly High and Low Temperatures")
        ax.set_xlabel("Month")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
    
    all_data.to_sql("temperature_data", con=engine, if_exists="replace", index=False)
    st.success("📂 Data saved to remote database")
