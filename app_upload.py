import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import sqlite3

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

        # 正确统计 duration：只在当前 container 内
        close_to_min = group[np.abs(group["temp"] - min_temp) <= 1]
        duration_hours = len(close_to_min) * 2

        summary_stats.append({
            "container": container_id,
            "avg_temp": avg_temp,
            "min_temp": min_temp,
            "max_temp": max_temp,
            "hours_below_0": hours_below_0,
            "duration_of_min": duration_hours
        })

    summary_stats_df = pd.DataFrame(summary_stats)


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

    # Replace "中国" with "China"
    summary_tbl["High_Location"] = summary_tbl["High_Location"].replace("中国", "China")
    summary_tbl["Low_Location"] = summary_tbl["Low_Location"].replace("中国", "China")

    # Layout tabs
    tab1, tab2 = st.tabs(["Summary Table", "Temperature Plot"])

    with tab1:
        st.subheader("Statistics by Month")
        st.dataframe(summary_tbl)

        st.subheader("Statistics by Container")
        st.dataframe(summary_stats_df)

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
    

     # 🔔 Save to SQLite
    os.makedirs("shared-data", exist_ok=True)
    conn = sqlite3.connect("shared-data/latest_data.db")
    all_data.to_sql("temperature_data", conn, if_exists="replace", index=False)
    conn.close()

    st.success("📂 Data saved to shared-data/latest_data.db")
