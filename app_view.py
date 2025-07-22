import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sqlalchemy import create_engine


DATABASE_URL = st.secrets["DATABASE_URL"]
engine = create_engine(DATABASE_URL)

st.set_page_config(layout="wide")
st.title("📊 Temperature Analysis Viewer")

try:
    df = pd.read_sql("SELECT * FROM temperature_data", con=engine)

    # 确保时间列正确解析并有 month 列
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df["month"] = df["time"].dt.strftime("%b")
    df = df.dropna(subset=["location"])

    st.sidebar.header("🔍 Search Options")

    # 选择 Container
    container_options = sorted(df["container"].dropna().unique().tolist())
    selected_container = st.sidebar.selectbox("Select Container", options=["All"] + container_options)

    # 选择 Month(s)
    month_options = sorted(df["month"].unique(), key=lambda m: pd.to_datetime(m, format='%b').month)
    selected_months = st.sidebar.multiselect("Select Month(s)", options=month_options)

    # 过滤数据
    df_filtered = df.copy()
    if selected_container != "All":
        df_filtered = df_filtered[df_filtered["container"] == selected_container]
    if selected_months:
        df_filtered = df_filtered[df_filtered["month"].isin(selected_months)]

    if df_filtered.empty:
        st.info("⚠️ No data matches current filter.")
    else:
        st.subheader(f"📈 Temperature Statistics for container: {selected_container}")
        summary_stats = []
        for container_id, group in df_filtered.groupby("container"):
            avg_temp = group["temp"].mean()
            min_temp = group["temp"].min()
            max_temp = group["temp"].max()
            hours_below_0 = np.sum(group["temp"] < 0) * 2

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
        summary_df = pd.DataFrame(summary_stats)

        high_temp = (
            df_filtered
            .loc[df_filtered.groupby("month")["temp"].idxmax()]
            [["month", "temp", "location"]]
            .rename(columns={"temp": "High_Temp", "location": "High_Location"})
        )

        # Statistics by month: Lows
        low_temp = (
            df_filtered
            .loc[df_filtered.groupby("month")["temp"].idxmin()]
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
            st.dataframe(summary_df)

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
    
except FileNotFoundError:
    st.warning("⚠️ No data available. Please contact administrator.")
