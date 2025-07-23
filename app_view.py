import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sqlalchemy import create_engine
import pydeck as pdk

#DATABASE_URL = st.secrets["DATABASE_URL"]

engine = create_engine(DATABASE_URL)

def compute_multi_segment_duration_hm(group, threshold=1.0, gap_multiplier=2):
    min_temp = group["temp"].min()
    near_min = group[np.abs(group["temp"] - min_temp) <= threshold].sort_values("time")
    if near_min.empty:
        return "1:00"
    near_min = near_min.reset_index(drop=True)

    time_diffs = near_min["time"].diff().dropna()
    if time_diffs.empty:
        return "1:00"

    typical_gap = time_diffs.median()
    max_allowed_gap = typical_gap * gap_multiplier  # 允许的最大间隔

    total_seconds = 0
    segment_start = near_min.loc[0, "time"]

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

def temp_to_color(temp, min_t, max_t):
    norm = (temp - min_t) / (max_t - min_t + 1e-6)
    r = int(255 * norm)
    g = int(255 * (1 - norm))
    b = int(255 * (1 - norm))
    return [r, g, b]

############################################
st.set_page_config(layout="wide")
#st.title("📊 Temperature Analysis Viewer")
st.markdown("### 🌡️ Temperature Report")
st.markdown("This dashboard shows temperature trends, extremes, and risk indicators during shipment.")


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

    # Generate the table
    if df_filtered.empty:
        st.info("⚠️ No data matches current filter.")
    else:
        summary_stats = []
        for container_id, group in df_filtered.groupby("container"):
            avg_temp = group["temp"].mean()
            min_temp = group["temp"].min()
            max_temp = group["temp"].max()
            hours_below_0 = np.sum(group["temp"] < 0) * 2

            duration_hours = compute_multi_segment_duration_hm(group)
            out_of_range = (group["temp"] < 0) | (group["temp"] > 25)
            percent_out = 100 * out_of_range.sum() / len(group)

            summary_stats.append({
                "container": container_id,
                "avg_temp": avg_temp,
                #"min_temp": min_temp,
                "min_temp": f"{min_temp:.1f}°C" + (" ❄️" if min_temp < 0 else ""),
                "max_temp": max_temp,
                "hours_below_0": f"{hours_below_0}h" + ("⚠️" if hours_below_0 >= 8 else ""),
                "out_of_range(%)": f"{percent_out:.1f}%",
                "duration_of_mintemp(h:m)": duration_hours
            })
        summary_df = pd.DataFrame(summary_stats)

        high_temp = (
            df_filtered
            .loc[df_filtered.groupby("month")["temp"].idxmax()]
            [["month", "temp", "location"]]
            .rename(columns={"temp": "High_Temp", "location": "High_Location"})
        )

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
        tab1, tab2, tab3 = st.tabs(["Summary Table", "Temperature Plot", "Map"])
        with tab1:
            st.subheader("Statistics by Month")
            st.dataframe(summary_tbl)

            st.subheader("Statistics by Container")
            st.dataframe(summary_df)

        with tab2:
            df_filtered["year_month"] = df_filtered["time"].dt.to_period("M").astype(str)
            high_temp = (
                df_filtered.loc[df_filtered.groupby("year_month")["temp"].idxmax()]
                [["year_month", "temp", "location"]]
                .rename(columns={"temp": "High_Temp", "location": "High_Location"})
            )
            low_temp = (
                df_filtered.loc[df_filtered.groupby("year_month")["temp"].idxmin()]
                [["year_month", "temp", "location"]]
                .rename(columns={"temp": "Low_Temp", "location": "Low_Location"})
            )
            summary_tbl = pd.merge(high_temp, low_temp, on="year_month", how="outer")
            summary_tbl = summary_tbl.dropna(subset=["year_month"])
            summary_tbl = summary_tbl.sort_values("year_month")

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = {
             "High_Temp": "#F28C8C",   # 柔和红 Coral Pink
             "Low_Temp": "#84C2F2"     # 柔和蓝 Sky Blue
            }
            plot_data = pd.melt( summary_tbl, id_vars=["year_month"], value_vars=["High_Temp", "Low_Temp"], 
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
        
        with tab3:
            st.subheader("Monthly Highs and Lows per Container")

            # 获取每个 container 每月的最高温和最低温对应的行
            high_points = df_filtered.loc[df_filtered.groupby(["month", "container"])["temp"].idxmax()]
            low_points = df_filtered.loc[df_filtered.groupby(["month", "container"])["temp"].idxmin()]

            high_points = high_points.copy()
            high_points["type"] = "High"

            low_points = low_points.copy()
            low_points["type"] = "Low"

            # 合并
            map_points = pd.concat([high_points, low_points])
            map_points = map_points.dropna(subset=["lat", "lon"])

            # 添加 popup label
            map_points["text"] = (
                map_points["type"] + " | " +
                map_points["container"] + " | " +
                map_points["month"] + " | " +
                map_points["temp"].round(1).astype(str) + "°C"
            )

            min_temp = map_points["temp"].min()
            max_temp = map_points["temp"].max()
            map_points["color"] = map_points["temp"].apply(lambda t: temp_to_color(t, min_temp, max_temp))
            
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_points,
                get_position='[lon, lat]',
                get_color='color',
                get_radius=30000,
                pickable=True,
                tooltip=True
            )

            view_state = pdk.ViewState(
                latitude=map_points["lat"].mean(),
                longitude=map_points["lon"].mean(),
                zoom=3,
                pitch=0
            )

            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "{text}"}
            )
            st.pydeck_chart(r)
    
except FileNotFoundError:
    st.warning("⚠️ No data available. Please contact administrator.")
