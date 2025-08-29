import sqlalchemy
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sqlalchemy import create_engine
import pydeck as pdk

DATABASE_URL = st.secrets["DATABASE_URL"]
engine = create_engine(DATABASE_URL)

st.set_page_config(
    page_title="Rate Card Viewer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

#########################################################################################
TEXT = {
    "English": {
        "title": "🌡️ Temperature Report",
        "desc": "This dashboard shows temperature trends, extremes, and risk indicators during shipment.",
        "search_header": "🔍 Search Options",
        "select_container": "Select Container",
        "select_year": "Select Year(s)",
        "select_month": "Select Month(s)",
        "no_data": "⚠️ No data matches current filter.",
        "tab1": "Summary Table",
        "tab2": "Temperature Plot",
        "tab3": "Map",
        "stats_month": "Statistics by Month",
        "stats_container": "Statistics by Container",
        "plot_title": "Monthly High and Low Temperatures",
        "map_title": "Monthly Highs and Lows per Container",
        "warning": "⚠️ File not found.",
        "db_error": "❌ Cannot connect to the database. Please check your connection.",
        "unknown_error": "❌ An unknown error occurred."
    },
    "中文": {
        "title": "🌡️ 温度报告",
        "desc": "本仪表板展示了运输过程中的温度趋势、极端情况与风险指标。",
        "search_header": "🔍 筛选选项",
        "select_container": "选择集装箱",
        "select_year": "选择年份",
        "select_month": "选择月份",
        "no_data": "⚠️ 没有符合条件的数据。",
        "tab1": "表格分析",
        "tab2": "图表分析",
        "tab3": "地图",
        "stats_month": "按月份统计",
        "stats_container": "按集装箱统计",
        "plot_title": "每月高低温折线图",
        "map_title": "每月各集装箱的高低温分布",
        "warning": "⚠️ 文件未找到。",
        "db_error": "❌ 无法连接数据库，请检查网络或数据库状态。",
        "unknown_error": "❌ 发生未知错误。"
    }
}

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

########################################################################################
##--------------------------Layout of MainPage------------------------------------------
# st.set_page_config(layout="wide")
# Language Selector
col1, col2 = st.columns([5, 1])
with col2:
    lang = st.selectbox("🌐", options=["English", "中文"], label_visibility="collapsed")
# Title
st.markdown(f"### {TEXT[lang]['title']}")
st.markdown(TEXT[lang]['desc'])
tab1, tab2, tab3 = st.tabs([TEXT[lang]["tab1"], TEXT[lang]["tab2"], TEXT[lang]["tab3"]])

main_placeholder = st.empty()
with st.spinner("Loading data, please wait..."):
    try:
        df = pd.read_sql("SELECT * FROM temperature_data", con=engine)
        df["year"] = df["time"].dt.year.astype(str)
        container_options = sorted(df["container"].dropna().unique().tolist())
        year_options = sorted(df["year"].unique())
        month_options = sorted(df["month"].unique(), key=lambda m: pd.to_datetime(m, format='%b').month)
        with main_placeholder.container():
            with st.sidebar:
                st.markdown(f"### {TEXT[lang]['search_header']}")
                #st.sidebar.expander(TEXT[lang]['search_header'], expanded = True)
                # Selector
                selected_container = st.sidebar.multiselect(TEXT[lang]['select_container'], options=["All"] + container_options)
                selected_years = st.sidebar.multiselect(TEXT[lang]['select_year'], options=year_options)
                selected_months = st.sidebar.multiselect(TEXT[lang]['select_month'], options=month_options)
            # Data Filtering
            df_filtered = df.copy()
            if selected_container and "All" not in selected_container:
                df_filtered = df_filtered[df_filtered["container"].isin(selected_container)]
            if selected_years:
                df_filtered = df_filtered[df_filtered["year"].isin(selected_years)]
            if selected_months:
                df_filtered = df_filtered[df_filtered["month"].isin(selected_months)]
            
            if selected_years or selected_months:
                if lang == "English":
                    st.markdown(f"📅 Showing results for: {', '.join(selected_years or ['All Years'])}, {', '.join(selected_months or ['All Months'])}")
                else:
                    st.markdown(f"📅 当前筛选结果：年份 - {', '.join(selected_years or ['全部年份'])}，月份 - {', '.join(selected_months or ['全部月份'])}")

            # Generate the table
            if df_filtered.empty:
                st.info(TEXT[lang]['no_data'])
            else:
                # Statistics by container
                summary_stats = []
                grouped = (df_filtered.copy().sort_values("time").groupby("container"))
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
                # 按 group_id 分组
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
                    "out_of_range(0-30°C)": f"{percent_out:.1f}%",
                    "duration_of_mintemp(h:m)": duration_hours
                    })

                summary_df = pd.DataFrame(summary_stats)
                summary_df = summary_df.sort_values(by=["departure_date", "container"])

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

                #valid_data = all_data.dropna(subset=["temp"])
                max_idx = df_filtered.groupby("year_month")["temp"].idxmax().dropna()
                min_idx = df_filtered.groupby("year_month")["temp"].idxmin().dropna()

                high_temp = (df_filtered.loc[df_filtered.groupby("year_month")["temp"].idxmax()]
                [["year_month", "temp", "location"]].rename(columns={"temp": "High_Temp", "location": "High_Location"}))
                low_temp = (df_filtered.loc[df_filtered.groupby("year_month")["temp"].idxmin()]
                [["year_month", "temp", "location"]].rename(columns={"temp": "Low_Temp", "location": "Low_Location"}))

                summary_tbl = pd.merge(high_temp, low_temp, on="year_month", how="outer")
                summary_tbl = summary_tbl.dropna(subset=["year_month"])
                summary_tbl = summary_tbl.sort_values("year_month")

                summary_tbl["High_Location"] = summary_tbl["High_Location"].replace("中国", "China")
                summary_tbl["Low_Location"] = summary_tbl["Low_Location"].replace("中国", "China")

                # Layout tabs
                with tab1:
                    st.subheader(TEXT[lang]["stats_month"])
                    st.dataframe(summary_tbl)

                    st.subheader(TEXT[lang]["stats_container"])
                    st.dataframe(summary_df)

                with tab2:
                    st.subheader(TEXT[lang]['plot_title'])
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = {
                    "High_Temp": "#F28C8C",
                    "Low_Temp": "#84C2F2" 
                    }
                    plot_data = pd.melt( summary_tbl, id_vars=["year_month"], value_vars=["High_Temp", "Low_Temp"], 
                                        var_name="Type", value_name="Temp")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for key, grp in plot_data.groupby("Type"):
                        ax.plot(grp["year_month"], grp["Temp"], marker='o', label="High" if "High" in key else "Low", color=colors[key])
                        for x, y in zip(grp["year_month"], grp["Temp"]):
                            ax.text(x, y + 0.3, f"{y:.1f}°C", ha='center', fontsize=8, color=colors[key])
                    ax.set_xlabel("Year-Month")
                    ax.set_ylabel("Temperature (°C)")
                    ax.legend()
                    ax.grid(False)
                    plt.xticks(rotation=45)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    st.pyplot(fig)
                
                with tab3:
                    st.subheader(TEXT[lang]['map_title'])
                    high_points = df_filtered.loc[df_filtered.groupby(["month", "container"])["temp"].idxmax()]
                    low_points = df_filtered.loc[df_filtered.groupby(["month", "container"])["temp"].idxmin()]
                    high_points = high_points.copy()
                    high_points["type"] = "High"
                    low_points = low_points.copy()
                    low_points["type"] = "Low"

                    map_points = pd.concat([high_points, low_points])
                    map_points = map_points.dropna(subset=["lat", "lon"])
                    map_points["text"] = (
                        map_points["type"] + " | " +
                        map_points["container"] + " | " +
                        map_points["month"] + " | " +
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

    except FileNotFoundError:
        st.warning(TEXT[lang]['warning'])
    except sqlalchemy.exc.OperationalError:
        st.error(TEXT[lang]["db_error"])
    except Exception:
        st.error(TEXT[lang]["unknown_error"])
        st.info("The database is empty, pleae upload data first.")