import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ================================
# Zomato Delivery Performance (Power BI-aligned preprocessing)
# ================================

st.set_page_config(page_title="Zomato Delivery Performance", layout="wide")
st.title("📊 Zomato Delivery Performance Overview")

# ---------- Helpers ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def trimmed_mean(s: pd.Series, pct: float = 0.078) -> float:
    """Robust mean ala Power BI notebook: buang pct terbawah & teratas."""
    arr = s.dropna().sort_values().values
    cut = int(pct * len(arr))
    if cut == 0 or 2*cut >= len(arr):
        return float(np.nanmean(arr))
    return float(np.mean(arr[cut: len(arr) - cut]))

def iqr_clip(s: pd.Series) -> pd.Series:
    """Winsorize (IQR clipping) untuk visualisasi agar grafik stabil."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower = max(0.0, q1 - 1.5*iqr)
    upper = q3 + 1.5*iqr
    return s.clip(lower, upper)

# ---------- File uploader ----------
uploaded_file = st.sidebar.file_uploader("Upload Zomato Dataset (CSV)", type=["csv"])
if not uploaded_file:
    st.info("Silakan upload file CSV Zomato Dataset untuk melihat analisis.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ---------- Preprocessing (selaras notebook/Power BI) ----------
df.columns = df.columns.str.strip()
df.drop_duplicates(inplace=True)

# Tanggal
if "Order_Date" in df.columns:
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce", dayfirst=True)

# Categorical: isi NaN -> "Unknown" (bukan dibuang)
for col in ["City", "Weather_conditions", "Road_traffic_density", "Festival",
            "Type_of_order", "Type_of_vehicle"]:
    if col in df.columns:
        df[col] = df[col].astype("object").str.strip()
        df[col] = df[col].fillna("Unknown")

# Jarak (km) dari koordinat
coord_cols = {"Restaurant_latitude", "Restaurant_longitude",
              "Delivery_location_latitude", "Delivery_location_longitude"}
if coord_cols.issubset(df.columns):
    df["Distance_km"] = haversine(
        df["Restaurant_latitude"], df["Restaurant_longitude"],
        df["Delivery_location_latitude"], df["Delivery_location_longitude"]
    )

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters")

if "City" in df.columns:
    city_vals = sorted(df["City"].unique().tolist())
    sel_city = st.sidebar.multiselect("Pilih Kota", options=city_vals, default=city_vals)
else:
    sel_city = None

if "Weather_conditions" in df.columns:
    weather_vals = sorted(df["Weather_conditions"].unique().tolist())
    sel_weather = st.sidebar.multiselect("Pilih Cuaca", options=weather_vals, default=weather_vals)
else:
    sel_weather = None

# Rentang tanggal
if "Order_Date" in df.columns and not df["Order_Date"].isna().all():
    min_date, max_date = df["Order_Date"].min(), df["Order_Date"].max()
    date_range = st.sidebar.date_input("Rentang Tanggal", [min_date, max_date])
else:
    date_range = []

# Terapkan filter
df_f = df.copy()
if sel_city is not None:
    df_f = df_f[df_f["City"].isin(sel_city)]
if sel_weather is not None:
    df_f = df_f[df_f["Weather_conditions"].isin(sel_weather)]
if len(date_range) == 2:
    df_f = df_f[(df_f["Order_Date"] >= pd.to_datetime(date_range[0])) &
                (df_f["Order_Date"] <= pd.to_datetime(date_range[1]))]

# ---------- Data utk visualisasi (winsorize agar grafik stabil) ----------
df_vis = df_f.copy()
if "Distance_km" in df_vis.columns:
    df_vis["Distance_km"] = iqr_clip(df_vis["Distance_km"])
if "Time_taken (min)" in df_vis.columns:
    # opsional: clip ringan agar histogram tidak terlalu ekor panjang
    q1, q3 = df_vis["Time_taken (min)"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df_vis["Time_taken (min)"] = df_vis["Time_taken (min)"].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)

# ================================
# Overview Metrics (selaras Power BI)
# ================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Orders", len(df_f))

with col2:
    st.metric("Unique Delivery Persons",
              df_f["Delivery_person_ID"].nunique() if "Delivery_person_ID" in df_f.columns else "-")

with col3:
    if "Time_taken (min)" in df_f.columns:
        st.metric("Rata-rata Waktu Antar (menit)", round(df_f["Time_taken (min)"].mean(), 2))
    else:
        st.metric("Rata-rata Waktu Antar (menit)", "-")

with col4:
    if "Distance_km" in df_f.columns:
        # 👉 Rata-rata Jarak ala Power BI: trimmed mean 7.8%
        dist_avg = trimmed_mean(df_f["Distance_km"], pct=0.078)
        st.metric("Rata-rata Jarak (km)", round(dist_avg, 2))
    else:
        st.metric("Rata-rata Jarak (km)", "-")

# ================================
# Visualisasi (pakai df_vis yg sdh di-winsorize)
# ================================
if "Time_taken (min)" in df_vis.columns:
    fig1 = px.histogram(df_vis, x="Time_taken (min)", nbins=40, title="Distribusi Waktu Antar (menit)")
    st.plotly_chart(fig1, use_container_width=True)

if "City" in df_vis.columns and "Time_taken (min)" in df_vis.columns:
    fig2 = px.box(df_vis, x="City", y="Time_taken (min)", color="City",
                  title="Perbandingan Waktu Antar per Kota")
    st.plotly_chart(fig2, use_container_width=True)

if "Order_Date" in df_vis.columns:
    daily_orders = df_f.groupby("Order_Date").size().reset_index(name="Orders")
    fig3 = px.line(daily_orders, x="Order_Date", y="Orders", title="Trend Jumlah Order Harian")
    st.plotly_chart(fig3, use_container_width=True)

if "Road_traffic_density" in df_vis.columns and "Time_taken (min)" in df_vis.columns:
    fig4 = px.box(df_vis, x="Road_traffic_density", y="Time_taken (min)",
                  color="Road_traffic_density", title="Waktu Antar vs Kepadatan Lalu Lintas")
    st.plotly_chart(fig4, use_container_width=True)

st.subheader("📋 Data Preview")
st.dataframe(df_f.head(50))
