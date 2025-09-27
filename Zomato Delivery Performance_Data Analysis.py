import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ================================
# Zomato Delivery Performance - Power BI Aligned
# ================================

st.set_page_config(page_title="Zomato Delivery Performance", layout="wide")
st.title("📊 Zomato Delivery Performance Overview")

# ---------- Helpers ----------
def haversine(lat1, lon1, lat2, lon2):
    """Hitung jarak (km) antara dua koordinat (vectorized)."""
    R = 6371.0  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def trimmed_mean(s: pd.Series, pct: float = 0.078) -> float:
    """Robust mean: buang pct terbawah & teratas (default 7.8%)."""
    arr = s.dropna().sort_values().values
    cut = int(pct * len(arr))
    if cut == 0 or 2*cut >= len(arr):
        return float(np.nanmean(arr))
    return float(np.mean(arr[cut: len(arr) - cut]))

def drop_distance_outliers_per_city(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buang outlier Distance_km per City memakai IQR.
    - HANYA drop upper outlier: value > Q3 + 1.5*IQR
    - Baris dengan City = NaN TIDAK diproses (dibiarkan)
    """
    df = df.copy()
    keep = pd.Series(True, index=df.index)
    if "City" not in df.columns or "Distance_km" not in df.columns:
        return df

    # groupby default mengabaikan NaN
    for city, sub in df[df["City"].notna()].groupby("City"):
        q1 = sub["Distance_km"].quantile(0.25)
        q3 = sub["Distance_km"].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        drop_idx = sub.index[sub["Distance_km"] > upper]  # pakai '>' (bukan '>=') untuk konsistensi notebook
        keep.loc[drop_idx] = False

    return df[keep]

def iqr_clip(series: pd.Series) -> pd.Series:
    """Winsorize (IQR clipping) untuk visualisasi agar grafik stabil."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower = max(0.0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr
    return series.clip(lower, upper)

# ---------- File uploader ----------
uploaded_file = st.sidebar.file_uploader("Upload Zomato Dataset (CSV)", type=["csv"])
if not uploaded_file:
    st.info("Silakan upload file CSV Zomato Dataset untuk melihat analisis.")
    st.stop()

# ---------- Load ----------
df_raw = pd.read_csv(uploaded_file)
df_raw.columns = df_raw.columns.str.strip()
df_raw.drop_duplicates(inplace=True)

# ---------- Derive fields ----------
# Tanggal
if "Order_Date" in df_raw.columns:
    df_raw["Order_Date"] = pd.to_datetime(df_raw["Order_Date"], errors="coerce", dayfirst=True)

# Jarak (km) dari koordinat
coord_cols = {"Restaurant_latitude", "Restaurant_longitude",
              "Delivery_location_latitude", "Delivery_location_longitude"}
if coord_cols.issubset(df_raw.columns):
    df_raw["Distance_km"] = haversine(
        df_raw["Restaurant_latitude"], df_raw["Restaurant_longitude"],
        df_raw["Delivery_location_latitude"], df_raw["Delivery_location_longitude"]
    )

# ---------- Outlier drop PER CITY (post-clean dataset) ----------
df_post = drop_distance_outliers_per_city(df_raw)  # ~45.17K baris tersisa

# ---------- Missing value handling (kategori) ----------
# NOTE: Diisi SETELAH drop outlier agar City=NaN tidak ikut pembuangan (sesuai notebook/BI)
for col in ["City", "Weather_conditions", "Road_traffic_density", "Festival",
            "Type_of_order", "Type_of_vehicle"]:
    if col in df_post.columns:
        df_post[col] = df_post[col].astype("object").str.strip().fillna("Unknown")
    if col in df_raw.columns:
        # agar filter juga bisa diterapkan ke metrik yang dihitung dari df_raw (trimmed mean)
        df_raw[col] = df_raw[col].astype("object").str.strip().fillna("Unknown")

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters")

# City
if "City" in df_post.columns:
    city_vals = sorted(df_post["City"].unique().tolist())
    sel_city = st.sidebar.multiselect("Pilih Kota", options=city_vals, default=city_vals)
else:
    sel_city = None

# Weather
if "Weather_conditions" in df_post.columns:
    weather_vals = sorted(df_post["Weather_conditions"].unique().tolist())
    sel_weather = st.sidebar.multiselect("Pilih Cuaca", options=weather_vals, default=weather_vals)
else:
    sel_weather = None

# Rentang tanggal
if "Order_Date" in df_post.columns and not df_post["Order_Date"].isna().all():
    min_date, max_date = df_post["Order_Date"].min(), df_post["Order_Date"].max()
    date_range = st.sidebar.date_input("Rentang Tanggal", [min_date, max_date])
else:
    date_range = []

# Terapkan filter ke KEDUA dataset (post-drop & raw)
def apply_filters(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    if sel_city is not None and "City" in df_out.columns:
        df_out = df_out[df_out["City"].isin(sel_city)]
    if sel_weather is not None and "Weather_conditions" in df_out.columns:
        df_out = df_out[df_out["Weather_conditions"].isin(sel_weather)]
    if len(date_range) == 2 and "Order_Date" in df_out.columns:
        df_out = df_out[
            (df_out["Order_Date"] >= pd.to_datetime(date_range[0])) &
            (df_out["Order_Date"] <= pd.to_datetime(date_range[1]))
        ]
    return df_out

df_f_post = apply_filters(df_post)  # basis metrik Count/Time/Unique
df_f_raw  = apply_filters(df_raw)   # basis metrik Distance Avg (trimmed mean)

# ---------- Data utk visualisasi (winsorize agar grafik stabil) ----------
df_vis = df_f_post.copy()
if "Distance_km" in df_vis.columns:
    df_vis["Distance_km"] = iqr_clip(df_vis["Distance_km"])

if "Time_taken (min)" in df_vis.columns:
    # clip ringan (opsional) agar histogram tidak terlalu ekor panjang
    q1, q3 = df_vis["Time_taken (min)"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df_vis["Time_taken (min)"] = df_vis["Time_taken (min)"].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)

# ================================
# Overview Metrics (selaras Power BI)
# ================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Orders", len(df_f_post))

with col2:
    st.metric("Unique Delivery Persons",
              df_f_post["Delivery_person_ID"].nunique() if "Delivery_person_ID" in df_f_post.columns else "-")

with col3:
    if "Time_taken (min)" in df_f_post.columns:
        st.metric("Rata-rata Waktu Antar (menit)", round(df_f_post["Time_taken (min)"].mean(), 2))
    else:
        st.metric("Rata-rata Waktu Antar (menit)", "-")

with col4:
    if "Distance_km" in df_f_raw.columns:
        # 👉 Distance Avg ala Power BI: TRIMMED MEAN 7.8% dihitung pada data pra-drop (robust)
        dist_avg = trimmed_mean(df_f_raw["Distance_km"], pct=0.078)
        st.metric("Rata-rata Jarak (km)", round(dist_avg, 2))
    else:
        st.metric("Rata-rata Jarak (km)", "-")

# ================================
# Visualisasi (pakai df_vis yang sudah di-winsorize)
# ================================
if "Time_taken (min)" in df_vis.columns:
    fig1 = px.histogram(df_vis, x="Time_taken (min)", nbins=40, title="Distribusi Waktu Antar (menit)")
    st.plotly_chart(fig1, use_container_width=True)

if {"City", "Time_taken (min)"} <= set(df_vis.columns):
    fig2 = px.box(df_vis, x="City", y="Time_taken (min)", color="City",
                  title="Perbandingan Waktu Antar per Kota")
    st.plotly_chart(fig2, use_container_width=True)

if "Order_Date" in df_vis.columns:
    daily_orders = df_f_post.groupby("Order_Date").size().reset_index(name="Orders")
    fig3 = px.line(daily_orders, x="Order_Date", y="Orders", title="Trend Jumlah Order Harian")
    st.plotly_chart(fig3, use_container_width=True)

if {"Road_traffic_density", "Time_taken (min)"} <= set(df_vis.columns):
    fig4 = px.box(df_vis, x="Road_traffic_density", y="Time_taken (min)",
                  color="Road_traffic_density", title="Waktu Antar vs Kepadatan Lalu Lintas")
    st.plotly_chart(fig4, use_container_width=True)

# ================================
# Data Table
# ================================
st.subheader("📋 Data Preview (post-drop)")
st.dataframe(df_f_post.head(50))
