# Streamlit app: Zomato Delivery Performance - Interactive Overview
# Save as: streamlit_app_zomato_delivery_overview.py
# Usage: streamlit run streamlit_app_zomato_delivery_overview.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Zomato Delivery Overview", layout="wide")

st.title("Zomato — Delivery Performance: Interactive Overview")
st.markdown(
    "Upload a dataset (CSV / Excel) exported from your analysis notebook. The app will try to infer common columns (timestamps, delivery duration, distance, coordinates) and show interactive visualizations and summaries."
)

# --- Sidebar: file upload & basic options ---
st.sidebar.header("Data input & options")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"]) 
use_example = st.sidebar.checkbox("Use example synthetic dataset (if you don't have a file)")

# Helper: load dataset
@st.cache_data
def load_df(file) -> pd.DataFrame:
    if file is None:
        return None
    try:
        if str(file.name).lower().endswith('.csv'):
            return pd.read_csv(file, low_memory=False)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

# A small synthetic example if user wants to try quickly
@st.cache_data
def example_df(n=1000):
    rng = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='H')
    df = pd.DataFrame({
        'order_id': np.arange(n),
        'order_ts': rng - pd.to_timedelta(np.random.randint(5,60,size=n), unit='m'),
        'dispatch_ts': rng - pd.to_timedelta(np.random.randint(0,10,size=n), unit='m'),
        'delivered_ts': rng + pd.to_timedelta(np.random.randint(5,60,size=n), unit='m'),
        'distance_km': np.round(np.random.exponential(scale=3, size=n), 2),
        'order_value': np.round(np.random.exponential(scale=100, size=n), 2),
        'city': np.random.choice(['Jakarta','Bandung','Surabaya','Medan'], size=n),
        'rating': np.random.choice([3,4,5], size=n, p=[0.15,0.5,0.35])
    })
    df['delivery_mins'] = (df['delivered_ts'] - df['dispatch_ts']).dt.total_seconds() / 60.0
    return df

if use_example and uploaded_file is None:
    df = example_df(2000)
else:
    df = load_df(uploaded_file)

if df is None:
    st.info("Silakan upload file CSV/XLSX dataset Anda atau centang 'Use example synthetic dataset'.")
    st.stop()

st.success(f"Dataset loaded — {df.shape[0]} baris x {df.shape[1]} kolom")

# --- Smart column inference ---

def infer_datetime_columns(df):
    dt_cols = []
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            dt_cols.append(c)
        else:
            # try parse if a string looks like a timestamp (only a sample)
            try:
                sample = df[c].dropna().astype(str).iloc[:10]
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().sum() >= 3:
                    dt_cols.append(c)
            except Exception:
                pass
    return dt_cols

# convert candidate datetime cols
cand_dt = infer_datetime_columns(df)
for c in cand_dt:
    try:
        df[c] = pd.to_datetime(df[c], errors='coerce')
    except Exception:
        pass

# Common patterns to detect
possible_delivery_duration_cols = [c for c in df.columns if 'delivery' in c.lower() and any(k in c.lower() for k in ['time','mins','duration'])]
possible_order_ts = [c for c in df.columns if any(k in c.lower() for k in ['order','ordered','order_ts','order_time','order_timestamp'])]
possible_dispatch_ts = [c for c in df.columns if any(k in c.lower() for k in ['dispatch','pickup','picked','driver'])]
possible_delivered_ts = [c for c in df.columns if any(k in c.lower() for k in ['deliver','delivered','dropoff','arrive'])]

# find numeric distance
possible_distance = [c for c in df.columns if any(k in c.lower() for k in ['distance','km','kms','mile'])]

st.sidebar.markdown("**Detected columns (inference):**")
st.sidebar.write(f"Datetime-like: {cand_dt}")
st.sidebar.write(f"Delivery duration candidates: {possible_delivery_duration_cols}")
st.sidebar.write(f"Order timestamps candidates: {possible_order_ts}")
st.sidebar.write(f"Dispatched timestamps candidates: {possible_dispatch_ts}")
st.sidebar.write(f"Delivered timestamps candidates: {possible_delivered_ts}")
st.sidebar.write(f"Distance-like: {possible_distance}")

# Let user choose which columns to use
st.sidebar.header("Pilih kolom untuk analisis")
col_order_ts = st.sidebar.selectbox("Order timestamp column", options=[None]+list(df.columns), index=0)
col_dispatch_ts = st.sidebar.selectbox("Dispatch/pickup timestamp column", options=[None]+list(df.columns), index=0)
col_delivered_ts = st.sidebar.selectbox("Delivered timestamp column", options=[None]+list(df.columns), index=0)
col_delivery_mins = st.sidebar.selectbox("Delivery duration (minutes) column", options=[None]+list(df.columns), index=0)
col_distance = st.sidebar.selectbox("Distance column", options=[None]+list(df.columns), index=0)
col_city = st.sidebar.selectbox("City column", options=[None]+list(df.columns), index=0)
col_rating = st.sidebar.selectbox("Rating column", options=[None]+list(df.columns), index=0)

# Compute derived delivery_mins if not present
if col_delivery_mins and col_delivery_mins in df.columns:
    df['delivery_mins'] = pd.to_numeric(df[col_delivery_mins], errors='coerce')
else:
    if col_dispatch_ts and col_delivered_ts and col_dispatch_ts in df.columns and col_delivered_ts in df.columns:
        df['delivery_mins'] = (pd.to_datetime(df[col_delivered_ts], errors='coerce') - pd.to_datetime(df[col_dispatch_ts], errors='coerce')).dt.total_seconds()/60.0
    elif col_order_ts and col_delivered_ts and col_order_ts in df.columns and col_delivered_ts in df.columns:
        df['delivery_mins'] = (pd.to_datetime(df[col_delivered_ts], errors='coerce') - pd.to_datetime(df[col_order_ts], errors='coerce')).dt.total_seconds()/60.0
    else:
        if 'delivery_mins' not in df.columns:
            df['delivery_mins'] = pd.to_numeric(df.get('delivery_time_mins', pd.Series(np.nan)), errors='coerce')

# ensure distance numeric
if col_distance and col_distance in df.columns:
    df['distance_km'] = pd.to_numeric(df[col_distance], errors='coerce')
else:
    if 'distance_km' not in df.columns:
        # try common names
        for c in ['distance','dist_km','kms','km']:
            if c in df.columns:
                try:
                    df['distance_km'] = pd.to_numeric(df[c], errors='coerce')
                    break
                except Exception:
                    pass

# parse order date for filters
date_col_for_filter = None
for c in [col_order_ts, col_dispatch_ts, col_delivered_ts]:
    if c and c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c]):
        date_col_for_filter = c
        break

if date_col_for_filter is None:
    # try to find any datetime-like
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            date_col_for_filter = c
            break

# --- Filters ---
st.sidebar.header("Filters")
if date_col_for_filter:
    min_date = df[date_col_for_filter].min().date()
    max_date = df[date_col_for_filter].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (pd.to_datetime(df[date_col_for_filter]).dt.date >= start_date) & (pd.to_datetime(df[date_col_for_filter]).dt.date <= end_date)
        df = df.loc[mask]

if col_city and col_city in df.columns:
    cities = st.sidebar.multiselect("City", options=df[col_city].dropna().unique().tolist(), default=df[col_city].dropna().unique().tolist())
    if cities:
        df = df[df[col_city].isin(cities)]

# --- Overview cards ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows", df.shape[0])
with col2:
    st.metric("Columns", df.shape[1])
with col3:
    mean_delivery = df['delivery_mins'].dropna().mean()
    st.metric("Rata-rata waktu antar (menit)", f"{mean_delivery:.1f}" if not np.isnan(mean_delivery) else "-")
with col4:
    median_delivery = df['delivery_mins'].dropna().median()
    st.metric("Median waktu antar (menit)", f"{median_delivery:.1f}" if not np.isnan(median_delivery) else "-")

# Missing values summary
st.subheader("Missing values / completeness")
missing = (df.isna().mean() * 100).sort_values(ascending=False)
st.dataframe(missing[missing>0].to_frame('pct_missing'))

# Numeric summary
st.subheader("Numeric summary")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) > 0:
    st.dataframe(df[num_cols].describe().T)

# --- Main interactive charts ---
st.subheader("Interaktif: Distribusi & Trends")

left, right = st.columns([2,3])
with left:
    # Histogram delivery minutes
    if 'delivery_mins' in df.columns and df['delivery_mins'].notna().sum()>0:
        bins = st.slider("Bins untuk histogram delivery (mins)", 10, 200, 40)
        fig = px.histogram(df, x='delivery_mins', nbins=bins, marginal='box', title='Distribusi waktu antar (menit)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tidak ada kolom 'delivery_mins' yang valid untuk histogram.")

    # Box by city
    if col_city and col_city in df.columns and 'delivery_mins' in df.columns:
        fig2 = px.box(df, x=col_city, y='delivery_mins', points='outliers', title='Waktu antar per kota')
        st.plotly_chart(fig2, use_container_width=True)

with right:
    # Time series orders per day
    if date_col_for_filter:
        ts = pd.to_datetime(df[date_col_for_filter]).dt.floor('D').value_counts().sort_index()
        ts = ts.rename_axis('date').reset_index(name='orders')
        fig3 = px.line(ts, x='date', y='orders', title='Order per hari')
        st.plotly_chart(fig3, use_container_width=True)
    # Scatter delivery vs distance
    if 'delivery_mins' in df.columns and 'distance_km' in df.columns and df['distance_km'].notna().sum()>0:
        fig4 = px.scatter(df.sample(min(len(df),2000)), x='distance_km', y='delivery_mins', trendline='ols', title='Delivery mins vs Distance (sample)')
        st.plotly_chart(fig4, use_container_width=True)

# Top / bottom performers
st.subheader("Top / Bottom — berdasarkan metrik yang dipilih")
metric = st.selectbox("Pilih metrik", options=['delivery_mins','order_value','distance_km'] + num_cols, index=0)
entity = st.selectbox("Group by (entity)", options=[None]+[col for col in df.columns if df[col].nunique() < 200 and df[col].dtype == 'object'])

if entity is not None:
    agg = df.groupby(entity)[metric].agg(['mean','median','count']).sort_values('mean')
    st.write(agg.head(10))
    st.write(agg.tail(10))

# Map if coords exist
st.subheader("Map (jika tersedia latitude/longitude)")
lat_cols = [c for c in df.columns if 'lat' in c.lower()]
lon_cols = [c for c in df.columns if 'lon' in c.lower() or 'lng' in c.lower()]
if lat_cols and lon_cols:
    lat = lat_cols[0]
    lon = lon_cols[0]
    coords = df[[lat, lon]].dropna()
    if not coords.empty:
        st.map(coords)
    else:
        st.info("Ditemukan kolom koordinat tetapi tidak ada nilai non-null.")
else:
    st.info("Tidak ditemukan kolom koordinat. Jika data Anda berisi latitude/longitude, beri nama kolom 'lat'/'lon' atau 'latitude'/'longitude'.")

# Download cleaned/filtered dataset
st.subheader("Export filtered dataset")
if st.button("Download CSV dari dataset terfilter"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label='Download CSV', data=csv, file_name='zomato_delivery_filtered.csv', mime='text/csv')

st.markdown("---")
st.caption("Aplikasi ini sifatnya generik: sesuaikan pemetaan kolom (sidebar) agar sesuai struktur file Anda. Jika butuh, saya bisa bantu sesuaikan app ini berdasarkan contoh file Anda.")

# --- Requirements.txt helper ---
st.markdown("## Requirements.txt")
st.code("""\nstreamlit\npandas\nnumpy\nplotly\nopenpyxl # jika ingin support excel\n""", language="text")
