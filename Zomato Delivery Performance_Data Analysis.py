import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ================================
# Streamlit App - Zomato Delivery Overview (No Map Version)
# ================================

st.set_page_config(page_title="Zomato Delivery Performance", layout="wide")
st.title("📊 Zomato Delivery Performance Overview")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Zomato Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Silakan upload file CSV Zomato Dataset untuk melihat analisis.")
    st.stop()

# Parsing tanggal
if "Order_Date" in df.columns:
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce", dayfirst=True)

# Hitung jarak jika ada koordinat
if {"Restaurant_latitude", "Restaurant_longitude", "Delivery_location_latitude", "Delivery_location_longitude"}.issubset(df.columns):
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    df["Distance_km"] = haversine(df["Restaurant_latitude"], df["Restaurant_longitude"],
                                   df["Delivery_location_latitude"], df["Delivery_location_longitude"])

# Sidebar Filters
st.sidebar.header("Filters")

if "City" in df.columns:
    city_options = st.sidebar.multiselect("Pilih Kota", options=df["City"].unique(), default=df["City"].unique())
    df = df[df["City"].isin(city_options)]

if "Weather_conditions" in df.columns:
    weather_options = st.sidebar.multiselect("Pilih Cuaca", options=df["Weather_conditions"].unique(), default=df["Weather_conditions"].unique())
    df = df[df["Weather_conditions"].isin(weather_options)]

if "Order_Date" in df.columns:
    min_date, max_date = df["Order_Date"].min(), df["Order_Date"].max()
    date_range = st.sidebar.date_input("Rentang Tanggal", [min_date, max_date])
    if len(date_range) == 2:
        df = df[(df["Order_Date"] >= pd.to_datetime(date_range[0])) & (df["Order_Date"] <= pd.to_datetime(date_range[1]))]

# ================================
# Overview Metrics
# ================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Orders", len(df))
with col2:
    st.metric("Unique Delivery Persons", df["Delivery_person_ID"].nunique())
with col3:
    if "Time_taken (min)" in df.columns:
        st.metric("Rata-rata Waktu Antar (menit)", round(df["Time_taken (min)"].mean(), 2))
with col4:
    if "Distance_km" in df.columns:
        st.metric("Rata-rata Jarak (km)", round(df["Distance_km"].mean(), 2))
    else:
        st.metric("Rata-rata Jarak (km)", "-")

# ================================
# Visualisasi
# ================================

# Distribusi waktu antar
if "Time_taken (min)" in df.columns:
    fig1 = px.histogram(df, x="Time_taken (min)", nbins=40, title="Distribusi Waktu Antar (menit)")
    st.plotly_chart(fig1, use_container_width=True)

# Boxplot waktu antar per kota
if "City" in df.columns and "Time_taken (min)" in df.columns:
    fig2 = px.box(df, x="City", y="Time_taken (min)", color="City", title="Perbandingan Waktu Antar per Kota")
    st.plotly_chart(fig2, use_container_width=True)

# Trend order harian
if "Order_Date" in df.columns:
    daily_orders = df.groupby("Order_Date").size().reset_index(name="Orders")
    fig3 = px.line(daily_orders, x="Order_Date", y="Orders", title="Trend Jumlah Order Harian")
    st.plotly_chart(fig3, use_container_width=True)

# Scatter plot: waktu antar vs kondisi lalu lintas
if "Road_traffic_density" in df.columns and "Time_taken (min)" in df.columns:
    fig4 = px.box(df, x="Road_traffic_density", y="Time_taken (min)", color="Road_traffic_density",
                  title="Waktu Antar vs Kepadatan Lalu Lintas")
    st.plotly_chart(fig4, use_container_width=True)

# ================================
# Data Table
# ================================

st.subheader("📋 Data Preview")
st.dataframe(df.head(50))
