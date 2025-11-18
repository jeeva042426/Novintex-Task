"""
Streamlit AQI App with Navigation Bar
Save as streamlit_aqi_nav.py and run: streamlit run streamlit_aqi_nav.py
This is a modular refactor of previous framework with a sidebar navbar to jump to pages.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="AQI Navigator", page_icon="📊")

# -------------------------
# Helper functions
# -------------------------
@st.cache_data
def read_csv_path_or_buffer(path_or_buf):
    try:
        df = pd.read_csv(path_or_buf)
    except Exception:
        df = pd.read_csv(path_or_buf, encoding='latin1', engine='python')
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        lc = c.lower().replace(' ', '_')
        if 'date' in lc:
            col_map[c] = 'Date'
        if 'city' in lc:
            col_map[c] = 'City'
        if 'index' in lc or 'aqi' in lc:
            col_map[c] = 'Index Value'
        if 'prominent' in lc or 'pollutant' in lc:
            col_map[c] = 'Prominent Pollutant'
        if 'no.' in lc or 'stations' in lc:
            col_map[c] = 'No. Stations'
    df = df.rename(columns=col_map)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

def prepare_city_df(df_raw, city_label):
    # filter if city column exists
    if 'City' in df_raw.columns:
        df_city = df_raw[df_raw['City'] == city_label].copy()
    else:
        df_city = df_raw.copy()
        df_city['City'] = city_label
    df_city = df_city.sort_values('Date').reset_index(drop=True)
    df_city['Index Value'] = pd.to_numeric(df_city['Index Value'], errors='coerce')
    # reindex to daily frequency to make missing visible
    df_city = df_city.set_index('Date').asfreq('D')
    df_city['Index_filled'] = df_city['Index Value'].interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    df_city = df_city.reset_index()
    return df_city

def create_lag_features(series, lags=7):
    df = pd.DataFrame({'y': series})
    for lag in range(1, lags+1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df = df.dropna()
    return df

# -------------------------
# Data loading (sidebar)
# -------------------------
st.sidebar.title("Data / Nav Controls")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=['csv'])
default_path = "/mnt/data/Agartala_AQIBulletins.csv"
use_default = False
try:
    open(default_path, 'rb').close()
    use_default = True
except Exception:
    use_default = False

if uploaded is not None:
    df_raw = read_csv_path_or_buffer(uploaded)
    st.sidebar.success("Using uploaded CSV")
elif use_default:
    df_raw = read_csv_path_or_buffer(default_path)
    st.sidebar.success(f"Using default file: {default_path}")
else:
    st.sidebar.error("No CSV provided. Upload a file or place one at the default path.")
    st.stop()

# Choose city (if present)
if 'City' in df_raw.columns:
    cities = sorted(df_raw['City'].dropna().unique().tolist())
    city = st.sidebar.selectbox("City", cities, index=0)
else:
    # user provides a label for the single-city CSV
    city = st.sidebar.text_input("City label (CSV has no City column)", value="City")

df_city = prepare_city_df(df_raw, city)

# -------------------------
# Navigation bar
# -------------------------
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate to", [
    "Overview",
    "Forecasting",
    "Clustering",
    "Seasonality (AI)",
    "Export"
])

# Quick top-level header
st.title(f"AQI Analytics — {city}")
st.markdown("Use the left navigation to jump to any analytics page.")

# -------------------------
# Page implementations
# -------------------------
def page_overview(df_city):
    st.header("Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Days", df_city['Date'].nunique())
    c2.metric("Date range", f"{df_city['Date'].min().date()} → {df_city['Date'].max().date()}")
    c3.metric("Missing Index", int(df_city['Index Value'].isna().sum()))
    st.subheader("Time series")
    tab1, tab2 = st.tabs(["Line chart", "Interactive Plotly"])
    with tab1:
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df_city['Date'], df_city['Index_filled'])
        ax.set_ylabel("Index Value")
        ax.grid(alpha=0.2)
        st.pyplot(fig)
    with tab2:
        figp = px.line(df_city, x='Date', y='Index_filled', title="Index (filled)")
        st.plotly_chart(figp, use_container_width=True)

    st.subheader("Prominent pollutant distribution")
    if 'Prominent Pollutant' in df_city.columns:
        poll_counts = df_city['Prominent Pollutant'].fillna("Unknown").value_counts().reset_index()
        poll_counts.columns = ['Pollutant','Count']
        st.plotly_chart(px.bar(poll_counts, x='Pollutant', y='Count'), use_container_width=True)
    else:
        st.info("No 'Prominent Pollutant' column in this data.")

    st.subheader("Missing / Abnormal readings")
    extremes = df_city[(df_city['Index Value'] < 0) | (df_city['Index Value'] > 1000)]
    spikes = ((df_city['Index_filled'] - df_city['Index_filled'].rolling(7, min_periods=1).mean()) /
              df_city['Index_filled'].rolling(7, min_periods=1).std().replace(0, np.nan)).abs() > 4
    st.write(f"- Extreme rows (neg or >1000): {len(extremes)}")
    if not extremes.empty:
        st.dataframe(extremes.head(10))
    st.write(f"- Rolling z-score spikes (threshold 4): {spikes.sum()}")
    if spikes.any():
        st.dataframe(df_city.loc[spikes].head(10))

def page_forecasting(df_city):
    st.header("Forecasting — Next Day")
    st.markdown("Create lag features and train a simple forecasting model (or use naive persistence).")

    col1, col2 = st.columns(2)
    lags = col1.number_input("Lag days (window)", min_value=3, max_value=28, value=7)
    test_frac = col2.slider("Test fraction", 0.05, 0.4, 0.2)

    series = pd.Series(df_city['Index_filled'].values, index=pd.to_datetime(df_city['Date']))
    df_lag = create_lag_features(series, lags=lags)
    n_test = max(1, int(len(df_lag) * test_frac))
    train = df_lag.iloc[:-n_test]
    test = df_lag.iloc[-n_test:]
    X_train, y_train = train.drop(columns=['y']), train['y']
    X_test, y_test = test.drop(columns=['y']), test['y']

    st.write(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    model_choice = st.selectbox("Model", ["Persistence (naive)", "RandomForest"])
    if model_choice == "Persistence (naive)":
        y_pred = X_test['lag_1'].values
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
    st.metric("RMSE", f"{mean_squared_error(y_test, y_pred, squared=False):.2f}")

    plot_df = pd.DataFrame({'Date': X_test.index, 'Actual': y_test.values, 'Predicted': y_pred}).set_index('Date')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Actual'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Predicted'], mode='lines', name='Predicted'))
    st.plotly_chart(fig, use_container_width=True)

    # Next day forecast
    last_vals = series.iloc[-lags:]
    if len(last_vals) < lags:
        st.warning("Not enough history for next-day forecast.")
    else:
        feat = {f'lag_{i}': last_vals.iloc[-i] for i in range(1, lags+1)}
        next_date = series.index[-1] + pd.Timedelta(days=1)
        feat['dayofweek'] = next_date.dayofweek
        feat['month'] = next_date.month
        feat_df = pd.DataFrame([feat])
        if model_choice == "Persistence (naive)":
            next_pred = feat['lag_1']
        else:
            next_pred = model.predict(feat_df)[0]
        st.success(f"Next-day forecast for {next_date.date()}: **{next_pred:.2f}**")

def page_clustering(df_city):
    st.header("Clustering — Group days by pollution level")
    n_clusters = st.slider("Number of clusters", 2, 5, 3)
    clust_df = df_city[['Date','Index_filled']].dropna().copy()
    clust_df['rolling_7'] = clust_df['Index_filled'].rolling(7, min_periods=1).mean()
    features = clust_df[['Index_filled','rolling_7']].fillna(method='bfill').values
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(features)
    clust_df['cluster'] = labels
    centers = km.cluster_centers_[:,0]
    order = np.argsort(centers)
    rank_map = {order[i]: i for i in range(len(order))}
    clust_df['cluster_rank'] = clust_df['cluster'].map(rank_map)
    # map names
    cluster_names = {}
    for r in sorted(clust_df['cluster_rank'].unique()):
        if r == 0:
            cluster_names[r] = 'Low Pollution'
        elif r == max(clust_df['cluster_rank'].unique()):
            cluster_names[r] = 'High Pollution'
        else:
            cluster_names[r] = 'Moderate Pollution'
    clust_df['cluster_name'] = clust_df['cluster_rank'].map(cluster_names)

    st.subheader("Cluster distribution")
    st.dataframe(clust_df['cluster_name'].value_counts().rename_axis('cluster').reset_index(name='count'))

    st.subheader("Cluster ranges (min, mean, max)")
    st.dataframe(clust_df.groupby('cluster_name')['Index_filled'].agg(['min','mean','max']).reset_index())

    st.plotly_chart(px.scatter(clust_df, x='Date', y='Index_filled', color='cluster_name', title="Daily Index by cluster"), use_container_width=True)

def page_seasonality(df_city):
    st.header("Seasonality / AI Seasonal Pattern Detector")
    monthly = df_city.set_index('Date').resample('M')['Index_filled'].mean().dropna()
    monthly_df = monthly.reset_index().rename(columns={'Index_filled':'monthly_avg'})
    month_agg = monthly_df.groupby(monthly_df['Date'].dt.month)['monthly_avg'].mean().reindex(range(1,13))
    ms = pd.DataFrame({'month': range(1,13), 'avg_index': month_agg.values})
    ms['month_name'] = ms['month'].apply(lambda m: datetime(2000, m, 1).strftime('%b'))
    q1, q2 = ms['avg_index'].quantile([0.33, 0.66])
    def categorize(v):
        if v <= q1:
            return "Clean"
        elif v <= q2:
            return "Moderate"
        else:
            return "High Pollution"
    ms['category'] = ms['avg_index'].apply(categorize)
    st.dataframe(ms[['month_name','avg_index','category']])
    st.plotly_chart(px.bar(ms, x='month_name', y='avg_index', color='category', title="Avg Index by Month"), use_container_width=True)

    high_months = ms[ms['category']=='High Pollution']['month_name'].tolist()
    insight = f"High Pollution months: {', '.join(high_months) if high_months else 'None'}."
    st.info("Automated insight:")
    st.write(insight)

def page_export(df_city):
    st.header("Export")
    st.write("Download processed data and summary.")
    proc = df_city[['Date','Index Value','Index_filled']].copy()
    csv = proc.to_csv(index=False).encode('utf-8')
    st.download_button("Download processed CSV", data=csv, file_name=f"{city}_aqi_processed.csv", mime="text/csv")
    st.markdown("You can extend this page to export PDF reports or PowerPoints.")

# -------------------------
# Router - show selected page
# -------------------------
if page == "Overview":
    page_overview(df_city)
elif page == "Forecasting":
    page_forecasting(df_city)
elif page == "Clustering":
    page_clustering(df_city)
elif page == "Seasonality (AI)":
    page_seasonality(df_city)
elif page == "Export":
    page_export(df_city)
else:
    st.write("Page not found.")

# -------------------------
# Footer
# -------------------------
st.sidebar.markdown("---")
st.sidebar.info("Tip: Use the left navigation to jump between analysis pages quickly.")
