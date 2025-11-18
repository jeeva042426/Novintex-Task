# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import timedelta
import zipfile

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Household Power - Analytics", layout="wide", page_icon="⚡")

# ---------- Helpers ----------
@st.cache_data
def download_ucidata():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    try:
        # Try direct read of a few rows to validate access
        _ = pd.read_csv(url, compression='zip', sep=';', nrows=5)
        return url
    except Exception:
        # fallback: use requests to fetch bytes and return BytesIO of contained file
        import requests, io
        r = requests.get(url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        fname = [n for n in z.namelist() if n.endswith('.txt')][0]
        df_bytes = z.read(fname)
        return BytesIO(df_bytes)


def load_data(filelike, nrows=None):
    """
    Loads the UCI electricity dataset from a file-like object or URL.
    Handles separator and missing values ('?').
    Accepts:
    - URL string to the zip file (pandas will handle compression='zip')
    - BytesIO containing the .txt content
    - BytesIO of the zip file (we will handle)
    """
    # If filelike is a BytesIO with zip content, extract the .txt inside
    if isinstance(filelike, BytesIO):
        # peek first bytes to guess if it's a zip
        head = filelike.getvalue()[:4]
        if head.startswith(b'PK'):  # zip magic
            z = zipfile.ZipFile(filelike)
            fname = [n for n in z.namelist() if n.endswith('.txt')][0]
            text_bytes = z.read(fname)
            filelike = BytesIO(text_bytes)
        else:
            # already the .txt content in BytesIO
            filelike.seek(0)

    # If it's an UploadedFile-like object with read(), convert to BytesIO
    if hasattr(filelike, "read") and not isinstance(filelike, str):
        try:
            raw_bytes = filelike.read()
            # create BytesIO of content for pandas
            filelike = BytesIO(raw_bytes)
        except Exception:
            # If reading fails, leave as-is and let pandas try
            pass

    # The dataset uses semicolon separator and "?" for missing
    df = pd.read_csv(filelike,
                     sep=';',
                     header=0,
                     low_memory=False,
                     na_values=['?'],
                     nrows=nrows)

    # Clean column names: strip spaces
    df.columns = df.columns.str.strip()

    # Combine Date and Time into datetime
    # Original Date format in dataset: dd/mm/yyyy
    if 'Date' in df.columns and 'Time' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                                        dayfirst=True, errors='coerce')
    else:
        # try to find alternate datetime column
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        else:
            raise ValueError("Could not find 'Date' and 'Time' columns to build a Datetime index.")

    # Convert numeric columns to float (they may be strings)
    numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            # attempt to coerce similarly-named columns
            alt = [col for col in df.columns if col.lower().replace(' ', '_') == c.lower()]
            if alt:
                df.rename(columns={alt[0]: c}, inplace=True)
                df[c] = pd.to_numeric(df[c], errors='coerce')

    # set index
    df = df.set_index('Datetime').sort_index()

    # drop redundant columns if present
    for col in ['Date', 'Time']:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def hourly_resample(df):
    """Resample to hourly mean (returns DataFrame with hourly index)."""
    # ensure Datetime index exists
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex to resample.")
    hourly = df.resample('H').mean()
    # Interpolate small gaps
    hourly = hourly.interpolate(limit=4)
    return hourly


def make_window_features(series, window=24):
    """
    series: pd.Series (hourly)
    returns DataFrame X (lag features) and y (next hour)
    """
    data = {}
    for i in range(window):
        data[f'lag_{i+1}'] = series.shift(i+1)
    X = pd.DataFrame(data)
    y = series.shift(-0)  # current hour aligned with lags
    y = y.shift(-1)  # predict next hour
    valid = X.join(y.rename('target')).dropna()
    X = valid.drop(columns=['target'])
    y = valid['target']
    return X, y


def train_test_time_split(X, y, test_size=0.2):
    n = len(X)
    split = int(n * (1 - test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true.replace(0, np.nan)))) * 100
    return mae, rmse, mape


# ---------- UI ----------
st.title("⚡ Household Electric Power — Full Analysis App")
st.markdown(
    """
    This app performs:
    - Task 1: EDA (time-series trend + missing/abnormal detection + hourly/daily patterns)
    - Task 2: Time-series forecasting (next-hour Global_active_power)
    - Task 3: Anomaly detection & clustering of daily consumption profiles
    - Task 4: Simple rule-based consumption category generator
    """
)

st.sidebar.header("Data Input")
use_download = st.sidebar.checkbox("Download UCI dataset automatically", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload CSV / TXT / ZIP file (semicolon separated)", type=["csv", "txt", "zip"])

# Load dataset
df_load_state = st.sidebar.empty()
df_raw = None
if use_download and uploaded_file is None:
    df_load_state.info("Downloading dataset from UCI (may take a little while)...")
    source = download_ucidata()
    try:
        df_raw = load_data(source)
    except Exception as e:
        st.sidebar.error("Automatic download failed. Please upload the dataset manually.")
        st.stop()
    df_load_state.success("Downloaded and loaded dataset.")
elif uploaded_file is not None:
    df_load_state.info("Loading uploaded file...")
    try:
        # Streamlit UploadedFile has .read()
        raw_bytes = uploaded_file.read()
        # if it's a zip file, check for PK header
        if raw_bytes[:2] == b'PK':
            # Try to open zip and extract a .txt
            z = zipfile.ZipFile(BytesIO(raw_bytes))
            fname = [n for n in z.namelist() if n.endswith('.txt')][0]
            txt_bytes = z.read(fname)
            df_raw = load_data(BytesIO(txt_bytes))
        else:
            # pass bytes directly
            df_raw = load_data(BytesIO(raw_bytes))
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()
    df_load_state.success("Uploaded file loaded.")
else:
    st.info("Choose to download from UCI or upload the dataset to begin.")
    st.stop()

# show overview
st.subheader("Data snapshot")
st.write("Rows:", df_raw.shape[0], " | Columns:", df_raw.shape[1])
st.dataframe(df_raw.head())

# ---------- Task 1: EDA ----------
st.header("Task 1 — EDA: Time-series and patterns")

if st.button("Run EDA"):
    st.subheader("1. Time-series trend: Global_active_power (original frequency)")
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))

    # Use DataFrame and column access (robust)
    if 'Global_active_power' not in df_raw.columns:
        st.error("Column 'Global_active_power' not found in dataset. Columns available: " + ", ".join(df_raw.columns))
    else:
        plot_series = df_raw['Global_active_power'].dropna()
        if len(plot_series) > 50000:
            plot_series = plot_series.iloc[:50000]
            ax.set_title("Global_active_power (first 50k rows shown)")
        sns.lineplot(x=plot_series.index, y=plot_series.values, ax=ax)
        ax.set_ylabel("Global_active_power (kW)")
        st.pyplot(fig)

        st.subheader("2. Missing / Abnormal readings")
        missing_count = df_raw['Global_active_power'].isna().sum()
        st.write(f"Missing Global_active_power values: **{missing_count}**")
        abnormal = df_raw[(df_raw['Global_active_power'] < 0) | (df_raw['Global_active_power'] > df_raw['Global_active_power'].quantile(0.999))]
        st.write(f"Found {len(abnormal)} abnormal rows (showing up to 100):")
        st.dataframe(abnormal.head(100))

        st.subheader("3. Hourly and daily patterns")
        st.write("Resampling original data to hourly mean for pattern analysis...")
        hourly_df = hourly_resample(df_raw)  # <-- DataFrame (hourly)
        if 'Global_active_power' not in hourly_df.columns:
            st.error("After resampling, 'Global_active_power' column is missing.")
        else:
            st.write("Hourly series sample:")
            # first 7 days (168 hours)
            sample = hourly_df['Global_active_power'].dropna().iloc[:168]
            if len(sample) > 0:
                st.line_chart(sample)
            else:
                st.write("Not enough hourly data to plot a 7-day sample.")

            st.write("Mean consumption by hour of day (0-23):")
            hourly_df['hour'] = hourly_df.index.hour
            hourly_by_hour = hourly_df.groupby('hour')['Global_active_power'].mean()
            fig2, ax2 = plt.subplots(1, 1, figsize=(9, 4))
            sns.barplot(x=hourly_by_hour.index, y=hourly_by_hour.values, ax=ax2)
            ax2.set_xlabel("Hour of day")
            ax2.set_ylabel("Average Global_active_power (kW)")
            st.pyplot(fig2)

            st.write("Mean consumption by day of week:")
            hourly_df['dow'] = hourly_df.index.dayofweek
            dow = hourly_df.groupby('dow')['Global_active_power'].mean()
            fig3, ax3 = plt.subplots(1, 1, figsize=(9, 4))
            sns.barplot(x=dow.index, y=dow.values, ax=ax3)
            ax3.set_xlabel("Day of week (0=Mon)")
            ax3.set_ylabel("Average Global_active_power (kW)")
            st.pyplot(fig3)

# ---------- Task 2: Forecasting ----------
st.header("Task 2 — Time-series Forecasting (next-hour Global_active_power)")

forecast_run = st.button("Run Forecasting")
if forecast_run:
    if 'Global_active_power' not in df_raw.columns:
        st.error("Column 'Global_active_power' not found — cannot run forecasting.")
    else:
        with st.spinner("Preparing hourly data and windowed features..."):
            hourly_df = hourly_resample(df_raw)
            hourly_series = hourly_df['Global_active_power'].dropna()
            st.write("Hourly data length:", len(hourly_series))
            if len(hourly_series) < 48:
                st.error("Not enough hourly data to build 24-lag features.")
            else:
                # build window features (previous 24 hours -> predict next hour)
                X, y = make_window_features(hourly_series, window=24)
                st.write("Feature matrix shape:", X.shape)
                # train-test split in time
                X_train, X_test, y_train, y_test = train_test_time_split(X, y, test_size=0.2)

                with st.spinner("Training RandomForestRegressor..."):
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)

                with st.spinner("Evaluating..."):
                    y_pred = model.predict(X_test)
                    mae, rmse, mape = compute_metrics(y_test, y_pred)
                    st.metric("MAE", f"{mae:.4f} kW")
                    st.metric("RMSE", f"{rmse:.4f} kW")
                    st.metric("MAPE", f"{mape:.2f} %")

                    st.subheader("Predicted vs Actual (test set — last 200 points)")
                    comp_df = pd.DataFrame({'actual': y_test.values, 'predicted': y_pred})
                    comp_df = comp_df.reset_index(drop=True)
                    fig4, ax4 = plt.subplots(1, 1, figsize=(12, 4))
                    ax4.plot(comp_df['actual'].values[-200:], label='Actual')
                    ax4.plot(comp_df['predicted'].values[-200:], label='Predicted', alpha=0.75)
                    ax4.legend()
                    ax4.set_ylabel("Global_active_power (kW)")
                    st.pyplot(fig4)

                    st.write("Download predictions as CSV:")
                    csv = comp_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download predictions.csv", csv, "predictions.csv", "text/csv")

# ---------- Task 3: Unsupervised Learning ----------
st.header("Task 3 — Anomaly Detection & Daily Clustering")

if st.button("Run Anomaly Detection & Clustering"):
    if 'Global_active_power' not in df_raw.columns:
        st.error("Column 'Global_active_power' not found — cannot run anomaly detection or clustering.")
    else:
        with st.spinner("Preparing hourly series for anomaly detection..."):
            hourly_df = hourly_resample(df_raw)
            hourly_series = hourly_df['Global_active_power'].dropna()
            hr_df = pd.DataFrame({'power': hourly_series})
            hr_df['timestamp'] = hr_df.index
            hr_df = hr_df.reset_index(drop=True)

        with st.spinner("Running IsolationForest for anomalies..."):
            iso = IsolationForest(contamination=0.01, random_state=42)  # tune as needed
            hr_df['anomaly'] = iso.fit_predict(hr_df[['power']])
            hr_df['anomaly_flag'] = hr_df['anomaly'].apply(lambda x: 'anomaly' if x == -1 else 'normal')
            anomalies = hr_df[hr_df['anomaly_flag'] == 'anomaly']
            st.write(f"Detected anomalies (hourly): {len(anomalies)}")
            st.dataframe(anomalies.head(50))

            fig5, ax5 = plt.subplots(1, 1, figsize=(12, 4))
            ax5.plot(hr_df['timestamp'], hr_df['power'], label='power', alpha=0.7)
            if not anomalies.empty:
                ax5.scatter(anomalies['timestamp'], anomalies['power'], color='red', s=20, label='anomaly')
            ax5.set_ylabel("Global_active_power (kW)")
            ax5.legend()
            st.pyplot(fig5)

        with st.spinner("Building daily profiles for clustering..."):
            # Build daily profiles properly from the hourly DataFrame (ensure 24 points per day)
            hourly_df_full = hourly_resample(df_raw)  # DataFrame with hourly rows
            # pivot: group by date and collect 24-hour vectors
            daily_profiles = []
            dates = []
            for day, group in hourly_df_full.groupby(hourly_df_full.index.date):
                # reindex group to full 24 hours for that day to ensure consistent length
                day_idx = pd.date_range(start=pd.to_datetime(day), periods=24, freq='H')
                gp = group.reindex(day_idx)['Global_active_power']
                if gp.isna().sum() > 6:
                    # too many missing hours — skip this day (tunable)
                    continue
                # interpolate remaining missing
                gp = gp.interpolate(limit=4).fillna(method='ffill').fillna(method='bfill')
                if len(gp) == 24:
                    daily_profiles.append(gp.values)
                    dates.append(pd.to_datetime(day))
            if len(daily_profiles) == 0:
                st.error("No daily profiles (24-hr) could be constructed. Check data density.")
            else:
                daily_profiles = np.array(daily_profiles)
                st.write("Daily profiles shape:", daily_profiles.shape)

                # scale before clustering
                scaler = StandardScaler()
                X_daily = scaler.fit_transform(daily_profiles)

                # choose number of clusters (let user choose)
                n_clusters = st.slider("Select number of clusters for daily profiles", min_value=2, max_value=6, value=3)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_daily)

                st.write("Cluster counts:")
                counts = pd.Series(labels).value_counts().sort_index()
                st.dataframe(counts.rename("count"))

                # show cluster centers (inverse transform)
                centers = scaler.inverse_transform(kmeans.cluster_centers_)

                fig6, ax6 = plt.subplots(figsize=(10, 5))
                hours = np.arange(24)
                for i, center in enumerate(centers):
                    ax6.plot(hours, center, label=f'Cluster {i}')
                ax6.set_xlabel("Hour of day")
                ax6.set_ylabel("Average Global_active_power (kW)")
                ax6.set_title("Cluster centers (daily consumption profiles)")
                ax6.legend()
                st.pyplot(fig6)

                # attach cluster label to dates and show one sample day per cluster
                sample_info = []
                for k in range(n_clusters):
                    idxs = np.where(labels == k)[0]
                    sample_day = dates[idxs[0]] if len(idxs) > 0 else None
                    avg_power = daily_profiles[idxs].mean() if len(idxs) > 0 else np.nan
                    sample_info.append({
                        'cluster': k,
                        'sample_day': sample_day,
                        'avg_daily_power': float(np.nanmean(daily_profiles[idxs])) if len(idxs) > 0 else np.nan,
                        'count': len(idxs)
                    })
                st.subheader("Cluster characteristics (one row per cluster)")
                st.dataframe(pd.DataFrame(sample_info))

# ---------- Task 4: Simple Rule-Based AI ----------
st.header("Task 4 — Consumption Category Generator (rule-based)")

st.write("Based on a _predicted_ Global_active_power value (kW), assign a category and suggestion.")

pred_val = st.number_input("Enter predicted Global_active_power (kW):", min_value=0.0, format="%.3f", value=1.25)


def categorize_power(val):
    # Category thresholds chosen heuristically; adjust as needed
    if val < 1.5:
        return "Low Usage", "Good — keep up the efficient usage. Continue monitoring and maintain energy-saving habits."
    elif 1.5 <= val < 3.0:
        return "Medium Usage", "Moderate usage — consider running heavy appliances during off-peak hours and check appliance efficiency."
    else:
        return "High Usage", "High usage — reduce simultaneous heavy loads, inspect appliances for faults, and consider energy audits."


if st.button("Generate Category & Suggestion"):
    cat, sug = categorize_power(pred_val)
    st.markdown(
        f"""
        <div style="padding:14px;border-radius:10px;background:#f7f9fb;">
        <h3>⚡ Usage Category: <b>{cat}</b></h3>
        <p><b>Suggestion:</b> {sug}</p>
        </div>
        """, unsafe_allow_html=True)
    st.write("### Example output")
    st.code(f"Predicted Power: {pred_val:.3f} kW  →  Category: {cat}  →  Suggestion: {sug}")

# ---------- Footer ----------
st.markdown("---")
st.write("Notes:")
st.write("""
- The forecasting here uses a simple RandomForest on lag-features (previous 24 hours). For production/time-series specialists, consider ARIMA/SARIMAX, Prophet, or deep learning models (LSTM/TCN/Transformer).
- Clustering uses KMeans on daily 24-hour profiles; interpret clusters by inspecting center curves and cluster sample days.
- Anomaly detection uses IsolationForest. Tune contamination and features to your needs.
""")
