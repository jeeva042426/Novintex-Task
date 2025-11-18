"""
Streamlit Wind Turbine SCADA Analyzer (fixed)
 - EDA: time-series and missing/abnormal detection
 - Scatter: Wind Speed vs LV ActivePower (power curve)
 - Forecasting: 1-step-ahead for 4 variables (persistence baseline + RandomForest)
 - Anomaly detection: underperformance vs theoretical power curve
 - AI: Performance score (0-100), categorize, suggestion

Save as streamlit_turbine_app.py and run:
    streamlit run streamlit_turbine_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Wind Turbine SCADA Analyzer", page_icon="🌀")

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_csv(path_or_buffer):
    try:
        df = pd.read_csv(path_or_buffer)
    except Exception:
        # fallback encoding/engine
        df = pd.read_csv(path_or_buffer, engine='python', encoding='latin1')
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # common name mapping (best-effort)
    rename_map = {}
    for c in df.columns:
        lc = c.lower().replace(' ', '_')
        if 'date' in lc or 'time' in lc:
            rename_map[c] = 'Date/Time'
        if 'active' in lc and 'power' in lc:
            rename_map[c] = 'LV ActivePower (kW)'
        if 'wind' in lc and 'speed' in lc:
            rename_map[c] = 'Wind Speed (m/s)'
        if 'theoretical' in lc or 'power_curve' in lc or 'theoretical_power' in lc:
            rename_map[c] = 'Theoretical_Power_Curve (kWh)'
        if 'direction' in lc:
            rename_map[c] = 'Wind Direction (°)'
    df = df.rename(columns=rename_map)
    # parse datetime if detected
    if 'Date/Time' in df.columns:
        df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')
    return df

def basic_checks(df):
    required = ['Date/Time','LV ActivePower (kW)','Wind Speed (m/s)','Theoretical_Power_Curve (kWh)','Wind Direction (°)']
    missing = [c for c in required if c not in df.columns]
    return missing

def create_lag_features(series, lags=6):
    """
    series: pd.Series with a DatetimeIndex
    returns: DataFrame with y and lag_1..lag_n and dayofweek/hour (index preserved, dropna applied)
    """
    s = series.copy()
    df = pd.DataFrame({'y': s})
    for lag in range(1, lags+1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    # use the timestamp index for features
    df['dayofweek'] = df.index.dayofweek
    df['hour'] = df.index.hour
    df = df.dropna()
    return df

def train_model(X_train, y_train, model_type='rf'):
    if model_type == 'rf':
        m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        m.fit(X_train, y_train)
        return m
    else:
        raise ValueError("Unsupported model")

def evaluate_and_plot(y_true, y_pred, title="Actual vs Predicted"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true.index, y=y_true.values, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=y_true.index, y=np.asarray(y_pred), mode='lines', name='Predicted'))
    fig.update_layout(title=f"{title} — MAE {mae:.3f} RMSE {rmse:.3f}", xaxis_title="Date/Time", yaxis_title="Value")
    return mae, rmse, fig

# -------------------------
# Sidebar: Data load + nav + duplicate handling
# -------------------------
st.sidebar.title("Controls")
uploaded = st.sidebar.file_uploader("Upload local CSV (Kaggle Wind Turbine SCADA) or leave blank", type=['csv'])
path_input = st.sidebar.text_input("Or paste local file path (optional)", value="")
use_default_sample = st.sidebar.checkbox("Use small built-in sample? (demo)", value=False)

# Duplicate handling options
dup_handling = st.sidebar.selectbox("Duplicate timestamps handling", options=[
    "aggregate_mean (recommended)",
    "keep_first",
    "keep_last",
    "make_unique (tiny offsets)"
], index=0)

if uploaded is not None:
    df_raw = load_csv(uploaded)
elif path_input.strip() != "":
    df_raw = load_csv(path_input.strip())
elif use_default_sample:
    # synthetic sample for demo
    rng = pd.date_range(end=pd.Timestamp.now().floor('H'), periods=500, freq='H')
    wind_speed = np.abs(np.random.normal(loc=8, scale=3, size=len(rng))).clip(0.1)
    theo = 0.5 * (wind_speed**3) / (wind_speed.max()**3) * 1500  # toy theoretical shape
    active = theo * (np.random.uniform(0.6,1.05,size=len(rng)))  # real production near theoretical
    direction = np.random.uniform(0,360,size=len(rng))
    df_raw = pd.DataFrame({
        'Date/Time': rng,
        'LV ActivePower (kW)': active,
        'Wind Speed (m/s)': wind_speed,
        'Theoretical_Power_Curve (kWh)': theo,
        'Wind Direction (°)': direction
    })
else:
    st.sidebar.info("Upload a CSV or enter a path, or enable demo sample.")
    st.stop()

# quick column check
missing_cols = basic_checks(df_raw)
if missing_cols:
    st.error(f"CSV missing these required columns (auto-detection failed): {missing_cols}. Please upload a compatible CSV.")
    st.dataframe(df_raw.head())
    st.stop()

# -------------------------
# Preprocessing: sort, handle duplicate timestamps, set index & ensure hourly index
# -------------------------
# Ensure Date/Time is datetime and sort
df_raw = df_raw.copy()
df_raw['Date/Time'] = pd.to_datetime(df_raw['Date/Time'], errors='coerce')
df_raw = df_raw.sort_values('Date/Time').reset_index(drop=True)

# Inspect duplicates
dup_count = df_raw['Date/Time'].duplicated().sum()
if dup_count > 0:
    st.sidebar.warning(f"Found {dup_count} duplicate timestamps in Date/Time — duplicate handling: {dup_handling}")

# Apply duplicate handling choice
if dup_handling.startswith('aggregate_mean'):
    # keep non-datetime columns aggregated by mean where possible (numeric), others by first
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = [c for c in df_raw.columns if c not in numeric_cols and c != 'Date/Time']
    # groupby Date/Time
    grouped = df_raw.groupby('Date/Time', as_index=False)
    df_agg = grouped.agg({**{c: 'mean' for c in numeric_cols}, **{c: 'first' for c in other_cols}})
    df_proc = df_agg
elif dup_handling == 'keep_first':
    df_proc = df_raw.drop_duplicates(subset='Date/Time', keep='first').copy()
elif dup_handling == 'keep_last':
    df_proc = df_raw.drop_duplicates(subset='Date/Time', keep='last').copy()
elif dup_handling == 'make_unique':
    # add tiny microsecond offsets to duplicates (not ideal for analysis but preserves rows)
    df_proc = df_raw.copy()
    counts = df_proc.groupby('Date/Time').cumcount()
    df_proc['Date/Time'] = df_proc['Date/Time'] + pd.to_timedelta(counts, unit='us')
else:
    df_proc = df_raw.copy()

# Now set index and try to asfreq to hourly. If fails, fall back to resample hourly with mean.
df_proc = df_proc.sort_values('Date/Time').reset_index(drop=True)
df_proc = df_proc.set_index('Date/Time')

# attempt hourly reindexing; if the index has duplicates (shouldn't after handling) or other problems, fallback to resample('H').mean()
try:
    # If original timestamps are not exactly hourly but regularly spaced, asfreq will create hourly index and fill with NaN
    df_proc = df_proc.asfreq('H')
except Exception as e:
    # fallback: resample hourly and aggregate numeric columns by mean
    st.sidebar.info("asfreq('H') failed; falling back to resample('H').mean() to produce hourly index.")
    # resample numeric columns by mean, forward/backward fill small gaps later
    df_proc = df_proc.resample('H').mean()

# Convert to numeric for expected columns (safe coercion)
for col in ['LV ActivePower (kW)','Wind Speed (m/s)','Theoretical_Power_Curve (kWh)','Wind Direction (°)']:
    if col in df_proc.columns:
        df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')

# Create filled columns (interpolate by time, with forward/backfill)
df_proc['LV_filled'] = df_proc['LV ActivePower (kW)'].interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
df_proc['WindSpeed_filled'] = df_proc['Wind Speed (m/s)'].interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
df_proc['Theor_filled'] = df_proc['Theoretical_Power_Curve (kWh)'].interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
df_proc['Direction_filled'] = df_proc['Wind Direction (°)'].interpolate(method='time').fillna(method='bfill').fillna(method='ffill')

# expose final df_raw used by app
df_raw = df_proc.copy()

# Navigation
page = st.sidebar.radio("Navigate", ["Overview / EDA", "Scatter: Power Curve", "Forecasting (all 4)", "Anomaly Detection", "AI Performance Score", "Export"])

st.title("Wind Turbine SCADA — Analyzer (fixed)")

# -------------------------
# Page: Overview / EDA
# -------------------------
if page == "Overview / EDA":
    st.header("Task 1 — EDA")
    st.markdown("Time-series trends and missing/abnormal detection for each variable.")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records (hourly index)", f"{len(df_raw)}")
    start_str = str(df_raw.index.min()) if not pd.isna(df_raw.index.min()) else "N/A"
    end_str = str(df_raw.index.max()) if not pd.isna(df_raw.index.max()) else "N/A"
    col2.metric("Start", start_str)
    col3.metric("End", end_str)
    col4.metric("Hourly freq?", "Yes (reindexed/resampled)")

    # Plot time-series for 4 parameters
    st.subheader("Time-series plots")
    vars_plot = [
        ('LV_filled','LV ActivePower (kW)'),
        ('WindSpeed_filled','Wind Speed (m/s)'),
        ('Theor_filled','Theoretical_Power_Curve (kWh)'),
        ('Direction_filled','Wind Direction (°)')
    ]
    for col_name, label in vars_plot:
        st.markdown(f"**{label}**")
        plot_df = df_raw[[col_name]].reset_index().rename(columns={'index':'Date/Time','Date/Time':'Date/Time'}) if False else df_raw.reset_index()
        # plotting
        if col_name in df_raw.columns:
            fig = px.line(df_raw.reset_index(), x=df_raw.reset_index().columns[0], y=col_name, title=label)
            # above px.line often auto-names x as 'Date/Time' when reset_index; ensure correct mapping:
            fig = px.line(df_raw.reset_index(), x='Date/Time', y=col_name, title=label)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"{col_name} not available in data.")
            continue

        # missing and abnormal detection
        raw_col = col_name.replace('_filled','')
        missing = 0
        if raw_col in df_raw.columns:
            missing = df_raw[raw_col].isna().sum()
        st.write(f"Missing raw values: {missing}")

        # abnormal simple rules:
        if 'ActivePower' in label or 'Power' in label:
            abnormal = df_raw[(df_raw[col_name] < 0) | (df_raw[col_name] > df_raw[col_name].quantile(0.999))]
        elif 'Wind Speed' in label:
            abnormal = df_raw[(df_raw[col_name] < 0) | (df_raw[col_name] > 60)]
        elif 'Direction' in label:
            abnormal = df_raw[(df_raw[col_name] < 0) | (df_raw[col_name] > 360)]
        else:
            abnormal = df_raw[(df_raw[col_name] < 0)]
        st.write(f"Abnormal rows detected: {len(abnormal)}")
        if len(abnormal) > 0:
            st.dataframe(abnormal[[col_name]].head(10))
        st.markdown("---")

# -------------------------
# Page: Scatter power curve
# -------------------------
elif page == "Scatter: Power Curve":
    st.header("Wind Speed vs LV ActivePower — Power Curve")
    st.markdown("Scatterplot and binned trend to observe power curve behavior.")
    df_plot = df_raw.reset_index()
    if 'WindSpeed_filled' not in df_plot.columns or 'LV_filled' not in df_plot.columns:
        st.error("Required filled columns not found for plotting. Check preprocessing.")
    else:
        fig = px.scatter(df_plot, x='WindSpeed_filled', y='LV_filled', opacity=0.4, title="LV ActivePower vs Wind Speed",
                         labels={'WindSpeed_filled':'Wind Speed (m/s)','LV_filled':'LV ActivePower (kW)'})
        st.plotly_chart(fig, use_container_width=True)
        # add binned mean curve
        max_ws = df_plot['WindSpeed_filled'].max()
        if pd.isna(max_ws) or max_ws <= 0:
            st.info("Wind speed values look empty or invalid; cannot compute binned mean.")
        else:
            bins = np.linspace(0, max_ws, 30)
            df_plot['wind_bin'] = pd.cut(df_plot['WindSpeed_filled'], bins=bins)
            bin_mean = df_plot.groupby('wind_bin').agg({'WindSpeed_filled':'mean','LV_filled':'mean'}).dropna()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=bin_mean['WindSpeed_filled'], y=bin_mean['LV_filled'], mode='lines+markers', name='Binned mean'))
            fig2.update_layout(title="Binned mean power curve", xaxis_title="Wind Speed (m/s)", yaxis_title="LV ActivePower (kW)")
            st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Page: Forecasting (all 4)
# -------------------------
elif page == "Forecasting (all 4)":
    st.header("Task 2 — Time-series Forecasting (1-step ahead for all 4 variables)")
    st.markdown("Creates lag features and trains separate models for each target (persistence baseline + RandomForest).")

    with st.expander("Forecast settings"):
        lags = st.number_input("Lag periods (hours)", min_value=1, max_value=48, value=6, step=1)
        test_frac = st.slider("Test fraction (time-based)", min_value=0.05, max_value=0.4, value=0.2)
        use_rf = st.checkbox("Use RandomForest models (else baseline only)", value=True)

    targets = {
        'LV ActivePower (kW)': 'LV_filled',
        'Wind Speed (m/s)': 'WindSpeed_filled',
        'Theoretical_Power_Curve (kWh)': 'Theor_filled',
        'Wind Direction (°)': 'Direction_filled'
    }

    results = {}
    for target_name, colname in targets.items():
        st.subheader(f"Forecast: {target_name}")
        if colname not in df_raw.columns:
            st.warning(f"Column {colname} not found. Skipping.")
            continue
        series = pd.Series(df_raw[colname].values, index=df_raw.index, name='y')
        df_feat = create_lag_features(series, lags=int(lags))
        if len(df_feat) < 10:
            st.warning("Not enough records after creating lag features — reduce lags or provide more data.")
            continue
        n_test = max(1, int(len(df_feat) * float(test_frac)))
        train = df_feat.iloc[:-n_test]
        test = df_feat.iloc[-n_test:]
        X_train, y_train = train.drop(columns=['y']), train['y']
        X_test, y_test = test.drop(columns=['y']), test['y']

        # persistence baseline (lag_1)
        y_pred_base = X_test['lag_1'].values
        mae_base = mean_absolute_error(y_test, y_pred_base)
        rmse_base = mean_squared_error(y_test, y_pred_base, squared=False)
        st.write(f"Baseline (persistence) MAE: {mae_base:.4f}  RMSE: {rmse_base:.4f}")

        # RF
        if use_rf:
            model = train_model(X_train, y_train, model_type='rf')
            y_pred = model.predict(X_test)
            mae, rmse = mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred, squared=False)
            st.write(f"RandomForest MAE: {mae:.4f}  RMSE: {rmse:.4f}")
            _, _, fig = evaluate_and_plot(pd.Series(y_test.values, index=X_test.index), y_pred, title=f"{target_name} — Actual vs Predicted")
            st.plotly_chart(fig, use_container_width=True)
            results[target_name] = {'mae':mae, 'rmse':rmse, 'y_test_index':X_test.index, 'y_test':y_test, 'y_pred':y_pred}
        else:
            _, _, fig = evaluate_and_plot(pd.Series(y_test.values, index=X_test.index), y_pred_base, title=f"{target_name} — Baseline Actual vs Predicted")
            st.plotly_chart(fig, use_container_width=True)
            results[target_name] = {'mae':mae_base, 'rmse':rmse_base, 'y_test_index':X_test.index, 'y_test':y_test, 'y_pred':y_pred_base}

# -------------------------
# Page: Anomaly Detection
# -------------------------
elif page == "Anomaly Detection":
    st.header("Task 3 — Anomaly Detection: Underperformance vs Theoretical Power Curve")
    st.markdown("Flag timepoints where actual power deviates strongly (underperforms) relative to theoretical power curve.")

    df = df_raw.copy()
    # compute relative deviation: (theoretical - actual) / theoretical
    df['deviation_rel'] = np.nan
    mask = (df['Theor_filled'] > 0)
    df.loc[mask, 'deviation_rel'] = (df.loc[mask,'Theor_filled'] - df.loc[mask,'LV_filled']) / df.loc[mask,'Theor_filled']
    df['deviation_pct'] = df['deviation_rel'] * 100

    st.write("Default underperformance threshold: deviation > 30%")
    thr = st.slider("Underperformance threshold (%)", min_value=5, max_value=100, value=30)
    anomalies = df[(df['deviation_pct'] > thr) & (df['Theor_filled']>0)].sort_values('deviation_pct', ascending=False)
    st.write(f"Detected {len(anomalies)} underperformance points (deviation > {thr}%).")
    if len(anomalies) > 0:
        st.dataframe(anomalies[['LV_filled','Theor_filled','deviation_pct']].head(200))
        # plot anomalies on time series
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['LV_filled'], mode='lines', name='Actual Power'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Theor_filled'], mode='lines', name='Theoretical Power'))
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['LV_filled'], mode='markers', marker=dict(color='red',size=6), name='Underperformance'))
        fig.update_layout(title="Actual vs Theoretical (anomalies highlighted)", xaxis_title="Date/Time", yaxis_title="Power (kW)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No anomalies detected at this threshold.")

# -------------------------
# Page: AI Performance Score
# -------------------------
elif page == "AI Performance Score":
    st.header("Task 4 — AI Turbine Performance Score Generator")
    st.markdown("Score = min(Actual / Theoretical, 1.0) scaled to 0–100. Categorize and suggest actions.")

    df = df_raw.copy()
    mask = df['Theor_filled'] > 0
    df['perf_ratio'] = 0.0
    df.loc[mask, 'perf_ratio'] = (df.loc[mask,'LV_filled'] / df.loc[mask,'Theor_filled']).clip(lower=0.0)
    df['perf_score'] = (df['perf_ratio'].clip(upper=1.0) * 100).round(1)

    def categorize(score):
        if score >= 80:
            return 'Good'
        elif score >= 50:
            return 'Moderate'
        else:
            return 'Poor'
    df['perf_cat'] = df['perf_score'].apply(categorize)

    st.metric("Overall mean performance score", f"{df['perf_score'].mean():.2f}")
    st.markdown("Distribution of performance categories")
    st.dataframe(df['perf_cat'].value_counts().rename_axis('category').reset_index(name='count'))

    # show time series colored by category
    max_show = min(10000, len(df))
    sample_n = st.slider("Show last N points (for plotting)", min_value=100, max_value=max_show, value=min(2000, max_show))
    df_plot = df.tail(sample_n).reset_index()
    if 'Date/Time' not in df_plot.columns:
        df_plot = df_plot.rename(columns={df_plot.columns[0]:'Date/Time'})
    fig = px.scatter(df_plot, x='Date/Time', y='perf_score', color='perf_cat', title="Performance Score over time", labels={'perf_score':'Score (0-100)'})
    st.plotly_chart(fig, use_container_width=True)

    # suggestions (simple)
    st.subheader("Automated suggestions (per category)")
    st.markdown("- **Good (≥80):** Turbine performing well. Continue standard monitoring and schedule routine maintenance.")
    st.markdown("- **Moderate (50–79):** Performance is degraded. Check blade condition, curtailment controls, and check for temporary wakes/icing.")
    st.markdown("- **Poor (<50):** Significant underperformance. Inspect gearbox, generator, yaw/misalignment, blade damage, or SCADA sensor errors. Consider immediate field inspection.")

    st.subheader("Show worst performing timestamps")
    worst = df[df['perf_cat']=='Poor'].sort_values('perf_score').head(200)
    if worst.empty:
        st.info("No Poor category points found.")
    else:
        st.dataframe(worst[['LV_filled','Theor_filled','perf_score']].head(200))

# -------------------------
# Page: Export
# -------------------------
elif page == "Export":
    st.header("Export processed results")
    df = df_raw.copy()
    mask = df['Theor_filled'] > 0
    df['perf_ratio'] = 0.0
    df.loc[mask, 'perf_ratio'] = (df.loc[mask,'LV_filled'] / df.loc[mask,'Theor_filled']).clip(lower=0.0)
    df['perf_score'] = (df['perf_ratio'].clip(upper=1.0) * 100).round(1)
    df['perf_cat'] = df['perf_score'].apply(lambda s: 'Good' if s>=80 else ('Moderate' if s>=50 else 'Poor'))
    out = df.reset_index()[['Date/Time','LV ActivePower (kW)','Wind Speed (m/s)','Theoretical_Power_Curve (kWh)','Wind Direction (°)','LV_filled','Theor_filled','perf_score','perf_cat']]
    csv_bytes = out.to_csv(index=False).encode('utf-8')
    st.download_button("Download processed CSV", data=csv_bytes, file_name="turbine_scada_processed.csv", mime="text/csv")
    st.markdown("You can extend this section to export PDF reports or charts.")

st.markdown("---")
st.info("Tips: For best results, use the Kaggle dataset CSV extracted locally. The app expects hourly or regularly spaced timestamps; adjust 'Duplicate timestamps handling' or the frequency if your data uses a different interval.")
