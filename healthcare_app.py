

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import importlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import sys

st.set_page_config(page_title="Healthcare EDA & Models — Enhanced", layout="wide")

# --- Small helpers -----------------------------------------------------------
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return None

@st.cache_data
def detect_sklearn_onehot_arg():
    """Detect whether OneHotEncoder expects 'sparse' or 'sparse_output'.
    Return a dict of kwargs to pass to OneHotEncoder."""
    try:
        from sklearn.preprocessing import OneHotEncoder as OHE
        import inspect
        params = inspect.signature(OHE).parameters
        if 'sparse_output' in params:
            return {'handle_unknown':'ignore','sparse_output':False}
        elif 'sparse' in params:
            return {'handle_unknown':'ignore','sparse':False}
    except Exception:
        pass
    # fallback
    return {'handle_unknown':'ignore'}

# Lottie embed helper
def lottie_embed(lottie_url, height=200):
    html = f"""
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <lottie-player src="{lottie_url}"  background="transparent"  speed="1"  style="width:100%; height:{height}px;"  loop  autoplay></lottie-player>
    """
    components.html(html, height=height+20)

# --- UI: header with animation ------------------------------------------------
st.title("🩺 Healthcare Dataset — EDA · Prediction · Anomalies")
with st.container():
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown("""
        **An interactive Streamlit app** for exploratory data analysis, supervised prediction of `Test Results`,
        billing anomaly detection, and a template-based clinical recommendation generator.
        """)
    with col2:
        # small Lottie animation
        lottie_embed("https://assets10.lottiefiles.com/packages/lf20_tz0a1h0x.json", height=120)

st.sidebar.header("Data & Controls")
csv_path = st.sidebar.text_input("CSV filename (in same folder)", "healthcare_dataset.csv")
if st.sidebar.button("Load dataset"):
    df = load_csv(csv_path)
else:
    df = load_csv(csv_path)

if df is None:
    st.stop()

# Normalize column names
df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

# --- Main layout with tabs ---------------------------------------------------
tab_eda, tab_model, tab_anom, tab_rec, tab_about = st.tabs(["EDA","Model","Anomaly","Recommendation","About"])

# Pre-detect OneHotEncoder kwargs
ONEHOT_KW = detect_sklearn_onehot_arg()

# ------------------- EDA Tab --------------------------------------------------
with tab_eda:
    st.header("Task 1 — Exploratory Data Analysis")
    st.subheader("Data preview")
    st.dataframe(df.head(8))

    with st.expander("Show full columns and dtypes"):
        st.write(df.dtypes)

    # Numeric plots
    st.subheader("Numeric summaries & distributions")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    quick_numeric = st.multiselect("Choose numeric columns to plot", numeric_cols, default=numeric_cols[:3])
    if quick_numeric:
        for c in quick_numeric:
            fig, axes = plt.subplots(1,2,figsize=(10,3))
            sns.histplot(df[c].dropna(), kde=True, ax=axes[0])
            axes[0].set_title(f"{c} — histogram")
            sns.boxplot(x=df[c].dropna(), ax=axes[1])
            axes[1].set_title(f"{c} — boxplot")
            st.pyplot(fig)

    # Categorical frequencies
    st.subheader("Categorical top frequencies")
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    quick_cat = st.multiselect("Choose categorical columns", cat_cols, default=cat_cols[:3])
    if quick_cat:
        for c in quick_cat:
            vc = df[c].fillna("Missing").value_counts().head(20)
            st.write(f"**{c}**")
            st.bar_chart(vc)

# ------------------- Model Tab ------------------------------------------------
with tab_model:
    st.header("Task 2 — Predict 'Test Results' (supervised)")
    if "Test Results" not in df.columns:
        st.warning("'Test Results' column not found. Upload a dataset with that column to train a model.")
    else:
        default_exclude = ["Name","Doctor","Hospital","Date of Admission","Discharge Date","Room Number"]
        features_all = [c for c in df.columns if c not in default_exclude + ["Test Results"]]
        st.write("Available features (choose which to use):")
        default_choices = [f for f in ["Age","Gender","Medical Condition","Medication","Billing Amount","Admission Type"] if f in df.columns]
        chosen = st.multiselect("Features", features_all, default=default_choices)

        if len(chosen) == 0:
            st.info("Choose at least one feature.")
        else:
            model_df = df[chosen + ["Test Results"]].copy()
            model_df = model_df.dropna(subset=["Test Results"])  # require label
            st.write("Rows available for training:", len(model_df))

            # Light sanitization
            numeric_feats = model_df[chosen].select_dtypes(include=[np.number]).columns.tolist()
            cat_feats = [c for c in chosen if c not in numeric_feats]

            for c in numeric_feats:
                model_df[c] = pd.to_numeric(model_df[c], errors='coerce')
                model_df[c] = model_df[c].fillna(model_df[c].median())
            for c in cat_feats:
                model_df[c] = model_df[c].fillna("Missing").astype(str)

            y_raw = model_df["Test Results"].astype(str)
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            X = model_df[chosen]

            test_size = st.slider("Test set size (%)", 10, 40, 25) / 100.0
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            except Exception as e:
                st.error(f"Error splitting data (maybe too few samples per class): {e}")
                st.stop()

            # Preprocessing
            numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]) if numeric_feats else None
            if cat_feats:
                # pass dynamic kwargs to OneHotEncoder
                cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')), ('onehot', OneHotEncoder(**ONEHOT_KW))])
            else:
                cat_pipe = None

            transformers = []
            if numeric_feats:
                transformers.append(('num', numeric_pipe, numeric_feats))
            if cat_feats:
                transformers.append(('cat', cat_pipe, cat_feats))

            if len(transformers) == 0:
                st.error("No valid numeric or categorical features selected for preprocessing.")
            else:
                preproc = ColumnTransformer(transformers, remainder='drop')
                clf = RandomForestClassifier(n_estimators=150, random_state=42)
                pipeline = Pipeline([('preproc', preproc), ('clf', clf)])

                # Show animated training indicator
                train_button = st.button("Train model")
                if train_button:
                    with st.spinner("Training model... this may take a moment"):
                        # small progress bar animation
                        prog = st.progress(0)
                        last = 0
                        try:
                            for i in range(5):
                                time.sleep(0.12)
                                last = (i+1)*20
                                prog.progress(last)
                            pipeline.fit(X_train, y_train)
                            prog.progress(100)
                            st.success("Training finished")

                            y_pred = pipeline.predict(X_test)
                            acc = accuracy_score(y_test, y_pred)
                            st.metric("Test accuracy", f"{acc:.4f}")

                            st.text("Classification report:")
                            st.text(classification_report(y_test, y_pred, target_names=le.classes_))

                            cm = confusion_matrix(y_test, y_pred)
                            st.write("Confusion matrix:")
                            st.dataframe(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

                            st.subheader("Sample Predicted vs Actual")
                            sample = X_test.reset_index(drop=True).copy()
                            sample['Actual'] = le.inverse_transform(y_test)
                            sample['Predicted'] = le.inverse_transform(y_pred)
                            st.dataframe(sample.head(50))

                            if st.button("Save trained model (model.joblib)"):
                                joblib.dump({'pipeline': pipeline, 'label_encoder': le, 'features': chosen}, 'model.joblib')
                                st.success('Saved model.joblib in current folder')

                        except Exception as e:
                            st.error(f"Training failed: {e}")

# ------------------- Anomaly Tab ---------------------------------------------
with tab_anom:
    st.header("Task 3 — Anomaly detection (Billing Amount)")
    if "Billing Amount" not in df.columns:
        st.warning("No 'Billing Amount' column found.")
    else:
        # coerce to numeric
        billing = pd.to_numeric(df['Billing Amount'], errors='coerce')
        billing = billing.dropna()
        if billing.empty:
            st.warning('No numeric billing records found after cleaning.')
        else:
            st.write(billing.describe())
            z_thresh = st.slider('Z-score threshold', 2.0, 6.0, 3.0)
            iso_cont = st.slider('IsolationForest contamination fraction', 0.001, 0.1, 0.01, step=0.001)

            mean = billing.mean()
            std = billing.std(ddof=0)
            if std == 0:
                z = pd.Series(0, index=billing.index)
                z_anom_idx = []
            else:
                z = (billing - mean) / std
                z_anom_idx = z[z.abs() > z_thresh].index

            iso = IsolationForest(contamination=float(iso_cont), random_state=42)
            try:
                iso_pred = iso.fit_predict(billing.values.reshape(-1,1))
                iso_anom_idx = billing.index[iso_pred == -1]
            except Exception as e:
                st.error(f'IsolationForest error: {e}')
                iso_anom_idx = []

            combined_idx = sorted(set(list(z_anom_idx) + list(iso_anom_idx)))
            st.write(f"Detected {len(combined_idx)} anomalies (combined) out of {len(billing)} billing records.")

            if len(combined_idx) > 0:
                anom_df = df.loc[combined_idx].copy()
                anom_df['Billing Amount'] = pd.to_numeric(anom_df['Billing Amount'], errors='coerce')
                st.subheader('Anomalous rows (preview)')
                st.dataframe(anom_df.head(200))

            st.markdown('**Interpretation examples:** Very high values may be rare procedures or errors; very low values may be discounts or missing data. Flagged rows should be manually reviewed.')

# ------------------- Recommendation Tab --------------------------------------
with tab_rec:
    st.header('Task 4 — AI Doctor Recommendation (template)')
    st.write('Choose a row index to generate a short recommendation (template-based).')
    max_idx = max(0, len(df)-1)
    row_index = st.number_input('Row index (0-based)', min_value=0, max_value=max_idx, value=0, step=1)
    patient = df.iloc[int(row_index)].to_dict()

    age = patient.get('Age', 'Unknown')
    med_cond = patient.get('Medical Condition', 'unspecified')
    medication = patient.get('Medication', 'none')
    predicted_test = patient.get('Test Results', 'Unknown')

    def make_recommendation(age, med_cond, medication, predicted_test):
        s = []
        s.append(f"Patient (Age: {age}) presenting with {med_cond}. Current medication: {medication}.")
        s.append(f"Test result (predicted/recorded): {predicted_test}.")
        pt = str(predicted_test).lower()
        if any(x in pt for x in ['positive','abnormal','high','critical']):
            s.append('Recommendation: Further diagnostic evaluation and confirmatory testing are advised.')
            s.append('Immediate actions: review current medications for interactions, treat symptoms, and consider specialist referral.')
            s.append('Follow-up: repeat testing and close monitoring; counsel patient on warning signs.')
        elif any(x in pt for x in ['negative','normal','low']):
            s.append('Recommendation: Findings are reassuring; continue routine management and follow-up.')
            s.append('Immediate actions: ensure medication adherence and lifestyle advice where appropriate.')
        else:
            s.append('Recommendation: Result unclear — consider repeat testing or alternative diagnostics as clinically indicated.')
        return ' '.join(s)

    if st.button('Generate recommendation'):
        with st.spinner('Generating...'):
            time.sleep(0.4)
            lottie_embed('https://assets4.lottiefiles.com/packages/lf20_puciaact.json', height=140)
            st.success('Done')
            st.write(make_recommendation(age, med_cond, medication, predicted_test))

# ------------------- About Tab ------------------------------------------------
with tab_about:
    st.header('About this app')
    st.markdown('- Uses `ColumnTransformer` + `Pipeline` for preprocessing and modeling.')
    st.markdown('- Compatible with different scikit-learn versions: dynamically uses `sparse_output` or `sparse` where available for `OneHotEncoder`.')
    st.markdown('- Animations use Lottie and are embedded using `components.html` for a lightweight, dependency-free UI flourish.')
    st.markdown('**Run:** `streamlit run healthcare_app_streamlit_enhanced.py`')
    st.markdown('**Notes:** This app generates template-based clinical suggestions — it is not a replacement for professional medical judgement.')


