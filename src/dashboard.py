import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math
import os
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Demand Forecasting | Sri Lanka Retail",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #f0f2f8; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0818 0%, #1a1040 50%, #0f0c29 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio > label {
    color: #94a3b8 !important; font-size: 10px; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 11px 16px; margin: 4px 0;
    color: #cbd5e0 !important; font-size: 13px; font-weight: 500; transition: all 0.2s;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(124,58,237,0.18); border-color: rgba(124,58,237,0.4);
}

.page-header {
    background: linear-gradient(135deg, #0a0818 0%, #1a1040 55%, #0f0c29 100%);
    padding: 28px 34px; border-radius: 16px; margin-bottom: 26px;
    border: 1px solid rgba(124,58,237,0.25); box-shadow: 0 8px 32px rgba(0,0,0,0.18);
}
.page-header h1 { color:#fff; font-size:22px; font-weight:700; margin:0 0 6px 0; letter-spacing:-0.3px; }
.page-header p  { color:#94a3b8; font-size:13px; margin:0; line-height:1.6; }

.metric-card {
    background:#fff; border-radius:14px; padding:20px 22px;
    border:1px solid #e8ecf4; box-shadow:0 2px 10px rgba(0,0,0,0.05); height:100%;
}
.metric-label { color:#64748b; font-size:11px; font-weight:700; letter-spacing:0.09em; text-transform:uppercase; margin-bottom:7px; }
.metric-value { color:#1e293b; font-size:28px; font-weight:700; font-family:'DM Mono',monospace; line-height:1; }
.metric-sub   { color:#94a3b8; font-size:11px; margin-top:6px; }

.insight-box {
    background: linear-gradient(135deg, #ede9fe, #ddd6fe);
    border-left: 4px solid #7c3aed; border-radius: 0 12px 12px 0;
    padding: 16px 20px; margin: 16px 0;
}
.insight-box p { color:#3b1f8c; font-size:13px; margin:0; line-height:1.7; }

.info-box {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border-left: 4px solid #3b82f6; border-radius: 0 12px 12px 0;
    padding: 16px 20px; margin: 16px 0;
}
.info-box p { color:#1e40af; font-size:13px; margin:0; line-height:1.7; }

.warn-box {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border-left: 4px solid #f59e0b; border-radius: 0 12px 12px 0;
    padding: 16px 20px; margin: 16px 0;
}
.warn-box p { color:#92400e; font-size:13px; margin:0; line-height:1.7; }

.section-title {
    color:#1e293b; font-size:15px; font-weight:700;
    margin:24px 0 14px 0; padding-bottom:8px; border-bottom: 2px solid #e8ecf4;
}

.styled-table { width:100%; border-collapse:separate; border-spacing:0; font-size:13px; }
.styled-table th {
    background:#1e293b; color:#fff; padding:12px 16px; text-align:left;
    font-weight:600; font-size:11px; letter-spacing:0.07em; text-transform:uppercase;
}
.styled-table th:first-child { border-radius:8px 0 0 0; }
.styled-table th:last-child  { border-radius:0 8px 0 0; }
.styled-table td { padding:12px 16px; border-bottom:1px solid #f1f4f9; color:#334155; vertical-align:middle; }
.styled-table tr:hover td { background:#f8fafc; }
.styled-table tr:last-child td { border-bottom:none; }
.best-row td { background:#f0fdf4 !important; font-weight:600; }

.badge-excellent { background:#dcfce7; color:#166534; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }
.badge-good      { background:#fef9c3; color:#854d0e; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }
.badge-hard      { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }

.order-required  { background:linear-gradient(135deg,#fff1f2,#ffe4e6); border:2px solid #f87171; border-radius:16px; padding:30px; text-align:center; margin:20px 0; }
.order-sufficient{ background:linear-gradient(135deg,#f0fdf4,#dcfce7); border:2px solid #4ade80; border-radius:16px; padding:30px; text-align:center; margin:20px 0; }
.order-critical  { background:linear-gradient(135deg,#fff7ed,#ffedd5); border:2px solid #fb923c; border-radius:16px; padding:30px; text-align:center; margin:20px 0; }
.order-qty-num   { font-size:56px; font-weight:800; font-family:'DM Mono',monospace; line-height:1; }

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white !important; border: none; padding: 12px 32px;
    border-radius: 10px; font-weight: 600; font-size: 14px;
    transition: all 0.2s; box-shadow: 0 4px 16px rgba(124,58,237,0.35);
    font-family: 'DM Sans', sans-serif;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 22px rgba(124,58,237,0.48); }

.white-card {
    background:#fff; border-radius:14px; padding:22px 26px;
    border:1px solid #e8ecf4; box-shadow:0 2px 10px rgba(0,0,0,0.04); margin-bottom:16px;
}

.sentiment-positive { background:linear-gradient(135deg,#f0fdf4,#dcfce7); border:1px solid #86efac; border-radius:14px; padding:20px; }
.sentiment-negative { background:linear-gradient(135deg,#fff1f2,#ffe4e6); border:1px solid #fca5a5; border-radius:14px; padding:20px; }
.sentiment-neutral  { background:linear-gradient(135deg,#f8fafc,#f1f5f9); border:1px solid #cbd5e1; border-radius:14px; padding:20px; }

.forecast-ready   { background:#f0fdf4; border:1px solid #86efac; border-radius:10px; padding:12px 18px; font-size:13px; color:#166534; }
.forecast-waiting { background:#fafafa; border:1px solid #e2e8f0; border-radius:10px; padding:12px 18px; font-size:13px; color:#94a3b8; }

.stSelectbox > label, .stSlider > label, .stNumberInput > label, .stDateInput > label {
    color:#475569 !important; font-size:12px !important;
    font-weight:700 !important; letter-spacing:0.05em !important; text-transform:uppercase !important;
}
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
button[kind="header"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

PRODUCT_NAMES = {
    265559:  "Grocery Staples B (Rice/Flour)",
    364606:  "Grocery Staples A (Rice/Flour)",
    502331:  "Bread A",
    564287:  "Baked Goods B",
    584028:  "Meat Product A",
    903285:  "Poultry A",
    1047679: "Soft Drinks A",
    1427659: "Dairy Product",
    1473474: "Vegetables B",
    1503844: "Vegetables A",
    1695835: "Fruit A (Bananas)"
}

PRODUCT_FAMILY = {
    265559:  "GROCERY I",    364606:  "GROCERY I",
    502331:  "BREAD/BAKERY", 564287:  "BREAD/BAKERY",
    584028:  "MEATS",        903285:  "POULTRY",
    1047679: "BEVERAGES",    1427659: "DAIRY",
    1473474: "PRODUCE",      1503844: "PRODUCE",  1695835: "PRODUCE"
}

STORE_NAMES = {44: "Store Colombo", 51: "Store Gampaha"}

MODEL_METRICS = {
    "Moving Average":    {"RMSE": 88.06,  "MAE": 48.54, "R2": 0.8129, "MAPE": 55.97, "WMAPE": 27.68},
    "Linear Regression": {"RMSE": 70.22,  "MAE": 37.46, "R2": 0.8810, "MAPE": 50.51, "WMAPE": 21.36},
    "Prophet":           {"RMSE": 74.55,  "MAE": 41.14, "R2": 0.8689, "MAPE": 53.48, "WMAPE": 23.25},
    "LSTM":              {"RMSE": 66.15,  "MAE": 35.40, "R2": 0.8944, "MAPE": 49.81, "WMAPE": 20.19},
    "XGBoost":           {"RMSE": 66.14,  "MAE": 34.90, "R2": 0.8944, "MAPE": 50.25, "WMAPE": 19.90},
}

TEST_METRICS = {"RMSE": 63.14, "MAE": 34.51, "R2": 0.8863, "MAPE": 58.89, "WMAPE": 21.78}

FEATURE_LABELS = {
    'sales_lag_1':           'Sales Yesterday',
    'sales_lag_2':           'Sales 2 Days Ago',
    'sales_lag_7':           'Sales 7 Days Ago',
    'sales_lag_14':          'Sales 14 Days Ago',
    'sales_lag_28':          'Sales 28 Days Ago',
    'sales_rolling_mean_7':  '7-Day Rolling Average',
    'sales_rolling_mean_28': '28-Day Rolling Average',
    'sales_rolling_std_7':   '7-Day Sales Variability',
    'dayofweek':             'Day of Week',
    'month':                 'Month of Year',
    'quarter':               'Quarter',
    'day':                   'Day of Month',
    'is_weekend':            'Weekend Flag',
    'is_month_start':        'Month Start Flag',
    'is_month_end':          'Month End Flag',
    'onpromotion':           'On Promotion',
    'holiday_impact':        'Sri Lankan Holiday Impact',
    'family_encoded':        'Product Category',
    'temperature_c':         'Temperature (C)',
    'rainfall_mm':           'Rainfall (mm)',
    'windspeed_kmh':         'Wind Speed (km/h)',
    'cpi_normalized':        'Consumer Price Index',
    'fuel_normalized':       'Fuel Price Index',
}

PRODUCT_CONFIG = {
    502331:  {'name': 'Bread A',           'perishable': True,  'safety_pct': 0.05},
    564287:  {'name': 'Baked Goods B',     'perishable': True,  'safety_pct': 0.05},
    584028:  {'name': 'Meat Product A',    'perishable': True,  'safety_pct': 0.10},
    903285:  {'name': 'Poultry A',         'perishable': True,  'safety_pct': 0.10},
    1503844: {'name': 'Vegetables A',      'perishable': True,  'safety_pct': 0.10},
    1473474: {'name': 'Vegetables B',      'perishable': True,  'safety_pct': 0.10},
    1695835: {'name': 'Fruit A (Bananas)', 'perishable': True,  'safety_pct': 0.10},
    1427659: {'name': 'Dairy Product',     'perishable': True,  'safety_pct': 0.10},
    1047679: {'name': 'Soft Drinks A',     'perishable': False, 'safety_pct': None},
    364606:  {'name': 'Staples A (Rice)',  'perishable': False, 'safety_pct': None},
    265559:  {'name': 'Staples B (Rice)',  'perishable': False, 'safety_pct': None},
}

Z_SCORES           = {90: 1.28, 95: 1.65, 99: 2.33}
PERISHABLE_IDS     = [502331, 564287, 584028, 903285, 1503844, 1473474, 1695835, 1427659]
NON_PERISHABLE_IDS = [1047679, 364606, 265559]
ENV_FEATURES       = {'temperature_c', 'rainfall_mm', 'windspeed_kmh', 'cpi_normalized', 'fuel_normalized'}

TEST_START = pd.Timestamp('2017-01-29')
TEST_END   = pd.Timestamp('2017-08-15')


@st.cache_data
def load_val_predictions():
    try:
        df = pd.read_csv('../results/xgboost_predictions.csv', parse_dates=['date'])
        if 'predicted' in df.columns and 'predictions' not in df.columns:
            df = df.rename(columns={'predicted': 'predictions'})
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def load_test_predictions():
    try:
        df = pd.read_csv('../results/xgboost_test_predictions.csv', parse_dates=['date'])
        if 'predicted' in df.columns and 'predictions' not in df.columns:
            df = df.rename(columns={'predicted': 'predictions'})
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def load_lr_predictions():
    try:
        df = pd.read_csv('../results/linear_regression_predictions.csv', parse_dates=['date'])
        for old, new in [('predicted', 'predictions'), ('pred', 'predictions')]:
            if old in df.columns and 'predictions' not in df.columns:
                df = df.rename(columns={old: new})
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def load_prophet_predictions():
    try:
        df = pd.read_csv('../results/prophet_predictions.csv', parse_dates=['date'])
        for old, new in [('predicted', 'predictions'), ('actuals', 'actual')]:
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        if 'error' not in df.columns and 'predictions' in df.columns and 'actual' in df.columns:
            df['error'] = df['actual'] - df['predictions']
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def load_ma_predictions():
    try:
        df = pd.read_csv('../results/moving_average_7day_predictions.csv', parse_dates=['date'])
        if 'ma_prediction' in df.columns and 'predictions' not in df.columns:
            df = df.rename(columns={'ma_prediction': 'predictions'})
        if 'unit_sales' in df.columns and 'actual' not in df.columns:
            df = df.rename(columns={'unit_sales': 'actual'})
        if 'error' not in df.columns and 'predictions' in df.columns and 'actual' in df.columns:
            df['error'] = df['actual'] - df['predictions']
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def load_shap_data():
    try:
        return pd.read_csv('../results/shap/complete_shap_values.csv', parse_dates=['date'])
    except:
        return pd.DataFrame()

@st.cache_data
def load_shap_importance():
    try:
        return pd.read_csv('../results/shap/shap_feature_importance.csv')
    except:
        return pd.DataFrame()

@st.cache_data
def load_train_data():
    try:
        return pd.read_csv('../data/processed/train_processed.csv', parse_dates=['date'])
    except:
        return pd.DataFrame()

@st.cache_data
def load_test_env():
    try:
        return pd.read_csv('../data/processed/test_env.csv', parse_dates=['date'])
    except:
        return pd.DataFrame()

@st.cache_data
def load_weather_data():
    try:
        return pd.read_csv('../data/external/weather_colombo_gampaha.csv', parse_dates=['date'])
    except:
        return pd.DataFrame()

@st.cache_data
def load_economic_data():
    try:
        return pd.read_csv('../data/external/economic_indicators.csv', parse_dates=['date'])
    except:
        return pd.DataFrame()

@st.cache_resource
def load_model_and_features():
    try:
        with open('../models/xgboost_model_v2.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('../models/feature_list.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, features
    except:
        return None, None


val_preds     = load_val_predictions()
lr_preds      = load_lr_predictions()
prophet_preds = load_prophet_predictions()
ma_preds      = load_ma_predictions()
shap_df       = load_shap_data()
shap_imp_df   = load_shap_importance()
train_data    = load_train_data()
test_env_df   = load_test_env()
weather_df    = load_weather_data()
economic_df   = load_economic_data()
test_preds    = load_test_predictions()


def create_lag_features(df):
    df_lagged = df.copy().sort_values(['store_nbr', 'item_nbr', 'date'])
    for lag in [1, 2, 7, 14, 28]:
        df_lagged[f'sales_lag_{lag}'] = df_lagged.groupby(
            ['store_nbr', 'item_nbr'])['unit_sales'].shift(lag)
    df_lagged['sales_rolling_mean_7'] = df_lagged.groupby(
        ['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=7).mean())
    df_lagged['sales_rolling_mean_28'] = df_lagged.groupby(
        ['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=28, min_periods=28).mean())
    df_lagged['sales_rolling_std_7'] = df_lagged.groupby(
        ['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=7).std())
    return df_lagged.dropna()


def page_header(icon, title, subtitle=""):
    st.markdown(f"""
    <div class="page-header">
        <h1>{icon} {title}</h1>
        {"<p>" + subtitle + "</p>" if subtitle else ""}
    </div>""", unsafe_allow_html=True)

def metric_card(label, value, sub=""):
    return f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>"""

def insight(text):
    st.markdown(f'<div class="insight-box"><p>{text}</p></div>', unsafe_allow_html=True)

def info(text):
    st.markdown(f'<div class="info-box"><p>{text}</p></div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box"><p>{text}</p></div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def chart_style(fig, height=380, title=""):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#334155', size=12),
        margin=dict(l=10, r=10, t=48 if title else 20, b=10),
        height=height,
        legend=dict(bgcolor='rgba(255,255,255,0.92)', bordercolor='#e8ecf4',
                    borderwidth=1, font=dict(size=11)),
        title=dict(text=title, font=dict(size=13, color='#1e293b', family='DM Sans'), x=0)
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f1f5f9', showline=True, linecolor='#e2e8f0', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#f1f5f9', showline=True, linecolor='#e2e8f0', zeroline=False)
    return fig

def product_label(pid):
    return PRODUCT_NAMES.get(pid, str(pid))

def status_badge(mae):
    if mae < 20:   return '<span class="badge-excellent">Excellent</span>'
    elif mae < 45: return '<span class="badge-good">Moderate</span>'
    else:          return '<span class="badge-hard">Challenging</span>'

def accuracy_badge(wmape):
    if wmape < 20:   return '<span class="badge-excellent">Accurate</span>'
    elif wmape < 35: return '<span class="badge-good">Moderate</span>'
    else:            return '<span class="badge-hard">High Error</span>'


st.sidebar.markdown("""
<div style="padding:22px 10px 14px 10px; text-align:center;">
    <div style="font-size:15px; font-weight:700; color:#f1f5f9; letter-spacing:-0.2px;">Demand Forecasting</div>
    <div style="font-size:10px; color:#475569; margin-top:4px; letter-spacing:0.1em; text-transform:uppercase;">Sri Lanka Retail</div>
</div>
<hr style="border-color:rgba(255,255,255,0.07); margin:8px 0 18px 0;">
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["Forecast View", "Why This Prediction?", "Order Calculator",
     "Model Analysis", "Market Intelligence"],
    label_visibility="collapsed"
)

st.sidebar.markdown("""
<hr style="border-color:rgba(255,255,255,0.07); margin:20px 0 16px 0;">
<div style="padding:0 6px;">
    <div style="font-size:10px; color:#475569; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:12px;">System Info</div>
    <div style="font-size:12px; color:#94a3b8; line-height:2.2;">
        Data: 2013-2017<br>
        Stores: 2<br>
        Products: 11<br>
        Features: 23<br>
    </div>
</div>
<hr style="border-color:rgba(255,255,255,0.07); margin:16px 0 14px 0;">
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 1 — FORECAST VIEW
# ─────────────────────────────────────────────
if page == "Forecast View":

    page_header("", "Sales Forecast",
                "Select a store and date range, then click Run Forecast to see predictions for all products.")

    section("Set Up Your Forecast")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_store = st.selectbox("Store", options=[44, 51], format_func=lambda x: STORE_NAMES[x])
    with fc2:
        start_date = st.date_input(
            "Forecast Start Date",
            value=pd.Timestamp('2017-02-01').date(),
            min_value=TEST_START.date(),
            max_value=(TEST_END - pd.Timedelta(days=6)).date(),
            help="Available range: Jan 29 to Aug 9, 2017"
        )
    with fc3:
        horizon = st.selectbox(
            "Forecast Length",
            options=[7, 14, 30],
            format_func=lambda x: f"{x} days {'(recommended)' if x==7 else ''}",
            index=0
        )

    st.markdown("<br>", unsafe_allow_html=True)
    btn_col, status_col = st.columns([1, 4])

    with btn_col:
        run_clicked = st.button("Run Forecast", use_container_width=True)

    if 'forecast_ran'        not in st.session_state: st.session_state.forecast_ran        = False
    if 'forecast_all'        not in st.session_state: st.session_state.forecast_all        = {}
    if 'forecast_params'     not in st.session_state: st.session_state.forecast_params     = {}

    if run_clicked:
        model, ALL_FEATURES = load_model_and_features()
        if model is None:
            st.error("Model file not found. Check ../models/xgboost_model_v2.pkl")
            st.stop()
        if test_env_df.empty:
            st.error("Test data not found. Check ../data/processed/test_env.csv")
            st.stop()

        start_ts = pd.Timestamp(start_date)
        end_ts   = start_ts + pd.Timedelta(days=horizon - 1)

        with st.spinner("Running forecast for all products..."):
            test_lagged = create_lag_features(test_env_df)
            all_results = {}

            for pid in PRODUCT_NAMES.keys():
                mask = (
                    (test_lagged['store_nbr'] == sel_store) &
                    (test_lagged['item_nbr']  == pid) &
                    (test_lagged['date']      >= start_ts) &
                    (test_lagged['date']      <= end_ts)
                )
                subset = test_lagged[mask].copy()
                if subset.empty:
                    continue
                preds = np.maximum(model.predict(subset[ALL_FEATURES]), 0)
                subset['predictions'] = preds
                subset['error']       = subset['unit_sales'] - subset['predictions']
                all_results[pid] = subset

            if not all_results:
                st.warning("No data found for this store and date range. Try a later start date (after Jan 29, 2017).")
                st.stop()

            st.session_state.forecast_ran    = True
            st.session_state.forecast_all    = all_results
            st.session_state.forecast_params = {
                'store': sel_store, 'start': start_ts,
                'end': end_ts, 'horizon': horizon
            }

    with status_col:
        if st.session_state.forecast_ran and st.session_state.forecast_all:
            p = st.session_state.forecast_params
            n = len(st.session_state.forecast_all)
            st.markdown(f"""
            <div class="forecast-ready">
                Forecast ready: {STORE_NAMES[p['store']]} /
                {p['start'].strftime('%d %b')} to {p['end'].strftime('%d %b %Y')} /
                {p['horizon']}-day forecast / {n} products
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="forecast-waiting">
                Set your filters above and click Run Forecast
            </div>""", unsafe_allow_html=True)
            st.stop()

    all_results = st.session_state.forecast_all
    if not all_results:
        st.stop()

    # ── SUMMARY TABLE ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("All Products Overview")

    summary_rows = []
    for pid, df in all_results.items():
        avg_actual = df['unit_sales'].mean()
        avg_pred   = df['predictions'].mean()
        mae        = df['error'].abs().mean()
        wmape      = df['error'].abs().sum() / df['unit_sales'].replace(0, np.nan).sum() * 100
        summary_rows.append({
            'pid':        pid,
            'name':       PRODUCT_NAMES[pid],
            'avg_actual': avg_actual,
            'avg_pred':   avg_pred,
            'mae':        mae,
            'wmape':      wmape,
        })

    summary_rows.sort(key=lambda x: x['wmape'])

    trows = ""
    for row in summary_rows:
        trows += f"""<tr>
            <td><strong>{row['name']}</strong></td>
            <td style="font-family:'DM Mono'; text-align:right;">{row['avg_actual']:.0f}</td>
            <td style="font-family:'DM Mono'; text-align:right;">{row['avg_pred']:.0f}</td>
            <td style="font-family:'DM Mono'; text-align:right;">{row['mae']:.1f}</td>
            <td style="font-family:'DM Mono'; text-align:right;">{row['wmape']:.1f}%</td>
        </tr>"""

    st.markdown(f"""
    <div style="border-radius:12px; overflow:hidden; border:1px solid #e8ecf4;">
    <table class="styled-table">
        <thead><tr>
            <th>Product</th>
            <th style="text-align:right">Avg Actual</th>
            <th style="text-align:right">Avg Forecast</th>
            <th style="text-align:right">MAE</th>
            <th style="text-align:right">Accuracy (WMAPE)</th>
        </tr></thead>
        <tbody>{trows}</tbody>
    </table></div>
    """, unsafe_allow_html=True)

    # ── INDIVIDUAL PRODUCT CHARTS ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("Product Forecasts")

    for row in summary_rows:
        pid = row['pid']
        df  = all_results[pid]

        wmape     = row['wmape']
        avg_pred  = row['avg_pred']
        avg_actual= row['avg_actual']

        st.markdown(f"""
        <div style="background:#fff; border-radius:14px; padding:18px 22px 6px 22px;
                    border:1px solid #e8ecf4; box-shadow:0 2px 8px rgba(0,0,0,0.04);
                    margin-bottom:4px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div style="font-size:15px; font-weight:700; color:#1e293b;">{row['name']}</div>
                <div style="display:flex; gap:20px; font-size:12px; color:#64748b;">
                    <span>Avg Actual: <strong style="color:#1e293b; font-family:'DM Mono';">{avg_actual:.0f}</strong></span>
                    <span>Avg Forecast: <strong style="color:#7c3aed; font-family:'DM Mono';">{avg_pred:.0f}</strong></span>
                    <span>WMAPE: <strong style="color:#1e293b; font-family:'DM Mono';">{wmape:.1f}%</strong></span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['unit_sales'], name='Actual Sales',
            line=dict(color='#1e293b', width=2), mode='lines+markers',
            marker=dict(size=5, color='#1e293b'),
            hovertemplate='<b>Actual</b>: %{y:.0f} units<br>%{x|%d %b %Y}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['predictions'], name='Forecast',
            line=dict(color='#7c3aed', width=2, dash='dash'), mode='lines+markers',
            marker=dict(size=5, color='#7c3aed', symbol='diamond'),
            hovertemplate='<b>Forecast</b>: %{y:.0f} units<br>%{x|%d %b %Y}<extra></extra>'
        ))
        fig.update_layout(
            hovermode='x unified',
            xaxis_title="", yaxis_title="Units",
            legend=dict(orientation='h', y=1.15, x=0),
            margin=dict(l=10, r=10, t=30, b=10),
            height=220,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='DM Sans', color='#334155', size=11),
        )
        fig.update_xaxes(showgrid=True, gridcolor='#f1f5f9', showline=True, linecolor='#e2e8f0', zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor='#f1f5f9', showline=True, linecolor='#e2e8f0', zeroline=False)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 2 — WHY THIS PREDICTION?
# ─────────────────────────────────────────────
elif page == "Why This Prediction?":

    page_header("", "Why Did the Model Predict This?",
                "See which factors pushed each prediction up or down, and by how much.")

    if shap_imp_df.empty or shap_df.empty:
        st.error("SHAP files not found. Check ../results/shap/ directory.")
        st.stop()

    section("What Influences Forecasts the Most")

    top_n = shap_imp_df.head(12).copy()
    top_n['label'] = top_n['feature'].map(FEATURE_LABELS).fillna(top_n['feature'])
    top_n['norm']  = top_n['mean_abs_shap'] / top_n['mean_abs_shap'].max() * 100
    top_n['color'] = top_n['feature'].apply(lambda f: '#10b981' if f in ENV_FEATURES else '#7c3aed')

    g1, g2 = st.columns([3, 2])

    with g1:
        fig_imp = go.Figure(go.Bar(
            x=top_n['mean_abs_shap'], y=top_n['label'], orientation='h',
            marker_color=top_n['color'].tolist(),
            text=[f"{v:.2f}" for v in top_n['mean_abs_shap']],
            textposition='outside', textfont=dict(size=11, family='DM Mono')
        ))
        fig_imp.update_layout(yaxis=dict(autorange='reversed'), xaxis_title="Average impact on forecast (units)")
        chart_style(fig_imp, height=420)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Purple: Sales history and operational factors   Green: Environmental factors (weather and economy)")

    with g2:
        st.markdown("<br>", unsafe_allow_html=True)
        for _, row in top_n.iterrows():
            bar_color = '#10b981' if row['feature'] in ENV_FEATURES else '#7c3aed'
            st.markdown(f"""
            <div style="margin-bottom:11px;">
                <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
                    <span style="color:#334155; font-weight:500;">{row['label']}</span>
                    <span style="color:{bar_color}; font-family:'DM Mono'; font-weight:600;">{row['mean_abs_shap']:.2f}</span>
                </div>
                <div style="background:#f1f5f9; border-radius:4px; height:6px;">
                    <div style="background:{bar_color}; width:{row['norm']}%; height:6px; border-radius:4px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Explain a Single Prediction")
    info("Pick a product and store to see exactly which factors drove a specific prediction.")

    e1, e2 = st.columns([1, 2])
    with e1:
        shap_product = st.selectbox("Product", options=sorted(PRODUCT_NAMES.keys()),
                                    format_func=product_label, key="shap_prod")
        shap_store   = st.selectbox("Store", options=[44, 51],
                                    format_func=lambda x: STORE_NAMES[x], key="shap_store")
        example_type = st.selectbox("Which prediction?",
                                    options=["Pick a specific date",
                                             "Typical prediction (median)",
                                             "Best prediction (smallest error)",
                                             "Worst prediction (largest error)"])

    sh_filt = shap_df[
        (shap_df['item_nbr'] == shap_product) &
        (shap_df['store_nbr'] == shap_store)
    ].copy()

    if sh_filt.empty:
        with e2:
            st.warning("No SHAP data for this combination.")
    else:
        sh_filt['abs_error'] = (sh_filt['actual'] - sh_filt['prediction']).abs()

    if "specific date" in example_type:
        available_dates = sorted(sh_filt['date'].dt.date.unique())
        picked_date = st.selectbox("Select Date", options=available_dates,
                                   format_func=lambda d: d.strftime('%d %b %Y'),
                                   key="shap_date")
        match   = sh_filt[sh_filt['date'].dt.date == picked_date]
        ex_row  = match.iloc[0] if not match.empty else sh_filt.iloc[len(sh_filt)//2]
    elif "Best" in example_type:
        ex_row = sh_filt.loc[sh_filt['abs_error'].idxmin()]
    elif "Worst" in example_type:
        ex_row = sh_filt.loc[sh_filt['abs_error'].idxmax()]
    else:
        ex_row = sh_filt.iloc[len(sh_filt)//2]

    base_val = float(ex_row.get('base_value', 148.63))
    actual   = float(ex_row['actual'])
    pred     = float(ex_row['prediction'])
    ex_date  = str(ex_row['date'])[:10]

    shap_feats = []
    for col in shap_df.columns:
        if col.startswith('shap_'):
            feat  = col[5:]
            label = FEATURE_LABELS.get(feat, feat)
            val   = float(ex_row[col])
            shap_feats.append({'label': label, 'shap': val, 'feature': feat})
    shap_feats = sorted(shap_feats, key=lambda x: abs(x['shap']), reverse=True)[:10]
    max_abs    = max(abs(s['shap']) for s in shap_feats) if shap_feats else 1

    with e2:
        err_val   = actual - pred
        err_color = '#dc2626' if abs(err_val) > 40 else '#16a34a'
        st.markdown(f"""
        <div class="white-card">
            <div style="font-size:11px; color:#94a3b8; margin-bottom:4px;">{ex_date}</div>
            <div style="font-size:16px; font-weight:700; color:#1e293b; margin-bottom:16px;">
                {PRODUCT_NAMES.get(shap_product,'?')} / {STORE_NAMES.get(shap_store,'?')}
            </div>
            <div style="display:flex; gap:30px; margin-bottom:20px; flex-wrap:wrap;">
                <div>
                    <div class="metric-label">Actual Sales</div>
                    <div style="font-size:22px; font-weight:700; font-family:'DM Mono'; color:#1e293b;">{actual:.0f} units</div>
                </div>
                <div>
                    <div class="metric-label">Forecast</div>
                    <div style="font-size:22px; font-weight:700; font-family:'DM Mono'; color:#7c3aed;">{pred:.0f} units</div>
                </div>
                <div>
                    <div class="metric-label">Difference</div>
                    <div style="font-size:22px; font-weight:700; font-family:'DM Mono'; color:{err_color};">{err_val:+.0f} units</div>
                </div>
            </div>
            <div style="font-size:12px; color:#64748b; margin-bottom:14px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">
                Starting point: {base_val:.0f} units average. Each factor adjusts it:
            </div>""", unsafe_allow_html=True)

        for item in shap_feats:
            sv      = item['shap']
            clr     = "#16a34a" if sv > 0 else "#dc2626"
            arrow   = "+" if sv > 0 else "-"
            bar_pct = min(abs(sv) / max_abs * 100, 100)
            align   = "left" if sv > 0 else "right"
            dot     = "ENV" if item['feature'] in ENV_FEATURES else "   "
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:9px;">
                <div style="width:180px; font-size:12px; color:#334155; font-weight:500; flex-shrink:0;">{dot} {item['label']}</div>
                <div style="flex:1; background:#f1f5f9; border-radius:4px; height:7px; position:relative; overflow:hidden;">
                    <div style="position:absolute; {align}:0; background:{clr}; width:{bar_pct}%; height:7px; border-radius:4px;"></div>
                </div>
                <div style="width:75px; text-align:right; font-size:12px; font-family:'DM Mono'; color:{clr}; font-weight:700;">
                    {arrow} {abs(sv):.1f}
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
            <div style="margin-top:16px; padding-top:12px; border-top:2px solid #e8ecf4;
                        font-size:14px; font-weight:700; color:#1e293b;">
                Final Forecast: {pred:.0f} units
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Model Facts")
    i1, i2, i3, i4 = st.columns(4)
    i1.markdown(metric_card("Top Factor",      "28-Day Avg",   "Most influential"),        unsafe_allow_html=True)
    i2.markdown(metric_card("Holiday Rank",    "21 of 23",     "Ecuador data limitation"), unsafe_allow_html=True)
    i3.markdown(metric_card("CPI Rank",        "12 of 23",     "Economy matters"),         unsafe_allow_html=True)
    i4.markdown(metric_card("Total Explained", "7,346",        "predictions with SHAP"),   unsafe_allow_html=True)


# PAGE 3 — ORDER CALCULATOR

elif page == "Order Calculator":

    page_header("", "Stock Order Calculator",
                "Enter your current stock level. The system calculates exactly how much to order.")

    if train_data.empty:
        st.error("Training data not found.")
        st.stop()
    if test_preds.empty:
        st.error("Forecast data not found.")
        st.stop()

    section("Your Store Details")
    r1, r2, r3 = st.columns(3)

    with r1:
        service_level = st.selectbox(
            "How Often Should You Never Run Out?",
            options=[90, 95, 99], index=1,
            format_func=lambda x: {
                90: "90% of the time",
                95: "95% of the time (recommended)",
                99: "99% of the time"
            }[x]
        )
    with r2:
        lead_time = st.slider("Days Until Delivery", 1, 14, 3,
                              help="How many days from ordering to stock arriving?")
    with r3:
        rep_store = st.selectbox("Store", options=[44, 51], format_func=lambda x: STORE_NAMES[x])

    section("Product and Stock on Hand")
    p1, p2 = st.columns([2, 1])
    with p1:
        rep_product = st.selectbox(
            "Product",
            options=sorted(PRODUCT_CONFIG.keys()),
            format_func=lambda x: f"{PRODUCT_CONFIG[x]['name']}  ({'Perishable' if PRODUCT_CONFIG[x]['perishable'] else 'Non-Perishable'})"
        )
    with p2:
        current_stock = st.number_input("Units Currently in Stock", min_value=0, value=100, step=10)

    st.markdown("<br>", unsafe_allow_html=True)
    calc_btn = st.button("Calculate My Order")

    if calc_btn:
        cfg = PRODUCT_CONFIG[rep_product]
        filt_preds = test_preds[
            (test_preds['item_nbr']  == rep_product) &
            (test_preds['store_nbr'] == rep_store)
        ].sort_values('date').tail(7)

        if filt_preds.empty:
            st.error("No forecast data for this product/store.")
            st.stop()

        forecast_val = filt_preds['predictions'].mean()
        filt_train   = train_data[(train_data['item_nbr']==rep_product) & (train_data['store_nbr']==rep_store)]
        demand_std   = filt_train['unit_sales'].std() if not filt_train.empty else 1.0
        z            = Z_SCORES[service_level]

        if cfg['perishable']:
            safety_stock       = forecast_val * cfg['safety_pct']
            method_str         = f"Perishable buffer: {cfg['safety_pct']*100:.0f}% of forecast"
            formula_comparison = z * demand_std * math.sqrt(lead_time)
        else:
            safety_stock       = z * demand_std * math.sqrt(lead_time)
            method_str         = f"Z({z}) x o({demand_std:.1f}) x sqrt({lead_time})"
            formula_comparison = None

        reorder_point  = (forecast_val * lead_time) + safety_stock
        order_quantity = max(0.0, reorder_point - current_stock)

        if current_stock == 0:    status = "CRITICAL"
        elif order_quantity > 0:  status = "ORDER REQUIRED"
        else:                     status = "STOCK SUFFICIENT"

        section("Results")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.markdown(metric_card("Daily Forecast", f"{forecast_val:.0f}", "units/day"),          unsafe_allow_html=True)
        rc2.markdown(metric_card("Safety Buffer",  f"{safety_stock:.0f}", method_str),           unsafe_allow_html=True)
        rc3.markdown(metric_card("Reorder Point",  f"{reorder_point:.0f}", "forecast + buffer"), unsafe_allow_html=True)
        rc4.markdown(metric_card("Stock on Hand",  f"{current_stock}",    "units"),              unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if status == "CRITICAL":
            st.markdown(f"""
            <div class="order-critical">
                <div style="font-size:14px; font-weight:700; color:#c2410c; margin-bottom:8px;">CRITICAL: OUT OF STOCK</div>
                <div class="order-qty-num" style="color:#c2410c;">{order_quantity:.0f}</div>
                <div style="font-size:16px; font-weight:600; color:#9a3412; margin-top:10px;">
                    units needed immediately: {cfg['name']} / {STORE_NAMES[rep_store]}
                </div>
            </div>""", unsafe_allow_html=True)
        elif status == "ORDER REQUIRED":
            st.markdown(f"""
            <div class="order-required">
                <div style="font-size:14px; font-weight:700; color:#b91c1c; margin-bottom:8px;">ORDER REQUIRED</div>
                <div class="order-qty-num" style="color:#dc2626;">{order_quantity:.0f}</div>
                <div style="font-size:16px; font-weight:600; color:#991b1b; margin-top:10px;">
                    units to order: {cfg['name']} / {STORE_NAMES[rep_store]}
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="order-sufficient">
                <div class="order-qty-num" style="color:#16a34a;">OK</div>
                <div style="font-size:20px; font-weight:700; color:#15803d; margin-top:10px;">STOCK SUFFICIENT</div>
                <div style="font-size:14px; color:#166534; margin-top:8px;">
                    You have {current_stock} units. You need {reorder_point:.0f} units (including safety buffer). No order needed yet.
                </div>
            </div>""", unsafe_allow_html=True)

        with st.expander("Show Calculation Steps"):
            if cfg['perishable']:
                st.markdown(f"""
**Product type:** Perishable. Small percentage buffer to minimise waste.

| Step | Calculation | Result |
|---|---|---|
| 1. Daily Forecast | Average of last 7 predictions | **{forecast_val:.1f} units/day** |
| 2. Safety Buffer | {forecast_val:.1f} x {cfg['safety_pct']*100:.0f}% | **{safety_stock:.1f} units** |
| 3. Reorder Point | ({forecast_val:.1f} x {lead_time} days) + {safety_stock:.1f} | **{reorder_point:.1f} units** |
| 4. Order Quantity | max(0, {reorder_point:.1f} - {current_stock}) | **{order_quantity:.1f} units** |

*Why a small percentage buffer for perishables? The statistical formula would suggest {formula_comparison:.0f} units, which is far too much for items that expire quickly. A small percentage prevents food waste. (Silver, Pyke and Peterson, 1998)*
                """)
            else:
                st.markdown(f"""
**Product type:** Non-Perishable. Statistical safety stock formula.

| Step | Calculation | Result |
|---|---|---|
| 1. Daily Forecast | Average of last 7 predictions | **{forecast_val:.1f} units/day** |
| 2. Safety Stock | Z({z}) x o({demand_std:.1f}) x sqrt({lead_time}) | **{safety_stock:.1f} units** |
| 3. Reorder Point | ({forecast_val:.1f} x {lead_time} days) + {safety_stock:.1f} | **{reorder_point:.1f} units** |
| 4. Order Quantity | max(0, {reorder_point:.1f} - {current_stock}) | **{order_quantity:.1f} units** |

*Z = {z} = service level {service_level}%. o = {demand_std:.2f} = historical daily sales variability (2013-2015). (Silver, Pyke and Peterson, 1998)*
                """)

        section("All Products Quick View")
        st.caption(f"{STORE_NAMES[rep_store]}: assuming 100 units on hand, {lead_time}-day delivery")

        all_rows = []
        for pid, pcfg in PRODUCT_CONFIG.items():
            fp = test_preds[(test_preds['item_nbr']==pid) & (test_preds['store_nbr']==rep_store)].sort_values('date').tail(7)
            if fp.empty: continue
            fc  = fp['predictions'].mean()
            tr  = train_data[(train_data['item_nbr']==pid) & (train_data['store_nbr']==rep_store)]
            std = tr['unit_sales'].std() if not tr.empty else 1.0
            ss  = fc * pcfg['safety_pct'] if pcfg['perishable'] else z * std * math.sqrt(lead_time)
            rp  = (fc * lead_time) + ss
            oq  = max(0.0, rp - 100)
            all_rows.append({
                'Product':        pcfg['name'],
                'Type':           'Perishable' if pcfg['perishable'] else 'Non-Perishable',
                'Daily Forecast': round(fc, 0),
                'Safety Buffer':  round(ss, 0),
                'Reorder Point':  round(rp, 0),
                'Order Qty':      round(oq, 0),
                'Action':         'Order Now' if oq > 0 else 'OK'
            })
        if all_rows:
            st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)



# PAGE 4 — MODEL ANALYSIS

elif page == "Model Analysis":

    page_header("", "Model Performance Analysis",
                "How XGBoost compares to other models, product accuracy, and category breakdown.")

    tab1, tab2, tab3 = st.tabs(["Product Performance", "Category Breakdown", "Model Comparison"])

    with tab3:
        section("2016 Validation Results")
        info("These results were used to select the best model. WMAPE is the primary metric. "
             "It handles days with near-zero sales correctly, unlike raw MAPE.")

        rows_html = ""
        for mname, m in MODEL_METRICS.items():
            is_best = "best-row" if mname == "XGBoost" else ""
            rows_html += f"""
            <tr class="{is_best}">
                <td><strong>{'[Best] ' if mname == 'XGBoost' else ''}{mname}</strong></td>
                <td style="font-family:'DM Mono',monospace; text-align:right;">{m['RMSE']:.2f}</td>
                <td style="font-family:'DM Mono',monospace; text-align:right;">{m['MAE']:.2f}</td>
                <td style="font-family:'DM Mono',monospace; text-align:right;">{m['R2']*100:.2f}%</td>
                <td style="font-family:'DM Mono',monospace; text-align:right;">{m['WMAPE']:.2f}%</td>
                <td style="font-family:'DM Mono',monospace; text-align:right;">{m['MAPE']:.2f}%</td>
            </tr>"""

        st.markdown(f"""
        <div style="border-radius:12px; overflow:hidden; border:1px solid #e8ecf4;">
        <table class="styled-table">
            <thead><tr>
                <th>Model</th>
                <th style="text-align:right">RMSE</th>
                <th style="text-align:right">MAE</th>
                <th style="text-align:right">R2</th>
                <th style="text-align:right">WMAPE (primary)</th>
                <th style="text-align:right">MAPE</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table></div>
        <div style="font-size:11px; color:#94a3b8; margin-top:8px;">
            WMAPE is the primary metric. Raw MAPE is shown for reference only.
            LSTM was evaluated for comparison. XGBoost was selected due to native SHAP explainability.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("2017 Test Results: Final Honest Evaluation")
        info("The model was trained on 2013-2015 only and never saw 2017 data. "
             "These are the official final results proving the model works on completely new data.")

        t1, t2, t3, t4, t5 = st.columns(5)
        t1.markdown(metric_card("R2 Score", f"{TEST_METRICS['R2']*100:.2f}%", "2017 test"), unsafe_allow_html=True)
        t2.markdown(metric_card("RMSE",     f"{TEST_METRICS['RMSE']:.2f}",    "units"),     unsafe_allow_html=True)
        t3.markdown(metric_card("MAE",      f"{TEST_METRICS['MAE']:.2f}",     "units"),     unsafe_allow_html=True)
        t4.markdown(metric_card("WMAPE",    f"{TEST_METRICS['WMAPE']:.2f}%",  "primary"),   unsafe_allow_html=True)
        t5.markdown(metric_card("MAPE",     f"{TEST_METRICS['MAPE']:.2f}%",   "reference"), unsafe_allow_html=True)

        val_r2   = MODEL_METRICS['XGBoost']['R2'] * 100
        test_r2  = TEST_METRICS['R2'] * 100
        r2_drop  = round(val_r2 - test_r2, 2)
        val_mae  = MODEL_METRICS['XGBoost']['MAE']
        test_mae = TEST_METRICS['MAE']

        insight(
            f"R2 dropped only {r2_drop}% from validation to test ({val_r2:.2f}% to {test_r2:.2f}%). "
            f"MAE {'improved' if test_mae < val_mae else 'changed'} slightly "
            f"({val_mae:.2f} to {test_mae:.2f} units). "
            "This small gap confirms the model generalises well to completely new data."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        names  = list(MODEL_METRICS.keys())
        wmapes = [m['WMAPE'] for m in MODEL_METRICS.values()]
        r2vals = [m['R2']*100 for m in MODEL_METRICS.values()]
        colors = ['#7c3aed' if n == 'XGBoost' else '#c4b5fd' for n in names]

        with mc1:
            fig_w = go.Figure(go.Bar(
                x=names, y=wmapes, marker_color=colors,
                text=[f"{v:.2f}%" for v in wmapes], textposition='outside',
                textfont=dict(size=11, family='DM Mono')
            ))
            fig_w.update_layout(yaxis=dict(title="WMAPE % (lower is better)"), xaxis_title="")
            chart_style(fig_w, height=320, title="WMAPE: Primary Accuracy Metric")
            st.plotly_chart(fig_w, use_container_width=True)

        with mc2:
            fig_r2 = go.Figure(go.Bar(
                x=names, y=r2vals, marker_color=colors,
                text=[f"{v:.2f}%" for v in r2vals], textposition='outside',
                textfont=dict(size=11, family='DM Mono')
            ))
            fig_r2.update_layout(yaxis=dict(range=[60, 93], title="R2 % (higher is better)"), xaxis_title="")
            chart_style(fig_r2, height=320, title="R2 Score")
            st.plotly_chart(fig_r2, use_container_width=True)

    with tab1:
        if val_preds.empty:
            st.error("Forecast data not loaded.")
        else:
            pf = val_preds.copy()
            pf['abs_error'] = pf['error'].abs()

            prod_stats = pf.groupby('item_nbr').agg(
                avg_actual=('actual', 'mean'),
                mae=('abs_error', 'mean')
            ).reset_index().sort_values('mae')

            wmape_series = pf.groupby('item_nbr').apply(
                lambda x: x['abs_error'].sum() / x['actual'].replace(0, np.nan).sum() * 100
            ).reset_index()
            wmape_series.columns = ['item_nbr', 'wmape']
            prod_stats = prod_stats.merge(wmape_series, on='item_nbr')

            prod_stats['name']       = prod_stats['item_nbr'].map(PRODUCT_NAMES)
            prod_stats['family']     = prod_stats['item_nbr'].map(PRODUCT_FAMILY)
            prod_stats['perishable'] = prod_stats['item_nbr'].apply(
                lambda x: "Perishable" if x in PERISHABLE_IDS else "Non-Perishable")

            section("All Products: Best to Worst Accuracy")
            medals = {0: "1st", 1: "2nd", 2: "3rd"}
            trows  = ""
            for i, (_, row) in enumerate(prod_stats.iterrows()):
                medal = medals.get(i, "")
                trows += f"""<tr>
                    <td>{medal} <strong>{row['name']}</strong></td>
                    <td style="font-size:11px; color:#64748b;">{row['family']}</td>
                    <td>{row['perishable']}</td>
                    <td style="font-family:'DM Mono'; text-align:right;">{row['avg_actual']:.0f}</td>
                    <td style="font-family:'DM Mono'; text-align:right;">{row['mae']:.1f}</td>
                    <td style="font-family:'DM Mono'; text-align:right;">{row['wmape']:.1f}%</td>
                </tr>"""

            st.markdown(f"""
            <div style="border-radius:12px; overflow:hidden; border:1px solid #e8ecf4;">
            <table class="styled-table">
                <thead><tr>
                    <th>Product</th><th>Category</th><th>Type</th>
                    <th style="text-align:right">Avg Daily Sales</th>
                    <th style="text-align:right">MAE</th>
                    <th style="text-align:right">WMAPE</th>   
                </tr></thead>
                <tbody>{trows}</tbody>
            </table></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            pp1, pp2 = st.columns(2)

            with pp1:
                fig_pp = go.Figure(go.Bar(
                    x=prod_stats['mae'], y=prod_stats['name'], orientation='h',
                    marker=dict(color=prod_stats['mae'],
                                colorscale=[[0, '#7c3aed'], [0.5, '#f59e0b'], [1, '#dc2626']],
                                showscale=False),
                    text=[f"{v:.1f}" for v in prod_stats['mae']],
                    textposition='outside', textfont=dict(size=10, family='DM Mono')
                ))
                fig_pp.update_layout(yaxis=dict(autorange='reversed'), xaxis_title="MAE (units)")
                chart_style(fig_pp, height=380, title="Error by Product")
                st.plotly_chart(fig_pp, use_container_width=True)

            with pp2:
                store_stats = pf.groupby('store_nbr').agg(
                    mae=('abs_error', 'mean'), avg=('actual', 'mean')
                ).reset_index()

                store_wmape = pf.groupby('store_nbr').apply(
                    lambda x: x['abs_error'].sum() / x['actual'].replace(0, np.nan).sum() * 100
                ).reset_index()
                store_wmape.columns = ['store_nbr', 'wmape']
                store_stats = store_stats.merge(store_wmape, on='store_nbr')
                store_stats['name'] = store_stats['store_nbr'].map(STORE_NAMES)

                fig_st = go.Figure()
                for idx, (_, sr) in enumerate(store_stats.iterrows()):
                    fig_st.add_trace(go.Bar(
                        name=sr['name'], x=['MAE (units)', 'WMAPE (%)'],
                        y=[sr['mae'], sr['wmape']],
                        marker_color=['#7c3aed', '#4f46e5'][idx],
                        text=[f"{sr['mae']:.1f}", f"{sr['wmape']:.1f}%"],
                        textposition='outside'
                    ))
                fig_st.update_layout(barmode='group', legend=dict(orientation='h', y=1.12))
                chart_style(fig_st, height=280, title="Colombo vs Gampaha")
                st.plotly_chart(fig_st, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                for _, sr in store_stats.iterrows():
                    st.markdown(
                        metric_card(sr['name'], f"MAE {sr['mae']:.1f}",
                                    f"Avg {sr['avg']:.0f} units/day / WMAPE {sr['wmape']:.1f}%"),
                        unsafe_allow_html=True)
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    with tab2:
        if val_preds.empty:
            st.error("Forecast data not loaded.")
        else:
            pf2 = val_preds.copy()
            pf2['abs_error'] = pf2['error'].abs()
            pf2['family']    = pf2['item_nbr'].map(PRODUCT_FAMILY)

            cat_stats = pf2.groupby('family').agg(
                mae=('abs_error', 'mean'), avg_sales=('actual', 'mean'), count=('actual', 'count')
            ).reset_index()

            cat_wmape = pf2.groupby('family').apply(
                lambda x: x['abs_error'].sum() / x['actual'].replace(0, np.nan).sum() * 100
            ).reset_index()
            cat_wmape.columns = ['family', 'wmape']
            cat_stats = cat_stats.merge(cat_wmape, on='family')
            cat_stats = cat_stats.sort_values('mae')

            section("Forecast Accuracy by Category")
            fig_cat = go.Figure(go.Bar(
                x=cat_stats['family'], y=cat_stats['mae'],
                marker_color='#7c3aed',
                text=[f"{v:.1f}" for v in cat_stats['mae']],
                textposition='outside', textfont=dict(size=12, family='DM Mono')
            ))
            fig_cat.update_layout(xaxis_title="Category", yaxis_title="Average MAE (units)")
            chart_style(fig_cat, height=340, title="Average Error by Category")
            st.plotly_chart(fig_cat, use_container_width=True)

            section("Category Summary")
            cat_disp = cat_stats.rename(columns={
                'family': 'Category', 'mae': 'Avg MAE', 'avg_sales': 'Avg Daily Sales',
                'count': 'Records', 'wmape': 'WMAPE'
            })
            cat_disp['Avg MAE']         = cat_disp['Avg MAE'].round(1)
            cat_disp['Avg Daily Sales'] = cat_disp['Avg Daily Sales'].round(1)
            cat_disp['WMAPE']           = cat_disp['WMAPE'].round(1).astype(str) + '%'
            st.dataframe(cat_disp, use_container_width=True, hide_index=True)



# PAGE 5 — MARKET INTELLIGENCE

elif page == "Market Intelligence":

    page_header("", "Market Intelligence",
                "Weather, economic conditions and live news sentiment: signals that influence demand.")

    warn("Environmental and economic data shown here were used as model features. "
         "The live sentiment indicator is a demonstration. It provides market context "
         "but does not feed into the current XGBoost predictions.")

    tab1, tab2, tab3 = st.tabs(["Weather", "Economic Indicators", "Live News Sentiment"])

    with tab1:
        if weather_df.empty:
            st.error("Weather data not found at ../data/external/weather_colombo_gampaha.csv")
        else:
            section("Temperature and Rainfall: Colombo and Gampaha (2013-2017)")
            info("Temperature, rainfall and wind speed are included as model features. "
                 "Their SHAP rankings (15 to 17 out of 23) indicate a modest but present effect on demand.")

            store_filter = st.selectbox("View data for",
                                        options=["Both Stores", "Store Colombo", "Store Gampaha"])

            w = weather_df.copy()
            if 'store_nbr' in w.columns:
                if store_filter == "Store Colombo":
                    w = w[w['store_nbr'] == 44]
                elif store_filter == "Store Gampaha":
                    w = w[w['store_nbr'] == 51]
                else:
                    w = w.groupby('date').mean(numeric_only=True).reset_index()

            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(
                x=w['date'], y=w['temperature_c'], name='Temperature (C)',
                line=dict(color='#f59e0b', width=1.5),
                hovertemplate='%{y:.1f}C<br>%{x|%d %b %Y}<extra></extra>'
            ))
            fig_temp.update_layout(yaxis_title="Temperature (C)", xaxis_title="")
            chart_style(fig_temp, height=260, title="Daily Temperature")
            st.plotly_chart(fig_temp, use_container_width=True)

            fig_rain = go.Figure()
            fig_rain.add_trace(go.Bar(
                x=w['date'], y=w['rainfall_mm'], name='Rainfall (mm)',
                marker_color='#3b82f6',
                hovertemplate='%{y:.1f}mm<br>%{x|%d %b %Y}<extra></extra>'
            ))
            fig_rain.update_layout(yaxis_title="Rainfall (mm)", xaxis_title="")
            chart_style(fig_rain, height=220, title="Daily Rainfall")
            st.plotly_chart(fig_rain, use_container_width=True)

            wc1, wc2, wc3 = st.columns(3)
            wc1.markdown(metric_card("Avg Temperature", f"{w['temperature_c'].mean():.1f}C", "2013-2017"), unsafe_allow_html=True)
            wc2.markdown(metric_card("Max Temperature", f"{w['temperature_c'].max():.1f}C",  "peak"),      unsafe_allow_html=True)
            wc3.markdown(metric_card("Avg Rainfall",    f"{w['rainfall_mm'].mean():.1f}mm",  "per day"),   unsafe_allow_html=True)

    with tab2:
        if economic_df.empty:
            st.error("Economic data not found at ../data/external/economic_indicators.csv")
        else:
            section("Consumer Price Index and Fuel Prices (2013-2017)")
            info("CPI ranked 12 out of 23 features: rising prices influence what customers buy. "
                 "Fuel prices ranked 22 out of 23 because they barely changed during this period.")

            e = economic_df.copy()
            ec1, ec2 = st.columns(2)

            with ec1:
                if 'cpi_normalized' in e.columns:
                    fig_cpi = go.Figure()
                    fig_cpi.add_trace(go.Scatter(
                        x=e['date'], y=e['cpi_normalized'], name='CPI',
                        line=dict(color='#7c3aed', width=2),
                        fill='tozeroy', fillcolor='rgba(124,58,237,0.08)',
                        hovertemplate='CPI: %{y:.3f}<br>%{x|%b %Y}<extra></extra>'
                    ))
                    fig_cpi.update_layout(yaxis_title="Normalised CPI", xaxis_title="")
                    chart_style(fig_cpi, height=260, title="Consumer Price Index")
                    st.plotly_chart(fig_cpi, use_container_width=True)
                    st.markdown(metric_card("CPI SHAP Rank", "12 of 23", "3.8% normalised importance"), unsafe_allow_html=True)

            with ec2:
                if 'fuel_normalized' in e.columns:
                    fig_fuel = go.Figure()
                    fig_fuel.add_trace(go.Scatter(
                        x=e['date'], y=e['fuel_normalized'], name='Fuel Price',
                        line=dict(color='#f59e0b', width=2),
                        fill='tozeroy', fillcolor='rgba(245,158,11,0.08)',
                        hovertemplate='Fuel: %{y:.3f}<br>%{x|%b %Y}<extra></extra>'
                    ))
                    fig_fuel.update_layout(yaxis_title="Normalised Fuel Price", xaxis_title="")
                    chart_style(fig_fuel, height=260, title="Fuel Price Index")
                    st.plotly_chart(fig_fuel, use_container_width=True)
                    st.markdown(metric_card("Fuel SHAP Rank", "22 of 23", "Near-zero importance"), unsafe_allow_html=True)

            insight("CPI shows a clear upward trend 2013-2017, and the model correctly identifies this as "
                    "a demand signal (ranked 12 of 23). Fuel prices are almost flat for the entire period. "
                    "This is a documented finding: feature importance reflects data variation, not concept importance.")

    with tab3:
        section("Live Economic Sentiment from Sri Lankan News")
        info("This module fetches current Sri Lankan news headlines and scores them for economic sentiment. "
             "It demonstrates how real-time market signals could be used in future model versions.")

        warn("Sentiment analysis is a demonstration feature. "
             "Scores are based on live news at the time of viewing and do not feed into XGBoost predictions.")

        run_sentiment = st.button("Fetch Latest News Sentiment")

        if run_sentiment:
            try:
                import requests
                from bs4 import BeautifulSoup

                try:
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                    analyzer  = SentimentIntensityAnalyzer()
                    use_vader = True
                except ImportError:
                    try:
                        from textblob import TextBlob
                        use_vader = False
                    except ImportError:
                        st.error("Please install: pip install vaderSentiment textblob beautifulsoup4 requests")
                        st.stop()

                sources = [
                    {"name": "Daily Mirror Sri Lanka", "url": "https://www.dailymirror.lk/",  "tag": "a"},
                    {"name": "Colombo Gazette",        "url": "https://colombogazette.com/",   "tag": "h2"},
                ]

                all_headlines = []
                with st.spinner("Fetching Sri Lankan news..."):
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                    for source in sources:
                        try:
                            resp = requests.get(source['url'], headers=headers, timeout=8)
                            if resp.status_code == 200:
                                soup = BeautifulSoup(resp.content, 'html.parser')
                                for tag in soup.find_all(source['tag'])[:30]:
                                    text = tag.get_text(strip=True)
                                    if 15 < len(text) < 200:
                                        all_headlines.append({"headline": text, "source": source['name']})
                        except:
                            continue

                if not all_headlines:
                    st.warning("Could not fetch headlines. Check your internet connection.")
                    st.stop()

                economic_keywords = [
                    'price', 'inflation', 'economy', 'market', 'trade', 'fuel', 'food', 'cost',
                    'retail', 'import', 'export', 'rupee', 'dollar', 'growth', 'gdp', 'tax',
                    'supply', 'demand', 'shortage', 'harvest', 'agriculture', 'commodity'
                ]

                scored = []
                for item in all_headlines:
                    is_econ = any(kw in item['headline'].lower() for kw in economic_keywords)
                    if use_vader:
                        compound = analyzer.polarity_scores(item['headline'])['compound']
                    else:
                        compound = TextBlob(item['headline']).sentiment.polarity
                    scored.append({
                        'headline':  item['headline'],
                        'source':    item['source'],
                        'score':     compound,
                        'economic':  is_econ,
                        'sentiment': 'Positive' if compound > 0.05 else 'Negative' if compound < -0.05 else 'Neutral'
                    })

                scored_df       = pd.DataFrame(scored)
                economic_news   = scored_df[scored_df['economic']]
                use_df          = economic_news if not economic_news.empty else scored_df
                avg_score       = use_df['score'].mean()

                section("Current Market Sentiment")

                if avg_score > 0.05:
                    st.markdown("""
                    <div class="sentiment-positive">
                        <div style="font-size:18px; font-weight:700; color:#166534; margin-bottom:10px;">Positive Market Conditions</div>
                        <div style="font-size:14px; color:#15803d; line-height:1.9;">
                            Economic news from Sri Lankan sources is broadly positive today.
                            Consumer confidence may be higher than usual.
                            Consider stocking slightly more on high-demand products this week,
                            especially groceries and beverages.
                        </div>
                    </div>""", unsafe_allow_html=True)
                elif avg_score < -0.05:
                    st.markdown("""
                    <div class="sentiment-negative">
                        <div style="font-size:18px; font-weight:700; color:#991b1b; margin-bottom:10px;">Cautious Market Conditions</div>
                        <div style="font-size:14px; color:#b91c1c; line-height:1.9;">
                            Economic news from Sri Lankan sources is broadly negative today.
                            Customers may be more price-sensitive than usual.
                            Consider tighter stock on premium or non-essential items.
                            Essential goods (rice, bread, vegetables) demand should remain stable.
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="sentiment-neutral">
                        <div style="font-size:18px; font-weight:700; color:#475569; margin-bottom:10px;">Neutral Market Conditions</div>
                        <div style="font-size:14px; color:#64748b; line-height:1.9;">
                            No strong economic signal in today's Sri Lankan news.
                            Standard forecast-based ordering is appropriate.
                            Use the Order Calculator page for your stock decisions.
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                section("Headlines Analysed")
                disp_df = use_df[['headline', 'source', 'sentiment', 'score']].copy()
                disp_df['score'] = disp_df['score'].round(3)
                disp_df.columns = ['Headline', 'Source', 'Sentiment', 'Score']
                st.dataframe(disp_df.head(15), use_container_width=True, hide_index=True)
                st.caption(f"Fetched {len(all_headlines)} headlines total. {len(economic_news)} matched economic keywords. Scored using VADER sentiment analyser.")
       

            except ImportError as e:
                st.error(f"Missing library: {e}. Run: pip install requests beautifulsoup4 vaderSentiment")
            except Exception as e:
                st.error(f"Error fetching sentiment: {e}")

        else:
            st.markdown("""
            <div style="text-align:center; padding:48px 20px; color:#94a3b8;">
                <div style="font-size:15px; font-weight:600; color:#64748b; margin-bottom:8px;">Click the button above to fetch live news</div>
                <div style="font-size:13px;">Sri Lankan news headlines will be scored for economic sentiment</div>
            </div>""", unsafe_allow_html=True)