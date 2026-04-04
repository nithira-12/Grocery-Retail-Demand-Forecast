"""
============================================================
EXPLAINABLE DEMAND FORECASTING DASHBOARD
Sri Lankan Grocery Retail | XGBoost + SHAP
============================================================

HOW TO RUN:
    cd src
    streamlit run dashboard.py

PAGES:
    1. Forecast View        - Run forecast, filter, see chart + table
    2. SHAP Explanations    - Feature importance + prediction explainer
    3. Replenishment        - Inventory order calculator
    4. Analysis             - Model comparison + product performance (tabbed)
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG - must be first Streamlit call
# ============================================================

st.set_page_config(
    page_title="Demand Forecasting | Sri Lanka",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
    
)

# ============================================================
# CSS STYLING
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #f5f6fa; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio > label {
    color: #94a3b8 !important;
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px; padding: 10px 14px;
    margin: 3px 0; color: #cbd5e0 !important;
    font-size: 13px; font-weight: 500; transition: all 0.15s;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(139,92,246,0.15);
    border-color: rgba(139,92,246,0.35);
}

/* ── PAGE HEADER ── */
.page-header {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 60%, #24243e 100%);
    padding: 26px 32px; border-radius: 14px; margin-bottom: 24px;
    border: 1px solid rgba(139,92,246,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
}
.page-header h1 { color:#fff; font-size:24px; font-weight:700; margin:0 0 5px 0; letter-spacing:-0.3px; }
.page-header p  { color:#94a3b8; font-size:13px; margin:0; }

/* ── METRIC CARDS ── */
.metric-card {
    background:#fff; border-radius:12px; padding:18px 20px;
    border:1px solid #e8ecf4; box-shadow:0 2px 8px rgba(0,0,0,0.05); height:100%;
}
.metric-label { color:#64748b; font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:6px; }
.metric-value { color:#1e293b; font-size:26px; font-weight:700; font-family:'JetBrains Mono',monospace; line-height:1; }
.metric-sub   { color:#94a3b8; font-size:11px; margin-top:5px; }

/* ── INSIGHT BOX ── */
.insight-box { background:linear-gradient(135deg,#ede9fe,#ddd6fe); border-left:4px solid #7c3aed; border-radius:0 10px 10px 0; padding:14px 18px; margin:14px 0; }
.insight-box p { color:#3b1f8c; font-size:13px; margin:0; line-height:1.6; }

/* ── SECTION TITLES ── */
.section-title { color:#1e293b; font-size:15px; font-weight:700; margin:22px 0 12px 0; padding-bottom:7px; border-bottom:2px solid #e8ecf4; }

/* ── TABLE ── */
.styled-table { width:100%; border-collapse:separate; border-spacing:0; font-size:13px; }
.styled-table th { background:#1e293b; color:#fff; padding:11px 14px; text-align:left; font-weight:600; font-size:11px; letter-spacing:0.06em; text-transform:uppercase; }
.styled-table td { padding:11px 14px; border-bottom:1px solid #f1f4f9; color:#334155; vertical-align:middle; }
.styled-table tr:hover td { background:#f8fafc; }
.best-row td { background:#f0fdf4 !important; font-weight:600; }

/* ── STATUS BADGES ── */
.badge-excellent { background:#dcfce7; color:#166534; padding:2px 9px; border-radius:20px; font-size:11px; font-weight:700; }
.badge-good      { background:#fef9c3; color:#854d0e; padding:2px 9px; border-radius:20px; font-size:11px; font-weight:700; }
.badge-hard      { background:#fee2e2; color:#991b1b; padding:2px 9px; border-radius:20px; font-size:11px; font-weight:700; }

/* ── ORDER BOXES ── */
.order-required  { background:linear-gradient(135deg,#fff1f2,#ffe4e6); border:2px solid #f87171; border-radius:14px; padding:28px; text-align:center; margin:20px 0; }
.order-sufficient{ background:linear-gradient(135deg,#f0fdf4,#dcfce7); border:2px solid #4ade80; border-radius:14px; padding:28px; text-align:center; margin:20px 0; }
.order-critical  { background:linear-gradient(135deg,#fff7ed,#ffedd5); border:2px solid #fb923c; border-radius:14px; padding:28px; text-align:center; margin:20px 0; }
.order-qty-num   { font-size:52px; font-weight:800; font-family:'JetBrains Mono',monospace; line-height:1; }

/* ── BUTTONS ── */
.stButton > button {
    background:linear-gradient(135deg,#7c3aed,#4f46e5);
    color:white !important; border:none; padding:11px 30px;
    border-radius:9px; font-weight:600; font-size:14px;
    transition:all 0.2s; box-shadow:0 4px 14px rgba(124,58,237,0.35);
}
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 6px 20px rgba(124,58,237,0.45); }

/* ── WHITE CARD ── */
.white-card { background:#fff; border-radius:12px; padding:20px 24px; border:1px solid #e8ecf4; box-shadow:0 2px 8px rgba(0,0,0,0.04); margin-bottom:16px; }

/* Inputs */
.stSelectbox > label, .stSlider > label, .stNumberInput > label {
    color:#475569 !important; font-size:12px !important;
    font-weight:700 !important; letter-spacing:0.04em !important; text-transform:uppercase !important;
}
            



[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

[data-testid="collapsedControl"] {
    display: none !important;
}

button[kind="header"] {
    display: none !important;
}
            
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================

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
    265559:  "GROCERY I",
    364606:  "GROCERY I",
    502331:  "BREAD/BAKERY",
    564287:  "BREAD/BAKERY",
    584028:  "MEATS",
    903285:  "POULTRY",
    1047679: "BEVERAGES",
    1427659: "DAIRY",
    1473474: "PRODUCE",
    1503844: "PRODUCE",
    1695835: "PRODUCE"
}

STORE_NAMES = {44: "Store Colombo", 51: "Store Gampaha"}

MODEL_METRICS = {
    "Moving Average":    {"RMSE": 87.91, "MAE": 49.12, "R2": 0.8167, "MAPE": 58.26},
    "Linear Regression": {"RMSE": 69.32, "MAE": 36.51, "R2": 0.8834, "MAPE": 52.09},
    "Prophet":           {"RMSE": 75.57, "MAE": 41.48, "R2": 0.8646, "MAPE": 53.87},
    "XGBoost ":        {"RMSE": 66.84, "MAE": 34.53, "R2": 0.8916, "MAPE": 50.23},
}

FEATURE_LABELS = {
    'sales_lag_7':           'Sales 7 Days Ago',
    'sales_lag_14':          'Sales 14 Days Ago',
    'sales_lag_28':          'Sales 28 Days Ago',
    'sales_rolling_mean_7':  '7-Day Rolling Avg',
    'sales_rolling_mean_28': '28-Day Rolling Avg',
    'sales_rolling_std_7':   '7-Day Rolling Std Dev',
    'dayofweek':             'Day of Week',
    'month':                 'Month',
    'quarter':               'Quarter',
    'day':                   'Day of Month',
    'is_weekend':            'Is Weekend',
    'is_month_start':        'Is Month Start',
    'is_month_end':          'Is Month End',
    'onpromotion':           'On Promotion',
    'holiday_impact':        'Holiday Impact (SL)',
    'family_encoded':        'Product Category'
}

PRODUCT_CONFIG = {
    502331:  {'name': 'Bread A',              'perishable': True,  'safety_pct': 0.05},
    564287:  {'name': 'Baked Goods B',        'perishable': True,  'safety_pct': 0.05},
    584028:  {'name': 'Meat Product A',       'perishable': True,  'safety_pct': 0.10},
    903285:  {'name': 'Poultry A',            'perishable': True,  'safety_pct': 0.10},
    1503844: {'name': 'Vegetables A',         'perishable': True,  'safety_pct': 0.10},
    1473474: {'name': 'Vegetables B',         'perishable': True,  'safety_pct': 0.10},
    1695835: {'name': 'Fruit A (Bananas)',    'perishable': True,  'safety_pct': 0.10},
    1427659: {'name': 'Dairy Product',        'perishable': True,  'safety_pct': 0.10},
    1047679: {'name': 'Soft Drinks A',        'perishable': False, 'safety_pct': None},
    364606:  {'name': 'Staples A (Rice)',     'perishable': False, 'safety_pct': None},
    265559:  {'name': 'Staples B (Rice)',     'perishable': False, 'safety_pct': None},
}

Z_SCORES = {90: 1.28, 95: 1.65, 99: 2.33}
PERISHABLE_IDS     = [502331, 564287, 584028, 903285, 1503844, 1473474, 1695835, 1427659]
NON_PERISHABLE_IDS = [1047679, 364606, 265559]

# ============================================================
# DATA LOADING (cached)
# ============================================================

@st.cache_data
def load_xgb_predictions():
    try:
        df = pd.read_csv('../results/xgboost_predictions.csv', parse_dates=['date'])
        if 'predicted' in df.columns and 'predictions' not in df.columns:
            df = df.rename(columns={'predicted': 'predictions'})
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def load_lr_predictions():
    try:
        df = pd.read_csv('../results/linear_regression_predictions.csv', parse_dates=['date'])
        # LR saves columns: date, store_nbr, item_nbr, actual, predictions, error
        for old, new in [('predicted','predictions'),('pred','predictions')]:
            if old in df.columns and 'predictions' not in df.columns:
                df = df.rename(columns={old: new})
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def load_prophet_predictions():
    try:
        df = pd.read_csv('../results/prophet_predictions.csv', parse_dates=['date'])
        for old, new in [('predicted','predictions'),('actuals','actual')]:
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

# Load at startup
xgb_preds     = load_xgb_predictions()
lr_preds      = load_lr_predictions()
prophet_preds = load_prophet_predictions()
ma_preds      = load_ma_predictions()
shap_df       = load_shap_data()
shap_imp_df   = load_shap_importance()
train_data    = load_train_data()

PRED_DATAFRAMES = {
    'XGBoost ':        xgb_preds,
    'Linear Regression': lr_preds,
    'Prophet':           prophet_preds,
    'Moving Average':    ma_preds,
}

# ============================================================
# HELPERS
# ============================================================

def page_header(icon, title, subtitle=""):
    st.markdown(f"""
    <div class="page-header">
        <h1>{icon} {title}</h1>
        {"<p>" + subtitle + "</p>" if subtitle else ""}
    </div>""", unsafe_allow_html=True)

def metric_card(label, value, sub=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>"""

def insight(text):
    st.markdown(f'<div class="insight-box"><p> {text}</p></div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def chart_style(fig, height=380, title=""):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#334155', size=12),
        margin=dict(l=10, r=10, t=45 if title else 20, b=10),
        height=height,
        legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#e8ecf4',
                    borderwidth=1, font=dict(size=11)),
        title=dict(text=title, font=dict(size=13, color='#1e293b'), x=0)
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f1f5f9', showline=True,
                     linecolor='#e2e8f0', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#f1f5f9', showline=True,
                     linecolor='#e2e8f0', zeroline=False)
    return fig

def product_label(pid):
    return f"{pid} – {PRODUCT_NAMES.get(pid, str(pid))}"

def status_badge(mae):
    if mae < 20:   return '<span class="badge-excellent"> Excellent</span>'
    elif mae < 45: return '<span class="badge-good">Moderate</span>'
    else:          return '<span class="badge-hard"> Challenging</span>'

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
<div style="padding:20px 8px 12px 8px; text-align:center;">
    <div style="font-size:30px; margin-bottom:6px;"></div>
    <div style="font-size:14px; font-weight:700; color:#f1f5f9; letter-spacing:-0.2px;">Demand Forecasting</div>
    <div style="font-size:10px; color:#64748b; margin-top:3px; letter-spacing:0.06em;">SRI LANKA RETAIL </div>
</div>
<hr style="border-color:rgba(255,255,255,0.07); margin:8px 0 16px 0;">
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["Forecast View", "SHAP Explanations", "Replenishment", "Analysis"],
    label_visibility="collapsed"
)

st.sidebar.markdown("""
<hr style="border-color:rgba(255,255,255,0.07); margin:20px 0 14px 0;">
<div style="padding:0 4px;">
    <div style="font-size:10px; color:#475569; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px;">System Info</div>
    <div style="font-size:12px; color:#94a3b8; line-height:2.1;">
         Data: 2013–2017<br> Stores: 2<br> Products: 11<br> Best R sq: 89.16%<br> Best MAE: 34.53 units
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAGE 1 — FORECAST VIEW
# ============================================================

if page == "Forecast View":

    page_header( "Forecast View",
                "Click Run Forecast to load all predictions \n  Use filters to explore")

    # ── RUN FORECAST BUTTON ──────────────────────────────
    btn_col, status_col = st.columns([1, 4])

    with btn_col:
        run_clicked = st.button("  Run Forecast", use_container_width=True)

    if 'forecast_ran' not in st.session_state:
        st.session_state.forecast_ran = False
    if run_clicked:
        st.session_state.forecast_ran = True

    with status_col:
        if st.session_state.forecast_ran:
            st.markdown("""
            <div style="background:#f0fdf4; border:1px solid #86efac; border-radius:8px;
                        padding:11px 16px; margin-top:4px; font-size:13px; color:#166534;">
                 <strong>Forecast loaded</strong> · Validation period: 2016-01-02 → 2016-12-31
                · all 11 products · 2 stores · 4 models ready
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#fafafa; border:1px solid #e2e8f0; border-radius:8px;
                        padding:11px 16px; margin-top:4px; font-size:13px; color:#94a3b8;">
                 Click <strong>Run Forecast</strong> above to load predictions for all products and stores
            </div>""", unsafe_allow_html=True)
            st.stop()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── FILTERS ─────────────────────────────────────────
    section(" Filter Results")
    fc1, fc2, fc3, fc4 = st.columns(4)

    with fc1:
        sel_model = st.selectbox("Model", options=list(PRED_DATAFRAMES.keys()), index=0)

    with fc2:
        sel_store = st.selectbox("Store", options=[44, 51], format_func=lambda x: STORE_NAMES[x])

    with fc3:
        categories = ["All Categories"] + sorted(set(PRODUCT_FAMILY.values()))
        sel_cat = st.selectbox("Category", options=categories)

    with fc4:
        if sel_cat == "All Categories":
            prod_opts = sorted(PRODUCT_NAMES.keys())
        else:
            prod_opts = sorted([p for p, f in PRODUCT_FAMILY.items() if f == sel_cat])
        sel_product = st.selectbox("Product", options=prod_opts, format_func=product_label)

    sel_days = st.select_slider(
        "Date range (days of validation period to show)",
        options=[30, 60, 90, 180, 366], value=90,
        format_func=lambda x: f"Last {x} days" if x < 366 else "Full year (2016)")

    # ── GET FILTERED DATA ────────────────────────────────
    pred_df = PRED_DATAFRAMES[sel_model]

    if pred_df.empty:
        st.error(f" No prediction data loaded for **{sel_model}**. "
                 "Check the results CSV files in `../results/`")
        st.stop()

    mask     = (pred_df['store_nbr'] == sel_store) & (pred_df['item_nbr'] == sel_product)
    filtered = pred_df[mask].sort_values('date').tail(sel_days).copy()

    if filtered.empty:
        st.warning(f"No data for {STORE_NAMES[sel_store]} · {product_label(sel_product)}")
        st.stop()

    if 'error' not in filtered.columns:
        filtered['error'] = filtered['actual'] - filtered['predictions']

    # ── KPI CARDS ────────────────────────────────────────
    section(" Key Metrics")
    avg_actual = filtered['actual'].mean()
    avg_pred   = filtered['predictions'].mean()
    avg_mae    = filtered['error'].abs().mean()
    avg_mape   = (filtered['error'].abs() / filtered['actual'].replace(0, np.nan)).mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(metric_card("Avg Actual Sales",  f"{avg_actual:.0f}", "units / day"), unsafe_allow_html=True)
    k2.markdown(metric_card("Avg Predicted",     f"{avg_pred:.0f}",  "units / day"), unsafe_allow_html=True)
    k3.markdown(metric_card("Mean Abs Error",    f"{avg_mae:.1f}",   "units"),       unsafe_allow_html=True)
    k4.markdown(metric_card("MAPE",              f"{avg_mape:.1f}%", "avg % error"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── FORECAST LINE CHART ──────────────────────────────
    section("Actual vs Predicted Sales")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pd.concat([filtered['date'], filtered['date'][::-1]]),
        y=pd.concat([filtered['actual'], filtered['predictions'][::-1]]),
        fill='toself', fillcolor='rgba(124,58,237,0.07)',
        line=dict(color='rgba(0,0,0,0)'), name='Error Band',
        showlegend=True, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=filtered['date'], y=filtered['actual'],
        name='Actual Sales',
        line=dict(color='#1e293b', width=2.5), mode='lines',
        hovertemplate='<b>Actual</b>: %{y:.0f} units<br>%{x|%d %b %Y}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=filtered['date'], y=filtered['predictions'],
        name=f'Predicted ({sel_model})',
        line=dict(color='#7c3aed', width=2.5, dash='dash'), mode='lines',
        hovertemplate='<b>Predicted</b>: %{y:.0f} units<br>%{x|%d %b %Y}<extra></extra>'
    ))
    fig.update_layout(hovermode='x unified', xaxis_title="Date",
                      yaxis_title="Sales (units)", legend=dict(orientation='h', y=1.08, x=0))
    chart_style(fig, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # ── PRODUCT FORECAST TABLE ───────────────────────────
    section("📋 All Products — Forecast Summary  (selected store · selected model)")
    st.caption("Average of last 7 predictions for each product")

    rows_summary = []
    for pid in sorted(PRODUCT_NAMES.keys()):
        sub = pred_df[(pred_df['store_nbr']==sel_store) &
                      (pred_df['item_nbr']==pid)].sort_values('date').tail(7)
        if sub.empty:
            continue
        err = (sub['actual'] - sub['predictions']).abs()
        rows_summary.append({
            'Product':      PRODUCT_NAMES[pid],
            'Category':     PRODUCT_FAMILY.get(pid, ''),
            'Avg Actual':   round(sub['actual'].mean(), 1),
            'Avg Forecast': round(sub['predictions'].mean(), 1),
            'MAE (7-day)':  round(err.mean(), 1),
        })

    if rows_summary:
        st.dataframe(pd.DataFrame(rows_summary), use_container_width=True, hide_index=True,
                     column_config={
                         "Avg Actual":   st.column_config.NumberColumn(format="%.1f"),
                         "Avg Forecast": st.column_config.NumberColumn(format="%.1f"),
                         "MAE (7-day)":  st.column_config.NumberColumn(format="%.1f"),
                     })

# ============================================================
# PAGE 2 — SHAP EXPLANATIONS
# ============================================================

elif page == "SHAP Explanations":

    page_header( "SHAP Explanations",
                "Understand WHY the XGBoost model makes each prediction · "
                )

    if shap_imp_df.empty or shap_df.empty:
        st.error(" SHAP files not found. Check `../results/shap/` directory.")
        st.stop()

    # ── GLOBAL IMPORTANCE ────────────────────────────────
    section("Global Feature Importance  (mean |SHAP value| )")

    top10 = shap_imp_df.head(10).copy()
    top10['label'] = top10['feature'].map(FEATURE_LABELS).fillna(top10['feature'])
    top10['norm']  = top10['mean_abs_shap'] / top10['mean_abs_shap'].max() * 100

    g1, g2 = st.columns([3, 2])

    with g1:
        fig_imp = go.Figure(go.Bar(
            x=top10['mean_abs_shap'], y=top10['label'], orientation='h',
            marker=dict(color=top10['mean_abs_shap'],
                        colorscale=[[0,'#c4b5fd'],[0.5,'#7c3aed'],[1,'#1e293b']],
                        showscale=False),
            text=[f"{v:.2f}" for v in top10['mean_abs_shap']],
            textposition='outside',
            textfont=dict(size=11, family='JetBrains Mono')
        ))
        fig_imp.update_layout(yaxis=dict(autorange='reversed'),
                              xaxis_title="Mean |SHAP Value| (units of impact on prediction)")
        chart_style(fig_imp, height=380, title="")
        st.plotly_chart(fig_imp, use_container_width=True)

    with g2:
        st.markdown("<br>", unsafe_allow_html=True)
        for _, row in top10.iterrows():
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:3px;">
                    <span style="color:#334155; font-weight:500;">{row['label']}</span>
                    <span style="color:#7c3aed; font-family:'JetBrains Mono'; font-weight:600;">{row['mean_abs_shap']:.2f}</span>
                </div>
                <div style="background:#f1f5f9; border-radius:4px; height:6px;">
                    <div style="background:linear-gradient(90deg,#7c3aed,#4f46e5); width:{row['norm']}%; height:6px; border-radius:4px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    insight("The 28-day rolling average dominates all predictions (100% normalised importance). "
            "The model's first question is always: 'What has this product been averaging over the past month?' "
            "Holiday impact ranks #15 — the lag features already capture seasonal patterns implicitly.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── INDIVIDUAL PREDICTION EXPLAINER ─────────────────
    section("🔎 Individual Prediction Explanation")

    e1, e2 = st.columns([1, 2])

    with e1:
        shap_product = st.selectbox("Product", options=sorted(PRODUCT_NAMES.keys()),
                                    format_func=product_label, key="shap_prod")
        shap_store   = st.selectbox("Store", options=[44, 51],
                                    format_func=lambda x: STORE_NAMES[x], key="shap_store")
        example_type = st.selectbox("Show example",
                                    options=["Median prediction",
                                             "Best prediction (lowest error)",
                                             "Worst prediction (highest error)"])

    sh_filt = shap_df[(shap_df['item_nbr']==shap_product) &
                      (shap_df['store_nbr']==shap_store)].copy()

    if sh_filt.empty:
        with e2:
            st.warning("No SHAP data for this combination.")
    else:
        sh_filt['abs_error'] = (sh_filt['actual'] - sh_filt['prediction']).abs()
        if "Best"   in example_type: ex_row = sh_filt.loc[sh_filt['abs_error'].idxmin()]
        elif "Worst" in example_type: ex_row = sh_filt.loc[sh_filt['abs_error'].idxmax()]
        else:                         ex_row = sh_filt.iloc[len(sh_filt)//2]

        base_val = float(ex_row.get('base_value', 148.90))
        actual   = float(ex_row['actual'])
        pred     = float(ex_row['prediction'])
        ex_date  = str(ex_row['date'])[:10]

        shap_feats = []
        for col in shap_df.columns:
            if col.startswith('shap_'):
                feat  = col[5:]
                label = FEATURE_LABELS.get(feat, feat)
                val   = float(ex_row[col])
                shap_feats.append({'label': label, 'shap': val})
        shap_feats = sorted(shap_feats, key=lambda x: abs(x['shap']), reverse=True)[:8]
        max_abs    = max(abs(s['shap']) for s in shap_feats) if shap_feats else 1

        with e2:
            err_color = '#dc2626' if abs(actual - pred) > 40 else '#16a34a'
            st.markdown(f"""
            <div class="white-card">
                <div style="font-size:11px; color:#94a3b8; margin-bottom:3px;">{ex_date}</div>
                <div style="font-size:15px; font-weight:700; color:#1e293b; margin-bottom:14px;">
                    {PRODUCT_NAMES.get(shap_product,'?')} · {STORE_NAMES.get(shap_store,'?')}
                </div>
                <div style="display:flex; gap:28px; margin-bottom:18px; flex-wrap:wrap;">
                    <div>
                        <div class="metric-label">Actual</div>
                        <div style="font-size:20px; font-weight:700; font-family:'JetBrains Mono'; color:#1e293b;">{actual:.0f} units</div>
                    </div>
                    <div>
                        <div class="metric-label">Predicted</div>
                        <div style="font-size:20px; font-weight:700; font-family:'JetBrains Mono'; color:#7c3aed;">{pred:.0f} units</div>
                    </div>
                    <div>
                        <div class="metric-label">Error</div>
                        <div style="font-size:20px; font-weight:700; font-family:'JetBrains Mono'; color:{err_color};">{actual-pred:+.0f} units</div>
                    </div>
                </div>
                <div style="font-size:12px; color:#64748b; margin-bottom:12px; font-weight:600;">
                    BASE PREDICTION: {base_val:.1f} units → each feature pushes it up or down:
                </div>
            """, unsafe_allow_html=True)

            for item in shap_feats:
                sv    = item['shap']
                clr   = "#16a34a" if sv > 0 else "#dc2626"
                arrow = "▲" if sv > 0 else "▼"
                bar   = min(abs(sv) / max_abs * 100, 100)
                align = "left" if sv > 0 else "right"
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                    <div style="width:155px; font-size:12px; color:#334155; font-weight:500; flex-shrink:0;">{item['label']}</div>
                    <div style="flex:1; background:#f1f5f9; border-radius:4px; height:7px; position:relative; overflow:hidden;">
                        <div style="position:absolute; {align}:0; background:{clr}; width:{bar}%; height:7px; border-radius:4px;"></div>
                    </div>
                    <div style="width:70px; text-align:right; font-size:12px; font-family:'JetBrains Mono'; color:{clr}; font-weight:700;">
                        {arrow} {abs(sv):.1f}
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
                <div style="margin-top:14px; padding-top:10px; border-top:2px solid #e8ecf4;
                            font-size:13px; font-weight:700; color:#1e293b;">
                    Final Prediction: {pred:.0f} units
                </div>
            </div>""", unsafe_allow_html=True)

    # ── KEY INSIGHTS ─────────────────────────────────────
    section("💡 Key SHAP Insights")
    i1, i2, i3 = st.columns(3)
    i1.markdown(metric_card("Top Feature",       "28-Day Avg",  "Dominates all predictions"), unsafe_allow_html=True)
    i2.markdown(metric_card("Holiday Impact Rank","#15 / 16",   "Lags capture it implicitly"), unsafe_allow_html=True)
    i3.markdown(metric_card("Computation Time",  "1.9 sec",     "7,346 via TreeExplainer"),   unsafe_allow_html=True)

# ============================================================
# PAGE 3 — REPLENISHMENT CALCULATOR
# ============================================================

elif page == "Replenishment":

    page_header("", "Replenishment Calculator",
                "Convert XGBoost forecasts into optimal inventory orders · "
                )

    if train_data.empty:
        st.error("⚠️ Training data not found at `../data/processed/train_processed.csv`")
        st.stop()
    if xgb_preds.empty:
        st.error("⚠️ XGBoost predictions not found at `../results/xgboost_predictions.csv`")
        st.stop()

   

    # ── INPUTS ───────────────────────────────────────────
    section(" Parameters")
    r1, r2, r3 = st.columns(3)

    with r1:
        service_level = st.selectbox("Service Level",
            options=[90, 95, 99], index=1,
            format_func=lambda x: f"{x}%  (Z = {Z_SCORES[x]})",
            help="90% = tolerate stockout 1 in 10 · 95% = 1 in 20 · 99% = 1 in 100")
    with r2:
        lead_time = st.slider("Lead Time (days)", 1, 14, 3,
                              help="Days between placing order and receiving stock")
    with r3:
        rep_store = st.selectbox("Store", options=[44, 51],
                                 format_func=lambda x: STORE_NAMES[x])

    section(" Product & Current Stock")
    p1, p2 = st.columns([2, 1])

    with p1:
        rep_product = st.selectbox("Product",
            options=sorted(PRODUCT_CONFIG.keys()),
            format_func=lambda x: (
                f"{PRODUCT_CONFIG[x]['name']}  "
                f"({' Perishable' if PRODUCT_CONFIG[x]['perishable'] else ' Non-Perishable'})"
            ))
    with p2:
        current_stock = st.number_input("Current Stock (units)", min_value=0, value=100, step=10)

    st.markdown("<br>", unsafe_allow_html=True)
    calc_btn = st.button("  Calculate Order")

    # ── CALCULATION ───────────────────────────────────────
    if calc_btn:
        cfg = PRODUCT_CONFIG[rep_product]

        # Forecast = average of last 7 XGBoost predictions for this product + store
        filt_preds = xgb_preds[
            (xgb_preds['item_nbr'] == rep_product) &
            (xgb_preds['store_nbr'] == rep_store)
        ].sort_values('date').tail(7)

        if filt_preds.empty:
            st.error("No forecast data for this product/store combination.")
            st.stop()

        forecast_val = filt_preds['predictions'].mean()

        # Std dev from training data (2013-2015)
        filt_train = train_data[
            (train_data['item_nbr'] == rep_product) &
            (train_data['store_nbr'] == rep_store)
        ]
        demand_std = filt_train['unit_sales'].std() if not filt_train.empty else 1.0
        z = Z_SCORES[service_level]

        # Safety stock calculation
        if cfg['perishable']:
            safety_stock = forecast_val * cfg['safety_pct']
            method_str   = f"Perishable: {cfg['safety_pct']*100:.0f}% × {forecast_val:.1f}"
            formula_comparison = z * demand_std * math.sqrt(lead_time)
        else:
            safety_stock = z * demand_std * math.sqrt(lead_time)
            method_str   = f"Z({z}) × σ({demand_std:.1f}) × √{lead_time}"
            formula_comparison = None

        reorder_point = (forecast_val * lead_time) + safety_stock
        order_quantity = max(0.0, reorder_point - current_stock)

        if current_stock == 0:           status = "CRITICAL"
        elif order_quantity > 0:         status = "ORDER REQUIRED"
        else:                            status = "STOCK SUFFICIENT"

        # ── RESULT CARDS ─────────────────────────────────
        section("Calculation Results")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.markdown(metric_card(" Forecast",      f"{forecast_val:.0f}",  "units · 7-day avg"),    unsafe_allow_html=True)
        rc2.markdown(metric_card(" Safety Stock",  f"{safety_stock:.0f}",  method_str),             unsafe_allow_html=True)
        rc3.markdown(metric_card(" Reorder Point", f"{reorder_point:.0f}", "forecast + safety"),    unsafe_allow_html=True)
        rc4.markdown(metric_card(" Current Stock", f"{current_stock}",     "units on hand"),        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ORDER RECOMMENDATION BOX ──────────────────────
        if status == "CRITICAL":
            st.markdown(f"""
            <div class="order-critical">
                <div style="font-size:13px; font-weight:700; color:#c2410c; margin-bottom:6px;"> CRITICAL — OUT OF STOCK</div>
                <div class="order-qty-num" style="color:#c2410c;">{order_quantity:.0f}</div>
                <div style="font-size:15px; font-weight:600; color:#9a3412; margin-top:8px;">
                    units to order immediately · {cfg['name']} · {STORE_NAMES[rep_store]}
                </div>
            </div>""", unsafe_allow_html=True)

        elif status == "ORDER REQUIRED":
            st.markdown(f"""
            <div class="order-required">
                <div style="font-size:13px; font-weight:700; color:#b91c1c; margin-bottom:6px;"> ORDER REQUIRED</div>
                <div class="order-qty-num" style="color:#dc2626;">{order_quantity:.0f}</div>
                <div style="font-size:15px; font-weight:600; color:#991b1b; margin-top:8px;">
                    units to order · {cfg['name']} · {STORE_NAMES[rep_store]}
                </div>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="order-sufficient">
                <div class="order-qty-num" style="color:#16a34a;"></div>
                <div style="font-size:18px; font-weight:700; color:#15803d; margin-top:10px;">STOCK SUFFICIENT</div>
                <div style="font-size:13px; color:#166534; margin-top:6px;">
                    Current stock ({current_stock} units) covers forecast ({forecast_val:.0f}) + safety buffer ({safety_stock:.0f}).
                    No order needed.
                </div>
            </div>""", unsafe_allow_html=True)

        # ── CALCULATION STEPS ─────────────────────────────
        with st.expander("🔍 Calculation Steps (academic reference)"):
            if cfg['perishable']:
                st.markdown(f"""
**Product Type:**  Perishable — percentage-based safety stock to minimise waste

**Step 1 — Forecast**
Average of last 7 XGBoost predictions for *{cfg['name']}* @ {STORE_NAMES[rep_store]}
= **{forecast_val:.1f} units**

**Step 2 — Safety Stock (Perishable)**
`Safety Stock = Forecast × {cfg['safety_pct']*100:.0f}%`
= {forecast_val:.1f} × {cfg['safety_pct']} = **{safety_stock:.1f} units**

*Why percentage instead of Z × σ × √L?*
The formula would give {formula_comparison:.0f} units — far too much buffer for a perishable item.
Unsold stock expires → financial waste. A small % is the industry-standard approach (Silver et al., 1998).

**Step 3 — Reorder Point**
`Reorder Point = (Forecast × Lead Time) + Safety Stock = ({forecast_val:.1f} × {lead_time}) + {safety_stock:.1f} = {reorder_point:.1f} units`

**Step 4 — Order Quantity**
`Order = max(0, {reorder_point:.1f} − {current_stock}) = {order_quantity:.1f} units`
                """)
            else:
                st.markdown(f"""
**Product Type:** 🥫 Non-Perishable — scientific Z × σ × √L formula

**Step 1 — Forecast**
Average of last 7 XGBoost predictions for *{cfg['name']}* @ {STORE_NAMES[rep_store]}
= **{forecast_val:.1f} units**

**Step 2 — Safety Stock (Non-Perishable)**
`Safety Stock = Z × σ × √L`

- **Z = {z}** → service level {service_level}% (stockout tolerated 1 in {round(1/(1-service_level/100))} times)
- **σ = {demand_std:.2f}** → std dev of historical daily sales (train data 2013–2015)
- **L = {lead_time}** days lead time, √L = {math.sqrt(lead_time):.3f}

= {z} × {demand_std:.2f} × {math.sqrt(lead_time):.3f} = **{safety_stock:.1f} units**

*Why formula for non-perishables?* They don't expire so more stock = no waste.
The formula automatically gives bigger buffers to more variable products.

**Step 3 — Reorder Point**
`({forecast_val:.1f} × {lead_time}) + {safety_stock:.1f} = {reorder_point:.1f} units`

**Step 4 — Order Quantity**
`max(0, {reorder_point:.1f} − {current_stock}) = {order_quantity:.1f} units`

*Reference: Silver, Pyke & Peterson (1998); Chopra & Meindl (2016)*
                """)

        # ── ALL PRODUCTS SUMMARY ──────────────────────────
        section("📋 All Products Summary  (same store & parameters · assuming 100 units current stock)")

        all_rows = []
        for pid, pcfg in PRODUCT_CONFIG.items():
            fp = xgb_preds[(xgb_preds['item_nbr']==pid) &
                           (xgb_preds['store_nbr']==rep_store)].sort_values('date').tail(7)
            if fp.empty:
                continue
            fc  = fp['predictions'].mean()
            tr  = train_data[(train_data['item_nbr']==pid) & (train_data['store_nbr']==rep_store)]
            std = tr['unit_sales'].std() if not tr.empty else 1.0
            ss  = fc * pcfg['safety_pct'] if pcfg['perishable'] else z * std * math.sqrt(lead_time)
            rp = (fc * lead_time) + ss
            oq  = max(0.0, rp - 100)
            all_rows.append({
                'Product':       pcfg['name'],
                'Type':          'Perishable' if pcfg['perishable'] else 'Non-Perishable',
                'Forecast':      round(fc, 0),
                'Safety Stock':  round(ss, 0),
                'Reorder Point': round(rp, 0),
                'Order Qty*':    round(oq, 0),
                'Status':        ' Order' if oq > 0 else 'OK!!'
            })

        if all_rows:
            st.caption("*Assuming 100 units current stock for all products in this summary view.")
            st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)

# ============================================================
# PAGE 4 — ANALYSIS (tabbed: Model Comparison + Product Perf + Category)
# ============================================================

elif page == "Analysis":

    page_header("📊", "Analysis",
                "Model comparison · product performance · category accuracy breakdown")

    tab1, tab2, tab3 = st.tabs([" Model Comparison", " Product Performance", " Category Breakdown"])

    # ── TAB 1: MODEL COMPARISON ───────────────────────────
    with tab1:
        section("Performance Metrics — Validation Set (2016)")

        rows_html = ""
        for mname, m in MODEL_METRICS.items():
            is_best = "best-row" if "XGBoost" in mname else ""
            rows_html += f"""
            <tr class="{is_best}">
                <td><strong>{mname}</strong></td>
                <td style="font-family:'JetBrains Mono',monospace; text-align:right;">{m['RMSE']:.2f}</td>
                <td style="font-family:'JetBrains Mono',monospace; text-align:right;">{m['MAE']:.2f}</td>
                <td style="font-family:'JetBrains Mono',monospace; text-align:right;">{m['R2']*100:.2f}%</td>
                <td style="font-family:'JetBrains Mono',monospace; text-align:right;">{m['MAPE']:.2f}%</td>
            </tr>"""

        st.markdown(f"""
        <div style="border-radius:12px; overflow:hidden; border:1px solid #e8ecf4; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
        <table class="styled-table">
            <thead><tr>
                <th>Model</th>
                <th style="text-align:right">RMSE ↓</th>
                <th style="text-align:right">MAE ↓</th>
                <th style="text-align:right">R² ↑</th>
                <th style="text-align:right">MAPE ↓</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table></div>""", unsafe_allow_html=True)

        insight("XGBoost achieves the best score across all four metrics — 89.16% R², "
                "a +7.49 percentage-point improvement over the Moving Average baseline.")

        st.markdown("<br>", unsafe_allow_html=True)

        mc1, mc2 = st.columns(2)
        names   = list(MODEL_METRICS.keys())
        r2vals  = [m['R2']*100  for m in MODEL_METRICS.values()]
        maevals = [m['MAE']     for m in MODEL_METRICS.values()]
        colors  = ['#7c3aed' if 'XGBoost' in n else '#c4b5fd' for n in names]

        with mc1:
            fig_r2 = go.Figure(go.Bar(
                x=names, y=r2vals, marker_color=colors,
                text=[f"{v:.2f}%" for v in r2vals], textposition='outside',
                textfont=dict(size=11, family='JetBrains Mono')
            ))
            fig_r2.update_layout(yaxis=dict(range=[75, 94], title="R² (%)"), xaxis_title="")
            chart_style(fig_r2, height=320, title="R² Score — Higher is Better")
            st.plotly_chart(fig_r2, use_container_width=True)

        with mc2:
            fig_mae = go.Figure(go.Bar(
                x=names, y=maevals, marker_color=colors,
                text=[f"{v:.2f}" for v in maevals], textposition='outside',
                textfont=dict(size=11, family='JetBrains Mono')
            ))
            fig_mae.update_layout(yaxis=dict(title="MAE (units)"), xaxis_title="")
            chart_style(fig_mae, height=320, title="MAE — Lower is Better")
            st.plotly_chart(fig_mae, use_container_width=True)

        section("Why XGBoost Won")
        st.markdown("""
        <div class="white-card">
        <ul style="color:#334155; font-size:13px; line-height:2.2; margin:0; padding-left:18px;">
            <li><strong>Non-linear interactions</strong> — captures promotion × weekend effects that linear models cannot represent</li>
            <li><strong>Lag features</strong> — 7/14/28-day history is the dominant signal; trees learn complex lag relationships automatically</li>
            <li><strong>Sri Lankan holiday encoding</strong> — <code>holiday_impact</code> feature captures Poya day meat restrictions and New Year surges</li>
            <li><strong>No feature scaling required</strong> — tree-based splits handle mixed magnitudes (unlike Linear Regression)</li>
            <li><strong>SHAP compatible</strong> — TreeExplainer provides fast, exact explanations for every prediction</li>
            <li><strong>Robust to outliers</strong> — gradient boosting is less sensitive to extreme sales spikes than linear models</li>
        </ul>
        </div>""", unsafe_allow_html=True)

    # ── TAB 2: PRODUCT PERFORMANCE ────────────────────────
    with tab2:
        if xgb_preds.empty:
            st.error("XGBoost predictions not loaded.")
        else:
            pf = xgb_preds.copy()
            pf['abs_error'] = pf['error'].abs()
            pf['pct_error'] = (pf['abs_error'] / pf['actual'].replace(0, np.nan)) * 100

            prod_stats = pf.groupby('item_nbr').agg(
                avg_actual=('actual','mean'),
                avg_pred=('predictions','mean'),
                mae=('abs_error','mean'),
                mape=('pct_error','mean')
            ).reset_index().sort_values('mae')

            prod_stats['name']       = prod_stats['item_nbr'].map(PRODUCT_NAMES)
            prod_stats['family']     = prod_stats['item_nbr'].map(PRODUCT_FAMILY)
            prod_stats['perishable'] = prod_stats['item_nbr'].apply(
                lambda x: "🥬 Perishable" if x in PERISHABLE_IDS else "🥫 Non-Perishable")

            section("All Products — Accuracy Ranking (best → worst MAE)")

            medals = {0:"🥇", 1:"🥈", 2:"🥉"}
            trows  = ""
            for i, (_, row) in enumerate(prod_stats.iterrows()):
                medal = medals.get(i, "")
                trows += f"""
                <tr>
                    <td>{medal} <strong>{row['name']}</strong></td>
                    <td style="font-size:11px; color:#64748b;">{row['family']}</td>
                    <td style="font-size:12px;">{row['perishable']}</td>
                    <td style="font-family:'JetBrains Mono'; text-align:right;">{row['avg_actual']:.0f}</td>
                    <td style="font-family:'JetBrains Mono'; text-align:right;">{row['mae']:.2f}</td>
                    <td style="font-family:'JetBrains Mono'; text-align:right;">{row['mape']:.1f}%</td>
                    <td>{status_badge(row['mae'])}</td>
                </tr>"""

            st.markdown(f"""
            <div style="border-radius:12px; overflow:hidden; border:1px solid #e8ecf4; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
            <table class="styled-table">
                <thead><tr>
                    <th>Product</th><th>Category</th><th>Type</th>
                    <th style="text-align:right">Avg Sales</th>
                    <th style="text-align:right">MAE ↑</th>
                    <th style="text-align:right">MAPE</th>
                    <th>Status</th>
                </tr></thead>
                <tbody>{trows}</tbody>
            </table></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            pp1, pp2 = st.columns(2)
            with pp1:
                fig_pp = go.Figure(go.Bar(
                    x=prod_stats['mae'], y=prod_stats['name'], orientation='h',
                    marker=dict(color=prod_stats['mae'],
                                colorscale=[[0,'#7c3aed'],[0.5,'#f59e0b'],[1,'#dc2626']],
                                showscale=False),
                    text=[f"{v:.1f}" for v in prod_stats['mae']],
                    textposition='outside', textfont=dict(size=10, family='JetBrains Mono')
                ))
                fig_pp.update_layout(yaxis=dict(autorange='reversed'), xaxis_title="MAE (units)")
                chart_style(fig_pp, height=380, title="MAE by Product")
                st.plotly_chart(fig_pp, use_container_width=True)

            with pp2:
                store_stats = pf.groupby('store_nbr').agg(
                    mae=('abs_error','mean'), mape=('pct_error','mean'), avg=('actual','mean')
                ).reset_index()
                store_stats['name'] = store_stats['store_nbr'].map(STORE_NAMES)

                fig_st = go.Figure()
                for idx, (_, sr) in enumerate(store_stats.iterrows()):
                    fig_st.add_trace(go.Bar(
                        name=sr['name'], x=['MAE (units)', 'MAPE (%)'],
                        y=[sr['mae'], sr['mape']],
                        marker_color=['#7c3aed','#4f46e5'][idx],
                        text=[f"{sr['mae']:.1f}", f"{sr['mape']:.1f}%"],
                        textposition='outside'
                    ))
                fig_st.update_layout(barmode='group', legend=dict(orientation='h', y=1.12))
                chart_style(fig_st, height=280, title="Store Colombo vs Store Gampaha")
                st.plotly_chart(fig_st, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                for _, sr in store_stats.iterrows():
                    st.markdown(metric_card(
                        sr['name'], f"MAE {sr['mae']:.1f}",
                        f"Avg {sr['avg']:.0f} units/day · MAPE {sr['mape']:.1f}%"
                    ), unsafe_allow_html=True)
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            insight("Store Gampaha achieves ~45% lower MAE than Store Colombo. "
                    "Higher-volume urban stores have naturally larger absolute errors, "
                    "though percentage error is similar across both locations.")

    # ── TAB 3: CATEGORY BREAKDOWN ─────────────────────────
    with tab3:
        if xgb_preds.empty:
            st.error("XGBoost predictions not loaded.")
        else:
            pf2 = xgb_preds.copy()
            pf2['abs_error'] = pf2['error'].abs()
            pf2['family']    = pf2['item_nbr'].map(PRODUCT_FAMILY)

            cat_stats = pf2.groupby('family').agg(
                mae=('abs_error','mean'),
                avg_sales=('actual','mean'),
                count=('actual','count')
            ).reset_index().sort_values('mae')

            section("Average MAE by Product Category")
            fig_cat = go.Figure(go.Bar(
                x=cat_stats['family'], y=cat_stats['mae'],
                marker_color='#7c3aed',
                text=[f"{v:.1f}" for v in cat_stats['mae']],
                textposition='outside', textfont=dict(size=12, family='JetBrains Mono')
            ))
            fig_cat.update_layout(xaxis_title="Category", yaxis_title="Average MAE (units)")
            chart_style(fig_cat, height=340, title="Category-Level Forecast Accuracy")
            st.plotly_chart(fig_cat, use_container_width=True)

            section("Category Summary Table")
            cat_disp = cat_stats.rename(columns={
                'family':'Category', 'mae':'Avg MAE',
                'avg_sales':'Avg Daily Sales', 'count':'Records'
            })
            cat_disp['Avg MAE']         = cat_disp['Avg MAE'].round(2)
            cat_disp['Avg Daily Sales'] = cat_disp['Avg Daily Sales'].round(1)
            st.dataframe(cat_disp, use_container_width=True, hide_index=True)

            insight("BREAD/BAKERY and GROCERY categories have the lowest absolute error because "
                    "their demand is stable and predictable. PRODUCE (vegetables) is most "
                    "challenging due to high daily variance and sensitivity to seasons and holidays.")