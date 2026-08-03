"""
MRP Planning Intelligence
=========================
Built by Rutwik Satish | MS Engineering Management + Graduate Certificate in Supply Chain
Northeastern University

A combined demand sensing and lot sizing optimization tool for manufacturing planners.
Takes raw historical usage data, classifies the demand pattern, generates a statistical
forecast, feeds it into the MRP lot sizing engine, and outputs a buyer-ready order schedule
with full cost justification — in under 3 minutes per component.

STAGE 1 — DEMAND INTELLIGENCE (from DemandIQ):
  CV classification (Vandeput, DFBP Ch. 13)
  Best forecasting method via Vandeput Score ranking (DFBP Ch. 8-9)
  Forward gross requirements with 80%/95% prediction intervals (FPP3 Ch. 5.5)

STAGE 2 — LOT SIZING OPTIMIZATION (from MRP Optimizer):
  6 APICS-standard algorithms (Jacobs, Berry, Whybark & Vollmann, 2011)
  Full MRP record: GR, SR, PAB, NR, POR, POL
  Savings vs current ERP method

SOURCES:
  FPP3  — Hyndman & Athanasopoulos, Forecasting: Principles and Practice, 3rd ed. (OTexts, 2021)
  DFBP  — Vandeput, Demand Forecasting Best Practices (Manning, 2023)
  FDPF  — Jain, Fundamentals of Demand Planning & Forecasting (Graceway, 2020)
  JB    — Jacobs, Berry, Whybark & Vollmann, Manufacturing Planning and Control (2011)
  SM    — Silver & Meal (1973), Management Science
  WW    — Wagner & Whitin (1958), Management Science
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from itertools import product
import streamlit as st

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MRP Planning Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────

def inject_css():
    css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #0a0f1a !important;
    color: #e2e8f0 !important;
}
[data-testid="stMain"], [data-testid="stAppViewContainer"],
[data-testid="block-container"], .main {
    background-color: #0a0f1a !important;
}
[data-testid="stSidebar"] {
    background-color: #060c16 !important;
    border-right: 1px solid #1e2d45 !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] hr { border-color: #1e2d45 !important; }
[data-testid="stSidebar"] .stButton button {
    background: #1B6EF3 !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 500 !important;
}
.main .block-container {
    padding-top: 1rem !important; padding-bottom: 2rem !important;
    max-width: 1400px; background-color: #0a0f1a !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important; border-bottom: 2px solid #1e2d45 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem !important; font-weight: 600 !important;
    padding: 0.55rem 1.1rem !important; color: #475569 !important;
    background: transparent !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #3b82f6 !important; border-bottom: 2px solid #3b82f6 !important;
}
.stButton > button {
    border-radius: 8px !important; font-weight: 500 !important;
    background: #1e2d45 !important; color: #e2e8f0 !important;
    border: 1px solid #2d4a6e !important;
}
.stButton > button:hover {
    background: #1B6EF3 !important; border-color: #1B6EF3 !important; color: white !important;
}
.stDownloadButton button {
    background: #1e2d45 !important; color: #e2e8f0 !important;
    border: 1px solid #2d4a6e !important; border-radius: 8px !important;
}
.stDownloadButton button:hover {
    background: #1B6EF3 !important; border-color: #1B6EF3 !important; color: white !important;
}
p, div, span, label, h1, h2, h3, h4 { color: #e2e8f0 !important; }
[data-testid="stMarkdownContainer"] p { color: #cbd5e1 !important; }
div[data-baseweb="select"] > div {
    background: #111827 !important; border-color: #1e2d45 !important; color: #e2e8f0 !important;
}
</style>
"""
    st.html(css)

inject_css()

# ── COLOUR TOKENS ─────────────────────────────────────────────────────────────

BLUE   = "#3b82f6"
GREEN  = "#22c55e"
RED    = "#f87171"
AMBER  = "#fbbf24"
ORANGE = "#fb923c"
NAVY   = "#060c16"
BG     = "#0a0f1a"
CARD   = "#111827"
BORDER = "#1e2d45"
T_PRI  = "#f1f5f9"
T_SEC  = "#94a3b8"
T_MUT  = "#475569"

CHART = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD,
    font=dict(family="Inter", size=12, color=T_SEC),
    margin=dict(l=40, r=20, t=40, b=40), hovermode="x unified",
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, color=T_SEC),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, color=T_SEC),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1,
                font=dict(color=T_SEC))
)

METHOD_COLORS = {
    "Lot-for-Lot": "#6366f1", "EOQ": "#0ea5e9",
    "Period Order Qty": "#8b5cf6", "Part Period Bal.": "#fbbf24",
    "Silver-Meal": "#22c55e", "Wagner-Whitin": "#f87171",
}

# ── HTML HELPERS ──────────────────────────────────────────────────────────────

def card(title, value, sub="", bc=None, vc=None):
    bc = bc or BLUE; vc = vc or T_PRI
    return f"""<div style="background:{CARD};border:1px solid {BORDER};border-left:4px solid {bc};
border-radius:10px;padding:1rem 1.25rem;margin-bottom:0.75rem;">
<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:{T_MUT};margin-bottom:0.2rem;">{title}</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:1.5rem;font-weight:600;color:{vc};line-height:1.2;">{value}</div>
{"<div style='font-size:0.75rem;color:"+T_SEC+";margin-top:0.2rem;'>"+sub+"</div>" if sub else ""}
</div>"""

def section(text):
    st.markdown(f"""<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;
letter-spacing:0.1em;color:{T_MUT};margin:1.5rem 0 0.6rem 0;padding-bottom:0.35rem;
border-bottom:1px solid {BORDER};">{text}</div>""", unsafe_allow_html=True)

def insight(text):
    st.markdown(f"""<div style="background:#0f1e3a;border-left:3px solid {BLUE};
padding:0.5rem 0.85rem;border-radius:0 6px 6px 0;font-size:0.8rem;color:#93c5fd;
margin-top:0.4rem;">{text}</div>""", unsafe_allow_html=True)

def cite(text):
    st.markdown(f'<div style="font-size:0.68rem;color:{T_MUT};font-style:italic;margin-top:0.25rem;">{text}</div>', unsafe_allow_html=True)

def alert(text, kind="info"):
    colors = {
        "danger":  ("#2d0f0f","#7f1d1d","#fca5a5"),
        "warning": ("#2a1f07","#78350f","#fcd34d"),
        "success": ("#0a2218","#14532d","#86efac"),
        "info":    ("#0d1b3e","#1e3a8a","#93c5fd"),
    }
    bg, border, fg = colors.get(kind, colors["info"])
    st.markdown(f"""<div style="background:{bg};border:1px solid {border};border-radius:8px;
padding:0.7rem 1rem;color:{fg};font-weight:500;font-size:0.85rem;margin:0.5rem 0;">{text}</div>""",
                unsafe_allow_html=True)

def stage_badge(n, label, active=True):
    bg = BLUE if active else "#1e2d45"
    tc = "#fff" if active else T_MUT
    return f"""<div style="display:inline-flex;align-items:center;gap:0.5rem;
background:{bg};border-radius:8px;padding:0.4rem 0.9rem;margin-right:0.5rem;">
<span style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;font-weight:700;
color:{tc};">{n}</span>
<span style="font-size:0.8rem;font-weight:600;color:{tc};">{label}</span>
</div>"""

# ── SAMPLE DATA ───────────────────────────────────────────────────────────────

SAMPLE_HISTORY = pd.DataFrame({
    "Period": list(range(1, 25)),
    "SKU": ["Sprocket-7A"] * 24,
    "Usage": [
        142, 158, 135, 190, 175, 162, 148, 220, 205, 168, 155, 182,
        150, 165, 145, 198, 185, 170, 152, 235, 215, 172, 160, 195
    ]
})

# ── DEMAND INTELLIGENCE FUNCTIONS (from DemandIQ) ────────────────────────────

def compute_profile(demand):
    n = len(demand)
    total = float(np.sum(demand))
    mean = total / n
    std = float(np.std(demand, ddof=1)) if n > 1 else 0.0
    cv = std / mean if mean > 0 else 0.0
    return {"n": n, "total": total, "mean": mean, "std": std, "cv": cv}


def classify_cv(cv):
    if cv < 0.2:
        return {"label": "X — Stable", "description": "Statistical forecasting is highly reliable",
                "color": GREEN, "category": "X",
                "recommended_lot": "Silver-Meal or EOQ",
                "recommended_fc": "SES"}
    elif cv <= 0.5:
        return {"label": "Y — Variable", "description": "Use adaptive methods with caution",
                "color": AMBER, "category": "Y",
                "recommended_lot": "Silver-Meal or Part Period Balancing",
                "recommended_fc": "Holt Linear"}
    else:
        return {"label": "Z — Lumpy", "description": "Lot-for-Lot recommended; statistical methods unreliable",
                "color": RED, "category": "Z",
                "recommended_lot": "Lot-for-Lot",
                "recommended_fc": "SES"}


@st.cache_data
def generate_forecast(demand_tuple, method_key, horizon, seasonal_periods=12):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
    arr = np.array(demand_tuple, dtype=float)
    n = len(arr)
    point_fc = np.full(horizon, np.nan)
    rmse_proxy = None
    try:
        if method_key == "SES":
            model = SimpleExpSmoothing(arr).fit(optimized=True)
            point_fc = np.array(model.forecast(horizon))
            rmse_proxy = float(np.sqrt(np.mean(np.array(model.resid)**2)))
        elif method_key == "Holt Linear":
            model = ExponentialSmoothing(arr, trend="add", seasonal=None,
                                         initialization_method="heuristic").fit(optimized=True)
            point_fc = np.array(model.forecast(horizon))
            rmse_proxy = float(np.sqrt(np.mean(np.array(model.resid)**2)))
        elif method_key == "Holt-Winters" and n >= 2 * seasonal_periods:
            model = ExponentialSmoothing(arr, trend="add", seasonal="add",
                                         seasonal_periods=seasonal_periods,
                                         initialization_method="heuristic").fit(optimized=True)
            point_fc = np.array(model.forecast(horizon))
            rmse_proxy = float(np.sqrt(np.mean(np.array(model.resid)**2)))
        else:
            model = SimpleExpSmoothing(arr).fit(optimized=True)
            point_fc = np.array(model.forecast(horizon))
            rmse_proxy = float(np.sqrt(np.mean(np.array(model.resid)**2)))
    except Exception:
        point_fc = np.full(horizon, float(np.mean(arr)))
        rmse_proxy = float(np.std(arr))
    if not rmse_proxy or rmse_proxy == 0:
        rmse_proxy = float(np.std(arr)) or 1.0
    h_arr = np.arange(1, horizon + 1, dtype=float)
    sigma_h = rmse_proxy * np.sqrt(h_arr)
    return {
        "point": np.maximum(0, point_fc),
        "lower_80": np.maximum(0, point_fc - 1.282 * sigma_h),
        "upper_80": point_fc + 1.282 * sigma_h,
        "rmse": rmse_proxy
    }

# ── LOT SIZING ALGORITHMS (from MRP Optimizer) ────────────────────────────────

def compute_mrp_record(gross_req, sched_receipts, initial_inventory, lot_sizes, lead_time):
    n = len(gross_req)
    pab = [0.0] * n
    nr  = [0.0] * n
    pol = [0.0] * n
    for t in range(n):
        prev_pab = initial_inventory if t == 0 else pab[t - 1]
        nr[t]  = max(0.0, gross_req[t] - prev_pab - sched_receipts[t])
        pab[t] = prev_pab + sched_receipts[t] + lot_sizes[t] - gross_req[t]
    for t in range(n):
        release_period = t - lead_time
        if release_period >= 0:
            pol[release_period] = lot_sizes[t]
    periods = [f"P{i+1}" for i in range(n)]
    return pd.DataFrame({
        "Period": periods,
        "Gross Req": [round(v) for v in gross_req],
        "Sched Receipts": [round(v) for v in sched_receipts],
        "Proj Available": [round(v) for v in pab],
        "Net Req": [round(v) for v in nr],
        "Planned Receipts": [round(v) for v in lot_sizes],
        "Order Releases": [round(v) for v in pol],
    })


def compute_costs(lot_sizes, gross_req, sched_receipts, initial_inventory,
                  ordering_cost, h_per_period):
    num_orders = sum(1 for q in lot_sizes if q > 0)
    total_ordering = num_orders * ordering_cost
    pab = 0.0
    total_holding = 0.0
    for t in range(len(gross_req)):
        pab = pab + sched_receipts[t] + lot_sizes[t] - gross_req[t]
        total_holding += max(0.0, pab) * h_per_period
    return total_ordering, total_holding, total_ordering + total_holding


def lot_for_lot(gross_req, sched_receipts, initial_inventory, moq=1):
    n = len(gross_req)
    orders = [0.0] * n
    pab = initial_inventory
    for t in range(n):
        nr = max(0.0, gross_req[t] - pab - sched_receipts[t])
        if nr > 0:
            orders[t] = max(moq, math.ceil(nr / moq) * moq)
        pab = pab + sched_receipts[t] + orders[t] - gross_req[t]
    return orders


def eoq_method(gross_req, sched_receipts, initial_inventory,
               ordering_cost, h_per_period, moq=1):
    D = sum(gross_req)
    if h_per_period <= 0 or D <= 0:
        return lot_for_lot(gross_req, sched_receipts, initial_inventory, moq)
    eoq = max(moq, math.ceil(math.sqrt(2 * D * ordering_cost / h_per_period) / moq) * moq)
    n = len(gross_req)
    orders = [0.0] * n
    pab = initial_inventory
    for t in range(n):
        nr = max(0.0, gross_req[t] - pab - sched_receipts[t])
        if nr > 0:
            orders[t] = max(eoq, math.ceil(nr / eoq) * eoq)
        pab = pab + sched_receipts[t] + orders[t] - gross_req[t]
    return orders


def period_order_qty(gross_req, sched_receipts, initial_inventory,
                     ordering_cost, h_per_period, moq=1):
    D = sum(gross_req)
    n = len(gross_req)
    avg = D / n if n > 0 else 1
    if h_per_period <= 0 or avg <= 0:
        return lot_for_lot(gross_req, sched_receipts, initial_inventory, moq)
    eoq = math.sqrt(2 * D * ordering_cost / h_per_period)
    P = max(1, round(eoq / avg))
    orders = [0.0] * n
    pab = initial_inventory
    t = 0
    while t < n:
        nr = max(0.0, gross_req[t] - pab - sched_receipts[t])
        if nr > 0:
            qty = sum(gross_req[t:t + P]) - pab - sched_receipts[t]
            qty = max(moq, math.ceil(max(qty, nr) / moq) * moq)
            orders[t] = qty
            pab = pab + sched_receipts[t] + orders[t] - gross_req[t]
            t += 1
        else:
            pab = pab + sched_receipts[t] - gross_req[t]
            t += 1
    return orders


def part_period_balancing(gross_req, sched_receipts, initial_inventory,
                           ordering_cost, h_per_period, moq=1):
    if h_per_period <= 0:
        return lot_for_lot(gross_req, sched_receipts, initial_inventory, moq)
    epp = ordering_cost / h_per_period
    n = len(gross_req)
    orders = [0.0] * n
    pab = initial_inventory
    t = 0
    while t < n:
        nr = max(0.0, gross_req[t] - pab - sched_receipts[t])
        if nr <= 0:
            pab = pab + sched_receipts[t] - gross_req[t]
            t += 1
            continue
        cum_pp = 0.0
        cum_qty = gross_req[t]
        j = t + 1
        while j < n:
            additional_pp = (j - t) * gross_req[j]
            if abs(cum_pp + additional_pp - epp) < abs(cum_pp - epp):
                cum_pp += additional_pp
                cum_qty += gross_req[j]
                j += 1
            else:
                break
        qty = max(moq, math.ceil(max(cum_qty, nr) / moq) * moq)
        orders[t] = qty
        pab = pab + sched_receipts[t] + orders[t] - gross_req[t]
        t = j if j > t + 1 else t + 1
    return orders


def silver_meal(gross_req, sched_receipts, initial_inventory,
                ordering_cost, h_per_period, moq=1):
    if h_per_period <= 0:
        return lot_for_lot(gross_req, sched_receipts, initial_inventory, moq)
    n = len(gross_req)
    orders = [0.0] * n
    pab = initial_inventory
    t = 0
    while t < n:
        nr = max(0.0, gross_req[t] - pab - sched_receipts[t])
        if nr <= 0:
            pab = pab + sched_receipts[t] - gross_req[t]
            t += 1
            continue
        best_T = 1
        best_cost = ordering_cost
        cum_hold = 0.0
        for j in range(t + 1, n):
            cum_hold += (j - t) * h_per_period * gross_req[j]
            cost_T = (ordering_cost + cum_hold) / (j - t + 1)
            if cost_T < best_cost:
                best_cost = cost_T
                best_T = j - t + 1
            else:
                break
        qty = sum(gross_req[t:t + best_T])
        qty = max(0.0, qty - pab - sched_receipts[t])
        qty = max(moq, math.ceil(qty / moq) * moq) if qty > 0 else 0.0
        orders[t] = qty
        pab = pab + sched_receipts[t] + orders[t] - gross_req[t]
        t += best_T
    return orders


def wagner_whitin(gross_req, sched_receipts, initial_inventory,
                  ordering_cost, h_per_period, moq=1):
    if h_per_period <= 0:
        return lot_for_lot(gross_req, sched_receipts, initial_inventory, moq)
    n = len(gross_req)
    adj = [max(0.0, gross_req[t] - (initial_inventory if t == 0 else 0) - sched_receipts[t])
           for t in range(n)]
    INF = float('inf')
    f = [INF] * (n + 1)
    split = [0] * (n + 1)
    f[n] = 0.0
    for t in range(n - 1, -1, -1):
        if adj[t] == 0 and all(adj[k] == 0 for k in range(t, n)):
            f[t] = 0.0; split[t] = n; continue
        for j in range(t, n):
            hold = sum((k - t) * h_per_period * adj[k] for k in range(t + 1, j + 1))
            cost = ordering_cost + hold + f[j + 1]
            if cost < f[t]:
                f[t] = cost; split[t] = j + 1
    orders = [0.0] * n
    t = 0
    while t < n:
        if adj[t] > 0 or (t == 0 and initial_inventory == 0 and sched_receipts[0] == 0):
            end = split[t]
            qty = sum(adj[t:end])
            qty = max(moq, math.ceil(qty / moq) * moq) if qty > 0 else 0.0
            orders[t] = qty
            t = end
        else:
            t += 1
    return orders


@st.cache_data
def run_all_lot_methods(gross_req_t, sched_rec_t, init_inv,
                        order_cost, h_per_period, moq, lead_time):
    gross_req = list(gross_req_t)
    sched_rec = list(sched_rec_t)
    methods = {
        "Lot-for-Lot":     lot_for_lot(gross_req, sched_rec, init_inv, moq),
        "EOQ":             eoq_method(gross_req, sched_rec, init_inv, order_cost, h_per_period, moq),
        "Period Order Qty":period_order_qty(gross_req, sched_rec, init_inv, order_cost, h_per_period, moq),
        "Part Period Bal.":part_period_balancing(gross_req, sched_rec, init_inv, order_cost, h_per_period, moq),
        "Silver-Meal":     silver_meal(gross_req, sched_rec, init_inv, order_cost, h_per_period, moq),
        "Wagner-Whitin":   wagner_whitin(gross_req, sched_rec, init_inv, order_cost, h_per_period, moq),
    }
    costs = {}
    for name, orders in methods.items():
        oc, hc, tc = compute_costs(orders, gross_req, sched_rec, init_inv, order_cost, h_per_period)
        costs[name] = {"orders": orders, "ordering": oc, "holding": hc,
                       "total": tc, "n_orders": sum(1 for q in orders if q > 0)}
    return methods, costs

# ── EXPORT FUNCTION ───────────────────────────────────────────────────────────

def generate_buyer_export(sku, profile, cv_class, fc_method,
                           fc_result, optimal_name, mrp_df, costs,
                           unit_cost, order_cost, lead_time):
    lines = [
        "=" * 65,
        "MRP PLANNING INTELLIGENCE — BUYER ORDER SCHEDULE",
        "=" * 65,
        f"Component: {sku}",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "STAGE 1: DEMAND ANALYSIS",
        f"  Historical periods analyzed: {profile['n']}",
        f"  Mean usage per period:        {profile['mean']:.1f} units",
        f"  Std deviation:                {profile['std']:.1f} units",
        f"  Coefficient of Variation:     {profile['cv']*100:.1f}%",
        f"  Demand pattern:               {cv_class['label']}",
        f"  Forecasting method used:      {fc_method}",
        "",
        "STAGE 2: GROSS REQUIREMENTS (forecast-generated)",
    ]
    if fc_result and "point" in fc_result:
        for i, v in enumerate(fc_result["point"]):
            lines.append(f"  Period {i+1:>2}: {int(round(v)):>6} units  "
                         f"[80% PI: {int(round(fc_result['lower_80'][i]))} – "
                         f"{int(round(fc_result['upper_80'][i]))}]")
    lines += [
        "",
        f"RECOMMENDED LOT SIZING METHOD: {optimal_name}",
        f"  Total cost (ordering + holding): ${costs[optimal_name]['total']:.2f}",
        f"  Number of orders:                {costs[optimal_name]['n_orders']}",
        f"  Unit cost: ${unit_cost:.2f}  |  Ordering cost per PO: ${order_cost:.2f}  |  Lead time: {lead_time} period(s)",
        "",
        "PLANNED ORDER RELEASES (action required by buyers):",
    ]
    for _, row in mrp_df.iterrows():
        if row["Order Releases"] > 0:
            lines.append(f"  {row['Period']}: PLACE ORDER for {int(row['Order Releases'])} units")
    lines += [
        "",
        "FULL MRP RECORD:",
        f"  {'Period':<8} {'Gross Req':>10} {'Proj Avail':>12} {'Net Req':>9} {'Pln Receipts':>14} {'Order Rel':>11}",
        f"  {'-'*66}",
    ]
    for _, row in mrp_df.iterrows():
        lines.append(
            f"  {row['Period']:<8} {row['Gross Req']:>10} {row['Proj Available']:>12} "
            f"{row['Net Req']:>9} {row['Planned Receipts']:>14} {row['Order Releases']:>11}"
        )
    lines += [
        "",
        "=" * 65,
        "SOURCES: Jacobs, Berry, Whybark & Vollmann, Manufacturing Planning and",
        "Control (2011) | Silver & Meal (1973) | Wagner & Whitin (1958) |",
        "Hyndman & Athanasopoulos, FPP3 (2021) | Vandeput, DFBP (2023)",
        "Built by Rutwik Satish | MS Engineering Management + Graduate Certificate",
        "in Supply Chain Engineering Management | Northeastern University",
        "=" * 65,
    ]
    return "\n".join(lines)

# ── SESSION STATE ─────────────────────────────────────────────────────────────

for k, v in [("hist_df", None), ("gross_req_override", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""<div style="padding:0.5rem 0 1rem;">
<div style="font-size:1.25rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.02em;">MRP Planning</div>
<div style="font-size:0.72rem;color:#64748b;margin-top:0.1rem;">Intelligence</div>
</div>""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(f'<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;margin-bottom:0.4rem;">Stage 1 — Historical Usage Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV: Period, SKU (opt), Usage", type=["csv"], label_visibility="collapsed")

    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            raw.columns = [c.strip().lower() for c in raw.columns]
            usage_col = next((c for c in raw.columns if c in ("usage","demand","qty","quantity")), None)
            if not usage_col:
                st.error("CSV must contain a Usage or Demand column.")
            else:
                raw[usage_col] = pd.to_numeric(raw[usage_col], errors="coerce")
                raw = raw[raw[usage_col] >= 0].dropna(subset=[usage_col])
                if "period" not in raw.columns:
                    raw["period"] = range(1, len(raw) + 1)
                if "sku" not in raw.columns:
                    raw["sku"] = "SKU-1"
                st.session_state["hist_df"] = raw.rename(
                    columns={"period": "Period", "sku": "SKU", usage_col: "Usage"}
                )[["Period", "SKU", "Usage"]].reset_index(drop=True)
                st.success(f"{len(raw)} periods loaded.")
        except Exception as ex:
            st.error(f"Could not read: {ex}")

    if st.button("Load Sample Data (Sprocket-7A)", use_container_width=True):
        st.session_state["hist_df"] = SAMPLE_HISTORY.copy()
        st.session_state["gross_req_override"] = None
        st.success("Sample data loaded.")

    st.markdown("---")
    st.markdown(f'<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;margin-bottom:0.4rem;">Stage 2 — MRP Parameters</div>', unsafe_allow_html=True)

    forecast_horizon = st.slider("Forecast horizon (periods)", 4, 24, 12)
    unit_cost   = st.number_input("Unit cost ($)", value=25.0, step=0.5, min_value=0.01)
    order_cost  = st.number_input("Ordering cost per PO ($)", value=150.0, step=10.0, min_value=1.0)
    hold_rate   = st.number_input("Annual holding rate (%)", value=25.0, step=1.0, min_value=1.0)
    lead_time   = st.number_input("Lead time (periods)", value=1, step=1, min_value=0, max_value=8)
    init_inv    = st.number_input("Initial inventory (units)", value=0, step=10, min_value=0)
    moq         = st.number_input("Min order qty (units)", value=1, step=1, min_value=1)
    sched_rec_input = st.number_input("Scheduled receipts period 1 (units)", value=0, step=10, min_value=0)

    h_per_period = (hold_rate / 100) * unit_cost / 12

    st.markdown("---")
    current_method_erp = st.selectbox(
        "Current ERP method (baseline for savings)",
        ["Lot-for-Lot", "EOQ", "Period Order Qty", "Fixed Batch"],
    )
    fixed_batch = 0
    if current_method_erp == "Fixed Batch":
        fixed_batch = st.number_input("Fixed batch size", value=100, step=10, min_value=1)

    st.markdown("---")
    st.markdown(f"""<div style="font-size:0.68rem;color:#475569;line-height:1.7;">
Built by <span style="color:#e2e8f0;font-weight:600;">Rutwik Satish</span><br>
MS Engineering Management<br>
Grad Certificate — Supply Chain<br>
Northeastern University
</div>""", unsafe_allow_html=True)

# ── LANDING PAGE ──────────────────────────────────────────────────────────────

if st.session_state["hist_df"] is None:
    st.markdown(f"""
<div style="background:linear-gradient(135deg,{NAVY} 0%,#0d1b3e 50%,#1a3a7a 100%);
border:1px solid #1e3a6e;border-radius:16px;padding:3rem 3.5rem;margin-bottom:2rem;">
  <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.15em;
  color:#60a5fa;margin-bottom:0.75rem;font-weight:600;">Manufacturing Planning Tool</div>
  <div style="font-size:2.6rem;font-weight:700;letter-spacing:-0.03em;line-height:1.15;
  margin-bottom:1rem;color:#f1f5f9;">MRP Planning Intelligence</div>
  <div style="font-size:1.05rem;color:#93c5fd;font-weight:300;max-width:620px;
  line-height:1.7;margin-bottom:1.75rem;">
    From raw component usage history to a buyer-ready order schedule in under 3 minutes.
    Upload historical usage data. The tool classifies your demand pattern, selects the
    best forecasting method, generates gross requirements, and runs all six APICS-standard
    lot sizing algorithms to find the lowest-cost ordering strategy.
  </div>
  <div style="display:flex;gap:1rem;flex-wrap:wrap;">""", unsafe_allow_html=True)

    badges = ["Stage 1: Demand Intelligence", "Stage 2: Lot Sizing Optimization",
              "Buyer-Ready Export", "APICS Grounded"]
    badge_html = "".join(
        f'<div style="background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.3);'
        f'border-radius:8px;padding:0.5rem 1rem;font-size:0.8rem;color:#93c5fd;">{b}</div>'
        for b in badges
    )
    st.markdown(badge_html + "</div></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
<div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:1.5rem;">
<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:{T_MUT};margin-bottom:0.75rem;font-weight:700;">How It Works</div>
<div style="display:grid;gap:1rem;">
{"".join(f'''<div style="display:flex;gap:0.75rem;align-items:flex-start;">
<div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:700;color:{BLUE};flex-shrink:0;width:1.5rem;">{n}</div>
<div><div style="font-size:0.88rem;font-weight:600;color:{T_PRI};margin-bottom:0.2rem;">{t}</div>
<div style="font-size:0.78rem;color:{T_SEC};">{d}</div></div></div>'''
for n, t, d in [
    ("1", "Upload usage history", "CSV with Period and Usage columns. Or load sample data to see the tool in action."),
    ("2", "Demand classification", "CV-based ABC-XYZ pattern detection (Vandeput, DFBP Ch. 13) with automatic forecasting method selection."),
    ("3", "Forecast generation", "Statistical forecast auto-populates gross requirements for your chosen horizon. Override any period manually."),
    ("4", "Lot sizing comparison", "All 6 APICS algorithms run simultaneously. Ranked by total cost. Savings vs your current ERP method shown."),
    ("5", "Export order schedule", "Download a buyer-ready planned order release schedule with full MRP record and cost justification."),
])}
</div></div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
<div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:1.5rem;">
<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:{T_MUT};margin-bottom:0.75rem;font-weight:700;">Sample Data Preview — Sprocket-7A</div>""", unsafe_allow_html=True)

        rows_html = "".join(
            f'<tr style="border-bottom:1px solid {BORDER};">'
            f'<td style="padding:0.3rem 0.75rem;color:{T_SEC};font-family:IBM Plex Mono,monospace;font-size:0.78rem;">{int(r.Period)}</td>'
            f'<td style="padding:0.3rem 0.75rem;color:#60a5fa;font-size:0.75rem;">{r.SKU}</td>'
            f'<td style="padding:0.3rem 0.75rem;color:{T_PRI};font-family:IBM Plex Mono,monospace;font-size:0.78rem;text-align:right;">{int(r.Usage):,}</td>'
            f'</tr>'
            for _, r in SAMPLE_HISTORY.iterrows()
        )
        st.markdown(f"""
<div style="background:#0d1724;border:1px solid {BORDER};border-radius:8px;overflow:hidden;max-height:320px;overflow-y:auto;">
<table style="width:100%;border-collapse:collapse;">
<thead><tr style="background:#060c16;border-bottom:2px solid #1e3a6e;">
  <th style="padding:0.4rem 0.75rem;text-align:left;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:{T_MUT};">Period</th>
  <th style="padding:0.4rem 0.75rem;text-align:left;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:{T_MUT};">SKU</th>
  <th style="padding:0.4rem 0.75rem;text-align:right;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:{T_MUT};">Usage</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
<div style="margin-top:2rem;padding:1rem 1.5rem;background:{NAVY};border-radius:10px;
font-size:0.68rem;color:#475569;line-height:1.9;text-align:center;">
  <span style="color:#93c5fd;font-weight:600;">Lot sizing:</span> Jacobs, Berry, Whybark & Vollmann, Manufacturing Planning and Control (2011) &nbsp;|&nbsp;
  <span style="color:#93c5fd;font-weight:600;">Silver-Meal:</span> Silver & Meal (1973), Management Science &nbsp;|&nbsp;
  <span style="color:#93c5fd;font-weight:600;">Wagner-Whitin:</span> Wagner & Whitin (1958), Management Science &nbsp;|&nbsp;
  <span style="color:#93c5fd;font-weight:600;">Forecasting:</span> Hyndman & Athanasopoulos, FPP3 (2021) &nbsp;|&nbsp;
  <span style="color:#93c5fd;font-weight:600;">Demand classification:</span> Vandeput, DFBP (2023)
</div>""", unsafe_allow_html=True)
    st.stop()

# ── MAIN APP — DATA LOADED ────────────────────────────────────────────────────

hist_df = st.session_state["hist_df"]
usage_arr = hist_df["Usage"].values.astype(float)
sku_label = hist_df["SKU"].iloc[0] if "SKU" in hist_df.columns else "Component"
n_hist = len(usage_arr)
profile = compute_profile(usage_arr)
cv_class = classify_cv(profile["cv"])

# Page header
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
margin-bottom:1rem;padding-bottom:0.75rem;border-bottom:1px solid {BORDER};">
  <div>
    <div style="font-size:1.4rem;font-weight:700;color:{T_PRI};letter-spacing:-0.02em;">MRP Planning Intelligence</div>
    <div style="font-size:0.78rem;color:{T_SEC};">
      Component: <b style="color:{T_PRI};">{sku_label}</b> &nbsp;|&nbsp;
      {n_hist} periods of usage history &nbsp;|&nbsp;
      Mean: <b style="color:{T_PRI};">{profile['mean']:.0f}</b> units &nbsp;|&nbsp;
      Pattern: <b style="color:{cv_class['color']};">{cv_class['label']}</b>
    </div>
  </div>
  <div>
    {stage_badge("S1", "Demand Intelligence")}
    {stage_badge("S2", "Lot Sizing")}
  </div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "Stage 1 — Demand Intelligence",
    "Stage 2 — Lot Sizing Optimization",
    "Buyer Export"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — DEMAND INTELLIGENCE
# ════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    section("Demand Pattern Classification")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(card("Mean Usage", f"{profile['mean']:.1f}", "units per period"), unsafe_allow_html=True)
    with c2: st.markdown(card("Std Deviation", f"{profile['std']:.1f}", "sample std, ddof=1"), unsafe_allow_html=True)
    with c3: st.markdown(card("Coefficient of Variation", f"{profile['cv']*100:.1f}%",
                               f"CV = std / mean", cv_class["color"], cv_class["color"]), unsafe_allow_html=True)

    col_badge, col_guide = st.columns([1, 2])
    with col_badge:
        st.markdown(f"""
<div style="background:{CARD};border:1px solid {BORDER};border-left:5px solid {cv_class['color']};
border-radius:10px;padding:1.25rem 1.5rem;">
  <span style="background:{cv_class['color']};color:#fff;font-size:0.85rem;font-weight:600;
  padding:0.3rem 0.9rem;border-radius:999px;">{cv_class['label']}</span>
  <div style="font-size:0.85rem;color:{T_SEC};margin:0.75rem 0 0.4rem;">{cv_class['description']}</div>
  <div style="font-size:0.75rem;color:{T_MUT};margin-top:0.5rem;">
    Recommended forecasting method: <b style="color:{T_PRI};">{cv_class['recommended_fc']}</b><br>
    Recommended lot sizing: <b style="color:{T_PRI};">{cv_class['recommended_lot']}</b>
  </div>
  <div style="font-size:0.68rem;color:{T_MUT};font-style:italic;margin-top:0.5rem;">
    Source: Vandeput, DFBP Ch. 13 — ABC-XYZ segmentation
  </div>
</div>""", unsafe_allow_html=True)

    with col_guide:
        st.markdown(f"""
<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:1.25rem 1.5rem;">
<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:{T_MUT};margin-bottom:0.75rem;">Classification Guide</div>
<table style="width:100%;font-size:0.8rem;border-collapse:collapse;">
<tr style="border-bottom:1px solid {BORDER};">
  <th style="text-align:left;padding:0.4rem 0.5rem;color:{T_MUT};font-weight:600;">Class</th>
  <th style="text-align:left;padding:0.4rem 0.5rem;color:{T_MUT};font-weight:600;">CV</th>
  <th style="text-align:left;padding:0.4rem 0.5rem;color:{T_MUT};font-weight:600;">Pattern</th>
  <th style="text-align:left;padding:0.4rem 0.5rem;color:{T_MUT};font-weight:600;">Best Lot Method</th>
</tr>
<tr style="border-bottom:1px solid {BORDER};">
  <td style="padding:0.4rem 0.5rem;"><span style="color:{GREEN};font-weight:700;">X</span></td>
  <td style="padding:0.4rem 0.5rem;font-family:IBM Plex Mono,monospace;">&lt; 0.20</td>
  <td style="padding:0.4rem 0.5rem;">Stable</td>
  <td style="padding:0.4rem 0.5rem;">Silver-Meal or EOQ</td>
</tr>
<tr style="border-bottom:1px solid {BORDER};">
  <td style="padding:0.4rem 0.5rem;"><span style="color:{AMBER};font-weight:700;">Y</span></td>
  <td style="padding:0.4rem 0.5rem;font-family:IBM Plex Mono,monospace;">0.20 – 0.50</td>
  <td style="padding:0.4rem 0.5rem;">Variable</td>
  <td style="padding:0.4rem 0.5rem;">Silver-Meal or PPB</td>
</tr>
<tr>
  <td style="padding:0.4rem 0.5rem;"><span style="color:{RED};font-weight:700;">Z</span></td>
  <td style="padding:0.4rem 0.5rem;font-family:IBM Plex Mono,monospace;">&gt; 0.50</td>
  <td style="padding:0.4rem 0.5rem;">Lumpy</td>
  <td style="padding:0.4rem 0.5rem;">Lot-for-Lot</td>
</tr>
</table>
</div>""", unsafe_allow_html=True)

    section("Usage History")
    sma6 = pd.Series(usage_arr).rolling(6).mean().values
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(x=list(range(1, n_hist+1)), y=usage_arr, name="Usage",
                               marker_color=BLUE, opacity=0.75, marker_line_width=0))
    fig_hist.add_trace(go.Scatter(x=list(range(1, n_hist+1)), y=sma6, name="6-Period SMA",
                                   mode="lines", line=dict(color=AMBER, width=2.5, dash="dash")))
    fig_hist.update_layout(height=280, xaxis_title="Period", yaxis_title="Usage (units)", **CHART)
    st.plotly_chart(fig_hist, width="stretch")
    insight("The 6-period moving average reveals the underlying usage trend. Consistent upward or downward movement in the dashed line should be reflected in your gross requirements.")

    section("Automatic Gross Requirements Generation")
    fc_method_options = ["SES", "Holt Linear"]
    if n_hist >= 24:
        fc_method_options.append("Holt-Winters")
    default_fc_idx = fc_method_options.index(cv_class["recommended_fc"]) if cv_class["recommended_fc"] in fc_method_options else 0
    fc_col1, fc_col2 = st.columns([1, 2])
    with fc_col1:
        fc_method_sel = st.selectbox("Forecasting method", fc_method_options, index=default_fc_idx,
                                      help="Pre-selected based on CV classification. Override if needed.")
        st.markdown(f'<div style="font-size:0.72rem;color:{T_MUT};margin-top:0.25rem;font-style:italic;">Recommended for {cv_class["label"]}: {cv_class["recommended_fc"]}</div>', unsafe_allow_html=True)

    with st.spinner("Generating forecast..."):
        fc_result = generate_forecast(tuple(usage_arr), fc_method_sel, forecast_horizon)

    forecast_gr = [max(0, int(round(v))) for v in fc_result["point"]]

    with fc_col2:
        st.markdown(f'<div style="font-size:0.78rem;color:{T_SEC};margin-bottom:0.5rem;">Forecast auto-populated below. Edit any period to override.</div>', unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:0.68rem;color:{T_MUT};margin-bottom:0.5rem;font-style:italic;">Tip: Edit the Gross Requirements table below to override any period. Changes flow through to Stage 2 automatically.</div>', unsafe_allow_html=True)

    periods_list = [f"P{i+1}" for i in range(forecast_horizon)]
    lower_80 = [max(0, int(round(v))) for v in fc_result["lower_80"]]
    upper_80 = [int(round(v)) for v in fc_result["upper_80"]]

    gr_default = pd.DataFrame({
        "Period": periods_list,
        "Gross Requirements": forecast_gr,
        "Lower 80% PI": lower_80,
        "Upper 80% PI": upper_80,
        "Scheduled Receipts": [sched_rec_input if i == 0 else 0 for i in range(forecast_horizon)]
    })

    gr_edited = st.data_editor(
        gr_default,
        num_rows="fixed",
        width="stretch",
        column_config={
            "Period":            st.column_config.TextColumn("Period", disabled=True, width="small"),
            "Gross Requirements":st.column_config.NumberColumn("Gross Requirements", min_value=0, step=1),
            "Lower 80% PI":      st.column_config.NumberColumn("Lower 80% PI", disabled=True),
            "Upper 80% PI":      st.column_config.NumberColumn("Upper 80% PI", disabled=True),
            "Scheduled Receipts":st.column_config.NumberColumn("Scheduled Receipts", min_value=0, step=1),
        }
    )
    cite("Prediction intervals: sigma_h = RMSE x sqrt(h), 80% PI = forecast +/- 1.282 x sigma_h (FPP3 Ch. 5.5). Edit Gross Requirements column to override any period.")

    gross_req_final = gr_edited["Gross Requirements"].fillna(0).tolist()
    sched_rec_final = gr_edited["Scheduled Receipts"].fillna(0).tolist()
    st.session_state["gross_req_override"] = (tuple(gross_req_final), tuple(sched_rec_final))

    # Forecast vs history chart
    section("Forecast Chart")
    hist_x = list(range(1, n_hist + 1))
    fwd_x  = list(range(n_hist + 1, n_hist + forecast_horizon + 1))
    fig_fwd = go.Figure()
    fig_fwd.add_trace(go.Scatter(x=hist_x, y=usage_arr, name="Historical Usage",
                                  mode="lines+markers", line=dict(color=T_SEC, width=2),
                                  marker=dict(size=4)))
    fig_fwd.add_trace(go.Scatter(
        x=fwd_x + fwd_x[::-1],
        y=upper_80 + lower_80[::-1],
        fill="toself", fillcolor=f"rgba(59,130,246,0.1)",
        line=dict(color="rgba(0,0,0,0)"), name="80% PI", hoverinfo="skip"
    ))
    fig_fwd.add_trace(go.Scatter(x=fwd_x, y=gross_req_final, name="Gross Requirements",
                                  mode="lines+markers", line=dict(color=BLUE, width=2.5),
                                  marker=dict(size=6)))
    fig_fwd.add_vline(x=n_hist + 0.5, line_dash="dash", line_color=T_MUT, line_width=1,
                       annotation_text="Forecast start", annotation_position="top")
    fig_fwd.update_layout(height=300, xaxis_title="Period", yaxis_title="Units", **CHART)
    st.plotly_chart(fig_fwd, width="stretch")
    insight("Blue line shows forecast-generated gross requirements feeding into Stage 2. Shaded band is the 80% prediction interval — your actual demand has an 80% chance of falling within this range.")

# ════════════════════════════════════════════════════════════════
# TAB 2 — LOT SIZING
# ════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    gr_data = st.session_state.get("gross_req_override")
    if gr_data is None:
        alert("Run Stage 1 first to generate gross requirements.", "info")
        st.stop()

    gross_req_t, sched_rec_t = gr_data

    with st.spinner("Running all 6 lot sizing algorithms..."):
        methods, costs = run_all_lot_methods(
            gross_req_t, sched_rec_t, init_inv,
            order_cost, h_per_period, moq, lead_time
        )

    sorted_costs = sorted(costs.items(), key=lambda x: x[1]["total"])
    optimal_name = sorted_costs[0][0]

    # Pre-select recommended method based on CV
    rec_lot = cv_class["recommended_lot"].split(" or ")[0].strip()

    section("Cost Comparison — All 6 APICS Methods")
    metric_cols = st.columns(6)
    for i, (name, c) in enumerate(sorted_costs):
        is_best = name == optimal_name
        is_rec  = name == rec_lot
        delta_vs_ww = round(c["total"] - costs["Wagner-Whitin"]["total"], 2)
        lbl = f"{name} ★" if is_best else name
        metric_cols[i].metric(
            label=lbl,
            value=f"${c['total']:.0f}",
            delta="Optimal" if is_best else f"+${delta_vs_ww:.0f} vs optimal",
            delta_color="normal" if is_best else "inverse"
        )

    names_list    = [n for n, _ in sorted_costs]
    ordering_vals = [costs[n]["ordering"] for n in names_list]
    holding_vals  = [costs[n]["holding"]  for n in names_list]
    colors_bar    = [METHOD_COLORS.get(n, T_MUT) for n in names_list]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="Ordering Cost", x=names_list, y=ordering_vals,
        marker_color=colors_bar, opacity=0.85,
        text=[f"${v:.0f}" for v in ordering_vals],
        textposition="inside", textfont=dict(color="white", size=11),
    ))
    fig_bar.add_trace(go.Bar(
        name="Holding Cost", x=names_list, y=holding_vals,
        marker_color=colors_bar, opacity=0.40,
        text=[f"${v:.0f}" for v in holding_vals],
        textposition="inside", textfont=dict(color="white", size=11),
    ))
    fig_bar.add_annotation(
        x=optimal_name, y=costs[optimal_name]["total"] + 5,
        text="Optimal", showarrow=False,
        font=dict(color=GREEN, size=12, family="Inter"),
    )
    fig_bar.update_layout(
        **CHART, barmode="stack", height=340,
        title=dict(text="Total Cost = Ordering Cost + Holding Cost", font=dict(size=13, color=T_SEC)),
        xaxis_title="", yaxis_title="Total Cost ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bar, width="stretch")
    cite("Jacobs, Berry, Whybark & Vollmann (2011) Ch. 4 | Silver & Meal (1973) | Wagner & Whitin (1958) — benchmark lower bound")

    section("Savings vs Current ERP Method")
    if current_method_erp == "Fixed Batch":
        fb_orders = lot_for_lot(list(gross_req_t), list(sched_rec_t), init_inv, fixed_batch)
        _, _, current_cost = compute_costs(fb_orders, list(gross_req_t), list(sched_rec_t),
                                            init_inv, order_cost, h_per_period)
        current_label = f"Fixed Batch ({fixed_batch} units)"
    else:
        current_cost  = costs[current_method_erp]["total"]
        current_label = current_method_erp

    savings = current_cost - costs[optimal_name]["total"]
    periods_per_year_est = 12
    ann_savings = savings * (periods_per_year_est / forecast_horizon) if forecast_horizon > 0 else 0

    if savings > 0:
        sc1, sc2, sc3 = st.columns(3)
        with sc1: st.markdown(card("Savings This Horizon", f"${savings:,.0f}", f"{current_label} vs {optimal_name}", GREEN, GREEN), unsafe_allow_html=True)
        with sc2: st.markdown(card("Annualized Savings", f"${ann_savings:,.0f}", "extrapolated to 12 months", GREEN, GREEN), unsafe_allow_html=True)
        with sc3: st.markdown(card("Across 200-SKU BOM", f"${ann_savings*200:,.0f}", "if all SKUs achieve same savings", GREEN, GREEN), unsafe_allow_html=True)
        insight(f"Switching from {current_label} to {optimal_name} on this one component saves ${savings:,.0f} over {forecast_horizon} periods with no capital investment. The ERP already supports all six methods — this is a configuration change, not an IT project.")
    elif savings == 0:
        alert(f"Your current method ({current_label}) is already optimal for this demand pattern.", "success")
    else:
        alert(f"Your current method is lower cost than the auto-selected optimal for this demand pattern. Review the full comparison above.", "warning")

    section("Recommended Method Rationale")
    cat = cv_class["category"]
    if cat == "X":
        rationale = f"Stable demand (CV={profile['cv']*100:.1f}%) means order quantities can be batched efficiently. Silver-Meal minimizes cost per period and performs within 8% of the Wagner-Whitin optimal (Silver & Meal, 1973). EOQ is a reliable alternative if your ERP does not support Silver-Meal."
    elif cat == "Y":
        rationale = f"Variable demand (CV={profile['cv']*100:.1f}%) means EOQ assumptions break down. Silver-Meal or Part Period Balancing dynamically adjust batch size as demand fluctuates. Lot-for-Lot maximizes ordering frequency and should only be used if holding cost significantly exceeds ordering cost."
    else:
        rationale = f"Highly variable / lumpy demand (CV={profile['cv']*100:.1f}%) means statistical lot sizing methods are unreliable. Lot-for-Lot minimizes excess inventory risk. Consider reviewing whether demand is genuinely lumpy or whether usage data quality is a factor."
    st.markdown(f"""
<div style="background:{CARD};border:1px solid {BORDER};border-left:4px solid {cv_class['color']};
border-radius:10px;padding:1.25rem 1.5rem;">
<div style="font-size:0.88rem;color:#cbd5e1;line-height:1.75;">
Demand pattern: <b style="color:{cv_class['color']};">{cv_class['label']}</b>.
Recommended method: <b style="color:{T_PRI};">{rec_lot}</b>. {rationale}
</div></div>""", unsafe_allow_html=True)

    section("Full MRP Record")
    sel_method = st.selectbox("View MRP record for method:",
                               list(methods.keys()),
                               index=list(methods.keys()).index(optimal_name))

    mrp_df = compute_mrp_record(
        list(gross_req_t), list(sched_rec_t), init_inv,
        methods[sel_method], lead_time
    )

    # HTML MRP table
    mrp_header = "".join(
        f'<th style="padding:0.4rem 0.75rem;text-align:{"right" if i>0 else "left"};'
        f'font-size:0.65rem;text-transform:uppercase;letter-spacing:0.07em;'
        f'color:{T_MUT};font-weight:600;">{c}</th>'
        for i, c in enumerate(mrp_df.columns)
    )
    mrp_rows_html = ""
    for _, row in mrp_df.iterrows():
        cells = ""
        for i, (col, val) in enumerate(zip(mrp_df.columns, row)):
            color = T_PRI
            bg = ""
            if col == "Order Releases" and val > 0:
                color = BLUE; bg = f"background:#0d1b3e;"
            elif col == "Net Req" and val > 0:
                color = AMBER; bg = f"background:#1a1200;"
            align = "right" if i > 0 else "left"
            cells += f'<td style="padding:0.38rem 0.75rem;text-align:{align};font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:{color};border-bottom:1px solid {BORDER};{bg}">{int(val):,}</td>'
        mrp_rows_html += f"<tr>{cells}</tr>"

    st.markdown(f"""
<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;overflow:hidden;overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;min-width:600px;">
<thead><tr style="background:#060c16;border-bottom:2px solid #1e3a6e;">{mrp_header}</tr></thead>
<tbody>{mrp_rows_html}</tbody>
</table></div>""", unsafe_allow_html=True)
    insight("Blue = periods where you need to PLACE AN ORDER (Planned Order Releases). Amber = periods with net shortage before the order arrives. Lead time offset already applied.")
    cite("MRP record format: Jacobs & Berry (2011) Ch. 3 — GR, SR, PAB, NR, POR, POL")

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(card("Number of Orders", str(costs[sel_method]["n_orders"]), "purchase orders to place"), unsafe_allow_html=True)
    with m2: st.markdown(card("Ordering Cost", f"${costs[sel_method]['ordering']:,.0f}", f"{costs[sel_method]['n_orders']} orders x ${order_cost:.0f}"), unsafe_allow_html=True)
    with m3: st.markdown(card("Holding Cost", f"${costs[sel_method]['holding']:,.0f}", f"avg inventory x ${h_per_period:.3f}/unit/period"), unsafe_allow_html=True)
    with m4: st.markdown(card("Total Cost", f"${costs[sel_method]['total']:,.0f}", "ordering + holding"), unsafe_allow_html=True)

    # Inventory profile chart
    section("Inventory Profile")
    pab_vals = mrp_df["Proj Available"].tolist()
    por_vals = mrp_df["Planned Receipts"].tolist()
    gr_vals  = mrp_df["Gross Req"].tolist()
    periods  = mrp_df["Period"].tolist()

    fig_inv = go.Figure()
    fig_inv.add_trace(go.Bar(x=periods, y=por_vals, name="Order Received",
                              marker_color=METHOD_COLORS.get(sel_method, BLUE),
                              opacity=0.7, marker_line_width=0))
    fig_inv.add_trace(go.Scatter(x=periods, y=pab_vals, name="Projected Available",
                                  mode="lines+markers", line=dict(color=T_PRI, width=2),
                                  marker=dict(size=6)))
    fig_inv.add_trace(go.Bar(x=periods, y=[-v for v in gr_vals], name="Gross Requirements",
                              marker_color=RED, opacity=0.4, marker_line_width=0))
    fig_inv.add_hline(y=0, line_dash="solid", line_color=T_MUT, line_width=1)
    fig_inv.update_layout(**CHART, height=300, barmode="relative",
                           xaxis_title="Period", yaxis_title="Units")
    st.plotly_chart(fig_inv, width="stretch")
    insight("Bars above zero are inbound orders. White line is projected available inventory. Bars below zero are gross requirements consuming inventory. The line should never drop below zero — if it does, review your scheduled receipts or increase safety stock.")

    # Store for export
    st.session_state["_export_data"] = {
        "sku": sku_label, "profile": profile, "cv_class": cv_class,
        "fc_method": fc_method_sel, "fc_result": fc_result,
        "optimal_name": optimal_name, "mrp_df": mrp_df, "costs": costs,
        "unit_cost": unit_cost, "order_cost": order_cost, "lead_time": int(lead_time)
    }

# ════════════════════════════════════════════════════════════════
# TAB 3 — BUYER EXPORT
# ════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    exp = st.session_state.get("_export_data")
    if not exp:
        alert("Complete Stage 1 and Stage 2 first, then return here to export.", "info")
    else:
        section("Buyer-Ready Order Schedule")
        st.markdown(f"""
<div style="background:{CARD};border:1px solid {BORDER};border-left:4px solid {GREEN};
border-radius:10px;padding:1rem 1.25rem;margin-bottom:1rem;">
<div style="font-size:0.78rem;color:#cbd5e1;">
Component: <b style="color:{T_PRI};">{exp['sku']}</b> &nbsp;|&nbsp;
Demand pattern: <b style="color:{exp['cv_class']['color']};">{exp['cv_class']['label']}</b> &nbsp;|&nbsp;
Recommended method: <b style="color:{T_PRI};">{exp['optimal_name']}</b> &nbsp;|&nbsp;
Total cost: <b style="color:{T_PRI};">${exp['costs'][exp['optimal_name']]['total']:,.0f}</b>
</div></div>""", unsafe_allow_html=True)

        section("Planned Order Releases — Action Required")
        order_rows = exp["mrp_df"][exp["mrp_df"]["Order Releases"] > 0]
        if len(order_rows) == 0:
            alert("No orders required in this planning horizon — current inventory covers all requirements.", "success")
        else:
            action_html = "".join(
                f'<div style="background:#0d1b3e;border:1px solid #1e3a6e;border-left:4px solid {BLUE};'
                f'border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.5rem;display:flex;'
                f'justify-content:space-between;align-items:center;">'
                f'<div><span style="font-family:IBM Plex Mono,monospace;font-size:0.9rem;'
                f'font-weight:700;color:{BLUE};">ORDER IN {row.Period}</span>'
                f'<span style="font-size:0.8rem;color:{T_SEC};margin-left:1rem;">Lead time: {int(lead_time)} period(s) before {row.Period}</span></div>'
                f'<span style="font-family:IBM Plex Mono,monospace;font-size:1.1rem;'
                f'font-weight:700;color:{T_PRI};">{int(row["Order Releases"]):,} units</span>'
                f'</div>'
                for _, row in order_rows.iterrows()
            )
            st.markdown(action_html, unsafe_allow_html=True)

        section("Export")
        summary = generate_buyer_export(
            exp["sku"], exp["profile"], exp["cv_class"],
            exp["fc_method"], exp["fc_result"],
            exp["optimal_name"], exp["mrp_df"], exp["costs"],
            exp["unit_cost"], exp["order_cost"], exp["lead_time"]
        )

        st.markdown(f"""
<div style="background:{NAVY};color:#f1f5f9;font-family:'IBM Plex Mono',monospace;
font-size:0.72rem;line-height:1.7;padding:1.25rem;border-radius:10px;
white-space:pre-wrap;overflow-x:auto;">{summary}</div>""", unsafe_allow_html=True)

        ec1, ec2 = st.columns(2)
        with ec1:
            st.download_button(
                "Download Order Schedule (.txt)",
                data=summary,
                file_name=f"MRP_{exp['sku'].replace(' ','_')}_order_schedule.txt",
                mime="text/plain",
                use_container_width=True
            )
        with ec2:
            order_csv = exp["mrp_df"].to_csv(index=False)
            st.download_button(
                "Download MRP Record (.csv)",
                data=order_csv,
                file_name=f"MRP_{exp['sku'].replace(' ','_')}_record.csv",
                mime="text/csv",
                use_container_width=True
            )

# ── FOOTER ────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="margin-top:3rem;padding:1.25rem 1.75rem;background:{NAVY};border-radius:10px;
font-size:0.68rem;color:#475569;line-height:1.9;text-align:center;">
  <span style="color:#93c5fd;font-weight:600;">Lot sizing algorithms:</span>
  Jacobs, Berry, Whybark & Vollmann, Manufacturing Planning and Control, 6th ed. (2011) &nbsp;|&nbsp;
  <span style="color:#93c5fd;font-weight:600;">Silver-Meal:</span> Silver & Meal (1973), Management Science &nbsp;|&nbsp;
  <span style="color:#93c5fd;font-weight:600;">Wagner-Whitin:</span> Wagner & Whitin (1958), Management Science, 5(1) &nbsp;|&nbsp;
  <span style="color:#93c5fd;font-weight:600;">Forecasting:</span> Hyndman & Athanasopoulos, FPP3 (2021) &nbsp;|&nbsp;
  <span style="color:#93c5fd;font-weight:600;">Demand classification:</span> Vandeput, DFBP (2023) &nbsp;|&nbsp;
  Built by <span style="color:#e2e8f0;font-weight:600;">Rutwik Satish</span>
  — MS Engineering Management + Graduate Certificate in Supply Chain Engineering Management, Northeastern University
</div>""", unsafe_allow_html=True)
