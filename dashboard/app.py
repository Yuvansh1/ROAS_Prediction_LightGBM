"""
ROAS Prediction — Audit Trail Dashboard
========================================
Streamlit app that reads logs/audit.jsonl and renders an interactive
audit trail with four tabs:

  Overview       — headline KPI cards + pipeline summary
  Training Runs  — RMSE history, hyperparameters, feature importance
  ROAS Forecast  — actual vs predicted ROAS time series + distribution
  Audit Log      — filterable raw event table with event timeline
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Ensure src/ is importable when running: streamlit run dashboard/app.py ──
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from roas_prediction.utils.audit import AuditLogger  # noqa: E402

# ────────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ROAS Prediction · Audit Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

AUDIT_LOG = ROOT / "logs" / "audit.jsonl"

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_audit_records(log_path: str) -> list[dict]:
    audit = AuditLogger(log_path=log_path)
    return audit.read_all()


def fmt_currency(v: float) -> str:
    return f"${v:,.2f}"


def fmt_pct(v: float) -> str:
    return f"{v:.2f}%"


def fmt_ts(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return iso


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.shields.io/badge/LightGBM-ROAS-blue?style=flat-square",
        use_container_width=False,
    )
    st.title("ROAS Prediction")
    st.caption("Audit Trail Dashboard")
    st.divider()

    custom_log = st.text_input(
        "Audit log path",
        value=str(AUDIT_LOG),
        help="Absolute or relative path to audit.jsonl",
    )
    refresh = st.button("🔄 Refresh data", use_container_width=True)
    if refresh:
        st.cache_data.clear()

    st.divider()
    st.caption(f"Log file: `{Path(custom_log).name}`")
    log_exists = Path(custom_log).exists()
    if log_exists:
        st.success("Log file found", icon="✅")
    else:
        st.warning("Log file not found — run `python main.py` first.", icon="⚠️")

# ────────────────────────────────────────────────────────────────────────────
# Load data
# ────────────────────────────────────────────────────────────────────────────
records = load_audit_records(custom_log) if log_exists else []

training_records = [r for r in records if r["event_type"] == "training_run"]
prediction_records = [r for r in records if r["event_type"] == "prediction"]
data_gen_records = [r for r in records if r["event_type"] == "data_generation"]
feat_eng_records = [r for r in records if r["event_type"] == "feature_engineering"]

# ────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────
st.title("📊 ROAS Prediction · Audit Trail")
st.caption(
    "A complete audit of every data-generation, feature-engineering, "
    "training, and inference event produced by the pipeline."
)

if not records:
    st.info(
        "No audit records found. Run `python main.py` to generate data and train the model, "
        "then refresh this dashboard.",
        icon="ℹ️",
    )
    st.stop()

# ────────────────────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────────────────────
tab_overview, tab_training, tab_roas, tab_log = st.tabs(
    ["🏠 Overview", "🎯 Training Runs", "📈 ROAS Forecast", "📋 Audit Log"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Pipeline Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events", len(records))
    col2.metric("Training Runs", len(training_records))
    col3.metric("Prediction Batches", len(prediction_records))
    col4.metric(
        "Total Predictions",
        sum(r["payload"].get("rows", 0) for r in prediction_records),
    )

    st.divider()

    # Latest run summary
    if training_records:
        latest = training_records[-1]
        p = latest["payload"]
        metrics = p.get("metrics", {})

        st.subheader("Latest Training Run")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", fmt_currency(metrics.get("rmse", 0)))
        c2.metric("MAE", fmt_currency(metrics.get("mae", 0)))
        c3.metric("R²", f"{metrics.get('r2', 0):.4f}")
        c4.metric("MAPE", fmt_pct(metrics.get("mape", 0)))

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Run metadata**")
            st.json(
                {
                    "run_id": p.get("run_id", "n/a"),
                    "timestamp": fmt_ts(latest["timestamp"]),
                    "train_rows": p.get("train_rows"),
                    "test_rows": p.get("test_rows"),
                    "best_iteration": p.get("best_iteration"),
                    "duration_s": p.get("duration_seconds"),
                    "model_path": p.get("model_path"),
                },
                expanded=True,
            )
        with col_b:
            st.markdown("**Hyperparameters**")
            st.json(p.get("params", {}), expanded=True)

    st.divider()

    # Event-type breakdown donut
    event_counts = pd.Series(
        [r["event_type"] for r in records]
    ).value_counts().reset_index()
    event_counts.columns = ["Event Type", "Count"]

    fig_donut = px.pie(
        event_counts,
        values="Count",
        names="Event Type",
        hole=0.55,
        title="Event Type Breakdown",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_donut.update_traces(textposition="outside", textinfo="percent+label")
    fig_donut.update_layout(showlegend=False, margin=dict(t=50, b=0, l=0, r=0))
    st.plotly_chart(fig_donut, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING RUNS
# ══════════════════════════════════════════════════════════════════════════════
with tab_training:
    if not training_records:
        st.info("No training runs found in the audit log.")
        st.stop()

    # Build runs table
    rows_list = []
    for r in training_records:
        p = r["payload"]
        m = p.get("metrics", {})
        rows_list.append(
            {
                "timestamp": fmt_ts(r["timestamp"]),
                "run_id": p.get("run_id", "")[:8] + "…",
                "run_id_full": p.get("run_id", ""),
                "train_rows": p.get("train_rows"),
                "test_rows": p.get("test_rows"),
                "best_iter": p.get("best_iteration"),
                "rmse": m.get("rmse"),
                "mae": m.get("mae"),
                "r2": m.get("r2"),
                "mape": m.get("mape"),
                "duration_s": p.get("duration_seconds"),
            }
        )
    runs_df = pd.DataFrame(rows_list)

    # ── RMSE over runs ──────────────────────────────────────────────────────
    st.subheader("RMSE Across Training Runs")
    fig_rmse = px.line(
        runs_df,
        x="timestamp",
        y="rmse",
        markers=True,
        labels={"rmse": "RMSE ($)", "timestamp": "Run Timestamp"},
        color_discrete_sequence=["#636EFA"],
    )
    fig_rmse.update_layout(hovermode="x unified", yaxis_tickprefix="$")
    st.plotly_chart(fig_rmse, use_container_width=True)

    col_left, col_right = st.columns(2)

    # ── R² over runs ────────────────────────────────────────────────────────
    with col_left:
        st.subheader("R² Over Runs")
        fig_r2 = px.line(
            runs_df,
            x="timestamp",
            y="r2",
            markers=True,
            color_discrete_sequence=["#00CC96"],
            labels={"r2": "R²", "timestamp": "Run Timestamp"},
        )
        fig_r2.update_layout(yaxis_range=[0, 1], hovermode="x unified")
        st.plotly_chart(fig_r2, use_container_width=True)

    # ── Feature importance (latest run) ────────────────────────────────────
    with col_right:
        st.subheader("Feature Importance (Latest Run)")
        latest_p = training_records[-1]["payload"]
        importance = latest_p.get("feature_importance", {})
        if importance:
            imp_df = (
                pd.DataFrame.from_dict(importance, orient="index", columns=["gain"])
                .sort_values("gain", ascending=True)
            )
            fig_imp = px.bar(
                imp_df,
                x="gain",
                y=imp_df.index,
                orientation="h",
                color="gain",
                color_continuous_scale="Blues",
                labels={"gain": "Gain", "y": "Feature"},
            )
            fig_imp.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance not recorded in this run.")

    # ── Runs table ──────────────────────────────────────────────────────────
    st.subheader("All Training Runs")
    display_df = runs_df.drop(columns=["run_id_full"]).copy()
    display_df["rmse"] = display_df["rmse"].apply(lambda v: f"${v:,.2f}" if v else "-")
    display_df["mae"] = display_df["mae"].apply(lambda v: f"${v:,.2f}" if v else "-")
    display_df["r2"] = display_df["r2"].apply(lambda v: f"{v:.4f}" if v else "-")
    display_df["mape"] = display_df["mape"].apply(lambda v: f"{v:.2f}%" if v else "-")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ROAS FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab_roas:
    if not prediction_records:
        st.info("No prediction records found.")
        st.stop()

    # Flatten all prediction rows across batches
    all_preds: list[dict] = []
    for r in prediction_records:
        p = r["payload"]
        for row in p.get("predictions", []):
            row["_run_id"] = p.get("run_id", "")[:8]
            row["_prediction_id"] = p.get("prediction_id", "")[:8]
            all_preds.append(row)

    if not all_preds:
        st.info("Prediction payloads are empty.")
        st.stop()

    pred_df = pd.DataFrame(all_preds)
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    pred_df = pred_df.sort_values("date")

    # Run selector
    run_ids = pred_df["_run_id"].unique().tolist()
    selected_run = st.selectbox(
        "Select training run",
        options=["All"] + run_ids,
        index=0,
    )
    if selected_run != "All":
        pred_df = pred_df[pred_df["_run_id"] == selected_run]

    # ── ROAS time-series ────────────────────────────────────────────────────
    st.subheader("Actual vs Predicted ROAS — Test Period")

    fig_roas = go.Figure()
    fig_roas.add_trace(
        go.Scatter(
            x=pred_df["date"],
            y=pred_df["actual_roas"],
            name="Actual ROAS",
            mode="lines",
            line=dict(color="#636EFA", width=2),
        )
    )
    fig_roas.add_trace(
        go.Scatter(
            x=pred_df["date"],
            y=pred_df["predicted_roas"],
            name="Predicted ROAS",
            mode="lines",
            line=dict(color="#EF553B", width=2, dash="dot"),
        )
    )
    fig_roas.update_layout(
        xaxis_title="Date",
        yaxis_title="ROAS",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_roas, use_container_width=True)

    # ── Revenue time-series ─────────────────────────────────────────────────
    st.subheader("Actual vs Predicted Revenue — Test Period")
    fig_rev = go.Figure()
    fig_rev.add_trace(
        go.Scatter(
            x=pred_df["date"],
            y=pred_df["actual_revenue"],
            name="Actual Revenue",
            mode="lines",
            line=dict(color="#00CC96", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,204,150,0.08)",
        )
    )
    fig_rev.add_trace(
        go.Scatter(
            x=pred_df["date"],
            y=pred_df["predicted_revenue"],
            name="Predicted Revenue",
            mode="lines",
            line=dict(color="#AB63FA", width=2, dash="dot"),
        )
    )
    fig_rev.update_layout(
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        yaxis_tickprefix="$",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    col_dist, col_scatter = st.columns(2)

    # ── ROAS distribution ────────────────────────────────────────────────────
    with col_dist:
        st.subheader("ROAS Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=pred_df["actual_roas"],
                name="Actual",
                opacity=0.65,
                marker_color="#636EFA",
                nbinsx=20,
            )
        )
        fig_hist.add_trace(
            go.Histogram(
                x=pred_df["predicted_roas"],
                name="Predicted",
                opacity=0.65,
                marker_color="#EF553B",
                nbinsx=20,
            )
        )
        fig_hist.update_layout(barmode="overlay", xaxis_title="ROAS", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Scatter: actual vs predicted ─────────────────────────────────────────
    with col_scatter:
        st.subheader("Actual vs Predicted Revenue (Scatter)")
        fig_scatter = px.scatter(
            pred_df,
            x="actual_revenue",
            y="predicted_revenue",
            trendline="ols",
            labels={
                "actual_revenue": "Actual Revenue ($)",
                "predicted_revenue": "Predicted Revenue ($)",
            },
            color_discrete_sequence=["#19D3F3"],
        )
        min_v = pred_df[["actual_revenue", "predicted_revenue"]].min().min()
        max_v = pred_df[["actual_revenue", "predicted_revenue"]].max().max()
        fig_scatter.add_shape(
            type="line",
            x0=min_v, y0=min_v, x1=max_v, y1=max_v,
            line=dict(color="gray", dash="dash"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Summary table ────────────────────────────────────────────────────────
    st.subheader("Prediction Records")

    summary_cols = [
        "date", "marketing_spend", "actual_revenue", "predicted_revenue",
        "actual_roas", "predicted_roas",
    ]
    available_cols = [c for c in summary_cols if c in pred_df.columns]
    show_df = pred_df[available_cols].copy()
    for money_col in ["marketing_spend", "actual_revenue", "predicted_revenue"]:
        if money_col in show_df:
            show_df[money_col] = show_df[money_col].apply(lambda v: f"${v:,.2f}")
    for roas_col in ["actual_roas", "predicted_roas"]:
        if roas_col in show_df:
            show_df[roas_col] = show_df[roas_col].apply(lambda v: f"{v:.4f}")

    st.dataframe(show_df, use_container_width=True, hide_index=True)

    # ── Prediction batch summaries ───────────────────────────────────────────
    st.subheader("Prediction Batch Summaries")
    summaries = []
    for r in prediction_records:
        p = r["payload"]
        s = p.get("summary", {})
        summaries.append(
            {
                "Timestamp": fmt_ts(r["timestamp"]),
                "Run ID": p.get("run_id", "")[:8] + "…",
                "Rows": p.get("rows"),
                "Date Range": f"{p.get('date_range', {}).get('start')} → {p.get('date_range', {}).get('end')}",
                "Mean Actual ROAS": f"{s.get('mean_actual_roas', 0):.4f}",
                "Mean Pred ROAS": f"{s.get('mean_predicted_roas', 0):.4f}",
                "Mean Actual Rev ($)": f"${s.get('mean_actual_revenue', 0):,.2f}",
                "Mean Pred Rev ($)": f"${s.get('mean_predicted_revenue', 0):,.2f}",
            }
        )
    st.dataframe(pd.DataFrame(summaries), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RAW AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════
with tab_log:
    st.subheader("Raw Audit Events")

    # Filters
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        all_types = sorted({r["event_type"] for r in records})
        selected_types = st.multiselect(
            "Filter by event type",
            options=all_types,
            default=all_types,
        )
    with col_f2:
        search_term = st.text_input("Search (event_id / run_id / keyword)", "")

    filtered = [
        r for r in records
        if r["event_type"] in selected_types
        and (
            search_term == ""
            or search_term.lower() in json.dumps(r).lower()
        )
    ]

    def _summary(r: dict) -> str:
        p = r.get("payload", {})
        et = r["event_type"]
        if et == "data_generation":
            return f"{p.get('rows_generated')} rows, seed={p.get('seed')}"
        if et == "feature_engineering":
            return f"{p.get('input_rows')} -> {p.get('output_rows')} rows, added: {', '.join(p.get('features_added', []))}"
        if et == "training_run":
            m = p.get("metrics", {})
            return f"RMSE=${m.get('rmse', 0):,.2f}, R2={m.get('r2', 0):.4f}, iter={p.get('best_iteration')}"
        if et == "prediction":
            s = p.get("summary", {})
            return f"{p.get('rows')} rows, mean ROAS={s.get('mean_predicted_roas', 0):.4f}"
        return json.dumps(p)[:80]

    # Re-build with proper summary
    table_rows = []
    for r in filtered:
        table_rows.append(
            {
                "Timestamp": fmt_ts(r["timestamp"]),
                "Event Type": r["event_type"],
                "Event ID": r["event_id"][:8] + "…",
                "Summary": _summary(r),
            }
        )

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(filtered)} of {len(records)} total events.")

    # ── Timeline chart ───────────────────────────────────────────────────────
    st.subheader("Event Timeline")
    timeline_data = pd.DataFrame(
        [
            {
                "timestamp": r["timestamp"],
                "event_type": r["event_type"],
                "event_id": r["event_id"][:8],
            }
            for r in filtered
        ]
    )
    timeline_data["timestamp"] = pd.to_datetime(timeline_data["timestamp"])

    fig_timeline = px.scatter(
        timeline_data,
        x="timestamp",
        y="event_type",
        color="event_type",
        hover_data=["event_id"],
        labels={"timestamp": "Time (UTC)", "event_type": "Event Type"},
        color_discrete_sequence=px.colors.qualitative.Set2,
        height=300,
    )
    fig_timeline.update_traces(marker=dict(size=12, symbol="circle"))
    fig_timeline.update_layout(showlegend=False, yaxis_title="")
    st.plotly_chart(fig_timeline, use_container_width=True)

    # ── Full JSON expander ────────────────────────────────────────────────────
    with st.expander("View raw JSON records"):
        for r in filtered[-20:]:
            st.json(r)
        if len(filtered) > 20:
            st.caption(f"Showing last 20 of {len(filtered)} filtered records.")
