"""sterileProcessing - Time-series signal peak/AUC analysis."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_io import find_signal_columns, find_time_column, load_dataframe
from signal_processing import analyze_peaks, detect_peaks_and_troughs

# --- Brand palette (blue / black / grey only) ---
COLOR_BG_MAIN = "#242422"        # main background
COLOR_BG_SECONDARY = "#3A3A3A"   # secondary background (grey)
COLOR_TEXT = "#E9E9E9"           # text / light
COLOR_ACCENT = "#9DBCD1"         # accent / brand (light blue)
COLOR_SUPPORTING = "#3A3A3A"     # supporting - chart panel surface (grey)

COLOR_SURFACE = COLOR_SUPPORTING
COLOR_GRID = "rgba(233, 233, 233, 0.12)"
COLOR_AXIS = "rgba(233, 233, 233, 0.35)"
COLOR_INK = COLOR_TEXT
COLOR_SIGNAL = COLOR_ACCENT                      # raw signal
COLOR_SIGNAL_FILL = "rgba(157, 188, 209, 0.25)"  # translucent accent - AUC shading
COLOR_PEAK = COLOR_TEXT                          # detected peak - bright highlight
COLOR_TROUGH = "rgba(233, 233, 233, 0.5)"        # baseline anchor points
COLOR_BASELINE = "rgba(233, 233, 233, 0.55)"     # baseline segment

st.set_page_config(page_title="Sterile Processing", layout="wide")

st.markdown(
    f"""
    <style>
    .stMultiSelect [data-baseweb="tag"] {{
        background-color: {COLOR_BG_SECONDARY} !important;
        color: {COLOR_ACCENT} !important;
    }}
    .stMultiSelect [data-baseweb="tag"] span {{
        color: {COLOR_ACCENT} !important;
    }}
    .stMultiSelect [data-baseweb="tag"] svg {{
        fill: {COLOR_ACCENT} !important;
    }}
    [data-testid="stPlotlyChart"] {{
        border-radius: 0.75rem;
        overflow: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

def banner(html_content: str) -> None:
    st.markdown(
        f"""
        <div style="
            background-color: rgba(157, 188, 209, 0.15);
            border-left: 4px solid {COLOR_ACCENT};
            color: {COLOR_TEXT};
            padding: 0.75rem 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        ">
            {html_content}
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
    <h1 style="font-size: 2.025rem; margin-bottom: 0;">Sterile Processing</h1>
    <p style="margin-top: 0; margin-bottom: 0.5rem; font-size: 0.875rem; color: rgba(233, 233, 233, 0.6);">
        Time-series signal peak detection, dynamic baselining, and net AUC analysis.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1. Data")
    uploaded_file = st.file_uploader("Upload .xlsx or .csv", type=["xlsx", "csv"])

    st.header("2. Peak detection")
    prominence_pct = st.slider(
        "Peak sensitivity (min prominence, % of signal range)",
        min_value=1, max_value=50, value=8, step=1,
        help="Lower values detect smaller/noisier peaks; higher values only detect dominant peaks.",
    )
    distance = st.slider(
        "Minimum spacing between peaks/troughs (samples)",
        min_value=1, max_value=20, value=2, step=1,
    )
    boundary_pct = st.slider(
        "Peak Boundary Sensitivity (Noise Cutoff, % of peak prominence)",
        min_value=1, max_value=50, value=10, step=1,
        help=(
            "How far the signal must decay back toward its local floor before a peak's "
            "start/end boundary is placed there. Lower values push the boundary further "
            "out (wider AUC window); higher values pull it in tighter around the apex."
        ),
    )

if uploaded_file is None:
    banner("Upload a time-series file to begin. Expected: a 'time' column plus one column per experimental condition.")
    st.stop()

try:
    df = load_dataframe(uploaded_file)
except Exception as exc:
    st.error(f"Could not read file: {exc}")
    st.stop()

time_col = find_time_column(df)
signal_cols = find_signal_columns(df, time_col)

if not signal_cols:
    st.warning("No usable Y-data columns found (everything is constant at 1, or the file has only a time column).")
    st.stop()

banner(
    f"""Loaded <b>{len(df)}</b> rows. X-axis: <b>{time_col}</b>.
    Found <b>{len(signal_cols)}</b> experimental condition column(s)
    ({df.shape[1] - 1 - len(signal_cols)} constant-1 column(s) dropped)."""
)

with st.sidebar:
    st.header("3. Columns to plot")
    st.caption("Pick which experimental condition column(s) to analyze.")
    chosen_cols = st.multiselect("Experimental conditions", options=signal_cols, default=[])

if not chosen_cols:
    banner("Select at least one column from the sidebar to view its analysis.")
    st.stop()

x_full = pd.to_numeric(df[time_col], errors="coerce").to_numpy()

all_summaries = []

for col in chosen_cols:
    y_full = pd.to_numeric(df[col], errors="coerce").to_numpy()
    mask = ~np.isnan(x_full) & ~np.isnan(y_full)
    x = x_full[mask]
    y = y_full[mask]

    if len(x) < 3:
        st.subheader(col)
        st.warning("Not enough valid data points to analyze.")
        continue

    y_range = float(np.max(y) - np.min(y))
    prominence = max(y_range * (prominence_pct / 100.0), 1e-9)

    peak_idx, trough_idx = detect_peaks_and_troughs(y, prominence=prominence, distance=distance)
    peaks = analyze_peaks(x, y, peak_idx, trough_idx, boundary_threshold=boundary_pct / 100.0)

    st.subheader(col)

    fig = go.Figure()

    # Raw signal
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", name="Signal",
            line=dict(color=COLOR_SIGNAL, width=2),
            hovertemplate=f"{time_col}: %{{x}}<br>{col}: %{{y:.4f}}<extra></extra>",
        )
    )

    # Trough anchor points (baseline endpoints), deduplicated
    if peaks:
        trough_pts = sorted({(p.x_start, p.y_start) for p in peaks} | {(p.x_end, p.y_end) for p in peaks})
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in trough_pts], y=[p[1] for p in trough_pts],
                mode="markers", name="Baseline anchor (trough)",
                marker=dict(color=COLOR_TROUGH, size=7, symbol="circle-open", line=dict(width=1.5)),
                hoverinfo="skip",
            )
        )

    for i, p in enumerate(peaks):
        show_legend = i == 0
        # Baseline segment (drawn first so the shaded trace can fill to it)
        fig.add_trace(
            go.Scatter(
                x=p.segment_x, y=p.baseline_y, mode="lines", name="Baseline",
                legendgroup="baseline", showlegend=show_legend,
                line=dict(color=COLOR_BASELINE, width=1.5, dash="dash"),
                hoverinfo="skip",
            )
        )
        # Shaded AUC region (fills to the baseline trace above)
        fig.add_trace(
            go.Scatter(
                x=p.segment_x, y=p.shaded_y, mode="lines", name="AUC area",
                legendgroup="auc", showlegend=show_legend,
                line=dict(width=0), fill="tonexty", fillcolor=COLOR_SIGNAL_FILL,
                hoverinfo="skip",
            )
        )

    if peaks:
        fig.add_trace(
            go.Scatter(
                x=[p.peak_x for p in peaks], y=[p.peak_y for p in peaks],
                mode="markers", name="Peak",
                marker=dict(color=COLOR_PEAK, size=11, symbol="triangle-up", line=dict(width=1, color=COLOR_SURFACE)),
                hovertemplate=f"Peak<br>{time_col}: %{{x}}<br>{col}: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        plot_bgcolor=COLOR_SURFACE,
        paper_bgcolor=COLOR_SURFACE,
        font=dict(color=COLOR_INK),
        xaxis=dict(title=time_col, gridcolor=COLOR_GRID, linecolor=COLOR_AXIS, zeroline=False),
        yaxis=dict(title=col, gridcolor=COLOR_GRID, linecolor=COLOR_AXIS, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=40, l=60, r=20, b=40),
        height=400,
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True, key=f"chart_{col}")

    if not peaks:
        st.caption("No peaks detected at the current sensitivity settings.")
        continue

    summary_df = pd.DataFrame(
        [
            {
                "Peak #": p.peak_number,
                "Peak Time": round(p.peak_x, 4),
                "Peak Height": round(p.peak_y, 4),
                "Baseline Range (Start - End Time)": f"{p.x_start:.4f} - {p.x_end:.4f}",
                "Net AUC": round(p.auc, 6),
            }
            for p in peaks
        ]
    )
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    export_df = summary_df.copy()
    export_df.insert(0, "Condition", col)
    all_summaries.append(export_df)

if all_summaries:
    combined = pd.concat(all_summaries, ignore_index=True)
    st.sidebar.header("4. Export")
    st.sidebar.download_button(
        "Download peak summary (CSV)",
        data=combined.to_csv(index=False).encode("utf-8"),
        file_name="peak_auc_summary.csv",
        mime="text/csv",
    )
