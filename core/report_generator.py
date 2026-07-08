"""
report_generator.py
====================
Phase 7 module — end-of-session HTML report with embedded charts.

Responsibility
--------------
Consume the CSV + JSON produced by SessionLogger and generate a single
self-contained HTML file containing:

    1. Session summary table  (from the JSON aggregates)
    2. Attention trend chart  (line — score over time)
    3. Gaze direction chart   (pie — % time in each h-direction)
    4. Blink / closure chart  (bar — events per 5-minute bucket)
    5. Face presence timeline (Gantt strip — present vs absent)
    6. Full per-tick data table (collapsible, for detailed inspection)
    7. Disclaimer footer

Charts are rendered headlessly with matplotlib's Agg backend and embedded
as base64 PNG strings — no temp files, no display required, no web server.

Dependencies
------------
    pandas     — DataFrame loading and column arithmetic
    matplotlib — chart generation (Agg backend)
    Standard library only for everything else

Usage
-----
    report_path = ReportGenerator(csv_path).export()
    print(f"Report saved to {report_path}")
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")            # must be set before any other matplotlib import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path




# ─────────────────────────────────────────────────────────────────────────────
# Chart appearance constants
# ─────────────────────────────────────────────────────────────────────────────

_FIGURE_DPI: int  = 100
_LINE_COLOR       = "#4A90D9"
_ATTENTIVE_COLOR  = "#27AE60"
_DISTRACTED_COLOR = "#E74C3C"
_FACE_PRESENT_COLOR = "#2ECC71"
_FACE_ABSENT_COLOR  = "#E74C3C"

# Attention category → colour for the trend chart background bands
_STATE_COLORS: dict[str, str] = {
    "HIGHLY_ATTENTIVE":    "#d4efdf",
    "ATTENTIVE":           "#a9dfbf",
    "PARTIALLY_ATTENTIVE": "#fef9e7",
    "DISTRACTED":          "#fde8d8",
    "HIGHLY_DISTRACTED":   "#fadbd8",
    "NO_FACE":             "#d5d8dc",
}

_DISCLAIMER = (
    "This score reflects visual engagement signals only — "
    "not a measurement of comprehension or mental focus."
)


# ─────────────────────────────────────────────────────────────────────────────
# ReportGenerator
# ─────────────────────────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Produces a self-contained HTML session report from a SessionLogger CSV file.

    Parameters
    ----------
    csv_path :
        Absolute or relative path to the session CSV produced byhtml_path = csv_path.replace(".csv", "_report.html")
        ``SessionLogger.close()``.  The JSON summary is derived automatically
        from the same stem: ``<stem>_summary.json``.

    Example
    -------
    ::

        report_path = ReportGenerator(csv_path).export()
    """

    def __init__(self, csv_path: str) -> None:
        self._csv_path  = os.path.abspath(csv_path)
        self._json_path = self._csv_path.replace(".csv", "_summary.json")

        csv = Path(self._csv_path)

        project_root = csv.parent.parent
        reports_dir = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)

        self._report_path = reports_dir / f"{csv.stem}_report.html"

        self._df: Optional[pd.DataFrame] = None
        self._summary: Optional[dict]   = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def export(self, format: str = "html") -> str:
        """
        Generate and write the report.

        Parameters
        ----------
        format :
            Currently only ``"html"`` is supported.

        Returns
        -------
        str
            Absolute path to the written report file.

        Raises
        ------
        FileNotFoundError
            If the CSV or JSON summary does not exist.
        ValueError
            If ``format`` is not ``"html"``.
        """
        if format != "html":
            raise ValueError(f"Unsupported report format: {format!r}. Only 'html' is supported.")

        self._load_data()
        html = self._build_html()

        with self._report_path.open("w", encoding="utf-8") as fh:
            fh.write(html)

        return str(self._report_path.resolve())

    def generate_summary(self) -> dict:
        """
        Return the parsed JSON summary dict (loaded from the companion file).

        Useful for programmatic access to session statistics without
        producing a full HTML report.
        """
        self._load_data()
        return dict(self._summary)

    def generate_charts(self, output_dir: str) -> list[str]:
        """
        Save the four chart PNGs to ``output_dir`` and return their paths.

        Provided for programmatic use (e.g. embedding charts in an external
        document or report builder).
        """
        self._load_data()
        os.makedirs(output_dir, exist_ok=True)
        paths: list[str] = []
        for name, fig in [
            ("attention_trend",    self._chart_attention_trend()),
            ("gaze_distribution",  self._chart_gaze_distribution()),
            ("blink_events",       self._chart_blink_events()),
            ("face_timeline",      self._chart_face_timeline()),
        ]:
            stem = os.path.splitext(os.path.basename(self._csv_path))[0]
            path = os.path.join(output_dir, f"{stem}_{name}.png")
            fig.savefig(path, dpi=_FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            paths.append(os.path.abspath(path))
        return paths

    # ─────────────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """Load CSV and JSON exactly once (idempotent)."""
        if self._df is not None:
            return

        if not os.path.exists(self._csv_path):
            raise FileNotFoundError(f"Session CSV not found: {self._csv_path}")
        if not os.path.exists(self._json_path):
            raise FileNotFoundError(f"Session JSON summary not found: {self._json_path}")

        self._df = pd.read_csv(self._csv_path)

        # Coerce numeric columns that may have been saved as strings
        for col in ("attention_score", "raw_score", "session_elapsed_sec",
                    "head_yaw", "head_pitch", "head_roll",
                    "gaze_h_ratio", "gaze_v_ratio"):
            if col in self._df.columns:
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce")

        # Boolean coercion (CSV stores True/False as strings)
        for col in ("face_present", "head_forward"):
            if col in self._df.columns:
                self._df[col] = self._df[col].map(
                    lambda v: str(v).strip().lower() in ("true", "1", "yes")
                )

        with open(self._json_path, encoding="utf-8") as fh:
            self._summary = json.load(fh)

    # ─────────────────────────────────────────────────────────────────────────
    # Chart generators — each returns a closed plt.Figure
    # ─────────────────────────────────────────────────────────────────────────

    def _chart_attention_trend(self) -> plt.Figure:
        """
        Line chart: attention score over session time.

        Background bands show which attention category each score falls in.
        """
        fig, ax = plt.subplots(figsize=(10, 3.5))
        df = self._df

        # Coloured band backgrounds for the five score ranges
        bands = [
            (90, 100, _STATE_COLORS["HIGHLY_ATTENTIVE"],    "Highly Attentive"),
            (75,  90, _STATE_COLORS["ATTENTIVE"],           "Attentive"),
            (50,  75, _STATE_COLORS["PARTIALLY_ATTENTIVE"], "Partially Attentive"),
            (25,  50, _STATE_COLORS["DISTRACTED"],          "Distracted"),
            ( 0,  25, _STATE_COLORS["HIGHLY_DISTRACTED"],   "Highly Distracted"),
        ]
        for ylo, yhi, color, _ in bands:
            ax.axhspan(ylo, yhi, alpha=0.35, color=color, linewidth=0)

        t = df["session_elapsed_sec"]
        ax.plot(t, df["attention_score"], color=_LINE_COLOR, linewidth=1.5,
                label="Smoothed score")
        if "raw_score" in df.columns:
            ax.plot(t, df["raw_score"], color=_LINE_COLOR, linewidth=0.6,
                    alpha=0.35, linestyle="--", label="Raw score")

        ax.set_xlim(t.min(), t.max())
        ax.set_ylim(0, 100)
        ax.set_xlabel("Session elapsed (s)")
        ax.set_ylabel("Attention score")
        ax.set_title("Attention Score Over Session")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        return fig

    def _chart_gaze_distribution(self) -> plt.Figure:
        """
        Pie chart: % time in each horizontal gaze direction.
        """
        fig, ax = plt.subplots(figsize=(5, 4))
        df = self._df

        if "gaze_h_direction" not in df.columns:
            ax.text(0.5, 0.5, "No gaze data", ha="center", va="center",
                    transform=ax.transAxes)
            fig.tight_layout()
            return fig

        counts = df["gaze_h_direction"].value_counts()
        colors = {
            "CENTER": "#27AE60",
            "LEFT":   "#2980B9",
            "RIGHT":  "#8E44AD",
        }
        wedge_colors = [colors.get(lbl, "#95A5A6") for lbl in counts.index]
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
               colors=wedge_colors, startangle=90,
               wedgeprops={"linewidth": 0.8, "edgecolor": "white"})
        ax.set_title("Horizontal Gaze Direction Distribution")
        fig.tight_layout()
        return fig

    def _chart_blink_events(self) -> plt.Figure:
        """
        Bar chart: cumulative blink and long-closure counts per 5-minute bucket.

        Uses the delta of the cumulative counter column so each bar shows
        events that occurred within that 5-minute window, not the total to date.
        """
        fig, ax = plt.subplots(figsize=(8, 3.5))
        df = self._df.copy()

        bucket_s  = 300   # 5 minutes
        max_t     = df["session_elapsed_sec"].max()

        if max_t < bucket_s:
            # Session shorter than one bucket — show totals as a single bar
            bucket_s = max(1, int(max_t))

        df["bucket"] = (df["session_elapsed_sec"] // bucket_s).astype(int)

        blink_col   = "blink_count_cumulative"
        closure_col = "long_closure_events_cumulative"

        def _events_in_bucket(col: str) -> pd.Series:
            if col not in df.columns:
                return pd.Series(dtype=float)
            last_per_bucket  = df.groupby("bucket")[col].last()
            first_per_bucket = df.groupby("bucket")[col].first()
            # Blinks in a bucket = last cumulative value − first cumulative value
            return (last_per_bucket - first_per_bucket).clip(lower=0)

        blink_counts   = _events_in_bucket(blink_col)
        closure_counts = _events_in_bucket(closure_col)

        x_labels = [
            f"{int(b * bucket_s // 60)}–{int((b + 1) * bucket_s // 60)} min"
            for b in blink_counts.index
        ]
        x = range(len(x_labels))

        bar_w = 0.4
        ax.bar([i - bar_w / 2 for i in x], blink_counts.values,
               width=bar_w, label="Blinks", color="#2980B9", alpha=0.85)
        ax.bar([i + bar_w / 2 for i in x], closure_counts.values,
               width=bar_w, label="Long closures", color="#E74C3C", alpha=0.85)

        ax.set_xticks(list(x))
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Event count")
        ax.set_title("Blink & Eye Closure Events (per 5-minute window)")
        ax.legend()
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        return fig

    def _chart_face_timeline(self) -> plt.Figure:
        """
        Horizontal Gantt-style strip: face present (green) vs absent (red)
        over the session timeline.
        """
        fig, ax = plt.subplots(figsize=(10, 1.4))
        df = self._df

        if "face_present" not in df.columns:
            ax.text(0.5, 0.5, "No face-presence data", ha="center", va="center",
                    transform=ax.transAxes)
            fig.tight_layout()
            return fig

        # Build contiguous runs
        t       = df["session_elapsed_sec"].values
        present = df["face_present"].values

        # Each second is one tick; fill rectangles for each run
        for i, (tp, fp) in enumerate(zip(t, present)):
            color = _FACE_PRESENT_COLOR if fp else _FACE_ABSENT_COLOR
            ax.barh(0, 1.0, left=tp, height=1, color=color, linewidth=0)

        max_t = float(t.max()) if len(t) else 1.0
        ax.set_xlim(0, max_t)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("Session elapsed (s)")
        ax.set_title("Face Presence Timeline")

        legend_patches = [
            mpatches.Patch(color=_FACE_PRESENT_COLOR, label="Face present"),
            mpatches.Patch(color=_FACE_ABSENT_COLOR,  label="Face absent"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
        fig.tight_layout()
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # HTML assembly
    # ─────────────────────────────────────────────────────────────────────────

    def _fig_to_b64(self, fig: plt.Figure) -> str:
        """Render a Figure to a base64-encoded PNG string and close the Figure."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=_FIGURE_DPI, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        plt.close(fig)
        return b64

    def _build_html(self) -> str:
        """Assemble the complete self-contained HTML report string."""
        s      = self._summary
        df     = self._df

        # ── Charts → inline base64 ─────────────────────────────────────
        trend_b64    = self._fig_to_b64(self._chart_attention_trend())
        gaze_b64     = self._fig_to_b64(self._chart_gaze_distribution())
        blink_b64    = self._fig_to_b64(self._chart_blink_events())
        timeline_b64 = self._fig_to_b64(self._chart_face_timeline())

        # ── Summary table rows ─────────────────────────────────────────
        dur_s    = s.get("session_duration_sec", 0)
        dur_fmt  = f"{int(dur_s // 60)}m {int(dur_s % 60)}s"

        dist     = s.get("attention_state_distribution_pct", {})
        dist_fmt = "  |  ".join(
            f"{k.replace('_', ' ').title()}: {v:.1f}%"
            for k, v in sorted(dist.items(), key=lambda x: -x[1])
        )

        summary_rows = [
            ("Session start",          s.get("session_start", "—")),
            ("Duration",               dur_fmt),
            ("Ticks logged",           str(s.get("ticks_logged", 0))),
            ("Average attention",      f"{s.get('average_attention_score', 0):.1f} / 100"),
            ("Min / Max attention",
             f"{s.get('minimum_attention_score', 0):.1f} / {s.get('maximum_attention_score', 0):.1f}"),
            ("Time attentive",
             f"{s.get('time_attentive_s', 0):.0f} s  ({s.get('time_attentive_s',0)/max(dur_s,1)*100:.1f}%)"),
            ("Time distracted",
             f"{s.get('time_distracted_s', 0):.0f} s  ({s.get('time_distracted_s',0)/max(dur_s,1)*100:.1f}%)"),
            ("Longest distracted streak",
             f"{s.get('longest_distracted_streak_s', 0):.0f} s"),
            ("State distribution",     dist_fmt),
            ("Total blinks",           str(s.get("total_blinks", 0))),
            ("Avg blink rate",         f"{s.get('avg_blink_rate_per_min', 0):.1f} blinks/min"),
            ("Long eye closures",      str(s.get("long_eye_closure_events", 0))),
            ("Time gaze at screen",    f"{s.get('pct_time_gaze_center', 0):.1f}%"),
            ("Face loss duration",
             f"{s.get('face_loss_duration_s', 0):.0f} s"),
        ]
        summary_html = "\n".join(
            f"<tr><td><b>{label}</b></td><td>{value}</td></tr>"
            for label, value in summary_rows
        )

        # ── Full data table (collapsible) ──────────────────────────────
        col_headers = "".join(f"<th>{c}</th>" for c in df.columns)
        data_rows   = ""
        for _, row in df.iterrows():
            data_rows += "<tr>" + "".join(
                f"<td>{row[c]}</td>" for c in df.columns
            ) + "</tr>\n"

        # ── Assemble HTML ──────────────────────────────────────────────
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Session Report — {s.get("session_start","")}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px;
         background: #f9f9f9; color: #222; }}
  h1   {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
  h2   {{ color: #34495e; margin-top: 32px; }}
  table.summary {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
  table.summary td {{ padding: 7px 12px; border: 1px solid #ddd; }}
  table.summary tr:nth-child(even) {{ background: #ecf0f1; }}
  .chart {{ text-align: center; margin: 16px 0; }}
  .chart img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px;
               box-shadow: 0 1px 4px rgba(0,0,0,0.12); }}
  details summary {{ cursor: pointer; font-weight: bold; color: #2980b9; margin-top: 20px; }}
  table.data {{ border-collapse: collapse; font-size: 0.78em; width: 100%; }}
  table.data th {{ background: #2c3e50; color: white; padding: 5px 7px; white-space: nowrap; }}
  table.data td {{ padding: 4px 7px; border: 1px solid #ccc; white-space: nowrap; }}
  table.data tr:nth-child(even) {{ background: #f0f3f4; }}
  .disclaimer {{ font-size: 0.85em; color: #7f8c8d; border-top: 1px solid #ccc;
                 margin-top: 32px; padding-top: 10px; font-style: italic; }}
</style>
</head>
<body>

<h1>Student Visual Engagement Session Report</h1>

<h2>Session Summary</h2>
<table class="summary">
  <tbody>
    {summary_html}
  </tbody>
</table>

<h2>Attention Score Over Time</h2>
<div class="chart">
  <img src="data:image/png;base64,{trend_b64}" alt="Attention trend chart">
</div>

<h2>Gaze Direction Distribution</h2>
<div class="chart">
  <img src="data:image/png;base64,{gaze_b64}" alt="Gaze distribution chart">
</div>

<h2>Blink &amp; Eye Closure Events</h2>
<div class="chart">
  <img src="data:image/png;base64,{blink_b64}" alt="Blink events chart">
</div>

<h2>Face Presence Timeline</h2>
<div class="chart">
  <img src="data:image/png;base64,{timeline_b64}" alt="Face presence timeline">
</div>

<details>
  <summary>Full Per-Tick Data Table ({len(df)} rows — click to expand)</summary>
  <div style="overflow-x:auto; margin-top: 10px;">
    <table class="data">
      <thead><tr>{col_headers}</tr></thead>
      <tbody>
        {data_rows}
      </tbody>
    </table>
  </div>
</details>

<p class="disclaimer">{_DISCLAIMER}</p>

</body>
</html>"""
        return html
