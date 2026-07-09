"""
session_logger.py
=================
Phase 6 module — per-second session logger.

Responsibility
--------------
Record one row of attention metrics per second during an active session.
On close(), flush all rows to a timestamped CSV and write a companion JSON
summary.  Performs NO computer vision and holds NO references to MediaPipe.

Design decisions
----------------
* In-memory list for accumulation:
    At 1 Hz, a 60-minute session is 3 600 rows.  Each row is ~30 fields ×
    average 10 chars = ~300 bytes → ~1 MB peak RAM.  No deque cap needed.
* No disk I/O on the hot path:
    Every log_tick() call is an O(1) list append.  flush() and close() write
    to disk, so they must NOT be called from the frame loop.
* CSV as primary format:
    Human-readable, directly loadable with pandas for Phase 7 (ReportGenerator).
* JSON summary as companion:
    Nested structure suits the session-level aggregates that are awkward in CSV.

Usage (from main.py)
--------------------
    logger = SessionLogger()                # auto-names output

    # Inside the frame loop, throttled to ~1 Hz:
    if attention_eng.calibration.is_complete and timestamp_s - _last_log_s >= 1.0:
        logger.log_tick(attention_result, blink_result, pose_result,
                        gaze_result, processor.is_valid, timestamp_s)
        _last_log_s = timestamp_s

    # In the finally block:
    csv_path = logger.close()
    if csv_path:
        print(f"[INFO] Session saved → {csv_path}")
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Optional

from .attention_engine import AttentionResult, SessionStatistics
from .blink_detector import BlinkResult
from .gaze_estimator import GazeResult
from .head_pose_estimator import HeadPoseResult


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR: str = "sessions"
"""Directory created automatically if absent."""

# Ordered list of CSV column names — defines file schema and column order.
# Adding a column here is the ONLY change needed to extend the schema.
_CSV_COLUMNS: tuple[str, ...] = (
    # ── Time ──────────────────────────────────────────────────────
    "timestamp",
    "session_elapsed_sec",
    # ── Attention ─────────────────────────────────────────────────
    "attention_score",          # smoothed EMA score 0-100
    "raw_score",                # unsmoothed frame score 0-100
    "attention_category",       # AttentionState enum value
    # ── Gaze ──────────────────────────────────────────────────────
    "gaze_h_direction",
    "gaze_v_direction",
    "gaze_h_ratio",
    "gaze_v_ratio",
    "gaze_penalty",               # penalty fraction 0-1
    # ── Blink / eye closure ───────────────────────────────────────
    "eye_state",
    "blink_count_cumulative",
    "closure_duration_s",
    "last_closure_type",
    "long_closure_events_cumulative",
    "blink_penalty",              # penalty fraction 0-1
    # ── Head pose ─────────────────────────────────────────────────
    "head_yaw",
    "head_pitch",
    "head_roll",
    "head_forward",
    "head_penalty",               # penalty fraction 0-1
    # ── Face presence ─────────────────────────────────────────────
    "face_present",
    "face_penalty",               # penalty fraction 0-1
    # ── Running session aggregates ────────────────────────────────
    "avg_attention_so_far",
    "time_attentive_s",
    "time_distracted_s",
    "face_loss_s",
)


# ─────────────────────────────────────────────────────────────────────────────
# SessionLogger
# ─────────────────────────────────────────────────────────────────────────────

class SessionLogger:
    """
    Write-on-demand session logger.  Thread-unsafe by design (single-threaded
    pipeline; no lock overhead needed).

    Typical lifecycle
    -----------------
    1. Instantiate once at startup.
    2. Call log_tick() ~1 Hz from the main loop (only after calibration).
    3. Call close() in the finally block to flush CSV and write JSON summary.
    """

    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
        """
        Parameters
        ----------
        output_dir :
            Directory for output files.  Created automatically if absent.
            Both the CSV and JSON summary land here with matching timestamps.
        """
        os.makedirs(output_dir, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path  = os.path.join(output_dir, f"session_{stamp}.csv")
        self._json_path = os.path.join(output_dir, f"session_{stamp}_summary.json")

        # Wall-clock session start for the JSON summary header
        self._session_start_wall: str = datetime.now().isoformat(timespec="seconds")

        # In-memory row buffer — the only allocation per tick is one dict append
        self._rows: list[dict] = []

        # Snapshot of the last statistics object received (for close-time summary)
        self._last_stats: Optional[SessionStatistics] = None
        self._last_elapsed_s: float = 0.0

        # Per-state tick counts for the summary distribution table
        self._state_ticks: dict[str, int] = {}
        self._gaze_center_ticks: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def log_tick(
        self,
        attention_result:  AttentionResult,
        blink_result:      BlinkResult,
        head_pose_result:  HeadPoseResult,
        gaze_result:       GazeResult,
        face_present:      bool,
        session_elapsed_s: float,
    ) -> None:
        """
        Append one row to the in-memory log.

        Call approximately once per second from main.py, AFTER calibration is
        complete and ONLY when ~1.0 second has elapsed since the previous call.
        No disk I/O occurs here.

        Parameters
        ----------
        attention_result :
            Current frame's AttentionResult (carries statistics snapshot).
        blink_result, head_pose_result, gaze_result :
            Current outputs from Phase 2-4 modules.
        face_present :
            LandmarkProcessor.is_valid for this tick.
        session_elapsed_s :
            Monotonic seconds elapsed since session_start_s in main.py
            (``timestamp_s`` as computed from timestamp_ms).
        """
        stats = attention_result.statistics

        row: dict = {
            # ── Time ──────────────────────────────────────────────
            "timestamp":                    datetime.now().isoformat(timespec="seconds"),
            "session_elapsed_sec":          round(session_elapsed_s, 1),
            # ── Attention ─────────────────────────────────────────
            "attention_score":              attention_result.smoothed_score,
            "raw_score":                    attention_result.raw_score,
            "attention_category":           attention_result.state.value,
            # ── Gaze ──────────────────────────────────────────────
            "gaze_h_direction":             gaze_result.horizontal_direction.value,
            "gaze_v_direction":             gaze_result.vertical_direction.value,
            "gaze_h_ratio":                 round(gaze_result.horizontal_ratio, 3),
            "gaze_v_ratio":                 round(gaze_result.vertical_ratio, 3),
            "gaze_penalty":                 attention_result.gaze_penalty,
            # ── Blink / eye closure ───────────────────────────────
            "eye_state":                    blink_result.eye_state.value,
            "blink_count_cumulative":       blink_result.blink_count,
            "closure_duration_s":           blink_result.closure_duration_s,
            "last_closure_type":            blink_result.last_closure_type.value,
            "long_closure_events_cumulative": stats.long_eye_closures,
            "blink_penalty":                attention_result.blink_penalty,
            # ── Head pose ─────────────────────────────────────────
            "head_yaw":                     head_pose_result.yaw,
            "head_pitch":                   head_pose_result.pitch,
            "head_roll":                    head_pose_result.roll,
            "head_forward":                 head_pose_result.is_forward,
            "head_penalty":                 attention_result.head_penalty,
            # ── Face presence ─────────────────────────────────────
            "face_present":                 face_present,
            "face_penalty":                 attention_result.face_penalty,
            # ── Running aggregates ────────────────────────────────
            "avg_attention_so_far":         round(stats.average_attention, 1),
            "time_attentive_s":             round(stats.time_attentive_s, 1),
            "time_distracted_s":            round(stats.time_distracted_s, 1),
            "face_loss_s":                  round(stats.face_loss_duration_s, 1),
        }

        self._rows.append(row)

        # Update summary accumulators
        self._last_stats     = stats
        self._last_elapsed_s = session_elapsed_s

        state_key = attention_result.state.value
        self._state_ticks[state_key] = self._state_ticks.get(state_key, 0) + 1

        if gaze_result.is_looking_center():
            self._gaze_center_ticks += 1

    def flush(self) -> None:
        """
        Write all buffered rows to the CSV file.

        Safe to call mid-session for crash resilience.  Each call overwrites
        the file (avoids partial-write corruption from incremental appends).
        No-op if no ticks have been logged yet.
        """
        if not self._rows:
            return

        with open(self._csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)

    def close(self) -> str:
        """
        Finalise the session: flush the CSV and write the JSON summary.

        Returns
        -------
        str
            Absolute path to the CSV file, or ``""`` if no ticks were logged
            (caller can skip printing/opening an empty file).
        """
        if not self._rows:
            return ""

        self.flush()
        self._write_json_summary()

        return os.path.abspath(self._csv_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Read-only properties (useful for live display or testing)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def tick_count(self) -> int:
        """Number of ticks logged so far."""
        return len(self._rows)

    @property
    def csv_path(self) -> str:
        """Absolute path the CSV will be written to."""
        return os.path.abspath(self._csv_path)

    @property
    def json_path(self) -> str:
        """Absolute path the JSON summary will be written to."""
        return os.path.abspath(self._json_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _write_json_summary(self) -> None:
        """
        Produce the session-level summary JSON.

        All values are drawn from the last received ``SessionStatistics``
        snapshot and the per-state tick counters accumulated in log_tick().
        No re-processing of CSV rows is needed.
        """
        total_ticks   = len(self._rows)        # guaranteed > 0 here
        elapsed_s     = self._last_elapsed_s
        stats         = self._last_stats

        # Per-state distribution as % of logged ticks
        state_pct: dict[str, float] = {
            state: round(count / total_ticks * 100.0, 1)
            for state, count in self._state_ticks.items()
        }

        # Gaze-at-screen %
        pct_center = round(self._gaze_center_ticks / total_ticks * 100.0, 1)

        # Blink rate (blinks per minute over the whole session)
        total_blinks = stats.blink_count if stats else 0
        blink_rate   = round(
            total_blinks / max(elapsed_s / 60.0, 1e-6), 1
        )

        summary: dict = {
            # ── Session metadata ──────────────────────────────────
            "session_start":                  self._session_start_wall,
            "session_duration_sec":           round(elapsed_s, 1),
            "ticks_logged":                   total_ticks,
            # ── Attention ─────────────────────────────────────────
            "average_attention_score":        round(stats.average_attention,  1) if stats else 0.0,
            "minimum_attention_score":        round(stats.minimum_attention,  1) if stats else 0.0,
            "maximum_attention_score":        round(stats.maximum_attention,  1) if stats else 0.0,
            "time_attentive_s":               round(stats.time_attentive_s,   1) if stats else 0.0,
            "time_distracted_s":              round(stats.time_distracted_s,  1) if stats else 0.0,
            "longest_distracted_streak_s":    round(stats.longest_distracted_streak_s, 1) if stats else 0.0,
            "attention_state_distribution_pct": state_pct,
            # ── Blink ─────────────────────────────────────────────
            "total_blinks":                   total_blinks,
            "avg_blink_rate_per_min":         blink_rate,
            "long_eye_closure_events":        stats.long_eye_closures if stats else 0,
            # ── Gaze ──────────────────────────────────────────────
            "pct_time_gaze_center":           pct_center,
            # ── Face ──────────────────────────────────────────────
            "face_loss_duration_s":           round(stats.face_loss_duration_s, 1) if stats else 0.0,
            # ── Disclosure ────────────────────────────────────────
            "disclaimer": (
                "This score reflects visual engagement signals only — "
                "not a measurement of comprehension or mental focus."
            ),
        }

        with open(self._json_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
