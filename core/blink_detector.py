"""
blink_detector.py
=================
Phase 2 module — Eye Aspect Ratio (EAR) blink and eye-closure detector.

Mathematics — Eye Aspect Ratio (Soukupová & Čech, 2016)
---------------------------------------------------------
Six landmarks per eye (p1..p6) placed at the corners and eyelid edges:

                    p2 ───── p3
                   /            |
        p1 -----                  ----- p4      <- horizontal (open)
                   |            /
                    p6 ───── p5

    EAR = ( ‖p2 – p6‖ + ‖p3 – p5‖ ) / ( 2 · ‖p1 – p4‖ )

Typical values:
    Open eye   : 0.25 – 0.35   (numerator is large)
    Blinking   : 0.10 – 0.20   (numerator shrinking)
    Closed eye : 0.00 – 0.08   (numerator near zero)

The ratio is dimensionless and scale-invariant (pixel-to-pixel distances
cancel out), so it does not change when the user moves closer to or further
from the camera.

Both eyes are computed and averaged:
    avg_EAR = (EAR_left + EAR_right) / 2

Averaging across both eyes reduces sensitivity to single-eye landmark noise
from glasses reflections, partial face shadows, or minor face rotation.

State machine
-------------

    OPEN  ─── EAR drops below threshold ───▶  CLOSED
    OPEN  ◀── EAR rises above threshold ───   CLOSED
                                          ↑
                         classify + count EXACTLY HERE (CLOSED→OPEN edge)

Rules:
  • While CLOSED: nothing is counted or classified.  Holding eyes shut does
    NOT accumulate blink_count.
  • At CLOSED→OPEN: the closure duration is measured and classified once.
    < 0.4 s   → BLINK         (normal involuntary blink, blink_count += 1)
    0.4–2.0 s → LONG_CLOSURE  (drowsiness/deliberate, flagged but not counted)
    ≥ 2.0 s   → POSSIBLE_SLEEP (significant flag, future scoring will penalize)

Face loss during closure:
  If the face disappears mid-blink (detection dropout from fast head turn or
  bad lighting), the state machine resets to OPEN rather than freezing in
  CLOSED, which would trigger a phantom long-closure or sleep event when the
  face re-appears.

Dependencies
------------
    landmark_processor.py  — must be updated for the same frame before calling
                             BlinkDetector.update().

Landmark indices
----------------
    Imported from ``landmark_processor`` — no index constants are duplicated
    here.  This enforces the single-source-of-truth rule from the design doc.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .landmark_processor import LandmarkProcessor, LEFT_EYE_EAR, RIGHT_EYE_EAR


# ─────────────────────────────────────────────────────────────────────────────
# Threshold constants
# Isolated here so Phase 5 (per-user calibration) can replace these with
# measured values by passing them into __init__, without touching any of the
# detection logic below.
# ─────────────────────────────────────────────────────────────────────────────

EAR_CLOSED_THRESHOLD: float = 0.21
"""
EAR below this value → eyes considered closed.
Design doc default (Section 4.4): 0.21.
Population open-eye range: 0.25–0.35.  0.21 is conservatively low so people
with smaller eyes or slightly hooded lids don't trigger false closures at rest.
"""

BLINK_MAX_DURATION_S: float = 0.40
"""
Closure shorter than this → classified as a normal blink (counted).
Human voluntary/involuntary blink duration: roughly 100–400 ms
(Sirevaag & Stern, 1994).  0.40 s matches the design doc's boundary.
"""

LONG_CLOSURE_MIN_S: float = 0.40   # same as BLINK_MAX_DURATION_S (boundary)
LONG_CLOSURE_MAX_S: float = 2.00
"""
Closure in [0.40, 2.00) s → classified as a long closure (flagged but NOT
counted as a blink).  Indicates deliberate closure, drowsiness, or
concentration with closed eyes.
"""

SLEEP_THRESHOLD_S: float = 2.00
"""
Closure ≥ 2.00 s → classified as possible sleep / severe inattention.
Will receive significant score penalty in Phase 5 (AttentionScoringEngine).
"""


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class EyeState(Enum):
    """Current binary eye state based on the most recently processed frame."""
    OPEN   = "OPEN"
    CLOSED = "CLOSED"


class ClosureType(Enum):
    """Classification of the most recently *completed* eye closure event."""
    NONE           = "NONE"           # No closure event has occurred yet
    BLINK          = "BLINK"          # < 0.40 s — normal blink, counted
    LONG_CLOSURE   = "LONG_CLOSURE"   # 0.40 – 2.00 s — prolonged, flagged
    POSSIBLE_SLEEP = "POSSIBLE_SLEEP" # ≥ 2.00 s — significant event, flagged


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BlinkResult:
    """
    Immutable snapshot of blink detector state after processing one frame.

    ``frozen=True`` ensures no downstream code mutates a stale result; it also
    makes ``BlinkResult`` hashable, which is useful for deque-based logging.

    Attributes
    ----------
    left_ear, right_ear, average_ear:
        Eye Aspect Ratios for this frame, rounded to 3 decimal places.
    eye_state:
        Current open/closed binary state.
    blink_count:
        Cumulative total of normal blinks (< 0.40 s closures) since session start.
    closure_duration_s:
        If currently closed: live elapsed closure duration.
        If just opened: duration of the closure that just completed.
        Otherwise: duration of the last completed closure.
    last_closure_type:
        Category of the most recently *completed* closure event.
    """
    left_ear: float
    right_ear: float
    average_ear: float
    eye_state: EyeState
    blink_count: int
    closure_duration_s: float
    last_closure_type: ClosureType


# ─────────────────────────────────────────────────────────────────────────────
# BlinkDetector
# ─────────────────────────────────────────────────────────────────────────────

class BlinkDetector:
    """
    Detects blinks and eye-closure events from MediaPipe face landmarks using
    the Eye Aspect Ratio (EAR).  No ML, no extra models — pure geometry.

    The detector is instantiated once per session.  Its ``update()`` method is
    called every processed frame, after ``LandmarkProcessor.update()`` has been
    called for the same frame.

    Example
    -------
    ::

        detector = BlinkDetector()

        # In the per-frame loop (after processor.update(result)):
        blink_result = detector.update(processor, timestamp_s)
        print(blink_result.blink_count, blink_result.eye_state.value)
    """

    def __init__(
        self,
        ear_threshold: float = EAR_CLOSED_THRESHOLD,
        blink_max_duration_s: float = BLINK_MAX_DURATION_S,
        sleep_threshold_s: float = SLEEP_THRESHOLD_S,
    ) -> None:
        """
        Parameters
        ----------
        ear_threshold:
            EAR below this → eyes closed.  Override for per-user calibration.
        blink_max_duration_s:
            Maximum closure duration (s) to count as a normal blink.
        sleep_threshold_s:
            Closure at or above this duration (s) → POSSIBLE_SLEEP flag.
        """
        # Configurable thresholds (replaceable by calibration in Phase 5)
        self._ear_threshold    = ear_threshold
        self._blink_max_s      = blink_max_duration_s
        self._sleep_s          = sleep_threshold_s

        # ── State machine ──────────────────────────────────────────────
        self._eye_state: EyeState = EyeState.OPEN
        # Wall-clock time (monotonic seconds from session_start_s) when the
        # current closure began; None when eyes are open.
        self._closure_start_s: Optional[float] = None

        # ── Accumulators ───────────────────────────────────────────────
        self._blink_count: int = 0
        self._last_closure_type: ClosureType = ClosureType.NONE
        self._last_closure_duration_s: float = 0.0

        # ── Per-frame EAR values (written each update, read by overlay) ─
        self._left_ear:  float = 0.0
        self._right_ear: float = 0.0
        self._avg_ear:   float = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Core per-frame update
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        processor: LandmarkProcessor,
        timestamp_s: float,
    ) -> BlinkResult:
        """
        Compute EAR for this frame and advance the blink state machine.

        Parameters
        ----------
        processor:
            A ``LandmarkProcessor`` already updated for this frame via
            ``processor.update(result)``.
        timestamp_s:
            Monotonic wall-clock time in seconds, measured from the session
            start.  In ``phase1_face_mesh.py`` this is::

                timestamp_s = time.monotonic() - session_start_s

            Using wall-clock time rather than frame count makes closure
            duration measurement robust to variable FPS.  Must be
            monotonically increasing (no resets, no system-time jumps).

        Returns
        -------
        BlinkResult
            Immutable snapshot of detector state after processing this frame.
        """
        if not processor.is_valid:
            # No face detected this frame.
            # Reset transient state so a detection dropout mid-blink does not
            # leave the machine stuck in CLOSED, which would create a phantom
            # long-closure or sleep event when the face reappears.
            self._on_face_lost()
            return self._build_result(timestamp_s)

        # ── Compute EAR for both eyes ──────────────────────────────────
        left_pts  = processor.get_landmarks(LEFT_EYE_EAR)
        right_pts = processor.get_landmarks(RIGHT_EYE_EAR)

        self._left_ear  = _compute_ear(left_pts)
        self._right_ear = _compute_ear(right_pts)
        self._avg_ear   = (self._left_ear + self._right_ear) / 2.0

        # ── Advance state machine ──────────────────────────────────────
        self._advance_state(self._avg_ear, timestamp_s)

        return self._build_result(timestamp_s)

    # ─────────────────────────────────────────────────────────────────────────
    # State machine
    # ─────────────────────────────────────────────────────────────────────────

    def _advance_state(self, avg_ear: float, timestamp_s: float) -> None:
        """
        Transition the OPEN/CLOSED state and handle event counting.

        The central invariant: ``blink_count`` is incremented ONLY at the
        CLOSED→OPEN edge.  It is never incremented while the eye remains
        closed.  This prevents "holding eyes shut" from inflating the count.
        """
        eyes_currently_closed = avg_ear < self._ear_threshold

        if eyes_currently_closed:
            if self._eye_state is EyeState.OPEN:
                # ── OPEN → CLOSED transition ──────────────────────────
                # Record when this closure started.
                self._eye_state       = EyeState.CLOSED
                self._closure_start_s = timestamp_s
            # While staying CLOSED: do nothing.  Duration accumulates in
            # wall time; it will be read in _build_result and at the edge.

        else:
            # Eyes are open (or became open again this frame).
            if self._eye_state is EyeState.CLOSED:
                # ── CLOSED → OPEN transition ──────────────────────────
                # Compute total closure duration and classify the event.
                start    = self._closure_start_s if self._closure_start_s is not None else timestamp_s
                duration = timestamp_s - start
                self._last_closure_duration_s = duration
                self._classify_and_count(duration)
                self._closure_start_s = None

            self._eye_state = EyeState.OPEN

    def _classify_and_count(self, duration_s: float) -> None:
        """
        Classify a completed closure event and update counters.

        Called exactly once per closure event, at the CLOSED→OPEN edge.
        Only BLINK events increment blink_count.

        Parameters
        ----------
        duration_s:
            Total duration of the closure that just ended.
        """
        if duration_s >= self._sleep_s:
            # ≥ 2.0 s — significant distraction / possible sleep
            self._last_closure_type = ClosureType.POSSIBLE_SLEEP

        elif duration_s >= self._blink_max_s:
            # 0.40 – 2.0 s — deliberate or drowsy extended closure
            self._last_closure_type = ClosureType.LONG_CLOSURE

        else:
            # < 0.40 s — normal involuntary blink
            self._last_closure_type = ClosureType.BLINK
            self._blink_count += 1

    def _on_face_lost(self) -> None:
        """
        Reset transient closure state when the face disappears.

        Does NOT reset blink_count or last_closure_type — those are session
        accumulators that should persist across brief detection dropouts.
        """
        self._eye_state       = EyeState.OPEN
        self._closure_start_s = None

    # ─────────────────────────────────────────────────────────────────────────
    # Result builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_result(self, timestamp_s: float) -> BlinkResult:
        """
        Assemble current detector state into an immutable BlinkResult.

        Live closure duration: while the eyes are still closed, the closure
        duration field shows how long they have been closed so far (useful
        for the overlay display).  Once the eyes open, it holds the duration
        of the closure that just completed.
        """
        if self._eye_state is EyeState.CLOSED and self._closure_start_s is not None:
            # Eyes are currently closed — report live elapsed duration.
            closure_dur = timestamp_s - self._closure_start_s
        else:
            # Eyes open — report duration of the last completed closure.
            closure_dur = self._last_closure_duration_s

        return BlinkResult(
            left_ear           = round(self._left_ear,  3),
            right_ear          = round(self._right_ear, 3),
            average_ear        = round(self._avg_ear,   3),
            eye_state          = self._eye_state,
            blink_count        = self._blink_count,
            closure_duration_s = round(closure_dur, 3),
            last_closure_type  = self._last_closure_type,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience properties
    # Provide direct attribute access without needing to unpack a BlinkResult.
    # Used by the overlay drawing code and (in later phases) the scoring engine.
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def blink_count(self) -> int:
        """Total normal blinks counted since session start."""
        return self._blink_count

    @property
    def left_ear(self) -> float:
        """Most recent left-eye EAR value (updated each frame)."""
        return self._left_ear

    @property
    def right_ear(self) -> float:
        """Most recent right-eye EAR value (updated each frame)."""
        return self._right_ear

    @property
    def average_ear(self) -> float:
        """Most recent averaged EAR value (updated each frame)."""
        return self._avg_ear

    @property
    def eye_closed(self) -> bool:
        """``True`` if eyes are currently classified as closed."""
        return self._eye_state is EyeState.CLOSED


# ─────────────────────────────────────────────────────────────────────────────
# Pure EAR computation function
# Module-level so it can be unit-tested independently of the class.
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ear(eye_points: np.ndarray | None) -> float:
    """
    Compute the Eye Aspect Ratio for one eye.

    Parameters
    ----------
    eye_points:
        numpy array of shape (6, 3) — pixel coordinates for p1..p6.
        Columns: [x_pixel, y_pixel, z_normalized].

    Returns
    -------
    float
        EAR value.  Returns 0.0 on ``None`` input or degenerate geometry.

    Implementation note
    -------------------
    Only x and y pixel coordinates are used (``[:, :2]`` slice).  The z
    column is the normalized relative depth MediaPipe provides — it does not
    contribute to a 2-D aspect ratio and is discarded here.

    Using pixel coordinates (not normalized) is correct: the formula is a
    ratio of distances and is scale-invariant.  Whether the distances are in
    pixels or any other unit, the result is the same dimensionless number.

    The six 2-D distance calls (``np.linalg.norm`` on 2-element vectors) take
    ~0.010 ms total per eye on the target hardware — negligible.
    """
    if eye_points is None or eye_points.shape != (6, 3):
        return 0.0

    # Unpack p1..p6, discarding z.
    p1, p2, p3, p4, p5, p6 = eye_points[:, :2]

    # Numerator: two vertical eyelid spans
    vertical_a = np.linalg.norm(p2 - p6)   # outer pair: upper-outer ↔ lower-outer
    vertical_b = np.linalg.norm(p3 - p5)   # inner pair: upper-inner ↔ lower-inner

    # Denominator: horizontal eye width × 2
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal < 1.0:
        # Degenerate frame: outer and inner corners are coincident (< 1 pixel
        # apart).  This can happen during extreme profile views or severely
        # corrupted landmark tracking.  Return 0 rather than a divide error.
        return 0.0

    return float((vertical_a + vertical_b) / (2.0 * horizontal))
