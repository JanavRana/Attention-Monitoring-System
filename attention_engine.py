from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from blink_detector import BlinkResult, EyeState, ClosureType
from head_pose_estimator import HeadPoseResult
from gaze_estimator import GazeResult, HorizontalDirection, VerticalDirection


# ---------------- Penalty System ----------------
CALIBRATION_DURATION_S = 3.0

FACE_PENALTY_SHORT = 20
FACE_PENALTY_MEDIUM = 50

BLINK_PENALTY_1S = 10
BLINK_PENALTY_2S = 25
BLINK_PENALTY_3S = 50
BLINK_HARD_CAP = 20

HEAD_SMALL = 10
HEAD_MEDIUM = 25
HEAD_LARGE = 40

ROLL_MEDIUM = 10
ROLL_LARGE = 20

GAZE_SHORT = 5
GAZE_MEDIUM = 15
GAZE_LONG = 25
GAZE_EXTREME = 35
GAZE_IGNORE = 0.08
GAZE_WARNING = 0.15

HEAD_GAZE_SYNERGY = 20
SLEEP_HEADDOWN_SYNERGY = 15

EMA_ALPHA: float = 0.025

class AttentionState(Enum):
    HIGHLY_ATTENTIVE     = "HIGHLY_ATTENTIVE"
    ATTENTIVE            = "ATTENTIVE"
    PARTIALLY_ATTENTIVE  = "PARTIALLY_ATTENTIVE"
    DISTRACTED           = "DISTRACTED"
    HIGHLY_DISTRACTED    = "HIGHLY_DISTRACTED"
    NO_FACE              = "NO_FACE"


def _classify_state(score_pct: float, face_present: bool) -> AttentionState:
    if not face_present:
        return AttentionState.NO_FACE
    if score_pct >= 90.0:
        return AttentionState.HIGHLY_ATTENTIVE
    if score_pct >= 75.0:
        return AttentionState.ATTENTIVE
    if score_pct >= 50.0:
        return AttentionState.PARTIALLY_ATTENTIVE
    if score_pct >= 25.0:
        return AttentionState.DISTRACTED
    return AttentionState.HIGHLY_DISTRACTED

@dataclass
class CalibrationProfile:
    neutral_head_pitch:      float = 0.0
    neutral_head_yaw:        float = 0.0
    neutral_gaze_horizontal: float = 0.5
    neutral_gaze_vertical:   float = 0.5
    normal_ear:              float = 0.30
    is_complete:             bool = False

@dataclass
class SessionStatistics:
    sample_count:               int = 0
    score_sum:                  float = 0.0
    average_attention:          float = 0.0
    minimum_attention:          float = 100.0
    maximum_attention:          float = 0.0
    time_attentive_s:           float = 0.0
    time_distracted_s:          float = 0.0
    longest_distracted_streak_s: float = 0.0
    blink_count:                int = 0
    long_eye_closures:          int = 0
    face_loss_duration_s:       float = 0.0

    _current_distracted_streak_s: float = field(default=0.0, repr=False)


@dataclass(frozen=True)
class AttentionResult:
    raw_score:        float           
    smoothed_score:   float            
    face_penalty: int
    blink_penalty: int
    head_penalty: int
    gaze_penalty: int
    state:            AttentionState
    is_calibrated:    bool
    calibration_progress: float      
    statistics:       SessionStatistics


class AttentionEngine:


    def __init__(self) -> None:
        self._calibration = CalibrationProfile()
        self._calib_start_s: Optional[float] = None
        self._calib_pitch_sum: float = 0.0
        self._calib_yaw_sum: float = 0.0
        self._calib_gh_sum: float = 0.0
        self._calib_gv_sum: float = 0.0
        self._calib_ear_sum: float = 0.0
        self._calib_samples: int = 0

        self._smoothed_score: float = 100.0
        self._ema_initialized: bool = False

        self._stats = SessionStatistics()

        self._prev_face_present: bool = True
        self._face_absent_start_s: Optional[float] = None
        self._gaze_away_start_s: Optional[float] = None
        self._last_blink_count_seen: int = 0
        self._last_long_closure_duration: float = -1.0


    def update(
        self,
        blink_result: BlinkResult,
        head_pose_result: HeadPoseResult,
        gaze_result: GazeResult,
        face_present: bool,
        timestamp_s: float,
        dt_s: float,
    ) -> AttentionResult:

        if not self._calibration.is_complete:
            self._run_calibration(blink_result, head_pose_result, gaze_result,
                                   face_present, timestamp_s)

        # ------------------------------------------------------------
        # Penalty-based scoring
        # ------------------------------------------------------------


        penalty = 0 
        blink_penalty = 0
        head_penalty = 0
        gaze_penalty = 0  

        # ---------------- Face ----------------
        if not face_present:

            if self._face_absent_start_s is None:
                self._face_absent_start_s = timestamp_s

            absent_time = timestamp_s - self._face_absent_start_s

            if absent_time > 10:
                raw_score = 0.0

            elif absent_time > 3:
                penalty += FACE_PENALTY_MEDIUM
                raw_score = max(0.0, 100.0 - penalty)

            elif absent_time > 1:
                penalty += FACE_PENALTY_SHORT
                raw_score = max(0.0, 100.0 - penalty)

            else:
                raw_score = 100.0

        else:
            self._face_absent_start_s = None

            if self._calibration.is_complete:

                blink_penalty = self._blink_penalty(blink_result)
                head_penalty = self._head_penalty(head_pose_result)
                gaze_penalty = self._gaze_penalty(gaze_result, timestamp_s)

                penalty += blink_penalty
                penalty += head_penalty
                penalty += gaze_penalty

                if (
                    head_penalty >= HEAD_MEDIUM
                    and gaze_penalty >= GAZE_MEDIUM
                ):
                    penalty += HEAD_GAZE_SYNERGY

                if (
                    blink_result.closure_duration_s >= 3.0
                    and head_pose_result.pitch >
                        self._calibration.neutral_head_pitch + 15
                ):
                    penalty += SLEEP_HEADDOWN_SYNERGY

                # print(
                #     f"Blink={blink_penalty} "
                #     f"Head={head_penalty} "
                #     f"Gaze={gaze_penalty} "
                #     f"Total={penalty}"
                # )
                raw_score = max(0.0, 100.0 - penalty)

            else:
                raw_score = 100.0
            

        if not self._ema_initialized:
            self._smoothed_score = raw_score
            self._ema_initialized = True
        else:
            self._smoothed_score = (
                EMA_ALPHA * raw_score + (1.0 - EMA_ALPHA) * self._smoothed_score
            )

        # print(self._smoothed_score)

        state = _classify_state(self._smoothed_score, face_present)

        self._update_statistics(state, face_present, blink_result, dt_s)

        return AttentionResult(
            raw_score               =round(raw_score, 1),
            smoothed_score          =round(self._smoothed_score, 1),

            face_penalty            =penalty if not face_present else 0,
            blink_penalty           =blink_penalty,
            head_penalty            =head_penalty,
            gaze_penalty            =gaze_penalty,

            state                   =state,
            is_calibrated           =self._calibration.is_complete,
            calibration_progress    =self._calibration_progress(timestamp_s),
            statistics              =self._stats,
        )

    @property
    def calibration(self) -> CalibrationProfile:
        return self._calibration

    @property
    def statistics(self) -> SessionStatistics:
        return self._stats


    def _run_calibration(
        self,
        blink_result: BlinkResult,
        head_pose_result: HeadPoseResult,
        gaze_result: GazeResult,
        face_present: bool,
        timestamp_s: float,
    ) -> None:
        if self._calib_start_s is None:
            self._calib_start_s = timestamp_s

        if not face_present:
            return  # skip frames with no face; do not pollute the baseline

        self._calib_pitch_sum += head_pose_result.pitch
        self._calib_yaw_sum   += head_pose_result.yaw
        self._calib_gh_sum    += gaze_result.horizontal_ratio
        self._calib_gv_sum    += gaze_result.vertical_ratio
        self._calib_ear_sum   += blink_result.average_ear
        self._calib_samples   += 1

        elapsed = timestamp_s - self._calib_start_s
        if elapsed >= CALIBRATION_DURATION_S and self._calib_samples > 0:
            n = self._calib_samples
            self._calibration.neutral_head_pitch      = self._calib_pitch_sum / n
            self._calibration.neutral_head_yaw        = self._calib_yaw_sum   / n
            self._calibration.neutral_gaze_horizontal  = self._calib_gh_sum   / n
            self._calibration.neutral_gaze_vertical    = self._calib_gv_sum   / n
            self._calibration.normal_ear               = self._calib_ear_sum  / n
            self._calibration.is_complete               = True

    def _calibration_progress(self, timestamp_s: float) -> float:
        if self._calibration.is_complete:
            return 1.0
        if self._calib_start_s is None:
            return 0.0
        elapsed = timestamp_s - self._calib_start_s
        return max(0.0, min(1.0, elapsed / CALIBRATION_DURATION_S))


    def _blink_penalty(self, blink_result: BlinkResult) -> int:

        if blink_result.eye_state is EyeState.OPEN:
            return 0

        duration = blink_result.closure_duration_s

        if duration >= 5.0:
            return BLINK_PENALTY_3S

        if duration >= 3.0:
            return BLINK_PENALTY_3S

        if duration >= 2.0:
            return BLINK_PENALTY_2S

        if duration >= 1.0:
            return BLINK_PENALTY_1S

        return 0


    def _head_penalty(self, head_pose_result: HeadPoseResult) -> int:

        pitch = abs(head_pose_result.pitch - self._calibration.neutral_head_pitch)
        yaw   = abs(head_pose_result.yaw   - self._calibration.neutral_head_yaw)
        roll  = abs(head_pose_result.roll)

        penalty = 0

        # ---------- Pitch ----------
        if pitch > 40:
            penalty += HEAD_LARGE
        elif pitch > 25:
            penalty += HEAD_MEDIUM
        elif pitch > 15:
            penalty += HEAD_SMALL

        # ---------- Yaw ----------
        if yaw > 40:
            penalty += HEAD_LARGE
        elif yaw > 25:
            penalty += HEAD_MEDIUM
        elif yaw > 15:
            penalty += HEAD_SMALL

        # ---------- Roll ----------
        if roll > 45:
            penalty += ROLL_LARGE
        elif roll > 20:
            penalty += ROLL_MEDIUM

        return penalty

    def _gaze_penalty(
        self,
        gaze_result: GazeResult,
        timestamp_s: float,
    ) -> int:

        h_dev = abs(
            gaze_result.horizontal_ratio -
            self._calibration.neutral_gaze_horizontal
        )

        v_dev = abs(
            gaze_result.vertical_ratio -
            self._calibration.neutral_gaze_vertical
        )

        # Horizontal gaze is much more reliable than vertical.
        combined_dev = (
            0.75 * h_dev +
            0.25 * v_dev
        )
        # print(
        #     f"H={h_dev:.3f}  "
        #     f"V={v_dev:.3f}  "
        #     f"C={combined_dev:.3f}"
        # )
        # Ignore natural MediaPipe jitter.
        if combined_dev < GAZE_IGNORE:
            self._gaze_away_start_s = None
            return 0

        # Small eye movement near the centre should not be punished.
        if combined_dev < GAZE_WARNING:
            self._gaze_away_start_s = None
            return 0

        # User has actually looked away.
        if self._gaze_away_start_s is None:
            self._gaze_away_start_s = timestamp_s
            return GAZE_SHORT

        away_time = timestamp_s - self._gaze_away_start_s

        if away_time >= 5.0:
            return GAZE_EXTREME
        elif away_time >= 3.0:
            return GAZE_LONG
        elif away_time >= 1.0:
            return GAZE_MEDIUM

        return GAZE_SHORT

    def _update_statistics(
        self,
        state: AttentionState,
        face_present: bool,
        blink_result: BlinkResult,
        dt_s: float,
    ) -> None:
        s = self._stats

        s.sample_count += 1
        s.score_sum += self._smoothed_score
        s.average_attention = s.score_sum / s.sample_count
        s.minimum_attention = min(s.minimum_attention, self._smoothed_score)
        s.maximum_attention = max(s.maximum_attention, self._smoothed_score)

        is_distracted = state in (
            AttentionState.DISTRACTED,
            AttentionState.HIGHLY_DISTRACTED,
            AttentionState.NO_FACE,
        )

        if is_distracted:
            s.time_distracted_s += dt_s
            s._current_distracted_streak_s += dt_s
            s.longest_distracted_streak_s = max(
                s.longest_distracted_streak_s, s._current_distracted_streak_s
            )
        else:
            s.time_attentive_s += dt_s
            s._current_distracted_streak_s = 0.0

        if not face_present:
            if self._face_absent_start_s is None:
                self._face_absent_start_s = 0.0  # marker; accumulate via dt
            s.face_loss_duration_s += dt_s
        else:
            self._face_absent_start_s = None

        if blink_result.blink_count > self._last_blink_count_seen:
            s.blink_count = blink_result.blink_count
            self._last_blink_count_seen = blink_result.blink_count

        if blink_result.eye_state is EyeState.OPEN and blink_result.last_closure_type in (
            ClosureType.LONG_CLOSURE, ClosureType.POSSIBLE_SLEEP
        ):

            if blink_result.closure_duration_s != self._last_long_closure_duration:
                s.long_eye_closures += 1
                self._last_long_closure_duration = blink_result.closure_duration_s


def draw_attention_overlay(frame_bgr, result: AttentionResult) -> None:
    import cv2

    h, w = frame_bgr.shape[:2]
    x = w // 2 - 110
    y_base = 25
    dy = 24

    state_colors = {
        AttentionState.HIGHLY_ATTENTIVE:    (0, 220, 0),
        AttentionState.ATTENTIVE:           (0, 200, 60),
        AttentionState.PARTIALLY_ATTENTIVE: (0, 200, 220),
        AttentionState.DISTRACTED:          (0, 120, 255),
        AttentionState.HIGHLY_DISTRACTED:   (0, 0, 255),
        AttentionState.NO_FACE:             (100, 100, 100),
    }
    color = state_colors.get(result.state, (200, 200, 200))

    calib_text = (
        "Calibrated" if result.is_calibrated
        else f"Calibrating {result.calibration_progress * 100:.0f}%"
    )

    lines = [
        (f"Attention: {result.smoothed_score:.0f}%", color),
        (f"State: {result.state.value}",             color),
        (calib_text,                                  (180, 180, 180)),
    ]

    for i, (text, c) in enumerate(lines):
        cv2.putText(
            frame_bgr, text, (x, y_base + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA,
        )
