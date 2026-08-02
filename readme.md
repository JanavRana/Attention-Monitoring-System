# Visual Engagement Analysis System

A real-time computer vision system that estimates a person's visual engagement during online learning sessions using facial landmarks. The system analyzes eye blinks, gaze direction, head pose, and face presence to generate an attention score, log session statistics, and produce an interactive HTML report.

> **Disclaimer:** This project measures visual engagement signals only. It is **not** a measurement of comprehension, learning ability, intelligence, or mental focus.

---

# Features

- Real-time face detection using MediaPipe Face Landmarker
- Eye blink detection using Eye Aspect Ratio (EAR)
- Head pose estimation using OpenCV `solvePnP`
- Gaze estimation using iris landmark geometry
- Personalized calibration before monitoring begins
- Attention scoring using a rule-based penalty engine
- Live attention overlay during monitoring
- Session logging to CSV
- Automatic JSON session summary
- Self-contained HTML report with charts and statistics
- Lightweight implementation suitable for CPU-only systems

---

# System Architecture

```
Webcam
    │
    ▼
Face Landmarker
    │
    ▼
Landmark Processor
    │
    ├─────────────┐
    ▼             ▼
Blink         Head Pose
Detector      Estimator
    │             │
    └──────┐ ┌────┘
           ▼ ▼
      Gaze Estimator
           │
           ▼
    Attention Engine
           │
     ┌─────┴─────┐
     ▼           ▼
 Live Overlay  Session Logger
                   │
          CSV + JSON Summary
                   │
                   ▼
          HTML Report Generator
```

---

# Technologies Used

- Python 3
- OpenCV
- MediaPipe Tasks API
- NumPy
- Pandas
- Matplotlib

---

# Folder Structure

```
Student-Attention-Monitoring-System/
│
├── core/
│   ├── landmark_processor.py
│   ├── blink_detector.py
│   ├── gaze_estimator.py
│   ├── head_pose_estimator.py
│   ├── attention_engine.py
│   ├── session_logger.py
│   └── report_generator.py
│
├── models/
│   └── face_landmarker.task
│
├── sessions/
│   ├── *.csv
│   └── *_summary.json
│
├── reports/
│   └── *_report.html
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone <repository-url>
cd Attention-Monitoring-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python main.py
```

The MediaPipe model is downloaded automatically the first time if it is not already present inside the `models/` directory.

---

# How It Works

## 1. Calibration

At startup, the system performs a short calibration period to determine the user's neutral:

- head orientation
- gaze position
- eye aspect ratio

These values are used as the personal baseline for the remainder of the session.

---

## 2. Attention Estimation

The system evaluates several visual cues:

- Face presence
- Blink duration
- Head pose
- Eye gaze direction

Each cue contributes penalties to the overall attention score. The score is updated continuously throughout the session.

---

## 3. Session Logging

Every second, the system records:

- Attention score
- Gaze direction
- Blink statistics
- Head pose
- Face presence
- Running session statistics

The data is stored as a CSV file.

---

## 4. Report Generation

When the session ends, the system automatically generates:

- CSV session log
- JSON summary
- Interactive HTML report

The HTML report contains:

- Attention trend over time
- Attention state distribution
- Blink statistics
- Gaze statistics
- Face presence timeline
- Session summary table

---

# Performance

Designed for lightweight CPU execution.

Typical configuration:

- Resolution: **640×480**
- Processing rate: **≈15 FPS**
- Tested on:
  - AMD Ryzen 5 3450U
  - 8 GB RAM

---

# Output Files

```
sessions/
    session_YYYYMMDD_HHMMSS.csv
    session_YYYYMMDD_HHMMSS_summary.json

reports/
    session_YYYYMMDD_HHMMSS_report.html
```

---

# Limitations

- Designed for a single user
- Requires the user's face to remain visible
- Performance depends on lighting conditions
- Visual engagement should not be interpreted as actual concentration or understanding
- Not intended for surveillance or automated decision-making

---

# Future Improvements

Potential extensions include:

- Persistent user calibration
- Multi-session analytics
- SQLite session database
- Improved gaze estimation models
- Machine learning–based attention scoring
- Classroom-level aggregate analytics (with informed consent)

---

# License

This project is intended for educational and research purposes.

---

# Acknowledgements

- Google MediaPipe
- OpenCV
- NumPy
- Pandas
- Matplotlib
