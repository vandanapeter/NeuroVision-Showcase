# NeuroVision: Real-Time CCTV Anomaly Detection
## Project Presentation

---

# Slide 1: Project Overview & Objectives

## 🎯 Project Title
**NeuroVision: Real-Time CCTV Anomaly Detection Using Deep Learning & Computer Vision**

## 📌 Objectives

1. **Automated Surveillance** – Reduce manual monitoring burden by automatically detecting unusual behavior in CCTV footage

2. **Real-Time Person Detection** – Accurately identify and locate all people in video frames using YOLOv8 neural network

3. **Multi-Person Tracking** – Follow individuals across frames with unique IDs using SORT algorithm

4. **Behavior Analysis** – Analyze motion patterns including speed, direction, and dwell time

5. **Anomaly Detection** – Flag suspicious activities:
   - Loitering (staying too long in one area)
   - Running (unusual speed)
   - Erratic Movement (frequent direction changes)
   - Crowd Formation (sudden gathering)

6. **Event Logging & Alerting** – Generate structured reports (CSV/JSON) for security review

---

# Slide 2: Methodology

## 🔬 Approach & Techniques

### Deep Learning Pipeline
```
Video Input → Frame Extraction → Person Detection → Tracking → Feature Extraction → Anomaly Detection → Event Logging
```

### Key Algorithms Used

| Stage | Algorithm | Purpose |
|-------|-----------|---------|
| **Detection** | YOLOv8 (You Only Look Once) | Real-time object detection CNN |
| **Tracking** | SORT + Kalman Filter | Predict & match person positions across frames |
| **Matching** | Hungarian Algorithm | Optimal assignment of detections to tracks |
| **Anomaly (Rule-Based)** | Threshold Comparison | Fast, interpretable detection |
| **Anomaly (ML-Based)** | Isolation Forest / One-Class SVM | Learn normal patterns, flag outliers |

### Processing Flow
1. **Extract frames** at configurable FPS (default: 10 FPS)
2. **Detect persons** using pre-trained YOLOv8 model
3. **Track identities** using Kalman Filter predictions + IoU matching
4. **Compute features** (speed, direction, loitering score, trajectory straightness)
5. **Apply rules/ML** to classify behavior as normal or anomalous
6. **Log events** with timestamps, severity, and details

---

# Slide 3: System Requirements

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| **RAM** | 8 GB | 16 GB |
| **GPU** | None (CPU mode) | NVIDIA GTX 1060+ (CUDA) |
| **Storage** | 5 GB | 20 GB (for videos/models) |

## 📦 Software Requirements

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Deep Learning** | PyTorch, Ultralytics (YOLOv8) |
| **Computer Vision** | OpenCV |
| **Tracking** | FilterPy (Kalman Filter), SciPy |
| **ML** | Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib |
| **Environment** | Jupyter Notebook / VS Code |

## 📥 Dependencies Installation
```bash
pip install ultralytics opencv-python filterpy scipy scikit-learn pandas matplotlib tqdm lap
```

---

# Slide 4: System Architecture

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NEUROVISION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   VIDEO      │───▶│   PERSON     │───▶│   MULTI-     │───▶│  FEATURE   │ │
│  │   LOADER     │    │  DETECTOR    │    │  TRACKER     │    │ EXTRACTOR  │ │
│  │  (OpenCV)    │    │  (YOLOv8)    │    │   (SORT)     │    │            │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └─────┬──────┘ │
│                                                                     │        │
│                                                                     ▼        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │  DASHBOARD   │◀───│   EVENT      │◀───│   ANOMALY    │◀───│   SCENE    │ │
│  │ (Matplotlib) │    │   LOGGER     │    │  DETECTOR    │    │  ANALYZER  │ │
│  │              │    │ (CSV/JSON)   │    │(Rules + ML)  │    │            │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📊 Data Flow

```
Input Video (.mp4/.avi)
    │
    ▼
Frames @ 10 FPS
    │
    ▼
Bounding Boxes [{bbox, confidence, center}]
    │
    ▼
Tracks [{track_id, bbox, history}]
    │
    ▼
Features [{speed, direction, loiter_score, straightness}]
    │
    ▼
Events [{event_type, severity, timestamp, details}]
    │
    ▼
Output: CSV/JSON + Annotated Video + Timeline Plot
```

---

# Slide 5: Module Descriptions

## 📦 Core Modules

### 1. VideoLoader Module
- **Purpose**: Load and preprocess video files
- **Features**: Configurable FPS, frame extraction, timestamp tracking
- **Formats**: MP4, AVI, MOV, MKV

### 2. PersonDetector Module (YOLOv8)
- **Purpose**: Detect all persons in each frame
- **Model**: YOLOv8n (nano) for speed, YOLOv8x for accuracy
- **Output**: Bounding boxes with confidence scores
- **Threshold**: Configurable (default: 0.3)

### 3. SORTTracker Module
- **Purpose**: Maintain identity across frames
- **Components**:
  - Kalman Filter: Predict next position
  - Hungarian Algorithm: Match detections to tracks
  - IoU Calculation: Measure box overlap
- **Parameters**: max_age=30, min_hits=3, iou_threshold=0.3

### 4. FeatureExtractor Module
- **Purpose**: Compute behavioral features per track
- **Features Computed**:
  | Feature | Description |
  |---------|-------------|
  | avg_speed | Mean velocity (px/s) |
  | direction_change_count | Number of turns > 45° |
  | loiter_score | Time in small radius |
  | straightness | Displacement / Distance ratio |

---

# Slide 6: Module Descriptions (Continued)

### 5. SceneAnalyzer Module
- **Purpose**: Monitor scene-level metrics
- **Features**:
  - Person count over time
  - Crowd density tracking
  - Sudden gathering detection

### 6. AnomalyDetector Module (Rule-Based)
- **Purpose**: Flag suspicious behavior using thresholds
- **Anomaly Types**:

| Type | Rule | Default Threshold |
|------|------|-------------------|
| Loitering | loiter_score > T | 3.0 seconds |
| Running | max_speed > S | 150 px/s |
| Erratic Movement | direction_changes > N | 5 turns |
| Crowd Formation | count_change > C | 5 people |

### 7. MLAnomalyDetector Module (Optional)
- **Purpose**: Learn normal patterns, detect outliers
- **Algorithms**: Isolation Forest, One-Class SVM
- **Training**: Requires normal behavior samples

### 8. EventLogger Module
- **Purpose**: Record and export detected anomalies
- **Outputs**: CSV, JSON with full event details

### 9. Dashboard Module
- **Purpose**: Visualize results
- **Components**: Timeline plots, event tables, trajectory visualization

---

# Slide 7: Results & Demonstration

## 📈 Sample Output

### Detected Events (UCSD Ped2 Dataset)
| Event Type | Count | Avg Severity |
|------------|-------|--------------|
| Crowd Formation | 2 | 1.0 |
| High Density | 2 | 0.78 |
| Erratic Movement | 20+ | 0.6 |
| Loitering | 3 | 0.52 |
| Running | 1 | 0.5 |

### Output Files Generated
```
./events/ucsd_ped2_test01_events.csv    # Spreadsheet format
./events/ucsd_ped2_test01_events.json   # Programmatic access
./output/ucsd_ped2_test01_annotated.mp4 # Video with boxes
./output/timeline.png                    # Anomaly timeline
```

## 🎯 Key Achievements
✅ Real-time processing at 10+ FPS on GPU  
✅ Multi-person tracking with unique IDs  
✅ Configurable thresholds for different environments  
✅ Extensible architecture (add new anomaly types easily)  
✅ Both rule-based and ML-based detection options  

---

## 📚 References
- YOLOv8: https://docs.ultralytics.com/
- SORT Algorithm: https://arxiv.org/abs/1602.00763
- UCSD Anomaly Dataset: http://www.svcl.ucsd.edu/projects/anomaly/

---

*NeuroVision - Intelligent Surveillance for Safer Spaces*
