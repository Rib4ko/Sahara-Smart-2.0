# Edge Vision & CCTV Multi-Object Tracking Rules

## 1. Distant Prone Detection Strategy
- When detecting falls/emergencies on wide-angle CCTV (1080p+), do not rely solely on pose keypoint models. A face-down or prone person occludes face and limb keypoints, resulting in 0 detections.
- Use standard object detection (e.g., `yolov8n.pt`) at higher inference resolution (`imgsz=960` for 1080p feeds) to maintain continuous bounding-box detection on distant figures (50–70px).
- Reserve pose estimation for close/medium range or head/face CNDP privacy masking.

## 2. Gravitational Head-Drop Verification Rule
- When evaluating height drops for collapse or syncope, verify that the top of the bounding box ($y_1$) drops downward toward the floor:
  $$\text{head\_drop\_ratio} = \frac{y_{1,\text{current}} - \text{baseline\_y}_1}{\text{baseline\_h}} \ge 0.22$$
- If height shrinks because $y_2$ moved upward while $y_1$ remained at standing baseline height, classify as **barrier occlusion** (person walking behind desks, low walls, railings, or sports equipment) and reject the alert.

## 3. Rolling Window Drift for Immobility Verification
- Never use instantaneous frame-to-frame velocity to determine immobility. Slow walkers travel 1–2 px per frame and falsely trigger immobility.
- Maintain a rolling spatial history (30 frames / ~1 second). Immobility is confirmed only if net drift across the entire window is below threshold ($\Delta d < 9$px).

## 4. Two-Stage IoU-First Association
- In dynamic scenes (sports courts, crowded corridors), pure centroid proximity causes frequent ID swaps during crossings.
- Association must be two-stage:
  1. Stage 1: Spatial overlap match ($IoU \ge 0.20$).
  2. Stage 2: Centroid proximity fallback for fast acceleration or scale shifts.

## 5. Pitfall of Pre-Trained Fall Classifiers
- Do not deploy generic single-person pre-trained fall classification weights on wide-angle multi-person CCTV. Geometric temporal tracking anchored on individual standing baselines delivers superior precision and zero false positives.

## 6. Multi-Camera CPU/GIL Saturation & Stream Decoupling
- In surveillance dashboards streaming multi-camera grids from a Python/FastAPI backend, never re-encode and stream multiple concurrent 30 FPS MJPEG endpoints in Python.
- Decouple ambient/simulation background feeds: Render them entirely client-side via HTML5 2D Canvas or looping HTML5 video (0% backend CPU / 0 GIL overhead).
- Allocate 100% of Python thread priority and model inference strictly to the dedicated active detection/search feed (e.g., CAM 04).

## 7. Part-Based Person Re-Identification (Re-ID) for Low-Pixel CCTV
- In public street CCTV, do not rely on facial recognition due to extreme resolution deficit (<20px face width), steep downward angles, motion blur, and privacy regulations (CNDP Law 09-08).
- Decompose person bounding boxes into two distinct anatomical regions:
  1. Upper Torso (20% to 55% vertical height)
  2. Lower Legs (55% to 92% vertical height)
- Apply center-weighted Gaussian spatial masking (inner 70% width/height) to discard surrounding sidewalk, wall, or road clutter.
- Feature extraction must combine:
  - Cyclic Hue angle moments ($\cos(H), \sin(H)$) to handle red wraparound at 0°/180°.
  - CIE-Lab color moments (perceptually uniform lightness & chromaticity $a^*, b^*$).
  - Gaussian-smoothed 16-bin Hue histogram.
- Require sustained temporal agreement (e.g., 3 consecutive frames exceeding threshold) before acquiring a target lock.

## 8. Continuous Git Synchronization Rule
- After each plan implementation that is successful and verified, always stage changes, commit with a concise descriptive message, and push to the remote repository:
  ```bash
  git add .
  git commit -m "feat: <summary of successful plan>"
  git push -u origin main
  ```
- Target repository: `https://github.com/Rib4ko/Sahara-Smart-2.0.git` (branch `main`).


