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
