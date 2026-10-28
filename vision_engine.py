"""
╔══════════════════════════════════════════════════════════════╗
║  Sahra-Eye Vision Engine                                     ║
║  Laâyoune Smart City Edge AI — Mode 1: Najda-Vision          ║
║  Fall & Medical Collapse Triage                               ║
║                                                               ║
║  Privacy-First: CNDP Law 09-08 — All faces masked by default ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import math
import time
import os
import logging
import base64
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="[Sahra-Eye] %(levelname)s: %(message)s")
logger = logging.getLogger("sahra-eye")

# ──────────────────────────────────────────────────────────────
#  COCO Pose Keypoint Indices
# ──────────────────────────────────────────────────────────────
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

FACE_KP_INDICES = [KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR]

# ──────────────────────────────────────────────────────────────
#  Fall & Faint Detection Thresholds
# ──────────────────────────────────────────────────────────────
FALL_ANGLE_THRESHOLD = 55      # Degrees from vertical for horizontal fall
FALL_SUSTAIN_FRAMES = 12       # Frames to confirm slip/horizontal fall (~0.4s)
FAINT_SUSTAIN_FRAMES = 14      # Frames to confirm fainting syncope (~0.5s of ground immobility)
FALL_ASPECT_RATIO = 1.25       # Width/height ratio for horizontal body across floor
FAINT_HEIGHT_DROP_RATIO = 0.65 # Current height < 65% of standing baseline indicates collapse
FAINT_MIN_ASPECT_RATIO = 0.50  # Crumpled/slumped aspect ratio
MIN_STANDING_FRAMES = 15       # Minimum frames standing upright before faint monitoring activates
TRACKER_MAX_DIST = 160         # Max pixel distance for centroid matching across frames
TRACKER_MAX_LOST = 25          # Frames before an unmatched non-victim track is deleted

# ──────────────────────────────────────────────────────────────
#  CNDP Privacy Mask Constants
# ──────────────────────────────────────────────────────────────
BLUR_KERNEL = (99, 99)
BLUR_SIGMA = 30
HEAD_EXPAND = 0.5              # Expand head bbox by this fraction on each side

# ──────────────────────────────────────────────────────────────
#  Green Wave Simulation
# ──────────────────────────────────────────────────────────────
GREEN_WAVE_DURATION = 30       # Seconds the green wave stays active after a fall


def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, float(xB - xA)) * max(0.0, float(yB - yA))
    areaA = max(1.0, float((boxA[2] - boxA[0]) * (boxA[3] - boxA[1])))
    areaB = max(1.0, float((boxB[2] - boxB[0]) * (boxB[3] - boxB[1])))
    union = areaA + areaB - inter
    return float(inter / max(union, 1.0))

def crop_to_base64(crop, max_size=120):
    """Encode an OpenCV crop as base64 JPEG URI for dashboard thumbnail streaming."""
    if crop is None or crop.size == 0:
        return ""
    h, w = crop.shape[:2]
    scale = max_size / max(h, w, 1)
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def hex_to_bgr(hex_str, default=(128, 128, 128)):
    try:
        hex_str = str(hex_str).lstrip('#')
        if len(hex_str) == 6:
            r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
            return (b, g, r)
    except Exception:
        pass
    return default


# ══════════════════════════════════════════════════════════════
#  Tifawt-Guard: Part-Based Person Re-Identification (Re-ID)
# ══════════════════════════════════════════════════════════════
class TifawtReIDMatcher:
    """
    3-Part Person Re-Identification (Re-ID) Engine for Mode 2 (Tifawt-Guard).
    Solves CCTV low-pixel facial blur using:
    1. Spatial part decomposition:
       - Head / Cap: 0% to 22% vertical height (critical for detecting red hats / headwear)
       - Upper Torso: 22% to 58% vertical height (shirts, jackets, hoodies)
       - Lower Legs: 58% to 94% vertical height (jeans, shorts, trackpants)
    2. Center-weighted Gaussian spatial masking (discards background asphalt/sidewalk clutter)
    3. Dual illumination-invariant color spaces:
       - HSV with cv2.bitwise_or for dual-band red wraparound [0,12] and [168,180]
       - Cyclic Hue angle moments (cos, sin)
       - CIE-Lab perceptually uniform color moments (L, a*, b*)
    4. Dominant color hex swatch extraction for operator UI preview
    5. Aspect ratio & child/minor scale prior
    6. Multi-part weighted cosine similarity
    """

    def __init__(self, threshold=0.65):
        self.threshold = threshold
        self.reference_target = None
        self.child_prior = True

    def has_target(self):
        return self.reference_target is not None

    def clear_target(self):
        self.reference_target = None

    def set_threshold(self, threshold):
        self.threshold = max(0.40, min(0.95, float(threshold)))

    def extract_features(self, crop):
        """Extract 3-part spatial color signature (Head 0-22%, Torso 22-58%, Legs 58-94%) from a person crop."""
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 24 or w < 12:
            return None

        # 1. Head / Cap region (0% to 22% vertical height)
        y_h1, y_h2 = 0, int(h * 0.22)
        head_crop = crop[y_h1:y_h2, :]

        # 2. Upper Torso (22% to 58% vertical height)
        y_t1, y_t2 = int(h * 0.22), int(h * 0.58)
        torso_crop = crop[y_t1:y_t2, :]

        # 3. Lower Body / Legs (58% to 94% vertical height)
        y_l1, y_l2 = int(h * 0.58), int(h * 0.94)
        legs_crop = crop[y_l1:y_l2, :]

        def get_part_signature(part):
            ph, pw = part.shape[:2]
            if ph < 3 or pw < 3:
                return np.zeros(32, dtype=np.float32), "#64748b"

            # Center crop to reduce surrounding sidewalk/background
            c_part = part[int(ph * 0.12):int(ph * 0.88), int(pw * 0.12):int(pw * 0.88)]
            if c_part.size == 0:
                c_part = part

            hsv = cv2.cvtColor(c_part, cv2.COLOR_BGR2HSV).astype(np.float32)
            c_part_uint8 = np.clip(c_part, 0, 255).astype(np.uint8)
            hsv_uint8 = cv2.cvtColor(c_part_uint8, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(c_part, cv2.COLOR_BGR2LAB).astype(np.float32)

            # Dual-band Red wraparound via cv2.bitwise_or
            mask_low = cv2.inRange(hsv_uint8, np.array([0, 50, 45]), np.array([12, 255, 255]))
            mask_high = cv2.inRange(hsv_uint8, np.array([168, 50, 45]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(mask_low, mask_high)
            red_ratio = float(cv2.countNonZero(red_mask)) / max(1.0, float(c_part.shape[0] * c_part.shape[1]))

            # Yellow/Amber mask (H in [16, 38], S > 45)
            yellow_mask = cv2.inRange(hsv_uint8, np.array([16, 45, 45]), np.array([38, 255, 255]))
            yellow_ratio = float(cv2.countNonZero(yellow_mask)) / max(1.0, float(c_part.shape[0] * c_part.shape[1]))

            # Blue / Denim mask (H in [95, 135], S > 35)
            blue_mask = cv2.inRange(hsv_uint8, np.array([95, 35, 35]), np.array([135, 255, 255]))
            blue_ratio = float(cv2.countNonZero(blue_mask)) / max(1.0, float(c_part.shape[0] * c_part.shape[1]))

            # Cyclic Hue moments (handles red wraparound at 0/180)
            h_cos = np.cos(hsv[:, :, 0] * np.pi / 90.0)
            h_sin = np.sin(hsv[:, :, 0] * np.pi / 90.0)

            mean_h_cos, mean_h_sin = float(np.mean(h_cos)), float(np.mean(h_sin))
            mean_s, std_s = float(np.mean(hsv[:, :, 1]) / 255.0), float(np.std(hsv[:, :, 1]) / 255.0)
            mean_v, std_v = float(np.mean(hsv[:, :, 2]) / 255.0), float(np.std(hsv[:, :, 2]) / 255.0)

            mean_l, std_l = float(np.mean(lab[:, :, 0]) / 255.0), float(np.std(lab[:, :, 0]) / 255.0)
            mean_a, std_a = float((np.mean(lab[:, :, 1]) - 128.0) / 128.0), float(np.std(lab[:, :, 1]) / 128.0)
            mean_b, std_b = float((np.mean(lab[:, :, 2]) - 128.0) / 128.0), float(np.std(lab[:, :, 2]) / 128.0)

            # 16-bin Hue histogram with soft Gaussian smoothing
            hist_h = cv2.calcHist([c_part], [0], None, [16], [0, 180])
            hist_h = cv2.GaussianBlur(hist_h, (5, 1), 1.0).flatten()
            hist_h = hist_h / (hist_h.sum() + 1e-6)

            vec = np.concatenate([
                [mean_h_cos, mean_h_sin, mean_s, std_s, mean_v, std_v,
                 mean_l, std_l, mean_a, std_a, mean_b, std_b,
                 red_ratio, yellow_ratio, blue_ratio],
                hist_h
            ]).astype(np.float32)
            norm = np.linalg.norm(vec)
            vec = vec / (norm + 1e-6)

            # Dominant color hex swatch from center
            med_bgr = np.median(c_part.reshape(-1, 3), axis=0).astype(int)
            hex_color = f"#{int(med_bgr[2]):02x}{int(med_bgr[1]):02x}{int(med_bgr[0]):02x}"

            return vec, hex_color

        h_vec, h_hex = get_part_signature(head_crop)
        t_vec, t_hex = get_part_signature(torso_crop)
        l_vec, l_hex = get_part_signature(legs_crop)
        ar = float(w) / max(float(h), 1.0)

        return {
            "head_vec": h_vec,
            "torso_vec": t_vec,
            "legs_vec": l_vec,
            "head_hex": h_hex,
            "torso_hex": t_hex,
            "legs_hex": l_hex,
            "aspect_ratio": ar,
            "height": h,
            "width": w
        }

    def set_parametric_target(self, label, head_hex, torso_hex, legs_hex, has_head_item=True, child_prior=True):
        """
        Synthesize reference target from structured parameters (e.g. parsed from Groq LLM).
        Renders a 3-part spatial candidate with natural textile texture noise.
        """
        h_bgr = hex_to_bgr(head_hex, (30, 30, 200))
        t_bgr = hex_to_bgr(torso_hex, (25, 180, 235))
        l_bgr = hex_to_bgr(legs_hex, (190, 85, 35))

        h, w = 180, 70
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:int(h * 0.22), :] = h_bgr
        img[int(h * 0.22):int(h * 0.58), :] = t_bgr
        img[int(h * 0.58):int(h * 0.94), :] = l_bgr
        img[int(h * 0.94):, :] = (35, 35, 35)

        noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        feats = self.extract_features(img)
        if feats is None:
            return False, "Failed to compile parametric features"

        feats["head_hex"] = head_hex
        feats["torso_hex"] = torso_hex
        feats["legs_hex"] = legs_hex

        thumb_b64 = crop_to_base64(img, max_size=160)
        self.child_prior = bool(child_prior)
        self.reference_target = {
            "label": label,
            "features": feats,
            "thumbnail": thumb_b64,
            "head_hex": head_hex,
            "torso_hex": torso_hex,
            "legs_hex": legs_hex,
            "has_head_item": bool(has_head_item),
            "aspect_ratio": feats["aspect_ratio"],
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        logger.info(f"Tifawt-Guard parametric target set: {label} (Head: {head_hex}, Torso: {torso_hex}, Legs: {legs_hex})")
        return True, "Parametric target active"

    def set_reference_target(self, image, label="Target Subject", det_model=None):
        """
        Set reference target from image or crop.
        If a detection model is passed, automatically detects and crops the most prominent person.
        """
        if image is None or image.size == 0:
            return False, "Invalid image data"

        crop = image
        h, w = image.shape[:2]

        if det_model is not None and (h > 150 and w > 100):
            try:
                results = det_model(image, classes=[0], imgsz=640, conf=0.35, verbose=False)
                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                    best_idx = int(np.argmax(areas))
                    x1, y1, x2, y2 = map(int, boxes[best_idx])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if (x2 - x1) > 20 and (y2 - y1) > 40:
                        crop = image[y1:y2, x1:x2]
            except Exception as e:
                logger.warning(f"Auto-crop detection fallback: {e}")

        feats = self.extract_features(crop)
        if feats is None:
            return False, "Could not extract person features (crop too small)"

        thumb_b64 = crop_to_base64(crop, max_size=160)
        self.reference_target = {
            "label": label,
            "features": feats,
            "thumbnail": thumb_b64,
            "head_hex": feats["head_hex"],
            "torso_hex": feats["torso_hex"],
            "legs_hex": feats["legs_hex"],
            "has_head_item": True,
            "aspect_ratio": feats["aspect_ratio"],
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        logger.info(f"Tifawt-Guard reference target set: {label} (Head: {feats['head_hex']}, Torso: {feats['torso_hex']}, Legs: {feats['legs_hex']})")
        return True, "Target registered successfully"

    def compute_similarity(self, candidate_crop):
        """
        Compute multi-part weighted similarity against reference target.
        Returns: (overall_sim, head_sim, torso_sim, legs_sim)
        """
        if not self.has_target() or candidate_crop is None:
            return 0.0, 0.0, 0.0, 0.0

        c_feats = self.extract_features(candidate_crop)
        if c_feats is None:
            return 0.0, 0.0, 0.0, 0.0

        r_feats = self.reference_target["features"]

        # Head cosine similarity
        sim_head = float(np.dot(c_feats["head_vec"], r_feats["head_vec"]))
        sim_head = max(0.0, min(1.0, sim_head))

        # Torso cosine similarity
        sim_torso = float(np.dot(c_feats["torso_vec"], r_feats["torso_vec"]))
        sim_torso = max(0.0, min(1.0, sim_torso))

        # Legs cosine similarity
        sim_legs = float(np.dot(c_feats["legs_vec"], r_feats["legs_vec"]))
        sim_legs = max(0.0, min(1.0, sim_legs))

        # Weighted composite: Head 25%, Torso 45%, Legs 30% if headwear active
        has_head_item = self.reference_target.get("has_head_item", True)
        if has_head_item:
            overall = 0.25 * sim_head + 0.45 * sim_torso + 0.30 * sim_legs
        else:
            overall = 0.10 * sim_head + 0.55 * sim_torso + 0.35 * sim_legs

        # Optional child prior: penalize if candidate aspect ratio is tall/slender adult
        if self.child_prior:
            c_ar = c_feats["aspect_ratio"]
            if c_ar < 0.28:
                overall *= 0.85

        return float(overall), float(sim_head), float(sim_torso), float(sim_legs)


# ══════════════════════════════════════════════════════════════
#  Person Tracker — With Standing Baseline & Ghost Victim Memory
# ══════════════════════════════════════════════════════════════
class PersonTracker:
    """
    Temporal tracker with:
    1. IoU-first matching (>0.20) + centroid proximity fallback (eliminates ID swaps)
    2. Standing baseline calibration with perspective & gravitational head-drop verification
    3. True window-drift immobility tracking over a 30-frame rolling window
    4. Syncope / Faint collapse detection with barrier occlusion immunity
    5. Persistent Fallen Victim Memory ('Ghost Anchor') retaining last known position
       even if detector drops detection on the floor or behind crowds.
    """

    def __init__(self, max_distance=60.0, max_lost=60):
        self.tracks = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.max_lost = max_lost

    def update(self, detections):
        matched = {}
        used_tracks = set()

        # Stage 1: IoU matching (> 0.20)
        for d_i, det in enumerate(detections):
            best_iou = 0.20
            best_id = None
            for tid, tr in self.tracks.items():
                if tid in used_tracks:
                    continue
                iou = box_iou(det['bbox'], tr['last_bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            if best_id is not None:
                matched[d_i] = best_id
                used_tracks.add(best_id)

        # Stage 2: Proximity matching for remaining
        for d_i, det in enumerate(detections):
            if d_i in matched:
                continue
            cx, cy = det['cx'], det['cy']
            best_dist = self.max_distance
            best_id = None
            for tid, tr in self.tracks.items():
                if tid in used_tracks:
                    continue
                dist = math.hypot(cx - tr['cx'], cy - tr['cy'])
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is not None:
                matched[d_i] = best_id
                used_tracks.add(best_id)

        results = []
        for d_i, det in enumerate(detections):
            cx, cy = det['cx'], det['cy']
            bbox = det['bbox']
            bw, bh = det['bw'], det['bh']
            ar = det['aspect_ratio']
            y1 = float(bbox[1])
            is_horiz = det.get('is_horizontal_fall', False)
            angle = det.get('angle', 0.0)
            has_angle = det.get('has_angle', False)

            if d_i in matched:
                tid = matched[d_i]
                tr = self.tracks[tid]
                tr['cx'] = cx
                tr['cy'] = cy
                tr['last_bbox'] = bbox
                tr['last_bh'] = bh
                tr['last_bw'] = bw
                tr['lost'] = 0

                # Window drift over 30 frames
                tr['pos_hist'].append((cx, cy))
                if len(tr['pos_hist']) > 30:
                    tr['pos_hist'].pop(0)
                old_x, old_y = tr['pos_hist'][0]
                window_drift = math.hypot(cx - old_x, cy - old_y)
                if window_drift < 9.0:
                    tr['immobile_frames'] += 1
                else:
                    tr['immobile_frames'] = 0

                # Upright Standing Calibration (ar <= 0.58 for upright running humans)
                is_upright = (ar <= 0.58) and (not has_angle or angle < 30)
                if is_upright:
                    tr['standing_frames'] += 1
                    if tr['baseline_h'] == 0:
                        tr['baseline_h'] = bh
                        tr['baseline_cy'] = cy
                        tr['baseline_y1'] = y1
                    elif bh > tr['baseline_h'] and bh <= tr['baseline_h'] * 1.30:
                        tr['baseline_h'] = bh
                        tr['baseline_cy'] = cy
                        tr['baseline_y1'] = y1

                # Height drop tracking (rolling 90 frames / 3 seconds)
                tr['h_hist'].append(bh)
                if len(tr['h_hist']) > 90:
                    tr['h_hist'].pop(0)
                recent_max_h = max(tr['h_hist'])
                recent_drop = (recent_max_h - bh) / max(1.0, recent_max_h)

                is_collapsing = False
                reason = ''

                # Perspective check: Reject if cy moved > 45 px upwards into background away from baseline
                baseline_drift = (tr.get('baseline_cy', cy) - cy)
                walked_to_background = baseline_drift > 45.0

                # Gravitational Head Drop verification: In a real collapse, the head (y1) MUST drop downward
                head_drop = (y1 - tr.get('baseline_y1', y1))
                head_drop_ratio = head_drop / max(1.0, tr.get('baseline_h', 1.0))

                # Mode 1 Faint / Collapse Rule
                if tr['standing_frames'] >= 5 and tr['baseline_h'] >= 110 and not walked_to_background:
                    h_ratio = bh / tr['baseline_h']
                    # Require height collapse, aspect ratio shift, AND gravitational head drop
                    if h_ratio <= 0.54 and recent_drop >= 0.25 and ar >= 0.50 and head_drop_ratio >= 0.22:
                        is_collapsing = True
                        pct = int((1.0 - h_ratio) * 100)
                        reason = f"FAINT / COLLAPSE (Height -{pct}%)"

                # Floor Sprawl: motionless sprawl across floor with low height
                if not is_collapsing and ar >= 1.25 and bh <= 80 and head_drop_ratio >= 0.20:
                    is_collapsing = True
                    reason = f"COLLAPSE ON FLOOR (AR {ar:.2f})"

                # Horizontal Fall (torso angle from pose)
                if not is_collapsing and is_horiz and head_drop_ratio >= 0.20:
                    is_collapsing = True
                    reason = f"FALL (Angle {angle:.0f}°)"

                if is_collapsing:
                    tr['fall_count'] += 1
                    tr['fall_reason'] = reason
                else:
                    tr['fall_count'] = max(0, tr['fall_count'] - 1)

                # Sustain confirmation: 8 consecutive frames (~0.25s)
                if tr['fall_count'] >= 8:
                    tr['is_victim'] = True
                    tr['victim_bbox'] = list(bbox)
                    tr['victim_time'] = time.time()
                elif tr.get('is_victim', False):
                    tr['victim_bbox'] = list(bbox)

                # Recovery Check: If victim stands back up for >= 30 frames, clear victim state
                if tr.get('is_victim', False):
                    if is_upright and bh >= tr['baseline_h'] * 0.85:
                        tr['recovery_frames'] += 1
                        if tr['recovery_frames'] >= 30:
                            tr['is_victim'] = False
                            tr['fall_count'] = 0
                            tr['recovery_frames'] = 0
                    else:
                        tr['recovery_frames'] = 0

                results.append((tid, tr.get('is_victim', False), tr.get('fall_reason', 'CRITICAL ALERT')))
            else:
                tid = self.next_id
                self.next_id += 1
                is_upright = (ar <= 0.52)
                self.tracks[tid] = {
                    'cx': cx,
                    'cy': cy,
                    'last_bbox': bbox,
                    'last_bh': bh,
                    'last_bw': bw,
                    'standing_frames': 1 if is_upright else 0,
                    'baseline_h': bh if is_upright else 0,
                    'baseline_cy': cy if is_upright else cy,
                    'baseline_y1': y1 if is_upright else y1,
                    'h_hist': [bh],
                    'pos_hist': [(cx, cy)],
                    'immobile_frames': 0,
                    'fall_count': 0,
                    'is_victim': False,
                    'victim_bbox': list(bbox),
                    'victim_time': 0,
                    'fall_reason': '',
                    'recovery_frames': 0,
                    'lost': 0,
                    'reid_sim': 0.0,
                    'reid_head_sim': 0.0,
                    'reid_torso_sim': 0.0,
                    'reid_legs_sim': 0.0,
                    'reid_match_count': 0,
                    'is_reid_match': False
                }
                results.append((tid, False, ''))

        # Age-out (never delete confirmed victims)
        to_del = []
        for tid, tr in self.tracks.items():
            if tid not in used_tracks:
                tr['lost'] += 1
                if not tr.get('is_victim', False) and tr['lost'] > self.max_lost:
                    to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

        return results

    def get_ghost_victims(self):
        """
        Return list of confirmed fallen victims currently NOT detected by YOLO
        (Ghost Anchor layer for occluded or floor-level victims).
        """
        ghosts = []
        for tid, tr in self.tracks.items():
            if tr.get('is_victim', False) and tr['lost'] > 0:
                ghosts.append({
                    'track_id': tid,
                    'bbox': tr.get('victim_bbox', tr.get('last_bbox')),
                    'lost_frames': tr['lost'],
                    'lost_seconds': round(tr['lost'] / 30.0, 1),
                    'reason': tr.get('fall_reason', 'FAINT / COLLAPSE')
                })
        return ghosts

    def clear_victims(self):
        """Manually clear all confirmed victims (operator dismissal)."""
        for tr in self.tracks.values():
            tr['is_victim'] = False
            tr['fall_count'] = 0
            tr['immobile_frames'] = 0
            tr['recovery_frames'] = 0


# ══════════════════════════════════════════════════════════════
#  Sahra-Eye Engine — Core Vision Pipeline
# ══════════════════════════════════════════════════════════════
class SahraEyeEngine:
    """
    Edge AI Vision Engine for Laâyoune Smart City.

    Phase 1 implements:
        Mode 1 (Najda-Vision) — Fall & Medical Collapse Triage
        CNDP Law 09-08 privacy masking on every frame
    """

    # ── Initialization ───────────────────────────────────────
    def __init__(self, source=0):
        self.source = source
        self.is_video_file = isinstance(source, str) and os.path.isfile(source)
        self.source_name = os.path.basename(source) if self.is_video_file else "Live Webcam"
        self.cap = None
        self.det_model = None
        self.pose_model = None
        self.current_mode = 1
        self.mode_names = {
            1: "Najda-Vision",
            2: "Tifawt-Guard",
            3: "Infra Guard",
        }

        # Telemetry state
        self.alerts = []
        self.max_alerts = 100
        self.fps = 0.0
        self.persons_count = 0
        self._prev_time = time.time()
        self._fps_samples = []

        # Green-wave corridor state
        self.green_wave_active = False
        self.green_wave_start = 0

        # Fall-detection & Person tracker
        self.tracker = PersonTracker()

        # Set of alert keys already logged (prevents duplicate alerts per track)
        self._logged_alert_keys = set()

        # Mode 2 (Tifawt-Guard) Re-ID Matcher
        self.reid_matcher = TifawtReIDMatcher(threshold=0.65)
        self._init_default_preset()

        # CNDP blur toggle (disabled for performance/testing per request)
        self.enable_cndp_blur = False

        # Boot
        self._init_camera()
        self._load_models()

        logger.info(f"Engine initialized — Mode 1: Najda-Vision active | Source: {self.source_name}")

    # ── Camera / Video Source ─────────────────────────────────
    def _init_camera(self):
        """Open video source (CCTV file or webcam) with DirectShow fallback."""
        if self.is_video_file:
            self.cap = cv2.VideoCapture(self.source)
        else:
            # Try DirectShow on Windows to avoid MSMF frame grab warnings
            try:
                self.cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
            except Exception:
                self.cap = cv2.VideoCapture(self.source)

            if not self.cap or not self.cap.isOpened():
                for src in [0, 1]:
                    try:
                        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                    except Exception:
                        self.cap = cv2.VideoCapture(src)
                    if self.cap.isOpened():
                        logger.info(f"Fallback: opened camera source {src}")
                        break

        if self.cap and self.cap.isOpened():
            if not self.is_video_file:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Source ready [{self.source_name}]: {w}×{h}")
        else:
            logger.warning(f"Could not open source [{self.source_name}] — demo mode active")

    def switch_source(self, source, source_name=None):
        """Seamlessly switch between webcam and uploaded CCTV video file."""
        if self.cap and self.cap.isOpened():
            self.cap.release()

        self.source = source
        self.is_video_file = isinstance(source, str) and os.path.isfile(source)
        self.source_name = source_name or (os.path.basename(source) if self.is_video_file else "Live Webcam")

        self._init_camera()

        # Reset tracking & alert states for the new video
        self.tracker = PersonTracker()
        self.alerts.clear()
        self._logged_alert_keys.clear()
        self.green_wave_active = False
        logger.info(f"Video source switched → {self.source_name}")

    def clear_victims(self):
        """Manually clear all confirmed victims and active alerts (operator dismissal)."""
        self.tracker.clear_victims()
        self.alerts.clear()
        self._logged_alert_keys.clear()
        self.green_wave_active = False
        logger.info("Cleared all confirmed victims and active alerts")

    # ── Tifawt-Guard Target Management ───────────────────────
    def _create_synthetic_target(self, head_bgr, torso_bgr, legs_bgr, h=180, w=70):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:int(h * 0.22), :] = head_bgr         # Head / Cap
        img[int(h * 0.22):int(h * 0.58), :] = torso_bgr  # Torso / Shirt
        img[int(h * 0.58):int(h * 0.94), :] = legs_bgr   # Legs / Pants
        img[int(h * 0.94):, :] = (35, 35, 35)            # Shoes
        noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    def _init_default_preset(self):
        """Initialize default missing person search preset (Missing Kid: Red Cap, Yellow Shirt & Jeans)."""
        self.set_target_preset("preset_1")

    def set_target_preset(self, preset_id):
        """Set one of the fast demo presets (Zero-latency preset bypass)."""
        presets = {
            "preset_1": {
                "label": "Missing Kid: Red Cap, Yellow Shirt & Jeans",
                "head_hex": "#dc2626",
                "torso_hex": "#eab308",
                "legs_hex": "#2563eb",
                "has_head_item": True,
                "child": True,
            },
            "preset_2": {
                "label": "Missing Boy: Blue Hoodie & Dark Pants",
                "head_hex": "#475569",
                "torso_hex": "#1d4ed8",
                "legs_hex": "#1e293b",
                "has_head_item": False,
                "child": True,
            },
            "preset_3": {
                "label": "Missing Child: Red Jacket & Blue Jeans",
                "head_hex": "#475569",
                "torso_hex": "#dc2626",
                "legs_hex": "#1d4ed8",
                "has_head_item": False,
                "child": True,
            },
            "preset_4": {
                "label": "Student: Sports Jersey & Shorts",
                "head_hex": "#475569",
                "torso_hex": "#ca8a04",
                "legs_hex": "#111827",
                "has_head_item": False,
                "child": True,
            }
        }
        p = presets.get(preset_id, presets["preset_1"])
        ok, msg = self.set_parametric_target(
            label=p["label"],
            head_hex=p["head_hex"],
            torso_hex=p["torso_hex"],
            legs_hex=p["legs_hex"],
            has_head_item=p.get("has_head_item", True),
            child_prior=p.get("child", True)
        )
        logger.info(f"Target preset selected: {p['label']}")
        return ok, p["label"]

    def set_parametric_target(self, label, head_hex, torso_hex, legs_hex, has_head_item=True, child_prior=True):
        """Set target from structured parameters (from Groq LLM or preset bypass)."""
        ok, msg = self.reid_matcher.set_parametric_target(
            label=label,
            head_hex=head_hex,
            torso_hex=torso_hex,
            legs_hex=legs_hex,
            has_head_item=has_head_item,
            child_prior=child_prior
        )
        if ok:
            self.alerts.clear()
            self._logged_alert_keys.clear()
            logger.info(f"Target set via parameters: {label}")
        return ok, msg

    def set_target_photo(self, image, label="Uploaded Reference Photo"):
        """Set reference target from an uploaded image (using YOLO auto-crop)."""
        ok, msg = self.reid_matcher.set_reference_target(image, label=label, det_model=self.det_model)
        if ok:
            self.alerts.clear()
            self._logged_alert_keys.clear()
            logger.info(f"Target photo updated: {label}")
        return ok, msg

    def clear_target(self):
        """Clear active search target."""
        self.reid_matcher.clear_target()
        self.alerts.clear()
        self._logged_alert_keys.clear()
        logger.info("Search target cleared.")

    def set_reid_threshold(self, threshold):
        """Update Re-ID similarity threshold."""
        self.reid_matcher.set_threshold(threshold)
        logger.info(f"Re-ID threshold updated to {self.reid_matcher.threshold:.2f}")

    def toggle_child_prior(self, active=None):
        if active is None:
            self.reid_matcher.child_prior = not self.reid_matcher.child_prior
        else:
            self.reid_matcher.child_prior = bool(active)

    # ── Model Loading ────────────────────────────────────────
    def _load_models(self):
        """
        Load high-accuracy models:
        1. YOLOv8n (Standard Object Detector) for sharp person detection at distance and on floor
        2. YOLOv8n-pose for keypoints and CNDP privacy masking
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed — run: pip install ultralytics")
            self.det_model = None
            self.pose_model = None
            return

        # 1. Standard Object Detector (YOLOv8n)
        try:
            self.det_model = YOLO("yolov8n.pt")
            logger.info("Loaded YOLOv8n (Standard Object Detector) ⚡")
        except Exception as e:
            logger.error(f"Detector load failed: {e}")
            self.det_model = None

        # 2. Pose Model (YOLOv8n-pose)
        try:
            ov_model_dir = "yolov8n-pose_openvino_model"
            if os.path.isdir(ov_model_dir):
                self.pose_model = YOLO(ov_model_dir, task="pose")
                logger.info("Loaded pre-exported OpenVINO pose model ⚡")
            else:
                self.pose_model = YOLO("yolov8n-pose.pt")
                logger.info("Loaded YOLOv8n-pose (PyTorch) ⚡")
        except Exception as e:
            logger.warning(f"Pose model optional load: {e}")
            self.pose_model = None

    # ─────────────────────────────────────────────────────────
    #  CNDP Law 09-08  — Privacy Mask Pipeline
    # ─────────────────────────────────────────────────────────
    def _get_head_region(self, keypoints, bbox, conf_threshold=0.5):
        """
        Compute head bounding box from face keypoints.
        Fallback: top 28 % of person bounding box.

        Returns (x1, y1, x2, y2) clipped to frame bounds, or None.
        """
        x1, y1, x2, y2 = map(int, bbox[:4])

        if keypoints is not None and len(keypoints) > 0:
            face_pts = []
            for idx in FACE_KP_INDICES:
                if idx < len(keypoints):
                    kp = keypoints[idx]
                    # keypoints shape: (x, y, conf)
                    if len(kp) >= 3 and float(kp[2]) > conf_threshold:
                        face_pts.append((int(kp[0]), int(kp[1])))

            if len(face_pts) >= 2:
                xs = [p[0] for p in face_pts]
                ys = [p[1] for p in face_pts]
                hx1, hy1 = min(xs), min(ys)
                hx2, hy2 = max(xs), max(ys)

                # Expand to cover full head
                w = max(hx2 - hx1, 1)
                h = max(hy2 - hy1, 1)
                expand = max(w, h) * HEAD_EXPAND
                hx1 = int(hx1 - expand)
                hy1 = int(hy1 - expand)
                hx2 = int(hx2 + expand)
                hy2 = int(hy2 + expand)
                return (max(0, hx1), max(0, hy1), hx2, hy2)

        # Fallback: top portion of person bbox
        head_h = int((y2 - y1) * 0.28)
        return (x1, y1, x2, y1 + head_h)

    def _apply_privacy_mask(self, frame, persons, alert_indices):
        """
        CNDP Law 09-08 compliance:
        Gaussian-blur every person's head region, EXCEPT those whose
        index appears in `alert_indices` (confirmed critical detection).
        Blurred heads get a green 'CNDP-0908 PRIVACY' badge.
        """
        h, w = frame.shape[:2]

        for idx, person in enumerate(persons):
            if idx in alert_indices:
                continue  # Un-mask: this person triggered a critical alert

            head = self._get_head_region(person.get("keypoints"), person["bbox"])
            if head is None:
                continue

            hx1, hy1, hx2, hy2 = head
            hx1, hy1 = max(0, hx1), max(0, hy1)
            hx2, hy2 = min(w, hx2), min(h, hy2)
            if hx2 <= hx1 or hy2 <= hy1:
                continue

            # Gaussian blur
            roi = frame[hy1:hy2, hx1:hx2]
            if roi.size > 0:
                blurred = cv2.GaussianBlur(roi, BLUR_KERNEL, BLUR_SIGMA)
                frame[hy1:hy2, hx1:hx2] = blurred

            # Green CNDP badge below the blurred region
            badge_y = min(hy2 + 2, h - 18)
            badge_w = 150
            cv2.rectangle(
                frame,
                (hx1, badge_y),
                (hx1 + badge_w, badge_y + 16),
                (0, 160, 80),
                -1,
            )
            cv2.putText(
                frame,
                "CNDP-0908 PRIVACY",
                (hx1 + 4, badge_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame

    # ─────────────────────────────────────────────────────────
    #  Mode 1 — Najda-Vision: Fall & Medical Collapse Triage
    # ─────────────────────────────────────────────────────────
    def _detect_fall(self, keypoints, bbox):
        """
        Dual-path fall & syncope/collapse analysis:
            Path 1: Horizontal fall (slip/trip) where torso angle > 55° OR AR > 1.25
            Returns:
                (is_horizontal_fall, torso_angle_deg, has_pose_angle, aspect_ratio, bw, bh)
        """
        bw = float(bbox[2]) - float(bbox[0])
        bh = float(bbox[3]) - float(bbox[1])
        aspect_ratio = bw / max(bh, 1.0)

        angle_from_vertical = 0.0
        has_pose_angle = False
        is_upright_upper_body = False

        if keypoints is not None and len(keypoints) >= 13:
            ls = keypoints[KP_LEFT_SHOULDER]
            rs = keypoints[KP_RIGHT_SHOULDER]
            lh = keypoints[KP_LEFT_HIP]
            rh = keypoints[KP_RIGHT_HIP]

            def conf(kp):
                return float(kp[2]) if len(kp) >= 3 else 1.0

            # Compute midpoint if at least one shoulder & one hip are detected
            s_pts = [p for p in [ls, rs] if conf(p) > 0.25]
            h_pts = [p for p in [lh, rh] if conf(p) > 0.25]

            # Check head-to-shoulder vertical relationship:
            # If face/head is vertically above the shoulders, the upper body is upright!
            face_kps = [keypoints[idx] for idx in FACE_KP_INDICES if idx < len(keypoints) and conf(keypoints[idx]) > 0.25]
            if face_kps and s_pts:
                s_mid_y = sum(float(p[1]) for p in s_pts) / len(s_pts)
                s_mid_x = sum(float(p[0]) for p in s_pts) / len(s_pts)
                head_y = sum(float(p[1]) for p in face_kps) / len(face_kps)
                head_x = sum(float(p[0]) for p in face_kps) / len(face_kps)
                head_dy = s_mid_y - head_y  # In image coords, y increases downwards. head_dy > 0 means head is above shoulders
                head_dx = abs(head_x - s_mid_x)
                if head_dy > 10 and head_dy > head_dx * 0.7:
                    is_upright_upper_body = True

            if s_pts and h_pts:
                s_mid = (sum(float(p[0]) for p in s_pts) / len(s_pts),
                         sum(float(p[1]) for p in s_pts) / len(s_pts))
                h_mid = (sum(float(p[0]) for p in h_pts) / len(h_pts),
                         sum(float(p[1]) for p in h_pts) / len(h_pts))

                dx = h_mid[0] - s_mid[0]
                dy = h_mid[1] - s_mid[1]
                if abs(dy) > 1e-6 or abs(dx) > 1e-6:
                    angle_from_vertical = math.degrees(math.atan2(abs(dx), abs(dy)))
                    has_pose_angle = True

        # Path 1: Horizontal fall condition (genuine horizontal body orientation)
        if has_pose_angle:
            is_horizontal_fall = (angle_from_vertical > FALL_ANGLE_THRESHOLD)
            if is_upright_upper_body and angle_from_vertical < 35:
                is_horizontal_fall = False
        else:
            # Fallback only when keypoints are missing on ground level: wider than tall
            if is_upright_upper_body:
                is_horizontal_fall = False
            else:
                is_horizontal_fall = (aspect_ratio > FALL_ASPECT_RATIO and bh < 180)

        return is_horizontal_fall, angle_from_vertical, has_pose_angle, aspect_ratio, bw, bh

    def _mode1_najda(self, frame, results):
        """
        Najda-Vision pipeline:
            1. Extract poses & boxes from YOLO
            2. Compute geometric indicators (angle, aspect ratio, height)
            3. Track temporal baseline height & confirm fall/faint via PersonTracker
            4. Support Ghost Anchor for victims lost by detector on the floor
            5. Trigger alerts + green-wave corridor
        """
        persons = []
        alert_indices = set()
        new_alerts = []

        if results is None or len(results) == 0:
            self.tracker.update([])
            ghost_victims = self.tracker.get_ghost_victims()
            return persons, alert_indices, new_alerts, ghost_victims

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            self.tracker.update([])
            ghost_victims = self.tracker.get_ghost_victims()
            return persons, alert_indices, new_alerts, ghost_victims

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        kps_data = None
        if result.keypoints is not None:
            kps_data = result.keypoints.data.cpu().numpy()

        tracker_detections = []

        for i in range(len(boxes)):
            bbox = boxes[i]
            keypoints = kps_data[i] if (kps_data is not None and i < len(kps_data)) else None

            is_horiz, angle, has_angle, aspect, bw, bh = self._detect_fall(keypoints, bbox)

            cx = (float(bbox[0]) + float(bbox[2])) / 2
            cy = (float(bbox[1]) + float(bbox[3])) / 2

            tracker_detections.append({
                "cx": cx,
                "cy": cy,
                "bbox": bbox,
                "bw": bw,
                "bh": bh,
                "aspect_ratio": aspect,
                "is_horizontal_fall": is_horiz,
                "angle": angle,
                "has_angle": has_angle,
            })

            person = {
                "bbox": bbox,
                "confidence": float(confs[i]),
                "keypoints": keypoints,
                "is_falling": is_horiz,
                "fall_angle": angle,
                "fall_aspect": aspect,
                "bh": bh,
            }
            persons.append(person)

        # Temporal confirmation via tracker with baseline standing height tracking
        track_results = self.tracker.update(tracker_detections)

        for i, (track_id, confirmed_fall, fall_reason) in enumerate(track_results):
            if i >= len(persons):
                break
            persons[i]["track_id"] = track_id
            persons[i]["confirmed_fall"] = confirmed_fall
            persons[i]["fall_reason"] = fall_reason

            if confirmed_fall:
                alert_indices.add(i)

                alert_key = f"fall_{track_id}"
                if alert_key not in self._logged_alert_keys:
                    self._logged_alert_keys.add(alert_key)
                    new_alerts.append(
                        {
                            "type": "fall",
                            "key": alert_key,
                            "severity": "critical",
                            "message": f"⚠ {fall_reason or 'COLLAPSE DETECTED'} — Person #{track_id}",
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "details": (
                                f"Detection: {fall_reason} | "
                                f"Angle {persons[i]['fall_angle']:.0f}° | "
                                f"Persistent Anchor Active"
                            ),
                            "action": (
                                "Protection Civile dispatched → "
                                "Green Wave corridor on Boulevard Mekka"
                            ),
                        }
                    )
                    # Activate green-wave corridor
                    self.green_wave_active = True
                    self.green_wave_start = time.time()
                    logger.warning(
                        f"🚨 {fall_reason or 'EMERGENCY COLLAPSE'} — Person #{track_id} | "
                        f"Green Wave on Blvd Mekka ({GREEN_WAVE_DURATION}s)"
                    )

        ghost_victims = self.tracker.get_ghost_victims()
        # If there are active ghost victims, sustain green wave
        if ghost_victims and not self.green_wave_active:
            self.green_wave_active = True
            self.green_wave_start = time.time()

        return persons, alert_indices, new_alerts, ghost_victims

    # ─────────────────────────────────────────────────────────
    #  Mode 2 — Tifawt-Guard: Missing Person & Amber Alert Search
    # ─────────────────────────────────────────────────────────
    def _mode2_tifawt(self, frame, results):
        """
        Mode 2: Tifawt-Guard — Missing Person & Amber Alert Search
        1. Extract bounding boxes from YOLO detector.
        2. Track persons temporally using PersonTracker.
        3. Extract part-based features (Torso + Legs) for each detected person crop.
        4. Match against active reference target via TifawtReIDMatcher.
        5. Trigger alert with snapshot thumbnail when sustained match >= threshold (3 frames).
        """
        persons = []
        alert_indices = set()
        new_alerts = []
        match_persons = []

        if results is None or len(results) == 0:
            self.tracker.update([])
            return persons, alert_indices, new_alerts, match_persons

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            self.tracker.update([])
            return persons, alert_indices, new_alerts, match_persons

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        tracker_detections = []
        h_frame, w_frame = frame.shape[:2]

        for i in range(len(boxes)):
            bbox = boxes[i]
            bw = float(bbox[2]) - float(bbox[0])
            bh = float(bbox[3]) - float(bbox[1])
            aspect = bw / max(bh, 1.0)
            cx = (float(bbox[0]) + float(bbox[2])) / 2.0
            cy = (float(bbox[1]) + float(bbox[3])) / 2.0

            tracker_detections.append({
                "cx": cx,
                "cy": cy,
                "bbox": bbox,
                "bw": bw,
                "bh": bh,
                "aspect_ratio": aspect,
                "is_horizontal_fall": False,
                "angle": 0.0,
                "has_angle": False,
            })

            person = {
                "bbox": bbox,
                "confidence": float(confs[i]),
                "keypoints": None,
                "aspect": aspect,
                "bw": bw,
                "bh": bh,
                "similarity": 0.0,
                "torso_sim": 0.0,
                "legs_sim": 0.0,
                "is_match": False,
                "crop": None
            }
            persons.append(person)

        # Update tracker
        track_results = self.tracker.update(tracker_detections)

        for i, (track_id, _, _) in enumerate(track_results):
            if i >= len(persons):
                break
            persons[i]["track_id"] = track_id

            # Extract crop
            bx1, by1, bx2, by2 = map(int, persons[i]["bbox"])
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w_frame, bx2), min(h_frame, by2)

            if bx2 - bx1 > 15 and by2 - by1 > 30:
                crop = frame[by1:by2, bx1:bx2]
                persons[i]["crop"] = crop

                tr = self.tracker.tracks.get(track_id)
                if tr is not None and self.reid_matcher.has_target():
                    sim, sim_h, sim_t, sim_l = self.reid_matcher.compute_similarity(crop)

                    # Exponential smoothing across track to prevent single-frame noise
                    prev_sim = tr.get("reid_sim", 0.0)
                    smoothed_sim = 0.65 * sim + 0.35 * prev_sim if prev_sim > 0 else sim
                    tr["reid_sim"] = smoothed_sim
                    tr["reid_head_sim"] = sim_h
                    tr["reid_torso_sim"] = sim_t
                    tr["reid_legs_sim"] = sim_l

                    persons[i]["similarity"] = smoothed_sim
                    persons[i]["head_sim"] = sim_h
                    persons[i]["torso_sim"] = sim_t
                    persons[i]["legs_sim"] = sim_l

                    if smoothed_sim >= self.reid_matcher.threshold:
                        tr["reid_match_count"] = tr.get("reid_match_count", 0) + 1
                    else:
                        tr["reid_match_count"] = max(0, tr.get("reid_match_count", 0) - 1)

                    # Require 3 frames of sustained match
                    if tr["reid_match_count"] >= 3:
                        persons[i]["is_match"] = True
                        tr["is_reid_match"] = True
                        alert_indices.add(i)
                        match_persons.append(persons[i])

                        alert_key = f"reid_{track_id}"
                        if alert_key not in self._logged_alert_keys:
                            self._logged_alert_keys.add(alert_key)
                            b64_thumb = crop_to_base64(crop, max_size=140)
                            target_ref = self.reid_matcher.reference_target
                            new_alerts.append({
                                "id": f"reid_{track_id}_{int(time.time())}",
                                "key": alert_key,
                                "type": "missing_person_match",
                                "severity": "critical",
                                "camera_id": "CAM 04",
                                "cam_id": "CAM 04 (Sector Al-Quds)",
                                "zone_name": "Sector Al-Quds / CCTV-04",
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "similarity": round(smoothed_sim * 100, 1),
                                "head_sim": round(sim_h * 100, 1),
                                "torso_sim": round(sim_t * 100, 1),
                                "legs_sim": round(sim_l * 100, 1),
                                "message": f"🎯 TARGET ACQUIRED ({round(smoothed_sim*100, 1)}%) — Person #{track_id}",
                                "details": f"Head: {round(sim_h*100, 1)}% | Torso: {round(sim_t*100, 1)}% | Legs: {round(sim_l*100, 1)}%",
                                "action": "Awaiting Operator Dispatch / Confirmation",
                                "candidate_thumb": b64_thumb,
                                "thumbnail": b64_thumb,
                                "reference_thumb": target_ref.get("thumbnail", ""),
                                "target_label": target_ref.get("label", "Target"),
                                "target_head_hex": target_ref.get("head_hex", "#dc2626"),
                                "target_torso_hex": target_ref.get("torso_hex", "#64748b"),
                                "target_legs_hex": target_ref.get("legs_hex", "#64748b"),
                                "status": "UNACKNOWLEDGED"
                            })
                            logger.warning(
                                f"🎯 [Tifawt-Guard] TARGET ACQUIRED ({round(smoothed_sim*100, 1)}%) — "
                                f"Person #{track_id} in CAM 04 (Sector Al-Quds)"
                            )

        return persons, alert_indices, new_alerts, match_persons

    # ─────────────────────────────────────────────────────────
    #  Drawing: Mode-Specific Overlays
    # ─────────────────────────────────────────────────────────
    def _draw_mode1_overlays(self, frame, persons, alert_indices, ghost_victims=None):
        """Draw bounding boxes, person IDs, fall/faint alerts, and Ghost Anchor markers."""
        # 1. Active person detections
        for idx, person in enumerate(persons):
            x1, y1, x2, y2 = map(int, person["bbox"][:4])

            if idx in alert_indices:
                # ── Critical alert: red box ──
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                reason = person.get("fall_reason") or "COLLAPSE DETECTED"
                label = f"{reason} #{person.get('track_id', '?')}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                )
                cv2.rectangle(
                    frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), (0, 0, 255), -1
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Diagnostic readout below box
                angle_lbl = f"Angle: {person.get('fall_angle', 0):.0f} deg | Aspect: {person.get('fall_aspect', 0):.2f}"
                cv2.putText(
                    frame,
                    angle_lbl,
                    (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 100, 255),
                    1,
                    cv2.LINE_AA,
                )
            else:
                # ── Normal person: subtle green box ──────────
                color = (0, 200, 100)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

                if "track_id" in person:
                    cv2.putText(
                        frame,
                        f"P#{person['track_id']}",
                        (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

        # 2. Ghost Anchor: Fallen Victims currently lost by YOLO (Occluded / Floor Level)
        if ghost_victims:
            for ghost in ghost_victims:
                gb = ghost.get("bbox")
                if not gb:
                    continue
                gx1, gy1, gx2, gy2 = map(int, gb[:4])
                gx1, gy1 = max(0, gx1), max(0, gy1)
                gx2, gy2 = min(frame.shape[1], gx2), min(frame.shape[0], gy2)

                # Pulsing emergency red
                pulse = 0.6 + 0.4 * math.sin(time.time() * 8)
                beacon_color = (0, 0, int(255 * pulse))

                # Draw beacon bounding box
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), beacon_color, 2)

                # Corner accent lines
                c_len = max(12, min(24, (gx2 - gx1) // 4))
                cv2.line(frame, (gx1, gy1), (gx1 + c_len, gy1), (0, 255, 255), 3)
                cv2.line(frame, (gx1, gy1), (gx1, gy1 + c_len), (0, 255, 255), 3)
                cv2.line(frame, (gx2, gy2), (gx2 - c_len, gy2), (0, 255, 255), 3)
                cv2.line(frame, (gx2, gy2), (gx2, gy2 - c_len), (0, 255, 255), 3)

                # Anchor Header
                lost_sec = ghost.get("lost_seconds", 0)
                g_lbl = f"VICTIM DOWN (LAST KNOWN POS) • Lost for {lost_sec:.1f}s #{ghost['track_id']}"
                (gw, gh), _ = cv2.getTextSize(g_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
                cv2.rectangle(frame, (gx1, gy1 - gh - 10), (gx1 + gw + 8, gy1), (0, 0, 180), -1)
                cv2.putText(
                    frame, g_lbl, (gx1 + 4, gy1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                )

                # Anchor Subtext
                cv2.putText(
                    frame, "OCCLUDED / FLOOR LEVEL - PERSISTENT BEACON",
                    (gx1, gy2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA
                )

        return frame

    def _draw_mode2_overlays(self, frame, persons, alert_indices, match_persons=None):
        """Draw bounding boxes, person IDs, Re-ID match locks, and confidence scores for Mode 2."""
        h, w = frame.shape[:2]

        for p in persons:
            x1, y1, x2, y2 = map(int, p["bbox"][:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            tid = p.get("track_id", "?")
            sim = p.get("similarity", 0.0)
            is_match = p.get("is_match", False)

            if is_match:
                # ── TARGET LOCK (Bright Emerald / Amber Corner Brackets) ──
                pulse = 0.7 + 0.3 * math.sin(time.time() * 8)
                color = (0, int(255 * pulse), int(120 * pulse))

                # Double border
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 200, 255), 1)

                # Target lock corner brackets
                c_len = min(25, max(8, (x2 - x1) // 3), max(8, (y2 - y1) // 3))
                cv2.line(frame, (x1, y1), (x1 + c_len, y1), (0, 255, 255), 4)
                cv2.line(frame, (x1, y1), (x1, y1 + c_len), (0, 255, 255), 4)
                cv2.line(frame, (x2, y1), (x2 - c_len, y1), (0, 255, 255), 4)
                cv2.line(frame, (x2, y1), (x2, y1 + c_len), (0, 255, 255), 4)
                cv2.line(frame, (x1, y2), (x1 + c_len, y2), (0, 255, 255), 4)
                cv2.line(frame, (x1, y2), (x1, y2 - c_len), (0, 255, 255), 4)
                cv2.line(frame, (x2, y2), (x2 - c_len, y2), (0, 255, 255), 4)
                cv2.line(frame, (x2, y2), (x2, y2 - c_len), (0, 255, 255), 4)

                # Top label banner
                lbl = f"TARGET MATCH: {sim*100:.1f}% [#{tid}]"
                lbl_sz = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
                lbl_y = max(24, y1 - 8)
                cv2.rectangle(frame, (x1, lbl_y - 20), (x1 + lbl_sz[0] + 12, lbl_y + 4), (0, 180, 80), -1)
                cv2.putText(frame, lbl, (x1 + 6, lbl_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                # Head / Torso / Leg match indicators
                h_lbl = f"Head:{p.get('head_sim', 0)*100:.0f}%"
                t_lbl = f"Torso:{p.get('torso_sim', 0)*100:.0f}%"
                l_lbl = f"Legs:{p.get('legs_sim', 0)*100:.0f}%"
                cv2.putText(frame, f"{h_lbl} | {t_lbl} | {l_lbl}", (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                # Normal pedestrian in Mode 2
                if self.reid_matcher.has_target() and sim > 0.35:
                    color = (255, 180, 50)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    lbl = f"ID #{tid} ({sim*100:.0f}%)"
                    cv2.putText(frame, lbl, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
                else:
                    color = (180, 160, 80)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    lbl = f"ID #{tid}"
                    cv2.putText(frame, lbl, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        return frame

    # ─────────────────────────────────────────────────────────
    #  Drawing: Heads-Up Display
    # ─────────────────────────────────────────────────────────
    def _draw_hud(self, frame):
        """Overlay status bar, FPS, mode label, and green-wave indicator."""
        h, w = frame.shape[:2]

        # ── Top bar ──────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(
            frame, "SAHRA-EYE", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2, cv2.LINE_AA,
        )

        mode_label = f"Mode {self.current_mode}: {self.mode_names.get(self.current_mode, '?')}  [{self.source_name}]"
        cv2.putText(
            frame, mode_label, (180, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
        )

        # FPS — colour-coded
        fps_colour = (
            (0, 255, 0) if self.fps >= 25
            else (0, 200, 255) if self.fps >= 15
            else (0, 0, 255)
        )
        cv2.putText(
            frame, f"FPS: {self.fps:.1f}", (w - 130, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_colour, 1, cv2.LINE_AA,
        )

        cv2.putText(
            frame, f"Persons: {self.persons_count}", (w - 270, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # ── Bottom CNDP bar ──────────────────────────────────
        if self.enable_cndp_blur:
            cv2.rectangle(frame, (0, h - 24), (w, h), (0, 70, 40), -1)
            cv2.putText(
                frame,
                "CNDP Law 09-08 | Privacy-First Active | All faces masked by default",
                (10, h - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (200, 255, 200),
                1,
                cv2.LINE_AA,
            )

        # ── Mode 2 HUD Info Bar ──────────────────────────────
        if self.current_mode == 2:
            if self.reid_matcher.has_target():
                t_lbl = self.reid_matcher.reference_target.get("label", "Target")
                thresh_pct = int(self.reid_matcher.threshold * 100)
                hud_txt = f"🔍 Tifawt-Guard ACTIVE | Target: {t_lbl} | Min Match: {thresh_pct}%"
                cv2.rectangle(frame, (0, h - 26), (w, h), (20, 30, 20), -1)
                cv2.putText(
                    frame, hud_txt, (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 180), 1, cv2.LINE_AA,
                )
            else:
                cv2.rectangle(frame, (0, h - 26), (w, h), (30, 20, 20), -1)
                cv2.putText(
                    frame, "⚠ Tifawt-Guard STANDBY — Awaiting Target Photo / Preset Selection", (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA,
                )

        # ── Green-wave flashing indicator ────────────────────
        if self.green_wave_active and self.current_mode == 1:
            elapsed = time.time() - self.green_wave_start
            if elapsed > GREEN_WAVE_DURATION:
                self.green_wave_active = False
            else:
                remaining = GREEN_WAVE_DURATION - elapsed
                alpha = 0.5 + 0.5 * math.sin(time.time() * 5)
                colour = (0, int(255 * alpha), int(80 * alpha))
                gw_txt = f"GREEN WAVE ACTIVE — Blvd Mekka  ({remaining:.0f}s)"
                cv2.putText(
                    frame, gw_txt, (10, h - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA,
                )

        return frame

    # ─────────────────────────────────────────────────────────
    #  Demo-Mode Synthetic Frame
    # ─────────────────────────────────────────────────────────
    def _generate_demo_frame(self):
        """Fallback frame when no camera is connected."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (25, 25, 35)

        # Subtle grid
        for x in range(0, 640, 40):
            cv2.line(frame, (x, 0), (x, 480), (35, 35, 45), 1)
        for y in range(0, 480, 40):
            cv2.line(frame, (0, y), (640, y), (35, 35, 45), 1)

        cv2.putText(
            frame, "SAHRA-EYE", (170, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 136), 3, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "No Camera — Demo Mode", (185, 250),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "Connect a webcam or RTSP feed to begin", (120, 290),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA,
        )
        return frame

    # ─────────────────────────────────────────────────────────
    #  FPS Calculation (Smoothed)
    # ─────────────────────────────────────────────────────────
    def _update_fps(self):
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            self._fps_samples.append(1.0 / dt)
            if len(self._fps_samples) > 30:
                self._fps_samples.pop(0)
            self.fps = sum(self._fps_samples) / len(self._fps_samples)

    # ─────────────────────────────────────────────────────────
    #  Main Processing Pipeline
    # ─────────────────────────────────────────────────────────
    def process_frame(self):
        """
        One tick of the vision pipeline.

        Returns:
            (annotated_frame: np.ndarray, telemetry: dict)
        """
        self._update_fps()

        # ── Read frame ───────────────────────────────────────
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.is_video_file:
                    # Seamless loop for uploaded CCTV / test video files
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                if not ret or frame is None:
                    frame = self._generate_demo_frame()
        else:
            frame = self._generate_demo_frame()
            return frame, self.get_telemetry()

        # ── Model Inference ───────────────────────────────────
        # Determine dynamic image resolution:
        # 960px for high-res / wide CCTV feeds (sharp detection of distant people & collapsed figures)
        # 640px for live webcams (blazing fast 30+ FPS)
        imgsz = 960 if (self.is_video_file or frame.shape[1] >= 1280) else 640

        results = None
        if self.det_model is not None:
            try:
                results = self.det_model(frame, classes=[0], imgsz=imgsz, conf=0.30, verbose=False)
            except Exception as e:
                logger.error(f"Detection inference error: {e}")
        elif self.pose_model is not None:
            try:
                results = self.pose_model(frame, imgsz=imgsz, conf=0.30, verbose=False)
            except Exception as e:
                logger.error(f"Pose inference error: {e}")

        # ── Mode dispatch ────────────────────────────────────
        persons = []
        alert_indices = set()
        new_alerts = []
        ghost_victims = []
        match_persons = []

        if self.current_mode == 1:
            persons, alert_indices, new_alerts, ghost_victims = self._mode1_najda(frame, results)
        elif self.current_mode == 2:
            persons, alert_indices, new_alerts, match_persons = self._mode2_tifawt(frame, results)

        self.persons_count = len(persons)

        # ── Store alerts ─────────────────────────────────────
        for alert in new_alerts:
            self.alerts.append(alert)
            if len(self.alerts) > self.max_alerts:
                self.alerts.pop(0)

        # ── Privacy mask (CNDP 09-08) [Disabled for testing/speed] ──
        if self.enable_cndp_blur:
            frame = self._apply_privacy_mask(frame, persons, alert_indices)

        # ── Mode overlays ────────────────────────────────────
        if self.current_mode == 1:
            frame = self._draw_mode1_overlays(frame, persons, alert_indices, ghost_victims)
        elif self.current_mode == 2:
            frame = self._draw_mode2_overlays(frame, persons, alert_indices, match_persons)

        # ── HUD ──────────────────────────────────────────────
        frame = self._draw_hud(frame)

        return frame, self.get_telemetry()

    # ─────────────────────────────────────────────────────────
    #  Public Interface
    # ─────────────────────────────────────────────────────────
    def set_mode(self, mode):
        """Switch operational mode (resets tracker state)."""
        if mode in self.mode_names:
            self.current_mode = mode
            self.tracker = PersonTracker()
            self._logged_alert_keys.clear()
            logger.info(f"Switched → Mode {mode}: {self.mode_names[mode]}")

    def clear_victims(self):
        """Manually clear all fallen victim markers and alerts (operator dismissal)."""
        self.tracker.clear_victims()
        self.green_wave_active = False
        self._logged_alert_keys.clear()
        logger.info("Cleared all fallen victim markers and emergency corridors.")

    def get_telemetry(self):
        """Snapshot of current engine telemetry for the dashboard."""
        gw_remaining = 0
        if self.green_wave_active:
            gw_remaining = max(
                0, GREEN_WAVE_DURATION - (time.time() - self.green_wave_start)
            )
        ghosts = self.tracker.get_ghost_victims()

        target_info = None
        if self.reid_matcher.has_target():
            rt = self.reid_matcher.reference_target
            target_info = {
                "label": rt.get("label", "Target"),
                "thumbnail": rt.get("thumbnail", ""),
                "head_hex": rt.get("head_hex", "#dc2626"),
                "torso_hex": rt.get("torso_hex", "#64748b"),
                "legs_hex": rt.get("legs_hex", "#64748b"),
                "has_head_item": rt.get("has_head_item", True),
                "aspect_ratio": round(rt.get("aspect_ratio", 0.35), 2),
                "timestamp": rt.get("timestamp", "")
            }

        return {
            "fps": round(self.fps, 1),
            "mode": self.current_mode,
            "mode_name": self.mode_names.get(self.current_mode, "Unknown"),
            "source_name": self.source_name,
            "is_video_file": self.is_video_file,
            "persons_count": self.persons_count,
            "ghost_victims_count": len(ghosts),
            "alerts": self.alerts[-20:],
            "alert_count": len(self.alerts),
            "green_wave_active": self.green_wave_active,
            "green_wave_remaining": round(gw_remaining, 1),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            # Mode 2: Tifawt-Guard fields
            "target_active": self.reid_matcher.has_target(),
            "target_info": target_info,
            "reid_threshold": round(self.reid_matcher.threshold, 2),
            "child_prior": self.reid_matcher.child_prior,
        }

    def release(self):
        """Release camera and clean up."""
        if self.cap:
            self.cap.release()
        logger.info("Engine released")
