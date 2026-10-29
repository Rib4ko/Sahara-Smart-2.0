"""
╔══════════════════════════════════════════════════════════════╗
║  Sahra-Eye — FastAPI Server & Dashboard                      ║
║  Laâyoune Smart City Edge AI Vision Engine                   ║
║                                                               ║
║  Endpoints:                                                   ║
║    GET  /            → Dark-mode Tailwind dashboard            ║
║    GET  /video_feed  → MJPEG live stream                       ║
║    GET  /telemetry   → JSON engine telemetry                   ║
║    POST /mode        → Switch operational mode                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import shutil
import threading
import time
import logging
import json
import requests as req

import cv2
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, FileResponse

from vision_engine import SahraEyeEngine

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[Sahra-Eye] %(levelname)s: %(message)s")
logger = logging.getLogger("sahra-eye-server")

# ── Load Groq API Key securely from environment or local .env (never committed) ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY="):
                        GROQ_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        except Exception as e:
            logger.warning(f"Could not read .env file: {e}")

if GROQ_API_KEY:
    logger.info("Groq API Key loaded securely from environment / .env")
else:
    logger.warning("GROQ_API_KEY not found in environment or .env")

# ──────────────────────────────────────────────────────────────
#  FastAPI Application
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sahra-Eye",
    description="Laâyoune Smart City Edge AI Vision Engine",
    version="1.0.0",
)

# ── Global State ─────────────────────────────────────────────
engine: SahraEyeEngine = None          # type: ignore
latest_frame_bytes: bytes = None       # type: ignore
latest_telemetry: dict = {}
frame_lock = threading.Lock()


# ── Background Processing Thread ─────────────────────────────
def processing_loop():
    """Continuously capture + process frames in background."""
    global latest_frame_bytes, latest_telemetry

    while True:
        if engine is None:
            time.sleep(0.1)
            continue
        try:
            frame, telemetry = engine.process_frame()
            if frame is not None:
                _, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                with frame_lock:
                    latest_frame_bytes = buffer.tobytes()
                    latest_telemetry = telemetry
        except Exception:
            logger.exception("Processing loop error")
            time.sleep(0.1)


def generate_mjpeg():
    """Yield MJPEG frames for the StreamingResponse."""
    while True:
        with frame_lock:
            frame = latest_frame_bytes
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(0.033)  # ~30 fps cap


# ── Lifecycle ────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global engine
    logger.info("Starting Sahra-Eye engine…")
    default_m1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Elder heart atack.mp4")
    src = default_m1_path if os.path.exists(default_m1_path) else 0
    engine = SahraEyeEngine(source=src)
    thread = threading.Thread(target=processing_loop, daemon=True)
    thread.start()
    logger.info(f"Engine running — dashboard at http://localhost:8000 (Source: {engine.source_name})")


@app.on_event("shutdown")
async def shutdown_event():
    if engine:
        engine.release()


# ── API Endpoints ────────────────────────────────────────────
@app.get("/video_feed")
async def video_feed():
    """Live annotated MJPEG video stream."""
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/cctv1_feed")
async def cctv1_feed():
    """Serve cctv1.mp4 video file with support for range streaming."""
    cctv1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cctv1.mp4")
    if not os.path.exists(cctv1_path):
        return JSONResponse(status_code=404, content={"error": "cctv1.mp4 not found"})
    return FileResponse(cctv1_path, media_type="video/mp4")


@app.get("/cctv2_feed")
async def cctv2_feed():
    """Serve cctv2.mp4 video file with support for range streaming."""
    cctv2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cctv2.mp4")
    if not os.path.exists(cctv2_path):
        return JSONResponse(status_code=404, content={"error": "cctv2.mp4 not found"})
    return FileResponse(cctv2_path, media_type="video/mp4")


@app.get("/cctv3_feed")
async def cctv3_feed():
    """Serve cctv3.jpg photo file."""
    cctv3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cctv3.jpg")
    if not os.path.exists(cctv3_path):
        return JSONResponse(status_code=404, content={"error": "cctv3.jpg not found"})
    return FileResponse(cctv3_path, media_type="image/jpeg")


@app.get("/telemetry")
async def telemetry():
    """Current engine telemetry as JSON."""
    with frame_lock:
        data = latest_telemetry.copy() if latest_telemetry else {}
    return JSONResponse(content=data)


@app.post("/mode")
async def set_mode(request: Request):
    """Switch operational mode and automatically load mode-specific default video."""
    body = await request.json()
    mode = int(body.get("mode", 1))
    if engine:
        engine.set_mode(mode)
        # Switch to mode-specific default video
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if mode == 1:
            m1_path = os.path.join(base_dir, "Elder heart atack.mp4")
            if os.path.exists(m1_path):
                engine.switch_source(m1_path, source_name="Elder heart atack.mp4")
        elif mode == 2:
            m2_path = os.path.join(base_dir, "kid missing.mp4")
            if os.path.exists(m2_path):
                engine.switch_source(m2_path, source_name="kid missing.mp4")
    return JSONResponse(content={"status": "ok", "mode": mode, "source": getattr(engine, "source_name", "Unknown")})


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a CCTV video file and immediately switch the vision engine to process it."""
    filename = file.filename or "cctv_upload.mp4"
    safe_name = "".join(c for c in filename if c.isalnum() or c in (".", "_", "-"))
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if engine:
        engine.switch_source(file_path, source_name=safe_name)
    logger.info(f"Uploaded and switched to CCTV video: {safe_name}")
    return JSONResponse(content={"status": "ok", "filename": safe_name})


@app.post("/reset_camera")
async def reset_camera():
    """Switch back to the live webcam."""
    if engine:
        engine.switch_source(0, source_name="Live Webcam")
    logger.info("Switched back to Live Webcam")
    return JSONResponse(content={"status": "ok", "source": "Live Webcam"})


@app.get("/list_videos")
async def list_videos():
    """List available test CCTV videos in uploads/."""
    videos = []
    if os.path.exists(UPLOAD_DIR):
        for f in os.listdir(UPLOAD_DIR):
            if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm")):
                videos.append(f)
    return JSONResponse(content={"videos": sorted(videos)})


@app.post("/switch_source")
async def switch_source(request: Request):
    """Switch vision engine source to any uploaded video file."""
    body = await request.json()
    filename = body.get("filename") or body.get("source")
    if not filename:
        return JSONResponse(status_code=400, content={"error": "Missing filename"})

    file_path = filename if os.path.isabs(filename) else os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": f"File not found: {filename}"})

    if engine:
        engine.switch_source(file_path, source_name=os.path.basename(file_path))
    return JSONResponse(content={"status": "ok", "source": os.path.basename(file_path)})


@app.post("/clear_alerts")
async def clear_alerts():
    """Clear all active fallen victim markers and alerts (operator dismissal)."""
    if engine:
        engine.clear_victims()
    logger.info("Alerts & victim markers dismissed by operator")
    return JSONResponse(content={"status": "ok"})


# ── Tifawt-Guard (Mode 2) Endpoints ─────────────────────────
@app.post("/search_by_description")
async def search_by_description(request: Request):
    """
    Accept text description, query Groq LLM (openai/gpt-oss-20b),
    extract structured 3-part attributes (Head, Torso, Legs), and activate search.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    description = body.get("description", "").strip()
    if not description:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Description cannot be empty"})

    if not GROQ_API_KEY:
        return JSONResponse(status_code=500, content={"status": "error", "message": "GROQ_API_KEY not configured in environment or .env"})

    prompt = f"""Extract physical appearance search parameters for a missing person from this text description:
"{description}"

Return a strict JSON object with:
- target_label: short descriptive title (e.g. "Missing Child - Red Cap & Yellow Shirt")
- head_hex: hex color string (e.g. #dc2626)
- head_item: description (e.g. "red cap")
- has_head_item: boolean (true if cap, hat, beanie, hood, or hijab mentioned)
- torso_hex: hex color string (e.g. #eab308)
- torso_item: description (e.g. "yellow shirt")
- legs_hex: hex color string (e.g. #2563eb)
- legs_item: description (e.g. "blue jeans")
- child_prior: boolean (true if child, kid, boy, girl, young, minor)
- summary: one-sentence summary of the extracted appearance
"""

    t0 = time.time()
    models_to_try = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    groq_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    parsed_data = None
    last_err = ""
    for model_name in models_to_try:
        try:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert CCTV emergency dispatch search parser. Always output strictly valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1
            }
            resp = req.post(groq_url, headers=headers, json=payload, timeout=8)
            if resp.status_code == 200:
                raw_content = resp.json()["choices"][0]["message"]["content"]
                parsed_data = json.loads(raw_content)
                logger.info(f"Groq LLM parsed search query using {model_name}: {parsed_data.get('target_label')}")
                break
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text}"
        except Exception as e:
            last_err = str(e)

    if not parsed_data:
        return JSONResponse(status_code=502, content={"status": "error", "message": f"Groq LLM query failed: {last_err}"})

    label = parsed_data.get("target_label") or description[:40]
    head_hex = parsed_data.get("head_hex", "#dc2626")
    torso_hex = parsed_data.get("torso_hex", "#eab308")
    legs_hex = parsed_data.get("legs_hex", "#2563eb")
    has_head_item = bool(parsed_data.get("has_head_item", True))
    child_prior = bool(parsed_data.get("child_prior", True))

    if engine:
        engine.set_parametric_target(
            label=label,
            head_hex=head_hex,
            torso_hex=torso_hex,
            legs_hex=legs_hex,
            has_head_item=has_head_item,
            child_prior=child_prior
        )

    return JSONResponse(content={
        "status": "ok",
        "target_info": {
            "label": label,
            "head_hex": head_hex,
            "torso_hex": torso_hex,
            "legs_hex": legs_hex,
            "has_head_item": has_head_item,
            "child_prior": child_prior
        },
        "data": parsed_data,
        "inference_ms": round((time.time() - t0) * 1000)
    })


@app.post("/search_by_structured_target")
async def search_by_structured_target(request: Request):
    """
    Direct zero-latency preset bypass receiving structured schema without external API call.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    label = body.get("label", "Custom Target")
    head_hex = body.get("head_hex", "#dc2626")
    torso_hex = body.get("torso_hex", "#eab308")
    legs_hex = body.get("legs_hex", "#2563eb")
    has_head_item = bool(body.get("has_head_item", True))
    child_prior = bool(body.get("child_prior", True))

    if engine:
        engine.set_parametric_target(
            label=label,
            head_hex=head_hex,
            torso_hex=torso_hex,
            legs_hex=legs_hex,
            has_head_item=has_head_item,
            child_prior=child_prior
        )
        return JSONResponse(content={
            "status": "ok",
            "label": label,
            "head_hex": head_hex,
            "torso_hex": torso_hex,
            "legs_hex": legs_hex,
            "has_head_item": has_head_item,
            "child_prior": child_prior
        })
    return JSONResponse(status_code=500, content={"status": "error", "message": "Engine not running"})


@app.post("/set_target_preset")
async def set_target_preset(request: Request):
    """Set preconfigured reference target preset for Tifawt-Guard."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    preset_id = body.get("preset_id", "preset_1")
    if engine:
        ok, label = engine.set_target_preset(preset_id)
        return JSONResponse(content={"status": "ok" if ok else "error", "preset": preset_id, "label": label})
    return JSONResponse(content={"status": "error", "message": "Engine not running"}, status_code=500)


@app.post("/upload_target_photo")
async def upload_target_photo(file: UploadFile = File(...)):
    """Upload custom missing person photo, auto-crop with YOLO, extract Re-ID features."""
    import numpy as np
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(content={"status": "error", "message": "Invalid image format"}, status_code=400)
    if engine:
        label = file.filename or "Uploaded Target Photo"
        ok, msg = engine.set_target_photo(img, label=label)
        return JSONResponse(content={"status": "ok" if ok else "error", "message": msg, "label": label})
    return JSONResponse(content={"status": "error", "message": "Engine not running"}, status_code=500)


@app.post("/set_reid_threshold")
async def set_reid_threshold(request: Request):
    """Update Re-ID similarity threshold."""
    body = await request.json()
    threshold = float(body.get("threshold", 0.65))
    if engine:
        engine.set_reid_threshold(threshold)
        return JSONResponse(content={"status": "ok", "threshold": threshold})
    return JSONResponse(content={"status": "error"}, status_code=500)


@app.post("/clear_target")
async def clear_target():
    """Clear active search target."""
    if engine:
        engine.clear_target()
        return JSONResponse(content={"status": "ok"})
    return JSONResponse(content={"status": "error"}, status_code=500)


@app.post("/toggle_child_prior")
async def toggle_child_prior(request: Request):
    """Toggle child biometric prior in Re-ID matching."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    active = body.get("active", None)
    if engine:
        engine.toggle_child_prior(active)
        return JSONResponse(content={"status": "ok", "child_prior": engine.reid_matcher.child_prior})
    return JSONResponse(content={"status": "error"}, status_code=500)


# ══════════════════════════════════════════════════════════════
#  Dashboard HTML — Dark-Mode Tailwind CSS
# ══════════════════════════════════════════════════════════════
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sahra-Eye | Laâyoune Smart City Vision</title>
    <meta name="description" content="Edge AI Vision Engine for Laâyoune Smart City — Real-time fall detection and missing person search with CNDP Law 09-08 compliance">
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'system-ui', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace']
                    },
                    colors: {
                        brand: {
                            50:  '#F0FDF4',
                            100: '#DCFCE7',
                            500: '#10B981',
                            600: '#059669',
                            700: '#047857'
                        }
                    }
                }
            }
        }
    </script>
    <style>
        body { font-family: 'Inter', sans-serif; }

        /* ── Animations & Effects ──────────────────────── */
        .pulse-live { animation: pulseLive 2s ease-in-out infinite; }
        @keyframes pulseLive { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.35; transform: scale(0.92); } }

        .video-border {
            border: 2px solid #E2E8F0;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }
        .video-border.alert-flash {
            border-color: #EF4444 !important;
            box-shadow: 0 0 25px rgba(239, 68, 68, 0.45) !important;
            animation: flashPulse 0.5s ease infinite alternate;
        }
        @keyframes flashPulse { from { border-color: #EF4444; } to { border-color: #F87171; } }

        .alert-entry { animation: slideIn 0.3s ease-out; }
        @keyframes slideIn { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }

        /* ── Clean Scrollbars ──────────────────────────── */
        .custom-scroll::-webkit-scrollbar { width: 5px; height: 5px; }
        .custom-scroll::-webkit-scrollbar-track { background: #F1F5F9; border-radius: 4px; }
        .custom-scroll::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 4px; }
        .custom-scroll::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

        /* ── Modern Card Interactions ─────────────────── */
        .card-interactive { transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1); }
        .card-interactive:hover { transform: translateY(-1px); box-shadow: 0 4px 12px -2px rgba(0,0,0,0.06); }
    </style>
</head>

<body class="bg-slate-50 min-h-screen text-slate-800 antialiased selection:bg-emerald-100 selection:text-emerald-900">

<!-- ═══════════════════════════════════════════════════════════
     CLEAN MINIMALIST HEADER
     ═══════════════════════════════════════════════════════════ -->
<header class="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-slate-200/80 shadow-xs">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-2.5 flex items-center justify-between gap-4">
        <!-- Logo -->
        <div class="flex items-center gap-3">
            <div class="w-9 h-9 rounded-xl bg-slate-900 text-white flex items-center justify-center text-base font-bold shadow-xs select-none">
                🏜️
            </div>
            <div>
                <h1 class="text-base font-extrabold tracking-tight text-slate-900 leading-none flex items-center gap-1.5">
                    <span>Sahra-Eye</span>
                    <span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 font-semibold border border-slate-200">v2.0</span>
                </h1>
                <p class="text-[10px] text-slate-400 font-mono tracking-wider mt-0.5">LAÂYOUNE SMART CITY VISION</p>
            </div>
        </div>

        <!-- Mode Switcher Pill Toggle -->
        <div class="flex items-center p-1 rounded-xl bg-slate-100 border border-slate-200/80 text-xs shadow-xs">
            <button id="navMode1Btn" onclick="setMode(1)" class="px-3.5 py-1.5 rounded-lg font-bold transition flex items-center gap-1.5 bg-white text-slate-900 shadow-xs cursor-pointer border border-slate-200/60">
                <span>❤️</span> Mode 1: Najda-Vision
            </button>
            <button id="navMode2Btn" onclick="setMode(2)" class="px-3.5 py-1.5 rounded-lg font-bold text-slate-500 hover:text-slate-900 transition flex items-center gap-1.5 cursor-pointer">
                <span>🔍</span> Mode 2: Tifawt-Guard
            </button>
        </div>

        <!-- Right System Badges -->
        <div class="flex items-center gap-2.5">
            <div class="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-50 border border-emerald-200/80 text-emerald-800 text-[11px] font-mono">
                <span class="w-2 h-2 rounded-full bg-emerald-500 pulse-live"></span>
                <span class="font-bold">CNDP 09-08</span>
                <span class="text-[10px] text-emerald-600">Privacy Active</span>
            </div>
            <div id="modeHeaderBadge" class="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-100 border border-slate-200 text-[11px] font-mono">
                <span id="modeHeaderNumber" class="font-bold text-rose-600">MODE 1</span>
                <span id="modeHeaderTitle" class="text-slate-500">Najda-Vision</span>
            </div>
            <div id="clock" class="text-xs font-mono font-semibold text-slate-500 hidden md:block"></div>
        </div>
    </div>
</header>

<!-- ═══════════════════════════════════════════════════════════
     MAIN WORKSPACE
     ═══════════════════════════════════════════════════════════ -->
<main class="max-w-7xl mx-auto px-4 sm:px-6 py-5 space-y-4">

    <!-- ── PROMINENT USE CASE HIGHLIGHT BANNERS ─────────────── -->
    <!-- Mode 1 Use Case -->
    <div id="useCaseBannerM1" class="bg-white rounded-2xl p-4 shadow-xs border border-slate-200/90 flex flex-wrap items-center justify-between gap-3">
        <div class="flex items-center gap-3.5">
            <div class="w-10 h-10 rounded-xl bg-rose-50 border border-rose-100 flex items-center justify-center text-xl shrink-0">
                ❤️
            </div>
            <div>
                <div class="flex items-center gap-2">
                    <span class="text-[10px] font-mono font-bold uppercase tracking-wider text-rose-600 px-2 py-0.5 rounded-full bg-rose-50 border border-rose-200/60">ACTIVE USE CASE</span>
                    <h2 class="text-sm sm:text-base font-extrabold text-slate-900">Heart attacks &amp; sudden medical collapses detected early</h2>
                </div>
                <p class="text-xs text-slate-500 mt-0.5">Real-time edge pose AI triage detects collapses in public spaces within 1.5s, alerting Protection Civile 150 &amp; activating green-wave ambulance corridors.</p>
            </div>
        </div>
        <div class="flex items-center gap-2 text-xs font-mono font-semibold text-rose-700 bg-rose-50 px-3 py-1.5 rounded-xl border border-rose-200/60">
            <span class="w-2 h-2 rounded-full bg-rose-500 pulse-live"></span>
            <span>AUTONOMOUS TRIAGE ACTIVE</span>
        </div>
    </div>

    <!-- Mode 2 Use Case -->
    <div id="useCaseBannerM2" class="hidden bg-white rounded-2xl p-4 shadow-xs border border-slate-200/90 flex flex-wrap items-center justify-between gap-3">
        <div class="flex items-center gap-3.5">
            <div class="w-10 h-10 rounded-xl bg-amber-50 border border-amber-100 flex items-center justify-center text-xl shrink-0">
                🔍
            </div>
            <div>
                <div class="flex items-center gap-2">
                    <span class="text-[10px] font-mono font-bold uppercase tracking-wider text-amber-700 px-2 py-0.5 rounded-full bg-amber-50 border border-amber-200/60">ACTIVE USE CASE</span>
                    <h2 class="text-sm sm:text-base font-extrabold text-slate-900">Missing children located in real-time across CCTV network</h2>
                </div>
                <p class="text-xs text-slate-500 mt-0.5">Groq AI parses physical appearance into 3-part anatomical color models (Cap, Torso, Legs) with biometric child scale priors to locate targets across city cameras.</p>
            </div>
        </div>
        <div class="flex items-center gap-2 text-xs font-mono font-semibold text-amber-800 bg-amber-50 px-3 py-1.5 rounded-xl border border-amber-200/60">
            <span class="w-2 h-2 rounded-full bg-amber-500 pulse-live"></span>
            <span>RE-ID SEARCH SCANNER ACTIVE</span>
        </div>
    </div>

    <!-- ── 2-COLUMN GRID (LEFT: Video & Main Features, RIGHT: Controls & Sidebar) ── -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-5">

        <!-- ═══════════════════════════════════════════════════
             LEFT COLUMN: Hero Camera + 3 CCTV Feeds + Stats + Map/Ticker (2/3)
             ═══════════════════════════════════════════════════ -->
        <div class="lg:col-span-2 space-y-4">

            <!-- ── MULTI-CAMERA SURVEILLANCE ARRAY ───────────── -->
            <div class="bg-white rounded-2xl p-3 shadow-xs border border-slate-200 space-y-3">

                <!-- Toolbar: Active Source & Quick Switch -->
                <div class="flex flex-wrap items-center justify-between gap-2.5 pb-2.5 border-b border-slate-100 text-xs">
                    <div class="flex items-center gap-2">
                        <span class="text-slate-400 font-semibold uppercase tracking-wider text-[11px]">Primary Feed:</span>
                        <span id="activeSourceBadge" class="font-mono font-bold px-2 py-0.5 rounded-md bg-emerald-50 text-emerald-800 border border-emerald-200 flex items-center gap-1.5 text-[11px]">
                            <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 pulse-live"></span>
                            <span id="activeSourceName">Elder heart atack.mp4</span>
                        </span>
                    </div>

                    <div class="flex items-center gap-1.5">
                        <select id="videoPresetSelect" onchange="switchPresetVideo(this.value)" class="px-2.5 py-1 text-xs bg-slate-50 hover:bg-slate-100 text-slate-700 border border-slate-200 rounded-lg outline-none cursor-pointer font-medium">
                            <option value="">-- Load CCTV Video --</option>
                        </select>
                        <button id="btnWebcam" onclick="switchToWebcam()" title="Switch to live local webcam" class="px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-50 hover:bg-slate-100 text-slate-700 border border-slate-200 transition flex items-center gap-1 cursor-pointer">
                            <span>📷</span> Webcam
                        </button>
                        <label class="cursor-pointer px-3 py-1 rounded-lg text-xs font-semibold bg-slate-900 hover:bg-slate-800 text-white transition flex items-center gap-1 shadow-xs">
                            <span>📁</span> Upload
                            <input type="file" id="videoUploadInput" accept="video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm" class="hidden" onchange="uploadCctvVideo(this)">
                        </label>
                        <span id="uploadStatus" class="text-xs text-emerald-600 font-mono hidden">Uploading…</span>
                    </div>
                </div>

                <!-- 1. HERO ACTIVE VIDEO (SIGNIFICANTLY BIGGER THAN AUXILIARY CAMS) -->
                <div id="videoBorder" class="video-border rounded-xl overflow-hidden bg-slate-950 relative shadow-sm">
                    <!-- Top HUD Overlay -->
                    <div class="absolute top-0 left-0 right-0 z-10 px-3.5 py-2.5 bg-gradient-to-b from-black/85 via-black/40 to-transparent flex items-center justify-between text-xs">
                        <div class="flex items-center gap-2">
                            <span id="heroPulseDot" class="w-2.5 h-2.5 rounded-full bg-rose-500 pulse-live"></span>
                            <span id="heroCamLabel" class="font-mono font-bold text-white text-xs sm:text-sm">CAM 04 · Elder Heart Attack Monitor</span>
                            <span id="heroCamSub" class="hidden sm:inline text-[10px] font-mono px-2 py-0.5 rounded bg-white/20 text-white/90">Active Fall &amp; Collapse Detection</span>
                        </div>
                        <div class="flex items-center gap-2">
                            <span id="heroSourceBadge" class="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-500/25 text-rose-300 border border-rose-500/40 font-semibold">Elder heart atack.mp4</span>
                            <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-black/60 text-emerald-400 border border-emerald-500/30 font-bold">LIVE AI • 30 FPS</span>
                        </div>
                    </div>

                    <!-- The Main Video Stream -->
                    <div class="relative w-full aspect-video flex items-center justify-center bg-black">
                        <img id="videoFeed" src="/video_feed" class="w-full h-full object-contain bg-black" alt="Sahra-Eye Active Vision Feed">
                    </div>

                    <!-- Bottom HUD Overlay -->
                    <div class="absolute bottom-0 left-0 right-0 z-10 px-3.5 py-2 bg-gradient-to-t from-black/85 via-black/40 to-transparent flex items-center justify-between text-[11px] font-mono text-white/80">
                        <span id="heroLocationLabel">📍 Place du Méchouar Sector · 640×480 @ OpenVINO</span>
                        <span id="heroPrivacyBadge" class="text-emerald-400 font-semibold">● CNDP-0908 PRIVACY MASK ENFORCED</span>
                    </div>
                </div>

                <!-- 2. THREE AUXILIARY CCTV FEEDS (SMALLER, 1/3 WIDTH EACH, ON REPEAT, NO DETECTION) -->
                <div>
                    <div class="flex items-center justify-between mb-1.5 px-0.5">
                        <span class="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Auxiliary CCTV Surveillance Sectors (No Detection · Continuous Feeds):</span>
                        <span class="text-[10px] font-mono text-slate-400">3 CHANNELS ONLINE</span>
                    </div>

                    <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
                        <!-- CAM 01 (Looping cctv1.mp4) -->
                        <div class="bg-slate-50 rounded-xl overflow-hidden border border-slate-200/90 shadow-2xs flex flex-col">
                            <div class="px-2.5 py-1.5 bg-white border-b border-slate-200/80 flex items-center justify-between text-[11px]">
                                <div class="flex items-center gap-1.5 truncate">
                                    <span class="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
                                    <span class="font-bold text-slate-800 truncate">CAM 01: Place Méchouar</span>
                                </div>
                                <span class="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-100 text-slate-500 shrink-0 font-medium">No Detection</span>
                            </div>
                            <div class="relative bg-black aspect-video flex items-center justify-center overflow-hidden">
                                <video id="cam1Video" src="/cctv1_feed" autoplay loop muted playsinline class="w-full h-full object-cover"></video>
                                <div class="absolute bottom-1.5 left-2 text-[9px] font-mono text-emerald-300 bg-black/75 px-1.5 py-0.5 rounded">
                                    ● cctv1.mp4 (Loop)
                                </div>
                            </div>
                        </div>

                        <!-- CAM 02 (Looping cctv2.mp4) -->
                        <div class="bg-slate-50 rounded-xl overflow-hidden border border-slate-200/90 shadow-2xs flex flex-col">
                            <div class="px-2.5 py-1.5 bg-white border-b border-slate-200/80 flex items-center justify-between text-[11px]">
                                <div class="flex items-center gap-1.5 truncate">
                                    <span class="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
                                    <span class="font-bold text-slate-800 truncate">CAM 02: Blvd Mohammed V</span>
                                </div>
                                <span class="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-100 text-slate-500 shrink-0 font-medium">No Detection</span>
                            </div>
                            <div class="relative bg-black aspect-video flex items-center justify-center overflow-hidden">
                                <video id="cam2Video" src="/cctv2_feed" autoplay loop muted playsinline class="w-full h-full object-cover"></video>
                                <div class="absolute bottom-1.5 left-2 text-[9px] font-mono text-emerald-300 bg-black/75 px-1.5 py-0.5 rounded">
                                    ● cctv2.mp4 (Loop)
                                </div>
                            </div>
                        </div>

                        <!-- CAM 03 (Photo cctv3.jpg) -->
                        <div class="bg-slate-50 rounded-xl overflow-hidden border border-slate-200/90 shadow-2xs flex flex-col">
                            <div class="px-2.5 py-1.5 bg-white border-b border-slate-200/80 flex items-center justify-between text-[11px]">
                                <div class="flex items-center gap-1.5 truncate">
                                    <span class="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
                                    <span class="font-bold text-slate-800 truncate">CAM 03: Al-Massira Souk</span>
                                </div>
                                <span class="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-100 text-slate-500 shrink-0 font-medium">No Detection</span>
                            </div>
                            <div class="relative bg-black aspect-video flex items-center justify-center overflow-hidden">
                                <img id="cam3Img" src="/cctv3_feed" class="w-full h-full object-cover" alt="CAM 03 CCTV Photo">
                                <div class="absolute bottom-1.5 left-2 text-[9px] font-mono text-cyan-300 bg-black/75 px-1.5 py-0.5 rounded">
                                    ● cctv3.jpg (Photo)
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

            </div>

            <!-- ── TELEMETRY STATS ROW ───────────────────────── -->
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div class="bg-white rounded-xl p-3.5 border border-slate-200/80 shadow-xs">
                    <div class="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Framerate</div>
                    <div class="flex items-baseline gap-1 mt-1">
                        <span id="statFps" class="text-2xl font-extrabold font-mono text-emerald-600">0</span>
                        <span class="text-xs text-slate-400">fps</span>
                    </div>
                </div>
                <div class="bg-white rounded-xl p-3.5 border border-slate-200/80 shadow-xs">
                    <div class="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Pedestrians</div>
                    <div class="flex items-baseline gap-1 mt-1">
                        <span id="statPersons" class="text-2xl font-extrabold font-mono text-slate-900">0</span>
                        <span class="text-xs text-slate-400">tracked</span>
                    </div>
                </div>
                <div class="bg-white rounded-xl p-3.5 border border-slate-200/80 shadow-xs">
                    <div class="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Alerts &amp; Incidents</div>
                    <div class="flex items-baseline gap-1 mt-1">
                        <span id="statAlerts" class="text-2xl font-extrabold font-mono text-rose-600">0</span>
                        <span class="text-xs text-slate-400">total</span>
                    </div>
                </div>
                <div class="bg-white rounded-xl p-3.5 border border-slate-200/80 shadow-xs">
                    <div class="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Green Wave Corridor</div>
                    <div class="mt-1">
                        <span id="statGreenWave" class="text-lg font-bold font-mono text-slate-400">INACTIVE</span>
                    </div>
                </div>
            </div>

            <!-- ── MODE 1 FEATURE: LAÂYOUNE SMART CITY CORRIDOR MAP ── -->
            <div id="mode1MainView" class="space-y-4">
                <div class="bg-white rounded-2xl overflow-hidden border border-slate-200 shadow-xs flex flex-col">
                    <!-- Map Toolbar -->
                    <div class="p-3.5 sm:p-4 border-b border-slate-200/80 flex flex-wrap items-center justify-between gap-3 bg-slate-50/70">
                        <div>
                            <div class="flex items-center gap-2">
                                <h2 class="text-xs sm:text-sm font-bold text-slate-900 tracking-tight">
                                    🗺️ Laâyoune Emergency Corridor Map — Hôpital Moulay Hassan to Place du Méchouar
                                </h2>
                                <span id="mapStatusBadge" class="px-2 py-0.5 text-[10px] font-bold uppercase rounded bg-slate-200 text-slate-700 font-mono">
                                    ORGANIC CITY CIRCULATION
                                </span>
                            </div>
                            <p id="mapSubtext" class="text-[11px] text-slate-500 mt-0.5">
                                Realistic multi-turn streets &amp; roundabouts with circulating traffic. Drag to pan or use zoom controls.
                            </p>
                        </div>

                        <div class="flex items-center gap-2">
                            <button type="button" onclick="randomizeCars()" title="Randomize Car Traffic Positions"
                                    class="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-semibold bg-white hover:bg-slate-100 text-slate-700 rounded-lg border border-slate-200 shadow-xs transition cursor-pointer">
                                <span>🔀</span> Randomize Cars
                            </button>
                            <button id="btnManualGreenWave" type="button" onclick="toggleManualGreenWave()"
                                    class="flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold rounded-lg transition cursor-pointer bg-emerald-600 text-white hover:bg-emerald-700 shadow-xs">
                                <span>🚑</span> Test Faint &amp; Dispatch Ambulance
                            </button>
                        </div>
                    </div>

                    <!-- Ambulance Navigation & GPS Bar -->
                    <div id="ambulanceStatusBar" class="px-4 py-2 border-b border-slate-200/80 flex flex-wrap items-center justify-between gap-3 text-xs bg-white text-slate-700">
                        <div class="flex items-center gap-2">
                            <span id="ambulancePulseDot" class="w-2.5 h-2.5 rounded-full bg-slate-400"></span>
                            <span class="text-sm">🚨</span>
                            <span class="font-bold text-slate-900">Ambulance GPS (AMB-LAA-01):</span>
                            <span id="ambulanceTurnStatus" class="text-slate-500 font-medium">Standby at Hôpital Moulay Hassan</span>
                        </div>
                        <div id="ambulanceTelemetryStats" class="flex items-center gap-3 font-mono tabular-nums text-[11px] text-slate-500">
                            <span>Active Traffic: <strong id="trafficCarCount" class="text-slate-800 font-mono">24 vehicles</strong> circulating</span>
                            <span>Destination: <strong class="text-emerald-700">Place du Méchouar</strong></span>
                        </div>
                    </div>

                    <!-- Map SVG Canvas -->
                    <div id="mapCanvasContainer" class="relative h-[420px] sm:h-[460px] w-full bg-[#111927] overflow-hidden select-none">
                        <!-- Top-Right Interactive Map Zoom Controls -->
                        <div class="absolute top-3 right-3 z-20 flex flex-col gap-1.5 bg-slate-900/90 backdrop-blur-md border border-slate-700 p-1.5 rounded-xl shadow-lg">
                            <button type="button" onclick="zoomInMap()" title="Zoom In" class="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-cyan-300 font-bold text-base cursor-pointer">+</button>
                            <div id="mapZoomDisplay" class="text-[10px] font-mono tabular-nums text-center text-white/60 py-0.5">1.2x</div>
                            <button type="button" onclick="zoomOutMap()" title="Zoom Out" class="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-cyan-300 font-bold text-base cursor-pointer">-</button>
                            <button type="button" onclick="resetMapZoom()" title="Reset Zoom" class="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white/80 text-xs cursor-pointer">⛶</button>
                        </div>

                        <!-- Top-Left Quick Focus Buttons -->
                        <div class="absolute top-3 left-3 z-20 flex flex-wrap items-center gap-1.5">
                            <button type="button" onclick="focusPlaceMechouar()" class="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-900/90 text-cyan-300 border border-cyan-500/40 backdrop-blur-md shadow cursor-pointer">
                                <span>🎯</span> Focus Place du Méchouar
                            </button>
                            <button id="btnFollowAmbulance" type="button" onclick="toggleFollowAmbulance()" class="hidden items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-900/90 text-emerald-400 border border-emerald-500/40 backdrop-blur-md shadow cursor-pointer">
                                <span>🚑</span> Follow Ambulance
                            </button>
                        </div>

                        <!-- The SVG Map Engine -->
                        <svg id="laayouneMapSvg" viewBox="0 0 1080 620" class="w-full h-full object-cover cursor-grab" aria-label="Laâyoune City Corridor Map"></svg>

                        <!-- Bottom Legend -->
                        <div class="absolute bottom-2.5 left-3 right-3 bg-slate-900/90 backdrop-blur-md border border-slate-700 rounded-lg px-3 py-1.5 text-xs flex flex-wrap items-center justify-between gap-2 pointer-events-none text-white/80">
                            <div id="mapLegendContent" class="flex flex-wrap items-center gap-4 text-[11px]">
                                <span><strong class="text-cyan-400">🧭</strong> City Traffic (24 Circulating Vehicles)</span>
                                <span><strong class="text-emerald-400">📍</strong> Place du Méchouar Central Plaza</span>
                            </div>
                            <span class="text-[10px] font-mono text-white/40 hidden sm:inline">Scroll or +/- to Zoom · Drag to Pan</span>
                        </div>
                    </div>

                    <!-- Sector Localization Bar -->
                    <div class="p-2.5 bg-slate-50 border-t border-slate-200 flex flex-wrap items-center justify-between gap-2">
                        <span class="text-xs text-slate-500 font-medium">Laâyoune Urban Camera Sectors:</span>
                        <div id="cameraNodesContainer" class="flex flex-wrap items-center gap-1.5"></div>
                    </div>
                </div>
            </div>

            <!-- ── MODE 2 FEATURE: LIVE MISSING PERSON MATCH TICKER & EVIDENCE GALLERY ── -->
            <div id="mode2MainView" class="space-y-4 hidden">
                <div class="bg-white rounded-2xl p-4 border border-slate-200 shadow-xs flex flex-col gap-3">
                    <div class="flex items-center justify-between pb-2 border-b border-slate-100">
                        <div class="flex items-center gap-2">
                            <span class="text-base">🎯</span>
                            <h3 class="text-xs font-bold text-slate-900 uppercase tracking-wider">
                                Live Missing Person Match Ticker &amp; Re-ID Evidence Gallery
                            </h3>
                            <span id="reidMatchCounterBadge" class="text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 font-bold border border-amber-200">
                                0 MATCHES
                            </span>
                        </div>
                        <button onclick="clearAllAlerts()" class="text-[10px] px-2.5 py-1 rounded-md bg-slate-100 hover:bg-slate-200 text-slate-600 border border-slate-200 transition cursor-pointer font-mono flex items-center gap-1 font-semibold">
                            <span>✖</span> Clear Gallery
                        </button>
                    </div>

                    <!-- Horizontal Scroll Stream of Candidate Matches -->
                    <div id="reidTickerContainer" class="flex items-center gap-3 overflow-x-auto custom-scroll py-2 min-h-[140px]">
                        <div id="reidTickerEmpty" class="w-full flex flex-col items-center justify-center py-6 text-center text-slate-400 gap-1.5">
                            <div class="text-2xl opacity-40">🔍</div>
                            <div class="text-xs font-mono font-medium">No target candidate detected yet — Scanning 4 CCTV sectors in real-time</div>
                            <div class="text-[11px] text-slate-400">Enter a description or load a preset in the right panel to begin search</div>
                        </div>
                    </div>
                </div>
            </div>

        </div>

        <!-- ═══════════════════════════════════════════════════
             RIGHT COLUMN: Mode Switcher & Operator Controls (1/3)
             ═══════════════════════════════════════════════════ -->
        <div class="space-y-4">

            <!-- Operational Mode Selection Card -->
            <div class="bg-white rounded-2xl p-4 border border-slate-200 shadow-xs">
                <h2 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2.5">Operational Mode</h2>
                <div class="space-y-2">
                    <button id="mode1Btn" onclick="setMode(1)"
                            class="card-interactive w-full text-left px-3.5 py-3 rounded-xl bg-rose-50 border-2 border-rose-500 cursor-pointer shadow-xs">
                        <div class="flex items-center justify-between">
                            <div>
                                <div class="text-xs font-bold text-rose-900 flex items-center gap-1.5">
                                    <span>❤️</span> 1 — Najda-Vision
                                </div>
                                <div class="text-[11px] text-rose-700/80 mt-0.5">Fall &amp; Heart Attack Collapse Triage</div>
                            </div>
                            <div id="mode1Indicator" class="w-2.5 h-2.5 rounded-full bg-rose-500 pulse-live"></div>
                        </div>
                    </button>

                    <button id="mode2Btn" onclick="setMode(2)"
                            class="card-interactive w-full text-left px-3.5 py-3 rounded-xl bg-slate-50 border border-slate-200 hover:border-amber-400/80 transition cursor-pointer">
                        <div class="flex items-center justify-between">
                            <div>
                                <div class="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                                    <span>🔍</span> 2 — Tifawt-Guard
                                </div>
                                <div class="text-[11px] text-slate-500 mt-0.5">Missing Child &amp; Person Re-ID Search</div>
                            </div>
                            <div id="mode2Indicator" class="w-2.5 h-2.5 rounded-full bg-slate-300"></div>
                        </div>
                    </button>
                </div>
            </div>

            <!-- ── MODE 1 SIDEBAR PANELS ──────────────────────── -->
            <div id="mode1Sidebar" class="space-y-4">
                <!-- Incident Alert Log -->
                <div class="bg-white rounded-2xl p-4 border border-slate-200 shadow-xs flex flex-col" style="max-height: 380px;">
                    <div class="flex items-center justify-between mb-2.5 pb-2 border-b border-slate-100">
                        <div class="flex items-center gap-1.5">
                            <h2 class="text-xs font-bold text-slate-500 uppercase tracking-wider">Incident Feed</h2>
                            <span id="alertBadge" class="hidden text-[10px] px-2 py-0.5 rounded-full bg-rose-100 text-rose-800 font-mono font-bold">0</span>
                        </div>
                        <button onclick="clearAllAlerts()" class="text-[10px] px-2 py-0.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-600 transition cursor-pointer font-mono font-semibold">
                            Dismiss
                        </button>
                    </div>

                    <div id="alertLog" class="flex-1 overflow-y-auto custom-scroll space-y-2 min-h-[90px]">
                        <div class="flex flex-col items-center justify-center text-center py-8 gap-1 text-slate-400">
                            <div class="text-2xl opacity-30">🛡️</div>
                            <div class="text-xs font-mono">No active incidents detected</div>
                            <div class="text-[10px]">Continuous AI monitoring on Place du Méchouar</div>
                        </div>
                    </div>
                </div>

                <!-- System Diagnostics -->
                <div class="bg-white rounded-2xl p-4 border border-slate-200 shadow-xs">
                    <h2 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2.5">Vision Engine Specs</h2>
                    <div class="space-y-2 text-xs">
                        <div class="flex items-center justify-between">
                            <span class="text-slate-500">Pose Model</span>
                            <span class="font-mono font-semibold text-slate-800">YOLOv8n-pose</span>
                        </div>
                        <div class="flex items-center justify-between">
                            <span class="text-slate-500">Inference Core</span>
                            <span class="font-mono font-semibold text-slate-800">CPU / OpenVINO</span>
                        </div>
                        <div class="flex items-center justify-between">
                            <span class="text-slate-500">Privacy Compliance</span>
                            <span class="font-mono font-bold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">Law 09-08 Active</span>
                        </div>
                    </div>
                </div>

                <!-- Emergency Dispatch Hotlines -->
                <div class="bg-white rounded-2xl p-4 border border-slate-200 shadow-xs">
                    <h2 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2.5">Emergency Dispatch</h2>
                    <div class="space-y-1.5 text-xs">
                        <div class="flex items-center justify-between px-3 py-2 rounded-xl bg-rose-50 border border-rose-100 text-rose-900 font-semibold">
                            <span>🚑 Protection Civile</span>
                            <span class="font-mono font-extrabold text-rose-700 text-sm">150</span>
                        </div>
                        <div class="flex items-center justify-between px-3 py-2 rounded-xl bg-slate-50 border border-slate-200 text-slate-800 font-semibold">
                            <span>🚔 Police Laâyoune</span>
                            <span class="font-mono font-extrabold text-slate-900 text-sm">19</span>
                        </div>
                        <div class="flex items-center justify-between px-3 py-2 rounded-xl bg-slate-50 border border-slate-200 text-slate-800 font-semibold">
                            <span>🏥 Hôpital Moulay Hassan</span>
                            <span class="font-mono font-semibold text-slate-700">0528-891-200</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ── MODE 2 SIDEBAR PANELS ──────────────────────── -->
            <div id="mode2Sidebar" class="space-y-4 hidden">

                <!-- AI Natural Language Search (Groq LLM) -->
                <div class="bg-white rounded-2xl p-4 border-2 border-amber-400 shadow-xs">
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-1.5">
                            <span class="text-sm">⚡</span>
                            <h2 class="text-xs font-bold text-amber-900 uppercase tracking-wider">AI Natural Language Search</h2>
                        </div>
                        <span class="text-[9px] font-mono px-2 py-0.5 rounded bg-amber-100 text-amber-800 border border-amber-300 font-bold">GROQ LLM</span>
                    </div>

                    <div class="relative mb-2.5">
                        <textarea id="nlpSearchInput" rows="2"
                                  class="w-full px-3 py-2 text-xs bg-slate-50 text-slate-800 border border-slate-200 rounded-xl focus:border-amber-500 focus:bg-white focus:outline-none custom-scroll placeholder-slate-400"
                                  placeholder="e.g. Young child wearing a red hat, yellow shirt, and jeans"></textarea>
                    </div>

                    <!-- Instant Zero-Latency Preset Pills -->
                    <div class="mb-3">
                        <div class="text-[10px] text-slate-400 mb-1.5 font-medium">Quick Presets (0ms Groq Bypass):</div>
                        <div class="flex flex-wrap gap-1.5">
                            <button type="button" onclick="loadNlpPreset('preset_1')" class="text-[10px] px-2 py-1 rounded-lg bg-rose-50 text-rose-800 border border-rose-200 hover:bg-rose-100 transition cursor-pointer flex items-center gap-1 font-semibold">
                                <span>🔴</span> Kid: Red Cap + Yellow Shirt + Jeans
                            </button>
                            <button type="button" onclick="loadNlpPreset('preset_2')" class="text-[10px] px-2 py-1 rounded-lg bg-blue-50 text-blue-800 border border-blue-200 hover:bg-blue-100 transition cursor-pointer flex items-center gap-1 font-semibold">
                                <span>🔵</span> Boy: Blue Hoodie
                            </button>
                            <button type="button" onclick="loadNlpPreset('preset_4')" class="text-[10px] px-2 py-1 rounded-lg bg-amber-50 text-amber-900 border border-amber-200 hover:bg-amber-100 transition cursor-pointer flex items-center gap-1 font-semibold">
                                <span>🟡</span> Sports Jersey
                            </button>
                        </div>
                    </div>

                    <!-- Search Button -->
                    <button id="btnGroqSearch" onclick="searchByGroqDescription()"
                            class="w-full py-2 px-3 rounded-xl bg-amber-500 hover:bg-amber-600 text-slate-950 font-bold text-xs shadow-xs transition cursor-pointer flex items-center justify-center gap-2">
                        <span id="groqSpinner" class="hidden animate-spin">⏳</span>
                        <span id="groqBtnIcon">🔍</span>
                        <span id="groqBtnText">Extract &amp; Search with Groq AI</span>
                    </button>
                    <div id="nlpStatusMsg" class="hidden text-[10px] font-mono mt-1.5 text-center text-amber-800"></div>
                </div>

                <!-- Active Reference Target Card -->
                <div class="bg-white rounded-2xl p-4 border border-slate-200 shadow-xs">
                    <div class="flex items-center justify-between mb-2.5 pb-2 border-b border-slate-100">
                        <div class="flex items-center gap-1.5">
                            <span class="text-sm">🎯</span>
                            <h2 class="text-xs font-bold text-slate-700 uppercase tracking-wider">Active Reference Target</h2>
                        </div>
                        <button onclick="clearSearchTarget()" title="Clear active target" class="text-[10px] text-slate-400 hover:text-rose-600 font-mono transition cursor-pointer">
                            ✕ Reset
                        </button>
                    </div>

                    <!-- Photo Upload Dropzone -->
                    <label class="border border-dashed border-slate-300 hover:border-amber-500 rounded-xl p-2.5 flex items-center justify-center gap-2.5 cursor-pointer transition bg-slate-50 hover:bg-amber-50/50 text-center mb-3">
                        <span class="text-base text-amber-600">📷</span>
                        <div class="text-left">
                            <div class="text-xs font-bold text-slate-800">Upload Reference Photo</div>
                            <div class="text-[10px] text-slate-400">YOLO auto-crops &amp; extracts 3-part colors</div>
                        </div>
                        <input type="file" id="targetPhotoInput" accept="image/*" class="hidden" onchange="uploadTargetPhotoFile(this)">
                    </label>

                    <!-- Target Profile Preview -->
                    <div class="bg-slate-50 rounded-xl p-3 border border-slate-200/80 flex gap-3 items-center">
                        <div class="relative w-16 h-20 rounded-lg overflow-hidden bg-slate-200 border border-slate-300 shrink-0 flex items-center justify-center">
                            <img id="targetRefThumb" src="" class="w-full h-full object-cover hidden" alt="Target Thumbnail">
                            <div id="targetThumbPlaceholder" class="text-[10px] text-slate-400 font-mono text-center">No Photo</div>
                        </div>

                        <div class="flex-1 min-w-0 space-y-1">
                            <div id="targetRefLabel" class="text-xs font-bold text-slate-900 truncate">
                                Standby — Search or Select
                            </div>

                            <!-- 3-Part Color Swatches -->
                            <div class="grid grid-cols-1 gap-0.5 text-[10px] font-mono">
                                <div class="flex items-center gap-1.5">
                                    <span class="text-slate-400 w-10">Head:</span>
                                    <span id="targetHeadSwatch" class="w-3.5 h-3.5 rounded border border-slate-300 inline-block bg-slate-300"></span>
                                    <span id="targetHeadHex" class="text-slate-600">—</span>
                                </div>
                                <div class="flex items-center gap-1.5">
                                    <span class="text-slate-400 w-10">Torso:</span>
                                    <span id="targetTorsoSwatch" class="w-3.5 h-3.5 rounded border border-slate-300 inline-block bg-slate-300"></span>
                                    <span id="targetTorsoHex" class="text-slate-600">—</span>
                                </div>
                                <div class="flex items-center gap-1.5">
                                    <span class="text-slate-400 w-10">Legs:</span>
                                    <span id="targetLegsSwatch" class="w-3.5 h-3.5 rounded border border-slate-300 inline-block bg-slate-300"></span>
                                    <span id="targetLegsHex" class="text-slate-600">—</span>
                                </div>
                            </div>

                            <div class="pt-1">
                                <button id="btnChildPrior" onclick="toggleChildPrior()" class="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-white text-slate-500 border border-slate-200 transition flex items-center gap-1 cursor-pointer shadow-2xs">
                                    <span>👶</span> Child Prior: OFF
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Re-ID Match Threshold Slider -->
                <div class="bg-white rounded-2xl p-4 border border-slate-200 shadow-xs space-y-2">
                    <div class="flex items-center justify-between">
                        <span class="text-xs font-bold text-slate-500 uppercase tracking-wider">Similarity Threshold</span>
                        <span id="reidThresholdVal" class="text-sm font-extrabold font-mono text-amber-600">65%</span>
                    </div>
                    <input type="range" id="thresholdSlider" min="45" max="90" value="65" oninput="updateReidThreshold(this.value)"
                           class="w-full accent-amber-500 cursor-pointer">
                    <div class="flex justify-between text-[10px] text-slate-400 font-mono">
                        <span>45% (Broad)</span>
                        <span>65% (Balanced)</span>
                        <span>90% (Strict)</span>
                    </div>
                </div>

            </div>

        </div>

    </div>

</main>

<!-- ═══════════════════════════════════════════════════════════
     CLEAN MINIMALIST FOOTER
     ═══════════════════════════════════════════════════════════ -->
<footer class="bg-white border-t border-slate-200 mt-8">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex flex-col sm:flex-row items-center justify-between gap-2 text-[11px] text-slate-400 font-mono">
        <span>🏜️ Sahra-Eye v2.0 — CNDP Law 09-08 Privacy Compliant Edge Vision</span>
        <span>École Supérieure de Technologie de Laâyoune (ESTL) • Zero Cloud Latency</span>
    </div>
</footer>

<!-- ═══════════════════════════════════════════════════════════
     JAVASCRIPT — Telemetry Polling & Interactions
     ═══════════════════════════════════════════════════════════ -->
<script>
    let prevAlertCount = 0;
    let alertFlashTimeout = null;

    // ── Telemetry Polling ───────────────────────────────
    async function fetchTelemetry() {
        try {
            const res = await fetch('/telemetry');
            if (!res.ok) return;
            const d = await res.json();

            // FPS
            const fpsEl = document.getElementById('statFps');
            if (fpsEl) {
                fpsEl.textContent = d.fps ?? '—';
                fpsEl.className = 'text-2xl font-extrabold font-mono ' + (
                    d.fps >= 25 ? 'text-emerald-600' : d.fps >= 15 ? 'text-amber-600' : 'text-rose-600'
                );
            }

            // Persons
            const pEl = document.getElementById('statPersons');
            if (pEl) pEl.textContent = d.persons_count ?? 0;

            // Alerts count
            const aEl = document.getElementById('statAlerts');
            if (aEl) aEl.textContent = d.alert_count ?? 0;

            // Source Name Badge
            if (d.source_name) {
                const nameEl = document.getElementById('activeSourceName');
                if (nameEl) {
                    nameEl.textContent = d.is_video_file ? d.source_name : d.source_name;
                }
            }

            // Green Wave
            const gwEl = document.getElementById('statGreenWave');
            if (gwEl) {
                if (d.green_wave_active) {
                    gwEl.textContent = Math.round(d.green_wave_remaining) + 's';
                    gwEl.className = 'text-lg font-bold font-mono text-emerald-600';
                    setMapGreenWave(true, d.green_wave_remaining);
                } else {
                    gwEl.textContent = 'INACTIVE';
                    gwEl.className = 'text-lg font-bold font-mono text-slate-400';
                    if (!isManualGreenWave) {
                        setMapGreenWave(false, 0);
                    }
                }
            }

            // Alert Log — only update when count changes
            if (d.alert_count !== prevAlertCount && d.alerts && d.alerts.length > 0) {
                prevAlertCount = d.alert_count;

                // Flash video border red briefly
                const vb = document.getElementById('videoBorder');
                if (vb) {
                    vb.classList.add('alert-flash');
                    clearTimeout(alertFlashTimeout);
                    alertFlashTimeout = setTimeout(() => vb.classList.remove('alert-flash'), 2500);
                }

                // Badge
                const badge = document.getElementById('alertBadge');
                if (badge) {
                    badge.textContent = d.alert_count;
                    badge.classList.remove('hidden');
                }

                // Render alerts (newest first)
                const log = document.getElementById('alertLog');
                if (log) {
                    log.innerHTML = d.alerts.slice().reverse().map(a => `
                        <div class="alert-entry p-3 rounded-xl border ${a.severity === 'critical' ? 'bg-rose-50/80 border-rose-200 text-rose-900' : 'bg-amber-50/80 border-amber-200 text-amber-900'} shadow-2xs">
                            <div class="flex items-center justify-between gap-2">
                                <span class="text-xs font-bold ${a.severity === 'critical' ? 'text-rose-700' : 'text-amber-700'}">
                                    ${a.message}
                                </span>
                                <span class="text-[10px] text-slate-400 font-mono shrink-0">${a.timestamp}</span>
                            </div>
                            ${a.details ? `<div class="text-[11px] text-slate-600 mt-1">${a.details}</div>` : ''}
                            ${a.action  ? `<div class="text-[11px] font-semibold text-emerald-700 mt-0.5">→ ${a.action}</div>` : ''}
                        </div>
                    `).join('');
                }
            }

            // ── Mode 2: Tifawt-Guard Telemetry & Ticker Update ──
            if (d.mode === 2 || currentMode === 2) {
                // Update Target Info in Sidebar
                if (d.target_info) {
                    const ti = d.target_info;
                    const thumb = document.getElementById('targetRefThumb');
                    const placeh = document.getElementById('targetThumbPlaceholder');
                    if (thumb && ti.thumbnail) {
                        thumb.src = ti.thumbnail;
                        thumb.classList.remove('hidden');
                        if (placeh) placeh.classList.add('hidden');
                    }
                    const lbl = document.getElementById('targetRefLabel');
                    if (lbl && ti.label) lbl.textContent = ti.label;

                    const hSwatch = document.getElementById('targetHeadSwatch');
                    const hHex = document.getElementById('targetHeadHex');
                    if (hSwatch && ti.head_hex) {
                        hSwatch.style.backgroundColor = ti.head_hex;
                        if (hHex) hHex.textContent = ti.head_hex;
                    }

                    const tSwatch = document.getElementById('targetTorsoSwatch');
                    const tHex = document.getElementById('targetTorsoHex');
                    if (tSwatch && ti.torso_hex) {
                        tSwatch.style.backgroundColor = ti.torso_hex;
                        if (tHex) tHex.textContent = ti.torso_hex;
                    }

                    const lSwatch = document.getElementById('targetLegsSwatch');
                    const lHex = document.getElementById('targetLegsHex');
                    if (lSwatch && ti.legs_hex) {
                        lSwatch.style.backgroundColor = ti.legs_hex;
                        if (lHex) lHex.textContent = ti.legs_hex;
                    }
                }

                // Render Re-ID Live Match Cards
                const reidMatches = (d.alerts || []).filter(a => a.type === 'missing_person_match');
                const matchCountBadge = document.getElementById('reidMatchCounterBadge');
                if (matchCountBadge) {
                    matchCountBadge.textContent = `${reidMatches.length} MATCHES`;
                    if (reidMatches.length > 0) {
                        matchCountBadge.className = "text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 font-bold border border-emerald-300 pulse-live";
                    } else {
                        matchCountBadge.className = "text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 font-bold border border-amber-200";
                    }
                }

                const tickerContainer = document.getElementById('reidTickerContainer');
                if (tickerContainer) {
                    if (reidMatches.length === 0) {
                        tickerContainer.innerHTML = `
                            <div id="reidTickerEmpty" class="w-full flex flex-col items-center justify-center py-6 text-center text-slate-400 gap-1.5">
                                <div class="text-2xl opacity-40">🔍</div>
                                <div class="text-xs font-mono font-medium">No target candidate detected yet — Scanning 4 CCTV sectors in real-time</div>
                                <div class="text-[11px] text-slate-400">Enter a description or load a preset in the right panel to begin search</div>
                            </div>
                        `;
                    } else {
                        tickerContainer.innerHTML = reidMatches.slice().reverse().map(m => `
                            <div class="alert-entry shrink-0 w-80 p-3.5 rounded-2xl bg-white border-2 border-emerald-500 shadow-md flex flex-col gap-2.5">
                                <div class="flex items-center justify-between">
                                    <span class="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-100 text-emerald-800 border border-emerald-300 flex items-center gap-1">
                                        <span class="w-1.5 h-1.5 rounded-full bg-emerald-600 animate-ping"></span>
                                        <span>🎯 ${m.similarity}% MATCH</span>
                                    </span>
                                    <span class="text-[10px] font-mono text-slate-400">${m.timestamp}</span>
                                </div>

                                <div class="flex items-center justify-between gap-2 bg-slate-50 p-2 rounded-xl border border-slate-200">
                                    <div class="flex flex-col items-center gap-1 flex-1">
                                        <span class="text-[9px] text-slate-500 uppercase font-mono font-semibold">Reference</span>
                                        <div class="w-14 h-18 rounded-lg overflow-hidden bg-slate-200 border border-slate-300 flex items-center justify-center">
                                            ${m.reference_thumb ? `<img src="${m.reference_thumb}" class="w-full h-full object-cover">` : '<span class="text-[8px] text-slate-400">Ref</span>'}
                                        </div>
                                    </div>
                                    <div class="text-xs text-emerald-600 font-bold font-mono">VS</div>
                                    <div class="flex flex-col items-center gap-1 flex-1">
                                        <span class="text-[9px] text-emerald-700 uppercase font-mono font-bold">Spotted Crop</span>
                                        <div class="w-14 h-18 rounded-lg overflow-hidden bg-slate-200 border-2 border-emerald-500 shadow-sm flex items-center justify-center">
                                            ${m.candidate_thumb ? `<img src="${m.candidate_thumb}" class="w-full h-full object-cover">` : '<span class="text-[8px] text-slate-400">Crop</span>'}
                                        </div>
                                    </div>
                                </div>

                                <div class="text-xs font-bold text-slate-900 truncate">${m.message}</div>
                                <div class="flex items-center justify-between text-[10px] text-slate-500 font-mono">
                                    <span>📍 ${m.camera_id} (${m.zone_name})</span>
                                    <span>H:${m.head_sim || 0}% | T:${m.torso_sim}% | L:${m.legs_sim}%</span>
                                </div>

                                <div class="flex items-center gap-2 pt-1 border-t border-slate-100">
                                    <button onclick="alert('🚨 Security Patrol dispatched to ' + '${m.zone_name}')" class="flex-1 py-1.5 px-2 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-[10px] transition cursor-pointer flex items-center justify-center gap-1 shadow-xs">
                                        <span>🚔</span> Dispatch Patrol
                                    </button>
                                    <button onclick="clearAllAlerts()" class="py-1.5 px-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-600 text-[10px] font-medium transition cursor-pointer">
                                        Dismiss
                                    </button>
                                </div>
                            </div>
                        `).join('');
                    }
                }
            }

        } catch (e) { /* silent retry */ }
    }

    // ── Video Source & CCTV Upload ───────────────────────
    async function uploadCctvVideo(input) {
        if (!input.files || input.files.length === 0) return;
        const file = input.files[0];
        const statusEl = document.getElementById('uploadStatus');
        if (statusEl) {
            statusEl.textContent = `Uploading ${file.name}…`;
            statusEl.classList.remove('hidden');
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/upload_video', {
                method: 'POST',
                body: formData,
            });
            const data = await res.json();
            if (data.status === 'ok') {
                if (statusEl) {
                    statusEl.textContent = `Playing: ${data.filename}`;
                    setTimeout(() => statusEl.classList.add('hidden'), 3500);
                }
                fetchTelemetry();
            } else {
                if (statusEl) {
                    statusEl.textContent = 'Upload failed!';
                    setTimeout(() => statusEl.classList.add('hidden'), 3500);
                }
            }
        } catch (e) {
            if (statusEl) {
                statusEl.textContent = 'Upload failed!';
                setTimeout(() => statusEl.classList.add('hidden'), 3500);
            }
        }
    }

    async function switchToWebcam() {
        const statusEl = document.getElementById('uploadStatus');
        if (statusEl) {
            statusEl.textContent = 'Switching to webcam…';
            statusEl.classList.remove('hidden');
        }
        try {
            const res = await fetch('/reset_camera', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'ok') {
                if (statusEl) {
                    statusEl.textContent = 'Webcam active';
                    setTimeout(() => statusEl.classList.add('hidden'), 2000);
                }
                fetchTelemetry();
            }
        } catch (e) {
            if (statusEl) {
                statusEl.textContent = 'Switch failed!';
                setTimeout(() => statusEl.classList.add('hidden'), 2000);
            }
        }
    }

    async function clearAllAlerts() {
        try {
            await fetch('/clear_alerts', { method: 'POST' });
            prevAlertCount = -1;
            fetchTelemetry();
        } catch (e) {
            console.error('Failed to dismiss alerts', e);
        }
    }

    // ── Mode Switching ──────────────────────────────────
    let currentMode = 1;

    async function setMode(mode) {
        currentMode = mode;
        try {
            await fetch('/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode })
            });
        } catch(e) {}
        updateModeUI(mode);
        fetchTelemetry();
    }

    function updateModeUI(mode) {
        const isM1 = mode === 1;

        // Navigation buttons in header
        const navM1 = document.getElementById('navMode1Btn');
        const navM2 = document.getElementById('navMode2Btn');
        if (navM1) navM1.className = isM1
            ? "px-3.5 py-1.5 rounded-lg font-bold transition flex items-center gap-1.5 bg-white text-slate-900 shadow-xs cursor-pointer border border-slate-200/60"
            : "px-3.5 py-1.5 rounded-lg font-bold text-slate-500 hover:text-slate-900 transition flex items-center gap-1.5 cursor-pointer";
        if (navM2) navM2.className = !isM1
            ? "px-3.5 py-1.5 rounded-lg font-bold transition flex items-center gap-1.5 bg-white text-slate-900 shadow-xs cursor-pointer border border-slate-200/60"
            : "px-3.5 py-1.5 rounded-lg font-bold text-slate-500 hover:text-slate-900 transition flex items-center gap-1.5 cursor-pointer";

        // Sidebar buttons
        const m1Btn = document.getElementById('mode1Btn');
        const m2Btn = document.getElementById('mode2Btn');
        const m1Ind = document.getElementById('mode1Indicator');
        const m2Ind = document.getElementById('mode2Indicator');
        if (m1Btn) m1Btn.className = isM1
            ? "card-interactive w-full text-left px-3.5 py-3 rounded-xl bg-rose-50 border-2 border-rose-500 cursor-pointer shadow-xs"
            : "card-interactive w-full text-left px-3.5 py-3 rounded-xl bg-slate-50 border border-slate-200 hover:border-slate-300 transition cursor-pointer";
        if (m2Btn) m2Btn.className = !isM1
            ? "card-interactive w-full text-left px-3.5 py-3 rounded-xl bg-amber-50 border-2 border-amber-500 cursor-pointer shadow-xs"
            : "card-interactive w-full text-left px-3.5 py-3 rounded-xl bg-slate-50 border border-slate-200 hover:border-amber-400/80 transition cursor-pointer";
        if (m1Ind) m1Ind.className = isM1 ? "w-2.5 h-2.5 rounded-full bg-rose-500 pulse-live" : "w-2.5 h-2.5 rounded-full bg-slate-300";
        if (m2Ind) m2Ind.className = !isM1 ? "w-2.5 h-2.5 rounded-full bg-amber-500 pulse-live" : "w-2.5 h-2.5 rounded-full bg-slate-300";

        // Header Mode badge
        const badgeNum = document.getElementById('modeHeaderNumber');
        const badgeTitle = document.getElementById('modeHeaderTitle');
        if (badgeNum) {
            badgeNum.textContent = isM1 ? "MODE 1" : "MODE 2";
            badgeNum.className = isM1 ? "font-bold text-rose-600" : "font-bold text-amber-700";
        }
        if (badgeTitle) badgeTitle.textContent = isM1 ? "Najda-Vision" : "Tifawt-Guard";

        // Use Case Banners
        const b1 = document.getElementById('useCaseBannerM1');
        const b2 = document.getElementById('useCaseBannerM2');
        if (b1) b1.classList.toggle('hidden', !isM1);
        if (b2) b2.classList.toggle('hidden', isM1);

        // Hero Camera Labels & Badges
        const heroPulse = document.getElementById('heroPulseDot');
        const heroCam = document.getElementById('heroCamLabel');
        const heroSub = document.getElementById('heroCamSub');
        const heroSrc = document.getElementById('heroSourceBadge');
        const heroLoc = document.getElementById('heroLocationLabel');

        if (isM1) {
            if (heroPulse) heroPulse.className = "w-2.5 h-2.5 rounded-full bg-rose-500 pulse-live";
            if (heroCam) heroCam.textContent = "CAM 04 · Elder Heart Attack Monitor";
            if (heroSub) heroSub.textContent = "Active Fall & Collapse Detection";
            if (heroSrc) {
                heroSrc.textContent = "Elder heart atack.mp4";
                heroSrc.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-rose-500/25 text-rose-300 border border-rose-500/40 font-semibold";
            }
            if (heroLoc) heroLoc.textContent = "📍 Place du Méchouar Sector · 640×480 @ OpenVINO";
        } else {
            if (heroPulse) heroPulse.className = "w-2.5 h-2.5 rounded-full bg-amber-500 pulse-live";
            if (heroCam) heroCam.textContent = "CAM 04 · Sector Al-Quds";
            if (heroSub) heroSub.textContent = "Active Re-ID Missing Person Search";
            if (heroSrc) {
                heroSrc.textContent = "kid missing.mp4";
                heroSrc.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/25 text-amber-300 border border-amber-500/40 font-semibold";
            }
            if (heroLoc) heroLoc.textContent = "📍 Sector Al-Quds Surveillance Grid · 640×480 @ YOLO Re-ID";
        }

        // Toggle Main Views
        const m1Main = document.getElementById('mode1MainView');
        const m2Main = document.getElementById('mode2MainView');
        if (m1Main) m1Main.classList.toggle('hidden', !isM1);
        if (m2Main) m2Main.classList.toggle('hidden', isM1);

        // Toggle Sidebars
        const m1Side = document.getElementById('mode1Sidebar');
        const m2Side = document.getElementById('mode2Sidebar');
        if (m1Side) m1Side.classList.toggle('hidden', !isM1);
        if (m2Side) m2Side.classList.toggle('hidden', isM1);
    }

    // ── Tifawt-Guard Handlers ────────────────────────────
    async function searchByGroqDescription() {
        const input = document.getElementById('nlpSearchInput');
        const spinner = document.getElementById('groqSpinner');
        const statusEl = document.getElementById('nlpStatusMsg');
        const btn = document.getElementById('btnGroqSearch');
        if (!input) return;

        const description = input.value.trim();
        if (!description) {
            if (statusEl) {
                statusEl.textContent = 'Please enter a target description.';
                statusEl.className = 'text-[11px] text-amber-400 font-mono';
                statusEl.classList.remove('hidden');
            }
            return;
        }

        if (spinner) spinner.classList.remove('hidden');
        if (btn) btn.disabled = true;
        if (statusEl) {
            statusEl.textContent = 'Querying Groq LLM (LLaMA-3) & extracting visual schema…';
            statusEl.className = 'text-[11px] text-amber-300 font-mono animate-pulse';
            statusEl.classList.remove('hidden');
        }

        try {
            const res = await fetch('/search_by_description', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ description })
            });
            const data = await res.json();
            if (data.status === 'ok') {
                if (statusEl) {
                    statusEl.textContent = `Target locked: ${data.target_info.label} (${data.inference_ms}ms)`;
                    statusEl.className = 'text-[11px] text-emerald-400 font-mono font-bold';
                }
                fetchTelemetry();
            } else {
                if (statusEl) {
                    statusEl.textContent = `Extraction failed: ${data.detail || data.message || 'Unknown error'}`;
                    statusEl.className = 'text-[11px] text-red-400 font-mono';
                }
            }
        } catch (e) {
            if (statusEl) {
                statusEl.textContent = `Network error: ${e.message}`;
                statusEl.className = 'text-[11px] text-red-400 font-mono';
            }
        } finally {
            if (spinner) spinner.classList.add('hidden');
            if (btn) btn.disabled = false;
        }
    }

    async function loadNlpPreset(presetId) {
        const input = document.getElementById('nlpSearchInput');
        const statusEl = document.getElementById('nlpStatusMsg');

        const presetDescriptions = {
            'preset_1': "A young child wearing a red hat, yellow shirt, and blue jeans",
            'preset_2': "Missing boy wearing a royal blue hoodie and dark pants",
            'preset_3': "Missing child in bright red hooded jacket and beige trousers",
            'preset_4': "Student in athletic sports jersey and dark track shorts"
        };

        if (input && presetDescriptions[presetId]) {
            input.value = presetDescriptions[presetId];
        }

        if (statusEl) {
            statusEl.textContent = 'Instant preset loaded (0ms Groq bypass active)';
            statusEl.className = 'text-[11px] text-emerald-400 font-mono';
            statusEl.classList.remove('hidden');
            setTimeout(() => statusEl.classList.add('hidden'), 3500);
        }

        await setTargetPreset(presetId);
    }

    async function setTargetPreset(presetId) {
        try {
            const res = await fetch('/set_target_preset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ preset_id: presetId })
            });
            const data = await res.json();
            if (data.status === 'ok') {
                fetchTelemetry();
            }
        } catch(e) {
            console.error('Failed to set preset', e);
        }
    }

    async function uploadTargetPhotoFile(input) {
        if (!input.files || input.files.length === 0) return;
        const file = input.files[0];
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/upload_target_photo', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.status === 'ok') {
                fetchTelemetry();
            } else {
                alert(data.message || 'Failed to process target photo');
            }
        } catch(e) {
            console.error('Upload target photo error', e);
        }
    }

    async function updateReidThreshold(val) {
        const threshold = parseFloat(val) / 100.0;
        const disp = document.getElementById('reidThresholdVal');
        if (disp) disp.textContent = val + '%';
        try {
            await fetch('/set_reid_threshold', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ threshold })
            });
        } catch(e) {}
    }

    async function clearSearchTarget() {
        try {
            await fetch('/clear_target', { method: 'POST' });
            fetchTelemetry();
        } catch(e) {}
    }

    async function toggleChildPrior() {
        try {
            const res = await fetch('/toggle_child_prior', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            const data = await res.json();
            const btn = document.getElementById('btnChildPrior');
            if (btn) {
                const active = data.child_prior;
                btn.className = active
                    ? "px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 transition flex items-center gap-1 cursor-pointer"
                    : "px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-white/5 text-white/40 border border-white/10 transition flex items-center gap-1 cursor-pointer";
                btn.innerHTML = `<span>👶</span> Child Scale Prior: ${active ? 'ACTIVE' : 'OFF'}`;
            }
        } catch(e) {}
    }

    // ── Clock ───────────────────────────────────────────
    function updateClock() {
        document.getElementById('clock').textContent =
            new Date().toLocaleTimeString('en-GB', { hour12: false });
    }

    // ── Video Presets ──────────────────────────────────
    async function loadVideoPresets() {
        try {
            const res = await fetch('/list_videos');
            const data = await res.json();
            const sel1 = document.getElementById('videoPresetSelect');
            const sel2 = document.getElementById('videoPresetSelect2');
            const optsHtml = '<option value="">-- Select CCTV Preset --</option>' + 
                (data.videos || []).map(v => `<option value="${v}">${v}</option>`).join('');
            if (sel1) sel1.innerHTML = optsHtml;
            if (sel2) sel2.innerHTML = optsHtml;
        } catch(e) {}
    }

    async function switchPresetVideo(filename) {
        if (!filename) return;
        const statusEl = document.getElementById('uploadStatus');
        statusEl.textContent = `Loading ${filename}…`;
        statusEl.classList.remove('hidden');
        try {
            const res = await fetch('/switch_source', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename })
            });
            const data = await res.json();
            if (data.status === 'ok') {
                statusEl.textContent = `Playing: ${data.source}`;
                setTimeout(() => statusEl.classList.add('hidden'), 2500);
                fetchTelemetry();
            }
        } catch(e) {
            statusEl.textContent = 'Switch failed!';
            setTimeout(() => statusEl.classList.add('hidden'), 2500);
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  LAÂYOUNE CORRIDOR MAP ENGINE (from src/laayoune-map-feature)
    // ═══════════════════════════════════════════════════════════
    const PLACE_MECHOUAR_CENTER = [825, 165];
    const PLACE_MECHOUAR_RADIUS = 58;
    const HOSPITAL_ORIGIN = [115, 495];

    const HOSPITAL_TO_MECHOUAR_PATH = [
        [115, 495], [255, 445], [305, 290], [515, 325], [610, 185], [730, 212], [770, 182]
    ];
    const OPPOSITE_CORRIDOR_PATH = [
        [764, 168], [725, 198], [600, 171], [505, 311], [292, 276], [242, 433], [110, 481]
    ];

    function createCirclePath(cx, cy, radius, steps = 28, clockwise = true) {
        const pts = [];
        for (let i = 0; i <= steps; i++) {
            const ratio = i / steps;
            const angle = (clockwise ? 1 : -1) * ratio * Math.PI * 2;
            pts.push([
                Math.round((cx + Math.cos(angle) * radius) * 10) / 10,
                Math.round((cy + Math.sin(angle) * radius) * 10) / 10
            ]);
        }
        return pts;
    }
    const MECHOUAR_RING_PATH = createCirclePath(PLACE_MECHOUAR_CENTER[0], PLACE_MECHOUAR_CENTER[1], PLACE_MECHOUAR_RADIUS, 28, false);

    const CITY_ROUTES = {
        'green-corridor': HOSPITAL_TO_MECHOUAR_PATH,
        'opp-corridor': OPPOSITE_CORRIDOR_PATH,
        'mechouar-ring': MECHOUAR_RING_PATH,
        'av-hassan-nw-se': [[355, 35], [330, 155], [305, 290], [255, 445], [215, 585]],
        'bd-mekka-we': [[35, 240], [175, 265], [305, 290], [515, 325], [715, 355], [1040, 405]],
        'av-massira-sn': [[415, 585], [465, 455], [515, 325], [610, 185], [695, 35]],
        'rue-smara-we': [[45, 115], [195, 140], [330, 155], [475, 168], [610, 185], [730, 212], [770, 182]],
        'bd-kairouan-ew': [[1035, 525], [820, 495], [640, 475], [465, 455], [255, 445], [115, 495], [35, 525]],
        'av-mohammed-v': [[895, 585], [820, 495], [715, 355], [730, 212], [778, 198]],
        'av-sakia-north': [[1040, 85], [878, 140]],
        'quartier-qods-loop': [[175, 265], [195, 140], [330, 155], [305, 290], [175, 265]],
    };

    const SECONDARY_STREETS = [
        [[195, 35], [195, 140], [175, 265], [135, 385], [115, 495]],
        [[475, 35], [475, 168], [415, 308], [365, 448], [335, 585]],
        [[715, 355], [640, 475], [595, 585]],
        [[883, 165], [1040, 185]],
        [[825, 223], [845, 375], [820, 495]],
        [[825, 107], [810, 30]],
        [[610, 185], [715, 355]],
        [[35, 375], [135, 385], [275, 380], [415, 308]],
        [[695, 35], [775, 120]],
        [[845, 375], [1040, 315]],
    ];

    const IRREGULAR_CITY_BLOCKS = [
        '55,45 180,45 180,125 55,102',
        '210,45 338,45 316,140 210,126',
        '368,45 460,45 460,152 345,142',
        '490,45 675,45 598,168 490,154',
        '50,130 180,152 162,250 48,225',
        '208,155 314,168 292,272 190,252',
        '342,170 462,182 405,292 318,276',
        '488,185 592,196 505,308 426,295',
        '628,182 745,120 755,165 718,195',
        '895,95 1025,52 1025,168 898,152',
        '895,182 1025,200 1025,298 858,352 842,235',
        '48,258 160,278 125,368 45,358',
        '186,282 288,302 265,368 148,370',
        '316,308 402,322 355,432 272,428',
        '425,326 498,338 452,438 375,434',
        '530,342 695,368 628,458 482,440',
        '622,212 715,226 702,338 625,295',
        '745,230 812,238 828,358 730,342',
        '135,462 242,460 202,570 112,568',
        '272,460 352,462 322,570 228,570',
        '378,464 452,468 402,570 348,570',
        '482,470 622,486 580,572 432,572',
        '655,490 802,508 775,575 612,572',
        '732,374 828,386 806,478 655,462',
        '858,388 1020,335 1022,388 845,475',
    ];

    const CORRIDOR_TURN_INTERSECTIONS = [
        { id: 'turn-1', label: 'Turn 1: Rond-Point Kairouan', x: 255, y: 445, isRoundabout: true },
        { id: 'turn-2', label: 'Turn 2: Av. Hassan II × Bd. Mekka', x: 305, y: 290, isRoundabout: false },
        { id: 'turn-3', label: 'Turn 3: Rond-Point Al Massira', x: 515, y: 325, isRoundabout: true },
        { id: 'turn-4', label: 'Turn 4: Rue Smara × Av. Med V', x: 610, y: 185, isRoundabout: false },
        { id: 'turn-5', label: 'Turn 5: Entrée Place du Méchouar', x: 730, y: 212, isRoundabout: false },
    ];

    const DEFAULT_CAMERA_NODES = [
        { id: 'cam-school-01', code: 'CAM-01', name: 'Place du Méchouar (Central Plaza)', sector: 'Central Plaza · Laâyoune' },
        { id: 'cam-mekka-02', code: 'CAM-02', name: 'Boulevard Mekka Crossing', sector: 'Main Avenue · Laâyoune' },
        { id: 'cam-onee-03', code: 'CAM-03', name: 'ONEE Water Plant Gate', sector: 'Restricted Zone · Laâyoune' },
        { id: 'cam-est-04', code: 'CAM-04', name: 'EST Laâyoune Campus Entrance', sector: 'Campus Zone · Laâyoune' },
    ];

    const RED_FEEDER_SEGS = [
        [[355, 35], [330, 155], [305, 275]],
        [[35, 240], [175, 265], [290, 286]],
        [[535, 328], [715, 355], [1040, 405]],
        [[415, 585], [465, 455], [508, 342]],
        [[695, 35], [618, 172]],
        [[45, 115], [195, 140], [330, 155], [475, 168], [595, 182]],
        [[895, 585], [820, 495], [715, 355], [730, 226]],
        [[275, 446], [465, 455], [640, 475], [820, 495], [1035, 525]],
    ];

    const CAR_COLORS = ['#38BDF8', '#FBBF24', '#E2E8F0', '#F8FAFC', '#A78BFA', '#FB923C', '#94A3B8', '#2DD4BF'];

    function getPointOnPolyline(points, t, laneOffset = 0) {
        const clamped = ((t % 1) + 1) % 1;
        if (!points || points.length < 2) return { x: points ? points[0][0] : 0, y: points ? points[0][1] : 0, angle: 0 };
        const segLengths = [];
        let totalLen = 0;
        for (let i = 0; i < points.length - 1; i++) {
            const dx = points[i + 1][0] - points[i][0];
            const dy = points[i + 1][1] - points[i][1];
            const len = Math.hypot(dx, dy);
            segLengths.push(len);
            totalLen += len;
        }
        let targetDist = clamped * totalLen;
        for (let i = 0; i < segLengths.length; i++) {
            if (targetDist <= segLengths[i] || i === segLengths.length - 1) {
                const segLen = Math.max(0.0001, segLengths[i]);
                const localT = Math.max(0, Math.min(1, targetDist / segLen));
                const dx = points[i + 1][0] - points[i][0];
                const dy = points[i + 1][1] - points[i][1];
                const rad = Math.atan2(dy, dx);
                const nx = -Math.sin(rad) * laneOffset;
                const ny = Math.cos(rad) * laneOffset;
                return {
                    x: points[i][0] + dx * localT + nx,
                    y: points[i][1] + dy * localT + ny,
                    angle: (rad * 180) / Math.PI
                };
            }
            targetDist -= segLengths[i];
        }
        const last = points[points.length - 1];
        return { x: last[0], y: last[1], angle: 0 };
    }

    function toSvgPoints(pts) {
        return pts.map(p => `${p[0]},${p[1]}`).join(' ');
    }

    function createRandomizedCars() {
        const specs = [
            { routeId: 'green-corridor', isGreen: true, count: 3 },
            { routeId: 'opp-corridor', isGreen: false, count: 3 },
            { routeId: 'mechouar-ring', isGreen: false, count: 3 },
            { routeId: 'av-hassan-nw-se', isGreen: false, count: 2 },
            { routeId: 'bd-mekka-we', isGreen: false, count: 3 },
            { routeId: 'av-massira-sn', isGreen: false, count: 2 },
            { routeId: 'rue-smara-we', isGreen: false, count: 2 },
            { routeId: 'bd-kairouan-ew', isGreen: false, count: 2 },
            { routeId: 'av-mohammed-v', isGreen: false, count: 2 },
            { routeId: 'quartier-qods-loop', isGreen: false, count: 2 },
        ];
        const list = [];
        let idx = 1;
        specs.forEach(({ routeId, isGreen, count }) => {
            for (let i = 0; i < count; i++) {
                const baseProgress = (i / count + Math.random() * 0.22) % 1;
                list.push({
                    id: `car-${idx++}`,
                    routeId,
                    isGreenCorridorDirection: isGreen,
                    progress: baseProgress,
                    baseSpeed: 0.0016 + Math.random() * 0.0017,
                    jitterPhase: Math.random() * Math.PI * 2,
                    sizeScale: 0.85 + Math.random() * 0.32,
                    laneOffset: (Math.random() - 0.5) * 4.5,
                    color: CAR_COLORS[Math.floor(Math.random() * CAR_COLORS.length)],
                });
            }
        });
        return list;
    }

    // Map State
    let mapCars = createRandomizedCars();
    let ambulanceProgress = 0;
    let mapZoom = 1.15;
    let mapPan = { x: 0, y: 0 };
    let isMapDragging = false;
    let mapDragStart = null;
    let followAmbulance = false;
    let mapGreenWave = false;
    let isManualGreenWave = false;
    let activeCameraId = 'cam-school-01';
    let mapTickCount = 0;

    function initLaayouneCorridorMap() {
        const svg = document.getElementById('laayouneMapSvg');
        if (!svg) return;

        let html = `
            <!-- Base Terrain Canvas -->
            <rect width="1080" height="620" fill="#111927" />

            <!-- Oued Sakia El Hamra Dry Riverbed -->
            <path d="M 0,18 Q 140,35 280,15 T 560,12 L 560,0 L 0,0 Z" fill="#102826" opacity="0.7" />

            <!-- Urban Parks -->
            <polygon points="615,52 745,42 758,108 625,155" fill="#112A28" stroke="#1B433E" stroke-width="1.2" />
            <polygon points="45,405 115,402 102,468 38,465" fill="#112A28" stroke="#1B433E" stroke-width="1.2" />

            <!-- City Blocks -->
            <g fill="#182234" stroke="#23324A" stroke-width="1.2" stroke-linejoin="round">
                ${IRREGULAR_CITY_BLOCKS.map(pts => `<polygon points="${pts}" />`).join('')}
            </g>

            <!-- Secondary Streets -->
            ${SECONDARY_STREETS.map(st => `
                <polyline points="${toSvgPoints(st)}" fill="none" stroke="#202D42" stroke-width="13" stroke-linecap="round" stroke-linejoin="round" />
            `).join('')}

            <!-- Main Avenues (Asphalt & Markings) -->
            ${[
                CITY_ROUTES['av-hassan-nw-se'],
                CITY_ROUTES['bd-mekka-we'],
                CITY_ROUTES['av-massira-sn'],
                CITY_ROUTES['rue-smara-we'],
                CITY_ROUTES['bd-kairouan-ew'],
                CITY_ROUTES['av-mohammed-v'],
                CITY_ROUTES['av-sakia-north'],
            ].map((pts, idx) => `
                <g>
                    <polyline points="${toSvgPoints(pts)}" fill="none" stroke="#2C3C55" stroke-width="${idx === 1 ? '32' : '26'}" stroke-linecap="round" stroke-linejoin="round" />
                    <polyline points="${toSvgPoints(pts)}" fill="none" stroke="#475569" stroke-width="1" stroke-dasharray="7 7" stroke-linecap="round" stroke-linejoin="round" />
                </g>
            `).join('')}

            <!-- Roundabouts (Turn 1 Kairouan & Turn 3 Al Massira) -->
            ${[
                { cx: 255, cy: 445, r: 22 },
                { cx: 515, cy: 325, r: 24 },
            ].map(rb => `
                <g>
                    <circle cx="${rb.cx}" cy="${rb.cy}" r="${rb.r}" fill="#111927" stroke="#2C3C55" stroke-width="14" />
                    <circle cx="${rb.cx}" cy="${rb.cy}" r="${rb.r - 8}" fill="none" stroke="#38BDF8" stroke-opacity="0.35" stroke-width="1.5" />
                </g>
            `).join('')}

            <!-- Place du Méchouar Circle Plaza -->
            <g id="placeMechouarGroup" transform="translate(825, 165)">
                <circle r="58" fill="none" stroke="#2C3C55" stroke-width="26" />
                <circle r="58" fill="none" stroke="#475569" stroke-width="1.2" stroke-dasharray="6 6" />
                <circle id="mechouarCenterCircle" r="44" fill="#111927" stroke="#06B6D4" stroke-width="2.5" />
                <circle id="mechouarInnerCircle" r="36" fill="none" stroke="#06B6D4" stroke-opacity="0.45" stroke-width="1.2" stroke-dasharray="5 4" />
                <text id="mechouarTitle" x="0" y="-4" text-anchor="middle" fill="#E2E8F0" font-size="10.5" font-weight="800">Place du Méchouar</text>
                <text id="mechouarArabic" x="0" y="10" text-anchor="middle" fill="#06B6D4" font-size="9" font-weight="700">ساحة المشور</text>
            </g>

            <!-- Emergency Corridor & Red Feeders Group -->
            <g id="emergencyCorridorGroup" style="display: none;">
                <!-- Opposite Lane Red -->
                <polyline points="${toSvgPoints(OPPOSITE_CORRIDOR_PATH)}" fill="none" stroke="#EF4444" stroke-opacity="0.28" stroke-width="14" stroke-linecap="round" stroke-linejoin="round" />
                <polyline points="${toSvgPoints(OPPOSITE_CORRIDOR_PATH)}" fill="none" stroke="#EF4444" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round" />

                <!-- Red Feeder Intersections -->
                ${RED_FEEDER_SEGS.map(seg => `
                    <g>
                        <polyline points="${toSvgPoints(seg)}" fill="none" stroke="#EF4444" stroke-opacity="0.20" stroke-width="20" stroke-linecap="round" stroke-linejoin="round" />
                        <polyline points="${toSvgPoints(seg)}" fill="none" stroke="#EF4444" stroke-width="4.5" stroke-dasharray="10 6" stroke-linecap="round" stroke-linejoin="round" />
                    </g>
                `).join('')}

                <!-- Green Emergency Polyline (Hospital -> Mechouar) -->
                <polyline points="${toSvgPoints(HOSPITAL_TO_MECHOUAR_PATH)}" fill="none" stroke="#10B981" stroke-opacity="0.35" stroke-width="24" stroke-linecap="round" stroke-linejoin="round" />
                <polyline points="${toSvgPoints(HOSPITAL_TO_MECHOUAR_PATH)}" fill="none" stroke="#10B981" stroke-width="9" stroke-linecap="round" stroke-linejoin="round" />
                <polyline points="${toSvgPoints(HOSPITAL_TO_MECHOUAR_PATH)}" fill="none" stroke="#D1FAE5" stroke-width="3" stroke-dasharray="8 12" stroke-linecap="round" stroke-linejoin="round" />

                <!-- Place du Mechouar Green Ring -->
                <circle cx="825" cy="165" r="58" fill="none" stroke="#10B981" stroke-opacity="0.45" stroke-width="12" />
            </g>

            <!-- Street Names -->
            <text x="415" y="302" fill="#CBD5E1" font-size="10.5" font-weight="700" transform="rotate(9, 415, 302)">Boulevard Mekka · شارع مكة</text>
            <text x="415" y="156" fill="#CBD5E1" font-size="10" font-weight="600" transform="rotate(6, 415, 156)">Rue Smara</text>
            <text x="175" y="462" fill="#CBD5E1" font-size="10" font-weight="600" transform="rotate(-19, 175, 462)">Rue de l'Hôpital</text>
            <text x="560" y="460" fill="#94A3B8" font-size="10" font-weight="600" transform="rotate(6, 560, 460)">Bd. Kairouan</text>
            <text x="306" y="215" fill="#CBD5E1" font-size="10" font-weight="600" transform="rotate(-79, 306, 215)">Av. Hassan II</text>
            <text x="560" y="245" fill="#CBD5E1" font-size="10" font-weight="600" transform="rotate(-55, 560, 245)">Av. Al Massira</text>
            <text x="738" y="295" fill="#CBD5E1" font-size="10" font-weight="600" transform="rotate(-84, 738, 295)">Av. Mohammed V</text>

            <!-- 5 Turn Traffic Signals -->
            <g id="mapTrafficSignalsGroup">
                ${CORRIDOR_TURN_INTERSECTIONS.map((node, i) => `
                    <g id="trafficSignal-${i}" transform="translate(${node.x}, ${node.y})">
                        <circle r="11" fill="#0B0F17" stroke="#64748B" stroke-width="2" class="sig-border" />
                        <circle cx="-3.5" cy="0" r="3" fill="#10B981" />
                        <circle cx="3.5" cy="0" r="3" fill="#F59E0B" class="sig-red-dot" />
                        <g class="sig-badge" transform="translate(0, 22)" style="display: none;">
                            <rect x="-46" y="-8" width="92" height="14" rx="3.5" fill="#0B0F17" stroke="#10B981" stroke-width="1" />
                            <text x="0" y="2" text-anchor="middle" fill="#10B981" font-size="8.5" font-weight="700">T${i + 1}: GREEN / RED</text>
                        </g>
                    </g>
                `).join('')}
            </g>

            <!-- Dynamic Cars Group -->
            <g id="mapCarsGroup"></g>

            <!-- Live Ambulance Group -->
            <g id="mapAmbulanceGroup" style="display: none;">
                <circle r="28" fill="#10B981" fill-opacity="0.24" />
                <circle r="17" fill="#EF4444" fill-opacity="0.28" />
                <g id="ambulanceBody" transform="rotate(0)">
                    <rect x="-15" y="-8" width="30" height="16" rx="4" fill="#FFFFFF" stroke="#0B0F17" stroke-width="2" />
                    <rect x="-4.5" y="-1.5" width="9" height="3" fill="#EF4444" />
                    <rect x="-1.5" y="-4.5" width="3" height="9" fill="#EF4444" />
                    <rect x="7" y="-5.5" width="3.5" height="4.5" fill="#EF4444" />
                    <rect x="7" y="1" width="3.5" height="4.5" fill="#3B82F6" />
                </g>
                <g transform="translate(0, -28)">
                    <rect x="-66" y="-13" width="132" height="20" rx="5" fill="#10B981" stroke="#0B0F17" stroke-width="1.5" />
                    <text id="ambulanceEtaBadgeText" x="0" y="0" text-anchor="middle" fill="#0B0F17" font-size="9.5" font-weight="800">
                        AMBULANCE · 120s
                    </text>
                </g>
            </g>

            <!-- Hospital Pin -->
            <g transform="translate(${HOSPITAL_ORIGIN[0]}, ${HOSPITAL_ORIGIN[1]})">
                <circle r="19" fill="#3B82F6" fill-opacity="0.22" stroke="#3B82F6" stroke-width="2" />
                <circle r="10.5" fill="#3B82F6" stroke="#FFFFFF" stroke-width="2" />
                <text x="0" y="3.5" text-anchor="middle" fill="#FFFFFF" font-size="10" font-weight="800">H</text>
                <g transform="translate(18, 29)">
                    <rect x="-84" y="-11" width="168" height="20" rx="4" fill="#0B0F17" stroke="#3B82F6" stroke-width="1.2" />
                    <text x="0" y="2.5" text-anchor="middle" fill="#FFFFFF" font-size="10" font-weight="700">Hôpital Moulay Hassan</text>
                </g>
            </g>

            <!-- Place du Méchouar Callout Badge -->
            <g id="mapMechouarCallout" transform="translate(825, 83)">
                <rect x="-104" y="-13" width="208" height="22" rx="5" fill="#0B0F17" stroke="#06B6D4" stroke-width="1.4" />
                <text id="mapMechouarCalloutText" x="0" y="2" text-anchor="middle" fill="#FFFFFF" font-size="10.5" font-weight="800">
                    Place du Méchouar (Test Loc)
                </text>
            </g>
        `;

        svg.innerHTML = html;

        // Render camera sector buttons
        const camContainer = document.getElementById('cameraNodesContainer');
        if (camContainer) {
            camContainer.innerHTML = DEFAULT_CAMERA_NODES.map(c => `
                <button type="button" onclick="selectCameraSector('${c.id}')"
                        id="camBtn-${c.id}"
                        class="px-2.5 py-1 text-xs rounded-md transition-colors cursor-pointer ${
                            c.id === activeCameraId ? 'bg-cyan-500 text-surface-950 font-bold' : 'bg-surface-800 text-white/70 hover:text-white'
                        }">
                    ${c.code}: ${c.name}
                </button>
            `).join('');
        }

        // Setup Drag & Pan
        svg.addEventListener('mousedown', (e) => {
            if (followAmbulance) toggleFollowAmbulance();
            isMapDragging = true;
            svg.classList.add('cursor-grabbing');
            svg.classList.remove('cursor-grab');
            mapDragStart = { clientX: e.clientX, clientY: e.clientY, panX: mapPan.x, panY: mapPan.y };
        });

        window.addEventListener('mousemove', (e) => {
            if (!isMapDragging || !mapDragStart) return;
            const dx = (e.clientX - mapDragStart.clientX) * (1.2 / mapZoom);
            const dy = (e.clientY - mapDragStart.clientY) * (1.2 / mapZoom);
            mapPan.x = mapDragStart.panX + dx;
            mapPan.y = mapDragStart.panY + dy;
            updateMapViewBox();
        });

        window.addEventListener('mouseup', () => {
            if (isMapDragging) {
                isMapDragging = false;
                svg.classList.remove('cursor-grabbing');
                svg.classList.add('cursor-grab');
                mapDragStart = null;
            }
        });

        const container = document.getElementById('mapCanvasContainer');
        if (container) {
            container.addEventListener('wheel', (e) => {
                e.preventDefault();
                if (e.deltaY < 0) {
                    mapZoom = Math.min(3.2, Math.round((mapZoom + 0.18) * 100) / 100);
                } else {
                    mapZoom = Math.max(1.0, Math.round((mapZoom - 0.18) * 100) / 100);
                }
                updateMapViewBox();
            }, { passive: false });
        }

        updateMapViewBox();
        setInterval(mapTick, 35);
    }

    function updateMapViewBox() {
        const svg = document.getElementById('laayouneMapSvg');
        if (!svg) return;
        const baseW = 1080;
        const baseH = 620;
        const viewW = baseW / mapZoom;
        const viewH = baseH / mapZoom;

        let cx = baseW / 2 - mapPan.x;
        let cy = baseH / 2 - mapPan.y;

        if (followAmbulance && mapGreenWave) {
            const ambPt = getPointOnPolyline(HOSPITAL_TO_MECHOUAR_PATH, ambulanceProgress);
            cx = ambPt.x;
            cy = ambPt.y;
        }

        const clampedCx = Math.max(viewW / 2, Math.min(baseW - viewW / 2, cx));
        const clampedCy = Math.max(viewH / 2, Math.min(baseH - viewH / 2, cy));
        const vbX = Math.round(clampedCx - viewW / 2);
        const vbY = Math.round(clampedCy - viewH / 2);

        svg.setAttribute('viewBox', `${vbX} ${vbY} ${Math.round(viewW)} ${Math.round(viewH)}`);
        const zoomDisp = document.getElementById('mapZoomDisplay');
        if (zoomDisp) zoomDisp.textContent = mapZoom.toFixed(1) + 'x';
    }

    function zoomInMap() {
        mapZoom = Math.min(3.2, Math.round((mapZoom + 0.35) * 100) / 100);
        updateMapViewBox();
    }

    function zoomOutMap() {
        mapZoom = Math.max(1.0, Math.round((mapZoom - 0.35) * 100) / 100);
        if (mapZoom === 1.0) mapPan = { x: 0, y: 0 };
        updateMapViewBox();
    }

    function resetMapZoom() {
        followAmbulance = false;
        mapZoom = 1.15;
        mapPan = { x: 0, y: 0 };
        updateMapViewBox();
        updateFollowBtnState();
    }

    function focusPlaceMechouar() {
        followAmbulance = false;
        mapZoom = 2.1;
        mapPan = {
            x: 1080 / 2 - PLACE_MECHOUAR_CENTER[0] + 45,
            y: 620 / 2 - PLACE_MECHOUAR_CENTER[1] - 25,
        };
        updateMapViewBox();
        updateFollowBtnState();
    }

    function toggleFollowAmbulance() {
        followAmbulance = !followAmbulance;
        if (followAmbulance && mapZoom < 1.6) mapZoom = 1.75;
        updateMapViewBox();
        updateFollowBtnState();
    }

    function updateFollowBtnState() {
        const btn = document.getElementById('btnFollowAmbulance');
        if (!btn) return;
        if (mapGreenWave) {
            btn.classList.remove('hidden');
            btn.classList.add('flex');
            if (followAmbulance) {
                btn.className = 'flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-emerald-500 text-surface-950 border border-emerald-400 shadow cursor-pointer';
                btn.innerHTML = '<span>🚑</span> Following Ambulance';
            } else {
                btn.className = 'flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-surface-950/90 text-emerald-400 border border-emerald-500/40 hover:bg-surface-900 shadow cursor-pointer';
                btn.innerHTML = '<span>🚑</span> Follow Ambulance';
            }
        } else {
            btn.classList.add('hidden');
            btn.classList.remove('flex');
        }
    }

    function randomizeCars() {
        mapCars = createRandomizedCars();
    }

    function selectCameraSector(camId) {
        activeCameraId = camId;
        DEFAULT_CAMERA_NODES.forEach(c => {
            const btn = document.getElementById(`camBtn-${c.id}`);
            if (btn) {
                if (c.id === camId) {
                    btn.className = 'px-2.5 py-1 text-xs rounded-md transition-colors cursor-pointer bg-cyan-500 text-surface-950 font-bold';
                } else {
                    btn.className = 'px-2.5 py-1 text-xs rounded-md transition-colors cursor-pointer bg-surface-800 text-white/70 hover:text-white';
                }
            }
        });
        if (camId === 'cam-school-01') focusPlaceMechouar();
    }

    function setMapGreenWave(active, remainingSec = 0) {
        if (mapGreenWave === active && !isManualGreenWave) return;
        mapGreenWave = active;

        const badge = document.getElementById('mapStatusBadge');
        const subtext = document.getElementById('mapSubtext');
        const emGroup = document.getElementById('emergencyCorridorGroup');
        const ambGroup = document.getElementById('mapAmbulanceGroup');
        const pulseDot = document.getElementById('ambulancePulseDot');
        const turnStatus = document.getElementById('ambulanceTurnStatus');
        const mechCenter = document.getElementById('mechouarCenterCircle');
        const mechInner = document.getElementById('mechouarInnerCircle');
        const mechCallout = document.getElementById('mapMechouarCallout');
        const mechCalloutText = document.getElementById('mapMechouarCalloutText');
        const legend = document.getElementById('mapLegendContent');
        const btnManual = document.getElementById('btnManualGreenWave');

        if (active) {
            if (badge) {
                badge.textContent = 'GREEN ZONE ACTIVE · 5 TURNS LOCKED';
                badge.className = 'px-2 py-0.5 text-[10px] font-bold uppercase rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/40';
            }
            if (subtext) subtext.textContent = 'Multi-turn route from Hospital → Place du Méchouar is now a GREEN ZONE. Opposite & cross streets turned RED.';
            if (emGroup) emGroup.style.display = 'block';
            if (ambGroup) ambGroup.style.display = 'block';
            if (pulseDot) pulseDot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-400 pulse-live';
            if (mechCenter) mechCenter.setAttribute('stroke', '#EF4444');
            if (mechInner) mechInner.setAttribute('stroke', '#EF4444');
            if (mechCallout && mechCallout.querySelector('rect')) {
                mechCallout.querySelector('rect').setAttribute('fill', '#EF4444');
            }
            if (mechCalloutText) mechCalloutText.textContent = 'FAINT ALERT: Place du Méchouar';
            if (btnManual) {
                btnManual.className = 'flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold rounded-lg transition cursor-pointer bg-red-500 text-white hover:bg-red-600';
                btnManual.innerHTML = '<span>🔄</span> Reset to Normal Traffic';
            }
            if (legend) {
                legend.innerHTML = `
                    <span class="flex items-center gap-1.5 text-emerald-400 font-bold">
                        <span>🛡️</span> 5-Turn Hospital → Place du Méchouar: GREEN ZONE
                    </span>
                    <span class="flex items-center gap-1.5 text-red-400 font-bold">
                        <span>🛑</span> Opposite & Cross Routes: RED (Stopped)
                    </span>
                `;
            }

            // Update Turn Signals to Green/Red
            CORRIDOR_TURN_INTERSECTIONS.forEach((_, i) => {
                const sig = document.getElementById(`trafficSignal-${i}`);
                if (sig) {
                    const border = sig.querySelector('.sig-border');
                    const redDot = sig.querySelector('.sig-red-dot');
                    const badgeEl = sig.querySelector('.sig-badge');
                    if (border) border.setAttribute('stroke', '#10B981');
                    if (redDot) redDot.setAttribute('fill', '#EF4444');
                    if (badgeEl) badgeEl.style.display = 'block';
                }
            });
        } else {
            ambulanceProgress = 0;
            followAmbulance = false;
            isManualGreenWave = false;
            if (badge) {
                badge.textContent = 'ORGANIC CITY CIRCULATION';
                badge.className = 'px-2 py-0.5 text-[10px] font-bold uppercase rounded bg-surface-800 text-white/50 border border-white/10';
            }
            if (subtext) subtext.textContent = 'Realistic Laâyoune streets & roundabouts with random traffic. Drag to pan or use zoom controls.';
            if (emGroup) emGroup.style.display = 'none';
            if (ambGroup) ambGroup.style.display = 'none';
            if (pulseDot) pulseDot.className = 'w-2.5 h-2.5 rounded-full bg-slate-500';
            if (turnStatus) turnStatus.textContent = 'Standby at Hôpital Moulay Hassan';
            if (mechCenter) mechCenter.setAttribute('stroke', '#06B6D4');
            if (mechInner) mechInner.setAttribute('stroke', '#06B6D4');
            if (mechCallout && mechCallout.querySelector('rect')) {
                mechCallout.querySelector('rect').setAttribute('fill', '#0B0F17');
            }
            if (mechCalloutText) mechCalloutText.textContent = 'Place du Méchouar (Test Loc)';
            if (btnManual) {
                btnManual.className = 'flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold rounded-lg transition cursor-pointer bg-emerald-500 text-surface-950 hover:bg-emerald-400';
                btnManual.innerHTML = '<span>🚑</span> Test Faint & Dispatch Ambulance';
            }
            if (legend) {
                legend.innerHTML = `
                    <span class="flex items-center gap-1.5 text-white/70 font-medium">
                        <span class="text-cyan-400">🧭</span> Organic City Traffic (${mapCars.length} Random Cars)
                    </span>
                    <span class="flex items-center gap-1.5 text-white/40">
                        <span class="text-cyan-400">📍</span> Place du Méchouar (Circular Plaza), Laâyoune
                    </span>
                `;
            }

            CORRIDOR_TURN_INTERSECTIONS.forEach((_, i) => {
                const sig = document.getElementById(`trafficSignal-${i}`);
                if (sig) {
                    const border = sig.querySelector('.sig-border');
                    const redDot = sig.querySelector('.sig-red-dot');
                    const badgeEl = sig.querySelector('.sig-badge');
                    if (border) border.setAttribute('stroke', '#64748B');
                    if (redDot) redDot.setAttribute('fill', '#F59E0B');
                    if (badgeEl) badgeEl.style.display = 'none';
                }
            });
        }
        updateFollowBtnState();
    }

    function toggleManualGreenWave() {
        if (mapGreenWave) {
            isManualGreenWave = false;
            setMapGreenWave(false, 0);
            fetch('/clear_alerts', { method: 'POST' }).catch(() => {});
        } else {
            isManualGreenWave = true;
            ambulanceProgress = 0;
            setMapGreenWave(true, 30);
        }
    }

    function mapTick() {
        mapTickCount++;
        const isEmergency = mapGreenWave;

        // 1. Advance Cars
        const carsGroup = document.getElementById('mapCarsGroup');
        let carsHtml = '';

        mapCars.forEach(car => {
            const isStoppedRed = isEmergency && !car.isGreenCorridorDirection;
            if (!isStoppedRed) {
                const wave = 1 + Math.sin(mapTickCount * 0.05 + car.jitterPhase) * 0.25;
                const boost = isEmergency && car.isGreenCorridorDirection ? 1.65 : wave;
                car.progress = (car.progress + car.baseSpeed * boost) % 1;
            }

            const path = CITY_ROUTES[car.routeId] || HOSPITAL_TO_MECHOUAR_PATH;
            const pt = getPointOnPolyline(path, car.progress, car.laneOffset);
            const fillCol = isStoppedRed ? '#EF4444' : car.color;

            carsHtml += `
                <g transform="translate(${pt.x.toFixed(1)}, ${pt.y.toFixed(1)}) rotate(${pt.angle.toFixed(1)}) scale(${car.sizeScale.toFixed(2)})">
                    <rect x="-8" y="-4.5" width="16" height="9" rx="2.5" fill="${fillCol}" stroke="#0B0F17" stroke-width="1.2" />
                    <rect x="2" y="-3" width="3.5" height="6" rx="1" fill="#0F172A" />
                    ${isStoppedRed ? '<circle cx="-9" cy="0" r="3.5" fill="#EF4444" fill-opacity="0.85" />' : ''}
                </g>
            `;
        });
        if (carsGroup) carsGroup.innerHTML = carsHtml;

        // 2. Advance Ambulance if Green Wave Active
        if (isEmergency) {
            ambulanceProgress = Math.min(1, ambulanceProgress + 0.0033);
            if (ambulanceProgress >= 1) ambulanceProgress = 0;

            const ambPt = getPointOnPolyline(HOSPITAL_TO_MECHOUAR_PATH, ambulanceProgress);
            const ambGroup = document.getElementById('mapAmbulanceGroup');
            const ambBody = document.getElementById('ambulanceBody');
            const etaBadge = document.getElementById('ambulanceEtaBadgeText');

            if (ambGroup) ambGroup.setAttribute('transform', `translate(${ambPt.x.toFixed(1)}, ${ambPt.y.toFixed(1)})`);
            if (ambBody) ambBody.setAttribute('transform', `rotate(${ambPt.angle.toFixed(1)})`);

            const remainingKm = Math.max(0, (1 - ambulanceProgress) * 3.4).toFixed(2);
            const etaSec = Math.max(0, Math.ceil((1 - ambulanceProgress) * 120));
            if (etaBadge) etaBadge.textContent = `AMBULANCE · ${etaSec}s`;

            const currentTurnText =
                ambulanceProgress < 0.16 ? "Rue de l'Hôpital → Approaching Rond-Point Kairouan (Turn 1)" :
                ambulanceProgress < 0.35 ? "Turn 1 Left → North-East on Av. Hassan II" :
                ambulanceProgress < 0.58 ? "Turn 2 Right → Diagonal on Boulevard Mekka" :
                ambulanceProgress < 0.75 ? "Turn 3 Left → Rond-Point Al Massira" :
                ambulanceProgress < 0.90 ? "Turn 4 Right → Rue Smara / Av. Mohammed V" :
                ambulanceProgress < 0.98 ? "Turn 5 Left → Entering Place du Méchouar Circle" :
                "Arrived at Place du Méchouar (Laâyoune)";

            const turnEl = document.getElementById('ambulanceTurnStatus');
            if (turnEl) turnEl.textContent = currentTurnText;

            const statsEl = document.getElementById('ambulanceTelemetryStats');
            if (statsEl) {
                statsEl.innerHTML = `
                    <span class="text-white/80">Speed: <strong class="text-emerald-400 font-mono">72 km/h</strong></span>
                    <span class="text-white/80">Dist: <strong class="text-white font-mono">${remainingKm} km</strong></span>
                    <span class="text-white/80">ETA: <strong class="text-emerald-400 font-mono">${etaSec}s</strong></span>
                `;
            }

            if (followAmbulance) updateMapViewBox();
        }
    }

    // ── Boot ────────────────────────────────────────────
    setInterval(fetchTelemetry, 500);
    setInterval(updateClock, 1000);
    updateClock();
    fetchTelemetry();
    loadVideoPresets();
    initLaayouneCorridorMap();
</script>
</body>
</html>
"""


# ── Dashboard Route ──────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the Sahra-Eye dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


# ──────────────────────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
