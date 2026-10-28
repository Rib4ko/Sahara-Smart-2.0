import urllib.request
import json
import time

time.sleep(2)

print("=== 1. Checking GET /telemetry ===")
req = urllib.request.Request("http://127.0.0.1:8000/telemetry")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    print("Current Mode:", data.get("current_mode"))
    print("Target info:", data.get("target_info"))

print("\n=== 2. Switch Mode to 2 (Tifawt-Guard) ===")
req_mode = urllib.request.Request(
    "http://127.0.0.1:8000/mode",
    data=json.dumps({"mode": 2}).encode(),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req_mode) as resp:
    print("Set mode response:", resp.read().decode())

print("\n=== 3. Switch Video to kid_missing.mp4 ===")
req_vid = urllib.request.Request(
    "http://127.0.0.1:8000/switch_source",
    data=json.dumps({"filename": "kid_missing.mp4"}).encode(),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req_vid) as resp:
    print("Switch video response:", resp.read().decode())

print("\n=== 4. Test Natural Language Search via Groq LLM ===")
prompt_desc = "A young child wearing a red hat, yellow shirt, and blue jeans"
req_groq = urllib.request.Request(
    "http://127.0.0.1:8000/search_by_description",
    data=json.dumps({"description": prompt_desc}).encode(),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req_groq) as resp:
    res_data = json.loads(resp.read().decode())
    print("Groq Response Status:", res_data.get("status"))
    print("Groq Extracted Target:", res_data.get("target_info"))
    print("Inference Time:", res_data.get("inference_ms"), "ms")

print("\n=== 5. Monitoring Target Detection on CAM 04 for 6 seconds ===")
for i in range(6):
    time.sleep(1)
    with urllib.request.urlopen("http://127.0.0.1:8000/telemetry") as resp:
        t = json.loads(resp.read().decode())
        reid_alerts = [a for a in t.get("alerts", []) if a.get("type") == "missing_person_match"]
        print(f"Second {i+1}: Found {len(reid_alerts)} Re-ID matches")
        for a in reid_alerts[-2:]:
            msg = a.get("message", "").encode("ascii", "replace").decode("ascii")
            h = a.get("head_sim", 0)
            tor = a.get("torso_sim", 0)
            l = a.get("legs_sim", 0)
            sim = a.get("similarity", 0)
            print(f"   -> {msg} | Sim: {sim}% (H:{h}% T:{tor}% L:{l}%)")
