# Sahra-Eye — System Schema (Simple Overview)

A simple, non-technical overview of how the smart vision platform operates.

---

```mermaid
flowchart LR
    %% Inputs
    subgraph IN["1. WHAT GOES IN"]
        CAM["📹 City Cameras Network\n(Live street video)"]
        NLP["💬 NLP Search\n('Find child in red cap & yellow shirt')"]
    end

    %% AI Core
    subgraph AI["2. SMART AI BRAIN"]
        direction TB
        MATCH["🎯 Finds Matching People\n(Matches clothes & colors across cameras)"]
        ALERT["🚨 Detects Sudden Falls\n(Immediate alert for medical emergencies)"]
        PRIV["🛡️ Protects Citizen Privacy\n(Automatically blurs all faces)"]
    end

    %% Outputs
    subgraph OUT["3. CITY CONTROL ROOM"]
        DASH["🖥️ Live Control Room\n• Real-time video grid\n• Instant location & red alert\n• Rapid emergency response"]
    end

    CAM --> AI
    NLP --> AI
    AI --> OUT
```

---

## How It Works in 3 Simple Steps:

### Step 1: What Goes In
* **City Cameras Network:** Ingests live video feeds from municipal cameras across the city.
* **NLP Search:** Operators type or speak a plain description (e.g., *"Lost kid in red hat and yellow shirt"*).

### Step 2: What the AI Does
* **Finds Matching People:** Continuously scans all cameras to locate persons matching the description.
* **Detects Medical Emergencies:** Identifies sudden collapses, falls, or cardiac distress in seconds.
* **Protects Citizen Privacy:** Automatically blurs faces so no personal biometric identities are exposed.

### Step 3: What Happens Next
* **City Control Room:** Operators instantly see the target locked on-screen with exact camera location for quick police or ambulance dispatch.
