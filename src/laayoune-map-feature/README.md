# Laâyoune Smart City Corridor & Ambulance Tracker Map (`laayoune-map-feature`)

A standalone, zero-API-key React + TypeScript + Tailwind CSS map module simulating **Laâyoune, Morocco** (`Hôpital Moulay Hassan` → 5 organic city turns → `Place du Méchouar` circular plaza).

## Folder Contents
- `types.ts` — Standalone TypeScript interfaces (`LaayouneCorridorMapProps`, `CameraSectorNode`, `SimCar`, `TurnIntersectionNode`).
- `mapData.ts` — Organic Laâyoune road network polylines, `Place du Méchouar` empty circle coordinates, roundabout generators, and randomized car traffic spawner.
- `LaayouneCorridorMap.tsx` — Interactive SVG map component with Zoom (`+`/`-`/wheel), Drag-to-Pan, `Place du Méchouar` empty circle focus, Green-Wave emergency route activation, Red opposite/cross traffic lock, and live `AMB-LAA-01` ambulance GPS tracker.
- `index.ts` — Barrel export file.

## How to Use in Any React Project

1. Copy the `laayoune-map-feature` folder into your project's `src/` directory.
2. Ensure `lucide-react` and Tailwind CSS are installed (`npm install lucide-react`).
3. Import and render `<LaayouneCorridorMap />`:

```tsx
import React, { useState } from 'react';
import { LaayouneCorridorMap } from './laayoune-map-feature';

export default function MySmartCityApp() {
  const [greenWaveActive, setGreenWaveActive] = useState(false);

  return (
    <div className="max-w-xl mx-auto p-4 bg-[#0B0F17]">
      <LaayouneCorridorMap
        greenWaveActive={greenWaveActive}
        onToggleManualGreenWave={() => setGreenWaveActive((prev) => !prev)}
      />
    </div>
  );
}
```
