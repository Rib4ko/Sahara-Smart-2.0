import { CameraSectorNode, SimCar, TurnIntersectionNode } from './types';

// Interpolate [x, y, angleDeg] along any multi-turn polyline path with optional perpendicular lane offset
export function getPointOnPolyline(
  points: [number, number][],
  t: number,
  laneOffset = 0
): { x: number; y: number; angle: number } {
  const clamped = ((t % 1) + 1) % 1;
  if (points.length < 2) return { x: points[0][0], y: points[0][1], angle: 0 };

  const segLengths: number[] = [];
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

      const x = points[i][0] + dx * localT + nx;
      const y = points[i][1] + dy * localT + ny;
      const angle = (rad * 180) / Math.PI;
      return { x, y, angle };
    }
    targetDist -= segLengths[i];
  }

  const last = points[points.length - 1];
  return { x: last[0], y: last[1], angle: 0 };
}

export function toSvgPoints(pts: [number, number][]): string {
  return pts.map(([x, y]) => `${x},${y}`).join(' ');
}

// Generate points around a circle for roundabout circulation
export function createCirclePath(
  cx: number,
  cy: number,
  radius: number,
  steps = 24,
  clockwise = true
): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i <= steps; i++) {
    const ratio = i / steps;
    const angle = (clockwise ? 1 : -1) * ratio * Math.PI * 2;
    pts.push([
      Math.round((cx + Math.cos(angle) * radius) * 10) / 10,
      Math.round((cy + Math.sin(angle) * radius) * 10) / 10,
    ]);
  }
  return pts;
}

export const PLACE_MECHOUAR_CENTER: [number, number] = [825, 165];
export const PLACE_MECHOUAR_RADIUS = 58; // Large prominent empty circle
export const HOSPITAL_ORIGIN: [number, number] = [115, 495];

// 1. MULTI-TURN EMERGENCY ROUTE (Hôpital Moulay Hassan -> 5 Realistic Organic Turns -> Place du Méchouar Circle)
export const HOSPITAL_TO_MECHOUAR_PATH: [number, number][] = [
  [115, 495], // Hôpital Moulay Hassan Gate
  [255, 445], // Turn 1: Rond-Point Kairouan / Av. Hassan II
  [305, 290], // Turn 2: Carrefour Av. Hassan II & Boulevard Mekka
  [515, 325], // Turn 3: Rond-Point Al Massira
  [610, 185], // Turn 4: Carrefour Rue Smara & Av. Mohammed V
  [730, 212], // Turn 5: Approche Sud Place du Méchouar
  [770, 182], // Entry onto Place du Méchouar Empty Circle Perimeter
];

// 2. OPPOSITE LANE ALONG THE SAME MULTI-TURN CORRIDOR (Place du Méchouar -> Hospital, turns RED & STOPS on emergency)
export const OPPOSITE_CORRIDOR_PATH: [number, number][] = [
  [764, 168],
  [725, 198],
  [600, 171],
  [505, 311],
  [292, 276],
  [242, 433],
  [110, 481],
];

// Circular ring path around Place du Méchouar empty circle
export const MECHOUAR_RING_PATH = createCirclePath(
  PLACE_MECHOUAR_CENTER[0],
  PLACE_MECHOUAR_CENTER[1],
  PLACE_MECHOUAR_RADIUS,
  28,
  false
);

// 3. ORGANIC CITY ROUTES (Angled avenues, diagonal streets & roundabouts across Laâyoune)
export const CITY_ROUTES: Record<string, [number, number][]> = {
  'green-corridor': HOSPITAL_TO_MECHOUAR_PATH,
  'opp-corridor': OPPOSITE_CORRIDOR_PATH,
  'mechouar-ring': MECHOUAR_RING_PATH,

  // Diagonal Avenue Hassan II (North-West to South-East across Turn 2 & Turn 1)
  'av-hassan-nw-se': [
    [355, 35],
    [330, 155],
    [305, 290],
    [255, 445],
    [215, 585],
  ],

  // Diagonal Boulevard Mekka (West-North-West to East-South-East across Turn 2 & Turn 3)
  'bd-mekka-we': [
    [35, 240],
    [175, 265],
    [305, 290],
    [515, 325],
    [715, 355],
    [1040, 405],
  ],

  // Angled Avenue Al Massira (South-West to North-East across Turn 3 & Turn 4)
  'av-massira-sn': [
    [415, 585],
    [465, 455],
    [515, 325],
    [610, 185],
    [695, 35],
  ],

  // Rue Smara Winding Collector (West to North-East across Av. Hassan II, Turn 4 & Place du Méchouar)
  'rue-smara-we': [
    [45, 115],
    [195, 140],
    [330, 155],
    [475, 168],
    [610, 185],
    [730, 212],
    [770, 182],
  ],

  // Boulevard Kairouan Southern Arterial (East to West-South-West across Turn 1)
  'bd-kairouan-ew': [
    [1035, 525],
    [820, 495],
    [640, 475],
    [465, 455],
    [255, 445],
    [115, 495],
    [35, 525],
  ],

  // Avenue Mohammed V Eastern Connector (South to Place du Méchouar Circle to North)
  'av-mohammed-v': [
    [895, 585],
    [820, 495],
    [715, 355],
    [730, 212],
    [778, 198],
  ],

  // Northern Corniche / Oued Sakia Connector into Place du Méchouar
  'av-sakia-north': [
    [1040, 85],
    [878, 140],
  ],

  // Western Quartier Al Qods Diagonal Loop
  'quartier-qods-loop': [
    [175, 265],
    [195, 140],
    [330, 155],
    [305, 290],
    [175, 265],
  ],
};

// Secondary organic neighborhood streets
export const SECONDARY_STREETS: [number, number][][] = [
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

// Irregular realistic city block polygons
export const IRREGULAR_CITY_BLOCKS: string[] = [
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

export const CORRIDOR_TURN_INTERSECTIONS: TurnIntersectionNode[] = [
  { id: 'turn-1', label: 'Turn 1: Rond-Point Kairouan', x: 255, y: 445, isRoundabout: true },
  { id: 'turn-2', label: 'Turn 2: Av. Hassan II × Bd. Mekka', x: 305, y: 290, isRoundabout: false },
  { id: 'turn-3', label: 'Turn 3: Rond-Point Al Massira', x: 515, y: 325, isRoundabout: true },
  { id: 'turn-4', label: 'Turn 4: Rue Smara × Av. Med V', x: 610, y: 185, isRoundabout: false },
  { id: 'turn-5', label: 'Turn 5: Entrée Place du Méchouar', x: 730, y: 212, isRoundabout: false },
];

export const DEFAULT_LAAYOUNE_CAMERA_NODES: CameraSectorNode[] = [
  {
    id: 'cam-school-01',
    code: 'CAM-01',
    name: 'Place du Méchouar (Test Localization)',
    sector: 'Central Plaza · Laâyoune',
    coordinates: 'Laâyoune, Morocco',
    xPct: 48,
    yPct: 46,
    defaultMode: 'najda',
  },
  {
    id: 'cam-mekka-02',
    code: 'CAM-02',
    name: 'Boulevard Mekka Crossing',
    sector: 'Main Avenue · Laâyoune',
    coordinates: 'Laâyoune, Morocco',
    xPct: 33,
    yPct: 58,
    defaultMode: 'tifawt',
  },
  {
    id: 'cam-onee-03',
    code: 'CAM-03',
    name: 'ONEE Water Plant Gate',
    sector: 'Restricted Zone · Laâyoune',
    coordinates: 'Laâyoune, Morocco',
    xPct: 19,
    yPct: 28,
    defaultMode: 'security',
  },
  {
    id: 'cam-est-04',
    code: 'CAM-04',
    name: 'EST Laâyoune Campus Entrance',
    sector: 'Campus Zone · Laâyoune',
    coordinates: 'Laâyoune, Morocco',
    xPct: 74,
    yPct: 67,
    defaultMode: 'security',
  },
];

const CAR_COLORS = [
  '#38BDF8',
  '#FBBF24',
  '#E2E8F0',
  '#F8FAFC',
  '#A78BFA',
  '#FB923C',
  '#94A3B8',
  '#2DD4BF',
];

export function createRandomizedCars(): SimCar[] {
  const specs: Array<{ routeId: string; isGreen: boolean; count: number }> = [
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

  const list: SimCar[] = [];
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
