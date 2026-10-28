export type MapOperationalMode = 'najda' | 'tifawt' | 'security';

export interface CameraSectorNode {
  id: string;
  code: string;
  name: string;
  sector: string;
  coordinates: string;
  xPct: number;
  yPct: number;
  defaultMode: MapOperationalMode;
}

export interface SimCar {
  id: string;
  routeId: string;
  isGreenCorridorDirection: boolean;
  progress: number;
  baseSpeed: number;
  jitterPhase: number;
  sizeScale: number;
  laneOffset: number;
  color: string;
}

export interface TurnIntersectionNode {
  id: string;
  label: string;
  x: number;
  y: number;
  isRoundabout: boolean;
}

export interface LaayouneCorridorMapProps {
  /** When true, activates the 5-turn Green Zone from Hospital to Place du Méchouar, locks opposite/cross streets RED, and dispatches the ambulance */
  greenWaveActive: boolean;
  /** Currently selected camera ID */
  activeCameraId?: string;
  /** Optional custom list of camera localization nodes */
  cameraNodes?: CameraSectorNode[];
  /** Callback when a camera node is clicked */
  onSelectCamera?: (camera: CameraSectorNode) => void;
  /** Callback to toggle/test the emergency green wave manually */
  onToggleManualGreenWave?: () => void;
}
