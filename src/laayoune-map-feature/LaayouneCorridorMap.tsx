import React, { useEffect, useState, useRef } from 'react';
import {
  Navigation,
  Siren,
  MapPin,
  ShieldCheck,
  OctagonAlert,
  Play,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Shuffle,
  Crosshair,
} from 'lucide-react';
import {
  LaayouneCorridorMapProps,
  SimCar,
} from './types';
import {
  PLACE_MECHOUAR_CENTER,
  PLACE_MECHOUAR_RADIUS,
  HOSPITAL_ORIGIN,
  HOSPITAL_TO_MECHOUAR_PATH,
  OPPOSITE_CORRIDOR_PATH,
  CITY_ROUTES,
  SECONDARY_STREETS,
  IRREGULAR_CITY_BLOCKS,
  CORRIDOR_TURN_INTERSECTIONS,
  DEFAULT_LAAYOUNE_CAMERA_NODES,
  getPointOnPolyline,
  toSvgPoints,
  createRandomizedCars,
} from './mapData';

export const LaayouneCorridorMap: React.FC<LaayouneCorridorMapProps> = ({
  greenWaveActive,
  activeCameraId = 'cam-school-01',
  cameraNodes = DEFAULT_LAAYOUNE_CAMERA_NODES,
  onSelectCamera,
  onToggleManualGreenWave,
}) => {
  const [cars, setCars] = useState<SimCar[]>(() => createRandomizedCars());
  const [ambulanceProgress, setAmbulanceProgress] = useState<number>(0);

  // Interactive Zoom & Pan State
  const [zoom, setZoom] = useState<number>(1.15);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [followAmbulance, setFollowAmbulance] = useState<boolean>(false);
  const dragStartRef = useRef<{ clientX: number; clientY: number; panX: number; panY: number } | null>(
    null
  );

  const greenWaveRef = useRef<boolean>(greenWaveActive);

  useEffect(() => {
    greenWaveRef.current = greenWaveActive;
    if (!greenWaveActive) {
      setAmbulanceProgress(0);
      setFollowAmbulance(false);
    }
  }, [greenWaveActive]);

  // 30 FPS Organic City Traffic & Multi-Turn Ambulance Tracker Loop
  useEffect(() => {
    let tick = 0;
    const interval = setInterval(() => {
      tick++;
      const isEmergency = greenWaveRef.current;

      setCars((prevCars) =>
        prevCars.map((car) => {
          // When emergency green wave is active, all cars on opposite & cross routes STOP at the red lights
          if (isEmergency && !car.isGreenCorridorDirection) {
            return car;
          }

          // Organic random speed variation so cars don't move robotically
          const wave = 1 + Math.sin(tick * 0.05 + car.jitterPhase) * 0.25;
          const boost = isEmergency && car.isGreenCorridorDirection ? 1.65 : wave;
          let nextProg = car.progress + car.baseSpeed * boost;
          if (nextProg > 1) nextProg -= 1;

          return {
            ...car,
            progress: nextProg,
          };
        })
      );

      if (isEmergency) {
        setAmbulanceProgress((prev) => {
          if (prev >= 1) return 0;
          return Math.min(1, prev + 0.0033);
        });
      }
    }, 35);

    return () => clearInterval(interval);
  }, []);

  const ambulanceCoord = getPointOnPolyline(
    HOSPITAL_TO_MECHOUAR_PATH,
    ambulanceProgress
  );
  const remainingKm = Math.max(0, (1 - ambulanceProgress) * 3.4).toFixed(2);
  const etaSeconds = Math.max(0, Math.ceil((1 - ambulanceProgress) * 120));

  const currentTurnStatus =
    ambulanceProgress < 0.16
      ? "Rue de l'Hôpital → Approaching Rond-Point Kairouan (Turn 1)"
      : ambulanceProgress < 0.35
      ? 'Turn 1 Left → North-East on Av. Hassan II'
      : ambulanceProgress < 0.58
      ? 'Turn 2 Right → Diagonal on Boulevard Mekka'
      : ambulanceProgress < 0.75
      ? 'Turn 3 Left → Rond-Point Al Massira'
      : ambulanceProgress < 0.90
      ? 'Turn 4 Right → Rue Smara / Av. Mohammed V'
      : ambulanceProgress < 0.98
      ? 'Turn 5 Left → Entering Place du Méchouar Circle'
      : 'Arrived at Place du Méchouar (Laâyoune)';

  // Dynamic SVG ViewBox calculation for smooth Zoom & Pan
  const baseW = 1080;
  const baseH = 620;
  const viewW = baseW / zoom;
  const viewH = baseH / zoom;

  let centerX = baseW / 2 - pan.x;
  let centerY = baseH / 2 - pan.y;

  if (followAmbulance && greenWaveActive) {
    centerX = ambulanceCoord.x;
    centerY = ambulanceCoord.y;
  }

  const clampedCenterX = Math.max(viewW / 2, Math.min(baseW - viewW / 2, centerX));
  const clampedCenterY = Math.max(viewH / 2, Math.min(baseH - viewH / 2, centerY));
  const viewBoxX = Math.round(clampedCenterX - viewW / 2);
  const viewBoxY = Math.round(clampedCenterY - viewH / 2);

  const handleZoomIn = () => {
    setZoom((z) => Math.min(3.2, Math.round((z + 0.35) * 100) / 100));
  };

  const handleZoomOut = () => {
    setZoom((z) => {
      const next = Math.max(1.0, Math.round((z - 0.35) * 100) / 100);
      if (next === 1.0) setPan({ x: 0, y: 0 });
      return next;
    });
  };

  const handleResetView = () => {
    setFollowAmbulance(false);
    setZoom(1.15);
    setPan({ x: 0, y: 0 });
  };

  const handleFocusPlaceMechouar = () => {
    setFollowAmbulance(false);
    setZoom(2.1);
    setPan({
      x: baseW / 2 - PLACE_MECHOUAR_CENTER[0] + 45,
      y: baseH / 2 - PLACE_MECHOUAR_CENTER[1] - 25,
    });
  };

  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    if (followAmbulance) setFollowAmbulance(false);
    setIsDragging(true);
    dragStartRef.current = {
      clientX: e.clientX,
      clientY: e.clientY,
      panX: pan.x,
      panY: pan.y,
    };
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!isDragging || !dragStartRef.current) return;
    const dx = (e.clientX - dragStartRef.current.clientX) * (1.2 / zoom);
    const dy = (e.clientY - dragStartRef.current.clientY) * (1.2 / zoom);
    setPan({
      x: dragStartRef.current.panX + dx,
      y: dragStartRef.current.panY + dy,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    dragStartRef.current = null;
  };

  const handleWheelZoom = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      setZoom((z) => Math.min(3.2, Math.round((z + 0.2) * 100) / 100));
    } else {
      setZoom((z) => Math.max(1.0, Math.round((z - 0.2) * 100) / 100));
    }
  };

  return (
    <div className="bg-[#111827] border border-[#1E293B] rounded-xl overflow-hidden flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-white">
              Laâyoune City Map — Hôpital Moulay Hassan to Place du Méchouar
            </h3>
            <span
              className={`px-2 py-0.5 text-[10px] font-bold uppercase rounded ${
                greenWaveActive
                  ? 'bg-[#10B981]/20 text-[#10B981] border border-[#10B981]/40'
                  : 'bg-[#1E293B] text-[#94A3B8] border border-[#334155]'
              }`}
            >
              {greenWaveActive ? 'GREEN ZONE ACTIVE · 5 TURNS LOCKED' : 'ORGANIC CITY CIRCULATION'}
            </span>
          </div>
          <p className="text-xs text-[#94A3B8] mt-0.5">
            {greenWaveActive
              ? 'Multi-turn route from Hospital → Place du Méchouar is now a GREEN ZONE. Opposite & cross streets turned RED.'
              : 'Realistic Laâyoune streets & roundabouts with random traffic. Drag to pan or use zoom controls.'}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setCars(createRandomizedCars())}
            title="Randomize Car Traffic Positions & Speeds"
            className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium bg-[#1E293B] hover:bg-[#334155] text-[#E2E8F0] rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            <Shuffle className="w-3.5 h-3.5 text-[#38BDF8]" />
            Randomize Cars
          </button>

          {onToggleManualGreenWave && (
            <button
              type="button"
              onClick={onToggleManualGreenWave}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors whitespace-nowrap shrink-0 ${
                greenWaveActive
                  ? 'bg-[#EF4444] text-white hover:bg-[#DC2626]'
                  : 'bg-[#10B981] text-[#0B0F17] hover:bg-[#34D399]'
              }`}
            >
              {greenWaveActive ? (
                <>
                  <RotateCcw className="w-3.5 h-3.5" />
                  Reset to Normal Traffic
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" />
                  Test Faint & Dispatch Ambulance
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {/* Live Ambulance Telemetry & Turn-by-Turn Navigation Bar */}
      <div
        className={`px-4 py-2.5 border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-3 text-xs transition-colors ${
          greenWaveActive ? 'bg-[#064E3B]/40' : 'bg-[#0B0F17]'
        }`}
      >
        <div className="flex items-center gap-2">
          <span
            className={`w-2.5 h-2.5 rounded-full ${
              greenWaveActive ? 'bg-[#10B981] animate-ping' : 'bg-[#64748B]'
            }`}
          />
          <Siren
            className={`w-4 h-4 ${
              greenWaveActive ? 'text-[#10B981]' : 'text-[#64748B]'
            }`}
          />
          <span className="font-semibold text-white">Ambulance GPS (AMB-LAA-01):</span>
          <span className={greenWaveActive ? 'text-[#34D399] font-medium' : 'text-[#94A3B8]'}>
            {greenWaveActive ? currentTurnStatus : 'Standby at Hôpital Moulay Hassan'}
          </span>
        </div>

        {greenWaveActive ? (
          <div className="flex items-center gap-3 font-mono tabular-nums text-[11px]">
            <span className="text-[#CBD5E1]">
              Speed: <strong className="text-[#10B981]">72 km/h</strong>
            </span>
            <span className="text-[#CBD5E1]">
              Dist: <strong className="text-white">{remainingKm} km</strong>
            </span>
            <span className="text-[#CBD5E1]">
              ETA: <strong className="text-[#10B981]">{etaSeconds}s</strong>
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-4 text-[11px] text-[#94A3B8]">
            <span>
              Active Traffic: <strong className="text-white">{cars.length} vehicles circulating</strong>
            </span>
            <span>
              Destination: <strong className="text-[#06B6D4]">Place du Méchouar (Circle)</strong>
            </span>
          </div>
        )}
      </div>

      {/* Bigger, Zoomable & Pannable Realistic Laâyoune City Map Canvas */}
      <div
        onWheel={handleWheelZoom}
        className="relative h-[475px] w-full bg-[#111927] overflow-hidden select-none"
      >
        {/* Top-Right Interactive Map Zoom & Focus Controls */}
        <div className="absolute top-3 right-3 z-20 flex flex-col gap-1.5 bg-[#0B0F17]/90 backdrop-blur-md border border-[#1E293B] p-1.5 rounded-xl shadow-lg">
          <button
            type="button"
            onClick={handleZoomIn}
            title="Zoom In"
            className="p-2 rounded-lg bg-[#161F30] hover:bg-[#1E293B] text-white transition-colors flex items-center justify-center"
          >
            <ZoomIn className="w-4 h-4 text-[#06B6D4]" />
          </button>
          <div className="text-[10px] font-mono tabular-nums text-center text-[#94A3B8] py-0.5">
            {zoom.toFixed(1)}x
          </div>
          <button
            type="button"
            onClick={handleZoomOut}
            title="Zoom Out"
            className="p-2 rounded-lg bg-[#161F30] hover:bg-[#1E293B] text-white transition-colors flex items-center justify-center"
          >
            <ZoomOut className="w-4 h-4 text-[#06B6D4]" />
          </button>
          <button
            type="button"
            onClick={handleResetView}
            title="Reset Map Zoom"
            className="p-2 rounded-lg bg-[#161F30] hover:bg-[#1E293B] text-[#CBD5E1] hover:text-white transition-colors flex items-center justify-center"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Top-Left Quick Focus Buttons (Zoom to Place du Méchouar / Follow Ambulance) */}
        <div className="absolute top-3 left-3 z-20 flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={handleFocusPlaceMechouar}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-[#0B0F17]/90 hover:bg-[#161F30] text-[#06B6D4] border border-[#06B6D4]/40 backdrop-blur-md shadow"
          >
            <Crosshair className="w-3.5 h-3.5" />
            Zoom to Place du Méchouar
          </button>

          {greenWaveActive && (
            <button
              type="button"
              onClick={() => {
                setFollowAmbulance((prev) => !prev);
                if (!followAmbulance && zoom < 1.6) setZoom(1.75);
              }}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold border backdrop-blur-md shadow transition-colors ${
                followAmbulance
                  ? 'bg-[#10B981] text-[#0B0F17] border-[#10B981]'
                  : 'bg-[#0B0F17]/90 text-[#10B981] border-[#10B981]/40 hover:bg-[#161F30]'
              }`}
            >
              <Siren className="w-3.5 h-3.5" />
              {followAmbulance ? 'Following Ambulance' : 'Follow Ambulance'}
            </button>
          )}
        </div>

        <svg
          viewBox={`${viewBoxX} ${viewBoxY} ${Math.round(viewW)} ${Math.round(viewH)}`}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className={`w-full h-full object-cover ${
            isDragging ? 'cursor-grabbing' : 'cursor-grab'
          }`}
          aria-label="Laâyoune Organic Multi-Turn Traffic Map with Place du Méchouar Circle"
        >
          {/* Base Laâyoune Terrain Canvas */}
          <rect width="1080" height="620" fill="#111927" />

          {/* Natural Oued Sakia El Hamra Dry Riverbed / Green Belt in North-West Corner */}
          <path
            d="M 0,18 Q 140,35 280,15 T 560,12 L 560,0 L 0,0 Z"
            fill="#102826"
            opacity="0.7"
          />

          {/* Urban Parks & Green Zones */}
          <polygon
            points="615,52 745,42 758,108 625,155"
            fill="#112A28"
            stroke="#1B433E"
            strokeWidth="1.2"
          />
          <polygon
            points="45,405 115,402 102,468 38,465"
            fill="#112A28"
            stroke="#1B433E"
            strokeWidth="1.2"
          />

          {/* Irregular Realistic City Blocks */}
          <g fill="#182234" stroke="#23324A" strokeWidth="1.2" strokeLinejoin="round">
            {IRREGULAR_CITY_BLOCKS.map((pts, i) => (
              <polygon key={`blk-${i}`} points={pts} />
            ))}
          </g>

          {/* Secondary Organic Neighborhood Streets */}
          {SECONDARY_STREETS.map((st, idx) => (
            <polyline
              key={`sec-st-${idx}`}
              points={toSvgPoints(st)}
              fill="none"
              stroke="#202D42"
              strokeWidth="13"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ))}

          {/* ================= MAIN ORGANIC LAÂYOUNE AVENUES (ASPHALT) ================= */}
          {[
            CITY_ROUTES['av-hassan-nw-se'],
            CITY_ROUTES['bd-mekka-we'],
            CITY_ROUTES['av-massira-sn'],
            CITY_ROUTES['rue-smara-we'],
            CITY_ROUTES['bd-kairouan-ew'],
            CITY_ROUTES['av-mohammed-v'],
            CITY_ROUTES['av-sakia-north'],
          ].map((pts, idx) => (
            <g key={`main-av-${idx}`}>
              <polyline
                points={toSvgPoints(pts)}
                fill="none"
                stroke="#2C3C55"
                strokeWidth={idx === 1 ? '32' : '26'}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <polyline
                points={toSvgPoints(pts)}
                fill="none"
                stroke="#475569"
                strokeWidth="1"
                strokeDasharray="7 7"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </g>
          ))}

          {/* Roundabout Asphalt Rings at Turn 1 (Kairouan) & Turn 3 (Al Massira) */}
          {[
            { cx: 255, cy: 445, r: 22 },
            { cx: 515, cy: 325, r: 24 },
          ].map((rb, i) => (
            <g key={`rb-${i}`}>
              <circle
                cx={rb.cx}
                cy={rb.cy}
                r={rb.r}
                fill="#111927"
                stroke="#2C3C55"
                strokeWidth="14"
              />
              <circle
                cx={rb.cx}
                cy={rb.cy}
                r={rb.r - 8}
                fill="none"
                stroke="#38BDF8"
                strokeOpacity="0.35"
                strokeWidth="1.5"
              />
            </g>
          ))}

          {/* ===================================================================== */}
          {/* PLACE DU MÉCHOUAR — LARGE EMPTY CIRCLE PLAZA & ROUNDABOUT RING        */}
          {/* ===================================================================== */}
          <g
            transform={`translate(${PLACE_MECHOUAR_CENTER[0]}, ${PLACE_MECHOUAR_CENTER[1]})`}
          >
            {/* Outer Circular Roadway of Place du Méchouar */}
            <circle
              r={PLACE_MECHOUAR_RADIUS}
              fill="none"
              stroke="#2C3C55"
              strokeWidth="26"
            />
            <circle
              r={PLACE_MECHOUAR_RADIUS}
              fill="none"
              stroke="#475569"
              strokeWidth="1.2"
              strokeDasharray="6 6"
            />

            {/* THE EMPTY CIRCLE OF PLACE DU MÉCHOUAR */}
            <circle
              r={PLACE_MECHOUAR_RADIUS - 14}
              fill="#111927"
              stroke={greenWaveActive ? '#EF4444' : '#06B6D4'}
              strokeWidth={greenWaveActive ? '3.5' : '2.5'}
            />
            <circle
              r={PLACE_MECHOUAR_RADIUS - 22}
              fill="none"
              stroke={greenWaveActive ? '#EF4444' : '#06B6D4'}
              strokeOpacity="0.45"
              strokeWidth="1.2"
              strokeDasharray="5 4"
            />

            <text
              x="0"
              y="-4"
              textAnchor="middle"
              fill={greenWaveActive ? '#FCA5A5' : '#E2E8F0'}
              fontSize="10.5"
              fontWeight="800"
            >
              Place du Méchouar
            </text>
            <text
              x="0"
              y="10"
              textAnchor="middle"
              fill={greenWaveActive ? '#EF4444' : '#06B6D4'}
              fontSize="9"
              fontWeight="700"
            >
              ساحة المشور
            </text>
          </g>

          {/* ================= EMERGENCY FAINT OVERLAYS (HIDDEN UNTIL FAINT!) ================= */}
          {greenWaveActive && (
            <g>
              {/* 1. OPPOSITE & INTERSECTING STREETS TURN RED (CARS STOPPED) */}
              <polyline
                points={toSvgPoints(OPPOSITE_CORRIDOR_PATH)}
                fill="none"
                stroke="#EF4444"
                strokeOpacity="0.28"
                strokeWidth="14"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <polyline
                points={toSvgPoints(OPPOSITE_CORRIDOR_PATH)}
                fill="none"
                stroke="#EF4444"
                strokeWidth="4.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {/* Intersecting Feeder Streets Locked RED */}
              {[
                [[355, 35], [330, 155], [305, 275]],
                [[35, 240], [175, 265], [290, 286]],
                [[535, 328], [715, 355], [1040, 405]],
                [[415, 585], [465, 455], [508, 342]],
                [[695, 35], [618, 172]],
                [[45, 115], [195, 140], [330, 155], [475, 168], [595, 182]],
                [[895, 585], [820, 495], [715, 355], [730, 226]],
                [[275, 446], [465, 455], [640, 475], [820, 495], [1035, 525]],
              ].map((seg, i) => (
                <g key={`red-seg-${i}`}>
                  <polyline
                    points={toSvgPoints(seg as [number, number][])}
                    fill="none"
                    stroke="#EF4444"
                    strokeOpacity="0.20"
                    strokeWidth="20"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <polyline
                    points={toSvgPoints(seg as [number, number][])}
                    fill="none"
                    stroke="#EF4444"
                    strokeWidth="4.5"
                    strokeDasharray="10 6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </g>
              ))}

              {/* 2. MULTI-TURN HOSPITAL -> PLACE DU MÉCHOUAR GREEN ZONE */}
              <polyline
                points={toSvgPoints(HOSPITAL_TO_MECHOUAR_PATH)}
                fill="none"
                stroke="#10B981"
                strokeOpacity="0.35"
                strokeWidth="24"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <polyline
                points={toSvgPoints(HOSPITAL_TO_MECHOUAR_PATH)}
                fill="none"
                stroke="#10B981"
                strokeWidth="9"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <polyline
                points={toSvgPoints(HOSPITAL_TO_MECHOUAR_PATH)}
                fill="none"
                stroke="#D1FAE5"
                strokeWidth="3"
                strokeDasharray="8 12"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {/* Green Ring Highlight around Place du Méchouar Empty Circle */}
              <circle
                cx={PLACE_MECHOUAR_CENTER[0]}
                cy={PLACE_MECHOUAR_CENTER[1]}
                r={PLACE_MECHOUAR_RADIUS}
                fill="none"
                stroke="#10B981"
                strokeOpacity="0.45"
                strokeWidth="12"
              />
            </g>
          )}

          {/* ================= STREET NAMES ALONG ANGLED AVENUES ================= */}
          <text
            x="415"
            y="302"
            fill="#CBD5E1"
            fontSize="10.5"
            fontWeight="700"
            transform="rotate(9, 415, 302)"
          >
            Boulevard Mekka · شارع مكة
          </text>
          <text
            x="415"
            y="156"
            fill="#CBD5E1"
            fontSize="10"
            fontWeight="600"
            transform="rotate(6, 415, 156)"
          >
            Rue Smara
          </text>
          <text
            x="175"
            y="462"
            fill="#CBD5E1"
            fontSize="10"
            fontWeight="600"
            transform="rotate(-19, 175, 462)"
          >
            Rue de l&apos;Hôpital
          </text>
          <text
            x="560"
            y="460"
            fill="#94A3B8"
            fontSize="10"
            fontWeight="600"
            transform="rotate(6, 560, 460)"
          >
            Bd. Kairouan
          </text>
          <text
            x="306"
            y="215"
            fill="#CBD5E1"
            fontSize="10"
            fontWeight="600"
            transform="rotate(-79, 306, 215)"
          >
            Av. Hassan II
          </text>
          <text
            x="560"
            y="245"
            fill="#CBD5E1"
            fontSize="10"
            fontWeight="600"
            transform="rotate(-55, 560, 245)"
          >
            Av. Al Massira
          </text>
          <text
            x="738"
            y="295"
            fill="#CBD5E1"
            fontSize="10"
            fontWeight="600"
            transform="rotate(-84, 738, 295)"
          >
            Av. Mohammed V
          </text>

          {/* ================= SIMULATED RANDOMIZED CITY CARS ================= */}
          {cars.map((car) => {
            const path = CITY_ROUTES[car.routeId] || HOSPITAL_TO_MECHOUAR_PATH;
            const pt = getPointOnPolyline(path, car.progress, car.laneOffset);
            const isStoppedRed = greenWaveActive && !car.isGreenCorridorDirection;

            return (
              <g
                key={car.id}
                transform={`translate(${pt.x}, ${pt.y}) rotate(${pt.angle}) scale(${car.sizeScale})`}
              >
                <rect
                  x="-8"
                  y="-4.5"
                  width="16"
                  height="9"
                  rx="2.5"
                  fill={isStoppedRed ? '#EF4444' : car.color}
                  stroke="#0B0F17"
                  strokeWidth="1.2"
                />
                <rect x="2" y="-3" width="3.5" height="6" rx="1" fill="#0F172A" />
                {isStoppedRed && (
                  <circle cx="-9" cy="0" r="3.5" fill="#EF4444" fillOpacity="0.85" />
                )}
              </g>
            );
          })}

          {/* ================= 5 MULTI-TURN INTERSECTION TRAFFIC SIGNALS ================= */}
          {CORRIDOR_TURN_INTERSECTIONS.map((node, index) => (
            <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
              <circle
                r="11"
                fill="#0B0F17"
                stroke={greenWaveActive ? '#10B981' : '#64748B'}
                strokeWidth="2"
              />
              <circle cx="-3.5" cy="0" r="3" fill="#10B981" />
              <circle
                cx="3.5"
                cy="0"
                r="3"
                fill={greenWaveActive ? '#EF4444' : '#F59E0B'}
              />
              {greenWaveActive && (
                <g transform="translate(0, 22)">
                  <rect
                    x="-46"
                    y="-8"
                    width="92"
                    height="14"
                    rx="3.5"
                    fill="#0B0F17"
                    stroke="#10B981"
                    strokeWidth="1"
                  />
                  <text
                    x="0"
                    y="2"
                    textAnchor="middle"
                    fill="#10B981"
                    fontSize="8.5"
                    fontWeight="700"
                  >
                    T{index + 1}: GREEN / RED
                  </text>
                </g>
              )}
            </g>
          ))}

          {/* ================= LIVE AMBULANCE TRACKER FOLLOWING ALL 5 TURNS ================= */}
          {greenWaveActive && (
            <g transform={`translate(${ambulanceCoord.x}, ${ambulanceCoord.y})`}>
              <circle r="28" fill="#10B981" fillOpacity="0.24" />
              <circle r="17" fill="#EF4444" fillOpacity="0.28" />

              <g transform={`rotate(${ambulanceCoord.angle})`}>
                <rect
                  x="-15"
                  y="-8"
                  width="30"
                  height="16"
                  rx="4"
                  fill="#FFFFFF"
                  stroke="#0B0F17"
                  strokeWidth="2"
                />
                <rect x="-4.5" y="-1.5" width="9" height="3" fill="#EF4444" />
                <rect x="-1.5" y="-4.5" width="3" height="9" fill="#EF4444" />
                <rect x="7" y="-5.5" width="3.5" height="4.5" fill="#EF4444" />
                <rect x="7" y="1" width="3.5" height="4.5" fill="#3B82F6" />
              </g>

              <g transform="translate(0, -28)">
                <rect
                  x="-66"
                  y="-13"
                  width="132"
                  height="20"
                  rx="5"
                  fill="#10B981"
                  stroke="#0B0F17"
                  strokeWidth="1.5"
                />
                <text
                  x="0"
                  y="0"
                  textAnchor="middle"
                  fill="#0B0F17"
                  fontSize="9.5"
                  fontWeight="800"
                >
                  AMBULANCE · {etaSeconds}s
                </text>
              </g>
            </g>
          )}

          {/* ================= HOSPITAL ORIGIN PIN (BOTTOM-LEFT) ================= */}
          <g transform={`translate(${HOSPITAL_ORIGIN[0]}, ${HOSPITAL_ORIGIN[1]})`}>
            <circle
              r="19"
              fill="#3B82F6"
              fillOpacity="0.22"
              stroke="#3B82F6"
              strokeWidth="2"
            />
            <circle r="10.5" fill="#3B82F6" stroke="#FFFFFF" strokeWidth="2" />
            <text x="0" y="3.5" textAnchor="middle" fill="#FFFFFF" fontSize="10" fontWeight="800">
              H
            </text>
            <g transform="translate(18, 29)">
              <rect
                x="-84"
                y="-11"
                width="168"
                height="20"
                rx="4"
                fill="#0B0F17"
                stroke="#3B82F6"
                strokeWidth="1.2"
              />
              <text
                x="0"
                y="2.5"
                textAnchor="middle"
                fill="#FFFFFF"
                fontSize="10"
                fontWeight="700"
              >
                Hôpital Moulay Hassan
              </text>
            </g>
          </g>

          {/* ================= PLACE DU MÉCHOUAR CALLOUT BADGE (ABOVE THE EMPTY CIRCLE) ================= */}
          <g
            transform={`translate(${PLACE_MECHOUAR_CENTER[0]}, ${
              PLACE_MECHOUAR_CENTER[1] - PLACE_MECHOUAR_RADIUS - 24
            })`}
          >
            <rect
              x="-104"
              y="-13"
              width="208"
              height="22"
              rx="5"
              fill={greenWaveActive ? '#EF4444' : '#0B0F17'}
              stroke={greenWaveActive ? '#FFFFFF' : '#06B6D4'}
              strokeWidth="1.4"
            />
            <text
              x="0"
              y="2"
              textAnchor="middle"
              fill="#FFFFFF"
              fontSize="10.5"
              fontWeight="800"
            >
              {greenWaveActive
                ? 'FAINT ALERT: Place du Méchouar'
                : 'Place du Méchouar (Test Loc)'}
            </text>
          </g>
        </svg>

        {/* Bottom Status Legend */}
        <div className="absolute bottom-2.5 left-3 right-3 bg-[#0B0F17]/90 backdrop-blur-md border border-[#1E293B] rounded-lg px-3 py-1.5 text-xs flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-4">
            {greenWaveActive ? (
              <>
                <span className="flex items-center gap-1.5 text-[#10B981] font-semibold">
                  <ShieldCheck className="w-4 h-4" />
                  5-Turn Hospital → Place du Méchouar: GREEN ZONE
                </span>
                <span className="flex items-center gap-1.5 text-[#EF4444] font-semibold">
                  <OctagonAlert className="w-4 h-4" />
                  Opposite & Cross Routes: RED (Stopped)
                </span>
              </>
            ) : (
              <>
                <span className="flex items-center gap-1.5 text-[#CBD5E1]">
                  <Navigation className="w-3.5 h-3.5 text-[#38BDF8]" />
                  Organic City Traffic ({cars.length} Random Cars)
                </span>
                <span className="flex items-center gap-1.5 text-[#94A3B8]">
                  <MapPin className="w-3.5 h-3.5 text-[#06B6D4]" />
                  Place du Méchouar (Circular Plaza), Laâyoune
                </span>
              </>
            )}
          </div>
          <span className="text-[11px] font-mono text-[#94A3B8]">
            Scroll or click +/- to Zoom · Drag to Pan
          </span>
        </div>
      </div>

      {/* Camera Localization Bar */}
      {cameraNodes.length > 0 && (
        <div className="p-3 bg-[#0B0F17] border-t border-[#1E293B] flex flex-wrap items-center justify-between gap-2">
          <span className="text-xs text-[#94A3B8]">Active Laâyoune Camera Localization:</span>
          <div className="flex flex-wrap items-center gap-1.5">
            {cameraNodes.map((cam) => (
              <button
                key={cam.id}
                type="button"
                onClick={() => onSelectCamera?.(cam)}
                className={`px-2.5 py-1 text-xs rounded-md transition-colors whitespace-nowrap shrink-0 ${
                  activeCameraId === cam.id
                    ? 'bg-[#06B6D4] text-[#0B0F17] font-semibold'
                    : 'bg-[#161F30] text-[#CBD5E1] hover:text-white'
                }`}
              >
                {cam.code}: {cam.name.split(' (')[0]}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
