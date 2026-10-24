// ============================================================================
// GEOSPATIAL INTELLIGENCE & TELEMETRY SHARED TYPE DEFINITIONS
// ============================================================================

export type ThemeMode = 'dark' | 'light';

export type WorkspaceLayoutMode = 'split' | 'map' | 'analytics';

export type TargetClassification = 
  | 'TIER-1 CORE' 
  | 'STRUCTURAL EXT' 
  | 'SHEAR ANOMALY' 
  | 'BLIND INFERRED';

export interface DrillTarget {
  id: string;
  name: string;
  easting: number;
  northing: number;
  lat: number;
  lon: number;
  depthMeters: number;
  prospectivityScore: number;
  epistemicUncertainty: number;
  conformalInterval: [number, number];
  primaryDriver: string;
  structuralDomain: string;
  classification: TargetClassification;
}

export interface CrosshairCoordinates {
  easting: number;
  northing: number;
  depth?: number;
}

export interface AnomalyNode {
  id: string;
  gridX: number;
  gridY: number;
  depthNorm: number;
  intensity: number;
  label: string;
}

export const TARGET_DATASET: DrillTarget[] = [
  {
    id: 'TGT-YIL-01',
    name: 'Boulder-Lefroy Shear Node A',
    easting: 356400,
    northing: 6598200,
    lat: -30.745,
    lon: 121.500,
    depthMeters: 380,
    prospectivityScore: 0.942,
    epistemicUncertainty: 0.048,
    conformalInterval: [0.894, 0.978],
    primaryDriver: 'Fault Dilation & Mag AS 1VD',
    structuralDomain: 'Kalgoorlie Terrane',
    classification: 'TIER-1 CORE'
  },
  {
    id: 'TGT-YIL-02',
    name: 'Zuleika Shear Dilational Jog',
    easting: 342150,
    northing: 6612800,
    lat: -30.612,
    lon: 121.352,
    depthMeters: 520,
    prospectivityScore: 0.887,
    epistemicUncertainty: 0.071,
    conformalInterval: [0.816, 0.945],
    primaryDriver: 'Hydrothermal K/Th Radiometrics',
    structuralDomain: 'Kurnalpi Terrane',
    classification: 'STRUCTURAL EXT'
  },
  {
    id: 'TGT-YIL-03',
    name: 'Laverton Tectonic Zone Crest',
    easting: 428900,
    northing: 6834100,
    lat: -28.618,
    lon: 122.271,
    depthMeters: 290,
    prospectivityScore: 0.915,
    epistemicUncertainty: 0.052,
    conformalInterval: [0.863, 0.957],
    primaryDriver: 'Banded Iron Contact Shear',
    structuralDomain: 'Laverton Domain',
    classification: 'TIER-1 CORE'
  },
  {
    id: 'TGT-YIL-04',
    name: 'Ida Fault Basement Splay',
    easting: 298400,
    northing: 6711500,
    lat: -29.712,
    lon: 120.915,
    depthMeters: 640,
    prospectivityScore: 0.794,
    epistemicUncertainty: 0.112,
    conformalInterval: [0.682, 0.891],
    primaryDriver: 'Gravity Bouguer HGM Gradient',
    structuralDomain: 'Youanmi Terrane Margin',
    classification: 'SHEAR ANOMALY'
  },
  {
    id: 'TGT-YIL-05',
    name: 'Kurnalpi Subsurface Diatreme',
    easting: 388100,
    northing: 6649700,
    lat: -30.278,
    lon: 121.835,
    depthMeters: 410,
    prospectivityScore: 0.832,
    epistemicUncertainty: 0.089,
    conformalInterval: [0.743, 0.908],
    primaryDriver: 'Copernicus DEM Slope & Curvature',
    structuralDomain: 'Edjudina Belt',
    classification: 'BLIND INFERRED'
  }
];

export const BRAILLE_FRAMES = ['⠁', '⠃', '⠇', '⠏', '⠟', '⠿'];
export const ASCII_GLYPH_RAMP = '·:-=+*#%@◈';
