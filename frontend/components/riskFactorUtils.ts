/**
 * Shared utilities for XAI Risk Factor Attribution and Radar Chart Scaling.
 * Ensures consistent signed directional logic and exact point-magnitude parity
 * between the Tornado/Waterfall bar chart and the Multi-Dimensional Radar Chart.
 */

export interface RiskFactorDataPoint {
  factor: string;
  project: number;     // Raw signed impact points (e.g. -7, +17)
  benchmark: number;   // Raw signed impact points (e.g. -2, +5)
}

export interface TransformedRadarPoint extends RiskFactorDataPoint {
  radialProject: number;    // Transformed 0-100 radial scale
  radialBenchmark: number;  // Transformed 0-100 radial scale
  projectColor: string;     // #ef5350 (red) or #26a69a (green)
  benchmarkColor: string;
}

export type BarChartDataSource =
  | Array<{ factor?: string; feature?: string; feature_label?: string; name?: string; pts?: number; impact?: number; val?: number; project?: number }>
  | Record<string, number>;

/**
 * Calculates the maximum absolute impact across all factors dynamically each render:
 * `maxAbsImpact = Math.ceil(Math.max(...data.flatMap(d => [Math.abs(d.project), Math.abs(d.benchmark)])) / 5) * 5`
 * Rounded up to a clean multiple of 5 (e.g., 10, 15, 20, 25) so the outer ring label
 * reflects the true largest value in the current dataset (e.g. +20 pts when max is +17).
 */
export function computeMaxAbsImpact(data: RiskFactorDataPoint[], minLimit = 5): number {
  if (!data || data.length === 0) return minLimit;

  // Flatten all project and benchmark magnitudes
  const values = data.flatMap((d) => [Math.abs(d.project || 0), Math.abs(d.benchmark || 0)]);
  const maxObserved = Math.max(...values, 0);

  if (maxObserved <= 0) return minLimit;

  // Dynamic ceiling to nearest multiple of 5
  return Math.ceil(maxObserved / 5) * 5;
}

/**
 * Transforms a raw signed impact value (-maxAbs to +maxAbs) into a 0-100 radial coordinate:
 *   - Neutral (0 points)           -> 50 (mid-ring)
 *   - Max Risk-Increasing (+maxAbs) -> 100 (outer ring)
 *   - Max Risk-Reducing   (-maxAbs) -> 0   (center)
 *
 * Formula: radialValue = ((impact - min) / (max - min)) * 100
 * where min = -maxAbsImpact, max = +maxAbsImpact
 */
export function toRadialScale(impact: number, maxAbsImpact: number): number {
  if (maxAbsImpact <= 0) return 50;
  const radial = 50 + (impact / maxAbsImpact) * 50;
  return Math.max(0, Math.min(100, Math.round(radial * 10) / 10));
}

/**
 * Validates directional fidelity between the raw signed impact and the 0-100 radial scale.
 * Logs a console warning if sign and neutral-ring crossing disagree.
 */
export function validateRadialDirection(rawImpact: number, radialValue: number, factorName: string): boolean {
  if (process.env.NODE_ENV !== 'production') {
    const isPositive = rawImpact > 0;
    const isNegative = rawImpact < 0;
    const isOuter = radialValue > 50.01;
    const isInner = radialValue < 49.99;

    if (isPositive && !isOuter) {
      console.warn(
        `[Radar Guardrail] Factor "${factorName}" has positive impact (+${rawImpact}) but radial value is <= 50 (${radialValue}).`
      );
      return false;
    }
    if (isNegative && !isInner) {
      console.warn(
        `[Radar Guardrail] Factor "${factorName}" has negative impact (${rawImpact}) but radial value is >= 50 (${radialValue}).`
      );
      return false;
    }
  }
  return true;
}

/**
 * Dev-mode data consistency guardrail:
 * Compares radar factor values directly against the reference bar chart source dataset.
 * If any factor differs by > tolerance (default 0.5 points), throws a visible console error:
 * "RiskFactorRadarChart data mismatch: {factor} shows {radarValue} but source data has {trueValue}"
 */
export function assertDataConsistencyWithBarChart(
  radarData: RiskFactorDataPoint[],
  barSource?: BarChartDataSource | null,
  tolerance = 0.5
): void {
  if (process.env.NODE_ENV === 'production' || !barSource || !radarData) return;

  const barMap = new Map<string, number>();

  if (Array.isArray(barSource)) {
    for (const item of barSource) {
      const name = item.factor || item.feature_label || item.feature || item.name;
      const val = item.pts !== undefined
        ? item.pts
        : (item.impact !== undefined
          ? item.impact
          : (item.project !== undefined ? item.project : item.val));
      if (name && typeof val === 'number') {
        barMap.set(name.trim().toLowerCase(), val);
      }
    }
  } else if (typeof barSource === 'object') {
    for (const [key, val] of Object.entries(barSource)) {
      if (typeof val === 'number') {
        barMap.set(key.trim().toLowerCase(), val);
      }
    }
  }

  if (barMap.size === 0) return;

  for (const item of radarData) {
    const key = item.factor.trim().toLowerCase();
    if (barMap.has(key)) {
      const trueValue = barMap.get(key)!;
      const radarValue = item.project;
      if (Math.abs(radarValue - trueValue) > tolerance) {
        console.error(
          `RiskFactorRadarChart data mismatch: ${item.factor} shows ${radarValue} but source data has ${trueValue}`
        );
      }
    }
  }
}

/**
 * Identifies the factor with the largest POSITIVE impact (top risk driver).
 * Specifically requires project > 0; never selects based on magnitude alone.
 */
export function getTopRiskDriver(data: RiskFactorDataPoint[]): RiskFactorDataPoint | null {
  if (!data || data.length === 0) return null;

  let topDriver: RiskFactorDataPoint | null = null;
  let maxPos = 0;

  for (const item of data) {
    if (item.project > maxPos) {
      maxPos = item.project;
      topDriver = item;
    }
  }

  return topDriver;
}

/**
 * Identifies the factor with the largest NEGATIVE impact (top protective mitigator).
 * Specifically requires project < 0; never selects based on magnitude alone.
 */
export function getTopMitigator(data: RiskFactorDataPoint[]): RiskFactorDataPoint | null {
  if (!data || data.length === 0) return null;

  let topMitigator: RiskFactorDataPoint | null = null;
  let maxNeg = 0;

  for (const item of data) {
    if (item.project < maxNeg) {
      maxNeg = item.project;
      topMitigator = item;
    }
  }

  return topMitigator;
}

/**
 * Formats a signed value with explicit + or - and units.
 */
export function formatSignedPoints(val: number): string {
  const rounded = Math.round(val);
  if (rounded > 0) return `+${rounded} pts`;
  if (rounded < 0) return `${rounded} pts`;
  return `0 pts`;
}

/**
 * Color and style tokens based on impact sign:
 *   - Positive (risk-increasing) -> Red #ef5350
 *   - Negative (risk-reducing)   -> Green #26a69a
 *   - Zero (neutral)             -> Gray #6b7280
 */
export function getFactorColorTokens(val: number) {
  if (val > 0) {
    return {
      color: '#ef5350',
      fill: 'rgba(239, 83, 80, 0.2)',
      label: 'risk-increasing',
      badgeClass: 'text-red-700 bg-red-50 border-red-200',
    };
  } else if (val < 0) {
    return {
      color: '#26a69a',
      fill: 'rgba(38, 166, 154, 0.2)',
      label: 'risk-reducing',
      badgeClass: 'text-teal-800 bg-teal-50 border-teal-200',
    };
  } else {
    return {
      color: '#6b7280',
      fill: 'rgba(107, 114, 128, 0.15)',
      label: 'neutral',
      badgeClass: 'text-gray-700 bg-gray-50 border-gray-200',
    };
  }
}

/**
 * Generates the unified executive finding matching the exact requirements.
 * Example:
 * "This project's risk is being pushed up most by Protest & Agitation Risk Factor (P_r) (+17 pts) — while Population Density is working in its favor (-7 pts)."
 */
export function generateExecutiveFinding(data: RiskFactorDataPoint[]): string | null {
  if (!data || data.length === 0) return null;

  const topDriver = getTopRiskDriver(data);
  const topMitigator = getTopMitigator(data);

  if (topDriver && topMitigator) {
    return `This project's risk is being pushed up most by ${topDriver.factor} (+${Math.round(
      topDriver.project
    )} pts) — while ${topMitigator.factor} is working in its favor (${Math.round(topMitigator.project)} pts).`;
  } else if (topDriver) {
    return `This project's risk is being pushed up most by ${topDriver.factor} (+${Math.round(
      topDriver.project
    )} pts), with no major protective mitigators detected.`;
  } else if (topMitigator) {
    return `All evaluated dimensions are protective or neutral, led by ${topMitigator.factor} (${Math.round(
      topMitigator.project
    )} pts).`;
  }

  return "This project's factor attribution profile shows neutral impact across all evaluated dimensions.";
}
