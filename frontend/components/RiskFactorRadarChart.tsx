import React, { useState, useMemo, useEffect } from 'react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import {
  RiskFactorDataPoint,
  BarChartDataSource,
  computeMaxAbsImpact,
  toRadialScale,
  formatSignedPoints,
  getFactorColorTokens,
  generateExecutiveFinding,
  validateRadialDirection,
  assertDataConsistencyWithBarChart,
} from './riskFactorUtils';

export interface RiskFactorRadarChartProps {
  data: RiskFactorDataPoint[];
  sourceBarData?: BarChartDataSource; // Optional source data reference from the adjacent bar chart for validation
  title?: string;
  subtitle?: string;
  onFactorClick?: (factor: string) => void;
  isLoading?: boolean;
  error?: string;
  onRetry?: () => void;
  className?: string;
}

/**
 * Custom Tooltip displaying ORIGINAL signed values and directional semantics.
 * Never exposes the internal 0-100 radial transform.
 */
interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    name: string;
    value: number;
    payload: {
      factor: string;
      originalProject: number;
      originalBenchmark: number;
      radialProject: number;
      radialBenchmark: number;
    };
  }>;
}

const CustomTooltip: React.FC<CustomTooltipProps> = ({ active, payload }) => {
  if (!active || !payload || !payload.length) return null;

  const dataPoint = payload[0]?.payload;
  if (!dataPoint) return null;

  const projVal = dataPoint.originalProject;
  const benchVal = dataPoint.originalBenchmark;
  const delta = projVal - benchVal;

  const projTokens = getFactorColorTokens(projVal);
  const benchTokens = getFactorColorTokens(benchVal);

  return (
    <div className="bg-white p-3.5 rounded-lg shadow-xl border border-gray-100 text-xs min-w-[220px] pointer-events-none z-50">
      <div className="font-semibold text-gray-900 border-b border-gray-100 pb-1.5 mb-2">
        {dataPoint.factor}
      </div>

      <div className="space-y-1.5">
        {/* Project Impact */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-gray-600">
            <span
              className="w-2.5 h-2.5 rounded-full inline-block"
              style={{ backgroundColor: projTokens.color }}
            />
            <span>This Project:</span>
          </div>
          <span className="font-bold" style={{ color: projTokens.color }}>
            {formatSignedPoints(projVal)}
          </span>
        </div>
        <div className="text-[10px] text-gray-400 pl-4 -mt-0.5">
          ({projTokens.label === 'risk-increasing' ? 'pushes risk up' : projTokens.label === 'risk-reducing' ? 'works in project favor' : 'neutral'})
        </div>

        {/* Portfolio Average Impact */}
        <div className="flex items-center justify-between pt-1">
          <div className="flex items-center gap-1.5 text-gray-600">
            <span
              className="w-2.5 h-2.5 rounded-full border border-dashed inline-block"
              style={{ borderColor: benchTokens.color, backgroundColor: benchTokens.fill }}
            />
            <span>Portfolio Average:</span>
          </div>
          <span className="font-medium text-gray-700">
            {formatSignedPoints(benchVal)}
          </span>
        </div>
      </div>

      {/* Variance vs Benchmark */}
      <div className="mt-2.5 pt-2 border-t border-gray-100 flex items-center justify-between">
        <span className="text-gray-500 font-medium">Variance vs Avg:</span>
        <span
          className={`font-semibold px-1.5 py-0.5 rounded text-[11px] ${
            delta > 0
              ? 'text-red-700 bg-red-50 border border-red-200'
              : delta < 0
              ? 'text-teal-800 bg-teal-50 border border-teal-200'
              : 'text-gray-600 bg-gray-50 border border-gray-200'
          }`}
        >
          {delta > 0 ? `+${Math.round(delta)} pts (higher risk)` : delta < 0 ? `${Math.round(delta)} pts (more protective)` : 'Equal to avg'}
        </span>
      </div>
    </div>
  );
};

/**
 * Custom Vertex Dot renderer encoding factor SIGN:
 *   - Positive (risk-increasing): Red #ef5350
 *   - Negative (risk-reducing): Green #26a69a
 *   - Neutral (0): Gray #6b7280
 */
interface CustomDotProps {
  cx?: number;
  cy?: number;
  payload?: {
    factor: string;
    originalProject: number;
    originalBenchmark: number;
  };
  isHovered?: boolean;
  onFactorClick?: (factor: string) => void;
  seriesType: 'project' | 'benchmark';
}

const CustomVertexDot: React.FC<CustomDotProps> = ({
  cx = 0,
  cy = 0,
  payload,
  isHovered = false,
  onFactorClick,
  seriesType,
}) => {
  if (!payload) return null;

  const value = seriesType === 'project' ? payload.originalProject : payload.originalBenchmark;
  const isPositive = value > 0;
  const isNegative = value < 0;

  const dotColor = isPositive ? '#ef5350' : isNegative ? '#26a69a' : '#6b7280';
  const radius = isHovered ? (seriesType === 'project' ? 7 : 5) : (seriesType === 'project' ? 5 : 3.5);

  return (
    <g className={onFactorClick ? 'cursor-pointer' : ''} onClick={() => onFactorClick && onFactorClick(payload.factor)}>
      {seriesType === 'project' ? (
        <>
          {/* Outer glow ring on hover */}
          {isHovered && (
            <circle cx={cx} cy={cy} r={radius + 4} fill={dotColor} fillOpacity={0.25} />
          )}
          <circle
            cx={cx}
            cy={cy}
            r={radius}
            fill={dotColor}
            stroke="#ffffff"
            strokeWidth={isHovered ? 2.5 : 2}
            className="transition-all duration-150"
          />
        </>
      ) : (
        <circle
          cx={cx}
          cy={cy}
          r={radius}
          fill="#ffffff"
          stroke={dotColor}
          strokeWidth={1.5}
          strokeDasharray="2 2"
          className="transition-all duration-150"
        />
      )}
    </g>
  );
};

/**
 * Custom Angle Axis Tick supporting responsive truncation and hover highlight
 */
interface CustomAngleTickProps {
  payload: { value: string };
  x: number;
  y: number;
  cx: number;
  cy: number;
  hoveredFactor: string | null;
  onFactorClick?: (factor: string) => void;
  onHover: (factor: string | null) => void;
}

const CustomAngleTick: React.FC<CustomAngleTickProps> = ({
  payload,
  x,
  y,
  cx,
  cy,
  hoveredFactor,
  onFactorClick,
  onHover,
}) => {
  const factor = payload.value;
  const isHovered = hoveredFactor === factor;

  const maxChars = 20;
  const isTruncated = factor.length > maxChars;
  const displayText = isTruncated ? `${factor.slice(0, maxChars - 2)}…` : factor;

  const dx = x > cx ? 6 : x < cx ? -6 : 0;
  const dy = y > cy ? 8 : y < cy ? -8 : 0;
  const textAnchor = x > cx + 10 ? 'start' : x < cx - 10 ? 'end' : 'middle';

  return (
    <g
      className={onFactorClick ? 'cursor-pointer' : ''}
      onClick={() => onFactorClick && onFactorClick(factor)}
      onMouseEnter={() => onHover(factor)}
      onMouseLeave={() => onHover(null)}
      tabIndex={onFactorClick ? 0 : -1}
      role={onFactorClick ? 'button' : undefined}
      aria-label={`Factor: ${factor}`}
    >
      <title>{factor}</title>
      <text
        x={x + dx}
        y={y + dy}
        textAnchor={textAnchor}
        className={`select-none text-[11px] sm:text-[12px] font-sans transition-colors ${
          isHovered
            ? 'fill-gray-900 font-bold underline decoration-[#ef5350] underline-offset-2'
            : 'fill-gray-600 font-medium'
        }`}
      >
        {displayText}
      </text>
    </g>
  );
};

export const RiskFactorRadarChart: React.FC<RiskFactorRadarChartProps> = ({
  data,
  sourceBarData,
  title = 'Risk Factor Profile',
  subtitle = 'How this project compares to the portfolio average',
  onFactorClick,
  isLoading = false,
  error,
  onRetry,
  className = '',
}) => {
  const [hoveredFactor, setHoveredFactor] = useState<string | null>(null);

  // 1. Calculate dynamic symmetric impact limit each render
  // maxAbsImpact = Math.ceil(Math.max(...data.flatMap(d => [Math.abs(d.project), Math.abs(d.benchmark)])) / 5) * 5
  const maxAbsImpact = useMemo(() => computeMaxAbsImpact(data, 5), [data]);

  // Dev-mode data consistency guardrail:
  // Assert that radar factor values numerically agree with the bar chart source dataset
  useEffect(() => {
    const barRef = sourceBarData || (typeof window !== 'undefined' ? (window as any).__BAR_CHART_SOURCE_DATA__ : null);
    if (barRef) {
      assertDataConsistencyWithBarChart(data, barRef, 0.5);
    }
  }, [data, sourceBarData]);

  // 2. Transform raw signed points into 0-100 radial scale
  const transformedData = useMemo(() => {
    if (!data || data.length === 0) return [];

    return data.map((item) => {
      const radialProject = toRadialScale(item.project, maxAbsImpact);
      const radialBenchmark = toRadialScale(item.benchmark, maxAbsImpact);

      // Dev-mode directional assertion
      validateRadialDirection(item.project, radialProject, item.factor);

      return {
        factor: item.factor,
        originalProject: item.project,
        originalBenchmark: item.benchmark,
        radialProject,
        radialBenchmark,
      };
    });
  }, [data, maxAbsImpact]);

  // 3. Auto-generate executive finding callout using signed driver and mitigator
  const executiveSummary = useMemo(() => generateExecutiveFinding(data), [data]);

  // Loading State
  if (isLoading) {
    return (
      <div
        className={`bg-white rounded-lg border border-gray-200 p-5 sm:p-6 shadow-sm flex flex-col justify-between h-full min-h-[460px] animate-pulse ${className}`}
        aria-busy="true"
        aria-label="Loading risk factor profile..."
      >
        <div>
          <div className="h-6 w-48 bg-gray-200 rounded mb-2" />
          <div className="h-4 w-72 bg-gray-100 rounded mb-4" />
          <div className="border-b border-gray-200 my-4" />
          <div className="h-10 w-full bg-blue-50 rounded mb-6" />
        </div>
        <div className="flex-1 flex items-center justify-center">
          <div className="w-56 h-56 rounded-full border-4 border-dashed border-gray-200 animate-spin" />
        </div>
        <div className="h-4 w-56 bg-gray-100 rounded mx-auto" />
      </div>
    );
  }

  // Error State
  if (error) {
    return (
      <div
        className={`bg-white rounded-lg border border-gray-200 p-5 sm:p-6 shadow-sm flex flex-col justify-between h-full min-h-[460px] ${className}`}
        role="alert"
      >
        <div>
          <h3 className="text-base font-bold text-gray-900">{title}</h3>
          <div className="border-b border-gray-200 my-4" />
        </div>
        <div className="flex-1 flex flex-col items-center justify-center text-center p-6 bg-red-50 rounded-lg">
          <p className="text-sm font-semibold text-red-800">Failed to load risk profile</p>
          <p className="text-xs text-red-600 mt-1 mb-4">{error}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="px-3.5 py-1.5 text-xs font-semibold text-white bg-red-600 hover:bg-red-700 rounded-md transition-colors shadow-sm"
            >
              Retry
            </button>
          )}
        </div>
        <div className="text-[11px] text-gray-400 text-center">Contact support if this persists.</div>
      </div>
    );
  }

  // Empty Data State
  if (!data || data.length < 3) {
    return (
      <div
        className={`bg-white rounded-lg border border-gray-200 p-5 sm:p-6 shadow-sm flex flex-col justify-between h-full min-h-[460px] ${className}`}
      >
        <div>
          <h3 className="text-base font-bold text-gray-900">{title}</h3>
          {subtitle && <p className="text-xs text-gray-500 mt-0.5">{subtitle}</p>}
          <div className="border-b border-gray-200 my-4" />
        </div>
        <div className="flex-1 flex flex-col items-center justify-center text-center p-8 text-gray-500 text-sm">
          Not enough factor data to display the risk profile (minimum 3 dimensions required).
        </div>
      </div>
    );
  }

  return (
    <div
      className={`bg-white rounded-lg border border-gray-200 p-5 sm:p-6 shadow-sm flex flex-col justify-between h-full min-h-[460px] ${className}`}
      role="region"
      aria-label={executiveSummary || title}
    >
      {/* 1. Header Block */}
      <div>
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-base sm:text-lg font-bold text-gray-900 tracking-tight">{title}</h3>
            {subtitle && <p className="text-xs sm:text-sm text-gray-500 mt-0.5">{subtitle}</p>}
          </div>
        </div>

        <div className="border-b border-gray-200 mt-3.5 mb-3" />

        {/* 2. Executive Finding Callout Pill */}
        {executiveSummary && (
          <div className="bg-blue-50/90 border-l-4 border-blue-600 rounded-r-md px-3.5 py-2 mb-3 text-xs sm:text-[13px] text-blue-900 font-medium flex items-center gap-2">
            <svg className="w-4 h-4 text-blue-600 shrink-0" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                clipRule="evenodd"
              />
            </svg>
            <span>
              <strong className="font-semibold">Executive Finding:</strong> {executiveSummary}
            </span>
          </div>
        )}
      </div>

      {/* 3. Recharts Radar Chart with Recentered Scale */}
      <div className="flex-1 w-full min-h-[340px] relative">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart
            data={transformedData}
            cx="50%"
            cy="50%"
            outerRadius="72%"
            margin={{ top: 12, right: 32, bottom: 12, left: 32 }}
          >
            {/* Concentric Grid Rings */}
            <PolarGrid stroke="#e5e7eb" strokeDasharray="2 2" />

            {/* Custom Angular Ticks */}
            <PolarAngleAxis
              dataKey="factor"
              tick={(props) => (
                <CustomAngleTick
                  {...props}
                  hoveredFactor={hoveredFactor}
                  onFactorClick={onFactorClick}
                  onHover={setHoveredFactor}
                />
              )}
            />

            {/* Radial Scale Axis (0 to 100 with 50 as 0/Neutral) */}
            <PolarRadiusAxis
              angle={90}
              domain={[0, 100]}
              ticks={[0, 50, 100]}
              stroke="#94a3b8"
              orientation="middle"
              tick={(props) => {
                const { payload, x, y } = props;
                let label = '';
                if (payload.value === 100) label = `+${maxAbsImpact} pts (higher risk)`;
                else if (payload.value === 50) label = `0 pts (neutral)`;
                else if (payload.value === 0) label = `-${maxAbsImpact} pts (lower risk)`;

                return (
                  <text
                    x={x + 4}
                    y={y - 4}
                    className={`font-sans text-[9px] select-none ${
                      payload.value === 50 ? 'fill-gray-700 font-bold' : 'fill-gray-400 font-medium'
                    }`}
                  >
                    {label}
                  </text>
                );
              }}
            />

            <Tooltip content={<CustomTooltip />} />

            {/* Portfolio Average Polygon: Dashed Outline */}
            <Radar
              name="Portfolio Average"
              dataKey="radialBenchmark"
              stroke="#0d9488"
              strokeWidth={1.75}
              strokeDasharray="4 4"
              fill="rgba(13, 148, 136, 0.05)"
              dot={(props) => (
                <CustomVertexDot
                  {...props}
                  seriesType="benchmark"
                  onFactorClick={onFactorClick}
                />
              )}
              isAnimationActive={true}
              animationDuration={500}
              animationEasing="ease-out"
            />

            {/* This Project Polygon: Neutral Gray Outline + Sign-Encoded Red/Green Dots */}
            <Radar
              name="This Project"
              dataKey="radialProject"
              stroke="#4b5563"
              strokeWidth={2}
              fill="rgba(75, 85, 99, 0.06)"
              dot={(props) => (
                <CustomVertexDot
                  {...props}
                  seriesType="project"
                  isHovered={hoveredFactor === props.payload?.factor}
                  onFactorClick={onFactorClick}
                />
              )}
              isAnimationActive={true}
              animationDuration={500}
              animationEasing="ease-out"
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* 4. Centered Legend & Sign Decoding */}
      <div className="flex flex-wrap items-center justify-center gap-5 pt-2 pb-1 text-xs text-gray-600">
        <div className="flex items-center gap-1.5 select-none">
          <span className="w-2.5 h-2.5 rounded-full bg-[#ef5350] inline-block" />
          <span className="font-semibold text-gray-800">Risk-Increasing (+)</span>
        </div>

        <div className="flex items-center gap-1.5 select-none">
          <span className="w-2.5 h-2.5 rounded-full bg-[#26a69a] inline-block" />
          <span className="font-semibold text-gray-800">Risk-Reducing (-)</span>
        </div>

        <div className="flex items-center gap-1.5 select-none pl-2 border-l border-gray-200">
          <span className="w-3.5 h-0.5 bg-[#4b5563] inline-block" />
          <span className="font-medium text-gray-700">This Project</span>
        </div>

        <div className="flex items-center gap-1.5 select-none">
          <span className="w-3.5 border-t-2 border-dashed border-[#0d9488] inline-block" />
          <span className="font-medium text-gray-700">Portfolio Average</span>
        </div>
      </div>

      {/* 5. How to Read This Footer */}
      <div className="border-t border-gray-100 mt-2.5 pt-2.5">
        <p className="text-[11px] leading-relaxed text-gray-500 font-normal">
          <strong className="font-semibold text-gray-700">How to read this:</strong> Points outside the dashed neutral ring are pushing this project's risk up (red); points inside it are pulling risk down (green). The further from the neutral ring, the bigger the effect. Compare the solid line (this project) to the dashed line (portfolio average) to see where this project deviates from typical.
        </p>
      </div>
    </div>
  );
};

export default RiskFactorRadarChart;
