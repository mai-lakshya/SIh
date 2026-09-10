import React, { useState } from 'react';
import { RiskFactorRadarChart } from './RiskFactorRadarChart';
import { RiskFactorDataPoint } from './riskFactorUtils';

// Canonical Project Explainability Dataset: Exact same dataset consumed by both the Bar Chart and Radar Chart
export const CURRENT_PROJECT_FACTORS: RiskFactorDataPoint[] = [
  { factor: 'Protest & Agitation Risk Factor (P_r)', project: 17, benchmark: 5 },
  { factor: 'Population Density', project: -7, benchmark: -2 },
  { factor: 'State', project: 6, benchmark: 2 },
  { factor: 'District', project: -5, benchmark: -1 },
  { factor: 'Log Land Area', project: 6, benchmark: 2 },
];

export const XaiSurvivalPageExample: React.FC = () => {
  const [selectedFactor, setSelectedFactor] = useState<string | null>(null);

  const handleFactorClick = (factor: string) => {
    setSelectedFactor(factor);
    const el = document.getElementById(`bar-factor-${encodeURIComponent(factor)}`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  };

  // Find max magnitude for bar chart scaling
  const maxBarMagnitude = Math.max(...CURRENT_PROJECT_FACTORS.map((f) => Math.abs(f.project)));

  return (
    <div className="w-full max-w-7xl mx-auto p-4 sm:p-6 lg:p-8 space-y-6 bg-gray-50 min-h-screen">
      <div>
        <h1 className="text-2xl font-extrabold text-gray-900 tracking-tight">
          XAI & Survival Analysis
        </h1>
        <p className="text-sm text-gray-500 mt-1">
          Dual-paradigm explainability: point-impact delay attributions and multidimensional risk shape.
        </p>
      </div>

      {/* Two-Column Responsive Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
        {/* Left Column: Existing "Why This Risk Score" Waterfall / Tornado Bar Chart */}
        <div className="bg-white rounded-lg border border-gray-200 p-5 sm:p-6 shadow-sm flex flex-col justify-between h-full min-h-[460px]">
          <div>
            <h3 className="text-base sm:text-lg font-bold text-gray-900 tracking-tight">
              Why This Risk Score
            </h3>
            <p className="text-xs sm:text-sm text-gray-500 mt-0.5">
              SHAP feature attributions on statutory delay points
            </p>
            <div className="border-b border-gray-200 mt-3.5 mb-4" />
          </div>

          <div className="flex-1 space-y-3 py-2">
            {CURRENT_PROJECT_FACTORS.map((item) => {
              const isInc = item.project > 0;
              const widthPct = Math.round((Math.abs(item.project) / (maxBarMagnitude || 20)) * 100);

              return (
                <div
                  key={item.factor}
                  id={`bar-factor-${encodeURIComponent(item.factor)}`}
                  onClick={() => setSelectedFactor(item.factor)}
                  className={`p-2.5 rounded-md transition-all cursor-pointer border ${
                    selectedFactor === item.factor
                      ? 'bg-blue-50/70 border-blue-300 ring-2 ring-blue-400'
                      : 'border-transparent hover:bg-gray-50'
                  }`}
                >
                  <div className="flex justify-between text-xs font-semibold mb-1">
                    <span className="text-gray-800">{item.factor}</span>
                    <span className={isInc ? 'text-[#ef5350]' : 'text-[#26a69a]'}>
                      {isInc ? `+${item.project} pts` : `${item.project} pts`}
                    </span>
                  </div>
                  <div className="w-full bg-gray-100 h-2.5 rounded-full overflow-hidden flex">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        isInc ? 'bg-[#ef5350]' : 'bg-[#26a69a]'
                      }`}
                      style={{ width: `${widthPct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="border-t border-gray-100 mt-3 pt-2.5">
            <p className="text-[11px] leading-relaxed text-gray-500">
              <strong className="font-semibold text-gray-700">How to read this:</strong> Red bars represent factors driving risk upward. Green bars represent mitigating factors working in the project's favor. Both charts consume the identical raw signed point dataset.
            </p>
          </div>
        </div>

        {/* Right Column: Corrected RiskFactorRadarChart with Recentered Scale & Dynamic Magnitude */}
        <RiskFactorRadarChart
          data={CURRENT_PROJECT_FACTORS}
          sourceBarData={CURRENT_PROJECT_FACTORS}
          title="Risk Factor Profile"
          subtitle="How this project compares to the portfolio average"
          onFactorClick={handleFactorClick}
        />
      </div>
    </div>
  );
};

export default XaiSurvivalPageExample;
