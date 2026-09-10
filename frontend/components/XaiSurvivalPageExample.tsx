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
        {/* Left Column: Corrected RiskFactorRadarChart with Recentered Scale & Dynamic Magnitude */}
        <RiskFactorRadarChart
          data={CURRENT_PROJECT_FACTORS}
          sourceBarData={CURRENT_PROJECT_FACTORS}
          title="Risk Factor Profile"
          subtitle="How this project compares to the portfolio average"
          onFactorClick={handleFactorClick}
        />

        {/* Right Column: When Is This Project Most Likely to Stall? */}
        <div className="bg-white rounded-lg border border-gray-200 p-5 sm:p-6 shadow-sm flex flex-col justify-between h-full min-h-[460px]">
          <div>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base sm:text-lg font-bold text-gray-900 tracking-tight">
                  When Is This Project Most Likely to Stall?
                </h3>
                <p className="text-xs sm:text-sm text-gray-500 mt-0.5">
                  Likelihood this project stays on track over the next 2 years
                </p>
              </div>
              <span className="px-2.5 py-1 text-[11px] font-semibold text-amber-800 bg-amber-50 border border-amber-200 rounded-md">
                Critical Window
              </span>
            </div>
            <div className="border-b border-gray-200 mt-3.5 mb-4" />
          </div>

          <div className="mb-3 px-3.5 py-2.5 bg-blue-50/70 border-l-4 border-blue-500 rounded-md text-xs text-gray-700 leading-relaxed">
            <strong className="text-blue-700">Survival Dynamics:</strong> The steepest drop shows when this project is most likely to face major delays.
          </div>

          <div className="flex-1 flex items-center justify-center p-4">
            <svg viewBox="0 0 500 240" className="w-full h-auto max-h-[260px]">
              <defs>
                <linearGradient id="survivalGradComp" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#0284c7" stopOpacity="0.3" />
                  <stop offset="100%" stopColor="#0284c7" stopOpacity="0.02" />
                </linearGradient>
              </defs>
              <line x1="40" y1="20" x2="480" y2="20" stroke="#f1f5f9" strokeDasharray="3 3" />
              <line x1="40" y1="70" x2="480" y2="70" stroke="#f1f5f9" strokeDasharray="3 3" />
              <line x1="40" y1="120" x2="480" y2="120" stroke="#f1f5f9" strokeDasharray="3 3" />
              <line x1="40" y1="170" x2="480" y2="170" stroke="#f1f5f9" strokeDasharray="3 3" />
              
              <text x="32" y="24" textAnchor="end" fontSize="10" fill="#94a3b8">100%</text>
              <text x="32" y="74" textAnchor="end" fontSize="10" fill="#94a3b8">75%</text>
              <text x="32" y="124" textAnchor="end" fontSize="10" fill="#94a3b8">50%</text>
              <text x="32" y="174" textAnchor="end" fontSize="10" fill="#94a3b8">25%</text>

              <rect x="130" y="20" width="90" height="180" fill="rgba(239, 68, 68, 0.08)" stroke="rgba(239, 68, 68, 0.2)" strokeDasharray="2 2" />
              <text x="175" y="36" textAnchor="middle" fontSize="9" fontWeight="bold" fill="#dc2626">High Stall Danger</text>

              <path d="M 40 20 L 80 28 L 130 46 L 175 90 L 220 135 L 300 162 L 390 185 L 480 195 L 480 200 L 40 200 Z" fill="url(#survivalGradComp)" />
              <polyline fill="none" stroke="#0284c7" strokeWidth="2.5" points="40,20 80,28 130,46 175,90 220,135 300,162 390,185 480,195" />

              <text x="40" y="216" textAnchor="middle" fontSize="9.5" fill="#64748b">Day 0</text>
              <text x="130" y="216" textAnchor="middle" fontSize="9.5" fill="#64748b">3 Mo</text>
              <text x="220" y="216" textAnchor="middle" fontSize="9.5" fill="#64748b">8 Mo</text>
              <text x="340" y="216" textAnchor="middle" fontSize="9.5" fill="#64748b">1 Yr</text>
              <text x="480" y="216" textAnchor="middle" fontSize="9.5" fill="#64748b">2 Yrs</text>
            </svg>
          </div>

          <div className="border-t border-gray-100 mt-3 pt-2.5">
            <p className="text-[11px] leading-relaxed text-gray-500">
              <strong className="font-semibold text-gray-700">Survival Dynamics:</strong> The steepest drop shows when this project is most likely to face major delays.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default XaiSurvivalPageExample;
