// Simulate DOM environment to test startNewAnalysisFromMap and live evaluation
const fs = require('fs');
const http = require('http');

// Mock DOM elements
const elements = {
  'inp-latitude': { value: '', dispatchEvent: () => {} },
  'inp-longitude': { value: '', dispatchEvent: () => {} },
  'inp-state': { value: '', options: [{ value: 'Rajasthan' }, { value: 'Maharashtra' }, { value: 'Andhra Pradesh' }] },
  'inp-district': { value: '' },
  'inp-road-type': { value: '' },
  'inp-site-address': { value: '' },
  'inp-project-type': { value: 'Highway' },
  'inp-terrain': { value: 'plain' },
  'inp-project-id': { value: '' },
  'inp-cost': { value: '' },
  'inp-land-area': { value: '' },
  'inp-sia-status': { value: '' },
  'inp-forest-status': { value: '' },
  'inp-fund-pct': { value: '' },
  'inp-sec11-days': { value: '' },
  'inp-comp-mult': { value: '' },
  'inp-affected-families': { value: '' },
  'inp-dispute-rate': { value: '' },
  'inp-protest-flag': { checked: false },
  'preset-selector': { value: '' },
  'out-prob': { textContent: '', style: {} },
  'out-crs': { innerHTML: '', style: {} },
  'out-delay-days': { textContent: '', style: {} },
  'out-median-surv': { textContent: '', style: {} },
  'out-confidence-tag': { textContent: '' },
  'out-crs-ci': { textContent: '' },
  'out-delay-ci': { textContent: '' },
  'out-surv-human': { textContent: '' },
  'output-badge-tier': { className: '', textContent: '', style: {} },
  'out-larr-badge': { className: '', textContent: '', style: {} },
  'out-larr-desc': { textContent: '' },
  'remoteness-analysis-section': { style: {} },
  'remoteness-tier-badge': { textContent: '', style: {} },
  'rem-settlement-name': { textContent: '' },
  'rem-settlement-tier': { textContent: '' },
  'rem-distance-km': { textContent: '' },
  'rem-road-type': { textContent: '' },
  'rem-delay-days': { textContent: '', style: {} },
  'rem-score-pct': { textContent: '' },
  'rem-score-bar': { style: {} },
  'rem-bar-label': { textContent: '' },
  'rem-breakdown-details': { innerHTML: '' },
  'view-predictor': { scrollIntoView: () => {} }
};

global.document = {
  getElementById: (id) => elements[id] || null
};
global.window = {
  referenceCoordinatesMap: {}
};
global.Event = class { constructor(type) { this.type = type; } };
global.reverseGeocodePopup = null;
global.map = null;
global.remotenessDebounceTimer = null;
global.switchMainTab = (tab) => console.log(`[Tab] Switched to: ${tab}`);
global.closeInspectorDrawer = () => {};
global.updateSec11LapseBadge = () => {};
global.showToast = (msg) => console.log(`[Toast]: ${msg}`);
global.setPredictorStateAndDistrict = (s, d) => {
  elements['inp-state'].value = s;
  elements['inp-district'].value = d;
};
global.getAuthHeaders = () => ({ "X-API-Key": "super-secret-token" });
global.getApiBase = () => "http://127.0.0.1:8000";
global.escapeHtml = (s) => String(s);

// Load the updated script from dashboard/index.html
const html = fs.readFileSync('dashboard/index.html', 'utf-8');

// Extract startNewAnalysisFromMap
const fnStartMatch = html.match(/function startNewAnalysisFromMap\(state, district, lat, lng\)[\s\S]*?\n    }/);
if (!fnStartMatch) throw new Error("Could not find startNewAnalysisFromMap in HTML");
eval(fnStartMatch[0]);

// Extract triggerLiveRemotenessEvaluation
const fnEvalMatch = html.match(/async function triggerLiveRemotenessEvaluation\(\)[\s\S]*?\n    }/);
if (!fnEvalMatch) throw new Error("Could not find triggerLiveRemotenessEvaluation in HTML");
eval(fnEvalMatch[0]);

// Extract renderRemotenessAnalysis
const fnRendMatch = html.match(/function renderRemotenessAnalysis\(remoteness, fullData\)[\s\S]*?\n    }/);
if (!fnRendMatch) throw new Error("Could not find renderRemotenessAnalysis in HTML");
eval(fnRendMatch[0]);

// Test execution: Click on Map at Guntur Coordinates (16.2904°N, 80.4542°E)
console.log("TEST 1: Simulating map click button 'Start New Analysis Here' for Guntur [16.2904, 80.4542]...");
startNewAnalysisFromMap('Andhra Pradesh', 'Guntur', 16.2904, 80.4542);

console.log("Latitude field filled:", elements['inp-latitude'].value);
console.log("Longitude field filled:", elements['inp-longitude'].value);
console.log("State field filled:", elements['inp-state'].value);
console.log("District field filled:", elements['inp-district'].value);

if (elements['inp-latitude'].value !== '16.2904') throw new Error(`Latitude expected 16.2904 but got ${elements['inp-latitude'].value}`);
if (elements['inp-longitude'].value !== '80.4542') throw new Error(`Longitude expected 80.4542 but got ${elements['inp-longitude'].value}`);

// Wait for live remoteness evaluation debounce timer (200ms)
setTimeout(() => {
  console.log("\nTEST 2: Checking Remoteness Diagnostic Card populated from live backend...");
  console.log("Nearest Settlement:", elements['rem-settlement-name'].textContent);
  console.log("Distance:", elements['rem-distance-km'].textContent);
  console.log("Delay Days:", elements['rem-delay-days'].textContent);
  console.log("Score:", elements['rem-score-pct'].textContent);

  if (!elements['rem-settlement-name'].textContent) {
    throw new Error("Expected settlement name to be populated!");
  }
  console.log("\nALL CHECKS PASSED: Coordinates are accurately auto-updated and the Remoteness Diagnostic Card updates immediately!");
  process.exit(0);
}, 600);
