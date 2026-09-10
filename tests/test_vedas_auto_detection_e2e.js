const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

// Look for Edge or Chrome executable
const candidatePaths = [
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
];

let BROWSER_PATH = candidatePaths.find(p => fs.existsSync(p));
if (!BROWSER_PATH) {
  console.error("No Edge or Chrome found in default paths.");
  process.exit(1);
}

const PORT = 9622;
const USER_DATA_DIR = path.join(__dirname, '..', 'scratch_browser_' + Date.now());

function sleep(ms) {
  return new Promise(res => setTimeout(res, ms));
}

function getJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try { resolve(JSON.parse(data)); } catch (e) { reject(e); }
      });
    }).on('error', reject);
  });
}

class CDPClient {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.id = 1;
    this.callbacks = new Map();
    this.ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id && this.callbacks.has(msg.id)) {
        const { resolve, reject } = this.callbacks.get(msg.id);
        this.callbacks.delete(msg.id);
        if (msg.error) reject(msg.error);
        else resolve(msg.result);
      }
    };
  }

  async ready() {
    if (this.ws.readyState === WebSocket.OPEN) return;
    return new Promise((resolve, reject) => {
      this.ws.onopen = () => resolve();
      this.ws.onerror = (e) => reject(e);
    });
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = this.id++;
      this.callbacks.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  close() {
    this.ws.close();
  }
}

async function run() {
  console.log('Launching headless browser at:', BROWSER_PATH);
  const browserProc = spawn(BROWSER_PATH, [
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${USER_DATA_DIR}`,
    '--headless=new',
    '--disable-gpu',
    '--no-sandbox',
    'about:blank'
  ]);

  try {
    let targets = null;
    for (let i = 0; i < 30; i++) {
      try {
        targets = await getJson(`http://127.0.0.1:${PORT}/json`);
        if (targets && targets.length > 0) break;
      } catch (e) {}
      await sleep(300);
    }

    if (!targets || targets.length === 0) {
      throw new Error('Failed to connect to browser CDP port ' + PORT);
    }

    const pageTarget = targets.find(t => t.type === 'page') || targets[0];
    const client = new CDPClient(pageTarget.webSocketDebuggerUrl);
    await client.ready();
    console.log('Connected to CDP target:', pageTarget.id);

    await client.send('Page.enable');
    await client.send('DOM.enable');
    await client.send('Runtime.enable');

    console.log('\n--- TEST 1: Navigating to http://localhost:8000/dashboard ---');
    await client.send('Page.navigate', { url: 'http://localhost:8000/dashboard' });
    await sleep(3500);

    // Check Initial VEDAS auto-detection on load
    console.log('Verifying initial VEDAS auto-detection on load...');
    const initialCheck = await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          const terrainEl = document.getElementById('inp-terrain');
          const badgeEl = document.getElementById('inp-terrain-vedas-badge');
          const badgeText = document.getElementById('inp-terrain-vedas-text')?.textContent.trim();
          const state = document.getElementById('inp-state')?.value;
          const district = document.getElementById('inp-district')?.value;
          return {
            state,
            district,
            terrain: terrainEl?.value,
            isAutofilled: terrainEl?.classList.contains('autofilled-field'),
            badgeDisplay: badgeEl?.style.display,
            badgeText
          };
        })()
      `,
      returnByValue: true
    });

    console.log('Initial State:', JSON.stringify(initialCheck.result.value, null, 2));
    const initRes = initialCheck.result.value;
    if (initRes.terrain !== 'Tribal_Schedule_V') {
      console.warn(`Initial terrain was ${initRes.terrain}, waiting another 1.5s for async VEDAS...`);
      await sleep(1500);
    }

    const recheckInit = await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          const terrainEl = document.getElementById('inp-terrain');
          const badgeEl = document.getElementById('inp-terrain-vedas-badge');
          return {
            terrain: terrainEl?.value,
            badgeDisplay: badgeEl?.style.display,
            badgeText: document.getElementById('inp-terrain-vedas-text')?.textContent.trim()
          };
        })()
      `,
      returnByValue: true
    });
    console.log('Verified Initial VEDAS Detection:', recheckInit.result.value);
    if (recheckInit.result.value.terrain !== 'Tribal_Schedule_V') {
      throw new Error(`Expected initial terrain Tribal_Schedule_V, got: ${recheckInit.result.value.terrain}`);
    }
    console.log('>>> TEST 1 PASSED: Initial load auto-detected Tribal_Schedule_V Area.\n');

    // TEST 2: Change to Himachal Pradesh -> Shimla
    console.log('--- TEST 2: Changing State/District to Himachal Pradesh -> Shimla ---');
    await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          const stateEl = document.getElementById('inp-state');
          stateEl.value = 'Himachal Pradesh';
          stateEl.dispatchEvent(new Event('change', { bubbles: true }));
          handlePredictorStateChange('Himachal Pradesh');
          const distEl = document.getElementById('inp-district');
          if (distEl) {
            distEl.value = 'Shimla';
            distEl.dispatchEvent(new Event('change', { bubbles: true }));
            onDistrictOrLocationChange();
          }
        })()
      `
    });

    await sleep(2000);

    const shimlaCheck = await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          const terrainEl = document.getElementById('inp-terrain');
          const badgeEl = document.getElementById('inp-terrain-vedas-badge');
          return {
            terrain: terrainEl?.value,
            badgeDisplay: badgeEl?.style.display,
            badgeText: document.getElementById('inp-terrain-vedas-text')?.textContent.trim()
          };
        })()
      `,
      returnByValue: true
    });
    console.log('Shimla Auto-Detection Result:', shimlaCheck.result.value);
    if (shimlaCheck.result.value.terrain !== 'Hilly') {
      throw new Error(`Expected Shimla terrain 'Hilly', got: ${shimlaCheck.result.value.terrain}`);
    }
    console.log('>>> TEST 2 PASSED: Shimla automatically classified as Hilly via CartoDEM.\n');

    // TEST 3: Coordinate Change to Mumbai Metropolitan Core
    console.log('--- TEST 3: Changing coordinates to Mumbai (19.0760, 72.8777) ---');
    await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          const latEl = document.getElementById('inp-latitude');
          const lonEl = document.getElementById('inp-longitude');
          latEl.value = '19.0760';
          lonEl.value = '72.8777';
          latEl.dispatchEvent(new Event('input', { bubbles: true }));
          lonEl.dispatchEvent(new Event('input', { bubbles: true }));
          latEl.dispatchEvent(new Event('change', { bubbles: true }));
          lonEl.dispatchEvent(new Event('change', { bubbles: true }));
        })()
      `
    });

    await sleep(2000);

    const mumbaiCheck = await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          const terrainEl = document.getElementById('inp-terrain');
          const badgeEl = document.getElementById('inp-terrain-vedas-badge');
          return {
            terrain: terrainEl?.value,
            badgeDisplay: badgeEl?.style.display,
            badgeText: document.getElementById('inp-terrain-vedas-text')?.textContent.trim()
          };
        })()
      `,
      returnByValue: true
    });
    console.log('Mumbai Auto-Detection Result:', mumbaiCheck.result.value);
    if (mumbaiCheck.result.value.terrain !== 'Urban') {
      throw new Error(`Expected Mumbai terrain 'Urban', got: ${mumbaiCheck.result.value.terrain}`);
    }
    console.log('>>> TEST 3 PASSED: Mumbai coordinates automatically classified as Urban High-Density.\n');

    // TEST 4: Map Reverse Geocode and VEDAS Telemetry Popup
    console.log('--- TEST 4: Verifying Map Popup VEDAS Telemetry ---');
    const mapPopupCheck = await client.send('Runtime.evaluate', {
      expression: `
        (async () => {
          switchMainTab('map');
          await new Promise(r => setTimeout(r, 600));
          // Trigger reverse geocode for Nainital forest region
          const res = await performReverseGeocodeLookup(29.5300, 78.7747);
          await new Promise(r => setTimeout(r, 1200));
          const popupText = document.getElementById('vedas-popup-terrain-text')?.textContent.trim();
          return {
            res,
            popupText
          };
        })()
      `,
      awaitPromise: true,
      returnByValue: true
    });
    console.log('Map Popup Result:', mapPopupCheck.result.value);
    const pText = mapPopupCheck.result.value.popupText;
    if (!pText || pText.includes('Querying')) {
      throw new Error(`Map popup did not resolve VEDAS telemetry: ${pText}`);
    }
    console.log('>>> TEST 4 PASSED: Map popup successfully displayed ISRO VEDAS satellite telemetry.\n');

    console.log('====================================================');
    console.log('ALL VEDAS AI TERRAIN AUTO-DETECTION TESTS COMPLETED!');
    console.log('====================================================');

    client.close();
  } finally {
    browserProc.kill();
    try {
      fs.rmSync(USER_DATA_DIR, { recursive: true, force: true });
    } catch (e) {}
  }
}

run().catch(err => {
  console.error('\n❌ Test execution failed:', err);
  process.exit(1);
});
