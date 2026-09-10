const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const USER_DATA_DIR = path.join(__dirname, '..', 'scratch_edge_vedas_' + Date.now());
const PORT = 9599;
const ARTIFACT_DIR = "C:\\Users\\Lakshya Valecha\\.gemini\\antigravity-ide\\brain\\13319db7-0978-4c87-9355-9ed19f3cd22c";

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
      this.ws.onerror = err => reject(err);
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
  console.log('Starting End-to-End Browser Verification for ISRO VEDAS GIS Integration...');
  const edgeProc = spawn(EDGE_PATH, [
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${USER_DATA_DIR}`,
    '--headless=new',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    'about:blank'
  ]);

  try {
    let version = null;
    for (let i = 0; i < 30; i++) {
      try {
        version = await getJson(`http://127.0.0.1:${PORT}/json/version`);
        break;
      } catch (e) {
        await sleep(200);
      }
    }

    if (!version) throw new Error('Could not connect to Edge debugging port.');

    const targets = await getJson(`http://127.0.0.1:${PORT}/json/list`);
    const pageTarget = targets.find(t => t.type === 'page');
    const client = new CDPClient(pageTarget.webSocketDebuggerUrl);
    await client.ready();

    await client.send('Page.enable');
    await client.send('DOM.enable');
    await client.send('Emulation.setDeviceMetricsOverride', {
      width: 1440,
      height: 1080,
      deviceScaleFactor: 1,
      mobile: false
    });

    // 1. Navigate to Dashboard
    console.log('Navigating to http://127.0.0.1:8000/dashboard...');
    await client.send('Page.navigate', { url: 'http://127.0.0.1:8000/dashboard' });
    
    // Wait for district boundaries to finish loading asynchronously
    console.log('Waiting for GeoJSON district boundaries to load...');
    await client.send('Runtime.evaluate', {
      expression: `
        new Promise((resolve) => {
          let count = 0;
          const check = () => {
            count++;
            if ((window.districtBoundingBoxes && window.districtBoundingBoxes.length > 0) || count > 40) {
              resolve(window.districtBoundingBoxes ? window.districtBoundingBoxes.length : 0);
            } else {
              setTimeout(check, 250);
            }
          };
          check();
        })
      `,
      awaitPromise: true
    });
    await sleep(1000);

    // 2. Trigger Map Click / Reverse Geocoding in Karbi Anglong, Assam
    console.log('Simulating Map Click at Karbi Anglong, Assam [25.8103°N, 93.4302°E]...');
    await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          performReverseGeocodeLookup(25.8103, 93.4302);
        })()
      `
    });

    // Wait for VEDAS satellite telemetry query to resolve in popup
    await sleep(2000);

    // 3. Inspect Map Popup Contents
    const popupCheck = await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          const textEl = document.getElementById('vedas-popup-terrain-text');
          const btn = document.getElementById('btn-start-analysis-here') || document.getElementById('btn-start-analysis-here-custom');
          return {
            hasPopup: !!document.querySelector('.leaflet-popup'),
            terrainText: textEl ? textEl.textContent.trim() : null,
            hasStartBtn: !!btn
          };
        })()
      `,
      returnByValue: true
    });

    console.log('Map Popup VEDAS Telemetry:', popupCheck.result.value);
    const pCheck = popupCheck.result.value;
    if (!pCheck.hasPopup) throw new Error('Leaflet popup did not open!');
    if (!pCheck.terrainText || !pCheck.terrainText.toLowerCase().includes('hilly')) {
      throw new Error(`Expected VEDAS terrain text to contain 'Hilly', got: ${pCheck.terrainText}`);
    }

    // Capture screenshot of Map with VEDAS Telemetry Popup
    console.log('Capturing screenshot of Map Popup with ISRO VEDAS Telemetry...');
    const popupShot = await client.send('Page.captureScreenshot', { format: 'png' });
    const popupShotPath = path.join(ARTIFACT_DIR, 'gis_vedas_popup_screenshot.png');
    fs.writeFileSync(popupShotPath, Buffer.from(popupShot.data, 'base64'));
    fs.writeFileSync(path.join(__dirname, 'gis_vedas_popup_screenshot.png'), Buffer.from(popupShot.data, 'base64'));
    console.log(`Saved popup screenshot to: ${popupShotPath}`);

    // 4. Click "Start New Analysis Here →"
    console.log('Clicking "Start New Analysis Here →"...');
    await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          const btn = document.getElementById('btn-start-analysis-here') || document.getElementById('btn-start-analysis-here-custom');
          if (btn) btn.click();
        })()
      `
    });

    // Wait for Predictor view, auto-fill, and VEDAS telemetry badge to render
    await sleep(2500);

    // 5. Inspect Risk Predictor Auto-Filled Inputs
    const predictorCheck = await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          const state = document.getElementById('inp-state')?.value;
          const district = document.getElementById('inp-district')?.value;
          const lat = document.getElementById('inp-latitude')?.value;
          const lon = document.getElementById('inp-longitude')?.value;
          const terrain = document.getElementById('inp-terrain')?.value;
          const terrainEl = document.getElementById('inp-terrain');
          const isAutofilled = terrainEl ? terrainEl.classList.contains('autofilled-field') : false;
          const badgeEl = document.getElementById('inp-terrain-vedas-badge');
          const badgeText = document.getElementById('inp-terrain-vedas-text')?.textContent.trim();
          const badgeVisible = badgeEl && badgeEl.style.display !== 'none';
          const activeTab = document.querySelector('.tab-button.active')?.dataset?.tab || 'unknown';

          return {
            state,
            district,
            lat,
            lon,
            terrain,
            isAutofilled,
            badgeVisible,
            badgeText,
            activeTab
          };
        })()
      `,
      returnByValue: true
    });

    console.log('Risk Predictor Auto-Filled State:');
    console.log(JSON.stringify(predictorCheck.result.value, null, 2));

    const pVals = predictorCheck.result.value;
    if (pVals.state !== 'Assam') throw new Error(`State mismatch: ${pVals.state}`);
    if (pVals.district !== 'Karbi Anglong') throw new Error(`District mismatch: ${pVals.district}`);
    if (pVals.terrain !== 'Hilly') throw new Error(`Terrain mismatch: ${pVals.terrain}`);
    if (!pVals.isAutofilled) throw new Error('Terrain element does not have .autofilled-field class!');
    if (!pVals.badgeVisible) throw new Error('inp-terrain-vedas-badge is not visible!');

    // 6. Capture screenshot of Auto-Filled Risk Predictor workbench
    console.log('Capturing screenshot of Auto-Filled Risk Predictor with VEDAS Telemetry...');
    const predictorShot = await client.send('Page.captureScreenshot', { format: 'png' });
    const predictorShotPath = path.join(ARTIFACT_DIR, 'predictor_vedas_autofilled_screenshot.png');
    fs.writeFileSync(predictorShotPath, Buffer.from(predictorShot.data, 'base64'));
    fs.writeFileSync(path.join(__dirname, 'predictor_vedas_autofilled_screenshot.png'), Buffer.from(predictorShot.data, 'base64'));
    console.log(`Saved predictor screenshot to: ${predictorShotPath}`);

    console.log('\n========================================');
    console.log('ALL ISRO VEDAS BROWSER E2E TESTS PASSED!');
    console.log('========================================');

    client.close();
  } finally {
    edgeProc.kill();
    try {
      fs.rmSync(USER_DATA_DIR, { recursive: true, force: true });
    } catch (e) {}
  }
}

run().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
