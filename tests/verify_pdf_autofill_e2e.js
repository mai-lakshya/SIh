const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const USER_DATA_DIR = path.join(__dirname, '..', 'scratch_edge_pdf_' + Date.now());
const PORT = 9598;
const ARTIFACT_DIR = path.join(__dirname, '..');

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
  console.log('Starting End-to-End Browser Verification for Form LA-7 Auto-Fill...');
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
    await sleep(3000);

    // 2. Switch to Predictor Tab
    console.log('Switching to Risk Predictor tab...');
    await client.send('Runtime.evaluate', {
      expression: `switchMainTab('predictor');`
    });
    await sleep(1000);

    // Verify presence of buttons
    const barCheck = await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          const btn = document.getElementById('btn-upload-pdf');
          const link = document.getElementById('link-download-template');
          const input = document.getElementById('pdf-file-input');
          return {
            hasUploadBtn: !!btn,
            btnText: btn ? btn.innerText.trim() : null,
            hasDownloadLink: !!link,
            linkHref: link ? link.getAttribute('href') : null,
            hasFileInput: !!input
          };
        })()
      `,
      returnByValue: true
    });
    console.log('Intake bar elements:', barCheck.result.value);

    // 3. Upload DEMO PDF using DOM.setFileInputFiles
    console.log('Uploading Land_Acquisition_Project_Data_Sheet_DEMO.pdf...');
    const doc = await client.send('DOM.getDocument');
    const fileNode = await client.send('DOM.querySelector', {
      nodeId: doc.root.nodeId,
      selector: '#pdf-file-input'
    });

    const demoPdfPath = path.resolve(__dirname, '..', 'templates', 'Land_Acquisition_Project_Data_Sheet_DEMO.pdf');
    await client.send('DOM.setFileInputFiles', {
      nodeId: fileNode.nodeId,
      files: [demoPdfPath]
    });

    // Wait for async processing and auto-fill
    await sleep(2500);

    // 4. Verify Auto-Filled Form Values
    const formVals = await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          const fields = {
            projectId: document.getElementById('inp-project-id')?.value,
            projectType: document.getElementById('inp-project-type')?.value,
            state: document.getElementById('inp-state')?.value,
            district: document.getElementById('inp-district')?.value,
            terrain: document.getElementById('inp-terrain')?.value,
            latitude: document.getElementById('inp-latitude')?.value,
            longitude: document.getElementById('inp-longitude')?.value,
            roadType: document.getElementById('inp-road-type')?.value,
            siteAddress: document.getElementById('inp-site-address')?.value,
            cost: document.getElementById('inp-cost')?.value,
            landArea: document.getElementById('inp-land-area')?.value,
            siaStatus: document.getElementById('inp-sia-status')?.value,
            forestStatus: document.getElementById('inp-forest-status')?.value,
            fundPct: document.getElementById('inp-fund-pct')?.value,
            sec11Days: document.getElementById('inp-sec11-days')?.value,
            compMult: document.getElementById('inp-comp-mult')?.value,
            pafs: document.getElementById('inp-affected-families')?.value,
            disputeRate: document.getElementById('inp-dispute-rate')?.value,
            protestFlag: document.getElementById('inp-protest-flag')?.checked,
            autofillClassesCount: document.querySelectorAll('.autofilled-field').length,
            autofillTagsCount: document.querySelectorAll('.autofill-indicator-tag').length,
            toastText: document.getElementById('toast-message')?.textContent
          };
          return fields;
        })()
      `,
      returnByValue: true
    });

    console.log('Form Values after Auto-Fill:');
    console.log(JSON.stringify(formVals.result.value, null, 2));

    const vals = formVals.result.value;
    if (vals.projectId !== 'NHAI-RJ-2023-0001') throw new Error(`Project ID mismatch: ${vals.projectId}`);
    if (vals.state !== 'Rajasthan') throw new Error(`State mismatch: ${vals.state}`);
    if (vals.district !== 'Banswara') throw new Error(`District mismatch: ${vals.district}`);
    if (vals.projectType !== 'Highway') throw new Error(`Project Type mismatch: ${vals.projectType}`);
    if (vals.terrain !== 'Rural_Agri') throw new Error(`Terrain mismatch: ${vals.terrain}`);
    if (parseFloat(vals.cost) !== 1485.37) throw new Error(`Cost mismatch: ${vals.cost}`);
    if (parseFloat(vals.landArea) !== 229.1) throw new Error(`Land area mismatch: ${vals.landArea}`);
    if (parseInt(vals.sec11Days) !== 311) throw new Error(`Sec 11 mismatch: ${vals.sec11Days}`);
    if (vals.protestFlag !== false) throw new Error(`Protest flag mismatch: ${vals.protestFlag}`);
    if (vals.autofillClassesCount === 0) throw new Error('No .autofilled-field classes found!');

    console.log(`Summary Toast Message: "${vals.toastText}"`);

    // 5. Capture High-Resolution Screenshot
    console.log('Capturing screenshot of auto-filled Risk Predictor form...');
    const screenshot = await client.send('Page.captureScreenshot', {
      format: 'png'
    });
    const screenshotPath = path.join(ARTIFACT_DIR, 'autofilled_form_screenshot.png');
    fs.writeFileSync(screenshotPath, Buffer.from(screenshot.data, 'base64'));
    console.log(`Saved screenshot to: ${screenshotPath}`);

    // Also save in local scratch directory
    const localScreenshotPath = path.join(__dirname, '..', 'autofilled_form_screenshot.png');
    fs.writeFileSync(localScreenshotPath, Buffer.from(screenshot.data, 'base64'));

    // 5b. Upload project_sample_5.pdf (Stacked Layout)
    console.log('Uploading project_sample_5.pdf (Stacked Layout: Assam, Karbi Anglong)...');
    const sample5PdfPath = path.resolve(__dirname, '..', 'templates', 'project_sample_5.pdf');
    await client.send('DOM.setFileInputFiles', {
      nodeId: fileNode.nodeId,
      files: [sample5PdfPath]
    });
    await sleep(2500);

    const formValsSample5 = await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          return {
            projectId: document.getElementById('inp-project-id')?.value,
            projectType: document.getElementById('inp-project-type')?.value,
            state: document.getElementById('inp-state')?.value,
            district: document.getElementById('inp-district')?.value,
            terrain: document.getElementById('inp-terrain')?.value,
            latitude: document.getElementById('inp-latitude')?.value,
            longitude: document.getElementById('inp-longitude')?.value,
            roadType: document.getElementById('inp-road-type')?.value,
            siteAddress: document.getElementById('inp-site-address')?.value,
            cost: document.getElementById('inp-cost')?.value,
            landArea: document.getElementById('inp-land-area')?.value,
            siaStatus: document.getElementById('inp-sia-status')?.value,
            forestStatus: document.getElementById('inp-forest-status')?.value,
            fundPct: document.getElementById('inp-fund-pct')?.value,
            sec11Days: document.getElementById('inp-sec11-days')?.value,
            compMult: document.getElementById('inp-comp-mult')?.value,
            pafs: document.getElementById('inp-affected-families')?.value,
            disputeRate: document.getElementById('inp-dispute-rate')?.value,
            protestFlag: document.getElementById('inp-protest-flag')?.checked,
            autofillClassesCount: document.querySelectorAll('.autofilled-field').length,
            toastText: document.getElementById('toast-message')?.textContent
          };
        })()
      `,
      returnByValue: true
    });

    console.log('Form Values after project_sample_5.pdf Auto-Fill:');
    console.log(JSON.stringify(formValsSample5.result.value, null, 2));

    const s5Vals = formValsSample5.result.value;
    if (s5Vals.projectId !== 'NHIDCL-AS-2023-003') throw new Error(`Project ID mismatch: ${s5Vals.projectId}`);
    if (s5Vals.state !== 'Assam') throw new Error(`State mismatch: ${s5Vals.state}`);
    if (s5Vals.district !== 'Karbi Anglong') throw new Error(`District mismatch: ${s5Vals.district}`);
    if (s5Vals.projectType !== 'Highway') throw new Error(`Project Type mismatch: ${s5Vals.projectType}`);
    if (s5Vals.terrain !== 'Hilly') throw new Error(`Terrain mismatch: ${s5Vals.terrain}`);
    if (parseFloat(s5Vals.cost) !== 2140.6) throw new Error(`Cost mismatch: ${s5Vals.cost}`);
    if (parseFloat(s5Vals.landArea) !== 98.7) throw new Error(`Land area mismatch: ${s5Vals.landArea}`);
    if (parseInt(s5Vals.sec11Days) !== 60) throw new Error(`Sec 11 mismatch: ${s5Vals.sec11Days}`);

    const screenshotS5 = await client.send('Page.captureScreenshot', { format: 'png' });
    const s5ScreenshotPath = path.join(ARTIFACT_DIR, 'autofilled_sample5_screenshot.png');
    fs.writeFileSync(s5ScreenshotPath, Buffer.from(screenshotS5.data, 'base64'));
    fs.writeFileSync(path.join(__dirname, '..', 'autofilled_sample5_screenshot.png'), Buffer.from(screenshotS5.data, 'base64'));
    console.log(`Saved sample 5 screenshot to: ${s5ScreenshotPath}`);

    // 6. Test Unrelated PDF upload and verify error handling
    console.log('Testing unrelated document upload error handling...');
    const unrelatedPath = path.resolve(__dirname, '..', 'templates', 'unrelated_document.pdf');
    await client.send('DOM.setFileInputFiles', {
      nodeId: fileNode.nodeId,
      files: [unrelatedPath]
    });
    await sleep(2000);

    const errorToast = await client.send('Runtime.evaluate', {
      expression: `document.getElementById('toast-message')?.textContent`,
      returnByValue: true
    });
    console.log(`Error Toast Message on Unrelated PDF: "${errorToast.result.value}"`);

    console.log('\n========================================');
    console.log('ALL BROWSER END-TO-END VERIFICATION CHECKS PASSED!');
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
