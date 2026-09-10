const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const USER_DATA_DIR = path.join(__dirname, '..', 'scratch_edge_banner_' + Date.now());
const PORT = 9595;

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

  async eval(expression) {
    const res = await this.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    return res.result ? res.result.value : undefined;
  }

  close() {
    this.ws.close();
  }
}

async function run() {
  console.log('Testing Government of India banner and main nav...');
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
    await client.send('Runtime.enable');
    await client.send('Emulation.setDeviceMetricsOverride', {
      width: 1440,
      height: 900,
      deviceScaleFactor: 1,
      mobile: false
    });

    console.log('Navigating to http://localhost:8000/dashboard...');
    await client.send('Page.navigate', { url: 'http://localhost:8000/dashboard' });

    // Wait for page and images to load
    await sleep(2000);

    const bannerInfo = await client.eval(`
      (() => {
        const banner = document.querySelector('.gov-identity-banner');
        const emblem = document.querySelector('.gov-emblem');
        const swachh = document.querySelector('.gov-swachh-logo');
        const nav = document.querySelector('.apple-sub-nav');
        return {
          bannerFound: !!banner,
          bannerRect: banner ? banner.getBoundingClientRect() : null,
          emblemSrc: emblem ? emblem.src : null,
          emblemLoaded: emblem ? (emblem.complete && emblem.naturalWidth > 0) : false,
          emblemDims: emblem ? { width: emblem.naturalWidth, height: emblem.naturalHeight } : null,
          swachhSrc: swachh ? swachh.src : null,
          swachhLoaded: swachh ? (swachh.complete && swachh.naturalWidth > 0) : false,
          swachhDims: swachh ? { width: swachh.naturalWidth, height: swachh.naturalHeight } : null,
          titleHi: document.querySelector('.gov-title-hi') ? document.querySelector('.gov-title-hi').textContent : null,
          titleEn: document.querySelector('.gov-title-en') ? document.querySelector('.gov-title-en').textContent : null,
          subEn: document.querySelector('.gov-sub-en') ? document.querySelector('.gov-sub-en').textContent : null,
          navFound: !!nav,
          navRect: nav ? nav.getBoundingClientRect() : null
        };
      })()
    `);

    console.log('Banner diagnostic results:', JSON.stringify(bannerInfo, null, 2));

    const shot = await client.send('Page.captureScreenshot', {
      format: 'png',
      clip: { x: 0, y: 0, width: 1440, height: 400, scale: 1 }
    });
    const buf = Buffer.from(shot.data, 'base64');
    const artifactPath = 'C:\\Users\\PRATYUSH\\.gemini\\antigravity-ide\\brain\\3a557fb1-3b6c-402e-955f-9a791e1a8ad5\\gov_banner_live.png';
    fs.writeFileSync(artifactPath, buf);
    console.log('Saved screenshot to', artifactPath);

    client.close();
  } finally {
    edgeProc.kill();
    try {
      fs.rmSync(USER_DATA_DIR, { recursive: true, force: true });
    } catch (e) {}
  }
}

run().catch(console.error);
