/* ===========================================================
   Face Emotion App — frontend logic
   =========================================================== */
'use strict';

const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const API = {
  health:        '/api/health',
  detectImage:   '/api/detect/image',
  detectFrame:   '/api/detect/frame',
  detectVideo:   '/api/detect/video',
  videoStatus:   (id) => `/api/detect/video/${id}/status`,
  videoDownload: (id) => `/api/detect/video/${id}/download`,
  cameraList:    '/api/camera/list',
  cameraStream:  (id) => `/api/camera/${id}/stream`,
  cameraStop:    (id) => `/api/camera/${id}/stop`,
};

// Mirror of app/config.py MAX_UPLOAD_MB. Keep these two values in sync —
// the client-side check below just gives a friendlier pre-flight error;
// the server-side check in app/routers/{image,video,detect}.py is the
// authoritative one and will return 413 if this drifts.
const MAX_UPLOAD_MB = 4096;
const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

// Extract FastAPI's "detail" field from a non-2xx response (e.g. 413).
// Falls back to a generic message so the user still gets something useful.
async function readErrorDetail(resp, fallback) {
  try {
    const data = await resp.clone().json();
    if (data && typeof data.detail === 'string') return data.detail;
  } catch (_) { /* not JSON, ignore */ }
  return fallback;
}

// ---- Tabs ---------------------------------------------------------------
$$('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    $$('.tab').forEach(t => t.classList.toggle('active', t === btn));
    $$('.panel').forEach(p => p.classList.toggle('active', p.dataset.panel === target));
  });
});

// ---- Health / status pill ----------------------------------------------
async function refreshHealth() {
  const pill = $('#status-pill');
  const text = $('#status-text');
  const footer = $('#footer-device');
  try {
    const r = await fetch(API.health);
    if (!r.ok) throw new Error('health ' + r.status);
    const data = await r.json();
    const ready = data.models_loaded && data.models_loaded.detector && data.models_loaded.classifier;
    pill.classList.toggle('ok', !!ready);
    pill.classList.toggle('bad', !ready);
    text.textContent = ready
      ? `就绪 · ${data.gpu_name || data.device}`
      : (data.cuda_available ? 'GPU 已检测,模型未加载' : '模型加载中…');
    footer.textContent = `device: ${data.device} · cuda: ${data.cuda_available}`;
  } catch (e) {
    pill.classList.remove('ok');
    pill.classList.add('bad');
    text.textContent = '服务不可达';
    console.error(e);
  }
}
refreshHealth();
setInterval(refreshHealth, 8000);

// =====================================================================
// IMAGE TAB
// =====================================================================
const imgDrop = $('#img-drop');
const imgInput = $('#img-input');
const imgOriginal = $('#img-original');
const imgResult = $('#img-result');
const imgChart = $('#img-emotion-chart');
const imgBars = $('#img-bars');

imgDrop.addEventListener('click', () => imgInput.click());
imgDrop.addEventListener('dragover', e => { e.preventDefault(); imgDrop.classList.add('drag'); });
imgDrop.addEventListener('dragleave', () => imgDrop.classList.remove('drag'));
imgDrop.addEventListener('drop', e => {
  e.preventDefault();
  imgDrop.classList.remove('drag');
  if (e.dataTransfer.files[0]) handleImage(e.dataTransfer.files[0]);
});
imgInput.addEventListener('change', e => { if (e.target.files[0]) handleImage(e.target.files[0]); });

const EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];

function renderEmotionBars(container, scores) {
  container.innerHTML = '';
  const max = Math.max(0.001, ...Object.values(scores));
  EMOTIONS.forEach(em => {
    const v = scores[em] || 0;
    const row = document.createElement('div');
    row.className = 'bar';
    row.innerHTML = `
      <span class="label">${em}</span>
      <div class="track"><div class="fill" style="width:${(v * 100).toFixed(1)}%"></div></div>
      <span class="pct">${(v * 100).toFixed(1)}%</span>
    `;
    container.appendChild(row);
  });
}

async function handleImage(file) {
  if (!file.type.startsWith('image/')) {
    alert('请选择图片文件');
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    alert(`图片超过 ${MAX_UPLOAD_MB / 1024}GB 限制(当前 ${(file.size / 1048576).toFixed(1)}MB)`);
    return;
  }
  // Preview
  const url = URL.createObjectURL(file);
  imgOriginal.src = url;
  imgResult.removeAttribute('src');
  imgChart.hidden = true;

  const fd = new FormData();
  fd.append('file', file);
  try {
    // Fire two parallel requests: annotated image + JSON stats
    const [imgResp, jsonResp] = await Promise.all([
      fetch(API.detectImage, { method: 'POST', body: fd }),
      (() => {
        const fd2 = new FormData();
        fd2.append('file', file);
        return fetch(API.detectFrame, { method: 'POST', body: fd2 });
      })(),
    ]);
    if (!imgResp.ok) {
      const detail = await readErrorDetail(imgResp, `HTTP ${imgResp.status}`);
      throw new Error(detail);
    }
    if (!jsonResp.ok) {
      const detail = await readErrorDetail(jsonResp, `HTTP ${jsonResp.status}`);
      throw new Error(detail);
    }

    const blob = await imgResp.blob();
    imgResult.src = URL.createObjectURL(blob);
    const count = imgResp.headers.get('X-Face-Count') || '0';
    const summary = imgResp.headers.get('X-Emotion-Summary') || '';

    const data = await jsonResp.json();
    if (data.count > 0) {
      // Sum scores across all faces for a "scene mood" distribution
      const agg = Object.fromEntries(EMOTIONS.map(e => [e, 0]));
      data.detections.forEach(d => {
        for (const [k, v] of Object.entries(d.all_scores || {})) {
          if (k in agg) agg[k] += Number(v) || 0;
        }
      });
      // Normalise
      const total = Object.values(agg).reduce((a, b) => a + b, 0) || 1;
      for (const k of Object.keys(agg)) agg[k] /= total;
      imgChart.hidden = false;
      renderEmotionBars(imgBars, agg);
    } else {
      imgChart.hidden = true;
    }
    console.log(`[image] ${count} face(s); ${summary}`);
  } catch (e) {
    alert('识别失败: ' + e.message);
    console.error(e);
  }
}

// =====================================================================
// VIDEO TAB
// =====================================================================
const vidDrop = $('#vid-drop');
const vidInput = $('#vid-input');
const vidStatus = $('#vid-status');
const vidProgress = $('#vid-progress');
const vidPercent = $('#vid-percent');
const vidMessage = $('#vid-message');
const vidDownload = $('#vid-download');

vidDrop.addEventListener('click', () => vidInput.click());
vidDrop.addEventListener('dragover', e => { e.preventDefault(); vidDrop.classList.add('drag'); });
vidDrop.addEventListener('dragleave', () => vidDrop.classList.remove('drag'));
vidDrop.addEventListener('drop', e => {
  e.preventDefault();
  vidDrop.classList.remove('drag');
  if (e.dataTransfer.files[0]) handleVideo(e.dataTransfer.files[0]);
});
vidInput.addEventListener('change', e => { if (e.target.files[0]) handleVideo(e.target.files[0]); });

async function handleVideo(file) {
  if (!file.type.startsWith('video/') && !/\.(mp4|mov|avi|mkv|webm)$/i.test(file.name)) {
    alert('请选择视频文件');
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    const sz = (file.size / 1048576).toFixed(1);
    vidStatus.hidden = false;
    vidDownload.hidden = true;
    vidProgress.style.width = '0%';
    vidPercent.textContent = '0%';
    vidMessage.textContent = `文件 ${sz}MB 超过 ${MAX_UPLOAD_MB / 1024}GB 限制,已取消上传`;
    return;
  }
  vidStatus.hidden = false;
  vidDownload.hidden = true;
  vidProgress.style.width = '0%';
  vidPercent.textContent = '0%';
  vidMessage.textContent = '上传中…';

  const fd = new FormData();
  fd.append('file', file);
  let resp;
  try {
    resp = await fetch(API.detectVideo, { method: 'POST', body: fd });
  } catch (e) {
    vidMessage.textContent = '上传失败: ' + e.message;
    return;
  }
  if (!resp.ok) {
    const detail = await readErrorDetail(resp, `HTTP ${resp.status}`);
    vidMessage.textContent = `上传失败: ${detail}`;
    return;
  }
  const { task_id } = await resp.json();
  vidMessage.textContent = '处理中…';

  // Poll
  const tick = setInterval(async () => {
    try {
      const r = await fetch(API.videoStatus(task_id));
      if (!r.ok) return;
      const d = await r.json();
      const pct = Math.round((d.progress || 0) * 100);
      vidProgress.style.width = pct + '%';
      vidPercent.textContent = pct + '%';
      vidMessage.textContent = d.message || d.status;
      if (d.status === 'done') {
        clearInterval(tick);
        if (d.download_url) {
          vidDownload.href = d.download_url;
          vidDownload.hidden = false;
        }
      } else if (d.status === 'error') {
        clearInterval(tick);
        vidMessage.textContent = '处理失败: ' + (d.error || '未知错误');
      }
    } catch (e) { /* ignore transient */ }
  }, 800);
}

// =====================================================================
// CAMERA TAB
// =====================================================================
const camSelect = $('#cam-select');
const camStart = $('#cam-start');
const camStop = $('#cam-stop');
const camRefresh = $('#cam-refresh');
const camFeed = $('#cam-feed');
const camStage = $('#cam-stage');

async function loadCameras() {
  camSelect.innerHTML = '<option>加载中…</option>';
  try {
    const r = await fetch(API.cameraList);
    const cams = await r.json();
    camSelect.innerHTML = '';
    const available = cams.filter(c => c.available);
    if (available.length === 0) {
      camSelect.innerHTML = '<option value="-1">未发现摄像头</option>';
      camStart.disabled = true;
      return;
    }
    available.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.id;
      opt.textContent = `${c.name}`;
      camSelect.appendChild(opt);
    });
    camStart.disabled = false;
  } catch (e) {
    camSelect.innerHTML = '<option>加载失败</option>';
    camStart.disabled = true;
    console.error(e);
  }
}
loadCameras();
camRefresh.addEventListener('click', loadCameras);

let activeCamId = null;
camStart.addEventListener('click', () => {
  const id = parseInt(camSelect.value, 10);
  if (Number.isNaN(id) || id < 0) return;
  // Add cache-buster to ensure reload
  camFeed.src = API.cameraStream(id) + '?t=' + Date.now();
  camFeed.hidden = false;
  camStart.disabled = true;
  camStop.disabled = false;
  activeCamId = id;
});

camStop.addEventListener('click', async () => {
  if (activeCamId !== null) {
    try { await fetch(API.cameraStop(activeCamId), { method: 'POST' }); } catch {}
  }
  camFeed.src = '';
  camFeed.hidden = true;
  camStart.disabled = false;
  camStop.disabled = true;
  activeCamId = null;
});
