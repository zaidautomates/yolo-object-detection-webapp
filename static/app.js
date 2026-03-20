/* ═══════════════════════════════════════════════
   YOLO Object Detector — Frontend Logic
   ═══════════════════════════════════════════════ */

const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const previewStrip= document.getElementById('previewStrip');
const previewImg  = document.getElementById('previewImg');
const fileName    = document.getElementById('fileName');
const fileSize    = document.getElementById('fileSize');
const btnDetect   = document.getElementById('btnDetect');
const resultsWrap = document.getElementById('resultsWrap');
const statsBar    = document.getElementById('statsBar');
const statTotal   = document.getElementById('statTotal');
const statAvgConf = document.getElementById('statAvgConf');
const statBest    = document.getElementById('statBest');
const annotatedImg= document.getElementById('annotatedImg');
const detBody     = document.getElementById('detBody');
const errorToast  = document.getElementById('errorToast');

// Colour palette (matches backend drawing colours roughly)
const DOT_COLORS = [
  '#00c8ff','#00ff64','#ff6400',
  '#c800ff','#ffff00','#0096ff',
  '#ff0096','#64ffc8','#ffc864',
  '#9600c8',
];

let selectedFile = null;

/* ── Drag-and-drop ────────────────────────────── */
['dragenter','dragover'].forEach(evt =>
  dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.add('drag-over'); })
);
['dragleave','drop'].forEach(evt =>
  dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.remove('drag-over'); })
);
dropZone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});

/* ── File input change ────────────────────────── */
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

/* ── Handle selected file ─────────────────────── */
function handleFile(file) {
  selectedFile = file;

  // show preview
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewStrip.classList.remove('hidden');
  };
  reader.readAsDataURL(file);

  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  btnDetect.disabled = false;
  hideError();
}

/* ── Detect button ────────────────────────────── */
btnDetect.addEventListener('click', async () => {
  if (!selectedFile) return;

  // UI → loading state
  btnDetect.classList.add('loading');
  btnDetect.disabled = true;
  resultsWrap.classList.add('hidden');
  statsBar.classList.add('hidden');
  hideError();

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const res = await fetch('/detect', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || 'Detection failed.');
      return;
    }

    renderResults(data);
  } catch (err) {
    showError('Could not reach the server. Is api.py running?');
  } finally {
    btnDetect.classList.remove('loading');
    btnDetect.disabled = false;
  }
});

/* ── Render results ───────────────────────────── */
function renderResults(data) {
  // Annotated image
  annotatedImg.src = 'data:image/jpeg;base64,' + data.annotated_image;

  // Stats
  const total = data.total_detections;
  const avgConf = total
    ? (data.detections.reduce((s, d) => s + d.confidence, 0) / total * 100).toFixed(1)
    : 0;
  const bestDet = total
    ? data.detections.reduce((a, b) => a.confidence > b.confidence ? a : b)
    : null;

  statTotal.textContent   = total;
  statAvgConf.textContent = avgConf + '%';
  statBest.textContent    = bestDet ? bestDet.class : '—';

  // Table
  detBody.innerHTML = '';
  data.detections.forEach((d, i) => {
    const color = DOT_COLORS[i % DOT_COLORS.length];
    const pct   = (d.confidence * 100).toFixed(1);
    const barColor = pct >= 70 ? '#34d399' : pct >= 40 ? '#fbbf24' : '#f87171';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>
        <span class="class-name">
          <span class="class-dot" style="background:${color}"></span>
          ${d.class}
        </span>
      </td>
      <td>
        <div class="conf-bar-wrap">
          <div class="conf-bar">
            <div class="conf-bar-fill" style="width:${pct}%;background:${barColor}"></div>
          </div>
          <span class="conf-pct">${pct}%</span>
        </div>
      </td>
      <td class="bbox-text">[${d.bbox.join(', ')}]</td>
    `;
    detBody.appendChild(tr);
  });

  statsBar.classList.remove('hidden');
  resultsWrap.classList.remove('hidden');
}

/* ── Helpers ──────────────────────────────────── */
function formatBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

function showError(msg) {
  errorToast.textContent = msg;
  errorToast.classList.remove('hidden');
}
function hideError() {
  errorToast.classList.add('hidden');
}
