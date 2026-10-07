// OmniBrain Quant Workspace Frontend Logic
import './style.css';
import { marked } from 'marked';

let latestMemoMarkdown = 'No memo generated yet.';
let currentUploadedImages = [];
let currentUploadedPdfName = null;
let currentArtifactList = [];
let currentArtifactIndex = 0;
window.chatHistory = [];

// Global Toast System
export function showToast(msg, durationMs = 3000) {
  const toast = document.getElementById('global-toast');
  const toastMsg = document.getElementById('toast-message');
  if (!toast || !toastMsg) return;
  toastMsg.textContent = msg;
  toast.classList.remove('hidden');
  clearTimeout(window._toastTimeout);
  window._toastTimeout = setTimeout(() => {
    toastMsg.textContent = 'Swarm Idle';
  }, durationMs);
}

// Keyboard shortcut: Cmd/Ctrl + K to focus query input
document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
    e.preventDefault();
    const input = document.getElementById('query-input');
    if (input) {
      input.focus();
      input.select();
    }
  }
});

// Apply suggestion chip
export function applySuggestion(text) {
  const input = document.getElementById('query-input');
  if (input) {
    input.value = text;
    executeAnalystQuery();
  }
}

// Toggle Trace Accordion
export function toggleTraceContent() {
  const content = document.getElementById('trace-content');
  const icon = document.getElementById('trace-expand-icon');
  if (!content || !icon) return;
  if (content.classList.contains('hidden')) {
    content.classList.remove('hidden');
    icon.textContent = 'expand_less';
  } else {
    content.classList.add('hidden');
    icon.textContent = 'expand_more';
  }
}

// Toggle Swarm Filter Pills
export function toggleSwarmFilter(type, btn) {
  document.querySelectorAll('.swarm-pill').forEach((b) => {
    b.className =
      'swarm-pill px-2.5 py-1 rounded-full bg-surface-container-low hover:bg-surface-container text-secondary hover:text-on-surface font-medium flex items-center gap-1 transition-colors';
  });
  if (btn) {
    btn.className =
      'swarm-pill px-2.5 py-1 rounded-full bg-primary-fixed text-primary font-semibold flex items-center gap-1 transition-colors';
  }
  showToast(`Swarm focus: ${type.toUpperCase()} filter active`);
}

// Filter Corpus Assets
export function filterCorpus(type, btn) {
  document.querySelectorAll('.corpus-tab-btn').forEach((b) => {
    b.className =
      'corpus-tab-btn px-2 py-1.5 text-secondary hover:text-on-surface hover:bg-surface-container transition-colors rounded-md flex items-center gap-1';
  });
  if (btn) {
    btn.className =
      'corpus-tab-btn px-2.5 py-1.5 text-primary font-medium bg-surface-container-lowest rounded-md shadow-xs flex items-center gap-1';
  }

  document.querySelectorAll('.artifact-card').forEach((card) => {
    if (type === 'all') {
      card.style.display = 'block';
    } else if (type === 'chart') {
      card.style.display = card.innerText.includes('Chart') ? 'block' : 'none';
    } else if (type === 'table') {
      card.style.display = card.innerText.includes('Table') ? 'block' : 'none';
    }
  });
}

// Switch Corpus Tab from Navigation
export function switchCorpusTab(tab) {
  showToast('Corpus view: 10-K document and multimodal tables active');
  const firstTabBtn = document.querySelector('.corpus-tab-btn');
  if (firstTabBtn) filterCorpus('all', firstTabBtn);
}

// Jump to Page
export function jumpToPage(val) {
  const pageNum = val.replace(/[^0-9]/g, '') || '1';
  showToast(`Navigated to PDF Page ${pageNum} • OCR context active`);
}

// Copy Markdown Memo
export function copyMarkdownMemo(btn) {
  const text = decodeURIComponent(btn.getAttribute('data-markdown') || '');
  navigator.clipboard
    .writeText(text)
    .then(() => {
      showToast('Markdown copied to clipboard! 📋');
    })
    .catch(() => {
      showToast('Failed to copy');
    });
}

// Artifact Inspection System
export async function loadWorkspaceArtifacts(targetDoc = null) {
  try {
    const url = targetDoc ? `/api/v1/visual/artifacts/recent?doc_name=${encodeURIComponent(targetDoc)}` : '/api/v1/visual/artifacts/recent';
    const res = await fetch(url);
    if (!res.ok) return false;
    const data = await res.json();
    if (!data.artifacts || data.artifacts.length === 0) return false;

    currentArtifactList = data.artifacts;
    currentUploadedImages = data.artifacts.map(a => a.filename);
    currentUploadedPdfName = data.pdf_name;

    const titleElem = document.getElementById('doc-card-title');
    const pgElem = document.getElementById('doc-card-pgcount');
    const subtitleElem = document.getElementById('doc-card-subtitle');
    const ocrStatus = document.getElementById('ocr-status');
    const ocrStatusIcon = document.getElementById('ocr-status-icon');
    const activeDocPages = document.getElementById('active-doc-pages');

    if (titleElem && data.pdf_name) titleElem.textContent = data.pdf_name;
    if (pgElem && data.total_pages) pgElem.textContent = `${data.total_pages} Pgs`;
    if (subtitleElem) subtitleElem.textContent = `Active Ingest Corpus • ${data.artifacts.length} Visual Exhibits`;
    if (ocrStatus) ocrStatus.textContent = 'OCR Completed';
    if (ocrStatusIcon) ocrStatusIcon.textContent = 'check_circle';
    if (activeDocPages && data.total_pages) activeDocPages.textContent = `(${data.total_pages} pgs)`;

    renderArtifactCards(data.artifacts);
    return true;
  } catch (err) {
    console.warn('Failed to load recent artifacts:', err);
    return false;
  }
}

export function renderArtifactCards(artifacts) {
  const artifactContainer = document.getElementById('artifact-cards-wrapper');
  if (!artifactContainer) return;
  artifactContainer.innerHTML = '';

  let tableCount = 0;
  let chartCount = 0;

  artifacts.forEach((art, idx) => {
    const isTable = art.type === 'table';
    if (isTable) tableCount++; else chartCount++;
    const icon = isTable ? 'table_chart' : 'bar_chart';

    artifactContainer.innerHTML += `
      <div onclick="inspectArtifactByIndex(${idx})" class="bg-surface-container-lowest rounded-xl p-2.5 border border-outline-variant/30 shadow-xs hover:border-primary/40 transition-all cursor-pointer group flex flex-col gap-2 artifact-card hover:shadow-md">
        <div class="flex items-center gap-2">
          <div class="w-7 h-7 bg-primary/10 rounded flex items-center justify-center text-primary shrink-0">
            <span class="material-symbols-outlined text-[15px]">${icon}</span>
          </div>
          <div class="overflow-hidden flex-1">
            <h3 class="text-xs font-semibold text-on-surface truncate group-hover:text-primary transition-colors">${art.title}</h3>
            <p class="text-[10px] text-secondary truncate">Page ${art.page} • ${art.filename}</p>
          </div>
        </div>
        <div class="w-full h-24 bg-surface-container rounded border border-outline-variant/20 overflow-hidden relative group-hover:shadow-inner transition-all flex items-center justify-center">
          <img src="${art.url}" alt="${art.title}" class="w-full h-full object-contain mix-blend-multiply opacity-90 group-hover:opacity-100 transition-opacity" loading="lazy">
          <div class="absolute inset-0 bg-primary/0 group-hover:bg-primary/5 transition-colors"></div>
        </div>
        <button type="button" onclick="event.stopPropagation(); inspectArtifactByIndex(${idx})" class="w-full py-1.5 px-2 rounded-lg bg-surface-container hover:bg-primary hover:text-on-primary text-secondary text-[11px] font-semibold flex items-center justify-center gap-1.5 transition-all shadow-xs cursor-pointer">
          <span class="material-symbols-outlined text-[14px]">visibility</span>
          <span>Inspect Exhibit</span>
        </button>
      </div>
    `;
  });

  const tabTables = document.getElementById('tab-tables');
  const tabCharts = document.getElementById('tab-charts');
  if (tabTables) {
    tabTables.innerHTML = `<span class="material-symbols-outlined text-sm">table_chart</span><span>Tables (${tableCount})</span>`;
  }
  if (tabCharts) {
    tabCharts.innerHTML = `<span class="material-symbols-outlined text-sm">bar_chart</span><span>Charts (${chartCount})</span>`;
  }
}

export async function inspectArtifactByIndex(idx) {
  if (!currentArtifactList || currentArtifactList.length === 0) {
    if (currentUploadedImages && currentUploadedImages.length > 0) {
      currentArtifactList = currentUploadedImages.map((img, i) => {
        const fname = img.split(/[/\\]/).pop();
        const isTbl = fname.startsWith('table_');
        const pMatch = fname.match(/_p(\d+)_/);
        return {
          index: i,
          filename: fname,
          url: `/api/v1/images/${fname}`,
          title: `${isTbl ? 'Extracted Table' : 'Extracted Chart'} ${i + 1}`,
          type: isTbl ? 'table' : 'chart',
          page: pMatch ? pMatch[1] : 1,
        };
      });
    } else {
      await loadWorkspaceArtifacts();
    }
  }

  if (!currentArtifactList || currentArtifactList.length === 0) {
    showToast('No artifacts to inspect. Please upload a PDF first.');
    return;
  }

  if (idx < 0) idx = 0;
  if (idx >= currentArtifactList.length) idx = currentArtifactList.length - 1;
  currentArtifactIndex = idx;

  const item = currentArtifactList[idx];
  const modal = document.getElementById('artifact-modal');
  if (!modal) return;

  const titleElem = document.getElementById('modal-title');
  const pageElem = document.getElementById('modal-page');
  const counterElem = document.getElementById('modal-counter');
  const imgElem = document.getElementById('modal-artifact-img');
  const iconElem = document.getElementById('modal-icon');
  const fullLink = document.getElementById('modal-full-img-link');

  if (titleElem) titleElem.textContent = item.title;
  if (pageElem) pageElem.textContent = `Source: ${item.filename} • PDF Page ${item.page}`;
  if (counterElem) counterElem.textContent = `Exhibit ${idx + 1} of ${currentArtifactList.length}`;
  if (iconElem) iconElem.textContent = item.type === 'chart' ? 'bar_chart' : 'table_chart';
  if (imgElem) {
    imgElem.src = item.url;
    imgElem.alt = item.title;
  }
  if (fullLink) {
    fullLink.href = item.url;
  }

  modal.classList.remove('hidden');
}

export function navigateArtifact(delta) {
  inspectArtifactByIndex(currentArtifactIndex + delta);
}

export async function inspectAllArtifacts() {
  if (!currentArtifactList || currentArtifactList.length === 0) {
    showToast('Loading exhibits for inspection...');
    const loaded = await loadWorkspaceArtifacts();
    if (!loaded || currentArtifactList.length === 0) {
      showToast('No artifacts available to inspect. Please upload a PDF first.');
      return;
    }
  }
  inspectArtifactByIndex(0);
}

export function previewArtifact(title, page, type, url = '') {
  if (url) {
    const foundIdx = currentArtifactList.findIndex(a => a.url === url || a.filename === url.split('/').pop());
    if (foundIdx !== -1) {
      inspectArtifactByIndex(foundIdx);
      return;
    }
  }
  inspectAllArtifacts();
}

export function closeArtifactModal() {
  document.getElementById('artifact-modal')?.classList.add('hidden');
}

// Keyboard navigation for exhibit inspector modal
document.addEventListener('keydown', (e) => {
  const modal = document.getElementById('artifact-modal');
  if (modal && !modal.classList.contains('hidden')) {
    if (e.key === 'Escape') {
      closeArtifactModal();
    } else if (e.key === 'ArrowLeft') {
      navigateArtifact(-1);
    } else if (e.key === 'ArrowRight') {
      navigateArtifact(1);
    }
  }
});

// DAG Modal
export function showGraphDAGModal() {
  document.getElementById('dag-modal')?.classList.remove('hidden');
}

export function closeDAGModal() {
  document.getElementById('dag-modal')?.classList.add('hidden');
}

// File Upload Handler
export async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  showToast(`Uploading ${file.name} to OmniBrain Ingest Pipeline...`);

  // Unhide containers
  document.getElementById('active-doc-pill')?.classList.remove('hidden');
  document.getElementById('active-doc-pill')?.classList.add('flex');
  document.getElementById('doc-card-container')?.classList.remove('hidden');
  document.getElementById('corpus-tab-strip')?.classList.remove('hidden');
  document.getElementById('artifact-list-container')?.classList.remove('hidden');

  const ocrContainer = document.getElementById('ocr-status-container');
  if (ocrContainer) {
    ocrContainer.classList.remove('hidden');
    ocrContainer.classList.add('flex');
    document.getElementById('ocr-status').textContent = 'OCR Pending';
    document.getElementById('ocr-status-icon').textContent = 'pending';
  }

  const activeName = document.getElementById('active-doc-name');
  const activeBadge = document.getElementById('active-doc-badge');
  if (activeName) activeName.textContent = file.name;
  if (activeBadge) {
    activeBadge.textContent = 'Ingesting...';
    activeBadge.className =
      'text-amber-700 bg-amber-50 px-1.5 py-0.2 rounded font-label text-[10px] font-semibold tracking-wide';
  }

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/v1/ingest/pdf', { method: 'POST', body: formData });
    if (res.ok) {
      const data = await res.json();
      if (activeBadge) {
        activeBadge.textContent = 'Processed';
        activeBadge.className =
          'text-emerald-700 bg-emerald-50 px-1.5 py-0.2 rounded font-label text-[10px] font-semibold tracking-wide';
      }

      const ocrStatus = document.getElementById('ocr-status');
      if (ocrStatus) {
        ocrStatus.textContent = 'OCR Completed';
        document.getElementById('ocr-status-icon').textContent = 'check_circle';
      }

      document.getElementById('active-doc-pages').textContent = `(${data.total_pages || 1} pgs)`;
      document.getElementById('doc-card-title').textContent = file.name;
      document.getElementById('doc-card-pgcount').textContent = `${data.total_pages || 1} Pgs`;
      currentUploadedPdfName = file.name;
      currentUploadedImages = data.extracted_image_paths || [];
      currentArtifactList = [];
      let tableCount = 0;
      let chartCount = 0;
      currentUploadedImages.forEach((img, idx) => {
        const filename = img.split(/[/\\]/).pop();
        const isTable = filename.startsWith('table_');
        if (isTable) tableCount++; else chartCount++;
        const label = `${isTable ? 'Extracted Table' : 'Extracted Chart'} ${isTable ? tableCount : chartCount}`;
        const pageMatch = filename.match(/_p(\d+)_/);
        const pageNum = pageMatch ? pageMatch[1] : 1;

        currentArtifactList.push({
          index: idx,
          filename: filename,
          url: `/api/v1/images/${filename}`,
          title: label,
          type: isTable ? 'table' : 'chart',
          page: pageNum,
        });
      });
      renderArtifactCards(currentArtifactList);
      showToast(
        `✅ Ingestion complete: ${data.chunks_indexed ?? data.total_chunks ?? 0} chunks & ${data.images_extracted ?? data.total_images ?? 0} figures indexed!`
      );
    } else {
      const errText = await res.text();
      if (activeBadge) activeBadge.textContent = 'Upload Failed';
      showToast(`Error uploading: ${res.status} ${errText}`);
    }
  } catch (err) {
    if (activeBadge) activeBadge.textContent = 'Error';
    showToast(`Upload failed: ${err.message}`);
  }
}

// Drag and Drop support
document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('pdf-file-input')?.closest('label');
  if (dropZone) {
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('bg-surface-container');
    });
    dropZone.addEventListener('dragleave', (e) => {
      e.preventDefault();
      dropZone.classList.remove('bg-surface-container');
    });
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('bg-surface-container');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        document.getElementById('pdf-file-input').files = e.dataTransfer.files;
        handleFileUpload({ target: { files: e.dataTransfer.files } });
      }
    });
  }
});

const sessionId = crypto.randomUUID();

// Main Analyst Query Execution (Connects to /api/v1/chat/query)
export async function executeAnalystQuery() {
  const input = document.getElementById('query-input');
  if (!input) return;
  const query = input.value.trim();
  if (!query) return;

  const execBtn = document.getElementById('execute-btn');
  if (execBtn && execBtn.disabled) return;

  // Immediately clear input so the prompt doesn't remain in the chat box
  input.value = '';
  input.focus();

  const now = new Date();
  const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // Dynamically inject User Bubble
  const traceContainer = document.getElementById('chat-stream');
  if (traceContainer) {
    const userBubble = `
      <div class="max-w-3xl mx-auto w-full flex items-start gap-3 justify-end mt-4">
        <div class="bg-primary text-on-primary rounded-2xl rounded-tr-sm px-4 py-3 max-w-xl shadow-xs">
          <p class="text-sm font-medium leading-relaxed">${query}</p>
          <div class="mt-1 flex items-center justify-end gap-1.5 text-[10px] text-on-primary/80">
            <span>${timeStr}</span><span>•</span><span>Lead Analyst Request</span>
          </div>
        </div>
        <div class="w-8 h-8 rounded-full bg-primary-fixed text-primary flex items-center justify-center font-bold text-xs shrink-0 ring-2 ring-primary/20">QA</div>
      </div>
    `;
    const traceLoading = `
      <div id="active-trace-card" class="max-w-3xl mx-auto w-full flex items-start gap-3 mt-4 animate-pulse">
        <div class="w-8 h-8 rounded-full bg-gradient-to-br from-primary-container to-primary flex items-center justify-center text-white font-bold text-xs shrink-0 ring-2 ring-primary/20">OB</div>
        <div class="bg-surface-container-lowest border border-outline-variant/30 rounded-2xl rounded-tl-sm px-4 py-3 shadow-xs">
          <p class="text-xs font-semibold text-secondary flex items-center gap-1.5">
            <span class="material-symbols-outlined text-sm">sync</span>
            <span>Agentic Swarm processing...</span>
          </p>
        </div>
      </div>
    `;
    traceContainer.insertAdjacentHTML('beforeend', userBubble + traceLoading);
    traceContainer.scrollTop = traceContainer.scrollHeight;
  }

  if (execBtn) {
    execBtn.disabled = true;
    execBtn.innerHTML =
      '<span class="animate-spin text-sm material-symbols-outlined">sync</span><span>Running</span>';
  }

  showToast('Swarm Active: Routing query across agents...', 5000);

  const startTime = performance.now();

  try {
    const response = await fetch('/api/v1/chat/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        pdf_name: currentUploadedPdfName,
        referenced_images: currentUploadedImages,
        thread_id: sessionId
      }),
    });

    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

    if (response.ok) {
      const result = await response.json();

      const activeTraceCard = document.getElementById('active-trace-card');
      if (activeTraceCard) activeTraceCard.remove();

      const score = result.is_grounded ? 100 : 0;
      const statElem = document.getElementById('faithfulness-stat');
      if (statElem) statElem.textContent = `${score}%`;
      
      window.chatHistory.push({
        query: query,
        response: result.final_response,
        citations: result.citations || []
      });

      // Create AI bubble
      const aiBubble = `
        <div class="max-w-3xl mx-auto w-full flex items-start gap-3 mt-4">
          <div class="w-8 h-8 rounded-full bg-gradient-to-br from-primary-container to-primary flex items-center justify-center text-white font-bold text-xs shrink-0 ring-2 ring-primary/20">OB</div>
          <div class="bg-surface-container-lowest border border-outline-variant/30 rounded-2xl rounded-tl-sm px-5 py-4 w-full shadow-xs">
            <div class="prose prose-sm prose-slate max-w-none font-headline overflow-hidden">
              ${marked.parse(result.final_response || "No response generated.")}
            </div>
            
            <div class="mt-4 pt-3 border-t border-surface-container flex items-center justify-between">
              <div class="flex items-center gap-3 text-[11px]">
                <span class="flex items-center gap-1.5 ${result.is_grounded ? 'text-emerald-700 bg-emerald-50' : 'text-amber-700 bg-amber-50'} px-2 py-1 rounded-md border ${result.is_grounded ? 'border-emerald-200' : 'border-amber-200'} font-semibold">
                  <span class="material-symbols-outlined text-[14px]">${result.is_grounded ? 'verified_user' : 'warning'}</span>
                  ${result.is_grounded ? '100% Grounded' : 'Unverified'}
                </span>
                <button onclick="openCitationsPanel(this.dataset.citations)" data-citations="${encodeURIComponent(JSON.stringify(result.citations || []))}" class="text-secondary font-mono hover:text-primary transition-colors cursor-pointer border-b border-dashed border-secondary hover:border-primary">Citations: ${result.citations ? result.citations.length : 0}</button>
                <span class="text-secondary font-mono">${result.execution_time_seconds ? result.execution_time_seconds.toFixed(2) : elapsed}s</span>
              </div>
              
              <button onclick="copyMarkdownMemo(this)" data-markdown="${encodeURIComponent(result.final_response)}" class="px-2.5 py-1.5 rounded bg-surface-container hover:bg-surface-container-high font-body text-xs font-semibold text-secondary flex items-center gap-1.5 transition-colors">
                <span class="material-symbols-outlined text-sm">content_copy</span>
                <span>Copy</span>
              </button>
            </div>
          </div>
        </div>
      `;

      if (traceContainer) {
        traceContainer.insertAdjacentHTML('beforeend', aiBubble);
        traceContainer.scrollTop = traceContainer.scrollHeight;
      }
      showToast(`✅ Swarm converged in ${elapsed}s`);
    } else {
      showChatError(query, elapsed, "API returned an error response.");
    }
  } catch (e) {
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
    showChatError(query, elapsed, e.message);
  } finally {
    if (execBtn) {
      execBtn.disabled = false;
      execBtn.innerHTML =
        '<span>Execute</span><span class="material-symbols-outlined text-sm">arrow_forward</span>';
    }
  }
}

function showChatError(query, elapsed, errorMessage) {
  const activeTraceCard = document.getElementById('active-trace-card');
  if (activeTraceCard) activeTraceCard.remove();

  const traceContainer = document.getElementById('chat-stream');
  if (traceContainer) {
    const errorBubble = `
      <div class="max-w-3xl mx-auto w-full flex items-start gap-3 mt-4">
        <div class="w-8 h-8 rounded-full bg-error text-on-error flex items-center justify-center font-bold text-xs shrink-0 ring-2 ring-error/20">!</div>
        <div class="bg-error-container text-on-error-container border border-error/20 rounded-2xl rounded-tl-sm px-4 py-3 w-full shadow-xs">
          <p class="text-sm font-semibold mb-1">Failed to process query</p>
          <p class="text-xs opacity-90">${errorMessage}</p>
        </div>
      </div>
    `;
    traceContainer.insertAdjacentHTML('beforeend', errorBubble);
    traceContainer.scrollTop = traceContainer.scrollHeight;
  }
  showToast(`❌ Swarm failed in ${elapsed}s`);
}

// Bind all functions to window for direct HTML inline event handlers
window.showToast = showToast;
window.applySuggestion = applySuggestion;
window.toggleTraceContent = toggleTraceContent;
window.toggleSwarmFilter = toggleSwarmFilter;
window.filterCorpus = filterCorpus;
window.switchCorpusTab = switchCorpusTab;
window.jumpToPage = jumpToPage;
window.copyMarkdownMemo = copyMarkdownMemo;
window.previewArtifact = previewArtifact;
window.inspectArtifactByIndex = inspectArtifactByIndex;
window.navigateArtifact = navigateArtifact;
window.inspectAllArtifacts = inspectAllArtifacts;
window.closeArtifactModal = closeArtifactModal;
window.loadWorkspaceArtifacts = loadWorkspaceArtifacts;
window.renderArtifactCards = renderArtifactCards;
window.showGraphDAGModal = showGraphDAGModal;
window.closeDAGModal = closeDAGModal;
window.handleFileUpload = handleFileUpload;
window.executeAnalystQuery = executeAnalystQuery;

function logTelemetry(action, metadata = {}) {
  fetch('/api/v1/chat/telemetry/log_action', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, metadata })
  }).catch(console.error);
}

function openCitationsPanel(encodedCitations) {
  try {
    const citations = JSON.parse(decodeURIComponent(encodedCitations));
    const container = document.getElementById('citations-list-container');
    const panel = document.getElementById('citations-side-panel');
    
    container.innerHTML = '';
    
    citations.forEach((cit, idx) => {
      const scoreVal = cit.relevance_score || cit.grounding_score || 1.0;
      const score = (scoreVal * 100).toFixed(0);
      let textContent = cit.text || cit.snippet || cit.executive_summary || JSON.stringify(cit, null, 2);
      let title = cit.figure_title || cit.pdf_name || `Citation ${idx + 1}`;
      
      const card = document.createElement('div');
      card.className = 'p-3 bg-surface border border-outline-variant/40 rounded-xl shadow-xs hover:border-primary/50 cursor-pointer transition-colors';
      card.onclick = () => openCitationContent(textContent, title, score, cit);
      
      card.innerHTML = `
        <div class="flex items-start gap-2 mb-2">
          <span class="material-symbols-outlined text-primary text-base">format_quote</span>
          <h4 class="text-xs font-semibold text-on-surface truncate">${title}</h4>
        </div>
        <p class="text-[11px] text-secondary line-clamp-3">${textContent}</p>
        <div class="mt-2 pt-2 border-t border-surface-container flex justify-between items-center text-[10px] font-label">
          <span class="text-emerald-700 font-medium">Faithfulness: ${score}%</span>
          <span class="text-primary font-semibold flex items-center gap-0.5 hover:underline">
            <span class="material-symbols-outlined text-xs">visibility</span>
            <span>Inspect</span>
          </span>
        </div>
      `;
      container.appendChild(card);
    });
    
    panel.classList.remove('hidden');
    // slight delay to allow display:block before translating
    setTimeout(() => {
      panel.classList.remove('translate-x-full');
    }, 10);

    const toast = document.getElementById('global-toast');
    if (toast) {
      toast.classList.add('right-[25rem]');
      toast.classList.remove('right-4');
    }
    
    logTelemetry('open_citations_panel', { count: citations.length });
  } catch (e) {
    console.error('Failed to parse citations', e);
  }
}

function closeCitationsPanel() {
  const panel = document.getElementById('citations-side-panel');
  panel.classList.add('translate-x-full');
  setTimeout(() => {
    panel.classList.add('hidden');
  }, 300);

  const toast = document.getElementById('global-toast');
  if (toast) {
    toast.classList.remove('right-[25rem]');
    toast.classList.add('right-4');
  }
}

function openCitationContent(content, title, score, rawData) {
  const modal = document.getElementById('citation-modal');
  const modalTitle = document.getElementById('citation-modal-title');
  const modalContent = document.getElementById('citation-modal-content');
  const modalScore = document.getElementById('citation-modal-score');

  if (modalTitle) modalTitle.textContent = title;
  if (modalScore) modalScore.textContent = `Confidence Score: ${score}%`;
  
  if (modalContent) {
    modalContent.innerHTML = '';
    
    // Check if rawData or title mentions an image filename
    let imgName = null;
    if (rawData) {
      if (typeof rawData.source === 'string' && rawData.source.match(/\.(png|jpe?g|webp)$/i)) {
        imgName = rawData.source.split(/[/\\]/).pop();
      } else if (typeof rawData.image_name === 'string') {
        imgName = rawData.image_name;
      } else if (typeof rawData.path === 'string') {
        imgName = rawData.path.split(/[/\\]/).pop();
      }
    }
    if (!imgName) {
      const strToSearch = `${title} ${content} ${JSON.stringify(rawData || {})}`;
      const imgMatch = strToSearch.match(/(?:chart|img|table)_[^\s"'<>\\]+\.(?:png|jpe?g|webp)/i);
      if (imgMatch) {
        imgName = imgMatch[0];
      }
    }
    
    if (imgName) {
      const imgBlock = document.createElement('div');
      imgBlock.className = 'mb-4 flex flex-col items-center p-3 bg-surface-container-low rounded-xl border border-outline-variant/30 gap-2';
      imgBlock.innerHTML = `
        <div class="relative max-w-full flex items-center justify-center">
          <img src="/api/v1/images/${imgName}" alt="${title}" class="max-h-[40vh] max-w-full object-contain rounded-lg shadow-xs" />
        </div>
        <div class="flex items-center gap-3 mt-1">
          <span class="text-[10px] text-secondary font-mono">${imgName}</span>
          <button type="button" onclick="closeCitationModal(); previewArtifact('${title}', 1, 'chart', '/api/v1/images/${imgName}')" class="px-2.5 py-1 rounded-md bg-primary text-on-primary text-[11px] font-semibold hover:bg-primary-container transition-colors flex items-center gap-1 shadow-xs cursor-pointer">
            <span class="material-symbols-outlined text-xs">zoom_in</span>
            <span>Open in Full Inspector</span>
          </button>
        </div>
      `;
      modalContent.appendChild(imgBlock);
    }
    
    const textElem = document.createElement('div');
    textElem.className = 'text-xs text-on-surface font-body leading-relaxed whitespace-pre-wrap';
    textElem.textContent = content;
    modalContent.appendChild(textElem);
  }
  
  modal.classList.remove('hidden');
  logTelemetry('open_citation_modal', { title, score });
}

function closeCitationModal() {
  document.getElementById('citation-modal').classList.add('hidden');
}

function switchMainTab(tabId) {
  const workspace = document.getElementById('workspace');
  const aboutPage = document.getElementById('about-page');
  const btnHome = document.getElementById('nav-home-btn');
  const btnAbout = document.getElementById('nav-about-btn');

  const activeClasses = ['text-primary', 'border-primary'];
  const inactiveClasses = ['text-secondary', 'border-transparent', 'hover:text-on-surface', 'hover:border-surface-variant'];

  if (tabId === 'home') {
    workspace.classList.remove('hidden');
    aboutPage.classList.add('hidden');
    
    btnHome.classList.add(...activeClasses);
    btnHome.classList.remove(...inactiveClasses);
    
    btnAbout.classList.add(...inactiveClasses);
    btnAbout.classList.remove(...activeClasses);
  } else if (tabId === 'about') {
    workspace.classList.add('hidden');
    aboutPage.classList.remove('hidden');
    
    btnAbout.classList.add(...activeClasses);
    btnAbout.classList.remove(...inactiveClasses);
    
    btnHome.classList.add(...inactiveClasses);
    btnHome.classList.remove(...activeClasses);
  }
}

window.openCitationsPanel = openCitationsPanel;
window.closeCitationsPanel = closeCitationsPanel;
window.openCitationContent = openCitationContent;
window.closeCitationModal = closeCitationModal;
window.switchMainTab = switchMainTab;

function toggleExportMenu() {
  const dropdown = document.getElementById('export-dropdown');
  if (dropdown) {
    dropdown.classList.toggle('hidden');
  }
}

async function exportChat(format) {
  toggleExportMenu(); // Close menu
  
  if (!window.chatHistory || window.chatHistory.length === 0) {
    // If chatHistory is empty, check if there is an active memo or query response on screen
    const traceBubbles = document.querySelectorAll('#chat-trace-container .prose');
    if (traceBubbles.length > 0) {
      window.chatHistory = Array.from(traceBubbles).map((bubble, idx) => ({
        query: `Research Query #${idx + 1}`,
        response: bubble.innerText || bubble.textContent,
        citations: []
      }));
    } else if (latestMemoMarkdown && latestMemoMarkdown !== 'No memo generated yet.') {
      window.chatHistory = [{
        query: 'Quantitative Investment Memorandum',
        response: latestMemoMarkdown,
        citations: []
      }];
    } else {
      showToast('Chat history is empty. Please execute a query first.');
      return;
    }
  }
  
  showToast(`Generating ${format.toUpperCase()} export...`);
  
  try {
    const res = await fetch('/api/v1/export/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        format: format,
        chat_history: window.chatHistory
      })
    });
    
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to export' }));
      throw new Error(err.detail || 'Failed to export');
    }
    
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `OmniBrain_Export_${new Date().toISOString().split('T')[0]}.${format}`;
    document.body.appendChild(a);
    a.click();
    
    setTimeout(() => {
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    }, 200);
    
    showToast(`${format.toUpperCase()} export downloaded successfully!`);
    logTelemetry('export_chat', { format, count: window.chatHistory.length });
  } catch (error) {
    console.error('Export failed:', error);
    showToast(`Failed to export: ${error.message}`);
  }
}

window.toggleExportMenu = toggleExportMenu;
window.exportChat = exportChat;

// Close dropdown when clicking outside
document.addEventListener('click', (event) => {
  const btn = document.getElementById('export-menu-button');
  const dropdown = document.getElementById('export-dropdown');
  if (btn && dropdown && !btn.contains(event.target) && !dropdown.contains(event.target)) {
    dropdown.classList.add('hidden');
  }
});

function initInspectAndModals() {
  loadWorkspaceArtifacts();

  // Backdrop click dismissal for modals
  const artifactModal = document.getElementById('artifact-modal');
  if (artifactModal) {
    artifactModal.addEventListener('click', (e) => {
      if (e.target === artifactModal) closeArtifactModal();
    });
  }

  const citationModal = document.getElementById('citation-modal');
  if (citationModal) {
    citationModal.addEventListener('click', (e) => {
      if (e.target === citationModal) closeCitationModal();
    });
  }

  const dagModal = document.getElementById('dag-modal');
  if (dagModal) {
    dagModal.addEventListener('click', (e) => {
      if (e.target === dagModal) closeDAGModal();
    });
  }

  const inspectBtn = document.getElementById('inspect-all-btn');
  if (inspectBtn) {
    inspectBtn.addEventListener('click', (e) => {
      e.preventDefault();
      inspectAllArtifacts();
    });
  }
}

// Automatically initialize workspace artifacts and modal listeners
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initInspectAndModals);
} else {
  initInspectAndModals();
}

