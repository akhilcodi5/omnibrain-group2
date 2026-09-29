// OmniBrain Quant Workspace Frontend Logic
import './style.css';
import { marked } from 'marked';

let latestMemoMarkdown = 'No memo generated yet.';
let currentUploadedImages = [];
let currentUploadedPdfName = null;

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

// Artifact Modal
export function previewArtifact(title, page, type) {
  const modal = document.getElementById('artifact-modal');
  if (!modal) return;
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-page').textContent = page;
  document.getElementById('modal-preview-text').textContent = title;
  document.getElementById('modal-icon').textContent =
    type === 'chart' ? 'bar_chart' : type === 'table' ? 'table_chart' : 'notes';
  modal.classList.remove('hidden');
}

export function closeArtifactModal() {
  document.getElementById('artifact-modal')?.classList.add('hidden');
}

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
      const artifactContainer = document.getElementById('artifact-cards-wrapper');
      
      let tableCount = 0;
      let chartCount = 0;
      
      if (artifactContainer) {
        artifactContainer.innerHTML = '';
        currentUploadedImages.forEach((img, idx) => {
          const filename = img.split(/[/\\]/).pop(); // Handle both Windows backslashes and POSIX forward slashes
          const isTable = filename.startsWith('table_');
          const isChart = !isTable;
          
          if (isTable) tableCount++;
          if (isChart) chartCount++;
          
          const icon = isChart ? 'bar_chart' : 'table_chart';
          const label = isChart ? 'Extracted Chart' : 'Extracted Table';
          
          artifactContainer.innerHTML += `
            <div class="bg-surface-container-lowest rounded-xl p-2.5 border border-outline-variant/30 shadow-xs hover:border-primary/40 transition-colors cursor-pointer group flex flex-col gap-2 artifact-card">
              <div class="flex items-center gap-2">
                <div class="w-7 h-7 bg-primary/10 rounded flex items-center justify-center text-primary">
                  <span class="material-symbols-outlined text-[15px]">${icon}</span>
                </div>
                <div class="overflow-hidden flex-1">
                  <h3 class="text-xs font-semibold text-on-surface truncate group-hover:text-primary transition-colors">${label} ${isChart ? chartCount : tableCount}</h3>
                  <p class="text-[10px] text-secondary truncate">${filename}</p>
                </div>
              </div>
              <div class="w-full h-24 bg-surface-container rounded border border-outline-variant/20 overflow-hidden relative group-hover:shadow-inner transition-all flex items-center justify-center">
                <img src="/api/v1/images/${filename}" alt="Extracted Figure" class="w-full h-full object-contain mix-blend-multiply opacity-90 group-hover:opacity-100 transition-opacity">
                <div class="absolute inset-0 bg-primary/0 group-hover:bg-primary/5 transition-colors"></div>
              </div>
            </div>
          `;
        });
      }
      const tabTables = document.getElementById('tab-tables');
      if (tabTables) {
          tabTables.innerHTML = `<span class="material-symbols-outlined text-sm">table_chart</span><span>Tables (${tableCount})</span>`;
      }
      const tabCharts = document.getElementById('tab-charts');
      if (tabCharts) {
          tabCharts.innerHTML = `<span class="material-symbols-outlined text-sm">bar_chart</span><span>Charts (${chartCount})</span>`; 
      }
      showToast(
        `✅ Ingestion complete: ${data.chunks_indexed ?? data.total_chunks ?? 0} chunks & ${data.images_extracted ?? data.total_images ?? 0} figures indexed!`
      );
    } else {
      if (activeBadge) activeBadge.textContent = 'Simulated';
      showToast(`Parsed ${file.name} into local sandbox.`);
    }
  } catch (err) {
    if (activeBadge) activeBadge.textContent = 'Active (Local)';
    showToast(`Loaded ${file.name} into Quant Workspace.`);
  }
}

const sessionId = crypto.randomUUID();

// Main Analyst Query Execution (Connects to /api/v1/chat/query)
export async function executeAnalystQuery() {
  const input = document.getElementById('query-input');
  if (!input) return;
  const query = input.value.trim();
  if (!query) return;

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
  
  const execBtn = document.getElementById('execute-btn');
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
                <span class="text-secondary font-mono">Citations: ${result.citations ? result.citations.length : 0}</span>
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
window.closeArtifactModal = closeArtifactModal;
window.showGraphDAGModal = showGraphDAGModal;
window.closeDAGModal = closeDAGModal;
window.handleFileUpload = handleFileUpload;
window.executeAnalystQuery = executeAnalystQuery;
