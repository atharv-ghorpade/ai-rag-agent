document.addEventListener('DOMContentLoaded', () => {

  // --- Tab Navigation ---
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      document.getElementById(targetId)?.classList.add('active');

      if (targetId === 'tab-corpus' && !corpusLoaded) {
        loadCorpus();
      }
    });
  });

  // --- Fetch System Status ---
  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        document.getElementById('header-status').textContent = data.vector_store_type ? `${data.vector_store_type.toUpperCase()} Online` : 'FAISS Online';
        document.getElementById('header-chunks').textContent = data.total_indexed_chunks || 519;
        document.getElementById('header-embed').textContent = (data.embedding_model || '').split('/').pop();
      }
    } catch (err) {
      console.warn('Backend status unreachable:', err);
      document.getElementById('header-status').textContent = 'Connecting...';
    }
  }
  fetchStatus();

  // --- Tab 1: Q&A Assistant ---
  const qaForm = document.getElementById('qa-form');
  const qaInput = document.getElementById('qa-input');
  const qaBtnText = document.getElementById('qa-btn-text');
  const qaSpinner = document.getElementById('qa-spinner');
  const qaSubmitBtn = document.getElementById('qa-submit-btn');
  const qaResultContainer = document.getElementById('qa-result-container');
  const qaAnswerText = document.getElementById('qa-answer-text');
  const qaLatency = document.getElementById('qa-latency');
  const qaCitationsGrid = document.getElementById('qa-citations-grid');

  qaForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = qaInput.value.trim();
    if (!query) return;

    // UI Loading state
    qaSubmitBtn.disabled = true;
    qaBtnText.textContent = 'Analyzing...';
    qaSpinner.classList.remove('hidden');
    qaResultContainer.classList.add('hidden');

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query, retrieval_k: 10, final_k: 4 })
      });

      if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
      const data = await res.json();

      qaAnswerText.innerHTML = renderMarkdown(data.answer || 'No answer generated.');
      qaLatency.textContent = `${data.latency_ms} ms`;

      // Citations
      qaCitationsGrid.innerHTML = '';
      if (data.citations && data.citations.length > 0) {
        data.citations.forEach(c => {
          const card = document.createElement('div');
          card.className = 'citation-card';
          card.innerHTML = `
            <div class="citation-info">
              <span class="citation-file">${escapeHtml(c.source_file)}</span>
              <span class="citation-page">Official Record • Page ${c.page}</span>
            </div>
            <span class="confidence-tag">${c.score > 0 ? '+' : ''}${c.score.toFixed(3)}</span>
          `;
          qaCitationsGrid.appendChild(card);
        });
      } else {
        qaCitationsGrid.innerHTML = '<span style="color:var(--text-dim); font-size:0.85rem;">No citations available.</span>';
      }

      qaResultContainer.classList.remove('hidden');
    } catch (err) {
      alert(`Query failed: ${err.message}`);
    } finally {
      qaSubmitBtn.disabled = false;
      qaBtnText.textContent = 'Generate Answer';
      qaSpinner.classList.add('hidden');
    }
  });

  // Suggestion chips
  document.querySelectorAll('.chip:not(.compare-chip)').forEach(chip => {
    chip.addEventListener('click', () => {
      qaInput.value = chip.getAttribute('data-query');
      qaForm.dispatchEvent(new Event('submit'));
    });
  });

  // --- Tab 2: Comparison (Vector vs Reranker) ---
  const compareForm = document.getElementById('compare-form');
  const compareInput = document.getElementById('compare-input');
  const compareSubmitBtn = document.getElementById('compare-submit-btn');
  const compareBtnText = document.getElementById('compare-btn-text');
  const compareSpinner = document.getElementById('compare-spinner');
  const compareResults = document.getElementById('compare-results');
  const stage1List = document.getElementById('stage1-list');
  const stage2List = document.getElementById('stage2-list');

  compareForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = compareInput.value.trim();
    if (!query) return;

    compareSubmitBtn.disabled = true;
    compareBtnText.textContent = 'Searching...';
    compareSpinner.classList.remove('hidden');
    compareResults.classList.add('hidden');

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, retrieval_k: 8, final_k: 4 })
      });

      if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
      const data = await res.json();

      // Render Stage 1 (Dense Vector Search)
      stage1List.innerHTML = '';
      data.vector_search_top_k.forEach((item, idx) => {
        const div = document.createElement('div');
        div.className = 'result-item';
        div.innerHTML = `
          <div class="result-meta">
            <span class="rank-badge ${idx === 0 ? 'top' : 'normal'}">Stage 1 Rank #${idx + 1}</span>
            <span class="score-badge">Cosine: ${item.score.toFixed(4)}</span>
          </div>
          <div class="result-text">${escapeHtml(item.text)}</div>
        `;
        stage1List.appendChild(div);
      });

      // Render Stage 2 (Cross-Encoder Reranked)
      stage2List.innerHTML = '';
      data.reranked_top_k.forEach((item) => {
        const div = document.createElement('div');
        div.className = 'result-item';
        const sourceLabel = item.source_file ? `${escapeHtml(item.source_file)} (P.${item.page})` : 'General Policy';
        div.innerHTML = `
          <div class="result-meta">
            <span class="rank-badge ${item.rank === 1 ? 'top' : 'normal'}">Reranked #${item.rank}</span>
            <span class="score-badge" style="color:#a855f7;">Confidence: ${item.score.toFixed(4)}</span>
          </div>
          <div style="font-size:0.75rem; color:#818cf8; margin-bottom:0.35rem;">Source: ${sourceLabel}</div>
          <div class="result-text">${escapeHtml(item.raw_text || item.text)}</div>
        `;
        stage2List.appendChild(div);
      });

      compareResults.classList.remove('hidden');
    } catch (err) {
      alert(`Comparison failed: ${err.message}`);
    } finally {
      compareSubmitBtn.disabled = false;
      compareBtnText.textContent = 'Compare Stages';
      compareSpinner.classList.add('hidden');
    }
  });

  // Comparison Chips
  document.querySelectorAll('.compare-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      compareInput.value = chip.getAttribute('data-query');
      compareForm.dispatchEvent(new Event('submit'));
    });
  });

  // --- Tab 3: Corpus Explorer ---
  let corpusLoaded = false;
  const corpusList = document.getElementById('corpus-list');
  const corpusSearch = document.getElementById('corpus-search');
  let cachedDocs = [];

  async function loadCorpus(search = '') {
    try {
      const url = search ? `/api/documents?limit=60&search=${encodeURIComponent(search)}` : '/api/documents?limit=60';
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
      const data = await res.json();
      cachedDocs = data.documents || [];
      renderCorpus(cachedDocs);
      corpusLoaded = true;
    } catch (err) {
      corpusList.innerHTML = `<div style="color:#ef4444; padding:2rem; text-align:center;">Failed to load corpus: ${err.message}</div>`;
    }
  }

  function renderCorpus(docs) {
    corpusList.innerHTML = '';
    if (!docs || docs.length === 0) {
      corpusList.innerHTML = '<div style="color:var(--text-muted); padding:2rem; text-align:center;">No matching documents found.</div>';
      return;
    }

    docs.forEach(doc => {
      const div = document.createElement('div');
      div.className = 'result-item';
      div.innerHTML = `
        <div class="result-meta">
          <span style="font-weight:600; color:#c7d2fe;">${escapeHtml(doc.source_file || 'Document')}</span>
          <span class="score-badge">Page ${doc.page} • ${escapeHtml(doc.academic_year || '')}</span>
        </div>
        <div style="font-size:0.75rem; color:#60a5fa; margin-bottom:0.35rem;">${escapeHtml(doc.program || '')}</div>
        <div class="result-text">${escapeHtml(doc.text)}</div>
      `;
      corpusList.appendChild(div);
    });
  }

  let searchTimeout;
  corpusSearch.addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      loadCorpus(e.target.value.trim());
    }, 350);
  });

  // Helper: Escape HTML to prevent injection and errors
  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
  const escapeHTML = escapeHtml;

  function renderMarkdown(md) {
    if (!md) return 'No answer generated.';
    if (window.marked && typeof window.marked.parse === 'function') {
      try {
        if (typeof window.marked.setOptions === 'function') {
          window.marked.setOptions({ gfm: true, breaks: true });
        }
        return window.marked.parse(md);
      } catch (e) {
        console.warn('Marked parse failed, using fallback:', e);
      }
    }

    // Robust Line-by-Line Fallback Markdown Parser
    const lines = md.split(/\r?\n/);
    const output = [];
    let inList = false;
    let inTable = false;
    let tableHtml = '';

    for (let rawLine of lines) {
      const line = rawLine.trim();

      // Table row
      if (line.startsWith('|') && line.endsWith('|')) {
        if (inList) {
          output.push('</ul>');
          inList = false;
        }
        if (line.includes('---')) continue;
        const cells = line.slice(1, -1).split('|').map(c => escapeHtml(c.trim()));
        if (!inTable) {
          inTable = true;
          tableHtml = '<table><thead><tr>' + cells.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
        } else {
          tableHtml += '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
        }
        continue;
      } else if (inTable) {
        tableHtml += '</tbody></table>';
        output.push(tableHtml);
        inTable = false;
        tableHtml = '';
      }

      // Empty line
      if (!line) {
        if (inList) {
          output.push('</ul>');
          inList = false;
        }
        continue;
      }

      // Headings
      if (line.startsWith('## ')) {
        if (inList) {
          output.push('</ul>');
          inList = false;
        }
        output.push(`<h2>${escapeHtml(line.slice(3).trim())}</h2>`);
        continue;
      }
      if (line.startsWith('# ')) {
        if (inList) {
          output.push('</ul>');
          inList = false;
        }
        output.push(`<h1>${escapeHtml(line.slice(2).trim())}</h1>`);
        continue;
      }

      // List items (- or • or * or numbers like 1.)
      const listMatch = line.match(/^([•\-\*]|\d+[\.\)])\s+(.*)/);
      if (listMatch) {
        if (!inList) {
          output.push('<ul>');
          inList = true;
        }
        output.push(`<li>${escapeHtml(listMatch[2])}</li>`);
        continue;
      }

      // Regular text paragraph
      if (inList) {
        output.push('</ul>');
        inList = false;
      }
      output.push(`<p>${escapeHtml(line)}</p>`);
    }

    if (inList) output.push('</ul>');
    if (inTable) output.push(tableHtml + '</tbody></table>');

    return output.join('\n');
  }

});
