// ── Toast notification system ─────────────────────────────────────────────────
function showToast(message, type = 'success') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const colors = {
    success: 'bg-green-50 border-green-200 text-green-800',
    error:   'bg-red-50 border-red-200 text-red-800',
    info:    'bg-slate-50 border-slate-200 text-slate-800',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
  };
  const icons = {
    success: '✓',
    error:   '✗',
    info:    'ℹ',
    warning: '⚠',
  };

  const toast = document.createElement('div');
  toast.className = `toast flex items-center gap-3 px-4 py-3 rounded-xl border shadow-lg text-sm font-medium min-w-[260px] max-w-[360px] ${colors[type] || colors.info}`;
  toast.innerHTML = `
    <span class="font-bold text-base leading-none">${icons[type] || '•'}</span>
    <span class="flex-1">${message}</span>
    <button onclick="this.closest('.toast').remove()" class="text-current opacity-50 hover:opacity-100 ml-1 text-xs">×</button>
  `;

  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// ── Chip input component ──────────────────────────────────────────────────────
function chipInput(initial = []) {
  return {
    items:   [...initial],
    newItem: '',
    add() {
      const v = this.newItem.trim().toLowerCase();
      if (v && !this.items.includes(v)) { this.items.push(v); }
      this.newItem = '';
    },
    remove(i) { this.items.splice(i, 1); }
  };
}

// ── Sort table helper ─────────────────────────────────────────────────────────
function sortTable(data, key, direction = 'asc') {
  return [...data].sort((a, b) => {
    const va = a[key], vb = b[key];
    if (va == null) return 1;
    if (vb == null) return -1;
    const cmp = typeof va === 'string' ? va.localeCompare(vb) : va - vb;
    return direction === 'asc' ? cmp : -cmp;
  });
}

// ── Logic rule preview ────────────────────────────────────────────────────────
function previewLogicRule(required, optional, blacklist, xorA, xorB) {
  const parts = [];
  if (required.length)  parts.push(required.join(' AND '));
  if (optional.length)  parts.push(`(${optional.join(' OR ')})`);
  if (blacklist.length) parts.push(`NOT (${blacklist.join(', ')})`);
  if (xorA && xorB)     parts.push(`(${xorA} XOR ${xorB})`);
  return parts.length ? parts.join(' → ') : 'Chưa có rule';
}

// ── Confirm danger actions ────────────────────────────────────────────────────
function confirmDanger(msg, action) {
  if (confirm(msg)) action();
}
