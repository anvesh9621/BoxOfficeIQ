/**
 * BoxOfficeIQ - Main Application Bootstrapper & SPA Router
 * Production-grade routing, telemetry sync & reactive view lifecycle
 */

import { state } from './utils/state.js';
import { checkApiHealth, getTheatres } from './api.js';

// Route Definitions
const ROUTES = {
  '#dashboard': () => import('./pages/dashboard.js'),
  '#forecasting': () => import('./pages/forecasting.js'),
  '#theatres': () => import('./pages/theatres.js'),
  '#performance': () => import('./pages/performance.js'),
  '#explainability': () => import('./pages/explainability.js'),
  '#sentiment': () => import('./pages/sentiment.js'),
  '#simulator': () => import('./pages/simulator.js'),
  '#assistant': () => import('./pages/assistant.js')
};

const PAGE_METADATA = {
  '#dashboard': { title: 'Executive Overview', context: 'CIRCUIT DEMAND' },
  '#forecasting': { title: 'Demand Forecasting', context: 'CIRCUIT DEMAND' },
  '#theatres': { title: 'Theatre Intelligence', context: 'CIRCUIT DEMAND' },
  '#performance': { title: 'Model Performance & Validation', context: 'PREDICTIVE MODELS' },
  '#explainability': { title: 'Explainable AI (TreeSHAP)', context: 'PREDICTIVE MODELS' },
  '#sentiment': { title: 'Sentiment Intelligence (DistilBERT)', context: 'PREDICTIVE MODELS' },
  '#simulator': { title: 'What-If Cinema Simulator', context: 'DECISION SUPPORT' },
  '#assistant': { title: 'CinePredict AI Copilot', context: 'DECISION SUPPORT' }
};

let currentPageModule = null;

/**
 * Toast Notification System
 */
export function showToast(message, type = 'info', title = '') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;

  const iconMap = {
    info: 'ℹ️',
    success: '✓',
    warning: '⚠',
    error: '✕'
  };

  const defaultTitles = {
    info: 'System Update',
    success: 'Completed',
    warning: 'Attention',
    error: 'Operation Error'
  };

  toast.innerHTML = `
    <span class="toast-icon">${iconMap[type] || 'ℹ️'}</span>
    <div class="toast-body">
      <div class="toast-title">${title || defaultTitles[type]}</div>
      <div class="toast-msg">${message}</div>
    </div>
    <button class="toast-close" aria-label="Dismiss notification">&times;</button>
  `;

  const closeBtn = toast.querySelector('.toast-close');
  closeBtn.addEventListener('click', () => dismissToast(toast));

  container.appendChild(toast);

  // Auto dismiss after 4s
  setTimeout(() => dismissToast(toast), 4000);
}

function dismissToast(toast) {
  if (!toast || toast.classList.contains('dismissing')) return;
  toast.classList.add('dismissing');
  setTimeout(() => {
    if (toast.parentNode) toast.parentNode.removeChild(toast);
  }, 240);
}

/**
 * Load SPA Route Page
 */
export async function loadPage(hash) {
  const mainContent = document.getElementById('main-content');
  const topbarTitle = document.getElementById('topbar-title');
  const contextTag = document.querySelector('.topbar-context-tag');
  const targetRoute = ROUTES[hash] ? hash : '#dashboard';

  // Highlight active link in sidebar
  document.querySelectorAll('.nav-link').forEach(link => {
    if (link.getAttribute('href') === targetRoute) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });

  // Update Topbar Title & Context Tag
  const meta = PAGE_METADATA[targetRoute] || { title: 'Executive Overview', context: 'CIRCUIT DEMAND' };
  if (topbarTitle) topbarTitle.textContent = meta.title;
  if (contextTag) contextTag.textContent = meta.context;

  // Update State
  state.set('activePage', targetRoute.replace('#', ''));

  // Cleanup current page if needed
  if (currentPageModule && typeof currentPageModule.cleanup === 'function') {
    try {
      currentPageModule.cleanup();
    } catch (e) {
      console.warn('Page cleanup error:', e);
    }
  }

  // Render high-fidelity skeleton loading state
  mainContent.innerHTML = `
    <div class="page-header">
      <div class="skeleton skeleton-title"></div>
      <div class="skeleton skeleton-text" style="width: 32%;"></div>
    </div>
    <div class="grid-4 section">
      <div class="skeleton skeleton-kpi"></div>
      <div class="skeleton skeleton-kpi"></div>
      <div class="skeleton skeleton-kpi"></div>
      <div class="skeleton skeleton-kpi"></div>
    </div>
    <div class="grid-2 section">
      <div class="skeleton skeleton-chart"></div>
      <div class="skeleton skeleton-chart"></div>
    </div>
  `;

  try {
    const moduleImporter = ROUTES[targetRoute];
    const module = await moduleImporter();
    currentPageModule = module;

    // Call init on page module
    await module.init(mainContent);
  } catch (err) {
    console.error(`Failed to load route [${targetRoute}]:`, err);
    mainContent.innerHTML = `
      <div class="card" style="text-align: center; padding: 48px 24px; max-width: 500px; margin: 40px auto;">
        <div style="font-size: 2rem; margin-bottom: 12px; color: var(--color-rose);">⚠</div>
        <h3 style="font-size: 1.1rem; margin-bottom: 8px;">View Render Exception</h3>
        <p style="font-size: 0.8rem; color: var(--color-text-body); margin-bottom: 18px;">
          ${err.message || 'An error occurred while loading this view component.'}
        </p>
        <button class="btn btn-primary btn-sm" onclick="window.location.reload()">Reload Application</button>
      </div>
    `;
    showToast('Failed to load page module. Check console for details.', 'error');
  }
}

/**
 * Topbar Live Telemetry Clock
 */
function startLiveClock() {
  const clockEl = document.getElementById('topbar-time');
  if (!clockEl) return;

  function update() {
    const now = new Date();
    clockEl.textContent = now.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    }) + ' IST';
  }
  update();
  setInterval(update, 1000);
}

/**
 * Check Backend API Health & Update Status Indicator
 */
async function syncApiStatus() {
  const dot = document.getElementById('api-status-dot');
  const text = document.getElementById('api-status-text');

  const isOnline = await checkApiHealth();
  state.set('apiOnline', isOnline);

  if (dot && text) {
    if (isOnline) {
      dot.className = 'status-indicator status-indicator--live';
      text.textContent = 'Backend Online (API)';
    } else {
      dot.className = 'status-indicator status-indicator--offline';
      text.textContent = 'Offline (Demo Mocks)';
    }
  }
}

/**
 * Initialize Application
 */
async function initApp() {
  const appLoader = document.getElementById('app-loader');
  const appShell = document.getElementById('app');

  startLiveClock();

  // Sidebar toggle for desktop/mobile
  const toggleBtn = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');
  const mainWrapper = document.querySelector('.main-wrapper');

  if (toggleBtn && sidebar && mainWrapper) {
    toggleBtn.addEventListener('click', () => {
      if (window.innerWidth <= 768) {
        sidebar.classList.toggle('mobile-open');
      } else {
        sidebar.classList.toggle('collapsed');
        mainWrapper.classList.toggle('full-width');
      }
    });
  }

  // Refresh button
  const refreshBtn = document.getElementById('refresh-btn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      showToast('Syncing real-time circuit predictions...', 'info');
      loadPage(window.location.hash || '#dashboard');
    });
  }

  // Listen to hash changes
  window.addEventListener('hashchange', () => {
    loadPage(window.location.hash);
  });

  // Preload initial theatres into state
  try {
    const theatres = await getTheatres();
    state.set('theatres', theatres);
    if (theatres.length > 0) {
      state.set('selectedTheatreId', theatres[0].theatre_id);
      state.set('selectedTheatre', theatres[0]);
    }
  } catch (e) {
    console.warn('Initial theatre fetch error:', e);
  }

  // Check API health
  await syncApiStatus();
  setInterval(syncApiStatus, 15000);

  // Initial Route Load
  const initialHash = window.location.hash || '#dashboard';
  await loadPage(initialHash);

  // Dismiss loader
  if (appLoader && appShell) {
    appLoader.classList.add('fade-out');
    appShell.style.display = 'flex';
    setTimeout(() => {
      appLoader.style.display = 'none';
    }, 450);
  }
}

// Auto-run on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
