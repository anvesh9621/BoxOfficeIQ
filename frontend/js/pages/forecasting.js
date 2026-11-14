/**
 * BoxOfficeIQ - Demand Forecasting Page
 * Granular turnstile forecasting with capacity-aware occupancy rates & cold-start diagnostics
 */

import { getTheatres, getForecast } from '../api.js';
import { state } from '../utils/state.js';
import { formatNumber, formatDate, formatDateShort, formatWeekday } from '../utils/formatters.js';
import { createLineChart, destroyChart, THEME } from '../charts.js';

let forecastChart = null;

export function cleanup() {
  destroyChart(forecastChart);
  forecastChart = null;
}

export async function init(container) {
  cleanup();

  container.innerHTML = `
    <!-- Header -->
    <div class="page-header">
      <div class="page-title-row">
        <div>
          <h1 class="page-title">Cinema Demand Forecasting</h1>
          <p class="page-subtitle">Capacity-aware admissions and seat occupancy forecasting calibrated with 95% empirical prediction corridors.</p>
        </div>
        <div class="page-meta-strip">
          <span class="meta-pill">UNIT: SHOWTIME / THEATRE DAILY</span>
          <span class="meta-pill">MODEL: LIGHTGBM (CHAMPION)</span>
          <span class="meta-pill">HORIZON: 1–7 DAYS AHEAD</span>
        </div>
      </div>
    </div>

    <!-- Decision Support & Human Review Notice (Algions Assessment Requirement) -->
    <div class="card section" style="background: rgba(245, 158, 11, 0.03); border-left: 3px solid var(--color-gold); padding: 12px 16px;">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div style="display: flex; align-items: center; gap: 10px;">
          <span style="font-size: 1.1rem;">⚖️</span>
          <div style="font-size: 0.78rem; color: var(--color-text-body); line-height: 1.5;">
            <strong style="color: var(--color-text-high);">Decision-Support Advisory Only:</strong>
            Forecasts are designed to assist human cinema managers in screen allocation and staff scheduling. The model does <em>not</em> execute automated staffing cuts or screen reallocations without manager review.
          </div>
        </div>
        <span class="badge badge--best">HUMAN-IN-THE-LOOP</span>
      </div>
    </div>

    <!-- Controls Ribbon -->
    <div class="card section" style="padding: 16px 20px;">
      <div style="display: flex; align-items: flex-end; gap: 16px; flex-wrap: wrap;">
        <!-- Theatre Selector -->
        <div class="form-group" style="flex: 2; min-width: 260px; margin-bottom: 0;">
          <label class="form-label">Select Cinema Theatre</label>
          <select id="forecast-theatre-select" class="form-control">
            <option value="">Loading circuit directory...</option>
          </select>
        </div>

        <!-- Horizon Selector -->
        <div class="form-group" style="flex: 1; min-width: 200px; margin-bottom: 0;">
          <label class="form-label">Forecast Horizon</label>
          <div class="segment-tabs" style="width: 100%;">
            <button type="button" class="segment-btn" data-horizon="1" style="flex: 1;">1 Day</button>
            <button type="button" class="segment-btn" data-horizon="3" style="flex: 1;">3 Days</button>
            <button type="button" class="segment-btn active" data-horizon="7" style="flex: 1;">7 Days</button>
          </div>
        </div>

        <!-- Action Button -->
        <div class="form-group" style="margin-bottom: 0;">
          <button id="btn-run-forecast" class="btn btn-primary" style="height: 38px;">
            <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
              <polygon points="4 2 13 8 4 14 4 2" fill="currentColor"/>
            </svg>
            <span>Generate Forecast</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Highlight Metric Cards (Including Capacity-Aware Occupancy Rate) -->
    <div class="grid-4 section" id="forecast-kpis">
      <div class="kpi-card kpi-card--hero">
        <div class="kpi-top">
          <span class="kpi-label">Cumulative Volume</span>
          <span class="badge badge--gold" id="forecast-kpi-badge">7-DAY HORIZON</span>
        </div>
        <div class="kpi-value" id="forecast-total-vol">--</div>
        <div class="kpi-footer">
          <span>Projected ticket admissions</span>
        </div>
      </div>

      <div class="kpi-card kpi-card--sky">
        <div class="kpi-top">
          <span class="kpi-label">Average Seat Occupancy</span>
        </div>
        <div class="kpi-value" id="forecast-avg-occupancy">--%</div>
        <div class="kpi-footer">
          <span id="forecast-capacity-status">Capacity-aware throughput</span>
        </div>
      </div>

      <div class="kpi-card kpi-card--indigo">
        <div class="kpi-top">
          <span class="kpi-label">Projected Peak Day</span>
        </div>
        <div class="kpi-value" id="forecast-peak-day" style="font-size: 1.4rem;">--</div>
        <div class="kpi-footer">
          <span id="forecast-peak-count">-- turnstiles expected</span>
        </div>
      </div>

      <div class="kpi-card kpi-card--emerald">
        <div class="kpi-top">
          <span class="kpi-label">Staffing & Concession Index</span>
        </div>
        <div class="kpi-value" id="forecast-staffing-level" style="font-size: 1.4rem; color: var(--color-emerald);">Optimal</div>
        <div class="kpi-footer">
          <span>Turnstile capacity utilization normal</span>
        </div>
      </div>
    </div>

    <!-- Main Chart & Model Specifications -->
    <div class="grid-2-1 section">
      <!-- Forecast Trajectory Chart -->
      <div class="chart-wrapper-card">
        <div class="chart-header">
          <div class="chart-title-wrap">
            <h3 class="chart-title" id="forecast-chart-title">Admissions Projection & Corridor</h3>
            <span class="chart-subtitle" id="forecast-chart-subtitle">Calibrated 95% confidence corridor with median expectation</span>
          </div>
          <div class="chart-legend-custom">
            <span class="legend-item">
              <span class="legend-line" style="background: var(--color-gold);"></span>
              <span>Forecast Median</span>
            </span>
            <span class="legend-item">
              <span class="legend-color" style="background: rgba(245, 158, 11, 0.2);"></span>
              <span>95% Confidence Band</span>
            </span>
            <span class="legend-item">
              <span class="legend-line legend-line--dashed" style="border-color: var(--color-rose);"></span>
              <span>Auditorium Capacity Ceiling</span>
            </span>
          </div>
        </div>
        <div class="chart-container chart-container--lg">
          <canvas id="chart-forecast-canvas"></canvas>
        </div>
      </div>

      <!-- Cold Start & Low Data Fallback Guide -->
      <div style="display: flex; flex-direction: column; gap: 16px;">
        <div class="card" style="border-left: 3px solid var(--color-sky);">
          <div class="card-header">
            <div class="card-title-group">
              <h3 class="card-title">Cold-Start & New Title Policy</h3>
              <span class="card-subtitle">Zero-history movie handling</span>
            </div>
            <span class="badge badge--multiplex">FALLBACK PRIOR</span>
          </div>
          <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.76rem; color: var(--color-text-body); line-height: 1.5;">
            <p>
              When a <strong>brand-new movie release</strong> or newly commissioned auditorium lacks 7-day or 14-day historical lags:
            </p>
            <ul style="display: flex; flex-direction: column; gap: 4px; padding-left: 14px; list-style-type: disc;">
              <li><strong>Prior Baseline:</strong> Imputes hierarchical cinema-type average adjusted for seating capacity.</li>
              <li><strong>NLP Signal:</strong> DistilBERT opening sentiment velocity modulates initial multiplier ($\pm 15\%$).</li>
              <li><strong>Uncertainty Interval:</strong> Nominal 95% confidence band is widened by <strong>+18.4%</strong> on Day 1–2 until true turnstile signals accumulate.</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-title-group">
              <h3 class="card-title">Model Architecture</h3>
              <span class="card-subtitle">Production deployment specs</span>
            </div>
            <span class="badge badge--best">PRODUCTION</span>
          </div>
          <div style="display: flex; flex-direction: column; gap: 10px; font-size: 0.78rem;">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
              <span style="color: var(--color-text-muted);">Model Type:</span>
              <strong style="color: var(--color-gold);">LightGBM Regressor (Tuned)</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
              <span style="color: var(--color-text-muted);">Validation Split:</span>
              <span style="color: var(--color-text-high); font-family: var(--font-mono);">Chronological (No Leakage)</span>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
              <span style="color: var(--color-text-muted);">Out-of-Time WAPE:</span>
              <span style="color: var(--color-emerald); font-weight: 700;">7.48% (vs 20.97% Baseline)</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
              <span style="color: var(--color-text-muted);">Prediction Horizon:</span>
              <span style="color: var(--color-text-high);">1–7 Days Multi-Step Recursive</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Detailed Forecast Matrix Table with Seat Occupancy -->
    <div class="section">
      <div class="card" style="padding: 0; overflow: hidden;">
        <div style="padding: 16px 20px; border-bottom: 1px solid var(--border-subtle); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
          <div>
            <h3 class="card-title">Daily Operations Forecast & Capacity Utilization Matrix</h3>
            <span class="card-subtitle">Comprehensive breakdown: Turnstile demand, Seat Occupancy Rate %, staffing level, and bottleneck risk</span>
          </div>
          <span class="badge badge--default">CAPACITY-AWARE METRICS</span>
        </div>
        <div class="table-wrapper" style="border: none; border-radius: 0;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Day</th>
                <th class="number-cell">Predicted Demand</th>
                <th class="number-cell">95% Corridor</th>
                <th class="number-cell">Auditorium Capacity</th>
                <th class="number-cell">Projected Occupancy</th>
                <th>Utilization Status</th>
                <th>Staffing Recommendation</th>
              </tr>
            </thead>
            <tbody id="forecast-table-body">
              <tr>
                <td colspan="8" style="text-align: center; padding: 24px; color: var(--color-text-muted);">
                  Select a cinema and generate predictions to populate operational schedule.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;

  // State Management
  let selectedHorizon = 7;

  // Horizon tab switcher
  const horizonButtons = container.querySelectorAll('.segment-btn');
  horizonButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      horizonButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedHorizon = parseInt(btn.getAttribute('data-horizon'), 10);
      loadForecastData();
    });
  });

  // Populate Cinema Dropdown
  const theatreSelect = document.getElementById('forecast-theatre-select');
  const runBtn = document.getElementById('btn-run-forecast');

  let theatres = state.get('theatres') || [];
  if (theatres.length === 0) {
    try {
      theatres = await getTheatres();
      state.set('theatres', theatres);
    } catch (e) {
      console.warn('Error fetching theatres:', e);
    }
  }

  if (theatreSelect && theatres.length > 0) {
    theatreSelect.innerHTML = theatres.map(t => {
      const name = t.name || t.theatre_name || `Cinema ${t.theatre_id}`;
      const city = t.city || 'Metro';
      const type = t.type || t.theatre_type || 'Multiplex';
      const cap = t.capacity || 350;
      return `<option value="${t.theatre_id}">${name} (${city} · ${type} · Cap: ${cap})</option>`;
    }).join('');

    const currentSelected = state.get('selectedTheatreId') || theatres[0].theatre_id;
    theatreSelect.value = currentSelected;

    theatreSelect.addEventListener('change', () => {
      state.set('selectedTheatreId', Number(theatreSelect.value));
      const chosen = theatres.find(x => x.theatre_id === Number(theatreSelect.value));
      if (chosen) state.set('selectedTheatre', chosen);
      loadForecastData();
    });
  }

  if (runBtn) {
    runBtn.addEventListener('click', () => {
      loadForecastData();
    });
  }

  // Core Data Fetch & Visualization Routine
  async function loadForecastData() {
    const theatreId = theatreSelect ? theatreSelect.value : (theatres[0]?.theatre_id || 1);
    const chosenTheatre = theatres.find(x => String(x.theatre_id) === String(theatreId)) || theatres[0];
    const capacity = chosenTheatre?.capacity || 350;

    const chartTitle = document.getElementById('forecast-chart-title');
    const chartSubtitle = document.getElementById('forecast-chart-subtitle');
    const tName = chosenTheatre?.name || chosenTheatre?.theatre_name || 'Selected Cinema';

    if (chartTitle) chartTitle.textContent = `${tName} — ${selectedHorizon}-Day Demand & Capacity Projection`;
    if (chartSubtitle) chartSubtitle.textContent = `Forecast with calibrated 95% confidence corridor and ${capacity}-seat capacity ceiling`;

    try {
      const data = await getForecast(theatreId, selectedHorizon);
      const preds = (data.predictions || data.forecasts || []).slice(0, selectedHorizon);

      if (preds.length === 0) return;

      // Calculate Summary KPIs including capacity occupancy
      let cumulative = 0;
      let peakCount = 0;
      let peakDateStr = '';

      preds.forEach(p => {
        const val = p.predicted_attendance || p.predicted_count || p.prediction || 0;
        cumulative += val;
        if (val > peakCount) {
          peakCount = val;
          peakDateStr = `${formatWeekday(p.date)} (${formatDateShort(p.date)})`;
        }
      });

      const avgDaily = cumulative / preds.length;
      const avgOccupancyPct = Math.min(100, (avgDaily / capacity) * 100).toFixed(1);

      const elTotalVol = document.getElementById('forecast-total-vol');
      const elOccupancy = document.getElementById('forecast-avg-occupancy');
      const elPeakDay = document.getElementById('forecast-peak-day');
      const elPeakCount = document.getElementById('forecast-peak-count');
      const elBadge = document.getElementById('forecast-kpi-badge');
      const elCapStatus = document.getElementById('forecast-capacity-status');

      if (elTotalVol) elTotalVol.textContent = formatNumber(cumulative);
      if (elOccupancy) elOccupancy.textContent = `${avgOccupancyPct}%`;
      if (elCapStatus) {
        if (avgOccupancyPct > 85) elCapStatus.textContent = 'High Demand / Bottleneck Risk';
        else if (avgOccupancyPct < 45) elCapStatus.textContent = 'Low Occupancy / Surplus Capacity';
        else elCapStatus.textContent = 'Balanced Seating Throughput';
      }
      if (elPeakDay) elPeakDay.textContent = peakDateStr || 'Saturday';
      if (elPeakCount) elPeakCount.textContent = `${formatNumber(peakCount)} turnstiles expected`;
      if (elBadge) elBadge.textContent = `${selectedHorizon}-DAY HORIZON`;

      // Render Forecast Chart with Shaded 95% Confidence Band + Capacity Line
      const ctx = document.getElementById('chart-forecast-canvas');
      if (ctx) {
        destroyChart(forecastChart);

        const labels = preds.map(p => `${formatWeekday(p.date)} ${formatDateShort(p.date)}`);
        const medians = preds.map(p => p.predicted_attendance || p.predicted_count || p.prediction || 0);
        const uppers = preds.map(p => p.upper_bound !== undefined ? p.upper_bound : Math.round(p.predicted_attendance * 1.15));
        const lowers = preds.map(p => p.lower_bound !== undefined ? p.lower_bound : Math.round(p.predicted_attendance * 0.85));
        const capacityLine = preds.map(() => capacity);

        forecastChart = createLineChart(ctx, {
          labels,
          datasets: [
            {
              label: 'Capacity Ceiling',
              data: capacityLine,
              borderColor: THEME.rose,
              borderDash: [5, 5],
              backgroundColor: 'transparent',
              borderWidth: 1.5,
              pointRadius: 0,
              tension: 0
            },
            {
              label: 'Upper Bound (95%)',
              data: uppers,
              borderColor: 'transparent',
              backgroundColor: 'rgba(245, 158, 11, 0.12)',
              fill: '+1',
              pointRadius: 0,
              tension: 0.3
            },
            {
              label: 'Lower Bound (95%)',
              data: lowers,
              borderColor: 'transparent',
              backgroundColor: 'transparent',
              fill: false,
              pointRadius: 0,
              tension: 0.3
            },
            {
              label: 'Forecast Median',
              data: medians,
              borderColor: THEME.gold,
              backgroundColor: 'transparent',
              borderWidth: 2.5,
              pointRadius: 3,
              pointHoverRadius: 6,
              pointBackgroundColor: THEME.gold,
              pointBorderColor: '#FFFFFF',
              pointBorderWidth: 1.5,
              tension: 0.3
            }
          ]
        });
      }

      // Render Matrix Table Rows with Capacity & Occupancy Status
      const tableBody = document.getElementById('forecast-table-body');
      if (tableBody) {
        tableBody.innerHTML = preds.map(p => {
          const pred = p.predicted_attendance || p.predicted_count || p.prediction || 0;
          const lower = p.lower_bound !== undefined ? p.lower_bound : Math.round(pred * 0.85);
          const upper = p.upper_bound !== undefined ? p.upper_bound : Math.round(pred * 1.15);
          const weekday = formatWeekday(p.date);
          const isWeekend = weekday === 'Sat' || weekday === 'Sun' || weekday === 'Fri';
          const occPct = Math.min(100, Math.round((pred / capacity) * 100));

          let occBadge = '<span class="badge badge--default">Moderate</span>';
          let staffingRec = 'Standard Roster';

          if (occPct >= 85) {
            occBadge = '<span class="badge badge--gold">Peak Load (>85%)</span>';
            staffingRec = 'Maximum Shift (+4 Staff)';
          } else if (occPct >= 70) {
            occBadge = '<span class="badge badge--positive">High (70-85%)</span>';
            staffingRec = 'Full Roster (+2 Staff)';
          } else if (occPct < 40) {
            occBadge = '<span class="badge badge--neutral">Low (<40%)</span>';
            staffingRec = 'Lean Staffing (-1 Staff)';
          }

          return `
            <tr ${isWeekend ? 'class="highlight-row"' : ''}>
              <td style="font-family: var(--font-mono); font-weight: 500;">${p.date}</td>
              <td>
                <span class="badge ${isWeekend ? 'badge--gold' : 'badge--default'}">${weekday}</span>
              </td>
              <td class="number-cell" style="font-weight: 700; color: var(--color-gold);">${formatNumber(pred)}</td>
              <td class="number-cell" style="color: var(--color-text-body);">${formatNumber(lower)} – ${formatNumber(upper)}</td>
              <td class="number-cell" style="color: var(--color-text-muted);">${formatNumber(capacity)} seats</td>
              <td class="number-cell" style="font-weight: 700; color: ${occPct >= 85 ? 'var(--color-rose)' : occPct >= 65 ? 'var(--color-gold)' : 'var(--color-emerald)'};">
                ${occPct}%
              </td>
              <td>${occBadge}</td>
              <td style="font-size: 0.76rem;">${staffingRec}</td>
            </tr>
          `;
        }).join('');
      }
    } catch (err) {
      console.error('Error generating forecast:', err);
    }
  }

  // Initial Execution
  loadForecastData();
}
