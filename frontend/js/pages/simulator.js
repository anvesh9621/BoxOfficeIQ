/**
 * BoxOfficeIQ - What-If Cinema Simulator
 * Interactive scenario stress-testing for holidays, seasonal surges, and operational planning
 */

import { runSimulation, getTheatres } from '../api.js';
import { state } from '../utils/state.js';
import { formatNumber, formatCurrency, formatDateShort } from '../utils/formatters.js';
import { createBarChart, destroyChart, THEME } from '../charts.js';

let comparisonChart = null;

export function cleanup() {
  destroyChart(comparisonChart);
  comparisonChart = null;
}

export async function init(container) {
  cleanup();

  container.innerHTML = `
    <!-- Header -->
    <div class="page-header">
      <div class="page-title-row">
        <div>
          <h1 class="page-title">What-If Operational Scenario Simulator</h1>
          <p class="page-subtitle">Stress-test cinema attendance, staffing schedules, and concession inventory against festival surges and pricing shifts.</p>
        </div>
        <div class="page-meta-strip">
          <span class="meta-pill">DECISION SUPPORT ENGINE</span>
          <span class="meta-pill">SENSITIVITY ANALYSIS</span>
        </div>
      </div>
    </div>

    <!-- Simulator Layout: 2 Columns (Parameters Left, Analytics Right) -->
    <div class="grid-2-1 section" style="grid-template-columns: 360px 1fr; gap: 20px;">

      <!-- Parameters Configuration Panel -->
      <div class="card" style="display: flex; flex-direction: column; gap: 16px;">
        <div class="card-header" style="margin-bottom: 0;">
          <div class="card-title-group">
            <h3 class="card-title">Scenario Parameters</h3>
            <span class="card-subtitle">Adjust variables to evaluate operational impact</span>
          </div>
        </div>

        <!-- Cinema Selector -->
        <div class="form-group" style="margin-bottom: 0;">
          <label class="form-label">Target Cinema Hub</label>
          <select id="sim-theatre-select" class="form-control">
            <option value="">Loading circuit locations...</option>
          </select>
        </div>

        <!-- Scenario Quick Presets -->
        <div>
          <label class="form-label" style="margin-bottom: 6px;">Scenario Presets</label>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
            <button type="button" class="btn btn-secondary btn-sm preset-btn" data-holiday="2.2" data-demand="30">Diwali Surge (+120%)</button>
            <button type="button" class="btn btn-secondary btn-sm preset-btn" data-holiday="1.8" data-demand="40">Blockbuster Debut</button>
            <button type="button" class="btn btn-secondary btn-sm preset-btn" data-holiday="0.7" data-demand="-20">Monsoon Rain (-25%)</button>
            <button type="button" class="btn btn-secondary btn-sm preset-btn" data-holiday="1.0" data-demand="0">Standard Baseline</button>
          </div>
        </div>

        <!-- Holiday Multiplier Slider -->
        <div class="form-group" style="margin-bottom: 0;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <label class="form-label" style="margin-bottom: 0;">Holiday / Festival Surge Factor</label>
            <span id="sim-holiday-val" style="font-family: var(--font-mono); font-weight: 700; color: var(--color-gold); font-size: 0.85rem;">1.4x</span>
          </div>
          <input type="range" id="sim-holiday-slider" class="form-range" min="0.5" max="3.0" step="0.1" value="1.4" />
          <div style="display: flex; justify-content: space-between; font-size: 0.65rem; color: var(--color-text-muted); margin-top: 4px;">
            <span>0.5x (Quiet Lull)</span>
            <span>1.0x (Normal)</span>
            <span>3.0x (Peak Festival)</span>
          </div>
        </div>

        <!-- Organic Demand Adjustment Slider -->
        <div class="form-group" style="margin-bottom: 0;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <label class="form-label" style="margin-bottom: 0;">Marketing & Word-of-Mouth Shift</label>
            <span id="sim-demand-val" style="font-family: var(--font-mono); font-weight: 700; color: var(--color-sky); font-size: 0.85rem;">+15%</span>
          </div>
          <input type="range" id="sim-demand-slider" class="form-range" min="-40" max="60" step="5" value="15" />
          <div style="display: flex; justify-content: space-between; font-size: 0.65rem; color: var(--color-text-muted); margin-top: 4px;">
            <span>-40% (Adverse)</span>
            <span>0%</span>
            <span>+60% (Viral)</span>
          </div>
        </div>

        <!-- Simulation Horizon Tabs -->
        <div class="form-group" style="margin-bottom: 0;">
          <label class="form-label">Simulation Time Window</label>
          <div class="segment-tabs" style="width: 100%;">
            <button type="button" class="segment-btn sim-horizon-btn" data-h="1" style="flex: 1;">1 Day</button>
            <button type="button" class="segment-btn sim-horizon-btn" data-h="3" style="flex: 1;">3 Days</button>
            <button type="button" class="segment-btn sim-horizon-btn active" data-h="7" style="flex: 1;">7 Days</button>
          </div>
        </div>

        <!-- Run Scenario Button -->
        <button id="btn-run-simulation" class="btn btn-primary" style="width: 100%; height: 40px; margin-top: 6px;">
          <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="4 2 13 8 4 14 4 2" fill="currentColor"/>
          </svg>
          <span>Calculate Scenario Impact</span>
        </button>

        <div style="padding: 10px 12px; background: rgba(255, 255, 255, 0.02); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); font-size: 0.7rem; color: var(--color-text-muted); line-height: 1.5;">
          ℹ️ Multipliers are calibrated against historical holiday elasticities and verified regression baselines.
        </div>
      </div>

      <!-- Simulation Output Panel (Right Pane) -->
      <div style="display: flex; flex-direction: column; gap: 16px;">
        <!-- Top Impact Cards -->
        <div class="grid-3" id="sim-impact-row">
          <div class="card" style="border-left: 3px solid var(--color-gold);">
            <div style="font-size: 0.68rem; font-weight: 600; text-transform: uppercase; color: var(--color-text-muted);">Simulated Admissions</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: var(--color-gold); margin: 4px 0;" id="res-total-admissions">--</div>
            <div style="font-size: 0.72rem; color: var(--color-emerald); font-weight: 600;" id="res-admissions-delta">+0% vs baseline</div>
          </div>

          <div class="card" style="border-left: 3px solid var(--color-sky);">
            <div style="font-size: 0.68rem; font-weight: 600; text-transform: uppercase; color: var(--color-text-muted);">Staffing Requirement</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: var(--color-sky); margin: 4px 0;" id="res-staff-val">-- Staff</div>
            <div style="font-size: 0.72rem; color: var(--color-text-muted);" id="res-staff-delta">+0 attendants needed</div>
          </div>

          <div class="card" style="border-left: 3px solid var(--color-emerald);">
            <div style="font-size: 0.68rem; font-weight: 600; text-transform: uppercase; color: var(--color-text-muted);">F&B Concessions Stock</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: var(--color-emerald); margin: 4px 0;" id="res-inventory-val">-- Units</div>
            <div style="font-size: 0.72rem; color: var(--color-text-muted);" id="res-inventory-delta">+0 F&B items</div>
          </div>
        </div>

        <!-- Comparative Chart -->
        <div class="chart-wrapper-card">
          <div class="chart-header">
            <div class="chart-title-wrap">
              <h3 class="chart-title">Baseline vs Scenario Attendance Trajectory</h3>
              <span class="chart-subtitle">Side-by-side volume comparison across target simulation window</span>
            </div>
            <div class="chart-legend-custom">
              <span class="legend-item"><span class="legend-color" style="background: rgba(99, 102, 241, 0.6);"></span> Baseline Forecast</span>
              <span class="legend-item"><span class="legend-color" style="background: var(--color-gold);"></span> Simulated Scenario</span>
            </div>
          </div>
          <div class="chart-container chart-container--md">
            <canvas id="chart-sim-comparison"></canvas>
          </div>
        </div>

        <!-- Scenario Delta Breakdown Table -->
        <div class="card" style="padding: 0; overflow: hidden;">
          <div style="padding: 14px 20px; border-bottom: 1px solid var(--border-subtle); display: flex; justify-content: space-between; align-items: center;">
            <h3 class="card-title">Daily Operations Shift Ledger</h3>
            <span class="badge badge--default">SIMULATION LEDGER</span>
          </div>
          <div class="table-wrapper" style="border: none; border-radius: 0;">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Date & Horizon</th>
                  <th class="number-cell">Baseline Turnstiles</th>
                  <th class="number-cell">Simulated Demand</th>
                  <th class="number-cell">Admissions Delta</th>
                  <th class="number-cell">Shift %</th>
                </tr>
              </thead>
              <tbody id="sim-table-body">
                <tr><td colspan="5" style="text-align: center; padding: 20px; color: var(--color-text-muted);">Run calculation to populate operational ledger.</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  `;

  // State
  let selectedHorizon = 7;

  const theatreSelect = document.getElementById('sim-theatre-select');
  const holidaySlider = document.getElementById('sim-holiday-slider');
  const holidayVal = document.getElementById('sim-holiday-val');
  const demandSlider = document.getElementById('sim-demand-slider');
  const demandVal = document.getElementById('sim-demand-val');
  const btnRun = document.getElementById('btn-run-simulation');

  // Horizon switcher
  container.querySelectorAll('.sim-horizon-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      container.querySelectorAll('.sim-horizon-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedHorizon = parseInt(btn.getAttribute('data-h'), 10);
      executeSimulation();
    });
  });

  // Preset Buttons
  container.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const h = btn.getAttribute('data-holiday');
      const d = btn.getAttribute('data-demand');
      if (holidaySlider && holidayVal) {
        holidaySlider.value = h;
        holidayVal.textContent = `${Number(h).toFixed(1)}x`;
      }
      if (demandSlider && demandVal) {
        demandSlider.value = d;
        demandVal.textContent = `${Number(d) >= 0 ? '+' : ''}${d}%`;
      }
      executeSimulation();
    });
  });

  // Sliders
  if (holidaySlider && holidayVal) {
    holidaySlider.addEventListener('input', () => {
      holidayVal.textContent = `${Number(holidaySlider.value).toFixed(1)}x`;
    });
  }

  if (demandSlider && demandVal) {
    demandSlider.addEventListener('input', () => {
      const v = Number(demandSlider.value);
      demandVal.textContent = `${v >= 0 ? '+' : ''}${v}%`;
    });
  }

  // Populate Theatres
  let theatres = state.get('theatres') || [];
  if (theatres.length === 0) {
    try {
      theatres = await getTheatres();
      state.set('theatres', theatres);
    } catch (e) {
      console.warn('Theatres fetch error:', e);
    }
  }

  if (theatreSelect && theatres.length > 0) {
    theatreSelect.innerHTML = theatres.map(t => {
      const name = t.name || t.theatre_name || `Cinema ${t.theatre_id}`;
      const city = t.city || 'Metro';
      return `<option value="${t.theatre_id}">${name} (${city})</option>`;
    }).join('');

    theatreSelect.value = state.get('selectedTheatreId') || theatres[0].theatre_id;
    theatreSelect.addEventListener('change', () => {
      state.set('selectedTheatreId', Number(theatreSelect.value));
      executeSimulation();
    });
  }

  if (btnRun) {
    btnRun.addEventListener('click', executeSimulation);
  }

  async function executeSimulation() {
    const theatreId = theatreSelect ? theatreSelect.value : (theatres[0]?.theatre_id || 1);
    const holidayFactor = parseFloat(holidaySlider.value);
    const demandAdj = parseInt(demandSlider.value, 10);

    if (btnRun) {
      btnRun.disabled = true;
      btnRun.innerHTML = `<span>Calculating...</span>`;
    }

    try {
      const simData = await runSimulation({
        theatre_id: theatreId,
        holiday_factor: holidayFactor,
        demand_adjustment_pct: demandAdj,
        horizon_days: selectedHorizon
      });

      if (btnRun) {
        btnRun.disabled = false;
        btnRun.innerHTML = `
          <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="4 2 13 8 4 14 4 2" fill="currentColor"/>
          </svg>
          <span>Calculate Scenario Impact</span>
        `;
      }

      // Compute total admissions
      const totalSim = (simData.simulated || []).reduce((a, b) => a + b, 0);
      const totalBase = (simData.baseline || []).reduce((a, b) => a + b, 0);
      const deltaAdmissions = totalSim - totalBase;
      const pctShift = totalBase > 0 ? ((deltaAdmissions / totalBase) * 100).toFixed(1) : '0.0';

      const elAdm = document.getElementById('res-total-admissions');
      const elDelta = document.getElementById('res-admissions-delta');
      const elStaff = document.getElementById('res-staff-val');
      const elStaffDelta = document.getElementById('res-staff-delta');
      const elInv = document.getElementById('res-inventory-val');
      const elInvDelta = document.getElementById('res-inventory-delta');

      if (elAdm) elAdm.textContent = formatNumber(totalSim);
      if (elDelta) {
        elDelta.textContent = `${Number(pctShift) >= 0 ? '+' : ''}${pctShift}% vs baseline`;
        elDelta.style.color = Number(pctShift) >= 0 ? 'var(--color-emerald)' : 'var(--color-rose)';
      }

      if (elStaff) elStaff.textContent = `${simData.recommendations?.recommended_staff || 18} Staff / Shift`;
      if (elStaffDelta) elStaffDelta.textContent = `${simData.recommendations?.staff_delta || '+4'} floor attendants vs base`;

      if (elInv) elInv.textContent = `${formatNumber(simData.recommendations?.recommended_concessions_inventory || 3200)} Units`;
      if (elInvDelta) elInvDelta.textContent = `${simData.recommendations?.inventory_delta || '+550'} F&B concession units`;

      // Render Chart
      const ctx = document.getElementById('chart-sim-comparison');
      if (ctx) {
        destroyChart(comparisonChart);

        const labels = (simData.dates || []).map(d => {
          const parts = String(d).split('-');
          return parts.length >= 3 ? `${parts[1]}/${parts[2]}` : d;
        });

        comparisonChart = createBarChart(ctx, {
          labels,
          datasets: [
            {
              label: 'Baseline Expected',
              data: simData.baseline || [],
              backgroundColor: 'rgba(99, 102, 241, 0.45)',
              borderColor: THEME.indigo,
              borderWidth: 1,
              borderRadius: 3
            },
            {
              label: 'Simulated Demand',
              data: simData.simulated || [],
              backgroundColor: 'rgba(245, 158, 11, 0.75)',
              borderColor: THEME.gold,
              borderWidth: 1,
              borderRadius: 3
            }
          ]
        });
      }

      // Populate Ledger Table
      const tableBody = document.getElementById('sim-table-body');
      if (tableBody && simData.dates) {
        tableBody.innerHTML = simData.dates.map((d, i) => {
          const base = simData.baseline[i] || 0;
          const sim = simData.simulated[i] || 0;
          const diff = sim - base;
          const pct = base > 0 ? ((diff / base) * 100).toFixed(1) : '0.0';

          return `
            <tr>
              <td style="font-family: var(--font-mono); font-weight: 500;">${d}</td>
              <td class="number-cell">${formatNumber(base)}</td>
              <td class="number-cell" style="font-weight: 700; color: var(--color-gold);">${formatNumber(sim)}</td>
              <td class="number-cell" style="font-weight: 600; color: ${diff >= 0 ? 'var(--color-emerald)' : 'var(--color-rose)'};">
                ${diff >= 0 ? '+' : ''}${formatNumber(diff)}
              </td>
              <td class="number-cell" style="color: ${diff >= 0 ? 'var(--color-emerald)' : 'var(--color-rose)'};">
                ${diff >= 0 ? '+' : ''}${pct}%
              </td>
            </tr>
          `;
        }).join('');
      }

    } catch (err) {
      console.error('Error running scenario simulation:', err);
      if (btnRun) {
        btnRun.disabled = false;
        btnRun.textContent = 'Calculate Scenario Impact';
      }
    }
  }

  // Initial Run
  executeSimulation();
}
