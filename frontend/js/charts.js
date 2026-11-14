/**
 * BoxOfficeIQ - Charts Factory Module
 * Built on Chart.js 4.x with refined enterprise dark cinema analytics styling
 */

export const THEME = {
  bg: '#0F1117',
  surface: '#151821',
  border: 'rgba(255, 255, 255, 0.08)',
  
  // Brand & Data Palette
  gold: '#F59E0B',
  goldDim: 'rgba(245, 158, 11, 0.15)',
  goldGlow: 'rgba(245, 158, 11, 0.35)',
  
  sky: '#38BDF8',
  skyDim: 'rgba(56, 189, 248, 0.15)',
  skyGlow: 'rgba(56, 189, 248, 0.3)',
  
  indigo: '#6366F1',
  indigoDim: 'rgba(99, 102, 241, 0.15)',
  
  emerald: '#10B981',
  emeraldDim: 'rgba(16, 185, 129, 0.15)',
  
  rose: '#F43F5E',
  roseDim: 'rgba(244, 63, 94, 0.15)',
  
  // Legacy aliases for backward compatibility with existing code
  cyan: '#38BDF8',
  cyanDim: 'rgba(56, 189, 248, 0.15)',
  cyanGlow: 'rgba(56, 189, 248, 0.3)',
  purple: '#6366F1',
  purpleDim: 'rgba(99, 102, 241, 0.15)',
  green: '#10B981',
  greenDim: 'rgba(16, 185, 129, 0.15)',
  red: '#F43F5E',
  
  // Neutrals
  text: '#F8FAFC',
  textMuted: '#94A3B8',
  textDim: '#64748B',
  gridColor: 'rgba(255, 255, 255, 0.05)',
  tooltipBg: 'rgba(15, 17, 23, 0.96)'
};

export const defaultChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: {
    mode: 'index',
    intersect: false
  },
  plugins: {
    legend: {
      display: false, // We use sleek bespoke HTML legend strips
      labels: {
        color: THEME.textMuted,
        font: { family: "'Inter', sans-serif", size: 12, weight: '500' },
        boxWidth: 8,
        boxHeight: 8,
        usePointStyle: true,
        padding: 16
      }
    },
    tooltip: {
      backgroundColor: THEME.tooltipBg,
      titleColor: THEME.text,
      bodyColor: THEME.textMuted,
      borderColor: 'rgba(255, 255, 255, 0.12)',
      borderWidth: 1,
      padding: { top: 10, bottom: 10, left: 14, right: 14 },
      cornerRadius: 6,
      caretSize: 4,
      boxPadding: 4,
      usePointStyle: true,
      titleFont: { family: "'Inter', sans-serif", weight: '600', size: 12 },
      bodyFont: { family: "'Inter', sans-serif", size: 12 },
      footerFont: { family: "'Inter', sans-serif", size: 11 }
    }
  },
  scales: {
    x: {
      grid: {
        color: THEME.gridColor,
        drawBorder: false
      },
      border: {
        display: false
      },
      ticks: {
        color: THEME.textDim,
        font: { family: "'Inter', sans-serif", size: 11, weight: '500' },
        maxRotation: 0,
        autoSkip: true,
        maxTicksLimit: 9,
        padding: 6
      }
    },
    y: {
      grid: {
        color: THEME.gridColor,
        drawBorder: false
      },
      border: {
        display: false
      },
      ticks: {
        color: THEME.textDim,
        font: { family: "'Inter', sans-serif", size: 11, weight: '500' },
        padding: 8
      }
    }
  }
};

export function destroyChart(chart) {
  if (chart && typeof chart.destroy === 'function') {
    chart.destroy();
  }
}

export function updateChartData(chart, newData) {
  if (!chart) return;
  if (newData.labels) chart.data.labels = newData.labels;
  if (newData.datasets) chart.data.datasets = newData.datasets;
  chart.update();
}

export function createLineChart(ctx, data, customOptions = {}) {
  destroyChart(ctx._chartInstance);

  const options = {
    ...defaultChartOptions,
    ...customOptions,
    plugins: {
      ...defaultChartOptions.plugins,
      ...(customOptions.plugins || {})
    },
    scales: {
      ...defaultChartOptions.scales,
      ...(customOptions.scales || {})
    }
  };

  const chart = new Chart(ctx, {
    type: 'line',
    data,
    options
  });

  ctx._chartInstance = chart;
  return chart;
}

export function createBarChart(ctx, data, customOptions = {}) {
  destroyChart(ctx._chartInstance);

  const options = {
    ...defaultChartOptions,
    ...customOptions,
    plugins: {
      ...defaultChartOptions.plugins,
      ...(customOptions.plugins || {})
    },
    scales: {
      ...defaultChartOptions.scales,
      ...(customOptions.scales || {})
    }
  };

  const chart = new Chart(ctx, {
    type: 'bar',
    data,
    options
  });

  ctx._chartInstance = chart;
  return chart;
}

export function createAreaChart(ctx, data, customOptions = {}) {
  destroyChart(ctx._chartInstance);

  const datasets = data.datasets.map(ds => ({
    fill: true,
    tension: 0.25,
    borderWidth: 2,
    pointRadius: 0,
    pointHoverRadius: 5,
    pointHoverBackgroundColor: ds.borderColor || THEME.sky,
    pointHoverBorderColor: '#FFFFFF',
    pointHoverBorderWidth: 2,
    ...ds
  }));

  const options = {
    ...defaultChartOptions,
    ...customOptions,
    plugins: {
      ...defaultChartOptions.plugins,
      ...(customOptions.plugins || {})
    },
    scales: {
      ...defaultChartOptions.scales,
      ...(customOptions.scales || {})
    }
  };

  const chart = new Chart(ctx, {
    type: 'line',
    data: { ...data, datasets },
    options
  });

  ctx._chartInstance = chart;
  return chart;
}

export function createDoughnutChart(ctx, data, customOptions = {}) {
  destroyChart(ctx._chartInstance);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: defaultChartOptions.plugins.tooltip
    },
    cutout: '76%',
    ...customOptions
  };

  const chart = new Chart(ctx, {
    type: 'doughnut',
    data,
    options
  });

  ctx._chartInstance = chart;
  return chart;
}

export function createScatterChart(ctx, data, customOptions = {}) {
  destroyChart(ctx._chartInstance);

  const options = {
    ...defaultChartOptions,
    ...customOptions,
    plugins: {
      ...defaultChartOptions.plugins,
      ...(customOptions.plugins || {})
    },
    scales: {
      x: {
        ...defaultChartOptions.scales.x,
        type: 'linear',
        position: 'bottom',
        title: {
          display: true,
          text: 'Actual Turnstile Count',
          color: THEME.textMuted,
          font: { family: "'Inter', sans-serif", size: 11, weight: '500' },
          padding: { top: 8 }
        }
      },
      y: {
        ...defaultChartOptions.scales.y,
        title: {
          display: true,
          text: 'Predicted Demand (LightGBM)',
          color: THEME.textMuted,
          font: { family: "'Inter', sans-serif", size: 11, weight: '500' },
          padding: { bottom: 8 }
        }
      }
    }
  };

  const chart = new Chart(ctx, {
    type: 'scatter',
    data,
    options
  });

  ctx._chartInstance = chart;
  return chart;
}
