/**
 * BoxOfficeIQ - Formatters Utility
 * Clean typography & semantic visual tokens
 */

export function formatNumber(n) {
  if (n === null || n === undefined || isNaN(n)) return '0';
  return Math.round(Number(n)).toLocaleString('en-US');
}

export function formatK(n) {
  if (n === null || n === undefined || isNaN(n)) return '0';
  const num = Number(n);
  if (Math.abs(num) >= 1_000_000) {
    return (num / 1_000_000).toFixed(1) + 'M';
  }
  if (Math.abs(num) >= 1_000) {
    return (num / 1_000).toFixed(1) + 'K';
  }
  return num.toLocaleString('en-US');
}

export function formatCurrency(n) {
  if (n === null || n === undefined || isNaN(n)) return '$0';
  const num = Number(n);
  if (Math.abs(num) >= 1_000_000) {
    return '$' + (num / 1_000_000).toFixed(2) + 'M';
  }
  if (Math.abs(num) >= 1_000) {
    return '$' + (num / 1_000).toFixed(1) + 'K';
  }
  return '$' + Math.round(num).toLocaleString('en-US');
}

export function formatPct(n) {
  if (n === null || n === undefined || isNaN(n)) return '0.0%';
  const num = Number(n);
  const formatted = Math.abs(num).toFixed(1) + '%';
  return num < 0 ? `-${formatted}` : `+${formatted}`;
}

export function formatDate(d) {
  if (!d) return '';
  const date = new Date(d);
  if (isNaN(date.getTime())) return String(d);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });
}

export function formatDateShort(d) {
  if (!d) return '';
  const date = new Date(d);
  if (isNaN(date.getTime())) return String(d);
  return `${date.toLocaleDateString('en-US', { month: 'short' })} ${date.getDate()}`;
}

export function formatWeekday(d) {
  if (!d) return '';
  const date = new Date(d);
  if (isNaN(date.getTime())) return '';
  return date.toLocaleDateString('en-US', { weekday: 'short' });
}

export function trendBadge(value, suffix = '%') {
  const num = Number(value);
  if (isNaN(num) || num === 0) {
    return `<span class="trend-badge trend-badge--neutral">0.0${suffix}</span>`;
  }
  if (num > 0) {
    return `<span class="trend-badge trend-badge--up">
      <svg viewBox="0 0 16 16" width="10" height="10" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="3 10 8 5 13 10"/></svg>
      +${num.toFixed(1)}${suffix}
    </span>`;
  }
  return `<span class="trend-badge trend-badge--down">
    <svg viewBox="0 0 16 16" width="10" height="10" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="3 6 8 11 13 6"/></svg>
    ${num.toFixed(1)}${suffix}
  </span>`;
}

export function getTheatreTypeBadge(type) {
  const t = (type || '').toLowerCase().replace(/\s+/g, '');
  if (t.includes('imax')) {
    return `<span class="badge badge--imax">IMAX</span>`;
  }
  if (t.includes('multiplex')) {
    return `<span class="badge badge--multiplex">Multiplex</span>`;
  }
  if (t.includes('single')) {
    return `<span class="badge badge--single">Single Screen</span>`;
  }
  if (t.includes('drive')) {
    return `<span class="badge badge--drivein">Drive-In</span>`;
  }
  return `<span class="badge badge--default">${type || 'Cinema'}</span>`;
}

export function getSentimentBadge(sentiment) {
  const s = (sentiment || '').toLowerCase();
  if (s.includes('pos')) {
    return `<span class="badge badge--positive">Positive</span>`;
  }
  if (s.includes('neg')) {
    return `<span class="badge badge--negative">Negative</span>`;
  }
  return `<span class="badge badge--neutral">Neutral</span>`;
}
