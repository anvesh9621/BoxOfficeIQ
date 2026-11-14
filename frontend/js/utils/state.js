/**
 * BoxOfficeIQ - State Management
 */

class AppState {
  constructor() {
    this._state = {
      selectedTheatreId: null,
      selectedTheatre: null,
      theatres: [],
      dateRange: {
        startDate: '2024-01-01',
        endDate: '2024-12-31'
      },
      forecastHorizon: 7,
      activePage: 'dashboard',
      apiOnline: false
    };
    this._listeners = [];
  }

  get(key) {
    return this._state[key];
  }

  getAll() {
    return { ...this._state };
  }

  set(key, value) {
    const oldVal = this._state[key];
    this._state[key] = value;
    this._dispatch(key, value, oldVal);
  }

  setMultiple(obj) {
    for (const [key, value] of Object.entries(obj)) {
      this.set(key, value);
    }
  }

  subscribe(callback) {
    this._listeners.push(callback);
    return () => {
      this._listeners = this._listeners.filter(cb => cb !== callback);
    };
  }

  _dispatch(key, newVal, oldVal) {
    const event = new CustomEvent('statechange', {
      detail: { key, newVal, oldVal, state: { ...this._state } }
    });
    window.dispatchEvent(event);

    for (const cb of this._listeners) {
      try {
        cb(key, newVal, oldVal, { ...this._state });
      } catch (e) {
        console.error('State listener error:', e);
      }
    }
  }
}

export const state = new AppState();
