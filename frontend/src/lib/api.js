/**
 * API client for the CCTV backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiClient {
  constructor() {
    this.baseUrl = API_BASE;
    this.token = null;
  }

  setToken(token) {
    this.token = token;
    if (typeof window !== 'undefined') {
      localStorage.setItem('cctv_token', token);
    }
  }

  getToken() {
    if (this.token) return this.token;
    if (typeof window !== 'undefined') {
      this.token = localStorage.getItem('cctv_token');
    }
    return this.token;
  }

  clearToken() {
    this.token = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('cctv_token');
    }
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    const token = this.getToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    // Remove Content-Type for FormData
    if (options.body instanceof FormData) {
      delete headers['Content-Type'];
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (response.status === 401) {
      this.clearToken();
      if (typeof window !== 'undefined') {
        window.location.href = '/login';
      }
      throw new Error('Unauthorized');
    }

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(error.detail || 'Request failed');
    }

    return response.json();
  }

  // ── Auth ──────────────────────────────────────────────────
  async login(email, password) {
    const data = await this.request('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    this.setToken(data.access_token);
    return data;
  }

  async getMe() {
    return this.request('/api/auth/me');
  }

  logout() {
    this.clearToken();
  }

  // ── Persons ───────────────────────────────────────────────
  async getPersons(params = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request(`/api/persons${query ? `?${query}` : ''}`);
  }

  async getPerson(id) {
    return this.request(`/api/persons/${id}`);
  }

  async createPerson(data) {
    return this.request('/api/persons', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updatePerson(id, data) {
    return this.request(`/api/persons/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deletePerson(id) {
    return this.request(`/api/persons/${id}`, { method: 'DELETE' });
  }

  async uploadFaceEncoding(personId, file) {
    const formData = new FormData();
    formData.append('file', file);
    return this.request(`/api/persons/${personId}/encodings`, {
      method: 'POST',
      body: formData,
    });
  }

  async getPersonHistory(id, params = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request(`/api/persons/${id}/history${query ? `?${query}` : ''}`);
  }

  // ── Cameras ───────────────────────────────────────────────
  async getCameras(params = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request(`/api/cameras${query ? `?${query}` : ''}`);
  }

  async getCamera(id) {
    return this.request(`/api/cameras/${id}`);
  }

  async createCamera(data) {
    return this.request('/api/cameras', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async startCamera(id) {
    return this.request(`/api/cameras/${id}/start`, { method: 'POST' });
  }

  async stopCamera(id) {
    return this.request(`/api/cameras/${id}/stop`, { method: 'POST' });
  }

  async getActivePersons() {
    return this.request('/api/cameras/active-persons');
  }

  // ── Events ────────────────────────────────────────────────
  async getEvents(params = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request(`/api/events${query ? `?${query}` : ''}`);
  }

  async getActiveSessions() {
    return this.request('/api/sessions/active');
  }

  // ── Unknown Faces ─────────────────────────────────────────
  async getUnknownFaces(params = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request(`/api/unknown-faces${query ? `?${query}` : ''}`);
  }

  async enrollUnknownFace(id, data) {
    return this.request(`/api/unknown-faces/${id}/enroll`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async dismissUnknownFace(id) {
    return this.request(`/api/unknown-faces/${id}/dismiss`, { method: 'POST' });
  }

  // ── Analytics ─────────────────────────────────────────────
  async getDashboard() {
    return this.request('/api/analytics/dashboard');
  }

  async getPeakTimes(days = 7) {
    return this.request(`/api/analytics/peak-times?days=${days}`);
  }

  async getOccupancy() {
    return this.request('/api/analytics/occupancy');
  }

  async getMovementLogs(params = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request(`/api/analytics/movement${query ? `?${query}` : ''}`);
  }

  async getReport(type = 'daily', date = null) {
    const params = { report_type: type };
    if (date) params.date = date;
    const query = new URLSearchParams(params).toString();
    return this.request(`/api/analytics/reports?${query}`);
  }

  // ── WebSocket ─────────────────────────────────────────────
  connectLiveEvents(onMessage, onError) {
    const wsUrl = this.baseUrl.replace('http', 'ws');
    const ws = new WebSocket(`${wsUrl}/api/events/live`);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (e) {
        console.error('Failed to parse WS message:', e);
      }
    };
    ws.onerror = onError || console.error;
    return ws;
  }

  connectCameraStream(cameraId, onFrame, onError) {
    const wsUrl = this.baseUrl.replace('http', 'ws');
    const ws = new WebSocket(`${wsUrl}/api/cameras/${cameraId}/stream`);
    ws.binaryType = 'arraybuffer';
    ws.onmessage = (event) => {
      // Binary JPEG frame received
      const blob = new Blob([event.data], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      onFrame(url);
    };
    ws.onerror = onError || console.error;
    ws.onopen = () => {
      // Send a keep-alive ping periodically
      ws._keepAlive = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send('ping');
        }
      }, 30000);
    };
    ws.onclose = () => {
      if (ws._keepAlive) clearInterval(ws._keepAlive);
    };
    return ws;
  }

  /**
   * Connect to live events WebSocket for real-time detection events.
   * Receives entry/exit/detection/unknown events from all cameras.
   */
  connectLiveEvents(onEvent, onError) {
    const wsUrl = this.baseUrl.replace(/^http/, 'ws') + '/api/events/live';
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (onEvent) onEvent(data);
      } catch (e) {
        console.warn('Failed to parse event:', e);
      }
    };
    ws.onerror = onError || console.error;
    ws.onopen = () => {
      console.log('Live events WebSocket connected');
      ws._keepAlive = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send('ping');
        }
      }, 30000);
    };
    ws.onclose = () => {
      console.log('Live events WebSocket disconnected');
      if (ws._keepAlive) clearInterval(ws._keepAlive);
    };
    return ws;
  }
}

const api = new ApiClient();
export default api;
