/**
 * API client for the CCTV backend.
 */

// If in browser, use relative paths to trigger Next.js rewrites. In SSR, default to localhost:8000
export const API_BASE = process.env.NEXT_PUBLIC_API_URL || (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000');

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

  async hardDeletePerson(id) {
    return this.request(`/api/persons/${id}?hard=true`, { method: 'DELETE' });
  }

  async uploadFaceEncoding(personId, file) {
    const formData = new FormData();
    formData.append('file', file);
    return this.request(`/api/persons/${personId}/encodings`, {
      method: 'POST',
      body: formData,
    });
  }

  async getPersonJourney(id, date = null) {
    const q = date ? `?date=${date}` : '';
    return this.request(`/api/persons/${id}/journey${q}`);
  }

  async getPersonHistory(id, params = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request(`/api/persons/${id}/history${query ? `?${query}` : ''}`);
  }

  // ── Encoding Management ────────────────────────────────────
  async getPersonEncodings(personId) {
    return this.request(`/api/persons/${personId}/encodings/list`);
  }

  async deletePersonEncoding(personId, encodingId) {
    return this.request(`/api/persons/${personId}/encodings/${encodingId}`, {
      method: 'DELETE',
    });
  }

  // ── Embedding Suggestions ──────────────────────────────────
  async getPersonSuggestions(personId, status = 'pending') {
    return this.request(`/api/persons/${personId}/suggestions?status=${status}`);
  }

  async getPersonSuggestionCount(personId) {
    return this.request(`/api/persons/${personId}/suggestions/count`);
  }

  async approveSuggestion(personId, suggestionId, replaceEncodingId = null) {
    const params = replaceEncodingId ? `?replace_encoding_id=${replaceEncodingId}` : '';
    return this.request(`/api/persons/${personId}/suggestions/${suggestionId}/approve${params}`, {
      method: 'POST',
    });
  }

  async rejectSuggestion(personId, suggestionId) {
    return this.request(`/api/persons/${personId}/suggestions/${suggestionId}/reject`, {
      method: 'POST',
    });
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

  async deleteCamera(id) {
    return this.request(`/api/cameras/${id}`, { method: 'DELETE' });
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

  async getSecurityAlerts(params = {}) {
    const query = new URLSearchParams({ event_type: 'security_alert', ...params }).toString();
    return this.request(`/api/events${query ? `?${query}` : ''}`);
  }

  async getVehicleEvents(params = {}) {
    const query = new URLSearchParams({ event_type: 'vehicle_entry', ...params }).toString();
    return this.request(`/api/events${query ? `?${query}` : ''}`);
  }

  async getTrafficAlerts(params = {}) {
    const query = new URLSearchParams({ event_type: 'security_alert', subtype: 'vehicle_proximity', ...params }).toString();
    return this.request(`/api/events${query ? `?${query}` : ''}`);
  }

  // ── Incidents ─────────────────────────────────────────────
  async getIncidents(params = {}) {
    const query = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== null && v !== undefined && v !== '')
    ).toString();
    return this.request(`/api/incidents${query ? `?${query}` : ''}`);
  }

  async getIncident(eventId) {
    return this.request(`/api/incidents/${eventId}`);
  }

  async setIncidentStatus(eventId, status) {
    return this.request(`/api/incidents/${eventId}`, {
      method: 'PATCH',
      body: JSON.stringify({ status }),
    });
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

  async mergeUnknownFace(unknownId, personId) {
    return this.request(`/api/unknown-faces/${unknownId}/merge`, {
      method: 'POST',
      body: JSON.stringify({ person_id: personId }),
    });
  }

  // ── Analytics ─────────────────────────────────────────────
  async getDashboard() {
    return this.request('/api/analytics/dashboard');
  }

  async getPeakTimes(days = 7) {
    return this.request(`/api/analytics/peak-times?days=${days}`);
  }

  async getDailySummary(date = null) {
    const q = date ? `?date=${date}` : '';
    return this.request(`/api/analytics/daily-summary${q}`);
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

  // ── Faculty Attendance ────────────────────────────────────
  async getAttendanceSummary(date = null) {
    const q = date ? `?date=${date}` : '';
    return this.request(`/api/attendance/summary${q}`);
  }

  async getAttendanceDay(params = {}) {
    const query = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== null && v !== undefined && v !== '')
    ).toString();
    return this.request(`/api/attendance/day${query ? `?${query}` : ''}`);
  }

  async getOnCampus(params = {}) {
    const query = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== null && v !== undefined && v !== '')
    ).toString();
    return this.request(`/api/attendance/on-campus${query ? `?${query}` : ''}`);
  }

  async getPersonAttendance(personId, from = null, to = null) {
    const params = new URLSearchParams();
    if (from) params.set('from', from);
    if (to) params.set('to', to);
    const q = params.toString();
    return this.request(`/api/attendance/person/${personId}${q ? `?${q}` : ''}`);
  }

  /**
   * Download an attendance export.
   *
   * Uses fetch + blob rather than a plain <a href> because the endpoint
   * requires an Authorization header, which a link cannot send. The filename
   * comes from Content-Disposition so the server stays the single source of
   * truth for naming. Mirrors downloadExport() on the analytics page.
   */
  async downloadAttendanceExport({ date, format = 'csv', department = null, status = null } = {}) {
    const params = new URLSearchParams({ format });
    if (date) params.set('date', date);
    if (department) params.set('department', department);
    if (status) params.set('status', status);

    const token = this.getToken();
    const resp = await fetch(`${this.baseUrl}/api/attendance/export?${params}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!resp.ok) {
      let detail = `Export failed (${resp.status})`;
      try {
        const body = await resp.json();
        if (body?.detail) detail = body.detail;
      } catch { /* non-JSON error body */ }
      throw new Error(detail);
    }

    const disposition = resp.headers.get('Content-Disposition') || '';
    const match = disposition.match(/filename="?([^"]+)"?/);
    const filename = match ? match[1] : `attendance.${format}`;

    const blob = await resp.blob();
    if (blob.size === 0) throw new Error('No records matched — nothing to export');

    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
    return filename;
  }

  // ── WebSocket ─────────────────────────────────────────────

  /**
   * Build an authenticated WebSocket URL.
   *
   * The server now REQUIRES ?token= on /api/events/live and
   * /api/cameras/{id}/stream (it previously validated the token only when one
   * happened to be present, so omitting it bypassed auth entirely).
   *
   * Browsers cannot set an Authorization header on a WebSocket handshake, so
   * the JWT travels as a query parameter. Use TLS in production -- query
   * strings are visible to proxies and land in access logs.
   */
  wsUrl(path) {
    const base = this.baseUrl.replace(/^http/, 'ws');
    const token = this.getToken();
    const qs = token ? `?token=${encodeURIComponent(token)}` : '';
    return `${base}${path}${qs}`;
  }

  connectCameraStream(cameraId, onFrame, onError) {
    const ws = new WebSocket(this.wsUrl(`/api/cameras/${cameraId}/stream`));
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
    const ws = new WebSocket(this.wsUrl('/api/events/live'));

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

  // ── Video Analysis ─────────────────────────────────────────
  async uploadVideo(file, onProgress) {
    const formData = new FormData();
    formData.append('file', file);

    const url = `${this.baseUrl}/api/video-analysis/upload`;
    const token = this.getToken();

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', url, true);

      if (token) {
        xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      }

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          try {
            const err = JSON.parse(xhr.responseText);
            reject(new Error(err.detail || 'Upload failed'));
          } catch {
            reject(new Error('Upload failed'));
          }
        }
      };

      xhr.onerror = () => reject(new Error('Network error'));
      xhr.send(formData);
    });
  }

  async getAnalysisStatus(jobId) {
    return this.request(`/api/video-analysis/${jobId}/status`);
  }

  async getAnalysisResults(jobId) {
    return this.request(`/api/video-analysis/${jobId}/results`);
  }

  async getAnalysisHistory() {
    return this.request('/api/video-analysis/history');
  }

  async deleteAnalysis(jobId) {
    return this.request(`/api/video-analysis/${jobId}`, { method: 'DELETE' });
  }

  connectAnalysisStream(jobId, onMessage, onError) {
    const ws = new WebSocket(this.wsUrl(`/api/video-analysis/${jobId}/stream`));
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (onMessage) onMessage(data);
      } catch (e) {
        console.warn('Failed to parse analysis WS message:', e);
      }
    };
    ws.onerror = onError || console.error;
    ws.onopen = () => {
      console.log('Analysis progress WebSocket connected');
    };
    ws.onclose = () => {
      console.log('Analysis progress WebSocket disconnected');
    };
    return ws;
  }

  // ── Search ──────────────────────────────────────────────
  async search(query, type = 'all', limit = 20) {
    const params = new URLSearchParams({ q: query, type, limit: String(limit) });
    return this.request(`/api/search?${params}`);
  }

  // ── System Health ──────────────────────────────────────
  async getSystemHealth() {
    return this.request('/api/system-health');
  }

  // ── AI Identity Review ─────────────────────────────────
  async getIdentityReviewDashboard() {
    return this.request('/api/identity-review/dashboard');
  }

  async getIdentityReviewCandidates(status = 'pending', limit = 20, offset = 0) {
    const params = new URLSearchParams({ status, limit: String(limit), offset: String(offset) });
    return this.request(`/api/identity-review/candidates?${params}`);
  }

  async getIdentityReviewCandidate(id) {
    return this.request(`/api/identity-review/candidates/${id}`);
  }

  // Dry-run an approval: which cluster members would actually reach the
  // gallery, which would not and why, and the measured coverage change.
  // Writes nothing. Pass selectedFaceIds to see the effect of excluding a
  // member; omit it for the default selection (everything but the outliers).
  async previewCandidate(id, selectedFaceIds = null) {
    return this.request(`/api/identity-review/candidates/${id}/preview`, {
      method: 'POST',
      body: JSON.stringify({ selected_face_ids: selectedFaceIds }),
    });
  }

  async approveCandidate(id, selectedFaceIds = null) {
    return this.request(`/api/identity-review/candidates/${id}/approve`, {
      method: 'POST',
      body: JSON.stringify({ selected_face_ids: selectedFaceIds }),
    });
  }

  async rejectCandidate(id) {
    return this.request(`/api/identity-review/candidates/${id}/reject`, { method: 'POST' });
  }

  async skipCandidate(id) {
    return this.request(`/api/identity-review/candidates/${id}/skip`, { method: 'POST' });
  }

  async getIdentityReviewClusters(status = 'active', limit = 20, offset = 0) {
    const params = new URLSearchParams({ status, limit: String(limit), offset: String(offset) });
    return this.request(`/api/identity-review/clusters?${params}`);
  }

  async getIdentityReviewCluster(id) {
    return this.request(`/api/identity-review/clusters/${id}`);
  }

  async ignoreCluster(id, reason = '') {
    return this.request(`/api/identity-review/clusters/${id}/ignore`, {
      method: 'POST', body: JSON.stringify({ reason }),
    });
  }

  async createPersonFromCluster(id, data) {
    return this.request(`/api/identity-review/clusters/${id}/create-person`, {
      method: 'POST', body: JSON.stringify(data),
    });
  }

  async getGalleryUpdates(status = null, limit = 20, offset = 0) {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (status) params.set('status', status);
    return this.request(`/api/identity-review/gallery-updates?${params}`);
  }

  async applyGalleryUpdate(id) {
    return this.request(`/api/identity-review/gallery-updates/${id}/apply`, { method: 'POST' });
  }

  async rollbackGalleryUpdate(id) {
    return this.request(`/api/identity-review/gallery-updates/${id}/rollback`, { method: 'POST' });
  }

  async getIdentityReviewActivity(limit = 50, offset = 0) {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    return this.request(`/api/identity-review/activity?${params}`);
  }
}

const api = new ApiClient();
export default api;
