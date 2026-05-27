"""
Self-contained share-camera page served directly by the backend.

This avoids Next.js WebSocket proxy issues — the phone connects
directly to the FastAPI backend, so WebSocket frames flow correctly.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ShareCamera"])

SHARE_CAMERA_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>SentinelAI Remote Camera</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    min-height: 100vh;
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    color: #e2e8f0;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    display: flex; flex-direction: column; align-items: center;
    padding: 20px;
  }
  h1 {
    font-size: 1.5rem; font-weight: 700; margin: 0;
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .subtitle { font-size: 0.813rem; color: #94a3b8; margin: 4px 0 16px; }
  .badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 16px; border-radius: 20px;
    font-size: 0.813rem; font-weight: 600; margin-bottom: 16px;
  }
  .badge-dot {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block;
  }
  .video-container {
    width: 100%; max-width: 480px; border-radius: 16px; overflow: hidden;
    background: #000; position: relative; aspect-ratio: 4/3;
    margin-bottom: 16px;
  }
  video {
    width: 100%; height: 100%; object-fit: cover; display: block;
  }
  .overlay {
    position: absolute; top: 10px; left: 10px;
    padding: 4px 10px; border-radius: 8px;
    background: rgba(0,0,0,0.6); backdrop-filter: blur(4px);
    font-size: 0.688rem; font-weight: 600; color: #22c55e;
    display: flex; gap: 12px;
  }
  .overlay span:last-child { color: #94a3b8; }
  .btn {
    flex: 1; padding: 14px; border-radius: 12px; border: none;
    cursor: pointer; font-weight: 700; font-size: 1rem; color: #fff;
    display: flex; align-items: center; justify-content: center; gap: 8px;
  }
  .btn-start { background: linear-gradient(135deg, #22c55e, #16a34a); }
  .btn-stop { background: linear-gradient(135deg, #ef4444, #dc2626); }
  .btn-switch {
    padding: 14px 18px; border-radius: 12px;
    border: 1px solid #334155; cursor: pointer;
    background: #1e293b; color: #e2e8f0; font-size: 1.25rem;
  }
  .controls { display: flex; gap: 10px; width: 100%; max-width: 480px; margin-bottom: 16px; }
  .settings {
    width: 100%; max-width: 480px; padding: 16px; border-radius: 12px;
    background: rgba(30,41,59,0.6); border: 1px solid #334155;
  }
  .settings-title { font-size: 0.75rem; font-weight: 600; color: #94a3b8; margin-bottom: 12px; }
  .setting-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .setting-row span { font-size: 0.813rem; }
  select {
    padding: 4px 8px; border-radius: 6px; background: #0f172a;
    border: 1px solid #334155; color: #e2e8f0; font-size: 0.75rem;
  }
  .info {
    width: 100%; max-width: 480px; margin-top: 16px;
    padding: 12px 16px; border-radius: 12px;
    background: rgba(30,41,59,0.4); border: 1px solid #1e293b;
    font-size: 0.688rem; color: #64748b;
  }
  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
</style>
</head>
<body>
  <h1>📡 SentinelAI Remote Camera</h1>
  <p class="subtitle">Stream your camera to SentinelAI for AI monitoring</p>

  <div class="badge" id="statusBadge" style="background:#64748b18;border:1px solid #64748b40;">
    <span class="badge-dot" id="statusDot" style="background:#64748b;"></span>
    <span id="statusText">Ready to stream</span>
  </div>

  <div class="video-container" id="videoContainer" style="border:2px solid #334155;">
    <video id="video" autoplay playsinline muted></video>
    <canvas id="canvas" style="display:none;"></canvas>
    <div class="overlay" id="overlay" style="display:none;">
      <span>● LIVE</span>
      <span id="frameInfo">0 frames</span>
    </div>
    <div id="placeholder" style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;background:rgba(15,23,42,0.8);">
      <div style="font-size:3rem;margin-bottom:8px;">📷</div>
      <div style="font-size:0.875rem;color:#94a3b8;" id="placeholderText">Tap "Start Streaming" to begin</div>
    </div>
  </div>

  <div class="controls" id="controls">
    <button class="btn btn-start" id="startBtn" onclick="startStreaming()">📡 Start Streaming</button>
    <button class="btn-switch" onclick="toggleCamera()">🔄</button>
  </div>

  <div class="settings" id="settings" style="display:none;">
    <div class="settings-title">STREAM SETTINGS</div>
    <div class="setting-row">
      <span>Quality</span>
      <select id="qualitySelect" onchange="quality=parseFloat(this.value);">
        <option value="0.2">Fastest</option>
        <option value="0.3">Low</option>
        <option value="0.4" selected>Medium (recommended)</option>
        <option value="0.7">High</option>
      </select>
    </div>
    <div class="setting-row">
      <span>FPS</span>
      <select id="fpsSelect" onchange="targetFps=parseInt(this.value);restartInterval();">
        <option value="5">5 FPS</option>
        <option value="10" selected>10 FPS</option>
        <option value="15">15 FPS</option>
        <option value="20">20 FPS</option>
      </select>
    </div>
  </div>

  <div class="info">
    <div><strong>Session:</strong> <span id="sessionInfo"></span></div>
    <div><strong>Server:</strong> <span id="serverInfo"></span></div>
    <div><strong>Camera:</strong> <span id="cameraInfo">Back</span></div>
  </div>

<script>
  const sessionId = 'bc_' + Math.random().toString(36).substring(2,10) + '_' + Date.now().toString(36);
  let facingMode = 'environment';
  let quality = 0.4;
  let targetFps = 10;
  let ws = null;
  let stream = null;
  let frameInterval = null;
  let frameCount = 0;
  let fpsCount = 0;
  let lastFpsTime = Date.now();
  let streaming = false;

  document.getElementById('sessionInfo').textContent = sessionId;
  document.getElementById('serverInfo').textContent = location.origin;

  function setStatus(status, text) {
    const colors = { idle:'#64748b', connecting:'#f59e0b', streaming:'#22c55e', error:'#ef4444', stopped:'#64748b' };
    const c = colors[status] || '#64748b';
    document.getElementById('statusDot').style.background = c;
    document.getElementById('statusDot').style.animation = status === 'streaming' ? 'pulse 2s ease-in-out infinite' : 'none';
    document.getElementById('statusBadge').style.background = c + '18';
    document.getElementById('statusBadge').style.border = '1px solid ' + c + '40';
    document.getElementById('statusText').textContent = text;
    document.getElementById('videoContainer').style.border = '2px solid ' + (status === 'streaming' ? '#22c55e40' : '#334155');
  }

  function restartInterval() {
    if (frameInterval) clearInterval(frameInterval);
    if (streaming) {
      frameInterval = setInterval(captureAndSend, 1000 / targetFps);
    }
  }

  function captureAndSend() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    if (!video || video.readyState < 2) return;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(function(blob) {
      if (blob && ws && ws.readyState === WebSocket.OPEN) {
        blob.arrayBuffer().then(function(buf) {
          ws.send(buf);
          frameCount++;
          fpsCount++;
          const now = Date.now();
          if (now - lastFpsTime >= 1000) {
            setStatus('streaming', 'Streaming • ' + fpsCount + ' FPS');
            fpsCount = 0;
            lastFpsTime = now;
          }
          document.getElementById('frameInfo').textContent = frameCount + ' frames';
        });
      }
    }, 'image/jpeg', quality);
  }

  async function startStreaming() {
    setStatus('connecting', 'Connecting...');
    document.getElementById('placeholderText').textContent = 'Connecting...';

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facingMode, width: { ideal: 640 }, height: { ideal: 480 } },

        audio: false,
      });

      const video = document.getElementById('video');
      video.srcObject = stream;
      await video.play();
      video.style.transform = facingMode === 'user' ? 'scaleX(-1)' : 'none';

      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = proto + '//' + location.host + '/api/browser-cam/stream/' + sessionId + '?label=Remote+Camera';
      ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';

      ws.onopen = function() {
        streaming = true;
        setStatus('streaming', 'Streaming • 0 FPS');
        document.getElementById('placeholder').style.display = 'none';
        document.getElementById('overlay').style.display = 'flex';
        document.getElementById('settings').style.display = 'block';
        document.getElementById('startBtn').textContent = '⬛ Stop Streaming';
        document.getElementById('startBtn').className = 'btn btn-stop';
        document.getElementById('startBtn').setAttribute('onclick', 'stopStreaming()');
        frameInterval = setInterval(captureAndSend, 1000 / targetFps);
      };

      ws.onerror = function() {
        setStatus('error', 'Connection failed');
        document.getElementById('placeholderText').textContent = 'WebSocket connection failed. Is the backend running?';
      };

      ws.onclose = function() {
        if (streaming) {
          streaming = false;
          setStatus('stopped', 'Disconnected');
        }
      };
    } catch (err) {
      setStatus('error', 'Error');
      document.getElementById('placeholderText').textContent = err.message || 'Camera access denied';
    }
  }

  function stopStreaming() {
    streaming = false;
    if (frameInterval) { clearInterval(frameInterval); frameInterval = null; }
    if (ws) { ws.close(); ws = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    document.getElementById('video').srcObject = null;
    document.getElementById('placeholder').style.display = 'flex';
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('settings').style.display = 'none';
    document.getElementById('startBtn').textContent = '📡 Start Streaming';
    document.getElementById('startBtn').className = 'btn btn-start';
    document.getElementById('startBtn').setAttribute('onclick', 'startStreaming()');
    setStatus('stopped', 'Stopped');
    document.getElementById('placeholderText').textContent = 'Tap "Start Streaming" to begin';
  }

  function toggleCamera() {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    document.getElementById('cameraInfo').textContent = facingMode === 'environment' ? 'Back' : 'Front';
    if (streaming) {
      stopStreaming();
      setTimeout(startStreaming, 500);
    }
  }
</script>
</body>
</html>
"""


@router.get("/share-camera", response_class=HTMLResponse)
async def share_camera_page():
    """Serve the self-contained remote camera streaming page."""
    return SHARE_CAMERA_HTML
