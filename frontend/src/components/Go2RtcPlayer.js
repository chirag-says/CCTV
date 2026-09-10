'use client';

/**
 * Live view: WebRTC video from go2rtc, detection boxes from us.
 *
 * WHAT CHANGED AND WHY
 * The old live view was a picture of a decision: the backend drew boxes into
 * the frame, JPEG-encoded it, and pushed the still down a WebSocket. The
 * overlay could not be styled, filtered or clicked, and it cost a CPU-bound
 * encode per frame per camera.
 *
 * Now the two halves travel separately. Video comes straight from go2rtc over
 * WebRTC and never enters the Python process at all. Boxes arrive on their
 * own WebSocket as normalised coordinates -- roughly 200 bytes where the JPEG
 * was 60 KB -- and are drawn on a canvas laid over the video.
 *
 * MATCHING THE TWO BY TIMESTAMP
 * They are not the same pipe, so they do not arrive together. Metadata takes
 * one network hop; video goes through an encoder, a jitter buffer and a
 * decoder, so it is displayed later than metadata describing the same moment.
 * Drawing whatever arrived last would put the boxes ahead of the people.
 *
 * So messages go into a short buffer and the renderer picks the one matching
 * the frame actually on screen: server time now, minus the video's own
 * measured latency (jitterBufferDelay from getStats, plus the decode/render
 * lag WebRTC reports). Server time is estimated from message arrivals -- the
 * MINIMUM observed (arrival - sent) over a sliding window, which is the least
 * contaminated sample of clock skew we can get without a time protocol.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import api from '@/lib/api';

// How long a metadata message stays eligible for display. Longer than any
// plausible video latency, short enough that a dead metadata socket goes
// blank instead of freezing boxes over live video.
const BUFFER_MS = 4000;

// Used until getStats() has produced a real number, and whenever it cannot.
const DEFAULT_LATENCY_MS = 150;

// A box older than this relative to the displayed frame is not drawn. At the
// default 10 fps analysis rate a frame is 100 ms, so this tolerates a few
// missed analyses without the overlay flickering out.
const MAX_STALENESS_MS = 700;

const COLOR = {
    known: '#22c55e',     // recognised person
    unknown: '#f97316',   // tracked, not identified
    face: '#60a5fa',      // face box inside a body box
    coasting: '#64748b',  // tracker believes it, is not seeing it
};

function boxStyle(obj) {
    if (obj.coasting) return { color: COLOR.coasting, dashed: true };
    if (obj.kind === 'face') return { color: COLOR.face, dashed: false };
    if (obj.identity) return { color: COLOR.known, dashed: false };
    return { color: COLOR.unknown, dashed: false };
}

export default function Go2RtcPlayer({
    camera,
    signallingUrl,
    showOverlay = true,
    onStatus,
    muted = true,
}) {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [status, setStatus] = useState('connecting');

    // Metadata buffer + clock estimation. Refs, not state: these are written
    // many times a second and read from a rAF loop. Putting them in state
    // would re-render the whole card at the analysis frame rate.
    const bufferRef = useRef([]);          // [{ t: serverSeconds, objects }]
    const clockOffsetRef = useRef(null);   // localSeconds - serverSeconds
    const latencyRef = useRef(DEFAULT_LATENCY_MS);

    const report = useCallback((next) => {
        setStatus(next);
        if (onStatus) onStatus(next);
    }, [onStatus]);

    // ── WebRTC: negotiate with go2rtc over its signalling socket ──────────
    useEffect(() => {
        if (!signallingUrl) return undefined;

        let mounted = true;
        let pc = null;
        let ws = null;
        let statsTimer = null;
        let reconnectTimer = null;

        const teardown = () => {
            if (statsTimer) { clearInterval(statsTimer); statsTimer = null; }
            if (ws) { ws.onclose = null; try { ws.close(); } catch (e) { /* closing */ } ws = null; }
            if (pc) { try { pc.close(); } catch (e) { /* closing */ } pc = null; }
        };

        const connect = async () => {
            if (!mounted) return;
            report('connecting');

            pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
                // go2rtc bundles everything on one transport; asking for more
                // just adds candidates that will never be used.
                bundlePolicy: 'max-bundle',
            });

            // Receive only. This page never sends media, and offering to would
            // make the browser ask for camera permission.
            pc.addTransceiver('video', { direction: 'recvonly' });
            pc.addTransceiver('audio', { direction: 'recvonly' });

            const stream = new MediaStream();
            pc.ontrack = (ev) => {
                stream.addTrack(ev.track);
                if (videoRef.current && videoRef.current.srcObject !== stream) {
                    videoRef.current.srcObject = stream;
                }
            };

            pc.onconnectionstatechange = () => {
                if (!mounted || !pc) return;
                if (pc.connectionState === 'connected') report('streaming');
                if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
                    report('reconnecting');
                    teardown();
                    reconnectTimer = setTimeout(connect, 2000);
                }
            };

            try {
                ws = new WebSocket(signallingUrl);
            } catch (err) {
                report('reconnecting');
                reconnectTimer = setTimeout(connect, 2000);
                return;
            }

            pc.onicecandidate = (ev) => {
                if (!ev.candidate || !ws || ws.readyState !== WebSocket.OPEN) return;
                ws.send(JSON.stringify({
                    type: 'webrtc/candidate',
                    value: ev.candidate.candidate,
                }));
            };

            ws.onopen = async () => {
                if (!mounted || !pc) return;
                try {
                    const offer = await pc.createOffer();
                    await pc.setLocalDescription(offer);
                    ws.send(JSON.stringify({ type: 'webrtc/offer', value: offer.sdp }));
                } catch (err) {
                    console.warn(`[WebRTC] offer failed for ${camera?.name}:`, err);
                }
            };

            ws.onmessage = async (ev) => {
                if (!mounted || !pc) return;
                let msg;
                try { msg = JSON.parse(ev.data); } catch (e) { return; }

                if (msg.type === 'webrtc/answer') {
                    try {
                        await pc.setRemoteDescription({ type: 'answer', sdp: msg.value });
                    } catch (err) {
                        console.warn(`[WebRTC] answer rejected for ${camera?.name}:`, err);
                    }
                } else if (msg.type === 'webrtc/candidate' && msg.value) {
                    try {
                        await pc.addIceCandidate({ candidate: msg.value, sdpMid: '0' });
                    } catch (e) { /* a candidate we cannot use is not an error */ }
                } else if (msg.type === 'error') {
                    console.warn(`[WebRTC] go2rtc error for ${camera?.name}:`, msg.value);
                    report('unavailable');
                }
            };

            ws.onclose = () => {
                if (!mounted) return;
                report('reconnecting');
                teardown();
                reconnectTimer = setTimeout(connect, 2000);
            };

            // How far behind live the displayed frame is. Measured, because
            // it varies by an order of magnitude between a LAN camera and a
            // congested link, and the overlay is only as aligned as this is.
            statsTimer = setInterval(async () => {
                if (!pc) return;
                try {
                    const stats = await pc.getStats();
                    stats.forEach((r) => {
                        if (r.type !== 'inbound-rtp' || r.kind !== 'video') return;
                        if (!r.jitterBufferEmittedCount) return;
                        const jitter = (r.jitterBufferDelay / r.jitterBufferEmittedCount) * 1000;
                        if (Number.isFinite(jitter)) {
                            latencyRef.current = Math.min(Math.max(jitter, 0), 2000);
                        }
                    });
                } catch (e) { /* stats are advisory */ }
            }, 2000);
        };

        connect();

        return () => {
            mounted = false;
            if (reconnectTimer) clearTimeout(reconnectTimer);
            teardown();
        };
    }, [signallingUrl, camera?.name, report]);

    // ── Metadata: one message per analysed frame ─────────────────────────
    useEffect(() => {
        if (!showOverlay || !camera?.id) return undefined;

        let mounted = true;
        let ws = null;
        let reconnectTimer = null;
        let keepAlive = null;
        const offsets = [];  // sliding window of (localSec - serverSec)

        const connect = () => {
            if (!mounted) return;
            try {
                ws = api.connectCameraMetadata(camera.id);
            } catch (err) {
                reconnectTimer = setTimeout(connect, 2000);
                return;
            }

            ws.onopen = () => {
                // The server reads this socket only to notice a disconnect;
                // a periodic ping is what keeps proxies from closing it.
                keepAlive = setInterval(() => {
                    if (ws && ws.readyState === WebSocket.OPEN) ws.send('ping');
                }, 25000);
            };

            ws.onmessage = (ev) => {
                if (!mounted) return;
                let msg;
                try { msg = JSON.parse(ev.data); } catch (e) { return; }
                if (msg.type !== 'detections') return;

                const localSec = Date.now() / 1000;

                // The MINIMUM observed (arrival - sent) is the sample least
                // inflated by queueing, so it is the best estimate of clock
                // skew available without a time protocol. The window makes it
                // follow a drifting clock instead of latching on one packet.
                offsets.push(localSec - msg.timestamp);
                if (offsets.length > 60) offsets.shift();
                clockOffsetRef.current = Math.min(...offsets);

                const buf = bufferRef.current;
                buf.push({ t: msg.timestamp, objects: msg.objects || [] });
                const cutoff = msg.timestamp - BUFFER_MS / 1000;
                while (buf.length && buf[0].t < cutoff) buf.shift();
            };

            ws.onclose = () => {
                if (keepAlive) { clearInterval(keepAlive); keepAlive = null; }
                if (!mounted) return;
                reconnectTimer = setTimeout(connect, 2000);
            };

            ws.onerror = () => { /* onclose handles the retry */ };
        };

        connect();

        return () => {
            mounted = false;
            if (reconnectTimer) clearTimeout(reconnectTimer);
            if (keepAlive) clearInterval(keepAlive);
            if (ws) { ws.onclose = null; try { ws.close(); } catch (e) { /* closing */ } }
            bufferRef.current = [];
        };
    }, [camera?.id, showOverlay]);

    // ── Draw: canvas over the video, matched by timestamp ────────────────
    useEffect(() => {
        if (!showOverlay) return undefined;

        let rafId = null;
        let vfcId = null;
        let mounted = true;

        // WHAT DRIVES A REDRAW.
        // requestVideoFrameCallback fires once per video frame the compositor
        // actually PRESENTS, which is exactly when the overlay has something
        // new to say. requestAnimationFrame fires at the display's refresh
        // rate instead -- 60 times a second over a 12 fps camera, so four
        // redraws in five repaint boxes that did not move. On a 3x3 wall that
        // is nine canvases redrawn 60 times a second to show 12 fps of
        // detections, which is the same "spend CPU to draw a picture nobody
        // asked for" that moving video off the JPEG path was meant to stop.
        //
        // rAF remains the fallback: Firefox has no rVFC, and a stalled video
        // presents no frames, so without it a camera that freezes would also
        // freeze the staleness check that clears its boxes.
        const video0 = videoRef.current;
        const useVfc = !!(video0 && typeof video0.requestVideoFrameCallback === 'function');

        const pickFrame = () => {
            const buf = bufferRef.current;
            if (!buf.length) return null;
            const offset = clockOffsetRef.current;
            if (offset === null) return buf[buf.length - 1];

            // The moment the frame currently on screen was captured, in the
            // server's clock.
            const shownAt = Date.now() / 1000 - offset - latencyRef.current / 1000;

            let best = null;
            let bestGap = Infinity;
            for (let i = buf.length - 1; i >= 0; i -= 1) {
                const gap = Math.abs(buf[i].t - shownAt);
                if (gap < bestGap) { bestGap = gap; best = buf[i]; }
                else if (buf[i].t < shownAt) break;  // buffer is time-ordered
            }
            return bestGap * 1000 <= MAX_STALENESS_MS ? best : null;
        };

        const draw = () => {
            if (!mounted) return;
            schedule();

            const canvas = canvasRef.current;
            const video = videoRef.current;
            if (!canvas || !video) return;

            // Match the canvas to the box the video is actually painted in.
            // objectFit: contain letterboxes, so the video's own rectangle is
            // narrower or shorter than the element -- drawing against the
            // element would offset every box by the size of the bars.
            const rect = video.getBoundingClientRect();
            if (!rect.width || !rect.height) return;

            const vw = video.videoWidth || 16;
            const vh = video.videoHeight || 9;
            const scale = Math.min(rect.width / vw, rect.height / vh);
            const drawnW = vw * scale;
            const drawnH = vh * scale;
            const padX = (rect.width - drawnW) / 2;
            const padY = (rect.height - drawnH) / 2;

            const dpr = window.devicePixelRatio || 1;
            if (canvas.width !== Math.round(rect.width * dpr)
                || canvas.height !== Math.round(rect.height * dpr)) {
                canvas.width = Math.round(rect.width * dpr);
                canvas.height = Math.round(rect.height * dpr);
            }

            const ctx = canvas.getContext('2d');
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, rect.width, rect.height);

            const frame = pickFrame();
            if (!frame) return;

            ctx.lineWidth = 2;
            ctx.font = '600 12px ui-sans-serif, system-ui, sans-serif';
            ctx.textBaseline = 'top';

            for (const obj of frame.objects) {
                const [nx1, ny1, nx2, ny2] = obj.box;
                const x = padX + nx1 * drawnW;
                const y = padY + ny1 * drawnH;
                const w = (nx2 - nx1) * drawnW;
                const h = (ny2 - ny1) * drawnH;

                const { color, dashed } = boxStyle(obj);
                ctx.strokeStyle = color;
                ctx.setLineDash(dashed ? [6, 4] : []);
                ctx.strokeRect(x, y, w, h);
                ctx.setLineDash([]);

                if (!obj.label || obj.kind === 'face') continue;

                const text = obj.label;
                const tw = ctx.measureText(text).width;
                const labelY = y - 17 >= 0 ? y - 17 : y + h + 1;
                ctx.fillStyle = color;
                ctx.fillRect(x - 1, labelY, tw + 10, 16);
                ctx.fillStyle = '#0b0f19';
                ctx.fillText(text, x + 4, labelY + 2);
            }
        };

        function schedule() {
            if (!mounted) return;
            const video = videoRef.current;
            if (useVfc && video) vfcId = video.requestVideoFrameCallback(draw);
            else rafId = requestAnimationFrame(draw);
        }

        schedule();
        return () => {
            mounted = false;
            if (rafId) cancelAnimationFrame(rafId);
            // video0, not videoRef.current: the callback was issued by the
            // element that existed when this effect ran, and that is the
            // element that has to cancel it.
            if (vfcId && video0 && video0.cancelVideoFrameCallback) {
                video0.cancelVideoFrameCallback(vfcId);
            }
        };
    }, [showOverlay]);

    return (
        <>
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted={muted}
                aria-label={`${camera?.name || 'Camera'} live video`}
                style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    display: status === 'streaming' ? 'block' : 'none',
                    background: '#000',
                }}
            />
            {showOverlay && (
                <canvas
                    ref={canvasRef}
                    style={{
                        position: 'absolute',
                        inset: 0,
                        width: '100%',
                        height: '100%',
                        pointerEvents: 'none',
                        display: status === 'streaming' ? 'block' : 'none',
                    }}
                />
            )}
        </>
    );
}
