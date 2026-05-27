'use client';

/**
 * DetectionOverlay — Browser-side RF-DETR object detection overlay.
 *
 * Uses a Web Worker + WebGPU to run RF-DETR inference on video frames
 * directly in the browser, drawing bounding boxes on a canvas overlay.
 *
 * Props:
 *   videoRef  — ref to the <img> or <video> element showing the camera feed
 *   enabled   — boolean toggle for detection
 *   modelId   — HuggingFace model ID (default: onnx-community/rfdetr_base-ONNX)
 *   threshold — confidence threshold (default: 0.35)
 *   onStats   — callback({ fps, inferenceMs, detectionCount })
 */

import { useState, useEffect, useRef, useCallback } from 'react';

// Color palette for different object categories
const CATEGORY_COLORS = {
    person: '#22c55e',
    car: '#3b82f6',
    truck: '#8b5cf6',
    bus: '#f59e0b',
    motorcycle: '#ef4444',
    bicycle: '#06b6d4',
    knife: '#dc2626',
    scissors: '#dc2626',
    default: '#a855f7',
};

function getColor(label) {
    const key = label?.toLowerCase() || '';
    return CATEGORY_COLORS[key] || CATEGORY_COLORS.default;
}

export default function DetectionOverlay({
    videoRef,
    enabled = false,
    modelId = 'onnx-community/rfdetr_base-ONNX',
    threshold = 0.35,
    onStats,
}) {
    const canvasRef = useRef(null);
    const workerRef = useRef(null);
    const captureCanvasRef = useRef(null);
    const [status, setStatus] = useState('idle'); // idle | downloading | loading | ready | error
    const [statusMsg, setStatusMsg] = useState('');
    const [detections, setDetections] = useState([]);
    const frameIdRef = useRef(0);
    const runningRef = useRef(false);
    const rafRef = useRef(null);

    // Initialize worker when enabled
    useEffect(() => {
        if (!enabled) {
            if (workerRef.current) {
                workerRef.current.terminate();
                workerRef.current = null;
            }
            setStatus('idle');
            setDetections([]);
            runningRef.current = false;
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
            return;
        }

        const worker = new Worker('/rfdetr-worker.js');
        workerRef.current = worker;

        worker.onmessage = (e) => {
            const { type } = e.data;

            if (type === 'status') {
                setStatus(e.data.status);
                setStatusMsg(e.data.message);
            } else if (type === 'ready') {
                setStatus('ready');
                setStatusMsg('');
                runningRef.current = true;
            } else if (type === 'result') {
                setDetections(e.data.detections || []);
                if (onStats) {
                    onStats({
                        inferenceMs: e.data.inferenceMs,
                        detectionCount: (e.data.detections || []).length,
                    });
                }
                // Schedule next frame capture
                if (runningRef.current) {
                    rafRef.current = requestAnimationFrame(() => captureAndDetect());
                }
            } else if (type === 'error') {
                setStatus('error');
                setStatusMsg(e.data.message);
            }
        };

        // Load model
        worker.postMessage({
            type: 'load',
            modelId,
            device: 'webgpu',
        });

        return () => {
            runningRef.current = false;
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
            worker.terminate();
            workerRef.current = null;
        };
    }, [enabled, modelId]);

    // Start detection loop when ready
    useEffect(() => {
        if (status === 'ready' && enabled) {
            runningRef.current = true;
            captureAndDetect();
        }
    }, [status, enabled]);

    // Capture frame from video/img element and send to worker
    const captureAndDetect = useCallback(() => {
        if (!runningRef.current || !workerRef.current || !videoRef?.current) return;

        const el = videoRef.current;
        const width = el.naturalWidth || el.videoWidth || el.width || 640;
        const height = el.naturalHeight || el.videoHeight || el.height || 480;

        if (width === 0 || height === 0) {
            rafRef.current = requestAnimationFrame(() => captureAndDetect());
            return;
        }

        // Create or reuse offscreen canvas for frame capture
        if (!captureCanvasRef.current) {
            captureCanvasRef.current = document.createElement('canvas');
        }
        const cc = captureCanvasRef.current;
        cc.width = width;
        cc.height = height;

        const ctx = cc.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(el, 0, 0, width, height);

        const imageData = ctx.getImageData(0, 0, width, height);

        frameIdRef.current += 1;
        workerRef.current.postMessage(
            {
                type: 'detect',
                imageData,
                id: frameIdRef.current,
                threshold,
            },
            [imageData.data.buffer], // Transfer ownership for performance
        );
    }, [videoRef, threshold]);

    // Draw detections on canvas overlay
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const parent = canvas.parentElement;
        if (!parent) return;

        // Match canvas size to parent container
        const rect = parent.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!enabled || detections.length === 0) return;

        // Get source image dimensions for coordinate mapping
        const el = videoRef?.current;
        const srcW = el?.naturalWidth || el?.videoWidth || 640;
        const srcH = el?.naturalHeight || el?.videoHeight || 480;

        const scaleX = canvas.width / srcW;
        const scaleY = canvas.height / srcH;

        for (const det of detections) {
            const { label, score, box } = det;
            const color = getColor(label);

            const x = box.x * scaleX;
            const y = box.y * scaleY;
            const w = box.w * scaleX;
            const h = box.h * scaleY;

            // Bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);

            // Label background
            const text = `${label} ${(score * 100).toFixed(0)}%`;
            ctx.font = 'bold 12px Inter, system-ui, sans-serif';
            const textMetrics = ctx.measureText(text);
            const textW = textMetrics.width + 8;
            const textH = 18;

            ctx.fillStyle = color;
            ctx.fillRect(x, y - textH, textW, textH);

            // Label text
            ctx.fillStyle = '#fff';
            ctx.fillText(text, x + 4, y - 5);
        }
    }, [detections, enabled, videoRef]);

    // Loading overlay
    if (enabled && status !== 'ready' && status !== 'idle') {
        return (
            <div
                style={{
                    position: 'absolute',
                    inset: 0,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: 'rgba(0, 0, 0, 0.6)',
                    backdropFilter: 'blur(4px)',
                    zIndex: 10,
                    borderRadius: 'inherit',
                    gap: '12px',
                }}
            >
                {status === 'error' ? (
                    <>
                        <div style={{ color: '#ef4444', fontWeight: 600, fontSize: '0.875rem' }}>
                            ⚠ Detection Error
                        </div>
                        <div style={{ color: '#999', fontSize: '0.75rem', maxWidth: '280px', textAlign: 'center' }}>
                            {statusMsg}
                        </div>
                    </>
                ) : (
                    <>
                        <div
                            style={{
                                width: '36px',
                                height: '36px',
                                border: '3px solid rgba(255,255,255,0.15)',
                                borderTopColor: '#818cf8',
                                borderRadius: '50%',
                                animation: 'spin 1s linear infinite',
                            }}
                        />
                        <div style={{ color: '#e2e8f0', fontWeight: 600, fontSize: '0.813rem' }}>
                            {status === 'downloading' ? 'Downloading RF-DETR Model...' :
                             status === 'loading' ? 'Initializing WebGPU...' :
                             status === 'fallback' ? 'Falling back to WASM...' :
                             'Loading...'}
                        </div>
                        <div style={{ color: '#94a3b8', fontSize: '0.688rem' }}>
                            {statusMsg || 'First load may take 30-60 seconds'}
                        </div>
                    </>
                )}
            </div>
        );
    }

    if (!enabled) return null;

    return (
        <canvas
            ref={canvasRef}
            style={{
                position: 'absolute',
                inset: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                zIndex: 5,
            }}
        />
    );
}
