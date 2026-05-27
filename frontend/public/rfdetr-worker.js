/**
 * RF-DETR Web Worker — Runs object detection off the main thread.
 * 
 * Uses @huggingface/transformers with WebGPU acceleration to run
 * RF-DETR ONNX models directly in the browser.
 * 
 * Messages IN:
 *   { type: 'load', modelId: string, device: 'webgpu'|'wasm' }
 *   { type: 'detect', imageData: ImageData, id: number, threshold: number }
 * 
 * Messages OUT:
 *   { type: 'status', status: string, message: string, progress?: number }
 *   { type: 'ready' }
 *   { type: 'result', id: number, detections: Array, inferenceMs: number }
 *   { type: 'error', message: string }
 */

let detector = null;
let isLoading = false;
let transformersModule = null;

self.onmessage = async (event) => {
    const { type } = event.data;

    if (type === 'load') {
        if (isLoading || detector) return;
        isLoading = true;
        await loadModel(event.data);
        isLoading = false;
    } else if (type === 'detect') {
        if (!detector) {
            self.postMessage({ type: 'error', message: 'Model not loaded yet' });
            return;
        }
        await runDetection(event.data);
    }
};

async function loadModel({ modelId, device }) {
    try {
        self.postMessage({
            type: 'status',
            status: 'downloading',
            message: 'Loading Transformers.js library...',
            progress: 0,
        });

        // Dynamic import of transformers.js
        transformersModule = await import(
            'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3'
        );
        const { pipeline, env } = transformersModule;

        // Disable local model caching in worker context
        env.allowLocalModels = false;

        const finalModelId = modelId || 'onnx-community/rfdetr_base-ONNX';

        // Check WebGPU availability
        let finalDevice = device || 'webgpu';
        if (finalDevice === 'webgpu') {
            try {
                if (!navigator.gpu) {
                    throw new Error('navigator.gpu not available');
                }
                const adapter = await navigator.gpu.requestAdapter();
                if (!adapter) {
                    throw new Error('No GPU adapter found');
                }
                self.postMessage({
                    type: 'status',
                    status: 'downloading',
                    message: 'WebGPU detected ✓ — Downloading model files...',
                    progress: 5,
                });
            } catch (gpuErr) {
                finalDevice = 'wasm';
                self.postMessage({
                    type: 'status',
                    status: 'downloading',
                    message: `WebGPU not available (${gpuErr.message}). Using WASM fallback — Downloading model...`,
                    progress: 5,
                });
            }
        }

        self.postMessage({
            type: 'status',
            status: 'downloading',
            message: `Downloading ${finalModelId.split('/').pop()} (${finalDevice})... This may take 1-3 minutes on first load.`,
            progress: 10,
        });

        // Use progress_callback to track download progress
        let lastProgressTime = Date.now();
        detector = await pipeline('object-detection', finalModelId, {
            device: finalDevice,
            dtype: 'fp32',
            progress_callback: (progressInfo) => {
                // progressInfo: { status, name, file, progress, loaded, total }
                const now = Date.now();
                // Throttle updates to avoid message spam (every 500ms)
                if (now - lastProgressTime < 500 && progressInfo.status === 'progress') return;
                lastProgressTime = now;

                if (progressInfo.status === 'initiate') {
                    const fileName = progressInfo.file || progressInfo.name || '';
                    self.postMessage({
                        type: 'status',
                        status: 'downloading',
                        message: `Downloading: ${fileName.split('/').pop() || 'model files'}...`,
                        progress: 15,
                    });
                } else if (progressInfo.status === 'progress') {
                    const pct = progressInfo.progress ?? 0;
                    // Map model download progress (0-100) to our range (15-85)
                    const mappedProgress = 15 + (pct / 100) * 70;
                    const loaded = progressInfo.loaded ? `${(progressInfo.loaded / 1024 / 1024).toFixed(0)}MB` : '';
                    const total = progressInfo.total ? `${(progressInfo.total / 1024 / 1024).toFixed(0)}MB` : '';
                    const sizeStr = loaded && total ? ` (${loaded}/${total})` : '';
                    self.postMessage({
                        type: 'status',
                        status: 'downloading',
                        message: `Downloading model: ${pct.toFixed(0)}%${sizeStr}`,
                        progress: Math.round(mappedProgress),
                    });
                } else if (progressInfo.status === 'done') {
                    self.postMessage({
                        type: 'status',
                        status: 'loading',
                        message: 'Initializing model on device...',
                        progress: 90,
                    });
                }
            },
        });

        self.postMessage({
            type: 'status',
            status: 'ready',
            message: `Model ready on ${finalDevice}`,
            progress: 100,
        });
        self.postMessage({ type: 'ready' });

    } catch (err) {
        const errMsg = err instanceof Error ? err.message : (typeof err === 'string' ? err : JSON.stringify(err));
        console.error('RF-DETR Worker load error:', errMsg);

        // Fallback to WASM if WebGPU was attempted
        if (device === 'webgpu' && !detector) {
            self.postMessage({
                type: 'status',
                status: 'fallback',
                message: `WebGPU failed: ${errMsg}. Retrying with WASM...`,
                progress: 10,
            });
            try {
                if (!transformersModule) {
                    transformersModule = await import(
                        'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3'
                    );
                }
                const { pipeline, env } = transformersModule;
                env.allowLocalModels = false;
                const finalModelId = modelId || 'onnx-community/rfdetr_base-ONNX';

                detector = await pipeline('object-detection', finalModelId, {
                    device: 'wasm',
                    dtype: 'fp32',
                    progress_callback: (progressInfo) => {
                        if (progressInfo.status === 'progress') {
                            const pct = progressInfo.progress ?? 0;
                            self.postMessage({
                                type: 'status',
                                status: 'downloading',
                                message: `[WASM] Downloading: ${pct.toFixed(0)}%`,
                                progress: 15 + (pct / 100) * 70,
                            });
                        }
                    },
                });
                self.postMessage({ type: 'ready' });
            } catch (fallbackErr) {
                const fbMsg = fallbackErr instanceof Error ? fallbackErr.message : String(fallbackErr);
                self.postMessage({
                    type: 'error',
                    message: `Model load failed on both WebGPU and WASM: ${fbMsg}`,
                });
            }
        } else {
            self.postMessage({
                type: 'error',
                message: `Model load failed: ${errMsg}`,
            });
        }
    }
}

async function runDetection({ imageData, id, threshold }) {
    try {
        const start = performance.now();

        // Convert ImageData to a RawImage-compatible object
        const { RawImage } = transformersModule;

        const image = new RawImage(
            new Uint8ClampedArray(imageData.data),
            imageData.width,
            imageData.height,
            4, // RGBA channels
        );

        const results = await detector(image, {
            threshold: threshold || 0.35,
            percentage: false,
        });

        const inferenceMs = performance.now() - start;

        // Map results to a clean format
        const detections = results.map((det) => ({
            label: det.label,
            score: Math.round(det.score * 1000) / 1000,
            box: {
                x: Math.round(det.box.xmin),
                y: Math.round(det.box.ymin),
                w: Math.round(det.box.xmax - det.box.xmin),
                h: Math.round(det.box.ymax - det.box.ymin),
            },
        }));

        self.postMessage({
            type: 'result',
            id,
            detections,
            inferenceMs: Math.round(inferenceMs),
        });
    } catch (err) {
        self.postMessage({
            type: 'error',
            message: `Detection failed: ${err.message}`,
            id,
        });
    }
}
