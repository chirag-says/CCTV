'use client';

/**
 * Ground-Plane Calibration — teaches each camera where the floor is.
 *
 * WHY THIS PAGE EXISTS
 * Two cameras facing each other cannot recognise the same person by
 * appearance: the one behind them sees no face, and too few body pixels to
 * match on. What they share is the FLOOR. Once each knows how its image maps
 * onto that floor, both compute the same position for a person standing there
 * — and "same place" becomes "same person".
 *
 * THE ORDERING TRAP THIS UI AVOIDS
 * The four points must be the SAME physical spots, entered in the SAME order,
 * in every camera of a zone. That is the only thing tying the cameras into one
 * coordinate frame. Asking for "near-left corner" per camera would guarantee
 * mistakes, because for two cameras facing each other, one's near-left corner
 * IS the other's far-right. So the points are defined once for the ZONE as
 * named landmarks, and each camera is asked to click those same landmarks in
 * that fixed order.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import AppShell from '@/components/AppShell';
import api from '@/lib/api';
import { VideoCameraIcon, UsersIcon, DetectionIcon } from '@/components/Icons';
import HandoffEditor from './HandoffEditor';

// Default landmark names. Deliberately about a rectangle on the floor, since
// that is what makes the world coordinates trivial to derive from two measured
// distances rather than needing a surveyed floor plan.
const DEFAULT_LANDMARKS = [
    '1 — near-left, by the wall',
    '2 — near-right, opposite wall',
    '3 — far-right, same wall as 2',
    '4 — far-left, same wall as 1',
];

const MARKER_COLORS = ['#ef4444', '#f59e0b', '#22c55e', '#3b82f6'];

export default function CalibrationPage() {
    const [zones, setZones] = useState([]);
    const [selectedZone, setSelectedZone] = useState(null);
    const [selectedCamera, setSelectedCamera] = useState(null);

    const [width, setWidth] = useState('4');
    const [length, setLength] = useState('10');
    const [landmarks, setLandmarks] = useState(DEFAULT_LANDMARKS);

    const [points, setPoints] = useState([]);          // normalised [x, y]
    const [snapshotUrl, setSnapshotUrl] = useState(null);
    const [snapshotError, setSnapshotError] = useState(null);
    const [loadingSnapshot, setLoadingSnapshot] = useState(false);

    const [saving, setSaving] = useState(false);
    const [message, setMessage] = useState(null);      // {type, text}
    const [zoneLive, setZoneLive] = useState(null);
    const [loading, setLoading] = useState(true);

    // The other cameras in this zone, with whatever points they already have.
    // Shown alongside the one being calibrated, because the whole difficulty
    // is knowing WHICH physical spot to click — and that is far easier to
    // judge by looking at both views at once than from any written
    // instruction.
    const [refShots, setRefShots] = useState([]);

    const snapshotUrlRef = useRef(null);
    const refUrlsRef = useRef([]);
    const livePollRef = useRef(null);

    // ── Data loading ──────────────────────────────────────────

    const loadZones = useCallback(async () => {
        try {
            const result = await api.getZones();
            setZones(result || []);
            return result || [];
        } catch (err) {
            setMessage({ type: 'error', text: `Could not load zones: ${err.message}` });
            return [];
        }
    }, []);

    useEffect(() => {
        (async () => {
            const result = await loadZones();
            // Open on the zone most likely to be worked on: the one with the
            // most cameras that is not yet ready for cross-camera tracking.
            const target =
                result.find((z) => z.camera_count > 1 && !z.cross_camera_ready) ||
                result.find((z) => z.camera_count > 1) ||
                result[0];
            if (target) setSelectedZone(target.zone);
            setLoading(false);
        })();
        return () => {
            if (snapshotUrlRef.current) URL.revokeObjectURL(snapshotUrlRef.current);
            refUrlsRef.current.forEach((u) => URL.revokeObjectURL(u));
            if (livePollRef.current) clearInterval(livePollRef.current);
        };
    }, [loadZones]);

    // Load the other cameras in the zone as visual reference.
    const loadReferences = useCallback(async (zoneObj, excludeId) => {
        refUrlsRef.current.forEach((u) => URL.revokeObjectURL(u));
        refUrlsRef.current = [];
        setRefShots([]);
        if (!zoneObj) return;

        const others = zoneObj.cameras.filter((c) => c.id !== excludeId);
        const loaded = [];
        for (const cam of others) {
            try {
                const [url, cal] = await Promise.all([
                    api.getCameraSnapshotUrl(cam.id),
                    api.getCalibration(cam.id).catch(() => null),
                ]);
                refUrlsRef.current.push(url);
                loaded.push({
                    id: cam.id,
                    name: cam.name,
                    url,
                    points: cal?.calibrated ? (cal.image_points || []).slice(0, 4) : [],
                });
            } catch {
                // A camera that is not streaming simply cannot be a reference.
            }
        }
        setRefShots(loaded);
    }, []);

    // Live zone state — the observable proof the feature works. A person seen
    // by 2 cameras appears ONCE here, with a camera count of 2.
    useEffect(() => {
        if (livePollRef.current) clearInterval(livePollRef.current);
        if (!selectedZone) return;

        const poll = async () => {
            try {
                setZoneLive(await api.getZoneLive(selectedZone));
            } catch {
                // A zone with no active tracker yet is normal, not an error.
            }
        };
        poll();
        livePollRef.current = setInterval(poll, 2000);
        return () => {
            if (livePollRef.current) clearInterval(livePollRef.current);
        };
    }, [selectedZone]);

    const zone = zones.find((z) => z.zone === selectedZone) || null;

    // ── Camera selection ──────────────────────────────────────

    async function selectCamera(camera) {
        setSelectedCamera(camera);
        setPoints([]);
        setMessage(null);
        setSnapshotError(null);

        // Load any existing calibration so an operator can see and adjust it
        // rather than being forced to start over.
        try {
            const existing = await api.getCalibration(camera.id);
            if (existing?.calibrated && existing.image_points?.length >= 4) {
                setPoints(existing.image_points.slice(0, 4));
                if (existing.world_points?.length >= 4) {
                    const w = existing.world_points[1]?.[0];
                    const l = existing.world_points[2]?.[1];
                    if (w) setWidth(String(w));
                    if (l) setLength(String(l));
                }
            }
        } catch {
            // No calibration yet is the normal case.
        }

        setLoadingSnapshot(true);
        if (snapshotUrlRef.current) {
            URL.revokeObjectURL(snapshotUrlRef.current);
            snapshotUrlRef.current = null;
        }
        setSnapshotUrl(null);
        try {
            const url = await api.getCameraSnapshotUrl(camera.id);
            snapshotUrlRef.current = url;
            setSnapshotUrl(url);
        } catch (err) {
            setSnapshotError(err.message);
        }
        setLoadingSnapshot(false);

        const zoneObj = zones.find((z) => z.zone === selectedZone);
        loadReferences(zoneObj, camera.id);
    }

    async function refreshSnapshot() {
        if (!selectedCamera) return;
        setLoadingSnapshot(true);
        setSnapshotError(null);
        try {
            const url = await api.getCameraSnapshotUrl(selectedCamera.id);
            if (snapshotUrlRef.current) URL.revokeObjectURL(snapshotUrlRef.current);
            snapshotUrlRef.current = url;
            setSnapshotUrl(url);
        } catch (err) {
            setSnapshotError(err.message);
        }
        setLoadingSnapshot(false);
    }

    // ── Point placement ───────────────────────────────────────

    function handleImageClick(e) {
        if (points.length >= 4) return;
        const rect = e.currentTarget.getBoundingClientRect();
        // Normalised to the displayed frame, so the calibration is independent
        // of capture resolution and of how large this page renders it.
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        setPoints((prev) => [...prev, [
            Math.min(1, Math.max(0, x)),
            Math.min(1, Math.max(0, y)),
        ]]);
        setMessage(null);
    }

    const worldPoints = (() => {
        const w = parseFloat(width);
        const l = parseFloat(length);
        if (!(w > 0) || !(l > 0)) return null;
        return [[0, 0], [w, 0], [w, l], [0, l]];
    })();

    async function save() {
        if (!selectedCamera || points.length < 4 || !worldPoints) return;
        setSaving(true);
        setMessage(null);
        try {
            const result = await api.saveCalibration(selectedCamera.id, {
                zone: selectedZone,
                imagePoints: points,
                worldPoints,
            });
            const cm = Math.round((result.residual || 0) * 100);
            setMessage({
                type: cm > 50 ? 'warn' : 'success',
                text:
                    cm > 50
                        ? `Saved, but the fit is loose (${cm} cm error on your own points). Check that you clicked the landmarks in the same order as the other camera.`
                        : `Saved. Fit error ${cm} cm. The camera picks this up within 30 seconds.`,
            });
            await loadZones();
        } catch (err) {
            setMessage({ type: 'error', text: err.message });
        }
        setSaving(false);
    }

    async function clearCalibration() {
        if (!selectedCamera) return;
        setSaving(true);
        try {
            await api.deleteCalibration(selectedCamera.id);
            setPoints([]);
            setMessage({ type: 'success', text: 'Calibration removed.' });
            await loadZones();
        } catch (err) {
            setMessage({ type: 'error', text: err.message });
        }
        setSaving(false);
    }

    const canSave = points.length === 4 && worldPoints && !saving;

    return (
        <AppShell>
            <div className="page-header">
                <h1>Cross-Camera Tracking</h1>
                <p>
                    Teach each camera where the floor is, so cameras in the same zone can
                    agree on where a person is standing — even when only one of them can
                    see their face — and mark where people pass between cameras that do not
                    overlap, so their number follows them.
                </p>
            </div>

            {loading ? (
                <div className="card"><div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>Loading zones…</div></div>
            ) : (
            <>
            {/* Zone picker */}
            <div className="card" style={{ marginBottom: '16px' }}>
                <div className="card-header"><h3 className="card-title">1 · Pick a zone</h3></div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                    {zones.map((z) => (
                        <button
                            key={z.zone}
                            className={`btn btn-sm ${selectedZone === z.zone ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => { setSelectedZone(z.zone); setSelectedCamera(null); setPoints([]); }}
                            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                        >
                            {z.zone}
                            <span className={`badge ${z.cross_camera_ready ? 'online' : 'offline'}`} style={{ fontSize: '0.625rem' }}>
                                {z.calibrated_count}/{z.camera_count}
                            </span>
                        </button>
                    ))}
                </div>
                {zone && !zone.cross_camera_ready && (
                    <div style={{ marginTop: '10px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        Floor calibration is <strong>optional</strong>: cameras learn where their views overlap
                        from people walking (step 5). Calibrate two or more here only if you also want
                        positions in metres. {zone.calibrated_count} of {zone.camera_count} usable.
                    </div>
                )}
            </div>

            {/* Reference rectangle */}
            <div className="card" style={{ marginBottom: '16px' }}>
                <div className="card-header"><h3 className="card-title">2 · Define the reference rectangle</h3></div>
                <div style={{
                    fontSize: '0.813rem', color: 'var(--text-secondary)', marginTop: 0,
                    background: 'var(--bg-tertiary)', padding: '12px 14px',
                    borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--accent-light)',
                }}>
                    <strong>Easiest reliable way:</strong> walk into the corridor and stick four bits of
                    masking tape on the floor, roughly a rectangle, somewhere <em>both</em> cameras can
                    see. Measure across and along it once. Then you are just clicking tape in each
                    view, which is unambiguous.
                    <div style={{ marginTop: '8px' }}>
                        No tape? Use things already on the floor that both cameras see — a chair leg,
                        the base of a door frame, the corner of a floor tile. Avoid anything that can
                        move.
                    </div>
                    <div style={{ marginTop: '8px', color: 'var(--warning)' }}>
                        The one thing that must be right is the <strong>order</strong>. Point 1 has to be the
                        same physical spot in every camera. For cameras facing each other this is
                        counter-intuitive: one camera&rsquo;s near-left corner is the other camera&rsquo;s
                        far-right, so the clicks land in mirrored places on screen. That is correct.
                    </div>
                </div>
                <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', alignItems: 'flex-end' }}>
                    <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        Width (m)
                        <input type="number" step="0.1" min="0.1" value={width}
                            onChange={(e) => setWidth(e.target.value)}
                            className="input" style={{ display: 'block', width: '110px', marginTop: '4px' }} />
                    </label>
                    <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        Length (m)
                        <input type="number" step="0.1" min="0.1" value={length}
                            onChange={(e) => setLength(e.target.value)}
                            className="input" style={{ display: 'block', width: '110px', marginTop: '4px' }} />
                    </label>
                    <div style={{ fontSize: '0.688rem', color: 'var(--text-muted)', maxWidth: '380px' }}>
                        Rough measurements are fine — they set the scale, and association
                        tolerates about half a metre of error. Getting the <em>order</em> right
                        matters far more than getting the tape measure right.
                    </div>
                </div>
                <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '8px' }}>
                    {landmarks.map((name, i) => (
                        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{
                                width: '22px', height: '22px', borderRadius: '50%', flexShrink: 0,
                                background: MARKER_COLORS[i], color: '#fff', fontWeight: 700,
                                fontSize: '0.688rem', display: 'flex', alignItems: 'center', justifyContent: 'center',
                            }}>{i + 1}</span>
                            <input
                                className="input"
                                value={name}
                                onChange={(e) => setLandmarks((prev) => prev.map((v, j) => (j === i ? e.target.value : v)))}
                                style={{ flex: 1, fontSize: '0.75rem' }}
                                placeholder={`Landmark ${i + 1}`}
                            />
                        </div>
                    ))}
                </div>
                <div style={{ marginTop: '8px', fontSize: '0.688rem', color: 'var(--text-muted)' }}>
                    Rename these to something you can recognise from both ends of the corridor
                    (&ldquo;corner by the water cooler&rdquo;). The names are just a memory aid for you —
                    what matters is that point 1 is the same physical spot in every camera.
                </div>
            </div>

            {/* Camera + clicking */}
            <div className="card" style={{ marginBottom: '16px' }}>
                <div className="card-header"><h3 className="card-title">3 · Mark the points in each camera</h3></div>
                {!zone ? (
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.813rem' }}>Pick a zone above.</div>
                ) : (
                    <>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '14px' }}>
                            {zone.cameras.map((cam) => (
                                <button
                                    key={cam.id}
                                    className={`btn btn-sm ${selectedCamera?.id === cam.id ? 'btn-primary' : 'btn-secondary'}`}
                                    onClick={() => selectCamera(cam)}
                                    style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                                >
                                    <VideoCameraIcon size={13} />
                                    {cam.name}
                                    {/* Saved is not usable: four clicks always fit themselves, so a
                                        residual says nothing about whether they are on the floor. */}
                                    {cam.calibrated && cam.usable !== false && (
                                        <span className="badge online" style={{ fontSize: '0.563rem' }}>
                                            {Math.round((cam.residual || 0) * 100)}cm
                                        </span>
                                    )}
                                    {cam.calibrated && cam.usable === false && (
                                        <span className="badge offline" style={{ fontSize: '0.563rem' }}>redo</span>
                                    )}
                                </button>
                            ))}
                        </div>

                        {selectedCamera && (
                            <>
                                {(zone.cameras.find((c) => c.id === selectedCamera.id)?.problems || []).length > 0 && (
                                    <div style={{
                                        marginBottom: '12px', padding: '10px 12px', borderRadius: 'var(--radius-sm)',
                                        fontSize: '0.813rem', background: 'var(--bg-tertiary)',
                                        borderLeft: '3px solid var(--danger)',
                                    }}>
                                        <strong>This camera&rsquo;s saved calibration is being ignored.</strong>
                                        {zone.cameras.find((c) => c.id === selectedCamera.id).problems.map((p, i) => (
                                            <div key={i} style={{ marginTop: '4px', color: 'var(--text-secondary)' }}>{p}</div>
                                        ))}
                                        <div style={{ marginTop: '4px', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                                            Clear the points and click four spots on the <em>floor</em> — tile corners work well —
                                            or remove the calibration. Cross-camera numbering does not depend on it.
                                        </div>
                                    </div>
                                )}
                                <div style={{
                                    display: 'flex', alignItems: 'center', gap: '10px',
                                    marginBottom: '10px', flexWrap: 'wrap',
                                }}>
                                    <strong style={{ fontSize: '0.875rem' }}>
                                        {points.length < 4
                                            ? `Click landmark ${points.length + 1}: ${landmarks[points.length]}`
                                            : 'All four points placed.'}
                                    </strong>
                                    <span style={{
                                        width: '18px', height: '18px', borderRadius: '50%',
                                        background: MARKER_COLORS[Math.min(points.length, 3)],
                                        display: points.length < 4 ? 'inline-block' : 'none',
                                    }} />
                                    <div style={{ flex: 1 }} />
                                    <button className="btn btn-sm btn-secondary" onClick={() => setPoints((p) => p.slice(0, -1))} disabled={!points.length}>
                                        Undo
                                    </button>
                                    <button className="btn btn-sm btn-secondary" onClick={() => setPoints([])} disabled={!points.length}>
                                        Clear
                                    </button>
                                    <button className="btn btn-sm btn-secondary" onClick={refreshSnapshot} disabled={loadingSnapshot}>
                                        {loadingSnapshot ? 'Loading…' : 'New frame'}
                                    </button>
                                </div>

                                {snapshotError ? (
                                    <div style={{
                                        padding: '30px', textAlign: 'center', background: 'var(--bg-tertiary)',
                                        borderRadius: 'var(--radius-sm)', color: 'var(--danger)', fontSize: '0.813rem',
                                    }}>
                                        {snapshotError}
                                        <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: '6px' }}>
                                            The camera has to be running to give a frame to click on.
                                        </div>
                                    </div>
                                ) : snapshotUrl ? (
                                    <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%', lineHeight: 0 }}>
                                        {/* eslint-disable-next-line @next/next/no-img-element */}
                                        <img
                                            src={snapshotUrl}
                                            alt={`${selectedCamera.name} frame for calibration`}
                                            onClick={handleImageClick}
                                            style={{
                                                maxWidth: '100%', display: 'block',
                                                cursor: points.length < 4 ? 'crosshair' : 'default',
                                                borderRadius: 'var(--radius-sm)',
                                            }}
                                        />
                                        <svg
                                            viewBox="0 0 100 100"
                                            preserveAspectRatio="none"
                                            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                                        >
                                            {points.length > 1 && (
                                                <polygon
                                                    points={points.map(([x, y]) => `${x * 100},${y * 100}`).join(' ')}
                                                    fill="rgba(99,102,241,0.18)"
                                                    stroke="rgba(99,102,241,0.9)"
                                                    strokeWidth="0.3"
                                                    vectorEffect="non-scaling-stroke"
                                                />
                                            )}
                                        </svg>
                                        {points.map(([x, y], i) => (
                                            <div
                                                key={i}
                                                style={{
                                                    position: 'absolute',
                                                    left: `${x * 100}%`,
                                                    top: `${y * 100}%`,
                                                    transform: 'translate(-50%, -50%)',
                                                    width: '22px', height: '22px', borderRadius: '50%',
                                                    background: MARKER_COLORS[i], color: '#fff',
                                                    border: '2px solid rgba(255,255,255,0.9)',
                                                    fontSize: '0.688rem', fontWeight: 700,
                                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                    pointerEvents: 'none',
                                                }}
                                            >
                                                {i + 1}
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.813rem' }}>
                                        {loadingSnapshot ? 'Fetching a frame…' : 'Select a camera to load a frame.'}
                                    </div>
                                )}

                                {/* The other cameras, so the same physical spot can be
                                    found by eye instead of from a description. */}
                                {refShots.length > 0 && (
                                    <div style={{ marginTop: '16px' }}>
                                        <div style={{ fontSize: '0.75rem', fontWeight: 600, marginBottom: '6px' }}>
                                            Same zone, for reference — find the <em>same physical spot</em> here
                                        </div>
                                        <div style={{
                                            display: 'grid',
                                            gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
                                            gap: '10px',
                                        }}>
                                            {refShots.map((ref) => (
                                                <div key={ref.id}>
                                                    <div style={{ position: 'relative', lineHeight: 0 }}>
                                                        {/* eslint-disable-next-line @next/next/no-img-element */}
                                                        <img
                                                            src={ref.url}
                                                            alt={`${ref.name} reference view`}
                                                            style={{ width: '100%', display: 'block', borderRadius: 'var(--radius-sm)' }}
                                                        />
                                                        {ref.points.map(([x, y], i) => (
                                                            <div key={i} style={{
                                                                position: 'absolute',
                                                                left: `${x * 100}%`, top: `${y * 100}%`,
                                                                transform: 'translate(-50%, -50%)',
                                                                width: '18px', height: '18px', borderRadius: '50%',
                                                                background: MARKER_COLORS[i], color: '#fff',
                                                                border: '2px solid rgba(255,255,255,0.9)',
                                                                fontSize: '0.625rem', fontWeight: 700,
                                                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                            }}>{i + 1}</div>
                                                        ))}
                                                    </div>
                                                    <div style={{ fontSize: '0.688rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                                                        {ref.name}
                                                        {ref.points.length
                                                            ? ` — ${ref.points.length} point${ref.points.length > 1 ? 's' : ''} marked`
                                                            : ' — not calibrated yet'}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                        <div style={{ fontSize: '0.688rem', color: 'var(--text-muted)', marginTop: '6px' }}>
                                            Cameras facing each other see the corridor reversed: what is close on
                                            one is far on the other, and left and right swap. Match by the
                                            <strong> objects on the floor</strong> (a chair leg, a doorway edge), never by
                                            where it sits in the picture.
                                        </div>
                                    </div>
                                )}

                                {message && (
                                    <div style={{
                                        marginTop: '12px', padding: '10px 12px', borderRadius: 'var(--radius-sm)',
                                        fontSize: '0.813rem',
                                        background: 'var(--bg-tertiary)',
                                        borderLeft: `3px solid ${message.type === 'error' ? 'var(--danger)'
                                            : message.type === 'warn' ? 'var(--warning)' : 'var(--success)'}`,
                                    }}>
                                        {message.text}
                                    </div>
                                )}

                                <div style={{ marginTop: '14px', display: 'flex', gap: '8px', alignItems: 'center' }}>
                                    <button className="btn btn-primary" onClick={save} disabled={!canSave}>
                                        {saving ? 'Saving…' : 'Save calibration'}
                                    </button>
                                    {selectedCamera.calibrated && (
                                        <button className="btn btn-secondary btn-sm" onClick={clearCalibration} disabled={saving}>
                                            Remove calibration
                                        </button>
                                    )}
                                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                        {points.length}/4 points placed
                                    </span>
                                </div>
                            </>
                        )}
                    </>
                )}
            </div>

            {zone && <HandoffEditor key={zone.zone} zone={zone.zone} cameras={zone.cameras} />}

            {/* Live proof */}
            <div className="card">
                <div className="card-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <UsersIcon size={16} /> Live in {selectedZone || 'zone'}
                    </h3>
                    {zoneLive?.stats && (
                        <span style={{ fontSize: '0.688rem', color: 'var(--text-muted)' }}>
                            {zoneLive.stats.multi_camera} of {zoneLive.stats.people} seen by more than one camera
                        </span>
                    )}
                </div>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 0 }}>
                    One row is one person, not one camera. A row listing two cameras is
                    cross-camera association working.
                </p>
                {zoneLive?.people?.length ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        {zoneLive.people.map((p) => (
                            <div key={p.zone_person_id} style={{
                                display: 'flex', alignItems: 'center', gap: '10px',
                                padding: '8px 10px', borderRadius: 'var(--radius-sm)',
                                background: 'var(--bg-tertiary)',
                                borderLeft: `3px solid ${p.camera_count > 1 ? 'var(--success)' : 'var(--border-color)'}`,
                            }}>
                                <DetectionIcon size={14} />
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{ fontSize: '0.813rem', fontWeight: 600 }}>
                                        {p.person_name || p.unknown_label || 'Unidentified'}
                                        {p.identified_by && (
                                            <span style={{ fontWeight: 400, color: 'var(--text-muted)', fontSize: '0.688rem' }}>
                                                {' '}· recognised by {p.identified_by}
                                            </span>
                                        )}
                                    </div>
                                    <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)' }}>
                                        ({p.x} m, {p.y} m) ±{p.sigma} m · {p.cameras.join(', ')}
                                    </div>
                                </div>
                                <span className={`badge ${p.camera_count > 1 ? 'online' : 'offline'}`} style={{ fontSize: '0.625rem' }}>
                                    {p.camera_count} cam{p.camera_count > 1 ? 's' : ''}
                                </span>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div style={{ padding: '16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                        Nobody in this zone right now, or no camera here is calibrated yet.
                    </div>
                )}
            </div>
            </>
            )}
        </AppShell>
    );
}
