'use client';

/**
 * Handoff areas — how a person's number survives a walk between cameras that
 * do not see the same floor (the COE room camera and the corridor).
 *
 * The operator marks, on each camera, the region where people LEAVE its view
 * towards the next camera and where they APPEAR coming from it, then links the
 * two regions with the longest plausible walk between them. A track that ends
 * in one linked area and a track that starts in the other within that time
 * are the same person. The door itself does not need to be visible: the edge
 * of the frame people walk out of is the area.
 *
 * Links are edited zone-wide in one document because a link has an end on two
 * cameras; editing it from one camera at a time is how the halves drift.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import api from '@/lib/api';
import { VideoCameraIcon, UsersIcon } from '@/components/Icons';

const AREA_COLORS = ['#f59e0b', '#22c55e', '#3b82f6', '#ec4899', '#a855f7', '#14b8a6', '#ef4444', '#84cc16'];
const MIN_SIZE = 0.02;
const DEFAULT_GAP = 10;

const refKey = (cameraId, areaId) => `${cameraId}|${areaId}`;

function newAreaId() {
    return `a${Date.now().toString(36)}${Math.random().toString(36).slice(2, 5)}`;
}

// The kinds of evidence a number can be carried by, in the order the engine
// trusts them. Keys are the engine's stats counters.
const EVIDENCE = [
    ['handoff_name', 'name'],
    ['handoff_floor', 'floor'],
    ['handoff_view', 'shared view'],
    ['handoff_stitch', 'same camera'],
    ['handoff_area', 'area'],
    ['handoff_route', 'route'],
    ['handoff_look', 'appearance'],
];

/**
 * What handoff has worked out about this zone by itself, and what is standing
 * in its way. This card exists because the first version of handoff could sit
 * at zero forever and nothing on screen said why: a calibration clicked on a
 * door post still showed "0 cm".
 */
function LearnedTopology({ zone, live }) {
    const [busy, setBusy] = useState(false);
    const [note, setNote] = useState(null);
    if (!live?.enabled) return null;

    const overlaps = live.learned?.overlaps || [];
    const routes = live.learned?.routes || [];
    const unusable = Object.entries(live.evidence?.floor_unusable || {});

    async function forget() {
        if (!window.confirm(`Forget everything handoff has learned about ${zone}? It relearns from ordinary traffic, usually within the hour. Calibrations, areas and links are not touched.`)) return;
        setBusy(true);
        try {
            const result = await api.forgetZoneLearned(zone);
            setNote(`Forgot ${result.forgotten} learned relation${result.forgotten === 1 ? '' : 's'}. Relearning from now.`);
        } catch (err) {
            setNote(err.message);
        }
        setBusy(false);
    }

    const row = { display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 10px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-tertiary)', fontSize: '0.75rem' };

    return (
        <div className="card" style={{ marginBottom: '16px' }}>
            <div className="card-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <h3 className="card-title">5 · What the cameras have learned</h3>
                {(overlaps.length > 0 || routes.length > 0) && (
                    <button className="btn btn-sm btn-secondary" onClick={forget} disabled={busy}>
                        {busy ? 'Forgetting…' : 'Forget and relearn'}
                    </button>
                )}
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 0 }}>
                Nothing here needs setting up. Handoff watches people walk and works out which cameras
                see the same floor and which ones people pass between. The steps above only add to it.
            </p>

            {unusable.map(([name, problems]) => (
                <div key={name} style={{ ...row, borderLeft: '3px solid var(--danger)', marginBottom: '6px', alignItems: 'flex-start' }}>
                    <div>
                        <strong>{name}: floor calibration is being ignored.</strong>
                        {problems.map((p, i) => <div key={i} style={{ color: 'var(--text-secondary)', marginTop: '2px' }}>{p}</div>)}
                        <div style={{ color: 'var(--text-muted)', marginTop: '2px' }}>
                            Redo it in step 3, or remove it — handoff does not depend on it.
                        </div>
                    </div>
                </div>
            ))}

            <div style={{ fontSize: '0.75rem', fontWeight: 600, margin: '8px 0 6px' }}>Cameras that see the same floor</div>
            <div style={{ fontSize: '0.688rem', color: 'var(--text-muted)', marginBottom: '6px' }}>
                Every pair that has had people walking at the same time is listed. Only pairs whose walks
                keep agreeing are used; the rest are coincidence and stay unconfirmed.
            </div>
            {overlaps.length === 0 ? (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    None found yet. This needs a few people to have walked through a place two cameras both see.
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    {overlaps.map((o) => (
                        <div key={`${o.a}~${o.b}`} style={{ ...row, borderLeft: `3px solid ${o.ready ? 'var(--success)' : 'var(--border-color)'}` }}>
                            <span style={{ flex: 1 }}><strong>{o.a_name}</strong>{'  ⇄  '}<strong>{o.b_name}</strong></span>
                            <span style={{ color: 'var(--text-muted)' }}>
                                {o.ready
                                    ? `in use · ${o.walks} walks, ${Math.round(o.agreement * 100)}% agree`
                                    : `not confirmed · ${o.walks >= o.walks_needed ? `${o.walks} walks` : `${o.walks} of ${o.walks_needed} walks`}, ${Math.round(o.agreement * 100)}% agree (needs ${Math.round(o.agreement_needed * 100)}%)`}
                            </span>
                        </div>
                    ))}
                </div>
            )}

            <div style={{ fontSize: '0.75rem', fontWeight: 600, margin: '12px 0 6px' }}>Routes people walk between cameras</div>
            {routes.length === 0 ? (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    None found yet. A route appears once several people have left one camera and turned up on another the same few seconds later.
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    {routes.map((r, i) => (
                        <div key={i} style={{ ...row, borderLeft: '3px solid var(--success)' }}>
                            <span style={{ flex: 1 }}><strong>{r.from_name}</strong>{'  →  '}<strong>{r.to_name}</strong></span>
                            <span style={{ color: 'var(--text-muted)' }}>
                                {Math.max(0, r.min_sec)}–{Math.max(0, r.max_sec)} s walk · seen on {r.episodes} occasions
                            </span>
                        </div>
                    ))}
                </div>
            )}

            <div style={{ fontSize: '0.688rem', color: 'var(--text-muted)', marginTop: '10px' }}>
                Appearance matching is {live.evidence?.appearance_active ? 'running' : 'not running (no body crops received in the last minute)'}.
                {' '}Drawn area links: {live.evidence?.area_links ?? 0}.
            </div>
            {note && <div style={{ fontSize: '0.75rem', marginTop: '8px', color: 'var(--text-secondary)' }}>{note}</div>}
        </div>
    );
}

export default function HandoffEditor({ zone, cameras }) {
    const [areas, setAreas] = useState({});        // cameraId -> [{id, name, rect}]
    const [links, setLinks] = useState([]);        // [{a:{camera_id, area_id}, b:{...}, max_gap_sec}]
    const [maxGap, setMaxGap] = useState(120);
    const [loading, setLoading] = useState(true);
    const [loadError, setLoadError] = useState(null);
    const [dirty, setDirty] = useState(false);
    const [saving, setSaving] = useState(false);
    const [message, setMessage] = useState(null);

    const [cameraId, setCameraId] = useState(null);
    const [snapshotUrl, setSnapshotUrl] = useState(null);
    const [snapshotError, setSnapshotError] = useState(null);
    const [loadingSnapshot, setLoadingSnapshot] = useState(false);
    const [draft, setDraft] = useState(null);      // {x0, y0, x1, y1} while dragging

    const [linkA, setLinkA] = useState('');
    const [linkB, setLinkB] = useState('');
    const [linkGap, setLinkGap] = useState(String(DEFAULT_GAP));

    const [live, setLive] = useState(null);

    const snapshotRef = useRef(null);
    const frameRef = useRef(null);
    const draftRef = useRef(null);

    // ── Load ─────────────────────────────────────────────────

    const applyDoc = useCallback((doc) => {
        const next = {};
        for (const cam of doc.cameras || []) next[cam.id] = cam.areas || [];
        setAreas(next);
        setLinks(doc.links || []);
        if (doc.max_gap_sec) setMaxGap(doc.max_gap_sec);
        setDirty(false);
    }, []);

    // The parent keys this component by zone, so a zone change remounts it
    // with fresh state; only the fetch belongs here.
    useEffect(() => {
        let cancelled = false;
        api.getZoneHandoff(zone)
            .then((doc) => { if (!cancelled) applyDoc(doc); })
            .catch((err) => { if (!cancelled) setLoadError(err.message); })
            .finally(() => { if (!cancelled) setLoading(false); });
        return () => { cancelled = true; };
    }, [zone, applyDoc]);

    useEffect(() => {
        let timer = null;
        const poll = async () => {
            try { setLive(await api.getZoneHandoffLive(zone)); } catch { /* backend restarting */ }
        };
        poll();
        timer = setInterval(poll, 2000);
        return () => clearInterval(timer);
    }, [zone]);

    useEffect(() => () => {
        if (snapshotRef.current) URL.revokeObjectURL(snapshotRef.current);
    }, []);

    const loadSnapshot = useCallback(async (id) => {
        setLoadingSnapshot(true);
        setSnapshotError(null);
        try {
            const url = await api.getCameraSnapshotUrl(id);
            if (snapshotRef.current) URL.revokeObjectURL(snapshotRef.current);
            snapshotRef.current = url;
            setSnapshotUrl(url);
        } catch (err) {
            setSnapshotUrl(null);
            setSnapshotError(err.message);
        }
        setLoadingSnapshot(false);
    }, []);

    function pickCamera(id) {
        setCameraId(id);
        updateDraft(null);
        loadSnapshot(id);
    }

    // ── Drawing ──────────────────────────────────────────────

    function pointAt(e) {
        const r = frameRef.current.getBoundingClientRect();
        return [
            Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
            Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
        ];
    }

    // The box being dragged lives in a ref as well as state: a quick drag can
    // deliver pointerup before React re-renders, and a handler reading the
    // state from its closure would then see no drag and leave it stuck.
    function updateDraft(next) {
        draftRef.current = next;
        setDraft(next);
    }

    function onPointerDown(e) {
        if (e.button !== 0) return;
        e.currentTarget.setPointerCapture(e.pointerId);
        const [x, y] = pointAt(e);
        updateDraft({ x0: x, y0: y, x1: x, y1: y });
    }

    function onPointerMove(e) {
        if (!draftRef.current) return;
        const [x, y] = pointAt(e);
        updateDraft({ ...draftRef.current, x1: x, y1: y });
    }

    function onPointerUp(e) {
        if (!draftRef.current) return;
        const [x, y] = pointAt(e);
        const draft = { ...draftRef.current, x1: x, y1: y };
        updateDraft(null);
        const rect = [
            Math.min(draft.x0, draft.x1), Math.min(draft.y0, draft.y1),
            Math.max(draft.x0, draft.x1), Math.max(draft.y0, draft.y1),
        ].map((v) => Math.round(v * 10000) / 10000);
        if (rect[2] - rect[0] < MIN_SIZE || rect[3] - rect[1] < MIN_SIZE) return;
        const existing = areas[cameraId] || [];
        if (existing.length >= 8) {
            setMessage({ type: 'error', text: 'At most 8 areas per camera.' });
            return;
        }
        setAreas((prev) => ({
            ...prev,
            [cameraId]: [...existing, { id: newAreaId(), name: `Area ${existing.length + 1}`, rect }],
        }));
        setDirty(true);
        setMessage(null);
    }

    function renameArea(camId, areaId, name) {
        setAreas((prev) => ({
            ...prev,
            [camId]: (prev[camId] || []).map((a) => (a.id === areaId ? { ...a, name } : a)),
        }));
        setDirty(true);
    }

    function deleteArea(camId, areaId) {
        setAreas((prev) => ({ ...prev, [camId]: (prev[camId] || []).filter((a) => a.id !== areaId) }));
        // A link to a deleted area would be rejected on save; drop it with the area.
        setLinks((prev) => prev.filter((l) =>
            !(l.a.camera_id === camId && l.a.area_id === areaId)
            && !(l.b.camera_id === camId && l.b.area_id === areaId)));
        setDirty(true);
    }

    // ── Links ────────────────────────────────────────────────

    const cameraName = (id) => cameras.find((c) => c.id === id)?.name || id.slice(0, 8);
    const areaName = (camId, areaId) => (areas[camId] || []).find((a) => a.id === areaId)?.name || '(deleted)';

    const allAreas = cameras.flatMap((cam) =>
        (areas[cam.id] || []).map((a) => ({ key: refKey(cam.id, a.id), cameraId: cam.id, areaId: a.id, label: `${cam.name} · ${a.name}` })));

    const linkProblem = (() => {
        if (!linkA || !linkB) return 'Pick an area at each end.';
        const [ca] = linkA.split('|');
        const [cb] = linkB.split('|');
        if (ca === cb) return 'Both ends are on the same camera — link areas on two different cameras.';
        const pair = [linkA, linkB].sort().join('~');
        if (links.some((l) => [refKey(l.a.camera_id, l.a.area_id), refKey(l.b.camera_id, l.b.area_id)].sort().join('~') === pair)) {
            return 'These two areas are already linked.';
        }
        const gap = parseFloat(linkGap);
        if (!(gap > 0) || gap > maxGap) return `Max gap must be between 1 and ${maxGap} seconds.`;
        return null;
    })();

    function addLink() {
        if (linkProblem) return;
        const [ca, aa] = linkA.split('|');
        const [cb, ab] = linkB.split('|');
        setLinks((prev) => [...prev, {
            a: { camera_id: ca, area_id: aa },
            b: { camera_id: cb, area_id: ab },
            max_gap_sec: parseFloat(linkGap),
        }]);
        setLinkA('');
        setLinkB('');
        setDirty(true);
    }

    function deleteLink(index) {
        setLinks((prev) => prev.filter((_, i) => i !== index));
        setDirty(true);
    }

    // ── Save ─────────────────────────────────────────────────

    async function save() {
        setSaving(true);
        setMessage(null);
        try {
            const doc = await api.saveZoneHandoff(zone, {
                cameras: cameras.map((c) => ({ id: c.id, areas: areas[c.id] || [] })),
                links,
            });
            applyDoc(doc);
            setMessage({ type: 'success', text: 'Saved. Handoff uses it immediately.' });
        } catch (err) {
            setMessage({ type: 'error', text: err.message });
        }
        setSaving(false);
    }

    // ── Render ───────────────────────────────────────────────

    const cameraAreas = cameraId ? areas[cameraId] || [] : [];
    const colorOf = (camId, areaId) => {
        const i = (areas[camId] || []).findIndex((a) => a.id === areaId);
        return AREA_COLORS[(i < 0 ? 0 : i) % AREA_COLORS.length];
    };

    return (
        <>
            <div className="card" style={{ marginBottom: '16px' }}>
                <div className="card-header"><h3 className="card-title">4 · Handoff areas between cameras</h3></div>
                <div style={{
                    fontSize: '0.813rem', color: 'var(--text-secondary)',
                    background: 'var(--bg-tertiary)', padding: '12px 14px',
                    borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--accent-light)',
                }}>
                    <strong>Optional.</strong> Handoff learns the routes between cameras by itself (step 5 shows what it
                    has found); draw areas only for a doorway it has not picked up, or to make one certain from the first day.
                    For cameras that do <strong>not</strong> see the same floor (the room and the corridor).
                    On each camera, drag a box over the part of the frame people <strong>walk out of</strong> towards
                    the next camera — usually a frame edge — and on the next camera, where they <strong>first appear</strong>.
                    Then link the two boxes. Draw them generously: what counts is where a person&rsquo;s
                    <em> feet</em> are when the camera last (or first) sees them.
                    <div style={{ marginTop: '8px' }}>
                        <strong>Max gap</strong> is the longest walk between the two views. Too short and slow
                        walkers get a new number; too long and the next stranger to appear can inherit it.
                    </div>
                </div>

                {loading ? (
                    <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.813rem' }}>Loading handoff areas…</div>
                ) : loadError ? (
                    <div style={{ padding: '24px', textAlign: 'center', color: 'var(--danger)', fontSize: '0.813rem' }}>
                        Could not load handoff areas: {loadError}
                    </div>
                ) : (
                    <>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', margin: '14px 0' }}>
                            {cameras.map((cam) => (
                                <button
                                    key={cam.id}
                                    className={`btn btn-sm ${cameraId === cam.id ? 'btn-primary' : 'btn-secondary'}`}
                                    onClick={() => pickCamera(cam.id)}
                                    style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                                >
                                    <VideoCameraIcon size={13} />
                                    {cam.name}
                                    {(areas[cam.id] || []).length > 0 && (
                                        <span className="badge online" style={{ fontSize: '0.563rem' }}>
                                            {(areas[cam.id] || []).length}
                                        </span>
                                    )}
                                </button>
                            ))}
                        </div>

                        {cameraId && (
                            <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', alignItems: 'flex-start' }}>
                                <div style={{ flex: '1 1 480px', minWidth: 0 }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                                        <strong style={{ fontSize: '0.813rem' }}>Drag on the frame to add an area</strong>
                                        <div style={{ flex: 1 }} />
                                        <button className="btn btn-sm btn-secondary" onClick={() => loadSnapshot(cameraId)} disabled={loadingSnapshot}>
                                            {loadingSnapshot ? 'Loading…' : 'New frame'}
                                        </button>
                                    </div>
                                    {snapshotError ? (
                                        <div style={{
                                            padding: '30px', textAlign: 'center', background: 'var(--bg-tertiary)',
                                            borderRadius: 'var(--radius-sm)', color: 'var(--danger)', fontSize: '0.813rem',
                                        }}>
                                            {snapshotError}
                                        </div>
                                    ) : snapshotUrl ? (
                                        <div
                                            ref={frameRef}
                                            onPointerDown={onPointerDown}
                                            onPointerMove={onPointerMove}
                                            onPointerUp={onPointerUp}
                                            onPointerCancel={() => updateDraft(null)}
                                            style={{ position: 'relative', lineHeight: 0, cursor: 'crosshair', touchAction: 'none', userSelect: 'none' }}
                                        >
                                            {/* eslint-disable-next-line @next/next/no-img-element */}
                                            <img
                                                src={snapshotUrl}
                                                alt={`${cameraName(cameraId)} frame for handoff areas`}
                                                draggable={false}
                                                style={{ width: '100%', display: 'block', borderRadius: 'var(--radius-sm)' }}
                                            />
                                            {cameraAreas.map((a) => (
                                                <div key={a.id} style={{
                                                    position: 'absolute',
                                                    left: `${a.rect[0] * 100}%`, top: `${a.rect[1] * 100}%`,
                                                    width: `${(a.rect[2] - a.rect[0]) * 100}%`, height: `${(a.rect[3] - a.rect[1]) * 100}%`,
                                                    border: `2px solid ${colorOf(cameraId, a.id)}`,
                                                    background: `${colorOf(cameraId, a.id)}33`,
                                                    pointerEvents: 'none', lineHeight: 'normal',
                                                }}>
                                                    <span style={{
                                                        background: colorOf(cameraId, a.id), color: '#fff',
                                                        fontSize: '0.688rem', fontWeight: 600, padding: '1px 5px',
                                                    }}>{a.name}</span>
                                                </div>
                                            ))}
                                            {draft && (
                                                <div style={{
                                                    position: 'absolute',
                                                    left: `${Math.min(draft.x0, draft.x1) * 100}%`,
                                                    top: `${Math.min(draft.y0, draft.y1) * 100}%`,
                                                    width: `${Math.abs(draft.x1 - draft.x0) * 100}%`,
                                                    height: `${Math.abs(draft.y1 - draft.y0) * 100}%`,
                                                    border: '2px dashed #fff', background: 'rgba(255,255,255,0.15)',
                                                    pointerEvents: 'none',
                                                }} />
                                            )}
                                        </div>
                                    ) : (
                                        <div style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.813rem' }}>
                                            {loadingSnapshot ? 'Fetching a frame…' : 'No frame loaded.'}
                                        </div>
                                    )}
                                </div>

                                <div style={{ flex: '0 1 260px', minWidth: '220px' }}>
                                    <div style={{ fontSize: '0.75rem', fontWeight: 600, marginBottom: '6px' }}>
                                        Areas on {cameraName(cameraId)}
                                    </div>
                                    {cameraAreas.length === 0 ? (
                                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                            None yet. Drag a box over the frame edge people leave through.
                                        </div>
                                    ) : cameraAreas.map((a) => (
                                        <div key={a.id} style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                                            <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: colorOf(cameraId, a.id), flexShrink: 0 }} />
                                            <input
                                                className="input"
                                                value={a.name}
                                                maxLength={64}
                                                onChange={(e) => renameArea(cameraId, a.id, e.target.value)}
                                                style={{ flex: 1, fontSize: '0.75rem' }}
                                                aria-label="Area name"
                                            />
                                            <button className="btn btn-sm btn-secondary" onClick={() => deleteArea(cameraId, a.id)} aria-label={`Delete ${a.name}`}>
                                                ✕
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Links */}
                        <div style={{ marginTop: '18px' }}>
                            <div style={{ fontSize: '0.75rem', fontWeight: 600, marginBottom: '6px' }}>Links</div>
                            {links.length === 0 ? (
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
                                    No links yet. Areas do nothing until they are linked to an area on another camera.
                                </div>
                            ) : (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginBottom: '10px' }}>
                                    {links.map((l, i) => (
                                        <div key={`${refKey(l.a.camera_id, l.a.area_id)}~${refKey(l.b.camera_id, l.b.area_id)}`} style={{
                                            display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.75rem',
                                            padding: '6px 10px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-sm)',
                                        }}>
                                            <span style={{ flex: 1 }}>
                                                <strong>{cameraName(l.a.camera_id)}</strong> · {areaName(l.a.camera_id, l.a.area_id)}
                                                {'  ⇄  '}
                                                <strong>{cameraName(l.b.camera_id)}</strong> · {areaName(l.b.camera_id, l.b.area_id)}
                                                <span style={{ color: 'var(--text-muted)' }}> · within {l.max_gap_sec} s</span>
                                            </span>
                                            <button className="btn btn-sm btn-secondary" onClick={() => deleteLink(i)} aria-label="Delete link">✕</button>
                                        </div>
                                    ))}
                                </div>
                            )}
                            {allAreas.length >= 2 && (
                                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'flex-end' }}>
                                    <label style={{ fontSize: '0.688rem', color: 'var(--text-muted)' }}>
                                        From
                                        <select className="input" value={linkA} onChange={(e) => setLinkA(e.target.value)}
                                            style={{ display: 'block', marginTop: '4px', fontSize: '0.75rem' }}>
                                            <option value="">Pick an area…</option>
                                            {allAreas.map((a) => <option key={a.key} value={a.key}>{a.label}</option>)}
                                        </select>
                                    </label>
                                    <label style={{ fontSize: '0.688rem', color: 'var(--text-muted)' }}>
                                        To
                                        <select className="input" value={linkB} onChange={(e) => setLinkB(e.target.value)}
                                            style={{ display: 'block', marginTop: '4px', fontSize: '0.75rem' }}>
                                            <option value="">Pick an area…</option>
                                            {allAreas.map((a) => <option key={a.key} value={a.key}>{a.label}</option>)}
                                        </select>
                                    </label>
                                    <label style={{ fontSize: '0.688rem', color: 'var(--text-muted)' }}>
                                        Max gap (s)
                                        <input type="number" className="input" min="1" max={maxGap} step="1" value={linkGap}
                                            onChange={(e) => setLinkGap(e.target.value)}
                                            style={{ display: 'block', width: '90px', marginTop: '4px', fontSize: '0.75rem' }} />
                                    </label>
                                    <button className="btn btn-sm btn-secondary" onClick={addLink} disabled={!!linkProblem}>
                                        Add link
                                    </button>
                                    {linkA && linkB && linkProblem && (
                                        <span style={{ fontSize: '0.688rem', color: 'var(--warning)' }}>{linkProblem}</span>
                                    )}
                                </div>
                            )}
                        </div>

                        {message && (
                            <div style={{
                                marginTop: '12px', padding: '10px 12px', borderRadius: 'var(--radius-sm)',
                                fontSize: '0.813rem', background: 'var(--bg-tertiary)',
                                borderLeft: `3px solid ${message.type === 'error' ? 'var(--danger)' : 'var(--success)'}`,
                            }}>
                                {message.text}
                            </div>
                        )}

                        <div style={{ marginTop: '14px', display: 'flex', gap: '8px', alignItems: 'center' }}>
                            <button className="btn btn-primary" onClick={save} disabled={!dirty || saving}>
                                {saving ? 'Saving…' : 'Save areas and links'}
                            </button>
                            {dirty && <span style={{ fontSize: '0.75rem', color: 'var(--warning)' }}>Unsaved changes</span>}
                        </div>
                    </>
                )}
            </div>

            <LearnedTopology zone={zone} live={live} />

            {/* Live numbers */}
            <div className="card" style={{ marginBottom: '16px' }}>
                <div className="card-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <UsersIcon size={16} /> Numbers in {zone}
                    </h3>
                    {live?.stats && (
                        <span style={{ fontSize: '0.688rem', color: 'var(--text-muted)' }}>
                            carried across by — {EVIDENCE.map(([key, label]) => `${label} ${live.stats[key] || 0}`).join(' · ')}
                        </span>
                    )}
                </div>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 0 }}>
                    The number on each live box. A path through two cameras is a handoff that worked.
                </p>
                {live && !live.enabled ? (
                    <div style={{ padding: '16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                        Handoff is switched off (HANDOFF_ENABLED=false).
                    </div>
                ) : live?.people?.length ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        {live.people.slice(0, 30).map((p) => (
                            <div key={p.global_track_id} style={{
                                display: 'flex', alignItems: 'center', gap: '10px',
                                padding: '8px 10px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-tertiary)',
                                borderLeft: `3px solid ${p.path.length > 1 ? 'var(--success)' : 'var(--border-color)'}`,
                            }}>
                                <strong style={{ fontSize: '0.875rem', minWidth: '44px' }}>#{p.global_track_id}</strong>
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{ fontSize: '0.813rem', fontWeight: 600 }}>{p.name || 'Unidentified'}</div>
                                    <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)' }}>
                                        {p.path.join(' → ')}
                                    </div>
                                </div>
                                <span className={`badge ${p.visible_on.length ? 'online' : 'offline'}`} style={{ fontSize: '0.625rem' }}>
                                    {p.visible_on.length ? `on ${p.visible_on.join(', ')}` : `${p.last_seen_sec_ago}s ago`}
                                </span>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div style={{ padding: '16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                        Nobody seen in this zone in the last minute.
                    </div>
                )}
            </div>
        </>
    );
}
