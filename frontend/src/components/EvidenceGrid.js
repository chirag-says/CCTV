'use client';

/**
 * EvidenceGrid — the images captured around an incident.
 *
 * Feeds the existing Lightbox for full-size inspection. Images are the only
 * visual evidence this system holds: there is no video recorder, so an
 * incident page shows the annotated alert frame plus the faces and wider
 * frames captured on the same camera in the surrounding window.
 *
 * When an alert predates snapshot capture it says so, rather than rendering
 * a broken image.
 */

import { useState } from 'react';
import Lightbox from './Lightbox';
import { API_BASE } from '@/lib/api';
import { UnknownFaceIcon } from './Icons';

function EvidenceTile({ url, caption, sub, onOpen, tall = false }) {
    return (
        <button
            type="button"
            onClick={() => onOpen(url, caption)}
            style={{
                padding: 0, border: '1px solid var(--border-color)',
                borderRadius: '10px', overflow: 'hidden', cursor: 'zoom-in',
                background: 'var(--bg-tertiary, transparent)', textAlign: 'left',
                font: 'inherit', color: 'inherit', display: 'block', width: '100%',
            }}
        >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
                src={`${API_BASE}${url}`}
                alt={caption}
                style={{
                    width: '100%', height: tall ? '170px' : '112px',
                    objectFit: 'cover', display: 'block', background: 'var(--border-color)',
                }}
            />
            <div style={{ padding: '7px 9px' }}>
                <div style={{ fontSize: '0.6875rem', fontWeight: 600 }}>{caption}</div>
                {sub && (
                    <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)' }}>{sub}</div>
                )}
            </div>
        </button>
    );
}

export default function EvidenceGrid({ evidence, windowSec }) {
    const [preview, setPreview] = useState(null);
    const open = (src, alt) => setPreview({ src: `${API_BASE}${src}`, alt });

    const { alert_snapshot, has_alert_snapshot, faces = [], full_frames = [] } = evidence || {};
    const nothing = !has_alert_snapshot && faces.length === 0 && full_frames.length === 0;

    if (nothing) {
        return (
            <div className="empty-state" style={{ padding: '32px 20px' }}>
                <div className="empty-state-icon"><UnknownFaceIcon size={28} /></div>
                <div className="empty-state-title">No images captured</div>
                <div className="empty-state-text">
                    Nothing was captured on this camera around the time of the alert.
                </div>
            </div>
        );
    }

    return (
        <>
            {!has_alert_snapshot && (
                <div style={{
                    fontSize: '0.75rem', color: 'var(--text-muted)',
                    marginBottom: '12px', padding: '8px 10px',
                    border: '1px solid var(--border-color)', borderRadius: '8px',
                }}>
                    No alert snapshot was captured for this incident — it predates alert
                    image capture. Related images from the same camera are shown instead.
                </div>
            )}

            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))',
                gap: '10px',
            }}>
                {has_alert_snapshot && (
                    <div style={{ gridColumn: 'span 2' }}>
                        <EvidenceTile
                            url={alert_snapshot}
                            caption="Alert frame"
                            sub="Annotated, at the moment of detection"
                            onOpen={open}
                            tall
                        />
                    </div>
                )}

                {faces.map((f) => (
                    <EvidenceTile
                        key={`face-${f.unknown_id}`}
                        url={f.snapshot_url}
                        caption="Face nearby"
                        sub={f.occurrence > 1 ? `seen ${f.occurrence}x` : 'first sighting'}
                        onOpen={open}
                    />
                ))}

                {full_frames.map((f) => (
                    <EvidenceTile
                        key={`frame-${f.unknown_id}`}
                        url={f.url}
                        caption="Full frame"
                        sub="Wider context"
                        onOpen={open}
                    />
                ))}
            </div>

            <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginTop: '10px' }}>
                Related images are those captured on the same camera within{' '}
                {Math.round((windowSec || 120) / 60)} minute
                {Math.round((windowSec || 120) / 60) === 1 ? '' : 's'} of the alert.
                They are not proof of involvement.
            </div>

            {preview && (
                <Lightbox src={preview.src} alt={preview.alt} onClose={() => setPreview(null)} />
            )}
        </>
    );
}
