'use client';

/**
 * FacultyPresenceCard — one teacher on the "Currently on Campus" board.
 *
 * Clicking opens their person page. Location is phrased from the presence
 * state: "Currently at" only while a camera is actually watching them,
 * "Last seen at" otherwise (SPEC §1.3 — the system observes cameras, not
 * the campus boundary).
 */

import { useRouter } from 'next/navigation';
import { API_BASE } from '@/lib/api';
import { PresenceStatus, timeLabel, durationLabel } from './PresenceState';
import { CameraIcon, ArrowRightIcon } from './Icons';

export default function FacultyPresenceCard({ person, highlight = false }) {
    const router = useRouter();
    const {
        person_id, name, designation, department, avatar_url,
        state, camera_name, camera_location, zone,
        first_seen, last_seen, minutes_since_seen, minutes_observed,
    } = person;

    const place = zone || camera_location || camera_name;
    const isHere = state === 'LIVE';
    const isGone = state === 'DEPARTED';

    return (
        <div
            className="card"
            onClick={() => router.push(`/persons/${person_id}`)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    router.push(`/persons/${person_id}`);
                }
            }}
            id={`presence-card-${person_id}`}
            style={{
                padding: '16px', cursor: 'pointer',
                opacity: isGone ? 0.6 : 1,
                borderColor: highlight ? 'var(--primary)' : undefined,
                transition: 'opacity 0.2s ease, border-color 0.2s ease',
            }}
        >
            {/* Identity */}
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                {avatar_url ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                        src={`${API_BASE}${avatar_url}`}
                        alt=""
                        className="avatar"
                        style={{ objectFit: 'cover', flexShrink: 0 }}
                    />
                ) : (
                    <div className="avatar" style={{ flexShrink: 0 }}>{name.charAt(0)}</div>
                )}

                <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                        display: 'flex', alignItems: 'center',
                        justifyContent: 'space-between', gap: '8px',
                    }}>
                        <span style={{
                            fontWeight: 700, fontSize: '0.9375rem',
                            color: 'var(--text-primary)',
                            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                        }}>
                            {name}
                        </span>
                        <PresenceStatus
                            state={state}
                            minutesSince={minutes_since_seen}
                            departedAt={last_seen}
                        />
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '1px' }}>
                        {department}{designation ? ` · ${designation}` : ''}
                    </div>
                </div>
            </div>

            {/* Location — asserted as current only while genuinely live */}
            {place && (
                <div style={{
                    display: 'flex', alignItems: 'center', gap: '6px',
                    marginTop: '12px', fontSize: '0.8125rem',
                    color: 'var(--text-secondary)',
                }}>
                    <span style={{ color: 'var(--text-muted)', display: 'flex' }}>
                        <CameraIcon size={13} />
                    </span>
                    <span style={{ color: 'var(--text-muted)' }}>
                        {isHere ? 'Currently at' : 'Last seen at'}
                    </span>
                    <span style={{ fontWeight: 600 }}>{place}</span>
                </div>
            )}

            {/* Times — "first detected", never "arrived" */}
            <div style={{
                display: 'flex', gap: '20px', marginTop: '12px',
                paddingTop: '12px', borderTop: '1px solid var(--border-color)',
            }}>
                <div>
                    <div style={{
                        fontSize: '0.625rem', color: 'var(--text-muted)',
                        textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 700,
                    }}>
                        First detected
                    </div>
                    <div style={{
                        fontFamily: "'JetBrains Mono', monospace",
                        fontSize: '0.875rem', fontWeight: 600, marginTop: '2px',
                    }}>
                        {timeLabel(first_seen)}
                    </div>
                </div>
                <div>
                    <div style={{
                        fontSize: '0.625rem', color: 'var(--text-muted)',
                        textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 700,
                    }}>
                        Time observed
                    </div>
                    <div style={{
                        fontFamily: "'JetBrains Mono', monospace",
                        fontSize: '0.875rem', fontWeight: 600, marginTop: '2px',
                    }}>
                        {durationLabel(minutes_observed)}
                    </div>
                </div>
                <div style={{
                    marginLeft: 'auto', alignSelf: 'flex-end',
                    fontSize: '0.6875rem', color: 'var(--primary)',
                    display: 'flex', alignItems: 'center', gap: '4px', fontWeight: 600,
                }}>
                    Journey <ArrowRightIcon size={11} />
                </div>
            </div>
        </div>
    );
}
