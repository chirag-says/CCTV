'use client';

/**
 * ActivityFeed Component — Real-time event feed display.
 * Supports standard events AND security alerts.
 */

import {
    EntryIcon,
    ExitIcon,
    DetectionIcon,
    UnknownFaceIcon,
    EventLogIcon,
    FireIcon,
    CrowdIcon,
    LoiterIcon,
    HazardIcon,
    SecurityIcon,
    VehicleIcon,
    PlateIcon,
    ProximityIcon,
} from './Icons';

export default function ActivityFeed({ events = [] }) {
    const getEventIcon = (event) => {
        const type = event.event_type;
        if (type === 'vehicle_entry') return <PlateIcon size={16} />;
        if (type === 'security_alert') {
            const subtype = event.subtype || event.metadata?.subtype || '';
            switch (subtype) {
                case 'gathering': return <CrowdIcon size={16} />;
                case 'loitering': return <LoiterIcon size={16} />;
                case 'vehicle_proximity': return <ProximityIcon size={16} />;
                case 'hazard':
                    const threat = event.metadata?.threat_class || '';
                    if (threat.includes('fire') || threat.includes('flame')) return <FireIcon size={16} />;
                    return <HazardIcon size={16} />;
                default: return <SecurityIcon size={16} />;
            }
        }
        switch (type) {
            case 'entry': return <EntryIcon size={16} />;
            case 'exit': return <ExitIcon size={16} />;
            case 'unknown': return <UnknownFaceIcon size={16} />;
            case 'detection': return <DetectionIcon size={16} />;
            default: return <DetectionIcon size={16} />;
        }
    };

    const getEventLabel = (event) => {
        if (event.event_type === 'vehicle_entry') {
            const plate = event.metadata?.plate || '???';
            const vtype = event.metadata?.vehicle_type || 'vehicle';
            return `${vtype.charAt(0).toUpperCase() + vtype.slice(1)} — plate ${plate}`;
        }
        if (event.event_type === 'security_alert') {
            const subtype = event.subtype || event.metadata?.subtype || 'alert';
            switch (subtype) {
                case 'gathering':
                    const count = event.person_count || event.metadata?.person_count || '?';
                    return `Crowd gathering detected (${count} people)`;
                case 'loitering':
                    const name = event.person_name || event.metadata?.person_name || 'Unknown';
                    return `${name} — loitering detected`;
                case 'vehicle_proximity':
                    const veh = event.metadata?.vehicle_type || 'vehicle';
                    return `⚠ Person too close to ${veh}`;
                case 'hazard':
                    const threat = event.threat_class || event.metadata?.threat_class || 'threat';
                    return `Hazard: ${threat} detected`;
                default:
                    return `Security alert: ${subtype}`;
            }
        }
        switch (event.event_type) {
            case 'entry':
                return `${event.person_name || 'Someone'} entered`;
            case 'exit':
                return `${event.person_name || 'Someone'} exited`;
            case 'unknown':
                return 'Unknown person detected';
            case 'detection':
                return `${event.person_name || 'Someone'} detected`;
            default:
                return 'Event';
        }
    };

    const getEventBadgeClass = (event) => {
        if (event.event_type === 'vehicle_entry') return 'vehicle-entry';
        if (event.event_type === 'security_alert') {
            const subtype = event.subtype || event.metadata?.subtype || '';
            if (subtype === 'hazard' || subtype === 'vehicle_proximity') return 'security-alert danger';
            return 'security-alert';
        }
        return event.event_type;
    };

    const formatTime = (timestamp) => {
        if (!timestamp) return '--:--';
        try {
            const date = new Date(timestamp);
            return date.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: false,
            });
        } catch {
            return '--:--';
        }
    };

    if (events.length === 0) {
        return (
            <div className="empty-state">
                <div className="empty-state-icon"><EventLogIcon size={32} /></div>
                <div className="empty-state-title">No recent events</div>
                <div className="empty-state-text">
                    Events will appear here when the system detects activity.
                </div>
            </div>
        );
    }

    return (
        <div className="activity-feed" role="feed" aria-label="Activity Feed">
            {events.map((event, index) => (
                <div
                    key={event.id || index}
                    className={`activity-item animate-in ${event.event_type === 'security_alert' ? 'security-alert-item' : ''}`}
                    style={{ animationDelay: `${index * 50}ms` }}
                >
                    <div className={`activity-icon ${getEventBadgeClass(event)}`}>
                        {getEventIcon(event)}
                    </div>
                    <div className="activity-content">
                        <div className="activity-title">{getEventLabel(event)}</div>
                        <div className="activity-subtitle">
                            Camera: {event.camera_name || event.camera_id || 'Unknown'}
                            {event.confidence > 0 && ` • ${(event.confidence * 100).toFixed(0)}% match`}
                            {event.event_type === 'security_alert' && event.metadata?.duration_sec &&
                                ` • ${Math.round(event.metadata.duration_sec)}s`}
                        </div>
                    </div>
                    <span className="activity-time">
                        {formatTime(event.created_at || event.timestamp)}
                    </span>
                </div>
            ))}
        </div>
    );
}
