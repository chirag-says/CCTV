'use client';

/**
 * ActivityFeed Component — Real-time event feed display.
 * Uses proper SVG icons instead of emoji/text icons.
 */

import {
    EntryIcon,
    ExitIcon,
    DetectionIcon,
    UnknownFaceIcon,
    EventLogIcon,
} from './Icons';

export default function ActivityFeed({ events = [] }) {
    const getEventIcon = (type) => {
        switch (type) {
            case 'entry': return <EntryIcon size={16} />;
            case 'exit': return <ExitIcon size={16} />;
            case 'unknown': return <UnknownFaceIcon size={16} />;
            case 'detection': return <DetectionIcon size={16} />;
            default: return <DetectionIcon size={16} />;
        }
    };

    const getEventLabel = (event) => {
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
                    className="activity-item animate-in"
                    style={{ animationDelay: `${index * 50}ms` }}
                >
                    <div className={`activity-icon ${event.event_type}`}>
                        {getEventIcon(event.event_type)}
                    </div>
                    <div className="activity-content">
                        <div className="activity-title">{getEventLabel(event)}</div>
                        <div className="activity-subtitle">
                            Camera: {event.camera_name || event.camera_id || 'Unknown'}
                            {event.confidence > 0 && ` • ${(event.confidence * 100).toFixed(0)}% match`}
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
