'use client';

/**
 * OccupancyRing Component — Circular progress showing current occupancy.
 */

export default function OccupancyRing({ current = 0, max = 100, label = 'People' }) {
    const radius = 68;
    const circumference = 2 * Math.PI * radius;
    const percentage = max > 0 ? Math.min(current / max, 1) : 0;
    const offset = circumference - (percentage * circumference);

    const getColor = () => {
        if (percentage < 0.5) return 'var(--success)';
        if (percentage < 0.8) return 'var(--warning)';
        return 'var(--danger)';
    };

    return (
        <div className="occupancy-ring">
            <svg width="160" height="160" viewBox="0 0 160 160">
                <circle
                    className="occupancy-ring-bg"
                    cx="80"
                    cy="80"
                    r={radius}
                    fill="none"
                    strokeWidth="8"
                />
                <circle
                    className="occupancy-ring-fill"
                    cx="80"
                    cy="80"
                    r={radius}
                    fill="none"
                    strokeWidth="8"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    style={{ stroke: getColor() }}
                />
            </svg>
            <div className="occupancy-ring-text">
                <div className="occupancy-ring-value animate-count">{current}</div>
                <div className="occupancy-ring-label">{label}</div>
            </div>
        </div>
    );
}
