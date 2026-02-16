'use client';

/**
 * HourlyChart Component — Simple bar chart for hourly distribution.
 */

export default function HourlyChart({ data = {}, maxValue = null, color = 'var(--primary)' }) {
    const hours = Array.from({ length: 24 }, (_, i) => String(i));
    const values = hours.map((h) => data[h] || 0);
    const max = maxValue || Math.max(...values, 1);

    return (
        <div className="chart-container" role="img" aria-label="Hourly distribution chart">
            {hours.map((hour, index) => {
                const value = values[index];
                const height = max > 0 ? (value / max) * 100 : 0;
                const isWorkHours = index >= 8 && index <= 18;

                return (
                    <div
                        key={hour}
                        className="chart-bar"
                        style={{
                            height: `${Math.max(height, 2)}%`,
                            background: value > 0
                                ? isWorkHours
                                    ? `linear-gradient(to top, ${color}, ${color}aa)`
                                    : `linear-gradient(to top, var(--text-muted), var(--text-muted)66)`
                                : 'var(--bg-tertiary)',
                            opacity: value > 0 ? 1 : 0.3,
                        }}
                        title={`${hour}:00 — ${value} events`}
                    >
                        <span className="chart-bar-value">{value}</span>
                        {index % 3 === 0 && (
                            <span className="chart-bar-label">{hour.padStart(2, '0')}</span>
                        )}
                    </div>
                );
            })}
        </div>
    );
}
