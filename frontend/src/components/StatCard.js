'use client';

/**
 * StatCard Component — Displays a metric with icon, value, and change indicator.
 */

export default function StatCard({
    label,
    value,
    icon,
    variant = 'primary',
    change = null,
    changeLabel = '',
    suffix = '',
}) {
    return (
        <div className={`stat-card ${variant}`} role="article" aria-label={label}>
            <div className="stat-card-top">
                <span className="stat-card-label">{label}</span>
                <div className="stat-card-icon">{icon}</div>
            </div>
            <div className="stat-card-value animate-count">
                {value}{suffix && <span style={{ fontSize: '1rem', fontWeight: 500, color: 'var(--text-muted)', marginLeft: '4px' }}>{suffix}</span>}
            </div>
            {change !== null && (
                <div className={`stat-card-change ${change >= 0 ? 'positive' : 'negative'}`}>
                    <span>{change >= 0 ? '↑' : '↓'}</span>
                    <span>{Math.abs(change)}%</span>
                    {changeLabel && <span style={{ color: 'var(--text-muted)' }}>{changeLabel}</span>}
                </div>
            )}
        </div>
    );
}
