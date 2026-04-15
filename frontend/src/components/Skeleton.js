'use client';

/**
 * Skeleton loading components — pulsing placeholders for content that hasn't loaded yet.
 */

/**
 * Base skeleton with shimmer animation.
 */
function SkeletonBase({ width, height, borderRadius = '8px', style = {} }) {
    return (
        <div
            className="skeleton-pulse"
            style={{
                width: width || '100%',
                height: height || '16px',
                borderRadius,
                background: 'var(--skeleton-bg, rgba(255,255,255,0.06))',
                ...style,
            }}
        />
    );
}

/**
 * Skeleton card — matches a typical Card component.
 */
export function SkeletonCard({ height = '180px' }) {
    return (
        <div className="card skeleton-card" style={{ padding: '20px' }}>
            <SkeletonBase width="40%" height="14px" style={{ marginBottom: '16px' }} />
            <SkeletonBase width="100%" height={height} borderRadius="10px" />
        </div>
    );
}

/**
 * Skeleton stat card — matches dashboard stat card dimensions.
 */
export function SkeletonStatCard() {
    return (
        <div className="card" style={{ padding: '20px' }}>
            <SkeletonBase width="60%" height="12px" style={{ marginBottom: '14px' }} />
            <SkeletonBase width="40%" height="32px" style={{ marginBottom: '10px' }} />
            <SkeletonBase width="80%" height="10px" />
        </div>
    );
}

/**
 * Skeleton table — rows of pulsing lines.
 */
export function SkeletonTable({ rows = 5, columns = 4 }) {
    return (
        <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
            {/* Header */}
            <div style={{
                display: 'flex',
                gap: '16px',
                padding: '14px 20px',
                borderBottom: '1px solid var(--border-color)',
            }}>
                {Array.from({ length: columns }).map((_, i) => (
                    <SkeletonBase key={`h-${i}`} width={`${100 / columns}%`} height="12px" />
                ))}
            </div>

            {/* Rows */}
            {Array.from({ length: rows }).map((_, rowIdx) => (
                <div
                    key={`r-${rowIdx}`}
                    style={{
                        display: 'flex',
                        gap: '16px',
                        padding: '14px 20px',
                        borderBottom: '1px solid var(--border-color)',
                        opacity: 1 - rowIdx * 0.1,
                    }}
                >
                    {Array.from({ length: columns }).map((_, colIdx) => (
                        <SkeletonBase
                            key={`c-${colIdx}`}
                            width={`${100 / columns}%`}
                            height="14px"
                        />
                    ))}
                </div>
            ))}
        </div>
    );
}

/**
 * Skeleton text — single line placeholder.
 */
export function SkeletonText({ width = '100%', height = '14px', style = {} }) {
    return <SkeletonBase width={width} height={height} style={style} />;
}

/**
 * Skeleton stats grid — matches the dashboard stat cards layout.
 */
export function SkeletonStatsGrid({ count = 4 }) {
    return (
        <div className="security-stats" style={{ marginBottom: '20px' }}>
            {Array.from({ length: count }).map((_, i) => (
                <SkeletonStatCard key={i} />
            ))}
        </div>
    );
}

/**
 * Full page skeleton — for pages that are loading.
 */
export function SkeletonPage() {
    return (
        <div style={{ padding: '0' }}>
            {/* Page header skeleton */}
            <div style={{ marginBottom: '24px' }}>
                <SkeletonBase width="200px" height="28px" style={{ marginBottom: '8px' }} />
                <SkeletonBase width="320px" height="14px" />
            </div>

            {/* Stats */}
            <SkeletonStatsGrid count={4} />

            {/* Content */}
            <SkeletonCard height="300px" />
        </div>
    );
}

export default SkeletonBase;
