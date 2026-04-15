'use client';

/**
 * ActivityHeatmap — Calendar-style heatmap showing daily event counts.
 * Similar to GitHub contribution graph.
 */

import { useState, useEffect, useMemo } from 'react';
import api from '@/lib/api';

const CELL_SIZE = 14;
const CELL_GAP = 3;
const WEEKS_TO_SHOW = 16; // ~4 months
const DAYS = ['Mon', '', 'Wed', '', 'Fri', '', ''];

const COLORS = [
    'var(--heatmap-0, rgba(255,255,255,0.04))',
    'rgba(99, 102, 241, 0.2)',
    'rgba(99, 102, 241, 0.4)',
    'rgba(99, 102, 241, 0.6)',
    'rgba(99, 102, 241, 0.85)',
];

export default function ActivityHeatmap() {
    const [data, setData] = useState({});
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadData();
    }, []);

    async function loadData() {
        try {
            // Fetch events to build the heatmap data
            const days = WEEKS_TO_SHOW * 7;
            const startDate = new Date();
            startDate.setDate(startDate.getDate() - days);
            const result = await api.getEvents({
                start_date: startDate.toISOString().split('T')[0],
                limit: 5000,
            });

            const events = result.data || [];
            const counts = {};
            events.forEach(event => {
                const dateKey = new Date(event.created_at).toISOString().split('T')[0];
                counts[dateKey] = (counts[dateKey] || 0) + 1;
            });
            setData(counts);
        } catch {
            setData({});
        } finally {
            setLoading(false);
        }
    }

    const { cells, months, maxCount } = useMemo(() => {
        const today = new Date();
        const cells = [];
        const months = [];
        let lastMonth = -1;

        const totalDays = WEEKS_TO_SHOW * 7;
        const startDate = new Date(today);
        startDate.setDate(startDate.getDate() - totalDays + 1);

        // Adjust to start on Monday
        const startDay = startDate.getDay();
        const offset = startDay === 0 ? 6 : startDay - 1; // Monday = 0
        startDate.setDate(startDate.getDate() - offset);

        let max = 0;
        for (let i = 0; i < WEEKS_TO_SHOW * 7; i++) {
            const d = new Date(startDate);
            d.setDate(d.getDate() + i);
            const dateKey = d.toISOString().split('T')[0];
            const count = data[dateKey] || 0;
            if (count > max) max = count;

            const week = Math.floor(i / 7);
            const day = i % 7;

            // Track month labels
            if (d.getMonth() !== lastMonth) {
                months.push({
                    label: d.toLocaleString('en-US', { month: 'short' }),
                    week,
                });
                lastMonth = d.getMonth();
            }

            cells.push({ dateKey, count, week, day, date: d });
        }

        return { cells, months, maxCount: max || 1 };
    }, [data]);

    function getColor(count) {
        if (count === 0) return COLORS[0];
        const ratio = count / maxCount;
        if (ratio < 0.25) return COLORS[1];
        if (ratio < 0.5) return COLORS[2];
        if (ratio < 0.75) return COLORS[3];
        return COLORS[4];
    }

    const [tooltip, setTooltip] = useState(null);

    const width = WEEKS_TO_SHOW * (CELL_SIZE + CELL_GAP) + 40;
    const height = 7 * (CELL_SIZE + CELL_GAP) + 28;

    return (
        <div className="card">
            <div className="card-header">
                <h3 className="card-title">Activity Map</h3>
                <div className="card-subtitle">Event frequency over the last {WEEKS_TO_SHOW} weeks</div>
            </div>

            {loading ? (
                <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
                    Loading heatmap...
                </div>
            ) : (
                <div style={{ overflowX: 'auto', padding: '0 0 8px' }}>
                    <svg width={width} height={height} style={{ display: 'block' }}>
                        {/* Day labels */}
                        {DAYS.map((label, i) => (
                            label && (
                                <text
                                    key={i}
                                    x={28}
                                    y={24 + i * (CELL_SIZE + CELL_GAP) + CELL_SIZE / 2}
                                    fill="var(--text-muted)"
                                    fontSize="9"
                                    fontWeight="500"
                                    textAnchor="end"
                                    dominantBaseline="middle"
                                >
                                    {label}
                                </text>
                            )
                        ))}

                        {/* Month labels */}
                        {months.map((m, i) => (
                            <text
                                key={i}
                                x={36 + m.week * (CELL_SIZE + CELL_GAP)}
                                y={12}
                                fill="var(--text-muted)"
                                fontSize="9"
                                fontWeight="500"
                            >
                                {m.label}
                            </text>
                        ))}

                        {/* Cells */}
                        {cells.map((cell, i) => (
                            <rect
                                key={i}
                                x={36 + cell.week * (CELL_SIZE + CELL_GAP)}
                                y={20 + cell.day * (CELL_SIZE + CELL_GAP)}
                                width={CELL_SIZE}
                                height={CELL_SIZE}
                                rx={3}
                                fill={getColor(cell.count)}
                                style={{ cursor: 'pointer', transition: 'fill 0.15s' }}
                                onMouseEnter={(e) => {
                                    setTooltip({
                                        x: e.clientX,
                                        y: e.clientY,
                                        text: `${cell.count} events on ${cell.date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}`,
                                    });
                                }}
                                onMouseLeave={() => setTooltip(null)}
                            />
                        ))}
                    </svg>

                    {/* Legend */}
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        justifyContent: 'flex-end',
                        paddingRight: '8px',
                        marginTop: '4px',
                    }}>
                        <span style={{ fontSize: '0.625rem', color: 'var(--text-muted)' }}>Less</span>
                        {COLORS.map((c, i) => (
                            <div key={i} style={{
                                width: 10, height: 10, borderRadius: 2,
                                background: c,
                            }} />
                        ))}
                        <span style={{ fontSize: '0.625rem', color: 'var(--text-muted)' }}>More</span>
                    </div>
                </div>
            )}

            {/* Tooltip */}
            {tooltip && (
                <div style={{
                    position: 'fixed',
                    left: tooltip.x + 10,
                    top: tooltip.y - 30,
                    background: 'var(--bg-primary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    padding: '4px 10px',
                    fontSize: '0.6875rem',
                    fontWeight: 500,
                    color: 'var(--text-primary)',
                    pointerEvents: 'none',
                    zIndex: 9999,
                    whiteSpace: 'nowrap',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
                }}>
                    {tooltip.text}
                </div>
            )}
        </div>
    );
}
