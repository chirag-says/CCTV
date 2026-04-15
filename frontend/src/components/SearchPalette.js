'use client';

/**
 * SearchPalette — Command palette style search (Ctrl+K).
 * Searches across persons, events, plates, and unknown faces.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import api from '@/lib/api';

const TYPE_CONFIG = {
    person: { icon: '👤', color: '#6366f1', label: 'Person', route: '/persons' },
    event: { icon: '📋', color: '#10b981', label: 'Event', route: '/events' },
    unknown_face: { icon: '❓', color: '#f59e0b', label: 'Unknown', route: '/unknown-faces' },
    vehicle: { icon: '🚗', color: '#3b82f6', label: 'Vehicle', route: '/traffic' },
};

export default function SearchPalette({ isOpen, onClose }) {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [selectedIdx, setSelectedIdx] = useState(0);
    const inputRef = useRef(null);
    const debounceRef = useRef(null);
    const router = useRouter();

    // Focus input when opened
    useEffect(() => {
        if (isOpen) {
            setQuery('');
            setResults([]);
            setSelectedIdx(0);
            setTimeout(() => inputRef.current?.focus(), 50);
        }
    }, [isOpen]);

    // Debounced search
    useEffect(() => {
        if (!query.trim()) {
            setResults([]);
            return;
        }

        if (debounceRef.current) clearTimeout(debounceRef.current);
        debounceRef.current = setTimeout(async () => {
            setLoading(true);
            try {
                const data = await api.search(query.trim());
                setResults(data.results || []);
                setSelectedIdx(0);
            } catch {
                setResults([]);
            } finally {
                setLoading(false);
            }
        }, 300);

        return () => {
            if (debounceRef.current) clearTimeout(debounceRef.current);
        };
    }, [query]);

    // Keyboard navigation
    const handleKeyDown = useCallback((e) => {
        if (e.key === 'Escape') {
            onClose();
            return;
        }
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setSelectedIdx(i => Math.min(i + 1, results.length - 1));
        }
        if (e.key === 'ArrowUp') {
            e.preventDefault();
            setSelectedIdx(i => Math.max(i - 1, 0));
        }
        if (e.key === 'Enter' && results.length > 0) {
            e.preventDefault();
            const result = results[selectedIdx];
            if (result) {
                const config = TYPE_CONFIG[result.type] || TYPE_CONFIG.event;
                router.push(config.route);
                onClose();
            }
        }
    }, [results, selectedIdx, onClose, router]);

    if (!isOpen) return null;

    return (
        <div
            style={{
                position: 'fixed',
                inset: 0,
                zIndex: 9998,
                display: 'flex',
                alignItems: 'flex-start',
                justifyContent: 'center',
                paddingTop: '15vh',
                background: 'rgba(0,0,0,0.6)',
                backdropFilter: 'blur(6px)',
                animation: 'fadeIn 0.15s ease',
            }}
            onClick={onClose}
        >
            <div
                style={{
                    width: '100%',
                    maxWidth: '560px',
                    background: 'var(--bg-secondary, #1a1b2e)',
                    border: '1px solid var(--border-color, rgba(255,255,255,0.1))',
                    borderRadius: '16px',
                    boxShadow: '0 24px 80px rgba(0,0,0,0.4)',
                    overflow: 'hidden',
                }}
                onClick={(e) => e.stopPropagation()}
            >
                {/* Search Input */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    padding: '16px 20px',
                    borderBottom: '1px solid var(--border-color)',
                }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                        stroke="var(--text-muted)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" />
                    </svg>
                    <input
                        ref={inputRef}
                        type="text"
                        placeholder="Search persons, events, plates..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={handleKeyDown}
                        style={{
                            flex: 1,
                            background: 'transparent',
                            border: 'none',
                            outline: 'none',
                            color: 'var(--text-primary)',
                            fontSize: '0.9375rem',
                            fontFamily: 'inherit',
                        }}
                    />
                    <kbd style={{
                        padding: '2px 8px',
                        borderRadius: '6px',
                        border: '1px solid var(--border-color)',
                        fontSize: '0.6875rem',
                        color: 'var(--text-muted)',
                        fontFamily: 'inherit',
                    }}>ESC</kbd>
                </div>

                {/* Results */}
                <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                    {loading ? (
                        <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
                            Searching...
                        </div>
                    ) : query && results.length === 0 ? (
                        <div style={{ padding: '32px 24px', textAlign: 'center' }}>
                            <div style={{ fontSize: '1.5rem', marginBottom: '8px', opacity: 0.3 }}>🔍</div>
                            <div style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
                                No results for "{query}"
                            </div>
                        </div>
                    ) : results.length > 0 ? (
                        <div style={{ padding: '8px' }}>
                            {results.map((result, idx) => {
                                const config = TYPE_CONFIG[result.type] || TYPE_CONFIG.event;
                                const isSelected = idx === selectedIdx;
                                return (
                                    <button
                                        key={`${result.type}-${result.id}`}
                                        onClick={() => {
                                            router.push(config.route);
                                            onClose();
                                        }}
                                        onMouseEnter={() => setSelectedIdx(idx)}
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '12px',
                                            width: '100%',
                                            padding: '10px 14px',
                                            border: 'none',
                                            borderRadius: '10px',
                                            background: isSelected ? 'var(--hover-surface, rgba(255,255,255,0.06))' : 'transparent',
                                            cursor: 'pointer',
                                            textAlign: 'left',
                                            transition: 'background 0.1s',
                                            fontFamily: 'inherit',
                                        }}
                                    >
                                        <span style={{
                                            width: '32px',
                                            height: '32px',
                                            borderRadius: '8px',
                                            background: `${config.color}15`,
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            fontSize: '0.875rem',
                                            flexShrink: 0,
                                        }}>
                                            {config.icon}
                                        </span>
                                        <div style={{ flex: 1, minWidth: 0 }}>
                                            <div style={{
                                                color: 'var(--text-primary)',
                                                fontSize: '0.8125rem',
                                                fontWeight: 600,
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                whiteSpace: 'nowrap',
                                            }}>
                                                {result.title}
                                            </div>
                                            <div style={{
                                                color: 'var(--text-muted)',
                                                fontSize: '0.6875rem',
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                whiteSpace: 'nowrap',
                                            }}>
                                                {result.subtitle}
                                            </div>
                                        </div>
                                        <span style={{
                                            padding: '2px 8px',
                                            borderRadius: '6px',
                                            background: `${config.color}15`,
                                            color: config.color,
                                            fontSize: '0.625rem',
                                            fontWeight: 600,
                                            textTransform: 'uppercase',
                                            letterSpacing: '0.05em',
                                            flexShrink: 0,
                                        }}>
                                            {config.label}
                                        </span>
                                    </button>
                                );
                            })}
                        </div>
                    ) : !query ? (
                        <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
                            Start typing to search across all data...
                        </div>
                    ) : null}
                </div>

                {/* Footer */}
                <div style={{
                    padding: '10px 16px',
                    borderTop: '1px solid var(--border-color)',
                    display: 'flex',
                    gap: '16px',
                    fontSize: '0.6875rem',
                    color: 'var(--text-muted)',
                }}>
                    <span>↑↓ Navigate</span>
                    <span>↵ Open</span>
                    <span>Esc Close</span>
                </div>
            </div>
        </div>
    );
}
