'use client';

/**
 * useKeyboardShortcuts — Global keyboard shortcuts for power users.
 *
 * Shortcuts:
 *   1-8  → Navigate to pages
 *   Esc  → Close any open modal
 *   R    → Refresh current page data
 *   ?    → Show shortcuts help overlay
 */

import { useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';

const PAGE_SHORTCUTS = {
    '1': '/',                    // Dashboard
    '2': '/monitor',             // Live Monitor
    '3': '/analytics',           // Analytics
    '4': '/persons',             // Persons
    '5': '/events',              // Events
    '6': '/unknown-faces',       // Unknown Faces
    '7': '/security-alerts',     // Security Alerts
    '8': '/video-analysis',      // Video Analysis
};

export const SHORTCUT_LIST = [
    { key: '1–8', description: 'Navigate to pages' },
    { key: 'Esc', description: 'Close any open modal' },
    { key: 'R', description: 'Refresh current page data' },
    { key: '?', description: 'Show keyboard shortcuts' },
];

export default function useKeyboardShortcuts({ onRefresh, onToggleHelp } = {}) {
    const router = useRouter();

    const handleKeyDown = useCallback((e) => {
        // Don't trigger shortcuts when typing in inputs
        const tag = e.target.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.target.isContentEditable) {
            return;
        }

        // Don't trigger on Ctrl/Cmd/Alt combos (except Ctrl+K for search)
        if (e.metaKey || e.altKey) return;
        if (e.ctrlKey && e.key !== 'k') return;

        const key = e.key;

        // Number key navigation (1-8)
        if (PAGE_SHORTCUTS[key] && !e.ctrlKey) {
            e.preventDefault();
            router.push(PAGE_SHORTCUTS[key]);
            return;
        }

        // Escape — close modals (handled by modal components, but we dispatch a custom event)
        if (key === 'Escape') {
            window.dispatchEvent(new CustomEvent('sentinel:close-modal'));
            return;
        }

        // R — Refresh page data
        if (key === 'r' || key === 'R') {
            e.preventDefault();
            if (onRefresh) {
                onRefresh();
            } else {
                // Dispatch a custom event that pages can listen to
                window.dispatchEvent(new CustomEvent('sentinel:refresh'));
            }
            return;
        }

        // ? — Toggle shortcuts help
        if (key === '?') {
            e.preventDefault();
            if (onToggleHelp) {
                onToggleHelp();
            }
            return;
        }

        // Ctrl+K — Focus search (future Task 3.4)
        if (e.ctrlKey && key === 'k') {
            e.preventDefault();
            window.dispatchEvent(new CustomEvent('sentinel:open-search'));
            return;
        }
    }, [router, onRefresh, onToggleHelp]);

    useEffect(() => {
        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [handleKeyDown]);
}
