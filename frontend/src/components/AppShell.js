'use client';

/**
 * AppShell — Mobile-aware layout wrapper.
 * Provides hamburger menu, sidebar drawer with overlay, and responsive main content area.
 */

import { useState, useEffect, useCallback } from 'react';
import { usePathname } from 'next/navigation';
import Sidebar from './Sidebar';
import ProtectedRoute from './ProtectedRoute';
import SearchPalette from './SearchPalette';
import useKeyboardShortcuts from '@/lib/useKeyboardShortcuts';
import { XIcon } from './Icons';

/**
 * Hamburger icon (three-line menu icon).
 */
function MenuIcon({ size = 24 }) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="18" x2="21" y2="18" />
        </svg>
    );
}

export default function AppShell({ children, unknownCount = 0 }) {
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [searchOpen, setSearchOpen] = useState(false);
    const pathname = usePathname();

    // Global keyboard shortcuts
    useKeyboardShortcuts();

    // Listen for Ctrl+K / search open event
    useEffect(() => {
        const handleOpenSearch = () => setSearchOpen(true);
        window.addEventListener('sentinel:open-search', handleOpenSearch);
        return () => window.removeEventListener('sentinel:open-search', handleOpenSearch);
    }, []);

    // Close sidebar on route change (mobile)
    useEffect(() => {
        setSidebarOpen(false);
    }, [pathname]);

    // Close sidebar on Escape key
    useEffect(() => {
        const handleKey = (e) => {
            if (e.key === 'Escape') setSidebarOpen(false);
        };
        document.addEventListener('keydown', handleKey);
        return () => document.removeEventListener('keydown', handleKey);
    }, []);

    // Prevent body scrolling when sidebar is open on mobile
    useEffect(() => {
        if (sidebarOpen) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }
        return () => { document.body.style.overflow = ''; };
    }, [sidebarOpen]);

    const closeSidebar = useCallback(() => setSidebarOpen(false), []);

    return (
        <ProtectedRoute>
            <div className="app-layout">
                {/* Mobile top bar */}
                <header className="mobile-header" id="mobile-header">
                    <button
                        className="mobile-menu-btn"
                        onClick={() => setSidebarOpen(true)}
                        aria-label="Open menu"
                        id="mobile-menu-btn"
                    >
                        <MenuIcon size={22} />
                    </button>
                    <span className="mobile-header-title">SentinelAI</span>
                </header>

                {/* Sidebar overlay (mobile only) */}
                {sidebarOpen && (
                    <div
                        className="sidebar-overlay"
                        onClick={closeSidebar}
                        aria-hidden="true"
                    />
                )}

                {/* Sidebar with close button */}
                <div className={`sidebar-wrapper ${sidebarOpen ? 'open' : ''}`}>
                    <button
                        className="sidebar-close-btn"
                        onClick={closeSidebar}
                        aria-label="Close menu"
                    >
                        <XIcon size={20} />
                    </button>
                    <Sidebar unknownCount={unknownCount} />
                </div>

                <main className="main-content">
                    {children}
                </main>

                {/* Search Palette (Ctrl+K) */}
                <SearchPalette isOpen={searchOpen} onClose={() => setSearchOpen(false)} />
            </div>
        </ProtectedRoute>
    );
}
