'use client';

/**
 * Sidebar Component — Main navigation for the CCTV dashboard.
 * Renders the nav content only — AppShell handles mobile wrapper / drawer.
 */

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useTheme } from '@/lib/ThemeContext';
import {
    DashboardIcon,
    MonitorIcon,
    UsersIcon,
    CameraIcon,
    UnknownFaceIcon,
    EventLogIcon,
    AnalyticsIcon,
    ShieldIcon,
    SecurityIcon,
    TrafficIcon,
    VideoUploadIcon,
} from './Icons';

function SunIcon({ size = 18 }) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="5" />
            <line x1="12" y1="1" x2="12" y2="3" />
            <line x1="12" y1="21" x2="12" y2="23" />
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
            <line x1="1" y1="12" x2="3" y2="12" />
            <line x1="21" y1="12" x2="23" y2="12" />
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
        </svg>
    );
}

function MoonIcon({ size = 18 }) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
    );
}

const navItems = [
    {
        section: 'Overview',
        items: [
            { href: '/', icon: DashboardIcon, label: 'Dashboard', id: 'nav-dashboard' },
            { href: '/monitor', icon: MonitorIcon, label: 'Live Monitor', id: 'nav-monitor' },
        ],
    },
    {
        section: 'Management',
        items: [
            { href: '/persons', icon: UsersIcon, label: 'Persons', id: 'nav-persons' },
            { href: '/cameras', icon: CameraIcon, label: 'Cameras', id: 'nav-cameras' },
            { href: '/unknown-faces', icon: UnknownFaceIcon, label: 'Unknown Faces', id: 'nav-unknown', badge: true },
        ],
    },
    {
        section: 'Security',
        items: [
            { href: '/security-alerts', icon: SecurityIcon, label: 'Security Alerts', id: 'nav-security', alertBadge: true },
            { href: '/traffic', icon: TrafficIcon, label: 'Traffic Monitor', id: 'nav-traffic' },
        ],
    },
    {
        section: 'Tools',
        items: [
            { href: '/video-analysis', icon: VideoUploadIcon, label: 'Video Analysis', id: 'nav-video-analysis' },
        ],
    },
    {
        section: 'Insights',
        items: [
            { href: '/events', icon: EventLogIcon, label: 'Events Log', id: 'nav-events' },
            { href: '/analytics', icon: AnalyticsIcon, label: 'Analytics', id: 'nav-analytics' },
        ],
    },
];

export default function Sidebar({ unknownCount = 0 }) {
    const pathname = usePathname();
    const { theme, toggleTheme, isTransitioning } = useTheme();

    return (
        <nav className="sidebar" role="navigation" aria-label="Main Navigation">
            {/* Brand */}
            <div className="sidebar-brand">
                <div className="sidebar-brand-icon">
                    <ShieldIcon size={20} />
                </div>
                <div className="sidebar-brand-text">
                    <h2>SentinelAI</h2>
                    <span>CCTV System</span>
                </div>
            </div>

            {/* Nav Links */}
            <div className="sidebar-nav">
                {navItems.map((section) => (
                    <div key={section.section}>
                        <div className="sidebar-section-title">{section.section}</div>
                        {section.items.map((item) => {
                            const IconComponent = item.icon;
                            return (
                                <Link
                                    key={item.href}
                                    href={item.href}
                                    id={item.id}
                                    className={`nav-item ${pathname === item.href ? 'active' : ''}`}
                                >
                                    <span className="nav-item-icon">
                                        <IconComponent size={18} />
                                    </span>
                                    <span>{item.label}</span>
                                    {item.badge && unknownCount > 0 && (
                                        <span className="nav-item-badge">{unknownCount}</span>
                                    )}
                                </Link>
                            );
                        })}
                    </div>
                ))}
            </div>

            {/* Footer */}
            <div style={{
                padding: '16px 12px',
                borderTop: '1px solid var(--border-color)',
                marginTop: 'auto'
            }}>
                {/* Theme Toggle */}
                <button
                    className="theme-toggle-btn"
                    onClick={toggleTheme}
                    aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                    title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                    id="theme-toggle"
                >
                    <span className={`theme-toggle-icon${isTransitioning ? ' spin-in' : ''}`}>
                        {theme === 'dark' ? <SunIcon size={16} /> : <MoonIcon size={16} />}
                    </span>
                    <span>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
                </button>

                {/* Admin Info */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    fontSize: '0.813rem',
                    color: 'var(--text-muted)'
                }}>
                    <div className="avatar sm">A</div>
                    <div>
                        <div style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>Admin</div>
                        <div style={{ fontSize: '0.688rem' }}>System Administrator</div>
                    </div>
                </div>
            </div>
        </nav>
    );
}
