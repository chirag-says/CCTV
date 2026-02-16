'use client';

/**
 * Sidebar Component — Main navigation for the CCTV dashboard.
 * Renders the nav content only — AppShell handles mobile wrapper / drawer.
 */

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    DashboardIcon,
    MonitorIcon,
    UsersIcon,
    CameraIcon,
    UnknownFaceIcon,
    EventLogIcon,
    AnalyticsIcon,
    ShieldIcon,
} from './Icons';

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
        section: 'Insights',
        items: [
            { href: '/events', icon: EventLogIcon, label: 'Events Log', id: 'nav-events' },
            { href: '/analytics', icon: AnalyticsIcon, label: 'Analytics', id: 'nav-analytics' },
        ],
    },
];

export default function Sidebar({ unknownCount = 0 }) {
    const pathname = usePathname();

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
