'use client';

/**
 * ProtectedRoute — Wraps page content and redirects to /login if not authenticated.
 * Shows a loading spinner while auth state is being checked.
 */

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/AuthContext';

export default function ProtectedRoute({ children }) {
    const { isAuthenticated, isLoading } = useAuth();
    const router = useRouter();

    useEffect(() => {
        if (!isLoading && !isAuthenticated) {
            router.push('/login');
        }
    }, [isLoading, isAuthenticated, router]);

    // While checking auth state, show loading spinner
    if (isLoading) {
        return (
            <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '100vh',
                background: 'var(--bg-primary)',
            }}>
                <div style={{ textAlign: 'center' }}>
                    <div style={{
                        width: '48px',
                        height: '48px',
                        border: '3px solid var(--border-color)',
                        borderTopColor: 'var(--accent)',
                        borderRadius: '50%',
                        animation: 'spin 1s linear infinite',
                        margin: '0 auto 16px',
                    }} />
                    <div style={{
                        color: 'var(--text-muted)',
                        fontSize: '0.875rem',
                        fontWeight: 500,
                    }}>
                        Authenticating...
                    </div>
                </div>
            </div>
        );
    }

    // If not authenticated (and not loading), don't render children
    if (!isAuthenticated) {
        return null;
    }

    return children;
}
