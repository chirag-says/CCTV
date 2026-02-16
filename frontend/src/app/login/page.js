'use client';

/**
 * Login Page — Admin authentication.
 */

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import api from '@/lib/api';
import { ShieldIcon } from '@/components/Icons';

export default function LoginPage() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const router = useRouter();

    async function handleSubmit(e) {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            await api.login(email, password);
            router.push('/');
        } catch (err) {
            setError(err.message || 'Login failed. Please check your credentials.');
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="login-page">
            <div className="login-card">
                <div style={{ marginBottom: '24px' }}>
                    <div style={{
                        width: '64px',
                        height: '64px',
                        borderRadius: '12px',
                        background: 'var(--primary)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '1.75rem',
                        margin: '0 auto 16px',
                    }}>
                        <ShieldIcon size={32} style={{ color: '#fff' }} />
                    </div>
                    <h1>SentinelAI</h1>
                    <p className="subtitle">AI-Powered Surveillance System</p>
                </div>

                {error && <div className="login-error">{error}</div>}

                <form onSubmit={handleSubmit}>
                    <div className="input-group">
                        <label className="input-label" htmlFor="login-email">Email Address</label>
                        <input
                            id="login-email"
                            type="email"
                            className="input"
                            placeholder="admin@cctv.local"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            autoFocus
                        />
                    </div>

                    <div className="input-group">
                        <label className="input-label" htmlFor="login-password">Password</label>
                        <input
                            id="login-password"
                            type="password"
                            className="input"
                            placeholder="Enter your password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            minLength={6}
                        />
                    </div>

                    <button
                        type="submit"
                        className="btn btn-primary"
                        disabled={loading}
                        id="login-submit"
                    >
                        {loading ? (
                            <>
                                <span className="loading-spinner" style={{ width: '16px', height: '16px' }}></span>
                                Signing in...
                            </>
                        ) : (
                            'Sign In'
                        )}
                    </button>
                </form>

                <div style={{
                    marginTop: '24px',
                    fontSize: '0.75rem',
                    color: 'var(--text-muted)',
                    padding: '12px',
                    background: 'var(--bg-tertiary)',
                    borderRadius: 'var(--radius-sm)',
                }}>
                    <strong>Demo Credentials:</strong><br />
                    Email: admin@cctv.local<br />
                    Password: admin123
                </div>
            </div>
        </div>
    );
}
