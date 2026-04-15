'use client';

/**
 * ErrorBoundary — Catches render errors in child components
 * and shows a friendly fallback UI instead of a white screen.
 */

import React from 'react';

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null, errorInfo: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        this.setState({ errorInfo });
        console.error('ErrorBoundary caught an error:', error, errorInfo);
    }

    handleReload = () => {
        this.setState({ hasError: false, error: null, errorInfo: null });
        window.location.reload();
    };

    handleReset = () => {
        this.setState({ hasError: false, error: null, errorInfo: null });
    };

    render() {
        if (this.state.hasError) {
            const isDev = process.env.NODE_ENV === 'development';

            return (
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: '100vh',
                    background: 'var(--bg-primary, #0a0a1a)',
                    padding: '20px',
                    fontFamily: "'Inter', -apple-system, sans-serif",
                }}>
                    <div style={{
                        maxWidth: '520px',
                        width: '100%',
                        background: 'var(--bg-secondary, #111827)',
                        border: '1px solid var(--border-color, rgba(255,255,255,0.08))',
                        borderRadius: '16px',
                        padding: '40px',
                        textAlign: 'center',
                        boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
                    }}>
                        {/* Error Icon */}
                        <div style={{
                            width: '72px',
                            height: '72px',
                            borderRadius: '50%',
                            background: 'rgba(239, 68, 68, 0.1)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            margin: '0 auto 24px',
                        }}>
                            <svg width="36" height="36" viewBox="0 0 24 24" fill="none"
                                stroke="#ef4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="12" cy="12" r="10" />
                                <line x1="12" y1="8" x2="12" y2="12" />
                                <line x1="12" y1="16" x2="12.01" y2="16" />
                            </svg>
                        </div>

                        <h2 style={{
                            color: 'var(--text-primary, #e5e7eb)',
                            fontSize: '1.375rem',
                            fontWeight: 700,
                            marginBottom: '8px',
                        }}>
                            Something went wrong
                        </h2>

                        <p style={{
                            color: 'var(--text-muted, #6b7280)',
                            fontSize: '0.875rem',
                            lineHeight: 1.6,
                            marginBottom: '24px',
                        }}>
                            An unexpected error occurred. You can try reloading the page or going back.
                        </p>

                        {/* Dev-only error details */}
                        {isDev && this.state.error && (
                            <details style={{
                                textAlign: 'left',
                                marginBottom: '24px',
                                padding: '14px',
                                background: 'rgba(239, 68, 68, 0.05)',
                                border: '1px solid rgba(239, 68, 68, 0.15)',
                                borderRadius: '10px',
                            }}>
                                <summary style={{
                                    color: '#ef4444',
                                    fontSize: '0.75rem',
                                    fontWeight: 600,
                                    cursor: 'pointer',
                                    marginBottom: '8px',
                                }}>
                                    Error Details (Development Only)
                                </summary>
                                <pre style={{
                                    color: '#fca5a5',
                                    fontSize: '0.688rem',
                                    fontFamily: "'JetBrains Mono', monospace",
                                    whiteSpace: 'pre-wrap',
                                    wordBreak: 'break-word',
                                    margin: 0,
                                    maxHeight: '200px',
                                    overflowY: 'auto',
                                }}>
                                    {this.state.error.toString()}
                                    {this.state.errorInfo?.componentStack}
                                </pre>
                            </details>
                        )}

                        <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
                            <button
                                onClick={this.handleReset}
                                style={{
                                    padding: '10px 24px',
                                    borderRadius: '10px',
                                    border: '1px solid var(--border-color, rgba(255,255,255,0.1))',
                                    background: 'transparent',
                                    color: 'var(--text-secondary, #9ca3af)',
                                    fontSize: '0.875rem',
                                    fontWeight: 500,
                                    cursor: 'pointer',
                                    transition: 'all 0.2s ease',
                                }}
                            >
                                Try Again
                            </button>
                            <button
                                onClick={this.handleReload}
                                style={{
                                    padding: '10px 24px',
                                    borderRadius: '10px',
                                    border: 'none',
                                    background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                                    color: '#fff',
                                    fontSize: '0.875rem',
                                    fontWeight: 600,
                                    cursor: 'pointer',
                                    boxShadow: '0 4px 12px rgba(99, 102, 241, 0.3)',
                                    transition: 'all 0.2s ease',
                                }}
                            >
                                Reload Page
                            </button>
                        </div>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
