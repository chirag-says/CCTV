'use client';

/**
 * ToastContext — Global toast notification system.
 * 
 * Usage:
 *   const { showToast } = useToast();
 *   showToast({ type: 'success', message: 'Person enrolled!' });
 */

import { createContext, useContext, useState, useCallback, useRef } from 'react';

const ToastContext = createContext(null);

let toastId = 0;

export function ToastProvider({ children }) {
    const [toasts, setToasts] = useState([]);
    const timersRef = useRef({});

    const removeToast = useCallback((id) => {
        if (timersRef.current[id]) {
            clearTimeout(timersRef.current[id]);
            delete timersRef.current[id];
        }
        setToasts(prev => prev.filter(t => t.id !== id));
    }, []);

    const showToast = useCallback(({ type = 'info', message, duration = 4000 }) => {
        const id = ++toastId;
        const toast = { id, type, message };

        setToasts(prev => [...prev.slice(-4), toast]); // Keep max 5 toasts

        if (duration > 0) {
            timersRef.current[id] = setTimeout(() => {
                removeToast(id);
            }, duration);
        }

        return id;
    }, [removeToast]);

    return (
        <ToastContext.Provider value={{ showToast, removeToast }}>
            {children}
            <ToastContainer toasts={toasts} onDismiss={removeToast} />
        </ToastContext.Provider>
    );
}

export function useToast() {
    const context = useContext(ToastContext);
    if (!context) {
        throw new Error('useToast must be used within a ToastProvider');
    }
    return context;
}

/* ── Toast Container ────────────────────────────────── */

function ToastContainer({ toasts, onDismiss }) {
    if (toasts.length === 0) return null;

    return (
        <div style={{
            position: 'fixed',
            top: '20px',
            right: '20px',
            zIndex: 10000,
            display: 'flex',
            flexDirection: 'column',
            gap: '10px',
            maxWidth: '380px',
            width: '100%',
            pointerEvents: 'none',
        }}>
            {toasts.map(toast => (
                <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
            ))}
        </div>
    );
}

/* ── Toast Item ─────────────────────────────────────── */

const TOAST_CONFIG = {
    success: { icon: '✓', color: '#10b981', bg: 'rgba(16, 185, 129, 0.12)', border: 'rgba(16, 185, 129, 0.25)' },
    error: { icon: '✕', color: '#ef4444', bg: 'rgba(239, 68, 68, 0.12)', border: 'rgba(239, 68, 68, 0.25)' },
    warning: { icon: '⚠', color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.12)', border: 'rgba(245, 158, 11, 0.25)' },
    info: { icon: 'ℹ', color: '#6366f1', bg: 'rgba(99, 102, 241, 0.12)', border: 'rgba(99, 102, 241, 0.25)' },
};

function ToastItem({ toast, onDismiss }) {
    const config = TOAST_CONFIG[toast.type] || TOAST_CONFIG.info;

    return (
        <div
            style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '14px 16px',
                background: 'var(--bg-secondary, #1a1b2e)',
                border: `1px solid ${config.border}`,
                borderLeft: `4px solid ${config.color}`,
                borderRadius: '12px',
                boxShadow: '0 8px 32px rgba(0,0,0,0.25)',
                animation: 'slideInRight 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
                pointerEvents: 'auto',
                backdropFilter: 'blur(12px)',
            }}
        >
            {/* Icon */}
            <div style={{
                width: '28px',
                height: '28px',
                borderRadius: '50%',
                background: config.bg,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '0.75rem',
                fontWeight: 700,
                color: config.color,
                flexShrink: 0,
            }}>
                {config.icon}
            </div>

            {/* Message */}
            <div style={{
                flex: 1,
                fontSize: '0.8125rem',
                fontWeight: 500,
                color: 'var(--text-primary, #e5e7eb)',
                lineHeight: 1.4,
            }}>
                {toast.message}
            </div>

            {/* Dismiss */}
            <button
                onClick={() => onDismiss(toast.id)}
                style={{
                    background: 'none',
                    border: 'none',
                    color: 'var(--text-muted, #6b7280)',
                    cursor: 'pointer',
                    padding: '4px',
                    fontSize: '1rem',
                    lineHeight: 1,
                    opacity: 0.6,
                    transition: 'opacity 0.2s',
                    flexShrink: 0,
                }}
                onMouseEnter={e => e.target.style.opacity = 1}
                onMouseLeave={e => e.target.style.opacity = 0.6}
            >
                ×
            </button>
        </div>
    );
}

export default ToastContext;
