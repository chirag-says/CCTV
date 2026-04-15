'use client';

/**
 * AuthContext — Global authentication state management.
 * 
 * Provides:
 * - user: current user object (or null)
 * - token: JWT token (or null)
 * - isAuthenticated: boolean
 * - isLoading: true while checking saved token
 * - login(email, password): authenticate
 * - logout(): clear session and redirect
 */

import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import api from './api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const router = useRouter();

    // ── Check saved token on mount ─────────────────────────────
    useEffect(() => {
        checkAuth();
    }, []);

    async function checkAuth() {
        const token = api.getToken();
        if (!token) {
            setIsLoading(false);
            return;
        }

        try {
            const userData = await api.getMe();
            setUser(userData);
        } catch (err) {
            // Token is invalid or expired — clear it
            console.warn('Auth check failed, clearing token:', err.message);
            api.clearToken();
            setUser(null);
        } finally {
            setIsLoading(false);
        }
    }

    // ── Login ──────────────────────────────────────────────────
    const login = useCallback(async (email, password) => {
        const data = await api.login(email, password);
        // After login, fetch user profile
        try {
            const userData = await api.getMe();
            setUser(userData);
        } catch {
            // If getMe fails, still set basic user info from login response
            setUser({ email });
        }
        return data;
    }, []);

    // ── Logout ─────────────────────────────────────────────────
    const logout = useCallback(() => {
        api.logout();
        setUser(null);
        router.push('/login');
    }, [router]);

    // ── Token Refresh (sliding expiry) ─────────────────────────
    useEffect(() => {
        if (!user) return;

        // Refresh token every 30 minutes
        const refreshInterval = setInterval(async () => {
            try {
                const token = api.getToken();
                if (!token) return;

                // Try to refresh via /api/auth/refresh
                try {
                    const data = await api.request('/api/auth/refresh', {
                        method: 'POST',
                    });
                    if (data.access_token) {
                        api.setToken(data.access_token);
                    }
                } catch {
                    // Refresh endpoint may not exist yet — validate current token
                    await api.getMe();
                }
            } catch (err) {
                console.warn('Token refresh failed:', err.message);
                logout();
            }
        }, 30 * 60 * 1000); // 30 minutes

        return () => clearInterval(refreshInterval);
    }, [user, logout]);

    const value = {
        user,
        token: typeof window !== 'undefined' ? api.getToken() : null,
        isAuthenticated: !!user,
        isLoading,
        login,
        logout,
    };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
}

export default AuthContext;
