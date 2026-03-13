'use client';

import { createContext, useContext, useState, useEffect, useCallback } from 'react';

const ThemeContext = createContext({
    theme: 'dark',
    toggleTheme: () => { },
    setTheme: () => { },
});

export function ThemeProvider({ children }) {
    const [theme, setThemeState] = useState('dark');
    const [mounted, setMounted] = useState(false);
    const [isTransitioning, setIsTransitioning] = useState(false);

    // On mount: read from localStorage or system preference
    useEffect(() => {
        const stored = localStorage.getItem('sentinel-theme');
        if (stored === 'light' || stored === 'dark') {
            setThemeState(stored);
        } else if (window.matchMedia('(prefers-color-scheme: light)').matches) {
            setThemeState('light');
        }
        setMounted(true);
    }, []);

    // Apply theme to <html> element and persist
    useEffect(() => {
        if (!mounted) return;
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('sentinel-theme', theme);
    }, [theme, mounted]);

    const toggleTheme = useCallback(() => {
        // 1. Add transitioning class BEFORE theme changes
        document.documentElement.classList.add('theme-transitioning');
        setIsTransitioning(true);

        // 2. Toggle theme (triggers CSS variable swap)
        setThemeState((prev) => (prev === 'dark' ? 'light' : 'dark'));

        // 3. Remove transitioning class after animation completes
        setTimeout(() => {
            document.documentElement.classList.remove('theme-transitioning');
            setIsTransitioning(false);
        }, 600);
    }, []);

    const setTheme = useCallback((t) => {
        if (t === 'dark' || t === 'light') setThemeState(t);
    }, []);

    return (
        <ThemeContext.Provider value={{ theme, toggleTheme, setTheme, isTransitioning }}>
            {children}
        </ThemeContext.Provider>
    );
}

export function useTheme() {
    return useContext(ThemeContext);
}
