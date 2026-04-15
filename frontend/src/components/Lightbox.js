'use client';

/**
 * Lightbox — Full-screen image preview overlay.
 * 
 * Usage:
 *   <Lightbox src={imageUrl} alt="Description" onClose={() => setOpen(false)} />
 * 
 * Or use the hook:
 *   const { openLightbox, LightboxComponent } = useLightbox();
 *   <img onClick={() => openLightbox(src, alt)} />
 *   <LightboxComponent />
 */

import { useState, useEffect, useCallback } from 'react';

export default function Lightbox({ src, alt = 'Preview', onClose }) {
    const [zoom, setZoom] = useState(1);
    const [loaded, setLoaded] = useState(false);

    // Close on Escape
    useEffect(() => {
        const handleKey = (e) => {
            if (e.key === 'Escape') onClose();
            if (e.key === '+' || e.key === '=') setZoom(z => Math.min(z + 0.25, 4));
            if (e.key === '-') setZoom(z => Math.max(z - 0.25, 0.25));
            if (e.key === '0') setZoom(1);
        };
        document.addEventListener('keydown', handleKey);
        document.body.style.overflow = 'hidden';
        return () => {
            document.removeEventListener('keydown', handleKey);
            document.body.style.overflow = '';
        };
    }, [onClose]);

    return (
        <div
            style={{
                position: 'fixed',
                inset: 0,
                zIndex: 9999,
                background: 'rgba(0, 0, 0, 0.92)',
                backdropFilter: 'blur(12px)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'zoom-out',
                animation: 'fadeIn 0.2s ease',
            }}
            onClick={onClose}
        >
            {/* Top bar */}
            <div
                style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '16px 24px',
                    background: 'linear-gradient(180deg, rgba(0,0,0,0.6) 0%, transparent 100%)',
                    zIndex: 10,
                }}
                onClick={(e) => e.stopPropagation()}
            >
                <span style={{
                    color: 'rgba(255,255,255,0.8)',
                    fontSize: '0.8125rem',
                    fontWeight: 500,
                }}>
                    {alt}
                </span>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    {/* Zoom controls */}
                    <button
                        onClick={() => setZoom(z => Math.max(z - 0.25, 0.25))}
                        style={zoomBtnStyle}
                        title="Zoom out (-)"
                    >−</button>
                    <span style={{
                        color: 'rgba(255,255,255,0.7)',
                        fontSize: '0.75rem',
                        fontFamily: "'JetBrains Mono', monospace",
                        minWidth: '40px',
                        textAlign: 'center',
                    }}>
                        {Math.round(zoom * 100)}%
                    </span>
                    <button
                        onClick={() => setZoom(z => Math.min(z + 0.25, 4))}
                        style={zoomBtnStyle}
                        title="Zoom in (+)"
                    >+</button>
                    <button
                        onClick={() => setZoom(1)}
                        style={{ ...zoomBtnStyle, fontSize: '0.6875rem', width: 'auto', padding: '4px 10px' }}
                        title="Reset zoom (0)"
                    >Reset</button>

                    <div style={{ width: '1px', height: '20px', background: 'rgba(255,255,255,0.15)', margin: '0 6px' }} />

                    {/* Download */}
                    <a
                        href={src}
                        download
                        onClick={(e) => e.stopPropagation()}
                        style={{ ...zoomBtnStyle, textDecoration: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                        title="Download"
                    >
                        ⬇
                    </a>

                    {/* Close */}
                    <button
                        onClick={onClose}
                        style={{
                            ...zoomBtnStyle,
                            fontSize: '1.125rem',
                            background: 'rgba(239, 68, 68, 0.2)',
                            borderColor: 'rgba(239, 68, 68, 0.3)',
                        }}
                        title="Close (Esc)"
                    >×</button>
                </div>
            </div>

            {/* Image */}
            <img
                src={src}
                alt={alt}
                onClick={(e) => e.stopPropagation()}
                onLoad={() => setLoaded(true)}
                style={{
                    maxWidth: '90vw',
                    maxHeight: '85vh',
                    objectFit: 'contain',
                    transform: `scale(${zoom})`,
                    transition: 'transform 0.2s ease',
                    cursor: zoom > 1 ? 'grab' : 'zoom-in',
                    borderRadius: '8px',
                    opacity: loaded ? 1 : 0,
                    boxShadow: '0 20px 80px rgba(0,0,0,0.5)',
                }}
            />

            {/* Loading */}
            {!loaded && (
                <div style={{
                    position: 'absolute',
                    color: 'rgba(255,255,255,0.5)',
                    fontSize: '0.875rem',
                }}>
                    Loading...
                </div>
            )}
        </div>
    );
}

const zoomBtnStyle = {
    width: '28px',
    height: '28px',
    borderRadius: '6px',
    border: '1px solid rgba(255,255,255,0.15)',
    background: 'rgba(255,255,255,0.08)',
    color: 'rgba(255,255,255,0.8)',
    cursor: 'pointer',
    fontSize: '0.875rem',
    fontWeight: 600,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.15s',
};

/**
 * useLightbox — Hook for easy lightbox usage.
 * 
 * const { openLightbox, LightboxComponent } = useLightbox();
 * <img onClick={() => openLightbox('/snapshot.jpg', 'Face crop')} />
 * {LightboxComponent}
 */
export function useLightbox() {
    const [lightbox, setLightbox] = useState(null);

    const openLightbox = useCallback((src, alt) => {
        setLightbox({ src, alt });
    }, []);

    const closeLightbox = useCallback(() => {
        setLightbox(null);
    }, []);

    const LightboxComponent = lightbox
        ? <Lightbox src={lightbox.src} alt={lightbox.alt} onClose={closeLightbox} />
        : null;

    return { openLightbox, closeLightbox, LightboxComponent };
}
