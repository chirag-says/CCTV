/**
 * SVG Icon Components — Heroicons-style inline SVGs.
 * All icons use currentColor so they inherit text/fill color from parent.
 * Default size is 20x20 (can be overridden with size prop).
 */

const Icon = ({ children, size = 20, className = '', style = {}, ...props }) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.75}
        strokeLinecap="round"
        strokeLinejoin="round"
        className={className}
        style={{ flexShrink: 0, ...style }}
        {...props}
    >
        {children}
    </svg>
);

// ── Navigation Icons ─────────────────────────────────

export const DashboardIcon = (props) => (
    <Icon {...props}>
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="4" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="11" width="7" height="10" rx="1" />
    </Icon>
);

export const MonitorIcon = (props) => (
    <Icon {...props}>
        <rect x="2" y="3" width="20" height="14" rx="2" />
        <line x1="8" y1="21" x2="16" y2="21" />
        <line x1="12" y1="17" x2="12" y2="21" />
    </Icon>
);

export const UsersIcon = (props) => (
    <Icon {...props}>
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
        <path d="M16 3.13a4 4 0 0 1 0 7.75" />
    </Icon>
);

export const CameraIcon = (props) => (
    <Icon {...props}>
        <path d="M23 7l-7 5 7 5V7z" />
        <rect x="1" y="5" width="15" height="14" rx="2" />
    </Icon>
);

export const UnknownFaceIcon = (props) => (
    <Icon {...props}>
        <circle cx="12" cy="8" r="5" />
        <path d="M20 21a8 8 0 1 0-16 0" />
        <line x1="9" y1="8" x2="9.01" y2="8" strokeWidth="2.5" />
        <line x1="15" y1="8" x2="15.01" y2="8" strokeWidth="2.5" />
    </Icon>
);

export const EventLogIcon = (props) => (
    <Icon {...props}>
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" y1="13" x2="8" y2="13" />
        <line x1="16" y1="17" x2="8" y2="17" />
        <polyline points="10 9 9 9 8 9" />
    </Icon>
);

export const AnalyticsIcon = (props) => (
    <Icon {...props}>
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6" y1="20" x2="6" y2="14" />
    </Icon>
);

// ── Status / Event Icons ─────────────────────────────

export const EntryIcon = (props) => (
    <Icon {...props}>
        <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" />
        <polyline points="10 17 15 12 10 7" />
        <line x1="15" y1="12" x2="3" y2="12" />
    </Icon>
);

export const ExitIcon = (props) => (
    <Icon {...props}>
        <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
        <polyline points="16 17 21 12 16 7" />
        <line x1="21" y1="12" x2="9" y2="12" />
    </Icon>
);

export const DetectionIcon = (props) => (
    <Icon {...props}>
        <circle cx="12" cy="12" r="3" fill="currentColor" stroke="none" />
        <circle cx="12" cy="12" r="8" />
    </Icon>
);

export const AlertIcon = (props) => (
    <Icon {...props}>
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
        <line x1="12" y1="9" x2="12" y2="13" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
    </Icon>
);

// ── Action Icons ─────────────────────────────────────

export const CheckIcon = (props) => (
    <Icon {...props}>
        <polyline points="20 6 9 17 4 12" />
    </Icon>
);

export const XIcon = (props) => (
    <Icon {...props}>
        <line x1="18" y1="6" x2="6" y2="18" />
        <line x1="6" y1="6" x2="18" y2="18" />
    </Icon>
);

export const PlusIcon = (props) => (
    <Icon {...props}>
        <line x1="12" y1="5" x2="12" y2="19" />
        <line x1="5" y1="12" x2="19" y2="12" />
    </Icon>
);

export const EditIcon = (props) => (
    <Icon {...props}>
        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
    </Icon>
);

export const TrashIcon = (props) => (
    <Icon {...props}>
        <polyline points="3 6 5 6 21 6" />
        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </Icon>
);

export const SettingsIcon = (props) => (
    <Icon {...props}>
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </Icon>
);

// ── UI / Layout Icons ────────────────────────────────

export const GridIcon = (props) => (
    <Icon {...props}>
        <rect x="3" y="3" width="7" height="7" />
        <rect x="14" y="3" width="7" height="7" />
        <rect x="3" y="14" width="7" height="7" />
        <rect x="14" y="14" width="7" height="7" />
    </Icon>
);

export const SquareIcon = (props) => (
    <Icon {...props}>
        <rect x="3" y="3" width="18" height="18" rx="2" />
    </Icon>
);

export const ShieldIcon = (props) => (
    <Icon {...props}>
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </Icon>
);

export const UserIcon = (props) => (
    <Icon {...props}>
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
        <circle cx="12" cy="7" r="4" />
    </Icon>
);

export const ClockIcon = (props) => (
    <Icon {...props}>
        <circle cx="12" cy="12" r="10" />
        <polyline points="12 6 12 12 16 14" />
    </Icon>
);

export const SignalIcon = (props) => (
    <Icon {...props}>
        <path d="M2 20h.01" strokeWidth="2.5" />
        <path d="M7 20v-4" />
        <path d="M12 20v-8" />
        <path d="M17 20V8" />
        <path d="M22 20V4" />
    </Icon>
);

export const FolderIcon = (props) => (
    <Icon {...props}>
        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    </Icon>
);

export const ImageIcon = (props) => (
    <Icon {...props}>
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor" stroke="none" />
        <polyline points="21 15 16 10 5 21" />
    </Icon>
);

export const LiveIcon = (props) => (
    <Icon {...props}>
        <circle cx="12" cy="12" r="2" fill="currentColor" stroke="none" />
        <path d="M16.24 7.76a6 6 0 0 1 0 8.49" />
        <path d="M7.76 16.24a6 6 0 0 1 0-8.49" />
        <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
        <path d="M4.93 19.07a10 10 0 0 1 0-14.14" />
    </Icon>
);

export const SearchIcon = (props) => (
    <Icon {...props}>
        <circle cx="11" cy="11" r="8" />
        <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </Icon>
);

export const FilterIcon = (props) => (
    <Icon {...props}>
        <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
    </Icon>
);

export const ArrowRightIcon = (props) => (
    <Icon {...props}>
        <line x1="5" y1="12" x2="19" y2="12" />
        <polyline points="12 5 19 12 12 19" />
    </Icon>
);

export const RefreshIcon = (props) => (
    <Icon {...props}>
        <polyline points="23 4 23 10 17 10" />
        <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
    </Icon>
);

export const CheckCircleIcon = (props) => (
    <Icon {...props}>
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
        <polyline points="22 4 12 14.01 9 11.01" />
    </Icon>
);

export const VideoCameraIcon = (props) => (
    <Icon {...props}>
        <path d="M23 7l-7 5 7 5V7z" />
        <rect x="1" y="5" width="15" height="14" rx="2" />
    </Icon>
);

export const VideoCameraOffIcon = (props) => (
    <Icon {...props}>
        <path d="M16 16v1a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h2m5.66 0H14a2 2 0 0 1 2 2v3.34l1 1L23 7v10" />
        <line x1="1" y1="1" x2="23" y2="23" />
    </Icon>
);

// ── Security / Safety Icons ────────────────────────────

export const FireIcon = (props) => (
    <Icon {...props}>
        <path d="M12 12c0-3 2.5-6 2.5-6s2.5 3 2.5 6a2.5 2.5 0 1 1-5 0z" />
        <path d="M12 22c-4.97 0-9-4.03-9-9 0-4 4-9 9-14 5 5 9 10 9 14 0 4.97-4.03 9-9 9z" />
    </Icon>
);

export const CrowdIcon = (props) => (
    <Icon {...props}>
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
        <path d="M16 3.13a4 4 0 0 1 0 7.75" />
        <circle cx="9" cy="7" r="2" fill="currentColor" stroke="none" />
    </Icon>
);

export const LoiterIcon = (props) => (
    <Icon {...props}>
        <circle cx="12" cy="5" r="3" />
        <path d="M12 8v6" />
        <path d="M9 20l3-6 3 6" />
        <path d="M18 12a6 6 0 0 0-12 0" strokeDasharray="2 2" />
    </Icon>
);

export const HazardIcon = (props) => (
    <Icon {...props}>
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
        <line x1="12" y1="9" x2="12" y2="13" />
        <circle cx="12" cy="17" r="1" fill="currentColor" stroke="none" />
    </Icon>
);

export const SecurityIcon = (props) => (
    <Icon {...props}>
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        <path d="M12 8v4" />
        <path d="M12 16h.01" />
    </Icon>
);

// ── Traffic / Vehicle Icons ──────────────────────────────

export const VehicleIcon = (props) => (
    <Icon {...props}>
        <path d="M5 17h14" />
        <path d="M5 17a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h1l1.5-3h9L18 7h1a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2" />
        <circle cx="7.5" cy="17" r="2" />
        <circle cx="16.5" cy="17" r="2" />
    </Icon>
);

export const PlateIcon = (props) => (
    <Icon {...props}>
        <rect x="2" y="6" width="20" height="12" rx="2" />
        <path d="M6 12h2" />
        <path d="M12 10v4" />
        <path d="M16 12h2" />
        <path d="M8 15h8" />
    </Icon>
);

export const TrafficIcon = (props) => (
    <Icon {...props}>
        <rect x="6" y="1" width="12" height="22" rx="2" />
        <circle cx="12" cy="6" r="2" fill="currentColor" stroke="none" />
        <circle cx="12" cy="12" r="2" />
        <circle cx="12" cy="18" r="2" />
    </Icon>
);

export const ProximityIcon = (props) => (
    <Icon {...props}>
        <circle cx="7" cy="5" r="2" />
        <path d="M5 9h4v5L7 20" />
        <path d="M9 14l-2 6" />
        <path d="M13 10l2-2 2 2-2 2z" fill="currentColor" stroke="none" />
        <path d="M17 7h3a2 2 0 0 1 2 2v4" />
        <path d="M22 15h-5v2h5" />
        <circle cx="18" cy="19" r="1.5" />
    </Icon>
);

export const VideoUploadIcon = (props) => (
    <Icon {...props}>
        <polygon points="23 7 16 12 23 17 23 7" />
        <rect x="1" y="5" width="14" height="14" rx="2" />
        <polyline points="8 9 8 15" />
        <polyline points="5 12 8 9 11 12" />
    </Icon>
);
