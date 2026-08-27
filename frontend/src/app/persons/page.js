'use client';

/**
 * Persons Management Page
 * All data fetched from backend API — no mock/dummy data.
 */

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import AppShell from '@/components/AppShell';
import api, { API_BASE } from '@/lib/api';
import PersonTimeline from '@/components/PersonTimeline';
import { PlusIcon, EditIcon, ImageIcon, UserIcon, XIcon, ClockIcon } from '@/components/Icons';

// Designations shown in the UI. These are what the seed scripts write and
// what the college actually uses — the previous list (employee/visitor/vip/
// banned) could not represent a single seeded teacher, so editing one
// silently reclassified them.
const ROLE_OPTIONS = [
    { value: 'professor', label: 'Professor', type: 'faculty' },
    { value: 'associate_professor', label: 'Associate Professor', type: 'faculty' },
    { value: 'assistant_professor', label: 'Assistant Professor', type: 'faculty' },
    { value: 'hod', label: 'Head of Department', type: 'faculty' },
    { value: 'lecturer', label: 'Lecturer', type: 'faculty' },
    { value: 'student', label: 'Student', type: 'student' },
    { value: 'employee', label: 'Staff', type: 'staff' },
    { value: 'visitor', label: 'Visitor', type: 'visitor' },
];

const PERSON_TYPES = [
    { value: 'faculty', label: 'Faculty' },
    { value: 'student', label: 'Student' },
    { value: 'staff', label: 'Staff' },
    { value: 'visitor', label: 'Visitor' },
];

const DEPARTMENTS = ['CSE', 'ECE', 'Mechanical', 'Civil'];

const roleLabel = (role) =>
    ROLE_OPTIONS.find((r) => r.value === role)?.label
    || (role || '—').replace(/_/g, ' ');

export default function PersonsPage() {
    const [persons, setPersons] = useState([]);
    const [total, setTotal] = useState(0);
    const [search, setSearch] = useState('');
    const [typeFilter, setTypeFilter] = useState('');
    const [deptFilter, setDeptFilter] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [showModal, setShowModal] = useState(false);
    const [editPerson, setEditPerson] = useState(null);
    const [formData, setFormData] = useState({
        name: '', role: 'visitor', person_type: 'visitor',
        department: '', email: '', phone: '',
    });
    const [timelinePerson, setTimelinePerson] = useState(null);
    const router = useRouter();

    useEffect(() => {
        loadPersons();
    }, [search, typeFilter, deptFilter]);

    async function loadPersons() {
        setLoading(true);
        setError(null);
        try {
            const params = { limit: 200 };
            if (search) params.search = search;
            if (typeFilter) params.person_type = typeFilter;
            if (deptFilter) params.department = deptFilter;
            const result = await api.getPersons(params);
            setPersons(result.data || []);
            setTotal(result.total || 0);
        } catch (e) {
            setError('Could not load persons. Make sure the backend is running.');
            setPersons([]);
            setTotal(0);
        } finally {
            setLoading(false);
        }
    }

    function openAddModal() {
        setEditPerson(null);
        setFormData({
            name: '', role: 'visitor', person_type: 'visitor',
            department: '', email: '', phone: '',
        });
        setShowModal(true);
    }

    function openEditModal(person) {
        setEditPerson(person);
        setFormData({
            name: person.name,
            role: person.role,
            person_type: person.person_type || 'visitor',
            department: person.department || '',
            email: person.email || '',
            phone: person.phone || '',
        });
        setShowModal(true);
    }

    /**
     * Picking a designation moves person_type with it, so an admin cannot
     * create an "Assistant Professor" who is classified as a visitor and is
     * then invisible to attendance. It stays editable underneath for the
     * cases that don't map cleanly.
     */
    function handleRoleChange(role) {
        const inferred = ROLE_OPTIONS.find((r) => r.value === role)?.type;
        setFormData((f) => ({
            ...f, role, person_type: inferred || f.person_type,
        }));
    }

    async function handleSave() {
        try {
            if (editPerson) {
                await api.updatePerson(editPerson.id, formData);
            } else {
                await api.createPerson(formData);
            }
            setShowModal(false);
            loadPersons();
        } catch (e) {
            alert(e.message);
        }
    }

    const getTypeBadgeClass = (personType) => {
        switch (personType) {
            case 'faculty': return 'active';
            case 'student': return 'detection';
            case 'staff': return 'enrolled';
            case 'visitor': return 'pending';
            default: return 'pending';
        }
    };

    return (
        <AppShell>
            <div className="page-header">
                <h1>Persons Management</h1>
                <p>Manage known individuals and their face encodings</p>
            </div>

            {/* Toolbar */}
            <div className="card" style={{ marginBottom: '20px' }}>
                <div className="toolbar-responsive" style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
                    <input
                        type="text"
                        className="input"
                        placeholder="Search by name..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        style={{ maxWidth: '300px', marginBottom: 0 }}
                        id="search-persons"
                    />
                    <select
                        className="input"
                        value={typeFilter}
                        onChange={(e) => setTypeFilter(e.target.value)}
                        style={{ maxWidth: '150px', marginBottom: 0 }}
                        id="filter-person-type"
                    >
                        <option value="">All Types</option>
                        {PERSON_TYPES.map((t) => (
                            <option key={t.value} value={t.value}>{t.label}</option>
                        ))}
                    </select>
                    <select
                        className="input"
                        value={deptFilter}
                        onChange={(e) => setDeptFilter(e.target.value)}
                        style={{ maxWidth: '160px', marginBottom: 0 }}
                        id="filter-department"
                    >
                        <option value="">All Departments</option>
                        {DEPARTMENTS.map((d) => (
                            <option key={d} value={d}>{d}</option>
                        ))}
                    </select>
                    <button className="btn btn-primary ml-auto" onClick={openAddModal} id="add-person-btn">
                        <PlusIcon size={16} /> Add Person
                    </button>
                </div>
            </div>

            {/* Persons Table */}
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">Known Persons ({total})</h3>
                </div>

                {loading ? (
                    <div style={{ padding: '60px 20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                        Loading persons...
                    </div>
                ) : error ? (
                    <div style={{ padding: '60px 20px', textAlign: 'center' }}>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '16px' }}>{error}</div>
                        <button className="btn btn-primary btn-sm" onClick={loadPersons}>Retry</button>
                    </div>
                ) : persons.length === 0 ? (
                    <div className="empty-state">
                        <div className="empty-state-icon"><UserIcon size={32} /></div>
                        <div className="empty-state-title">No persons found</div>
                        <div className="empty-state-text">
                            {search || typeFilter || deptFilter
                                ? 'No persons match your current filters. Try adjusting your search.'
                                : 'Add a person to start recognising individuals.'}
                        </div>
                        {!search && !typeFilter && !deptFilter && (
                            <button className="btn btn-primary" onClick={openAddModal} style={{ marginTop: '16px' }}>
                                <PlusIcon size={16} /> Add Person
                            </button>
                        )}
                    </div>
                ) : (
                    <div className="table-responsive-wrapper">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Person</th>
                                    <th>Type</th>
                                    <th>Designation</th>
                                    <th>Department</th>
                                    <th>Encodings</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {persons.map((person) => (
                                    <tr key={person.id}>
                                        <td>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                                {person.avatar_url ? (
                                                    <img
                                                        src={`${API_BASE}${person.avatar_url}`}
                                                        alt={person.name}
                                                        className="avatar"
                                                        style={{ objectFit: 'cover' }}
                                                    />
                                                ) : (
                                                    <div className="avatar">{person.name.charAt(0)}</div>
                                                )}
                                                <div>
                                                    <div
                                                        style={{ fontWeight: 600, color: 'var(--primary)', cursor: 'pointer', textDecoration: 'none' }}
                                                        onClick={() => router.push(`/persons/${person.id}`)}
                                                        onMouseEnter={e => e.target.style.textDecoration = 'underline'}
                                                        onMouseLeave={e => e.target.style.textDecoration = 'none'}
                                                    >
                                                        {person.name}
                                                    </div>
                                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{person.email || '—'}</div>
                                                </div>
                                            </div>
                                        </td>
                                        <td>
                                            <span className={`badge ${getTypeBadgeClass(person.person_type)}`}>
                                                {person.person_type || 'visitor'}
                                            </span>
                                        </td>
                                        <td style={{ color: 'var(--text-secondary)' }}>
                                            {roleLabel(person.role)}
                                        </td>
                                        <td>{person.department || '—'}</td>
                                        <td>
                                            <span style={{
                                                fontFamily: "'JetBrains Mono', monospace",
                                                color: person.encoding_count > 0 ? 'var(--success)' : 'var(--text-muted)',
                                            }}>
                                                {person.encoding_count} face{person.encoding_count !== 1 ? 's' : ''}
                                            </span>
                                        </td>
                                        <td>
                                            <span className={`badge ${person.is_active ? 'active' : 'offline'}`}>
                                                {person.is_active ? 'Active' : 'Inactive'}
                                            </span>
                                        </td>
                                        <td>
                                            <div style={{ display: 'flex', gap: '6px' }}>
                                                <button
                                                    className="btn btn-secondary btn-sm"
                                                    onClick={() => openEditModal(person)}
                                                    style={{ display: 'flex', alignItems: 'center', gap: '4px' }}
                                                >
                                                <EditIcon size={14} /> Edit
                                                </button>
                                                <button
                                                    className="btn btn-secondary btn-sm"
                                                    onClick={() => setTimelinePerson(person)}
                                                    style={{ display: 'flex', alignItems: 'center', gap: '4px' }}
                                                >
                                                    <ClockIcon size={14} /> History
                                                </button>
                                                <button className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                                    <ImageIcon size={14} /> Add Face
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {/* Add/Edit Modal */}
            {showModal && (
                <div className="modal-overlay" onClick={() => setShowModal(false)}>
                    <div className="modal" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h3 className="modal-title">{editPerson ? 'Edit Person' : 'Add New Person'}</h3>
                            <button className="modal-close" onClick={() => setShowModal(false)}><XIcon size={18} /></button>
                        </div>

                        <div className="input-group">
                            <label className="input-label">Full Name *</label>
                            <input
                                className="input"
                                value={formData.name}
                                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                placeholder="Enter full name"
                                id="person-name"
                            />
                        </div>

                        <div className="input-group">
                            <label className="input-label">Designation</label>
                            <select
                                className="input"
                                value={formData.role}
                                onChange={(e) => handleRoleChange(e.target.value)}
                                id="person-role"
                            >
                                {ROLE_OPTIONS.map((r) => (
                                    <option key={r.value} value={r.value}>{r.label}</option>
                                ))}
                            </select>
                        </div>

                        <div className="input-group">
                            <label className="input-label">Type</label>
                            <select
                                className="input"
                                value={formData.person_type}
                                onChange={(e) => setFormData({ ...formData, person_type: e.target.value })}
                                id="person-type"
                            >
                                {PERSON_TYPES.map((t) => (
                                    <option key={t.value} value={t.value}>{t.label}</option>
                                ))}
                            </select>
                            <div style={{
                                fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px',
                            }}>
                                Only people marked <strong>Faculty</strong> appear in attendance
                                and campus presence.
                            </div>
                        </div>

                        <div className="input-group">
                            <label className="input-label">Department</label>
                            <input
                                className="input"
                                list="department-options"
                                value={formData.department}
                                onChange={(e) => setFormData({ ...formData, department: e.target.value })}
                                placeholder="e.g., CSE"
                                id="person-department"
                            />
                            <datalist id="department-options">
                                {DEPARTMENTS.map((d) => <option key={d} value={d} />)}
                            </datalist>
                        </div>

                        <div className="input-group">
                            <label className="input-label">Email</label>
                            <input
                                className="input"
                                type="email"
                                value={formData.email}
                                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                placeholder="person@company.com"
                                id="person-email"
                            />
                        </div>

                        <div className="input-group">
                            <label className="input-label">Phone</label>
                            <input
                                className="input"
                                value={formData.phone}
                                onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                                placeholder="+91-XXXXXXXXXX"
                                id="person-phone"
                            />
                        </div>

                        <div className="modal-actions">
                            <button className="btn btn-secondary" onClick={() => setShowModal(false)}>Cancel</button>
                            <button className="btn btn-primary" onClick={handleSave} id="save-person-btn">
                                {editPerson ? 'Update' : 'Add Person'}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Person History Timeline Panel */}
            {timelinePerson && (
                <>
                    <div
                        style={{
                            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)',
                            zIndex: 8999, animation: 'fadeIn 0.2s ease',
                        }}
                        onClick={() => setTimelinePerson(null)}
                    />
                    <PersonTimeline person={timelinePerson} onClose={() => setTimelinePerson(null)} />
                </>
            )}
        </AppShell>
    );
}
