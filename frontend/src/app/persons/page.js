'use client';

/**
 * Persons Management Page
 * All data fetched from backend API — no mock/dummy data.
 */

import { useState, useEffect } from 'react';
import AppShell from '@/components/AppShell';
import api from '@/lib/api';
import { PlusIcon, EditIcon, ImageIcon, UserIcon, XIcon } from '@/components/Icons';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function PersonsPage() {
    const [persons, setPersons] = useState([]);
    const [total, setTotal] = useState(0);
    const [search, setSearch] = useState('');
    const [roleFilter, setRoleFilter] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [showModal, setShowModal] = useState(false);
    const [editPerson, setEditPerson] = useState(null);
    const [formData, setFormData] = useState({ name: '', role: 'visitor', department: '', email: '', phone: '' });

    useEffect(() => {
        loadPersons();
    }, [search, roleFilter]);

    async function loadPersons() {
        setLoading(true);
        setError(null);
        try {
            const params = {};
            if (search) params.search = search;
            if (roleFilter) params.role = roleFilter;
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
        setFormData({ name: '', role: 'visitor', department: '', email: '', phone: '' });
        setShowModal(true);
    }

    function openEditModal(person) {
        setEditPerson(person);
        setFormData({
            name: person.name,
            role: person.role,
            department: person.department || '',
            email: person.email || '',
            phone: person.phone || '',
        });
        setShowModal(true);
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

    const getRoleBadgeClass = (role) => {
        switch (role) {
            case 'employee': return 'active';
            case 'vip': return 'detection';
            case 'visitor': return 'pending';
            case 'banned': return 'exit';
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
                        value={roleFilter}
                        onChange={(e) => setRoleFilter(e.target.value)}
                        style={{ maxWidth: '160px', marginBottom: 0 }}
                        id="filter-role"
                    >
                        <option value="">All Roles</option>
                        <option value="employee">Employee</option>
                        <option value="visitor">Visitor</option>
                        <option value="vip">VIP</option>
                        <option value="banned">Banned</option>
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
                            {search || roleFilter
                                ? 'No persons match your current filters. Try adjusting your search.'
                                : 'Add a person to start tracking individuals.'}
                        </div>
                        {!search && !roleFilter && (
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
                                    <th>Role</th>
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
                                                    <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{person.name}</div>
                                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{person.email || '—'}</div>
                                                </div>
                                            </div>
                                        </td>
                                        <td>
                                            <span className={`badge ${getRoleBadgeClass(person.role)}`}>
                                                {person.role}
                                            </span>
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
                            <label className="input-label">Role</label>
                            <select
                                className="input"
                                value={formData.role}
                                onChange={(e) => setFormData({ ...formData, role: e.target.value })}
                                id="person-role"
                            >
                                <option value="visitor">Visitor</option>
                                <option value="employee">Employee</option>
                                <option value="vip">VIP</option>
                                <option value="banned">Banned</option>
                            </select>
                        </div>

                        <div className="input-group">
                            <label className="input-label">Department</label>
                            <input
                                className="input"
                                value={formData.department}
                                onChange={(e) => setFormData({ ...formData, department: e.target.value })}
                                placeholder="e.g., Engineering"
                                id="person-department"
                            />
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
        </AppShell>
    );
}
