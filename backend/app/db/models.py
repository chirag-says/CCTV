"""
SQLAlchemy ORM models for SentinelAI.

These define the actual PostgreSQL table schemas.
"""

from sqlalchemy import (
    Column, String, Boolean, Float, Integer, Text, DateTime, JSON, LargeBinary,
    ForeignKey, Index, func
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, timezone
from uuid import uuid4

Base = declarative_base()


def gen_uuid():
    return str(uuid4())


def utcnow():
    return datetime.now(timezone.utc)


# ── Admin Users ──────────────────────────────────────────────────
class AdminUser(Base):
    __tablename__ = "admin_users"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    role = Column(String(50), default="operator")  # superadmin/admin/operator
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


# ── Cameras ──────────────────────────────────────────────────────
class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(100), nullable=False)
    location = Column(String(255), default="")
    stream_url = Column(String(500), default="0")  # RTSP URL or webcam index
    camera_type = Column(String(50), default="webcam")  # webcam/rtsp/ip
    is_active = Column(Boolean, default=True)
    config = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    events = relationship("DetectionEvent", back_populates="camera", lazy="dynamic")
    tracking_sessions = relationship("TrackingSession", back_populates="camera", lazy="dynamic")
    unknown_faces = relationship("UnknownFace", back_populates="camera", lazy="dynamic")


# ── Persons ──────────────────────────────────────────────────────
class Person(Base):
    __tablename__ = "persons"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(255), nullable=False, index=True)
    role = Column(String(50), default="visitor")  # employee/visitor/vip/banned
    department = Column(String(100), nullable=True)
    phone = Column(String(20), nullable=True)
    email = Column(String(255), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    encodings = relationship("FaceEncoding", back_populates="person", lazy="dynamic",
                             cascade="all, delete-orphan")
    events = relationship("DetectionEvent", back_populates="person", lazy="dynamic")
    tracking_sessions = relationship("TrackingSession", back_populates="person", lazy="dynamic")


# ── Face Encodings ───────────────────────────────────────────────
class FaceEncoding(Base):
    __tablename__ = "face_encodings"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    person_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    encoding_data = Column(LargeBinary, nullable=False)  # pickled numpy array
    source_image = Column(String(500), nullable=True)
    quality = Column(Float, default=1.0)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    person = relationship("Person", back_populates="encodings")


# ── Detection Events ────────────────────────────────────────────
class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    person_id = Column(String(36), ForeignKey("persons.id", ondelete="SET NULL"), nullable=True, index=True)
    camera_id = Column(String(36), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)  # entry/exit/detection/unknown/security_alert/vehicle
    subtype = Column(String(50), nullable=True)  # crowd/loitering/hazard/plate/proximity
    confidence = Column(Float, default=0.0)
    snapshot_url = Column(String(500), nullable=True)
    metadata_json = Column(JSON, default=dict)  # renamed from metadata to avoid SQLAlchemy conflicts
    created_at = Column(DateTime(timezone=True), default=utcnow, index=True)

    # Relationships
    person = relationship("Person", back_populates="events")
    camera = relationship("Camera", back_populates="events")

    # Indexes for analytics queries
    __table_args__ = (
        Index("ix_events_type_created", "event_type", "created_at"),
        Index("ix_events_camera_created", "camera_id", "created_at"),
    )


# ── Tracking Sessions ───────────────────────────────────────────
class TrackingSession(Base):
    __tablename__ = "tracking_sessions"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    person_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    camera_id = Column(String(36), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    entry_time = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    exit_time = Column(DateTime(timezone=True), nullable=True)
    duration_sec = Column(Integer, nullable=True)
    status = Column(String(20), default="active")  # active/completed
    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    person = relationship("Person", back_populates="tracking_sessions")
    camera = relationship("Camera", back_populates="tracking_sessions")


# ── Unknown Faces ────────────────────────────────────────────────
class UnknownFace(Base):
    __tablename__ = "unknown_faces"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    camera_id = Column(String(36), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    snapshot_url = Column(String(500), nullable=True)
    full_frame = Column(String(500), nullable=True)
    face_encoding = Column(LargeBinary, nullable=True)  # pickled numpy array for matching
    occurrence = Column(Integer, default=1)
    first_seen = Column(DateTime(timezone=True), default=utcnow)
    last_seen = Column(DateTime(timezone=True), default=utcnow)
    status = Column(String(20), default="pending")  # pending/enrolled/dismissed
    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    camera = relationship("Camera", back_populates="unknown_faces")
