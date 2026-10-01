"""
SQLAlchemy ORM models for SentinelAI.

These define the actual PostgreSQL table schemas.
"""

from sqlalchemy import (
    Column, String, Boolean, Float, Integer, Text, DateTime, JSON, LargeBinary,
    ForeignKey, Index, Sequence, func
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
    events = relationship("DetectionEvent", back_populates="camera", lazy="dynamic", cascade="all, delete-orphan", passive_deletes=True)
    tracking_sessions = relationship("TrackingSession", back_populates="camera", lazy="dynamic", cascade="all, delete-orphan", passive_deletes=True)
    unknown_faces = relationship("UnknownFace", back_populates="camera", lazy="dynamic", cascade="all, delete-orphan", passive_deletes=True)


# ── Persons ──────────────────────────────────────────────────────
class Person(Base):
    __tablename__ = "persons"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(255), nullable=False, index=True)
    # `role` is the human-facing designation shown in the UI:
    # assistant_professor / professor / hod / student / employee / visitor…
    # It is free text and has always been so — the seed scripts write
    # designations, the old UI dropdown wrote access flags.
    role = Column(String(50), default="visitor")
    # `person_type` is the machine-facing class that attendance, presence and
    # the daily summary filter on: faculty / student / staff / visitor.
    # Added in migration 009 — see app/migrations/009_college_features.sql.
    person_type = Column(String(20), default="visitor", index=True)
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
    suggested_encodings = relationship("SuggestedEncoding", back_populates="person", lazy="dynamic",
                                       cascade="all, delete-orphan")
    events = relationship("DetectionEvent", back_populates="person", lazy="dynamic")
    tracking_sessions = relationship("TrackingSession", back_populates="person", lazy="dynamic")


# ── Face Encodings ───────────────────────────────────────────────
class FaceEncoding(Base):
    __tablename__ = "face_encodings"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    person_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    encoding_data = Column(LargeBinary, nullable=False)  # raw float32 bytes (512 * 4 = 2048 bytes)
    source_image = Column(String(500), nullable=True)
    quality = Column(Float, default=1.0)

    # Pose metadata (from InsightFace 5-point landmarks)
    yaw = Column(Float, nullable=True)              # Head yaw angle (degrees, -90 to +90)
    pitch = Column(Float, nullable=True)             # Head pitch angle (degrees)
    roll = Column(Float, nullable=True)              # Head roll angle (degrees)

    # Quality metadata
    det_score = Column(Float, nullable=True)         # SCRFD detection confidence
    blur_score = Column(Float, nullable=True)        # Laplacian variance (higher = sharper)

    # Capture context (invaluable for debugging recognition failures)
    camera_id = Column(String(36), nullable=True)    # Which camera captured this
    capture_distance = Column(Float, nullable=True)  # Estimated distance from face size (meters)
    lighting = Column(String(20), nullable=True)     # "bright" / "dim" / "backlit"

    # Versioning
    model_version = Column(String(50), default="arcface_r50")  # Embedding model identifier
    version = Column(Integer, default=1)             # Row version for incremental sync

    # Health tracking (for smart replacement decisions)
    usage_count = Column(Integer, default=0)         # How many times this encoding was a Top-1 candidate
    success_count = Column(Integer, default=0)       # How many times it contributed to a successful match
    last_matched = Column(DateTime(timezone=True), nullable=True)  # Last time this encoding was the best match
    pose_bucket = Column(String(20), nullable=True)  # "frontal" / "left" / "right" / "extreme_left" / "extreme_right"
    appearance_tags = Column(String(255), nullable=True)  # Comma-separated: "glasses,cap,mask"

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

    # Stable identity of an unenrolled person (migration 011).
    # Deliberately NOT a foreign key to unknown_faces: that row can be
    # enrolled or dismissed, and neither should cascade into or block the
    # permanent event log. `person_id` and this are mutually exclusive --
    # a subject is either enrolled or they are not.
    unknown_person_id = Column(String(36), nullable=True, index=True)

    created_at = Column(DateTime(timezone=True), default=utcnow, index=True)

    # Relationships
    person = relationship("Person", back_populates="events")
    camera = relationship("Camera", back_populates="events")

    # Indexes for analytics queries
    __table_args__ = (
        Index("ix_events_type_created", "event_type", "created_at"),
        Index("ix_events_camera_created", "camera_id", "created_at"),
        # "Where has this stranger been?" -- see migration 011.
        Index("ix_events_unknown_created", "unknown_person_id", "created_at"),
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
    face_encoding = Column(LargeBinary, nullable=True)  # raw float32 bytes for matching
    occurrence = Column(Integer, default=1)
    first_seen = Column(DateTime(timezone=True), default=utcnow)
    last_seen = Column(DateTime(timezone=True), default=utcnow)
    status = Column(String(20), default="pending")  # pending/enrolled/dismissed
    cluster_id = Column(String(36), nullable=True, index=True)  # Groups same unknown person across sightings
    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Pose + quality metadata (migration 010). Mirrors face_encodings so an
    # unknown face and a gallery encoding are described in the same vocabulary.
    # NULL means "pose unknown" — rows written before 010, or faces detected
    # without SCRFD landmarks. Never treat NULL as "frontal".
    yaw = Column(Float, nullable=True)
    pitch = Column(Float, nullable=True)
    roll = Column(Float, nullable=True)
    pose_bucket = Column(String(20), nullable=True)
    quality = Column(Float, nullable=True)
    blur_score = Column(Float, nullable=True)
    det_score = Column(Float, nullable=True)
    lighting = Column(String(20), nullable=True)

    # Relationships
    camera = relationship("Camera", back_populates="unknown_faces")


# ── Unknown Face Views (multi-view re-identification) ────────────
class UnknownFaceView(Base):
    """
    One additional face view of an unknown identity (migration 012).

    WHY THIS EXISTS
    `unknown_faces.face_encoding` holds a single vector: the first view ever
    seen. UnknownRegistry matches best-of-K across SEVERAL views, so that a
    stranger first seen frontally is still matched in profile -- but before
    this table those extra views lived only in one camera thread's memory.
    Every refresh, restart and second camera rebuilt the identity from that one
    frozen vector, and best-of-K silently became best-of-1.

    Measured consequence: 0.0% cross-camera re-identification across 7 cameras.

    Mirrors `face_encodings`, which does the same job for enrolled people.
    """
    __tablename__ = "unknown_face_views"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    unknown_face_id = Column(
        String(36),
        ForeignKey("unknown_faces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    encoding = Column(LargeBinary, nullable=False)  # raw float32 bytes
    # Pose/quality of THIS view. The Gallery Optimizer's diversity gate reads
    # them when an unknown is eventually promoted to a person.
    pose_bucket = Column(String(20), nullable=True)
    quality = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


# ── Encoding Version (for incremental cache sync) ────────────────
class EncodingVersion(Base):
    __tablename__ = "encoding_versions"

    id = Column(Integer, primary_key=True, autoincrement=False, default=1)
    version = Column(Integer, default=0, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow)


# ── Suggested Encodings (admin approval queue) ────────────────────
class SuggestedEncoding(Base):
    __tablename__ = "suggested_encodings"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    person_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    encoding_data = Column(LargeBinary, nullable=False)  # raw float32 bytes
    snapshot_url = Column(String(500), nullable=True)     # Face crop image path
    camera_id = Column(String(36), nullable=True)

    # Quality & pose metrics
    quality = Column(Float, nullable=True)
    yaw = Column(Float, nullable=True)
    pitch = Column(Float, nullable=True)
    roll = Column(Float, nullable=True)
    det_score = Column(Float, nullable=True)
    blur_score = Column(Float, nullable=True)
    lighting = Column(String(20), nullable=True)
    pose_bucket = Column(String(20), nullable=True)       # "frontal" / "left" / "right" etc.
    appearance_tags = Column(String(255), nullable=True)   # "glasses,cap" etc.

    # Match context
    similarity = Column(Float, nullable=True)             # Cosine similarity to the matched person
    reason = Column(String(255), nullable=True)           # "New left profile", "Better quality", etc.

    # Status
    status = Column(String(20), default="pending", index=True)  # pending/approved/rejected
    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    person = relationship("Person", back_populates="suggested_encodings")


# ── Unknown Clusters (AI Identity Review) ────────────────────────
class UnknownCluster(Base):
    __tablename__ = "unknown_clusters"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    representative_snapshot = Column(String(500), nullable=True)
    representative_face_id = Column(String(36), nullable=True)
    embedding_centroid = Column(LargeBinary, nullable=True)
    member_count = Column(Integer, default=0)
    first_seen = Column(DateTime(timezone=True), nullable=True)
    last_seen = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), default="active", index=True)  # active/reviewed/merged/ignored
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    candidates = relationship("IdentityReviewCandidate", back_populates="cluster", lazy="dynamic")


# ── Identity Review Candidates ───────────────────────────────────
class IdentityReviewCandidate(Base):
    __tablename__ = "identity_review_candidates"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    cluster_id = Column(String(36), ForeignKey("unknown_clusters.id", ondelete="CASCADE"), nullable=False, index=True)
    person_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)

    # Similarity scores
    embedding_similarity = Column(Float, nullable=True)
    verifier_similarity = Column(Float, nullable=True)
    vlm_confidence = Column(Float, nullable=True)
    image_quality_score = Column(Float, nullable=True)
    pose_quality_score = Column(Float, nullable=True)
    cluster_consistency = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=True)

    # AI analysis
    vlm_explanation = Column(Text, nullable=True)
    recommended_action = Column(String(20), default="review")  # approve/reject/review

    # Gallery Optimizer enrichment
    gallery_improvement_value = Column(String(20), nullable=True)    # none/low/medium/high
    gallery_improvement_reason = Column(String(255), nullable=True)  # "New Extreme Left profile"
    coverage_before = Column(Float, nullable=True)                   # e.g. 63.0
    coverage_after = Column(Float, nullable=True)                    # e.g. 74.0
    vlm_detected_pose = Column(String(20), nullable=True)            # extreme_left
    vlm_quality_assessment = Column(String(20), nullable=True)       # good/fair/poor
    review_priority = Column(String(20), default="low")              # critical/high/medium/low

    # Review status
    status = Column(String(20), default="pending", index=True)  # pending/approved/rejected/skipped
    reviewed_by = Column(String(36), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    cluster = relationship("UnknownCluster", back_populates="candidates")
    person = relationship("Person")


# ── Gallery Updates (Approved Learning Events) ───────────────────
class GalleryUpdate(Base):
    __tablename__ = "gallery_updates"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    person_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    candidate_id = Column(String(36), ForeignKey("identity_review_candidates.id", ondelete="SET NULL"), nullable=True)
    cluster_id = Column(String(36), ForeignKey("unknown_clusters.id", ondelete="SET NULL"), nullable=True)
    encoding_data = Column(LargeBinary, nullable=True)
    snapshot_url = Column(String(500), nullable=True)
    pose_bucket = Column(String(20), nullable=True)
    quality = Column(Float, nullable=True)
    action = Column(String(20), default="append")  # append/replace
    replaced_encoding_id = Column(String(36), nullable=True)
    gallery_version = Column(Integer, nullable=True)              # Per-person version for ordered rollback

    # Member-level provenance (migration 010). One row per individual unknown
    # face that was actually added — not one row per averaged cluster centroid.
    source_face_id = Column(String(36), nullable=True)            # unknown_faces.id this embedding came from
    selection_reason = Column(String(255), nullable=True)         # Why the diversity gate accepted it
    encoding_id = Column(String(36), nullable=True)               # face_encodings.id created on apply (for rollback)
    coverage_before = Column(Float, nullable=True)                # Gallery coverage % before this addition
    coverage_after = Column(Float, nullable=True)                 # Gallery coverage % after this addition
    status = Column(String(20), default="pending", index=True)  # pending/applied/rolled_back
    applied_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    person = relationship("Person")
    candidate = relationship("IdentityReviewCandidate")


# ── Incident Reviews (Security Incident Workflow) ────────────────
class IncidentReview(Base):
    """
    Operator review state for a security incident.

    An "incident" IS a detection_events row with event_type='security_alert';
    this table only carries the human workflow layered on top. Keeping it
    separate means the event log stays immutable and append-only (it is
    written from the vision event path on a hot table), and "list all open
    incidents" is an indexed filter rather than a JSON scan of metadata_json.

    Rows are created lazily -- an alert has no review row until someone
    changes its status. Absence of a row means status='open'.

    V1 uses `status` only. `severity`, `assignee` and `notes` are defined
    now because adding a column later needs hand-applied SQL (there is no
    Alembic), and unused columns cost nothing.
    """
    __tablename__ = "incident_reviews"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    event_id = Column(
        String(36),
        ForeignKey("detection_events.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    # Human-facing incident number ("INCIDENT #1042"). Starts at 1000 so it
    # reads as a case number rather than a row count.
    seq = Column(
        Integer,
        Sequence("incident_review_seq", start=1000),
        server_default=Sequence("incident_review_seq", start=1000).next_value(),
        unique=True, nullable=False,
    )
    status = Column(String(20), default="open", index=True)  # open/investigating/resolved
    severity = Column(String(20), nullable=True)   # operator override of the derived severity
    assignee = Column(String(36), nullable=True)   # admin_users.id — not used in V1
    notes = Column(JSON, default=list)             # [{author, text, at}] — not used in V1
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    event = relationship("DetectionEvent")


# ── Identity Review Activity (Audit Trail) ───────────────────────
class IdentityReviewActivity(Base):
    __tablename__ = "identity_review_activity"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    candidate_id = Column(String(36), nullable=True, index=True)
    cluster_id = Column(String(36), nullable=True)
    action = Column(String(50), nullable=False)  # approved/rejected/skipped/merged_cluster/created_person/ignored_forever
    actor = Column(String(36), nullable=True)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
