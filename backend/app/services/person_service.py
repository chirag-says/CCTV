import logging
from typing import Optional, Dict, Any
from uuid import uuid4
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.session import SessionLocal
from app.db.models import Person, FaceEncoding, TrackingSession, EncodingVersion
from app.core.exceptions import NotFoundException
from app.models.person import infer_person_type
from app.vision.recognizer import serialize_encoding, deserialize_encoding

logger = logging.getLogger(__name__)


def _person_to_dict(person: Person, encoding_count: int = 0) -> dict:
    """Convert a Person ORM object to a dict matching the API response format."""
    return {
        "id": person.id,
        "name": person.name,
        "role": person.role,
        # Older rows written before migration 009 can still be NULL if the
        # migration has not been applied; fall back rather than emit null.
        "person_type": person.person_type or infer_person_type(person.role),
        "department": person.department,
        "phone": person.phone,
        "email": person.email,
        "avatar_url": person.avatar_url,
        "is_active": person.is_active,
        "created_at": person.created_at.isoformat() if person.created_at else None,
        "updated_at": person.updated_at.isoformat() if person.updated_at else None,
        "encoding_count": encoding_count,
    }


class PersonService:
    """Service for person management operations."""

    @staticmethod
    def list_persons(
        role: Optional[str] = None,
        person_type: Optional[str] = None,
        department: Optional[str] = None,
        is_active: bool = True,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List persons with optional filtering."""
        db: Session = SessionLocal()
        try:
            query = db.query(Person)

            if role:
                query = query.filter(Person.role == role)
            if person_type:
                query = query.filter(Person.person_type == person_type)
            if department:
                # Departments are hand-typed, so 'CSE' and 'cse' both occur.
                query = query.filter(
                    func.upper(func.trim(Person.department)) == department.strip().upper()
                )
            if is_active is not None:
                query = query.filter(Person.is_active == is_active)
            if search:
                query = query.filter(Person.name.ilike(f"%{search}%"))

            total = query.count()
            persons_orm = (
                query.order_by(Person.name.asc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            # Get encoding counts for each person
            persons = []
            for p in persons_orm:
                enc_count = (
                    db.query(func.count(FaceEncoding.id))
                    .filter(FaceEncoding.person_id == p.id)
                    .scalar()
                )
                persons.append(_person_to_dict(p, enc_count or 0))

            return {
                "data": persons,
                "total": total,
            }
        finally:
            db.close()

    @staticmethod
    def get_person(person_id: str) -> dict:
        """Get a specific person by ID."""
        db: Session = SessionLocal()
        try:
            person = db.query(Person).filter(Person.id == person_id).first()
            if not person:
                raise NotFoundException("Person", person_id)

            enc_count = (
                db.query(func.count(FaceEncoding.id))
                .filter(FaceEncoding.person_id == person_id)
                .scalar()
            )

            return _person_to_dict(person, enc_count or 0)
        finally:
            db.close()

    @staticmethod
    def create_person(data: dict) -> dict:
        """Create a new person."""
        db: Session = SessionLocal()
        try:
            role = data.get("role", "visitor")
            person = Person(
                id=str(uuid4()),
                name=data["name"],
                role=role,
                # An explicit person_type wins; otherwise derive it from the
                # designation so callers that predate this field (and the
                # seed scripts) still classify correctly.
                person_type=data.get("person_type") or infer_person_type(role),
                department=data.get("department") or None,
                phone=data.get("phone") or None,
                email=data.get("email") or None,
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            db.add(person)
            db.commit()
            db.refresh(person)

            # Update event queue's in-memory FK guard
            try:
                from app.core.event_queue import get_event_queue
                get_event_queue().add_person_id(person.id)
            except Exception:
                pass

            return _person_to_dict(person, 0)
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create person: {e}")
            raise Exception("Failed to create person")
        finally:
            db.close()

    @staticmethod
    def update_person(person_id: str, data: dict) -> dict:
        """Update a person's information."""
        db: Session = SessionLocal()
        try:
            person = db.query(Person).filter(Person.id == person_id).first()
            if not person:
                raise NotFoundException("Person", person_id)

            update_data = {k: v for k, v in data.items() if v is not None}
            for key, value in update_data.items():
                if hasattr(person, key):
                    setattr(person, key, value)

            # Changing the designation without saying what class it belongs to
            # must not silently leave a teacher classified as a visitor. An
            # explicit person_type in the same request still wins.
            if "role" in update_data and "person_type" not in update_data:
                person.person_type = infer_person_type(update_data["role"])

            person.updated_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(person)

            enc_count = (
                db.query(func.count(FaceEncoding.id))
                .filter(FaceEncoding.person_id == person_id)
                .scalar()
            )
            return _person_to_dict(person, enc_count or 0)
        except NotFoundException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update person: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def delete_person(person_id: str) -> bool:
        """Soft-delete a person."""
        db: Session = SessionLocal()
        try:
            person = db.query(Person).filter(Person.id == person_id).first()
            if not person:
                raise NotFoundException("Person", person_id)

            person.is_active = False
            person.updated_at = datetime.now(timezone.utc)
            db.commit()

            # Update event queue's in-memory FK guard
            try:
                from app.core.event_queue import get_event_queue
                get_event_queue().remove_person_id(person_id)
            except Exception:
                pass

            return True
        except NotFoundException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete person: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def add_face_encoding(
        person_id: str,
        encoding: np.ndarray,
        source_image: Optional[str] = None,
        quality: float = 1.0,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
        det_score: Optional[float] = None,
        blur_score: Optional[float] = None,
        camera_id: Optional[str] = None,
        capture_distance: Optional[float] = None,
        lighting: Optional[str] = None,
        model_version: str = "arcface_r50",
    ) -> dict:
        """Store a face encoding for a person with metadata."""
        db: Session = SessionLocal()
        try:
            # Verify person exists
            person = db.query(Person).filter(Person.id == person_id).first()
            if not person:
                raise NotFoundException("Person", person_id)

            # Get current version for the new encoding
            current_version = PersonService._get_or_create_version(db)
            new_version = current_version + 1

            enc = FaceEncoding(
                id=str(uuid4()),
                person_id=person_id,
                encoding_data=serialize_encoding(encoding),  # raw float32 bytes
                source_image=source_image,
                quality=quality,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                det_score=det_score,
                blur_score=blur_score,
                camera_id=camera_id,
                capture_distance=capture_distance,
                lighting=lighting,
                model_version=model_version,
                version=new_version,
                created_at=datetime.now(timezone.utc),
            )
            db.add(enc)

            # Increment global encoding version
            PersonService._set_version(db, new_version)

            db.commit()

            return {
                "id": enc.id,
                "person_id": person_id,
                "source_image": source_image,
                "quality": quality,
                "created_at": enc.created_at.isoformat() if enc.created_at else None,
            }
        except NotFoundException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to store face encoding: {e}")
            raise Exception("Failed to store face encoding")
        finally:
            db.close()

    @staticmethod
    def get_all_encodings() -> list:
        """
        Load all face encodings from database.
        Returns list of dicts ready for FaceRecognizer.load_encodings().
        Handles both new float32 format and legacy pickle format.
        """
        db: Session = SessionLocal()
        try:
            results = (
                db.query(FaceEncoding, Person.name)
                .join(Person, Person.id == FaceEncoding.person_id)
                .filter(Person.is_active == True)
                .all()
            )

            encodings = []
            for enc, person_name in results:
                try:
                    arr = deserialize_encoding(enc.encoding_data)
                    if arr is not None:
                        encodings.append({
                            "encoding_id": enc.id,
                            "person_id": enc.person_id,
                            "person_name": person_name or "Unknown",
                            "encoding": arr,
                            "quality": enc.quality or 1.0,
                        })
                    else:
                        logger.warning(f"Failed to deserialize encoding {enc.id}")
                except Exception as e:
                    logger.warning(f"Failed to decode encoding {enc.id}: {e}")

            logger.info(f"Loaded {len(encodings)} encodings from database")
            return encodings
        finally:
            db.close()

    @staticmethod
    def get_encoding_version() -> int:
        """Get the current global encoding version number."""
        db: Session = SessionLocal()
        try:
            return PersonService._get_or_create_version(db)
        finally:
            db.close()

    @staticmethod
    def _get_or_create_version(db: Session) -> int:
        """Get or create the encoding version row (internal, uses existing session)."""
        row = db.query(EncodingVersion).filter(EncodingVersion.id == 1).first()
        if row is None:
            row = EncodingVersion(id=1, version=0, updated_at=datetime.now(timezone.utc))
            db.add(row)
            db.flush()
        return row.version

    @staticmethod
    def _set_version(db: Session, version: int):
        """Set the encoding version (internal, uses existing session)."""
        row = db.query(EncodingVersion).filter(EncodingVersion.id == 1).first()
        if row:
            row.version = version
            row.updated_at = datetime.now(timezone.utc)
        else:
            db.add(EncodingVersion(id=1, version=version, updated_at=datetime.now(timezone.utc)))


    @staticmethod
    def get_person_history(
        person_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get tracking session history for a person."""
        db: Session = SessionLocal()
        try:
            query = (
                db.query(TrackingSession)
                .filter(TrackingSession.person_id == person_id)
            )

            total = query.count()
            sessions = (
                query.order_by(TrackingSession.entry_time.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            return {
                "data": [
                    {
                        "id": s.id,
                        "person_id": s.person_id,
                        "camera_id": s.camera_id,
                        "entry_time": s.entry_time.isoformat() if s.entry_time else None,
                        "exit_time": s.exit_time.isoformat() if s.exit_time else None,
                        "duration_sec": s.duration_sec,
                        "status": s.status,
                        "created_at": s.created_at.isoformat() if s.created_at else None,
                    }
                    for s in sessions
                ],
                "total": total,
            }
        finally:
            db.close()
