"""
Person Service — CRUD operations for persons and face encodings.
Now uses SQLAlchemy + PostgreSQL instead of Supabase / MockStore.
"""

import pickle
import logging
from typing import Optional, Dict, Any
from uuid import uuid4
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.session import SessionLocal
from app.db.models import Person, FaceEncoding, TrackingSession
from app.core.exceptions import NotFoundException

logger = logging.getLogger(__name__)


def _person_to_dict(person: Person, encoding_count: int = 0) -> dict:
    """Convert a Person ORM object to a dict matching the API response format."""
    return {
        "id": person.id,
        "name": person.name,
        "role": person.role,
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
            if is_active is not None:
                query = query.filter(Person.is_active == is_active)
            if search:
                query = query.filter(Person.name.ilike(f"%{search}%"))

            total = query.count()
            persons_orm = (
                query.order_by(Person.created_at.desc())
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
            person = Person(
                id=str(uuid4()),
                name=data["name"],
                role=data.get("role", "visitor"),
                department=data.get("department"),
                phone=data.get("phone"),
                email=data.get("email"),
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            db.add(person)
            db.commit()
            db.refresh(person)
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
    ) -> dict:
        """Store a face encoding for a person."""
        db: Session = SessionLocal()
        try:
            # Verify person exists
            person = db.query(Person).filter(Person.id == person_id).first()
            if not person:
                raise NotFoundException("Person", person_id)

            enc = FaceEncoding(
                id=str(uuid4()),
                person_id=person_id,
                encoding_data=pickle.dumps(encoding),  # store as binary
                source_image=source_image,
                quality=quality,
                created_at=datetime.now(timezone.utc),
            )
            db.add(enc)
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
                    arr = pickle.loads(enc.encoding_data)
                    encodings.append({
                        "encoding_id": enc.id,
                        "person_id": enc.person_id,
                        "person_name": person_name or "Unknown",
                        "encoding": arr,
                        "quality": enc.quality or 1.0,
                    })
                except Exception as e:
                    logger.warning(f"Failed to decode encoding {enc.id}: {e}")

            logger.info(f"Loaded {len(encodings)} encodings from database")
            return encodings
        finally:
            db.close()

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
