"""
One-off script: Promote Abhitha from student → faculty (CSE teacher).

Updates person_type and role in the database. Her face encodings are
kept as-is — no re-enrollment needed.
"""

import sys
import os

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, ".")

from datetime import datetime, timezone
from app.db.session import SessionLocal
from app.db.models import Person

def promote():
    db = SessionLocal()
    try:
        person = (
            db.query(Person)
            .filter(Person.name == "Abhitha", Person.is_active == True)
            .first()
        )
        if not person:
            print("❌ Abhitha not found in the database.")
            print("   Run seed_student.py first to enroll her.")
            return

        print(f"Found: {person.name} (id={person.id[:12]}...)")
        print(f"  Current role        : {person.role}")
        print(f"  Current person_type : {person.person_type}")
        print(f"  Current department  : {person.department}")

        # Promote to faculty
        person.role = "assistant_professor"
        person.person_type = "faculty"
        person.department = "CSE"
        person.updated_at = datetime.now(timezone.utc)

        db.commit()

        print()
        print(f"✅ Abhitha promoted to teacher!")
        print(f"  New role        : {person.role}")
        print(f"  New person_type : {person.person_type}")
        print(f"  New department  : {person.department}")

    finally:
        db.close()


if __name__ == "__main__":
    promote()
