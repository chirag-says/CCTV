-- 013: Backfill camera zones into `cameras.location`.
--
-- WHY
-- The monitor's "Bunch" layout and cross-camera reasoning group cameras by
-- zone, and `camera_group()` (services/camera_service.py) uses `location` as
-- that zone. 11 of the 13 registered cameras have location='' -- their zone
-- is only recoverable from naming conventions, which this migration makes
-- explicit once, in data, instead of forever in code.
--
-- Data-only migration: no schema change, so verify_schema() / --check are
-- unaffected. Idempotent: every UPDATE is guarded by the current value.
--
-- ROLLBACK (DOWN)
--   UPDATE cameras SET location = '' WHERE location IN ('COE', 'Front Gate', 'Security Cabin');
--   UPDATE cameras SET location = 'Main Entrance' WHERE name = 'MainGate'  AND location = 'Main Gate';
--   UPDATE cameras SET location = 'Main Building' WHERE name = 'MainStream' AND location = 'Main Gate';

BEGIN;

UPDATE cameras SET location = 'COE'
 WHERE location = '' AND LOWER(name) LIKE 'coe%';

-- 'frontside parking' is the front-gate parking view; it belongs with the
-- gate cameras, which its name-token alone would not have grouped.
UPDATE cameras SET location = 'Front Gate'
 WHERE location = '' AND (LOWER(name) LIKE 'frontgate%' OR LOWER(name) LIKE 'frontside%');

UPDATE cameras SET location = 'Security Cabin'
 WHERE location = '' AND LOWER(name) LIKE 'security cabin%';

-- MainGate / MainStream are ONE physical device (same IP, two substreams)
-- registered twice with different location strings. Unify the zone so they
-- bunch together instead of forming two single-camera groups.
UPDATE cameras SET location = 'Main Gate'
 WHERE name IN ('MainGate', 'MainStream')
   AND location IN ('Main Entrance', 'Main Building');

COMMIT;
