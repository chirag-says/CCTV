"""
Cross-camera handoff — one person, one number, on every camera.

WHAT THIS SOLVES
----------------
Each camera's ByteTrack numbers people on its own ("Track #37"), so a person
walking from the COE room into the corridor becomes a stranger with a new
number. This module gives every person a short global number ("#12") and
carries it with them from camera to camera.

It runs in the MAIN process, on the per-frame detection metadata every camera
already publishes (see frame_metadata.py and main.handle_pipeline_event). That
is deliberate: camera pipelines live in several worker processes, so any
process-local registry would split the fleet at a process boundary. The main
process is the one place that sees every camera.

EVIDENCE
--------
Nothing here needs configuring to work. The first version of this module had
three kinds of evidence and all three needed an operator, or an enrolled face;
on the COE corridor that came to zero handoffs, because the floor calibration
had been clicked on a door post and no areas were ever drawn. What a person
can be followed by, most reliable first:

  1. NAME. A track recognised as an enrolled person takes the number that
     person already holds elsewhere. Only ever fills a void: a number that
     already carries a different name is never renamed by this.

  2. FLOOR (overlapping views, operator-calibrated). Cameras calibrated onto
     the same floor project a person's foot point to the same (x, y);
     ZoneTracker owns that association. Calibrations that cannot locate a
     person are ignored, see ground_plane.calibration_problems.

  3. VIEW (overlapping views, learned). Where two cameras see the same floor,
     topology.py learns from ordinary walkers where a foot in one appears in
     the other. Two tracks that keep landing on each other's predicted
     position are one person. Unlike everything below, this is not limited to
     a track's first seconds: a person can walk into the shared region at any
     point in their life.

  4. STITCH (one camera). A track that starts where another just ended is the
     same person the tracker briefly lost. Measured on COE footage this is the
     commonest reason one person held several numbers.

  5. AREA (non-overlapping views, operator-drawn). A track that ends inside
     one handoff area and a new track that starts inside the linked area
     within the gap are the same person.

  6. ROUTE (non-overlapping views, learned). The same thing without the
     drawing: topology.py learns where people who leave one camera turn up on
     another, and how long the walk takes.

  7. LOOK (body appearance). appearance.py embeds each track's body crop with
     a re-identification network. It settles WHICH of several people who took
     the same route is which, and when a look is unmistakable it carries a
     number on its own, with no route known at all. Measured here: 0.85-0.93
     for one person seen from the same side on two cameras against at most
     0.61 for two different people, but only 0.5-0.8 when one camera sees the
     front and the other the back. So it decides alone only with a margin.

RULES THAT KEEP IT HONEST
-------------------------
  * Within-camera distinctness is authoritative. Two live tracks on the same
    camera never share a number; that camera's tracker had the pixels. It is
    enforced after every frame, so a wrong stitch is undone the moment the
    original track comes back.
  * A person still visible somewhere cannot be handed off. They are there, not
    here. (Overlapping views are VIEW's job, which is evidence of exactly
    that.)
  * A track may only take over an earlier number in its first few seconds
    (claim window). After that its number changes only if a shared view shows
    it to be someone already numbered.
  * Each departure can be claimed once.
  * A recognised name outranks everything: nothing joins two different names.

The known failure: people walking as a tight group are one blob to a camera
thirty metres away, so within a group numbers can end up on the wrong member.
The group keeps its numbers; who holds which may swap.

THREADING
---------
handle_pipeline_event is called from camera threads (thread mode) or the event
pump (server mode); everything below runs under one RLock. Config is swapped as
an immutable object by a background refresher, never mutated in place.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np

from app.vision.ground_plane import GroundPlane
from app.vision.topology import Topology

logger = logging.getLogger(__name__)

Key = Tuple[str, int]        # (camera_id, local track_id)
AreaRef = Tuple[str, str]    # (camera_id, area_id)
Point = Tuple[float, float]

# Upper bound for a link's gap. Longer than any walk between adjacent COE
# cameras; anything beyond it is a person who went somewhere else first, and
# accepting it would hand their number to whoever happens to appear next.
MAX_GAP_SEC = 120.0

# Normalised step for differencing the homography: about one pixel at 1000 px.
_FLOOR_STEP = 1e-3


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HandoffArea:
    camera_id: str
    area_id: str
    name: str
    rect: Tuple[float, float, float, float]   # normalised x1, y1, x2, y2

    def contains(self, u: float, v: float) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= u <= x2 and y1 <= v <= y2

    @classmethod
    def parse(cls, camera_id: str, raw: dict) -> Optional["HandoffArea"]:
        """Lenient: a malformed stored area is skipped, never fatal."""
        try:
            x1, y1, x2, y2 = (float(c) for c in raw["rect"])
            area_id = str(raw["id"]).strip()
        except (KeyError, TypeError, ValueError):
            return None
        if not area_id or not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            return None
        return cls(camera_id, area_id, str(raw.get("name") or area_id), (x1, y1, x2, y2))


@dataclass(frozen=True)
class HandoffConfig:
    """Everything the engine needs from the cameras table, parsed once."""

    areas: Mapping[str, Tuple[HandoffArea, ...]] = field(default_factory=dict)
    links: Mapping[AreaRef, Tuple[Tuple[AreaRef, float], ...]] = field(default_factory=dict)
    planes: Mapping[str, GroundPlane] = field(default_factory=dict)
    names: Mapping[str, str] = field(default_factory=dict)
    # Which cameras may share people at all. Learned evidence never links
    # cameras in different groups: nobody walks from COE to the front gate in
    # four seconds, and a look-alike there is a coincidence.
    groups: Mapping[str, str] = field(default_factory=dict)
    # Calibrations that are stored but cannot locate a person, and why.
    unusable_planes: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    @property
    def max_gap(self) -> float:
        return max((gap for links in self.links.values() for _, gap in links), default=0.0)

    def area_at(self, camera_id: str, u: float, v: float) -> Optional[HandoffArea]:
        for area in self.areas.get(camera_id, ()):
            if area.contains(u, v):
                return area
        return None

    def camera_name(self, camera_id: str) -> str:
        return self.names.get(camera_id) or camera_id[:8]

    def group_of(self, camera_id: str) -> str:
        return self.groups.get(camera_id, "")

    @classmethod
    def from_rows(cls, rows: Iterable[tuple]) -> "HandoffConfig":
        """Build from (camera_id, camera_name, cameras.config[, group]) rows.

        Stored shape, per camera:
            config["handoff"] = {
                "areas": [{"id", "name", "rect": [x1, y1, x2, y2]}],
                "links": [{"area_id", "to_camera_id", "to_area_id", "max_gap_sec"}],
            }
        A link is stored once, on the camera owning its first end, and is
        undirected: people walk both ways through the same door.
        """
        areas: Dict[str, List[HandoffArea]] = defaultdict(list)
        planes: Dict[str, GroundPlane] = {}
        unusable: Dict[str, Tuple[str, ...]] = {}
        names: Dict[str, str] = {}
        groups: Dict[str, str] = {}
        raw_links: List[Tuple[str, dict]] = []

        for row in rows:
            camera_id, name, config = str(row[0]), row[1], row[2]
            names[camera_id] = name or ""
            groups[camera_id] = str(row[3] or "") if len(row) > 3 else ""
            config = config if isinstance(config, dict) else {}
            plane = GroundPlane.from_config(camera_id, config)
            if plane is not None:
                problems = plane.problems()
                if problems:
                    unusable[camera_id] = tuple(problems)
                else:
                    planes[camera_id] = plane
            handoff = config.get("handoff")
            if not isinstance(handoff, dict):
                continue
            for raw in handoff.get("areas") or ():
                area = HandoffArea.parse(camera_id, raw) if isinstance(raw, dict) else None
                if area is not None:
                    areas[camera_id].append(area)
            for raw in handoff.get("links") or ():
                if isinstance(raw, dict):
                    raw_links.append((camera_id, raw))

        known = {(a.camera_id, a.area_id) for group in areas.values() for a in group}
        links: Dict[AreaRef, List[Tuple[AreaRef, float]]] = defaultdict(list)
        for camera_id, raw in raw_links:
            try:
                a = (camera_id, str(raw["area_id"]))
                b = (str(raw["to_camera_id"]), str(raw["to_area_id"]))
                gap = float(raw["max_gap_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            # Same-camera links are meaningless here: a track that leaves and
            # re-enters one view is the stitcher's job, and within-camera
            # distinctness forbids the handoff anyway.
            if a not in known or b not in known or a[0] == b[0] or not 0.0 < gap <= MAX_GAP_SEC:
                continue
            links[a].append((b, gap))
            links[b].append((a, gap))

        return cls(
            areas={cam: tuple(group) for cam, group in areas.items()},
            links={ref: tuple(group) for ref, group in links.items()},
            planes=planes,
            names=names,
            groups=groups,
            unusable_planes=unusable,
        )


# ── State ────────────────────────────────────────────────────────────────────

@dataclass
class _Binding:
    """One camera-local track and the global number it carries."""
    gid: int
    first_seen: float
    last_seen: float
    foot: Point = (0.0, 0.0)              # where it stood when last seen
    first_foot: Point = (0.0, 0.0)        # where it first appeared
    height: float = 0.0                   # body box height when last seen
    last_area: Optional[str] = None       # area the track was in when last seen
    zone_person_id: Optional[str] = None
    settled: bool = False                 # took a number by evidence; no more claims
    handed_off: bool = False              # its departure was claimed from another camera
    continued: bool = False               # a later track on this camera carried on from it
    continuation: bool = False            # it carried on from an earlier track; not an arrival
    route_voted: bool = False             # its arrival has been offered to route learning
    unchecked_from: Optional[Key] = None  # track its number was taken from, not yet confirmed by a look
    unchecked_floor: float = 0.0          # ... and how alike the two must look for it to stand
    refused: Set[int] = field(default_factory=set)   # numbers this track was found NOT to be
    trail: Deque[Tuple[float, float, float]] = field(default_factory=deque)
    look: Optional[np.ndarray] = None     # sum of unit appearance vectors
    looks: int = 0

    @property
    def lifetime(self) -> float:
        return self.last_seen - self.first_seen


@dataclass
class _Person:
    gid: int
    created_at: float
    last_seen: float
    last_camera: str
    name: Optional[str] = None
    keys: Set[Key] = field(default_factory=set)
    cameras_visited: List[str] = field(default_factory=list)
    camera_seen: Dict[str, float] = field(default_factory=dict)   # camera -> when last seen there


def _foot(box) -> Optional[Tuple[float, float, float]]:
    """Normalised [x1, y1, x2, y2] -> (u, v, box_height) at the feet."""
    try:
        x1, y1, x2, y2 = (float(c) for c in box)
    except (TypeError, ValueError):
        return None
    return (x1 + x2) / 2.0, y2, max(0.0, y2 - y1)


def _label(obj: dict, gid: int, known_name: Optional[str] = None) -> str:
    """'#12' for an unnamed person, 'Chirag #12' / 'Unknown-3F2A #12' otherwise.

    `known_name` is the name this number already carries from another camera.
    It ranks below this camera's own recognition and above a stranger label.
    """
    base = obj.get("identity") or known_name
    if not base:
        label = str(obj.get("label") or "")
        if label and not label.startswith("Track #") and label != obj.get("kind"):
            base = label
    return f"{base} #{gid}" if base else f"#{gid}"


def _dist(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


# ── Engine ───────────────────────────────────────────────────────────────────

class HandoffEngine:
    def __init__(
        self,
        *,
        claim_window_sec: float = 5.0,
        live_grace_sec: float = 1.0,
        memory_sec: float = 60.0,
        zone_lookup: Optional[Callable[[str], object]] = None,
        topology: Optional[Topology] = None,
        learn: bool = True,
        stitch_gap_sec: float = 4.0,
        look_match: float = 0.82,
        look_margin: float = 0.10,
        look_veto: float = 0.50,
        look_memory_sec: float = 180.0,
        label_settle_sec: float = 0.0,
        numbers_path: Optional[str] = None,
    ):
        """
        Args:
            claim_window_sec: how long a new track may still take over an
                earlier number. Covers the delay before it reaches a handoff
                area, is floor-associated, or is recognised.
            live_grace_sec: a track unseen this long no longer counts as
                visible. Above the analysis interval, so frame skipping by the
                load governor does not make a present person look departed.
            memory_sec: how long a departed person's number can still be
                handed over. Raised automatically to cover the longest link.
            zone_lookup: zone name -> ZoneTracker; None disables floor
                association (tests, or no zone_tracker available).
            topology: learned camera relations; a private empty one if None.
            learn: keep learning view maps and routes from traffic.
            stitch_gap_sec: longest tracker dropout one camera's track is
                carried across.
            look_match / look_margin: similarity at which appearance alone
                carries a number, and how far the runner-up must trail.
            look_veto: below this, a candidate that fits by timing is still
                refused -- it visibly is not the same person.
            look_memory_sec: how long a departed person can be re-found by
                appearance alone.
            label_settle_sec: how long a brand-new track's number is withheld
                from its label while it may still take over an earlier one.
                0 shows every number at once.
            numbers_path: file in which recognised people's numbers for the
                day are kept, so a restart does not renumber them. None keeps
                them in memory only.
        """
        self.claim_window_sec = float(claim_window_sec)
        self.live_grace_sec = float(live_grace_sec)
        self.memory_sec = float(memory_sec)
        self._zone_lookup = zone_lookup
        self.topology = topology if topology is not None else Topology()
        self.learn = bool(learn)
        self.stitch_gap_sec = float(stitch_gap_sec)
        self.look_match = float(look_match)
        self.look_margin = float(look_margin)
        self.look_veto = float(look_veto)
        self.look_memory_sec = float(look_memory_sec)
        self.label_settle_sec = float(label_settle_sec)

        # Fixed behaviour, named so the reasoning sits next to the number.
        self.look_claim_window_sec = 10.0   # a first usable crop can take a few seconds
        self.look_wait_sec = 1.5            # hold an ambiguous choice this long for a crop
        self.clock_slack_sec = 1.5          # two cameras' streams are not equally late
        self.view_interval_sec = 0.1        # shared-view work, per camera
        self.view_sync_sec = 0.35           # positions older than this are not "now"
        self.view_tol_min = 0.03            # agreement tolerance, normalised
        self.view_tol_height = 0.25         # ... or this fraction of the body height
        self.agree_window_sec = 3.0
        self.agree_hits = 10                # sustained, not a crossing
        self.agree_span_sec = 1.5
        self.route_settle_sec = 3.0         # an arrival must outlive tracker noise
        self.trail_sec = 3.5                # history kept per track to tell walking from fidgeting
        self.look_min_looks = 2             # crops before a look may decide alone
        self.look_stitch_min = 0.55         # two halves of one track must look this alike

        self._config = HandoffConfig()
        # Pairs of tracks that may be one person walking through a shared
        # view, held until they have walked far enough to believe. See
        # _learn_view.
        self._walks: Dict[Tuple[Key, Key], dict] = {}
        self._bindings: Dict[Key, _Binding] = {}
        self._people: Dict[int, _Person] = {}
        self._by_camera: Dict[str, Set[Key]] = defaultdict(set)
        self._agreement: Dict[Tuple[Key, Key], Deque[float]] = {}
        self._last_view: Dict[str, float] = {}
        self._next_gid = 1
        # name -> (number, day). A recognised person holds ONE number for the
        # whole calendar day, however long they are out of sight. See
        # _keep_number.
        self._numbers_path = numbers_path
        self._day_numbers: Dict[str, Tuple[int, str]] = {}
        self._load_day_numbers()
        self._last_prune = 0.0
        self._last_routes = 0.0
        self._last_look = float("-inf")
        self._lock = threading.RLock()
        self.stats: Counter = Counter()

    # ── Public API ────────────────────────────────────────────────────────

    def set_config(self, config: HandoffConfig) -> None:
        with self._lock:
            self._config = config

    @property
    def config(self) -> HandoffConfig:
        return self._config

    def process(self, message: dict) -> None:
        """Assign global numbers to one frame_metadata message, in place.

        Adds `global_track_id` to every person object that has a local track
        id and rewrites its `label`. Never raises on malformed input: this
        runs on the live-view path and a bad message must cost one frame of
        labels, not the overlay.
        """
        camera_id = str(message.get("camera_id") or "")
        objects = message.get("objects")
        if not camera_id or not isinstance(objects, list):
            return
        try:
            now = float(message.get("timestamp") or time.time())
        except (TypeError, ValueError):
            now = time.time()

        with self._lock:
            self._prune(now)
            config = self._config

            live: List[Tuple[dict, Key, _Binding, Optional[HandoffArea], tuple]] = []
            for obj in objects:
                if not isinstance(obj, dict) or obj.get("kind") != "person" or obj.get("coasting"):
                    continue
                track_id = obj.get("track_id")
                foot = _foot(obj.get("box"))
                if track_id is None or foot is None:
                    continue
                key = (camera_id, int(track_id))
                binding = self._observe(key, now, foot)
                area = config.area_at(camera_id, foot[0], foot[1])
                binding.last_area = area.area_id if area else None
                live.append((obj, key, binding, area, foot))

            self._associate_floor(camera_id, live, now)
            self._shared_views(camera_id, live, now)

            seen_now = {key for _obj, key, _b, _a, _f in live}
            for obj, key, binding, area, _foot_pt in live:
                name = obj.get("identity") or None
                age = now - binding.first_seen
                if not binding.settled and age <= max(self.claim_window_sec, self.look_claim_window_sec):
                    in_window = age <= self.claim_window_sec
                    if not (in_window and self._claim_by_floor(key, binding, now)):
                        if not (in_window and self._claim_by_stitch(key, binding, seen_now, now)):
                            self._claim_departed(key, binding, area, name, now, in_window)
                if name:
                    self._claim_by_name(key, binding, name, now)
                    person = self._people[binding.gid]
                    if not person.name:
                        person.name = name
                    if person.name == name:
                        self._keep_number(person, now)

            self._enforce_distinct(live, now)
            if self.learn:
                self._learn_routes(now)

            # Label everything that has a number, including coasting tracks:
            # a dashed box must say the same number its solid box did.
            for obj in objects:
                if not isinstance(obj, dict) or obj.get("kind") != "person":
                    continue
                track_id = obj.get("track_id")
                binding = (
                    self._bindings.get((camera_id, int(track_id)))
                    if track_id is not None else None
                )
                if binding is None:
                    continue
                obj["global_track_id"] = binding.gid
                # A track that has just appeared usually takes over an earlier
                # number within a second or so. Showing the throwaway number
                # it holds until then reads as "it lost me": on the live walk
                # every new camera flashed a number like #791 before settling
                # on #510. So the number is held back for that moment.
                if (
                    self.label_settle_sec > 0.0 and not binding.settled
                    and not obj.get("identity")
                    and now - binding.first_seen < self.label_settle_sec
                ):
                    obj["label"] = ""
                    continue
                # The name is shown wherever the person is, not only on the
                # camera that can see their face -- the camera behind them
                # never will. It goes in the LABEL only: `identity` stays this
                # camera's own recognition, so a displayed name can never come
                # back in as evidence for itself.
                person = self._people.get(binding.gid)
                known = person.name if person is not None else None
                if known and not obj.get("identity"):
                    obj["global_name"] = known
                obj["label"] = _label(obj, binding.gid, known)

    def add_appearance(self, camera_id: str, track_id: int, vector, timestamp: Optional[float] = None) -> bool:
        """One body-appearance embedding for a track. Unit length is enforced here.

        Returns False for a track this engine is not following (it ended
        before its crop was embedded), which is routine.
        """
        try:
            vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return False
        norm = float(np.linalg.norm(vec))
        if vec.size == 0 or not np.isfinite(norm) or norm <= 1e-9:
            return False
        vec = vec / norm
        with self._lock:
            binding = self._bindings.get((str(camera_id), int(track_id)))
            if binding is None:
                return False
            if binding.look is None or binding.look.shape != vec.shape:
                binding.look, binding.looks = vec.copy(), 1
            else:
                binding.look = binding.look + vec
                binding.looks += 1
            self._last_look = float(timestamp) if timestamp is not None else time.time()
            self.stats["looks"] += 1
            self._check_claim((str(camera_id), int(track_id)), binding)
            return True

    def snapshot(self, camera_ids: Optional[Iterable[str]] = None) -> dict:
        """People currently known, for the live API. Filtered to cameras if given."""
        wanted = set(camera_ids) if camera_ids is not None else None
        now = time.time()
        with self._lock:
            config = self._config
            people = []
            for person in self._people.values():
                visible = sorted({
                    key[0] for key in person.keys
                    if key in self._bindings and self._is_live(self._bindings[key], now)
                })
                if wanted is not None and not (set(person.cameras_visited) & wanted):
                    continue
                people.append({
                    "global_track_id": person.gid,
                    "name": person.name,
                    "visible_on": [config.camera_name(c) for c in visible],
                    "path": [config.camera_name(c) for c in person.cameras_visited],
                    "last_seen_sec_ago": round(max(0.0, now - person.last_seen), 1),
                })
            people.sort(key=lambda p: p["last_seen_sec_ago"])
            return {
                "people": people,
                "stats": dict(self.stats),
                "learned": self._describe_learned(wanted),
                "evidence": self._describe_evidence(wanted, now),
            }

    def reset(self) -> None:
        with self._lock:
            self._bindings.clear()
            self._people.clear()
            self._by_camera.clear()
            self._agreement.clear()
            self._walks.clear()

    def forget_learned(self, camera_ids: Iterable[str]) -> int:
        """Drop learned view maps and routes for these cameras (one was re-aimed)."""
        with self._lock:
            return self.topology.forget(camera_ids)

    def save_topology(self, path: str, force: bool = False) -> bool:
        with self._lock:
            if not (force or self.topology.dirty):
                return False
            return self.topology.save(path)

    # ── Reporting ─────────────────────────────────────────────────────────

    def _describe_learned(self, wanted: Optional[Set[str]]) -> dict:
        config = self._config
        learned = self.topology.describe(wanted)
        for row in learned["overlaps"]:
            row["a_name"], row["b_name"] = config.camera_name(row["a"]), config.camera_name(row["b"])
        for row in learned["routes"]:
            row["from_name"], row["to_name"] = config.camera_name(row["from"]), config.camera_name(row["to"])
        return learned

    def _describe_evidence(self, wanted: Optional[Set[str]], now: float) -> dict:
        """Which kinds of evidence can fire right now, so "nothing is handing
        off" can be read off a screen instead of out of a database."""
        config = self._config

        def mine(camera_id: str) -> bool:
            return wanted is None or camera_id in wanted

        return {
            "floor_cameras": sorted(config.camera_name(c) for c in config.planes if mine(c)),
            "floor_unusable": {
                config.camera_name(c): list(problems)
                for c, problems in config.unusable_planes.items() if mine(c)
            },
            "area_links": sum(1 for (cam, _), links in config.links.items() if mine(cam) for _ in links) // 2,
            "appearance_active": (now - self._last_look) <= 60.0,
        }

    # ── Bookkeeping ───────────────────────────────────────────────────────

    def _new_person(self, camera_id: str, now: float) -> _Person:
        person = _Person(gid=self._next_gid, created_at=now, last_seen=now, last_camera=camera_id)
        self._next_gid += 1
        self._people[person.gid] = person
        return person

    def _observe(self, key: Key, now: float, foot: Tuple[float, float, float]) -> _Binding:
        point = (foot[0], foot[1])
        binding = self._bindings.get(key)
        if binding is None or binding.gid not in self._people:
            person = self._new_person(key[0], now)
            binding = _Binding(gid=person.gid, first_seen=now, last_seen=now, first_foot=point)
            self._bindings[key] = binding
            self._by_camera[key[0]].add(key)
            person.keys.add(key)
            self.stats["tracks"] += 1
        binding.last_seen = max(binding.last_seen, now)
        binding.foot = point
        binding.height = foot[2]
        trail = binding.trail
        trail.append((now, point[0], point[1]))
        while trail and now - trail[0][0] > self.trail_sec:
            trail.popleft()
        person = self._people[binding.gid]
        person.last_seen = max(person.last_seen, now)
        person.last_camera = key[0]
        self._note_camera(person, key[0], now)
        return binding

    def _prune(self, now: float) -> None:
        if now - self._last_prune < 1.0:
            return
        self._last_prune = now
        memory = max(
            self.memory_sec,
            self._config.max_gap + self.live_grace_sec + 1.0,
            self.look_memory_sec if (now - self._last_look) <= 60.0 else 0.0,
        )
        for key, binding in list(self._bindings.items()):
            if now - binding.last_seen > memory:
                del self._bindings[key]
                self._by_camera[key[0]].discard(key)
                person = self._people.get(binding.gid)
                if person is not None:
                    person.keys.discard(key)
                    if not person.keys:
                        del self._people[binding.gid]
        for pair, hits in list(self._agreement.items()):
            if not hits or now - hits[-1] > self.agree_window_sec:
                del self._agreement[pair]
        for pair, walk in list(self._walks.items()):
            if now - walk["last"] > 5.0:
                del self._walks[pair]

    def _is_live(self, binding: _Binding, now: float) -> bool:
        return now - binding.last_seen <= self.live_grace_sec

    def _visible(self, gid: int, now: float, exclude: Key, camera_id: Optional[str] = None) -> bool:
        """Is `gid` currently seen by some other track (on `camera_id`, if given)?"""
        person = self._people.get(gid)
        if person is None:
            return False
        for key in person.keys:
            if key == exclude or (camera_id is not None and key[0] != camera_id):
                continue
            if self._is_live(self._bindings[key], now):
                return True
        return False

    def _note_camera(self, person: _Person, camera_id: str, now: float) -> None:
        """Record the person's path: a camera is added when they ARRIVE on it.

        Not whenever it differs from the last one. Someone standing where two
        cameras overlap is reported by both, alternately, twenty times a
        second; noting each change filled the path with left, right, left,
        right and pushed where they actually came from off the front.
        """
        away = now - person.camera_seen.get(camera_id, float("-inf"))
        person.camera_seen[camera_id] = max(now, person.camera_seen.get(camera_id, now))
        if away <= 3.0:
            return
        if not person.cameras_visited or person.cameras_visited[-1] != camera_id:
            person.cameras_visited.append(camera_id)
            del person.cameras_visited[:-16]

    def _rebind(self, key: Key, binding: _Binding, gid: int, reason: str, detail: str) -> None:
        """Move one track onto an existing number, dropping its old one if orphaned."""
        old = self._people.get(binding.gid)
        target = self._people[gid]
        if old is not None:
            old.keys.discard(key)
            if not old.keys:
                del self._people[old.gid]
            if not target.name and old.name:
                target.name = old.name
        binding.gid = gid
        binding.settled = True
        target.keys.add(key)
        target.last_seen = max(target.last_seen, binding.last_seen)
        target.last_camera = key[0]
        self._note_camera(target, key[0], binding.last_seen)
        self.stats[f"handoff_{reason}"] += 1
        logger.info(
            f"[handoff] #{gid} -> {self._config.camera_name(key[0])} T{key[1]} by {reason} ({detail})"
        )
        if target.name:
            self._keep_number(target, binding.last_seen)

    # ── A recognised person's number lasts the day ────────────────────────

    @staticmethod
    def _day_of(now: float) -> str:
        return time.strftime("%Y-%m-%d", time.localtime(now))

    def _keep_number(self, person: _Person, now: float) -> None:
        """Give a recognised person the number they already had today.

        A stranger's number is a tag for one visit: three minutes out of sight
        and it is forgotten, because the longer it is kept the more past people
        a newcomer can be mistaken for. A recognised person carries no such
        risk -- their face says who they are -- so their number is reserved
        under their name for the calendar day. Whenever they are recognised
        again, an hour later or after a restart, they get it back.

        The reservation is by name only. It is never evidence that some track
        IS that person; recognition decides that, and this only decides which
        number a recognised person wears.
        """
        name, day = person.name, self._day_of(now)
        if not name:
            return
        reserved = self._day_numbers.get(name)
        if reserved is None or reserved[1] != day:
            self._day_numbers[name] = (person.gid, day)
            self._save_day_numbers()
            return
        number = reserved[0]
        if number == person.gid or number in self._people:
            # Already wearing it -- or someone visible right now is, which
            # means two tracks are being called by one name. That is the
            # recogniser's disagreement to settle, not something to paper over
            # by putting one number on both.
            return
        old = person.gid
        del self._people[old]
        person.gid = number
        self._people[number] = person
        for key in person.keys:
            self._bindings[key].gid = number
        self.stats["kept_for_day"] += 1
        logger.info(f"[handoff] {name} is #{number} today (was shown as #{old})")

    def _load_day_numbers(self) -> None:
        if not self._numbers_path:
            return
        try:
            with open(self._numbers_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            day = str(data["day"])
            self._day_numbers = {str(n): (int(g), day) for n, g in data["named"].items()}
        except FileNotFoundError:
            return
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            logger.warning(f"[handoff] ignoring unreadable {self._numbers_path}: {e}")
            return
        # New numbers start above every reserved one, so nobody is handed a
        # number that belongs to someone who has not come back yet.
        self._next_gid = max([self._next_gid] + [g + 1 for g, _d in self._day_numbers.values()])

    def _save_day_numbers(self) -> None:
        """Written at once, not on a timer: it changes about once per person
        per day, and a restart seconds after someone is first recognised is
        exactly when it is needed."""
        if not self._numbers_path or not self._day_numbers:
            return
        day = max(d for _g, d in self._day_numbers.values())
        today = {n: g for n, (g, d) in self._day_numbers.items() if d == day}
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._numbers_path)), exist_ok=True)
            tmp = f"{self._numbers_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"day": day, "named": today}, f, indent=1)
            os.replace(tmp, self._numbers_path)
        except OSError as e:
            logger.warning(f"[handoff] could not save {self._numbers_path}: {e}")

    def _may_be_same(self, a: _Person, b: _Person) -> bool:
        """Nothing already known rules out these being one person."""
        if a.name and b.name and a.name != b.name:
            return False
        # One camera never held both at once: its tracker saw two people.
        for key_a in a.keys:
            for key_b in b.keys:
                if key_a[0] != key_b[0]:
                    continue
                one, two = self._bindings[key_a], self._bindings[key_b]
                if one.first_seen <= two.last_seen and two.first_seen <= one.last_seen:
                    return False
        return True

    def _merge(self, gid_a: int, gid_b: int, reason: str, detail: str) -> bool:
        """Two numbers turned out to be one person. The older number survives,
        so a number is stable for as long as the person has been followed."""
        a, b = self._people.get(gid_a), self._people.get(gid_b)
        if a is None or b is None or a is b or not self._may_be_same(a, b):
            return False
        keep, drop = (a, b) if a.created_at <= b.created_at else (b, a)
        for key in drop.keys:
            self._bindings[key].gid = keep.gid
        keep.keys |= drop.keys
        for key in keep.keys:
            self._bindings[key].settled = True
        keep.name = keep.name or drop.name
        keep.last_seen = max(keep.last_seen, drop.last_seen)
        for camera_id in drop.cameras_visited:
            if camera_id not in keep.cameras_visited:
                keep.cameras_visited.append(camera_id)
        del keep.cameras_visited[:-16]
        for camera_id, seen in drop.camera_seen.items():
            keep.camera_seen[camera_id] = max(seen, keep.camera_seen.get(camera_id, seen))
        del self._people[drop.gid]
        self.stats[f"handoff_{reason}"] += 1
        logger.info(f"[handoff] #{drop.gid} is #{keep.gid} by {reason} ({detail})")
        if keep.name:
            self._keep_number(keep, keep.last_seen)
        return True

    def _enforce_distinct(self, live, now: float) -> None:
        """Two live tracks on one camera never share a number.

        Every claim above is a judgement and can be wrong; this is the one
        thing known for certain. When the tracker shows two people where a
        number says one, the track that joined later was the mistake and gets
        a number of its own.
        """
        holders: Dict[int, Tuple[Key, _Binding]] = {}
        for _obj, key, binding, _area, _foot_pt in live:
            held = holders.get(binding.gid)
            if held is None:
                holders[binding.gid] = (key, binding)
                continue
            if binding.first_seen >= held[1].first_seen:
                loser_key, loser = key, binding
            else:
                loser_key, loser = held
                holders[binding.gid] = (key, binding)
            person = self._people[loser.gid]
            person.keys.discard(loser_key)
            fresh = self._new_person(loser_key[0], now)
            fresh.cameras_visited.append(loser_key[0])
            fresh.camera_seen[loser_key[0]] = now
            fresh.keys.add(loser_key)
            loser.gid = fresh.gid
            loser.settled = True
            loser.continuation = False
            self.stats["split"] += 1
            logger.info(
                f"[handoff] {self._config.camera_name(loser_key[0])} T{loser_key[1]} was not "
                f"#{person.gid} (both visible at once); now #{fresh.gid}"
            )

    # ── Appearance ────────────────────────────────────────────────────────

    @staticmethod
    def _look_of(binding: _Binding) -> Optional[np.ndarray]:
        if binding.look is None or binding.looks <= 0:
            return None
        norm = float(np.linalg.norm(binding.look))
        return binding.look / norm if norm > 1e-9 else None

    def _similarity(self, look: Optional[np.ndarray], person: _Person, exclude: Optional[Key] = None) -> Optional[float]:
        """Best agreement between a look and any track this person has held."""
        if look is None:
            return None
        best: Optional[float] = None
        for key in person.keys:
            if key == exclude:
                continue
            other = self._look_of(self._bindings[key])
            if other is None or other.shape != look.shape:
                continue
            score = float(np.dot(look, other))
            if best is None or score > best:
                best = score
        return best

    @property
    def _appearance_expected(self) -> bool:
        return self.stats.get("looks", 0) > 0

    # ── Evidence: floor (operator-calibrated overlap) ─────────────────────

    def _associate_floor(self, camera_id: str, live, now: float) -> None:
        plane = self._config.planes.get(camera_id)
        if plane is None or self._zone_lookup is None:
            return
        from app.vision.zone_tracker import TrackObservation

        observations = []
        for _obj, key, binding, _area, (u, v, height) in live:
            binding.zone_person_id = None
            # Foot-point noise grows with the box: a near person's feet are
            # often clipped or occluded by a few percent of their height.
            obs = plane.observe(u, v, noise_px=max(0.004, 0.04 * height), step=_FLOOR_STEP)
            if obs is not None:
                observations.append(TrackObservation(track_id=key[1], x=obs[0], y=obs[1], sigma=obs[2]))
        if not observations:
            return
        assignments = self._zone_lookup(plane.zone).publish(camera_id, observations, now)
        for _obj, key, binding, _area, _foot_pt in live:
            assignment = assignments.get(key[1])
            if assignment is not None:
                binding.zone_person_id = assignment.zone_person_id

    def _claim_by_floor(self, key: Key, binding: _Binding, now: float) -> bool:
        """Same floor spot as a person another camera is tracking right now."""
        if binding.zone_person_id is None:
            return False
        best: Optional[_Person] = None
        for other_key, other in self._bindings.items():
            if (
                other_key[0] == key[0]
                or other.zone_person_id != binding.zone_person_id
                or not self._is_live(other, now)
            ):
                continue
            person = self._people.get(other.gid)
            if person is not None and (best is None or person.created_at < best.created_at):
                best = person
        if best is None or best.gid == binding.gid or self._visible(best.gid, now, key, key[0]):
            return False
        self._rebind(key, binding, best.gid, "floor", f"same floor position as {self._config.camera_name(best.last_camera)}")
        return True

    # ── Evidence: view (learned overlap) ──────────────────────────────────

    def _moving(self, binding: _Binding) -> bool:
        """Has this track actually walked somewhere in the last few seconds.

        Only walkers teach anything. Two people who are each sitting still
        would otherwise 'correspond' at the same two spots for an hour.

        Measured over the whole trail rather than frame to frame: someone
        walking away down a long corridor crosses only a few pixels a second,
        while someone fidgeting at a desk moves more than that per frame but
        ends up where they started.
        """
        trail = binding.trail
        if len(trail) < 3:
            return False
        t1, u1, v1 = trail[-1]
        t0, u0, v0 = trail[0]
        if t1 - t0 < 1.2:
            return False
        return float(np.hypot(u1 - u0, v1 - v0)) >= max(0.008, 0.12 * binding.height)

    def _shared_views(self, camera_id: str, live, now: float) -> None:
        if not live or now - self._last_view.get(camera_id, float("-inf")) < self.view_interval_sec:
            return
        self._last_view[camera_id] = now
        config = self._config
        group = config.group_of(camera_id)
        mine = [(key, binding) for _obj, key, binding, _area, _foot_pt in live]
        for other_cam, keys in self._by_camera.items():
            if other_cam == camera_id or config.group_of(other_cam) != group:
                continue
            theirs = [
                (key, self._bindings[key]) for key in keys
                if now - self._bindings[key].last_seen <= self.view_sync_sec
            ]
            if not theirs:
                continue
            if self.learn:
                self._learn_view(camera_id, mine, other_cam, theirs, now)
            self._match_view(camera_id, mine, other_cam, theirs, now)

    @staticmethod
    def _walking_party(movers) -> Optional[Tuple[Point, float, int]]:
        """(where, body height, who) if these walkers are one party, else None.

        People who walk together stand in one place as far as a floor map is
        concerned, so a group of friends is as good a measurement as a person
        alone -- and in a college corridor far more common. Two walkers in
        different places are not: there is no telling which is which.
        """
        if not movers or len(movers) > 4:
            return None
        height = max(b.height for _k, b in movers)
        feet = [b.foot for _k, b in movers]
        for i in range(len(feet)):
            for j in range(i + 1, len(feet)):
                if _dist(feet[i], feet[j]) > 1.0 * height:
                    return None
        where = (sum(f[0] for f in feet) / len(feet), sum(f[1] for f in feet) / len(feet))
        return where, height, min(b.gid for _k, b in movers)

    def _learn_view(self, cam_a: str, mine, cam_b: str, theirs, now: float) -> None:
        """One walking party here and one there, at the same instant.

        That is a candidate, not yet a lesson. Samples are held back until the
        pair has walked a real distance on BOTH cameras: one person crossing a
        shared floor moves in both views, whereas a walker here paired by
        chance with someone pottering about a lab there moves in one. Without
        the hold-back the busy robotics lab 'overlapped' every corridor.

        A walk is identified by the people's numbers, not their track ids, so
        it survives the tracker dropping and re-finding them halfway down.
        """
        party_a = self._walking_party([(k, b) for k, b in mine if self._moving(b)])
        if party_a is None:
            return
        party_b = self._walking_party([(k, b) for k, b in theirs if self._moving(b)])
        if party_b is None:
            return
        if cam_a > cam_b:
            cam_a, party_a, cam_b, party_b = cam_b, party_b, cam_a, party_a
        (foot_a, height_a, who_a), (foot_b, height_b, who_b) = party_a, party_b
        pair = ((cam_a, who_a), (cam_b, who_b))
        walk = self._walks.get(pair)
        if walk is None:
            walk = self._walks[pair] = {
                "since": now, "last": now, "held": [], "trusted": False,
                "from_a": foot_a, "from_b": foot_b,
                "need_a": max(0.04, 0.6 * height_a), "need_b": max(0.04, 0.6 * height_b),
            }
        walk["last"] = now
        if walk["trusted"]:
            self.topology.add_correspondence(cam_a, foot_a, cam_b, foot_b, pair, height_a, height_b)
            return
        held = walk["held"]
        if len(held) < 400:
            held.append((foot_a, foot_b, height_a, height_b))
        if (
            now - walk["since"] >= 2.0
            and _dist(foot_a, walk["from_a"]) >= walk["need_a"]
            and _dist(foot_b, walk["from_b"]) >= walk["need_b"]
        ):
            walk["trusted"] = True
            for held_a, held_b, tall_a, tall_b in held:
                self.topology.add_correspondence(cam_a, held_a, cam_b, held_b, pair, tall_a, tall_b)
            held.clear()

    def _view_error(self, predicted, other: _Binding) -> Optional[float]:
        """How far `other` stands from where a view map says the person is,
        as a fraction of what is tolerable. None when they cannot be one."""
        if predicted is None:
            return None
        # The size that place in the picture implies.
        tall = predicted[4] or other.height
        if not 0.6 <= other.height / max(tall, 1e-6) <= 1.7:
            return None
        tolerance = max(self.view_tol_min, self.view_tol_height * tall, 2.5 * predicted[2])
        error = _dist((predicted[0], predicted[1]), other.foot) / tolerance
        return error if error <= 1.0 else None

    def _match_view(self, cam_a: str, mine, cam_b: str, theirs, now: float) -> None:
        """Pair tracks that stand where each camera says the other's person is.

        BOTH ways round. The cameras here face each other down thirty metres
        of corridor, so each sees the far half as a handful of pixels: from
        the far camera, people five metres apart land on the same spot, and a
        one-way check joined them. The near camera resolves that easily -- so
        its verdict is required too. Where only one camera can tell people
        apart, the other's map has no consensus there and nothing is joined,
        which is the correct answer for evidence that weak.
        """
        forward = self.topology.view_if_ready(cam_a, cam_b)
        backward = self.topology.view_if_ready(cam_b, cam_a)
        if forward is None or backward is None:
            return
        predict_a = [forward.predict(a.foot) for _key, a in mine]
        if not any(p is not None for p in predict_a):
            return
        predict_b = [backward.predict(b.foot) for _key, b in theirs]
        cost = np.full((len(mine), len(theirs)), 1e6, dtype=np.float64)
        moving_b = [self._moving(b) for _key, b in theirs]
        for i, (key_a, a) in enumerate(mine):
            look_a = self._look_of(a)
            moving_a = self._moving(a)
            for j, (key_b, b) in enumerate(theirs):
                if a.gid == b.gid:
                    cost[i, j] = 0.0     # already one person: keep them paired
                    continue
                # One person is walking on both cameras or standing on both.
                # This is what stops a walker being joined to whoever they
                # pass behind: seen through the lab's glass wall, people in
                # the hallway cross right over the feet of someone at a bench.
                if moving_a != moving_b[j]:
                    continue
                there = self._view_error(predict_a[i], b)
                if there is None:
                    continue
                here = self._view_error(predict_b[j], a)
                if here is None:
                    continue
                d = max(there, here)
                # Position decides whether they CAN be one person; a look
                # only helps choose between people standing together.
                look_b = self._look_of(b)
                if look_a is not None and look_b is not None and look_a.shape == look_b.shape:
                    d = min(1.0, max(0.0, d - 0.6 * (float(np.dot(look_a, look_b)) - 0.5)))
                cost[i, j] = d
        from app.vision.bytetrack import _linear_assignment

        matches, _, _ = _linear_assignment(cost, thresh=1.0)
        for i, j in matches:
            (key_a, a), (key_b, b) = mine[i], theirs[j]
            if a.gid != b.gid and cost[i, j] <= 1.0:
                self._agree(key_a, a, key_b, b, now)

    def _agree(self, key_a: Key, a: _Binding, key_b: Key, b: _Binding, now: float) -> None:
        """Two tracks on two cameras stood in the same place once more."""
        pair = (key_a, key_b) if key_a <= key_b else (key_b, key_a)
        hits = self._agreement.get(pair)
        if hits is None:
            hits = self._agreement[pair] = deque()
        hits.append(now)
        while hits and now - hits[0] > self.agree_window_sec:
            hits.popleft()
        if len(hits) < self.agree_hits or hits[-1] - hits[0] < self.agree_span_sec:
            return
        config = self._config
        if self._merge(
            a.gid, b.gid, "view",
            f"{config.camera_name(key_a[0])} T{key_a[1]} and {config.camera_name(key_b[0])} "
            f"T{key_b[1]} stood in the same place for {hits[-1] - hits[0]:.1f}s",
        ):
            self._agreement.pop(pair, None)
        else:
            hits.clear()

    # ── Evidence: stitch (one camera) ─────────────────────────────────────

    def _claim_by_stitch(self, key: Key, binding: _Binding, seen_now: Set[Key], now: float) -> bool:
        """Carry on from a track this camera's tracker just dropped.

        The candidate must be where that track would have walked to by now,
        and must have gone unseen at least a frame. If the tracker then finds
        the original again, _enforce_distinct takes this number back; and if
        the first crop of this track does not look like the person it took
        over from, add_appearance does (see _check_claim).
        """
        look = self._look_of(binding)
        waiting_for_look = (
            look is None and self._appearance_expected
            and now - binding.first_seen < self.look_wait_sec
        )
        # In a knot of people "the track that was lost here" could be any of
        # them, and position cannot say which. There, only a look decides.
        crowded = any(
            other_key != key
            and _dist(self._bindings[other_key].foot, binding.first_foot)
            <= 1.2 * max(binding.height, self._bindings[other_key].height)
            for other_key in seen_now
        )
        best: Optional[Tuple[float, Key, _Binding]] = None
        second = float("inf")
        for other_key in self._by_camera.get(key[0], ()):
            other = self._bindings[other_key]
            if (
                other_key == key or other_key in seen_now
                or other.continued or other.handed_off
                or other.gid == binding.gid or other.gid in binding.refused
                or other.lifetime < 0.4
            ):
                continue
            gap = binding.first_seen - other.last_seen
            if not 0.0 < gap <= self.stitch_gap_sec:
                continue
            # A blink of the tracker is believed on position alone. Anything
            # longer is long enough for someone else to have stepped into the
            # same spot, so it waits the second it takes for a crop.
            if waiting_for_look and gap > 1.0:
                continue
            person = self._people.get(other.gid)
            # "Is that person still on this camera" is answered by THIS frame,
            # not by the grace period used across cameras: this camera has
            # just said exactly which tracks it can see, so a stitch need not
            # wait a second showing a wrong number first.
            if person is None or any(k in seen_now for k in person.keys if k != key):
                continue
            height = max(binding.height, other.height, 1e-3)
            expected = self._extrapolate(other, gap)
            # Measured on COE footage: a re-found person starts within a
            # quarter of their own height of where they were lost, mostly
            # regardless of the gap, because the ones lost for seconds are the
            # ones standing still.
            distance = _dist(expected, binding.first_foot) / ((0.25 + 0.08 * gap) * height)
            if distance > 1.0:
                continue
            similarity = self._similarity(look, person)
            if similarity is not None and similarity < self.look_stitch_min:
                continue
            # ... and if neither side can be looked at (too small, too far),
            # there is nothing to confirm it with: leave them two people.
            if similarity is None and (gap > 1.0 or crowded) and self._appearance_expected:
                continue
            if best is None or distance < best[0]:
                second = best[0] if best is not None else second
                best = (distance, other_key, other)
            else:
                second = min(second, distance)
        # Two dropped tracks equally near is a coin toss; leave it to the
        # evidence that can tell them apart.
        if best is None or (second <= 1.0 and best[0] > 0.6 * second):
            return False
        _distance, other_key, other = best
        other.continued = True
        binding.continuation = True
        binding.unchecked_from, binding.unchecked_floor = other_key, self.look_stitch_min
        self._rebind(
            key, binding, other.gid, "stitch",
            f"carries on from T{other_key[1]}, lost {binding.first_seen - other.last_seen:.1f}s earlier",
        )
        self._check_claim(key, binding)
        return True

    def _check_claim(self, key: Key, binding: _Binding) -> None:
        """Undo a number taken before the track had been looked at, if the
        first look says it is someone else.

        A stitch, or a handoff along a route, is made on position and timing
        the instant a track appears -- usually before its first crop exists.
        That is right: the number should not wait. But it means the look
        arrives after the decision, so the decision is checked when it does.
        On one camera, seconds apart, one person scores about 0.9, so a stitch
        must clear a high bar; across cameras a person seen from behind can
        score as low as 0.5, so a handoff is only undone below that.
        """
        source_key = binding.unchecked_from
        if source_key is None:
            return
        person = self._people.get(binding.gid)
        # Compared with EVERY track the person has held, exactly as the claim
        # itself is. Comparing with only the track it was taken from let the
        # two disagree: seen live on 2026-11-20, a far-off 90-pixel fragment
        # scored 0.51 against the fragment before it and above the bar against
        # the same man's close-up, so the stitch was made, undone, made and
        # undone a dozen times in one second, each time under a new number.
        similarity = self._similarity(self._look_of(binding), person, exclude=key) if person else None
        if similarity is None:
            return                       # nothing to compare with yet
        binding.unchecked_from = None
        if similarity >= binding.unchecked_floor:
            return
        if len(person.keys) < 2:
            return
        source = self._bindings.get(source_key)
        # Decided once. Without this the next frame makes the same claim again.
        binding.refused.add(person.gid)
        person.keys.discard(key)
        fresh = self._new_person(key[0], binding.first_seen)
        fresh.last_seen = binding.last_seen
        fresh.cameras_visited.append(key[0])
        fresh.camera_seen[key[0]] = binding.last_seen
        fresh.keys.add(key)
        binding.gid = fresh.gid
        binding.settled = False
        # Whoever it was taken from is free to be carried on by the right track.
        if binding.continuation:
            binding.continuation = False
            if source is not None:
                source.continued = False
            self.stats["unstitch"] += 1
        else:
            if source is not None:
                source.handed_off = False
            self.stats["unclaim"] += 1
        logger.info(
            f"[handoff] {self._config.camera_name(key[0])} T{key[1]} does not look like "
            f"#{person.gid} ({similarity:.2f}); now #{fresh.gid}"
        )

    @staticmethod
    def _extrapolate(binding: _Binding, gap: float) -> Point:
        """Where a track that was walking would be `gap` seconds after it was lost."""
        trail = binding.trail
        if len(trail) >= 3:
            t1, u1, v1 = trail[-1]
            t0, u0, v0 = trail[0]
            if t1 - t0 >= 0.4:
                # Capped: a second of extrapolation is a guess, five is fiction.
                dt = min(gap, 1.5) / (t1 - t0)
                return u1 + (u1 - u0) * dt, v1 + (v1 - v0) * dt
        return binding.foot

    # ── Evidence: area, route, look (someone who left another view) ───────

    def _claim_departed(
        self, key: Key, binding: _Binding, area: Optional[HandoffArea],
        name: Optional[str], now: float, in_window: bool,
    ) -> bool:
        """Is this new track someone who recently left a view?

        Candidates that fit a known way between the cameras -- a drawn link,
        or a learned route -- are 'expected': the timing alone would do, and a
        look is used to pick between them. Everyone else in the camera group
        is only a candidate if the look is unmistakable.
        """
        config = self._config
        camera_id = key[0]
        group = config.group_of(camera_id)
        look = self._look_of(binding)
        drawn = dict(config.links.get((camera_id, area.area_id), ())) if (area is not None and in_window) else {}

        expected: List[Tuple[Key, _Binding, Optional[float], str]] = []
        by_look: Dict[int, Tuple[float, Key, _Binding]] = {}

        for other_key, other in self._bindings.items():
            if other.handed_off or other.continued or other.gid == binding.gid or other.gid in binding.refused:
                continue
            gap = binding.first_seen - other.last_seen
            # Still being watched after this track appeared: two people.
            if gap < -self.clock_slack_sec:
                continue
            person = self._people.get(other.gid)
            if person is None or self._visible(other.gid, now, key):
                continue
            # Recognition outranks everything: a departure that was someone
            # else by face is not this person, however well it fits.
            if name and person.name and person.name != name:
                continue
            similarity = self._similarity(look, person, exclude=key)

            how: Optional[str] = None
            if other_key[0] != camera_id and in_window:
                max_gap = drawn.get((other_key[0], other.last_area)) if other.last_area else None
                if max_gap is not None and gap <= max_gap:
                    how = "area"
                elif config.group_of(other_key[0]) == group:
                    model = self.topology.transition_if_any(other_key[0], camera_id)
                    window = model.window(other.foot, binding.first_foot) if model is not None else None
                    if window is not None and window[0] - 1.0 <= gap <= window[1] + 1.5:
                        how = "route"
            if how is not None:
                if similarity is None or similarity >= self.look_veto:
                    expected.append((other_key, other, similarity, how))
                continue

            if (
                similarity is not None
                and binding.looks >= self.look_min_looks
                and gap <= self.look_memory_sec
                and config.group_of(other_key[0]) == group
                and (other_key[0] != camera_id or gap > self.stitch_gap_sec)
            ):
                held = by_look.get(other.gid)
                if held is None or similarity > held[0]:
                    by_look[other.gid] = (similarity, other_key, other)

        if expected:
            people = {other.gid for _k, other, _s, _h in expected}
            if (
                len(people) > 1 and look is None and self._appearance_expected
                and now - binding.first_seen < self.look_wait_sec
            ):
                # Several people took this route together. A crop is on its
                # way; a second's wait beats a coin toss that then sticks.
                return False
            if any(s is not None for _k, _o, s, _h in expected):
                other_key, other, similarity, how = max(
                    expected, key=lambda c: (c[2] if c[2] is not None else self.look_veto, -c[1].last_seen)
                )
            else:
                # First out, first in: what a doorway queue actually does.
                other_key, other, similarity, how = min(expected, key=lambda c: c[1].last_seen)
            return self._take_departure(key, binding, area, other_key, other, how, similarity)

        if by_look:
            ranked = sorted(by_look.values(), key=lambda c: c[0], reverse=True)
            best = ranked[0]
            runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
            if best[0] >= self.look_match and best[0] - runner_up >= self.look_margin:
                return self._take_departure(key, binding, area, best[1], best[2], "look", best[0])
        return False

    def _take_departure(
        self, key: Key, binding: _Binding, area: Optional[HandoffArea],
        other_key: Key, other: _Binding, how: str, similarity: Optional[float],
    ) -> bool:
        config = self._config
        gap = max(binding.first_seen - other.last_seen, 0.0)
        other.handed_off = True
        source = config.camera_name(other_key[0])
        if how == "area":
            left = next(
                (a.name for a in config.areas.get(other_key[0], ()) if a.area_id == other.last_area),
                other.last_area,
            )
            detail = f"left {source} '{left}' {gap:.1f}s earlier, entered '{area.name if area else '?'}'"
        elif how == "route":
            detail = f"left {source} {gap:.1f}s earlier, by a route people are known to walk"
        else:
            detail = f"left {source} {gap:.1f}s earlier"
        if similarity is not None:
            detail += f", looks the same ({similarity:.2f})"
        elif how != "look":
            # Taken on timing alone; the first crop gets a say (_check_claim).
            binding.unchecked_from, binding.unchecked_floor = other_key, self.look_veto
        self._rebind(key, binding, other.gid, how, detail)
        return True

    def _learn_routes(self, now: float) -> None:
        """Offer each settled arrival to route learning, once.

        Every departure in the camera group from the half minute before it is
        a vote; topology.TransitionModel finds the routes among the
        coincidences. Tracker stutter is kept out on both sides -- a track
        that carried on from another is not an arrival, and one that was
        carried on from is not a departure.
        """
        if now - self._last_routes < 1.0:
            return
        self._last_routes = now
        config = self._config
        for key, arrival in self._bindings.items():
            if arrival.route_voted or now - arrival.first_seen < self.route_settle_sec:
                continue
            arrival.route_voted = True
            # Someone who arrived walked in. A track that appears and stays
            # put is a person the detector had been missing, not an arrival.
            if (
                arrival.continuation or arrival.lifetime < 2.0
                or _dist(arrival.first_foot, arrival.foot) < max(0.02, 0.3 * arrival.height)
            ):
                continue
            group = config.group_of(key[0])
            for other_key, departure in self._bindings.items():
                if other_key[0] == key[0] or config.group_of(other_key[0]) != group:
                    continue
                # ... and someone who left was walking when last seen.
                if (
                    departure.continued or departure.lifetime < 2.0
                    or self._is_live(departure, now) or not self._moving(departure)
                ):
                    continue
                self.topology.add_transition(
                    other_key[0], departure.foot, key[0], arrival.first_foot,
                    arrival.first_seen - departure.last_seen, arrival.first_seen,
                )

    # ── Evidence: name ────────────────────────────────────────────────────

    def _claim_by_name(self, key: Key, binding: _Binding, name: str, now: float) -> bool:
        """Recognised as someone who already holds a number elsewhere."""
        current = self._people.get(binding.gid)
        if current is None or current.name:
            return False
        best: Optional[_Person] = None
        for person in self._people.values():
            if person.gid == binding.gid or person.name != name:
                continue
            if self._visible(person.gid, now, key, key[0]):
                continue
            if best is None or person.last_seen > best.last_seen:
                best = person
        if best is None:
            return False
        self._rebind(key, binding, best.gid, "name", f"recognised as {name}")
        return True


# ── Singleton + config refresh ───────────────────────────────────────────────

_engine: Optional[HandoffEngine] = None
_engine_lock = threading.Lock()


def load_config_from_db() -> HandoffConfig:
    from app.db.models import Camera
    from app.db.session import SessionLocal
    from app.services.camera_service import camera_group

    with SessionLocal() as db:
        rows = db.query(Camera.id, Camera.name, Camera.config, Camera.location).all()
    return HandoffConfig.from_rows(
        (str(r[0]), r[1], r[2], camera_group(r[1] or "", r[3] or "")) for r in rows
    )


def reload_handoff_config() -> None:
    """Re-read areas, links and calibrations now. Called after an edit."""
    engine = get_handoff()
    if engine is None:
        return
    try:
        engine.set_config(load_config_from_db())
    except Exception as e:
        logger.warning(f"[handoff] config reload failed: {e}")


def _state_path(setting: str, default: str) -> str:
    """Relative paths are anchored at backend/, not the working directory, so
    it is the same file however the server was started."""
    from app.config import settings

    path = str(getattr(settings, setting, default))
    if not os.path.isabs(path):
        backend = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(backend, path)
    return path


def topology_path() -> str:
    """Where learned topology lives."""
    return _state_path("HANDOFF_TOPOLOGY_PATH", "data/handoff_topology.json")


def numbers_path() -> str:
    """Where recognised people's numbers for the day live."""
    return _state_path("HANDOFF_NUMBERS_PATH", "data/handoff_numbers.json")


def _refresh_loop(engine: HandoffEngine, interval: float) -> None:
    while True:
        try:
            engine.set_config(load_config_from_db())
        except Exception as e:
            logger.warning(f"[handoff] config refresh failed: {e}")
        try:
            engine.save_topology(topology_path())
        except Exception as e:
            logger.warning(f"[handoff] topology save failed: {e}")
        time.sleep(interval)


def get_handoff() -> Optional[HandoffEngine]:
    """Process-wide engine, or None when HANDOFF_ENABLED is off."""
    global _engine
    if _engine is not None:
        return _engine
    from app.config import settings

    if not getattr(settings, "HANDOFF_ENABLED", True):
        return None
    with _engine_lock:
        if _engine is None:
            from app.vision.zone_tracker import get_zone_registry

            topology = Topology()
            if topology.load(topology_path()):
                learned = topology.describe()
                logger.info(
                    f"[handoff] loaded learned topology: {len(learned['overlaps'])} shared "
                    f"view(s), {len(learned['routes'])} route(s)"
                )
            engine = HandoffEngine(
                claim_window_sec=settings.HANDOFF_CLAIM_WINDOW_SEC,
                live_grace_sec=settings.HANDOFF_LIVE_GRACE_SEC,
                memory_sec=settings.HANDOFF_MEMORY_SEC,
                zone_lookup=get_zone_registry().zone,
                topology=topology,
                learn=bool(getattr(settings, "HANDOFF_LEARN_TOPOLOGY", True)),
                stitch_gap_sec=float(getattr(settings, "HANDOFF_STITCH_GAP_SEC", 4.0)),
                look_match=float(getattr(settings, "HANDOFF_LOOK_MATCH", 0.82)),
                look_margin=float(getattr(settings, "HANDOFF_LOOK_MARGIN", 0.10)),
                label_settle_sec=float(getattr(settings, "HANDOFF_LABEL_SETTLE_SEC", 1.5)),
                numbers_path=numbers_path() if getattr(settings, "HANDOFF_KEEP_NAMED_ALL_DAY", True) else None,
            )
            threading.Thread(
                target=_refresh_loop,
                args=(engine, max(5.0, float(settings.HANDOFF_CONFIG_REFRESH_SEC))),
                name="handoff-config",
                daemon=True,
            ).start()
            _engine = engine
    return _engine
