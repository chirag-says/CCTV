"""
Camera topology, learned by watching people walk.

WHY THIS EXISTS
---------------
Cross-camera handoff needs to know how the cameras relate: which pairs see the
same floor at the same moment, and which pairs a person walks between. Until
now an operator had to supply both -- four floor points per camera for the
first, drawn areas and a guessed gap for the second. On the COE corridor that
produced nothing: the floor points were clicked up a door post (and reported a
perfect fit), no areas were ever saved, and the fisheye lenses there bend the
floor enough that a four-point homography would have been poor even if the
clicks had been right.

The cameras already hold the answer. Every person who walks through is a
measurement of the layout, and there are hundreds a day:

  * VIEW MAPS (cameras that overlap). When two cameras each watch one moving
    person at the same instant, that is very likely one person, and the two
    foot points are one floor point seen twice. Enough of those and we know,
    for any foot position in camera A, where the same person stands in camera
    B. No model of the lens, the floor, or the mounting is assumed: the map is
    a table of observed correspondences, so it is exactly as curved as the
    optics are.

  * TRANSITIONS (cameras that do not). People who leave camera A near one
    spot keep turning up in camera B near another, a consistent few seconds
    later. Correlating every departure with every later arrival makes the
    real routes stand out as a sharp peak in the delay, over a flat background
    of coincidences. This is the classic result of Makris, Ellis and Black,
    "Bridging the gaps between cameras" (CVPR 2004).

WHY WRONG GUESSES DO NOT POISON IT
----------------------------------
Nothing here is told which pairings are true, so some samples are wrong: two
different people who happened to be walking at once. What separates them is
that the truth repeats and coincidence does not. A real correspondence lands
on the same spot every time anyone walks there; a false one lands wherever the
other stranger happened to be.

So every read is a consensus, never a single sample: a view map answers with
the MEDIAN of nearby samples, refuses when they disagree, and refuses unless
they come from several separate walks; a transition needs a peak several times
above chance, built from several separate episodes.

That also makes it self-healing. Samples are kept per small cell of the frame
and the oldest fall out, so a camera that gets re-aimed first goes quiet
(old and new samples disagree) and then relearns.

Pure data structures: no camera, model, database or clock in here. The handoff
engine owns the instance and its lock.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict, deque
from typing import Deque, Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Point = Tuple[float, float]

FORMAT_VERSION = 1


# ── Overlapping views ────────────────────────────────────────────────────────

class ViewMap:
    """Where a foot seen at (u, v) in one camera appears in another. Directed."""

    def __init__(
        self,
        grid: int = 32,
        per_cell: int = 16,
        per_walk: int = 2,
        k: int = 7,
        k_min: int = 4,
        max_radius: float = 0.06,
        max_spread: float = 0.03,
        min_walks: int = 2,
        ready_walks: int = 5,
        ready_samples: int = 60,
        ready_cells: int = 6,
        ready_agreement: float = 0.6,
    ):
        """
        Args:
            grid: cells per axis over the normalised frame.
            per_cell: samples kept per cell; the oldest falls out.
            per_walk: samples one walk may leave in one cell, so somebody
                standing still cannot become the consensus by repetition.
            k / k_min: neighbours consulted, and the fewest that count.
            max_radius: how far (normalised) a neighbour may be from the query.
            max_spread: neighbours that disagree by more than this are not a
                correspondence, they are coincidences sharing a cell.
            min_walks: separate walks a single answer must rest on.
            ready_walks / ready_samples / ready_cells: evidence before the map
                is used at all.
            ready_agreement: of the places where separate walks can be
                compared, the share where they must AGREE. This is the test
                that cannot be passed by volume. Measured on eight minutes of
                COE traffic: the two corridor cameras that really face each
                other agree in 64-79% of comparable places and improve with
                every walk; a pair that merely had people walking at the same
                time agrees in about half, by chance, and gets worse.
        """
        self.grid = int(grid)
        self.per_cell = int(per_cell)
        self.per_walk = int(per_walk)
        self.k = int(k)
        self.k_min = int(k_min)
        self.max_radius = float(max_radius)
        self.max_spread = float(max_spread)
        self.min_walks = int(min_walks)
        self.ready_walks = int(ready_walks)
        self.ready_samples = int(ready_samples)
        self.ready_cells = int(ready_cells)
        self.ready_agreement = float(ready_agreement)

        # cell index -> rows of [src_u, src_v, dst_u, dst_v, walk, dst_height]
        self._cells: Dict[int, List[List[float]]] = {}
        self._walk_ids: "OrderedDict[Hashable, int]" = OrderedDict()
        self._next_walk = 1
        self._samples = 0
        self._walk_count_cache: Optional[int] = None
        self._agreeing_cells = 0
        self._comparable_cells = 0
        self._adds_since_check = 0
        self._checked = False

    # -- writing --

    def _cell(self, u: float, v: float) -> Tuple[int, int]:
        g = self.grid
        return min(g - 1, max(0, int(u * g))), min(g - 1, max(0, int(v * g)))

    def _walk_id(self, walk: Hashable) -> int:
        wid = self._walk_ids.get(walk)
        if wid is None:
            wid = self._next_walk
            self._next_walk += 1
            self._walk_ids[walk] = wid
            while len(self._walk_ids) > 512:
                self._walk_ids.popitem(last=False)
        return wid

    def add(self, src: Point, dst: Point, walk: Hashable, dst_height: float = 0.0) -> bool:
        """Record that `src` here and `dst` there were one moving person.

        `walk` identifies the pair of tracks the sample came from. Returns
        False when this walk has already said enough about this cell.

        `dst_height` is how tall the person stood in the other camera. Kept
        because it is half the answer: a place in a picture implies a size, so
        "a small figure at the far end" and "someone at a bench in front of
        it" can share a foot point and still be told apart.
        """
        cu, cv = self._cell(src[0], src[1])
        wid = self._walk_id(walk)
        rows = self._cells.setdefault(cv * self.grid + cu, [])
        if sum(1 for r in rows if r[4] == wid) >= self.per_walk:
            return False
        rows.append([float(src[0]), float(src[1]), float(dst[0]), float(dst[1]),
                     float(wid), float(dst_height)])
        self._samples += 1
        if len(rows) > self.per_cell:
            rows.pop(0)
            self._samples -= 1
        self._walk_count_cache = None
        self._adds_since_check += 1
        return True

    # -- reading --

    @property
    def samples(self) -> int:
        return self._samples

    @property
    def walks(self) -> int:
        if self._walk_count_cache is None:
            self._walk_count_cache = len({r[4] for rows in self._cells.values() for r in rows})
        return self._walk_count_cache

    def _survey(self) -> None:
        """Count the cells where separate walks can be compared, and where
        they agree. Recomputed every so often, not on every sample."""
        if self._checked and self._adds_since_check < 20:
            return
        self._checked = True
        self._adds_since_check = 0
        comparable = agreeing = 0
        for rows in self._cells.values():
            if not rows:
                continue
            arr = np.asarray(rows, dtype=np.float64)
            found = self._neighbours((float(arr[:, 0].mean()), float(arr[:, 1].mean())))
            if found is None:
                continue
            comparable += 1
            if self._answer(found) is not None:
                agreeing += 1
        self._comparable_cells, self._agreeing_cells = comparable, agreeing

    @property
    def agreeing_cells(self) -> int:
        """Cells where several separate walks point at the same place."""
        self._survey()
        return self._agreeing_cells

    @property
    def agreement(self) -> float:
        """Of the cells where walks can be compared, the share where they agree."""
        self._survey()
        return self._agreeing_cells / self._comparable_cells if self._comparable_cells else 0.0

    @property
    def ready(self) -> bool:
        return (
            self._samples >= self.ready_samples
            and self.walks >= self.ready_walks
            and self.agreeing_cells >= self.ready_cells
            and self.agreement >= self.ready_agreement
        )

    def predict(self, src: Point) -> Optional[Tuple[float, float, float, int, float]]:
        """(u, v, spread, n, height) in the other camera, or None where nothing
        is known. `height` is 0.0 when no sample recorded one.

        None is the common answer and a correct one: the map only covers
        places where people have actually been seen by both cameras.
        """
        if not self.ready:
            return None
        return self._consensus(src)

    def _consensus(self, src: Point) -> Optional[Tuple[float, float, float, int, float]]:
        near = self._neighbours(src)
        return self._answer(near) if near is not None else None

    def _neighbours(self, src: Point) -> Optional[np.ndarray]:
        """Samples near `src` from enough separate walks to compare, or None."""
        cu, cv = self._cell(src[0], src[1])
        g = self.grid
        rows: List[List[float]] = []
        for dv in (-1, 0, 1):
            y = cv + dv
            if not 0 <= y < g:
                continue
            for du in (-1, 0, 1):
                x = cu + du
                if 0 <= x < g:
                    cell = self._cells.get(y * g + x)
                    if cell:
                        rows.extend(cell)
        if len(rows) < self.k_min:
            return None
        arr = np.asarray(rows, dtype=np.float64)
        dist = np.hypot(arr[:, 0] - src[0], arr[:, 1] - src[1])
        order = np.argsort(dist)[: self.k]
        order = order[dist[order] <= self.max_radius]
        if len(order) < self.k_min:
            return None
        near = arr[order]
        if len(set(near[:, 4].tolist())) < self.min_walks:
            return None
        return near

    def _answer(self, near: np.ndarray) -> Optional[Tuple[float, float, float, int, float]]:
        """Where those samples say the person is, or None if they disagree."""
        u = float(np.median(near[:, 2]))
        v = float(np.median(near[:, 3]))
        spread = 1.4826 * float(np.median(np.hypot(near[:, 2] - u, near[:, 3] - v)))
        if spread > self.max_spread:
            return None
        heights = near[:, 5][near[:, 5] > 0]
        height = float(np.median(heights)) if len(heights) else 0.0
        return u, v, spread, int(len(near)), height

    # -- persistence --

    def to_dict(self) -> dict:
        return {
            "next_walk": self._next_walk,
            "cells": {
                str(index): [[round(r[0], 4), round(r[1], 4), round(r[2], 4), round(r[3], 4),
                              int(r[4]), round(r[5], 4)] for r in rows]
                for index, rows in self._cells.items() if rows
            },
        }

    def load_dict(self, data: dict) -> None:
        self._cells = {}
        self._samples = 0
        limit = self.grid * self.grid
        for index, rows in (data.get("cells") or {}).items():
            index = int(index)
            if not 0 <= index < limit:
                continue
            kept = [
                [float(c) for c in row[:5]] + [float(row[5]) if len(row) > 5 else 0.0]
                for row in rows if len(row) >= 5
            ][-self.per_cell:]
            if kept:
                self._cells[index] = kept
                self._samples += len(kept)
        self._next_walk = max(int(data.get("next_walk") or 1), 1 + int(max(
            (r[4] for rows in self._cells.values() for r in rows), default=0)))
        self._walk_ids.clear()
        self._walk_count_cache = None
        self._checked = False


# ── Sequential views ─────────────────────────────────────────────────────────

class TransitionModel:
    """Departures from one camera that turn into arrivals on another. Directed."""

    def __init__(
        self,
        max_samples: int = 800,
        radius: float = 0.15,
        peak_width: float = 3.0,
        span: Tuple[float, float] = (-2.0, 30.0),
        min_count: int = 6,
        min_episodes: int = 4,
        lift: float = 4.0,
        dominance: float = 0.5,
        episode_gap: float = 8.0,
    ):
        """
        Args:
            radius: how close (normalised) a past exit or entry must be to the
                one being asked about to count as the same doorway.
            peak_width: seconds; real walks between two spots agree this well.
            span: delays considered at all. Slightly negative, because two
                cameras' streams do not arrive equally late.
            min_count / min_episodes: votes in the peak, and separate
                occasions they must come from -- one group of five walking
                through together is one piece of evidence, not five.
            lift: how many times above chance the peak must stand.
            dominance: share of ALL votes between these two spots that must
                fall in the peak. A real route is how people get from there to
                here; a camera whose tracker stutters produces peaks by sheer
                volume, but never a majority.
        """
        self.radius = float(radius)
        self.peak_width = float(peak_width)
        self.span = (float(span[0]), float(span[1]))
        self.min_count = int(min_count)
        self.min_episodes = int(min_episodes)
        self.lift = float(lift)
        self.dominance = float(dominance)
        self.episode_gap = float(episode_gap)
        # rows of (exit_u, exit_v, entry_u, entry_v, gap, when)
        self._votes: Deque[Tuple[float, float, float, float, float, float]] = deque(maxlen=int(max_samples))
        self._cache: Dict[Tuple[int, int, int, int], Optional[Tuple[float, float, int]]] = {}

    def vote(self, exit_pt: Point, entry_pt: Point, gap: float, when: float) -> None:
        if not self.span[0] <= gap <= self.span[1]:
            return
        self._votes.append((float(exit_pt[0]), float(exit_pt[1]),
                            float(entry_pt[0]), float(entry_pt[1]), float(gap), float(when)))
        self._cache.clear()

    @property
    def votes(self) -> int:
        return len(self._votes)

    def window(self, exit_pt: Point, entry_pt: Point) -> Optional[Tuple[float, float, int]]:
        """(shortest, longest, episodes) walk seen between these two spots, or
        None if people leaving there are not known to arrive here."""
        key = (int(exit_pt[0] * 20), int(exit_pt[1] * 20), int(entry_pt[0] * 20), int(entry_pt[1] * 20))
        if key in self._cache:
            return self._cache[key]
        result = self._window(exit_pt, entry_pt)
        if len(self._cache) > 2048:
            self._cache.clear()
        self._cache[key] = result
        return result

    def _window(self, exit_pt: Point, entry_pt: Point) -> Optional[Tuple[float, float, int]]:
        if len(self._votes) < self.min_count:
            return None
        arr = np.asarray(self._votes, dtype=np.float64)
        near = arr[
            (np.hypot(arr[:, 0] - exit_pt[0], arr[:, 1] - exit_pt[1]) <= self.radius)
            & (np.hypot(arr[:, 2] - entry_pt[0], arr[:, 3] - entry_pt[1]) <= self.radius)
        ]
        if len(near) < self.min_count:
            return None
        near = near[np.argsort(near[:, 4])]
        gaps = near[:, 4]
        # Densest stretch of `peak_width` seconds.
        ends = np.searchsorted(gaps, gaps + self.peak_width, side="right")
        counts = ends - np.arange(len(gaps))
        start = int(np.argmax(counts))
        peak = near[start: int(ends[start])]
        chance = len(near) * self.peak_width / (self.span[1] - self.span[0])
        if (
            len(peak) < self.min_count
            or len(peak) < self.lift * chance
            or len(peak) < self.dominance * len(near)
        ):
            return None
        when = np.sort(peak[:, 5])
        episodes = 1 + int(np.sum(np.diff(when) > self.episode_gap))
        if episodes < self.min_episodes:
            return None
        # The walk itself, not the whole stretch searched: a stray coincidence
        # inside the three seconds must not widen what counts as on time.
        low, high = np.percentile(peak[:, 4], [10, 90])
        return float(low), float(high), episodes

    def routes(self) -> List[dict]:
        """Distinct learned routes, for display. Coarse on purpose."""
        if len(self._votes) < self.min_count:
            return []
        out: List[dict] = []
        seen: List[Tuple[float, float, float, float]] = []
        for row in list(self._votes)[::-1]:
            exit_pt, entry_pt = (row[0], row[1]), (row[2], row[3])
            if any(np.hypot(exit_pt[0] - s[0], exit_pt[1] - s[1]) <= self.radius
                   and np.hypot(entry_pt[0] - s[2], entry_pt[1] - s[3]) <= self.radius for s in seen):
                continue
            seen.append((row[0], row[1], row[2], row[3]))
            found = self.window(exit_pt, entry_pt)
            if found is not None:
                out.append({
                    "exit": [round(row[0], 2), round(row[1], 2)],
                    "entry": [round(row[2], 2), round(row[3], 2)],
                    "min_sec": round(found[0], 1),
                    "max_sec": round(found[1], 1),
                    "episodes": found[2],
                })
        return out

    def to_dict(self) -> dict:
        return {"votes": [[round(v, 4) if i < 4 else round(v, 2) for i, v in enumerate(row)]
                          for row in self._votes]}

    def load_dict(self, data: dict) -> None:
        self._votes.clear()
        self._cache.clear()
        for row in data.get("votes") or []:
            if len(row) >= 6:
                self._votes.append(tuple(float(c) for c in row[:6]))


# ── The whole network ────────────────────────────────────────────────────────

class Topology:
    """Every learned relation between cameras. Not thread-safe by itself."""

    def __init__(self) -> None:
        self._views: Dict[Tuple[str, str], ViewMap] = {}
        self._transitions: Dict[Tuple[str, str], TransitionModel] = {}
        self.dirty = False

    def view(self, src: str, dst: str) -> ViewMap:
        found = self._views.get((src, dst))
        if found is None:
            found = self._views[(src, dst)] = ViewMap()
        return found

    def view_if_ready(self, src: str, dst: str) -> Optional[ViewMap]:
        found = self._views.get((src, dst))
        return found if found is not None and found.ready else None

    def transition(self, src: str, dst: str) -> TransitionModel:
        found = self._transitions.get((src, dst))
        if found is None:
            found = self._transitions[(src, dst)] = TransitionModel()
        return found

    def transition_if_any(self, src: str, dst: str) -> Optional[TransitionModel]:
        return self._transitions.get((src, dst))

    def add_correspondence(self, cam_a: str, foot_a: Point, cam_b: str, foot_b: Point,
                           walk: Hashable, height_a: float = 0.0, height_b: float = 0.0) -> None:
        """One moving person seen by two cameras at once. Learned both ways."""
        added = self.view(cam_a, cam_b).add(foot_a, foot_b, walk, height_b)
        added = self.view(cam_b, cam_a).add(foot_b, foot_a, walk, height_a) or added
        self.dirty = self.dirty or added

    def add_transition(self, src: str, exit_pt: Point, dst: str, entry_pt: Point,
                       gap: float, when: float) -> None:
        self.transition(src, dst).vote(exit_pt, entry_pt, gap, when)
        self.dirty = True

    def forget(self, camera_ids: Iterable[str]) -> int:
        """Drop everything learned about these cameras (after one is re-aimed)."""
        wanted = set(camera_ids)
        dropped = 0
        for table in (self._views, self._transitions):
            for key in [k for k in table if k[0] in wanted or k[1] in wanted]:
                del table[key]
                dropped += 1
        self.dirty = self.dirty or bool(dropped)
        return dropped

    def describe(self, camera_ids: Optional[Iterable[str]] = None) -> dict:
        """What has been learned, in terms an operator can check against the building."""
        wanted = set(camera_ids) if camera_ids is not None else None

        def keep(pair: Tuple[str, str]) -> bool:
            return wanted is None or (pair[0] in wanted and pair[1] in wanted)

        overlaps = []
        for (a, b), view in sorted(self._views.items()):
            if a < b and keep((a, b)) and view.samples:
                back = self.view(b, a)
                overlaps.append({
                    "a": a, "b": b,
                    "ready": view.ready and back.ready,
                    "walks": view.walks,
                    "samples": view.samples,
                    "walks_needed": view.ready_walks,
                    "agreement": round(min(view.agreement, back.agreement), 2),
                    "agreement_needed": view.ready_agreement,
                })
        routes = []
        for (a, b), model in sorted(self._transitions.items()):
            if not keep((a, b)):
                continue
            for route in model.routes():
                routes.append({"from": a, "to": b, **route})
        return {"overlaps": overlaps, "routes": routes}

    # -- persistence --

    def to_dict(self) -> dict:
        return {
            "version": FORMAT_VERSION,
            "views": [{"src": a, "dst": b, **view.to_dict()}
                      for (a, b), view in self._views.items() if view.samples],
            "transitions": [{"src": a, "dst": b, **model.to_dict()}
                            for (a, b), model in self._transitions.items() if model.votes],
        }

    def load_dict(self, data: dict) -> None:
        if not isinstance(data, dict) or data.get("version") != FORMAT_VERSION:
            return
        for raw in data.get("views") or []:
            self.view(str(raw["src"]), str(raw["dst"])).load_dict(raw)
        for raw in data.get("transitions") or []:
            self.transition(str(raw["src"]), str(raw["dst"])).load_dict(raw)
        self.dirty = False

    def save(self, path: str) -> bool:
        """Atomic write: a crash mid-save must not cost what took days to learn."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, separators=(",", ":"))
            os.replace(tmp, path)
            self.dirty = False
            return True
        except OSError as e:
            logger.warning(f"[topology] could not save to {path}: {e}")
            return False

    def load(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.load_dict(json.load(f))
            return True
        except FileNotFoundError:
            return False
        except (OSError, ValueError, KeyError, TypeError) as e:
            # A damaged file means relearning, which takes a day of ordinary
            # traffic. It must not mean handoff fails to start.
            logger.warning(f"[topology] ignoring unreadable {path}: {e}")
            return False
