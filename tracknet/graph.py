"""Camera topology: directed graph with travel-time windows and BFS utilities."""
from __future__ import annotations

from collections import deque

from . import config


class CameraGraph:
    def __init__(self, edges: dict[tuple[str, str], tuple[float, float]] | None = None,
                 cameras: dict[str, dict] | None = None):
        self.edges = dict(config.EDGES if edges is None else edges)
        self.cameras = dict(config.CAMERAS if cameras is None else cameras)
        self.forward: dict[str, dict[str, tuple[float, float]]] = {c: {} for c in self.cameras}
        self.backward: dict[str, dict[str, tuple[float, float]]] = {c: {} for c in self.cameras}
        for (src, dst), window in self.edges.items():
            lo, hi = window
            if lo < 0 or hi < lo:
                raise ValueError(f"Invalid travel window for {src}->{dst}: {window}")
            for c in (src, dst):
                self.forward.setdefault(c, {})
                self.backward.setdefault(c, {})
            self.forward[src][dst] = (lo, hi)
            self.backward[dst][src] = (lo, hi)

    @property
    def nodes(self) -> list[str]:
        return sorted(self.forward)

    def _adj(self, direction: str) -> dict[str, dict[str, tuple[float, float]]]:
        if direction == "forward":
            return self.forward
        if direction == "backward":
            return self.backward
        raise ValueError("direction must be 'forward' or 'backward'")

    def neighbors(self, camera: str, direction: str) -> list[str]:
        return sorted(self._adj(direction).get(camera, {}))

    def levels(self, start: str, direction: str, max_hops: int) -> dict[str, int]:
        """Breadth-first search: camera -> number of hops from `start` (start itself = 0)."""
        adj = self._adj(direction)
        if start not in adj:
            return {}
        seen = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if seen[node] >= max_hops:
                continue
            for nb in adj.get(node, {}):
                if nb not in seen:
                    seen[nb] = seen[node] + 1
                    queue.append(nb)
        return seen

    def bidirectional_levels(self, start: str, max_hops: int) -> dict[str, dict[str, int]]:
        """Expand from `start` both forward (where it may go) and backward (where it may have come from)."""
        return {"forward": self.levels(start, "forward", max_hops),
                "backward": self.levels(start, "backward", max_hops)}

    def shortest_path(self, src: str, dst: str) -> list[str] | None:
        """Bidirectional BFS: search forward from src and backward from dst until the frontiers meet."""
        if src not in self.forward or dst not in self.forward:
            return None
        if src == dst:
            return [src]
        dist_f, dist_b = {src: 0}, {dst: 0}
        par_f: dict[str, str | None] = {src: None}
        par_b: dict[str, str | None] = {dst: None}
        front_f, front_b = [src], [dst]

        while front_f and front_b:
            expand_forward = len(front_f) <= len(front_b)
            frontier = front_f if expand_forward else front_b
            dist, par = (dist_f, par_f) if expand_forward else (dist_b, par_b)
            other = dist_b if expand_forward else dist_f
            adj = self.forward if expand_forward else self.backward

            nxt: list[str] = []
            meetings: list[str] = []
            for node in frontier:
                for nb in adj.get(node, {}):
                    if nb in dist:
                        continue
                    dist[nb] = dist[node] + 1
                    par[nb] = node
                    nxt.append(nb)
                    if nb in other:
                        meetings.append(nb)
            if meetings:  # finish the whole layer, then take the meeting node with the shortest total length
                meet = min(meetings, key=lambda n: dist_f.get(n, 0) + dist_b.get(n, 0))
                left, n = [], meet
                while n is not None:
                    left.append(n)
                    n = par_f.get(n)
                left.reverse()
                right, n = [], par_b.get(meet)
                while n is not None:
                    right.append(n)
                    n = par_b.get(n)
                return left + right
            if expand_forward:
                front_f = nxt
            else:
                front_b = nxt
        return None

    def hops(self, src: str, dst: str) -> int | None:
        path = self.shortest_path(src, dst)
        return None if path is None else len(path) - 1

    def transit_window(self, src: str, dst: str) -> tuple[float, float] | None:
        """(min, max) seconds to get from src to dst along the shortest path."""
        path = self.shortest_path(src, dst)
        if path is None:
            return None
        lo = hi = 0.0
        for a, b in zip(path, path[1:]):
            w = self.forward[a][b]
            lo, hi = lo + w[0], hi + w[1]
        return lo, hi
