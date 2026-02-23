from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Dict, List, Optional, Tuple

MAP_WIDTH = 960
MAP_HEIGHT = 560
CELL_SIZE = 20

SIM_STEP_MS = 50
HISTORY_INTERVAL_MS = 1000
ASSIGN_INTERVAL_MS = 250

SITE_CANDIDATES = [
    {"id": "S1", "name": "物流中心", "x": 120.0, "y": 282.0},
    {"id": "S2", "name": "北区配送点", "x": 420.0, "y": 98.0},
    {"id": "S3", "name": "东区配送点", "x": 844.0, "y": 178.0},
    {"id": "S4", "name": "南区配送点", "x": 722.0, "y": 478.0},
    {"id": "S5", "name": "西南配送点", "x": 250.0, "y": 500.0},
    {"id": "S6", "name": "中东配送点", "x": 620.0, "y": 120.0},
    {"id": "S7", "name": "东北配送点", "x": 860.0, "y": 90.0},
    {"id": "S8", "name": "中南配送点", "x": 520.0, "y": 522.0},
    {"id": "S9", "name": "西北配送点", "x": 220.0, "y": 110.0},
]

MIN_SITE_COUNT = 3
MAX_SITE_COUNT = len(SITE_CANDIDATES) - 1
DEFAULT_SITE_COUNT = 3

OBSTACLES = [
    {"x": 298.0, "y": 218.0, "w": 188.0, "h": 98.0},
    {"x": 580.0, "y": 304.0, "w": 142.0, "h": 168.0},
    {"x": 444.0, "y": 394.0, "w": 116.0, "h": 84.0},
]

DRONE_START_SITE_INDEX = [0, 0]


def normalize_site_count(site_count: int) -> int:
    return max(MIN_SITE_COUNT, min(MAX_SITE_COUNT, int(site_count)))


def build_active_sites(site_count: int) -> List[dict]:
    active_count = normalize_site_count(site_count)
    return [dict(site) for site in SITE_CANDIDATES[: active_count + 1]]


def clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_in_rect(x: float, y: float, rect: dict, margin: float = 0.0) -> bool:
    return (
        x >= rect["x"] - margin
        and x <= rect["x"] + rect["w"] + margin
        and y >= rect["y"] - margin
        and y <= rect["y"] + rect["h"] + margin
    )


class Mulberry32:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        x = self.state
        x = ((x ^ (x >> 15)) * (x | 1)) & 0xFFFFFFFF
        x ^= (x + (((x ^ (x >> 7)) * (x | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((x ^ (x >> 14)) & 0xFFFFFFFF) / 4294967296.0


class GridPathPlanner:
    def __init__(self, width: int, height: int, cell_size: int, obstacles: List[dict], sites: List[dict]):
        self.width = width
        self.height = height
        self.cell = cell_size
        self.cols = width // cell_size
        self.rows = height // cell_size
        self.obstacles = obstacles
        self.sites = sites
        self.blocked = [0] * (self.cols * self.rows)
        self.path_cache: Dict[Tuple[int, int], Tuple[List[Tuple[int, int]], float]] = {}
        self._build_blocked_grid()

    def _build_blocked_grid(self) -> None:
        margin = 4.0
        for cy in range(self.rows):
            for cx in range(self.cols):
                wx, wy = self.to_world((cx, cy))
                blocked = any(point_in_rect(wx, wy, o, margin) for o in self.obstacles)
                self.blocked[self.to_index(cx, cy)] = 1 if blocked else 0

        for site in self.sites:
            cx, cy = self.to_cell((site["x"], site["y"]))
            self.blocked[self.to_index(cx, cy)] = 0

    def to_index(self, cx: int, cy: int) -> int:
        return cy * self.cols + cx

    def from_index(self, idx: int) -> Tuple[int, int]:
        return idx % self.cols, idx // self.cols

    def in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.cols and 0 <= cy < self.rows

    def to_cell(self, point: Tuple[float, float]) -> Tuple[int, int]:
        return (
            int(clamp(math.floor(point[0] / self.cell), 0, self.cols - 1)),
            int(clamp(math.floor(point[1] / self.cell), 0, self.rows - 1)),
        )

    def to_world(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        return (cell[0] * self.cell + self.cell * 0.5, cell[1] * self.cell + self.cell * 0.5)

    def is_blocked(self, cx: int, cy: int) -> bool:
        if not self.in_bounds(cx, cy):
            return True
        return self.blocked[self.to_index(cx, cy)] == 1

    def nearest_free_cell(self, cell: Tuple[int, int]) -> Tuple[int, int]:
        if not self.is_blocked(*cell):
            return cell

        queue: List[Tuple[int, int]] = [cell]
        visited = {cell}
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        while queue:
            cur = queue.pop(0)
            if not self.is_blocked(*cur):
                return cur
            for dx, dy in dirs:
                nxt = (cur[0] + dx, cur[1] + dy)
                if self.in_bounds(*nxt) and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return cell

    def _a_star_cells(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        start_idx = self.to_index(*start)
        goal_idx = self.to_index(*goal)
        if start_idx == goal_idx:
            return [start]

        size = self.cols * self.rows
        g_score = [math.inf] * size
        came_from = [-1] * size
        closed = [0] * size

        def heuristic(idx: int) -> float:
            cx, cy = self.from_index(idx)
            return math.hypot(cx - goal[0], cy - goal[1])

        heap: List[Tuple[float, int, int]] = []
        counter = 0
        g_score[start_idx] = 0.0
        heapq.heappush(heap, (heuristic(start_idx), counter, start_idx))

        dirs = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)),
            (-1, -1, math.sqrt(2)),
        ]

        while heap:
            _, _, cur_idx = heapq.heappop(heap)
            if closed[cur_idx]:
                continue
            if cur_idx == goal_idx:
                cells: List[Tuple[int, int]] = []
                walk = cur_idx
                while walk != -1:
                    cells.append(self.from_index(walk))
                    walk = came_from[walk]
                cells.reverse()
                return cells

            closed[cur_idx] = 1
            ccx, ccy = self.from_index(cur_idx)

            for dx, dy, move_cost in dirs:
                nx, ny = ccx + dx, ccy + dy
                if not self.in_bounds(nx, ny) or self.is_blocked(nx, ny):
                    continue
                if dx != 0 and dy != 0:
                    if self.is_blocked(ccx + dx, ccy) or self.is_blocked(ccx, ccy + dy):
                        continue

                n_idx = self.to_index(nx, ny)
                if closed[n_idx]:
                    continue

                tentative = g_score[cur_idx] + move_cost
                if tentative < g_score[n_idx]:
                    g_score[n_idx] = tentative
                    came_from[n_idx] = cur_idx
                    counter += 1
                    heapq.heappush(heap, (tentative + heuristic(n_idx), counter, n_idx))

        return [start, goal]

    def _cached_cells(self, start_idx: int, goal_idx: int) -> Tuple[List[Tuple[int, int]], float]:
        key = (start_idx, goal_idx)
        if key in self.path_cache:
            return self.path_cache[key]

        rev_key = (goal_idx, start_idx)
        if rev_key in self.path_cache:
            rev_cells, rev_len = self.path_cache[rev_key]
            flipped = (list(reversed(rev_cells)), rev_len)
            self.path_cache[key] = flipped
            return flipped

        start = self.from_index(start_idx)
        goal = self.from_index(goal_idx)
        cells = self._a_star_cells(start, goal)

        length = 0.0
        for i in range(1, len(cells)):
            a = self.to_world(cells[i - 1])
            b = self.to_world(cells[i])
            length += dist(a, b)

        packed = (cells, length)
        self.path_cache[key] = packed
        return packed

    def find_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], float]:
        s_cell = self.nearest_free_cell(self.to_cell(start))
        g_cell = self.nearest_free_cell(self.to_cell(goal))
        cells, _ = self._cached_cells(self.to_index(*s_cell), self.to_index(*g_cell))

        points: List[Tuple[float, float]] = [start]
        if len(cells) >= 2:
            for c in cells[1:-1]:
                points.append(self.to_world(c))
        points.append(goal)

        plen = 0.0
        for i in range(1, len(points)):
            plen += dist(points[i - 1], points[i])
        return points, plen

    def estimate_distance(self, start: Tuple[float, float], goal: Tuple[float, float]) -> float:
        _, plen = self.find_path(start, goal)
        return plen


@dataclass
class Task:
    id: str
    source_index: int
    target_index: int
    created_at_ms: float
    priority: int
    status: str = "pending"
    assigned_at_ms: Optional[float] = None
    loaded_at_ms: Optional[float] = None
    completed_at_ms: Optional[float] = None
    assigned_drone: Optional[int] = None


@dataclass
class Drone:
    id: int
    x: float
    y: float
    heading: float
    speed: float
    phase: str = "idle"  # idle, delivering, returning
    task_id: Optional[str] = None  # current delivery target task id
    cargo_task_ids: List[str] = None
    path: List[Tuple[float, float]] = None
    path_index: int = 0
    completed_count: int = 0
    total_distance: float = 0.0
    active_batch_id: Optional[str] = None
    batch_loaded_at_ms: Optional[float] = None
    batch_route_est_distance: float = 0.0
    batch_size: int = 0

    def __post_init__(self) -> None:
        if self.path is None:
            self.path = []
        if self.cargo_task_ids is None:
            self.cargo_task_ids = []


class DroneSimulation:
    def __init__(
        self,
        name: str,
        policy: str,
        planner: GridPathPlanner,
        rng: Mulberry32,
        payload_capacity: int,
        sites: List[dict],
    ):
        self.name = name
        self.policy = policy
        self.planner = planner
        self.rng = rng
        self.payload_capacity = max(1, int(payload_capacity))
        self.sites = sites
        self.reset()

    def set_payload_capacity(self, payload_capacity: int) -> None:
        self.payload_capacity = max(1, int(payload_capacity))

    def reset(self) -> None:
        self.time_ms = 0.0
        self.assign_acc_ms = 0.0
        self.history_acc_ms = 0.0

        self.tasks: List[Task] = []
        self.task_map: Dict[str, Task] = {}

        self.drones: List[Drone] = []
        for idx, site_idx in enumerate(DRONE_START_SITE_INDEX):
            site = self.sites[site_idx]
            self.drones.append(
                Drone(
                    id=idx,
                    x=site["x"] + (-8.0 if idx == 0 else 8.0),
                    y=site["y"] + (-12.0 if idx == 0 else 12.0),
                    heading=0.0,
                    speed=102.0 if idx == 0 else 96.0,
                )
            )

        self.stats = {
            "generated": 0,
            "completed": 0,
            "total_completion_ms": 0.0,
            "total_wait_ms": 0.0,
            "loaded_count": 0,
            "total_distance": 0.0,
            "batch_count": 0,
            "total_loaded_tasks": 0,
            "total_batch_route_est_distance": 0.0,
        }
        self.history: List[dict] = []
        self.decision_log: List[dict] = []
        self.event_log: List[dict] = []
        self.batch_results: List[dict] = []
        self.phase_trace: List[dict] = []
        self.batch_counter = 1

    def _center_point(self) -> Tuple[float, float]:
        center = self.sites[0]
        return center["x"], center["y"]

    def _is_at_center(self, drone: Drone) -> bool:
        cx, cy = self._center_point()
        return dist((drone.x, drone.y), (cx, cy)) <= 12.0

    def _has_active_path(self, drone: Drone) -> bool:
        return len(drone.path) >= 2 and drone.path_index < len(drone.path)

    def _append_decision(self, payload: dict) -> None:
        self.decision_log.append(payload)
        if len(self.decision_log) > 80:
            self.decision_log.pop(0)

    def _append_event(self, payload: dict) -> None:
        self.event_log.append(payload)
        if len(self.event_log) > 140:
            self.event_log.pop(0)

    def _metrics_brief(self) -> dict:
        return {
            "pending": len(self.get_pending_tasks()),
            "completion_rate": round(self.get_completion_rate(), 4),
            "avg_wait_s": round(self.get_avg_wait_ms() / 1000.0, 3),
        }

    def enqueue_task(self, template: dict) -> None:
        task = Task(
            id=template["id"],
            source_index=template["source_index"],
            target_index=template["target_index"],
            created_at_ms=template["created_at_ms"],
            priority=template["priority"],
        )
        self.tasks.append(task)
        self.task_map[task.id] = task
        self.stats["generated"] += 1

    def get_task(self, task_id: Optional[str]) -> Optional[Task]:
        if task_id is None:
            return None
        return self.task_map.get(task_id)

    def get_pending_tasks(self) -> List[Task]:
        return [t for t in self.tasks if t.status == "pending"]

    def get_open_tasks(self) -> List[Task]:
        return [t for t in self.tasks if t.status != "completed"]

    def step(self, dt_ms: float) -> None:
        self.time_ms += dt_ms
        self.assign_acc_ms += dt_ms
        self.history_acc_ms += dt_ms

        while self.assign_acc_ms >= ASSIGN_INTERVAL_MS:
            self.assign_acc_ms -= ASSIGN_INTERVAL_MS
            self.assign_tasks()

        dt_sec = dt_ms / 1000.0
        for drone in self.drones:
            self._move_drone(drone, dt_sec)

        while self.history_acc_ms >= HISTORY_INTERVAL_MS:
            self.history_acc_ms -= HISTORY_INTERVAL_MS
            t_s = round(self.time_ms / 1000.0, 2)
            phases = [d.phase for d in self.drones]
            self.history.append(
                {
                    "t": t_s,
                    "completed": self.stats["completed"],
                    "avg_wait": round(self.get_avg_wait_ms() / 1000.0, 4),
                    "avg_completion": round(self.get_avg_completion_ms() / 1000.0, 4),
                    "distance": round(self.stats["total_distance"], 2),
                    "pending": len(self.get_pending_tasks()),
                    "phases": phases,
                }
            )
            if len(self.history) > 360:
                self.history.pop(0)
            self.phase_trace.append({"t": t_s, "phases": phases})
            if len(self.phase_trace) > 420:
                self.phase_trace.pop(0)

    def assign_tasks(self) -> None:
        # Keep idle drones returning to center for the next loading cycle.
        for drone in self.drones:
            if drone.phase == "idle" and drone.task_id is None and not self._has_active_path(drone) and not self._is_at_center(drone):
                self._set_path(drone, self._center_point())
                drone.phase = "returning"
                self._append_event(
                    {
                        "t": round(self.time_ms / 1000.0, 2),
                        "event": "reposition_to_center",
                        "drone_id": drone.id,
                        "metrics": self._metrics_brief(),
                    }
                )

        idle_at_center = [
            d
            for d in self.drones
            if d.phase == "idle" and d.task_id is None and self._is_at_center(d) and not self._has_active_path(d)
        ]
        if not idle_at_center or not self.get_pending_tasks():
            return

        if self.policy == "astar":
            self._assign_batches_astar(idle_at_center)
        else:
            self._assign_batches_astar_plain(idle_at_center)

    def _assign_batches_astar(self, idle_drones: List[Drone]) -> None:
        idle_sorted = sorted(idle_drones, key=lambda d: (d.completed_count, d.id))
        for drone in idle_sorted:
            pending = self.get_pending_tasks()
            if not pending:
                break
            selected, ranking = self._select_tasks_astar(drone, pending, self.payload_capacity)
            self._load_tasks(drone, selected, order_mode="astar", ranking=ranking)

    def _assign_batches_astar_plain(self, idle_drones: List[Drone]) -> None:
        idle_sorted = sorted(idle_drones, key=lambda d: (d.completed_count, d.id))
        for drone in idle_sorted:
            pending = self.get_pending_tasks()
            if not pending:
                break
            selected, ranking = self._select_tasks_astar_plain(pending, self.payload_capacity)
            self._load_tasks(drone, selected, order_mode="fifo", ranking=ranking)

    def _select_tasks_astar(self, drone: Drone, pending: List[Task], cap: int) -> Tuple[List[Task], List[dict]]:
        center = self._center_point()
        scored: List[Tuple[float, Task, dict]] = []
        for task in pending:
            dst = self.sites[task.target_index]
            delivery = self.planner.estimate_distance(center, (dst["x"], dst["y"]))
            age = self.time_ms - task.created_at_ms
            priority_bonus = 180.0 * task.priority
            score = delivery - age * 0.020 - priority_bonus
            scored.append(
                (
                    score,
                    task,
                    {
                        "task_id": task.id,
                        "target_index": task.target_index,
                        "score": round(score, 3),
                        "distance_term": round(delivery, 3),
                        "age_term": round(age * 0.020, 3),
                        "priority_term": round(priority_bonus, 3),
                    },
                )
            )

        scored.sort(key=lambda x: x[0])
        selected = [task for _, task, _ in scored[:cap]]
        ranking = [row for _, _, row in scored[: min(5, len(scored))]]
        return selected, ranking

    def _select_tasks_astar_plain(self, pending: List[Task], cap: int) -> Tuple[List[Task], List[dict]]:
        # Plain A*: FCFS-like dispatch, no greedy scoring optimization.
        ordered = sorted(
            pending,
            key=lambda t: (-t.priority, t.created_at_ms, t.id),
        )
        selected = ordered[:cap]
        ranking = [
            {
                "task_id": t.id,
                "target_index": t.target_index,
                "score": None,
                "mode": "fcfs",
                "age_s": round((self.time_ms - t.created_at_ms) / 1000.0, 3),
                "priority": t.priority,
            }
            for t in ordered[: min(5, len(ordered))]
        ]
        return selected, ranking

    def _order_delivery_queue(
        self,
        drone: Drone,
        tasks: List[Task],
        mode: str,
        start_point: Optional[Tuple[float, float]] = None,
    ) -> List[Task]:
        if not tasks:
            return []

        current = start_point if start_point is not None else (drone.x, drone.y)
        remaining = tasks[:]
        ordered: List[Task] = []

        while remaining:
            if mode == "astar":
                best = min(
                    remaining,
                    key=lambda t: self.planner.estimate_distance(
                        current,
                        (self.sites[t.target_index]["x"], self.sites[t.target_index]["y"]),
                    ),
                )
            elif mode == "fifo":
                best = remaining[0]
            else:
                def utility(task: Task) -> float:
                    dst = self.sites[task.target_index]
                    d = self.planner.estimate_distance(current, (dst["x"], dst["y"]))
                    age = (self.time_ms - task.created_at_ms) / 1000.0
                    return age * 1.2 - d / 300.0 + task.priority * 0.9

                best = max(remaining, key=utility)

            ordered.append(best)
            remaining.remove(best)
            dst = self.sites[best.target_index]
            current = (dst["x"], dst["y"])

        return ordered

    def _load_tasks(
        self,
        drone: Drone,
        tasks: List[Task],
        order_mode: str,
        ranking: Optional[List[dict]] = None,
    ) -> None:
        if not tasks:
            return

        center = self._center_point()
        ordered = self._order_delivery_queue(drone, tasks, order_mode, start_point=center)

        cargo_ids: List[str] = []
        for task in ordered:
            if task.status != "pending":
                continue
            task.status = "loaded"
            task.assigned_drone = drone.id
            if task.assigned_at_ms is None:
                task.assigned_at_ms = self.time_ms
            if task.loaded_at_ms is None:
                task.loaded_at_ms = self.time_ms
                self.stats["total_wait_ms"] += task.loaded_at_ms - task.created_at_ms
                self.stats["loaded_count"] += 1
            cargo_ids.append(task.id)

        if not cargo_ids:
            return

        batch_id = f"B{self.batch_counter}"
        self.batch_counter += 1

        # Estimate route distance for the loaded batch to expose behavior-level evidence.
        route_points = [center]
        for task in ordered:
            dst = self.sites[task.target_index]
            route_points.append((dst["x"], dst["y"]))
        batch_route_distance = 0.0
        for i in range(1, len(route_points)):
            batch_route_distance += self.planner.estimate_distance(route_points[i - 1], route_points[i])

        self.stats["batch_count"] += 1
        self.stats["total_loaded_tasks"] += len(cargo_ids)
        self.stats["total_batch_route_est_distance"] += batch_route_distance

        decision_payload = {
            "t": round(self.time_ms / 1000.0, 2),
            "batch_id": batch_id,
            "drone_id": drone.id,
            "policy": self.policy,
            "selected_tasks": cargo_ids,
            "selected_target_indices": [task.target_index for task in ordered],
            "batch_size": len(cargo_ids),
            "batch_route_est_distance": round(batch_route_distance, 3),
            "ranking_top": ranking[:3] if ranking else [],
            "route_points": [{"x": round(p[0], 3), "y": round(p[1], 3)} for p in route_points],
        }
        self._append_decision(decision_payload)

        self._append_event(
            {
                "t": round(self.time_ms / 1000.0, 2),
                "event": "load_batch",
                "batch_id": batch_id,
                "drone_id": drone.id,
                "tasks": cargo_ids,
                "batch_size": len(cargo_ids),
                "batch_route_est_distance": round(batch_route_distance, 3),
                "metrics": self._metrics_brief(),
            }
        )

        drone.cargo_task_ids = cargo_ids
        drone.task_id = cargo_ids[0]
        drone.phase = "delivering"
        drone.active_batch_id = batch_id
        drone.batch_loaded_at_ms = self.time_ms
        drone.batch_route_est_distance = batch_route_distance
        drone.batch_size = len(cargo_ids)

        first_task = self.get_task(drone.task_id)
        if first_task is not None:
            dst = self.sites[first_task.target_index]
            self._set_path(drone, (dst["x"], dst["y"]))

    def _set_path(self, drone: Drone, target: Tuple[float, float]) -> None:
        start = (drone.x, drone.y)
        points, _ = self.planner.find_path(start, target)
        drone.path = points
        drone.path_index = 1 if len(points) > 1 else 0
        if len(points) < 2:
            drone.path = [start, target]
            drone.path_index = 1

    def _move_drone(self, drone: Drone, dt_sec: float) -> None:
        if len(drone.path) < 2 or drone.path_index >= len(drone.path):
            return

        remaining = drone.speed * dt_sec
        while remaining > 0 and drone.path_index < len(drone.path):
            tx, ty = drone.path[drone.path_index]
            dx = tx - drone.x
            dy = ty - drone.y
            seg = math.hypot(dx, dy)

            if seg < 1e-6:
                drone.path_index += 1
                if drone.path_index >= len(drone.path):
                    self._on_drone_path_finished(drone)
                continue

            travel = min(seg, remaining)
            drone.heading = math.atan2(dy, dx)
            drone.x += (dx / seg) * travel
            drone.y += (dy / seg) * travel
            drone.total_distance += travel
            self.stats["total_distance"] += travel
            remaining -= travel

            if travel >= seg - 1e-4:
                drone.x, drone.y = tx, ty
                drone.path_index += 1
                if drone.path_index >= len(drone.path):
                    self._on_drone_path_finished(drone)

    def _on_drone_path_finished(self, drone: Drone) -> None:
        if drone.phase == "delivering":
            finished_task_id = drone.task_id
            current_task = self.get_task(drone.task_id)
            if current_task is not None and current_task.status != "completed":
                current_task.status = "completed"
                current_task.completed_at_ms = self.time_ms
                self.stats["completed"] += 1
                self.stats["total_completion_ms"] += current_task.completed_at_ms - current_task.created_at_ms
                drone.completed_count += 1

                self._append_event(
                    {
                        "t": round(self.time_ms / 1000.0, 2),
                        "event": "deliver_task",
                        "batch_id": drone.active_batch_id,
                        "drone_id": drone.id,
                        "task_id": current_task.id,
                        "target_index": current_task.target_index,
                        "remaining_cargo": max(0, len(drone.cargo_task_ids) - 1),
                        "metrics": self._metrics_brief(),
                    }
                )

            if drone.task_id in drone.cargo_task_ids:
                drone.cargo_task_ids = [tid for tid in drone.cargo_task_ids if tid != drone.task_id]

            if drone.cargo_task_ids:
                drone.task_id = drone.cargo_task_ids[0]
                nxt = self.get_task(drone.task_id)
                if nxt is not None:
                    dst = self.sites[nxt.target_index]
                    self._set_path(drone, (dst["x"], dst["y"]))
                else:
                    drone.phase = "returning"
                    drone.task_id = None
                    self._set_path(drone, self._center_point())
            else:
                drone.phase = "returning"
                drone.task_id = None
                self._set_path(drone, self._center_point())

            if finished_task_id is not None and not drone.cargo_task_ids:
                batch_elapsed_s = 0.0
                if drone.batch_loaded_at_ms is not None:
                    batch_elapsed_s = max(0.0, (self.time_ms - drone.batch_loaded_at_ms) / 1000.0)
                if drone.active_batch_id is not None:
                    self.batch_results.append(
                        {
                            "batch_id": drone.active_batch_id,
                            "drone_id": drone.id,
                            "loaded_t_s": round((drone.batch_loaded_at_ms or self.time_ms) / 1000.0, 3),
                            "completed_t_s": round(self.time_ms / 1000.0, 3),
                            "batch_elapsed_s": round(batch_elapsed_s, 3),
                            "batch_route_est_distance": round(drone.batch_route_est_distance, 3),
                            "batch_size": drone.batch_size,
                        }
                    )
                    if len(self.batch_results) > 160:
                        self.batch_results.pop(0)
                self._append_event(
                    {
                        "t": round(self.time_ms / 1000.0, 2),
                        "event": "batch_completed_returning",
                        "batch_id": drone.active_batch_id,
                        "drone_id": drone.id,
                        "last_task_id": finished_task_id,
                        "batch_elapsed_s": round(batch_elapsed_s, 3),
                        "batch_route_est_distance": round(drone.batch_route_est_distance, 3),
                        "metrics": self._metrics_brief(),
                    }
                )
            return

        if drone.phase == "returning":
            drone.phase = "idle"
            drone.task_id = None
            drone.path = []
            drone.path_index = 0
            drone.cargo_task_ids = []
            drone.active_batch_id = None
            drone.batch_loaded_at_ms = None
            drone.batch_route_est_distance = 0.0
            drone.batch_size = 0
            self._append_event(
                {
                    "t": round(self.time_ms / 1000.0, 2),
                    "event": "arrive_center",
                    "drone_id": drone.id,
                    "metrics": self._metrics_brief(),
                }
            )
            return

        drone.phase = "idle"
        drone.task_id = None
        drone.path = []
        drone.path_index = 0
        drone.cargo_task_ids = []
        drone.active_batch_id = None
        drone.batch_loaded_at_ms = None
        drone.batch_route_est_distance = 0.0
        drone.batch_size = 0

    def get_avg_completion_ms(self) -> float:
        if self.stats["completed"] == 0:
            return 0.0
        return self.stats["total_completion_ms"] / self.stats["completed"]

    def get_avg_wait_ms(self) -> float:
        if self.stats["loaded_count"] == 0:
            return 0.0
        return self.stats["total_wait_ms"] / self.stats["loaded_count"]

    def get_completion_rate(self) -> float:
        if self.stats["generated"] == 0:
            return 0.0
        return self.stats["completed"] / self.stats["generated"]

    def metrics_dict(self) -> dict:
        avg_batch_size = (
            self.stats["total_loaded_tasks"] / self.stats["batch_count"]
            if self.stats["batch_count"] > 0
            else 0.0
        )
        avg_batch_route_est = (
            self.stats["total_batch_route_est_distance"] / self.stats["batch_count"]
            if self.stats["batch_count"] > 0
            else 0.0
        )
        return {
            "generated": self.stats["generated"],
            "completed": self.stats["completed"],
            "pending": len(self.get_pending_tasks()),
            "avg_completion_s": round(self.get_avg_completion_ms() / 1000.0, 4),
            "avg_wait_s": round(self.get_avg_wait_ms() / 1000.0, 4),
            "completion_rate": round(self.get_completion_rate(), 6),
            "total_distance_px": round(self.stats["total_distance"], 3),
            "batch_count": self.stats["batch_count"],
            "avg_batch_size": round(avg_batch_size, 4),
            "avg_batch_route_est_distance": round(avg_batch_route_est, 3),
        }

    def state_dict(self) -> dict:
        open_tasks = []
        for task in self.get_open_tasks():
            open_tasks.append(
                {
                    "id": task.id,
                    "source_index": task.source_index,
                    "target_index": task.target_index,
                    "status": task.status,
                    "priority": task.priority,
                    "age_s": round((self.time_ms - task.created_at_ms) / 1000.0, 3),
                }
            )

        drones = []
        for drone in self.drones:
            path_remaining = [
                {"x": round(p[0], 3), "y": round(p[1], 3)}
                for p in (drone.path[drone.path_index :] if drone.path_index < len(drone.path) else [])
            ]
            drones.append(
                {
                    "id": drone.id,
                    "x": round(drone.x, 3),
                    "y": round(drone.y, 3),
                    "heading_rad": round(drone.heading, 4),
                    "phase": drone.phase,
                    "task_id": drone.task_id,
                    "cargo_count": len(drone.cargo_task_ids),
                    "cargo_task_ids": drone.cargo_task_ids,
                    "active_batch_id": drone.active_batch_id,
                    "path_remaining": path_remaining,
                }
            )

        return {
            "time_s": round(self.time_ms / 1000.0, 3),
            "metrics": self.metrics_dict(),
            "drones": drones,
            "open_tasks": open_tasks,
            "history": self.history,
            "phase_trace": self.phase_trace[-180:],
            "batch_results": self.batch_results[-80:],
            "decision_log": self.decision_log[-18:],
            "event_log": self.event_log[-24:],
        }


class ComparisonSandbox:
    def __init__(
        self,
        seed: int = 42,
        interval_ms: int = 4000,
        speed: float = 1.5,
        payload_capacity: int = 2,
        site_count: int = DEFAULT_SITE_COUNT,
    ):
        self.seed = seed
        self.interval_ms = interval_ms
        self.speed = speed
        self.payload_capacity = max(1, int(payload_capacity))
        self.site_count = normalize_site_count(site_count)
        self.sites = build_active_sites(self.site_count)
        self.running = False
        self.planner = GridPathPlanner(MAP_WIDTH, MAP_HEIGHT, CELL_SIZE, OBSTACLES, self.sites)
        self.reset(
            seed=seed,
            interval_ms=interval_ms,
            payload_capacity=payload_capacity,
            site_count=self.site_count,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        interval_ms: Optional[int] = None,
        payload_capacity: Optional[int] = None,
        site_count: Optional[int] = None,
    ) -> None:
        if seed is not None:
            self.seed = max(1, int(seed))
        if interval_ms is not None:
            self.interval_ms = int(clamp(float(interval_ms), 1000.0, 60000.0))
        if payload_capacity is not None:
            self.payload_capacity = max(1, int(payload_capacity))
        if site_count is not None:
            self.site_count = normalize_site_count(site_count)
        self.sites = build_active_sites(self.site_count)
        self.planner = GridPathPlanner(MAP_WIDTH, MAP_HEIGHT, CELL_SIZE, OBSTACLES, self.sites)

        self.global_time_ms = 0.0
        self.acc_ms = 0.0
        self.task_acc_ms = 0.0
        self.task_counter = 1

        self.rand = Mulberry32(self.seed)
        rng_astar = Mulberry32(self.seed + 101)
        rng_plain = Mulberry32(self.seed + 202)

        self.simulators = {
            "astar": DroneSimulation(
                "A* + 贪心",
                "astar",
                self.planner,
                rng_astar,
                payload_capacity=self.payload_capacity,
                sites=self.sites,
            ),
            "mappo": DroneSimulation(
                "A* 基线",
                "astar_plain",
                self.planner,
                rng_plain,
                payload_capacity=self.payload_capacity,
                sites=self.sites,
            ),
        }

        for _ in range(2):
            self.generate_task()

    def set_speed(self, speed: float) -> None:
        self.speed = clamp(float(speed), 0.5, 5.0)

    def set_interval_ms(self, interval_ms: float) -> None:
        self.interval_ms = int(clamp(float(interval_ms), 1000.0, 60000.0))

    def set_payload_capacity(self, payload_capacity: int) -> None:
        self.payload_capacity = max(1, int(payload_capacity))
        for sim in self.simulators.values():
            sim.set_payload_capacity(self.payload_capacity)

    def set_site_count(self, site_count: int) -> None:
        normalized = normalize_site_count(site_count)
        if normalized == self.site_count:
            return
        self.reset(site_count=normalized)

    def generate_task(self) -> dict:
        source_index = 0
        target_index = 1 + int(self.rand.random() * (len(self.sites) - 1))
        priority = 1 if self.rand.random() > 0.82 else 0

        template = {
            "id": f"T{self.task_counter}",
            "source_index": source_index,
            "target_index": target_index,
            "created_at_ms": self.global_time_ms,
            "priority": priority,
        }
        self.task_counter += 1

        self.simulators["astar"].enqueue_task(template)
        self.simulators["mappo"].enqueue_task(template)
        return template

    def advance_by(self, ms: float) -> None:
        self.acc_ms += ms
        while self.acc_ms >= SIM_STEP_MS:
            self.acc_ms -= SIM_STEP_MS
            self.step(SIM_STEP_MS)

    def step(self, dt_ms: float) -> None:
        self.global_time_ms += dt_ms
        self.task_acc_ms += dt_ms

        while self.task_acc_ms >= self.interval_ms:
            self.task_acc_ms -= self.interval_ms
            self.generate_task()

        self.simulators["astar"].step(dt_ms)
        self.simulators["mappo"].step(dt_ms)

    def metrics_snapshot(self) -> dict:
        return {
            "astar": self.simulators["astar"].metrics_dict(),
            "mappo": self.simulators["mappo"].metrics_dict(),
        }

    def state_dict(self) -> dict:
        return {
            "mode": "running" if self.running else "paused",
            "coordinate_system": "origin=(0,0) at top-left, x right, y down; unit=pixel",
            "time_s": round(self.global_time_ms / 1000.0, 3),
            "controls": {
                "seed": self.seed,
                "task_interval_s": round(self.interval_ms / 1000.0, 3),
                "speed_multiplier": round(self.speed, 3),
                "payload_capacity": self.payload_capacity,
                "site_count": self.site_count,
            },
            "map": {
                "width": MAP_WIDTH,
                "height": MAP_HEIGHT,
                "sites": self.sites,
                "obstacles": OBSTACLES,
            },
            "algorithms": {
                "astar": self.simulators["astar"].state_dict(),
                "mappo": self.simulators["mappo"].state_dict(),
            },
        }
