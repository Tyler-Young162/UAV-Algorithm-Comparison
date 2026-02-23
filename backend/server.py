from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from threading import Lock
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .evaluation import run_evaluation, run_payload_sweep
from .simulation import ComparisonSandbox, DEFAULT_SITE_COUNT, MAX_SITE_COUNT, MIN_SITE_COUNT

ROOT_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="Drone Algorithm Comparison Sandbox")

sandbox = ComparisonSandbox(
    seed=42,
    interval_ms=4000,
    speed=1.5,
    payload_capacity=2,
    site_count=DEFAULT_SITE_COUNT,
)
state_lock = Lock()


class ResetRequest(BaseModel):
    seed: int = Field(default=42, ge=1)
    interval_s: float = Field(default=4.0, ge=1.0, le=20.0)
    speed: float = Field(default=1.5, ge=0.5, le=5.0)
    payload_capacity: int = Field(default=2, ge=1, le=12)
    site_count: int = Field(default=DEFAULT_SITE_COUNT, ge=MIN_SITE_COUNT, le=MAX_SITE_COUNT)


class ConfigRequest(BaseModel):
    interval_s: Optional[float] = Field(default=None, ge=1.0, le=20.0)
    speed: Optional[float] = Field(default=None, ge=0.5, le=5.0)
    payload_capacity: Optional[int] = Field(default=None, ge=1, le=12)
    site_count: Optional[int] = Field(default=None, ge=MIN_SITE_COUNT, le=MAX_SITE_COUNT)


class AdvanceRequest(BaseModel):
    ms: float = Field(default=16.67, gt=0, le=5000)


class EvaluateRequest(BaseModel):
    episodes: int = Field(default=24, ge=4, le=200)
    horizon_s: float = Field(default=120.0, ge=10.0, le=1200.0)
    interval_s: float = Field(default=4.0, ge=1.0, le=20.0)
    base_seed: int = Field(default=100, ge=1, le=1000000)
    payload_capacity: int = Field(default=2, ge=1, le=12)
    site_count: int = Field(default=DEFAULT_SITE_COUNT, ge=MIN_SITE_COUNT, le=MAX_SITE_COUNT)
    payload_values: Optional[List[int]] = None


async def simulation_loop() -> None:
    last = time.perf_counter()
    while True:
        await asyncio.sleep(0.05)
        now = time.perf_counter()
        dt_ms = (now - last) * 1000.0
        last = now

        with state_lock:
            if sandbox.running:
                sandbox.advance_by(dt_ms * sandbox.speed)


@app.on_event("startup")
async def _startup() -> None:
    app.state.sim_task = asyncio.create_task(simulation_loop())


@app.on_event("shutdown")
async def _shutdown() -> None:
    task = getattr(app.state, "sim_task", None)
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/state")
def get_state() -> dict:
    with state_lock:
        return sandbox.state_dict()


@app.post("/api/control/start")
def start() -> dict:
    with state_lock:
        sandbox.running = True
        return {"ok": True, "running": True}


@app.post("/api/control/pause")
def pause() -> dict:
    with state_lock:
        sandbox.running = False
        return {"ok": True, "running": False}


@app.post("/api/control/reset")
def reset(req: ResetRequest) -> dict:
    with state_lock:
        sandbox.set_speed(req.speed)
        sandbox.reset(
            seed=req.seed,
            interval_ms=int(req.interval_s * 1000.0),
            payload_capacity=req.payload_capacity,
            site_count=req.site_count,
        )
        sandbox.running = False
        return {"ok": True, "state": sandbox.state_dict()}


@app.post("/api/control/config")
def configure(req: ConfigRequest) -> dict:
    with state_lock:
        if req.speed is not None:
            sandbox.set_speed(req.speed)
        if req.interval_s is not None:
            sandbox.set_interval_ms(req.interval_s * 1000.0)
        if req.payload_capacity is not None:
            sandbox.set_payload_capacity(req.payload_capacity)
        if req.site_count is not None:
            sandbox.set_site_count(req.site_count)
            sandbox.running = False
        return {"ok": True, "controls": sandbox.state_dict()["controls"]}


@app.post("/api/control/advance")
def advance(req: AdvanceRequest) -> dict:
    with state_lock:
        sandbox.advance_by(req.ms)
        return {"ok": True, "state": sandbox.state_dict()}


@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest) -> dict:
    if req.payload_values:
        report = await asyncio.to_thread(
            run_payload_sweep,
            payload_values=req.payload_values,
            episodes=req.episodes,
            horizon_s=req.horizon_s,
            interval_s=req.interval_s,
            base_seed=req.base_seed,
            site_count=req.site_count,
        )
    else:
        report = await asyncio.to_thread(
            run_evaluation,
            episodes=req.episodes,
            horizon_s=req.horizon_s,
            interval_s=req.interval_s,
            base_seed=req.base_seed,
            payload_capacity=req.payload_capacity,
            site_count=req.site_count,
        )
    return {"ok": True, "report": report}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT_DIR / "index.html")


# keep api routes above this mount
app.mount("/", StaticFiles(directory=str(ROOT_DIR), html=True), name="static")
