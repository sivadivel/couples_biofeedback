"""
server.py
aiohttp HTTP + WebSocket server for the couples biofeedback dashboard.
"""

from __future__ import annotations
import asyncio
import json
import os
import time
from pathlib import Path

from aiohttp import web, WSMsgType

from processor import PartnerProcessor, DyadicProcessor

STATIC_DIR = Path(__file__).parent / "static"


class BiofeedbackServer:
    def __init__(
        self,
        proc_a: PartnerProcessor,
        proc_b: PartnerProcessor,
        port: int = 8765,
    ):
        self.proc_a = proc_a
        self.proc_b = proc_b
        self.dyadic = DyadicProcessor(proc_a.name, proc_b.name)
        self.port = port
        self.clients: set[web.WebSocketResponse] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── callbacks for BLE / simulator ────────────────────────────────────────

    def on_rr(self, idx: int, rr_ms: float) -> None:
        proc = self.proc_a if idx == 0 else self.proc_b
        if self._loop is not None:
            self._loop.call_soon_threadsafe(proc.push_rr, rr_ms)
        else:
            proc.push_rr(rr_ms)

    def on_bpm(self, idx: int, bpm: float) -> None:
        # bpm notifications not used directly; RR intervals are the source of truth
        pass

    # ── WebSocket handler ─────────────────────────────────────────────────────

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        self.clients.add(ws)
        await ws.send_str(json.dumps({
            "type": "session_init",
            "names": {"A": self.proc_a.name, "B": self.proc_b.name},
        }))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    mtype = data.get("type")
                    if mtype == "ping":
                        await ws.send_str(json.dumps({"type": "pong"}))
                    elif mtype == "set_baseline":
                        partner = data.get("partner", "A")
                        proc = self.proc_a if partner == "A" else self.proc_b
                        ok = proc.set_baseline()
                        await self.broadcast({
                            "type": "baseline_status",
                            "partner": partner,
                            "ok": ok,
                            "name": proc.name,
                        })
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            self.clients.discard(ws)
        return ws

    # ── static file handlers ─────────────────────────────────────────────────

    _NO_CACHE = {"Cache-Control": "no-store"}

    async def index_handler(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "index.html", headers=self._NO_CACHE)

    async def mode_b_handler(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "mode_b.html", headers=self._NO_CACHE)

    async def static_handler(self, request: web.Request) -> web.FileResponse:
        filename = request.match_info["filename"]
        path = STATIC_DIR / filename
        if not path.exists():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers=self._NO_CACHE)

    # ── broadcast ─────────────────────────────────────────────────────────────

    async def broadcast(self, msg: dict) -> None:
        if not self.clients:
            return
        text = json.dumps(msg)
        dead = set()
        for ws in list(self.clients):
            try:
                await ws.send_str(text)
            except Exception:
                dead.add(ws)
        self.clients -= dead

    # ── metrics loop ─────────────────────────────────────────────────────────

    async def metrics_loop(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            now = time.monotonic()
            for msg in self.proc_a.get_updates(now):
                await self.broadcast(msg)
            for msg in self.proc_b.get_updates(now):
                await self.broadcast(msg)
            for msg in self.dyadic.get_updates(self.proc_a, self.proc_b, now):
                await self.broadcast(msg)

    # ── app factory ──────────────────────────────────────────────────────────

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_get("/", self.index_handler)
        app.router.add_get("/mode_b", self.mode_b_handler)
        app.router.add_get("/static/{filename}", self.static_handler)
        return app

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        app = self.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", self.port)
        await site.start()
        print(f"Server running at http://localhost:{self.port}/")
        asyncio.create_task(self.metrics_loop())
        # run forever
        await asyncio.Event().wait()
