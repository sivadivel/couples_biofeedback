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
        proc_b: PartnerProcessor | None = None,
        port: int = 8765,
    ):
        self.proc_a = proc_a
        self.proc_b = proc_b
        self.dyadic = DyadicProcessor(proc_a.name, proc_b.name) if proc_b else None
        self.port = port
        self.clients: set[web.WebSocketResponse] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sensor_online: dict[int, bool] = {0: False, 1: False}

    # ── callbacks for BLE / simulator ────────────────────────────────────────

    def on_rr(self, idx: int, rr_ms: float) -> None:
        if idx == 0:
            proc = self.proc_a
        elif idx == 1 and self.proc_b:
            proc = self.proc_b
        else:
            return
        if self._loop is not None:
            self._loop.call_soon_threadsafe(proc.push_rr, rr_ms)
        else:
            proc.push_rr(rr_ms)

    def on_bpm(self, idx: int, bpm: float) -> None:
        # bpm notifications not used directly; RR intervals are the source of truth
        pass

    def on_sensor_connect(self, idx: int) -> None:
        self._sensor_online[idx] = True
        partner = "A" if idx == 0 else "B"
        if self._loop:
            msg = {"type": "sensor_status", "partner": partner, "online": True}
            self._loop.call_soon_threadsafe(asyncio.create_task, self.broadcast(msg))

    def on_sensor_disconnect(self, idx: int) -> None:
        self._sensor_online[idx] = False
        partner = "A" if idx == 0 else "B"
        if self._loop:
            msg = {"type": "sensor_status", "partner": partner, "online": False}
            self._loop.call_soon_threadsafe(asyncio.create_task, self.broadcast(msg))

    def on_battery(self, idx: int, level: int) -> None:
        partner = "A" if idx == 0 else "B"
        if self._loop:
            msg = {"type": "battery", "partner": partner, "level": level}
            self._loop.call_soon_threadsafe(asyncio.create_task, self.broadcast(msg))
        else:
            # called before loop starts — queue via broadcast at first opportunity
            self._pending_battery = getattr(self, "_pending_battery", {})
            self._pending_battery[partner] = level

    # ── WebSocket handler ─────────────────────────────────────────────────────

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        self.clients.add(ws)
        names = {"A": self.proc_a.name}
        if self.proc_b:
            names["B"] = self.proc_b.name
        await ws.send_str(json.dumps({
            "type": "session_init",
            "names": names,
            "single_partner": self.proc_b is None,
        }))
        if self.proc_a.baseline is not None:
            await ws.send_str(json.dumps({
                "type": "baseline_status",
                "partner": "A",
                "ok": True,
                "name": self.proc_a.name,
            }))
        if self.proc_b and self.proc_b.baseline is not None:
            await ws.send_str(json.dumps({
                "type": "baseline_status",
                "partner": "B",
                "ok": True,
                "name": self.proc_b.name,
            }))
        # re-send sensor online state and any pending battery levels
        for idx, partner in ((0, "A"), (1, "B")):
            if idx == 1 and not self.proc_b:
                continue
            await ws.send_str(json.dumps({
                "type": "sensor_status",
                "partner": partner,
                "online": self._sensor_online.get(idx, False),
            }))
        for partner, level in getattr(self, "_pending_battery", {}).items():
            await ws.send_str(json.dumps({
                "type": "battery", "partner": partner, "level": level
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
                        if partner == "B" and not self.proc_b:
                            continue
                        proc = self.proc_a if partner == "A" else self.proc_b
                        ok = proc.set_baseline()
                        await self.broadcast({
                            "type": "baseline_status",
                            "partner": partner,
                            "ok": ok,
                            "name": proc.name,
                        })
                    elif mtype == "clear_baseline":
                        partner = data.get("partner", "A")
                        if partner == "B" and not self.proc_b:
                            continue
                        proc = self.proc_a if partner == "A" else self.proc_b
                        proc.clear_baseline()
                        await self.broadcast({
                            "type": "baseline_status",
                            "partner": partner,
                            "ok": False,
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

    async def whitepaper_handler(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(Path(__file__).parent / "whitepaper.html", headers=self._NO_CACHE)

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
            if self.proc_b:
                for msg in self.proc_b.get_updates(now):
                    await self.broadcast(msg)
            if self.dyadic:
                for msg in self.dyadic.get_updates(self.proc_a, self.proc_b, now):
                    await self.broadcast(msg)

    # ── app factory ──────────────────────────────────────────────────────────

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_get("/", self.index_handler)
        app.router.add_get("/mode_b", self.mode_b_handler)
        app.router.add_get("/whitepaper", self.whitepaper_handler)
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
