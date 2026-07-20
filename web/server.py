"""
server.py — multi-tenant aiohttp server for the simplified web biofeedback app.

Unlike the local app's server.py (one process = one couple, server-side BLE),
this app hosts many concurrent Rooms, each joined by up to two browsers that
pair their own heart rate strap via the Web Bluetooth API and relay RR
intervals over the WebSocket connection. No recording, transcription, dyadic
coupling, or meditation — just live per-partner metrics and a recovery/
breathing mode, reusing the same PartnerProcessor metric engine as the full
app (see processor.py at the repo root, imported unmodified via room.py).

Everything is in-memory and ephemeral: no session logs, no accounts. A room
is identified only by an unguessable code and is pruned once empty and idle.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from aiohttp import web, WSMsgType

from room import Room, RoomRegistry

STATIC_DIR = Path(__file__).parent / "static"
_NO_CACHE = {"Cache-Control": "no-store"}

PRUNE_INTERVAL_S = 60.0


class WebApp:
    def __init__(self):
        self.rooms = RoomRegistry()

    # ── static pages ──────────────────────────────────────────────────────────

    async def index_handler(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "index.html", headers=_NO_CACHE)

    async def static_handler(self, request: web.Request) -> web.FileResponse:
        filename = request.match_info["filename"]
        path = (STATIC_DIR / filename).resolve()
        if STATIC_DIR.resolve() not in path.parents or not path.exists():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers=_NO_CACHE)

    # ── room REST API ─────────────────────────────────────────────────────────

    async def create_room_handler(self, request: web.Request) -> web.Response:
        room = self.rooms.create()
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": room.code}),
            headers=_NO_CACHE,
        )

    async def room_status_handler(self, request: web.Request) -> web.Response:
        code = request.match_info["code"]
        room = self.rooms.get(code)
        if room is None:
            return web.Response(
                status=404, content_type="application/json",
                text=json.dumps({"exists": False}),
            )
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "exists": True,
                "names": room.names,
                "occupied": {p: room.occupied(p) for p in ("A", "B")},
            }),
            headers=_NO_CACHE,
        )

    # ── WebSocket ─────────────────────────────────────────────────────────────

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        code = request.query.get("room", "")
        partner = request.query.get("partner", "A")
        name = (request.query.get("name") or "").strip() or f"Partner {partner}"

        room = self.rooms.get(code)
        ws = web.WebSocketResponse(heartbeat=30)

        if partner not in ("A", "B") or room is None:
            await ws.prepare(request)
            await ws.close(code=4404, message=b"room not found")
            return ws
        if room.occupied(partner):
            await ws.prepare(request)
            await ws.close(code=4409, message=b"partner slot already connected")
            return ws

        await ws.prepare(request)
        proc = room.claim(partner, name)
        room.clients[partner] = ws
        room.touch()

        await ws.send_str(json.dumps({
            "type": "session_init",
            "partner": partner,
            "names": room.names,
            "sensor_online": room.sensor_online,
        }))
        await self._broadcast_peer_update(room)

        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    if msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                        break
                    continue
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                room.touch()
                await self._handle_message(room, partner, proc, data, ws)
        finally:
            if room.clients.get(partner) is ws:
                del room.clients[partner]
            room.sensor_online[partner] = False
            room.touch()
            await self._broadcast_peer_update(room)
        return ws

    async def _handle_message(self, room: Room, partner: str, proc, data: dict,
                               ws: web.WebSocketResponse) -> None:
        mtype = data.get("type")

        if mtype == "ping":
            await ws.send_str(json.dumps({"type": "pong"}))

        elif mtype == "rr_sample":
            for rr_ms in data.get("rr_ms", []):
                try:
                    proc.push_rr(float(rr_ms))
                except (TypeError, ValueError):
                    continue

        elif mtype == "sensor_connect":
            room.sensor_online[partner] = True
            await self._broadcast(room, {"type": "sensor_status", "partner": partner, "online": True})

        elif mtype == "sensor_disconnect":
            room.sensor_online[partner] = False
            await self._broadcast(room, {"type": "sensor_status", "partner": partner, "online": False})

        elif mtype == "battery":
            level = data.get("level")
            if isinstance(level, (int, float)):
                await self._broadcast(room, {"type": "battery", "partner": partner, "level": level})

        elif mtype == "set_baseline":
            ok = proc.set_baseline()
            await self._broadcast(room, {
                "type": "baseline_status", "partner": partner, "ok": ok, "name": proc.name,
            })

        elif mtype == "clear_baseline":
            proc.clear_baseline()
            await self._broadcast(room, {
                "type": "baseline_status", "partner": partner, "ok": False, "name": proc.name,
            })

        elif mtype == "set_mode":
            proc.set_mode(data.get("mode", "conversation"))

    async def _broadcast_peer_update(self, room: Room) -> None:
        await self._broadcast(room, {
            "type": "peer_update",
            "names": room.names,
            "occupied": {p: room.occupied(p) for p in ("A", "B")},
        })

    async def _broadcast(self, room: Room, msg: dict) -> None:
        if not room.clients:
            return
        text = json.dumps(msg)
        dead = []
        for partner, ws in list(room.clients.items()):
            try:
                await ws.send_str(text)
            except Exception:
                dead.append(partner)
        for partner in dead:
            room.clients.pop(partner, None)

    # ── background loops ─────────────────────────────────────────────────────

    async def metrics_loop(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            try:
                now = time.monotonic()
                for room in self.rooms.active():
                    for partner, proc in room.procs.items():
                        if proc is None:
                            continue
                        for msg in proc.get_updates(now):
                            await self._broadcast(room, msg)
            except Exception as exc:
                print(f"[metrics_loop] ERROR (tick skipped): {exc}")

    async def prune_loop(self) -> None:
        while True:
            await asyncio.sleep(PRUNE_INTERVAL_S)
            removed = self.rooms.prune_idle()
            if removed:
                print(f"[prune_loop] removed {removed} idle room(s)")

    # ── app bootstrap ────────────────────────────────────────────────────────

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.index_handler)
        app.router.add_get("/static/{filename}", self.static_handler)
        app.router.add_post("/api/rooms", self.create_room_handler)
        app.router.add_get("/api/rooms/{code}", self.room_status_handler)
        app.router.add_get("/ws", self.ws_handler)
        return app

    async def run(self, port: int) -> None:
        app = self.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        print(f"Web app running at http://localhost:{port}/")
        asyncio.create_task(self.metrics_loop())
        asyncio.create_task(self.prune_loop())
        await asyncio.Event().wait()


def parse_args():
    p = argparse.ArgumentParser(description="Simplified biofeedback web app (multi-tenant)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    return p.parse_args()


async def main():
    args = parse_args()
    webapp = WebApp()
    await webapp.run(args.port)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")
