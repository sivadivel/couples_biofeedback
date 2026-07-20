"""
room.py — per-session state for the simplified, multi-tenant web app.

A Room is the web app's equivalent of the local app's singleton
BiofeedbackServer: it owns two PartnerProcessors and the set of connected
WebSocket clients. Unlike the local app, many Rooms exist at once (keyed by
an unguessable room code) and there is no server-side BLE or session log —
state is purely in-memory and disappears once the room goes idle.
"""

from __future__ import annotations

import secrets
import string
import time

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from processor import PartnerProcessor  # noqa: E402 — shared metric engine, unmodified

ROOM_CODE_ALPHABET = string.ascii_uppercase.replace("O", "").replace("I", "") + "23456789"
ROOM_CODE_LEN = 6
ROOM_IDLE_TIMEOUT_S = 5 * 60  # prune a room once empty this long


def _generate_code() -> str:
    return "".join(secrets.choice(ROOM_CODE_ALPHABET) for _ in range(ROOM_CODE_LEN))


class Room:
    def __init__(self, code: str):
        self.code = code
        self.created_at = time.monotonic()
        self.last_active = time.monotonic()

        self.names: dict[str, str | None] = {"A": None, "B": None}
        self.procs: dict[str, PartnerProcessor | None] = {"A": None, "B": None}
        self.clients: dict[str, object] = {}  # "A"/"B" -> WebSocketResponse
        self.sensor_online: dict[str, bool] = {"A": False, "B": False}

    def touch(self) -> None:
        self.last_active = time.monotonic()

    def is_idle(self, now: float) -> bool:
        return not self.clients and (now - self.last_active) > ROOM_IDLE_TIMEOUT_S

    def claim(self, partner: str, name: str) -> PartnerProcessor:
        """Assign a partner slot a display name and (re)create its processor."""
        self.names[partner] = name
        proc = PartnerProcessor(name, partner, mode="conversation")
        self.procs[partner] = proc
        return proc

    def occupied(self, partner: str) -> bool:
        """True if a live WebSocket already holds this slot."""
        return partner in self.clients

    def other(self, partner: str) -> str:
        return "B" if partner == "A" else "A"


class RoomRegistry:
    def __init__(self):
        self._rooms: dict[str, Room] = {}

    def create(self) -> Room:
        code = _generate_code()
        while code in self._rooms:
            code = _generate_code()
        room = Room(code)
        self._rooms[code] = room
        return room

    def get(self, code: str) -> Room | None:
        return self._rooms.get(code.upper())

    def active(self) -> list[Room]:
        """Rooms with at least one connected client — the only ones worth ticking."""
        return [r for r in self._rooms.values() if r.clients]

    def prune_idle(self) -> int:
        now = time.monotonic()
        dead = [c for c, r in self._rooms.items() if r.is_idle(now)]
        for c in dead:
            del self._rooms[c]
        return len(dead)
