"""
users.py — persistent registry of known users, shared by the couples dashboard
and standalone meditate mode.

A "user" is just a display name. The same slugification is used as the key for
both this registry and per-user meditation history files (see server.py's
history/*.ndjson), so selecting an existing user always lines up with their
history.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

USERS_FILE = Path(__file__).parent / "users.json"


def slug(name: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_").lower()
    return key or "anonymous"


def list_users() -> list[dict]:
    if not USERS_FILE.exists():
        return []
    try:
        data = json.loads(USERS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return data.get("users", [])


def create_user(name: str) -> dict:
    """Idempotent: returns the existing user if one with the same slug already exists."""
    name = (name or "").strip()
    if not name:
        raise ValueError("name is required")
    existing = list_users()
    key = slug(name)
    for u in existing:
        if u["id"] == key:
            return u
    user = {"id": key, "name": name, "created_at": datetime.now(timezone.utc).isoformat()}
    existing.append(user)
    USERS_FILE.write_text(json.dumps({"users": existing}, indent=2))
    return user


def delete_user(user_id: str) -> bool:
    """Removes a user from the registry. Returns True if a matching user was found."""
    existing = list_users()
    remaining = [u for u in existing if u["id"] != user_id]
    if len(remaining) == len(existing):
        return False
    USERS_FILE.write_text(json.dumps({"users": remaining}, indent=2))
    return True
