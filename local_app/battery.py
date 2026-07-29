"""
battery.py — one-shot BLE battery level read.

Opens its own connection, reads the Battery Level characteristic, then closes.
Must be called before ble.stream() acquires the persistent connection.
"""

from bleak import BleakClient

BATTERY_LEVEL_UUID = "00002a19-0000-1000-8000-00805f9b34fb"


async def read_battery_once(address: str) -> int | None:
    """Return battery percentage (0-100) or None if unsupported / unreachable."""
    try:
        async with BleakClient(address, timeout=8.0) as client:
            data = await client.read_gatt_char(BATTERY_LEVEL_UUID)
            return int(data[0])
    except Exception:
        return None
