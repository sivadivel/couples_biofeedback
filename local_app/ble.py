"""BLE heart rate monitor connection using the standard Heart Rate Service (0x180D)."""

import asyncio
import struct
from bleak import BleakScanner, BleakClient

HR_SERVICE_UUID     = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


def parse_hr_measurement(data: bytearray) -> tuple[int, list[float]]:
    """
    Parse BLE Heart Rate Measurement characteristic value (BT spec 3.106).
    Returns (bpm, [rr_ms, ...]).  R-R list is empty if the device doesn't send it.
    """
    flags = data[0]
    idx = 1

    if flags & 0x01:        # bit 0: HR value is UINT16
        bpm = struct.unpack_from("<H", data, idx)[0]
        idx += 2
    else:                   # HR value is UINT8
        bpm = data[idx]
        idx += 1

    if flags & 0x08:        # bit 3: Energy Expended field present (skip it)
        idx += 2

    rr_intervals: list[float] = []
    if flags & 0x10:        # bit 4: RR-Interval fields present
        while idx + 1 < len(data):
            raw = struct.unpack_from("<H", data, idx)[0]
            rr_intervals.append(raw * 1000.0 / 1024.0)   # units of 1/1024 s → ms
            idx += 2

    return bpm, rr_intervals


async def scan_for_hr_monitors(timeout: float = 10.0) -> list:
    """Return BLE devices that advertise the Heart Rate Service."""
    print(f"Scanning for heart rate monitors ({timeout:.0f}s)...")
    return list(await BleakScanner.discover(
        timeout=timeout,
        service_uuids=[HR_SERVICE_UUID],
    ))


async def stream(address: str, name: str, on_bpm, on_rr=None,
                 on_connect=None, on_disconnect=None):
    """
    Connect to a BLE HR monitor and call on_bpm(bpm) and on_rr(rr_ms) on each
    notification.  Reconnects automatically on drop.
    """
    while True:
        try:
            async with BleakClient(address) as client:
                if on_connect:
                    on_connect(name)

                def handler(_, data):
                    bpm, rr_list = parse_hr_measurement(bytearray(data))
                    on_bpm(bpm)
                    if on_rr:
                        for rr in rr_list:
                            on_rr(rr)

                await client.start_notify(HR_MEASUREMENT_UUID, handler)
                while client.is_connected:
                    await asyncio.sleep(0.5)
                await client.stop_notify(HR_MEASUREMENT_UUID)

        except Exception as exc:
            if on_disconnect:
                on_disconnect(name, str(exc))
            await asyncio.sleep(3)
