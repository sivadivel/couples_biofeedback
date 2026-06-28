"""
debug_ble.py — print raw BLE heart rate notifications from one device.

Usage:
    python3 debug_ble.py <address>

Shows the raw flag byte, parsed BPM, and whether R-R intervals are present.
Run this first to confirm what the sensor is actually sending.
"""

import argparse
import asyncio
import struct
import sys

from bleak import BleakClient

HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


def decode(data: bytearray) -> None:
    flags = data[0]
    idx = 1

    hr16  = bool(flags & 0x01)
    has_ee = bool(flags & 0x08)
    has_rr = bool(flags & 0x10)

    if hr16:
        bpm = struct.unpack_from("<H", data, idx)[0]; idx += 2
    else:
        bpm = data[idx]; idx += 1

    if has_ee:
        idx += 2  # skip energy expended

    rr_list = []
    if has_rr:
        while idx + 1 < len(data):
            raw = struct.unpack_from("<H", data, idx)[0]
            rr_list.append(raw * 1000.0 / 1024.0)
            idx += 2

    rr_str = ", ".join(f"{r:.1f}" for r in rr_list) if rr_list else "— none —"
    print(f"flags=0x{flags:02X}  bpm={bpm:3d}  RR intervals (ms): {rr_str}")


async def main(address: str) -> None:
    print(f"Connecting to {address} …")
    async with BleakClient(address) as client:
        print("Connected. Listening for notifications (Ctrl-C to stop).\n")
        print(f"{'flags':8}  {'bpm':6}  RR intervals (ms)")
        print("-" * 60)

        def handler(_, data):
            decode(bytearray(data))

        await client.start_notify(HR_MEASUREMENT_UUID, handler)
        await asyncio.Event().wait()   # run until Ctrl-C


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("address", help="BLE device address / UUID from scan.py")
    args = p.parse_args()
    try:
        asyncio.run(main(args.address))
    except KeyboardInterrupt:
        pass
