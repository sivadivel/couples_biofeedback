"""
scan.py — discover nearby BLE heart rate monitors and print their addresses.

Usage:
    python3 scan.py
    python3 scan.py --timeout 15
"""

import argparse
import asyncio
import sys

from ble import scan_for_hr_monitors


async def main(timeout: float) -> None:
    print(f"Scanning for heart rate monitors ({timeout:.0f}s) …")
    devices = await scan_for_hr_monitors(timeout=timeout)

    if not devices:
        print("No heart rate monitors found.")
        print("Make sure the sensors are powered on and worn (or held against skin).")
        sys.exit(1)

    print(f"\nFound {len(devices)} device(s):\n")
    for i, d in enumerate(devices):
        name = d.name or "Unknown"
        print(f"  [{i}]  {name:<30}  {d.address}")

    addresses = [d.address for d in devices]
    names = [d.name or f"Partner {chr(65 + i)}" for i, d in enumerate(devices)]

    print("\nTo launch with these devices, run:\n")
    if len(devices) >= 2:
        print(
            f'  python3 launch.py --names "{names[0]}" "{names[1]}"'
            f" \\\n"
            f"    --addresses {addresses[0]} {addresses[1]}"
        )
    else:
        print(
            f'  python3 launch.py --names "{names[0]}" "Partner B"'
            f" \\\n"
            f"    --addresses {addresses[0]} <address-of-second-device>"
        )
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Scan for BLE heart rate monitors")
    p.add_argument("--timeout", type=float, default=10.0,
                   help="Scan duration in seconds (default: 10)")
    args = p.parse_args()
    asyncio.run(main(args.timeout))
