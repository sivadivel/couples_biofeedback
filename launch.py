"""
launch.py — entry point for the couples biofeedback web application.

Usage:
    python launch.py --simulate
    python launch.py --simulate --names "Alex" "Jordan" --bpm 68 75
    python launch.py --addresses UUID1 UUID2
"""

import argparse
import asyncio
import sys
import webbrowser

from processor import PartnerProcessor
from server import BiofeedbackServer


def parse_args():
    p = argparse.ArgumentParser(
        description="Couples biofeedback web server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--simulate", action="store_true",
                   help="Use simulated HR data (no hardware needed)")
    p.add_argument("--bpm", nargs=2, type=int, default=[68, 75], metavar="BPM",
                   help="Base BPM for simulated monitors (default: 68 75)")
    p.add_argument("--names", nargs=2, metavar="NAME", default=["Partner A", "Partner B"],
                   help="Display names for the two partners")
    p.add_argument("--addresses", nargs=2, metavar="ADDR",
                   help="BLE addresses to connect to directly")
    p.add_argument("--port", type=int, default=8765,
                   help="HTTP/WS port (default: 8765)")
    return p.parse_args()


async def main():
    args = parse_args()

    names = args.names or ["Partner A", "Partner B"]
    proc_a = PartnerProcessor(names[0], "A")
    proc_b = PartnerProcessor(names[1], "B")

    server = BiofeedbackServer(proc_a, proc_b, port=args.port)

    if args.simulate:
        from simulator import simulate_stream

        async def sim_a():
            await simulate_stream(
                name=names[0],
                base_bpm=args.bpm[0],
                on_bpm=lambda bpm: server.on_bpm(0, bpm),
                on_rr=lambda rr: server.on_rr(0, rr),
                on_connect=lambda label: print(f"[sim] {label} connected"),
            )

        async def sim_b():
            await simulate_stream(
                name=names[1],
                base_bpm=args.bpm[1],
                on_bpm=lambda bpm: server.on_bpm(1, bpm),
                on_rr=lambda rr: server.on_rr(1, rr),
                on_connect=lambda label: print(f"[sim] {label} connected"),
            )

        asyncio.create_task(sim_a())
        asyncio.create_task(sim_b())

    elif args.addresses:
        from ble import stream as ble_stream

        async def ble_a():
            await ble_stream(
                address=args.addresses[0],
                name=names[0],
                on_bpm=lambda bpm: server.on_bpm(0, bpm),
                on_rr=lambda rr: server.on_rr(0, rr),
                on_connect=lambda label: print(f"[ble] {label} connected"),
                on_disconnect=lambda label, reason: print(f"[ble] {label} disconnected: {reason}"),
            )

        async def ble_b():
            await ble_stream(
                address=args.addresses[1],
                name=names[1],
                on_bpm=lambda bpm: server.on_bpm(1, bpm),
                on_rr=lambda rr: server.on_rr(1, rr),
                on_connect=lambda label: print(f"[ble] {label} connected"),
                on_disconnect=lambda label, reason: print(f"[ble] {label} disconnected: {reason}"),
            )

        asyncio.create_task(ble_a())
        asyncio.create_task(ble_b())

    else:
        # No addresses provided — scan interactively
        from ble import scan_for_hr_monitors, stream as ble_stream

        print("Scanning for heart rate monitors (10 s) …")
        print("Make sure both sensors are powered on and worn.\n")
        devices = await scan_for_hr_monitors(timeout=10.0)

        if not devices:
            print("No heart rate monitors found.")
            print("Tip: run  python3 scan.py  to debug discovery, or use  --simulate  for testing.")
            sys.exit(1)

        print(f"Found {len(devices)} device(s):\n")
        for i, d in enumerate(devices):
            print(f"  [{i}]  {d.name or 'Unknown':<30}  {d.address}")

        if len(devices) < 2:
            print("\nOnly one device found — need two. Power on the second sensor and retry.")
            sys.exit(1)

        print()
        try:
            raw_a = input("Select index for Partner A: ").strip()
            raw_b = input("Select index for Partner B: ").strip()
            idx_a, idx_b = int(raw_a), int(raw_b)
            if idx_a == idx_b or not (0 <= idx_a < len(devices)) or not (0 <= idx_b < len(devices)):
                raise ValueError
        except (ValueError, EOFError):
            print("Invalid selection. Exiting.")
            sys.exit(1)

        addr_a, addr_b = devices[idx_a].address, devices[idx_b].address
        print(f"\nConnecting  {names[0]} → {addr_a}")
        print(f"            {names[1]} → {addr_b}\n")

        async def ble_a():
            await ble_stream(
                address=addr_a, name=names[0],
                on_bpm=lambda bpm: server.on_bpm(0, bpm),
                on_rr=lambda rr: server.on_rr(0, rr),
                on_connect=lambda label: print(f"[ble] {label} connected"),
                on_disconnect=lambda label, reason: print(f"[ble] {label} disconnected: {reason}"),
            )

        async def ble_b():
            await ble_stream(
                address=addr_b, name=names[1],
                on_bpm=lambda bpm: server.on_bpm(1, bpm),
                on_rr=lambda rr: server.on_rr(1, rr),
                on_connect=lambda label: print(f"[ble] {label} connected"),
                on_disconnect=lambda label, reason: print(f"[ble] {label} disconnected: {reason}"),
            )

        asyncio.create_task(ble_a())
        asyncio.create_task(ble_b())

    # open browser after a short delay
    async def open_browser():
        await asyncio.sleep(1.0)
        webbrowser.open(f"http://localhost:{args.port}/")

    asyncio.create_task(open_browser())

    await server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")
