"""
launch.py — entry point for the couples biofeedback web application.

Usage (one person):
    python launch.py --simulate --names "Alex" --bpm 68
    python launch.py --addresses UUID1 --names "Alex"

Usage (two people):
    python launch.py --simulate --names "Alex" "Jordan" --bpm 68 75
    python launch.py --addresses UUID1 UUID2 --names "Alex" "Jordan"
    python launch.py --names "Alex" "Jordan"   # interactive BLE scan
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
    p.add_argument("--bpm", nargs="+", type=int, default=[68, 75], metavar="BPM",
                   help="Base BPM for simulated monitors — 1 or 2 values (default: 68 75)")
    p.add_argument("--names", nargs="+", metavar="NAME", default=["Partner A", "Partner B"],
                   help="Display name(s) — 1 or 2 values")
    p.add_argument("--addresses", nargs="+", metavar="ADDR",
                   help="BLE address(es) to connect to directly — 1 or 2 values")
    p.add_argument("--port", type=int, default=8765,
                   help="HTTP/WS port (default: 8765)")
    args = p.parse_args()

    # validate counts
    for flag, values in [("--names", args.names), ("--bpm", args.bpm)]:
        if values and len(values) > 2:
            p.error(f"{flag} accepts 1 or 2 values, got {len(values)}")
    if args.addresses and len(args.addresses) > 2:
        p.error(f"--addresses accepts 1 or 2 values, got {len(args.addresses)}")

    return args


async def main():
    args = parse_args()

    names = args.names or ["Partner A", "Partner B"]
    bpms  = args.bpm   or [68, 75]

    proc_a = PartnerProcessor(names[0], "A")
    proc_b = PartnerProcessor(names[1], "B") if len(names) > 1 else None

    if args.simulate:
        server = BiofeedbackServer(proc_a, proc_b, port=args.port)
        from simulator import simulate_stream

        async def sim_a():
            server.on_sensor_connect(0)
            await simulate_stream(
                name=names[0],
                base_bpm=bpms[0],
                on_bpm=lambda bpm: server.on_bpm(0, bpm),
                on_rr=lambda rr: server.on_rr(0, rr),
                on_connect=lambda label: print(f"[sim] {label} connected"),
            )

        asyncio.create_task(sim_a())

        if proc_b:
            async def sim_b():
                server.on_sensor_connect(1)
                await simulate_stream(
                    name=names[1],
                    base_bpm=bpms[1] if len(bpms) > 1 else 75,
                    on_bpm=lambda bpm: server.on_bpm(1, bpm),
                    on_rr=lambda rr: server.on_rr(1, rr),
                    on_connect=lambda label: print(f"[sim] {label} connected"),
                )
            asyncio.create_task(sim_b())

    elif args.addresses:
        server = BiofeedbackServer(proc_a, proc_b, port=args.port)
        from ble import stream as ble_stream
        from battery import read_battery_once

        batt_a = await read_battery_once(args.addresses[0])
        if batt_a is not None:
            server.on_battery(0, batt_a)

        async def ble_a():
            await ble_stream(
                address=args.addresses[0],
                name=names[0],
                on_bpm=lambda bpm: server.on_bpm(0, bpm),
                on_rr=lambda rr: server.on_rr(0, rr),
                on_connect=lambda label: (print(f"[ble] {label} connected"), server.on_sensor_connect(0)),
                on_disconnect=lambda label, reason: (print(f"[ble] {label} disconnected: {reason}"), server.on_sensor_disconnect(0)),
            )

        asyncio.create_task(ble_a())

        if proc_b and len(args.addresses) > 1:
            batt_b = await read_battery_once(args.addresses[1])
            if batt_b is not None:
                server.on_battery(1, batt_b)

            async def ble_b():
                await ble_stream(
                    address=args.addresses[1],
                    name=names[1],
                    on_bpm=lambda bpm: server.on_bpm(1, bpm),
                    on_rr=lambda rr: server.on_rr(1, rr),
                    on_connect=lambda label: (print(f"[ble] {label} connected"), server.on_sensor_connect(1)),
                    on_disconnect=lambda label, reason: (print(f"[ble] {label} disconnected: {reason}"), server.on_sensor_disconnect(1)),
                )
            asyncio.create_task(ble_b())

    else:
        # Setup mode: browser dialog handles scanning and device assignment
        server = BiofeedbackServer(proc_a, proc_b, port=args.port, setup_mode=True)
        print(f"Open http://localhost:{args.port}/ to configure the session.")

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
