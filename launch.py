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

    server = BiofeedbackServer(proc_a, proc_b, port=args.port)

    if args.simulate:
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
        # No addresses provided — scan interactively
        from ble import scan_for_hr_monitors, stream as ble_stream
        from battery import read_battery_once

        n_needed = 2 if proc_b else 1
        print(f"Scanning for heart rate monitor(s) (10 s) …")
        print("Make sure sensor(s) are powered on and worn.\n")
        devices = await scan_for_hr_monitors(timeout=10.0)

        if not devices:
            print("No heart rate monitors found.")
            print("Tip: run  python3 scan.py  to debug discovery, or use  --simulate  for testing.")
            sys.exit(1)

        if len(devices) < n_needed:
            print(f"Found {len(devices)} device(s) but {n_needed} needed.")
            print("Power on all sensors and retry, or use --simulate for testing.")
            sys.exit(1)

        print(f"Found {len(devices)} device(s):\n")
        for i, d in enumerate(devices):
            print(f"  [{i}]  {d.name or 'Unknown':<30}  {d.address}")

        addr_a = None
        addr_b = None

        if len(devices) == n_needed:
            # exact match — auto-assign without prompting
            addr_a = devices[0].address
            addr_b = devices[1].address if proc_b else None
            print(f"\nAuto-assigning:")
            print(f"  {names[0]:<20} → {devices[0].name or devices[0].address}")
            if addr_b:
                print(f"  {names[1]:<20} → {devices[1].name or devices[1].address}")
        else:
            # more devices found than needed — let user pick
            print()
            try:
                raw_a = input("Select index for Partner A: ").strip()
                idx_a = int(raw_a)
                if not (0 <= idx_a < len(devices)):
                    raise ValueError
            except (ValueError, EOFError):
                print("Invalid selection. Exiting.")
                sys.exit(1)

            addr_a = devices[idx_a].address

            if proc_b:
                try:
                    raw_b = input("Select index for Partner B: ").strip()
                    idx_b = int(raw_b)
                    if idx_b == idx_a or not (0 <= idx_b < len(devices)):
                        raise ValueError
                except (ValueError, EOFError):
                    print("Invalid selection. Exiting.")
                    sys.exit(1)
                addr_b = devices[idx_b].address

            print(f"\nConnecting  {names[0]} → {addr_a}")
            if addr_b:
                print(f"            {names[1]} → {addr_b}")

        print()

        batt_a = await read_battery_once(addr_a)
        if batt_a is not None:
            server.on_battery(0, batt_a)

        async def ble_a():
            await ble_stream(
                address=addr_a, name=names[0],
                on_bpm=lambda bpm: server.on_bpm(0, bpm),
                on_rr=lambda rr: server.on_rr(0, rr),
                on_connect=lambda label: (print(f"[ble] {label} connected"), server.on_sensor_connect(0)),
                on_disconnect=lambda label, reason: (print(f"[ble] {label} disconnected: {reason}"), server.on_sensor_disconnect(0)),
            )

        asyncio.create_task(ble_a())

        if addr_b:
            batt_b = await read_battery_once(addr_b)
            if batt_b is not None:
                server.on_battery(1, batt_b)

            async def ble_b():
                await ble_stream(
                    address=addr_b, name=names[1],
                    on_bpm=lambda bpm: server.on_bpm(1, bpm),
                    on_rr=lambda rr: server.on_rr(1, rr),
                    on_connect=lambda label: (print(f"[ble] {label} connected"), server.on_sensor_connect(1)),
                    on_disconnect=lambda label, reason: (print(f"[ble] {label} disconnected: {reason}"), server.on_sensor_disconnect(1)),
                )
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
