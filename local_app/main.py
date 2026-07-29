"""
Real-time dual heart rate monitor.

Usage:
    python main.py --simulate                  # two fake monitors (no hardware)
    python main.py                             # scan & pick BLE devices
    python main.py --addresses AA:BB:... CC:DD:...   # connect directly by address
"""

import argparse
import asyncio
import sys
import threading

from plot import HeartRateDashboard


def _run_loop(loop: asyncio.AbstractEventLoop, coros):
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(asyncio.gather(*coros))
    except KeyboardInterrupt:
        pass


def _build_dashboard_and_coros(args):
    """Return (dashboard, list_of_coroutines) for either simulated or real mode."""

    if args.simulate:
        from simulator import simulate_stream

        names = args.names or ["Athlete A", "Athlete B"]
        bases = args.bpm
        dashboard = HeartRateDashboard(names)

        coros = [
            simulate_stream(
                name=names[i],
                base_bpm=bases[i],
                on_bpm=lambda bpm, i=i: dashboard.add_bpm(i, bpm),
                on_rr=lambda rr, i=i: dashboard.add_rr(i, rr),
                on_connect=lambda label, i=i: dashboard.mark_connected(i, label),
            )
            for i in range(len(names))
        ]
        return dashboard, coros

    # --- real BLE mode ---
    from ble import scan_for_hr_monitors, stream as ble_stream

    if args.addresses:
        addresses = args.addresses
        names = args.names or [f"Monitor {i+1}" for i in range(len(addresses))]
    else:
        # Interactive scan
        scan_loop = asyncio.new_event_loop()
        devices = scan_loop.run_until_complete(scan_for_hr_monitors(timeout=args.scan_timeout))
        scan_loop.close()

        if not devices:
            print(
                "\nNo heart rate monitors found.\n"
                "Tips:\n"
                "  • Make sure the monitor is powered on and in range.\n"
                "  • Grant Bluetooth permission to Terminal in System Settings → Privacy → Bluetooth.\n"
                "  • Try --simulate to test the UI without hardware."
            )
            sys.exit(1)

        print("\nDevices found:")
        for i, d in enumerate(devices):
            print(f"  [{i}] {d.name or 'Unknown'}  ({d.address})")

        if len(devices) == 1:
            selected = [devices[0]]
        else:
            raw = input("\nEnter indices of up to 2 devices (e.g. '0 1'): ").split()
            selected = [devices[int(x)] for x in raw[:2]]

        addresses = [d.address for d in selected]
        names = args.names or [d.name or f"Monitor {i+1}" for i, d in enumerate(selected)]

    dashboard = HeartRateDashboard(names[:len(addresses)])

    coros = [
        ble_stream(
            address=addresses[i],
            name=names[i],
            on_bpm=lambda bpm, i=i: dashboard.add_bpm(i, bpm),
            on_rr=lambda rr, i=i: dashboard.add_rr(i, rr),
            on_connect=lambda label, i=i: dashboard.mark_connected(i, label),
            on_disconnect=lambda label, reason, i=i: dashboard.mark_disconnected(i, reason),
        )
        for i in range(len(addresses))
    ]
    return dashboard, coros


def main():
    parser = argparse.ArgumentParser(
        description="Real-time dual BLE heart rate monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--simulate", action="store_true",
                        help="Use simulated HR data (no hardware needed)")
    parser.add_argument("--bpm", nargs=2, type=int, default=[68, 75],
                        metavar="BPM",
                        help="Base BPM for simulated monitors (default: 68 75)")
    parser.add_argument("--names", nargs="+", metavar="NAME",
                        help="Display names for monitors")
    parser.add_argument("--addresses", nargs="+", metavar="ADDR",
                        help="BLE addresses to connect to directly (skip scan)")
    parser.add_argument("--scan-timeout", type=float, default=10.0, metavar="SEC",
                        help="BLE scan duration in seconds (default: 10)")
    args = parser.parse_args()

    dashboard, coros = _build_dashboard_and_coros(args)

    loop = asyncio.new_event_loop()
    bg = threading.Thread(
        target=_run_loop,
        args=(loop, coros),
        daemon=True,
        name="ble-thread",
    )
    bg.start()

    try:
        dashboard.run()
    except KeyboardInterrupt:
        pass
    finally:
        loop.call_soon_threadsafe(loop.stop)


if __name__ == "__main__":
    main()
