"""Standalone servo bring-up test — drives the ESP8266 over MQTT, no camera/model.

Use this during hardware bring-up to confirm wiring, power, and the MQTT path
before running the full vision pipeline.

Usage:
    python servo_test.py                 # run a LEFT/CENTER/RIGHT/SCAN sequence
    python servo_test.py --interactive   # type commands by hand
    python servo_test.py --host 192.168.1.100

Requires the broker running (tools/setup_broker.ps1) and the ESP8266 flashed
and connected. The default port (1884) comes from config.json.
"""
from __future__ import annotations

import argparse
import time

from src.config import Config
from src.mqtt_client import MqttPublisher


def run_sequence(pub: MqttPublisher):
    seq = [
        ("CENTER", 1.0, "center the camera"),
        ("LEFT:8", 1.2, "pan left (big step)"),
        ("LEFT:8", 1.2, "pan left again"),
        ("CENTER", 1.2, "back to center"),
        ("RIGHT:8", 1.2, "pan right (big step)"),
        ("RIGHT:8", 1.2, "pan right again"),
        ("CENTER", 1.2, "back to center"),
        ("SCAN", 4.0, "autonomous search sweep (watch it sweep 30-150)"),
        ("CENTER", 1.0, "exit scan, recenter"),
        ("STOP", 0.5, "hold"),
    ]
    print("[servo_test] running motion sequence — watch the servo:")
    for token, hold, desc in seq:
        print(f"  -> {token:10s}  ({desc})")
        pub.publish_command(token)
        time.sleep(hold)
    print("[servo_test] sequence complete.")


def run_interactive(pub: MqttPublisher):
    print("[servo_test] interactive mode. Commands: LEFT:n RIGHT:n CENTER STOP SCAN "
          "(q to quit)")
    while True:
        try:
            cmd = input("cmd> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd.lower() in ("q", "quit", "exit"):
            break
        if cmd:
            pub.publish_command(cmd)
            print(f"  published: {cmd}")


def main():
    ap = argparse.ArgumentParser(description="MQTT servo bring-up test.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--host", default=None, help="override broker host")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.host:
        cfg["mqtt"]["host"] = args.host

    pub = MqttPublisher(cfg)
    if not pub.connect(timeout=5.0):
        raise SystemExit(f"[servo_test] could not connect to broker "
                         f"{cfg.get('mqtt.host')}:{cfg.get('mqtt.port')} — "
                         f"is tools/setup_broker.ps1 running?")
    try:
        if args.interactive:
            run_interactive(pub)
        else:
            run_sequence(pub)
    finally:
        pub.publish_command("CENTER")
        time.sleep(0.3)
        pub.close()


if __name__ == "__main__":
    main()
