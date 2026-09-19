"""Local administration; no public key-management or spend-control endpoint."""

import argparse
import asyncio
import json
import os
import platform
import subprocess
import uuid
from decimal import Decimal
from pathlib import Path

from .auth import create_key
from .policy import KeyPolicy
from .storage.postgres import Postgres


def hardware():
    result = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "memory_bandwidth_bytes_per_second": None,
        "bandwidth_source": "unavailable",
    }
    if platform.system() == "Darwin":
        for field, name in (
            ("memory_bytes", "hw.memsize"),
            ("cpu", "machdep.cpu.brand_string"),
        ):
            value = subprocess.run(
                ["sysctl", "-n", name], check=True, capture_output=True, text=True
            ).stdout.strip()
            result[field] = int(value) if field == "memory_bytes" else value
    return result


async def run(args):
    if args.command == "hardware":
        print(json.dumps(hardware(), indent=2))
        return
    store = await Postgres.connect(os.environ["OMLX_GATEWAY_POSTGRES_DSN"])
    try:
        if args.command == "migrate":
            await store.migrate()
        elif args.command == "prune-traces":
            print(await store.prune_traces(args.batch_size))
        elif args.command == "policy":
            await store.put_policy(
                args.id, KeyPolicy.model_validate_json(Path(args.file).read_text())
            )
        elif args.command == "key":
            plaintext, digest = create_key(os.environ["OMLX_GATEWAY_PEPPER"].encode())
            key_id = str(uuid.uuid4())
            await store.put_key(key_id, digest, args.policy)
            print(json.dumps({"id": key_id, "key": plaintext}))
        elif args.command == "revoke":
            await store.revoke(args.id)
        elif args.command == "cloud":
            await store.cloud_switch(args.state == "on")
        elif args.command == "status":
            print(json.dumps(await store.runtime_status(), indent=2))
        elif args.command == "mode":
            await store.set_routing_mode(args.mode)
            if args.wait:
                async with asyncio.timeout(args.timeout):
                    while True:
                        status = await store.runtime_status()
                        if (
                            status.get("fresh")
                            and status.get("mode") == args.mode
                            and (args.mode == "auto" or status.get("hardware_released"))
                        ):
                            print(json.dumps(status, indent=2))
                            break
                        await asyncio.sleep(0.5)
            else:
                print(json.dumps({"requested_mode": args.mode}))
        elif args.command == "reconcile":
            await store.reconcile(args.id, Decimal(args.cost), args.tokens)
    finally:
        await store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate")
    prune = commands.add_parser("prune-traces")
    prune.add_argument("--batch-size", type=int, default=1000)
    commands.add_parser("hardware")
    commands.add_parser("status")
    mode = commands.add_parser("mode")
    mode.add_argument("mode", choices=["auto", "cloud_only"])
    mode.add_argument("--wait", action="store_true")
    mode.add_argument("--timeout", type=float, default=600)
    policy = commands.add_parser("policy")
    policy.add_argument("id")
    policy.add_argument("file")
    key = commands.add_parser("key")
    key.add_argument("--policy", required=True)
    revoke = commands.add_parser("revoke")
    revoke.add_argument("id")
    cloud = commands.add_parser("cloud")
    cloud.add_argument("state", choices=["on", "off"])
    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("id")
    reconcile.add_argument("--cost", required=True)
    reconcile.add_argument("--tokens", required=True, type=int)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
