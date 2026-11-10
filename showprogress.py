"""Replay a saved conversation, or follow a running task in another terminal."""

import argparse
import json
import time
from pathlib import Path

from workshop.trace import display


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path,
                        help="JSONL trace; defaults to the latest run")
    parser.add_argument("--follow", action="store_true", help="Wait for new events until the result")
    parser.add_argument("--tools", action="store_true", help="Show tools as well as the conversation")
    parser.add_argument("--delay", type=float, default=0, help="Seconds between displayed messages")
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    path = args.trace
    if path is None:
        latest = Path("runs/latest.txt")
        if not latest.exists():
            parser.error("No run yet. Run telecom.py or supply an examples/*.jsonl file.")
        path = latest.parent / latest.read_text(encoding="utf-8").strip()
    try:
        with path.open(encoding="utf-8") as stream:
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line or not line.endswith("\n"):
                    if not args.follow:
                        break
                    stream.seek(position)  # JSONL writes can be observed mid-line.
                    time.sleep(0.25)
                    continue
                event = json.loads(line)
                display(event, tools=args.tools)
                if event["event"] == "result":
                    break
                if event["event"] == "message" and args.delay:
                    time.sleep(args.delay)
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
