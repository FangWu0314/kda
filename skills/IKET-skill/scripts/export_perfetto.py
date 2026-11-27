#!/usr/bin/env python3
"""Export cuda-in-kernel-events/v1 to Chrome Trace JSON accepted by Perfetto.

Pair by (rank, device, launch, name, span), not by the reporting warp. Each
GPU/launch has a separate origin; the exported processes are NOT clock aligned.
"""

import argparse
from collections import defaultdict
import heapq
import json
from pathlib import Path
import statistics
import sys


SCHEMA = "cuda-in-kernel-events/v1"
INTEGER_FIELDS = ("rank", "device", "launch", "cta", "warp", "sm", "ts_ns", "span")
CLOCK_NOTE = "Independent origin per (rank, device, launch); no cross-GPU or cross-launch clock alignment."


class InvalidTrace(ValueError):
    def __init__(self, summary):
        self.summary = summary
        super().__init__("; ".join(summary["diagnostics"]))


def process_key(event):
    return event["rank"], event["device"], event["launch"]


def attribution(event):
    return {**{k: event[k] for k in ("cta", "warp", "sm")},
            "ts_ns": str(event["ts_ns"]), "args": event.get("args", {})}


def export(document, allow_incomplete=False):
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("Expected schema " + SCHEMA)
    if not isinstance(document.get("events"), list):
        raise ValueError("events must be an array")
    if not isinstance(document.get("metadata", {}), dict):
        raise ValueError("metadata must be an object")
    dropped = document.get("dropped_events")
    if type(dropped) is not int or dropped < 0:
        raise ValueError("dropped_events must be a nonnegative integer")

    diagnostics, valid = [], []
    if dropped:
        diagnostics.append(f"Recorder dropped {dropped} events")
    for index, event in enumerate(document["events"]):
        problem = None
        if not isinstance(event, dict):
            problem = "event must be an object"
        elif any(type(event.get(k)) is not int or not 0 <= event[k] < 2**64
                 for k in INTEGER_FIELDS):
            problem = "rank/device/launch/cta/warp/sm/ts_ns/span must be uint64 integers"
        elif not isinstance(event.get("name"), str) or not event["name"]:
            problem = "name must be a nonempty string"
        elif event.get("kind") not in ("B", "E", "i"):
            problem = "kind must be B, E, or i"
        elif not isinstance(event.get("args", {}), dict):
            problem = "args must be an object"
        if problem:
            diagnostics.append(f"event[{index}]: {problem}")
        else:
            valid.append(event)

    pairs, instants, intervals = defaultdict(list), [], []
    for event in valid:
        pairs[process_key(event) + (event["name"], event["span"])].append(event)
    for key, events in pairs.items():
        if len(events) == 1 and events[0]["kind"] == "i":
            instants.append((events[0], None))
            continue
        begins = [e for e in events if e["kind"] == "B"]
        ends = [e for e in events if e["kind"] == "E"]
        problem = None
        if any(e["kind"] == "i" for e in events):
            problem = f"span {key!r}: instant span reused or mixed with B/E"
        elif len(begins) != 1 or len(ends) != 1:
            problem = f"span {key!r}: expected one B and one E, got {len(begins)} B / {len(ends)} E"
        elif ends[0]["ts_ns"] < begins[0]["ts_ns"]:
            problem = f"span {key!r}: negative duration"
        if problem:
            diagnostics.append(problem)
            # Keep the observed boundaries, never fabricate a duration or select
            # an arbitrary pair when a span was reused.
            instants.extend((event, problem) for event in events)
        else:
            intervals.append((begins[0], ends[0]))

    origins = {}
    for event in valid:
        key = process_key(event)
        origins[key] = min(origins.get(key, event["ts_ns"]), event["ts_ns"])
    durations = defaultdict(list)
    for begin, end in intervals:
        durations[process_key(begin) + (begin["name"],)].append(end["ts_ns"] - begin["ts_ns"])
    stats = []
    for (rank, device, launch, name), ns in sorted(durations.items()):
        stats.append({"rank": rank, "device": device, "launch": launch, "name": name,
                      "count": len(ns), "min_us": min(ns) / 1000,
                      "median_us": statistics.median(ns) / 1000,
                      "mean_us": statistics.mean(ns) / 1000, "max_us": max(ns) / 1000})
    summary = {
        "schema": SCHEMA, "complete": not diagnostics, "dropped_events": dropped,
        "input_events": len(document["events"]), "valid_events": len(valid),
        "complete_spans": len(intervals), "instant_or_incomplete_boundaries": len(instants),
        "diagnostics": diagnostics, "clock_note": CLOCK_NOTE,
        "duration_note": "Per-span observed duration; concurrent warp durations are not summed into wall time.",
        "processes": [{"rank": k[0], "device": k[1], "launch": k[2], "origin_ns": str(v)}
                      for k, v in sorted(origins.items())],
        "durations": stats,
    }
    if diagnostics and not allow_incomplete:
        raise InvalidTrace(summary)

    trace, process_ids, thread_ids = [], {}, {}
    for pid, key in enumerate(sorted(origins), 1):
        process_ids[key] = pid
        trace.append({"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
                      "args": {"name": f"rank {key[0]} / GPU {key[1]} / launch {key[2]} [independent clock]"}})

    def track(event, lane):
        key = process_key(event)
        identity = key + (event["cta"], event["warp"], event["name"], lane)
        if identity not in thread_ids:
            tid = len(thread_ids) + 1
            thread_ids[identity] = tid
            trace.append({"ph": "M", "name": "thread_name", "pid": process_ids[key], "tid": tid,
                          "args": {"name": f"CTA {event['cta']} / warp {event['warp']} / {event['name']} / lane {lane}"}})
        return process_ids[key], thread_ids[identity]

    # Greedy interval coloring produces disjoint tracks, including for crossed
    # asynchronous spans whose end is reported by a different warp or CTA.
    by_track = defaultdict(list)
    for begin, end in intervals:
        by_track[process_key(begin) + (begin["cta"], begin["warp"], begin["name"])].append((begin, end))
    for spans in by_track.values():
        active, free, next_lane = [], [], 0
        for begin, end in sorted(spans, key=lambda p: (p[0]["ts_ns"], p[1]["ts_ns"], p[0]["span"])):
            while active and active[0][0] <= begin["ts_ns"]:
                _, lane = heapq.heappop(active)
                heapq.heappush(free, lane)
            if free:
                lane = heapq.heappop(free)
            else:
                lane, next_lane = next_lane, next_lane + 1
            heapq.heappush(active, (end["ts_ns"], lane))
            pid, tid = track(begin, lane)
            delta = end["ts_ns"] - begin["ts_ns"]
            trace.append({"ph": "X", "name": begin["name"], "cat": "cuda-in-kernel",
                          "pid": pid, "tid": tid,
                          "ts": (begin["ts_ns"] - origins[process_key(begin)]) / 1000,
                          "dur": delta / 1000,
                          "args": {"span": str(begin["span"]), "duration_ns": str(delta),
                                   "begin": attribution(begin), "end": attribution(end)}})
    for event, problem in instants:
        pid, tid = track(event, 0)
        args = {"span": str(event["span"]), **attribution(event)}
        if problem:
            args.update(incomplete=True, original_kind=event["kind"], diagnostic=problem)
        trace.append({"ph": "i", "s": "t", "name": event["name"], "cat": "cuda-in-kernel",
                      "pid": pid, "tid": tid,
                      "ts": (event["ts_ns"] - origins[process_key(event)]) / 1000, "args": args})
    return {"traceEvents": trace, "displayTimeUnit": "ns",
            "otherData": {"metadata": document.get("metadata", {}), "summary": summary}}, summary


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or (
            args.summary and args.summary.resolve() in (args.input.resolve(), args.output.resolve())):
        parser.error("input, output, and summary must be different files")
    try:
        document = json.loads(args.input.read_text())
        trace, summary = export(document, args.allow_incomplete)
        write_json(args.output, trace)
        if args.summary:
            write_json(args.summary, summary)
        print(f"Exported {summary['complete_spans']} spans; complete={summary['complete']}")
        return 0
    except InvalidTrace as exc:
        if args.summary:
            write_json(args.summary, exc.summary)
        print(f"Refusing incomplete trace: {exc}. No trace was written.", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
