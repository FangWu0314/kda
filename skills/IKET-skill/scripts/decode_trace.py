#!/usr/bin/env python3
"""Decode in_kernel_trace.cuh captures to the exporter event schema (CPU only)."""
import argparse
import json
from pathlib import Path
import struct

EVENT = struct.Struct('<QQIHHII')
KINDS = {0: 'B', 1: 'E', 2: 'i'}


def decode(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != 'cuda-in-kernel-capture/v1':
        raise ValueError('Expected schema cuda-in-kernel-capture/v1')
    dims = [manifest[x] for x in ('num_ctas', 'warps_per_cta', 'capacity_per_warp')]
    if any(type(x) is not int or x <= 0 for x in dims) or dims[1] > 32:
        raise ValueError('Invalid allocation dimensions')
    ct, wp, cap = dims
    writers = ct * wp
    arrays = {}
    for field, size in [('events_file', writers * cap * EVENT.size),
                        ('counts_file', writers * 4), ('dropped_file', writers * 4)]:
        data = (path.parent / manifest[field]).read_bytes()
        if len(data) != size:
            raise ValueError(f'{field}: expected {size} bytes, got {len(data)}')
        arrays[field] = data
    counts = [v[0] for v in struct.iter_unpack('<I', arrays['counts_file'])]
    drops = [v[0] for v in struct.iter_unpack('<I', arrays['dropped_file'])]
    names = manifest['event_names']
    if not isinstance(names, list) or not all(isinstance(s, str) and s for s in names):
        raise ValueError('event_names must be a nonempty-name list')
    if len(names) != len(set(names)):
        raise ValueError('event_names must be unique')
    identity = {k: manifest[k] for k in ('rank', 'device', 'launch')}
    if any(type(v) is not int or v < 0 for v in identity.values()):
        raise ValueError('rank, device and launch must be nonnegative integers')
    events = []
    for writer, count in enumerate(counts):
        if count > cap:
            raise ValueError(f'Writer {writer}: count {count} exceeds capacity {cap}')
        for index in range(count):
            offset = (writer * cap + index) * EVENT.size
            stamp, span, event, kind, sm, cta, warp = EVENT.unpack_from(arrays['events_file'], offset)
            if cta * wp + warp != writer or cta >= ct or warp >= wp:
                raise ValueError(f'Writer {writer}: record identity disagrees with buffer position')
            if event >= len(names) or kind not in KINDS:
                raise ValueError(f'Writer {writer}: unknown event ID/kind')
            events.append(dict(identity, cta=cta, warp=warp, sm=sm, name=names[event],
                               kind=KINDS[kind], ts_ns=stamp, span=span))
    metadata = dict(manifest.get('metadata', {}))
    metadata['capture_manifest'] = str(path)
    metadata['trace_allocation'] = dict(zip(('num_ctas', 'warps_per_cta', 'capacity_per_warp'), dims))
    return {'schema': 'cuda-in-kernel-events/v1', 'metadata': metadata,
            'dropped_events': sum(drops), 'events': events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('output')
    args = parser.parse_args()
    try:
        manifest_path = Path(args.manifest)
        manifest = json.loads(manifest_path.read_text())
        inputs = [manifest_path] + [manifest_path.parent / manifest[key]
                                   for key in ('events_file', 'counts_file', 'dropped_file')]
        if Path(args.output).resolve() in {p.resolve() for p in inputs}:
            parser.error('output must not overwrite the manifest or binary inputs')
        result = decode(args.manifest)
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'decode error: {exc}\n')
    print(f"Decoded {len(result['events'])} events; dropped {result['dropped_events']}")


if __name__ == '__main__':
    main()
