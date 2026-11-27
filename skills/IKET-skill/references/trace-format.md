# Capture, Export and Reading in Perfetto

## A. Host Side of the Reference Recorder

```cpp
const size_t writers = size_t(num_ctas) * warps_per_cta;
iket_trace::Buffer trace{};
trace.capacity_per_warp = capacity_per_warp;
trace.sample_cta = -1;             // or one linear CTA ID
trace.warp_mask = selected_warps;  // uint32_t bitmask
cudaMalloc(&trace.events, writers * capacity_per_warp * sizeof(iket_trace::Event));
cudaMalloc(&trace.counts, writers * sizeof(uint32_t));
cudaMalloc(&trace.dropped, writers * sizeof(uint32_t));

// Before each captured call, outside the timed interval:
cudaMemsetAsync(trace.counts, 0, writers * sizeof(uint32_t), stream);
cudaMemsetAsync(trace.dropped, 0, writers * sizeof(uint32_t), stream);
// timing_start -> original_kernel(..., trace) -> timing_end
// Wait for the stream, then copy back events/counts/dropped and write them
// unchanged to events.bin, counts.bin and dropped.bin.
// cudaFree the three buffers after the last capture.
```

Keep the original launch method, stream and cluster dimensions. Each captured
call gets its own set of files. One GPU buffer can be reused once its
copy-back is done; if copying back between calls would change the benchmark's
call rhythm, allocate one region per call and copy them all back at the end.

Binary layout, little endian, 32 B per record:
`uint64 timestamp, uint64 span, uint32 event_id, uint16 kind, uint16 sm,
uint32 cta, uint32 warp`. Kind 0/1/2 is begin/end/instant. `counts` and
`dropped` are little-endian uint32 arrays.

Save a `capture.json` next to each sample:

```json
{
  "schema": "cuda-in-kernel-capture/v1",
  "rank": 0, "device": 0, "launch": 0,
  "num_ctas": 148, "warps_per_cta": 16, "capacity_per_warp": 64,
  "events_file": "events.bin",
  "counts_file": "counts.bin",
  "dropped_file": "dropped.bin",
  "event_names": ["mma.issue", "mma.ready_observed", "mma.wait"],
  "metadata": {
    "instrumented": true,
    "source_commit": "<commit, plus saved dirty diff>",
    "sample_cta": -1,
    "warp_mask": "<bitmask>",
    "clock": "GPU globaltimer, nanosecond scale",
    "sample_kind": "measured",
    "span_encoding": "<how span maps to task/stage/iteration>"
  }
}
```

```bash
python3 "$SKILL_DIR/scripts/decode_trace.py" capture.json events.json
python3 "$SKILL_DIR/scripts/export_perfetto.py" events.json trace.perfetto.json --summary trace-summary.json
```

To keep the trace argument out of production code, add it under a
compile-time branch: the instrumented build has the parameter and the calls,
and the production build has neither.

## B. Event JSON Format

Existing instrumentation can skip the recorder and write this format
directly:

```json
{
  "schema": "cuda-in-kernel-events/v1",
  "metadata": {"instrumented": true, "source_commit": "<commit>"},
  "dropped_events": 0,
  "events": [
    {"rank":0,"device":0,"launch":0,"cta":3,"warp":0,"sm":12,
     "name":"mma.ready_observed","kind":"B","ts_ns":10000000000000000,"span":7,
     "args":{"stage":1,"iteration":7}},
    {"rank":0,"device":0,"launch":0,"cta":3,"warp":12,"sm":12,
     "name":"mma.ready_observed","kind":"E","ts_ns":10000000000001000,"span":7,
     "args":{"stage":1,"iteration":7}}
  ]
}
```

| Field | Meaning |
|---|---|
| `ts_ns` | Raw GPU timestamp as an integer. Keep it an integer end to end. |
| `rank`, `device`, `launch` | Together identify one clock group. |
| `cta`, `warp`, `sm` | Where the event was recorded. `cta` is the linear CTA ID. |
| `name` | Event name with a defined time meaning. |
| `kind` | `B`, `E` or lowercase `i`. |
| `span` | Transaction ID, unique within `(rank, device, launch, name)`. Begin and end share it, also across warps. |
| `args` | Optional object: expert, tile, stage, iteration, chunk, bytes, role. Write 64-bit values beyond 2^53 as strings. |
| `dropped_events` | Actual number of dropped events. When it is unknown, treat the capture as incomplete. |

The binary recorder stores only `span`. Record the span encoding in the
capture metadata, and use it after decoding to fill `args`. Example for a
fixed K-iteration count: low 32 bits `tile_sequence * K_iters + k_iter`, high
32 bits the CTA ID; recover the stage as `k_iter % pipeline_stages`. Kernels
with clusters, variable K or dynamic scheduling need their own shared task
identity.

When several ranks are merged into one JSON, keep each event's
rank/device/launch and sum the dropped counts.

## C. Exporter Behaviour

- Timestamps stay integers. Each `(rank, device, launch)` is shifted by its
  earliest recorded event, then converted to microseconds. That origin is the
  kernel start only when a kernel-entry event is recorded.
- B/E are paired by `(rank, device, launch, name, span)`.
- Output uses `X` complete slices. Overlapping spans of one CTA/warp/event go
  to separate lanes, so a lane is a display row; the track name and `args`
  give the owner.
- Raw timestamps and spans are kept as strings in `args`. JSON `ts`/`dur` are
  in microseconds.
- Dropped events, duplicate or missing endpoints and negative durations make
  the export fail with exit code 2. `--allow-incomplete` writes the trace
  anyway, marks the affected events and lists the problems in the summary.
- The summary reports the duration distribution of each event per clock
  group.

## D. Reading and Querying in Perfetto

Open `https://ui.perfetto.dev/` and load `trace.perfetto.json`. Pick one rank
and one measured sample, expand the CTA/warp/event tracks and check B/E
ownership and spans. Query slices from the SQL tab:

```sql
SELECT s.name, COUNT(*) AS n,
       MIN(s.dur)/1000.0 AS min_us,
       AVG(s.dur)/1000.0 AS mean_us,
       MAX(s.dur)/1000.0 AS max_us
FROM slice AS s
WHERE s.dur >= 0
GROUP BY s.name;
```

The JSON input is in microseconds; `slice.ts` and `slice.dur` in Perfetto SQL
are in nanoseconds. Filter by process or track first when the file holds
several ranks or launches. For dependencies, overlap and critical paths, see
[performance-analysis.md](performance-analysis.md).
