# Instrumentation

## 1. Define Events Before Adding Timestamps

Read the kernel's producer/consumer structure, warp specialization, stage and
phase handling, and barrier protocol. Then write an event table. One
asynchronous stage usually needs three events: issue, wait, and observed
completion.

| Event | Begin | End | What the interval is |
|---|---|---|---|
| `tma.issue` | Before the async copy is issued | After the issue returns | Submission |
| `tma.ready_observed` | Before the matching issue | After the consumer's existing wait returns | Issue to observed availability |
| `mma.issue` | Before the tcgen05/WGMMA issue | After issue or commit returns | Submission; state whether commit is included |
| `mma.wait` | Before the epilogue's existing wait | After that wait returns | Time the consumer is blocked |
| `mma.ready_observed` | Before the matching issue | After the matching wait returns | Issue to observed completion |
| `epilogue.store` | Start of convert/scale/write-back | Last output store of this round | State whether conversion and publish are included |
| `comm.put_issue` | Before the PUT/send call | After the call returns | Submission, including any queue backpressure |
| `comm.wait` | Before the existing ready/signal wait | After that wait returns | Local wait for remote data |
| `reduce` | First read of the inputs | Last partial store | One reduction unit, such as a row or a chunk |
| `publish_ready` | Before the existing ready publish | After it | The publishing code; consumers observe it later |
| `kernel.path_exit` | Instant at each role's last control-flow exit | n/a | Software tail of that role |

Record stages, and add per-iteration events only when the question needs
them. Estimate the event count first. For a question about startup or tail,
the first and last tasks are enough. When sampling a subset of stages, keep
both the begin and the end of each sampled span, even when different warps
record them.

## 2. Two Integration Paths

### The project already uses the FlashInfer profiler

Keep `PROFILER_INIT`, `PROFILER_EVENT_START/END` and
`flashinfer.profiler.export_to_perfetto_trace(buffer, event_names, output_path)`
from one pinned version, and use the header and the Python decoder from that
same version. Its exporter needs `tg4perfetto`; see [sources](sources.md).

Before using its multi-group or instant paths, check four things in the
pinned version: tag encoding matches decoding, the write pointer advances on
every path, writes stay inside the buffer, and the 32-bit clock does not wrap
within the capture. Its START/END macros contain a block fence; count that as
instrumentation overhead.

### The kernel has no profiler

Include [`in_kernel_trace.cuh`](../assets/in_kernel_trace.cuh). Lane 0 of each
warp writes to its own region with a 64-bit `%globaltimer` stamp, an explicit
span and a dropped-event counter. It needs no FlashInfer and no change to the
communication backend.

The recorder adds no fence or barrier. `asm volatile` with a `memory` clobber
limits compiler reordering around the timestamp read, and arithmetic can
still be scheduled across it. For very short intervals, check the SASS. The
host reads the log after the kernel finishes.

```cpp
// Instrumented build only: -DIKET_TRACE_ENABLED=1
#include "in_kernel_trace.cuh"
using iket_trace::Kind;
enum : uint32_t { MMA_ISSUE = 0, MMA_READY_OBSERVED = 1, MMA_WAIT = 2 };

// cta and task_sequence come from the kernel's scheduler.
uint64_t span = (uint64_t(cta) << 32) | task_sequence;
iket_trace::record(trace, MMA_ISSUE, Kind::Begin, span);
iket_trace::record(trace, MMA_READY_OBSERVED, Kind::Begin, span);
issue_existing_mma_and_commit(...);          // unchanged
iket_trace::record(trace, MMA_ISSUE, Kind::End, span);

// ... later, possibly in the consumer warp, with the same span value ...
iket_trace::record(trace, MMA_WAIT, Kind::Begin, span);
existing_completion_wait(...);               // unchanged
iket_trace::record(trace, MMA_WAIT, Kind::End, span);
iket_trace::record(trace, MMA_READY_OBSERVED, Kind::End, span);
```

Span rules:

- Each `(launch, event name, span)` gets one begin and one end.
- When several warps wait on the same task, pick one warp as the completion
  observer, and put the warp ID into the span of each per-warp `wait`.
- In a 2-CTA cluster, choose which CTA owns the completion event and derive
  the span from a task identity both CTAs share.
- When a tile runs the K loop repeatedly, include the K iteration or the
  commit batch in the span. One hardware barrier covering a batch of MMAs is
  one span for that batch.

## 3. Buffer Sizing and Lifetime

Let `C = grid.x * grid.y * grid.z`, `W = ceil(threads_per_CTA / 32)` and
`K` = capacity per warp.

| Buffer | Shape | Bytes |
|---|---|---|
| `events` | `[C, W, K]`, 32 B each | `C*W*K*32` |
| `counts` | `[C, W]`, uint32 | `C*W*4` |
| `dropped` | `[C, W]`, uint32 | `C*W*4` |

With C=148, W=16, K=64, `events` is 4,849,664 B (4.625 MiB). A begin and an
end take one record each; size K for instants, branches and loop counts too.
Unsampled warps still get a region, so estimate memory first for large grids.

- Allocate the three buffers as ordinary device memory and pass an
  `iket_trace::Buffer` as an extra kernel argument. Update the host binding,
  the JIT signature and the cache key together.
- Before each call, zero `counts` and `dropped` on the same stream. `events`
  needs no clearing; decode only the first `counts[writer]` records.
- Keep allocation, zeroing and copy-back outside the timed interval, and
  reuse a buffer only after its kernel has finished.
- `sample_cta = -1` records every CTA; any other value selects one linear CTA
  ID. `warp_mask` selects physical warps. Call `record()` where lane 0 of the
  warp executes, because lane 0 is the writer.
- Each writer region belongs to one warp. Application synchronization stays
  on the kernel's own barriers.

## 4. Capture Checklist

- The kernel's issue order, barrier phases, fences, ready/credit protocol and
  buffer reuse are unchanged.
- The build keeps line info. Save the ptxas output and compare register
  count, `setmaxnreg` behaviour and spills against the uninstrumented build.
- Every rank runs the same binary and ABI. Per-rank filtering changes only
  which events are recorded.
- Under CUDA Graph replay, zero the counters before each replay or give each
  replay its own region, and let the host label each sample.
- After the stream completes, copy back and write the binaries and the
  capture manifest described in [trace-format.md](trace-format.md).
