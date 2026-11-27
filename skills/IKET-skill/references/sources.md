# Sources and Format Differences

Checked on 2026-11-27. FlashInfer references are pinned to
`063b0454940d466dcf785cdf1272116989b42653`.

- [FlashInfer profiler C++ header](https://github.com/flashinfer-ai/flashinfer/blob/063b0454940d466dcf785cdf1272116989b42653/include/flashinfer/profiler.cuh)
- [FlashInfer Perfetto exporter](https://github.com/flashinfer-ai/flashinfer/blob/063b0454940d466dcf785cdf1272116989b42653/flashinfer/profiler/__init__.py)
- [FlashInfer profiler usage](https://github.com/flashinfer-ai/flashinfer/blob/063b0454940d466dcf785cdf1272116989b42653/profiler/README.md)
- [Trace formats supported by Perfetto](https://perfetto.dev/docs/getting-started/other-formats)
- [NVIDIA PTX ISA: special registers, globaltimer, tcgen05](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA IKET for CuTe DSL (Python) kernels](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/iket_profiling.html)

## Relation to FlashInfer

The method is FlashInfer's: explicit in-kernel events, a designated recording
thread, GPU timestamps, a device buffer, host decoding and a timeline view.
The recorder and exporter in this skill are an independent implementation
with their own format. FlashInfer is Apache-2.0.

| Aspect | FlashInfer (pinned version) | This skill |
|---|---|---|
| Timestamp | 32-bit `%globaltimer_lo` | 64-bit `%globaltimer` |
| Record | Packed tag + timestamp | Explicit CTA/warp/SM/event/span, 32 B per record |
| Synchronization | Block fence inside START/END | None added; completion endpoints use the kernel's own waits |
| Writers | Interleaved per group | Fixed-capacity region per warp, single writer |
| Async pairing | Begin/end events | Unique span; begin and end may come from different warps |
| Output | Native trace through `tg4perfetto` | Chrome Trace JSON from the standard library, overlapping spans on separate lanes |

The two binary formats are separate: decode each with its own decoder.

Two details of the pinned FlashInfer version to check before using its
multi-group or instant paths: `group_idx` is part of the write address while
the tag base carries the block ID and the Python decoder splits it as a
block-group ID, and the INSTANT path does not advance the write pointer the
way START/END do.

## What the Timeline Shows

- `%globaltimer` is a per-device time reference on a nanosecond scale. Its
  update granularity is coarser than 1 ns.
- A wait return on an mbarrier defines when software observed completion.
- Chrome Trace JSON requires duration slices on one thread track to nest, so
  the exporter places overlapping spans on separate lanes. A lane is a
  display row.
