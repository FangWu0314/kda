---
name: iket-skill
description: In-kernel event tracing for CUDA and CuTe C++ kernels. Instrument a kernel with GPU timestamps, export per-CTA/warp/stage timelines to Perfetto, and find pipeline dependencies, overlap and critical-path stalls. Use when asked to trace or time TMA, MMA, epilogue, reduction or GPU communication stages inside a kernel, or to explain where the time goes within one kernel launch.
---

# In-Kernel Event Tracing

Record timestamped events inside a running kernel, turn them into a Perfetto
timeline, and use that timeline to answer one specific performance question.
The recorder follows FlashInfer's approach: device-side event log, host
decode, Perfetto view. Background and format differences are in
[sources](references/sources.md).

`SKILL_DIR` below is the directory that contains this file.

## Workflow

1. **Read the current source.** Identify the call chain, the role of each
   warp, the existing barriers and waits, and the question the trace should
   answer. Take line numbers, register settings and machine details from the
   code as it is now.
2. **Write the event table first.** Follow the
   [instrumentation guide](references/instrumentation.md): for each event,
   list its name, begin and end location, recording warp, span ID and what
   the interval means. Start with a few representative CTAs and warps and
   state the sampling scope; widen it when the question is about cross-CTA
   tails.
3. **Add the recorder.** Include
   [`assets/in_kernel_trace.cuh`](assets/in_kernel_trace.cuh), or convert
   existing instrumentation to the
   [event format](references/trace-format.md). A project that already uses
   the FlashInfer profiler can keep its API.
4. **Build and capture.** Save the compile log (registers, stack, spills),
   source version, event-name mapping and sampling configuration with each
   capture. Follow the project's own measurement protocol for warmup, run
   count and cache state.
5. **Decode and export.**

   ```bash
   python3 "$SKILL_DIR/scripts/decode_trace.py" capture.json events.json
   python3 "$SKILL_DIR/scripts/export_perfetto.py" events.json trace.perfetto.json --summary trace-summary.json
   ```

   Open `trace.perfetto.json` at `https://ui.perfetto.dev/` with **Open trace
   file**. The output is Chrome Trace JSON; the scripts use only the Python
   standard library.
6. **Analyze.** Follow the
   [analysis guide](references/performance-analysis.md): check trace quality,
   then trace dependencies along the critical path. Report evidence,
   conclusions it supports, open questions and the next experiment.

To try the export without a GPU, use `$SKILL_DIR/assets/example-events.json`
as the input. It contains synthetic events.

## Measurement Rules

1. **Completion is observed at the kernel's existing wait.** An issue call
   that returns marks submission. Name the two intervals separately:
   `x.issue` covers the call, and `x.ready_observed` runs from before the
   issue to after the existing consumer wait returns. Keep the kernel's
   barriers, fences, flushes and waits exactly as they are.
2. **One time origin per (rank, device, launch).** `%globaltimer` values from
   different GPUs share no origin, so compare GPUs side by side, each in its
   own relative time.
3. **One begin and one end per (launch, event name, span).** Build `span`
   from the full task identity, such as CTA, task sequence and iteration.
   Stage slots and phase bits repeat and need the iteration added. Begin and
   end may be recorded by different warps.
4. **Quantify only complete traces.** The exporter fails on dropped events
   and on missing or duplicate endpoints. `--allow-incomplete` is for
   debugging the instrumentation.
5. **Intervals describe software stages.** Report overlap as overlap between
   software stages. Durations of concurrent warps are separate measurements;
   use the envelope or the union to describe a stage.
6. **Tracing changes the kernel.** Stores, register pressure and code layout
   all shift. Use the instrumented build to locate a bottleneck and the
   uninstrumented build for final numbers, on the same machine, workload and
   protocol.

## Deliverables

- Event table with source anchors, compile configuration, sampling scope,
  buffer capacity and dropped-event count.
- Raw events and one Perfetto trace per measured call, keeping rank, launch,
  CTA, warp and task identity.
- A report with the measurement protocol, the critical-path timeline of one
  real call, and recommendations ordered by evidence.
- The verification level reached: host export only, compiled, or captured on
  GPU.
