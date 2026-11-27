# Pipeline Timeline Analysis

How to turn recorded TMA, MMA, synchronization, communication and reduction
events into a bottleneck diagnosis that someone else can check against the
raw data. The events are software intervals; the analysis links them through
the kernel's dependencies.

## 1. Check the Trace First

Record these with the trace. When one is missing, name it and limit the
conclusions to what the remaining data covers.

| Check | What to record or verify |
|---|---|
| Build identity | Commit and dirty diff, compiler, CUDA and communication library versions, target architecture, the cubin actually loaded |
| Run configuration | Machine, GPU, rank, launch number, shape, routing, grid/block/cluster, sampled CTAs and warps |
| Event meaning | The source statement for each endpoint, and whether the mark sits before or after the call, wait or store |
| Capacity | Buffer capacity, records written, dropped count |
| Completeness | B/E pairing, coverage of active paths, event order, tasks that are legitimately empty |
| Clock | Timer register, bit width, unit, wrap handling, which ranks are on which GPU |
| Perturbation | Registers, spills, shared memory and added stores before and after instrumentation |

- Reset the log outside the timed interval and store each launch separately.
- Include only roles that actually ran. A warp with no task is left out of
  "earliest start" minimums.
- A stage with unpaired endpoints, overflow or an uncovered key role is
  incomplete: describe it qualitatively and fix the capture before
  quantifying it.
- A single sampled CTA supports conclusions about that CTA.
- Compare `setmaxnreg`, spills and scheduling in the SASS or compile log.
  Report added static spills.
- Report instrumented and uninstrumented timings side by side.

## 2. Time Domain and Statistics

- The key of a record is at least
  `(rank, GPU, launch, CTA, warp, event, task/stage/chunk)`, plus the full
  iteration number when stage numbers repeat. A transaction that crosses
  warps is linked by its span.
- Compute within one `(rank, GPU, launch)` first. Time zero is the earliest
  recorded kernel-entry event, the "observed entry".
- For communication between GPUs, use each side's own relative timeline and
  the local wait intervals.
- Store 64-bit timestamps. With 32-bit values, unwrap along one record stream
  using the known measurement length, and discard the capture when more than
  one wrap is possible.
- `clock64` is a per-SM cycle counter. Use one timer per trace.
- The trace store completes slightly after the endpoint it records. This
  matters near the kernel tail.
- When external timing uses another time domain and only durations are
  available, report the difference as unattributed.

Derive one row of metrics per call, then summarize with P50/P10/P90. State
whether the summary pools rank-calls, is per rank, or takes the slowest rank
per round. Sixteen ranks times five calls give eighty descriptive samples
from five repetitions. Report `median(end - begin)`. When showing a median
timeline, say that its endpoints can come from different calls, and use one
real call to explain causality.

## 3. Map Events to Dependencies

Map the kernel's actual events onto this table.

| Role / stage | Minimum endpoints | Meaning |
|---|---|---|
| Producer (TMA) | Stage reusable, issue, consumer wait returns | Issue is submission; the wait return is observed availability |
| MMA | Operand wait returns, issue/commit, result wait returns | Separates preparation, async execution and observed completion |
| Epilogue | Write-back end, done publish | The done publish covers every output tile the consumer needs |
| Input communication | PUT call before/after, incoming wait before/after | The call is the submission path; the wait is local observation of arrival |
| Metadata | Local count, remote count, index publish, barrier before/after | Separates computation, communication dependency and synchronization |
| Reduce | Dependency observed ready, reduce begin/end, output ready | Links "dependencies met" to "result usable" |
| Sender | Per chunk: ready query before/after, PUT before/after, flush before/after | Separates waiting for data, serial submission and completion |
| Final combine | Return-signal wait before/after, final add before/after | Separates waiting on the peer from the last computation |
| Tail | Exit mark per role, kernel start/end, full API start/end | Separates the last computation from kernel and API end |

Two marks around an async instruction give the issue interval. Take the
completion endpoint at the existing dependent wait: it is the moment the
consumer observed completion, and the data may have been ready earlier. In
CUTLASS `synclog`, some wait marks are placed before the wait, so check where
each mark sits in the
[barrier implementation](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/barrier.h).

## 4. Per-Call Metrics

All sets contain valid records from one time domain.

- **Stage envelope:** `span(S) = max(end_i) - min(begin_i)`. Includes stagger
  and gaps between executors.
- **Stage active union:** `active(S) = length(union([begin_i, end_i)))`.
  Counts parallel executors once; waits inside an interval are still
  included.
- **Observed overlap:** `overlap(A,B) = length(union(A) ∩ union(B))`.
- **Conditional activity:** `active(S ∩ W) / length(W)`. State the window W,
  the sampled roles and the definition of active.
- **Per-warp work:** `end - begin` for one warp. It differs from the
  cross-warp envelope when one CTA starts early.
- **Dependency ready time:** `required_done(x) = max(done(dep))` over the
  dependencies of x that are observed on the same GPU.
- **Scheduling delay:** `begin(x) - required_done(x)`, when the dependencies
  really gate x.
- **Gap after a stage:** `min(next_begin) - max(stage_end)`, when the next
  stage depends on all executors of the stage.

Name communication intervals by what was observed. `first PUT < last compute`
shows that a submission overlapped compute: call it "PUT submission overlaps
compute". `outgoing_put_before → incoming_ready` on one GPU includes the
progress difference between the two ends: call it the "exchange window".

## 5. Follow the Critical Path

Follow dependencies to the output: producer publishes → consumer observes →
eligible to run → actually begins → completes → next dependency. For each
large gap, answer two questions: which condition was it waiting for, and
would the end move earlier if the gap shrank? Overlapping stages are not
additive, and the longest stage can be fully hidden behind another path.

| Observation | Hypothesis to test | Evidence needed |
|---|---|---|
| Dependencies ready, but the owning warp is still busy elsewhere | The role starts too late | Ready observation, end of the warp's previous work, begin of the stage |
| Data ready, sender has not reached its chunk | Sender work or queue serialization | Chunk order, previous PUT return, next query/submit start |
| Long chunk wait | Not produced yet, or head-of-line blocking on an earlier slot | Slot-to-item map, ready time of each slot and of later slots |
| Everything ready, last PUT still late | Remaining serial submissions, queries or loop overhead | Sender intervals clipped at all-ready and summed |
| Long TMA consumer wait | Late issue, memory supply, or stage recycling | Issue time of the same stage, wait for a free slot, consumer arrival |
| Long return wait | Peer produced, submitted or completed late | The peer's own relative timeline and the local wait |
| Final step starts long after local work ends | Grid sync, peer contribution or scheduling | Grid and return-wait endpoints on the same warp |
| Last computation done, kernel still running | Cleanup, flush, waits or exit tail in other warps | Last event of every role and the real kernel interval |

For a sender, "ready" and "submitted" are different states. A single thread
that submits chunk by chunk handles the next chunk after the previous call
returns, even when every payload is ready. At `R = max(ready)` count the
PUTs that have returned, are in flight and have not started. Split
`[R, last_put_return)` into clipped PUT calls, ready queries and other gaps;
they come from one sequential stream, so they add up. With compacted slots, a
late slot at the head holds back later slots that are already ready.

## 6. Worked Example

Hypothetical data from one GPU and one launch, in µs.

- Compute ends at 100. The first PUT is `[82,90)`: submission starts 18
  before compute ends, with 8 of observed overlap.
- All rows are ready at R=120. Three more PUTs are `[114,126)`, `[130,138)`
  and `[141,145)`.
- At R, one PUT has returned, one is in flight and two have not started. The
  tail to the last PUT return is `145-120 = 25`.
- Clipped PUT time is `6+8+4 = 18`, the ready query `[126,129)` is 3, and the
  remaining gaps are `1+3 = 4`. So `25 = 18+3+4`. The rows were ready for all
  25; the 18 is PUT call time, which can include queue wait inside the call.
- Local work ends at 122, grid sync on one warp is `[122,124)`, return wait
  is `[124,150)` and the final add is `[150,154)`. The gap after local work
  is 28: grid 2, return wait 26.
- The latest exit mark of another warp is 158 and the kernel ends at 160:
  report an observed exit tail of 4 and an unattributed 2.

## 7. Comparison Rules

1. Compare on the same machine, shape and routing, communication library and
   configuration, and cache state. Record clock and power settings and other
   load.
2. Follow the project's measurement protocol for warmup count, measured-call
   count and cache state. Report the protocol with the numbers.
3. Capture trace and timing from the same measured calls where possible.
4. The instrumented build locates; the uninstrumented build gives the final
   number.
5. List main-kernel duration, CUDA-Event-bracketed API time and host wall
   time separately, and state which calls each interval contains.
6. Change one thing per comparison.

## 8. Report Template

```text
Subject:  source/patch summary; GPU/node; CUDA and comm library; shape/routing; sampled CTAs/warps
Protocol: warmup and measured calls; cache state; timed range; how rank-calls are summarized
Quality:  valid/missing/dropped events; clock width; role coverage; compile differences

Stage | endpoint definition (source) | per-call P50 [P10,P90] | envelope or per-warp
<one row per stage in the event table>

One real call: rank=... launch=..., observed entry = 0
<dependency chain with timestamps, from first input to last exit>
Sender after all-ready: returned / in flight / not started; PUT, query, other gaps

Confirmed:  facts backed by events and source.
Hypotheses: possible limits, and the evidence each still needs.
Next:       smallest next experiment; which endpoint should move; how to confirm without instrumentation.
Artifacts:  raw events, schema, source patch, compile log, analysis script, tables, Perfetto trace.
```
