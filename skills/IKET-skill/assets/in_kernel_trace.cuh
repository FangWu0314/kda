#pragma once

// In-kernel event recorder. The record layout is decoded by
// scripts/decode_trace.py; background is in ../references/sources.md.
// Build with -DIKET_TRACE_ENABLED=1 to record. Without it, record() is empty.
#include <cuda_runtime.h>
#include <cstdint>

#ifndef IKET_TRACE_ENABLED
#define IKET_TRACE_ENABLED 0
#endif

namespace iket_trace {

enum class Kind : uint16_t { Begin = 0, End = 1, Instant = 2 };

// Binary layout consumed by scripts/decode_trace.py (little endian, 32 bytes).
struct alignas(16) Event {
    uint64_t ts_ns;
    uint64_t span;
    uint32_t event_id;
    uint16_t kind;
    uint16_t sm;
    uint32_t cta;
    uint32_t warp;
};
static_assert(sizeof(Event) == 32, "Trace event layout changed");

struct Buffer {
    Event* events;                // [num_ctas, warps_per_cta, capacity_per_warp]
    uint32_t* counts;              // [num_ctas, warps_per_cta], reset before launch
    uint32_t* dropped;             // same shape, reset before launch
    uint32_t capacity_per_warp;
    int32_t sample_cta;            // -1: all CTAs; otherwise one linear CTA ID
    uint32_t warp_mask;            // selected warp IDs in each CTA, max 32 warps
};

// Call at uniform warp-level boundaries. Exactly the warp's physical lane 0
// records; a call made only by lane 1 (etc.) intentionally records nothing.
// The same writer must not concurrently/reentrantly reuse its buffer slot.
__device__ __forceinline__ void record(const Buffer& b, uint32_t event_id,
                                       Kind kind, uint64_t span) {
#if IKET_TRACE_ENABLED
    const uint32_t thread = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const uint32_t warp = thread / 32;
    if (thread % 32 != 0 || !(b.warp_mask & (uint32_t(1) << warp))) return;
    const uint32_t cta = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    if (b.sample_cta >= 0 && cta != static_cast<uint32_t>(b.sample_cta)) return;
    const uint32_t warps = (blockDim.x * blockDim.y * blockDim.z + 31) / 32;
    const uint64_t writer = static_cast<uint64_t>(cta) * warps + warp;
    const uint32_t pos = b.counts[writer];
    if (pos >= b.capacity_per_warp) {
        b.dropped[writer] += 1;
        return;
    }
    uint64_t stamp;
    uint32_t sm;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamp) :: "memory");
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    b.events[writer * b.capacity_per_warp + pos] =
        Event{stamp, span, event_id, static_cast<uint16_t>(kind),
              static_cast<uint16_t>(sm), cta, warp};
    b.counts[writer] = pos + 1;
    // No device memory fence: the host reads only after kernel/stream completion.
    // These trace stores do not publish application data to another GPU thread.
#endif
}

}  // namespace iket_trace
