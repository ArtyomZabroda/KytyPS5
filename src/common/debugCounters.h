#ifndef KYTY_COMMON_DEBUG_COUNTERS_H_
#define KYTY_COMMON_DEBUG_COUNTERS_H_

#include <atomic>
#include <chrono>
#include <cstdint>

namespace Common::DebugCounters {

static_assert(std::atomic<uint64_t>::is_always_lock_free && std::atomic<bool>::is_always_lock_free);

// Collect events only while the panel is visible. Relaxed, lock-free atomics keep updates
// cheap and safe in signal handlers. Rates come from comparing two snapshots.
enum class Counter {
	// GPU work.
	Draws,
	DrawInstances,
	IndirectDraws,
	Dispatches,
	QueueSubmits,
	EarlySubmits, // Of those, recorded work handed to an idle GPU between guest packets.
	GpuWaits,     // CPU blocked on a GPU timeline semaphore.
	GpuWaitNs,
	// GPU waits by call site, as count and duration pairs in GpuWaitSite order.
	GpuWaitOther,
	GpuWaitOtherNs,
	GpuWaitPresent, // Presentation waits for a frame or swapchain slot.
	GpuWaitPresentNs,
	GpuWaitStreamBuffer, // Stream buffer wraps onto memory the GPU still reads.
	GpuWaitStreamBufferNs,
	GpuWaitBufferReadback, // Synchronous buffer downloads and download retirement.
	GpuWaitBufferReadbackNs,
	GpuWaitColorMetadata, // DCC/CMASK clear-code readbacks.
	GpuWaitColorMetadataNs,
	GpuWaitFaultBuffer,
	GpuWaitFaultBufferNs,
	GpuWaitOcclusion, // Occlusion query results published to the guest.
	GpuWaitOcclusionNs,
	GpuWaitUnmap, // Guest unmaps of memory the GPU caches own.
	GpuWaitUnmapNs,
	GpuWaitGds, // End-of-pipe GDS reads.
	GpuWaitGdsNs,
	GpuWaitBackground, // The asynchronous callback thread, which does not block draws.
	GpuWaitBackgroundNs,
	PriorityCallbackWaits, // Callers blocked until asynchronous GPU callbacks finish.
	PriorityCallbackWaitNs,
	// Synchronous buffer readbacks by source: calls, downloads that waited on the GPU, bytes,
	// and wait time, in ReadbackSource order.
	ReadbackOtherCalls,
	ReadbackOtherDownloads,
	ReadbackOtherBytes,
	ReadbackOtherWaitNs,
	ReadbackReadFaultCalls, // Guest CPU reads of GPU-written memory that faulted.
	ReadbackReadFaultDownloads,
	ReadbackReadFaultBytes,
	ReadbackReadFaultWaitNs,
	ReadbackWriteCalls, // Guest writes and invalidations over GPU-written memory.
	ReadbackWriteDownloads,
	ReadbackWriteBytes,
	ReadbackWriteWaitNs,
	ReadbackIndirectArgsCalls, // Indirect dispatch arguments read for shared memory sizing.
	ReadbackIndirectArgsDownloads,
	ReadbackIndirectArgsBytes,
	ReadbackIndirectArgsWaitNs,
	ReadbackGpuThreadReadCalls, // GPU-thread reads of guest buffers, such as constants and heaps.
	ReadbackGpuThreadReadDownloads,
	ReadbackGpuThreadReadBytes,
	ReadbackGpuThreadReadWaitNs,
	ReadbackColorMetadataCalls,
	ReadbackColorMetadataDownloads,
	ReadbackColorMetadataBytes,
	ReadbackColorMetadataWaitNs,
	ReadbackRetireCalls, // Buffers retired with downloads still queued.
	ReadbackRetireDownloads,
	ReadbackRetireBytes,
	ReadbackRetireWaitNs,
	SyncGpuCommands,       // Commands other threads ran on the GPU thread and waited for.
	SyncGpuCommandQueueNs, // Time before the GPU thread started them.
	SyncGpuCommandNs,      // Total caller wait.
	// Synchronous readbacks that submit the recording command buffer, split by what they wait on.
	ReadbackSplitWaits,
	ReadbackSplitSubmitNs,     // Submitting the command buffer that holds the copy.
	ReadbackSplitBacklogNs,    // Waiting for submissions made before it.
	ReadbackSplitCurrentNs,    // Waiting for that command buffer itself.
	ReadbackSplitBacklogTicks, // Submissions the GPU had not finished when the readback began.
	// Guest memory the GPU thread reads directly on the CPU while it builds draws and dispatches.
	NumInstancesReads,         // Instance counts taken from the last indirect draw record.
	NumInstancesReadsGpuDirty, // Of those, records whose dword was written by the GPU.
	NumInstancesReadsOne,      // Of those, instance counts equal to one.
	RestartIndexScans,         // Index buffers scanned for a custom primitive restart value.
	RestartIndexScansGpuDirty, // Of those, index buffers holding GPU-written bytes.
	DispatchArgsReads,         // Indirect dispatch arguments read for thread dimensions.
	DispatchArgsReadsGpuDirty, // Of those, arguments written by the GPU.
	SrtWalkerBackingLoads,     // Faulting SRT walker loads of clean bytes, served from the backing.
	SrtWalkerReadbacks,        // Faulting SRT walker loads of GPU-written bytes, downloaded first.
	GpuFaults,                 // Guest writes to GPU-tracked memory.
	ShadersCompiled,
	ShaderCompileNs,
	PipelinesCreated,
	PipelineCreateNs,
	// Caches.
	TextureUploadBytes,
	TextureDownloadBytes,
	TextureEvictions,
	ColorMetadataReadbacks, // CPU blocked reading back GPU-written DCC/CMASK metadata.
	ColorMetadataReadbackNs,
	ColorMetadataChecks, // GPU clear-key checks of DCC/CMASK metadata.
	BufferUploadBytes,
	BufferDownloadBytes,
	BufferEvictions,
	BufferObtains,            // Guest buffer ranges prepared for draws and dispatches.
	BufferObtainStreamed,     // Of those, small CPU-written ranges copied to the stream buffer.
	BufferObtainUploads,      // Of those, ranges that uploaded CPU-written pages.
	BufferObtainWritten,      // Of those, ranges the shader writes.
	BufferObtainCleanReads,   // Of those, read-only ranges with nothing to upload.
	BufferObtainCleanRepeats, // Of those, ranges already prepared in the same command buffer.
	ImageHeapUnmaps,          // Guest unmaps that reach the image heaps.
	ImageHeapBuries,          // Draws and dispatches that bury image heap elements after unmaps.
	ImageHeapBuryChecks,      // Element-range overlap tests while burying.
	TextureUploads,           // Guest images uploaded to the host.
	TextureUploadsFromGpu,    // Of those, sourced from GPU-written buffer memory.
	TextureUploadsUnchanged,  // Guest bytes equal to the image's previous upload.
	TextureUploadUnchangedBytes,
	TextureUploadNewBytes, // Guest bytes that differ from the previous upload, or a first one.
	TextureUploadFromGpuBytes,
	TextureImagesCreated,
	TextureImagesFreed,
	TextureShapeLookups,    // Texture descriptors resolved for draws and dispatches.
	TextureShapeMisses,     // Of those, T# encodings whose image description was computed anew.
	GuestSparseReads,       // Guest memory copies through the direct-memory backing.
	GuestSparseReadMisses,  // Of those, copies that walked the backing mappings.
	ImageRegionLookups,     // Texture cache searches of the image page index.
	ImageRegionCacheHits,   // Of those, answered from a recent identical search.
	ImageRegionPages,       // Index pages those searches visited.
	ImageRegionOwnerChecks, // Image overlap tests on those pages, duplicates included.
	ImageRegionMatches,     // Distinct images the searches returned.
	FindImageCalls,
	FindImageCandidates,      // Overlapping images FindImage weighed.
	FindImageSameStart,       // Of those, images that start at the requested address.
	FindImageOverlapResolves, // Calls with no same-backing image, which resolved overlaps.
	// Presentation.
	FlipsRequested,
	FlipsRejected, // Guest flip queue was full.
	LateVblanks,   // The present thread fell behind the vblank rate.
	SwapchainRecreations,
	DroppedPresents,
	// Emulator.
	EmulatedInstructions, // Host CPU lacked an instruction the guest used.
	UnresolvedImportCalls,
	GuestFileReadBytes,
	GuestFileWriteBytes,
	AudioUnderruns,
	Count
};

// Which caller a GPU wait on this thread belongs to. The outermost scope wins, so a readback
// requested by a texture lookup stays attributed to that lookup.
enum class GpuWaitSite {
	Other,
	Present,
	StreamBuffer,
	BufferReadback,
	ColorMetadata,
	FaultBuffer,
	Occlusion,
	Unmap,
	Gds,
	Background,
	Count
};

static_assert(static_cast<int>(Counter::GpuWaitBackgroundNs) -
                      static_cast<int>(Counter::GpuWaitOther) + 1 ==
                  2 * static_cast<int>(GpuWaitSite::Count),
              "GpuWaitSite and its counters must stay in the same order");

// What a synchronous buffer readback on this thread was for. The outermost scope wins.
enum class ReadbackSource {
	Other,
	ReadFault,
	Write,
	IndirectArgs,
	GpuThreadRead,
	ColorMetadata,
	Retire,
	Count
};

static_assert(static_cast<int>(Counter::ReadbackRetireWaitNs) -
                      static_cast<int>(Counter::ReadbackOtherCalls) + 1 ==
                  4 * static_cast<int>(ReadbackSource::Count),
              "ReadbackSource and its counters must stay in the same order");

enum class Peak { ShaderCompileNs, PipelineCreateNs, Count };

namespace Detail {
inline std::atomic<bool>           g_enabled {false};
inline std::atomic<uint64_t>       g_counters[static_cast<int>(Counter::Count)] {};
inline std::atomic<uint64_t>       g_peaks[static_cast<int>(Peak::Count)] {};
inline thread_local GpuWaitSite    t_gpu_wait_site   = GpuWaitSite::Other;
inline thread_local ReadbackSource t_readback_source = ReadbackSource::Other;
} // namespace Detail

[[nodiscard]] inline bool Enabled() noexcept {
	return Detail::g_enabled.load(std::memory_order_relaxed);
}

inline void SetEnabled(bool enabled) noexcept {
	Detail::g_enabled.store(enabled, std::memory_order_relaxed);
}

inline void Add(Counter counter, uint64_t value = 1) noexcept {
	if (!Enabled()) {
		return;
	}
	Detail::g_counters[static_cast<int>(counter)].fetch_add(value, std::memory_order_relaxed);
}

class ScopedTimer final {
public:
	ScopedTimer(Counter count, Counter duration, Peak peak = Peak::Count)
	    : m_count(count), m_duration(duration), m_peak(peak),
	      m_start(Enabled() ? std::chrono::steady_clock::now()
	                        : std::chrono::steady_clock::time_point {}) {}
	~ScopedTimer() {
		if (m_start == std::chrono::steady_clock::time_point {} || !Enabled()) {
			return;
		}
		const auto elapsed =
		    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                              std::chrono::steady_clock::now() - m_start)
		                              .count());
		Add(m_count);
		Add(m_duration, elapsed);
		if (m_peak != Peak::Count) {
			auto& peak     = Detail::g_peaks[static_cast<int>(m_peak)];
			auto  previous = peak.load(std::memory_order_relaxed);
			while (previous < elapsed &&
			       !peak.compare_exchange_weak(previous, elapsed, std::memory_order_relaxed)) {
			}
		}
	}
	ScopedTimer(const ScopedTimer&)            = delete;
	ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
	Counter                               m_count;
	Counter                               m_duration;
	Peak                                  m_peak;
	std::chrono::steady_clock::time_point m_start;
};

[[nodiscard]] inline Counter GpuWaitCount(GpuWaitSite site) noexcept {
	return static_cast<Counter>(static_cast<int>(Counter::GpuWaitOther) +
	                            2 * static_cast<int>(site));
}

[[nodiscard]] inline Counter GpuWaitDuration(GpuWaitSite site) noexcept {
	return static_cast<Counter>(static_cast<int>(GpuWaitCount(site)) + 1);
}

[[nodiscard]] inline GpuWaitSite CurrentGpuWaitSite() noexcept {
	return Detail::t_gpu_wait_site;
}

// The last split readback wait on this thread, for logging by its caller.
struct ReadbackSplit {
	uint64_t backlog_ticks = 0;
	uint64_t submit_ns     = 0;
	uint64_t backlog_ns    = 0;
	uint64_t current_ns    = 0;
};

inline thread_local ReadbackSplit t_last_readback_split;

class ScopedGpuWaitSite final {
public:
	explicit ScopedGpuWaitSite(GpuWaitSite site) noexcept: m_previous(Detail::t_gpu_wait_site) {
		if (m_previous == GpuWaitSite::Other) {
			Detail::t_gpu_wait_site = site;
		}
	}
	~ScopedGpuWaitSite() { Detail::t_gpu_wait_site = m_previous; }
	ScopedGpuWaitSite(const ScopedGpuWaitSite&)            = delete;
	ScopedGpuWaitSite& operator=(const ScopedGpuWaitSite&) = delete;

private:
	GpuWaitSite m_previous;
};

enum class ReadbackStat { Calls, Downloads, Bytes, WaitNs };

[[nodiscard]] inline Counter ReadbackCounter(ReadbackSource source, ReadbackStat stat) noexcept {
	return static_cast<Counter>(static_cast<int>(Counter::ReadbackOtherCalls) +
	                            4 * static_cast<int>(source) + static_cast<int>(stat));
}

[[nodiscard]] inline ReadbackSource CurrentReadbackSource() noexcept {
	return Detail::t_readback_source;
}

class ScopedReadbackSource final {
public:
	explicit ScopedReadbackSource(ReadbackSource source) noexcept
	    : m_previous(Detail::t_readback_source) {
		if (m_previous == ReadbackSource::Other) {
			Detail::t_readback_source = source;
		}
	}
	~ScopedReadbackSource() { Detail::t_readback_source = m_previous; }
	ScopedReadbackSource(const ScopedReadbackSource&)            = delete;
	ScopedReadbackSource& operator=(const ScopedReadbackSource&) = delete;

private:
	ReadbackSource m_previous;
};

// Guest read faults on GPU-written memory, by the PM4 packet the GPU thread was running.
// Slots 0-255 are opcodes; the last two are the GPU thread outside a packet and other threads.
constexpr uint32_t READ_FAULT_SLOTS        = 258;
constexpr uint32_t READ_FAULT_GPU_OUTSIDE  = 256;
constexpr uint32_t READ_FAULT_OTHER_THREAD = 257;

namespace Detail {
inline std::atomic<uint64_t> g_read_fault_counts[READ_FAULT_SLOTS] {};
inline std::atomic<uint64_t> g_read_fault_ns[READ_FAULT_SLOTS] {};
inline thread_local uint32_t t_pm4_op = READ_FAULT_OTHER_THREAD;
} // namespace Detail

// The GPU thread marks the packet it runs, and READ_FAULT_GPU_OUTSIDE between packets.
inline void SetPm4Op(uint32_t op) noexcept {
	Detail::t_pm4_op = op;
}

[[nodiscard]] inline uint32_t CurrentPm4Op() noexcept {
	return Detail::t_pm4_op;
}

inline void AddReadFault(uint32_t slot, uint64_t ns) noexcept {
	if (!Enabled() || slot >= READ_FAULT_SLOTS) {
		return;
	}
	Detail::g_read_fault_counts[slot].fetch_add(1, std::memory_order_relaxed);
	Detail::g_read_fault_ns[slot].fetch_add(ns, std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t GetReadFaults(uint32_t slot) noexcept {
	return Detail::g_read_fault_counts[slot].load(std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t GetReadFaultNs(uint32_t slot) noexcept {
	return Detail::g_read_fault_ns[slot].load(std::memory_order_relaxed);
}

// Read faults by the draw phase the GPU thread was in. Phase names are string literals,
// compared by address. The table keeps the first phases that fault and drops the rest.
constexpr uint32_t           FAULT_PHASE_SLOTS = 16;
inline constexpr const char* NO_DRAW_PHASE     = "no draw phase";

namespace Detail {
inline std::atomic<const char*> g_fault_phase_names[FAULT_PHASE_SLOTS] {};
inline std::atomic<uint64_t>    g_fault_phase_counts[FAULT_PHASE_SLOTS] {};
inline std::atomic<uint64_t>    g_fault_phase_ns[FAULT_PHASE_SLOTS] {};
inline thread_local const char* t_draw_phase = nullptr;
} // namespace Detail

inline void SetDrawPhase(const char* phase) noexcept {
	Detail::t_draw_phase = phase;
}

[[nodiscard]] inline const char* CurrentDrawPhase() noexcept {
	return Detail::t_draw_phase != nullptr ? Detail::t_draw_phase : NO_DRAW_PHASE;
}

class ScopedDrawPhase final {
public:
	explicit ScopedDrawPhase(const char* phase) noexcept: m_previous(Detail::t_draw_phase) {
		Detail::t_draw_phase = phase;
	}
	~ScopedDrawPhase() { Detail::t_draw_phase = m_previous; }
	ScopedDrawPhase(const ScopedDrawPhase&)            = delete;
	ScopedDrawPhase& operator=(const ScopedDrawPhase&) = delete;

private:
	const char* m_previous;
};

inline void AddPhaseFault(const char* phase, uint64_t ns) noexcept {
	if (!Enabled()) {
		return;
	}
	for (uint32_t i = 0; i < FAULT_PHASE_SLOTS; i++) {
		const char* name = Detail::g_fault_phase_names[i].load(std::memory_order_acquire);
		if (name == nullptr) {
			const char* expected = nullptr;
			if (Detail::g_fault_phase_names[i].compare_exchange_strong(
			        expected, phase, std::memory_order_acq_rel, std::memory_order_acquire)) {
				name = phase;
			} else {
				name = expected;
			}
		}
		if (name == phase) {
			Detail::g_fault_phase_counts[i].fetch_add(1, std::memory_order_relaxed);
			Detail::g_fault_phase_ns[i].fetch_add(ns, std::memory_order_relaxed);
			return;
		}
	}
}

[[nodiscard]] inline const char* GetPhaseFaultName(uint32_t slot) noexcept {
	return Detail::g_fault_phase_names[slot].load(std::memory_order_acquire);
}

[[nodiscard]] inline uint64_t GetPhaseFaults(uint32_t slot) noexcept {
	return Detail::g_fault_phase_counts[slot].load(std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t GetPhaseFaultNs(uint32_t slot) noexcept {
	return Detail::g_fault_phase_ns[slot].load(std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t Get(Counter counter) noexcept {
	return Detail::g_counters[static_cast<int>(counter)].load(std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t Get(Peak peak) noexcept {
	return Detail::g_peaks[static_cast<int>(peak)].load(std::memory_order_relaxed);
}

} // namespace Common::DebugCounters

#endif /* KYTY_COMMON_DEBUG_COUNTERS_H_ */
