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
	// --zero-copy-buffers: small CPU-written ranges answered from imported guest memory, and
	// command-stream writes that waited for the GPU to finish reading their pages.
	BufferObtainZeroCopy,
	ZeroCopyWriteWaits,
	// --threaded-recording: time the recording thread spends calling the driver, and waits for it
	// to catch up (before a submit or direct recording).
	RecorderBusyNs,
	RecorderWaits,
	RecorderWaitNs,
	// --parallel-draw-prep: draws the worker thread prepares before their commit.
	DrawPrepJobs,            // Draws it took.
	DrawPrepReady,           // Of those, prepared completely.
	DrawPrepSkipped,         // Of those, draws the commit drops or turns into special operations.
	DrawPrepSkippedEmpty,    // Of those, draws without vertices or instances.
	DrawPrepSkippedSpecial,  // Of those, metadata, resolve and depth-copy operations.
	DrawPrepSkippedShader,   // Of those, draws without a vertex shader address.
	DrawPrepSkippedTopology, // Of those, draws with an unsupported primitive type.
	DrawPrepMissing,         // Of those, draws needing a compile, tessellation or GPU calls.
	DrawPrepFaults,          // Of those, draws whose walker read GPU-tracked pages.
	DrawPrepMemory,          // Of those, draws whose specialization read GPU memory.
	DrawPrepHeap,            // Of those, draws using an image heap.
	DrawPrepNs,              // Worker time.
	DrawPrepUsed,            // Draws committed with the worker's preparation.
	DrawPrepStale,           // Prepared draws redone after GPU writes or compiles came between.
	DrawPrepStolen,          // Draws the GPU thread took before the worker started them.
	DrawPrepWaitNs,          // GPU thread time waiting for the worker.
	DrawPrepMismatches,      // --verify-draw-prep: worker results unlike the GPU thread's.
	DrawPrepHeapRedo,        // --parallel-heap-prep: prepared draws whose heap views changed.
	DrawPrepBufferCopies,    // --parallel-buffer-prep: buffer ranges the worker copied ahead.
	DrawPrepBufferHits,      // Of those, ranges a commit used.
	DrawPrepBufferFull,      // Ranges not copied because the worker's ring was full.
	DrawXframeChecks,        // --xframe-draw-stats: draws keyed.
	DrawXframeFull,          // Of those, keys with user data and arguments seen this or last frame.
	DrawXframeStructure,     // Of those, the same keys without user data and arguments.
	DrawXframeUserConfig,    // Structure keys with UserConfig seen this or last frame.
	DrawXframeArgs,          // Structure keys with the draw arguments seen this or last frame.
	DrawXframeStage,         // Structure keys with the stage registers seen this or last frame.
	DrawXframeUserData,      // Structure keys with the user SGPRs seen this or last frame.
	DrawXframeUdDraws,       // Repeats by arguments whose user SGPRs were compared.
	DrawXframeUdWords,       // Of those, user SGPRs that differ from the earlier repeat.
	DrawXframeUdPointers,    // Of those, words of a pair that addresses mapped guest memory.
	DrawXframeUdRelocatable, // Compared draws whose differing words are all such pointers.
	DrawXframeBindCompared,  // Repeats by arguments whose resource descriptors were compared.
	DrawXframeSameTextures,  // Of those, the same T# and S# words in every stage.
	DrawXframeSameBuffers,   // Of those, the same V# words in every stage.
	DrawXframeSameBoth,      // Of those, both.
	DrawXframeVsharpNoBase,  // Of those, the same V# words apart from the base address.
	DrawXframeVsharpShape,   // Of those, the same V# words apart from the base and the size.
	DrawXframeVsharpMoved,   // Of those, draws whose V# bases differ with the same V# count.
	DrawXframeVsharpShift,   // Of the moved, all bases moved by one and the same delta.
	DrawXframeRelocatable,   // Same T#/S# and the same V# apart from the base.
	DrawXframeSkipped,       // --xframe-skip-repeats: repeated draws not rendered.
	DrawRecordChecks,        // --draw-records-verify: draws looked up by programs and arguments.
	DrawRecordExcluded,      // Of those, draws a record would not cover.
	DrawRecordNew,           // Of those, no record yet.
	DrawRecordRegs,          // Of those, a record whose registers differ.
	DrawRecordTextures,      // Of those, the same registers but other T#/S# words.
	DrawRecordBuffers,       // Of those, V# words that differ beyond the base address.
	DrawRecordImages,        // Of those, all the same but the texture cache changed since.
	DrawRecordHits,          // Of those, draws the record would have prepared.
	DrawRecordWrong,         // Hits whose record differs from the draw's own preparation.
	DrawRecordWrongPrograms, // Of those, other shader modules.
	DrawRecordWrongTextures, // Of those, other images, views or samplers.
	DrawRecordWrongTargets,  // Of those, other color or depth targets.
	DrawRecordWrongPipeline, // Of those, another pipeline.
	DrawRecordCacheChanges,  // Texture cache generation changes seen between draws.
	DrawRecordRegsContext,   // Regs misses whose context registers differ.
	DrawRecordRegsUcfg,      // Regs misses whose UserConfig differs.
	DrawRecordRegsStage,     // Regs misses whose stage registers differ.
	DrawRecordRegsArgs,      // Regs misses whose draw arguments differ.
	DrawRecordHitsKept,      // Hits after a cache change, with every image still the same.
	DrawRecordWrongKept,     // Of those, records that differ from the draw's preparation.
	DrawRecordHitsOlder,     // Hits on the older of the two records of a key.
	DrawRecordHitsScissor,   // Hits whose screen scissor differs from the record's.
	DrawReplayChecks,        // --draw-records: draws looked up.
	DrawReplayMissRegs,      // Of those, no record with the same registers.
	DrawReplayMissResources, // Of those, other programs, T#/S# or V# words.
	DrawReplayMissImages,    // Of those, an image of the record is gone or replaced.
	DrawReplayHits,          // Of those, draws prepared from their record.
	DrawReplayStores,        // Records stored.
	DrawRecordMetaStable,    // Hits whose color clear stamps all still hold.
	DrawRecordMetaChanged,   // Hits with a recorded color clear stamp that no longer holds.
	DrawRecordMetaUnstamped, // Hits with a DCC/CMASK image that had no stamp when recorded.
	// Exclusive holds of the guest address-space lock, and the time each was held.
	AddressLockMaps,
	AddressLockMapNs,
	AddressLockUnmaps,
	AddressLockUnmapNs,
	AddressLockCommits,
	AddressLockCommitNs,
	AddressLockReleases,
	AddressLockReleaseNs,
	AddressLockProtects,
	AddressLockProtectNs,
	// Time spent waiting for the address-space lock: guest reads and tracker protections.
	AddressLockReadWaitNs,
	AddressLockProtectWaitNs,
	// Graphics draws and their shader stages compared with the previous draw.
	DrawSamePipeline,  // Same host pipeline.
	DrawSameTargets,   // Same color and depth target images.
	DrawSameAll,       // Same pipeline, targets, and every stage's bindings.
	StageBinds,        // Shader stages bound by draws.
	StageSameShader,   // Of those, the slot's previous stage ran the same shader.
	StageSameSnapshot, // And read the same descriptors, SRT and user data.
	StageSameImages,   // And bound the same image views.
	StageSameRanges,   // And the same guest buffer ranges.
	StageSameBuffers,  // And the same host buffer descriptors (handle, offset, range).
	StageSameData,     // And the same shader data dwords.
	StageSameAll,      // And the same images, buffers, data and samplers.
	// Where the GPU thread's wall time goes. Idle and blocked time sit between work items;
	// GPU waits happen inside PM4 processing or synchronous calls.
	GpuThreadIdleNs,    // No submission or call queued.
	GpuThreadBlockedNs, // Every queued submission waits on a memory condition.
	GpuThreadBlocks,    // Waits of that kind.
	GpuThreadPm4Ns,     // Processing guest command buffers.
	GpuThreadCallNs,    // Running calls other threads sent to the GPU thread.
	GpuThreadGpuWaitNs, // Of PM4 and call time, blocked on GPU timeline semaphores.
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
	SubmissionsStarted,        // Guest submissions the GPU thread took from its queues.
	SubmissionQueueNs, // Their time between the guest's submit and the GPU thread taking them.
	SubmissionBacklog, // Sum of submissions still queued behind each one taken.
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
	GpuFaultNs,                // Handler time of those, without the host's exception dispatch.
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
	BufferStreamBytes,        // Bytes those ranges copied.
	BufferStreamReuses,       // Of those ranges, ones answered with an earlier copy.
	BufferStreamReuseChecks,  // Earlier copies checked against the guest bytes.
	BufferStreamReuseMisses,  // Of those, copies whose bytes differed.
	BufferObtainUploads,      // Of those, ranges that uploaded CPU-written pages.
	BufferObtainWritten,      // Of those, ranges the shader writes.
	BufferObtainCleanReads,   // Of those, read-only ranges with nothing to upload.
	BufferObtainCleanRepeats, // Of those, ranges already prepared in the same command buffer.
	BufferObtainCleanHits,    // Read-only ranges answered from the clean read cache.
	BufferObtainCleanChecks,  // Clean read cache hits checked against the full path.
	BufferObtainCleanMisses,  // Of those, hits the full path answered differently.
	TrackingLockContended,    // Tracker and texture cache spin locks found held on entry.
	TrackingLockYields,       // Times their waiters gave the processor away.
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
	TextureShapeLookups,      // Texture descriptors resolved for draws and dispatches.
	TextureShapeMisses,       // Of those, T# encodings whose image description was computed anew.
	GuestSparseReads,         // Guest memory copies through the direct-memory backing.
	GuestSparseReadMisses,    // Of those, copies that walked the backing mappings.
	GuestSparseReadSplit,     // Of those, copies no single mapping held, so none was hinted.
	ImageRegionLookups,       // Texture cache searches of the image page index.
	ImageRegionCacheHits,     // Of those, answered from a recent identical search.
	ImageRegionPages,         // Index pages those searches visited.
	ImageRegionOwnerChecks,   // Image overlap tests on those pages, duplicates included.
	ImageRegionMatches,       // Distinct images the searches returned.
	FindImageCalls,           // Full lookups, not answered from the result cache.
	FindImageCandidates,      // Overlapping images FindImage weighed.
	FindImageSameStart,       // Of those, images that start at the requested address.
	FindImageOverlapResolves, // Calls with no same-backing image, which resolved overlaps.
	FindImageTextures,        // FindImage calls for sampled textures.
	FindImageStorage,         // FindImage calls for storage images.
	FindImageColorTargets,    // FindImage calls for color render targets.
	FindImageDepthTargets,    // FindImage calls for depth render targets.
	FindImageCacheHits,       // Calls answered from the FindImage result cache.
	FindImageVerifies,        // Cache hits checked against the full lookup's choice.
	FindImageMismatches,      // Of those, hits the full lookup would have answered differently.
	ColorClearMaterializes,   // FindImage calls on DCC/CMASK images that inspect the metadata.
	ColorClearGuestReads,     // Metadata slices read through the guest address.
	ColorClearGuestChecks,    // Calls that also read the backing to compare.
	ColorClearGuestMismatch,  // Of those, slices where guest and backing bytes differed.
	ColorClearStampHits,      // Calls skipped because the image's stamp still held.
	ColorClearStampVerifies,  // Of those, calls that ran the full path to compare.
	ColorClearStampMismatch,  // Of those, calls where the full path had work to do.
	ColorClearStampMisses,    // Calls that ran the full path with stamps on.
	ColorClearStampStale,     // Of those, calls whose stamp for the same inputs no longer held.
	ColorClearStampsStored,   // Stamps left by a full path that found nothing to do.
	ColorClearStampUploads,   // CPU-dirty metadata uploaded so that CPU writes are seen.
	ColorClearStampUntracked, // Full paths with nothing to do whose metadata spans regions.
	GraphicsPipelineBindsSkipped, // Draws whose pipeline was already bound in the command buffer.
	ImageBarriersMerged, // Image barriers skipped: the last one already covered the request.
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

// Adds the nanoseconds since `start`, a time taken while counters were enabled.
inline void AddSince(Counter counter, std::chrono::steady_clock::time_point start) noexcept {
	if (start == std::chrono::steady_clock::time_point {} || !Enabled()) {
		return;
	}
	Add(counter, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                       std::chrono::steady_clock::now() - start)
	                                       .count()));
}

// A start time for AddSince, or none while counters are off.
[[nodiscard]] inline std::chrono::steady_clock::time_point TimerStart() noexcept {
	return Enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
}

// Set once on the GPU thread, so shared code can attribute its time to that thread.
inline thread_local bool t_gpu_thread = false;

// F9: graphics draws return without rendering, to measure how fast the guest runs without them.
inline std::atomic<bool> g_skip_draws {false};

// Guest flips requested, counted always: frame boundaries for --xframe-draw-stats.
inline std::atomic<uint64_t> g_flips {0};

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

// GPU-thread time by draw phase, measured between phase changes while counters are enabled.
// Time outside any phase (between draws) is not counted.
constexpr uint32_t PHASE_TIME_SLOTS = 64;

namespace Detail {
inline std::atomic<const char*> g_phase_time_names[PHASE_TIME_SLOTS] {};
inline std::atomic<uint64_t>    g_phase_time_ns[PHASE_TIME_SLOTS] {};
inline thread_local int64_t     t_phase_start_ns = 0;
} // namespace Detail

inline void AddPhaseTime(const char* phase, uint64_t ns) noexcept {
	for (uint32_t i = 0; i < PHASE_TIME_SLOTS; i++) {
		const char* name = Detail::g_phase_time_names[i].load(std::memory_order_acquire);
		if (name == nullptr) {
			const char* expected = nullptr;
			if (Detail::g_phase_time_names[i].compare_exchange_strong(
			        expected, phase, std::memory_order_acq_rel, std::memory_order_acquire)) {
				name = phase;
			} else {
				name = expected;
			}
		}
		if (name == phase) {
			Detail::g_phase_time_ns[i].fetch_add(ns, std::memory_order_relaxed);
			return;
		}
	}
}

inline void SwitchDrawPhase(const char* next) noexcept {
	const char* previous = Detail::t_draw_phase;
	Detail::t_draw_phase = next;
	if (!t_gpu_thread || !Enabled()) {
		Detail::t_phase_start_ns = 0;
		return;
	}
	const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
	                        std::chrono::steady_clock::now().time_since_epoch())
	                        .count();
	if (previous != nullptr && Detail::t_phase_start_ns != 0 && now > Detail::t_phase_start_ns) {
		AddPhaseTime(previous, static_cast<uint64_t>(now - Detail::t_phase_start_ns));
	}
	Detail::t_phase_start_ns = now;
}

[[nodiscard]] inline const char* GetPhaseTimeName(uint32_t slot) noexcept {
	return Detail::g_phase_time_names[slot].load(std::memory_order_acquire);
}

[[nodiscard]] inline uint64_t GetPhaseTimeNs(uint32_t slot) noexcept {
	return Detail::g_phase_time_ns[slot].load(std::memory_order_relaxed);
}

inline void SetDrawPhase(const char* phase) noexcept {
	SwitchDrawPhase(phase);
}

[[nodiscard]] inline const char* CurrentDrawPhase() noexcept {
	return Detail::t_draw_phase != nullptr ? Detail::t_draw_phase : NO_DRAW_PHASE;
}

class ScopedDrawPhase final {
public:
	explicit ScopedDrawPhase(const char* phase) noexcept: m_previous(Detail::t_draw_phase) {
		SwitchDrawPhase(phase);
	}
	~ScopedDrawPhase() { SwitchDrawPhase(m_previous); }
	ScopedDrawPhase(const ScopedDrawPhase&)            = delete;
	ScopedDrawPhase& operator=(const ScopedDrawPhase&) = delete;

private:
	const char* m_previous;
};

// A part of the current draw phase, named "Phase/Part". Outside a draw it does nothing, so code
// that dispatches share can mark its parts without timing the dispatch as a draw.
class ScopedDrawSubPhase final {
public:
	explicit ScopedDrawSubPhase(const char* phase) noexcept: m_previous(Detail::t_draw_phase) {
		if (m_previous != nullptr) {
			SwitchDrawPhase(phase);
		}
	}
	~ScopedDrawSubPhase() {
		if (m_previous != nullptr) {
			SwitchDrawPhase(m_previous);
		}
	}
	ScopedDrawSubPhase(const ScopedDrawSubPhase&)            = delete;
	ScopedDrawSubPhase& operator=(const ScopedDrawSubPhase&) = delete;

private:
	const char* m_previous;
};

// Render passes the GPU thread ended, by the source location that ended them.
constexpr uint32_t RENDER_PASS_END_SLOTS = 32;

namespace Detail {
inline std::atomic<const char*> g_pass_end_files[RENDER_PASS_END_SLOTS] {};
inline std::atomic<uint32_t>    g_pass_end_lines[RENDER_PASS_END_SLOTS] {};
inline std::atomic<uint64_t>    g_pass_end_counts[RENDER_PASS_END_SLOTS] {};
} // namespace Detail

inline void CountRenderPassEnd(const char* file, uint32_t line) noexcept {
	if (!t_gpu_thread || !Enabled()) {
		return;
	}
	for (uint32_t i = 0; i < RENDER_PASS_END_SLOTS; i++) {
		const char* name = Detail::g_pass_end_files[i].load(std::memory_order_acquire);
		if (name == nullptr) {
			Detail::g_pass_end_lines[i].store(line, std::memory_order_relaxed);
			Detail::g_pass_end_files[i].store(file, std::memory_order_release);
			name = file;
		}
		if (name == file && Detail::g_pass_end_lines[i].load(std::memory_order_relaxed) == line) {
			auto& count = Detail::g_pass_end_counts[i];
			count.store(count.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
			return;
		}
	}
}

[[nodiscard]] inline const char* GetRenderPassEndFile(uint32_t slot) noexcept {
	return Detail::g_pass_end_files[slot].load(std::memory_order_acquire);
}

[[nodiscard]] inline uint32_t GetRenderPassEndLine(uint32_t slot) noexcept {
	return Detail::g_pass_end_lines[slot].load(std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t GetRenderPassEnds(uint32_t slot) noexcept {
	return Detail::g_pass_end_counts[slot].load(std::memory_order_relaxed);
}

// Time spent inside Vulkan entry points on the GPU thread (--vk-call-timing), in TSC ticks.
// Every wrapped entry point owns a slot; the same time is also split by the draw phase the call
// was made in. Only the GPU thread writes, so a plain load and store replaces a locked add.
constexpr uint32_t DRIVER_CALL_SLOTS  = 96;
constexpr uint32_t DRIVER_PHASE_SLOTS = 32;

namespace Detail {
inline std::atomic<const char*> g_driver_call_names[DRIVER_CALL_SLOTS] {};
inline std::atomic<uint64_t>    g_driver_call_ticks[DRIVER_CALL_SLOTS] {};
inline std::atomic<uint64_t>    g_driver_call_counts[DRIVER_CALL_SLOTS] {};
inline std::atomic<const char*> g_driver_phase_names[DRIVER_PHASE_SLOTS] {};
inline std::atomic<uint64_t>    g_driver_phase_ticks[DRIVER_PHASE_SLOTS] {};
inline std::atomic<uint64_t>    g_driver_timer_ticks {0};
inline std::atomic<uint64_t>    g_driver_tsc_frequency {0};
inline thread_local const char* t_driver_phase      = nullptr;
inline thread_local uint32_t    t_driver_phase_slot = DRIVER_PHASE_SLOTS;

inline void SingleWriterAdd(std::atomic<uint64_t>& value, uint64_t delta) noexcept {
	value.store(value.load(std::memory_order_relaxed) + delta, std::memory_order_relaxed);
}

inline uint32_t DriverPhaseSlot(const char* phase) noexcept {
	for (uint32_t i = 0; i < DRIVER_PHASE_SLOTS; i++) {
		const char* name = g_driver_phase_names[i].load(std::memory_order_acquire);
		if (name == nullptr) {
			g_driver_phase_names[i].store(phase, std::memory_order_release);
			return i;
		}
		if (name == phase) {
			return i;
		}
	}
	return DRIVER_PHASE_SLOTS;
}
} // namespace Detail

inline void RegisterDriverCall(uint32_t slot, const char* name) noexcept {
	Detail::g_driver_call_names[slot].store(name, std::memory_order_release);
}

inline void SetDriverTiming(uint64_t tsc_frequency, uint64_t timer_ticks) noexcept {
	Detail::g_driver_tsc_frequency.store(tsc_frequency, std::memory_order_relaxed);
	Detail::g_driver_timer_ticks.store(timer_ticks, std::memory_order_relaxed);
}

inline void AddDriverCall(uint32_t slot, uint64_t ticks) noexcept {
	Detail::SingleWriterAdd(Detail::g_driver_call_ticks[slot], ticks);
	Detail::SingleWriterAdd(Detail::g_driver_call_counts[slot], 1);
	const char* phase = Detail::t_draw_phase != nullptr ? Detail::t_draw_phase : NO_DRAW_PHASE;
	if (phase != Detail::t_driver_phase) {
		Detail::t_driver_phase      = phase;
		Detail::t_driver_phase_slot = Detail::DriverPhaseSlot(phase);
	}
	if (Detail::t_driver_phase_slot < DRIVER_PHASE_SLOTS) {
		Detail::SingleWriterAdd(Detail::g_driver_phase_ticks[Detail::t_driver_phase_slot], ticks);
	}
}

[[nodiscard]] inline const char* GetDriverCallName(uint32_t slot) noexcept {
	return Detail::g_driver_call_names[slot].load(std::memory_order_acquire);
}

[[nodiscard]] inline uint64_t GetDriverCallTicks(uint32_t slot) noexcept {
	return Detail::g_driver_call_ticks[slot].load(std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t GetDriverCallCount(uint32_t slot) noexcept {
	return Detail::g_driver_call_counts[slot].load(std::memory_order_relaxed);
}

[[nodiscard]] inline const char* GetDriverPhaseName(uint32_t slot) noexcept {
	return Detail::g_driver_phase_names[slot].load(std::memory_order_acquire);
}

[[nodiscard]] inline uint64_t GetDriverPhaseTicks(uint32_t slot) noexcept {
	return Detail::g_driver_phase_ticks[slot].load(std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t GetDriverTscFrequency() noexcept {
	return Detail::g_driver_tsc_frequency.load(std::memory_order_relaxed);
}

// Cost of the back-to-back TSC reads that bracket each call, included in every measured call.
[[nodiscard]] inline uint64_t GetDriverTimerTicks() noexcept {
	return Detail::g_driver_timer_ticks.load(std::memory_order_relaxed);
}

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

// Guest HLE calls by function. Names are string literals (usually __func__), compared by address.
// Each slot also keeps a mask of the threads that called it, sampled on every 64th call.
constexpr uint32_t HLE_CALL_SLOTS = 1024;

namespace Detail {
inline std::atomic<const char*> g_hle_call_names[HLE_CALL_SLOTS] {};
inline std::atomic<uint64_t>    g_hle_call_counts[HLE_CALL_SLOTS] {};
inline std::atomic<uint64_t>    g_hle_call_threads[HLE_CALL_SLOTS] {};
inline std::atomic<uint32_t>    g_hle_thread_bits {0};
inline thread_local uint32_t    t_hle_calls      = 0;
inline thread_local uint64_t    t_hle_thread_bit = 0;
} // namespace Detail

inline void CountHleCall(const char* name) noexcept {
	if (!Enabled()) {
		return;
	}
	const auto hash = static_cast<uint32_t>(
	    ((reinterpret_cast<uintptr_t>(name) >> 3u) * 0x9e3779b97f4a7c15ull) >> 40u);
	for (uint32_t probe = 0; probe < 32; probe++) {
		const auto  slot    = (hash + probe) % HLE_CALL_SLOTS;
		auto&       entry   = Detail::g_hle_call_names[slot];
		const char* current = entry.load(std::memory_order_acquire);
		if (current == nullptr) {
			const char* expected = nullptr;
			if (entry.compare_exchange_strong(expected, name, std::memory_order_acq_rel,
			                                  std::memory_order_acquire)) {
				current = name;
			} else {
				current = expected;
			}
		}
		if (current != name) {
			continue;
		}
		Detail::g_hle_call_counts[slot].fetch_add(1, std::memory_order_relaxed);
		if ((Detail::t_hle_calls++ & 63u) == 0) {
			if (Detail::t_hle_thread_bit == 0) {
				Detail::t_hle_thread_bit =
				    1ull << (Detail::g_hle_thread_bits.fetch_add(1, std::memory_order_relaxed) %
				             64u);
			}
			Detail::g_hle_call_threads[slot].fetch_or(Detail::t_hle_thread_bit,
			                                          std::memory_order_relaxed);
		}
		return;
	}
}

[[nodiscard]] inline const char* GetHleCallName(uint32_t slot) noexcept {
	return Detail::g_hle_call_names[slot].load(std::memory_order_acquire);
}

[[nodiscard]] inline uint64_t GetHleCalls(uint32_t slot) noexcept {
	return Detail::g_hle_call_counts[slot].load(std::memory_order_relaxed);
}

// Threads that called the slot since the previous take, from sampled calls.
[[nodiscard]] inline uint64_t TakeHleCallThreads(uint32_t slot) noexcept {
	return Detail::g_hle_call_threads[slot].exchange(0, std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t Get(Counter counter) noexcept {
	return Detail::g_counters[static_cast<int>(counter)].load(std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t Get(Peak peak) noexcept {
	return Detail::g_peaks[static_cast<int>(peak)].load(std::memory_order_relaxed);
}

} // namespace Common::DebugCounters

#endif /* KYTY_COMMON_DEBUG_COUNTERS_H_ */
