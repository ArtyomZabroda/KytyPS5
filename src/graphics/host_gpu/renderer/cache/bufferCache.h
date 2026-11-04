#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS = 14;
	static constexpr uint64_t CACHING_PAGESIZE = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES =
	    (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE = CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	// Changes whenever guest memory becomes GPU-written, so draws prepared on another thread
	// before it can tell they may have read stale bytes.
	[[nodiscard]] uint64_t GpuDirtyEpoch() const noexcept {
		return m_gpu_dirty_epoch.load(std::memory_order_acquire);
	}
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer*       GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	// Guest memory was written in command-stream order: earlier stream copies may be stale.
	void ForgetStreamedCopies() { ++m_stream_epoch; }
	// --zero-copy-buffers: a command-stream write at parse time waits for submitted work that
	// reads its pages from imported memory, and those pages stop being served from it.
	void NoteCommandWrite(uint64_t vaddr, uint64_t size);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// Color metadata the GPU found free of the clear keys in `keys` (bits of code >> 5) stays so
	// until a GPU write or download.
	void               MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys);
	[[nodiscard]] bool IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const;
	// Advances when a GPU write or download drops checked color metadata.
	[[nodiscard]] uint64_t ColorCheckGeneration() const {
		return m_color_check_generation.load(std::memory_order_acquire);
	}
	// The CPU- and GPU-dirty generations of the tracking region holding the range: zeros when the
	// range spans regions or its region is not tracked yet.
	[[nodiscard]] std::pair<uint64_t, uint64_t> DirtyGenerations(uint64_t vaddr,
	                                                             uint64_t size) const {
		return {m_memory_tracker.CpuDirtyGeneration(vaddr, size),
		        m_memory_tracker.GpuDirtyGeneration(vaddr, size)};
	}
	void ProcessFaultBuffer();
	void SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	// Queue after the range's state change becomes visible, or a drain could consume it first.
	void                   QueueBdaSync(uint64_t vaddr, uint64_t size);
	[[nodiscard]] RangeSet TakeBdaSync();
	void                   RunGarbageCollector();

	// --parallel-buffer-prep: small CPU-written ranges the draw prep worker copied ahead into its
	// own ring. The draw's commit uses a copy instead of the stream buffer while the range's
	// region keeps the CPU-dirty generation the worker saw.
	struct PrestreamCopy {
		uint64_t vaddr      = 0;
		uint64_t size       = 0;
		uint64_t offset     = 0;
		uint64_t generation = 0;
	};
	struct PrestreamDraw {
		std::vector<PrestreamCopy> copies;
		uint64_t                   entry = 0; // Its ring space, until the commit retires it.
	};
	[[nodiscard]] bool PrestreamEnabled() const noexcept { return m_prestream_ring != nullptr; }
	// Worker thread.
	void BeginPrestream(PrestreamDraw& draw);
	void Prestream(PrestreamDraw& draw, uint64_t vaddr, uint64_t size);
	// GPU thread: ObtainBuffer answers from the draw's copies until reset, and each committed
	// draw retires its space after its commands are recorded.
	void UsePrestream(const PrestreamDraw* draw) noexcept { m_prestream_draw = draw; }
	void RetirePrestream(PrestreamDraw& draw);

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const { return !id || m_slot_buffers[id].is_deleted; }

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void                   Unregister(BufferId id);
	template <bool insert>
	void                     ChangeRegister(BufferId id);
	void                     DeleteBuffer(BufferId id);
	[[nodiscard]] bool       SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                           bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void               ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size);
	void               ReportStreamedCopyMismatch(uint64_t vaddr, uint64_t size);

	GraphicContext&                                    m_graphics;
	CommandScheduler&                                  m_scheduler;
	FaultManager                                       m_fault_manager;
	Buffer                                             m_gds_buffer;
	Buffer                                             m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                         m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                          m_buffers;
	PageTable                                          m_page_table;
	RangeSet                                           m_gpu_modified_ranges;
	std::atomic<uint64_t>                              m_gpu_dirty_epoch {0};
	// One per TileManager::ColorClearSlots key slot.
	std::array<RangeSet, 8> m_checked_color_metadata;
	std::atomic<uint64_t>   m_color_check_generation {1};
	MemoryTracker           m_memory_tracker;
	std::mutex              m_bda_sync_mutex;
	RangeSet                m_bda_sync;
	StreamBuffer            m_staging_buffer;
	StreamBuffer            m_stream_buffer;
	StreamBuffer            m_download_buffer;
	StreamBuffer            m_device_buffer;
	TextureCache&           m_texture_cache;
	uint64_t                m_total_used_memory  = 0;
	uint64_t                m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t                m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t                m_gc_tick            = 0;
	// Read-only ranges ObtainBuffer found with nothing to upload. One stays valid while no
	// buffer is registered or unregistered and the CPU dirties no page of its tracking region.
	struct CleanRead {
		uint64_t vaddr              = 0;
		uint64_t size               = 0;
		uint64_t dirty_generation   = 0;
		uint64_t buffers_generation = 0;
		BufferId id {};
	};
	std::array<CleanRead, 4096> m_clean_reads {};
	uint64_t                    m_buffers_generation    = 1;
	uint32_t                    m_clean_read_hits       = 0;
	uint32_t                    m_clean_read_mismatches = 0;
	// Small CPU-written ranges copied to the stream buffer. A copy answers the same range again
	// within its command buffer while no shader and no command-stream write followed it; the
	// guest must not change memory that submitted work still reads.
	struct StreamedCopy {
		uint64_t vaddr  = 0;
		uint64_t size   = 0;
		uint64_t tick   = 0;
		uint64_t epoch  = 0;
		uint64_t offset = 0;
		// Set on every 16th copy: the guest bytes' hash, checked by the copy's next reuse.
		uint64_t hash  = 0;
		bool     check = false;
	};
	std::array<StreamedCopy, 4096> m_streamed_copies {};
	uint64_t                       m_stream_epoch             = 1;
	uint32_t                       m_streamed_copy_count      = 0;
	uint32_t                       m_streamed_copy_mismatches = 0;
	bool                           m_streamed_copy_reuse      = true;
	// --zero-copy-buffers: guest direct memory imported in chunks, which a background thread
	// imports on first use; the import pins the pages.
	struct ZeroCopyChunk;
	void               StartZeroCopy();
	void               StopZeroCopy();
	void               ZeroCopyImportThread(std::stop_token stop);
	[[nodiscard]] bool ObtainZeroCopy(uint64_t vaddr, uint64_t size,
	                                  std::pair<Buffer*, uint64_t>& result);

	std::vector<std::unique_ptr<ZeroCopyChunk>> m_zero_copy_chunks;
	std::mutex                                  m_zero_copy_mutex;
	std::condition_variable_any                 m_zero_copy_available;
	std::vector<uint32_t>                       m_zero_copy_requests;
	std::jthread                                m_zero_copy_thread;
	uint64_t                                    m_zero_copy_base        = 0;
	uint64_t                                    m_zero_copy_size        = 0;
	uint32_t                                    m_zero_copy_memory_type = 0;
	uint32_t                                    m_zero_copy_requested   = 0;
	bool                                        m_zero_copy             = false;

	// --parallel-buffer-prep: ring space in allocation order; the oldest entry's begin up to
	// m_prestream_head is in use.
	struct PrestreamEntry {
		uint64_t id        = 0;
		uint64_t begin     = 0;
		uint64_t end       = 0;
		uint64_t tick      = 0;
		bool     committed = false;
	};
	[[nodiscard]] bool AllocatePrestream(PrestreamDraw& draw, uint64_t size, uint64_t& offset);

	std::unique_ptr<Buffer>    m_prestream_ring;
	uint64_t                   m_prestream_alignment = 256;
	std::mutex                 m_prestream_mutex; // The entries and the head.
	std::deque<PrestreamEntry> m_prestream_entries;
	uint64_t                   m_prestream_head    = 0;
	uint64_t                   m_prestream_next_id = 1;
	const PrestreamDraw*       m_prestream_draw    = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
