#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                          StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                          vk::BufferUsageFlags usage, const auto& write) {
	const auto [mapped, offset] = staging.Map(size, alignment);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	// Large transfers use the same one-time host allocation as ordinary buffer uploads.
	auto temporary =
	    std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0, usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset             = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto               command = m_scheduler.Current().Recorder();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = destination_offset;
		before.size                = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	m_buffers_generation++;
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages   = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset, addresses.data(),
		                addresses.size() * sizeof(vk::DeviceAddress));
		QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset, size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    ForgetColorMetadataChecks(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);

	auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped    = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Recorder();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.Handle(), static_cast<uint32_t>(copies.size()),
	                  copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	const auto copy_count = copies.size();
	auto publish = [this, mapped, offset, total_size, buffer_address, copies = std::move(copies),
	                owner = std::move(temporary)] {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
	};
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish), vaddr, size);
	} else {
		namespace Counters   = Common::DebugCounters;
		const auto source    = Counters::CurrentReadbackSource();
		const auto downloads = Counters::ReadbackCounter(source, Counters::ReadbackStat::Downloads);
		const auto wait_ns   = Counters::ReadbackCounter(source, Counters::ReadbackStat::WaitNs);
		Counters::Add(Counters::ReadbackCounter(source, Counters::ReadbackStat::Bytes), total_size);
		const auto tick                 = m_scheduler.CurrentTick();
		Counters::t_last_readback_split = {};
		{
			const Counters::ScopedGpuWaitSite wait_site(Counters::GpuWaitSite::BufferReadback);
			const Counters::ScopedTimer       timer(downloads, wait_ns);
			m_scheduler.Wait(tick);
			m_scheduler.WaitPriorityOperations(tick);
		}
		publish();
		static std::atomic<uint32_t> logged {0};
		if (Counters::Enabled() && logged.fetch_add(1, std::memory_order_relaxed) < 64) {
			const auto& split = Counters::t_last_readback_split;
			LOGF("Readback: op=0x%02x phase=%s source=%u addr=0x%016" PRIx64 " size=0x%" PRIx64
			     " bytes=%" PRIu64 " copies=%zu behind=%" PRIu64 " submit=%.2f earlier=%.2f "
			     "own=%.2f ms\n",
			     Counters::CurrentPm4Op(), Counters::CurrentDrawPhase(),
			     static_cast<uint32_t>(source), vaddr, size, total_size, copy_count,
			     split.backlog_ticks, static_cast<double>(split.submit_ns) / 1e6,
			     static_cast<double>(split.backlog_ns) / 1e6,
			     static_cast<double>(split.current_ns) / 1e6);
		}
	}
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	if (Config::ParallelBufferPrepEnabled() && Config::ParallelDrawPrepEnabled() &&
	    !Config::ZeroCopyBuffersEnabled()) {
		m_prestream_ring = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Stream, 0,
		                                            AllFlags, 64 * MiB);
		if (m_prestream_ring->Mapped().empty()) {
			m_prestream_ring.reset();
		}
		const auto& limits    = m_graphics.physical_device_properties.limits;
		m_prestream_alignment = std::max({uint64_t {limits.minUniformBufferOffsetAlignment},
		                                  uint64_t {limits.minStorageBufferOffsetAlignment},
		                                  uint64_t {limits.nonCoherentAtomSize}, uint64_t {16}});
	}
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	m_scheduler.BeginCommand();
	for (auto* buffer: {&m_bda_pagetable_buffer, m_fault_manager.GetFaultBuffer()}) {
		buffer->Fill(0, buffer->Size(), 0);
	}
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	StartZeroCopy();
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	StopZeroCopy();
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
	QueueBdaSync(vaddr, size);
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	namespace Counters = Common::DebugCounters;
	auto source        = Counters::CurrentReadbackSource();
	if (source == Counters::ReadbackSource::Other && is_write) {
		source = Counters::ReadbackSource::Write;
	}
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write, source] {
		const Counters::ScopedReadbackSource readback_source(source);
		Counters::Add(Counters::ReadbackCounter(source, Counters::ReadbackStat::Calls));
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];

		if (DownloadBufferMemory<false>(buffer, vaddr, size)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			QueueBdaSync(vaddr, size);
		}
	});
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice
			// versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE
				                                     ? LOWER_ADDRESS_SIZE
				                                     : LibKernel::Memory::kExtendedMemoryBase +
				                                           LibKernel::Memory::kExtendedMemorySize) -
				                                    end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end     = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr              = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

namespace {

// Guest bytes of a stream copy that records or checks its hash; streamed ranges fit in one
// caching page.
thread_local std::array<uint8_t, BufferCache::CACHING_PAGESIZE> t_streamed_bytes;

// Set when SynchronizeBuffer records an upload, so ObtainBuffer can count clean ranges.
thread_local bool t_synchronize_uploaded = false;

// Recent ranges ObtainBuffer prepared, by command buffer, to count repeated clean reads.
struct ObtainedRange {
	uint64_t vaddr = 0;
	uint64_t size  = 0;
	uint64_t tick  = 0;
};
constexpr size_t                                             OBTAINED_RANGE_SLOTS = 1024;
thread_local std::array<ObtainedRange, OBTAINED_RANGE_SLOTS> t_obtained_ranges {};

// Times parts of ObtainBuffer only for draw bindings, so vertex buffers keep their own phase.
class BindingBufferPhase final {
public:
	BindingBufferPhase()
	    : m_active(std::strcmp(Common::DebugCounters::CurrentDrawPhase(),
	                           "PrepareGraphicsBindings/Buffers") == 0) {}
	void Switch(const char* part) {
		m_phase.reset();
		if (m_active) m_phase.emplace(part);
	}
	void Reset() { m_phase.reset(); }

private:
	bool                                                     m_active;
	std::optional<Common::DebugCounters::ScopedDrawSubPhase> m_phase;
};

bool RepeatsInCommandBuffer(uint64_t vaddr, uint64_t size, uint64_t tick) {
	const auto hash   = (vaddr ^ (size * 0x9e3779b97f4a7c15ull)) * 0xff51afd7ed558ccdull;
	auto&      entry  = t_obtained_ranges[(hash >> 32u) % OBTAINED_RANGE_SLOTS];
	const bool repeat = entry.vaddr == vaddr && entry.size == size && entry.tick == tick;
	entry             = {vaddr, size, tick};
	return repeat;
}

} // namespace

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
	if (source) {
		t_synchronize_uploaded = true;
		auto& command          = m_scheduler.Current();
		command.EndRendering();
		const auto              native = command.Recorder();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(
		    vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
		    vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(
		    vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
		    vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);

	const auto source = UploadMemory(
	    m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		    for (auto& copy: copies) {
			    const auto address = buffer.CpuAddress() + copy.dstOffset;
			    if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset,
			                                             copy.size)) {
				    EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			    }
			    copy.srcOffset += base_offset;
		    }
	    });
	return source.first->Handle();
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}
	using Common::DebugCounters::Counter;
	const bool counting = Common::DebugCounters::Enabled();
	if (counting) {
		Common::DebugCounters::Add(Counter::BufferObtains);
		if (is_written) {
			Common::DebugCounters::Add(Counter::BufferObtainWritten);
		}
	}
	if (is_written) {
		ForgetStreamedCopies();
	}

	// Read before the checks below: a CPU write after this point advances the generation, so a
	// range recorded with it is never answered from the cache after that write.
	CleanRead* clean            = nullptr;
	uint64_t   dirty_generation = 0;
	BufferId   cached_id {};
	if (!is_written && !is_texel_buffer) {
		dirty_generation = m_memory_tracker.CpuDirtyGeneration(vaddr, size);
		const auto hash  = (vaddr ^ (size * 0x9e3779b97f4a7c15ull)) * 0xff51afd7ed558ccdull;
		clean            = &m_clean_reads[(hash >> 32u) % m_clean_reads.size()];
		if (dirty_generation != 0 && clean->vaddr == vaddr && clean->size == size &&
		    clean->dirty_generation == dirty_generation &&
		    clean->buffers_generation == m_buffers_generation) {
			// Every 16th hit takes the full path and compares.
			if ((++m_clean_read_hits & 15u) != 0) {
				Common::DebugCounters::Add(Counter::BufferObtainCleanHits);
				auto& buffer = m_slot_buffers[clean->id];
				TouchBuffer(buffer);
				return {&buffer, buffer.Offset(vaddr)};
			}
			Common::DebugCounters::Add(Counter::BufferObtainCleanChecks);
			cached_id = clean->id;
		}
	}
	const auto report_mismatch = [&](const char* reason, BufferId found) {
		Common::DebugCounters::Add(Counter::BufferObtainCleanMisses);
		if (m_clean_read_mismatches < 64) {
			++m_clean_read_mismatches;
			LOGF_COLOR(Log::Color::BrightRed,
			           "BufferCache: clean read cache mismatch #%u (%s): 0x%016" PRIx64
			           "+0x%" PRIx64 " cached id=%u found id=%u\n",
			           m_clean_read_mismatches, reason, vaddr, size, cached_id.index, found.index);
		}
	};

	BindingBufferPhase phase;
	if (!is_written && size <= CACHING_PAGESIZE &&
	    m_memory_tracker.IsRegionOnlyCpuModified(vaddr, size)) {
		phase.Switch("PrepareGraphicsBindings/BStream");
		const auto streamed = [&](uint64_t offset) -> std::pair<Buffer*, uint64_t> {
			Common::DebugCounters::Add(Counter::BufferObtainStreamed);
			if (cached_id) {
				report_mismatch("streamed", {});
			}
			if (clean != nullptr && clean->vaddr == vaddr && clean->size == size) {
				*clean = {};
			}
			return {&m_stream_buffer, offset};
		};
		if (m_zero_copy && !is_texel_buffer) {
			std::pair<Buffer*, uint64_t> zero_copy {};
			if (ObtainZeroCopy(vaddr, size, zero_copy)) {
				Common::DebugCounters::Add(Counter::BufferObtainZeroCopy);
				if (cached_id) {
					report_mismatch("zero-copy", {});
				}
				if (clean != nullptr && clean->vaddr == vaddr && clean->size == size) {
					*clean = {};
				}
				return zero_copy;
			}
		}
		if (m_prestream_draw != nullptr) {
			const auto& copies = m_prestream_draw->copies;
			const auto  copy   = std::ranges::find_if(copies, [&](const PrestreamCopy& candidate) {
				return candidate.vaddr == vaddr && candidate.size == size;
			});
			if (copy != copies.end() &&
			    copy->generation == m_memory_tracker.CpuDirtyGeneration(vaddr, size)) {
				Common::DebugCounters::Add(Counter::DrawPrepBufferHits);
				if (cached_id) {
					report_mismatch("prestreamed", {});
				}
				if (clean != nullptr && clean->vaddr == vaddr && clean->size == size) {
					*clean = {};
				}
				return {m_prestream_ring.get(), copy->offset};
			}
		}
		const auto read_guest = [&](void* destination) {
			if (!LibKernel::Memory::CopySparseMemory(vaddr, destination, size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
		};
		// The stream buffer reuses no space its current command buffer allocated.
		const auto tick      = m_scheduler.CurrentTick();
		const auto slot_hash = (vaddr ^ (size * 0x9e3779b97f4a7c15ull)) * 0xff51afd7ed558ccdull;
		auto&      copy      = m_streamed_copies[(slot_hash >> 32u) % m_streamed_copies.size()];
		if (m_streamed_copy_reuse && copy.vaddr == vaddr && copy.size == size &&
		    copy.tick == tick && copy.epoch == m_stream_epoch) {
			bool same = true;
			if (copy.check) {
				// The stream buffer may be uncached device memory, so the guest bytes are
				// compared by hash rather than read back.
				Common::DebugCounters::Add(Counter::BufferStreamReuseChecks);
				read_guest(t_streamed_bytes.data());
				same       = XXH3_64bits(t_streamed_bytes.data(), size) == copy.hash;
				copy.check = false;
				if (!same) {
					ReportStreamedCopyMismatch(vaddr, size);
				}
			}
			if (same) {
				Common::DebugCounters::Add(Counter::BufferStreamReuses);
				return streamed(copy.offset);
			}
		}
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			const bool check = (++m_streamed_copy_count & 15u) == 0;
			uint64_t   hash  = 0;
			if (check) {
				read_guest(t_streamed_bytes.data());
				hash = XXH3_64bits(t_streamed_bytes.data(), size);
				std::memcpy(mapped, t_streamed_bytes.data(), size);
			} else {
				read_guest(mapped);
			}
			m_stream_buffer.Commit();
			Common::DebugCounters::Add(Counter::BufferStreamBytes, size);
			copy = {vaddr, size, tick, m_stream_epoch, offset, hash, check};
			return streamed(offset);
		}
	}

	phase.Reset();
	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		phase.Switch("PrepareGraphicsBindings/BFind");
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	t_synchronize_uploaded = false;
	phase.Switch("PrepareGraphicsBindings/BSync");
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	phase.Reset();
	if (counting) {
		if (t_synchronize_uploaded) {
			Common::DebugCounters::Add(Counter::BufferObtainUploads);
		} else if (!is_written && !is_texel_buffer) {
			Common::DebugCounters::Add(Counter::BufferObtainCleanReads);
			if (RepeatsInCommandBuffer(vaddr, size, m_scheduler.CurrentTick())) {
				Common::DebugCounters::Add(Counter::BufferObtainCleanRepeats);
			}
		}
	}
	if (is_written) {
		if (!m_gpu_modified_ranges.Contains(vaddr, size)) {
			m_gpu_dirty_epoch.fetch_add(1, std::memory_order_release);
		}
		m_gpu_modified_ranges.Add(vaddr, size);
		ForgetColorMetadataChecks(vaddr, size);
	}
	if (cached_id) {
		if (cached_id != id) {
			report_mismatch("different buffer", id);
		} else if (t_synchronize_uploaded) {
			report_mismatch("uploaded", id);
		}
	}
	// After this call the range is clean: anything CPU-dirty was just uploaded.
	if (clean != nullptr && dirty_generation != 0) {
		*clean = {vaddr, size, dirty_generation, m_buffers_generation, id};
	}
	return {&buffer, buffer.Offset(vaddr)};
}

void BufferCache::BeginPrestream(PrestreamDraw& draw) {
	draw.copies.clear();
	draw.entry = 0;
}

bool BufferCache::AllocatePrestream(PrestreamDraw& draw, uint64_t size, uint64_t& offset) {
	const auto             capacity = m_prestream_ring->Size();
	const std::scoped_lock lock {m_prestream_mutex};

	auto& entries = m_prestream_entries;
	while (!entries.empty() && entries.front().committed &&
	       m_scheduler.IsFree(entries.front().tick)) {
		entries.pop_front();
	}
	auto start = Common::AlignUp(m_prestream_head, m_prestream_alignment);
	if (entries.empty()) {
		if (start > capacity || size > capacity - start) {
			start = 0;
		}
	} else if (const auto tail = entries.front().begin; m_prestream_head >= tail) {
		// In use: [tail, head).
		if (start > capacity || size > capacity - start) {
			if (size >= tail) {
				return false;
			}
			start = 0;
		}
	} else if (start >= tail || size >= tail - start) {
		// In use: [tail, capacity) and [0, head).
		return false;
	}
	offset           = start;
	m_prestream_head = start + size;
	if (draw.entry == 0) {
		draw.entry = m_prestream_next_id++;
		entries.push_back({.id = draw.entry, .begin = start, .end = m_prestream_head});
	} else {
		entries.back().end = m_prestream_head;
	}
	return true;
}

void BufferCache::Prestream(PrestreamDraw& draw, uint64_t vaddr, uint64_t size) {
	using Common::DebugCounters::Counter;
	if (m_prestream_ring == nullptr || size > CACHING_PAGESIZE ||
	    !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	for (const auto& copy: draw.copies) {
		if (copy.vaddr == vaddr && copy.size == size) {
			return;
		}
	}
	// Read before the checks: a CPU write after this point advances the generation.
	const auto generation = m_memory_tracker.CpuDirtyGeneration(vaddr, size);
	if (generation == 0 || !m_memory_tracker.IsRegionOnlyCpuModified(vaddr, size)) {
		return;
	}
	const auto reserved = Common::AlignUp(size, m_prestream_alignment);
	uint64_t   offset   = 0;
	if (!AllocatePrestream(draw, reserved, offset)) {
		Common::DebugCounters::Add(Counter::DrawPrepBufferFull);
		return;
	}
	if (!LibKernel::Memory::CopySparseMemory(vaddr, m_prestream_ring->Mapped().data() + offset,
	                                         size)) {
		return;
	}
	m_prestream_ring->Flush(offset, reserved);
	draw.copies.push_back({vaddr, size, offset, generation});
	Common::DebugCounters::Add(Counter::DrawPrepBufferCopies);
}

void BufferCache::RetirePrestream(PrestreamDraw& draw) {
	if (draw.entry == 0) {
		return;
	}
	const auto             tick = m_scheduler.CurrentTick();
	const std::scoped_lock lock {m_prestream_mutex};
	for (auto& entry: m_prestream_entries) {
		if (entry.id == draw.entry) {
			entry.committed = true;
			entry.tick      = tick;
			break;
		}
	}
	draw.entry = 0;
	draw.copies.clear();
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	return UploadMemory(
	    m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		    if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size)) {
			    EXIT("BufferCache: failed to read mapped guest image backing\n");
		    }
	    });
}

void BufferCache::ReportStreamedCopyMismatch(uint64_t vaddr, uint64_t size) {
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferStreamReuseMisses);
	m_streamed_copy_reuse = false;
	if (m_streamed_copy_mismatches < 64) {
		++m_streamed_copy_mismatches;
		LOGF_COLOR(Log::Color::BrightRed,
		           "BufferCache: stream copy reuse mismatch #%u: 0x%016" PRIx64 "+0x%" PRIx64
		           ", reuse disabled\n",
		           m_streamed_copy_mismatches, vaddr, size);
	}
}

static constexpr uint64_t ZERO_COPY_CHUNK_SIZE = uint64_t {256} << 20u;
static constexpr uint32_t ZERO_COPY_MAX_CHUNKS = 8;
static constexpr uint64_t ZERO_COPY_PAGE_BITS  = 12;
// A page tick that keeps the page out of imported memory.
static constexpr uint64_t ZERO_COPY_EXCLUDED = UINT64_MAX;

struct BufferCache::ZeroCopyChunk {
	enum State : uint8_t { None, Requested, Ready, Failed };
	std::atomic<uint8_t>    state {None};
	VkDeviceMemory          memory = VK_NULL_HANDLE;
	VkBuffer                handle = VK_NULL_HANDLE;
	std::unique_ptr<Buffer> buffer;
	// Per page: the tick of the last command buffer that read it, or ZERO_COPY_EXCLUDED.
	std::unique_ptr<uint64_t[]> page_ticks;
};

void BufferCache::StartZeroCopy() {
	if (!Config::ZeroCopyBuffersEnabled()) {
		return;
	}
	auto&      d    = VULKAN_HPP_DEFAULT_DISPATCHER;
	const auto base = LibKernel::Memory::BackingBase();
	const auto size = LibKernel::Memory::BackingSize();

	VkMemoryHostPointerPropertiesEXT properties {
	    VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
	if (d.vkGetMemoryHostPointerPropertiesEXT == nullptr || base == 0 || (base & 0xfffu) != 0 ||
	    d.vkGetMemoryHostPointerPropertiesEXT(
	        m_graphics.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
	        reinterpret_cast<const void*>(base), &properties) != VK_SUCCESS ||
	    properties.memoryTypeBits == 0) {
		LOGF_COLOR(Log::Color::BrightRed,
		           "BufferCache: zero-copy buffers unavailable, guest memory cannot be imported\n");
		return;
	}
	const auto& memory = m_graphics.physical_device_memory_properties;
	uint32_t    type   = UINT32_MAX;
	for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
		const bool coherent =
		    bool(memory.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent);
		if ((properties.memoryTypeBits & (1u << i)) != 0 && (type == UINT32_MAX || coherent)) {
			type = i;
		}
	}
	m_zero_copy_base        = base;
	m_zero_copy_size        = size;
	m_zero_copy_memory_type = type;
	m_zero_copy_chunks.resize((size + ZERO_COPY_CHUNK_SIZE - 1) / ZERO_COPY_CHUNK_SIZE);
	for (auto& chunk: m_zero_copy_chunks) {
		chunk = std::make_unique<ZeroCopyChunk>();
	}
	m_zero_copy_thread =
	    std::jthread([this](std::stop_token stop) { ZeroCopyImportThread(std::move(stop)); });
	m_zero_copy = true;
	LOGF_COLOR(Log::Color::BrightGreen,
	           "BufferCache: zero-copy buffers on, memory type %u, up to %u chunks of %" PRIu64
	           " MB\n",
	           type, ZERO_COPY_MAX_CHUNKS, ZERO_COPY_CHUNK_SIZE >> 20u);
}

void BufferCache::StopZeroCopy() {
	if (!m_zero_copy) {
		return;
	}
	m_zero_copy_thread.request_stop();
	if (m_zero_copy_thread.joinable()) {
		m_zero_copy_thread.join();
	}
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	for (auto& chunk: m_zero_copy_chunks) {
		chunk->buffer.reset();
		if (chunk->handle != VK_NULL_HANDLE) {
			d.vkDestroyBuffer(m_graphics.device, chunk->handle, nullptr);
		}
		if (chunk->memory != VK_NULL_HANDLE) {
			d.vkFreeMemory(m_graphics.device, chunk->memory, nullptr);
		}
	}
	m_zero_copy_chunks.clear();
	m_zero_copy = false;
}

void BufferCache::ZeroCopyImportThread(std::stop_token stop) {
	auto&          d      = VULKAN_HPP_DEFAULT_DISPATCHER;
	const VkDevice device = m_graphics.device;
	for (;;) {
		uint32_t index = 0;
		{
			std::unique_lock lock(m_zero_copy_mutex);
			if (!m_zero_copy_available.wait(lock, stop,
			                                [this] { return !m_zero_copy_requests.empty(); })) {
				return;
			}
			index = m_zero_copy_requests.back();
			m_zero_copy_requests.pop_back();
		}
		auto&      chunk  = *m_zero_copy_chunks[index];
		const auto offset = uint64_t {index} * ZERO_COPY_CHUNK_SIZE;
		const auto bytes  = std::min(ZERO_COPY_CHUNK_SIZE, m_zero_copy_size - offset);
		const auto start  = std::chrono::steady_clock::now();

		VkImportMemoryHostPointerInfoEXT import {
		    VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
		import.handleType   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
		import.pHostPointer = reinterpret_cast<void*>(m_zero_copy_base + offset);
		VkMemoryAllocateInfo allocate {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &import};
		allocate.allocationSize  = bytes;
		allocate.memoryTypeIndex = m_zero_copy_memory_type;
		VkExternalMemoryBufferCreateInfo external {
		    VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
		external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
		VkBufferCreateInfo create {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
		create.size  = bytes;
		create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
		               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
		               VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		bool ok = d.vkAllocateMemory(device, &allocate, nullptr, &chunk.memory) == VK_SUCCESS &&
		          d.vkCreateBuffer(device, &create, nullptr, &chunk.handle) == VK_SUCCESS &&
		          d.vkBindBufferMemory(device, chunk.handle, chunk.memory, 0) == VK_SUCCESS;
		if (ok) {
			chunk.page_ticks = std::make_unique<uint64_t[]>(bytes >> ZERO_COPY_PAGE_BITS);
			chunk.buffer =
			    std::make_unique<Buffer>(m_graphics, m_scheduler, vk::Buffer {chunk.handle}, bytes);
		}
		const auto ms =
		    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
		        .count();
		LOGF_COLOR(ok ? Log::Color::BrightGreen : Log::Color::BrightRed,
		           "BufferCache: zero-copy chunk %u (%" PRIu64 " MB at 0x%016" PRIx64
		           ") %s in %.1f ms\n",
		           index, bytes >> 20u, offset, ok ? "imported" : "failed", ms);
		chunk.state.store(ok ? ZeroCopyChunk::Ready : ZeroCopyChunk::Failed,
		                  std::memory_order_release);
	}
}

bool BufferCache::ObtainZeroCopy(uint64_t vaddr, uint64_t size,
                                 std::pair<Buffer*, uint64_t>& result) {
	uint64_t offset = 0;
	if (!LibKernel::Memory::TranslateBacking(vaddr, size, &offset)) {
		return false;
	}
	const auto index = offset / ZERO_COPY_CHUNK_SIZE;
	if (index >= m_zero_copy_chunks.size() || (offset + size - 1) / ZERO_COPY_CHUNK_SIZE != index) {
		return false;
	}
	auto&      chunk = *m_zero_copy_chunks[index];
	const auto state = chunk.state.load(std::memory_order_acquire);
	if (state != ZeroCopyChunk::Ready) {
		if (state == ZeroCopyChunk::None && m_zero_copy_requested < ZERO_COPY_MAX_CHUNKS) {
			chunk.state.store(ZeroCopyChunk::Requested, std::memory_order_relaxed);
			++m_zero_copy_requested;
			{
				std::lock_guard lock(m_zero_copy_mutex);
				m_zero_copy_requests.push_back(static_cast<uint32_t>(index));
			}
			m_zero_copy_available.notify_one();
		}
		return false;
	}
	const auto local = offset - index * ZERO_COPY_CHUNK_SIZE;
	const auto first = local >> ZERO_COPY_PAGE_BITS;
	const auto last  = (local + size - 1) >> ZERO_COPY_PAGE_BITS;
	for (auto page = first; page <= last; page++) {
		if (chunk.page_ticks[page] == ZERO_COPY_EXCLUDED) {
			return false;
		}
	}
	const auto tick = m_scheduler.CurrentTick();
	for (auto page = first; page <= last; page++) {
		chunk.page_ticks[page] = tick;
	}
	result = {chunk.buffer.get(), local};
	return true;
}

void BufferCache::NoteCommandWrite(uint64_t vaddr, uint64_t size) {
	if (!m_zero_copy || size == 0) {
		return;
	}
	bool wait = false;
	for (auto page = vaddr >> ZERO_COPY_PAGE_BITS;
	     page <= (vaddr + size - 1) >> ZERO_COPY_PAGE_BITS; page++) {
		uint64_t offset = 0;
		if (!LibKernel::Memory::TranslateBacking(page << ZERO_COPY_PAGE_BITS,
		                                         uint64_t {1} << ZERO_COPY_PAGE_BITS, &offset)) {
			continue;
		}
		const auto index = offset / ZERO_COPY_CHUNK_SIZE;
		if (index >= m_zero_copy_chunks.size()) {
			continue;
		}
		auto& chunk = *m_zero_copy_chunks[index];
		if (chunk.state.load(std::memory_order_acquire) != ZeroCopyChunk::Ready) {
			continue;
		}
		auto& tick =
		    chunk.page_ticks[(offset - index * ZERO_COPY_CHUNK_SIZE) >> ZERO_COPY_PAGE_BITS];
		if (tick != 0 && tick != ZERO_COPY_EXCLUDED && !m_scheduler.IsFree(tick)) {
			wait = true;
			tick = ZERO_COPY_EXCLUDED;
		}
	}
	if (wait) {
		// Submitted work must read the bytes from before this write; later work copies them.
		Common::DebugCounters::Add(Common::DebugCounters::Counter::ZeroCopyWriteWaits);
		m_scheduler.Finish();
	}
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	ForgetStreamedCopies();
	if (!is_gds) {
		NoteCommandWrite(vaddr, size);
	}
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	ForgetStreamedCopies();
	if (!dst_gds) {
		NoteCommandWrite(dst_vaddr, size);
	}
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) &&
	    !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

void BufferCache::MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0) {
			m_checked_color_metadata[slot].Add(vaddr, size);
		}
	}
}

bool BufferCache::IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0 && !m_checked_color_metadata[slot].Contains(vaddr, size)) {
			return false;
		}
	}
	return true;
}

void BufferCache::ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size) {
	bool forgot = false;
	for (auto& checked: m_checked_color_metadata) {
		if (checked.Intersects(vaddr, size)) {
			checked.Subtract(vaddr, size);
			forgot = true;
		}
	}
	if (forgot) {
		m_color_check_generation.fetch_add(1, std::memory_order_acq_rel);
	}
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(
			    !DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	namespace Counters             = Common::DebugCounters;
	constexpr auto retire          = Counters::ReadbackSource::Retire;
	const auto     completion_tick = m_scheduler.CurrentTick();
	Counters::Add(Counters::ReadbackCounter(retire, Counters::ReadbackStat::Calls));
	{
		const Counters::ScopedGpuWaitSite wait_site(Counters::GpuWaitSite::BufferReadback);
		const Counters::ScopedTimer       timer(
		    Counters::ReadbackCounter(retire, Counters::ReadbackStat::Downloads),
		    Counters::ReadbackCounter(retire, Counters::ReadbackStat::WaitNs));
		m_scheduler.Wait(completion_tick);
		m_scheduler.WaitPriorityOperations(completion_tick);
	}
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::QueueBdaSync(uint64_t vaddr, uint64_t size) {
	const auto      begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto      end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	std::lock_guard lock(m_bda_sync_mutex);
	m_bda_sync.Add(begin, end - begin);
}

RangeSet BufferCache::TakeBdaSync() {
	std::lock_guard lock(m_bda_sync_mutex);
	return std::exchange(m_bda_sync, {});
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

} // namespace Libs::Graphics
