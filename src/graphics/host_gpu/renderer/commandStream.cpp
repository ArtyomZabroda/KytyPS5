#include "graphics/host_gpu/renderer/commandStream.h"

#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"

#include <chrono>
#include <cstdio>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#endif

namespace Libs::Graphics {

namespace {

namespace Counters = Common::DebugCounters;

// While the GPU thread draws, batches arrive tens of microseconds apart; spinning this long avoids
// most sleeps and wake-ups between them without keeping a core busy when the GPU thread is idle.
constexpr auto SpinTime = std::chrono::microseconds(20);

template <typename Ready>
bool SpinUntil(const Ready& ready) {
	const auto deadline = std::chrono::steady_clock::now() + SpinTime;
	for (;;) {
		for (int i = 0; i < 64; i++) {
			if (ready()) {
				return true;
			}
			_mm_pause();
		}
		if (std::chrono::steady_clock::now() >= deadline) {
			return false;
		}
	}
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
std::vector<GROUP_AFFINITY> PhysicalCores() {
	DWORD length = 0;
	GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
	std::vector<std::byte> buffer(length);
	if (length == 0 ||
	    GetLogicalProcessorInformationEx(
	        RelationProcessorCore,
	        reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data()),
	        &length) == FALSE) {
		return {};
	}
	std::vector<GROUP_AFFINITY> cores;
	for (DWORD offset = 0; offset < length;) {
		const auto* entry = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
		    buffer.data() + offset);
		if (entry->Relationship == RelationProcessorCore) {
			cores.push_back(entry->Processor.GroupMask[0]);
		}
		offset += entry->Size;
	}
	return cores;
}
#endif

} // namespace

void PlaceGpuThread(bool recording_thread) {
	if (!Config::PinGpuThreadsEnabled()) {
		return;
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const auto cores = PhysicalCores();
	if (cores.size() < 3) {
		return;
	}
	// The last two cores: the first ones take more of the system's interrupts. Sharing one core
	// through SMT made both threads slower than letting the OS place them.
	const size_t   core = recording_thread ? cores.size() - 2 : cores.size() - 1;
	GROUP_AFFINITY affinity {};
	affinity.Mask  = cores[core].Mask;
	affinity.Group = cores[core].Group;
	const bool ok  = SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr) != FALSE;
	char       text[128];
	std::snprintf(text, sizeof(text),
	              "GPU thread placement: %s thread on core %zu (mask 0x%llx)%s\n",
	              recording_thread ? "recording" : "PM4", core,
	              static_cast<unsigned long long>(affinity.Mask), ok ? "" : " failed");
	Log::WriteToConsoleAndLog(text);
#else
	(void)recording_thread;
#endif
}

CommandStream::CommandStream() {
	m_blocks.push_back(std::make_unique<Block>());
	m_block      = m_blocks.back().get();
	m_read_block = m_block;
	m_thread     = std::thread([this] { Run(); });
}

CommandStream::~CommandStream() {
	Drain();
	m_stop.store(true, std::memory_order_seq_cst);
	WakeWorker();
	m_thread.join();
}

void CommandStream::RecordTarget(vk::CommandBuffer buffer) {
	Record([this, buffer](vk::CommandBuffer /*previous*/) { m_target = buffer; });
}

void CommandStream::Drain() {
	if (Idle()) {
		return;
	}
	Publish();
	const auto start = Counters::TimerStart();
	if (!SpinUntil([this] { return Idle(); })) {
		m_producer_waiting.store(true, std::memory_order_seq_cst);
		for (;;) {
			const auto consumed = m_consumed.load(std::memory_order_seq_cst);
			if (consumed == m_produced) {
				break;
			}
			m_consumed.wait(consumed, std::memory_order_acquire);
		}
		m_producer_waiting.store(false, std::memory_order_relaxed);
	}
	if (start != std::chrono::steady_clock::time_point {}) {
		Counters::Add(Counters::Counter::RecorderWaits);
		Counters::AddSince(Counters::Counter::RecorderWaitNs, start);
	}
}

void CommandStream::NextBlock(size_t bytes) {
	EXIT_IF(bytes > BlockBytes);
	Block* block = nullptr;
	{
		std::lock_guard lock(m_blocks_mutex);
		if (!m_free_blocks.empty()) {
			block = m_free_blocks.back();
			m_free_blocks.pop_back();
		} else {
			m_blocks.push_back(std::make_unique<Block>());
			block = m_blocks.back().get();
		}
	}
	// Everything written to the old block is published before the link, so the recording thread
	// that sees the link has seen all of it.
	Publish();
	m_block->next.store(block, std::memory_order_seq_cst);
	m_block     = block;
	m_write     = 0;
	m_published = 0;
}

void CommandStream::Retire(Block* block) {
	block->committed.store(0, std::memory_order_relaxed);
	block->next.store(nullptr, std::memory_order_relaxed);
	std::lock_guard lock(m_blocks_mutex);
	m_free_blocks.push_back(block);
}

void CommandStream::WakeWorker() {
	m_wake.fetch_add(1, std::memory_order_release);
	m_wake.notify_one();
}

bool CommandStream::HasWork() const noexcept {
	return m_read_block->committed.load(std::memory_order_seq_cst) != m_read ||
	       m_read_block->next.load(std::memory_order_seq_cst) != nullptr;
}

bool CommandStream::ReplayAvailable() {
	const auto start    = Counters::TimerStart();
	bool       replayed = false;
	for (;;) {
		const auto committed = m_read_block->committed.load(std::memory_order_acquire);
		if (m_read < committed) {
			const auto* data = m_read_block->data.get();
			while (m_read < committed) {
				Header header {};
				std::memcpy(&header, data + m_read, sizeof(header));
				if (header.replay != nullptr) {
					header.replay(m_target, data + m_read + HeaderBytes);
					m_replayed++;
				}
				m_read += header.bytes;
			}
			replayed = true;
			continue;
		}
		auto* next = m_read_block->next.load(std::memory_order_acquire);
		if (next == nullptr) {
			break;
		}
		if (m_read_block->committed.load(std::memory_order_acquire) != m_read) {
			continue;
		}
		Retire(m_read_block);
		m_read_block = next;
		m_read       = 0;
	}
	if (!replayed) {
		return false;
	}
	// Paired with Drain(): either it sees the new count, or this thread sees it waiting.
	m_consumed.store(m_replayed, std::memory_order_seq_cst);
	if (m_producer_waiting.load(std::memory_order_seq_cst)) {
		m_consumed.notify_one();
	}
	if (start != std::chrono::steady_clock::time_point {}) {
		Counters::AddSince(Counters::Counter::RecorderBusyNs, start);
	}
	return true;
}

void CommandStream::Run() {
	KYTY_PROFILER_THREAD("Thread_GpuRecord");
	PlaceGpuThread(true);
	for (;;) {
		if (ReplayAvailable()) {
			continue;
		}
		// The destructor drains first, so nothing is left to replay.
		if (m_stop.load(std::memory_order_acquire)) {
			return;
		}
		if (SpinUntil([this] { return HasWork() || m_stop.load(std::memory_order_relaxed); })) {
			continue;
		}
		const auto wake = m_wake.load(std::memory_order_acquire);
		m_worker_sleeping.store(true, std::memory_order_seq_cst);
		if (!HasWork() && !m_stop.load(std::memory_order_seq_cst)) {
			m_wake.wait(wake, std::memory_order_acquire);
		}
		m_worker_sleeping.store(false, std::memory_order_relaxed);
	}
}

} // namespace Libs::Graphics
