#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSTREAM_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSTREAM_H_

#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <vector>

namespace Libs::Graphics {

// --threaded-recording: carries the renderer's Vulkan commands from the thread that prepares them
// to a recording thread, which calls the driver in the same order. Each command is a closure
// written into a fixed-size block, with copies of the arrays it points to just before it. One
// thread at a time records (the renderer's callers are already serialized); the recording thread
// replays into the target command buffer.
//
// Commands may also end and submit command buffers (see CommandScheduler::Submit), and
// RecordTarget() switches the buffer that later commands go to. Each command buffer has its own
// pool, so the producer can begin the next buffer while this thread still finishes the previous
// one. Code that records into the current buffer directly calls Drain() first.
// --pin-gpu-threads: pins the calling thread, the GPU (PM4) thread or the recording thread, to its
// own physical core, so the two never share one through SMT. Does nothing without the option.
void PlaceGpuThread(bool recording_thread);

class CommandStream final {
public:
	CommandStream();
	~CommandStream();
	KYTY_CLASS_NO_COPY(CommandStream);

	// Commands recorded after this go to `buffer`, which the producer has already begun.
	void RecordTarget(vk::CommandBuffer buffer);
	// Returns once the recording thread has replayed every command recorded so far.
	void               Drain();
	[[nodiscard]] bool Idle() const noexcept {
		return m_consumed.load(std::memory_order_acquire) == m_produced;
	}

	// Records a command whose arguments are all values. `fn` is called with the command buffer.
	template <typename Fn>
	void Record(const Fn& fn) {
		Reserve(MaxCommandBytes);
		Emplace(fn);
		Commit();
	}

	// Records a command whose arguments point to memory. build() copies that memory with Copy(),
	// at most `data_bytes` (a sum of DataBytes()), and returns the closure that uses the copies.
	template <typename Build>
	void RecordWithData(size_t data_bytes, const Build& build) {
		Reserve(data_bytes + MaxCommandBytes);
		const auto fn = build();
		Emplace(fn);
		Commit();
	}

	// Space that Copy() takes for `count` elements of T.
	template <typename T>
	[[nodiscard]] static constexpr size_t DataBytes(size_t count) noexcept {
		return count == 0 ? 0 : HeaderBytes + AlignUp(sizeof(T) * count);
	}

	// Copies `count` elements into the stream, where they stay until the command that uses them
	// is replayed. Returns null for no elements. Only inside a RecordWithData build().
	template <typename T>
	[[nodiscard]] T* Copy(const T* source, size_t count) {
		static_assert(std::is_trivially_copyable_v<T> && alignof(T) <= Alignment);
		if (source == nullptr || count == 0) {
			return nullptr;
		}
		auto* data = AllocateData(sizeof(T) * count);
		std::memcpy(data, source, sizeof(T) * count);
		return std::launder(reinterpret_cast<T*>(data));
	}

private:
	static constexpr size_t Alignment       = 16;
	static constexpr size_t HeaderBytes     = 16;
	static constexpr size_t MaxClosureBytes = 128;
	static constexpr size_t MaxCommandBytes = HeaderBytes + MaxClosureBytes;
	static constexpr size_t BlockBytes      = size_t {1} << 20;
	// Commands are handed over in batches. Publishing each one made the producer's and the
	// recording thread's cores trade the same cache line for every command.
	static constexpr uint32_t PublishCommands = 64;
	static constexpr size_t   PublishBytes    = 16 * 1024;

	using ReplayFn = void (*)(vk::CommandBuffer buffer, const void* closure);

	// Starts every record. A record without a replay function holds data for a later command.
	struct Header {
		ReplayFn replay;
		uint32_t bytes;
	};
	static_assert(sizeof(Header) <= HeaderBytes);

	struct Block {
		Block(): data(new std::byte[BlockBytes]) {}

		std::unique_ptr<std::byte[]> data;
		// Bytes the recording thread may replay.
		std::atomic<uint32_t> committed {0};
		// Set once the producer has moved on; nothing more is committed to this block.
		std::atomic<Block*> next {nullptr};
	};

	[[nodiscard]] static constexpr size_t AlignUp(size_t value) noexcept {
		return (value + Alignment - 1) & ~(Alignment - 1);
	}

	template <typename Fn>
	static void Replay(vk::CommandBuffer buffer, const void* closure) {
		(*static_cast<const Fn*>(closure))(buffer);
	}

	void Reserve(size_t bytes) {
		if (m_write + bytes > BlockBytes) {
			NextBlock(bytes);
		}
		m_reserve_end = m_write + bytes;
	}

	std::byte* AllocateData(size_t bytes) {
		const auto total = HeaderBytes + AlignUp(bytes);
		EXIT_IF(m_write + total > m_reserve_end);
		auto*        record = m_block->data.get() + m_write;
		const Header header {nullptr, static_cast<uint32_t>(total)};
		std::memcpy(record, &header, sizeof(header));
		m_write += total;
		return record + HeaderBytes;
	}

	template <typename Fn>
	void Emplace(const Fn& fn) {
		static_assert(std::is_trivially_copyable_v<Fn> && std::is_trivially_destructible_v<Fn>,
		              "recorded commands must capture values and stream copies only");
		static_assert(sizeof(Fn) <= MaxClosureBytes && alignof(Fn) <= Alignment);
		const auto total = AlignUp(HeaderBytes + sizeof(Fn));
		EXIT_IF(m_write + total > m_reserve_end);
		auto*        record = m_block->data.get() + m_write;
		const Header header {&Replay<Fn>, static_cast<uint32_t>(total)};
		std::memcpy(record, &header, sizeof(header));
		::new (static_cast<void*>(record + HeaderBytes)) Fn(fn);
		m_write += total;
	}

	void Commit() {
		m_produced++;
		if (++m_unpublished >= PublishCommands || m_write - m_published >= PublishBytes) {
			Publish();
		}
	}

public:
	// Hands the recorded commands to the recording thread now instead of with the next batch.
	void Publish() {
		if (m_write == m_published) {
			return;
		}
		// Sequentially consistent, paired with the recording thread's sleep flag: either it sees
		// these commands before it sleeps, or this thread sees it asleep and wakes it.
		m_block->committed.store(static_cast<uint32_t>(m_write), std::memory_order_seq_cst);
		m_published   = m_write;
		m_unpublished = 0;
		if (m_worker_sleeping.load(std::memory_order_seq_cst)) {
			WakeWorker();
		}
	}

private:
	void               NextBlock(size_t bytes);
	void               WakeWorker();
	void               Run();
	bool               ReplayAvailable();
	[[nodiscard]] bool HasWork() const noexcept;
	void               Retire(Block* block);

	// Producer side.
	Block*   m_block       = nullptr;
	size_t   m_write       = 0;
	size_t   m_published   = 0;
	size_t   m_reserve_end = 0;
	uint64_t m_produced    = 0;
	uint32_t m_unpublished = 0;

	// Recording thread side.
	Block*            m_read_block = nullptr;
	size_t            m_read       = 0;
	uint64_t          m_replayed   = 0;
	vk::CommandBuffer m_target     = nullptr;

	// Shared.
	std::atomic<uint64_t> m_consumed {0};
	std::atomic<uint32_t> m_wake {0};
	std::atomic<bool>     m_worker_sleeping {false};
	std::atomic<bool>     m_producer_waiting {false};
	std::atomic<bool>     m_stop {false};

	std::mutex                          m_blocks_mutex;
	std::vector<std::unique_ptr<Block>> m_blocks;
	std::vector<Block*>                 m_free_blocks;

	std::thread m_thread;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSTREAM_H_
