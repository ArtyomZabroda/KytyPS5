#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/commandStream.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler {
public:
	// With threaded_recording, a CommandStream thread calls the driver for recorded commands. The
	// stream is declared after the command pool, so its thread stops before the pool is destroyed.
	CommandScheduler(RenderContext& context, GraphicContext& graphics,
	                 bool threaded_recording = false);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering(std::source_location location = std::source_location::current());
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void Shutdown();
	void Wait(uint64_t tick);
	void PopPendingOperations();
	void DrainPriorityOperations();
	void WaitPriorityOperations(uint64_t tick);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void DeferOperation(Common::UniqueFunction<void>&& operation);
	void DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	// The operation touches guest memory only in [vaddr, vaddr + size); size 0 touches none.
	void DeferPriorityOperation(Common::UniqueFunction<void>&& operation, uint64_t vaddr,
	                            uint64_t size);
	[[nodiscard]] bool HasPendingPriorityOperations();
	// Whether a queued or running priority operation may touch guest memory in the range.
	[[nodiscard]] bool        PriorityOperationsTouch(uint64_t vaddr, uint64_t size);
	[[nodiscard]] static bool InDeferredOperation() noexcept;
	// Called between guest packets after a draw or dispatch. Guest command buffers can hold a
	// whole frame; without this the GPU idles until the next readback or flush, which then waits
	// for all of the recorded work.
	void SubmitIfGpuIdle();

	[[nodiscard]] bool     Active() const noexcept { return m_command.m_registers != nullptr; }
	void                   CheckActive() const;
	CommandBuffer&         Current();
	[[nodiscard]] uint64_t CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool     IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&  m_graphics;
		MasterSemaphore& m_master;
		// One pool per buffer: with --threaded-recording, the GPU thread begins a buffer while the
		// recording thread still records and submits another, and a pool cannot be used by two
		// threads at once.
		std::vector<vk::CommandPool>   m_pools;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
		// The guest memory the operation may touch; by default all of it.
		uint64_t touch_begin = 0;
		uint64_t touch_end   = UINT64_MAX;
	};

	void     BeginNext();
	uint64_t SubmitRecorded(SubmitInfo& submit);
	void     PriorityOperationsThread(std::stop_token stop);
	void     QueueOperation(Common::UniqueFunction<void>&& operation, bool priority,
	                        uint64_t touch_begin = 0, uint64_t touch_end = UINT64_MAX);
	void     RunOperation(Common::UniqueFunction<void>&& operation);
	// --perf-overlay: reads the start and end timestamps of command buffers whose ticks completed.
	void PollGpuTime();

	MasterSemaphore                       m_master;
	RenderContext&                        m_context;
	GraphicContext&                       m_graphics;
	CommandPool                           m_command_pool;
	std::unique_ptr<CommandStream>        m_stream;
	CommandBuffer                         m_command;
	DynamicState                          m_dynamic_state;
	std::chrono::steady_clock::time_point m_command_begin;
	// --perf-overlay: a pair of timestamp queries per command buffer, at its start and its end.
	struct GpuTimeSlot {
		uint32_t slot = 0;
		uint64_t tick = 0;
	};
	static constexpr uint32_t             GpuTimeSlots    = 256;
	vk::QueryPool                         m_gpu_time_pool = nullptr;
	std::deque<GpuTimeSlot>               m_gpu_time_pending;
	uint32_t                              m_gpu_time_next     = 0;
	int32_t                               m_gpu_time_current  = -1;
	uint64_t                              m_gpu_time_last_end = 0;
	std::deque<PendingOperation>          m_pending_operations;
	std::deque<PendingOperation>          m_priority_operations;
	std::mutex                            m_operation_mutex;
	std::condition_variable               m_operation_available;
	std::jthread                          m_priority_thread;
	bool                                  m_priority_active       = false;
	uint64_t                              m_priority_active_tick  = 0;
	uint64_t                              m_priority_active_begin = 0;
	uint64_t                              m_priority_active_end   = 0;
	OperationState                        m_operation_state       = OperationState::Open;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
