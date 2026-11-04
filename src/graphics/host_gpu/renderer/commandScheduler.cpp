#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <optional>

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
}

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
}

CommandScheduler::CommandPool::~CommandPool() {
	for (const auto pool: m_pools) {
		m_graphics.device.destroyCommandPool(pool, nullptr);
	}
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_pools.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	for (size_t index = first; index < first + GrowStep; index++) {
		vk::CommandPoolCreateInfo create {};
		create.queueFamilyIndex = m_graphics.queue_family;
		create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
		                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		const auto result = m_graphics.device.createCommandPool(&create, nullptr, &m_pools[index]);
		EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

		vk::CommandBufferAllocateInfo allocate {};
		allocate.commandPool        = m_pools[index];
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = 1;
		EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, &m_buffers[index]) !=
		        vk::Result::eSuccess);
	}
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics,
                                   bool threaded_recording)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master),
      m_stream(threaded_recording ? std::make_unique<CommandStream>() : nullptr), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {
	m_command.m_stream = m_stream.get();
	if (Common::DebugCounters::Enabled() &&
	    graphics.physical_device_properties.limits.timestampComputeAndGraphics != VK_FALSE) {
		const vk::QueryPoolCreateInfo info {.queryType  = vk::QueryType::eTimestamp,
		                                    .queryCount = GpuTimeSlots * 2};
		if (graphics.device.createQueryPool(&info, nullptr, &m_gpu_time_pool) ==
		    vk::Result::eSuccess) {
			graphics.device.resetQueryPool(m_gpu_time_pool, 0, GpuTimeSlots * 2);
		} else {
			m_gpu_time_pool = nullptr;
		}
	}
}

CommandScheduler::~CommandScheduler() {
	Shutdown();
	if (m_gpu_time_pool) {
		m_graphics.device.destroyQueryPool(m_gpu_time_pool, nullptr);
	}
}

void CommandScheduler::PollGpuTime() {
	if (m_gpu_time_pending.empty()) {
		return;
	}
	m_master.Refresh();
	const auto   known  = m_master.KnownGpuTick();
	const double period = m_graphics.physical_device_properties.limits.timestampPeriod;
	while (!m_gpu_time_pending.empty() && m_gpu_time_pending.front().tick <= known) {
		const auto slot = m_gpu_time_pending.front().slot;
		m_gpu_time_pending.pop_front();
		std::array<uint64_t, 2> stamps {};
		const auto              result = m_graphics.device.getQueryPoolResults(
		    m_gpu_time_pool, slot * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		    vk::QueryResultFlagBits::e64);
		m_graphics.device.resetQueryPool(m_gpu_time_pool, slot * 2, 2);
		if (result != vk::Result::eSuccess || stamps[1] < stamps[0]) {
			continue;
		}
		// Buffers run in order on the one queue; overlap with the previous one counts once.
		const auto begin = std::max(stamps[0], m_gpu_time_last_end);
		if (stamps[1] > begin) {
			Common::DebugCounters::Add(
			    Common::DebugCounters::Counter::GpuBusyNs,
			    static_cast<uint64_t>(static_cast<double>(stamps[1] - begin) * period));
		}
		m_gpu_time_last_end = std::max(m_gpu_time_last_end, stamps[1]);
	}
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering(std::source_location location) {
	if (Active() && !m_command.IsInvalid()) {
		Current().EndRendering(location);
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::FlushAndWait() {
	const auto tick = Submit();
	m_master.Wait(tick);
	BeginNext();
}

void CommandScheduler::Finish() {
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

// Nanoseconds since `start`, which moves to now.
static uint64_t LapNs(std::chrono::steady_clock::time_point& start) {
	const auto now     = std::chrono::steady_clock::now();
	const auto elapsed = now - start;
	start              = now;
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
}

void CommandScheduler::Wait(uint64_t tick) {
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		namespace Counters = Common::DebugCounters;
		// Readbacks wait for earlier submissions, then for the one holding their copy.
		const bool split = Counters::Enabled() &&
		                   Counters::CurrentGpuWaitSite() == Counters::GpuWaitSite::BufferReadback;
		Counters::ReadbackSplit               times;
		std::chrono::steady_clock::time_point start;
		if (split) {
			m_master.Refresh();
			const auto done     = m_master.KnownGpuTick();
			times.backlog_ticks = tick - 1 > done ? tick - 1 - done : 0;
			start               = std::chrono::steady_clock::now();
		}
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		if (split) {
			times.submit_ns = LapNs(start);
			m_master.Wait(tick - 1);
			times.backlog_ns = LapNs(start);
		}
		m_master.Wait(tick);
		if (split) {
			times.current_ns = LapNs(start);
			Counters::Add(Counters::Counter::ReadbackSplitWaits);
			Counters::Add(Counters::Counter::ReadbackSplitSubmitNs, times.submit_ns);
			Counters::Add(Counters::Counter::ReadbackSplitBacklogNs, times.backlog_ns);
			Counters::Add(Counters::Counter::ReadbackSplitCurrentNs, times.current_ns);
			Counters::Add(Counters::Counter::ReadbackSplitBacklogTicks, times.backlog_ticks);
			Counters::t_last_readback_split = times;
		}
		BeginNext();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations() {
	m_master.Refresh();
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop_front();
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), false);
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), true);
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation,
                                              uint64_t vaddr, uint64_t size) {
	QueueOperation(std::move(operation), true, vaddr, vaddr + size);
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation, bool priority,
                                      uint64_t touch_begin, uint64_t touch_end) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		auto& queue = priority ? m_priority_operations : m_pending_operations;
		queue.push_back({std::move(operation), CurrentTick(), touch_begin, touch_end});
		lock.unlock();
		if (priority) {
			m_operation_available.notify_one();
		}
		return;
	}
	if (g_deferred_callback_scheduler != this) {
		m_operation_available.wait(lock,
		                           [this] { return m_operation_state == OperationState::Closed; });
	}
	lock.unlock();
	operation();
}

bool CommandScheduler::HasPendingPriorityOperations() {
	std::lock_guard lock(m_operation_mutex);
	return !m_priority_operations.empty() || m_priority_active;
}

bool CommandScheduler::PriorityOperationsTouch(uint64_t vaddr, uint64_t size) {
	const auto end     = vaddr + size;
	const auto touches = [&](uint64_t begin, uint64_t finish) {
		return begin < finish && begin < end && vaddr < finish;
	};
	std::lock_guard lock(m_operation_mutex);
	if (m_priority_active && touches(m_priority_active_begin, m_priority_active_end)) {
		return true;
	}
	return std::ranges::any_of(m_priority_operations, [&](const PendingOperation& operation) {
		return touches(operation.touch_begin, operation.touch_end);
	});
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	const Common::DebugCounters::ScopedGpuWaitSite wait_site(
	    Common::DebugCounters::GpuWaitSite::Background);
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop_front();
			m_priority_active       = true;
			m_priority_active_tick  = operation.tick;
			m_priority_active_begin = operation.touch_begin;
			m_priority_active_end   = operation.touch_end;
		}
		m_master.Wait(operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active       = false;
			m_priority_active_tick  = 0;
			m_priority_active_begin = 0;
			m_priority_active_end   = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	const auto       done = [this, tick] {
		const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	};
	if (done()) {
		return;
	}
	const Common::DebugCounters::ScopedTimer timer(
	    Common::DebugCounters::Counter::PriorityCallbackWaits,
	    Common::DebugCounters::Counter::PriorityCallbackWaitNs);
	m_operation_available.wait(lock, done);
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	EXIT_IF(m_command.IsInvalid());
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	// Commit() reuses only buffers whose submission the GPU has finished, so the recording thread
	// is done with the one it returns.
	m_command.m_buffer = m_command_pool.Commit();
	m_command.Begin();
	if (m_stream != nullptr) {
		m_stream->RecordTarget(m_command.m_buffer);
	}
	m_gpu_time_current = -1;
	if (m_gpu_time_pool && m_gpu_time_pending.size() < GpuTimeSlots) {
		const uint32_t slot = m_gpu_time_next;
		m_gpu_time_next     = (m_gpu_time_next + 1) % GpuTimeSlots;
		m_gpu_time_current  = static_cast<int32_t>(slot);
		const auto pool     = m_gpu_time_pool;
		const auto write    = [pool, slot](vk::CommandBuffer buffer) {
			buffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, pool, slot * 2);
		};
		if (m_stream != nullptr) {
			m_stream->Record(write);
		} else {
			write(m_command.m_buffer);
		}
	}
	m_dynamic_state.Invalidate();
	m_command_begin = std::chrono::steady_clock::now();
	return m_command;
}

void CommandScheduler::SubmitIfGpuIdle() {
	// Each submit costs tens of microseconds; batches shorter than this are not worth one.
	constexpr auto MinRecording = std::chrono::milliseconds(1);
	if (!Active() || m_command.IsInvalid() ||
	    std::chrono::steady_clock::now() - m_command_begin < MinRecording) {
		return;
	}
	// While the GPU still runs earlier submissions it has work, and the batch keeps growing.
	m_master.Refresh();
	if (m_master.KnownGpuTick() + 1 < CurrentTick()) {
		return;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::EarlySubmits);
	Flush();
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	EXIT_IF(m_command.IsInvalid());
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);
	if (m_stream != nullptr) {
		return SubmitRecorded(submit);
	}

	m_command.EndRendering();
	if (m_gpu_time_current >= 0) {
		m_command.m_buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_gpu_time_pool,
		                                  static_cast<uint32_t>(m_gpu_time_current) * 2 + 1);
	}
	m_command.End();
	const auto buffer   = m_command.m_buffer;
	auto&      graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);

	vk::Result result;
	uint64_t   tick;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		tick = m_master.NextTick();
		submit.AddSignal(m_master.Handle(), tick);

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		result = graphics.queue.submit(1, &submit_info, nullptr);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);

	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, tick, m_command.m_debug_op,
		                  m_command.m_debug_submit_id, m_command.m_debug_arg0,
		                  m_command.m_debug_arg1, m_command.m_debug_arg2, m_command.m_debug_arg3,
		                  m_command.m_debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	m_command.m_buffer = nullptr;
	if (m_gpu_time_current >= 0) {
		m_gpu_time_pending.push_back({static_cast<uint32_t>(m_gpu_time_current), tick});
		m_gpu_time_current = -1;
	}
	PollGpuTime();
	return tick;
}

// --threaded-recording: the recording thread ends and submits the buffer after replaying its
// commands, and the GPU thread goes on at once. The tick is taken here, so CurrentTick() and the
// ticks callers wait for mean the same as without the thread; waiting for a tick that is not
// submitted yet is valid for a timeline semaphore. Only this stream signals the master
// semaphore, and it submits in tick order.
uint64_t CommandScheduler::SubmitRecorded(SubmitInfo& submit) {
	m_command.EndRendering();
	if (m_gpu_time_current >= 0) {
		const auto pool  = m_gpu_time_pool;
		const auto query = static_cast<uint32_t>(m_gpu_time_current) * 2 + 1;
		m_stream->Record([pool, query](vk::CommandBuffer buffer) {
			buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, pool, query);
		});
	}
	const auto tick = m_master.NextTick();
	submit.AddSignal(m_master.Handle(), tick);

	struct Debug {
		uint32_t op;
		uint64_t submit_id;
		uint32_t arg0, arg1, arg2, arg3;
		uint64_t arg4;
	};
	const Debug debug {m_command.m_debug_op,   m_command.m_debug_submit_id, m_command.m_debug_arg0,
	                   m_command.m_debug_arg1, m_command.m_debug_arg2,      m_command.m_debug_arg3,
	                   m_command.m_debug_arg4};
	auto*       graphics = &m_graphics;
	m_stream->RecordWithData(
	    CommandStream::DataBytes<SubmitInfo>(1) + CommandStream::DataBytes<Debug>(1), [&] {
		    const auto* info       = m_stream->Copy(&submit, 1);
		    const auto* debug_copy = m_stream->Copy(&debug, 1);
		    return [=](vk::CommandBuffer buffer) {
			    auto result = buffer.end();
			    EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

			    vk::TimelineSemaphoreSubmitInfo timeline_info {};
			    timeline_info.waitSemaphoreValueCount   = info->num_wait_semaphores;
			    timeline_info.pWaitSemaphoreValues      = info->wait_ticks.data();
			    timeline_info.signalSemaphoreValueCount = info->num_signal_semaphores;
			    timeline_info.pSignalSemaphoreValues    = info->signal_ticks.data();

			    vk::SubmitInfo submit_info {};
			    submit_info.pNext                = &timeline_info;
			    submit_info.waitSemaphoreCount   = info->num_wait_semaphores;
			    submit_info.pWaitSemaphores      = info->wait_semaphores.data();
			    submit_info.pWaitDstStageMask    = info->wait_stages.data();
			    submit_info.commandBufferCount   = 1;
			    submit_info.pCommandBuffers      = &buffer;
			    submit_info.signalSemaphoreCount = info->num_signal_semaphores;
			    submit_info.pSignalSemaphores    = info->signal_semaphores.data();
			    {
				    Common::LockGuard lock(graphics->queue_mutex);
				    result = graphics->queue.submit(1, &submit_info, nullptr);
			    }
			    if (result != vk::Result::eSuccess) {
				    ReportVulkanFatal("vkQueueSubmit", result, tick, debug_copy->op,
				                      debug_copy->submit_id, debug_copy->arg0, debug_copy->arg1,
				                      debug_copy->arg2, debug_copy->arg3, debug_copy->arg4);
			    }
			    EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
		    };
	    });
	// Callers may wait for the tick right away.
	m_stream->Publish();
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);

	m_command.m_buffer = nullptr;
	if (m_gpu_time_current >= 0) {
		m_gpu_time_pending.push_back({static_cast<uint32_t>(m_gpu_time_current), tick});
		m_gpu_time_current = -1;
	}
	PollGpuTime();
	return tick;
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
