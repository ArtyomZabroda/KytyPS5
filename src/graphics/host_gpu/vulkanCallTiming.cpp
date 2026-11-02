#include "graphics/host_gpu/vulkanCallTiming.h"

#include "common/debugCounters.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/pthread.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace Libs::Graphics {
namespace {

namespace Counters = Common::DebugCounters;

uint32_t g_next_slot = 0;

// One instantiation per wrapped entry point (Id keeps entry points with equal signatures apart).
// Calls from other threads, or while the counters are off, go straight to the driver.
template <int Id, typename R, typename... Args>
struct TimedEntry {
	static inline R(VKAPI_PTR* original)(Args...) = nullptr;
	static inline uint32_t slot                   = 0;

	static R VKAPI_PTR Call(Args... args) {
		if (!Counters::t_gpu_thread || !Counters::Enabled()) {
			return original(args...);
		}
		const uint64_t start = __rdtsc();
		if constexpr (std::is_void_v<R>) {
			original(args...);
			Counters::AddDriverCall(slot, __rdtsc() - start);
		} else {
			R result = original(args...);
			Counters::AddDriverCall(slot, __rdtsc() - start);
			return result;
		}
	}
};

template <int Id, typename R, typename... Args>
void Wrap(R(VKAPI_PTR*& entry)(Args...), const char* name) {
	using Entry = TimedEntry<Id, R, Args...>;
	if (entry == nullptr || entry == &Entry::Call || g_next_slot >= Counters::DRIVER_CALL_SLOTS) {
		return;
	}
	Entry::original = entry;
	Entry::slot     = g_next_slot++;
	Counters::RegisterDriverCall(Entry::slot, name);
	entry = &Entry::Call;
}

// The smallest cost of two back-to-back TSC reads: each measured call includes about this much.
uint64_t MeasureTimerTicks() {
	uint64_t best = std::numeric_limits<uint64_t>::max();
	for (int i = 0; i < 4096; i++) {
		const uint64_t start = __rdtsc();
		const uint64_t end   = __rdtsc();
		best                 = std::min(best, end - start);
	}
	return best;
}

} // namespace

#define KYTY_TIME_VK(name) Wrap<__LINE__>(dispatcher.name, #name)

void InstallVulkanCallTiming() {
	auto& dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;

	// Recording: the entry points the renderer uses (core names and the extension aliases it may
	// be called through).
	KYTY_TIME_VK(vkCmdBeginConditionalRenderingEXT);
	KYTY_TIME_VK(vkCmdEndConditionalRenderingEXT);
	KYTY_TIME_VK(vkCmdBeginQuery);
	KYTY_TIME_VK(vkCmdEndQuery);
	KYTY_TIME_VK(vkCmdResetQueryPool);
	KYTY_TIME_VK(vkCmdBeginRendering);
	KYTY_TIME_VK(vkCmdBeginRenderingKHR);
	KYTY_TIME_VK(vkCmdEndRendering);
	KYTY_TIME_VK(vkCmdEndRenderingKHR);
	KYTY_TIME_VK(vkCmdBindDescriptorSets);
	KYTY_TIME_VK(vkCmdBindIndexBuffer);
	KYTY_TIME_VK(vkCmdBindPipeline);
	KYTY_TIME_VK(vkCmdBindVertexBuffers);
	KYTY_TIME_VK(vkCmdBindVertexBuffers2);
	KYTY_TIME_VK(vkCmdBindVertexBuffers2EXT);
	KYTY_TIME_VK(vkCmdPushConstants);
	KYTY_TIME_VK(vkCmdPushDescriptorSet);
	KYTY_TIME_VK(vkCmdPushDescriptorSetKHR);
	KYTY_TIME_VK(vkCmdPipelineBarrier);
	KYTY_TIME_VK(vkCmdPipelineBarrier2);
	KYTY_TIME_VK(vkCmdPipelineBarrier2KHR);
	KYTY_TIME_VK(vkCmdDraw);
	KYTY_TIME_VK(vkCmdDrawIndexed);
	KYTY_TIME_VK(vkCmdDrawIndirect);
	KYTY_TIME_VK(vkCmdDrawIndexedIndirect);
	KYTY_TIME_VK(vkCmdDrawMeshTasksEXT);
	KYTY_TIME_VK(vkCmdDispatch);
	KYTY_TIME_VK(vkCmdDispatchBase);
	KYTY_TIME_VK(vkCmdDispatchIndirect);
	KYTY_TIME_VK(vkCmdBlitImage);
	KYTY_TIME_VK(vkCmdClearAttachments);
	KYTY_TIME_VK(vkCmdClearColorImage);
	KYTY_TIME_VK(vkCmdClearDepthStencilImage);
	KYTY_TIME_VK(vkCmdCopyBuffer);
	KYTY_TIME_VK(vkCmdCopyBufferToImage);
	KYTY_TIME_VK(vkCmdCopyImage);
	KYTY_TIME_VK(vkCmdCopyImageToBuffer);
	KYTY_TIME_VK(vkCmdFillBuffer);
	KYTY_TIME_VK(vkCmdUpdateBuffer);
	KYTY_TIME_VK(vkCmdResolveImage);
	KYTY_TIME_VK(vkCmdSetAttachmentFeedbackLoopEnableEXT);
	KYTY_TIME_VK(vkCmdSetBlendConstants);
	KYTY_TIME_VK(vkCmdSetColorWriteEnableEXT);
	KYTY_TIME_VK(vkCmdSetCullMode);
	KYTY_TIME_VK(vkCmdSetDepthBias);
	KYTY_TIME_VK(vkCmdSetDepthBiasEnable);
	KYTY_TIME_VK(vkCmdSetDepthBounds);
	KYTY_TIME_VK(vkCmdSetDepthBoundsTestEnable);
	KYTY_TIME_VK(vkCmdSetDepthCompareOp);
	KYTY_TIME_VK(vkCmdSetDepthTestEnable);
	KYTY_TIME_VK(vkCmdSetDepthWriteEnable);
	KYTY_TIME_VK(vkCmdSetFrontFace);
	KYTY_TIME_VK(vkCmdSetLineWidth);
	KYTY_TIME_VK(vkCmdSetScissor);
	KYTY_TIME_VK(vkCmdSetScissorWithCount);
	KYTY_TIME_VK(vkCmdSetStencilCompareMask);
	KYTY_TIME_VK(vkCmdSetStencilOp);
	KYTY_TIME_VK(vkCmdSetStencilReference);
	KYTY_TIME_VK(vkCmdSetStencilTestEnable);
	KYTY_TIME_VK(vkCmdSetStencilWriteMask);
	KYTY_TIME_VK(vkCmdSetViewport);
	KYTY_TIME_VK(vkCmdSetViewportWithCount);

	// Command buffer lifetime, descriptor writes and submission. Waits are left out: the overlay
	// already times them per call site.
	KYTY_TIME_VK(vkBeginCommandBuffer);
	KYTY_TIME_VK(vkEndCommandBuffer);
	KYTY_TIME_VK(vkResetCommandBuffer);
	KYTY_TIME_VK(vkAllocateCommandBuffers);
	KYTY_TIME_VK(vkUpdateDescriptorSets);
	KYTY_TIME_VK(vkAllocateDescriptorSets);
	KYTY_TIME_VK(vkQueueSubmit);
	KYTY_TIME_VK(vkQueueSubmit2);
	KYTY_TIME_VK(vkGetQueryPoolResults);

	Counters::SetDriverTiming(LibKernel::KernelGetTscFrequency(), MeasureTimerTicks());
}

#undef KYTY_TIME_VK

} // namespace Libs::Graphics
