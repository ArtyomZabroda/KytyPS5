#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANCALLTIMING_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANCALLTIMING_H_

namespace Libs::Graphics {

// Diagnostics (--vk-call-timing): wraps the Vulkan entry points the renderer records and submits
// with in the default dispatcher, so the performance overlay can show how long the GPU thread
// spends inside the driver. Call once, right after the dispatcher is initialized with the device.
void InstallVulkanCallTiming();

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANCALLTIMING_H_
