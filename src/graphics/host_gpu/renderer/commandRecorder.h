#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <span>

namespace Libs::Graphics {

// Every Vulkan command the renderer records on the GPU thread goes through this class. It has the
// subset of vk::CommandBuffer methods the renderer uses, with the same names and arguments, so call
// sites read as before. For now each method calls the driver at once; keeping the commands in one
// place lets them be recorded on another thread later.
class CommandRecorder final {
public:
	CommandRecorder() = default;
	explicit CommandRecorder(vk::CommandBuffer buffer) noexcept: m_buffer(buffer) {}

	[[nodiscard]] explicit operator bool() const noexcept { return m_buffer != nullptr; }

	// Synchronization.
	void pipelineBarrier(vk::PipelineStageFlags src_stages, vk::PipelineStageFlags dst_stages,
	                     vk::DependencyFlags flags, uint32_t memory_count,
	                     const vk::MemoryBarrier* memory, uint32_t buffer_count,
	                     const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
	                     const vk::ImageMemoryBarrier* images) const {
		m_buffer.pipelineBarrier(src_stages, dst_stages, flags, memory_count, memory, buffer_count,
		                         buffers, image_count, images);
	}
	void pipelineBarrier2(const vk::DependencyInfo& dependency) const {
		m_buffer.pipelineBarrier2(dependency);
	}

	// Rendering.
	void beginRendering(const vk::RenderingInfo& info) const { m_buffer.beginRendering(info); }
	void beginRendering(const vk::RenderingInfo* info) const { m_buffer.beginRendering(info); }
	void endRendering() const { m_buffer.endRendering(); }
	void beginConditionalRenderingEXT(const vk::ConditionalRenderingBeginInfoEXT* info) const {
		m_buffer.beginConditionalRenderingEXT(info);
	}
	void endConditionalRenderingEXT() const { m_buffer.endConditionalRenderingEXT(); }
	void clearAttachments(uint32_t attachment_count, const vk::ClearAttachment* attachments,
	                      uint32_t rect_count, const vk::ClearRect* rects) const {
		m_buffer.clearAttachments(attachment_count, attachments, rect_count, rects);
	}

	// Bindings.
	void bindPipeline(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) const {
		m_buffer.bindPipeline(bind_point, pipeline);
	}
	void bindDescriptorSets(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                        uint32_t first_set, uint32_t set_count, const vk::DescriptorSet* sets,
	                        uint32_t dynamic_offset_count, const uint32_t* dynamic_offsets) const {
		m_buffer.bindDescriptorSets(bind_point, layout, first_set, set_count, sets,
		                            dynamic_offset_count, dynamic_offsets);
	}
	void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                          uint32_t set, uint32_t write_count,
	                          const vk::WriteDescriptorSet* writes) const {
		m_buffer.pushDescriptorSetKHR(bind_point, layout, set, write_count, writes);
	}
	void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                          uint32_t set, std::span<const vk::WriteDescriptorSet> writes) const {
		m_buffer.pushDescriptorSetKHR(bind_point, layout, set, static_cast<uint32_t>(writes.size()),
		                              writes.data());
	}
	void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset,
	                   uint32_t size, const void* values) const {
		m_buffer.pushConstants(layout, stages, offset, size, values);
	}
	void bindVertexBuffers2(uint32_t first_binding, uint32_t binding_count,
	                        const vk::Buffer* buffers, const vk::DeviceSize* offsets,
	                        const vk::DeviceSize* sizes, const vk::DeviceSize* strides) const {
		m_buffer.bindVertexBuffers2(first_binding, binding_count, buffers, offsets, sizes, strides);
	}
	void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) const {
		m_buffer.bindIndexBuffer(buffer, offset, type);
	}

	// Draws and dispatches.
	void draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
	          uint32_t first_instance) const {
		m_buffer.draw(vertex_count, instance_count, first_vertex, first_instance);
	}
	void drawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index,
	                 int32_t vertex_offset, uint32_t first_instance) const {
		m_buffer.drawIndexed(index_count, instance_count, first_index, vertex_offset,
		                     first_instance);
	}
	void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t draw_count,
	                  uint32_t stride) const {
		m_buffer.drawIndirect(buffer, offset, draw_count, stride);
	}
	void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t draw_count,
	                         uint32_t stride) const {
		m_buffer.drawIndexedIndirect(buffer, offset, draw_count, stride);
	}
	void drawMeshTasksEXT(uint32_t group_count_x, uint32_t group_count_y,
	                      uint32_t group_count_z) const {
		m_buffer.drawMeshTasksEXT(group_count_x, group_count_y, group_count_z);
	}
	void dispatch(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z) const {
		m_buffer.dispatch(group_count_x, group_count_y, group_count_z);
	}
	void dispatchBase(uint32_t base_x, uint32_t base_y, uint32_t base_z, uint32_t group_count_x,
	                  uint32_t group_count_y, uint32_t group_count_z) const {
		m_buffer.dispatchBase(base_x, base_y, base_z, group_count_x, group_count_y, group_count_z);
	}
	void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) const {
		m_buffer.dispatchIndirect(buffer, offset);
	}

	// Transfers.
	void copyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t region_count,
	                const vk::BufferCopy* regions) const {
		m_buffer.copyBuffer(source, destination, region_count, regions);
	}
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, uint32_t region_count,
	               const vk::ImageCopy* regions) const {
		m_buffer.copyImage(source, source_layout, destination, destination_layout, region_count,
		                   regions);
	}
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, const vk::ImageCopy& region) const {
		copyImage(source, source_layout, destination, destination_layout, 1, &region);
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
	                       uint32_t region_count, const vk::BufferImageCopy* regions) const {
		m_buffer.copyBufferToImage(source, destination, layout, region_count, regions);
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
	                       const vk::BufferImageCopy& region) const {
		copyBufferToImage(source, destination, layout, 1, &region);
	}
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
	                       uint32_t region_count, const vk::BufferImageCopy* regions) const {
		m_buffer.copyImageToBuffer(source, layout, destination, region_count, regions);
	}
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
	                       const vk::BufferImageCopy& region) const {
		copyImageToBuffer(source, layout, destination, 1, &region);
	}
	void resolveImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	                  vk::ImageLayout destination_layout, const vk::ImageResolve& region) const {
		m_buffer.resolveImage(source, source_layout, destination, destination_layout, 1, &region);
	}
	void fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
	                uint32_t data) const {
		m_buffer.fillBuffer(buffer, offset, size, data);
	}
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue* color,
	                     uint32_t range_count, const vk::ImageSubresourceRange* ranges) const {
		m_buffer.clearColorImage(image, layout, color, range_count, ranges);
	}
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
	                            const vk::ClearDepthStencilValue* value, uint32_t range_count,
	                            const vk::ImageSubresourceRange* ranges) const {
		m_buffer.clearDepthStencilImage(image, layout, value, range_count, ranges);
	}

	// Queries.
	void beginQuery(vk::QueryPool pool, uint32_t query, vk::QueryControlFlags flags) const {
		m_buffer.beginQuery(pool, query, flags);
	}
	void endQuery(vk::QueryPool pool, uint32_t query) const { m_buffer.endQuery(pool, query); }

	// Dynamic state.
	void setViewport(uint32_t first, uint32_t count, const vk::Viewport* viewports) const {
		m_buffer.setViewport(first, count, viewports);
	}
	void setScissor(uint32_t first, uint32_t count, const vk::Rect2D* scissors) const {
		m_buffer.setScissor(first, count, scissors);
	}
	void setViewportWithCount(uint32_t count, const vk::Viewport* viewports) const {
		m_buffer.setViewportWithCount(count, viewports);
	}
	void setScissorWithCount(uint32_t count, const vk::Rect2D* scissors) const {
		m_buffer.setScissorWithCount(count, scissors);
	}
	void setDepthTestEnable(vk::Bool32 enable) const { m_buffer.setDepthTestEnable(enable); }
	void setDepthWriteEnable(vk::Bool32 enable) const { m_buffer.setDepthWriteEnable(enable); }
	void setDepthCompareOp(vk::CompareOp op) const { m_buffer.setDepthCompareOp(op); }
	void setDepthBoundsTestEnable(vk::Bool32 enable) const {
		m_buffer.setDepthBoundsTestEnable(enable);
	}
	void setDepthBounds(float min, float max) const { m_buffer.setDepthBounds(min, max); }
	void setDepthBiasEnable(vk::Bool32 enable) const { m_buffer.setDepthBiasEnable(enable); }
	void setDepthBias(float constant, float clamp, float slope) const {
		m_buffer.setDepthBias(constant, clamp, slope);
	}
	void setStencilTestEnable(vk::Bool32 enable) const { m_buffer.setStencilTestEnable(enable); }
	void setStencilOp(vk::StencilFaceFlags faces, vk::StencilOp fail, vk::StencilOp pass,
	                  vk::StencilOp depth_fail, vk::CompareOp compare) const {
		m_buffer.setStencilOp(faces, fail, pass, depth_fail, compare);
	}
	void setStencilReference(vk::StencilFaceFlags faces, uint32_t reference) const {
		m_buffer.setStencilReference(faces, reference);
	}
	void setStencilWriteMask(vk::StencilFaceFlags faces, uint32_t mask) const {
		m_buffer.setStencilWriteMask(faces, mask);
	}
	void setStencilCompareMask(vk::StencilFaceFlags faces, uint32_t mask) const {
		m_buffer.setStencilCompareMask(faces, mask);
	}
	void setCullMode(vk::CullModeFlags mode) const { m_buffer.setCullMode(mode); }
	void setFrontFace(vk::FrontFace face) const { m_buffer.setFrontFace(face); }
	void setBlendConstants(const float constants[4]) const {
		m_buffer.setBlendConstants(constants);
	}
	void setColorWriteEnableEXT(uint32_t count, const vk::Bool32* enables) const {
		m_buffer.setColorWriteEnableEXT(count, enables);
	}
	void setLineWidth(float width) const { m_buffer.setLineWidth(width); }
	void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags aspects) const {
		m_buffer.setAttachmentFeedbackLoopEnableEXT(aspects);
	}

private:
	vk::CommandBuffer m_buffer = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
