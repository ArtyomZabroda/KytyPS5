#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_

#include "common/assert.h"
#include "graphics/host_gpu/renderer/commandStream.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace Libs::Graphics {

// Every Vulkan command the renderer records on the GPU thread goes through this class. It has the
// subset of vk::CommandBuffer methods the renderer uses, with the same names and arguments, so call
// sites read as before. Without a stream each method calls the driver at once. With one
// (--threaded-recording), the command and copies of the memory its arguments point to go to the
// stream, and its recording thread calls the driver later; callers may reuse that memory as soon
// as the method returns. The stream cannot copy pNext chains, so recorded structures have none.
class CommandRecorder final {
public:
	CommandRecorder() = default;
	explicit CommandRecorder(vk::CommandBuffer buffer, CommandStream* stream = nullptr) noexcept
	    : m_buffer(buffer), m_stream(stream) {}

	[[nodiscard]] explicit operator bool() const noexcept { return m_buffer != nullptr; }

	// Synchronization.
	void pipelineBarrier(vk::PipelineStageFlags src_stages, vk::PipelineStageFlags dst_stages,
	                     vk::DependencyFlags flags, uint32_t memory_count,
	                     const vk::MemoryBarrier* memory, uint32_t buffer_count,
	                     const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
	                     const vk::ImageMemoryBarrier* images) const {
		if (m_stream == nullptr) {
			m_buffer.pipelineBarrier(src_stages, dst_stages, flags, memory_count, memory,
			                         buffer_count, buffers, image_count, images);
			return;
		}
		ExpectNoChains(memory, memory_count);
		ExpectNoChains(buffers, buffer_count);
		ExpectNoChains(images, image_count);
		m_stream->RecordWithData(
		    Bytes(memory, memory_count) + Bytes(buffers, buffer_count) + Bytes(images, image_count),
		    [&] {
			    const auto* memory_copy  = m_stream->Copy(memory, memory_count);
			    const auto* buffers_copy = m_stream->Copy(buffers, buffer_count);
			    const auto* images_copy  = m_stream->Copy(images, image_count);
			    return [=](vk::CommandBuffer buffer) {
				    buffer.pipelineBarrier(src_stages, dst_stages, flags, memory_count, memory_copy,
				                           buffer_count, buffers_copy, image_count, images_copy);
			    };
		    });
	}
	void pipelineBarrier2(const vk::DependencyInfo& dependency) const {
		if (m_stream == nullptr) {
			m_buffer.pipelineBarrier2(dependency);
			return;
		}
		EXIT_IF(dependency.pNext != nullptr);
		ExpectNoChains(dependency.pMemoryBarriers, dependency.memoryBarrierCount);
		ExpectNoChains(dependency.pBufferMemoryBarriers, dependency.bufferMemoryBarrierCount);
		ExpectNoChains(dependency.pImageMemoryBarriers, dependency.imageMemoryBarrierCount);
		m_stream->RecordWithData(
		    Bytes(&dependency, 1) +
		        Bytes(dependency.pMemoryBarriers, dependency.memoryBarrierCount) +
		        Bytes(dependency.pBufferMemoryBarriers, dependency.bufferMemoryBarrierCount) +
		        Bytes(dependency.pImageMemoryBarriers, dependency.imageMemoryBarrierCount),
		    [&] {
			    auto* copy = m_stream->Copy(&dependency, 1);
			    copy->pMemoryBarriers =
			        m_stream->Copy(dependency.pMemoryBarriers, dependency.memoryBarrierCount);
			    copy->pBufferMemoryBarriers = m_stream->Copy(dependency.pBufferMemoryBarriers,
			                                                 dependency.bufferMemoryBarrierCount);
			    copy->pImageMemoryBarriers  = m_stream->Copy(dependency.pImageMemoryBarriers,
			                                                 dependency.imageMemoryBarrierCount);
			    return [copy](vk::CommandBuffer buffer) { buffer.pipelineBarrier2(*copy); };
		    });
	}

	// Rendering.
	void beginRendering(const vk::RenderingInfo& info) const {
		if (m_stream == nullptr) {
			m_buffer.beginRendering(info);
			return;
		}
		EXIT_IF(info.pNext != nullptr);
		ExpectNoChains(info.pColorAttachments, info.colorAttachmentCount);
		ExpectNoChains(info.pDepthAttachment, 1);
		ExpectNoChains(info.pStencilAttachment, 1);
		m_stream->RecordWithData(
		    Bytes(&info, 1) + Bytes(info.pColorAttachments, info.colorAttachmentCount) +
		        Bytes(info.pDepthAttachment, 1) + Bytes(info.pStencilAttachment, 1),
		    [&] {
			    auto* copy = m_stream->Copy(&info, 1);
			    copy->pColorAttachments =
			        m_stream->Copy(info.pColorAttachments, info.colorAttachmentCount);
			    copy->pDepthAttachment   = m_stream->Copy(info.pDepthAttachment, 1);
			    copy->pStencilAttachment = m_stream->Copy(info.pStencilAttachment, 1);
			    return [copy](vk::CommandBuffer buffer) { buffer.beginRendering(copy); };
		    });
	}
	void beginRendering(const vk::RenderingInfo* info) const { beginRendering(*info); }
	void endRendering() const {
		Record([](vk::CommandBuffer buffer) { buffer.endRendering(); });
	}
	void beginConditionalRenderingEXT(const vk::ConditionalRenderingBeginInfoEXT* info) const {
		if (m_stream == nullptr) {
			m_buffer.beginConditionalRenderingEXT(info);
			return;
		}
		EXIT_IF(info->pNext != nullptr);
		m_stream->Record([copy = *info](vk::CommandBuffer buffer) {
			buffer.beginConditionalRenderingEXT(&copy);
		});
	}
	void endConditionalRenderingEXT() const {
		Record([](vk::CommandBuffer buffer) { buffer.endConditionalRenderingEXT(); });
	}
	void clearAttachments(uint32_t attachment_count, const vk::ClearAttachment* attachments,
	                      uint32_t rect_count, const vk::ClearRect* rects) const {
		if (m_stream == nullptr) {
			m_buffer.clearAttachments(attachment_count, attachments, rect_count, rects);
			return;
		}
		m_stream->RecordWithData(
		    Bytes(attachments, attachment_count) + Bytes(rects, rect_count), [&] {
			    const auto* attachments_copy = m_stream->Copy(attachments, attachment_count);
			    const auto* rects_copy       = m_stream->Copy(rects, rect_count);
			    return [=](vk::CommandBuffer buffer) {
				    buffer.clearAttachments(attachment_count, attachments_copy, rect_count,
				                            rects_copy);
			    };
		    });
	}

	// Bindings.
	void bindPipeline(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) const {
		Record([=](vk::CommandBuffer buffer) { buffer.bindPipeline(bind_point, pipeline); });
	}
	void bindDescriptorSets(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                        uint32_t first_set, uint32_t set_count, const vk::DescriptorSet* sets,
	                        uint32_t dynamic_offset_count, const uint32_t* dynamic_offsets) const {
		if (m_stream == nullptr) {
			m_buffer.bindDescriptorSets(bind_point, layout, first_set, set_count, sets,
			                            dynamic_offset_count, dynamic_offsets);
			return;
		}
		m_stream->RecordWithData(
		    Bytes(sets, set_count) + Bytes(dynamic_offsets, dynamic_offset_count), [&] {
			    const auto* sets_copy    = m_stream->Copy(sets, set_count);
			    const auto* offsets_copy = m_stream->Copy(dynamic_offsets, dynamic_offset_count);
			    return [=](vk::CommandBuffer buffer) {
				    buffer.bindDescriptorSets(bind_point, layout, first_set, set_count, sets_copy,
				                              dynamic_offset_count, offsets_copy);
			    };
		    });
	}
	void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                          uint32_t set, uint32_t write_count,
	                          const vk::WriteDescriptorSet* writes) const {
		if (m_stream == nullptr) {
			m_buffer.pushDescriptorSetKHR(bind_point, layout, set, write_count, writes);
			return;
		}
		size_t bytes = Bytes(writes, write_count);
		for (uint32_t i = 0; i < write_count; i++) {
			bytes += DescriptorBytes(writes[i]);
		}
		m_stream->RecordWithData(bytes, [&] {
			auto* copy = m_stream->Copy(writes, write_count);
			for (uint32_t i = 0; i < write_count; i++) {
				CopyDescriptors(copy[i]);
			}
			return [=](vk::CommandBuffer buffer) {
				buffer.pushDescriptorSetKHR(bind_point, layout, set, write_count, copy);
			};
		});
	}
	void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                          uint32_t set, std::span<const vk::WriteDescriptorSet> writes) const {
		pushDescriptorSetKHR(bind_point, layout, set, static_cast<uint32_t>(writes.size()),
		                     writes.data());
	}
	void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset,
	                   uint32_t size, const void* values) const {
		if (m_stream == nullptr) {
			m_buffer.pushConstants(layout, stages, offset, size, values);
			return;
		}
		const auto* bytes = static_cast<const std::byte*>(values);
		m_stream->RecordWithData(Bytes(bytes, size), [&] {
			const auto* copy = m_stream->Copy(bytes, size);
			return [=](vk::CommandBuffer buffer) {
				buffer.pushConstants(layout, stages, offset, size, copy);
			};
		});
	}
	void bindVertexBuffers2(uint32_t first_binding, uint32_t binding_count,
	                        const vk::Buffer* buffers, const vk::DeviceSize* offsets,
	                        const vk::DeviceSize* sizes, const vk::DeviceSize* strides) const {
		if (m_stream == nullptr) {
			m_buffer.bindVertexBuffers2(first_binding, binding_count, buffers, offsets, sizes,
			                            strides);
			return;
		}
		m_stream->RecordWithData(
		    Bytes(buffers, binding_count) + Bytes(offsets, binding_count) +
		        Bytes(sizes, binding_count) + Bytes(strides, binding_count),
		    [&] {
			    const auto* buffers_copy = m_stream->Copy(buffers, binding_count);
			    const auto* offsets_copy = m_stream->Copy(offsets, binding_count);
			    const auto* sizes_copy   = m_stream->Copy(sizes, binding_count);
			    const auto* strides_copy = m_stream->Copy(strides, binding_count);
			    return [=](vk::CommandBuffer buffer) {
				    buffer.bindVertexBuffers2(first_binding, binding_count, buffers_copy,
				                              offsets_copy, sizes_copy, strides_copy);
			    };
		    });
	}
	void bindIndexBuffer(vk::Buffer index_buffer, vk::DeviceSize offset, vk::IndexType type) const {
		Record(
		    [=](vk::CommandBuffer buffer) { buffer.bindIndexBuffer(index_buffer, offset, type); });
	}

	// Draws and dispatches.
	void draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
	          uint32_t first_instance) const {
		Record([=](vk::CommandBuffer buffer) {
			buffer.draw(vertex_count, instance_count, first_vertex, first_instance);
		});
	}
	void drawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index,
	                 int32_t vertex_offset, uint32_t first_instance) const {
		Record([=](vk::CommandBuffer buffer) {
			buffer.drawIndexed(index_count, instance_count, first_index, vertex_offset,
			                   first_instance);
		});
	}
	void drawIndirect(vk::Buffer args, vk::DeviceSize offset, uint32_t draw_count,
	                  uint32_t stride) const {
		Record([=](vk::CommandBuffer buffer) {
			buffer.drawIndirect(args, offset, draw_count, stride);
		});
	}
	void drawIndexedIndirect(vk::Buffer args, vk::DeviceSize offset, uint32_t draw_count,
	                         uint32_t stride) const {
		Record([=](vk::CommandBuffer buffer) {
			buffer.drawIndexedIndirect(args, offset, draw_count, stride);
		});
	}
	void drawMeshTasksEXT(uint32_t group_count_x, uint32_t group_count_y,
	                      uint32_t group_count_z) const {
		Record([=](vk::CommandBuffer buffer) {
			buffer.drawMeshTasksEXT(group_count_x, group_count_y, group_count_z);
		});
	}
	void dispatch(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z) const {
		Record([=](vk::CommandBuffer buffer) {
			buffer.dispatch(group_count_x, group_count_y, group_count_z);
		});
	}
	void dispatchBase(uint32_t base_x, uint32_t base_y, uint32_t base_z, uint32_t group_count_x,
	                  uint32_t group_count_y, uint32_t group_count_z) const {
		Record([=](vk::CommandBuffer buffer) {
			buffer.dispatchBase(base_x, base_y, base_z, group_count_x, group_count_y,
			                    group_count_z);
		});
	}
	void dispatchIndirect(vk::Buffer args, vk::DeviceSize offset) const {
		Record([=](vk::CommandBuffer buffer) { buffer.dispatchIndirect(args, offset); });
	}

	// Transfers.
	void copyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t region_count,
	                const vk::BufferCopy* regions) const {
		if (m_stream == nullptr) {
			m_buffer.copyBuffer(source, destination, region_count, regions);
			return;
		}
		m_stream->RecordWithData(Bytes(regions, region_count), [&] {
			const auto* copy = m_stream->Copy(regions, region_count);
			return [=](vk::CommandBuffer buffer) {
				buffer.copyBuffer(source, destination, region_count, copy);
			};
		});
	}
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, uint32_t region_count,
	               const vk::ImageCopy* regions) const {
		if (m_stream == nullptr) {
			m_buffer.copyImage(source, source_layout, destination, destination_layout, region_count,
			                   regions);
			return;
		}
		m_stream->RecordWithData(Bytes(regions, region_count), [&] {
			const auto* copy = m_stream->Copy(regions, region_count);
			return [=](vk::CommandBuffer buffer) {
				buffer.copyImage(source, source_layout, destination, destination_layout,
				                 region_count, copy);
			};
		});
	}
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, const vk::ImageCopy& region) const {
		copyImage(source, source_layout, destination, destination_layout, 1, &region);
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
	                       uint32_t region_count, const vk::BufferImageCopy* regions) const {
		if (m_stream == nullptr) {
			m_buffer.copyBufferToImage(source, destination, layout, region_count, regions);
			return;
		}
		m_stream->RecordWithData(Bytes(regions, region_count), [&] {
			const auto* copy = m_stream->Copy(regions, region_count);
			return [=](vk::CommandBuffer buffer) {
				buffer.copyBufferToImage(source, destination, layout, region_count, copy);
			};
		});
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
	                       const vk::BufferImageCopy& region) const {
		copyBufferToImage(source, destination, layout, 1, &region);
	}
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
	                       uint32_t region_count, const vk::BufferImageCopy* regions) const {
		if (m_stream == nullptr) {
			m_buffer.copyImageToBuffer(source, layout, destination, region_count, regions);
			return;
		}
		m_stream->RecordWithData(Bytes(regions, region_count), [&] {
			const auto* copy = m_stream->Copy(regions, region_count);
			return [=](vk::CommandBuffer buffer) {
				buffer.copyImageToBuffer(source, layout, destination, region_count, copy);
			};
		});
	}
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
	                       const vk::BufferImageCopy& region) const {
		copyImageToBuffer(source, layout, destination, 1, &region);
	}
	void resolveImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	                  vk::ImageLayout destination_layout, const vk::ImageResolve& region) const {
		if (m_stream == nullptr) {
			m_buffer.resolveImage(source, source_layout, destination, destination_layout, 1,
			                      &region);
			return;
		}
		m_stream->Record([=, copy = region](vk::CommandBuffer buffer) {
			buffer.resolveImage(source, source_layout, destination, destination_layout, 1, &copy);
		});
	}
	void fillBuffer(vk::Buffer destination, vk::DeviceSize offset, vk::DeviceSize size,
	                uint32_t data) const {
		Record(
		    [=](vk::CommandBuffer buffer) { buffer.fillBuffer(destination, offset, size, data); });
	}
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue* color,
	                     uint32_t range_count, const vk::ImageSubresourceRange* ranges) const {
		if (m_stream == nullptr) {
			m_buffer.clearColorImage(image, layout, color, range_count, ranges);
			return;
		}
		m_stream->RecordWithData(Bytes(ranges, range_count), [&] {
			const auto* ranges_copy = m_stream->Copy(ranges, range_count);
			return [=, color_copy = *color](vk::CommandBuffer buffer) {
				buffer.clearColorImage(image, layout, &color_copy, range_count, ranges_copy);
			};
		});
	}
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
	                            const vk::ClearDepthStencilValue* value, uint32_t range_count,
	                            const vk::ImageSubresourceRange* ranges) const {
		if (m_stream == nullptr) {
			m_buffer.clearDepthStencilImage(image, layout, value, range_count, ranges);
			return;
		}
		m_stream->RecordWithData(Bytes(ranges, range_count), [&] {
			const auto* ranges_copy = m_stream->Copy(ranges, range_count);
			return [=, value_copy = *value](vk::CommandBuffer buffer) {
				buffer.clearDepthStencilImage(image, layout, &value_copy, range_count, ranges_copy);
			};
		});
	}

	// Queries.
	void beginQuery(vk::QueryPool pool, uint32_t query, vk::QueryControlFlags flags) const {
		Record([=](vk::CommandBuffer buffer) { buffer.beginQuery(pool, query, flags); });
	}
	void endQuery(vk::QueryPool pool, uint32_t query) const {
		Record([=](vk::CommandBuffer buffer) { buffer.endQuery(pool, query); });
	}

	// Dynamic state.
	void setViewport(uint32_t first, uint32_t count, const vk::Viewport* viewports) const {
		if (m_stream == nullptr) {
			m_buffer.setViewport(first, count, viewports);
			return;
		}
		m_stream->RecordWithData(Bytes(viewports, count), [&] {
			const auto* copy = m_stream->Copy(viewports, count);
			return [=](vk::CommandBuffer buffer) { buffer.setViewport(first, count, copy); };
		});
	}
	void setScissor(uint32_t first, uint32_t count, const vk::Rect2D* scissors) const {
		if (m_stream == nullptr) {
			m_buffer.setScissor(first, count, scissors);
			return;
		}
		m_stream->RecordWithData(Bytes(scissors, count), [&] {
			const auto* copy = m_stream->Copy(scissors, count);
			return [=](vk::CommandBuffer buffer) { buffer.setScissor(first, count, copy); };
		});
	}
	void setViewportWithCount(uint32_t count, const vk::Viewport* viewports) const {
		if (m_stream == nullptr) {
			m_buffer.setViewportWithCount(count, viewports);
			return;
		}
		m_stream->RecordWithData(Bytes(viewports, count), [&] {
			const auto* copy = m_stream->Copy(viewports, count);
			return [=](vk::CommandBuffer buffer) { buffer.setViewportWithCount(count, copy); };
		});
	}
	void setScissorWithCount(uint32_t count, const vk::Rect2D* scissors) const {
		if (m_stream == nullptr) {
			m_buffer.setScissorWithCount(count, scissors);
			return;
		}
		m_stream->RecordWithData(Bytes(scissors, count), [&] {
			const auto* copy = m_stream->Copy(scissors, count);
			return [=](vk::CommandBuffer buffer) { buffer.setScissorWithCount(count, copy); };
		});
	}
	void setDepthTestEnable(vk::Bool32 enable) const {
		Record([=](vk::CommandBuffer buffer) { buffer.setDepthTestEnable(enable); });
	}
	void setDepthWriteEnable(vk::Bool32 enable) const {
		Record([=](vk::CommandBuffer buffer) { buffer.setDepthWriteEnable(enable); });
	}
	void setDepthCompareOp(vk::CompareOp op) const {
		Record([=](vk::CommandBuffer buffer) { buffer.setDepthCompareOp(op); });
	}
	void setDepthBoundsTestEnable(vk::Bool32 enable) const {
		Record([=](vk::CommandBuffer buffer) { buffer.setDepthBoundsTestEnable(enable); });
	}
	void setDepthBounds(float min, float max) const {
		Record([=](vk::CommandBuffer buffer) { buffer.setDepthBounds(min, max); });
	}
	void setDepthBiasEnable(vk::Bool32 enable) const {
		Record([=](vk::CommandBuffer buffer) { buffer.setDepthBiasEnable(enable); });
	}
	void setDepthBias(float constant, float clamp, float slope) const {
		Record([=](vk::CommandBuffer buffer) { buffer.setDepthBias(constant, clamp, slope); });
	}
	void setStencilTestEnable(vk::Bool32 enable) const {
		Record([=](vk::CommandBuffer buffer) { buffer.setStencilTestEnable(enable); });
	}
	void setStencilOp(vk::StencilFaceFlags faces, vk::StencilOp fail, vk::StencilOp pass,
	                  vk::StencilOp depth_fail, vk::CompareOp compare) const {
		Record([=](vk::CommandBuffer buffer) {
			buffer.setStencilOp(faces, fail, pass, depth_fail, compare);
		});
	}
	void setStencilReference(vk::StencilFaceFlags faces, uint32_t reference) const {
		Record([=](vk::CommandBuffer buffer) { buffer.setStencilReference(faces, reference); });
	}
	void setStencilWriteMask(vk::StencilFaceFlags faces, uint32_t mask) const {
		Record([=](vk::CommandBuffer buffer) { buffer.setStencilWriteMask(faces, mask); });
	}
	void setStencilCompareMask(vk::StencilFaceFlags faces, uint32_t mask) const {
		Record([=](vk::CommandBuffer buffer) { buffer.setStencilCompareMask(faces, mask); });
	}
	void setCullMode(vk::CullModeFlags mode) const {
		Record([=](vk::CommandBuffer buffer) { buffer.setCullMode(mode); });
	}
	void setFrontFace(vk::FrontFace face) const {
		Record([=](vk::CommandBuffer buffer) { buffer.setFrontFace(face); });
	}
	void setBlendConstants(const float constants[4]) const {
		const std::array<float, 4> copy {constants[0], constants[1], constants[2], constants[3]};
		Record([=](vk::CommandBuffer buffer) { buffer.setBlendConstants(copy.data()); });
	}
	void setColorWriteEnableEXT(uint32_t count, const vk::Bool32* enables) const {
		if (m_stream == nullptr) {
			m_buffer.setColorWriteEnableEXT(count, enables);
			return;
		}
		m_stream->RecordWithData(Bytes(enables, count), [&] {
			const auto* copy = m_stream->Copy(enables, count);
			return [=](vk::CommandBuffer buffer) { buffer.setColorWriteEnableEXT(count, copy); };
		});
	}
	void setLineWidth(float width) const {
		Record([=](vk::CommandBuffer buffer) { buffer.setLineWidth(width); });
	}
	void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags aspects) const {
		Record(
		    [=](vk::CommandBuffer buffer) { buffer.setAttachmentFeedbackLoopEnableEXT(aspects); });
	}

private:
	// Calls the driver now, or records a command whose arguments are all values.
	template <typename Fn>
	void Record(const Fn& fn) const {
		if (m_stream == nullptr) {
			fn(m_buffer);
		} else {
			m_stream->Record(fn);
		}
	}

	template <typename T>
	[[nodiscard]] static size_t Bytes(const T* items, size_t count) noexcept {
		return items == nullptr ? 0 : CommandStream::DataBytes<T>(count);
	}

	template <typename T>
	static void ExpectNoChains(const T* items, size_t count) {
		if (items == nullptr) {
			return;
		}
		for (size_t i = 0; i < count; i++) {
			EXIT_IF(items[i].pNext != nullptr);
		}
	}

	[[nodiscard]] static bool IsImageDescriptor(vk::DescriptorType type) noexcept {
		return type == vk::DescriptorType::eSampler ||
		       type == vk::DescriptorType::eCombinedImageSampler ||
		       type == vk::DescriptorType::eSampledImage ||
		       type == vk::DescriptorType::eStorageImage ||
		       type == vk::DescriptorType::eInputAttachment;
	}
	[[nodiscard]] static bool IsBufferDescriptor(vk::DescriptorType type) noexcept {
		return type == vk::DescriptorType::eUniformBuffer ||
		       type == vk::DescriptorType::eStorageBuffer ||
		       type == vk::DescriptorType::eUniformBufferDynamic ||
		       type == vk::DescriptorType::eStorageBufferDynamic;
	}
	[[nodiscard]] static bool IsTexelBufferDescriptor(vk::DescriptorType type) noexcept {
		return type == vk::DescriptorType::eUniformTexelBuffer ||
		       type == vk::DescriptorType::eStorageTexelBuffer;
	}

	// Space for the descriptor array a write points to; only the one its type reads.
	[[nodiscard]] static size_t DescriptorBytes(const vk::WriteDescriptorSet& write) {
		EXIT_IF(write.pNext != nullptr);
		const auto type = write.descriptorType;
		if (IsImageDescriptor(type)) {
			return Bytes(write.pImageInfo, write.descriptorCount);
		}
		if (IsBufferDescriptor(type)) {
			return Bytes(write.pBufferInfo, write.descriptorCount);
		}
		EXIT_IF(!IsTexelBufferDescriptor(type));
		return Bytes(write.pTexelBufferView, write.descriptorCount);
	}

	// Points a copied write at stream copies of its descriptors.
	void CopyDescriptors(vk::WriteDescriptorSet& write) const {
		const auto type  = write.descriptorType;
		const auto count = write.descriptorCount;
		write.pImageInfo =
		    IsImageDescriptor(type) ? m_stream->Copy(write.pImageInfo, count) : nullptr;
		write.pBufferInfo =
		    IsBufferDescriptor(type) ? m_stream->Copy(write.pBufferInfo, count) : nullptr;
		write.pTexelBufferView =
		    IsTexelBufferDescriptor(type) ? m_stream->Copy(write.pTexelBufferView, count) : nullptr;
	}

	vk::CommandBuffer m_buffer = nullptr;
	CommandStream*    m_stream = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
