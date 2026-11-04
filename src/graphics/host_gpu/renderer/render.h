#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <array>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
#include <span>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
class RenderContext;
class CommandScheduler;

// The T# dwords and the shader's use of them, which together fix a texture's image description.
using TextureShapeKey = std::array<uint32_t, 14>;

struct TextureShapeKeyHash {
	[[nodiscard]] size_t operator()(const TextureShapeKey& key) const noexcept;
};

struct RenderExecutorTestAccess;

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	DispatchIndirect = 9,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// Nonzero address: the GPU reads count Vulkan indirect draw records, stride bytes apart.
struct DrawIndirectArgs {
	uint64_t              address = 0;
	uint32_t              count   = 0;
	uint32_t              stride  = 0;
	ShaderIndirectOffsets offsets;
};

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// With indirect records, index_count is INDEX_BUFFER_SIZE.
	DrawIndirectArgs indirect;
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	DrawIndirectArgs indirect;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const;

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	void BeginRendering(const RenderState& state) const;
	// The caller's location is counted for the overlay's render pass breakdown.
	void EndRendering(std::source_location location = std::source_location::current()) const;
	// Binding state lasts for the whole command buffer, so a draw that uses the pipeline already
	// bound skips vkCmdBindPipeline.
	void BindGraphicsPipeline(vk::Pipeline pipeline) const;
	// Code that binds a graphics pipeline itself calls this, so the next draw binds again.
	void InvalidateGraphicsPipeline() const noexcept { m_graphics_pipeline = nullptr; }

	// Records the renderer's commands. Handle() remains for the scheduler and for code that records
	// through other APIs (presentation, ImGui). With --threaded-recording it first waits until the
	// recording thread has replayed every command, so direct recording keeps its order.
	[[nodiscard]] CommandRecorder   Recorder() const;
	[[nodiscard]] vk::CommandBuffer Handle() const;
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&      GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] HW::UserConfig&   GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] HW::Shader&       GetShaders() const noexcept { return *m_shaders; }

	// The registers the accessors above return. A queued draw commits with its own copies.
	struct RegisterSet {
		HW::Context*    registers   = nullptr;
		HW::UserConfig* user_config = nullptr;
		HW::Shader*     shaders     = nullptr;
	};
	RegisterSet SwapRegisters(const RegisterSet& set) noexcept {
		const RegisterSet previous {m_registers, m_user_config, m_shaders};
		m_registers   = set.registers;
		m_user_config = set.user_config;
		m_shaders     = set.shaders;
		return previous;
	}

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	vk::CommandBuffer   m_buffer          = nullptr;
	CommandStream*      m_stream          = nullptr;
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	// The graphics pipeline bound in this command buffer; null until a draw binds one.
	mutable vk::Pipeline m_graphics_pipeline = nullptr;

	friend class CommandScheduler;
};

namespace Pssr {
struct Request;
}

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	~RenderExecutor();
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode,
	                    bool async_compute);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode, bool async_compute);
	void ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request);

	// --parallel-draw-prep: draws wait in order, each with a copy of the registers, until they
	// commit, while a worker thread prepares them. Without the flag they are drawn at once.
	void QueueDrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	void QueueDrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);
	void CommitQueuedDraws(CommandBuffer& buffer);
	[[nodiscard]] bool HasQueuedDraws() const noexcept { return m_queue_head != m_queue_tail; }

	// Decoding reads only the stage's snapshot; the other steps use the caches.
	void DecodeBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	// With decoded, takes its decoding instead of decoding again.
	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared,
	                     PreparedBindings* decoded = nullptr);
	void DecodeBuffers(std::span<PreparedBindings* const> stages);
	void FindDecodedBuffers(std::span<PreparedBindings* const> stages);
	void FindBuffers(std::span<PreparedBindings* const> stages);
	void RebindBuffers(PreparedBindings& bindings);
	void RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);
	[[nodiscard]] DecodedTexture DecodeTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                           const ShaderRecompiler::IR::DescriptorValue& value);
	[[nodiscard]] TextureBinding BindTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                         DecodedTexture&                            decoded);

private:
	// False, with nothing drawn, when the draw's indirect records need CPU counts.
	bool DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	bool DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo>         colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	// The registers alone decide whether DepthStencilCopy copies.
	struct DepthStencilCopies {
		bool depth   = false;
		bool stencil = false;
	};
	[[nodiscard]] static DepthStencilCopies FindDepthStencilCopies(const HW::Context& hw);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                    const DrawEmitInfo& emit, DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                          uint32_t         render_target_slice_offset,
	                                          DrawRenderState& state);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, const DrawEmitInfo& emit,
	                         const DrawIndexBufferSource& index_source);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t       render_target_slice_offset);
	[[nodiscard]] bool        ConsumeMetadataColorOperation(CommandBuffer& buffer,
	                                                        uint32_t       render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeFill(const ShaderComputeInputInfo& input,
	                                         CommandBuffer& command, uint32_t group_x,
	                                         uint32_t group_y, uint32_t group_z, uint32_t mode);

	// A queued draw with copies of its registers, and what the worker prepared for it.
	struct QueuedDraw {
		// Changed under m_prep_mutex. The worker owns the draw while it is Preparing.
		enum class Prep : uint8_t { Queued, Preparing, Ready, Serial };
		HW::Context    registers;
		HW::UserConfig user_config;
		HW::Shader     shaders;
		uint64_t       submit_id = 0;
		bool           indexed   = false;
		DrawIndexArgs  index_args;
		DrawAutoArgs   auto_args;
		Prep           prep = Prep::Queued;
		// What the preparation depends on besides the registers, checked at commit.
		uint64_t dirty_epoch        = 0;
		uint64_t program_generation = 0;
		// Whether bindings hold the decoding of the stages: vertex 0-2, pixel 3.
		bool                              decoded = false;
		DrawRenderState                   state;
		std::array<DrawStageResources, 4> resources;
		std::array<PreparedBindings, 4>   bindings;
		// --parallel-heap-prep: the image heap views the programs used, acquired again at commit.
		std::vector<ShaderRecompiler::IR::ImageHeapPeek> heap_peeks;
		// --parallel-buffer-prep: buffer bytes the worker copied for the commit.
		BufferCache::PrestreamDraw prestream;
	};
	static constexpr size_t MaxQueuedDraws = 32;

	QueuedDraw&        QueueDraw(uint64_t submit_id, CommandBuffer& buffer);
	void               PublishQueuedDraw(CommandBuffer& buffer);
	[[nodiscard]] bool OldestDrawSettled();
	void               CommitOldestDraw(CommandBuffer& buffer);
	void               RunDrawPrepWorker(const std::stop_token& stop);
	[[nodiscard]] bool PrepareQueuedDraw(QueuedDraw& draw);
	// --verify-draw-prep compares these with the GPU thread's own results.
	void VerifyPreparedPrograms(const CommandBuffer& buffer, DrawRenderState& state);
	// Acquires the views the worker used; false if any changed since.
	[[nodiscard]] bool AcquirePeekedHeaps(const QueuedDraw& draw);
	// Worker: copies the draw's small read-only CPU-written buffers ahead of its commit.
	void PrestreamBuffers(QueuedDraw& draw);
	void AdoptDecodedBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared,
	                          PreparedBindings& decoded);

	RenderContext&       m_context;
	GraphicsBindings     m_graphics_bindings;
	PreparedBindings     m_compute_bindings;
	std::vector<ImageId> m_bound_images;
	// Per color slot: the registers of the last resolved render target and their decoding.
	struct ColorTargetDecode {
		HW::RenderTarget registers;
		uint32_t         mask         = 0;
		uint32_t         slice_offset = 0;
		bool             valid        = false;
		RenderColorInfo  info;
	};
	std::array<ColorTargetDecode, 8> m_color_target_decodes;
	// Draws bind the same textures every frame; their descriptions depend on nothing else.
	std::unordered_map<TextureShapeKey, TextureShape, TextureShapeKeyHash> m_texture_shapes;
	std::vector<vk::DescriptorImageInfo>                                   m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>                                    m_descriptor_writes;
	vk::DescriptorSetLayout m_pssr_descriptor_layout = nullptr;
	vk::PipelineLayout      m_pssr_pipeline_layout   = nullptr;
	vk::Pipeline            m_pssr_pipeline          = nullptr;
	// The resources of the draw being prepared; reused, so they keep their capacity.
	std::array<DrawStageResources, 4> m_draw_resources;
	// The ring of queued draws: committed up to m_queue_head, published up to m_queue_tail, and
	// taken by the worker from m_prep_next. The GPU thread moves head and tail under m_prep_mutex.
	std::array<std::unique_ptr<QueuedDraw>, MaxQueuedDraws> m_queued_draws;
	size_t                                                  m_queue_head = 0;
	size_t                                                  m_queue_tail = 0;
	size_t                                                  m_prep_next  = 0;
	std::mutex                                              m_prep_mutex;
	std::condition_variable_any                             m_prep_work;
	std::condition_variable                                 m_prep_done;
	// The draw committing with its preparation, if any.
	QueuedDraw*                      m_commit_job = nullptr;
	std::unique_ptr<DrawRenderState> m_verify_state;
	// Last, so it stops before the members it uses are destroyed.
	std::jthread m_prep_worker;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
