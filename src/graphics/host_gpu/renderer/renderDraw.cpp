#include "graphics/host_gpu/renderer/renderDraw.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

// Diagnostics: how much of a draw's bindings equals the previous draw's. Runs on the GPU
// thread, and only while counters are enabled.
class RepeatHash {
public:
	template <typename T>
	void Add(const T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		AddBytes(&value, sizeof(value));
	}
	template <typename T>
	void AddRange(std::span<const T> values) {
		static_assert(std::is_trivially_copyable_v<T>);
		Add(values.size());
		AddBytes(values.data(), values.size_bytes());
	}
	[[nodiscard]] uint64_t Value() const { return m_hash; }

private:
	void AddBytes(const void* data, size_t size) {
		const auto* bytes = static_cast<const uint8_t*>(data);
		for (size_t i = 0; i < size; i++) {
			m_hash = (m_hash ^ bytes[i]) * 0x100000001b3ull;
		}
	}
	uint64_t m_hash = 0xcbf29ce484222325ull;
};

struct StageRepeatState {
	const void* module   = nullptr;
	uint64_t    snapshot = 0;
	uint64_t    images   = 0;
	uint64_t    ranges   = 0;
	uint64_t    buffers  = 0;
	uint64_t    data     = 0;
	uint64_t    samplers = 0;
	bool        valid    = false;
};

struct DrawRepeatState {
	std::array<StageRepeatState, 4> stages {};
	const void*                     pipeline = nullptr;
	uint64_t                        targets  = 0;
};

DrawRepeatState g_draw_repeats;

// --xframe-draw-stats: keys of the draws of this frame and the previous one.
struct CrossFrameKeys {
	std::unordered_set<uint64_t> current;
	std::unordered_set<uint64_t> previous;

	void Rotate() {
		previous.swap(current);
		current.clear();
	}
	// Whether the key was seen this frame or the previous one; records it.
	[[nodiscard]] bool Seen(uint64_t key) {
		const bool seen = previous.contains(key) || current.contains(key);
		current.insert(key);
		return seen;
	}
};

CrossFrameKeys g_xframe_full;
CrossFrameKeys g_xframe_structure;
// The structure key with one more part each: UserConfig, arguments, stage registers, user SGPRs.
std::array<CrossFrameKeys, 4> g_xframe_parts;
// The user SGPRs (HS, GS, PS) of the last draw with each argument key.
using CrossFrameUserData = std::array<uint32_t, 3 * HW::UserSgprInfo::SGPRS_MAX>;
std::unordered_map<uint64_t, CrossFrameUserData> g_xframe_user_data;
// The argument key of the draw being committed, for CountCrossFrameBindings.
uint64_t g_xframe_draw_key   = 0;
bool     g_xframe_draw_keyed = false;

struct CrossFrameDescriptors {
	uint64_t              textures = 0;
	uint64_t              buffers  = 0;
	uint64_t              no_base  = 0; // V# words without the base address.
	uint64_t              shape    = 0; // V# words without the base address and the size.
	std::vector<uint64_t> bases;
};
std::unordered_map<uint64_t, CrossFrameDescriptors> g_xframe_descriptors;

// Compares the user SGPRs of a repeat with those of the last draw with its key: how many differ
// and whether each is half of a pair that addresses mapped guest memory.
void CountUserDataChanges(const RenderContext& context, uint64_t key,
                          const CrossFrameUserData& current) {
	using Common::DebugCounters::Counter;
	constexpr uint32_t Sgprs = HW::UserSgprInfo::SGPRS_MAX;
	const auto         found = g_xframe_user_data.find(key);
	if (found != g_xframe_user_data.end()) {
		const auto& previous = found->second;
		const auto  pointer  = [&](uint32_t low) {
			const auto value = (uint64_t {current[low + 1]} << 32u) | current[low];
			return value != 0 && context.IsMapped(value, 1);
		};
		uint64_t changed  = 0;
		uint64_t pointers = 0;
		for (uint32_t i = 0; i < current.size(); i++) {
			if (current[i] == previous[i]) continue;
			changed++;
			const auto slot = i % Sgprs;
			if ((slot + 1 < Sgprs && pointer(i)) || (slot > 0 && pointer(i - 1))) {
				pointers++;
			}
		}
		Common::DebugCounters::Add(Counter::DrawXframeUdDraws);
		Common::DebugCounters::Add(Counter::DrawXframeUdWords, changed);
		Common::DebugCounters::Add(Counter::DrawXframeUdPointers, pointers);
		if (changed == pointers) Common::DebugCounters::Add(Counter::DrawXframeUdRelocatable);
	}
	if (g_xframe_user_data.size() >= 1u << 20u) {
		g_xframe_user_data.clear();
	}
	g_xframe_user_data[key] = current;
}
uint64_t g_xframe_flip = 0;

// Counts whether the draw's registers repeat those of a draw of this or the previous frame.
// Returns true when --xframe-skip-repeats drops it (an upper-bound experiment: the frame renders
// wrong).
bool CrossFrameRepeat(const RenderContext& context, const CommandBuffer& buffer,
                      std::span<const uint64_t> draw_words) {
	using Common::DebugCounters::Counter;
	if (!Config::XframeDrawStatsEnabled()) {
		return false;
	}
	Common::DebugCounters::SetDrawPhase("(xframe) Key");
	const auto flip = Common::DebugCounters::g_flips.load(std::memory_order_relaxed);
	if (flip != g_xframe_flip) {
		g_xframe_flip = flip;
		g_xframe_full.Rotate();
		g_xframe_structure.Rotate();
		for (auto& part: g_xframe_parts) {
			part.Rotate();
		}
	}
	const auto&                   registers = buffer.GetRegisters();
	const auto&                   vs        = buffer.GetShaders().GetVs();
	const auto&                   ps        = buffer.GetShaders().GetPs();
	const std::array<uint64_t, 5> programs {vs.es_regs.data_addr, vs.ls_regs.data_addr,
	                                        vs.hs_regs.data_addr, vs.gs_regs.data_addr,
	                                        ps.ps_regs.data_addr};

	auto structure = XXH3_64bits(&registers, sizeof(registers));
	structure      = XXH3_64bits_withSeed(programs.data(), sizeof(programs), structure);
	auto full = XXH3_64bits_withSeed(&buffer.GetUserConfig(), sizeof(HW::UserConfig), structure);
	full      = XXH3_64bits_withSeed(&vs, sizeof(vs), full);
	full      = XXH3_64bits_withSeed(&ps, sizeof(ps), full);
	full      = XXH3_64bits_withSeed(draw_words.data(), draw_words.size_bytes(), full);

	const auto with_user_config =
	    XXH3_64bits_withSeed(&buffer.GetUserConfig(), sizeof(HW::UserConfig), structure);
	const auto with_args =
	    XXH3_64bits_withSeed(draw_words.data(), draw_words.size_bytes(), structure);
	auto with_stage = XXH3_64bits_withSeed(&vs.es_regs, sizeof(vs.es_regs), structure);
	with_stage      = XXH3_64bits_withSeed(&vs.ls_regs, sizeof(vs.ls_regs), with_stage);
	with_stage      = XXH3_64bits_withSeed(&vs.hs_regs, sizeof(vs.hs_regs), with_stage);
	with_stage      = XXH3_64bits_withSeed(&vs.gs_regs, sizeof(vs.gs_regs), with_stage);
	with_stage      = XXH3_64bits_withSeed(&ps.ps_regs, sizeof(ps.ps_regs), with_stage);
	auto with_user_data =
	    XXH3_64bits_withSeed(&vs.hs_user_sgpr, sizeof(vs.hs_user_sgpr), structure);
	with_user_data =
	    XXH3_64bits_withSeed(&vs.gs_user_sgpr, sizeof(vs.gs_user_sgpr), with_user_data);
	with_user_data =
	    XXH3_64bits_withSeed(&ps.ps_user_sgpr, sizeof(ps.ps_user_sgpr), with_user_data);

	const bool full_seen      = g_xframe_full.Seen(full);
	const bool structure_seen = g_xframe_structure.Seen(structure);
	const bool args_seen      = g_xframe_parts[1].Seen(with_args);
	Common::DebugCounters::Add(Counter::DrawXframeChecks);
	if (full_seen) Common::DebugCounters::Add(Counter::DrawXframeFull);
	if (structure_seen) Common::DebugCounters::Add(Counter::DrawXframeStructure);
	if (g_xframe_parts[0].Seen(with_user_config)) {
		Common::DebugCounters::Add(Counter::DrawXframeUserConfig);
	}
	if (args_seen) Common::DebugCounters::Add(Counter::DrawXframeArgs);
	g_xframe_draw_key   = with_args;
	g_xframe_draw_keyed = true;
	if (Common::DebugCounters::Enabled()) {
		CrossFrameUserData user_data {};
		constexpr auto     Sgprs = HW::UserSgprInfo::SGPRS_MAX;
		std::copy_n(vs.hs_user_sgpr.value, Sgprs, user_data.begin());
		std::copy_n(vs.gs_user_sgpr.value, Sgprs, user_data.begin() + Sgprs);
		std::copy_n(ps.ps_user_sgpr.value, Sgprs, user_data.begin() + 2 * Sgprs);
		CountUserDataChanges(context, with_args, user_data);
	}
	if (g_xframe_parts[2].Seen(with_stage)) Common::DebugCounters::Add(Counter::DrawXframeStage);
	if (g_xframe_parts[3].Seen(with_user_data)) {
		Common::DebugCounters::Add(Counter::DrawXframeUserData);
	}
	// The same state drawing the same mesh: user data may differ (records would relocate it).
	const bool skip = args_seen && Config::XframeSkipRepeatsEnabled();
	if (skip) Common::DebugCounters::Add(Counter::DrawXframeSkipped);
	return skip;
}

// Compares the T#, S# and V# words of each stage with those of the last draw with the same
// argument key: whether the changed user data changed which resources the draw binds.
void CountCrossFrameBindings(std::span<const PreparedBindings* const, 4> slots) {
	using Common::DebugCounters::Counter;
	using ShaderRecompiler::IR::DescriptorValue;
	if (!std::exchange(g_xframe_draw_keyed, false)) {
		return;
	}
	RepeatHash            textures;
	RepeatHash            buffers;
	RepeatHash            no_base;
	RepeatHash            shape;
	std::vector<uint64_t> bases;
	for (const auto* stage: slots) {
		if (stage == nullptr) {
			textures.Add(uint64_t {0});
			buffers.Add(uint64_t {0});
			no_base.Add(uint64_t {0});
			shape.Add(uint64_t {0});
			continue;
		}
		const auto& resources = *stage->runtime->resources;
		textures.Add(stage->runtime->module);
		textures.AddRange(std::span<const DescriptorValue> {resources.images});
		textures.AddRange(std::span<const DescriptorValue> {resources.samplers});
		buffers.AddRange(std::span<const DescriptorValue> {resources.buffers});
		no_base.Add(resources.buffers.size());
		shape.Add(resources.buffers.size());
		for (auto vsharp: resources.buffers) {
			// V#: base[31:0] in dword 0, base[47:32] in the low half of dword 1, size in dword 2.
			bases.push_back((uint64_t {vsharp.dwords[1] & 0xFFFFu} << 32u) | vsharp.dwords[0]);
			vsharp.dwords[0] = 0;
			vsharp.dwords[1] &= 0xFFFF0000u;
			no_base.Add(vsharp);
			vsharp.dwords[2] = 0;
			shape.Add(vsharp);
		}
	}
	CrossFrameDescriptors current {textures.Value(), buffers.Value(), no_base.Value(),
	                               shape.Value(), std::move(bases)};
	if (g_xframe_descriptors.size() >= 1u << 20u) {
		g_xframe_descriptors.clear();
	}
	const auto [found, inserted] = g_xframe_descriptors.try_emplace(g_xframe_draw_key);
	if (inserted) {
		found->second = std::move(current);
		return;
	}
	const bool same_textures = found->second.textures == current.textures;
	const bool same_buffers  = found->second.buffers == current.buffers;
	Common::DebugCounters::Add(Counter::DrawXframeBindCompared);
	if (same_textures) Common::DebugCounters::Add(Counter::DrawXframeSameTextures);
	if (same_buffers) Common::DebugCounters::Add(Counter::DrawXframeSameBuffers);
	if (same_textures && same_buffers) Common::DebugCounters::Add(Counter::DrawXframeSameBoth);
	const bool same_no_base = found->second.no_base == current.no_base;
	if (same_no_base) Common::DebugCounters::Add(Counter::DrawXframeVsharpNoBase);
	if (found->second.shape == current.shape) {
		Common::DebugCounters::Add(Counter::DrawXframeVsharpShape);
	}
	if (same_textures && same_no_base) Common::DebugCounters::Add(Counter::DrawXframeRelocatable);
	const auto& previous = found->second.bases;
	if (previous.size() == current.bases.size() && previous != current.bases) {
		Common::DebugCounters::Add(Counter::DrawXframeVsharpMoved);
		std::optional<uint64_t> shift;
		bool                    one_shift = true;
		for (size_t i = 0; i < previous.size() && one_shift; i++) {
			if (previous[i] == current.bases[i]) continue;
			const auto delta = current.bases[i] - previous[i];
			one_shift        = !shift.has_value() || *shift == delta;
			shift            = delta;
		}
		if (one_shift) Common::DebugCounters::Add(Counter::DrawXframeVsharpShift);
	}
	found->second = std::move(current);
}

StageRepeatState HashStage(const PreparedBindings& prepared) {
	StageRepeatState state;
	state.module          = prepared.runtime->module;
	state.valid           = true;
	const auto& resources = *prepared.runtime->resources;
	{
		RepeatHash hash;
		hash.AddRange(std::span<const ShaderRecompiler::IR::DescriptorValue> {resources.buffers});
		hash.AddRange(std::span<const ShaderRecompiler::IR::DescriptorValue> {resources.images});
		hash.AddRange(std::span<const ShaderRecompiler::IR::DescriptorValue> {resources.samplers});
		hash.AddRange(std::span<const uint32_t> {resources.flattened_srt});
		hash.AddRange(std::span<const uint32_t> {resources.user_data});
		state.snapshot = hash.Value();
	}
	{
		RepeatHash hash;
		for (const auto& image: prepared.images) {
			hash.Add(image.image_id.index);
			hash.Add(static_cast<VkImageView>(image.image_view));
		}
		state.images = hash.Value();
	}
	{
		RepeatHash ranges;
		for (const auto& source: prepared.buffer_sources) {
			ranges.Add(source.address);
			ranges.Add(source.size);
		}
		state.ranges = ranges.Value();
		RepeatHash buffers;
		for (const auto& buffer: prepared.buffers) {
			buffers.Add(static_cast<VkBuffer>(buffer.buffer));
			buffers.Add(buffer.offset);
			buffers.Add(buffer.range);
		}
		state.buffers = buffers.Value();
	}
	{
		RepeatHash hash;
		hash.AddRange(std::span<const uint32_t> {prepared.shader_data});
		state.data = hash.Value();
	}
	{
		RepeatHash hash;
		for (const auto sampler: prepared.samplers) {
			hash.Add(static_cast<VkSampler>(sampler));
		}
		state.samplers = hash.Value();
	}
	return state;
}

void CountBindingRepeats(std::span<const PreparedBindings* const, 4> slots, const void* pipeline,
                         const DrawRenderState& draw) {
	namespace Counters = Common::DebugCounters;
	if (!Counters::Enabled()) {
		return;
	}
	CountCrossFrameBindings(slots);
	auto& last      = g_draw_repeats;
	bool  draw_same = true;
	for (size_t slot = 0; slot < slots.size(); slot++) {
		auto&      previous = last.stages[slot];
		const auto current = slots[slot] != nullptr ? HashStage(*slots[slot]) : StageRepeatState {};
		if (current.valid) {
			Counters::Add(Counters::Counter::StageBinds);
			const bool shader   = previous.valid && previous.module == current.module;
			const bool snapshot = shader && previous.snapshot == current.snapshot;
			const bool images   = shader && previous.images == current.images;
			const bool ranges   = shader && previous.ranges == current.ranges;
			const bool buffers  = shader && previous.buffers == current.buffers;
			const bool data     = shader && previous.data == current.data;
			const bool all = images && buffers && data && previous.samplers == current.samplers;
			if (shader) Counters::Add(Counters::Counter::StageSameShader);
			if (snapshot) Counters::Add(Counters::Counter::StageSameSnapshot);
			if (images) Counters::Add(Counters::Counter::StageSameImages);
			if (ranges) Counters::Add(Counters::Counter::StageSameRanges);
			if (buffers) Counters::Add(Counters::Counter::StageSameBuffers);
			if (data) Counters::Add(Counters::Counter::StageSameData);
			if (all) Counters::Add(Counters::Counter::StageSameAll);
			draw_same = draw_same && all;
		} else {
			draw_same = draw_same && !previous.valid;
		}
		previous = current;
	}
	RepeatHash targets;
	for (uint32_t i = 0; i < draw.color_count; i++) {
		targets.Add(draw.color_info[i].image_id.index);
	}
	targets.Add(draw.depth_info.image_id.index);
	const bool same_pipeline = last.pipeline == pipeline;
	const bool same_targets  = last.targets == targets.Value();
	if (same_pipeline) Counters::Add(Counters::Counter::DrawSamePipeline);
	if (same_targets) Counters::Add(Counters::Counter::DrawSameTargets);
	if (same_pipeline && same_targets && draw_same) {
		Counters::Add(Counters::Counter::DrawSameAll);
	}
	last.pipeline = pipeline;
	last.targets  = targets.Value();
}

} // namespace

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t                     index_offset,
                                                const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.has_dynamic_vertex_fetch) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = vs_input_info.stage.module->info;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count = 0;
static std::atomic<uint32_t> g_draw_input_log_count = 0;
static std::atomic<uint32_t> g_mrt_state_log_count  = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

static void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF(
	    "DrawFramebufferSkip[%u]: %s color=%s color_addr=0x%010" PRIx64 " color_size=0x%016" PRIx64
	    " color_image=%s depth_format=%s depth_image=%s depth_vaddr_num=%d target_mask=0x%08" PRIx32
	    " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	    log_id, draw_name, RenderColorTypeName(color), color.desc.info.data.address,
	    color.desc.info.data.size, color.image_id ? "yes" : "no",
	    vk::to_string(depth.desc.view_info.format).c_str(), depth.image_id ? "yes" : "no",
	    static_cast<int>(!depth.desc.info.data.Empty()) +
	        static_cast<int>(depth.desc.info.HasStencil()),
	    ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.module->info.info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op, bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend,
	    bc.color_comb_fcn, static_cast<int>(ps_resources.images.size()),
	    static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
					const auto& r  = vs_input_info.resources[ai];
					const auto& rd = vs_input_info.resources_dst[ai];
					if (rd.buffer_index != bi) {
						continue;
					}
					const auto offset = static_cast<uint32_t>(r.Base48() - b.addr);
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
			const auto& r  = vs_input_info.resources[ai];
			const auto& rd = vs_input_info.resources_dst[ai];
			if (rd.buffer_index != bi) {
				continue;
			}
			LOGF("DrawInputState[%u]: attr[%d] offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, static_cast<uint32_t>(r.Base48() - b.addr), rd.register_start,
			     rd.registers_num, rd.fetch_index, r.fields[0], r.fields[1], r.fields[2],
			     r.fields[3]);
		}
	}
}

static void UpdateGraphicsDynamicState(const CommandBuffer& buffer, DynamicState& dynamic_state,
                                       const ShaderVertexInputInfo& vs_input_info,
                                       const RenderDepthInfo& depth, const RenderState& rendering) {
	KYTY_PROFILER_FUNCTION();

	const auto&        ctx = buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	const auto&        outputs = vs_input_info.stage.module->info.info.outputs;
	const bool         indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	std::array<vk::Viewport, viewport_slots> viewports {};
	std::array<vk::Rect2D, viewport_slots>   scissors {};
	const uint32_t viewport_count = indexed_viewports ? viewport_slots : 1;
	for (uint32_t i = 0; i < viewport_count; i++) {
		const auto& guest    = vp.viewports[i];
		auto&       viewport = viewports[i];
		if (ctx.GetClipControl().clip_disable) {
			const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
			viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
			viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
		} else {
			viewport.x      = guest.xoffset - guest.xscale;
			viewport.y      = guest.yoffset - guest.yscale;
			viewport.width  = guest.xscale * 2.0f;
			viewport.height = guest.yscale * 2.0f;
		}
		viewport.minDepth =
		    guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
		viewport.maxDepth = guest.zscale + guest.zoffset;

		const auto final_scissor =
		    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
		auto& scissor  = scissors[i];
		scissor.offset = {final_scissor.left, final_scissor.top};
		scissor.extent = {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                  static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)};
		if (viewport.width == 0.0f) {
			// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
			viewport.width = 1.0f;
			scissor.extent = {0, 0};
		}
	}
	dynamic_state.SetViewports(std::span {viewports}.first(viewport_count));
	dynamic_state.SetScissors(std::span {scissors}.first(viewport_count));

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	dynamic_state.SetLineWidth(line_width);
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	dynamic_state.SetBlendConstants(blend_constants);
	dynamic_state.SetDepthTestEnabled(depth.depth_test_enable);
	dynamic_state.SetDepthWriteEnabled(depth.depth_write_enable);
	dynamic_state.SetDepthCompareOp(depth.depth_compare_op);
#if !defined(__APPLE__)
	dynamic_state.SetDepthBoundsTestEnabled(depth.depth_bounds_test_enable);
	if (depth.depth_bounds_test_enable) {
		dynamic_state.SetDepthBounds(ctx.GetDepthBoundsMin(), ctx.GetDepthBoundsMax());
	}
#endif

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	dynamic_state.SetDepthBiasEnabled(depth_bias_enable);
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		dynamic_state.SetDepthBias(constant_factor, poly_offset.clamp, slope_factor);
	}

	// Rect lists are drawn without culling.
	vk::CullModeFlags cull_mode {};
	if (!Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		if (mode.cull_front) {
			cull_mode |= vk::CullModeFlagBits::eFront;
		}
		if (mode.cull_back) {
			cull_mode |= vk::CullModeFlagBits::eBack;
		}
	}
	dynamic_state.SetCullMode(cull_mode);
	dynamic_state.SetFrontFace(mode.face ? vk::FrontFace::eClockwise
	                                     : vk::FrontFace::eCounterClockwise);

	dynamic_state.SetStencilTestEnabled(depth.stencil_test_enable);
	if (depth.stencil_test_enable) {
		const auto& front = depth.stencil_front;
		const auto& back  = depth.stencil_back;
		dynamic_state.SetStencilOps(
		    {front.failOp, front.passOp, front.depthFailOp, front.compareOp},
		    {back.failOp, back.passOp, back.depthFailOp, back.compareOp});
		dynamic_state.SetStencilCompareMasks(front.compareMask, back.compareMask);
		dynamic_state.SetStencilWriteMasks(front.writeMask, back.writeMask);
		dynamic_state.SetStencilReferences(front.reference, back.reference);
	}

#if defined(__APPLE__)
	// MoltenVK has no VK_EXT_color_write_enable; the pipeline is created without the
	// eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead.
#else
	vk::Bool32 enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t slot = 0; slot < rendering.num_color_attachments; slot++) {
		enable[slot] = rendering.color_attachments[slot].image_view != nullptr;
	}
	dynamic_state.SetColorWriteEnables(std::span {enable}.first(rendering.num_color_attachments));
#endif
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

struct DrawCallInfo {
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;

	[[nodiscard]] bool IsIndexed() const { return debug_op == CommandBufferDebugOp::DrawIndex; }
	[[nodiscard]] const char* Name() const { return IsIndexed() ? "DrawIndex" : "DrawIndexAuto"; }
};

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects  = {};
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto& owner = cache.m_slot_images[target.image_id];
		if ((!owner.registered && !owner.info.data.Empty()) || owner.binding.needs_rebind) {
			EXIT("color target changed after render-state discovery\n");
		}
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Recorder());
		const auto extent           = target.Extent();
		state.width                 = std::min(state.width, extent.width);
		state.height                = std::min(state.height, extent.height);
		state.num_layers            = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view       = image_view;
		attachment.image_layout     = layout;
	}
	if (depth.image_id) {
		const auto& owner = cache.m_slot_images[depth.image_id];
		if (!owner.registered || owner.binding.needs_rebind) {
			EXIT("depth target changed after render-state discovery\n");
		}
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		const bool  meta_clear =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
		depth.load_clear_aspects = {};
		if (depth.depth_clear_enable || meta_clear)
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eDepth;
		// Apply pending clears to compressed stencil as well as depth.
		if (depth.stencil_clear_enable || (meta_clear && metadata.stencil_compressed))
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eStencil;
		if (meta_clear &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		const auto           draw_writes = depth.AttachmentWriteAspects();
		vk::ImageAspectFlags sampled_aspects;
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture)
					continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |=
				    DepthFeedbackAspects(draw_writes, depth.desc.view_info, native->info);
			}
		}
		if (feedback_aspects && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("depth attachment feedback loop is not supported by the host\n");
			}
		}
		auto layout = depth_attachment_layout(depth);
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		const auto access               = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                                  vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Recorder());
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.has_stencil = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eStencil);
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

bool RenderExecutor::ConsumeMetadataColorOperation(CommandBuffer& buffer,
                                                   uint32_t       render_target_slice_offset) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	if (mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear)) {
		// FCE can precede any ordinary draw. Discover the target to materialize its native
		// clear metadata before the expanded image is sampled.
		RenderColorInfo target {};
		ResolveRenderColorTarget(buffer, target, render_target_slice_offset, 0, true);
		return true;
	}
	// Metadata/decompression shaders must not export colors into the expanded image.
	return mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

// vertex_offset is the first vertex of non-indexed draws.
struct DrawEmitInfo {
	int32_t          vertex_offset  = 0;
	uint32_t         first_instance = 0;
	DrawIndirectArgs indirect;
};

struct DrawIndexBufferSource {
	uint64_t      address            = 0;
	const void*   host_data          = nullptr;
	uint64_t      size               = 0;
	vk::IndexType type               = vk::IndexType::eUint16;
	uint32_t      guest_element_size = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

static uint64_t VertexBufferDescriptorSize(int binding, const ShaderVertexInputInfo& info) {
	const auto& buffer = info.buffers[binding];
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < info.resources_num; i++) {
		if (info.resources_dst[i].buffer_index != binding) {
			continue;
		}
		const auto& resource = info.resources[i];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent =
		    resource.OutOfBounds() == 2
		        ? resource.Base48() - buffer.addr +
		              ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		        : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

struct VertexBufferRange {
	uint64_t                     base_address  = 0;
	uint64_t                     requested_end = 0;
	uint64_t                     acquired_end  = 0;
	std::pair<Buffer*, uint64_t> binding;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }
};

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers {};
	std::array<vk::DeviceSize, MaxBuffers> offsets {};
	std::array<vk::DeviceSize, MaxBuffers> sizes {};
	uint32_t                               count = 0;
};

// A draw's guest vertex ranges, merged and clamped without the buffer cache.
struct VertexBufferPlan {
	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes {};
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges {};
	uint32_t                                                      count = 0;
};

static VertexBufferPlan DecodeVertexBuffers(const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	// Collect the non-empty guest vertex ranges.
	VertexBufferPlan                                              plan;
	auto&                                                         sizes = plan.sizes;
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges {};
	uint32_t                                                      range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(i, vs_input_info);
		sizes[i]           = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vertex.addr, size);
		}
		ranges[range_count++] = {vertex.addr, vertex.addr + size};
	}

	std::sort(ranges.begin(), ranges.begin() + range_count,
	          [](const VertexBufferRange& left, const VertexBufferRange& right) {
		          return left.base_address < right.base_address;
	          });

	// Merge overlapping or touching ranges before acquiring host buffers.
	auto&     merged_ranges = plan.ranges;
	uint32_t& merged_count  = plan.count;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = ranges[i];
		if (merged_count != 0 &&
		    merged_ranges[merged_count - 1].requested_end >= range.base_address) {
			merged_ranges[merged_count - 1].requested_end =
			    std::max(merged_ranges[merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		merged_ranges[merged_count++] = {range.base_address, range.requested_end};
	}

	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		// PPSA20298
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		range.acquired_end = range.base_address + size;
	}
	return plan;
}

static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&               buffer,
                                                  const ShaderVertexInputInfo& vs_input_info,
                                                  VertexBufferPlan&            plan) {
	const auto& sizes         = plan.sizes;
	auto&       merged_ranges = plan.ranges;
	const auto  merged_count  = plan.count;
	auto&       cache         = buffer.GetContext().GetBufferCache();
	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		range.binding =
		    cache.ObtainBuffer(range.base_address, range.acquired_end - range.base_address, false);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto range = std::find_if(merged_ranges.begin(), merged_ranges.begin() + merged_count,
		                                [&](const VertexBufferRange& value) {
			                                return vertex.addr >= value.base_address &&
			                                       vertex.addr < value.acquired_end;
		                                });
		if (range == merged_ranges.begin() + merged_count) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->binding.first->Handle();
		prepared.offsets[i] = range->binding.second + vertex.addr - range->base_address;
		prepared.sizes[i]   = std::min(size, range->acquired_end - vertex.addr);
	}

	return prepared;
}

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              uint32_t phase) {
	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count, 0,
	                    draw.instance_count, draw.first_instance);
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kPolygon:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

static bool ResolvePrimitiveRestart(const CommandBuffer&         buffer,
                                    const DrawIndexBufferSource& source) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}

	const auto element_size = source.guest_element_size;
	const auto index_mask   = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index  = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	const auto restart_index = reset_index & index_mask;
	if (restart_index == index_mask) {
		// Use native restart; the 8-bit path widens its marker to 0xffff.
		return true;
	}

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before preparing draw resources: readback can restart the command buffer.
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	namespace Counters = Common::DebugCounters;
	Counters::Add(Counters::Counter::RestartIndexScans);
	if (Counters::Enabled() &&
	    buffer.GetContext().GetBufferCache().HasGpuDirtyBytes(source.address, source.size)) {
		Counters::Add(Counters::Counter::RestartIndexScansGpuDirty);
	}
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

// Every draw times its phases; only indexed draws log them, as before.
static void MarkDrawPhase(const DrawCallInfo& draw, const char* phase) {
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), phase);
	} else {
		Common::DebugCounters::SetDrawPhase(phase);
	}
}

namespace {

// --draw-records-verify: what a record of the previous frame's draw with the same programs and
// arguments would prepare, checked against this draw's own preparation.
struct DrawRecordStage {
	const void*                                        module = nullptr;
	std::vector<ShaderRecompiler::IR::DescriptorValue> textures; // T# words, then S# words.
	std::vector<ShaderRecompiler::IR::DescriptorValue> buffers;  // V# words without the base.
	std::vector<uint32_t>                              image_ids;
	std::vector<vk::ImageView>                         image_views;
	std::vector<vk::Sampler>                           samplers;
};

struct DrawRecord {
	// Context and stage registers without user SGPRs, UserConfig and the draw arguments.
	std::vector<uint8_t>                               state;
	std::array<DrawRecordStage, 4>                     stages;
	uint32_t                                           stage_count = 0;
	std::array<uint32_t, RENDER_COLOR_ATTACHMENTS_MAX> color_ids {};
	uint32_t                                           color_count = 0;
	uint32_t                                           depth_id    = 0;
	const void*                                        pipeline    = nullptr;
	std::pair<uint64_t, uint64_t>                      generations;
	// Where the context registers, UserConfig, stage registers and arguments end in state.
	std::array<size_t, 4> part_ends {};
	// Every image the draw binds, with its serial: ImageId slots are reused.
	std::vector<std::pair<uint32_t, uint64_t>> images;
	// Left out of state: it only reaches the dynamic state that CommitDraw sets on every draw.
	std::array<int, 4> scissor {};
	// The color clear stamps of its DCC/CMASK images; unstamped when one had none.
	std::vector<ColorClearStamp> color_stamps;
	bool                         color_unstamped = false;
};

// The two latest records of a key, for draws that alternate between two states (ping-pong
// targets).
struct DrawRecordPair {
	std::array<DrawRecord, 2> slots;
	uint32_t                  newest = 0;
	uint32_t                  filled = 0;
};

std::unordered_map<uint64_t, DrawRecordPair> g_draw_records;
std::pair<uint64_t, uint64_t>                g_draw_record_generations;
// Regs misses by the part and the 4-byte word that differs first in it: part << 16 | word.
std::unordered_map<uint32_t, uint64_t> g_draw_record_reg_words;
uint64_t                               g_draw_record_reg_misses = 0;

void CountRegisterDifference(const DrawRecord& record, const DrawRecord& current) {
	using Common::DebugCounters::Counter;
	constexpr std::array<Counter, 4> Counters {
	    Counter::DrawRecordRegsContext, Counter::DrawRecordRegsUcfg, Counter::DrawRecordRegsStage,
	    Counter::DrawRecordRegsArgs};
	constexpr std::array<const char*, 4> Names {"context", "ucfg", "stage", "args"};
	size_t                               begin = 0;
	for (uint32_t part = 0; part < 4; part++) {
		const auto end    = current.part_ends[part];
		const auto [a, b] = std::mismatch(record.state.begin() + begin, record.state.begin() + end,
		                                  current.state.begin() + begin);
		if (a != record.state.begin() + end) {
			Common::DebugCounters::Add(Counters[part]);
			const auto word = static_cast<uint32_t>((a - record.state.begin() - begin) / 4);
			g_draw_record_reg_words[(part << 16u) | word]++;
		}
		begin = end;
	}
	if (++g_draw_record_reg_misses % 50000 != 0) {
		return;
	}
	std::vector<std::pair<uint32_t, uint64_t>> words(g_draw_record_reg_words.begin(),
	                                                 g_draw_record_reg_words.end());
	std::ranges::sort(words, std::ranges::greater {}, &std::pair<uint32_t, uint64_t>::second);
	std::string text = "Draw records: first differing register words of regs misses:";
	for (size_t i = 0; i < std::min<size_t>(words.size(), 8); i++) {
		text += fmt::format(" {} +{} x{};", Names[words[i].first >> 16u],
		                    (words[i].first & 0xFFFFu) * 4u, words[i].second);
	}
	Log::WriteToConsoleAndLog(text + "\n");
	g_draw_record_reg_words.clear();
}

// Whether every image of the record is the one it recorded and still registered.
bool DrawRecordImagesKept(const TextureCache& textures, const DrawRecord& record) {
	return std::ranges::all_of(record.images, [&](const auto& image) {
		return textures.RegisteredSerial(ImageId {image.first}) == image.second;
	});
}

template <typename T>
void AppendBytes(std::vector<uint8_t>& out, const T& value) {
	static_assert(std::is_trivially_copyable_v<T>);
	const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
	out.insert(out.end(), bytes, bytes + sizeof(T));
}

DrawRecordStage RecordStage(const PreparedBindings& prepared) {
	const auto&     resources = *prepared.runtime->resources;
	DrawRecordStage stage;
	stage.module   = prepared.runtime->module;
	stage.textures = resources.images;
	stage.textures.insert(stage.textures.end(), resources.samplers.begin(),
	                      resources.samplers.end());
	stage.buffers = resources.buffers;
	for (auto& vsharp: stage.buffers) {
		// V#: base[31:0] in dword 0, base[47:32] in the low half of dword 1.
		vsharp.dwords[0] = 0;
		vsharp.dwords[1] &= 0xFFFF0000u;
	}
	for (const auto& image: prepared.images) {
		stage.image_ids.push_back(image.image_id.index);
		stage.image_views.push_back(image.image_view);
	}
	stage.samplers = prepared.samplers;
	return stage;
}

void VerifyDrawRecord(TextureCache& textures, const CommandBuffer& buffer,
                      std::span<const uint64_t> args, bool excluded,
                      std::span<PreparedBindings* const> stages,
                      std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
                      const void* pipeline) {
	using Common::DebugCounters::Add;
	using Common::DebugCounters::Counter;
	Common::DebugCounters::SetDrawPhase("(verify) DrawRecords");
	Add(Counter::DrawRecordChecks);
	const auto generations = textures.Generations();
	if (generations != g_draw_record_generations) {
		g_draw_record_generations = generations;
		Add(Counter::DrawRecordCacheChanges);
	}
	if (excluded || stages.size() > 4) {
		Add(Counter::DrawRecordExcluded);
		return;
	}
	const auto&                   vs = buffer.GetShaders().GetVs();
	const auto&                   ps = buffer.GetShaders().GetPs();
	const std::array<uint64_t, 5> programs {vs.es_regs.data_addr, vs.ls_regs.data_addr,
	                                        vs.hs_regs.data_addr, vs.gs_regs.data_addr,
	                                        ps.ps_regs.data_addr};
	const auto key = XXH3_64bits_withSeed(args.data(), args.size_bytes(),
	                                      XXH3_64bits(programs.data(), sizeof(programs)));

	DrawRecord current;
	current.state.reserve(4096);
	AppendBytes(current.state, buffer.GetRegisters());
	{
		const auto& registers = buffer.GetRegisters();
		const auto& viewport  = registers.GetScreenViewport();
		const auto* base      = reinterpret_cast<const uint8_t*>(&registers);
		const auto  first = reinterpret_cast<const uint8_t*>(&viewport.screen_scissor_left) - base;
		const auto  last =
		    reinterpret_cast<const uint8_t*>(&viewport.screen_scissor_bottom + 1) - base;
		std::fill(current.state.begin() + first, current.state.begin() + last, 0);
		current.scissor = {viewport.screen_scissor_left, viewport.screen_scissor_top,
		                   viewport.screen_scissor_right, viewport.screen_scissor_bottom};
	}
	current.part_ends[0] = current.state.size();
	AppendBytes(current.state, buffer.GetUserConfig());
	current.part_ends[1] = current.state.size();
	AppendBytes(current.state, vs.es_regs);
	AppendBytes(current.state, vs.ls_regs);
	AppendBytes(current.state, vs.hs_regs);
	AppendBytes(current.state, vs.gs_regs);
	AppendBytes(current.state, ps.ps_regs);
	current.part_ends[2]  = current.state.size();
	const auto* arg_bytes = reinterpret_cast<const uint8_t*>(args.data());
	current.state.insert(current.state.end(), arg_bytes, arg_bytes + args.size_bytes());
	current.part_ends[3] = current.state.size();
	for (const auto* stage: stages) {
		current.stages[current.stage_count++] = RecordStage(*stage);
	}
	for (const auto& color: colors) {
		current.color_ids[current.color_count++] = color.image_id.index;
	}
	current.depth_id     = depth.image_id.index;
	current.pipeline     = pipeline;
	current.generations  = generations;
	const auto add_image = [&](uint32_t index) {
		if (index != ImageId::INVALID_INDEX) {
			current.images.emplace_back(index, textures.RegisteredSerial(ImageId {index}));
		}
	};
	for (uint32_t i = 0; i < current.stage_count; i++) {
		std::ranges::for_each(current.stages[i].image_ids, add_image);
	}
	std::ranges::for_each(std::span {current.color_ids}.first(current.color_count), add_image);
	add_image(current.depth_id);
	for (const auto& [index, serial]: current.images) {
		if (!textures.CopyColorClearStamps(ImageId {index}, current.color_stamps)) {
			current.color_unstamped = true;
		}
	}

	if (g_draw_records.size() >= 1u << 16u) {
		g_draw_records.clear();
	}
	auto&      pair     = g_draw_records[key];
	const bool inserted = pair.filled == 0;
	uint32_t   slot     = pair.newest;
	for (const uint32_t candidate: {pair.newest, 1u - pair.newest}) {
		if (candidate < pair.filled && pair.slots[candidate].state == current.state) {
			slot = candidate;
			break;
		}
	}
	auto&      record  = pair.slots[slot];
	const bool matched = !inserted && record.state == current.state;
	const auto staged  = std::span {record.stages.data(), record.stage_count};
	const auto now     = std::span {current.stages.data(), current.stage_count};
	const auto same    = [&](auto field) {
		return staged.size() == now.size() && std::ranges::equal(staged, now, {}, field, field);
	};
	if (inserted) {
		Add(Counter::DrawRecordNew);
	} else if (record.state != current.state) {
		Add(Counter::DrawRecordRegs);
		CountRegisterDifference(record, current);
	} else if (!same(&DrawRecordStage::textures)) {
		Add(Counter::DrawRecordTextures);
	} else if (!same(&DrawRecordStage::buffers)) {
		Add(Counter::DrawRecordBuffers);
	} else if (record.generations != current.generations &&
	           !DrawRecordImagesKept(textures, record)) {
		Add(Counter::DrawRecordImages);
	} else {
		const bool kept = record.generations != current.generations;
		Add(Counter::DrawRecordHits);
		if (kept) Add(Counter::DrawRecordHitsKept);
		if (slot != pair.newest) Add(Counter::DrawRecordHitsOlder);
		if (record.scissor != current.scissor) Add(Counter::DrawRecordHitsScissor);
		if (record.color_unstamped) {
			Add(Counter::DrawRecordMetaUnstamped);
		} else if (!textures.ColorClearStampsHold(record.color_stamps)) {
			Add(Counter::DrawRecordMetaChanged);
		} else {
			Add(Counter::DrawRecordMetaStable);
		}
		const bool programs_ok = same(&DrawRecordStage::module);
		const bool textures_ok = same(&DrawRecordStage::image_ids) &&
		                         same(&DrawRecordStage::image_views) &&
		                         same(&DrawRecordStage::samplers);
		const bool targets_ok  = record.color_count == current.color_count &&
		                         record.color_ids == current.color_ids &&
		                         record.depth_id == current.depth_id;
		const bool pipeline_ok = record.pipeline == current.pipeline;
		if (!programs_ok) Add(Counter::DrawRecordWrongPrograms);
		if (!textures_ok) Add(Counter::DrawRecordWrongTextures);
		if (!targets_ok) Add(Counter::DrawRecordWrongTargets);
		if (!pipeline_ok) Add(Counter::DrawRecordWrongPipeline);
		if (!(programs_ok && textures_ok && targets_ok && pipeline_ok)) {
			Add(Counter::DrawRecordWrong);
			if (kept) Add(Counter::DrawRecordWrongKept);
		}
	}
	// A matching record is renewed; otherwise the draw replaces the older record.
	const uint32_t target = matched ? slot : (pair.filled < 2 ? pair.filled : 1u - pair.newest);
	pair.slots[target]    = std::move(current);
	pair.filled           = std::max(pair.filled, target + 1);
	pair.newest           = target;
}

} // namespace

namespace {

// --draw-records: what the previous frame's preparation of the same draw left behind. A draw
// repeats it when its registers, programs and T#/S# words are the same, its V# words differ at
// most in the base address and every image of the record is still the same registered image.
struct DrawReplayStage {
	const void*                                        module = nullptr;
	std::vector<ShaderRecompiler::IR::DescriptorValue> textures; // T# words, then S# words.
	std::vector<ShaderRecompiler::IR::DescriptorValue> buffers;  // V# words without the base.
	std::vector<TextureBinding>                        images;
	std::vector<vk::Sampler>                           samplers;
};

struct DrawReplay {
	// As DrawRecord::state: the screen scissor is left out.
	std::vector<uint8_t>                                      state;
	std::array<DrawReplayStage, 4>                            stages;
	uint32_t                                                  stage_count = 0;
	std::array<RenderColorInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	// The base layer each color target lookup started from, for its clear materialization.
	std::array<uint32_t, RENDER_COLOR_ATTACHMENTS_MAX> color_layers {};
	uint32_t                                           color_count = 0;
	std::vector<std::pair<ImageId, uint64_t>>          images;
	// The GC tick in which its images were last touched.
	mutable uint64_t touched_tick = 0;
};

struct DrawReplayPair {
	std::array<DrawReplay, 2> slots;
	uint32_t                  newest = 0;
	uint32_t                  filled = 0;
};

// The draw being prepared.
struct DrawReplayDraw {
	bool              active     = false;
	bool              capturable = false;
	DrawReplayPair*   pair       = nullptr;
	int32_t           matched    = -1;
	const DrawReplay* hit        = nullptr;
	// This draw's state and, for a store, what its preparation produced.
	DrawReplay current;
	// What the record's state is built from when the draw stores one.
	const CommandBuffer*    buffer = nullptr;
	std::array<uint64_t, 8> args {};
	size_t                  arg_count = 0;
};

std::unordered_map<uint64_t, DrawReplayPair> g_draw_replays;
DrawReplayDraw                               g_draw_replay;

void MaskVsharps(std::vector<ShaderRecompiler::IR::DescriptorValue>& buffers) {
	for (auto& vsharp: buffers) {
		// V#: base[31:0] in dword 0, base[47:32] in the low half of dword 1.
		vsharp.dwords[0] = 0;
		vsharp.dwords[1] &= 0xFFFF0000u;
	}
}

bool SameVsharpShapes(const std::vector<ShaderRecompiler::IR::DescriptorValue>& current,
                      const std::vector<ShaderRecompiler::IR::DescriptorValue>& shapes) {
	return current.size() == shapes.size() &&
	       std::ranges::equal(current, shapes, [](auto vsharp, const auto& shape) {
		       vsharp.dwords[0] = 0;
		       vsharp.dwords[1] &= 0xFFFF0000u;
		       return vsharp == shape;
	       });
}

bool SameTextureWords(const ShaderRecompiler::IR::ResourceSnapshot&             resources,
                      const std::vector<ShaderRecompiler::IR::DescriptorValue>& textures) {
	const auto images = resources.images.size();
	return textures.size() == images + resources.samplers.size() &&
	       std::equal(resources.images.begin(), resources.images.end(), textures.begin()) &&
	       std::equal(resources.samplers.begin(), resources.samplers.end(),
	                  textures.begin() + static_cast<std::ptrdiff_t>(images));
}

// The draw's registers, programs and arguments as a record keeps them, without the screen
// scissor: it only reaches the dynamic state that CommitDraw sets on every draw.
void BuildDrawReplayState(const CommandBuffer& buffer, std::span<const uint64_t> args,
                          std::vector<uint8_t>& state) {
	state.clear();
	const auto& registers = buffer.GetRegisters();
	AppendBytes(state, registers);
	{
		const auto& viewport = registers.GetScreenViewport();
		const auto* base     = reinterpret_cast<const uint8_t*>(&registers);
		const auto  first = reinterpret_cast<const uint8_t*>(&viewport.screen_scissor_left) - base;
		const auto  last =
		    reinterpret_cast<const uint8_t*>(&viewport.screen_scissor_bottom + 1) - base;
		std::fill(state.begin() + first, state.begin() + last, 0);
	}
	const auto& vs = buffer.GetShaders().GetVs();
	const auto& ps = buffer.GetShaders().GetPs();
	AppendBytes(state, buffer.GetUserConfig());
	AppendBytes(state, vs.es_regs);
	AppendBytes(state, vs.ls_regs);
	AppendBytes(state, vs.hs_regs);
	AppendBytes(state, vs.gs_regs);
	AppendBytes(state, ps.ps_regs);
	const auto* arg_bytes = reinterpret_cast<const uint8_t*>(args.data());
	state.insert(state.end(), arg_bytes, arg_bytes + args.size_bytes());
}

// Whether a record's state is the draw's, compared in place instead of building it.
bool SameDrawReplayState(const std::vector<uint8_t>& state, const CommandBuffer& buffer,
                         std::span<const uint64_t> args) {
	const auto& registers = buffer.GetRegisters();
	const auto& ucfg      = buffer.GetUserConfig();
	const auto& vs        = buffer.GetShaders().GetVs();
	const auto& ps        = buffer.GetShaders().GetPs();
	const auto  total = sizeof(registers) + sizeof(ucfg) + sizeof(vs.es_regs) + sizeof(vs.ls_regs) +
	                    sizeof(vs.hs_regs) + sizeof(vs.gs_regs) + sizeof(ps.ps_regs) +
	                    args.size_bytes();
	if (state.size() != total) {
		return false;
	}
	size_t offset = 0;

	// Compares the next bytes of the state.
	const auto same = [&](const void* data, size_t size) {
		const bool equal = std::memcmp(state.data() + offset, data, size) == 0;
		offset += size;
		return equal;
	};
	const auto& viewport = registers.GetScreenViewport();
	const auto* base     = reinterpret_cast<const uint8_t*>(&registers);
	const auto  first =
	    static_cast<size_t>(reinterpret_cast<const uint8_t*>(&viewport.screen_scissor_left) - base);
	const auto last = static_cast<size_t>(
	    reinterpret_cast<const uint8_t*>(&viewport.screen_scissor_bottom + 1) - base);
	if (!same(base, first)) {
		return false;
	}
	offset = last;
	return same(base + last, sizeof(registers) - last) && same(&ucfg, sizeof(ucfg)) &&
	       same(&vs.es_regs, sizeof(vs.es_regs)) && same(&vs.ls_regs, sizeof(vs.ls_regs)) &&
	       same(&vs.hs_regs, sizeof(vs.hs_regs)) && same(&vs.gs_regs, sizeof(vs.gs_regs)) &&
	       same(&ps.ps_regs, sizeof(ps.ps_regs)) && same(args.data(), args.size_bytes());
}

// Looks the draw up by programs and arguments; returns whether a record has the same registers,
// so that their validation can be skipped.
bool BeginDrawReplay(const CommandBuffer& buffer, std::span<const uint64_t> args) {
	using Common::DebugCounters::Add;
	using Common::DebugCounters::Counter;
	auto& draw   = g_draw_replay;
	draw.active  = false;
	draw.hit     = nullptr;
	draw.matched = -1;
	if (!Config::DrawRecordsEnabled() || args.size() > draw.args.size()) {
		return false;
	}
	Common::DebugCounters::SetDrawPhase("(records) Key");
	const auto&                   vs = buffer.GetShaders().GetVs();
	const auto&                   ps = buffer.GetShaders().GetPs();
	const std::array<uint64_t, 5> programs {vs.es_regs.data_addr, vs.ls_regs.data_addr,
	                                        vs.hs_regs.data_addr, vs.gs_regs.data_addr,
	                                        ps.ps_regs.data_addr};

	const auto seed = XXH3_64bits(programs.data(), sizeof(programs));
	const auto key  = XXH3_64bits_withSeed(args.data(), args.size_bytes(), seed);
	if (g_draw_replays.size() >= 1u << 16u) {
		g_draw_replays.clear();
	}
	draw.pair = &g_draw_replays[key];
	for (const uint32_t candidate: {draw.pair->newest, 1u - draw.pair->newest}) {
		if (candidate < draw.pair->filled &&
		    SameDrawReplayState(draw.pair->slots[candidate].state, buffer, args)) {
			draw.matched = static_cast<int32_t>(candidate);
			break;
		}
	}
	draw.buffer    = &buffer;
	draw.arg_count = args.size();
	std::ranges::copy(args, draw.args.begin());
	draw.active              = true;
	draw.capturable          = true;
	draw.current.stage_count = 0;
	draw.current.color_count = 0;
	Add(Counter::DrawReplayChecks);
	if (draw.matched < 0) {
		Add(Counter::DrawReplayMissRegs);
	}
	return draw.matched >= 0;
}

// The record the draw repeats, once its programs and resources are known.
const DrawReplay* FindDrawReplay(TextureCache& textures, const DrawRenderState& state) {
	using Common::DebugCounters::Add;
	using Common::DebugCounters::Counter;
	auto& draw = g_draw_replay;
	if (!draw.active || draw.matched < 0) {
		return nullptr;
	}
	Common::DebugCounters::SetDrawPhase("(records) Check");
	const auto&                              record = draw.pair->slots[draw.matched];
	std::array<const ShaderStageRuntime*, 4> runtimes {};
	uint32_t                                 count = 0;
	for (const auto& stage: state.VertexStages()) {
		runtimes[count++] = &stage.stage;
	}
	if (state.ps_active) {
		runtimes[count++] = &state.ps_input_info.stage;
	}
	bool same = count == record.stage_count;
	for (uint32_t i = 0; i < count && same; i++) {
		const auto& stage = record.stages[i];

		same = runtimes[i]->module == stage.module &&
		       SameTextureWords(*runtimes[i]->resources, stage.textures) &&
		       SameVsharpShapes(runtimes[i]->resources->buffers, stage.buffers);
	}
	if (!same) {
		Add(Counter::DrawReplayMissResources);
		return nullptr;
	}
	if (!std::ranges::all_of(record.images, [&](const auto& image) {
		    return textures.RegisteredSerial(image.first) == image.second;
	    })) {
		Add(Counter::DrawReplayMissImages);
		return nullptr;
	}
	Add(Counter::DrawReplayHits);
	// Touches repeat nothing within a GC tick.
	if (record.touched_tick != textures.GcTick()) {
		for (const auto& image: record.images) {
			textures.RepeatTouch(image.first);
		}
		record.touched_tick = textures.GcTick();
	}
	draw.hit = &record;
	return draw.hit;
}

// Keeps what a draw without a record found for its stage.
void CaptureDrawReplayStage(const PreparedBindings& prepared) {
	auto& draw = g_draw_replay;
	if (!draw.active || draw.hit != nullptr || draw.current.stage_count >= 4) {
		draw.capturable = false;
		return;
	}
	const auto& resources = *prepared.runtime->resources;
	auto&       stage     = draw.current.stages[draw.current.stage_count++];
	stage.module          = prepared.runtime->module;
	stage.textures        = resources.images;
	stage.textures.insert(stage.textures.end(), resources.samplers.begin(),
	                      resources.samplers.end());
	stage.buffers = resources.buffers;
	MaskVsharps(stage.buffers);
	const auto& program = prepared.runtime->module->info.info;
	for (uint32_t i = 0; i < prepared.images.size(); i++) {
		// A lookup that found nothing can create the image: the next draw must look again.
		if (!program.images[i].heap && !prepared.decoded_images[i].null &&
		    !prepared.images[i].found_id) {
			draw.capturable = false;
		}
	}
	stage.images = prepared.images;
	for (auto& image: stage.images) {
		image.mip_views.clear();
		image.image_view = nullptr;
	}
	stage.samplers = prepared.samplers;
}

// Ends the draw: a draw without a record stores what it prepared.
void FinishDrawReplay(const TextureCache& textures, bool excluded) {
	using Common::DebugCounters::Add;
	using Common::DebugCounters::Counter;
	auto& draw = g_draw_replay;
	if (!draw.active) {
		return;
	}
	draw.active = false;
	auto& pair  = *draw.pair;
	if (draw.hit != nullptr) {
		pair.newest = static_cast<uint32_t>(draw.matched);
		return;
	}
	if (excluded || !draw.capturable) {
		return;
	}
	auto& images = draw.current.images;
	images.clear();
	const auto add = [&](ImageId id) {
		if (id) {
			images.emplace_back(id, textures.RegisteredSerial(id));
		}
	};
	for (uint32_t i = 0; i < draw.current.stage_count; i++) {
		for (const auto& image: draw.current.stages[i].images) {
			add(image.found_id);
			add(image.image_id);
		}
	}
	for (uint32_t i = 0; i < draw.current.color_count; i++) {
		add(draw.current.colors[i].image_id);
	}
	if (std::ranges::any_of(images, [](const auto& image) { return image.second == 0; })) {
		return;
	}
	const uint32_t target = draw.matched >= 0 ? static_cast<uint32_t>(draw.matched)
	                        : pair.filled < 2 ? pair.filled
	                                          : 1u - pair.newest;
	BuildDrawReplayState(*draw.buffer, std::span {draw.args}.first(draw.arg_count),
	                     draw.current.state);
	draw.current.touched_tick = textures.GcTick();
	std::swap(pair.slots[target], draw.current);
	pair.filled = std::max(pair.filled, target + 1);
	pair.newest = target;
	Add(Counter::DrawReplayStores);
}

} // namespace

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t         render_target_slice_offset,
                                            DrawRenderState& state) {
	m_context.GetImageHeap().BeginDraw();
	MarkDrawPhase(draw, "GetGraphicsPrograms");
	if (m_commit_job == nullptr) {
		state.resources = &m_draw_resources;
		m_context.GetPipelineCache().GetGraphicsPrograms(buffer, state);
	} else if (Config::VerifyDrawPrepEnabled()) {
		VerifyPreparedPrograms(buffer, state);
	} else if (!m_commit_job->heap_peeks.empty() && !AcquirePeekedHeaps(*m_commit_job)) {
		// The GPU thread's own programs replace the worker's where they differ.
		Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawPrepHeapRedo);
		VerifyPreparedPrograms(buffer, state);
	}
	// Vertex programs beyond the host mesh limits or with unsupported call targets are not
	// compiled.
	if (state.vertex_info[0].stage.module == nullptr) {
		return false;
	}
	if (const auto* replay = FindDrawReplay(m_context.GetTextureCache(), state);
	    replay != nullptr) {
		MarkDrawPhase(draw, "(records) Targets");
		auto& textures = m_context.GetTextureCache();
		for (uint32_t i = 0; i < replay->color_count; i++) {
			auto& color = state.color_info[i];
			color       = replay->colors[i];
			textures.RepeatColorClear(color.image_id, color.desc, replay->color_layers[i]);
			BindRenderTarget(color.image_id);
		}
		state.color_count = replay->color_count;
	} else {
		MarkDrawPhase(draw, "ResolveRenderColorTarget");
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			if ((state.mrt_mask & (1u << slot)) != 0) {
				ResolveRenderColorTarget(buffer, state.color_info[state.color_count],
				                         render_target_slice_offset, slot);
				if (state.color_info[state.color_count].image_id) {
					state.color_count++;
				}
			}
		}
		auto& record = g_draw_replay;
		if (record.active) {
			for (uint32_t i = 0; i < state.color_count; i++) {
				const auto& color  = state.color_info[i];
				const auto& decode = m_color_target_decodes[color.target_slot];
				// The lookup started from the decoded description.
				record.capturable &= decode.valid;
				record.current.colors[i]       = color;
				record.current.color_layers[i] = decode.info.desc.view_info.base_layer;
			}
			record.current.color_count = state.color_count;
		}
	}
	MarkDrawPhase(draw, "ResolveRenderDepthTarget");
	ResolveRenderDepthTarget(buffer, state.depth_info);
	// TODO(PPSA09243): Vulkan passes take one sample count. Where PS5 attachments mix counts,
	// keep those matching the raster count instead of resolving multisampled shadows.
	const auto colors       = std::span {state.color_info, state.color_count};
	const auto pass_samples = state.depth_info.image_id ? state.depth_info.desc.info.samples
	                                                    : state.color_info[0].desc.info.samples;
	if (std::ranges::any_of(colors, [&](const RenderColorInfo& color) {
		    return color.desc.info.samples != pass_samples;
	    })) {
		const auto raster =
		    render_sample_count(buffer.GetRegisters().GetAaConfig().msaa_num_samples);
		const auto dropped = std::ranges::remove_if(colors, [&](const RenderColorInfo& color) {
			return color.desc.info.samples != raster;
		});
		state.color_count -= static_cast<uint32_t>(dropped.size());
		if (state.depth_info.image_id && state.depth_info.desc.info.samples != raster) {
			state.depth_info = {};
		}
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			Log::WriteToConsoleAndLog(
			    "Warning: dropping attachments whose sample count differs from the raster.\n");
		}
	}

	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active) {
		LogFramebufferSkip(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   draw.index_count, 0);
		return false;
	}

	return true;
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type = source.type;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
	}
	return prepared;
}

static void CommitVertexBuffers(CommandRecorder vk_buffer, const PreparedVertexBuffers& prepared) {
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF(prepared.buffers[i] == nullptr);
	}
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(CommandRecorder vk_buffer, const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
                                 const DrawRenderState& state, uint32_t index_type_and_size,
                                 const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.IsIndexed() && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vertex_info[0], index_type_and_size,
	                  draw.index_count, index_addr);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, CommandRecorder vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kPolygon:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.IsIndexed()) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count,
				               static_cast<uint32_t>(emit.vertex_offset), emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.IsIndexed()) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count,
					               i + static_cast<uint32_t>(emit.vertex_offset),
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

bool RenderExecutor::CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
                                      const DrawEmitInfo& emit, DrawRenderState& state) {
	if (!buffer.GetRegisters().GetRenderControl().copy_depth_to_color) {
		return false;
	}
	auto&       ucfg  = buffer.GetUserConfig();
	const auto& hw    = buffer.GetRegisters();
	auto&       depth = state.depth_info;
	auto&       color = state.color_info[0];
	EXIT_NOT_IMPLEMENTED(
	    !m_context.GetGraphics().maintenance8_enabled ||
	    !Prospero::IsRectList(ucfg.GetPrimType()) || draw.IsIndexed() || draw.index_count != 3 ||
	    draw.instance_count != 1 || emit.vertex_offset != 0 || emit.first_instance != 0 ||
	    state.color_count != 1 || color.target_slot != 0 || !depth.image_id ||
	    color.desc.info.samples != 1 || depth.desc.info.extent.width != color.Extent().width ||
	    depth.desc.info.extent.height != color.Extent().height ||
	    depth.desc.view_info.layer_count != 1 || color.desc.view_info.layer_count != 1 ||
	    hw.GetBlendControl(0).enable || state.ps_input_info.ps_pixel_kill_enable ||
	    state.ps_input_info.ps_depth_export_enable ||
	    state.ps_input_info.ps_sample_mask_export_enable);
	vk::ImageAspectFlags feedback;
	const auto           rendering = AcquireRenderTargets(buffer, &color, 1, depth, feedback);
	if (depth.load_clear_aspects) {
		m_context.GetCommandScheduler().BeginRendering(rendering);
	}
	auto& cache       = m_context.GetTextureCache();
	auto& source      = cache.GetImage(depth.image_id);
	auto& destination = cache.GetImage(color.image_id);
	EXIT_NOT_IMPLEMENTED((source.backing.format != vk::Format::eD32Sfloat &&
	                      source.backing.format != vk::Format::eD32SfloatS8Uint) ||
	                     destination.backing.format != vk::Format::eR32Sfloat);
	const auto rect = calc_final_scissor(hw.GetScreenViewport(), hw.GetScanModeControl(),
	                                     {rendering.width, rendering.height}, 0);
	if (rect.right > rect.left && rect.bottom > rect.top) {
		const vk::ImageCopy region {
		    {vk::ImageAspectFlagBits::eDepth, depth.desc.view_info.base_level,
		     depth.desc.view_info.base_layer, 1},
		    {rect.left, rect.top, 0},
		    {vk::ImageAspectFlagBits::eColor, color.desc.view_info.base_level,
		     color.desc.view_info.base_layer, 1},
		    {rect.left, rect.top, 0},
		    {static_cast<uint32_t>(rect.right - rect.left),
		     static_cast<uint32_t>(rect.bottom - rect.top), 1}};
		m_context.GetCommandScheduler().EndRendering();
		const auto command = buffer.Recorder();
		source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
		               command);
		destination.Transit(vk::ImageLayout::eTransferDstOptimal,
		                    vk::AccessFlagBits2::eTransferWrite, {}, command);
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                  destination.backing.image, vk::ImageLayout::eTransferDstOptimal, 1,
		                  &region);
	}
	return true;
}

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         const DrawEmitInfo&          emit,
                                         const DrawIndexBufferSource& index_source) {
	auto&      ucfg                     = buffer.GetUserConfig();
	const bool primitive_restart_enable = state.primitive_restart_enable;
	if (CopyDepthToColor(buffer, draw, emit, state)) {
		return;
	}
	const auto vertex_stages = state.VertexStages();
	const bool mesh_active   = state.vertex_info[0].stage.module->info.stage == ShaderType::Mesh;
	uint32_t   mesh_groups   = 0;
	if (mesh_active) {
		const auto& mesh = state.vertex_info[0].mesh;
		EXIT_NOT_IMPLEMENTED(mesh.fast_launch && (draw.IsIndexed() || primitive_restart_enable));
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed());
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed(),
			     primitive_restart_enable);
		}
		const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return;
		}
		mesh_groups        = (primitives - 1u) / mesh.primitives_per_group + 1u;
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		if (mesh_groups > limits.maxMeshWorkGroupCount[0] ||
		    draw.instance_count > limits.maxMeshWorkGroupCount[1] ||
		    static_cast<uint64_t>(mesh_groups) * draw.instance_count >
		        limits.maxMeshWorkGroupTotalCount) {
			EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", mesh_groups,
			     draw.instance_count);
		}
	}

	if (mesh_active && draw.IsIndexed()) {
		// Register the original guest indices for shader reads; PrepareGraphicsBindings
		// synchronizes registered BDA ranges before any draw commands are committed.
		(void)m_context.GetBufferCache().FindBuffer(index_source.address,
		                                            static_cast<uint64_t>(draw.index_count) *
		                                                index_source.guest_element_size);
	}
	LogDrawPhase(draw.Name(), "PrepareBindings");
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	// The worker decoded the stages of a draw it prepared.
	const auto decoded = [&](uint32_t stage) {
		return m_commit_job != nullptr && m_commit_job->decoded ? &m_commit_job->bindings[stage]
		                                                        : nullptr;
	};
	const auto* replay = g_draw_replay.hit;

	// A draw with a record takes its texture bindings and samplers from it.
	const auto prepare = [&](const ShaderStageRuntime& runtime, PreparedBindings& prepared,
	                         PreparedBindings* worker) {
		if (replay != nullptr) {
			const auto& stage        = replay->stages[stage_count];
			prepared.replay_images   = &stage.images;
			prepared.replay_samplers = &stage.samplers;
		}
		PrepareBindings(runtime, prepared, worker);
		prepared.replay_images   = nullptr;
		prepared.replay_samplers = nullptr;
		if (g_draw_replay.active) {
			CaptureDrawReplayStage(prepared);
		}
		descriptor_stages[stage_count++] = &prepared;
	};
	for (uint32_t i = 0; i < vertex_stages.size(); i++) {
		prepare(state.vertex_info[i].stage, bindings.vertex[i], decoded(i));
	}
	if (state.ps_active) {
		if (!bindings.pixel) bindings.pixel.emplace();
		prepare(state.ps_input_info.stage, *bindings.pixel, decoded(3));
	}
	const auto stages = std::span {descriptor_stages.data(), stage_count};
	{
		const Common::DebugCounters::ScopedDrawPhase phase("PrepareGraphicsBindings");
		PrepareGraphicsBindings(stages, std::span {state.color_info, state.color_count});
	}
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	if (!mesh_active) {
		LogDrawPhase(draw.Name(), "PrepareVertexBuffers");
		Common::DebugCounters::SetDrawPhase("PrepareVertexBuffers/Vertex");
		auto plan       = DecodeVertexBuffers(state.vertex_info[0]);
		vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0], plan);
		Common::DebugCounters::SetDrawPhase("PrepareVertexBuffers/Index");
		index_binding = PrepareIndexBuffer(buffer, index_source);
	}
	std::pair<Buffer*, uint64_t> args_buffer {};
	if (emit.indirect.address != 0) {
		Common::DebugCounters::SetDrawPhase("IndirectArgs");
		const auto size = static_cast<uint64_t>(emit.indirect.count - 1) * emit.indirect.stride +
		                  (draw.IsIndexed() ? sizeof(vk::DrawIndexedIndirectCommand)
		                                    : sizeof(vk::DrawIndirectCommand));
		args_buffer = m_context.GetBufferCache().ObtainBuffer(emit.indirect.address, size, false);
		EXIT_IF((args_buffer.second & 3u) != 0);
	}
	MarkDrawPhase(draw, "CreatePipeline");
	auto&                pipeline = m_context.GetPipelineCache().GetGraphicsPipeline(buffer, state);
	vk::ImageAspectFlags feedback_aspects;
	Common::DebugCounters::SetDrawPhase("AcquireRenderTargets");
	const auto rendering = AcquireRenderTargets(buffer, state.color_info, state.color_count,
	                                            state.depth_info, feedback_aspects, stages);
	if (Common::DebugCounters::Enabled()) {
		// Overlay-only statistics, kept out of the CommitDraw time.
		Common::DebugCounters::SetDrawPhase("(overlay) CountBindingRepeats");
		std::array<const PreparedBindings*, 4> slots {};
		for (uint32_t i = 0; i < vertex_stages.size() && i < 3; i++) {
			slots[i] = &bindings.vertex[i];
		}
		if (state.ps_active) {
			slots[3] = &*bindings.pixel;
		}
		CountBindingRepeats(slots, &pipeline, state);
	}
	if (Config::DrawRecordsVerifyEnabled()) {
		const std::array<uint64_t, 8> args {
		    draw.index_count,     draw.instance_count,
		    draw.first_instance,  static_cast<uint32_t>(emit.vertex_offset),
		    emit.first_instance,  emit.indirect.address,
		    index_source.address, static_cast<uint64_t>(draw.debug_op)};
		// Indexed mesh draws register their indices with the buffer cache.
		VerifyDrawRecord(m_context.GetTextureCache(), buffer, args, mesh_active && draw.IsIndexed(),
		                 stages, std::span {state.color_info, state.color_count}, state.depth_info,
		                 &pipeline);
	}
	if (g_draw_replay.active) {
		Common::DebugCounters::SetDrawPhase("(records) Store");
		// Indexed mesh draws register their indices with the buffer cache.
		FinishDrawReplay(m_context.GetTextureCache(), mesh_active && draw.IsIndexed());
	}
	Common::DebugCounters::SetDrawPhase("CommitDraw");

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory.
	auto vk_buffer = buffer.Recorder();
	SetDrawDebugPhase(buffer, submit_id, draw, draw.IsIndexed() ? 0x100u : 0x200u);
	if (!mesh_active) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	if (state.ps_active && !draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x300u);
	}
	CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
	if (!mesh_active) {
		CommitIndexBuffer(vk_buffer, index_binding);
	}
	using ShaderRecompiler::IR::PushData;
	const auto draw_stages =
	    vk::ShaderStageFlagBits::eFragment |
	    (mesh_active ? vk::ShaderStageFlagBits::eMeshEXT : vk::ShaderStageFlagBits::eVertex);
	const auto draw_dwords =
	    PushData::GraphicsDrawDwords(mesh_active, state.ps_input_info.object_primitive_id);
	if (draw_dwords != 0) {
		std::array<uint32_t, PushData::MeshDrawDwordCount> draw_data {};
		if (mesh_active) {
			draw_data = {draw.index_count, static_cast<uint32_t>(emit.vertex_offset),
			             index_source.guest_element_size,
			             static_cast<uint32_t>(index_source.address),
			             static_cast<uint32_t>(index_source.address >> 32u)};
		} else {
			draw_data[0] = buffer.GetUserConfig().GetObjectId();
		}
		vk_buffer.pushConstants(pipeline.pipeline_layout, draw_stages, 0,
		                        draw_dwords * sizeof(uint32_t), draw_data.data());
	}
	if (const auto planes = vertex_stages.back().user_clip_planes; planes != 0) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, draw_stages,
		                        PushData::UserClipPlaneDword * sizeof(uint32_t),
		                        std::bit_width(planes) * 4u * sizeof(uint32_t),
		                        buffer.GetRegisters().GetUserClipPlanes().data());
	}

	auto& dynamic_state = m_context.GetCommandScheduler().GetDynamicState();
	UpdateGraphicsDynamicState(buffer, dynamic_state, vertex_stages.back(), state.depth_info,
	                           rendering);
	dynamic_state.SetAttachmentFeedbackLoopAspects(feedback_aspects);
	dynamic_state.Commit(m_context.GetGraphics(), vk_buffer);

	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages) {
		// As for dispatches, buffer writes wait for earlier accesses to the memory.
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteHazardBarrier(vk_buffer, shader_write_stages);
	}
	LogDrawPhase(draw.Name(), "BeginRendering");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x400u);
	}
	m_context.GetCommandScheduler().BeginRendering(rendering);
	buffer.BindGraphicsPipeline(pipeline.pipeline);
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x500u);
	}
	auto&         occlusion_queries = m_context.GetOcclusionQueries();
	const int32_t occlusion_query   = occlusion_queries.Begin(vk_buffer);
	if (mesh_active) {
		vk_buffer.drawMeshTasksEXT(mesh_groups, draw.instance_count, 1);
	} else if (args_buffer.first == nullptr) {
		EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
	} else if (draw.IsIndexed()) {
		vk_buffer.drawIndexedIndirect(args_buffer.first->Handle(), args_buffer.second,
		                              emit.indirect.count, emit.indirect.stride);
	} else {
		vk_buffer.drawIndirect(args_buffer.first->Handle(), args_buffer.second, emit.indirect.count,
		                       emit.indirect.stride);
	}
	occlusion_queries.End(vk_buffer, occlusion_query);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Draws);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawInstances, draw.instance_count);

	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x600u);
	}
	if (shader_write_stages) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
	LogDrawPhase(draw.Name(), "DrawComplete");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x700u);
	}
}

// Vulkan offsets for a prepared draw, or nullopt when its indirect records need the CPU: mesh
// and quad-list draws for counts, tessellation draws for the index/SGPR prolog. Indirect records
// carry their own offsets.
static std::optional<DrawEmitInfo> ResolveDrawEmit(const HW::UserConfig&        ucfg,
                                                   const ShaderVertexInputInfo& vs,
                                                   DrawOffsetSource source, int32_t first_vertex,
                                                   uint32_t                first_instance,
                                                   const DrawIndirectArgs& indirect) {
	if ((indirect.address != 0 || source == DrawOffsetSource::IndirectArgs) &&
	    ucfg.GetIndexOffset() != 0) {
		static std::once_flag warning_once;
		std::call_once(warning_once, [&] {
			Log::WriteToConsoleAndLog(fmt::format(
			    "GPU: indirect draw ignores GE_INDX_OFFSET=0x{:08x}; geometry may be incorrect.\n",
			    ucfg.GetIndexOffset()));
		});
	}
	if (indirect.address != 0) {
		if (!vs.indirect_draw || vs.stage.module->info.stage == ShaderType::Mesh ||
		    ucfg.GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy ||
		    vs.indirect_offsets != indirect.offsets) {
			return std::nullopt;
		}
		return DrawEmitInfo {.indirect = indirect};
	}
	if (source == DrawOffsetSource::IndirectArgs) {
		// Native fetches consume the patched SGPRs; rewritten fetches need Vulkan offsets.
		return vs.has_dynamic_vertex_fetch ? DrawEmitInfo {first_vertex, first_instance}
		                                   : DrawEmitInfo {};
	}
	const auto [vertex_offset, instance_offset] = ResolveDrawOffsets(ucfg.GetIndexOffset(), vs);
	return DrawEmitInfo {vertex_offset + first_vertex, instance_offset};
}

bool RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& args) {
	KYTY_PROFILER_FUNCTION();
	if (Common::DebugCounters::g_skip_draws.load(std::memory_order_relaxed)) {
		return true;
	}
	if (const std::array<uint64_t, 8> words {
	        args.index_count, reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
	        args.index_type_and_size, static_cast<uint32_t>(args.base_vertex), args.first_instance,
	        args.render_target_slice_offset, args.indirect.address};
	    CrossFrameRepeat(m_context, buffer, words)) {
		return true;
	}
	Common::DebugCounters::SetDrawPhase("DrawSetup/Pending");

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	Common::DebugCounters::SetDrawPhase("DrawSetup/Lock");
	Common::LockGuard lock(m_context.GetMutex());
	if (args.index_count == 0 || (args.instance_count == 0 && args.indirect.address == 0)) {
		return true;
	}

	Common::DebugCounters::SetDrawPhase("DrawSetup/Special");
	if (ConsumeMetadataColorOperation(buffer, args.render_target_slice_offset) ||
	    DepthStencilCopy(buffer) || ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return true;
	}

	Common::DebugCounters::SetDrawPhase("DrawSetup/Checks");
	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	const std::array<uint64_t, 8> replay_args {args.index_count,
	                                           reinterpret_cast<uint64_t>(args.index_addr),
	                                           args.instance_count,
	                                           args.index_type_and_size,
	                                           static_cast<uint32_t>(args.base_vertex),
	                                           args.first_instance,
	                                           args.render_target_slice_offset,
	                                           args.indirect.address};

	const bool registers_repeat = BeginDrawReplay(buffer, replay_args);
	Common::DebugCounters::SetDrawPhase("DrawSetup/Checks");
	// A record holds registers that passed these checks.
	if (!registers_repeat) {
		uc_check(ucfg);
		hw_check(buffer);
	}

	// A draw the worker prepared commits with the state it prepared. Only the other draws
	// value-initialize a state of their own: it is about 17 KB.
	std::optional<DrawRenderState> serial_state;
	if (m_commit_job == nullptr) {
		serial_state.emplace();
		serial_state->indirect_offsets = args.indirect.offsets;
		serial_state->indirect_draw    = args.indirect.address != 0;
	}
	DrawRenderState& state = m_commit_job != nullptr ? m_commit_job->state : *serial_state;
	if (!GetDrawTopology(ucfg, state.topology)) {
		return true;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16:
			index_source.type               = vk::IndexType::eUint16;
			index_source.guest_element_size = 2;
			break;
		case Prospero::IndexType::kIndex32:
			index_source.type               = vk::IndexType::eUint32;
			index_source.guest_element_size = 4;
			break;
		case Prospero::IndexType::kIndex8: index_source.guest_element_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	Common::DebugCounters::SetDrawPhase("DrawSetup/Restart");
	state.primitive_restart_enable = ResolvePrimitiveRestart(buffer, index_source);

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] =
			    state.primitive_restart_enable && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count, args.instance_count,
	                         args.first_instance};
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return true;
	}

	Common::DebugCounters::SetDrawPhase("ResolveDrawEmit");
	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size, args.index_addr);
	const auto emit = ResolveDrawEmit(ucfg, state.vertex_info[0], args.offset_source,
	                                  args.base_vertex, args.first_instance, args.indirect);
	if (emit) {
		ExecutePreparedDraw(submit_id, buffer, draw, state, *emit, index_source);
	}
	ResetBindings();
	return emit.has_value();
}

RenderExecutor::QueuedDraw& RenderExecutor::QueueDraw(uint64_t submit_id, CommandBuffer& buffer) {
	static_assert(std::is_trivially_copyable_v<HW::Context> &&
	              std::is_trivially_copyable_v<HW::UserConfig> &&
	              std::is_trivially_copyable_v<HW::Shader>);
	if (m_queue_tail - m_queue_head == MaxQueuedDraws) {
		CommitOldestDraw(buffer);
	}
	auto& slot = m_queued_draws[m_queue_tail % MaxQueuedDraws];
	if (slot == nullptr) {
		slot = std::make_unique<QueuedDraw>();
	}
	auto& draw       = *slot;
	draw.registers   = buffer.GetRegisters();
	draw.user_config = buffer.GetUserConfig();
	draw.shaders     = buffer.GetShaders();
	draw.submit_id   = submit_id;
	return draw;
}

void RenderExecutor::PublishQueuedDraw(CommandBuffer& buffer) {
	{
		const std::scoped_lock lock(m_prep_mutex);
		m_queued_draws[m_queue_tail % MaxQueuedDraws]->prep = QueuedDraw::Prep::Queued;
		m_queue_tail++;
	}
	if (!m_prep_worker.joinable()) {
		m_prep_worker =
		    std::jthread([this](const std::stop_token& stop) { RunDrawPrepWorker(stop); });
	}
	m_prep_work.notify_one();
	// Draws the worker has finished commit at once, so the queue stays short.
	while (m_queue_head != m_queue_tail && OldestDrawSettled()) {
		CommitOldestDraw(buffer);
	}
}

void RenderExecutor::QueueDrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                                    const DrawIndexArgs& args) {
	if (!Config::ParallelDrawPrepEnabled()) {
		(void)DrawIndex(submit_id, buffer, args);
		return;
	}
	auto& draw      = QueueDraw(submit_id, buffer);
	draw.indexed    = true;
	draw.index_args = args;
	PublishQueuedDraw(buffer);
}

void RenderExecutor::QueueDrawAuto(uint64_t submit_id, CommandBuffer& buffer,
                                   const DrawAutoArgs& args) {
	if (!Config::ParallelDrawPrepEnabled()) {
		(void)DrawAuto(submit_id, buffer, args);
		return;
	}
	auto& draw     = QueueDraw(submit_id, buffer);
	draw.indexed   = false;
	draw.auto_args = args;
	PublishQueuedDraw(buffer);
}

void RenderExecutor::CommitQueuedDraws(CommandBuffer& buffer) {
	while (m_queue_head != m_queue_tail) {
		CommitOldestDraw(buffer);
	}
}

bool RenderExecutor::OldestDrawSettled() {
	const std::scoped_lock lock(m_prep_mutex);
	const auto             prep = m_queued_draws[m_queue_head % MaxQueuedDraws]->prep;
	return prep == QueuedDraw::Prep::Ready || prep == QueuedDraw::Prep::Serial;
}

void RenderExecutor::CommitOldestDraw(CommandBuffer& buffer) {
	using Common::DebugCounters::Counter;
	auto& draw = *m_queued_draws[m_queue_head % MaxQueuedDraws];
	{
		std::unique_lock lock(m_prep_mutex);
		if (draw.prep == QueuedDraw::Prep::Queued) {
			// Not started: preparing it here is faster than waiting for it.
			draw.prep = QueuedDraw::Prep::Serial;
			Common::DebugCounters::Add(Counter::DrawPrepStolen);
		}
		if (draw.prep == QueuedDraw::Prep::Preparing) {
			const auto start = Common::DebugCounters::TimerStart();
			while (draw.prep == QueuedDraw::Prep::Preparing) {
				// The worker can wait for a command it sent to this thread.
				lock.unlock();
				{
					const Common::DebugCounters::ScopedDrawPhase phase("QueueCommit/Commands");
					GuestGpu::RunPendingCommands();
				}
				lock.lock();
				if (draw.prep == QueuedDraw::Prep::Preparing) {
					const Common::DebugCounters::ScopedDrawPhase phase("QueueCommit/Wait");
					m_prep_done.wait_for(lock, std::chrono::microseconds(100));
				}
			}
			Common::DebugCounters::AddSince(Counter::DrawPrepWaitNs, start);
		}
	}
	m_commit_job = nullptr;
	if (draw.prep == QueuedDraw::Prep::Ready) {
		if (draw.dirty_epoch != m_context.GetBufferCache().GpuDirtyEpoch() ||
		    draw.program_generation != m_context.GetPipelineCache().ProgramGeneration()) {
			Common::DebugCounters::Add(Counter::DrawPrepStale);
		} else {
			Common::DebugCounters::Add(Counter::DrawPrepUsed);
			m_commit_job = &draw;
		}
	}
	auto& buffers = m_context.GetBufferCache();
	if (m_commit_job != nullptr && draw.prestream.entry != 0) {
		buffers.UsePrestream(&draw.prestream);
	}
	const auto live = buffer.SwapRegisters({&draw.registers, &draw.user_config, &draw.shaders});
	if (draw.indexed) {
		(void)DrawIndex(draw.submit_id, buffer, draw.index_args);
	} else {
		(void)DrawAuto(draw.submit_id, buffer, draw.auto_args);
	}
	buffer.SwapRegisters(live);
	buffers.UsePrestream(nullptr);
	// After the draw's commands, so the space stays until the GPU has read it.
	buffers.RetirePrestream(draw.prestream);
	m_commit_job = nullptr;
	const std::scoped_lock lock(m_prep_mutex);
	m_queue_head++;
}

void RenderExecutor::RunDrawPrepWorker(const std::stop_token& stop) {
	std::unique_lock lock(m_prep_mutex);
	while (m_prep_work.wait(lock, stop, [&] { return m_prep_next < m_queue_tail; })) {
		m_prep_next = std::max(m_prep_next, m_queue_head);
		if (m_prep_next == m_queue_tail) {
			continue;
		}
		auto& draw = *m_queued_draws[m_prep_next++ % MaxQueuedDraws];
		if (draw.prep != QueuedDraw::Prep::Queued) {
			continue;
		}
		draw.prep = QueuedDraw::Prep::Preparing;
		lock.unlock();
		const bool ready = PrepareQueuedDraw(draw);
		lock.lock();
		draw.prep = ready ? QueuedDraw::Prep::Ready : QueuedDraw::Prep::Serial;
		m_prep_done.notify_all();
	}
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool RenderExecutor::PrepareQueuedDraw(QueuedDraw& draw) {
	using Common::DebugCounters::Counter;
	namespace IR         = ShaderRecompiler::IR;
	const auto  start    = Common::DebugCounters::TimerStart();
	const auto& ctx      = draw.registers;
	const auto& ucfg     = draw.user_config;
	const auto& indirect = draw.indexed ? draw.index_args.indirect : draw.auto_args.indirect;
	const auto& index    = draw.index_args;
	const auto& vertices = draw.auto_args;
	const bool  no_index =
	    index.index_count == 0 || (index.instance_count == 0 && indirect.address == 0);
	const bool no_vertices =
	    indirect.address == 0 && (vertices.vertex_count == 0 || vertices.instance_count == 0);
	const bool empty  = draw.indexed ? no_index : no_vertices;
	const auto copies = FindDepthStencilCopies(ctx);

	vk::PrimitiveTopology topology {};
	Common::DebugCounters::Add(Counter::DrawPrepJobs);
	// The commit drops these draws, or turns them into other operations, before their programs.
	if (empty) {
		Common::DebugCounters::Add(Counter::DrawPrepSkipped);
		Common::DebugCounters::Add(Counter::DrawPrepSkippedEmpty);
		return false;
	}
	// Disable and Normal draw; the other modes are metadata and resolve operations.
	const auto mode = ctx.GetColorControl().mode;
	if ((mode != static_cast<uint8_t>(CbColorMode::Disable) &&
	     mode != static_cast<uint8_t>(CbColorMode::Normal)) ||
	    copies.depth || copies.stencil) {
		Common::DebugCounters::Add(Counter::DrawPrepSkipped);
		Common::DebugCounters::Add(Counter::DrawPrepSkippedSpecial);
		return false;
	}
	if (!DrawHasValidVertexShader(draw.shaders)) {
		Common::DebugCounters::Add(Counter::DrawPrepSkipped);
		Common::DebugCounters::Add(Counter::DrawPrepSkippedShader);
		return false;
	}
	if (!GetDrawTopology(ucfg, topology)) {
		Common::DebugCounters::Add(Counter::DrawPrepSkipped);
		Common::DebugCounters::Add(Counter::DrawPrepSkippedTopology);
		return false;
	}
	if (Common::DebugCounters::g_skip_draws.load(std::memory_order_relaxed) ||
	    graphics_debug_dump_enabled()) {
		Common::DebugCounters::Add(Counter::DrawPrepSkipped);
		return false;
	}
	auto& state = draw.state;
	state       = DrawRenderState {.indirect_offsets = indirect.offsets,
	                               .indirect_draw    = indirect.address != 0,
	                               .resources        = &draw.resources};

	draw.dirty_epoch        = m_context.GetBufferCache().GpuDirtyEpoch();
	draw.program_generation = m_context.GetPipelineCache().ProgramGeneration();
	draw.decoded            = false;
	draw.heap_peeks.clear();
	PipelineCache::SetHeapPeeks(&draw.heap_peeks);
	IR::BeginSpeculation();
	bool ready =
	    m_context.GetPipelineCache().PrepareGraphicsPrograms(ctx, ucfg, draw.shaders, state) &&
	    state.vertex_info[0].stage.module != nullptr;
	if (ready) {
		// Decoding can stop the emulator on descriptors of draws the commit drops, so only draws
		// that reach their bindings whatever their targets are decoded here.
		const auto& vs   = state.vertex_info[0];
		const bool  mesh = vs.stage.module->info.stage == ShaderType::Mesh;
		const bool  rect_skip =
		    !draw.indexed && Prospero::IsRectList(ucfg.GetPrimType()) && vs.buffers_num == 0 &&
		    vs.stage.module->info.param_export_mask == 0 && state.ps_input_info.input_num != 0;
		const auto emit = draw.indexed
		                      ? ResolveDrawEmit(ucfg, vs, index.offset_source, index.base_vertex,
		                                        index.first_instance, indirect)
		                      : ResolveDrawEmit(ucfg, vs, vertices.offset_source,
		                                        static_cast<int32_t>(vertices.first_vertex),
		                                        vertices.first_instance, indirect);
		if (state.ps_active && !mesh && !rect_skip && emit.has_value() &&
		    !ctx.GetRenderControl().copy_depth_to_color) {
			const auto stages = state.VertexStages();
			for (uint32_t i = 0; i < stages.size(); i++) {
				DecodeBindings(state.vertex_info[i].stage, draw.bindings[i]);
			}
			DecodeBindings(state.ps_input_info.stage, draw.bindings[3]);
			draw.decoded = true;
		}
	}
	const auto miss = IR::EndSpeculation();
	PipelineCache::SetHeapPeeks(nullptr);
	switch (miss) {
		case IR::SpeculativeMiss::None: break;
		case IR::SpeculativeMiss::WalkerFault:
			Common::DebugCounters::Add(Counter::DrawPrepFaults);
			break;
		case IR::SpeculativeMiss::Memory:
			Common::DebugCounters::Add(Counter::DrawPrepMemory);
			break;
		case IR::SpeculativeMiss::ImageHeap:
			Common::DebugCounters::Add(Counter::DrawPrepHeap);
			break;
		case IR::SpeculativeMiss::Program:
			Common::DebugCounters::Add(Counter::DrawPrepMissing);
			break;
	}
	ready = ready && miss == IR::SpeculativeMiss::None;
	if (ready) {
		Common::DebugCounters::Add(Counter::DrawPrepReady);
		if (draw.decoded && m_context.GetBufferCache().PrestreamEnabled()) {
			PrestreamBuffers(draw);
		}
	}
	Common::DebugCounters::AddSince(Counter::DrawPrepNs, start);
	return ready;
}

void RenderExecutor::PrestreamBuffers(QueuedDraw& draw) {
	std::array<PreparedBindings*, 4> stages {};
	uint32_t                         count = 0;
	for (uint32_t i = 0; i < draw.state.VertexStages().size(); i++) {
		stages[count++] = &draw.bindings[i];
	}
	if (draw.state.ps_active) {
		stages[count++] = &draw.bindings[3];
	}
	const auto decoded = std::span {stages.data(), count};
	// The same ranges the commit's FindBuffers decodes.
	DecodeBuffers(decoded);
	auto& cache = m_context.GetBufferCache();
	cache.BeginPrestream(draw.prestream);
	for (const auto* stage: decoded) {
		for (const auto& resource: stage->runtime->module->info.info.buffers) {
			if (resource.descriptor_index == UINT32_MAX || resource.written) continue;
			const auto& source = stage->buffer_sources[resource.descriptor_index];
			if (source.address != 0 && source.size != 0) {
				cache.Prestream(draw.prestream, source.address, source.size);
			}
		}
	}
}

bool RenderExecutor::AcquirePeekedHeaps(const QueuedDraw& draw) {
	auto& heap = m_context.GetImageHeap();
	bool  same = true;
	// Every view is acquired, as the GPU thread's own programs would, even after a change.
	for (const auto& peek: draw.heap_peeks) {
		auto request = peek.request;
		request.root = &peek.root;
		ShaderRecompiler::IR::ImageHeapView view;
		const bool                          acquired = heap.Acquire(request, view);
		same = same && acquired && view.slots == peek.slots && view.slot_count == peek.slot_count &&
		       std::ranges::equal(view.kinds, peek.kinds);
	}
	return same;
}

void RenderExecutor::VerifyPreparedPrograms(const CommandBuffer& buffer, DrawRenderState& state) {
	if (m_verify_state == nullptr) {
		m_verify_state = std::make_unique<DrawRenderState>();
	}
	auto& check = *m_verify_state;
	check       = DrawRenderState {.indirect_offsets = state.indirect_offsets,
	                               .indirect_draw    = state.indirect_draw,
	                               .resources        = &m_draw_resources};
	m_context.GetPipelineCache().GetGraphicsPrograms(buffer, check);
	const auto same_stage = [&](const ShaderStageRuntime& prepared,
	                            const ShaderStageRuntime& serial, uint32_t stage) {
		if (prepared.module != serial.module) return false;
		if (serial.module == nullptr) return true;
		const auto& x = *prepared.resources;
		const auto& y = *serial.resources;
		return x.buffers == y.buffers && x.images == y.images && x.samplers == y.samplers &&
		       x.flattened_srt == y.flattened_srt && x.user_data == y.user_data &&
		       (*state.resources)[stage].specialization == m_draw_resources[stage].specialization;
	};
	bool same =
	    check.ps_active == state.ps_active && check.mrt_mask == state.mrt_mask &&
	    (!check.ps_active || same_stage(state.ps_input_info.stage, check.ps_input_info.stage, 3));
	for (uint32_t i = 0; i < check.vertex_info.size(); i++) {
		same = same && same_stage(state.vertex_info[i].stage, check.vertex_info[i].stage, i);
	}
	if (same) {
		return;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawPrepMismatches);
	static std::atomic<uint32_t> logged = 0;
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		const auto* vs = check.vertex_info[0].stage.module;
		Log::WriteToConsoleAndLog(fmt::format(
		    "Draw prep: worker programs differ from the GPU thread's (vs module 0x{:016x}).\n",
		    vs != nullptr ? vs->id : 0));
	}
	state.vertex_info     = check.vertex_info;
	state.ps_input_info   = check.ps_input_info;
	state.ps_active       = check.ps_active;
	state.mrt_mask        = check.mrt_mask;
	state.resources       = &m_draw_resources;
	m_commit_job->decoded = false;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_PROFILER_FUNCTION();
	if (Common::DebugCounters::g_skip_draws.load(std::memory_order_relaxed)) {
		return true;
	}
	if (const std::array<uint64_t, 6> words {
	        args.vertex_count, args.instance_count, args.first_vertex, args.first_instance,
	        args.render_target_slice_offset, args.indirect.address};
	    CrossFrameRepeat(m_context, buffer, words)) {
		return true;
	}
	Common::DebugCounters::SetDrawPhase("DrawSetup/Pending");

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);

	Common::DebugCounters::SetDrawPhase("DrawSetup/Lock");
	Common::LockGuard lock(m_context.GetMutex());
	if (args.indirect.address == 0 && (args.vertex_count == 0 || args.instance_count == 0)) {
		return true;
	}

	Common::DebugCounters::SetDrawPhase("DrawSetup/Special");
	if (ConsumeMetadataColorOperation(buffer, args.render_target_slice_offset) ||
	    DepthStencilCopy(buffer) || ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return true;
	}

	Common::DebugCounters::SetDrawPhase("DrawSetup/Checks");
	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	const std::array<uint64_t, 7> replay_args {args.vertex_count,
	                                           args.instance_count,
	                                           static_cast<uint32_t>(args.first_vertex),
	                                           args.first_instance,
	                                           args.render_target_slice_offset,
	                                           args.indirect.address,
	                                           uint64_t {1} << 63u};

	const bool registers_repeat = BeginDrawReplay(buffer, replay_args);
	Common::DebugCounters::SetDrawPhase("DrawSetup/Checks");
	// A record holds registers that passed these checks.
	if (!registers_repeat) {
		uc_check(ucfg);
		hw_check(buffer);
	}

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndexAuto, args.vertex_count,
	                         args.instance_count, args.first_instance};

	// A draw the worker prepared commits with the state it prepared. Only the other draws
	// value-initialize a state of their own: it is about 17 KB.
	std::optional<DrawRenderState> serial_state;
	if (m_commit_job == nullptr) {
		serial_state.emplace();
		serial_state->indirect_offsets = args.indirect.offsets;
		serial_state->indirect_draw    = args.indirect.address != 0;
	}
	DrawRenderState& state = m_commit_job != nullptr ? m_commit_job->state : *serial_state;
	if (!GetDrawTopology(ucfg, state.topology)) {
		ResetBindings();
		return true;
	}
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return true;
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.module->info.param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return true;
	}

	Common::DebugCounters::SetDrawPhase("ResolveDrawEmit");
	LogDrawStateIfNeeded(buffer, draw, state, 0, nullptr);
	const auto emit = ResolveDrawEmit(ucfg, state.vertex_info[0], args.offset_source,
	                                  static_cast<int32_t>(args.first_vertex), args.first_instance,
	                                  args.indirect);
	if (emit) {
		ExecutePreparedDraw(submit_id, buffer, draw, state, *emit, {});
	}
	ResetBindings();
	return emit.has_value();
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer,
                                         uint32_t       render_target_slice_offset) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, render_target_slice_offset, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, render_target_slice_offset, 1, true, true);
	if (!src.image_id || !dst.image_id) {
		return false;
	}
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source,
	                    {src.desc.view_info.base_level, 1, src.desc.view_info.base_layer, 1},
	                    {dst.desc.view_info.base_level, 1, dst.desc.view_info.base_layer, 1});
	return true;
}

} // namespace Libs::Graphics
