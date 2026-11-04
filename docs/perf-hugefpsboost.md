# Performance work on `hugefpsboost`

Notes on the CPU/GPU performance work for Cyberpunk 2077. Measured values are
marked as measurements; everything else is an estimate or a plan.

Test machine of record: user's PC, preset `x64-relwithdebinfo`. LTO + PGO gives
about +5 FPS on top (user measurement).

## Command-line flags (all off by default)

| Flag | What it does |
|------|--------------|
| `--perf-overlay` | Debug counters and the overlay (draw phases, GPU busy, buffer cleanliness). |
| `--vk-call-timing` | TSC-timed trampolines around Vulkan entry points. |
| `--deferred-eop-labels` | Writes end-of-pipe labels only after the GPU finishes the work before them. |
| `--zero-copy-buffers` | Binds guest memory directly instead of copying into the stream buffer. |
| `--parallel-draw-prep`, `--verify-draw-prep` | Prepares draw state on workers; verify mode compares with the serial path. |
| `--parallel-heap-prep`, `--parallel-buffer-prep` | Parallel image-heap and buffer preparation. |
| `--fast-unmap` | An unmap waits for the GPU only when GPU-written bytes or queued completions overlap the range. |
| `--fast-guest-reads` | Guest reads inside backing mappings skip the address-space lock. |
| `--fast-color-clear-checks` | MaterializeColorClear skips unchanged images; every 16th skip is checked. |
| `--xframe-draw-stats` | Share of draws whose registers repeat across frames. |
| `--xframe-skip-repeats` | Upper-bound experiment: repeated draws are not rendered (the frame is wrong). |
| `--draw-records`, `--draw-records-verify` | Repeated draws take targets and texture bindings from a record; verify mode. |
| `--no-image-barrier-merge` | Restores the old image barrier behavior. |

## Measurements (one Cyberpunk scene)

- FPS without overlay: 17–19.
- CPU cost per draw on the GPU thread: about 9.86 µs without
  `--draw-records`. Largest parts: PrepareGraphicsBindings 2.3–3.0 µs,
  PrepareBindings about 1.4, vertex buffers about 1.0,
  ResolveRenderColorTarget 0.9–1.1, CommitDraw about 0.9.
- `--draw-records`: 9.86 → 10.95 µs per draw (slower).
- GPU busy (timestamps): 17.7 ms per frame without `--zero-copy-buffers`
  (ceiling about 56 FPS), 29.6–32 ms per frame with it (ceiling about
  31–34 FPS). GPU clocks were near maximum (nvidia-smi: 2835/3105 MHz, P0).
  Zero-copy cuts the BStream phase from 1.21 to 0.50 µs per draw but makes
  the GPU read guest memory over the bus.
- Conclusion: the frame is bound by the GPU thread on the CPU, not by the GPU.

## Closed directions

- **Replaying whole draws across frames** (`ReplayUnsafeDraw`, removed): SRT user
  data and descriptors are the same as last frame in 0% of draws. Best gain was about 3%. Code was rolled back.
- **Buffer cleanliness**: 0% of draws have all their buffers clean between
  frames. Per-object records keyed on buffer contents (chenxiao07's "native"
  mode) will not hit in Cyberpunk; the keys must exclude data that changes
  every frame.

## Porting chenxiao07's work

Source: github.com/chenxiao07/KytyPS5 at 86819f4 (GPLv2, same as this repo;
common ancestor upstream 1c301c9). His fork is tuned for Demon's Souls, has no
ImageHeap, and does not boot Cyberpunk (the Cyberpunk fixes landed upstream
later). His doc `docs/GPU-DRIVEN-XPR.md` reports +16–19% from table mode,
about 1.4 µs per table draw and 3.6 µs per native draw.

| His part | File | Plan here |
|----------|------|-----------|
| ResolvedBuffer: direct-mapped V# → buffer cache (65536 entries, epochs) | `src/local/table-xpr.inc` | Step A: port behind a flag into FindBuffers. |
| RingCopy: shared copies of CPU-written ranges across draws (4096) | `src/local/table-xpr.inc` | Step A, together with the cache. |
| HeldBuffer | `src/local/table-xpr.inc` | After step A. |
| Table mode: TablePair, TableImageSet keyed by T#/S#, TableEvaluate | `src/local/table-xpr.inc` + translator TablePlan/BindingLayout | Later; needs translator changes. |
| Native object records with prebuilt descriptor sets | `src/local/native-xpr.inc` | Not planned for Cyberpunk (see buffer cleanliness). |

Step A target: Find + Buffers + BStream phases, about 2.3 µs per draw
together. Success is measured with the overlay's draw phases and FPS without
overlay.
