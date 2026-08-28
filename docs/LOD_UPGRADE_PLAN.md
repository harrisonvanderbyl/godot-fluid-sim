# Upgrade plan: VoxelLodTerrain + LODpaged velocity field

Status: **plan** (not yet implemented). Written 2026-08-28.

This plan replaces the fixed-density `VoxelTerrain` child and the single dense
chunk grid with `VoxelLodTerrain` and a paged, LOD-aware velocity field. It is
organized as 7 phases designed to be implemented and debugged incrementally —
the simulation stays binary-compatible (all particles at LOD 0) until Phase 4.

---

## 0. Terminology & coordinate spaces

| Term | Meaning |
|---|---|
| **Voxel space** | The simulation's integer cell space (current `grid_w/h/d` box, 1 unit per cell at LOD 0). |
| **LOD k cell** | A cell covering `2^k` voxel-space units per axis. |
| **Page** | One allocated block of the arena: `16³` cells at LOD 0, `8³` at LOD 1, `4³` at LOD 2, `2³` at LOD 3. Every page therefore covers the same *voxel-space volume* (`16³` cells worth). |
| **Arena** | One big GPU storage buffer of `ChunkCell`s, subdivided into page slots. |
| **Page table** | Per-LOD fixed 3D grid of 16-byte entries mapping `(lod, page_coord) → arena base offset + flags`. |
| **Terrain block** | A `VoxelLodTerrain` data block: `get_data_block_size() == 16` voxels at LOD k, i.e. covers `16 << k` voxels — exactly one of our pages. 1 terrain block ⇔ 1 page. |

Because one terrain block at any LOD covers the same voxel volume as one of our
pages, the page table is filled directly from terrain signals with
**no coordinate math beyond `>> k`**.

---

## 1. GPU memory redesign (final layout)

### 1.1 Arena (`cell_arena_buf`) — replaces `chunk_buf`

Same `ChunkCell` struct (32 B: `occupant`, `vel_x/y/z_bits`, `sdf_bits`,
`count`), but the buffer is sized as:

```
max_arena_cells = max_pages * page_cells        (page_cells = 4096)
```

`max_pages` defaults to 4096 (⇒ 4096 × 4096 cells × 32 B = 512 MB worst case;
typical usage will be a few hundred pages). Grown by doubling + rebuild when
exhausted. For the 256³ default sim box, full coverage is 256³/16³ = 4096
pages at LOD 0, but terrain proximity means typical allocation will be a
fraction of that.

Key property: **a particle's reads/writes and the SDF bake target are the same
struct as today**, so the physics shader's collision path (`sdf_bits`) keeps
working per-page — SDF for LOD k pages is sourced from the terrain's LOD k
blocks (naturally coarser far away, 8× cheaper bufs).

### 1.2 Page table (`lod_table_buf`) — new

One entry per (lod, page coord) over the sim box:

```
struct LodPageEntry {        // 16 B, std430
    uint32_t cell_base;      // arena cell offset of the page
    uint32_t flags;          // bit0=allocated, bits 8..15 = generation
    int32_t  occupant_hint;  // -1 = page surely empty (skip), else unknown
    uint32_t reserved;
};
```

- Table size per LOD: `ceil(grid / (16 << k))` per axis. LOD 0 covers the full
  box; LOD 1 covers `ceil(grid/32)²` etc.
- Lookups are O(1) direct indexing with no hashing (sim box is bounded).
- `occupant_hint` is a cheap early-out maintained by clear/alloc (cleared on
  full-page-dealloc; `atomicOr`-compatible).

### 1.3 Store-to-disk request ring (`store_ring_buf`) — new

```
uint32_t ring[N];           // N = 65536, values = particle indices
uint32_t write_cursor;      // atomicAdd'd by particles
```

Particles whose region has **no allocated page** append their index here
exactly once per gap (guard: only when `flags.bit0 == 0` AND a second CAS on a
dedicated `particle_stored` bit in `custom0.w` — set once, cleared by the CPU
drain). CPU drains the ring every frame, persists those particles, and marks
them inactive (`NaN` position sentinel — the pool already supports this).

### 1.4 Dispatch-time constants

`PushConstants` gains (in currently-unused spare bytes; still ≤ 128 B):

```
int32_t lod_levels;          // page-count exponent table
int32_t sim_stride_mask;     // phase mask = 2^maxlod - 1
int32_t arena_pages;
int32_t lod0_table_stride;   // for index math
```

(LOD dims per level are derivable in-shader from `grid_w/h/d`.)

---

## 2. Terrain swap — `VoxelTerrain` → `VoxelLodTerrain` (Phase 1)

All changes stay inside `_refresh_sdf_from_child()` and the `_set/_get`
`terrain_` property forwarding (no API-shape change: we never link
godot_voxel, everything goes through `Object::call`).

1. `ClassDB::instantiate("VoxelLodTerrain")` instead of `"VoxelTerrain"`.
2. Defaults applied to the child: `lod_count = 4`, `mesh_block_size = 16`,
   `lod_distance = 48`, `streaming_system = CLIPBOX` (supports multiple
   viewers; legacy octree also fine), `collision_lod_count = 1` (only near LOD
   needs colliders if used), `full_load_mode = false`.
3. Terrain AABB: since VLT is centered on viewers, the sim box should be
   anchored to this node's global transform. The fluid grid origin = node
   origin; particles clamp inside as today.
4. New signal set, with **runtime arity probe**:

```
block_loaded(position, lod?)         block_unloaded(position, lod?)
mesh_block_entered(position, lod?)   mesh_block_exited(position, lod?)
```

   Bind a single C++ method `_on_terrain_block_event(kind:int, position:Variant,
   lod:Variant)` and, on connect, check `get_signal_list()` for the arg count
   of `block_loaded`; if the build only passes `position`, derive `lod` on our
   side (see §3.1). This tolerates both the current build
   (`position,lod`) and older `position`-only builds.
5. SDF refresh path kept temporarily for LOD0: the existing dense
   `sdf_storage_buf` + `clear_grid` bake keeps running, but only dispatches the
   clear over pages that are *allocated* after Phase 3 (controlled by the
   arena-driven page loop, so cost scales with resident pages, not 16.7 M
   cells).

Interim risk guard: until Phase 3 lands, **all pages are force-allocated** at
LOD 0 over the sim box (both for the arena and the LOD0 page table), making
Phase 1–2 observably identical to the current dense grid.

---

## 3. LOD tracking: signals → page lifecycle (Phases 2–3)

### 3.1 Building our own octree view

VLT emits per-(position, lod) events; it does not emit "this region split /
merged". We reconstruct transitions locally:

- Maintain `HashMap<Vector3i, uint8_t> active_page_lod` keyed by **page coord
  in a shared viewport** (page coords at LOD k, and voxel-space range
  `[pos << k, (pos<<k)+16)`) — the map key is `page_pos << (12 + lod)` styles
  or (simpler, since bounded) `Dictionary[(lod, page_coord)]`.
- On `block_loaded(p, lod)`  → mark (p, lod) active.
- On `block_unloaded(p, lod)` → clear (p, lod) active.

A page is **resident for simulation** at the finest LOD among active entries
covering its voxel range. Because one page covers exactly one terrain block at
any lod, a "transition" is always one of:

- **coarsen**  : `(p, k)` became inactive while `(p >> 1, k+1)` covering the
  same range became/becomes active.
- **refine**   : the mirror image.
- **unload**   : range no longer covered at any lod.
- **load**     : range gains coverage.

The debouncer accumulated in `_process` (same pattern as today's
`sdf_refresh_pending`, but now a list) batches events to one GPU sync per
frame.

Fallback when a signal might be missed: re-resolve ambiguity by calling
`voxel_to_data_block_position(vpos, k)` + `debug_get_data_block_info(pos, k)`
(`loading_state` 0/1/2) for candidate lods — used only by the recovery path /
an inspector "Resync LODs" button, never per-frame.

### 3.2 Page allocator (CPU)

`VoxelLodArena` (new `src/lod_arena.{hpp,cpp}`):

- `HashMap<PageKey{lod, coord}, uint32_t>` resident→slot,
  `std::vector<uint32_t> freelist`, `std::vector<uint8_t> slot_lod/gen`.
- `alloc(lod, coord)` / `free(slot)` / `retarget(slot, new_lod)` — the retarget
  is where migration is scheduled (§3.3). Freelist exhaustion → grow `max_pages`
  (double + full rebuild, same as grid-size changes today) or, if growth is
  disabled, refuse and leave the page unallocated (particles then skip-physics,
  which is the *safe* failure mode).
- Stale-END sanity: a page whose voxel range left view entirely is freed with
  1 deferred GPU frame (reuse `_queue_free_rid` discipline if we ever make the
  arena itself swappable; in practice the arena is persistent and only its
  *contents* churn, so page reuse is immediate and safe within one GPU sync
  boundary).

### 3.3 LOD transition — migrate shader (`lod_migrate.glsl`, new)

Dispatched per transition (CPU fills a small `MigrateJob { src_base, dst_base,
src_lod, dst_lod }` list, one push constant each):

- **coarsen (16³ → 8³ equivalent cell volume, i.e. 2:1 per axis)**: each dst
  cell gathers its 8 src children:
  - `count  = Σ child.count`
  - `vel    = Σ child.vel` (momentum is extensive — sum conserves it; divide by
    count when redistributing later)
  - `occupant = first non-(-1) child idx` (phase hint only — same semantics as
    today's representative)
  - `sdf     = min(children)` (currently valid: belongs to a coarser terrain
    level, but `min` keeps terrain conservative)
- **refine (2:1 up)**: each dst child copies the src parent, with `vel /= 8`
  (or `< 8` for corner children outside the refined terrain — out-of-terrain
  children get `occupant=-1` and `vel=0`).
- Uncovered range (unload): nothing to copy — page freed; particles in that
  range take the store-to-disk path that frame (§5.3).
- New load at finer lod (load): page zero-initialized by `lod_alloc_clear.glsl`
  (the block-scoped replacement for full-grid `clear_grid.glsl`) with SDF
  sampled from the terrain at that lod (VLT has per-lod SDF).
  Clear/alloc can be merged into one shader: `lod_alloc` writes the page entry +
  zeros cells + bakes SDF in one dispatch.

GPU ordering note: everything runs on the local `RenderingDevice`, single
`submit()/sync()` per frame (existing discipline). Allocation page-table
writes happen via a tiny `buffer_update` on `lod_table_buf` *before* the
compute list; migrations and physics run in the same compute list after it, so
ordering is guaranteed without extra syncs.

---

## 4. Physics shader changes (`velocity_spread.glsl`)

### 4.1 Per-particle LOD resolution

At dispatch start, each particle resolves its sim LOD:

1. Read cached `lod` from `custom0.w` (0 = invalid / must resolve).
2. Walk the page table **finest → coarsest** for the cell containing
   `position`: first level whose entry has `flags.bit0` wins. Store
   `custom0.w = lod | 0x40` (invalidation marker bit).
3. If **no level allocates** the containing page: the particle is beyond
   simulation range → take the store-ring path (§5.3) and return (skip
   everything; costs one table walk + one atomic CAS).

On any subsequent frame, the cached entry's `flags/gen` is compared against the
page table first; a mismatch (page moved/freed/refined) triggers a fresh walk.

### 4.2 Strided time stepping

Particles at LOD k run every `stride = 1 << k` frames, individually phased so
work spreads evenly:

```
phase  = (particle_idx * 0x9E3779B1) & (stride - 1)   // cheap hash
run    = ((frame_count + phase) & (stride - 1)) == 0
```

When running, scale the step by `stride`:

- gravity term `× stride`
- spread/momentum代言 unchanged (the impulse pool is consumed & re-deposited
  within one particle-step, so larger steps are self-consistent)
- `opacity_fade` and smoothing (`neighbors_filled`, viscosity mix) scale their
  `mix` factors by `1/stride` so visual smoothing stays frame-rate-independent.

Cap scaling ("more particles fit in one cell"): `my_cap = max_occupancy <<
(3 * k)` (physical volume semantics: an LOD1 cell holds the volume of 8 LOD0
cells). Neighbour phase checks (`occ >= 0` particles on the other side of a LOD
border) must resolve the *representative's* phase via `Particle.color_packed`
alpha — already how solid/liquid is detected today.

### 4.3 Cell addressing

All `cell_index(p)` call sites become `resolve_cell(p, lod)`:

```
entry = lod_table[lod][p >> (4 + ... )]      // per-lod dims in push consts
assert entry allocated (caller guaranteed)
return entry.cell_base + local_index(p & page_mask)
```

`in_bounds` semantics unchanged (still voxel-space box), out-of-range
neighbours behave exactly as today.

### 4.4 LOD borders

When a particle's step wants to cross to a cell in a *different* page:

- Neighbour page at same lod missing → treat as boundary: reflect momentum
  (same code path as box boundary), do not allocate.
- Neighbour at coarser lod → allowed: coarse alcohol reads harvest from the
  coarser page (overlapping coverage means both lods may be resident — beats
  hard reflection, and transitions keep working during the migrate frame).
  Keep a define `LOD_BORDER_MODE` between reflect / coarsified-read.

### 4.5 Sources / sinks (`fluid_source_sink.glsl`)

Spawn resolve: `resolve_cell(spawn_pos, finest)`; if no page → do not spawn,
increment a `spawn_deferred` counter (source retries next frame). Sinks always
fine. Push consts gain the source/sink radius in voxel units (unchanged).

---

## 5. Frame loop changes (C++)

### 5.1 `_process` order

```
sync rd
process deferred frees
if pending_terrain_events:      apply_pages(events)   // alloc/free/migrate
if sdf_refresh_pending:         rebake SDF on dirty pages (lod_alloc clear)
assemble container force (unchanged)
dispatch_physics (unchanged pipeline; new ptr push consts)
sources/sinks dispatch (unchanged)
drain store ring → persist inactive (§5.3)
render (unchanged)
submit / sync (unchanged)
```

### 5.2 Dispatch indirection

Two viable options; start with (a):

(a) **Whole-range dispatch** — all `num_particles` threads, early-out on
unresolved/unallocated. Simple, correct; wasted lanes for far particles are
cheap (a table walk + return).

(b) **Runnable segmentation** — every ~16 frames rebuild `runnable_indices`
sorted by particle LOD (radix on `custom0.w`), push per-segment
`{begin, count, lod}` and dispatch one group range per lod. Do this only if
profiling shows step (a) wastes > 15% ON the dispatch.

### 5.3 Store-to-disk drain (CPU)

Each frame, `buffer_get_data` on a tiny 2-int header (`write_cursor`,
`overflow`) + unpacked indices tail since last drain (keep the ring's CPU mirror
of write_cursor):

- For each drained particle idx: pack its state (position, color, custom0) into
  a serialized chunk (per-node ring file `user://fluid_spill/<node>_<chunk>.bin`,
  or a caller-provided `VoxelStream`-like callable later).
- Set that particle inactive (NaN position, `buffer_update` of a small list of
  4-float strips; batch ≤ 4096/frame).
- Clear its stored bit in `custom0.w` so a future contiguous page can respawn
  it (via an `lod_restore` pass or by a FluidSource).

Overflow behavior: if the ring fills in one frame, the shader skips further
writes and increments `overflow`; the CPU prints (throttled) — particles beyond
overflow simply skip-physics silently, no corruption.

### 5.4 Stats / debug

- `get_lod_stats()` → Dictionary { pages_by_lod, particles_by_lod, skipped,
  store_ring_pending, arena_free_slots } from a 32-int stats buffer zeroed by
  `lod_alloc`/physics prologues each frame.
- Inspector toggle `debug_tint_by_lod`: multiply color by a per-lod tint in the
  physics shader epilogue for visual verification.
- Existing grid gizmo: also draw allocated page wireframes (per-resident page
  AABB in voxel space) — cheap clear+redraw on an ImmediateMesh.

---

## 6. Implementation order (7 PRs)

| # | Scope | Acceptance |
|---|---|---|
| **P1** | Terrain swap (`VoxelLodTerrain`, property fwd, 2-arg signals w/ arity probe); force-alloc LOD0 pages | Scene runs identical to today; lod events logged |
| **P2** | Arena + page table + store-ring buffers; `VoxelLodArena` C++; all-lod0 force alloc; block-scoped clear | Buffers explorable in debugger; sim unchanged |
| **P3** | Signal → page lifecycle, `lod_migrate.glsl`, `lod_alloc.glsl`; CPU octree view & debounce | Toggle `lod_distance` in inspector; migration visibly preserves momentum on coarsen/refine; no nan/leak over 1000 transitions |
| **P4** | Physics LOD resolution, strided stepping, cap scaling, store-ring skip path | Particle count scales with distance; skipped particles logged + persisted; fps↑ at same visual density |
| **P5** | Source/sink LOD spawning; render sort/debug tint; `get_lod_stats` | Sources still fill near LOD0; tint shows correct lods |
| **P6** | Runnable segment dispatch (opt-in), menu polish, README update | Optional |
| **P7** | Persistence format + restore (FluidSource-compatible spawner reading spill files) | Restart restores spilled particles |

---

## 7. Risks & mitigations

1. **Signal arity across godot_voxel builds** — probe `get_signal_list()` and
   write callbacks tolerant to both `(position)` and `(position, lod)`.
2. **Transition missing → stale page** — every N s run a cheap audit pass
   (compare our octree against `debug_get_data_block_info` sampling) in debug
   builds only.
3. **Arena exhaustion** — graceful degrade = particles skip-physics (should
   *never* crash or lose live near particles: LOD0 has alloc priority,
   `alloc()` reserves a small LOD0-only pool).
4. **Momentum double-count across the migrate frame** — migrate runs in the
   same compute list BEFORE physics, physics reads only dst pages that frame;
   src page is marked freed-in-table before physics so no one reads both.
5. **`internal_local_device` single-submission constraint** — unchanged; all
   new work rides the existing `submit()/sync()` point.
6. **64 MB → ~sub-10 MB SDF uploads** — SDF for far pages comes from VLT LOD
   blocks (16³ floats each), uploaded per-page on alloc rather than one 16.7 M
   float blob; the dense `sdf_storage_buf` path is deleted in P5.
7. **godot_voxel absent / wrong version** — node falls back to no-terrain mode
   (free-run particles, all pages LOD0 forced) with a one-time warning; never a
   hard error path.

---

## 8. What does NOT change

- `RenderingDevice` ownership, pipeline creation, hot reload of
  `clear/physics/sortkey` shaders, deferred-free queue, composite rendering,
  camera UBO, `FluidSource`/`FluidSink` public API, inspector layout style.
- The `ChunkCell` struct (momentum pool + CAS exchange semantics) — it just
  moves from a dense grid into arena pages.
