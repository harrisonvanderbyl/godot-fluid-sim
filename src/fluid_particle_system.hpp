#pragma once
#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/classes/standard_material3d.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/classes/shader_material.hpp>
#include <godot_cpp/classes/shader.hpp>
#include <godot_cpp/classes/camera3d.hpp>
#include <godot_cpp/classes/texture2drd.hpp>
#include <godot_cpp/classes/scene_tree.hpp>
#include <godot_cpp/classes/window.hpp>
#include <godot_cpp/classes/viewport.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>
#include <godot_cpp/templates/hash_map.hpp>

#include <vector>
#include <unordered_set>

#include "lod_arena.hpp"

namespace godot {

// ──────────────────────────────────────────────────────────────────────────────
// Mirrors the GPU-side ChunkCell struct (std430, 32 bytes)
// ──────────────────────────────────────────────────────────────────────────────
struct GPUChunkCell {
    int32_t  occupant;
    uint32_t vel_x_bits, vel_y_bits, vel_z_bits, sdf_bits;
    int32_t  count;        // real occupancy (matches ArenaCell.count in GLSL)
    uint32_t _pad[2];
};  // 32 bytes

// ──────────────────────────────────────────────────────────────────────────────
// FluidParticleSystem — main node
//
// GPU resource lifetime, in one place:
//
//   _ensure_device()          creates the local RenderingDevice ONCE and keeps
//                             it across rebuilds. create_local_rendering_device
//                             transfers ownership to us, so it is memdelete'd
//                             exactly once, in _shutdown_gpu().
//
//   _build_gpu_resources()    simulation buffers + compute pipelines. Bumps
//                             gpu_generation so child sources know their cached
//                             RIDs are stale.
//
//   _ensure_rd_render_pipeline()  render shader, vertex array, render targets,
//                             camera UBO, pipeline, composite node. Idempotent
//                             and self-cleaning on every failure path.
//
//   _destroy_gpu_resources()  releases everything but KEEPS the device. This is
//                             what a rebuild uses.
//
//   _shutdown_gpu()           _destroy_gpu_resources + drain the deferred queue
//                             + destroy the device. Idempotent; called from
//                             _exit_tree, NOTIFICATION_PREDELETE and ~dtor.
// ──────────────────────────────────────────────────────────────────────────────
class FluidParticleSystem : public Node3D {
    GDCLASS(FluidParticleSystem, Node3D)

public:
    FluidParticleSystem();
    ~FluidParticleSystem();

    // Godot lifecycle
    void _enter_tree() override;
    void _process(double delta) override;
    void _exit_tree() override;

    // Scripting interface (Inspector-visible properties)
    void set_grid_width(int v);     int get_grid_width()     const { return grid_width; }
    void set_grid_height(int v);    int get_grid_height()    const { return grid_height; }
    void set_grid_depth(int v);     int get_grid_depth()     const { return grid_depth; }
    void set_grid_size(Vector3i v); Vector3i get_grid_size() const { return Vector3i(grid_width, grid_height, grid_depth); }
    void set_num_particles(int v);  int get_num_particles()  const { return num_particles; }

    // Rebuild all GPU resources (buffers, pipelines, render targets) after a
    // property change that affects buffer sizes. No-op if not in the tree.
    void set_gravity(Vector3 v)    { gravity_vec = v; }
    Vector3 get_gravity()           const { return gravity_vec; }
    void set_gravity_local(bool v)  { gravity_local = v; }
    bool get_gravity_local()        const { return gravity_local; }
    void set_surface_tension(float v) { surface_tension = v; }
    float get_surface_tension()     const { return surface_tension; }
    void set_water_viscosity(float v) { water_viscosity = v; }
    float get_water_viscosity()     const { return water_viscosity; }
    void set_attraction_force(float v) { attraction_force = v; }
    float get_attraction_force()    const { return attraction_force; }
    void set_neighbor_mode(int v)   { neighbor_mode = (v >= 15 ? 15 : 6); }
    int  get_neighbor_mode()        const { return neighbor_mode; }
    void set_max_occupancy(int v)   { max_occupancy = v; }
    int  get_max_occupancy()        const { return max_occupancy; }
    void set_back_pressure(float v) { back_pressure = v; }
    float get_back_pressure()       const { return back_pressure; }

    // Simulation toggle
    void set_simulation_active(bool v) { simulation_active = v; }
    bool get_simulation_active()   const { return simulation_active; }

    // ── SDF terrain collision ─────────────────────────────────────────────
    // Connect a VoxelBuffer (from the godot_voxel addon) whose CHANNEL_SDF
    // holds a signed distance field. When apply_sdf_to_velocity_field() is
    // called (or reset_grid() while a buffer is set), the clear_grid shader
    // bakes the SDF gradient (surface normal) into the persistent velocity
    // field of the chunk grid. The physics shader then reads that velocity
    // each frame, pushing particles out of the terrain — a collision-like
    // reaction with no per-frame CPU work.
    // Re-dispatch clear_grid, baking the SDF normals into the velocity field.
    // No-op if no SDF buffer is connected or the GPU is not ready.
    void   apply_sdf_to_velocity_field();
    // Re-reads the connected VoxelBuffer's SDF channel (picks up terrain
    // edits), uploads it to the GPU, then clears the grid and bakes the SDF
    // normals into the velocity field. Exposed as an inspector button.
    void   reload_sdf_and_clear_grid();
    Callable get_reload_sdf_and_clear_grid() const {
        return callable_mp(const_cast<FluidParticleSystem *>(this), &FluidParticleSystem::reload_sdf_and_clear_grid);
    }

    // Initial chunk fill
    void set_use_initial_chunk(bool v) { use_initial_chunk = v; }
    bool get_use_initial_chunk()  const { return use_initial_chunk; }
    void set_initial_chunk_origin(Vector3 v) { initial_chunk_origin = v; }
    Vector3 get_initial_chunk_origin() const { return initial_chunk_origin; }
    void set_initial_chunk_size(Vector3i v) { initial_chunk_size = v; }
    Vector3i get_initial_chunk_size() const { return initial_chunk_size; }
    void set_initial_chunk_color(Color v) { initial_chunk_color = v; }
    Color get_initial_chunk_color() const { return initial_chunk_color; }
    void set_initial_chunk_attraction(float v) { initial_chunk_attraction = v; }
    float get_initial_chunk_attraction() const { return initial_chunk_attraction; }

    // Shader paths (Inspector-editable, default to addon). Setting a new path
    // hot-reloads the corresponding compute shader if the node is in the tree.
    void   set_clear_shader_path(const String &v);
    String get_clear_shader_path()  const { return clear_shader_path; }
    void   set_physics_shader_path(const String &v);
    String get_physics_shader_path() const { return physics_shader_path; }
    void   set_sortkey_shader_path(const String &v);
    String get_sortkey_shader_path() const { return sortkey_shader_path; }
    void   set_migrate_shader_path(const String &v);
    String get_migrate_shader_path() const { return migrate_shader_path; }
    // Composite material resource. If null, an internal material built from
    // fluid_composite.gdshader is used. If set in the inspector with an empty
    // shader, the default composite code is copied in so the user can edit code
    // + uniforms per-instance. Edits reflect instantly.
    void   set_render_material(const Ref<ShaderMaterial> &v);
    Ref<ShaderMaterial> get_render_material() const;

    // Composite shader uniforms (Inspector-editable)
    void set_fluid_particle_size(float v);
    float get_fluid_particle_size() const { return fluid_particle_size; }
    void set_smooth_radius(float v);
    float get_smooth_radius() const { return smooth_radius; }
    void set_tint_strength(float v);
    float get_tint_strength() const { return tint_strength; }
    void set_refraction_strength(float v);
    float get_refraction_strength() const { return refraction_strength; }
    void set_absorption_dist(float v);
    float get_absorption_dist() const { return absorption_dist; }
    void set_fluid_tint(Color v);
    Color get_fluid_tint() const { return fluid_tint; }
    void set_smooth_falloff(float v);
    float get_smooth_falloff() const { return smooth_falloff; }
    void set_coverage_threshold(float v);
    float get_coverage_threshold() const { return coverage_threshold; }

    // Spawn helpers callable from GDScript
    void spawn_block(Vector3 origin, int w, int h, int d, Color color, float attraction);
    void add_velocity_impulse(Vector3 impulse);
    void add_rotational_impulse(Vector3 center, Vector3 axis_amount);
    void reset_grid();  // explicitly clear chunk grid (occupants + velocities)

    // Reload the physics compute shader from disk (recompiles velocity_spread.glsl
    // and rebuilds the pipeline + uniform set). Safe to call at runtime; if the
    // new shader fails to compile, the old one is kept. No-op if not in the tree.
    void reload_physics_shader();

    Callable get_reload_physics_shader() const {
        return callable_mp(const_cast<FluidParticleSystem *>(this), &FluidParticleSystem::reload_physics_shader);
    }

    // Reload the render (vertex+fragment) shader from disk and rebuild the
    // render pipeline + uniform set. Use this after editing fluid_depth.glsl,
    // including changes that alter the pipeline. On compile failure the current
    // shader is kept and rendering continues.
    void reload_render_shader();

    Callable get_reload_render_shader() const {
        return callable_mp(const_cast<FluidParticleSystem *>(this), &FluidParticleSystem::reload_render_shader);
    }

    // Grid bounding box, used by the editor gizmo (plugin.gd). Includes the
    // camera-window origin so the gizmo tracks the moving simulation volume.
    AABB  get_grid_aabb()  const {
        return AABB(Vector3(grid_window_origin), Vector3((float)grid_width,(float)grid_height,(float)grid_depth));
    }

    // ── Access for child source/sink nodes ────────────────────────────────
    // These expose our vertex/attribute RD storage buffers (position and
    // color/custom0, respectively) plus the chunk grid, so FluidSource /
    // FluidSink can dispatch their own compute shaders directly against the
    // same buffers the renderer reads — no CPU round-trip.
    //
    // is_gpu_ready() and get_gpu_generation() let a child detect that we tore
    // down and rebuilt underneath it. Any child holding RIDs built against an
    // older generation must drop them before dispatching again.
    RenderingDevice *get_rd()             const { return rd; }
    bool             is_gpu_ready()       const { return gpu_ready; }
    uint64_t         get_gpu_generation() const { return gpu_generation; }
    RID              get_vertex_buf()     const { return vertex_buf; }
    RID              get_attribute_buf()  const { return attrib_buf; }
    RID              get_chunk_buf()      const { return chunk_buf; }
    RID              get_render_vertex_buf() const { return render_vertex_buf; }
    // P5: LOD buffers exposed for child shaders (source/sink page tests).
    RID              get_cell_arena_buf() const { return cell_arena_buf; }
    RID              get_lod_table_buf()  const { return lod_table_buf; }
    int              get_lod_levels()     const { return lod_levels; }
    int              get_grid_w()         const { return grid_width; }
    int              get_grid_h()         const { return grid_height; }
    int              get_grid_d()         const { return grid_depth; }
    int              get_vertex_stride_floats() const { return vertex_stride_floats; }
    int              get_attrib_stride_words()  const { return attrib_stride_words; }
    int              get_color_offset_words()   const { return color_offset_words; }
    int              get_custom0_offset_words() const { return custom0_offset_words; }

protected:
    static void _bind_methods();

    // Fires on NOTIFICATION_PREDELETE whether or not the node ever entered the
    // tree, and whether or not _exit_tree ran. No `override` — godot-cpp's
    // GDCLASS detects this by name, it is not a virtual on Object.
    void _notification(int p_what);

    // Dynamic property forwarding for the child VoxelTerrain. Exposes all
    // VoxelTerrain-specific properties on this node under a "Terrain" group
    // with a "terrain_" prefix, so they can be edited in the inspector without
    // selecting the internal child node. Values are cached so they survive
    // child recreation and are applied when the child is created.
    bool _set(const StringName &p_name, const Variant &p_value);
    bool _get(const StringName &p_name, Variant &r_ret) const;

private:
    // ── shader paths (Inspector-editable) ────────────────────────────────
    String clear_shader_path   = "res://addons/fluid_particles/shaders/clear_grid.glsl";    String physics_shader_path = "res://addons/fluid_particles/shaders/velocity_spread.glsl";
    String sortkey_shader_path = "res://addons/fluid_particles/shaders/depth_sort_key.glsl";
    String migrate_shader_path = "res://addons/fluid_particles/shaders/lod_migrate.glsl";
    // User-facing ShaderMaterial property. null → use composite_material.
    Ref<ShaderMaterial> render_material;

    // Main camera resolved by ObjectID so a freed camera can never be
    // dereferenced — the ID goes stale and we re-resolve.
    ObjectID main_camera_id;

    Camera3D          *_resolve_main_camera();
    // World-space anchor the sim window follows: a VoxelViewer (drives godot_voxel
    // block streaming) when one exists, else the main camera.
    Node3D            *_resolve_follow_anchor();
    ObjectID           follow_anchor_id;   // cache (avoid full-scene scan per frame)
    Ref<ShaderMaterial> _active_composite_material() const;
    // Push every node-owned composite uniform onto a material. One place to
    // edit when a uniform is added.
    void                _push_composite_uniforms(const Ref<ShaderMaterial> &mat) const;

    // ── simulation parameters ─────────────────────────────────────────────
    int   grid_width      = 256;    // awidth
    int   grid_height     = 256;    // aheight
    int   grid_depth      = 256;    // adepth
    int   num_particles   = 262144;
    Vector3 gravity_vec   = Vector3(0, 0.1f, 0);  // gravity vector (world or local)
    bool    gravity_local = false;   // if true, transformed by node basis into grid space
    float surface_tension = 0.0f;   // surfaceTension
    float water_viscosity = 0.0f;   // waterviscosity
    float attraction_force = 0.1f;  // global multiplier on per-particle attraction/repulsion
    int   neighbor_mode   = 15;      // 6+center (reference uses 7)
    int   max_occupancy   = 1;       // max liquid particles per grid cell
    float back_pressure   = 0.0f;    // outward bias as a cell fills
    float fluid_particle_size = 0.1f;  // composite shader uniform

    // ── simulation toggle ───────────────────────────────────────────────────
    bool  simulation_active = true;

    // ── initial chunk fill ─────────────────────────────────────────────────
    bool      use_initial_chunk       = true;
    Vector3   initial_chunk_origin    = Vector3(2, 2, 2);
    Vector3i  initial_chunk_size      = Vector3i(64, 64, 64);
    Color     initial_chunk_color     = Color(1.0f, 1.0f, 1.0f, 1.0f);  // solid white
    float     initial_chunk_attraction = 1.0f;

    // ── SDF terrain collision ─────────────────────────────────────────────
    // VoxelBuffer reference (Variant because the godot_voxel addon is a
    // separate GDExtension whose headers we don't link against — methods are
    // called through Object::call). CHANNEL_SDF is read once, decoded to float
    // and uploaded to sdf_storage_buf. The clear_grid shader samples it to
    // bake the SDF gradient into the persistent chunk velocity field.
    Variant sdf_buffer_var;
    // Terrain child supplying the SDF data. P1 swaps this to VoxelLodTerrain
    // (forced to a single LOD so output matches the old VoxelTerrain setup);
    // a legacy VoxelTerrain is still accepted/created as a fallback. The node
    // is godot_voxel (separate GDExtension), so all access goes through
    // Object::call / Object::set / Object::get.
    ObjectID voxel_terrain_child_id;
    bool terrain_is_vlt = false;  // child is a VoxelLodTerrain (not legacy VoxelTerrain)
    bool terrain_no_signal_warned = false;  // one-time "no block signals" memo
    float sdf_strength = 1.0f;              // velocity push magnitude (unused now, kept for future)
    // SDF storage buffer: float[grid_w * grid_h * grid_d], allocated once at
    // grid size in _build_gpu_resources and never reallocated. _upload_sdf_data
    // just buffer_update's into it. clear_grid.glsl reads it directly at binding 5.
    RID     sdf_storage_buf;
    int     sdf_w = 0, sdf_h = 0, sdf_d = 0;  // SDF buffer dimensions (for push constants)

    // Cache of terrain property values set before the child VoxelTerrain
    // exists (e.g. in the editor before _enter_tree). Applied to the child
    // when it is created in _refresh_sdf_from_child.
    HashMap<StringName, Variant> _terrain_property_cache;
    Object *_get_voxel_terrain_child() const;

    // Set by the block_loaded / mesh_block_entered signal callback. _process
    // checks it and runs a debounced SDF recopy (re-extract + upload +
    // clear_grid) so a burst of block loads doesn't trigger N recopies.
    bool sdf_refresh_pending = false;
    // Cached arity of the terrain block signals (1 or 2 args). Probed from
    // get_signal_list() at connect time; -1 = not probed yet / not available.
    int  terrain_signal_arity = -1;
    // Throttled terrain event log (land/unload spam would flood the output).
    uint64_t terrain_event_log_last_ms = 0;
    uint32_t terrain_event_log_count = 0;
    void _on_terrain_block_loaded(const Variant &a, const Variant &b, const Variant &c, const Variant &d);
    void _on_terrain_block_unloaded(const Variant &a, const Variant &b, const Variant &c, const Variant &d);
    // Debug hook: inject a block event via the same path as real signals.
    void debug_simulate_block_event(Vector3i position, int lod, bool loaded);
    void _on_terrain_block_event(const Variant &p_position, const Variant &p_lod, bool p_entered);

    // ── RenderingDevice ─────────────────────────────────────────────────────
    // A LOCAL RenderingDevice (created via create_local_rendering_device) that
    // we fully control. All compute + render pipelines run on this device, so
    // we can call submit()/sync() without conflicting with Godot's render
    // thread. The output color+depth textures are shared to the shared device
    // (via texture_create_from_extension) so the composite shader can sample them.
    //
    // OWNERSHIP: create_local_rendering_device() hands the device to us. It is
    // created once by _ensure_device() and destroyed once by _shutdown_gpu().
    // A rebuild must NOT churn it.
    RenderingDevice *rd = nullptr;          // local device (compute + render)
    RenderingDevice *shared_rd = nullptr;   // shared device (for compositor textures)

    // GPU buffers (RIDs on the local device). Standalone storage buffers —
    // not tied to any mesh.
    RID vertex_buf;
    RID attrib_buf;
    RID chunk_buf;
    RID render_vertex_buf;  // interpolated display positions (render reads this)
    RID runnable_buf;
    RID sort_key_buf;

    // ── LOD arena resources (P2) ──────────────────────────────────────────
    // Paged counterpart of the dense chunk grid. The simulation still runs on
    // chunk_buf in P2; these exist and are maintained purely from terrain
    // signals so P3/P4 can switch the shaders over. See docs/LOD_UPGRADE_PLAN.md
    // §1 and src/lod_arena.hpp.
    //
    // LIFECYCLE: pages are allocated/freed ONLY inside _apply_pending_lod_events,
    // driven exclusively by _on_terrain_block_event — no polling, no per-frame
    // reconciliation.
    RID cell_arena_buf;   // max_pages * 4096 * 32B ChunkCell slots
    RID lod_table_buf;    // per-LOD page table, 16B LodPageEntry per coord
    RID store_ring_buf;   // u32 ring of particle indices + 2-word header
    RID lod_stats_buf;    // 32 ints RSS/debug stats, zeroed per frame in P4+
    VoxelLodArena lod_arena;               // CPU freelist allocator
    std::vector<LodPageEntry> lod_table_cpu; // CPU mirror of lod_table_buf
    // Tracks which pages have had SDF baked from the terrain. Pages allocated
    // start without SDF (far_outside); the poller bakes
    // SDF into them lazily as the terrain streams data.
    std::unordered_set<int64_t> sdf_baked_pages;
    int32_t lod_levels     = 4;   // matches default VoxelLodTerrain lod_count
    // Arena capacity. Must cover ALL LODs, not just LOD-0: for a 256³ grid
    // that's 4096 (L0) + 512 (L1) + 64 (L2) + 8 (L3) = 4680. The old 4096
    // left zero headroom, so once the poller filled L0 every coarser page
    // alloc failed. Kept as a property in case the user wants to limit it
    // deliberately (memory: pages × 4096 cells × 32 B = 128 KB/page).
    int32_t max_arena_pages = 8192;
    void  set_max_arena_pages(int v) {
        if (max_arena_pages == v) return;
        max_arena_pages = v > 64 ? v : 64;
        // Only the LOD arena + page table need rebuilding — particle buffers,
        // pipelines, and uniform sets are unaffected by arena page count.
        if (is_inside_tree() && gpu_ready) {
            _destroy_lod_resources();
            _build_lod_resources();
            _rebuild_compute_uniform_sets();
        }
    }
    int   get_max_arena_pages() const { return max_arena_pages; }

    // P5: recolor particles by sim LOD (blue/green/orange/red/magenta),
    // frozen = dark grey. Purely visual; one extra attribute write per
    // particle in the physics epilogue.
    bool    debug_tint_by_lod = false;
    void    set_debug_tint_by_lod(bool v) { debug_tint_by_lod = v; }
    bool    get_debug_tint_by_lod() const { return debug_tint_by_lod; }
    // P5: {pages_by_lod, particles_by_lod, skipped, arena_free_slots}.
    // Syncs the device + small readback of the stats buffer.
    Dictionary get_lod_stats();

    // P7: persistence of frozen (out-of-page) particles.
    bool    spill_to_disk = true;          // persist drained spill particles
    void    set_spill_to_disk(bool v)      { spill_to_disk = v; }
    bool    get_spill_to_disk() const      { return spill_to_disk; }
    int64_t get_spilled_total() const      { return spilled_total; }
    int     get_spill_batch_per_frame() const { return spill_batch_per_frame; }
    void    set_spill_batch_per_frame(int v)  { spill_batch_per_frame = v > 0 ? v : 1; }
    // P5.5 accessors.
    void  set_poll_terrain_blocks(bool v)     { poll_terrain_blocks = v; }
    bool  get_poll_terrain_blocks() const     { return poll_terrain_blocks; }
    void  set_poll_interval_seconds(float v)  { poll_interval_seconds = v < 0.05f ? 0.05f : v; }
    float get_poll_interval_seconds() const   { return poll_interval_seconds; }
    // Debug overlay: a child Node3D holding one transparent BoxMesh per loaded
    // page, colored by LOD (same palette as the particle debug tint). Meshes
    // are pooled — the overlay rebuilds when the poller's page set changes,
    // not per frame.
    void    _update_page_debug_overlay();
    void    _destroy_page_debug_overlay();
    bool    show_page_boxes = false;
    void    set_show_page_boxes(bool v) {
        if (show_page_boxes == v) return;
        show_page_boxes = v;
        if (is_inside_tree()) {
            if (v) _update_page_debug_overlay();
            else   _destroy_page_debug_overlay();
        }
    }
    bool    get_show_page_boxes() const { return show_page_boxes; }

    // ── Infinite world: camera-following sim window ─────────────────────
    // The GPU buffers are a finite 256³ box, but the box doesn't have to sit
    // at the node origin. grid_window_origin is the world-space (terrain
    // voxel-space) origin of the simulated window, snapped to the coarsest
    // page size (16 << (lod_levels-1)) so every LOD page table stays 1:1
    // aligned with terrain blocks. When follow is on and the camera drifts
    // inside margin_ratio of the window edge, the window shifts in world
    // space: particle positions are rebased (world-fixed), the model matrix
    // picks up the offset so rendering stays glued to the world, and the SDF
    // is re-extracted at the new window.
    bool     grid_window_follow  = false;
    float    grid_window_margin  = 0.25f;   // fraction of window size
    Vector3i grid_window_origin;            // voxels; 0 = classic fixed box
    void    set_grid_window_follow(bool v) { grid_window_follow = v; }
    bool    get_grid_window_follow() const { return grid_window_follow; }
    void    set_grid_window_margin(float v){ grid_window_margin = v < 0.05f ? 0.05f : (v > 0.45f ? 0.45f : v); }
    float   get_grid_window_margin() const { return grid_window_margin; }
    // Snap step and per-frame check + the shift itself (rebases positions).
    Vector3i _window_snap_step() const {
        return Vector3i(1, 1, 1) * (LOD_PAGE_SIZE << (lod_levels > 0 ? lod_levels - 1 : 0));
    }
    void     _shift_grid_window(Vector3i new_origin);
    void     _update_grid_window_follow();
    // P6: optional runnable-segment dispatch. When on, every ~16 frames the
    // active particles are sorted by sim LOD and physics is dispatched as one
    // group-range per segment, cutting wasted lanes on far particles.
    bool    segment_dispatch = false;
    void    set_segment_dispatch(bool v)     { segment_dispatch = v; }
    bool    get_segment_dispatch() const     { return segment_dispatch; }
    void    _rebuild_runnable_segments();
    int     segment_rebuild_frame = -1;
    struct LodSegment { int32_t begin; int32_t count; int32_t lod; };
    std::vector<LodSegment> lod_segments;

    // P7: re-activate spilled particles through the normal source path.
    int  restore_spilled_particles(int max_count);
    // P5.5: signal-free terrain block polling (VLT emits no block signals in
    // this godot_voxel build). Polls debug_get_mesh_block_info and mirrors
    // transitions into the normal LodEvent pipeline.
    void _poll_terrain_block_states();
    bool     poll_terrain_blocks = true;
    float    poll_interval_seconds = 0.5;
    uint64_t last_block_poll_ms = 0;
    std::vector<int32_t> lod_table_base;   // flat index of each LOD's region
    bool lod_buffers_valid = false;

    // Terrain signal events awaiting application at the next _process.
    struct LodEvent { Vector3i page_pos; int32_t lod; bool loaded; };
    std::vector<LodEvent> pending_lod_events;

    // CPU page-table geometry / accessors (shared by alloc + free paths).
    Vector3i _lod_table_dims(int lod) const;
    int64_t _lod_table_base(int lod) const;
    int64_t _lod_table_index(int lod, Vector3i page_pos) const;
    // Push one (16 B) entry of the CPU mirror to the GPU buffer.
    void _lod_table_upload(int64_t index);
    // Apply a single terrain event to the arena + page table (alloc/free +
    // zero-fill of the page's cells on alloc).
    void _apply_lod_event(const LodEvent &ev);
    // Drain pending_lod_events (must be called with RD synced, before dispatch).
    void _apply_pending_lod_events();
    // Bake SDF data from the terrain into a page's arena cells at alloc time.
    // Queries the terrain at the page's LOD and writes sdf_bits per cell.
    void _bake_sdf_for_page(int lod, Vector3i page_pos, uint32_t cell_base);

    // ── Phase 3: LOD migration (signal-driven, like alloc/free) ─────────────
    // One migrate job per destination page. The GPU work itself is dispatched
    // in _dispatch_lod_migrations at the END of _apply_pending_lod_events, so
    // all buffer_updates (zero-fill, table entries) and all migrations land in
    // the same submit/sync boundary.
    struct MigrateJob {
        uint32_t dst_base;      // arena cell offset of destination page
        uint32_t src_base[8];   // child (coarsen) or parent (refine) pages
        uint32_t mode;          // 0 = coarsen, 1 = refine
        uint32_t dst_octant;    // refine: which child page (0..7)
    };
    std::vector<MigrateJob> pending_migrate_jobs;
    // Slots detached from the arena whose GPU data a pending job still reads.
    // Released back to the freelist AFTER the migration dispatch, so slot
    // reuse cannot clobber a source page mid-batch.
    std::vector<uint32_t> detached_slots;
    // Detect coarsen/refine opportunities across the JUST-DRAINED event batch:
    // coarsen = load at k+1 while 8 k pages detached intact → migrate k→k+1;
    // refine = 8 loads at k while k+1 page detached → migrate k+1→k.
    void _detect_batch_lod_transitions();
    // Record all pending_migrate_jobs into the RD compute list.
    void _dispatch_lod_migrations();
    // Phase 4: drain the store ring — read skipped particle indices, log,
    // reset the cursor. Particles freeze (render-visible, no physics).
    void _drain_store_ring();
    // P7: persist drained spill particles to user:// and deactivate them on
    // the GPU (NaN position). Batches writes; capped per frame.
    void _spill_particles(const PackedInt32Array &indices);
    // P7: write a spill file for a FluidSource to pick up later.
    String _spill_dir() const;
    // P7: re-activate spilled particles through the normal source path.
    // Stats / logging state for the store-ring drain.
    uint64_t last_skip_log_ms = 0;
    uint32_t skipped_particle_count = 0;
    uint32_t last_ring_overflow = 0;
    uint32_t ring_pending_unreported = 0;
    // P7 spill state.
    int64_t spilled_total = 0;
    int     spill_batch_per_frame = 1024;   // GPU deactivate writes per frame
    // P7: in-memory mirror of spilled records for AUTO-restore. When the
    // block poller loads a page, any queued record whose position falls in
    // that page's voxel range is re-activated (position written back, custom0
    // cleared) and removed from the queue. Disk file remains the long-term
    // backup for restarts.
    struct SpillRec { int32_t idx; float x, y, z; };
    std::vector<SpillRec> spill_queue;
    void _auto_restore_for_page(int lod, Vector3i page_pos);

    // Stride/offset (in 4-byte words) describing how particle data is packed
    // inside vertex_buf/attrib_buf. Fixed layout; _build_gpu_resources reasserts
    // these, but the defaults must already be valid because _create_vertex_array
    // reads them.
    int vertex_stride_floats  = 3;   // vec3 position
    int attrib_stride_words   = 8;   // vec4 color + vec4 custom0
    int color_offset_words    = 0;
    int custom0_offset_words  = 4;

    // Compute pipelines
    RID clear_pipeline;
    RID physics_pipeline;
    RID sortkey_pipeline;
    RID migrate_pipeline;
    // Shader RIDs (needed for uniform_set_create)
    RID clear_shader;
    RID physics_shader;
    RID sortkey_shader;
    RID migrate_shader;

    // Uniform sets (one per pipeline)
    RID clear_uniform_set;
    RID physics_uniform_set;
    RID sortkey_uniform_set;
    RID migrate_uniform_set;

    // ── runtime state ──────────────────────────────────────────────────────
    uint64_t  frame_count  = 0;
    // Explicit rotational/transform impulses: accumulated as a delta transform.
    // add_velocity_impulse composes a translation; add_rotational_impulse
    // composes a rotation about a center. Both are applied per-particle in the
    // shader via: dp = delta_basis * p + delta_origin - p
    Basis    pending_delta_basis;
    Vector3  pending_delta_origin;
    bool     delta_impulse_pending = false;
    Transform3D prev_global_transform;  // for computing node-movement impulses
    Transform3D prev_delta_transform;   // differencing baseline
    bool      has_prev_delta_transform = false;
    bool      has_prev_transform = false;
    bool      gpu_ready    = false;

    // Incremented by every _build_gpu_resources. Child sources stamp the value
    // they built against and rebuild when it moves, so a grid or particle-count
    // change can never leave them dispatching against freed buffers.
    uint64_t  gpu_generation = 0;

    // ── composite shader uniforms (Inspector-editable) ───────────────────
    float   smooth_radius       = 2.0f;
    float   tint_strength       = 0.6f;
    float   refraction_strength = 0.04f;
    float   absorption_dist     = 4.0f;
    Color   fluid_tint          = Color(0.7f, 0.85f, 1.0f);
    float   smooth_falloff      = 0.5f;
    float   coverage_threshold  = 0.5f;

    // ── RD offscreen render pipeline ─────────────────────────────────────────
    // Particles are rendered directly via a RenderingDevice render pipeline
    // into an RD framebuffer with a color attachment (RGBA8: rgb=color, a=1)
    // and a depth attachment (D32_SFLOAT). The output textures are shared to
    // the shared device and exposed to the composite shader via Texture2DRD.
    //
    // The composite quad (composite_node) is a MeshInstance3D with a spatial
    // shader that samples the color+depth textures and blurs/refracts/tints
    // the screen. Created exactly once by _ensure_composite_node().
    MeshInstance3D  *composite_node     = nullptr;
    // Debug page boxes (per-LOD transparent AABBs) — pooled meshes under a
    // single child node; rebuilt when the allocated-page set changes.
    Node3D          *page_overlay_node  = nullptr;
    std::vector<MeshInstance3D *> page_box_pool;
    std::vector<Ref<StandardMaterial3D>> page_lod_materials;   // one shared mat per LOD
    int              page_overlay_dirty = 0;   // needs-rebuild flag (alloc/free count)
    int              last_overlay_page_count = -1;
    Ref<ShaderMaterial> composite_material;
    Ref<Shader>         composite_shader;     // fluid_composite.gdshader

    // RD render pipeline resources. All on the local device (rd) except the
    // shared-device texture wrappers.
    RID     rd_render_shader;       // shader RID (vertex+fragment SPIR-V)
    RID     rd_render_pipeline;     // render pipeline RID
    RID     rd_camera_ubo;          // uniform buffer for camera matrices
    RID     rd_render_uniform_set;  // uniform set (camera UBO at binding 0)
    RID     rd_color_tex;           // RGBA8 color attachment (local device)
    RID     rd_depth_tex;           // D32_SFLOAT depth attachment (local device)
    RID     rd_framebuffer;         // framebuffer (color + depth, local device)
    int64_t rd_framebuffer_format = -1;  // cached framebuffer format ID
    RID     rd_vertex_array;        // vertex array wrapping local storage buffers
    int64_t rd_vertex_format   = -1;     // cached vertex format ID
    Vector2i rd_vp_size;                // current offscreen texture size

    // Shared-device texture wrappers (so the composite shader can sample them).
    // These alias the VkImages owned by rd_color_tex / rd_depth_tex, so they
    // must always be released BEFORE the textures they alias.
    RID     shared_color_tex;       // shared-device view of rd_color_tex
    RID     shared_depth_tex;       // shared-device view of rd_depth_tex

    // Texture2DRD wrappers so the composite shader can sample the RD textures.
    Ref<Texture2DRD> rd_color_tex_2d;
    Ref<Texture2DRD> rd_depth_tex_2d;

    // Per-frame data prepared on the main thread, consumed by the draw.
    PackedByteArray rd_ubo_bytes;       // camera UBO data (updated each frame)
    PackedByteArray rd_pc_bytes;        // push constant data (updated each frame)
    bool rd_submitted = false;          // true after submit(), cleared by sync()

    // ── Deferred RID destruction ────────────────────────────────────────────
    // Godot's render thread samples our color/depth textures through Texture2DRD
    // and can still be reading them for a few frames after we unbind. Freeing
    // the underlying VkImage inline is a use-after-free — it shows up as a
    // driver crash while the window is being resized. Anything the main
    // renderer can observe is retired through this queue instead.
    struct DeferredFree {
        RID  rid;
        int  frames_left      = 0;
        bool on_shared_device = false;   // free on shared_rd rather than rd
    };
    std::vector<DeferredFree> deferred_frees;

    void _queue_free_rid(RID r, bool on_shared_device);
    void _process_deferred_frees(bool flush_now);
    // Free a local-device RID and null the handle. Safe with an invalid RID or
    // a null device.
    void _free_local_rid(RID &r);

    // ── Device lifetime ─────────────────────────────────────────────────────
    bool _ensure_device();
    void _shutdown_gpu();

    // Tell every FluidSourceBase descendant to drop RIDs built against our
    // buffers. Must run before any buffer teardown. Pass nullptr to start at
    // `this`; the parameter exists for the recursive walk.
    void _notify_children_gpu_reset(Node *from);

    // ── Render resource construction ────────────────────────────────────────
    // Each returns false and leaves nothing behind on failure, so a retry
    // starts clean. _ensure_rd_render_pipeline orchestrates them.
    void _ensure_rd_render_pipeline();
    RID  _compile_render_shader();
    bool _create_vertex_array();
    bool _create_render_uniform_set();
    bool _create_render_targets(Vector2i size);
    void _destroy_render_targets(bool deferred);
    bool _create_render_pipeline_state();
    void _bind_render_targets_to_composite();
    void _ensure_composite_node();
    void _destroy_composite_node();

    void _sync_rd_viewport_size();
    // Records the particle draw. Returns true if commands were recorded.
    // Submission belongs to _process — an early return here must not strand
    // the frame's compute lists unflushed.
    bool _render_particles_rd();
    void _sync_offscreen_camera();
    // Recursively collect all Viewport descendants of `node` into `out`.
    // Used to find the editor's 3D viewport in editor mode.
    static void _collect_viewports(Node *node, TypedArray<Viewport> &out);

    // ── Build / teardown ────────────────────────────────────────────────────
    void _build_gpu_resources();
    void _rebuild_compute_uniform_sets();
    // Release everything but KEEP the device. Idempotent.
    void _destroy_gpu_resources();
    void _destroy_render_resources(bool deferred_targets);
    void _destroy_sim_resources();
    void _rebuild_gpu_resources();

    // Create/destroy the LOD arena + page table + store ring + stats buffers.
    void _build_lod_resources();
    void _destroy_lod_resources();

    // ── SDF terrain collision ─────────────────────────────────────────────
    // Read CHANNEL_SDF from the connected VoxelBuffer, decode it to float
    // (handling 8/16/32-bit depths + quantization scales), and upload to a
    // GPU storage buffer the clear_grid shader samples. No-op if no buffer.
    void _upload_sdf_data();
    // Instantiate a VoxelTerrain child sized to the fluid grid (if the
    // godot_voxel addon is available) and extract its SDF into the chunk
    // buffer's sdf_bits field. Returns true if an SDF buffer is available.
    bool _refresh_sdf_from_child();

    // Sync pending GPU work on the local device (no-op if nothing submitted).
    void _sync_rd();
    // Reload a single compute shader by index: 0=clear, 1=physics, 2=sortkey.
    // Syncs the GPU, compiles the new shader, tears down the old pipeline/
    // uniform_set/shader (in that order), and rebuilds them. On compile failure
    // the old shader is kept.
    void _reload_compute_shader(int which);
    void _dispatch_clear_grid(bool keep_occupant = false);
    void _dispatch_physics(Vector3 global_add_velocity,
                           const Basis &delta_basis,
                           Vector3 delta_origin,
                           bool has_delta);
    void _dispatch_sortkey();

    // Compile a .glsl compute shader resource into an RD shader RID.
    // Uses CACHE_MODE_REPLACE so edits on disk are picked up.
    RID _compile_compute_shader(const String &res_path);
    // Build a single storage-buffer RDUniform (helper for uniform-set creation).
    static Ref<RDUniform> _make_storage_uniform(RID buf, uint32_t binding);
};

}  // namespace godot