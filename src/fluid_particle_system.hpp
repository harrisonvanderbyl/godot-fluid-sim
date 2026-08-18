#pragma once
#include <godot_cpp/classes/node3d.hpp>
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

#include <vector>

namespace godot {

// ──────────────────────────────────────────────────────────────────────────────
// Mirrors the GPU-side ChunkCell struct (std430, 32 bytes)
// ──────────────────────────────────────────────────────────────────────────────
struct GPUChunkCell {
    int32_t  occupant;
    uint32_t vel_x_bits, vel_y_bits, vel_z_bits, vel_w_bits;
    uint32_t _pad[3];
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
    // Composite material resource. If null, an internal material built from
    // fluid_composite.gdshader is used. If set in the inspector with an empty
    // shader, the default composite code is copied in so the user can edit code
    // + uniforms per-instance. Edits reflect instantly.
    void   set_render_material(const Ref<ShaderMaterial> &v);
    Ref<ShaderMaterial> get_render_material() const;

    // Composite shader uniforms (Inspector-editable)
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

    // Grid bounding box, used by the editor gizmo (plugin.gd)
    AABB  get_grid_aabb()  const {
        return AABB(Vector3(0,0,0), Vector3((float)grid_width,(float)grid_height,(float)grid_depth));
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

private:
    // ── shader paths (Inspector-editable) ────────────────────────────────
    String clear_shader_path   = "res://addons/fluid_particles/shaders/clear_grid.glsl";
    String physics_shader_path = "res://addons/fluid_particles/shaders/velocity_spread.glsl";
    String sortkey_shader_path = "res://addons/fluid_particles/shaders/depth_sort_key.glsl";
    // User-facing ShaderMaterial property. null → use composite_material.
    Ref<ShaderMaterial> render_material;

    // Main camera resolved by ObjectID so a freed camera can never be
    // dereferenced — the ID goes stale and we re-resolve.
    ObjectID main_camera_id;

    Camera3D          *_resolve_main_camera();
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

    // ── simulation toggle ───────────────────────────────────────────────────
    bool  simulation_active = true;

    // ── initial chunk fill ─────────────────────────────────────────────────
    bool      use_initial_chunk       = true;
    Vector3   initial_chunk_origin    = Vector3(2, 2, 2);
    Vector3i  initial_chunk_size      = Vector3i(64, 64, 64);
    Color     initial_chunk_color     = Color(1.0f, 1.0f, 1.0f, 1.0f);  // solid white
    float     initial_chunk_attraction = 1.0f;

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
    RID runnable_buf;
    RID sort_key_buf;

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
    // Shader RIDs (needed for uniform_set_create)
    RID clear_shader;
    RID physics_shader;
    RID sortkey_shader;

    // Uniform sets (one per pipeline)
    RID clear_uniform_set;
    RID physics_uniform_set;
    RID sortkey_uniform_set;

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

    // Sync pending GPU work on the local device (no-op if nothing submitted).
    void _sync_rd();
    // Reload a single compute shader by index: 0=clear, 1=physics, 2=sortkey.
    // Syncs the GPU, compiles the new shader, tears down the old pipeline/
    // uniform_set/shader (in that order), and rebuilds them. On compile failure
    // the old shader is kept.
    void _reload_compute_shader(int which);
    void _dispatch_clear_grid();
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