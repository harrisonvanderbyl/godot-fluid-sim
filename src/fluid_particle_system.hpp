#pragma once
#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/shader_material.hpp>
#include <godot_cpp/classes/shader.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>

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
    void set_num_particles(int v);  int get_num_particles()  const { return num_particles; }

    // Rebuild all GPU resources (buffers, pipelines, render mesh) after a
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

    // Shader paths (Inspector-editable, default to addon)
    void   set_clear_shader_path(const String &v)  { clear_shader_path = v; }
    String get_clear_shader_path()  const { return clear_shader_path; }
    void   set_physics_shader_path(const String &v) { physics_shader_path = v; }
    String get_physics_shader_path() const { return physics_shader_path; }
    void   set_sortkey_shader_path(const String &v) { sortkey_shader_path = v; }
    String get_sortkey_shader_path() const { return sortkey_shader_path; }
    // Render material resource. If null, uses the addon's particle_render.gdshader
    // internally (not editable). If set in the inspector with an empty shader, the
    // default particle render code is copied in so the user can edit code + uniforms
    // per-instance. Edits to the shader or uniform values reflect instantly.
    void   set_render_material(const Ref<ShaderMaterial> &v);
    Ref<ShaderMaterial> get_render_material() const;

    // Spawn helpers callable from GDScript
    void spawn_block(Vector3 origin, int w, int h, int d, Color color, float attraction);
    void add_velocity_impulse(Vector3 impulse);
    void add_rotational_impulse(Vector3 center, Vector3 axis_amount);
    void reset_grid();  // explicitly clear chunk grid (occupants + velocities)

    // Grid bounding box, used by the editor gizmo (plugin.gd)
    AABB  get_grid_aabb()  const {
        return AABB(Vector3(0,0,0), Vector3((float)grid_width,(float)grid_height,(float)grid_depth));
    }

    // ── Access for child source/sink nodes ────────────────────────────────
    // These expose the render mesh's own vertex/attribute RD storage buffers
    // (bindings for position and color/custom0, respectively) plus the chunk
    // grid, so FluidSource / FluidSink can dispatch their own compute shaders
    // directly against the same buffers the renderer reads — no CPU round-trip.
    RenderingDevice *get_rd()             const { return rd; }
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

private:
    // ── shader paths (Inspector-editable) ────────────────────────────────
    String clear_shader_path   = "res://addons/fluid_particles/shaders/clear_grid.glsl";
    String physics_shader_path = "res://addons/fluid_particles/shaders/velocity_spread.glsl";
    String sortkey_shader_path = "res://addons/fluid_particles/shaders/depth_sort_key.glsl";
    // User-facing ShaderMaterial property. null → use internal_material (default).
    Ref<ShaderMaterial> render_material;
    // Internal default material, created when render_material is null.
    Ref<ShaderMaterial> internal_material;

    // ── simulation parameters (matching particles.cpp reference) ──────────
    int   grid_width      = 256;    // awidth
    int   grid_height     = 256;    // aheight
    int   grid_depth      = 256;    // adepth
    int   num_particles   = 262144; // 100*100*100
    Vector3 gravity_vec   = Vector3(0, 0.1f, 0);  // gravity vector (world or local)
    bool    gravity_local = true;   // if true, transformed by node basis into grid space
    float surface_tension = 1.1f;   // surfaceTension
    float water_viscosity = 1.0f;   // waterviscosity
    float attraction_force = 1.0f;  // global multiplier on per-particle attraction/repulsion
    int   neighbor_mode   = 7;      // 6+center (reference uses 7)

    // ── simulation toggle ───────────────────────────────────────────────────
    bool  simulation_active = true;

    // ── initial chunk fill (matching particles.cpp reference spawn) ────────
    bool      use_initial_chunk       = true;
    Vector3   initial_chunk_origin    = Vector3(2, 2, 2);
    Vector3i  initial_chunk_size      = Vector3i(64, 64, 64);
    Color     initial_chunk_color     = Color(1.0f, 1.0f, 1.0f, 1.0f);  // solid white
    float     initial_chunk_attraction = 0.0f;  // solid particles have 0 attraction

    // ── rendering ────────────────────────────────────────────────
    // A normal MeshInstance3D (render_node) rebuilds its ArrayMesh every frame
    // from a CPU readback of particle_buf, shaded by particle_render.gdshader
    // (real ray-sphere/liquid rendering). See _update_render_mesh().

    // ── RenderingDevice compute pipeline ────────────────────────
    // Shared RenderingDevice (RenderingServer's main device) — required so
    // buffer RIDs fetched via mesh_surface_get_vertex_buffer_rd_rid()/
    // mesh_surface_get_attribute_buffer_rd_rid() can be bound directly in our
    // own compute pipelines (RIDs are not portable across separate
    // RenderingDevice instances, so a local device would not work here).
    RenderingDevice *rd = nullptr;

    // GPU buffers (RIDs). Position/color/custom0 live inside the render
    // mesh's own vertex/attribute storage buffers (see _ensure_render_node).
    RID vertex_buf;
    RID attrib_buf;
    RID chunk_buf;
    RID runnable_buf;
    RID sort_key_buf;

    // Byte-stride/offset (in 4-byte words) describing how particle data is
    // packed inside vertex_buf/attrib_buf. Computed once from the mesh's
    // array format via RenderingServer::mesh_surface_get_format_*, and passed
    // to every compute shader via push constants.
    int vertex_stride_floats  = 3;
    int attrib_stride_words   = 0;
    int color_offset_words    = 0;
    int custom0_offset_words  = 0;

    // Pipelines
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
    bool      has_prev_transform = false;
    bool      gpu_ready    = false;

    // ── render mesh ──────────────────────────────────────────────────────────
    // Created once (not rebuilt per frame): a persistent ArrayMesh whose
    // vertex/attribute storage buffers are written to directly by the compute
    // shaders. See _ensure_render_node().
    MeshInstance3D  *render_node  = nullptr;
    Ref<ArrayMesh>   render_mesh;

    void _ensure_render_node();
    void _build_gpu_resources();
    void _destroy_gpu_resources();
    void _rebuild_gpu_resources();
    void _dispatch_clear_grid();
    void _dispatch_physics(Vector3 global_add_velocity,
                           const Basis &delta_basis,
                           Vector3 delta_origin,
                           bool has_delta);
    void _dispatch_sortkey();
};

}  // namespace godot
