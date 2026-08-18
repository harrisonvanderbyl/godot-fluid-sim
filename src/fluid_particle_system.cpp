#include "fluid_particle_system.hpp"
#include "fluid_source_sink.hpp"

#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>
#include <godot_cpp/classes/rd_vertex_attribute.hpp>
#include <godot_cpp/classes/rd_pipeline_rasterization_state.hpp>
#include <godot_cpp/classes/rd_pipeline_multisample_state.hpp>
#include <godot_cpp/classes/rd_pipeline_depth_stencil_state.hpp>
#include <godot_cpp/classes/rd_pipeline_color_blend_state.hpp>
#include <godot_cpp/classes/rd_pipeline_color_blend_state_attachment.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/classes/immediate_mesh.hpp>
#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/shader.hpp>
#include <godot_cpp/classes/shader_material.hpp>
#include <godot_cpp/classes/quad_mesh.hpp>
#include <godot_cpp/classes/plane_mesh.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/memory.hpp>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace godot;

// ─────────────────────────────────────────────────────────────────────────────
// Push-constant layout sent to every compute dispatch (std430 / 16-byte align)
// ─────────────────────────────────────────────────────────────────────────────
struct PushConstants {
    int32_t grid_w, grid_h, grid_d;
    int32_t num_particles;
    float   surface_tension;
    float   water_viscosity;
    float   attraction_force;
    int32_t neighbor_mode;
    float   gravity[3];      // vec3 in grid space
    int32_t frame_count;
    int32_t num_runnable;
    int32_t vertex_stride_floats;
    int32_t attrib_stride_words;
    int32_t color_offset_words;
    int32_t custom0_offset_words;
    float   has_delta;       // 1.0 if delta transform is non-identity
    int32_t max_occupancy;  // max liquid particles per grid cell
    float   back_pressure;   // outward bias as a cell fills
    // Delta transform as 3 vec4 rows (row-major 3x4 matrix):
    //   row0 = (m00, m01, m02, origin_x)
    //   row1 = (m10, m11, m12, origin_y)
    //   row2 = (m20, m21, m22, origin_z)
    // Per-particle displacement: dp = M * p + origin - p
    float   delta_row[12];
};  // 128 bytes

static_assert(sizeof(PushConstants) == 128,
              "PushConstants must stay within the 128-byte Vulkan guarantee");

// ─────────────────────────────────────────────────────────────────────────────
// Camera uniform buffer layout. MUST match CameraUBO in fluid_depth.glsl (std140).
// Declared at file scope so the allocation size and the per-frame upload can
// never drift apart — this used to be a local struct plus a hardcoded 352.
// ─────────────────────────────────────────────────────────────────────────────
struct CameraUniforms {
    float view_matrix[16];      // offset   0
    float proj_matrix[16];      // offset  64
    float inv_view_matrix[16];  // offset 128
    float inv_proj_matrix[16];  // offset 192
    float model_matrix[16];     // offset 256
    float cam_pos_world[3];     // offset 320
    float _pad0;                // offset 332
};                              // 336 bytes

// UBO allocation, rounded up to a 16-byte multiple.
static constexpr uint32_t CAMERA_UBO_SIZE =
    ((sizeof(CameraUniforms) + 15u) / 16u) * 16u;

// ─────────────────────────────────────────────────────────────────────────────
// Render-state constants. Kept in one place so a pipeline change (reverse-Z,
// MSAA, blending) is a single edit rather than two copies that can diverge.
//
// NOTE: this must agree with the depth qualifier in fluid_depth.glsl. If you
// switch to COMPARE_OP_GREATER you must also enable REVERSE_Z in that shader.
// ─────────────────────────────────────────────────────────────────────────────
static constexpr RenderingDevice::CompareOperator FLUID_DEPTH_COMPARE =
    RenderingDevice::COMPARE_OP_LESS;
static constexpr float FLUID_DEPTH_CLEAR = 1.0f;   // 0.0f under reverse-Z

// How many frames a GPU resource stays alive after we stop using it. The main
// renderer samples our color/depth textures through Texture2DRD on its own
// thread, so freeing the underlying VkImage the same frame we unbind it is a
// use-after-free. Godot buffers up to 3 frames; 4 gives us margin.
static constexpr int DEFERRED_FREE_FRAMES = 4;


// ─────────────────────────────────────────────────────────────────────────────
FluidParticleSystem::FluidParticleSystem() {}

FluidParticleSystem::~FluidParticleSystem() {
    // Last line of defence. Normally _exit_tree or NOTIFICATION_PREDELETE has
    // already run; _shutdown_gpu is idempotent so a second call is harmless.
    _shutdown_gpu();
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_PREDELETE:
            // Fires whether or not the node was ever in the tree, and whether
            // or not _exit_tree ran. Do NOT touch child nodes here — they may
            // already be mid-deletion.
            _shutdown_gpu();
            break;
        default:
            break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_grid_width","v"),    &FluidParticleSystem::set_grid_width);
    ClassDB::bind_method(D_METHOD("get_grid_width"),        &FluidParticleSystem::get_grid_width);
    ClassDB::bind_method(D_METHOD("set_grid_height","v"),   &FluidParticleSystem::set_grid_height);
    ClassDB::bind_method(D_METHOD("get_grid_height"),       &FluidParticleSystem::get_grid_height);
    ClassDB::bind_method(D_METHOD("set_grid_depth","v"),    &FluidParticleSystem::set_grid_depth);
    ClassDB::bind_method(D_METHOD("get_grid_depth"),        &FluidParticleSystem::get_grid_depth);
    ClassDB::bind_method(D_METHOD("set_grid_size","v"),    &FluidParticleSystem::set_grid_size);
    ClassDB::bind_method(D_METHOD("get_grid_size"),         &FluidParticleSystem::get_grid_size);
    ClassDB::bind_method(D_METHOD("set_num_particles","v"), &FluidParticleSystem::set_num_particles);
    ClassDB::bind_method(D_METHOD("get_num_particles"),     &FluidParticleSystem::get_num_particles);
    ClassDB::bind_method(D_METHOD("set_gravity","v"),       &FluidParticleSystem::set_gravity);
    ClassDB::bind_method(D_METHOD("get_gravity"),           &FluidParticleSystem::get_gravity);
    ClassDB::bind_method(D_METHOD("set_gravity_local","v"), &FluidParticleSystem::set_gravity_local);
    ClassDB::bind_method(D_METHOD("get_gravity_local"),     &FluidParticleSystem::get_gravity_local);
    ClassDB::bind_method(D_METHOD("set_surface_tension","v"),  &FluidParticleSystem::set_surface_tension);
    ClassDB::bind_method(D_METHOD("get_surface_tension"),      &FluidParticleSystem::get_surface_tension);
    ClassDB::bind_method(D_METHOD("set_water_viscosity","v"),  &FluidParticleSystem::set_water_viscosity);
    ClassDB::bind_method(D_METHOD("get_water_viscosity"),      &FluidParticleSystem::get_water_viscosity);
    ClassDB::bind_method(D_METHOD("set_attraction_force","v"), &FluidParticleSystem::set_attraction_force);
    ClassDB::bind_method(D_METHOD("get_attraction_force"),     &FluidParticleSystem::get_attraction_force);
    ClassDB::bind_method(D_METHOD("set_neighbor_mode","v"), &FluidParticleSystem::set_neighbor_mode);
    ClassDB::bind_method(D_METHOD("get_neighbor_mode"),     &FluidParticleSystem::get_neighbor_mode);
    ClassDB::bind_method(D_METHOD("set_max_occupancy","v"), &FluidParticleSystem::set_max_occupancy);
    ClassDB::bind_method(D_METHOD("get_max_occupancy"),      &FluidParticleSystem::get_max_occupancy);
    ClassDB::bind_method(D_METHOD("set_back_pressure","v"), &FluidParticleSystem::set_back_pressure);
    ClassDB::bind_method(D_METHOD("get_back_pressure"),      &FluidParticleSystem::get_back_pressure);
    ClassDB::bind_method(D_METHOD("set_simulation_active","v"), &FluidParticleSystem::set_simulation_active);
    ClassDB::bind_method(D_METHOD("get_simulation_active"),     &FluidParticleSystem::get_simulation_active);
    ClassDB::bind_method(D_METHOD("set_use_initial_chunk","v"), &FluidParticleSystem::set_use_initial_chunk);
    ClassDB::bind_method(D_METHOD("get_use_initial_chunk"),     &FluidParticleSystem::get_use_initial_chunk);
    ClassDB::bind_method(D_METHOD("set_initial_chunk_origin","v"), &FluidParticleSystem::set_initial_chunk_origin);
    ClassDB::bind_method(D_METHOD("get_initial_chunk_origin"),     &FluidParticleSystem::get_initial_chunk_origin);
    ClassDB::bind_method(D_METHOD("set_initial_chunk_size","v"), &FluidParticleSystem::set_initial_chunk_size);
    ClassDB::bind_method(D_METHOD("get_initial_chunk_size"),     &FluidParticleSystem::get_initial_chunk_size);
    ClassDB::bind_method(D_METHOD("set_initial_chunk_color","v"), &FluidParticleSystem::set_initial_chunk_color);
    ClassDB::bind_method(D_METHOD("get_initial_chunk_color"),     &FluidParticleSystem::get_initial_chunk_color);
    ClassDB::bind_method(D_METHOD("set_initial_chunk_attraction","v"), &FluidParticleSystem::set_initial_chunk_attraction);
    ClassDB::bind_method(D_METHOD("get_initial_chunk_attraction"),     &FluidParticleSystem::get_initial_chunk_attraction);
    ClassDB::bind_method(D_METHOD("spawn_block","origin","w","h","d","color","attraction"),
                         &FluidParticleSystem::spawn_block);
    ClassDB::bind_method(D_METHOD("add_velocity_impulse","impulse"),
                         &FluidParticleSystem::add_velocity_impulse);
    ClassDB::bind_method(D_METHOD("add_rotational_impulse","center","axis_amount"),
                         &FluidParticleSystem::add_rotational_impulse);
    ClassDB::bind_method(D_METHOD("reset_grid"),
                         &FluidParticleSystem::reset_grid);

    ClassDB::bind_method(D_METHOD("get_grid_aabb"),      &FluidParticleSystem::get_grid_aabb);

    // Shader paths
    ClassDB::bind_method(D_METHOD("set_clear_shader_path","v"),   &FluidParticleSystem::set_clear_shader_path);
    ClassDB::bind_method(D_METHOD("get_clear_shader_path"),        &FluidParticleSystem::get_clear_shader_path);
    ClassDB::bind_method(D_METHOD("set_physics_shader_path","v"), &FluidParticleSystem::set_physics_shader_path);
    ClassDB::bind_method(D_METHOD("get_physics_shader_path"),      &FluidParticleSystem::get_physics_shader_path);
    ClassDB::bind_method(D_METHOD("set_sortkey_shader_path","v"), &FluidParticleSystem::set_sortkey_shader_path);
    ClassDB::bind_method(D_METHOD("get_sortkey_shader_path"),      &FluidParticleSystem::get_sortkey_shader_path);
    ClassDB::bind_method(D_METHOD("set_render_material","v"),  &FluidParticleSystem::set_render_material);
    ClassDB::bind_method(D_METHOD("get_render_material"),       &FluidParticleSystem::get_render_material);
    ClassDB::bind_method(D_METHOD("set_smooth_radius","v"),        &FluidParticleSystem::set_smooth_radius);
    ClassDB::bind_method(D_METHOD("get_smooth_radius"),             &FluidParticleSystem::get_smooth_radius);
    ClassDB::bind_method(D_METHOD("set_tint_strength","v"),       &FluidParticleSystem::set_tint_strength);
    ClassDB::bind_method(D_METHOD("get_tint_strength"),            &FluidParticleSystem::get_tint_strength);
    ClassDB::bind_method(D_METHOD("set_refraction_strength","v"), &FluidParticleSystem::set_refraction_strength);
    ClassDB::bind_method(D_METHOD("get_refraction_strength"),      &FluidParticleSystem::get_refraction_strength);
    ClassDB::bind_method(D_METHOD("set_absorption_dist","v"),     &FluidParticleSystem::set_absorption_dist);
    ClassDB::bind_method(D_METHOD("get_absorption_dist"),          &FluidParticleSystem::get_absorption_dist);
    ClassDB::bind_method(D_METHOD("set_fluid_tint","v"),          &FluidParticleSystem::set_fluid_tint);
    ClassDB::bind_method(D_METHOD("get_fluid_tint"),               &FluidParticleSystem::get_fluid_tint);
    ClassDB::bind_method(D_METHOD("set_smooth_falloff","v"),      &FluidParticleSystem::set_smooth_falloff);
    ClassDB::bind_method(D_METHOD("get_smooth_falloff"),           &FluidParticleSystem::get_smooth_falloff);
    ClassDB::bind_method(D_METHOD("set_coverage_threshold","v"),   &FluidParticleSystem::set_coverage_threshold);
    ClassDB::bind_method(D_METHOD("get_coverage_threshold"),        &FluidParticleSystem::get_coverage_threshold);

    ClassDB::bind_method(D_METHOD("reload_physics_shader"), &FluidParticleSystem::reload_physics_shader);
    ClassDB::bind_method(D_METHOD("get_reload_physics_shader"), &FluidParticleSystem::get_reload_physics_shader);
    ClassDB::bind_method(D_METHOD("reload_render_shader"),  &FluidParticleSystem::reload_render_shader);
    ClassDB::bind_method(D_METHOD("get_reload_render_shader"), &FluidParticleSystem::get_reload_render_shader);

    // ── Inspector layout ───────────────────────────────────────────────────
    // Reload button at the top for quick iteration.
    ADD_GROUP("Shaders", "");
    ADD_PROPERTY(PropertyInfo(Variant::CALLABLE, "reload_physics_shader",
            PROPERTY_HINT_TOOL_BUTTON, "Reload Physics Shader,Reload",
            PROPERTY_USAGE_EDITOR),
            "", "get_reload_physics_shader");
    ADD_PROPERTY(PropertyInfo(Variant::CALLABLE, "reload_render_shader",
            PROPERTY_HINT_TOOL_BUTTON, "Reload Render Shader,Reload",
            PROPERTY_USAGE_EDITOR),
            "", "get_reload_render_shader");

    // Compute shader paths.
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "clear_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_clear_shader_path",   "get_clear_shader_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "physics_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_physics_shader_path", "get_physics_shader_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "sortkey_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_sortkey_shader_path", "get_sortkey_shader_path");
    // Composite material — when set, pre-fills with fluid_composite.gdshader
    // so the user can edit the composite shader code + uniforms in the inspector.
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "composite_material",
        PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial"), "set_render_material", "get_render_material");
    ADD_GROUP("", "");

    // Simulation parameters.
    ADD_GROUP("Simulation", "sim_");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,  "sim_active"), "set_simulation_active", "get_simulation_active");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "sim_grid_size"), "set_grid_size", "get_grid_size");
    ADD_PROPERTY(PropertyInfo(Variant::INT,   "sim_num_particles"),   "set_num_particles",   "get_num_particles");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "sim_gravity"),         "set_gravity",         "get_gravity");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,    "sim_gravity_local"),   "set_gravity_local",   "get_gravity_local");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sim_surface_tension"), "set_surface_tension", "get_surface_tension");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sim_water_viscosity"), "set_water_viscosity", "get_water_viscosity");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sim_attraction_force", PROPERTY_HINT_RANGE, "-2,2,0.01"), "set_attraction_force", "get_attraction_force");
    ADD_PROPERTY(PropertyInfo(Variant::INT,   "sim_neighbor_mode"),   "set_neighbor_mode",   "get_neighbor_mode");
    ADD_PROPERTY(PropertyInfo(Variant::INT,   "sim_max_occupancy"),   "set_max_occupancy",   "get_max_occupancy");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sim_back_pressure",   PROPERTY_HINT_RANGE, "0,2,0.01"), "set_back_pressure", "get_back_pressure");
    ADD_GROUP("", "");

    // Initial chunk fill group.
    ADD_GROUP("Initial Chunk", "initial_chunk_");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,     "initial_chunk_use"),                             "set_use_initial_chunk",        "get_use_initial_chunk");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3,  "initial_chunk_origin"),                          "set_initial_chunk_origin",     "get_initial_chunk_origin");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "initial_chunk_size"),                            "set_initial_chunk_size",       "get_initial_chunk_size");
    ADD_PROPERTY(PropertyInfo(Variant::COLOR,    "initial_chunk_color"),                           "set_initial_chunk_color",      "get_initial_chunk_color");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT,    "initial_chunk_attraction", PROPERTY_HINT_RANGE, "-2,2,0.01"), "set_initial_chunk_attraction", "get_initial_chunk_attraction");

    // Composite shader uniforms group.
    ADD_GROUP("Composite", "composite_");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "composite_smooth_radius",       PROPERTY_HINT_RANGE, "0,10,0.01"),  "set_smooth_radius",       "get_smooth_radius");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "composite_tint_strength",       PROPERTY_HINT_RANGE, "0,2,0.01"),   "set_tint_strength",        "get_tint_strength");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "composite_refraction_strength", PROPERTY_HINT_RANGE, "0,0.2,0.001"),"set_refraction_strength",  "get_refraction_strength");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "composite_absorption_dist",     PROPERTY_HINT_RANGE, "0,20,0.01"),  "set_absorption_dist",      "get_absorption_dist");
    ADD_PROPERTY(PropertyInfo(Variant::COLOR, "composite_tint",                PROPERTY_HINT_COLOR_NO_ALPHA),      "set_fluid_tint",           "get_fluid_tint");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "composite_smooth_falloff",      PROPERTY_HINT_RANGE, "0,5,0.01"),   "set_smooth_falloff",      "get_smooth_falloff");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "composite_coverage_threshold",   PROPERTY_HINT_RANGE, "0,1,0.01"),   "set_coverage_threshold",  "get_coverage_threshold");
    ADD_GROUP("", "");
}

// ─────────────────────────────────────────────────────────────────────────────
// Composite uniform setters — push to the active composite material.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::set_smooth_radius(float v) {
    smooth_radius = v;
    Ref<ShaderMaterial> comp = _active_composite_material();
    if (comp.is_valid()) comp->set_shader_parameter("smooth_radius", v);
}
void FluidParticleSystem::set_tint_strength(float v) {
    tint_strength = v;
    Ref<ShaderMaterial> comp = _active_composite_material();
    if (comp.is_valid()) comp->set_shader_parameter("tint_strength", v);
}
void FluidParticleSystem::set_refraction_strength(float v) {
    refraction_strength = v;
    Ref<ShaderMaterial> comp = _active_composite_material();
    if (comp.is_valid()) comp->set_shader_parameter("refraction_strength", v);
}
void FluidParticleSystem::set_absorption_dist(float v) {
    absorption_dist = v;
    Ref<ShaderMaterial> comp = _active_composite_material();
    if (comp.is_valid()) comp->set_shader_parameter("absorption_dist", v);
}
void FluidParticleSystem::set_fluid_tint(Color v) {
    fluid_tint = v;
    Ref<ShaderMaterial> comp = _active_composite_material();
    if (comp.is_valid()) comp->set_shader_parameter("fluid_tint", v);
}
void FluidParticleSystem::set_smooth_falloff(float v) {
    smooth_falloff = v;
    Ref<ShaderMaterial> comp = _active_composite_material();
    if (comp.is_valid()) comp->set_shader_parameter("smooth_falloff", v);
}
void FluidParticleSystem::set_coverage_threshold(float v) {
    coverage_threshold = v;
    Ref<ShaderMaterial> comp = _active_composite_material();
    if (comp.is_valid()) comp->set_shader_parameter("coverage_threshold", v);
}

// ─────────────────────────────────────────────────────────────────────────────
// Shader path setters — hot-reload the corresponding compute shader when the
// path changes while the node is in the tree. If the new shader fails to
// compile, the old one is kept and an error is printed.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::set_clear_shader_path(const String &v) {
    if (clear_shader_path == v) return;
    clear_shader_path = v;
    if (is_inside_tree()) _reload_compute_shader(0);
}
void FluidParticleSystem::set_physics_shader_path(const String &v) {
    if (physics_shader_path == v) return;
    physics_shader_path = v;
    if (is_inside_tree()) _reload_compute_shader(1);
}
void FluidParticleSystem::set_sortkey_shader_path(const String &v) {
    if (sortkey_shader_path == v) return;
    sortkey_shader_path = v;
    if (is_inside_tree()) _reload_compute_shader(2);
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::set_grid_width(int v)   { if (grid_width  == v) return; grid_width  = v; _rebuild_gpu_resources(); }
void FluidParticleSystem::set_grid_height(int v)  { if (grid_height == v) return; grid_height = v; _rebuild_gpu_resources(); }
void FluidParticleSystem::set_grid_depth(int v)   { if (grid_depth  == v) return; grid_depth  = v; _rebuild_gpu_resources(); }
void FluidParticleSystem::set_grid_size(Vector3i v) {
    if (grid_width == v.x && grid_height == v.y && grid_depth == v.z) return;
    grid_width = v.x; grid_height = v.y; grid_depth = v.z;
    _rebuild_gpu_resources();
}
void FluidParticleSystem::set_num_particles(int v) {
    if (num_particles == v) return;
    num_particles = v;
    _rebuild_gpu_resources();
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::set_render_material(const Ref<ShaderMaterial> &v) {
    render_material = v;

    if (render_material.is_valid()) {
        // If the material has no shader or empty code, pre-fill with the default
        // composite shader so the user can edit code + uniforms immediately.
        Ref<Shader> shader = render_material->get_shader();
        if (shader.is_null() || shader->get_code().is_empty()) {
            Ref<Shader> def;
            def.instantiate();
            Ref<FileAccess> sf = FileAccess::open(
                "res://addons/fluid_particles/shaders/fluid_composite.gdshader",
                FileAccess::READ);
            if (sf.is_valid()) def->set_code(sf->get_as_text());
            render_material->set_shader(def);
        }
        // Re-bind the RD color texture in case the user replaced the material.
        if (rd_color_tex_2d.is_valid()) {
            render_material->set_shader_parameter("fluid_tex", rd_color_tex_2d);
        }
        if (rd_depth_tex_2d.is_valid()) {
            render_material->set_shader_parameter("fluid_depth_tex", rd_depth_tex_2d);
        }
        // Push all composite uniform values from node properties.
        _push_composite_uniforms(render_material);
    }

    // The user-facing material is the COMPOSITE material (the one that draws
    // the blurred/refracted result to screen). The RD render pipeline writes
    // color+depth directly to RD textures — the composite shader samples them.
    if (composite_node) {
        if (render_material.is_valid()) {
            composite_node->set_material_override(render_material);
        } else if (composite_material.is_valid()) {
            composite_node->set_material_override(composite_material);
        }
    }
}

Ref<ShaderMaterial> FluidParticleSystem::get_render_material() const {
    return render_material;
}

// Push every node-owned composite uniform onto a material in one place, so
// adding a uniform means editing one function instead of three.
void FluidParticleSystem::_push_composite_uniforms(const Ref<ShaderMaterial> &mat) const {
    if (mat.is_null()) return;
    mat->set_shader_parameter("smooth_radius", smooth_radius);
    mat->set_shader_parameter("tint_strength", tint_strength);
    mat->set_shader_parameter("refraction_strength", refraction_strength);
    mat->set_shader_parameter("absorption_dist", absorption_dist);
    mat->set_shader_parameter("fluid_tint", fluid_tint);
    mat->set_shader_parameter("smooth_falloff", smooth_falloff);
    mat->set_shader_parameter("coverage_threshold", coverage_threshold);
}

// ─────────────────────────────────────────────────────────────────────────────
// Deferred RID destruction
//
// The color/depth textures are shared to the main renderer via Texture2DRD.
// Godot's render thread can still be sampling them for up to a few frames after
// we stop using them, so freeing on the spot is a use-after-free that shows up
// as a driver crash while resizing the window. Everything that the main
// renderer can observe goes through this queue instead.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_queue_free_rid(RID r, bool on_shared_device) {
    if (!r.is_valid()) return;
    DeferredFree d;
    d.rid              = r;
    d.frames_left      = DEFERRED_FREE_FRAMES;
    d.on_shared_device = on_shared_device;
    deferred_frees.push_back(d);
}

void FluidParticleSystem::_process_deferred_frees(bool flush_now) {
    if (deferred_frees.empty()) return;

    if (flush_now) {
        // Draining immediately (teardown): make sure our own submissions are
        // done first. The caller is responsible for the fact that the main
        // renderer is also finished — which is true during _exit_tree.
        _sync_rd();
    }

    std::vector<DeferredFree> keep;
    keep.reserve(deferred_frees.size());

    // Insertion order matters: shared-device wrappers were pushed before the
    // local textures they alias, so iterating forward frees the wrapper first.
    for (DeferredFree &d : deferred_frees) {
        if (!flush_now && --d.frames_left > 0) {
            keep.push_back(d);
            continue;
        }
        RenderingDevice *dev = d.on_shared_device ? shared_rd : rd;
        if (dev && d.rid.is_valid()) {
            dev->free_rid(d.rid);
        }
    }
    deferred_frees.swap(keep);
}

// Free a local-device RID immediately and null the handle. Safe to call with an
// invalid RID or a null device.
void FluidParticleSystem::_free_local_rid(RID &r) {
    if (rd && r.is_valid()) rd->free_rid(r);
    r = RID();
}

// ─────────────────────────────────────────────────────────────────────────────
// Local RenderingDevice lifetime.
//
// The device is created ONCE and reused across every rebuild. The old code
// created a fresh device in _build_gpu_resources and abandoned the previous one
// in _destroy_gpu_resources ("Godot manages its lifetime" — it does not:
// create_local_rendering_device hands ownership to the caller). Every inspector
// edit that touched grid size or particle count leaked a whole device plus all
// its buffers, pipelines and textures.
// ─────────────────────────────────────────────────────────────────────────────
bool FluidParticleSystem::_ensure_device() {
    if (rd) return true;

    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) return false;

    shared_rd = rs->get_rendering_device();
    if (!shared_rd) {
        UtilityFunctions::printerr(
            "FluidParticleSystem: no shared RenderingDevice (headless or GL renderer?).");
        return false;
    }
    rd = rs->create_local_rendering_device();
    if (!rd) {
        UtilityFunctions::printerr("FluidParticleSystem: could not create local RenderingDevice.");
        shared_rd = nullptr;
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Full shutdown: destroy every GPU resource AND the local device itself.
// Idempotent — safe to call from _exit_tree, NOTIFICATION_PREDELETE and the
// destructor in any order.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_shutdown_gpu() {
    if (!rd && deferred_frees.empty()) {
        shared_rd = nullptr;
        return;
    }

    _destroy_gpu_resources();

    // Drain anything still parked in the deferred queue. By the time we get
    // here the node is leaving the tree, so the main renderer is no longer
    // going to sample our textures.
    _process_deferred_frees(true);
    deferred_frees.clear();

    if (rd) {
        // create_local_rendering_device() transfers ownership to us. This is
        // the call that was missing.
        memdelete(rd);
        rd = nullptr;
    }
    shared_rd = nullptr;
    rd_submitted = false;
    gpu_ready = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build the RD offscreen render pipeline:
//   FluidParticleSystem (this)
//     └── MeshInstance3D (composite_node)  — fullscreen quad with the
//                                            blur+depth composite shader
//
// Particles are rendered directly via a RenderingDevice render pipeline into
// an RD framebuffer (color RGBA8 + depth D32_SFLOAT). The vertex/fragment
// shader (fluid_depth.glsl) reads the mesh's storage buffers and writes
// depth + color. The output textures are exposed to the composite shader via
// Texture2DRD.
//
// This function is now strictly idempotent and cleans up after itself on every
// failure path, so a shader compile error no longer strands a set of textures
// that the next call re-creates.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_ensure_rd_render_pipeline() {
    if (rd_render_pipeline.is_valid()) return;  // already built
    if (!rd) return;
    if (!vertex_buf.is_valid() || !attrib_buf.is_valid()) return;

    // Anything created below is torn down by _destroy_render_resources() if we
    // bail out partway, so a retry starts from a clean slate.

    // ── Compile the render shader (fluid_depth.glsl: vertex + fragment) ──────
    if (!rd_render_shader.is_valid()) {
        rd_render_shader = _compile_render_shader();
        if (!rd_render_shader.is_valid()) {
            _destroy_render_resources(false);
            return;
        }
    }

    // ── Vertex array wrapping the mesh's storage buffers ────────────────────
    if (!_create_vertex_array()) {
        _destroy_render_resources(false);
        return;
    }

    // ── Color + depth textures, framebuffer, shared wrappers ────────────────
    // Start at 512x512; _sync_rd_viewport_size() resizes to match the main
    // viewport on the first frame.
    if (!_create_render_targets(Vector2i(512, 512))) {
        _destroy_render_resources(false);
        return;
    }

    // ── Camera uniform buffer + uniform set ──────────────────────────────────
    // Camera matrices (5 x mat4 + vec3 + pad) exceed Vulkan's 128-byte push
    // constant limit, so they go in a uniform buffer at binding 0.
    // Layout MUST match CameraUBO in fluid_depth.glsl (std140).
    if (!rd_camera_ubo.is_valid()) {
        PackedByteArray zero;
        zero.resize(CAMERA_UBO_SIZE);
        zero.fill(0);
        rd_camera_ubo = rd->uniform_buffer_create(CAMERA_UBO_SIZE, zero);
        if (!rd_camera_ubo.is_valid()) {
            UtilityFunctions::printerr("FluidParticleSystem: failed to create camera UBO");
            _destroy_render_resources(false);
            return;
        }
    }
    if (!_create_render_uniform_set()) {
        _destroy_render_resources(false);
        return;
    }

    // ── Render pipeline ──────────────────────────────────────────────────────
    if (!_create_render_pipeline_state()) {
        _destroy_render_resources(false);
        return;
    }

    // ── Composite quad (fullscreen, samples the RD textures) ────────────────
    _ensure_composite_node();

    UtilityFunctions::print("FluidParticleSystem: RD render pipeline ready.");
}

// ─────────────────────────────────────────────────────────────────────────────
RID FluidParticleSystem::_compile_render_shader() {
    if (!rd) return RID();

    Ref<RDShaderFile> sf = ResourceLoader::get_singleton()->load(
        "res://addons/fluid_particles/shaders/fluid_depth.glsl", "",
        ResourceLoader::CACHE_MODE_REPLACE);
    if (!sf.is_valid()) {
        UtilityFunctions::printerr("FluidParticleSystem: cannot load fluid_depth.glsl");
        return RID();
    }
    Ref<RDShaderSPIRV> spirv = sf->get_spirv();
    if (!spirv.is_valid()) {
        UtilityFunctions::printerr("FluidParticleSystem: no SPIR-V in fluid_depth.glsl");
        return RID();
    }
    String verr = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_VERTEX);
    String ferr = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_FRAGMENT);
    if (!verr.is_empty() || !ferr.is_empty()) {
        UtilityFunctions::printerr("FluidParticleSystem: fluid_depth.glsl compile error:\n",
                                   "  vertex: ", verr, "\n  fragment: ", ferr);
        return RID();
    }
    RID s = rd->shader_create_from_spirv(spirv);
    if (!s.is_valid()) {
        UtilityFunctions::printerr("FluidParticleSystem: failed to create render shader RID");
    }
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// The render mesh uses ARRAY_FLAG_USE_STORAGE_BUFFER, so its vertex and
// attribute buffers are RD storage buffers. We build a vertex array describing
// the layout (vec3 position, vec4 color, vec4 custom0) and bind them directly.
// ─────────────────────────────────────────────────────────────────────────────
bool FluidParticleSystem::_create_vertex_array() {
    if (!rd) return false;
    if (rd_vertex_array.is_valid()) return true;

    TypedArray<Ref<RDVertexAttribute>> attrs;
    {
        // Binding 0: vertex buffer (position: vec3 float).
        Ref<RDVertexAttribute> a;
        a.instantiate();
        a->set_location(0);   // matches layout(location=0) in the GLSL
        a->set_binding(0);
        a->set_offset(0);
        a->set_format(RenderingDevice::DATA_FORMAT_R32G32B32_SFLOAT);
        a->set_stride(vertex_stride_floats * 4);  // bytes
        a->set_frequency(RenderingDevice::VERTEX_FREQUENCY_VERTEX);
        attrs.append(a);
    }
    {
        // Color: vec4 float at offset 0 in the attribute buffer.
        Ref<RDVertexAttribute> a;
        a.instantiate();
        a->set_location(1);   // matches layout(location=1)
        a->set_binding(1);
        a->set_offset(color_offset_words * 4);
        a->set_format(RenderingDevice::DATA_FORMAT_R32G32B32A32_SFLOAT);
        a->set_stride(attrib_stride_words * 4);   // bytes
        a->set_frequency(RenderingDevice::VERTEX_FREQUENCY_VERTEX);
        attrs.append(a);
    }
    {
        // Custom0: vec4 float at custom0_offset in the attribute buffer.
        Ref<RDVertexAttribute> a;
        a.instantiate();
        a->set_location(2);   // matches layout(location=2)
        a->set_binding(1);
        a->set_offset(custom0_offset_words * 4);  // bytes
        a->set_format(RenderingDevice::DATA_FORMAT_R32G32B32A32_SFLOAT);
        a->set_stride(attrib_stride_words * 4);
        a->set_frequency(RenderingDevice::VERTEX_FREQUENCY_VERTEX);
        attrs.append(a);
    }

    rd_vertex_format = rd->vertex_format_create(attrs);
    if (rd_vertex_format < 0) {
        UtilityFunctions::printerr("FluidParticleSystem: failed to create vertex format");
        return false;
    }

    TypedArray<RID> src_buffers;
    src_buffers.append(vertex_buf);
    src_buffers.append(attrib_buf);
    rd_vertex_array = rd->vertex_array_create(num_particles, rd_vertex_format, src_buffers);
    if (!rd_vertex_array.is_valid()) {
        UtilityFunctions::printerr("FluidParticleSystem: failed to create vertex array");
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
bool FluidParticleSystem::_create_render_uniform_set() {
    if (!rd || !rd_render_shader.is_valid() || !rd_camera_ubo.is_valid()) return false;
    if (rd_render_uniform_set.is_valid()) return true;

    Ref<RDUniform> u;
    u.instantiate();
    u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_UNIFORM_BUFFER);
    u->set_binding(0);
    u->add_id(rd_camera_ubo);
    TypedArray<RDUniform> uniforms;
    uniforms.append(u);
    rd_render_uniform_set = rd->uniform_set_create(uniforms, rd_render_shader, 0);
    if (!rd_render_uniform_set.is_valid()) {
        UtilityFunctions::printerr("FluidParticleSystem: failed to create render uniform set");
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Create the color + depth textures, the framebuffer, and the shared-device
// wrappers the compositor samples. Single source of truth — this used to be
// duplicated verbatim between _ensure_rd_render_pipeline and
// _sync_rd_viewport_size, which is exactly the kind of duplication that rots
// when the pipeline changes.
//
// On failure nothing is left behind and rd_vp_size is NOT advanced, so the
// next frame retries instead of silently deciding the size already matches.
// ─────────────────────────────────────────────────────────────────────────────
bool FluidParticleSystem::_create_render_targets(Vector2i size) {
    if (!rd || !shared_rd) return false;
    if (size.x <= 0 || size.y <= 0) return false;

    auto make_format = [&](RenderingDevice::DataFormat fmt,
                           BitField<RenderingDevice::TextureUsageBits> usage) {
        Ref<RDTextureFormat> f;
        f.instantiate();
        f->set_texture_type(RenderingDevice::TEXTURE_TYPE_2D);
        f->set_format(fmt);
        f->set_width(size.x);
        f->set_height(size.y);
        f->set_depth(1);
        f->set_array_layers(1);
        f->set_mipmaps(1);
        f->set_samples(RenderingDevice::TEXTURE_SAMPLES_1);
        f->set_usage_bits(usage);
        return f;
    };

    Ref<RDTextureView> view;
    view.instantiate();

    RID color = rd->texture_create(
        make_format(RenderingDevice::DATA_FORMAT_R8G8B8A8_UNORM,
                    RenderingDevice::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT |
                    RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT |
                    RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT),
        view);
    RID depth = rd->texture_create(
        make_format(RenderingDevice::DATA_FORMAT_D32_SFLOAT,
                    RenderingDevice::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                    RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT |
                    RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT),
        view);

    if (!color.is_valid() || !depth.is_valid()) {
        UtilityFunctions::printerr("FluidParticleSystem: failed to create RD render targets");
        if (color.is_valid()) rd->free_rid(color);
        if (depth.is_valid()) rd->free_rid(depth);
        return false;
    }

    TypedArray<RID> fb_textures;
    fb_textures.append(color);
    fb_textures.append(depth);
    RID fb = rd->framebuffer_create(fb_textures);
    if (!fb.is_valid()) {
        UtilityFunctions::printerr("FluidParticleSystem: failed to create RD framebuffer");
        rd->free_rid(color);
        rd->free_rid(depth);
        return false;
    }

    // Share to the shared device for the compositor. The color+depth textures
    // live on the local device; the composite shader is a spatial shader on the
    // shared device, so we wrap the local device's VkImage handles.
    uint64_t color_vkimage = rd->get_driver_resource(
        RenderingDevice::DRIVER_RESOURCE_TEXTURE, color, 0);
    uint64_t depth_vkimage = rd->get_driver_resource(
        RenderingDevice::DRIVER_RESOURCE_TEXTURE, depth, 0);

    RID shared_color = shared_rd->texture_create_from_extension(
        RenderingDevice::TEXTURE_TYPE_2D,
        RenderingDevice::DATA_FORMAT_R8G8B8A8_UNORM,
        RenderingDevice::TEXTURE_SAMPLES_1,
        RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT |
        RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT,
        color_vkimage, size.x, size.y, 1, 1, 1);
    RID shared_depth = shared_rd->texture_create_from_extension(
        RenderingDevice::TEXTURE_TYPE_2D,
        RenderingDevice::DATA_FORMAT_D32_SFLOAT,
        RenderingDevice::TEXTURE_SAMPLES_1,
        RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT |
        RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT,
        depth_vkimage, size.x, size.y, 1, 1, 1);

    if (!shared_color.is_valid() || !shared_depth.is_valid()) {
        UtilityFunctions::printerr(
            "FluidParticleSystem: failed to share textures to shared device");
        if (shared_color.is_valid()) shared_rd->free_rid(shared_color);
        if (shared_depth.is_valid()) shared_rd->free_rid(shared_depth);
        rd->free_rid(fb);
        rd->free_rid(color);
        rd->free_rid(depth);
        return false;
    }

    // Commit.
    rd_color_tex     = color;
    rd_depth_tex     = depth;
    rd_framebuffer   = fb;
    shared_color_tex = shared_color;
    shared_depth_tex = shared_depth;
    rd_framebuffer_format = rd->framebuffer_get_format(rd_framebuffer);
    rd_vp_size = size;

    // Texture2DRD wrappers for the composite shader.
    if (rd_color_tex_2d.is_null()) rd_color_tex_2d.instantiate();
    if (rd_depth_tex_2d.is_null()) rd_depth_tex_2d.instantiate();
    rd_color_tex_2d->set_texture_rd_rid(shared_color_tex);
    rd_depth_tex_2d->set_texture_rd_rid(shared_depth_tex);

    _bind_render_targets_to_composite();
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Release the current render targets. `deferred` routes them through the
// frame-delay queue (use this while the node is live and the main renderer may
// still be reading them); pass false only during teardown.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_destroy_render_targets(bool deferred) {
    // Unbind first so the renderer stops picking them up for NEW frames. This
    // does not help with frames already in flight — that is what the delay is
    // for.
    if (rd_color_tex_2d.is_valid()) rd_color_tex_2d->set_texture_rd_rid(RID());
    if (rd_depth_tex_2d.is_valid()) rd_depth_tex_2d->set_texture_rd_rid(RID());

    if (deferred) {
        // Order matters: the shared wrappers alias the local textures' VkImage,
        // so they must be released first. The old code freed the local textures
        // first and left the wrappers pointing at destroyed memory.
        _queue_free_rid(shared_color_tex, true);
        _queue_free_rid(shared_depth_tex, true);
        _queue_free_rid(rd_framebuffer,   false);
        _queue_free_rid(rd_color_tex,     false);
        _queue_free_rid(rd_depth_tex,     false);
    } else {
        _sync_rd();
        if (shared_rd) {
            if (shared_color_tex.is_valid()) shared_rd->free_rid(shared_color_tex);
            if (shared_depth_tex.is_valid()) shared_rd->free_rid(shared_depth_tex);
        }
        _free_local_rid(rd_framebuffer);
        _free_local_rid(rd_color_tex);
        _free_local_rid(rd_depth_tex);
    }

    shared_color_tex = RID();
    shared_depth_tex = RID();
    rd_framebuffer   = RID();
    rd_color_tex     = RID();
    rd_depth_tex     = RID();
    rd_framebuffer_format = -1;
}

// ─────────────────────────────────────────────────────────────────────────────
bool FluidParticleSystem::_create_render_pipeline_state() {
    if (!rd) return false;
    if (!rd_render_shader.is_valid()) return false;
    if (rd_framebuffer_format < 0 || rd_vertex_format < 0) return false;
    if (rd_render_pipeline.is_valid()) return true;

    Ref<RDPipelineRasterizationState> raster;
    raster.instantiate();
    raster->set_cull_mode(RenderingDevice::POLYGON_CULL_DISABLED);

    Ref<RDPipelineMultisampleState> ms;
    ms.instantiate();
    ms->set_sample_count(RenderingDevice::TEXTURE_SAMPLES_1);

    Ref<RDPipelineDepthStencilState> ds;
    ds.instantiate();
    ds->set_enable_depth_test(true);
    ds->set_enable_depth_write(true);
    ds->set_depth_compare_operator(FLUID_DEPTH_COMPARE);

    Ref<RDPipelineColorBlendState> blend;
    blend.instantiate();
    {
        Ref<RDPipelineColorBlendStateAttachment> att;
        att.instantiate();
        att->set_enable_blend(false);
        att->set_write_r(true);
        att->set_write_g(true);
        att->set_write_b(true);
        att->set_write_a(true);
        TypedArray<Ref<RDPipelineColorBlendStateAttachment>> atts;
        atts.append(att);
        blend->set_attachments(atts);
    }

    rd_render_pipeline = rd->render_pipeline_create(
        rd_render_shader, rd_framebuffer_format, rd_vertex_format,
        RenderingDevice::RENDER_PRIMITIVE_POINTS, raster, ms, ds, blend);
    if (!rd_render_pipeline.is_valid()) {
        UtilityFunctions::printerr("FluidParticleSystem: failed to create render pipeline");
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_bind_render_targets_to_composite() {
    Ref<ShaderMaterial> comp = _active_composite_material();
    if (comp.is_null()) return;
    comp->set_shader_parameter("fluid_tex", rd_color_tex_2d);
    comp->set_shader_parameter("fluid_depth_tex", rd_depth_tex_2d);
    if (rd_vp_size.x > 0 && rd_vp_size.y > 0) {
        comp->set_shader_parameter(
            "fluid_pixel_size",
            Vector2(1.0f / (float)rd_vp_size.x, 1.0f / (float)rd_vp_size.y));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Create the fullscreen composite quad, once. The old code memnew'd a fresh
// MeshInstance3D on every _ensure_rd_render_pipeline call — and since
// _exit_tree does not free children, re-entering the tree stacked up a new
// composite node each cycle, each one still holding a material bound to
// textures from a destroyed device.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_ensure_composite_node() {
    if (composite_shader.is_null()) {
        composite_shader.instantiate();
        Ref<FileAccess> sf = FileAccess::open(
            "res://addons/fluid_particles/shaders/fluid_composite.gdshader",
            FileAccess::READ);
        if (sf.is_valid()) {
            composite_shader->set_code(sf->get_as_text());
        } else {
            UtilityFunctions::printerr(
                "FluidParticleSystem: cannot open fluid_composite.gdshader");
        }
    }

    if (composite_material.is_null()) {
        composite_material.instantiate();
        composite_material->set_shader(composite_shader);
        composite_material->set_shader_parameter("offscreen_far", 100.0);
        composite_material->set_shader_parameter("offscreen_near", 0.1);
        _push_composite_uniforms(composite_material);
    }

    _bind_render_targets_to_composite();

    if (composite_node) return;   // already built and still ours

    // Fullscreen quad mesh — a 2-triangle quad in NDC; the vertex shader writes
    // POSITION directly so it covers the whole screen regardless of camera.
    Ref<QuadMesh> quad;
    quad.instantiate();
    quad->set_size(Vector2(2.0, 2.0));   // NDC-sized
    quad->set_orientation(PlaneMesh::FACE_Z);

    composite_node = memnew(MeshInstance3D);
    composite_node->set_name("FluidComposite");
    composite_node->set_mesh(quad);
    composite_node->set_material_override(
        render_material.is_valid() ? render_material : composite_material);
    composite_node->set_cast_shadows_setting(GeometryInstance3D::SHADOW_CASTING_SETTING_OFF);
    // Prevent frustum culling of the POSITION-rewriting fullscreen quad.
    composite_node->set_custom_aabb(AABB(Vector3(-1e6f, -1e6f, -1e6f), Vector3(2e6f, 2e6f, 2e6f)));
    composite_node->set_extra_cull_margin(1e6f);
    add_child(composite_node);
}

void FluidParticleSystem::_destroy_composite_node() {
    if (!composite_node) return;
    // queue_free rather than memdelete: we may be inside a tree notification,
    // and SceneTree validates instance IDs before running its deletion queue,
    // so this stays correct even if the engine frees the child first.
    composite_node->queue_free();
    composite_node = nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Resize the RD color+depth textures + framebuffer to match the main viewport.
// Called every frame from _sync_offscreen_camera().
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_sync_rd_viewport_size() {
    if (!rd || !shared_rd) return;
    if (!rd_render_shader.is_valid()) return;   // nothing to resize into

    Camera3D *main_cam = _resolve_main_camera();
    Vector2i main_size(0, 0);
    if (main_cam) {
        if (Viewport *vp = main_cam->get_viewport())
            main_size = vp->get_visible_rect().size;
    }
    if (main_size.x <= 0 || main_size.y <= 0) return;
    if (main_size == rd_vp_size && rd_framebuffer.is_valid()) return;  // no change

    // Our own in-flight work must finish before we retire the framebuffer it
    // draws into. The main renderer's in-flight frames are handled by routing
    // the old targets through the deferred-free queue instead of freeing now.
    _sync_rd();

    // The pipeline is tied to the framebuffer format, so it goes too. It is
    // local-only and we just synced, so it can be freed immediately.
    _free_local_rid(rd_render_pipeline);

    Vector2i old_size = rd_vp_size;
    _destroy_render_targets(/*deferred=*/true);

    if (!_create_render_targets(main_size)) {
        // Leave rd_vp_size at the old value so the next frame retries rather
        // than concluding the size already matches.
        rd_vp_size = old_size;
        UtilityFunctions::printerr(
            "FluidParticleSystem: render target resize failed; rendering paused.");
        return;
    }
    if (!_create_render_pipeline_state()) {
        UtilityFunctions::printerr(
            "FluidParticleSystem: pipeline rebuild after resize failed; rendering paused.");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Record the particle draw into the RD framebuffer. Called every frame from
// _process() AFTER the compute dispatches (so the storage buffers are current)
// and AFTER _sync_offscreen_camera().
//
// Returns true if commands were recorded. Submission is now the caller's job:
// this function used to own submit(), so an early return here (no camera, for
// example) left the frame's compute lists recorded but never flushed, and they
// accumulated in the device forever.
// ─────────────────────────────────────────────────────────────────────────────
bool FluidParticleSystem::_render_particles_rd() {
    if (!rd) return false;
    if (!rd_render_pipeline.is_valid() || !rd_framebuffer.is_valid()) return false;
    if (!rd_vertex_array.is_valid() || !rd_render_uniform_set.is_valid()) return false;
    if (!rd_camera_ubo.is_valid()) return false;

    Camera3D *main_cam = _resolve_main_camera();
    if (!main_cam) return false;

    Projection  proj        = main_cam->get_camera_projection();
    Transform3D view_xform  = main_cam->get_global_transform();
    Transform3D model_xform = get_global_transform();

    // Camera view matrix = inverse of the camera's global transform.
    Transform3D view_t     = view_xform.affine_inverse();
    // inv_view = camera global transform (world <- view).
    Transform3D inv_view_t = view_xform;
    Projection  inv_proj   = proj.inverse();

    CameraUniforms ubo{};

    // Projection -> float[16] (column-major, matching GLSL mat4).
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
            ubo.proj_matrix[c * 4 + r] = proj.columns[c][r];
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
            ubo.inv_proj_matrix[c * 4 + r] = inv_proj.columns[c][r];

    // Flip Y in clip space: negate row 1 (Y row) of proj and inv_proj.
    ubo.proj_matrix[1]  = -ubo.proj_matrix[1];
    ubo.proj_matrix[5]  = -ubo.proj_matrix[5];
    ubo.proj_matrix[9]  = -ubo.proj_matrix[9];
    ubo.proj_matrix[13] = -ubo.proj_matrix[13];
    ubo.inv_proj_matrix[1]  = -ubo.inv_proj_matrix[1];
    ubo.inv_proj_matrix[5]  = -ubo.inv_proj_matrix[5];
    ubo.inv_proj_matrix[9]  = -ubo.inv_proj_matrix[9];
    ubo.inv_proj_matrix[13] = -ubo.inv_proj_matrix[13];

    // Transform3D -> mat4 (column-major, matching GLSL mat4 memory layout).
    auto xform_to_mat4 = [](const Transform3D &t, float *out) {
        out[0]  = t.basis.rows[0].x;  out[1]  = t.basis.rows[1].x;  out[2]  = t.basis.rows[2].x;  out[3]  = 0.0f;
        out[4]  = t.basis.rows[0].y;  out[5]  = t.basis.rows[1].y;  out[6]  = t.basis.rows[2].y;  out[7]  = 0.0f;
        out[8]  = t.basis.rows[0].z;  out[9]  = t.basis.rows[1].z;  out[10] = t.basis.rows[2].z;  out[11] = 0.0f;
        out[12] = t.origin.x;         out[13] = t.origin.y;         out[14] = t.origin.z;         out[15] = 1.0f;
    };
    xform_to_mat4(view_t,      ubo.view_matrix);
    xform_to_mat4(inv_view_t,  ubo.inv_view_matrix);
    xform_to_mat4(model_xform, ubo.model_matrix);

    Vector3 cam_pos = view_xform.origin;
    ubo.cam_pos_world[0] = cam_pos.x;
    ubo.cam_pos_world[1] = cam_pos.y;
    ubo.cam_pos_world[2] = cam_pos.z;
    ubo._pad0 = 0.0f;

    rd_ubo_bytes.resize(sizeof(CameraUniforms));
    memcpy(rd_ubo_bytes.ptrw(), &ubo, sizeof(CameraUniforms));

    // ── Small push constant (viewport size + far plane, 16 bytes) ───────────
    struct RenderPushConstants {
        float viewport_w;
        float viewport_h;
        float offscreen_far;
        float _pad0;
    };
    RenderPushConstants pc{};
    pc.viewport_w    = (float)rd_vp_size.x;
    pc.viewport_h    = (float)rd_vp_size.y;
    pc.offscreen_far = (float)main_cam->get_far();
    pc._pad0 = 0.0f;

    rd_pc_bytes.resize(sizeof(RenderPushConstants));
    memcpy(rd_pc_bytes.ptrw(), &pc, sizeof(RenderPushConstants));

    rd->buffer_update(rd_camera_ubo, 0, rd_ubo_bytes.size(), rd_ubo_bytes);

    PackedColorArray clear_colors;
    clear_colors.append(Color(0, 0, 0, 0));

    int64_t draw_list = rd->draw_list_begin(
        rd_framebuffer,
        RenderingDevice::DRAW_CLEAR_COLOR_0 | RenderingDevice::DRAW_CLEAR_DEPTH,
        clear_colors, FLUID_DEPTH_CLEAR, 0);
    if (draw_list < 0) {
        UtilityFunctions::printerr("FluidParticleSystem: draw_list_begin failed");
        return false;
    }
    rd->draw_list_bind_render_pipeline(draw_list, rd_render_pipeline);
    rd->draw_list_bind_uniform_set(draw_list, rd_render_uniform_set, 0);
    rd->draw_list_bind_vertex_array(draw_list, rd_vertex_array);
    rd->draw_list_set_push_constant(draw_list, rd_pc_bytes, rd_pc_bytes.size());
    rd->draw_list_draw(draw_list, false, 1);
    rd->draw_list_end();
    return true;
}

// Resolve + cache the main camera by ObjectID so we never dereference a freed
// pointer. Returns nullptr if none found this frame.
Camera3D *FluidParticleSystem::_resolve_main_camera() {
    if (main_camera_id.is_valid()) {
        if (Camera3D *cam = Object::cast_to<Camera3D>(ObjectDB::get_instance(main_camera_id)))
            return cam;
        main_camera_id = ObjectID();   // stale — fall through and re-resolve
    }

    Camera3D *found = nullptr;

    if (Engine::get_singleton()->is_editor_hint()) {
        // Editor: the preview camera lives in the editor's 3D SubViewport.
        // Pick the largest viewport that has a camera.
        SceneTree *tree = get_tree();
        Window *root = tree ? tree->get_root() : nullptr;
        if (root) {
            TypedArray<Viewport> viewports;
            _collect_viewports(root, viewports);
            int best_area = 0;
            for (int i = 0; i < viewports.size(); i++) {
                Viewport *v = Object::cast_to<Viewport>(viewports[i]);
                if (!v) continue;
                Camera3D *cam = v->get_camera_3d();
                if (!cam) continue;
                Vector2i sz = v->get_visible_rect().size;
                int area = sz.x * sz.y;
                if (area > best_area) { best_area = area; found = cam; }
            }
        }
    }

    // Runtime: the camera driving THIS node's viewport (not the root — they can
    // differ if the game renders into a nested SubViewport).
    if (!found) {
        if (Viewport *vp = get_viewport()) found = vp->get_camera_3d();
    }
    // Fallbacks.
    if (!found) {
        SceneTree *tree = get_tree();
        if (Viewport *vp = tree ? tree->get_root() : nullptr) {
            found = vp->get_camera_3d();
            if (!found)
                found = Object::cast_to<Camera3D>(vp->find_child("Camera3D", true, false));
        }
    }

    if (found) main_camera_id = found->get_instance_id();
    return found;
}

// The material actually drawn by composite_node: the user's override if set,
// else the internal composite material. Per-frame uniforms go here.
Ref<ShaderMaterial> FluidParticleSystem::_active_composite_material() const {
    return render_material.is_valid() ? render_material : composite_material;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sync the RD render pipeline to the main camera: resize the offscreen
// textures to match the main viewport, push per-frame uniforms to the composite
// shader, and position the composite quad in front of the camera. Called every
// frame from _process() BEFORE the compute dispatches and the RD render draw.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_sync_offscreen_camera() {
    Camera3D *main_cam = _resolve_main_camera();
    if (!main_cam) return;

    // ── Resize the RD textures to match the main viewport ────────────────────
    // This must happen before we read the projection — get_camera_projection()
    // derives aspect from the viewport the camera sits in.
    _sync_rd_viewport_size();

    // ── Push per-frame uniforms to the ACTIVE composite material ─────────────
    Ref<ShaderMaterial> comp = _active_composite_material();
    if (comp.is_valid()) {
        Projection  proj   = main_cam->get_camera_projection();
        Transform3D xform  = main_cam->get_global_transform();
        // Match the Y-flip applied to the render projection in _render_particles_rd:
        // P' = D*P where D=diag(1,-1,1,1), so P'^{-1} = P^{-1}*D, which negates
        // column 1 of the inverse projection.
        Projection inv_proj = proj.inverse();
        inv_proj.columns[1].x = -inv_proj.columns[1].x;
        inv_proj.columns[1].y = -inv_proj.columns[1].y;
        inv_proj.columns[1].z = -inv_proj.columns[1].z;
        inv_proj.columns[1].w = -inv_proj.columns[1].w;
        comp->set_shader_parameter("offscreen_inv_proj", inv_proj);
        comp->set_shader_parameter("offscreen_inv_view", xform);
        comp->set_shader_parameter("offscreen_far", (double)main_cam->get_far());
        comp->set_shader_parameter("offscreen_near", (double)main_cam->get_near());

        if (rd_vp_size.x > 0 && rd_vp_size.y > 0) {
            comp->set_shader_parameter(
                "fluid_pixel_size",
                Vector2(1.0f / (float)rd_vp_size.x, 1.0f / (float)rd_vp_size.y));
        }
    }

    // Park the fullscreen quad just in front of the camera (position is mostly
    // cosmetic since the vertex shader writes POSITION; the huge AABB set in
    // _ensure_composite_node is what actually keeps it from being culled).
    if (composite_node) {
        if (Node3D *parent3d = Object::cast_to<Node3D>(composite_node->get_parent())) {
            Transform3D cam_xform    = main_cam->get_global_transform();
            Transform3D parent_glob  = parent3d->get_global_transform();
            Transform3D desired      = cam_xform * Transform3D(Basis(), Vector3(0, 0, -0.1f));
            composite_node->set_transform(parent_glob.affine_inverse() * desired);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_enter_tree() {
    if (!_ensure_device()) return;
    _build_gpu_resources();
    _ensure_rd_render_pipeline();
}

void FluidParticleSystem::_exit_tree() {
    _destroy_composite_node();
    _shutdown_gpu();
}

// ─────────────────────────────────────────────────────────────────────────────
// Recursively collect all Viewport descendants of `node` into `out`. Used to
// find the editor's 3D viewport in editor mode.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_collect_viewports(Node *node, TypedArray<Viewport> &out) {
    if (!node) return;
    int child_count = node->get_child_count();
    for (int i = 0; i < child_count; i++) {
        Node *child = node->get_child(i);
        Viewport *vp = Object::cast_to<Viewport>(child);
        if (vp) {
            out.append(vp);
        }
        _collect_viewports(child, out);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tell every FluidSourceBase descendant to drop its pipeline. Sources build
// uniform sets against OUR storage buffers and cache a raw pointer to OUR
// device; if we free those out from under them the next dispatch binds dangling
// RIDs. Must run before any buffer teardown.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_notify_children_gpu_reset(Node *from) {
    Node *root = from ? from : this;
    int n = root->get_child_count();
    for (int i = 0; i < n; i++) {
        Node *child = root->get_child(i);
        if (FluidSourceBase *ss = Object::cast_to<FluidSourceBase>(child)) {
            ss->on_parent_gpu_reset();
        }
        _notify_children_gpu_reset(child);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_build_gpu_resources() {
    if (!_ensure_device()) return;

    // Idempotent: if we are already built, tear down first rather than leaking
    // the previous generation on top of itself.
    if (gpu_ready) _destroy_gpu_resources();

    // Bumping the generation tells sources/sinks their cached RIDs are stale.
    gpu_generation++;

    if (num_particles <= 0 || grid_width <= 0 || grid_height <= 0 || grid_depth <= 0) {
        UtilityFunctions::printerr(
            "FluidParticleSystem: invalid dimensions (particles=", num_particles,
            " grid=", grid_width, "x", grid_height, "x", grid_depth, ")");
        return;
    }

    // ── Standalone storage buffers on the local device ──────────────────────
    // vertex_buf: vec3 position per particle (12 bytes stride)
    // attrib_buf: vec4 color (16 bytes) + vec4 custom0 (16 bytes) = 32 bytes
    {
        const uint32_t nan_bits = 0x7FC00000u;
        float nan_val;
        std::memcpy(&nan_val, &nan_bits, sizeof(float));

        // Vertex buffer: vec3 position per particle.
        uint32_t vbuf_size = num_particles * 3 * sizeof(float);
        PackedByteArray vbuf_data;
        vbuf_data.resize(vbuf_size);
        float *vp = reinterpret_cast<float *>(vbuf_data.ptrw());
        for (int i = 0; i < num_particles; i++) {
            vp[i * 3 + 0] = nan_val;  // inactive sentinel
            vp[i * 3 + 1] = 0.0f;
            vp[i * 3 + 2] = 0.0f;
        }

        // Attribute buffer: vec4 color + vec4 custom0 per particle.
        uint32_t abuf_size = num_particles * 8 * sizeof(float);
        PackedByteArray abuf_data;
        abuf_data.resize(abuf_size);
        float *ap = reinterpret_cast<float *>(abuf_data.ptrw());
        for (int i = 0; i < num_particles; i++) {
            // color (vec4)
            ap[i * 8 + 0] = initial_chunk_color.r;
            ap[i * 8 + 1] = initial_chunk_color.g;
            ap[i * 8 + 2] = initial_chunk_color.b;
            ap[i * 8 + 3] = initial_chunk_color.a;
            // custom0 (vec4): attraction, opacity_fade, neighbors_filled, unused
            ap[i * 8 + 4] = initial_chunk_attraction;
            ap[i * 8 + 5] = 1.0f;
            ap[i * 8 + 6] = 0.0f;
            ap[i * 8 + 7] = 0.0f;
        }

        // If initial chunk is enabled, activate particles in a block.
        if (use_initial_chunk) {
            Color solid_col = Color(0.0f, 0.0f, 0.0f, 17.0f / 255.0f);
            int idx = 0;
            for (int iz = 0; iz < initial_chunk_size.z && idx < num_particles; iz++) {
                for (int iy = 0; iy < initial_chunk_size.y && idx < num_particles; iy++) {
                    for (int ix = 0; ix < initial_chunk_size.x && idx < num_particles; ix++, idx++) {
                        vp[idx * 3 + 0] = initial_chunk_origin.x + ix;
                        vp[idx * 3 + 1] = initial_chunk_origin.y + iy;
                        vp[idx * 3 + 2] = initial_chunk_origin.z + iz;
                        ap[idx * 8 + 0] = solid_col.r;
                        ap[idx * 8 + 1] = solid_col.g;
                        ap[idx * 8 + 2] = solid_col.b;
                        ap[idx * 8 + 3] = solid_col.a;
                        ap[idx * 8 + 4] = initial_chunk_attraction;
                        ap[idx * 8 + 5] = 1.0f;
                        ap[idx * 8 + 6] = 0.0f;
                        ap[idx * 8 + 7] = 0.0f;
                    }
                }
            }
        }

        // Create as vertex buffers (not storage buffers) so vertex_array_create
        // accepts them. Pass BUFFER_CREATION_AS_STORAGE_BIT so compute shaders
        // can also bind them as storage buffers.
        BitField<RenderingDevice::BufferCreationBits> buf_flags =
            RenderingDevice::BUFFER_CREATION_AS_STORAGE_BIT;
        vertex_buf = rd->vertex_buffer_create(vbuf_size, vbuf_data, buf_flags);
        attrib_buf = rd->vertex_buffer_create(abuf_size, abuf_data, buf_flags);
        if (!vertex_buf.is_valid() || !attrib_buf.is_valid()) {
            UtilityFunctions::printerr("FluidParticleSystem: failed to create particle buffers.");
            _destroy_gpu_resources();
            return;
        }

        // Fixed layout (not from mesh format):
        vertex_stride_floats = 3;           // vec3 = 3 floats = 12 bytes
        attrib_stride_words  = 8;           // vec4 + vec4 = 8 words = 32 bytes
        color_offset_words   = 0;           // color at offset 0 in attribute buffer
        custom0_offset_words = 4;           // custom0 at offset 4 words (16 bytes)
    }

    // ── Chunk grid buffer ────────────────────────────────────────────────────
    {
        int64_t cell_count = (int64_t)grid_width * grid_height * grid_depth;
        int64_t buf_size   = cell_count * sizeof(GPUChunkCell);
        PackedByteArray data;
        data.resize(buf_size);
        data.fill(0);
        GPUChunkCell *cells = reinterpret_cast<GPUChunkCell*>(data.ptrw());
        for (int64_t i = 0; i < cell_count; i++) {
            cells[i].occupant = -1;
        }
        chunk_buf = rd->storage_buffer_create(buf_size, data);
    }

    // ── Runnable index buffer (identity permutation) ─────────────────────────
    {
        int64_t buf_size = (int64_t)num_particles * sizeof(int32_t);
        PackedByteArray data;
        data.resize(buf_size);
        int32_t *idx = reinterpret_cast<int32_t*>(data.ptrw());
        for (int i = 0; i < num_particles; i++) idx[i] = i;
        runnable_buf = rd->storage_buffer_create(buf_size, data);
    }

    // ── Sort-key buffer ───────────────────────────────────────────────────────
    {
        int64_t buf_size = (int64_t)num_particles * sizeof(float);
        PackedByteArray data;
        data.resize(buf_size);
        data.fill(0);
        sort_key_buf = rd->storage_buffer_create(buf_size, data);
    }

    if (!chunk_buf.is_valid() || !runnable_buf.is_valid() || !sort_key_buf.is_valid()) {
        UtilityFunctions::printerr("FluidParticleSystem: failed to create simulation buffers.");
        _destroy_gpu_resources();
        return;
    }

    // ── Load and compile compute shaders ─────────────────────────────────────
    clear_shader   = _compile_compute_shader(clear_shader_path);
    physics_shader = _compile_compute_shader(physics_shader_path);
    sortkey_shader = _compile_compute_shader(sortkey_shader_path);

    if (clear_shader.is_valid())   clear_pipeline   = rd->compute_pipeline_create(clear_shader);
    if (physics_shader.is_valid()) physics_pipeline = rd->compute_pipeline_create(physics_shader);
    if (sortkey_shader.is_valid()) sortkey_pipeline = rd->compute_pipeline_create(sortkey_shader);

    // ── Build uniform sets ────────────────────────────────────────────────────
    _rebuild_compute_uniform_sets();

    // ── Rendering ──────────────────────────────────────────────────────────
    // Rendering goes through the RD render pipeline, which reads the vertex/
    // attribute RD buffers directly (see _render_particles_rd).
    gpu_ready = true;
    UtilityFunctions::print("FluidParticleSystem: GPU resources ready (gen ",
                            (int64_t)gpu_generation, ").");
}

// All three compute pipelines share the same buffer bindings:
//   binding 0 -> vertex buffer (position)
//   binding 1 -> attribute buffer (color + custom0)
//   binding 2 -> chunk_grid
//   binding 3 -> runnable_indices
//   binding 4 -> sort_keys
void FluidParticleSystem::_rebuild_compute_uniform_sets() {
    if (!rd) return;

    TypedArray<RDUniform> uniforms;
    uniforms.append(_make_storage_uniform(vertex_buf,   0));
    uniforms.append(_make_storage_uniform(attrib_buf,   1));
    uniforms.append(_make_storage_uniform(chunk_buf,    2));
    uniforms.append(_make_storage_uniform(runnable_buf, 3));
    uniforms.append(_make_storage_uniform(sort_key_buf, 4));

    if (clear_shader.is_valid() && !clear_uniform_set.is_valid())
        clear_uniform_set   = rd->uniform_set_create(uniforms, clear_shader,   0);
    if (physics_shader.is_valid() && !physics_uniform_set.is_valid())
        physics_uniform_set = rd->uniform_set_create(uniforms, physics_shader, 0);
    if (sortkey_shader.is_valid() && !sortkey_uniform_set.is_valid())
        sortkey_uniform_set = rd->uniform_set_create(uniforms, sortkey_shader, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Release the render-side resources. Does NOT touch the device.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_destroy_render_resources(bool deferred_targets) {
    if (!rd) {
        // Device already gone; the RIDs died with it. Just drop the handles,
        // but still detach the Texture2DRDs so no material holds a stale RID.
        if (rd_color_tex_2d.is_valid()) rd_color_tex_2d->set_texture_rd_rid(RID());
        if (rd_depth_tex_2d.is_valid()) rd_depth_tex_2d->set_texture_rd_rid(RID());
        rd_render_uniform_set = RID();
        rd_render_pipeline    = RID();
        rd_vertex_array       = RID();
        rd_framebuffer        = RID();
        rd_color_tex          = RID();
        rd_depth_tex          = RID();
        rd_camera_ubo         = RID();
        rd_render_shader      = RID();
        shared_color_tex      = RID();
        shared_depth_tex      = RID();
        rd_framebuffer_format = -1;
        rd_vertex_format      = -1;
        return;
    }

    _sync_rd();

    _destroy_render_targets(deferred_targets);

    // Dependency order: uniform set -> pipeline -> vertex array -> buffers/shader.
    _free_local_rid(rd_render_uniform_set);
    _free_local_rid(rd_render_pipeline);
    _free_local_rid(rd_vertex_array);
    _free_local_rid(rd_camera_ubo);
    _free_local_rid(rd_render_shader);
    rd_vertex_format = -1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Release the simulation-side resources. Does NOT touch the device.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_destroy_sim_resources() {
    if (!rd) {
        clear_uniform_set = RID();  physics_uniform_set = RID();  sortkey_uniform_set = RID();
        clear_pipeline    = RID();  physics_pipeline    = RID();  sortkey_pipeline    = RID();
        clear_shader      = RID();  physics_shader      = RID();  sortkey_shader      = RID();
        vertex_buf = RID(); attrib_buf = RID(); chunk_buf = RID();
        runnable_buf = RID(); sort_key_buf = RID();
        return;
    }

    _sync_rd();

    _free_local_rid(clear_uniform_set);
    _free_local_rid(physics_uniform_set);
    _free_local_rid(sortkey_uniform_set);

    _free_local_rid(clear_pipeline);
    _free_local_rid(physics_pipeline);
    _free_local_rid(sortkey_pipeline);

    _free_local_rid(clear_shader);
    _free_local_rid(physics_shader);
    _free_local_rid(sortkey_shader);

    _free_local_rid(vertex_buf);
    _free_local_rid(attrib_buf);
    _free_local_rid(chunk_buf);
    _free_local_rid(runnable_buf);
    _free_local_rid(sort_key_buf);
}

// ─────────────────────────────────────────────────────────────────────────────
// Release every GPU resource but KEEP the device, so a rebuild does not churn
// devices. Idempotent.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_destroy_gpu_resources() {
    gpu_ready = false;

    // Sources hold uniform sets built against our buffers and a cached pointer
    // to our device. They must let go before we free anything.
    _notify_children_gpu_reset(nullptr);

    // Sync any in-flight GPU work before freeing resources. Without this,
    // freeing uniform sets / pipelines / buffers still referenced by a pending
    // command buffer can corrupt the device.
    _sync_rd();

    _destroy_render_resources(/*deferred_targets=*/true);
    _destroy_sim_resources();
}

// ─────────────────────────────────────────────────────────────────────────────
// Called from property setters (num_particles, grid_*) when a size-affecting
// property changes while the node is in the tree. Tears down the GPU resources
// and rebuilds them with the new dimensions — reusing the same device.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_rebuild_gpu_resources() {
    if (!is_inside_tree()) return;  // _enter_tree will build on first entry
    if (!rd) return;                // device not up yet

    _destroy_gpu_resources();
    _build_gpu_resources();
    _ensure_rd_render_pipeline();
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_process(double delta) {
    // Wait for the previous frame's GPU work to complete before starting new
    // work. The local device only allows one pending submission at a time.
    _sync_rd();

    // Retire anything whose frame delay has elapsed.
    _process_deferred_frees(false);

    if (!rd) return;

    // Sync the RD render pipeline to the main camera (resize textures, push
    // composite uniforms, position the composite quad). This must run BEFORE
    // the early returns so the composite stays positioned even when the
    // simulation is paused.
    _sync_offscreen_camera();

    // If the render pipeline never came up (bad shader on load, resize failure)
    // keep trying — this makes shader hot-reload recover on its own.
    if (!rd_render_pipeline.is_valid() && vertex_buf.is_valid()) {
        _ensure_rd_render_pipeline();
    }

    bool recorded = false;

    if (gpu_ready && simulation_active) {
        // ── Assemble this frame's displacement FORCE ────────────────────────
        // The shader applies  dp = M*p + origin - p  and subtracts dp from momentum.
        // For correct inertia dp must be a *force* (the frame-to-frame CHANGE in the
        // container's displacement field), not the raw displacement — otherwise
        // steady container motion keeps accelerating the fluid instead of letting it
        // settle into a matching drift.
        //
        //   displacement field this frame : dispC(p) = (Bc - I)p + oc
        //   displacement field last frame : dispP(p) = (Bp - I)p + op
        //   container force = dispC - dispP           : (Bc - Bp)p + (oc - op)
        //   explicit one-shot impulse (kick/stir)     : (Be - I)p + oe
        //
        // Solve  M*p + origin - p  ==  fContainer(p) + fImpulse(p)  for M, origin:
        //   M      = I + (Bc - Bp) + (Be - I) = (Bc - Bp) + Be
        //   origin = (oc - op) + oe

        // 1. Raw container delta transform for THIS frame (frame-to-frame).
        Transform3D cur_xform = get_global_transform();
        Basis   Bc;               // identity by default (first frame)
        Vector3 oc;
        if (has_prev_transform) {
            Transform3D delta_xform = prev_global_transform.affine_inverse() * cur_xform;
            Bc = delta_xform.basis;
            oc = delta_xform.origin;
        }

        // 2. Previous frame's container delta (differencing baseline). Falls back to
        //    identity/zero so the first moving frame reads as pure acceleration.
        Basis   Bp = has_prev_delta_transform ? prev_delta_transform.basis  : Basis();
        Vector3 op = has_prev_delta_transform ? prev_delta_transform.origin : Vector3();

        // 3. Explicit one-shot impulse (add_velocity_impulse / add_rotational_impulse).
        //    Applied fresh this frame and NOT stored into prev_delta, so it isn't
        //    subtracted back out next frame — a genuine impulse.
        bool    had_impulse = delta_impulse_pending;
        Basis   Be;               // identity if none pending
        Vector3 oe;
        if (had_impulse) {
            Be = pending_delta_basis;
            oe = pending_delta_origin;
            delta_impulse_pending = false;
        }

        // 4. Assemble the force matrix M and origin (row-wise on the basis).
        Basis M;
        M.rows[0] = (Bc.rows[0] - Bp.rows[0]) + Be.rows[0];
        M.rows[1] = (Bc.rows[1] - Bp.rows[1]) + Be.rows[1];
        M.rows[2] = (Bc.rows[2] - Bp.rows[2]) + Be.rows[2];
        Vector3 force_origin = (oc - op) + oe;

        // A force exists if the container has history to difference against, or an
        // explicit impulse was queued. Gravity is applied separately in the shader,
        // so it does NOT gate has_delta.
        bool has_delta = has_prev_transform || had_impulse;

        // 5. Roll history forward. Store ONLY the raw container delta (Bc, oc) as
        //    next frame's differencing baseline — never the impulse.
        prev_global_transform    = cur_xform;
        has_prev_transform       = true;
        prev_delta_transform     = Transform3D(Bc, oc);
        has_prev_delta_transform = true;

        // ── Dispatch ────────────────────────────────────────────────────────
        // No per-frame chunk clear: the grid holds persistent pooled velocity.
        _dispatch_physics(Vector3(), M, force_origin, has_delta);
        recorded = true;

        for (int i = 0; i < get_child_count(); i++) {
            FluidSourceBase *ss = Object::cast_to<FluidSourceBase>(get_child(i));
            if (ss) ss->dispatch_into_parent();
        }

        if (frame_count % 60 == 0) _dispatch_sortkey();
        frame_count++;
    }

    // ── Record the particle draw into the RD framebuffer ─────────────────────
    // Runs every frame (even when paused) so the composite shader always has a
    // current fluid texture. Must run AFTER the compute dispatches so the
    // storage buffers are up to date.
    if (_render_particles_rd()) recorded = true;

    // Single submission point. Previously this lived inside
    // _render_particles_rd, so any early return there (no camera resolved, for
    // instance) left this frame's compute lists recorded and unflushed, and
    // they piled up in the device indefinitely.
    if (recorded) {
        rd->submit();
        rd_submitted = true;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a compute list for one pipeline (no submit — the caller owns
// the submit/sync cycle).
// ─────────────────────────────────────────────────────────────────────────────
static void dispatch_compute(RenderingDevice *rd,
                             RID pipeline, RID uniform_set,
                             const PushConstants &pc,
                             uint32_t x_groups, uint32_t y_groups = 1, uint32_t z_groups = 1)
{
    if (!rd) return;
    if (!pipeline.is_valid() || !uniform_set.is_valid()) return;
    if (x_groups == 0 || y_groups == 0 || z_groups == 0) return;

    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(PushConstants));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(PushConstants));

    int64_t cl = rd->compute_list_begin();
    if (cl < 0) return;
    rd->compute_list_bind_compute_pipeline(cl, pipeline);
    rd->compute_list_bind_uniform_set(cl, uniform_set, 0);
    rd->compute_list_set_push_constant(cl, pc_bytes, sizeof(PushConstants));
    rd->compute_list_dispatch(cl, x_groups, y_groups, z_groups);
    rd->compute_list_end();
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_dispatch_clear_grid() {
    if (!rd) return;

    PushConstants pc{};
    pc.grid_w         = grid_width;
    pc.grid_h         = grid_height;
    pc.grid_d         = grid_depth;
    pc.num_particles  = num_particles;

    int64_t cell_count = (int64_t)grid_width * grid_height * grid_depth;
    uint32_t groups = (uint32_t)((cell_count + 63) / 64);
    dispatch_compute(rd, clear_pipeline, clear_uniform_set, pc, groups);
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_dispatch_physics(Vector3 global_add_velocity,
                                             const Basis &delta_basis,
                                             Vector3 delta_origin,
                                             bool has_delta) {
    if (!rd) return;

    // Resolve gravity into grid (local) space.
    Vector3 grav = gravity_vec;
    if (!gravity_local) {
        // Basis::inverse() does a full matrix inverse (cofactor method),
        // so it handles scale correctly (unlike Transform3D::inverse()).
        Basis inv_basis = get_global_transform().basis.inverse();
        grav = inv_basis.xform(grav);
    }

    PushConstants pc{};
    pc.grid_w           = grid_width;
    pc.grid_h           = grid_height;
    pc.grid_d           = grid_depth;
    pc.num_particles    = num_particles;
    pc.surface_tension  = surface_tension;
    pc.water_viscosity  = water_viscosity;
    pc.attraction_force = attraction_force;
    pc.neighbor_mode    = neighbor_mode;
    pc.gravity[0]       = grav.x;
    pc.gravity[1]       = grav.y;
    pc.gravity[2]       = grav.z;
    pc.frame_count      = (int32_t)(frame_count & 0x7fffffff);
    pc.num_runnable     = num_particles;
    pc.vertex_stride_floats = vertex_stride_floats;
    pc.attrib_stride_words  = attrib_stride_words;
    pc.color_offset_words   = color_offset_words;
    pc.custom0_offset_words = custom0_offset_words;
    pc.has_delta        = has_delta ? 1.0f : 0.0f;
    pc.max_occupancy    = max_occupancy;
    pc.back_pressure    = back_pressure;
    // Pack 3x3 basis + origin into 3 vec4 rows (row-major):
    //   row0 = (m00, m01, m02, origin_x)
    //   row1 = (m10, m11, m12, origin_y)
    //   row2 = (m20, m21, m22, origin_z)
    pc.delta_row[0]  = delta_basis.rows[0].x;
    pc.delta_row[1]  = delta_basis.rows[0].y;
    pc.delta_row[2]  = delta_basis.rows[0].z;
    pc.delta_row[3]  = delta_origin.x;
    pc.delta_row[4]  = delta_basis.rows[1].x;
    pc.delta_row[5]  = delta_basis.rows[1].y;
    pc.delta_row[6]  = delta_basis.rows[1].z;
    pc.delta_row[7]  = delta_origin.y;
    pc.delta_row[8]  = delta_basis.rows[2].x;
    pc.delta_row[9]  = delta_basis.rows[2].y;
    pc.delta_row[10] = delta_basis.rows[2].z;
    pc.delta_row[11] = delta_origin.z;

    uint32_t groups = (uint32_t)((num_particles + 63) / 64);
    dispatch_compute(rd, physics_pipeline, physics_uniform_set, pc, groups);
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_dispatch_sortkey() {
    // PushConstants pc{};
    // pc.grid_w        = grid_width;
    // pc.grid_h        = grid_height;
    // pc.grid_d        = grid_depth;
    // pc.num_particles = num_particles;
    // // Camera position is baked into the sort key shader via global_vel[0..2]
    // // We pass (0,0,0) here; in _process you could pass the camera world position.
    // uint32_t groups = (uint32_t)((num_particles + 63) / 64);
    // dispatch_compute(rd, sortkey_pipeline, sortkey_uniform_set, pc, groups);
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::spawn_block(Vector3 origin, int w, int h, int d,
                                      Color color, float attraction)
{
    if (!rd) return;

    int64_t idx = 0;
    for (int iz = 0; iz < d && idx < num_particles; iz++) {
        for (int iy = 0; iy < h && idx < num_particles; iy++) {
            for (int ix = 0; ix < w && idx < num_particles; ix++, idx++) {
                Vector3 pos = Vector3(origin.x + ix, origin.y + iy, origin.z + iz);
                // Write directly into the vertex/attribute buffers. No CPU round-trip.
                // rd->buffer_update(vertex_buf, ...);
                (void)pos;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::add_velocity_impulse(Vector3 impulse) {
    // Compose a pure translation into the pending delta transform.
    // D(p) = pending_delta_basis * p + (pending_delta_origin + impulse)
    if (!delta_impulse_pending) {
        pending_delta_basis = Basis();  // identity
        pending_delta_origin = Vector3();
        delta_impulse_pending = true;
    }
    pending_delta_origin += impulse;
}

void FluidParticleSystem::add_rotational_impulse(Vector3 center, Vector3 axis_amount) {
    // Build a rotation matrix about the given center by the angle |axis_amount|
    // around the axis axis_amount.normalized(). Then compose it into the
    // pending delta transform so the shader applies it per-particle.
    //
    // The rotation transform is: R(p) = M * (p - center) + center
    //   = M * p + (center - M * center)
    // So delta_basis = M, delta_origin = center - M * center.
    float angle = axis_amount.length();
    if (angle < 1e-8f) return;
    Vector3 axis = axis_amount / angle;

    // Rodrigues' rotation formula → 3x3 matrix
    float c = cosf(angle);
    float s = sinf(angle);
    float t = 1.0f - c;
    float x = axis.x, y = axis.y, z = axis.z;
    Basis M(
        Vector3(t*x*x + c,    t*x*y - s*z,  t*x*z + s*y),
        Vector3(t*x*y + s*z,  t*y*y + c,    t*y*z - s*x),
        Vector3(t*x*z - s*y,  t*y*z + s*x,  t*z*z + c)
    );

    Vector3 origin = center - M.xform(center);

    if (!delta_impulse_pending) {
        pending_delta_basis = M;
        pending_delta_origin = origin;
        delta_impulse_pending = true;
    } else {
        // Compose: new_delta(p) = M * (old_delta(p)) + origin
        //   = M * (old_basis * p + old_origin) + origin
        //   = (M * old_basis) * p + (M * old_origin + origin)
        pending_delta_basis = M * pending_delta_basis;
        pending_delta_origin = M.xform(pending_delta_origin) + origin;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::reset_grid() {
    if (!gpu_ready || !rd) return;
    // Drain any pending submission first — the local device allows only one.
    _sync_rd();
    _dispatch_clear_grid();
    rd->submit();
    rd->sync();
    rd_submitted = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sync pending GPU work on the local device. Must be called before freeing any
// RD resource that might still be referenced by an in-flight command buffer
// (uniform sets, pipelines, shaders, buffers, textures). No-op if rd is null
// or nothing has been submitted.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_sync_rd() {
    if (rd && rd_submitted) {
        rd->sync();
        rd_submitted = false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reload the render (vertex+fragment) shader from disk and rebuild the pipeline.
// Safe against compile failures: the old shader is kept if the new one does not
// build. Exposed so pipeline-affecting edits to fluid_depth.glsl can be picked
// up without restarting.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::reload_render_shader() {
    if (!rd) {
        UtilityFunctions::printerr(
            "FluidParticleSystem::reload_render_shader: no RenderingDevice.");
        return;
    }

    _sync_rd();

    RID new_shader = _compile_render_shader();
    if (!new_shader.is_valid()) {
        UtilityFunctions::printerr(
            "FluidParticleSystem: render shader failed to compile; keeping previous.");
        return;
    }

    // Dependency order: uniform set references the shader, pipeline references
    // the shader, so both go before the shader itself.
    _free_local_rid(rd_render_uniform_set);
    _free_local_rid(rd_render_pipeline);
    _free_local_rid(rd_render_shader);

    rd_render_shader = new_shader;

    if (!_create_render_uniform_set() || !_create_render_pipeline_state()) {
        UtilityFunctions::printerr(
            "FluidParticleSystem: render pipeline rebuild failed after shader reload.");
        return;
    }
    UtilityFunctions::print("FluidParticleSystem: render shader reloaded.");
}

// ─────────────────────────────────────────────────────────────────────────────
// Reload a single compute shader by index:
//   0 = clear, 1 = physics, 2 = sortkey
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_reload_compute_shader(int which) {
    if (!rd) {
        UtilityFunctions::printerr(
            "FluidParticleSystem::_reload_compute_shader: RenderingDevice not "
            "available (node not in tree?).");
        return;
    }

    // Pick the path + RID/pipeline/uniform-set slots for this shader.
    const String *path   = nullptr;
    RID          *p_shader       = nullptr;
    RID          *p_pipeline     = nullptr;
    RID          *p_uniform_set  = nullptr;
    const char   *name = "?";
    switch (which) {
        case 0:
            path = &clear_shader_path;   p_shader = &clear_shader;
            p_pipeline = &clear_pipeline; p_uniform_set = &clear_uniform_set;
            name = "clear";   break;
        case 1:
            path = &physics_shader_path; p_shader = &physics_shader;
            p_pipeline = &physics_pipeline; p_uniform_set = &physics_uniform_set;
            name = "physics"; break;
        case 2:
            path = &sortkey_shader_path; p_shader = &sortkey_shader;
            p_pipeline = &sortkey_pipeline; p_uniform_set = &sortkey_uniform_set;
            name = "sortkey"; break;
        default:
            UtilityFunctions::printerr(
                "FluidParticleSystem::_reload_compute_shader: invalid index ", which);
            return;
    }

    // 1. Sync any in-flight GPU work that may reference the resources we're
    //    about to free.
    _sync_rd();

    // 2. Compile the new shader FIRST so a compile failure leaves the old one
    //    intact. _compile_compute_shader uses CACHE_MODE_REPLACE so on-disk
    //    edits are picked up.
    RID new_shader = _compile_compute_shader(*path);
    if (!new_shader.is_valid()) {
        UtilityFunctions::printerr(
            "FluidParticleSystem::_reload_compute_shader: new ", name,
            " shader failed to compile; keeping previous shader.");
        return;
    }

    // 3. Tear down the old resources in dependency order: uniform set first
    //    (references pipeline + shader), then pipeline, then shader.
    _free_local_rid(*p_uniform_set);
    _free_local_rid(*p_pipeline);
    _free_local_rid(*p_shader);

    // 4. Install the new shader and rebuild the pipeline + uniform set against
    //    the same storage buffers all three pipelines share.
    *p_shader   = new_shader;
    *p_pipeline = rd->compute_pipeline_create(*p_shader);
    _rebuild_compute_uniform_sets();

    UtilityFunctions::print(
        "FluidParticleSystem: ", name, " shader reloaded from ", *path);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public reload entry point — kept for the Inspector "Reload Physics Shader"
// tool button and GDScript callers. Delegates to the generalized helper.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::reload_physics_shader() {
    _reload_compute_shader(1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Load a .glsl compute shader resource and compile it into an RD shader RID.
// Uses CACHE_MODE_REPLACE so on-disk edits are picked up on every call.
// ─────────────────────────────────────────────────────────────────────────────
RID FluidParticleSystem::_compile_compute_shader(const String &res_path) {
    if (!rd) return RID();
    if (res_path.is_empty()) return RID();

    Ref<RDShaderFile> sf = ResourceLoader::get_singleton()->load(
        res_path, "", ResourceLoader::CACHE_MODE_REPLACE);
    if (!sf.is_valid()) {
        UtilityFunctions::printerr("FluidParticleSystem: cannot load shader resource: ", res_path);
        return RID();
    }
    Ref<RDShaderSPIRV> spirv = sf->get_spirv();
    if (!spirv.is_valid()) {
        UtilityFunctions::printerr("FluidParticleSystem: no SPIR-V in: ", res_path);
        return RID();
    }
    String err = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_COMPUTE);
    if (!err.is_empty()) {
        UtilityFunctions::printerr("FluidParticleSystem: shader error (", res_path, "):\n", err);
        return RID();
    }
    return rd->shader_create_from_spirv(spirv);
}

// ─────────────────────────────────────────────────────────────────────────────
// Build a single storage-buffer RDUniform.
// ─────────────────────────────────────────────────────────────────────────────
Ref<RDUniform> FluidParticleSystem::_make_storage_uniform(RID buf, uint32_t binding) {
    Ref<RDUniform> u;
    u.instantiate();
    u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
    u->set_binding(binding);
    u->add_id(buf);
    return u;
}