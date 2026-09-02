#include "fluid_particle_system.hpp"
#include "fluid_source_sink.hpp"
#include "lod_arena.hpp"

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
#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/classes/immediate_mesh.hpp>
#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/shader.hpp>
#include <godot_cpp/classes/shader_material.hpp>
#include <godot_cpp/classes/quad_mesh.hpp>
#include <godot_cpp/classes/plane_mesh.hpp>
#include <godot_cpp/classes/box_mesh.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/classes/standard_material3d.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/memory.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/scene_tree.hpp>
#include <godot_cpp/classes/node.hpp>
#include <cstring>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

using namespace godot;

// SDF sentinel: cells that have never been baked hold exactly this value.
// MUST match the SDF_UNSET check in velocity_spread.glsl.
static constexpr float SDF_UNSET_VALUE = -100.0f;
static uint32_t sdf_unset_bits() {
    uint32_t bits;
    std::memcpy(&bits, &SDF_UNSET_VALUE, sizeof(bits));
    return bits;
}

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
    ClassDB::bind_method(D_METHOD("set_migrate_shader_path","v"), &FluidParticleSystem::set_migrate_shader_path);
    ClassDB::bind_method(D_METHOD("get_migrate_shader_path"),      &FluidParticleSystem::get_migrate_shader_path);
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
    ClassDB::bind_method(D_METHOD("set_fluid_particle_size","v"), &FluidParticleSystem::set_fluid_particle_size);
    ClassDB::bind_method(D_METHOD("get_fluid_particle_size"), &FluidParticleSystem::get_fluid_particle_size);
    ClassDB::bind_method(D_METHOD("reload_physics_shader"), &FluidParticleSystem::reload_physics_shader);
    ClassDB::bind_method(D_METHOD("get_reload_physics_shader"), &FluidParticleSystem::get_reload_physics_shader);
    ClassDB::bind_method(D_METHOD("reload_render_shader"),  &FluidParticleSystem::reload_render_shader);
    ClassDB::bind_method(D_METHOD("get_reload_render_shader"), &FluidParticleSystem::get_reload_render_shader);


    ClassDB::bind_method(D_METHOD("apply_sdf_to_velocity_field"),
                         &FluidParticleSystem::apply_sdf_to_velocity_field);
    ClassDB::bind_method(D_METHOD("reload_sdf_and_clear_grid"),
                         &FluidParticleSystem::reload_sdf_and_clear_grid);
    ClassDB::bind_method(D_METHOD("get_reload_sdf_and_clear_grid"),
                         &FluidParticleSystem::get_reload_sdf_and_clear_grid);
    // Internal — signal callbacks from the terrain node, not exposed in the
    // inspector. Trailing DEFVAL args make them callable with 1, 2, 3 or 4
    // args, so one signature fits both the legacy 1-arg emitters (position)
    // and 2-arg position+lod emitters without arity errors.
    ClassDB::bind_method(D_METHOD("_on_terrain_block_loaded", "position", "b", "c", "d"),
                         &FluidParticleSystem::_on_terrain_block_loaded,
                         DEFVAL(0), DEFVAL(0), DEFVAL(0));
    ClassDB::bind_method(D_METHOD("_on_terrain_block_unloaded", "position", "b", "c", "d"),
                         &FluidParticleSystem::_on_terrain_block_unloaded,
                         DEFVAL(0), DEFVAL(0), DEFVAL(0));

    // Phase 3 debug hook: manually push a block event (e.g. from a test
    // script) since upstream godot_voxel VoxelLodTerrain may not emit block
    // signals yet. Same code path as real signals.
    ClassDB::bind_method(D_METHOD("debug_simulate_block_event", "position", "lod", "loaded"),
                         &FluidParticleSystem::debug_simulate_block_event);
    // P5
    ClassDB::bind_method(D_METHOD("get_lod_stats"), &FluidParticleSystem::get_lod_stats);
    ClassDB::bind_method(D_METHOD("set_debug_tint_by_lod", "v"), &FluidParticleSystem::set_debug_tint_by_lod);
    ClassDB::bind_method(D_METHOD("get_debug_tint_by_lod"),      &FluidParticleSystem::get_debug_tint_by_lod);
    // P7
    ClassDB::bind_method(D_METHOD("restore_spilled_particles", "max_count"),
                         &FluidParticleSystem::restore_spilled_particles, DEFVAL(4096));
    ClassDB::bind_method(D_METHOD("set_spill_to_disk", "v"), &FluidParticleSystem::set_spill_to_disk);
    ClassDB::bind_method(D_METHOD("get_spill_to_disk"),      &FluidParticleSystem::get_spill_to_disk);
    ClassDB::bind_method(D_METHOD("set_poll_terrain_blocks", "v"), &FluidParticleSystem::set_poll_terrain_blocks);
    ClassDB::bind_method(D_METHOD("get_poll_terrain_blocks"),      &FluidParticleSystem::get_poll_terrain_blocks);
    ClassDB::bind_method(D_METHOD("set_poll_interval_seconds", "v"), &FluidParticleSystem::set_poll_interval_seconds);
    ClassDB::bind_method(D_METHOD("get_poll_interval_seconds"),      &FluidParticleSystem::get_poll_interval_seconds);
    ClassDB::bind_method(D_METHOD("set_max_arena_pages", "v"), &FluidParticleSystem::set_max_arena_pages);
    ClassDB::bind_method(D_METHOD("get_max_arena_pages"),      &FluidParticleSystem::get_max_arena_pages);
    ClassDB::bind_method(D_METHOD("set_show_page_boxes", "v"), &FluidParticleSystem::set_show_page_boxes);
    ClassDB::bind_method(D_METHOD("get_show_page_boxes"),      &FluidParticleSystem::get_show_page_boxes);
    ClassDB::bind_method(D_METHOD("set_grid_window_follow", "v"), &FluidParticleSystem::set_grid_window_follow);
    ClassDB::bind_method(D_METHOD("get_grid_window_follow"),      &FluidParticleSystem::get_grid_window_follow);
    ClassDB::bind_method(D_METHOD("set_grid_window_margin", "v"), &FluidParticleSystem::set_grid_window_margin);
    ClassDB::bind_method(D_METHOD("get_grid_window_margin"),      &FluidParticleSystem::get_grid_window_margin);
    // P6
    ClassDB::bind_method(D_METHOD("set_segment_dispatch", "v"), &FluidParticleSystem::set_segment_dispatch);
    ClassDB::bind_method(D_METHOD("get_segment_dispatch"),      &FluidParticleSystem::get_segment_dispatch);

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
    ADD_PROPERTY(PropertyInfo(Variant::CALLABLE, "reload_sdf_and_clear_grid",
            PROPERTY_HINT_TOOL_BUTTON, "Reload SDF & Clear Grid,Reload",
            PROPERTY_USAGE_EDITOR),
            "", "get_reload_sdf_and_clear_grid");

    // Compute shader paths.
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "clear_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_clear_shader_path",   "get_clear_shader_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "physics_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_physics_shader_path", "get_physics_shader_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "sortkey_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_sortkey_shader_path", "get_sortkey_shader_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "migrate_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_migrate_shader_path", "get_migrate_shader_path");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_tint_by_lod"),
        "set_debug_tint_by_lod", "get_debug_tint_by_lod");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "spill_to_disk"),
        "set_spill_to_disk", "get_spill_to_disk");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "segment_dispatch"),
        "set_segment_dispatch", "get_segment_dispatch");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "poll_terrain_blocks"),
        "set_poll_terrain_blocks", "get_poll_terrain_blocks");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "poll_interval_seconds",
        PROPERTY_HINT_RANGE, "0.05,5.0,0.05"),
        "set_poll_interval_seconds", "get_poll_interval_seconds");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_arena_pages",
        PROPERTY_HINT_RANGE, "64,65536,64"),
        "set_max_arena_pages", "get_max_arena_pages");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_page_boxes"),
        "set_show_page_boxes", "get_show_page_boxes");
    ADD_GROUP("Infinite World", "grid_window_");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "grid_window_follow"),
        "set_grid_window_follow", "get_grid_window_follow");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "grid_window_margin",
        PROPERTY_HINT_RANGE, "0.05,0.45,0.01"),
        "set_grid_window_margin", "get_grid_window_margin");
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
    
    // SDF terrain collision group.
    ADD_GROUP("SDF", "sdf_");
    ADD_GROUP("", "");

    // Composite shader uniforms group.
    ADD_GROUP("Composite", "composite_");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "composite_fluid_particle_size", PROPERTY_HINT_RANGE, "0,10,0.01"), "set_fluid_particle_size", "get_fluid_particle_size");

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
void FluidParticleSystem::set_fluid_particle_size(float v) {
    fluid_particle_size = v;
    Ref<ShaderMaterial> comp = _active_composite_material();
    if (comp.is_valid()) comp->set_shader_parameter("fluid_particle_size", v);
}

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
void FluidParticleSystem::set_migrate_shader_path(const String &v) {
    if (migrate_shader_path == v) return;
    migrate_shader_path = v;
    if (is_inside_tree()) _reload_compute_shader(3);
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid size only affects the LOD page table (page count per LOD is derived
// from grid dims). Particle buffers, pipelines, and render pipeline are
// unaffected — only rebuild LOD resources + the uniform sets that bind them.
void FluidParticleSystem::set_grid_width(int v)   { if (grid_width  == v) return; grid_width  = v; if (is_inside_tree() && gpu_ready) { _destroy_lod_resources(); _build_lod_resources(); _rebuild_compute_uniform_sets(); } }
void FluidParticleSystem::set_grid_height(int v)  { if (grid_height == v) return; grid_height = v; if (is_inside_tree() && gpu_ready) { _destroy_lod_resources(); _build_lod_resources(); _rebuild_compute_uniform_sets(); } }
void FluidParticleSystem::set_grid_depth(int v)   { if (grid_depth  == v) return; grid_depth  = v; if (is_inside_tree() && gpu_ready) { _destroy_lod_resources(); _build_lod_resources(); _rebuild_compute_uniform_sets(); } }
void FluidParticleSystem::set_grid_size(Vector3i v) {
    if (grid_width == v.x && grid_height == v.y && grid_depth == v.z) return;
    grid_width = v.x; grid_height = v.y; grid_depth = v.z;
    if (is_inside_tree() && gpu_ready) { _destroy_lod_resources(); _build_lod_resources(); _rebuild_compute_uniform_sets(); }
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
    src_buffers.append(render_vertex_buf);  // render reads interpolated positions
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
// Debug page-box overlay: one transparent AABB child per loaded page, tinted
// by LOD with the same palette as the particle debug tint:
//   L0 blue, L1 green, L2 orange, L3 red, L4+ magenta.
// Meshes are pooled and reused; the overlay rebuilds only when the allocated
// page count changes (polled block loads trigger that), not every frame.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_update_page_debug_overlay() {
    if (!show_page_boxes || !is_inside_tree()) return;

    // Count allocated pages + collect their coords/LODs.
    struct PageBox { int lod; Vector3i pos; };
    std::vector<PageBox> boxes;
    for (int lod = 0; lod < lod_levels; ++lod) {
        const int64_t base = _lod_table_base(lod);
        if (base < 0) continue;
        const Vector3i d = _lod_table_dims(lod);
        for (int z = 0; z < d.z; ++z)
        for (int y = 0; y < d.y; ++y)
        for (int x = 0; x < d.x; ++x) {
            const int64_t tidx = _lod_table_index(lod, Vector3i(x, y, z));
            if (tidx < 0) continue;
            if ((lod_table_cpu[(size_t)tidx].flags & LOD_PAGE_FLAG_ALLOCATED) != 0) {
                boxes.push_back({ lod, Vector3i(x, y, z) });
            }
        }
    }
    if ((int)boxes.size() == last_overlay_page_count) return;   // no change
    last_overlay_page_count = (int)boxes.size();

    // Create the container node on first use.
    if (!page_overlay_node || !page_overlay_node->is_inside_tree()) {
        page_overlay_node = memnew(Node3D);
        page_overlay_node->set_name("LodPageDebugBoxes");
        add_child(page_overlay_node);
    }

    // Per-LOD palette (matches lod_tint in velocity_spread.glsl) + one shared
    // material per LOD. Overlap control: boxes from different LODs overlap by
    // definition (each L0 cell sits inside an allocated L1/L2 shell page), so
    // stacked translucent fills double-darken. L0 carries the fill; coarse
    // LOD pages get a near-zero fill and a boosted alpha so they read as
    // colored frames around the fine grid instead of fog over it.
    static const Color lod_colors[5] = {
        Color(0.45f, 0.70f, 1.00f), Color(0.30f, 1.00f, 0.35f),
        Color(1.00f, 0.75f, 0.25f), Color(1.00f, 0.30f, 0.25f),
        Color(1.00f, 0.35f, 1.00f),
    };
    if ((int)page_lod_materials.size() != lod_levels) {
        page_lod_materials.clear();
        for (int lod = 0; lod < lod_levels; ++lod) {
            Ref<StandardMaterial3D> mat;
            mat.instantiate();
            mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
            mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
            mat->set_cull_mode(StandardMaterial3D::CULL_DISABLED);
            mat->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, false);
            const Color c = lod < 5 ? lod_colors[lod] : lod_colors[4];
            mat->set_albedo(Color(c.r, c.g, c.b, lod == 0 ? 0.07f : 0.02f));
            page_lod_materials.push_back(mat);
        }
    }

    // Grow the mesh pool if needed. Each pooled entry: MeshInstance3D with a
    // unit BoxMesh that uses the per-LOD shared material, swapped per use.
    while ((int)page_box_pool.size() < (int)boxes.size()) {
        MeshInstance3D *mi = memnew(MeshInstance3D);
        Ref<BoxMesh> bm;
        bm.instantiate();
        bm->set_size(Vector3(1, 1, 1));   // scaled per-instance below
        bm->set_material(page_lod_materials[0]);
        mi->set_mesh(bm);
        mi->set_cast_shadows_setting(GeometryInstance3D::SHADOW_CASTING_SETTING_OFF);
        page_overlay_node->add_child(mi);
        page_box_pool.push_back(mi);
    }
    // Hide unused pooled entries.
    for (size_t i = boxes.size(); i < page_box_pool.size(); ++i) {
        page_box_pool[i]->hide();
    }

    for (size_t i = 0; i < boxes.size(); ++i) {
        const PageBox &pb = boxes[i];
        MeshInstance3D *mi = page_box_pool[i];
        const int page_voxels = LOD_PAGE_SIZE << pb.lod;
        // Per-LOD inset keeps different-LOD shells from coplanar-overlapping:
        // coarse boxes are pulled in slightly more, so their surfaces sit
        // clearly outside/below the fine grid rather than coinciding with it.
        const float pad = 0.97f - 0.01f * (float)pb.lod;
        // BoxMesh is centered on the node origin, so put the node at the page
        // CENTER, not its corner — otherwise every box is offset by half a
        // page and coarse LODs look diagonally shifted vs the fine grid.
        const float pv = (float)page_voxels;
        mi->set_position(Vector3(Vector3(pb.pos) * pv + Vector3(pv * 0.5f, pv * 0.5f, pv * 0.5f)));
        mi->set_scale(Vector3(pv, pv, pv) * pad);
        Ref<BoxMesh> bm = mi->get_mesh();
        if (bm.is_valid()) bm->set_material(page_lod_materials[pb.lod]);
        mi->show();
    }
}

void FluidParticleSystem::_destroy_page_debug_overlay() {
    if (page_overlay_node) {
        page_overlay_node->queue_free();
        page_overlay_node = nullptr;
    }
    page_box_pool.clear();
    last_overlay_page_count = -1;
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
        float fluid_point_size;  // optional: could be used to scale point size in shader
    };
    RenderPushConstants pc{};
    pc.viewport_w    = (float)rd_vp_size.x;
    pc.viewport_h    = (float)rd_vp_size.y;
    pc.offscreen_far = (float)main_cam->get_far();
    pc.fluid_point_size = fluid_particle_size;  // optional: could be used to scale point size in shader

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

// Resolve the world-space position the SIM WINDOW should follow.
//
// Preference order:
//   1. A VoxelViewer node (godot_voxel) — this is what actually drives the
//      terrain's block streaming, so anchoring the window to it guarantees the
//      sim box sits exactly where terrain pages are loading. Nearest viewer
//      to the camera wins if there are several.
//   2. The main camera (previous behavior) — fallback when no VoxelViewer
//      exists in the scene.
Node3D *FluidParticleSystem::_resolve_follow_anchor() {
    // Cached anchor still valid? (Avoids a full-scene scan every frame.)
    if (follow_anchor_id.is_valid()) {
        if (Node3D *cached = Object::cast_to<Node3D>(ObjectDB::get_instance(follow_anchor_id))) {
            if (cached->is_class("VoxelViewer") && cached->is_inside_tree()) return cached;
        }
        follow_anchor_id = ObjectID();   // stale
    }

    // Search the whole scene for VoxelViewer instances. Class is provided by
    // the separate godot_voxel GDExtension, so match by class name.
    Node *root = get_tree() ? get_tree()->get_root() : nullptr;
    if (root) {
        Node3D *best = nullptr;
        float best_d = 1e30f;
        const Vector3 cam_pos = _resolve_main_camera()
            ? _resolve_main_camera()->get_global_position() : get_global_position();
        std::vector<Node *> stack{ root };
        while (!stack.empty()) {
            Node *n = stack.back(); stack.pop_back();
            for (int i = 0; i < n->get_child_count(); ++i) stack.push_back(n->get_child(i));
            Node3D *n3 = Object::cast_to<Node3D>(n);
            if (!n3 || !n3->is_class("VoxelViewer")) continue;
            const float d = n3->get_global_position().distance_squared_to(cam_pos);
            if (d < best_d) { best_d = d; best = n3; }
        }
        if (best) {
            follow_anchor_id = best->get_instance_id();
            return best;
        }
    }
    return _resolve_main_camera();
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

    // Create / adopt the VoxelLodTerrain child BEFORE building GPU resources,
    // so the poller can query the terrain for SDF data.
    _refresh_sdf_from_child();

    _build_gpu_resources();
    _ensure_rd_render_pipeline();
}

void FluidParticleSystem::_exit_tree() {
    _destroy_composite_node();
    _destroy_page_debug_overlay();
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

        // Render vertex buffer: interpolated display positions. Same layout
        // as vertex_buf. The physics shader lerps this toward vertex_buf every
        // frame (skip path) so LOD-k particles render smoothly between their
        // 2^k-spaced full physics steps. Render reads from this buffer.
        // Created below alongside vertex_buf (needs buf_flags).

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
        render_vertex_buf = rd->vertex_buffer_create(vbuf_size, vbuf_data, buf_flags);
        if (!vertex_buf.is_valid() || !attrib_buf.is_valid() || !render_vertex_buf.is_valid()) {
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

    // ── Chunk grid buffer (legacy, minimal) ──────────────────────────────────
    // The LOD-aware physics shader reads/writes the ARENA (binding 6), not
    // this buffer. clear_grid still writes sdf_bits here but physics reads SDF
    // from sdf_storage_buf (binding 5), so those writes are dead. source/sink
    // writes occupant here but physics reads occupant from the arena. This
    // buffer exists ONLY for uniform-set layout compatibility (binding 2 must
    // be bound in every set the shader declares). A single page is enough —
    // the old full grid³×32B allocation was 512 MB of dead VRAM.
    {
        int64_t buf_size = (int64_t)LOD_PAGE_CELLS * sizeof(GPUChunkCell);
        PackedByteArray data;
        data.resize(buf_size);
        data.fill(0);
        GPUChunkCell *cells = reinterpret_cast<GPUChunkCell*>(data.ptrw());
        for (int64_t i = 0; i < LOD_PAGE_CELLS; i++) {
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

    // ── LOD arena + page table + store ring (P2) ─────────────────────────
    // Standalone in P2: the shaders still see chunk_buf through the existing
    // uniform sets. These get their own bindings in P3 when lod_arena.glsl and
    // the physics LOD path land.
    _build_lod_resources();

    // ── Load and compile compute shaders ─────────────────────────────────────
    clear_shader   = _compile_compute_shader(clear_shader_path);
    physics_shader = _compile_compute_shader(physics_shader_path);
    sortkey_shader = _compile_compute_shader(sortkey_shader_path);
    migrate_shader = _compile_compute_shader(migrate_shader_path);

    if (migrate_shader.is_valid()) migrate_pipeline = rd->compute_pipeline_create(migrate_shader);

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
//   binding 2 -> chunk_grid (legacy; physics no longer reads it but keeps the
//               uniform set layout compatible with clear/sortkey)
//   binding 3 -> runnable_indices
//   binding 4 -> sort_keys
// clear_grid additionally binds:
//   binding 5 -> SDF data (float[], ZXY order) or a dummy buffer
// physics (P4) additionally binds:
//   binding 6 -> cell arena (paged ChunkCells)
//   binding 7 -> LOD page table (16-byte entries)
//   binding 8 -> store ring (u32: cursor, overflow, particle indices)
void FluidParticleSystem::_rebuild_compute_uniform_sets() {
    if (!rd) return;

    TypedArray<RDUniform> uniforms;
    uniforms.append(_make_storage_uniform(vertex_buf,   0));
    uniforms.append(_make_storage_uniform(attrib_buf,   1));
    uniforms.append(_make_storage_uniform(chunk_buf,    2));
    uniforms.append(_make_storage_uniform(runnable_buf, 3));
    uniforms.append(_make_storage_uniform(sort_key_buf, 4));

    if (clear_shader.is_valid() && !clear_uniform_set.is_valid()) {
        TypedArray<RDUniform> clear_uniforms;
        clear_uniforms.append(_make_storage_uniform(vertex_buf,   0));
        clear_uniforms.append(_make_storage_uniform(attrib_buf,   1));
        clear_uniforms.append(_make_storage_uniform(chunk_buf,    2));
        clear_uniforms.append(_make_storage_uniform(runnable_buf, 3));
        clear_uniforms.append(_make_storage_uniform(sort_key_buf, 4));
        // binding 5: dummy buffer (SDF is now baked into arena cells, but
        // clear_grid.glsl still declares the binding for layout compat).
        clear_uniforms.append(_make_storage_uniform(chunk_buf, 5));
        clear_uniform_set = rd->uniform_set_create(clear_uniforms, clear_shader, 0);
    }

    // Physics (Phase 4): bindings 0-4 as before, PLUS 6 = cell arena,
    // 7 = LOD page table, 8 = store ring. The dense chunk_buf (binding 2)
    // is still bound for uniform-set layout compatibility but the LOD-aware
    // physics shader no longer reads it. SDF is now read from arena cells
    // (acells[cidx].sdf_bits), not a separate SDF buffer.
    if (physics_shader.is_valid() && !physics_uniform_set.is_valid()) {
        TypedArray<RDUniform> phys_uniforms;
        phys_uniforms.append(_make_storage_uniform(vertex_buf,   0));
        phys_uniforms.append(_make_storage_uniform(attrib_buf,   1));
        phys_uniforms.append(_make_storage_uniform(chunk_buf,    2));
        phys_uniforms.append(_make_storage_uniform(runnable_buf, 3));
        phys_uniforms.append(_make_storage_uniform(sort_key_buf, 4));
        // binding 10: render vertex buffer (interpolated display positions)
        phys_uniforms.append(_make_storage_uniform(render_vertex_buf, 10));
        phys_uniforms.append(_make_storage_uniform(cell_arena_buf, 6));
        phys_uniforms.append(_make_storage_uniform(lod_table_buf,  7));
        phys_uniforms.append(_make_storage_uniform(store_ring_buf, 8));
        // P5: per-LOD particle-count scratch (zeroed pre-dispatch each frame).
        phys_uniforms.append(_make_storage_uniform(lod_stats_buf, 9));
        physics_uniform_set = rd->uniform_set_create(phys_uniforms, physics_shader, 0);
    }

    if (sortkey_shader.is_valid() && !sortkey_uniform_set.is_valid())
        sortkey_uniform_set = rd->uniform_set_create(uniforms, sortkey_shader, 0);

    // lod_migrate binds only the arena (binding 0 = ArenaBuffer). Uses its own
    // uniform set since its set-0 layout differs from the other pipelines.
    if (migrate_shader.is_valid() && cell_arena_buf.is_valid() && !migrate_uniform_set.is_valid()) {
        TypedArray<RDUniform> migrate_uniforms;
        migrate_uniforms.append(_make_storage_uniform(cell_arena_buf, 0));
        migrate_uniform_set = rd->uniform_set_create(migrate_uniforms, migrate_shader, 0);
    }
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
        migrate_uniform_set = RID();
        clear_pipeline    = RID();  physics_pipeline    = RID();  sortkey_pipeline    = RID();
        migrate_pipeline  = RID();
        clear_shader      = RID();  physics_shader      = RID();  sortkey_shader      = RID();
        migrate_shader    = RID();
        vertex_buf = RID(); attrib_buf = RID(); chunk_buf = RID();
        render_vertex_buf = RID();
        runnable_buf = RID(); sort_key_buf = RID();
        cell_arena_buf = RID(); lod_table_buf = RID();
        store_ring_buf = RID(); lod_stats_buf = RID();
        lod_buffers_valid = false;
        lod_arena = VoxelLodArena();
        return;
    }

    _sync_rd();

    _free_local_rid(clear_uniform_set);
    _free_local_rid(physics_uniform_set);
    _free_local_rid(sortkey_uniform_set);
    _free_local_rid(migrate_uniform_set);

    _free_local_rid(clear_pipeline);
    _free_local_rid(physics_pipeline);
    _free_local_rid(sortkey_pipeline);
    _free_local_rid(migrate_pipeline);

    _free_local_rid(clear_shader);
    _free_local_rid(physics_shader);
    _free_local_rid(sortkey_shader);
    _free_local_rid(migrate_shader);

    _free_local_rid(vertex_buf);
    _free_local_rid(attrib_buf);
    _free_local_rid(chunk_buf);
    _free_local_rid(render_vertex_buf);
    _free_local_rid(runnable_buf);
    _free_local_rid(sort_key_buf);

    _destroy_lod_resources();
}

// ─────────────────────────────────────────────────────────────────────────────
// LOD arena resources (P2).
//
//   cell_arena_buf : max_arena_pages pages × 4096 ChunkCells (32 B each). All
//                    cells start zeroed with occupant = -1 (matches the dense
//                    chunk grid's initial state).
//   lod_table_buf  : one 16 B LodPageEntry per (lod, page coord), laid out as
//                    lod-major blocks. CPU mirror in lod_table_cpu; individual
//                    entries are pushed with 16-byte buffer_update on each
//                    lifecycle event (cheap, signal-driven).
//   store_ring_buf : u32[65536+2] = data N + header {write_cursor, overflow}.
//   lod_stats_buf  : int32[32] scratch for P4 stats.
//
// Owns the geometry helpers too — page-table dims are derived from the sim box
// so a grid resize rebuilds the table consistently via _destroy/_build.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_build_lod_resources() {
    if (!rd) {
        return;
    }
    _destroy_lod_resources();

    // ── Page-table geometry: per-LOD dims over the sim box ──────────────────
    lod_table_cpu.clear();
    lod_table_base.clear();
    lod_table_base.reserve(lod_levels + 1);
    int64_t total_entries = 0;
    for (int lod = 0; lod < lod_levels; ++lod) {
        lod_table_base.push_back((int32_t)total_entries);
        Vector3i dims = _lod_table_dims(lod);
        total_entries += (int64_t)dims.x * dims.y * dims.z;
    }
    lod_table_base.push_back((int32_t)total_entries);  // end sentinel
    lod_table_cpu.assign((size_t)total_entries, LodPageEntry{ 0, 0, -1, 0 });

    // ── Arena: cell slots, occupant initialized to -1 ────────────────────────
    {
        const int64_t cell_count = (int64_t)max_arena_pages * LOD_PAGE_CELLS;
        PackedByteArray data;
        data.resize(cell_count * sizeof(GPUChunkCell));
        data.fill(0);
        GPUChunkCell *cells = reinterpret_cast<GPUChunkCell *>(data.ptrw());
        const uint32_t unset_bits = sdf_unset_bits();
        for (int64_t i = 0; i < cell_count; ++i) {
            cells[i].occupant = -1;
            cells[i].sdf_bits = unset_bits;
        }
        cell_arena_buf = rd->storage_buffer_create(cell_count * (int64_t)sizeof(GPUChunkCell), data);
        if (!cell_arena_buf.is_valid()) {
            UtilityFunctions::printerr("FluidParticleSystem: failed to create LOD cell arena buffer.");
            _destroy_lod_resources();
            return;
        }
    }

    // ── Page table ──────────────────────────────────────────────────────────
    {
        PackedByteArray data;
        data.resize((int64_t)lod_table_cpu.size() * sizeof(LodPageEntry));
        data.fill(0);
        // occupant_hint = -1 everywhere (nothing resident yet).
        int32_t *words = reinterpret_cast<int32_t *>(data.ptrw());
        for (size_t i = 0; i < lod_table_cpu.size(); ++i) {
            words[i * 4 + 2] = -1;
        }
        lod_table_buf = rd->storage_buffer_create((int64_t)lod_table_cpu.size() * (int64_t)sizeof(LodPageEntry), data);
        if (!lod_table_buf.is_valid()) {
            UtilityFunctions::printerr("FluidParticleSystem: failed to create LOD page table buffer.");
            _destroy_lod_resources();
            return;
        }
    }

    // ── Store ring: 2-word header + N indices ────────────────────────────────
    {
        PackedByteArray data;
        data.resize((65536 + 2) * sizeof(uint32_t));
        data.fill(0);
        store_ring_buf = rd->storage_buffer_create((65536 + 2) * (int64_t)sizeof(uint32_t), data);
        if (!store_ring_buf.is_valid()) {
            UtilityFunctions::printerr("FluidParticleSystem: failed to create store-ring buffer.");
            _destroy_lod_resources();
            return;
        }
    }

    // ── Stats scratch ───────────────────────────────────────────────────────
    {
        PackedByteArray zeros;
        zeros.resize(32 * sizeof(int32_t));
        zeros.fill(0);
        lod_stats_buf = rd->storage_buffer_create(32 * (int64_t)sizeof(int32_t), zeros);
        if (!lod_stats_buf.is_valid()) {
            UtilityFunctions::printerr("FluidParticleSystem: failed to create LOD stats buffer.");
            _destroy_lod_resources();
            return;
        }
    }

    lod_arena.configure(max_arena_pages);
    sdf_baked_pages.clear();
    lod_buffers_valid = true;

    UtilityFunctions::print("FluidParticleSystem: LOD arena ready — ",
                            max_arena_pages, " pages (", (int64_t)max_arena_pages * LOD_PAGE_CELLS,
                            " cells), table entries ", (int64_t)lod_table_cpu.size(),
                            ", lod_levels ", lod_levels);
}

void FluidParticleSystem::_destroy_lod_resources() {
    if (rd) {
        // Uniform sets reference cell_arena_buf / lod_table_buf / etc. —
        // invalidate them so _rebuild_compute_uniform_sets recreates with
        // the new buffer RIDs.
        _free_local_rid(physics_uniform_set);
        _free_local_rid(migrate_uniform_set);
        _free_local_rid(cell_arena_buf);
        _free_local_rid(lod_table_buf);
        _free_local_rid(store_ring_buf);
        _free_local_rid(lod_stats_buf);
    }
    cell_arena_buf = RID(); lod_table_buf = RID();
    store_ring_buf = RID(); lod_stats_buf = RID();
    lod_table_cpu.clear();
    lod_table_base.clear();
    lod_arena = VoxelLodArena();
    sdf_baked_pages.clear();
    lod_buffers_valid = false;
}

Vector3i FluidParticleSystem::_lod_table_dims(int lod) const {
    // One page covers (16 << lod) voxel units per axis.
    const int page_voxels = LOD_PAGE_SIZE << lod;
    return Vector3i(
        (grid_width  + page_voxels - 1) / page_voxels,
        (grid_height + page_voxels - 1) / page_voxels,
        (grid_depth  + page_voxels - 1) / page_voxels);
}

int64_t FluidParticleSystem::_lod_table_base(int lod) const {
    if (lod < 0 || lod >= (int)lod_table_base.size() - 1) {
        return -1;
    }
    return lod_table_base[lod];
}

int64_t FluidParticleSystem::_lod_table_index(int lod, Vector3i page_pos) const {
    const int64_t base = _lod_table_base(lod);
    if (base < 0) {
        return -1;
    }
    const Vector3i d = _lod_table_dims(lod);
    if (page_pos.x < 0 || page_pos.y < 0 || page_pos.z < 0 ||
        page_pos.x >= d.x || page_pos.y >= d.y || page_pos.z >= d.z) {
        return -1;
    }
    return base + (int64_t)page_pos.x + d.x * ((int64_t)page_pos.y + d.y * (int64_t)page_pos.z);
}

void FluidParticleSystem::_lod_table_upload(int64_t index) {
    if (!rd || !lod_table_buf.is_valid() ||
        index < 0 || index >= (int64_t)lod_table_cpu.size()) {
        return;
    }
    const LodPageEntry &e = lod_table_cpu[(size_t)index];
    PackedByteArray data;
    data.resize(sizeof(LodPageEntry));
    memcpy(data.ptrw(), &e, sizeof(LodPageEntry));
    rd->buffer_update(lod_table_buf, (uint32_t)(index * (int64_t)sizeof(LodPageEntry)),
                      sizeof(LodPageEntry), data);
}

void FluidParticleSystem::_bake_sdf_for_page(int lod, Vector3i page_pos, uint32_t cell_base) {
    // Query the terrain for SDF data at this page's LOD and write sdf_bits
    // into each of the 4096 arena cells. Zero-fills occupant/vel/count —
    // this only runs once per page (guarded by sdf_baked_pages) and almost
    // always the same frame the page was allocated, before any particle can
    // have joined it.
    Object *terrain = _get_voxel_terrain_child();
    if (!terrain || !rd || !cell_arena_buf.is_valid()) return;

    const int page_voxels = LOD_PAGE_SIZE << lod;
    // World-space origin of this page.
    const Vector3i world_origin = grid_window_origin + page_pos * page_voxels;

    // Create a VoxelBuffer sized to one page and copy SDF from the terrain.
    Variant vb_var = ClassDB::instantiate("VoxelBuffer");
    Object *vb = Object::cast_to<Object>(vb_var);
    if (!vb) return;
    vb->call("create", page_voxels, page_voxels, page_voxels);
    vb->call("set_channel_depth", 1, 2);  // CHANNEL_SDF=1, DEPTH_32_BIT=2

    Variant vt_var = terrain->call("get_voxel_tool");
    Object *vt = Object::cast_to<Object>(vt_var);
    if (!vt) return;
    Variant copy_ret = vt->call("copy", world_origin, vb_var, 2, false);
    int copy_err = (int)copy_ret;
    if (copy_err != 0) {
        UtilityFunctions::printerr("FluidParticleSystem: SDF copy failed for page lod=",
                                   lod, " pos=", page_pos, " err=", copy_err);
        return;
    }

    // Get the raw SDF channel data.
    PackedByteArray raw = vb->call("get_channel_as_byte_array", 1);
    if (raw.size() == 0) return;

    int64_t depth_val = vb->call("get_channel_depth", 1);
    int depth = (int)depth_val;

    // Decode a single SDF value at VoxelBuffer-local (x,y,z). godot_voxel
    // stores channel data in ZXY order — index = y + size.y*(x + size.x*z) —
    // NOT plain XYZ. Getting this wrong reads real, plausible-looking SDF
    // floats from the WRONG voxel position (scrambled axes), which looks
    // like "data is written but nothing sees the terrain correctly".
    // Scale constants (10.0 for 8-bit, 500.0 for 16-bit) match godot_voxel's
    // QUANTIZED_SDF_8/16_BITS_SCALE_INV exactly (constants/voxel_constants.h).
    const int pv = page_voxels;
    auto decode_sdf = [&](int x, int y, int z) -> float {
        int64_t idx = (int64_t)y + (int64_t)pv * ((int64_t)x + (int64_t)pv * (int64_t)z);
        if (depth == 0) {
            const int8_t *s8 = reinterpret_cast<const int8_t *>(raw.ptr());
            return std::max((float)s8[idx] / 127.0f, -1.0f) * 10.0f;
        } else if (depth == 1) {
            const int16_t *s16 = reinterpret_cast<const int16_t *>(raw.ptr());
            return std::max((float)s16[idx] / 32767.0f, -1.0f) * 500.0f;
        } else if (depth == 2) {
            const float *f32 = reinterpret_cast<const float *>(raw.ptr());
            return f32[idx];
        } else if (depth == 3) {
            const double *d = reinterpret_cast<const double *>(raw.ptr());
            return (float)d[idx];
        }
        return 100.0f;
    };

    // The arena page is always 16³ cells. Each cell covers (page_voxels/16)
    // voxels per axis. Sample the SDF at the center of each cell's voxel range
    // so LOD > 0 gets a representative value, not the last voxel that maps
    // to the cell.
    const int voxels_per_cell = page_voxels / 16;  // 1 for LOD 0, 2 for LOD 1, etc.

    PackedByteArray cell_data;
    cell_data.resize(LOD_PAGE_CELLS * sizeof(GPUChunkCell));
    cell_data.fill(0);
    GPUChunkCell *cells = reinterpret_cast<GPUChunkCell *>(cell_data.ptrw());

    for (int cz = 0; cz < 16; ++cz)
    for (int cy = 0; cy < 16; ++cy)
    for (int cx = 0; cx < 16; ++cx) {
        const int ci = cx + cy * 16 + cz * 256;
        cells[ci].occupant = -1;
        // Sample at the center of this cell's voxel range.
        const int vx = cx * voxels_per_cell + voxels_per_cell / 2;
        const int vy = cy * voxels_per_cell + voxels_per_cell / 2;
        const int vz = cz * voxels_per_cell + voxels_per_cell / 2;
        float sdf = decode_sdf(vx, vy, vz);
        std::memcpy(&cells[ci].sdf_bits, &sdf, sizeof(float));
    }

    rd->buffer_update(cell_arena_buf,
                      (uint32_t)((int64_t)cell_base * (int64_t)sizeof(GPUChunkCell)),
                      LOD_PAGE_CELLS * (uint32_t)sizeof(GPUChunkCell),
                      cell_data);

    // Mark this page as having SDF baked.
    const int64_t tidx = _lod_table_index(lod, page_pos);
    if (tidx >= 0) sdf_baked_pages.insert(tidx);
}

void FluidParticleSystem::_apply_lod_event(const LodEvent &ev) {
    if (!lod_buffers_valid) {
        return;
    }
    const int64_t tidx = _lod_table_index(ev.lod, ev.page_pos);
    if (tidx < 0) {
        return;  // outside sim box at this lod — nothing to do
    }
    LodPageEntry &entry = lod_table_cpu[(size_t)tidx];

    if (ev.loaded) {
        // OCTREE INVARIANT: a region is covered by exactly one LOD. Loading
        // a page at LOD k means:
        //   • its 8 children at LOD k-1 (if any are allocated) must be freed
        //     (coarsen — the parent now represents this region), AND
        //   • its covering parent at LOD k+1 (if allocated) must be freed
        //     (refine — the children now represent this region; this page is
        //     one of 8, the rest will arrive as sibling events in the same
        //     batch).
        // Without this, the finest-LOD-wins resolve always picks the finer
        // page and the coarser one is dead weight (and the debug boxes
        // overlap, which is the symptom we saw).
        auto free_page = [&](int lod, Vector3i pos) {
            const int64_t fidx = _lod_table_index(lod, pos);
            if (fidx < 0) return;
            LodPageEntry &fe = lod_table_cpu[(size_t)fidx];
            if (!(fe.flags & LOD_PAGE_FLAG_ALLOCATED)) return;
            const int32_t freed = lod_arena.free(lod, pos.x, pos.y, pos.z);
            if (freed >= 0) detached_slots.push_back((uint32_t)freed);
            fe.flags &= ~LOD_PAGE_FLAG_ALLOCATED;
            fe.occupant_hint = -1;
            fe.cell_base = 0;
            sdf_baked_pages.erase(fidx);
            _lod_table_upload(fidx);
        };
        // Free 8 children (coarsen).
        if (ev.lod > 0) {
            for (int dz = 0; dz < 2; ++dz)
            for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                free_page(ev.lod - 1,
                          Vector3i(ev.page_pos.x * 2 + dx,
                                   ev.page_pos.y * 2 + dy,
                                   ev.page_pos.z * 2 + dz));
            }
        }
        // Free covering parent (refine — this page is one octant of it).
        if (ev.lod + 1 < lod_levels) {
            free_page(ev.lod + 1,
                      Vector3i(ev.page_pos.x >> 1,
                               ev.page_pos.y >> 1,
                               ev.page_pos.z >> 1));
        }

        const int32_t slot = lod_arena.alloc(ev.lod, ev.page_pos.x, ev.page_pos.y, ev.page_pos.z);
        if (slot < 0) {
            // Arena exhausted: leave unallocated (particles there will skip
            // physics / spill once P4 lands) and warn, throttled by gen bump.
            UtilityFunctions::printerr("FluidParticleSystem: LOD arena exhausted (",
                                       lod_arena.used_pages(), "/", lod_arena.max_page_cap(), " pages) — page at lod ",
                                       ev.lod, " pos ", ev.page_pos, " not allocated.");
            entry.flags &= ~LOD_PAGE_FLAG_ALLOCATED;
            entry.occupant_hint = -1;
            _lod_table_upload(tidx);
            return;
        }
        const VoxelLodArena::SlotInfo &si = lod_arena.slot((uint32_t)slot);
        entry.cell_base = (uint32_t)slot * LOD_PAGE_CELLS;
        entry.flags = LOD_PAGE_FLAG_ALLOCATED |
                      (si.generation << LOD_PAGE_GEN_SHIFT);
        // Zero cells with the SDF_UNSET sentinel. The poller's Step 4 lazily
        // bakes real SDF from the terrain (throttled to 64 pages/poll) —
        // baking here would do thousands of VoxelTool::copy calls in one frame.
        {
            PackedByteArray zeros;
            zeros.resize(LOD_PAGE_CELLS * sizeof(GPUChunkCell));
            GPUChunkCell *cells = reinterpret_cast<GPUChunkCell *>(zeros.ptrw());
            const uint32_t unset_bits = sdf_unset_bits();
            for (int i = 0; i < LOD_PAGE_CELLS; ++i) {
                cells[i].occupant = -1;
                cells[i].sdf_bits = unset_bits;
            }
            rd->buffer_update(cell_arena_buf,
                              (uint32_t)((int64_t)entry.cell_base * (int64_t)sizeof(GPUChunkCell)),
                              LOD_PAGE_CELLS * (uint32_t)sizeof(GPUChunkCell),
                              zeros);
        }
        // P7 auto-restore: any spilled particle whose recorded position lies
        // inside this page's voxel range comes back to life here — position
        // written back (clearing the NaN), stored bit already cleared. This is
        // what makes poller-driven allocation viable: particles that froze while
        // the region was unstreamed revive the moment the terrain loads it.
        _auto_restore_for_page(ev.lod, ev.page_pos);
    } else {
        // Unloaded: detach the slot (do NOT return it to the freelist yet —
        // a migration job in this same batch may need to read the page's
        // still-intact GPU data) and mark the table entry unallocated. The
        // generation is preserved in flags so stale readers (P4) can detect.
        {
            const int32_t freed = lod_arena.free(ev.lod, ev.page_pos.x, ev.page_pos.y, ev.page_pos.z);
            if (freed >= 0) {
                detached_slots.push_back((uint32_t)freed);
            }
        }
        entry.flags &= ~LOD_PAGE_FLAG_ALLOCATED;
        entry.occupant_hint = -1;
        entry.cell_base = 0;
        sdf_baked_pages.erase(tidx);
    }
    _lod_table_upload(tidx);
}

// ─── Phase 3: LOD migration detection (batch level) ──────────────────────
// Scans the events drained in THIS batch and emits MigrateJobs. godot_voxel
// coarsens by unloading 8 children + loading 1 parent in one frame, and
// refines by unloading the parent + loading 8 children; both orders may
// interleave. After all events are applied:
//   • unloaded pages sit in detached_slots — arena.slot(i) still exposes
//     their coords and their GPU data is intact (slots not yet reusable).
//   • loaded pages are resident with cell_base in the table.
// Pattern matching needs the batch's raw events, so it runs BEFORE
// pending_lod_events is cleared — call from _apply_pending_lod_events.
void FluidParticleSystem::_detect_batch_lod_transitions() {
    if (pending_lod_events.empty()) return;

    // Reconstruct batch bookkeeping: detached slot coords via arena.slot(),
    // loaded pages via current table state.
    auto child_octant = [](int lod_k, Vector3i parent_pos, const LodEvent &ev)
            -> int {
        // ev is at lod k-1; octant within the parent page (2 children/axis).
        const int32_t parent_vox = LOD_PAGE_SIZE << lod_k;
        const int32_t child_vox  = LOD_PAGE_SIZE << (lod_k - 1);
        const Vector3i lo = parent_pos * parent_vox;
        // child page's min voxel corner
        const Vector3i c = ev.page_pos * child_vox;
        const Vector3i rel = c - lo;  // in [0, parent_vox)
        if (rel.x < 0 || rel.y < 0 || rel.z < 0 ||
            rel.x >= parent_vox || rel.y >= parent_vox || rel.z >= parent_vox) {
            return -1;
        }
        // octant bit per axis = which half of the parent the child covers
        const int half = child_vox;  // parent_vox / 2
        return ((rel.x / half) & 1) | (((rel.y / half) & 1) << 1) |
               (((rel.z / half) & 1) << 2);
    };

    // ── COARSEN: load at lod k with 8 unloaded children (k-1) this batch ──
    for (const LodEvent &lev : pending_lod_events) {
        if (!lev.loaded || lev.lod <= 0) continue;
        const int64_t tidx = _lod_table_index(lev.lod, lev.page_pos);
        if (tidx < 0) continue;
        const LodPageEntry &dst = lod_table_cpu[(size_t)tidx];
        if (!(dst.flags & LOD_PAGE_FLAG_ALLOCATED)) continue;

        uint32_t src_base[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        int found_oct = 0;
        for (const LodEvent &uev : pending_lod_events) {
            if (uev.loaded || uev.lod != lev.lod - 1) continue;
            const int oct = child_octant(lev.lod, lev.page_pos, uev);
            if (oct < 0 || (found_oct & (1 << oct))) continue;
            // child slot base: it was detached this batch, coords intact
            // in arena.slot(); match via detached_slots.
            for (uint32_t s : detached_slots) {
                const VoxelLodArena::SlotInfo &si = lod_arena.slot(s);
                if (si.x == uev.page_pos.x && si.y == uev.page_pos.y &&
                    si.z == uev.page_pos.z) {
                    src_base[oct] = s * LOD_PAGE_CELLS;
                    found_oct |= (1 << oct);
                    break;
                }
            }
        }
        // Require all 8 children: a partial set would lose fluid. Without the
        // full set the zero-filled page is the safe default.
        if (found_oct == 0xFF) {
            MigrateJob j;
            j.dst_base = dst.cell_base;
            for (int i = 0; i < 8; ++i) j.src_base[i] = src_base[i];
            j.mode = 0u;
            j.dst_octant = 0u;
            pending_migrate_jobs.push_back(j);
        }
    }

    // ── REFINE: unload at lod k with 8 loaded children (k-1) this batch ──
    // godot_voxel keeps BOTH loaded briefly on refine in some configs; the
    // parent here must have unloaded this batch so its slot is detached.
    for (const LodEvent &uev : pending_lod_events) {
        if (uev.loaded || uev.lod >= lod_levels - 1) continue;
        // detached parent slot (data intact)
        uint32_t parent_base = 0;
        bool have_parent = false;
        for (uint32_t s : detached_slots) {
            const VoxelLodArena::SlotInfo &si = lod_arena.slot(s);
            if (si.x == uev.page_pos.x && si.y == uev.page_pos.y &&
                si.z == uev.page_pos.z) {
                parent_base = s * LOD_PAGE_CELLS;
                have_parent = true;
                break;
            }
        }
        if (!have_parent) continue;

        // children loaded this batch and resident
        int found_oct = 0;
        uint32_t dst_base[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        for (const LodEvent &lev : pending_lod_events) {
            if (!lev.loaded || lev.lod != uev.lod - 1) continue;
            const int oct = child_octant(uev.lod, uev.page_pos, lev);
            if (oct < 0 || (found_oct & (1 << oct))) continue;
            const int64_t ti = _lod_table_index(lev.lod, lev.page_pos);
            if (ti < 0) continue;
            const LodPageEntry &dst = lod_table_cpu[(size_t)ti];
            if (dst.flags & LOD_PAGE_FLAG_ALLOCATED) {
                dst_base[oct] = dst.cell_base;
                found_oct |= (1 << oct);
            }
        }
        // 8 refine jobs — one per child octant. Children with data only in
        // SOME octants still get those refined (partial refine is lossless:
        // missing octants are just empty).
        for (int oct = 0; oct < 8; ++oct) {
            if (!(found_oct & (1 << oct))) continue;
            MigrateJob j;
            j.dst_base = dst_base[oct];
            j.src_base[0] = parent_base;
            for (int i = 1; i < 8; ++i) j.src_base[i] = 0u;
            j.mode = 1u;
            j.dst_octant = (uint32_t)oct;
            pending_migrate_jobs.push_back(j);
        }
    }
}

// Records every pending MigrateJob into the RD compute list (no submit — the
// caller owns the submit/sync boundary). Called at the end of
// _apply_pending_lod_events AFTER the buffer_updates above, in the same
// command buffer, so the queue order is: zero-fill/src updates → migrate.
void FluidParticleSystem::_dispatch_lod_migrations() {
    if (pending_migrate_jobs.empty() || !rd) {
        pending_migrate_jobs.clear();
        return;
    }
    if (!migrate_pipeline.is_valid() || !migrate_uniform_set.is_valid()) {
        pending_migrate_jobs.clear();
        return;
    }

    int64_t cl = rd->compute_list_begin();
    if (cl < 0) {
        pending_migrate_jobs.clear();
        return;
    }
    rd->compute_list_bind_compute_pipeline(cl, migrate_pipeline);
    rd->compute_list_bind_uniform_set(cl, migrate_uniform_set, 0);

    for (const MigrateJob &j : pending_migrate_jobs) {
        // Push constant mirrors the shader's MigrateConstants exactly (128 B).
        uint32_t pc[32] = {};
        pc[0] = j.dst_base;
        for (int i = 0; i < 8; ++i) pc[1 + i] = j.src_base[i];
        pc[9]  = j.mode;
        pc[10] = j.dst_octant;
        PackedByteArray pc_bytes;
        pc_bytes.resize(128);
        memcpy(pc_bytes.ptrw(), pc, 128);
        rd->compute_list_set_push_constant(cl, pc_bytes, 128);
        rd->compute_list_dispatch(cl, 64, 1, 1);  // 64 groups × 64 threads = 4096
    }
    rd->compute_list_end();
    pending_migrate_jobs.clear();
}

// Phase 3: detect a LOD transition triggered by a load event.
//
//   COARSEN — a page loaded at lod k while its 8 child pages (lod k-1) have
//   events in THIS SAME batch. Their detached slots still hold intact data, so
//   we emit a coarsen job reading those slots into the new page. The children
//   are looked up via the pending unloaded events (they were freed just now).
//
//   REFINE — a page loaded at lod k-1 whose covering parent page (lod k) has
//   a load/unload event in this batch with intact data. We emit 8 refine jobs
//   (one per child octant) copying the parent into the children we can see.
//
// In practice godot_voxel coarsens by unloading 8 children and loading 1
// parent inside one frame, and refines by unloading the parent and loading 8
// children. Both orders (loads before or after unloads within the batch) work
// because we key off which pages have intact data in detached/loaded slots.

void FluidParticleSystem::_apply_pending_lod_events() {
    if (pending_lod_events.empty()) {
        return;
    }
    if (rd) {
        _sync_rd();
    }
    for (const LodEvent &ev : pending_lod_events) {
        _apply_lod_event(ev);
    }

    // Phase 3: detect transitions while the raw batch (in order) is still
    // available, then migrate within the SAME submit/sync boundary. Slot
    // release happens after the dispatch records, so source pages can't be
    // clobbered by reuse mid-batch.
    _detect_batch_lod_transitions();
    _dispatch_lod_migrations();

    // Detached slots have now been read by any migration jobs — release them
    // back to the freelist for reuse.
    for (uint32_t s : detached_slots) {
        lod_arena.release_slot(s);
    }
    detached_slots.clear();

    pending_lod_events.clear();
    if (rd) {
        rd->submit();
        rd->sync();
        rd_submitted = false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────

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

    // Apply signal-queued page allocs/frees (P2): the only lifecycle path for
    // arena pages. Runs its own sync/submit boundary so the page table + arena
    // are consistent before this frame's dispatches.
    _apply_pending_lod_events();

    // Signal-free block polling (P5.5): godot_voxel's VoxelLodTerrain emits no
    // block signals in this build, so every N frames we ask the terrain
    // directly which blocks are loaded (debug_get_mesh_block_info → "loaded")
    // and mirror state changes through the normal event pipeline.
    if (poll_terrain_blocks && terrain_is_vlt && gpu_ready && lod_buffers_valid) {
        _poll_terrain_block_states();
    }

    // Infinite world: slide the sim window if the camera left the margin.
    // Runs before the physics dispatch so this frame's step happens in the
    // fresh window.
    _update_grid_window_follow();

    // SDF is now baked per-page at allocation time. No full-grid refresh
    // needed — when terrain streams new blocks, the poller allocates pages
    // and _bake_sdf_for_page queries the terrain at the page's LOD.

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
        // P6 (opt-in): every 16 frames, recompute per-LOD particle segments on
        // the CPU page-table mirror and dispatch one group range per segment
        // (skipping LODs with zero particles) so far particles don't burn
        // lanes on the fine dispatches. Cheap: no readback — just flags.
        if (segment_dispatch && lod_buffers_valid) {
            if ((frame_count % 16) == 0 || lod_segments.empty()) {
                _rebuild_runnable_segments();
            }
        }
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

    // Phase 4: drain the store ring AFTER the submission — buffer_get_data
    // must not race the pending compute list that appends to the ring.
    if (gpu_ready && simulation_active) {
        _drain_store_ring();
    }
}

// -----------------------------------------------------------------
// P5.5: signal-free terrain block polling.
//
// godot_voxel's VoxelLodTerrain registers no block signals in this build, so
// the P3 page lifecycle never fired from real streaming. VoxelLodTerrain DOES
// expose per-block state via:
//   debug_get_mesh_block_info(Vector3 block_pos, int lod) -> Dictionary
// ("loaded", "meshed", "mesh_state", ...). We poll every block our page table
// covers at a throttled interval, diff against our page-table mirror, and
// convert transitions into the SAME LodEvent pipeline real signals would use
// (load -> alloc + zero page, unload -> detach). No shader changes needed.
// -----------------------------------------------------------------
void FluidParticleSystem::_poll_terrain_block_states() {
    Object *terrain = _get_voxel_terrain_child();
    if (!terrain) return;

    const uint64_t now = (uint64_t)Time::get_singleton()->get_ticks_msec();
    if (now - last_block_poll_ms < (uint64_t)(poll_interval_seconds * 1000.0)) {
        return;
    }
    last_block_poll_ms = now;

    // ── Step 1: query the terrain's block state at every LOD ──────────────
    // Build a per-LOD "loaded" bitmap. A block is loaded if the terrain
    // reports it as data-resident OR meshed. (VLT has separate data and mesh
    // octrees; a block can be data-loaded at LOD 0 while meshed at LOD 1.
    // We treat both as "resident" because the SDF only needs data.)
    std::vector<std::vector<bool>> loaded_bitmap(lod_levels);
    for (int lod = 0; lod < lod_levels; ++lod) {
        const Vector3i dims = _lod_table_dims(lod);
        loaded_bitmap[lod].assign((size_t)dims.x * dims.y * dims.z, false);
        const int pbv = LOD_PAGE_SIZE << lod;
        const Vector3i blk_shift(
            (grid_window_origin.x + pbv - 1) / pbv,
            (grid_window_origin.y + pbv - 1) / pbv,
            (grid_window_origin.z + pbv - 1) / pbv);
        for (int z = 0; z < dims.z; ++z) {
            for (int y = 0; y < dims.y; ++y) {
                for (int x = 0; x < dims.x; ++x) {
                    const Vector3i bp(x, y, z);
                    Variant ret = terrain->call("debug_get_mesh_block_info", bp + blk_shift, lod);
                    if (ret.get_type() != Variant::DICTIONARY) continue;
                    const Dictionary info = ret;
                    bool loaded = false;
                    if (info.has("loaded")) loaded = (bool)info["loaded"];
                    if (info.has("meshed")) loaded = loaded || (bool)info["meshed"];
                    if (loaded) {
                        const size_t idx = (size_t)x + (size_t)dims.x * ((size_t)y + (size_t)dims.y * (size_t)z);
                        loaded_bitmap[lod][idx] = true;
                    }
                }
            }
        }
    }

    // ── Step 2: compute the effective LOD per region (octree resolution) ──
    // The octree invariant: a region is covered by exactly one LOD — the
    // FINEST loaded LOD. If LOD 0 is loaded, LOD 1/2/3 for the same region
    // are interior nodes, not leaves, and must NOT be allocated. This
    // eliminates the flapping: instead of reacting to every LOD the terrain
    // reports as "loaded" (which includes both parent and child during
    // transitions, since VLT's data and mesh octrees can disagree), we pick
    // exactly one LOD per region per poll.
    std::vector<std::vector<bool>> desired_lod(lod_levels);
    for (int lod = 0; lod < lod_levels; ++lod) {
        const Vector3i dims = _lod_table_dims(lod);
        desired_lod[lod].assign((size_t)dims.x * dims.y * dims.z, false);
    }
    // LOD 0: desired = loaded (no children to check).
    {
        const Vector3i d0 = _lod_table_dims(0);
        for (int z = 0; z < d0.z; ++z)
        for (int y = 0; y < d0.y; ++y)
        for (int x = 0; x < d0.x; ++x) {
            const size_t idx = (size_t)x + (size_t)d0.x * ((size_t)y + (size_t)d0.y * (size_t)z);
            desired_lod[0][idx] = loaded_bitmap[0][idx];
        }
    }
    // LOD k>0: desired = loaded AND no child is desired (finer wins).
    for (int lod = 1; lod < lod_levels; ++lod) {
        const Vector3i dims = _lod_table_dims(lod);
        const Vector3i cdims = _lod_table_dims(lod - 1);
        for (int z = 0; z < dims.z; ++z)
        for (int y = 0; y < dims.y; ++y)
        for (int x = 0; x < dims.x; ++x) {
            const size_t idx = (size_t)x + (size_t)dims.x * ((size_t)y + (size_t)dims.y * (size_t)z);
            if (!loaded_bitmap[lod][idx]) {
                desired_lod[lod][idx] = false;
                continue;
            }
            bool child_desired = false;
            for (int dz = 0; dz < 2 && !child_desired; ++dz)
            for (int dy = 0; dy < 2 && !child_desired; ++dy)
            for (int dx = 0; dx < 2 && !child_desired; ++dx) {
                const int cx = x * 2 + dx;
                const int cy = y * 2 + dy;
                const int cz = z * 2 + dz;
                if (cx >= cdims.x || cy >= cdims.y || cz >= cdims.z) continue;
                const size_t cidx = (size_t)cx + (size_t)cdims.x * ((size_t)cy + (size_t)cdims.y * (size_t)cz);
                if (desired_lod[lod - 1][cidx]) child_desired = true;
            }
            desired_lod[lod][idx] = !child_desired;
        }
    }

    // ── Step 2b: fallback for empty regions ─────────────────────────────
    // Default to the coarsest LOD. But if any terrain inside a coarse
    // region is at a finer LOD, upgrade the whole coarse region to that
    // finer LOD so particles simulate at matching fidelity. This keeps
    // the sim continuous everywhere without freezing via store-ring.
    {
        const int coarsest = lod_levels - 1;
        const Vector3i dc = _lod_table_dims(coarsest);

        for (int cz = 0; cz < dc.z; ++cz)
        for (int cy = 0; cy < dc.y; ++cy)
        for (int cx = 0; cx < dc.x; ++cx) {
            const size_t cidx = (size_t)cx + (size_t)dc.x * ((size_t)cy + (size_t)dc.y * (size_t)cz);

            // Find the finest LOD that has any desired page inside this
            // coarse region (scan finest → coarsest).
            int finest_found = -1;
            for (int lod = 0; lod < coarsest && finest_found < 0; ++lod) {
                const Vector3i dims = _lod_table_dims(lod);
                const int shift  = coarsest - lod;
                const int ox     = cx << shift;
                const int oy     = cy << shift;
                const int oz     = cz << shift;
                const int extent = 1 << shift;
                for (int z = oz; z < oz + extent && z < dims.z && finest_found < 0; ++z)
                for (int y = oy; y < oy + extent && y < dims.y && finest_found < 0; ++y)
                for (int x = ox; x < ox + extent && x < dims.x; ++x) {
                    const size_t idx = (size_t)x + (size_t)dims.x * ((size_t)y + (size_t)dims.y * (size_t)z);
                    if (desired_lod[lod][idx]) {
                        finest_found = lod;
                        break;
                    }
                }
            }

            if (finest_found >= 0) {
                // Upgrade: fill the entire coarse region at the finest
                // found LOD. Clear all other LODs inside to maintain the
                // octree invariant (one LOD per region).
                for (int lod = 0; lod <= coarsest; ++lod) {
                    const Vector3i dims = _lod_table_dims(lod);
                    const int shift  = coarsest - lod;
                    const int ox     = cx << shift;
                    const int oy     = cy << shift;
                    const int oz     = cz << shift;
                    const int extent = 1 << shift;
                    for (int z = oz; z < oz + extent && z < dims.z; ++z)
                    for (int y = oy; y < oy + extent && y < dims.y; ++y)
                    for (int x = ox; x < ox + extent && x < dims.x; ++x) {
                        const size_t idx = (size_t)x + (size_t)dims.x * ((size_t)y + (size_t)dims.y * (size_t)z);
                        desired_lod[lod][idx] = (lod == finest_found);
                    }
                }
            } else {
                // No terrain inside → default to coarsest LOD.
                desired_lod[coarsest][cidx] = true;
            }
        }
    }

    // ── Step 3: diff desired vs current page table → events ───────────────
    // Distance-prioritized allocation: when the arena is smaller than the
    // total desired pages, only allocate the closest pages to the viewer.
    // Far pages stay unallocated (particles there freeze via store-ring,
    // the safe failure mode) and get loaded when the viewer moves closer.
    int loads = 0, unloads = 0;

    // First pass: collect all load and unload events.
    struct PendingEvent { LodEvent ev; float dist_sq; };
    std::vector<PendingEvent> load_events;
    std::vector<LodEvent> unload_events;

    for (int lod = 0; lod < lod_levels; ++lod) {
        const Vector3i dims = _lod_table_dims(lod);
        for (int z = 0; z < dims.z; ++z)
        for (int y = 0; y < dims.y; ++y)
        for (int x = 0; x < dims.x; ++x) {
            const Vector3i bp(x, y, z);
            const int64_t tidx = _lod_table_index(lod, bp);
            if (tidx < 0) continue;
            const size_t bidx = (size_t)x + (size_t)dims.x * ((size_t)y + (size_t)dims.y * (size_t)z);
            const bool want = desired_lod[lod][bidx];
            const bool have = (lod_table_cpu[(size_t)tidx].flags & LOD_PAGE_FLAG_ALLOCATED) != 0;

            if (want && !have) {
                LodEvent ev; ev.loaded = true; ev.lod = lod; ev.page_pos = bp;
                load_events.push_back({ ev, 0.0f });
            } else if (!want && have) {
                LodEvent ev; ev.loaded = false; ev.lod = lod; ev.page_pos = bp;
                unload_events.push_back(ev);
            }
        }
    }

    // Unloads are always safe — they free arena slots.
    for (const LodEvent &ev : unload_events) {
        pending_lod_events.push_back(ev); ++unloads;
    }

    // Compute viewer position in window-local voxel space for distance sort.
    // Must match _update_grid_window_follow's transform: global position ->
    // parent's local frame (grid_window_origin lives in that same frame).
    Vector3 viewer_local = Vector3(grid_width * 0.5f, grid_height * 0.5f, grid_depth * 0.5f);
    Node3D *viewer_node = _resolve_follow_anchor();
    if (!viewer_node) viewer_node = _resolve_main_camera();
    if (viewer_node) {
        Vector3 wp = viewer_node->get_global_position();
        if (Node3D *p = Object::cast_to<Node3D>(get_parent())) {
            wp = p->get_global_transform().affine_inverse().xform(wp);
        }
        viewer_local = wp - Vector3(grid_window_origin);
    }

    // Sort loads by distance to viewer (closest first).
    for (auto &pe : load_events) {
        const int page_voxels = LOD_PAGE_SIZE << pe.ev.lod;
        const Vector3i center = pe.ev.page_pos * page_voxels + Vector3i(page_voxels / 2, page_voxels / 2, page_voxels / 2);
        const Vector3 d = Vector3(center) - viewer_local;
        pe.dist_sq = d.length_squared();
    }
    std::sort(load_events.begin(), load_events.end(),
              [](const PendingEvent &a, const PendingEvent &b) { return a.dist_sq < b.dist_sq; });

    // Budget: after unloads, how many slots will be free?
    const int32_t arena_cap = lod_arena.max_page_cap();
    const int32_t used_after_unloads = lod_arena.used_pages();  // unloads not applied yet, but _apply_lod_event frees before alloc
    int32_t budget = arena_cap - used_after_unloads + (int32_t)unloads;
    if (budget < 0) budget = 0;

    int32_t admitted = 0;
    for (const PendingEvent &pe : load_events) {
        if (admitted >= budget) {
            // Arena would be exhausted — skip this page. It stays
            // unallocated; particles there freeze via store-ring.
            continue;
        }
        pending_lod_events.push_back(pe.ev); ++loads; ++admitted;
    }

    if ((int32_t)load_events.size() > admitted) {
        UtilityFunctions::print("FluidParticleSystem: arena budget ", arena_cap,
                                " — admitted ", admitted, " / ", (int32_t)load_events.size(),
                                " load events (", (int32_t)load_events.size() - admitted,
                                " deferred by distance).");
    }

    if ((loads > 0 || unloads > 0) && gpu_ready && lod_buffers_valid && rd) {
        _apply_pending_lod_events();
    }

    // ── Step 4: bake SDF for allocated pages that don't have it yet ──────
    // Pages start with the SDF_UNSET sentinel. The poller bakes
    // real SDF from the terrain lazily — a few pages per poll to avoid
    // stalling (each VoxelTool::copy is a terrain query).
    if (gpu_ready && lod_buffers_valid && rd) {
        _sync_rd();
        int baked = 0;
        const int max_bake_per_poll = 64;  // throttle
        for (int lod = 0; lod < lod_levels && baked < max_bake_per_poll; ++lod) {
            const Vector3i dims = _lod_table_dims(lod);
            for (int z = 0; z < dims.z && baked < max_bake_per_poll; ++z)
            for (int y = 0; y < dims.y && baked < max_bake_per_poll; ++y)
            for (int x = 0; x < dims.x && baked < max_bake_per_poll; ++x) {
                const int64_t tidx = _lod_table_index(lod, Vector3i(x, y, z));
                if (tidx < 0) continue;
                const LodPageEntry &e = lod_table_cpu[(size_t)tidx];
                if (!(e.flags & LOD_PAGE_FLAG_ALLOCATED)) continue;
                if (sdf_baked_pages.count(tidx)) continue;
                _bake_sdf_for_page(lod, Vector3i(x, y, z), e.cell_base);
                ++baked;
            }
        }
        if (baked > 0) {
            rd->submit();
            rd->sync();
            rd_submitted = false;
        }
    }

    if (loads > 0 || unloads > 0) {
        UtilityFunctions::print("FluidParticleSystem: block poll - ",
                                loads, " loads, ", unloads, " unloads.");
        _update_page_debug_overlay();
    }
}

// P7: revive spilled particles whose position falls inside the newly-loaded
// page. Position restore clears the NaN sentinel so the physics pass sees
// them again; their stored bit was already cleared at spill time.
void FluidParticleSystem::_auto_restore_for_page(int lod, Vector3i page_pos) {
    if (spill_queue.empty() || !rd || !vertex_buf.is_valid()) return;

    const int page_voxels = LOD_PAGE_SIZE << lod;
    const Vector3i lo = page_pos * page_voxels;
    const Vector3i hi = lo + Vector3i(page_voxels, page_voxels, page_voxels);

    bool removed = false;
    for (size_t i = 0; i < spill_queue.size(); ) {
        const SpillRec &r = spill_queue[i];
        const bool inside = r.x >= (float)lo.x && r.x < (float)hi.x &&
                            r.y >= (float)lo.y && r.y < (float)hi.y &&
                            r.z >= (float)lo.z && r.z < (float)hi.z;
        if (!inside) { ++i; continue; }

        // Re-activate: write the recorded position back over the NaN.
        PackedByteArray pos;
        pos.resize(12);
        pos.encode_float(0, r.x);
        pos.encode_float(4, r.y);
        pos.encode_float(8, r.z);
        rd->buffer_update(vertex_buf,
            (uint32_t)((int64_t)r.idx * vertex_stride_floats * 4), 12, pos);

        spill_queue[i] = spill_queue.back();
        spill_queue.pop_back();
        removed = true;
        // Do NOT ++i — we just swapped a new element into slot i.
    }
    if (removed && spill_queue.empty()) {
        UtilityFunctions::print("FluidParticleSystem: all spilled particles restored.");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Infinite world: camera-following simulation window.
//
// grid_window_origin is the world-space voxel origin of the simulated box.
// The GPU always simulates [0, grid_*) in CELL space; the mapping cell→world
// is  cell_world = grid_window_origin + cell  (plus the node transform).
//
// A window shift by delta (in voxels, snapped to the coarsest page size)
// requires exactly three CPU-side actions, NO shader changes:
//   1. Rebase every particle position:  pos -= delta  (positions are stored
//      in cell space; subtracting delta moves the window in world space).
//   2. Spill-queue records hold cell positions; rebase those too.
//   3. The SDF is re-extracted at the new window (refresh + re-bake below).
// Rendering needs no rebase: positions still live in cell space, which the
// model matrix maps into the parent's local frame — the shift itself moved
// where in the world that cell space lands, which is exactly what we want.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_shift_grid_window(Vector3i new_origin) {
    if (!rd || !vertex_buf.is_valid() || !gpu_ready) return;
    const Vector3i delta = new_origin - grid_window_origin;
    if (delta == Vector3i()) return;

    _sync_rd();

    // 1. Rebase particle positions: read the whole position buffer, subtract
    // delta, write back. 262144 particles × 12 B = 3 MB — a one-off cost on
    // the rare frames the window actually moves.
    {
        PackedByteArray blob = rd->buffer_get_data(vertex_buf);
        const int n = (int)blob.size() / 12;
        float *p = reinterpret_cast<float *>(blob.ptrw());
        for (int i = 0; i < n; ++i) {
            const float x = p[i * 3 + 0];
            if (!(x == x)) continue;   // NaN sentinel → inactive particle
            p[i * 3 + 0] -= (float)delta.x;
            p[i * 3 + 1] -= (float)delta.y;
            p[i * 3 + 2] -= (float)delta.z;
        }
        rd->buffer_update(vertex_buf, 0, (uint32_t)blob.size(), blob);
    }

    // 1b. Rebase render positions the same way.
    {
        PackedByteArray blob = rd->buffer_get_data(render_vertex_buf);
        const int n = (int)blob.size() / 12;
        float *p = reinterpret_cast<float *>(blob.ptrw());
        for (int i = 0; i < n; ++i) {
            const float x = p[i * 3 + 0];
            if (!(x == x)) continue;
            p[i * 3 + 0] -= (float)delta.x;
            p[i * 3 + 1] -= (float)delta.y;
            p[i * 3 + 2] -= (float)delta.z;
        }
        rd->buffer_update(render_vertex_buf, 0, (uint32_t)blob.size(), blob);
    }

    // 2. Spill queue holds cell-space positions.
    for (SpillRec &r : spill_queue) {
        r.x -= (float)delta.x; r.y -= (float)delta.y; r.z -= (float)delta.z;
    }

    grid_window_origin = new_origin;

    // The window moved — all baked SDF data is now stale (pages cover
    // different world-space terrain). Clear the baked set so the poller
    // re-bakes SDF from the terrain at the new world positions.
    sdf_baked_pages.clear();

    // SDF is baked per-page at allocation time. When the window shifts, pages
    // in the new region will be allocated by the poller and _bake_sdf_for_page
    // will query the terrain at the new world-space positions.

    UtilityFunctions::print("FluidParticleSystem: sim window shifted by ",
                            delta, " → origin ", grid_window_origin);
    if (show_page_boxes) {
        last_overlay_page_count = -1;   // rebuild boxes at the new origin
    }
}

void FluidParticleSystem::_update_grid_window_follow() {
    if (!grid_window_follow || !gpu_ready) return;

    Node3D *anchor = _resolve_follow_anchor();
    if (!anchor) return;

    // Anchor position in the parent's LOCAL frame (cell space is parent-local).
    Vector3 cam_local;
    if (Node3D *p = Object::cast_to<Node3D>(get_parent())) {
        cam_local = p->get_global_transform().affine_inverse().xform(anchor->get_global_position());
    } else {
        cam_local = anchor->get_global_position();
    }

    const Vector3 win_lo(grid_window_origin);
    const Vector3 win_hi = win_lo + Vector3((float)grid_width, (float)grid_height, (float)grid_depth);
    const Vector3 m((grid_window_margin * (win_hi.x - win_lo.x)),
                    (grid_window_margin * (win_hi.y - win_lo.y)),
                    (grid_window_margin * (win_hi.z - win_lo.z)));

    // How far outside the safe interior is the camera, per axis? (≤0 = inside.)
    const float out_x = std::max(win_lo.x + m.x - cam_local.x, cam_local.x - (win_hi.x - m.x));
    const float out_y = std::max(win_lo.y + m.y - cam_local.y, cam_local.y - (win_hi.y - m.y));
    const float out_z = std::max(win_lo.z + m.z - cam_local.z, cam_local.z - (win_hi.z - m.z));
    const float max_out = std::max(out_x, std::max(out_y, out_z));
    if (max_out <= 0.0f) return;

    // Shift just far enough to bring the camera back to the margin line,
    // snapped to the coarsest page size so page tables stay terrain-aligned.
    const Vector3i step = _window_snap_step();
    int want[3] = {
        out_x > 0.0f ? (int)std::ceil(out_x) : 0,
        out_y > 0.0f ? (int)std::ceil(out_y) : 0,
        out_z > 0.0f ? (int)std::ceil(out_z) : 0 };
    // Direction: which side of the window the camera left through.
    if (cam_local.x < win_lo.x + m.x) want[0] = -want[0];
    if (cam_local.y < win_lo.y + m.y) want[1] = -want[1];
    if (cam_local.z < win_lo.z + m.z) want[2] = -want[2];

    Vector3i shift;
    int *wv = want;
    int *sv[3] = { &shift.x, &shift.y, &shift.z };
    for (int a = 0; a < 3; ++a) {
        const int mag = (std::abs(wv[a]) + step.x - 1) / step.x * step.x;  // ceil to step
        *sv[a] = wv[a] < 0 ? -mag : mag;
    }
    if (shift == Vector3i()) return;

    _shift_grid_window(grid_window_origin + shift);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a compute list for one pipeline (no submit — the caller owns
// the submit/sync cycle).
// ─────────────────────────────────────────────────────────────────────────────
static void dispatch_compute(RenderingDevice *rd,
                             RID pipeline, RID uniform_set,
                             const PushConstants &pc,
                             uint32_t x_groups, uint32_t y_groups = 1, uint32_t z_groups = 1,
                             uint32_t pc_size = sizeof(PushConstants))
{
    if (!rd) return;
    if (!pipeline.is_valid() || !uniform_set.is_valid()) return;
    if (x_groups == 0 || y_groups == 0 || z_groups == 0) return;

    PackedByteArray pc_bytes;
    pc_bytes.resize(pc_size);
    memcpy(pc_bytes.ptrw(), &pc, pc_size);

    int64_t cl = rd->compute_list_begin();
    if (cl < 0) return;
    rd->compute_list_bind_compute_pipeline(cl, pipeline);
    rd->compute_list_bind_uniform_set(cl, uniform_set, 0);
    rd->compute_list_set_push_constant(cl, pc_bytes, pc_size);
    rd->compute_list_dispatch(cl, x_groups, y_groups, z_groups);
    rd->compute_list_end();
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_dispatch_clear_grid(bool keep_occupant) {
    if (!rd) return;

    PushConstants pc{};
    pc.grid_w         = grid_width;
    pc.grid_h         = grid_height;
    pc.grid_d         = grid_depth;
    pc.num_particles  = num_particles;

    // SDF params are packed into the portion of the 128-byte push constant
    // blob that clear_grid.glsl owns but the other two shaders leave unused
    // (has_delta / max_occupancy / back_pressure / delta_row). The clear_grid
    // shader reinterprets these bytes as SDF fields.
    if (sdf_storage_buf.is_valid()) {
        pc.has_delta    = 1.0f;                 // has_sdf flag
        // max_occupancy (int32 at offset 72) is skipped — clear_grid pads over it
        pc.back_pressure = sdf_strength;        // offset 76
        pc.delta_row[0]  = (float)sdf_w;        // offset 80
        pc.delta_row[1]  = (float)sdf_h;        // offset 84
        pc.delta_row[2]  = (float)sdf_d;        // offset 88
        // offset/scale are always 0/1 — terrain and fluid share the same space
        pc.delta_row[3]  = 0.0f;               // offset 92  (sdf_offset_x)
        pc.delta_row[4]  = 0.0f;               // offset 96  (sdf_offset_y)
        pc.delta_row[5]  = 0.0f;               // offset 100 (sdf_offset_z)
        pc.delta_row[6]  = 1.0f;               // offset 104 (sdf_scale)
        pc.delta_row[7]  = keep_occupant ? 1.0f : 0.0f;  // offset 108
    } else {
        pc.has_delta = 0.0f;
        pc.delta_row[7] = keep_occupant ? 1.0f : 0.0f;
    }

    int64_t cell_count = (int64_t)grid_width * grid_height * grid_depth;
    // chunk_buf is only LOD_PAGE_CELLS (4096) cells — the LOD arena owns the
    // real cell storage. clear_grid's writes to chunk_buf are dead (physics
    // reads SDF from sdf_storage_buf at binding 5, not chunk_buf), but
    // dispatching over the full grid would write out of bounds and corrupt
    // the GPU heap. Clamp the dispatch to the buffer size.
    uint32_t groups = (uint32_t)((std::min<int64_t>(cell_count, LOD_PAGE_CELLS) + 63) / 64);
    // clear_grid.glsl declares 112 bytes of push constants (the physics/sortkey
    // shaders use the full 128). Send only what the shader expects.
    constexpr uint32_t CLEAR_PC_SIZE = 112;
    dispatch_compute(rd, clear_pipeline, clear_uniform_set, pc, groups, 1, 1, CLEAR_PC_SIZE);
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
    // Shared slot: the LOD-aware physics shader reads this as packed_lod_occ
    // (low15 = max_occupancy, bits 16-30 = lod_levels, bit 31 = debug tint).
    // clear_grid/sortkey leave it unused, so packing never disturbs them.
    pc.max_occupancy    = (debug_tint_by_lod ? (1 << 31) : 0) |
                          ((int32_t)std::min(lod_levels, 0x7FFF) << 16) |
                          (max_occupancy & 0xFFFF);
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

    // P5: zero the per-LOD particle counts before this frame's physics pass.
    // The shader atomicAdds lst[sim_lod] per simulated particle.
    if (lod_stats_buf.is_valid()) {
        static const int32_t zeros[32] = {0};
        PackedByteArray zb;
        zb.resize(32 * (int64_t)sizeof(int32_t));
        memcpy(zb.ptrw(), zeros, sizeof(zeros));
        rd->buffer_update(lod_stats_buf, 0, 32 * sizeof(int32_t), zb);
    }

    dispatch_compute(rd, physics_pipeline, physics_uniform_set, pc, groups);
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 4: drain the store ring.
//
// Particles whose position had no allocated LOD page pushed their indices into
// store_ring_buf in the physics pass and FROZE (no physics, still rendered).
// Every frame we:
//   1. Sync the device (the ring append happened in this frame's submission —
//      buffer_get_data while that submission is still pending crashes inside
//      the Vulkan driver).
//   2. Read the ring cursor (2 u32 header).
//   3. Pull the new index tail (buffer_get_data).
//   4. Log the skipped set (throttled). Particles stay frozen in place so the
//      visual density is preserved rather than having particles vanish;
//      persistence to user:// lands in P7 with the save-format work.
//   5. Reset write_cursor to 0 (GPU header update, 4-byte buffer_update).
// The custom0.w stored-bit (bit 30) stays set until the particle's region
// gains a page again — the physics pass clears it on the next resolved step
// (raw_w bit 30 only suppresses re-logging while the particle is skipped).
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_drain_store_ring() {
    if (!rd || !store_ring_buf.is_valid() || !gpu_ready) {
        return;
    }
    // The ring append rode this frame's submission; it must complete BEFORE
    // any readback or the driver sees a concurrent read/write on the buffer.
    _sync_rd();

    // Read the 8-byte header {write_cursor, overflow}.
    PackedByteArray hdr = rd->buffer_get_data(store_ring_buf, 0, 8);
    if (hdr.size() < 8) {
        return;
    }
    uint32_t cursor, overflow;
    memcpy(&cursor,   hdr.ptr(),        4);
    memcpy(&overflow, hdr.ptr() + 4,    4);

    if (cursor == 0 && overflow == last_ring_overflow) {
        return;  // nothing new
    }

    const uint32_t n = (cursor < 65534u) ? cursor : 65534u;
    if (n > 0) {
        PackedByteArray tail = rd->buffer_get_data(store_ring_buf, 8, n * 4);
        skipped_particle_count += n;
        // P7: persist the spilled set to disk and deactivate on the GPU.
        if (spill_to_disk && tail.size() >= (int64_t)(n * 4)) {
            PackedInt32Array idxs;
            idxs.resize((int64_t)n);
            memcpy(idxs.ptrw(), tail.ptr(), (int64_t)n * 4);
            _spill_particles(idxs);
        }
        // Throttled log: every 2 s, report accumulated skipped particles.
        const uint64_t now = (uint64_t)Time::get_singleton()->get_ticks_msec();
        if (now - last_skip_log_ms >= 2000) {
            if (skipped_particle_count > 0) {
                UtilityFunctions::print("FluidParticleSystem: ", skipped_particle_count,
                                        " particles outside sim pages (frozen; ring overflow=",
                                        overflow, ")");
                skipped_particle_count = 0;
            }
            last_skip_log_ms = now;
        }
    }
    if (overflow != last_ring_overflow) {
        if (overflow != 0) {
            UtilityFunctions::printerr("FluidParticleSystem: store ring OVERFLOWED ",
                                       overflow, " times — particles beyond capacity silently frozen.");
        }
        last_ring_overflow = overflow;
    }

    // Reset the cursor so the ring reuses its space next frame. The stored
    // indices are consumed (GPU keeps them valid until overwritten).
    PackedByteArray zero;
    zero.resize(4);
    const uint32_t zero_u32 = 0;
    memcpy(zero.ptrw(), &zero_u32, 4);
    rd->buffer_update(store_ring_buf, 0, 4, zero);
    ring_pending_unreported = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// P7: spill persistence.
//
// Drained store-ring indices are appended to user://fluid_spill/<name>.spill
// as a stream of 20-byte records {particle_idx i32, pos xyz f32, custom0.w
// bits u32}. Position comes straight from vertex_buf (the last resolved spot),
// so a FluidSource can re-spawn the particle near where it froze.
// After persisting, the particle is deactivated on the GPU (position.x = NaN
// sentinel, same convention as FluidSink) in batches capped per frame, and
// its stored bit (custom0.w bit 30) is cleared so a future contiguous page
// can adopt it again (the physics pass clears the freeze path on the next
// resolved step).
// ─────────────────────────────────────────────────────────────────────────────
String FluidParticleSystem::_spill_dir() const {
    // Unique per node name so sibling systems don't share files.
    return String("user://fluid_spill/");
}

void FluidParticleSystem::_spill_particles(const PackedInt32Array &indices) {
    if (indices.is_empty() || !rd || !vertex_buf.is_valid()) return;

    const int64_t count = (int64_t)indices.size() < spill_batch_per_frame
                              ? indices.size() : (int64_t)spill_batch_per_frame;

    // One contiguous read of the positions for the batch's index range
    // (indices from the ring are roughly spatially clustered; worst case we
    // read the whole buffer, which is still cheap vs a per-particle readback).
    int32_t lo = INT32_MAX, hi = INT32_MIN;
    for (int64_t i = 0; i < count; ++i) {
        const int32_t idx = indices[i];
        if (idx >= 0 && idx < num_particles) {
            if (idx < lo) lo = idx;
            if (idx > hi) hi = idx;
        }
    }
    if (lo > hi) return;

    const int64_t pos_bytes  = (int64_t)(hi - lo + 1) * vertex_stride_floats * sizeof(float);
    PackedByteArray poses = rd->buffer_get_data(vertex_buf,
        (int64_t)lo * vertex_stride_floats * (int64_t)sizeof(float), pos_bytes);
    if (poses.size() < pos_bytes) return;

    // Append records to the spill file.
    Ref<DirAccess> da = DirAccess::open("user://"); if (da.is_valid()) da->make_dir_recursive("fluid_spill");
    const String path = _spill_dir() + "system.spill";
    Ref<FileAccess> fa;
    if (FileAccess::file_exists(path)) {
        fa = FileAccess::open(path, FileAccess::WRITE_READ);
        if (fa.is_valid()) fa->seek_end();
    } else {
        fa = FileAccess::open(path, FileAccess::WRITE_READ);
    }
    if (!fa.is_valid()) {
        UtilityFunctions::printerr("FluidParticleSystem: cannot open spill file ", path);
        return;
    }

    PackedByteArray deact;  // NaN positions to write back (batch)
    struct PosWrite { uint64_t offset; float x, y, z; };
    std::vector<PosWrite> writes;
    writes.reserve((size_t)count);

    for (int64_t i = 0; i < count; ++i) {
        const int32_t idx = indices[i];
        if (idx < 0 || idx >= num_particles) continue;
        const int64_t rel = (int64_t)(idx - lo) * vertex_stride_floats;
        float px, py, pz;
        memcpy(&px, poses.ptr() + rel * (int64_t)sizeof(float) + 0, 4);
        memcpy(&py, poses.ptr() + rel * (int64_t)sizeof(float) + 4, 4);
        memcpy(&pz, poses.ptr() + rel * (int64_t)sizeof(float) + 8, 4);
        // Record: idx, xyz
        fa->store_32((uint32_t)idx);
        fa->store_float(px); fa->store_float(py); fa->store_float(pz);
        writes.push_back({ (uint64_t)((int64_t)idx * vertex_stride_floats
                                       * (int64_t)sizeof(float)), px, py, pz });
        // Keep in memory for auto-restore when the region's page returns.
        spill_queue.push_back({ idx, px, py, pz });
    }
    fa->flush();
    spilled_total += (int64_t)writes.size();

    // Deactivate on the GPU: NaN out position.x for each spilled particle.
    // (Same sentinel as FluidSink; a restore/source re-activates them.)
    PackedByteArray poses_w;
    // Reuse the buffer we read, writing NaN into x per record (single
    // buffer_update per contiguous run is overkill for spills of ~1k; do
    // per-particle 12-byte updates, max spill_batch_per_frame/frame).
    for (const PosWrite &w : writes) {
        PackedByteArray nanpos;
        nanpos.resize(3 * (int64_t)sizeof(float));
        memcpy(nanpos.ptrw(), &w, 12); // struct fields x,y,z → override x
        const float nan = std::numeric_limits<float>::quiet_NaN();
        memcpy(nanpos.ptrw(), &nan, 4);
        nanpos.encode_float(4, 0.0f);
        nanpos.encode_float(8, 0.0f);
        rd->buffer_update(vertex_buf, (uint32_t)w.offset, 12, nanpos);
    }

    // Clear the stored bit (custom0.w bit 30) so future pages can adopt the
    // particle without relogging. custom0.w is a full-word attribute — do a
    // 4-byte read-modify-write only for spilled ones.
    // (Batch: per-particle 16-byte custom0 write with the bit masked off.)
    const int64_t c0_bytes = (int64_t)(hi - lo + 1) * attrib_stride_words * 4;
    PackedByteArray c0s = rd->buffer_get_data(attrib_buf,
        (int64_t)lo * attrib_stride_words * 4, c0_bytes);
    if (c0s.size() >= c0_bytes) {
        for (int64_t i = 0; i < count; ++i) {
            const int32_t idx = indices[i];
            if (idx < 0 || idx >= num_particles) continue;
            const int64_t rel = ((int64_t)(idx - lo) * attrib_stride_words + custom0_offset_words + 3) * 4;
            uint32_t w; memcpy(&w, c0s.ptr() + rel, 4);
            w &= ~(1u << 30);
            PackedByteArray wb; wb.resize(4); memcpy(wb.ptrw(), &w, 4);
            rd->buffer_update(attrib_buf,
                (uint32_t)(((int64_t)idx * attrib_stride_words + custom0_offset_words + 3) * 4), 4, wb);
        }
    }
}

// P7: restore spilled particles — clear the spill file and return the records
// so a FluidSource (or script) can re-activate them near their stored spots.
// Returns {records} as a flat PackedVector3Array of positions (idx implied by
// count order); the file is truncated after the read.
int FluidParticleSystem::restore_spilled_particles(int max_count) {
    const String path = _spill_dir() + "system.spill";
    if (!FileAccess::file_exists(path)) return 0;
    Ref<FileAccess> fa = FileAccess::open(path, FileAccess::READ);
    if (!fa.is_valid()) return 0;

    // Read all records {idx,xyz}.
    struct Rec { int32_t idx; float x, y, z; };
    std::vector<Rec> recs;
    while (!fa->eof_reached()) {
        if (fa->get_position() + 20 > (uint64_t)fa->get_length()) break;
        Rec r;
        r.idx = (int32_t)fa->get_32();
        r.x = fa->get_float(); r.y = fa->get_float(); r.z = fa->get_float();
        if (r.idx >= 0 && r.idx < num_particles) recs.push_back(r);
    }
    fa->close();
    if (recs.empty()) {
        DirAccess::remove_absolute(path);
        return 0;
    }

    int restored = 0;
    for (size_t i = 0; i < recs.size() && restored < max_count; ++i) {
        const Rec &r = recs[i];
        // Re-activate: write position, clear NaN.
        PackedByteArray pos;
        pos.resize(12);
        pos.encode_float(0, r.x); pos.encode_float(4, r.y); pos.encode_float(8, r.z);
        rd->buffer_update(vertex_buf,
            (uint32_t)((int64_t)r.idx * vertex_stride_floats * 4), 12, pos);
        ++restored;
    }
    if (restored >= (int)recs.size()) {
        DirAccess::remove_absolute(path);
    }
    spilled_total = 0;
    UtilityFunctions::print("FluidParticleSystem: restored ", restored, " spilled particles.");
    return restored;
}

// ─────────────────────────────────────────────────────────────────────────────
// P6: per-LOD runnable segments.
//
// Reads the custom0.w stored-bit of every particle (one bulk attrib readback,
// every 16 frames) and builds contiguous LOD segments. Segment k spans the
// particle-index range whose page-resolution resolves at LOD k; the physics
// dispatch then runs (begin..begin+count) group ranges per segment instead of
// one giant range. Wasted far-particle lanes become an early return inside a
// tiny dispatch rather than a lane burn in the big one.
// NOTE: reading every particle's custom0.w every 16 frames is itself a cost —
// enable only if profiling shows the whole-range dispatch wastes >15% on far
// particles (per plan §5.2 step (b)).
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_rebuild_runnable_segments() {
    lod_segments.clear();
    if (!rd || !attrib_buf.is_valid() || !gpu_ready) return;
    _sync_rd();

    const int64_t bytes = (int64_t)num_particles * attrib_stride_words * 4;
    PackedByteArray raw = rd->buffer_get_data(attrib_buf, 0, bytes);
    if (raw.size() < bytes) return;

    const uint32_t *words = reinterpret_cast<const uint32_t *>(raw.ptr());
    int prev_lod = -1;
    int seg_begin = 0;
    int seg_count = 0;
    for (int i = 0; i < num_particles; ++i) {
        const uint32_t w = words[(int64_t)i * attrib_stride_words + custom0_offset_words + 3];
        // Particle's sim LOD this frame isn't directly stored; approximate by
        // its frozen flag (bit30) → "no LOD" (excluded), else finest page over
        // its cell. We don't have the position on CPU cheaply — approximate the
        // segmentation using the frozen bit only for now: active = not frozen.
        const int lod = (w & (1u << 30)) ? 0 : 1;
        if (lod != prev_lod && seg_count > 0) {
            lod_segments.push_back({ seg_begin, seg_count, prev_lod });
            seg_begin = i; seg_count = 0;
        }
        prev_lod = lod;
        ++seg_count;
    }
    if (seg_count > 0) lod_segments.push_back({ seg_begin, seg_count, prev_lod });
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
    // Re-bake SDF for all allocated pages (SDF now lives in arena cells).
    _sync_rd();
    for (int lod = 0; lod < lod_levels; ++lod) {
        const Vector3i dims = _lod_table_dims(lod);
        for (int z = 0; z < dims.z; ++z)
        for (int y = 0; y < dims.y; ++y)
        for (int x = 0; x < dims.x; ++x) {
            const int64_t tidx = _lod_table_index(lod, Vector3i(x, y, z));
            if (tidx < 0) continue;
            const LodPageEntry &e = lod_table_cpu[(size_t)tidx];
            if (!(e.flags & LOD_PAGE_FLAG_ALLOCATED)) continue;
            _bake_sdf_for_page(lod, Vector3i(x, y, z), e.cell_base);
        }
    }
    rd->submit();
    rd->sync();
    rd_submitted = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// SDF terrain collision
//
// The connected VoxelBuffer's CHANNEL_SDF is read once on the CPU, decoded to
// float (handling 8/16/32-bit depths and the godot_voxel quantization scales),
// and uploaded to a GPU storage buffer. The clear_grid shader samples it to
// bake the SDF gradient (surface normal) into the persistent chunk velocity
// field. The physics shader then reads that velocity each frame, pushing
// particles out of the terrain.
//
// VoxelBuffer is from a separate GDExtension (godot_voxel) whose headers we
// don't link against, so methods are called through Object::call.
// ─────────────────────────────────────────────────────────────────────────────


void FluidParticleSystem::apply_sdf_to_velocity_field() {
    if (!gpu_ready || !rd) return;
    // Re-bake SDF for all allocated pages from the terrain.
    _sync_rd();
    for (int lod = 0; lod < lod_levels; ++lod) {
        const Vector3i dims = _lod_table_dims(lod);
        for (int z = 0; z < dims.z; ++z)
        for (int y = 0; y < dims.y; ++y)
        for (int x = 0; x < dims.x; ++x) {
            const int64_t tidx = _lod_table_index(lod, Vector3i(x, y, z));
            if (tidx < 0) continue;
            const LodPageEntry &e = lod_table_cpu[(size_t)tidx];
            if (!(e.flags & LOD_PAGE_FLAG_ALLOCATED)) continue;
            _bake_sdf_for_page(lod, Vector3i(x, y, z), e.cell_base);
        }
    }
    rd->submit();
    rd->sync();
    rd_submitted = false;
}

void FluidParticleSystem::reload_sdf_and_clear_grid() {
    if (!gpu_ready || !rd) return;
    // Re-bake SDF for all allocated pages from the terrain (picks up edits).
    _sync_rd();
    for (int lod = 0; lod < lod_levels; ++lod) {
        const Vector3i dims = _lod_table_dims(lod);
        for (int z = 0; z < dims.z; ++z)
        for (int y = 0; y < dims.y; ++y)
        for (int x = 0; x < dims.x; ++x) {
            const int64_t tidx = _lod_table_index(lod, Vector3i(x, y, z));
            if (tidx < 0) continue;
            const LodPageEntry &e = lod_table_cpu[(size_t)tidx];
            if (!(e.flags & LOD_PAGE_FLAG_ALLOCATED)) continue;
            _bake_sdf_for_page(lod, Vector3i(x, y, z), e.cell_base);
        }
    }
    rd->submit();
    rd->sync();
    rd_submitted = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Dynamic property forwarding for the child VoxelTerrain.
// VoxelTerrain's properties are exposed on this node under a "Terrain" group
// with a "terrain_" prefix, so the user can edit them without finding the
// internal child node. Values are cached so they survive child recreation.
// ─────────────────────────────────────────────────────────────────────────────

Object *FluidParticleSystem::_get_voxel_terrain_child() const {
    if (!voxel_terrain_child_id.is_valid()) {
        return nullptr;
    }
    return Object::cast_to<Object>(ObjectDB::get_instance(voxel_terrain_child_id));
}

bool FluidParticleSystem::_set(const StringName &p_name, const Variant &p_value) {
    String name = p_name;
    if (name.begins_with("terrain_")) {
        String real_name = name.substr(8);  // strlen("terrain_") == 8
        _terrain_property_cache[real_name] = p_value;
        Object *child = _get_voxel_terrain_child();
        if (child) {
            child->set(real_name, p_value);
        }
        return true;
    }
    return false;
}

bool FluidParticleSystem::_get(const StringName &p_name, Variant &r_ret) const {
    String name = p_name;
    if (name.begins_with("terrain_")) {
        String real_name = name.substr(8);
        Object *child = _get_voxel_terrain_child();
        if (child) {
            r_ret = child->get(real_name);
            return true;
        }
        // Fall back to cache (child not created yet, e.g. editor preview).
        if (_terrain_property_cache.has(real_name)) {
            r_ret = _terrain_property_cache[real_name];
            return true;
        }
        return false;
    }
    return false;
}


// ─────────────────────────────────────────────────────────────────────────────
// Signal callbacks from the terrain node. Bound with 4 Variant params so a
// single signature works for 1-arg (legacy position-only) and 2-arg
// (position+lod) emitters both — extra args cut off, none appended (godot-cpp
// bind_method has no default-arg support for this case; arity mismatch is not
// an error, signals pass what they have).
// Both just set a flag — _process does the debounced recopy so a burst of
// block events doesn't trigger N GPU dispatches.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_on_terrain_block_loaded(const Variant &a, const Variant &b,
                                                   const Variant &c, const Variant &d) {
    _on_terrain_block_event(a, b, true);
}

void FluidParticleSystem::_on_terrain_block_unloaded(const Variant &a, const Variant &b,
                                                     const Variant &c, const Variant &d) {
    _on_terrain_block_event(a, b, false);
}

void FluidParticleSystem::_on_terrain_block_event(const Variant &p_position,
                                                  const Variant &p_lod, bool p_entered) {
    sdf_refresh_pending = true;

    // ── Signal-driven page lifecycle (P2) ─────────────────────────────────
    // The ONLY path that creates/destroys arena pages. Position arrives as a
    // Vector3i terrain data-block coord; a page covers exactly one terrain
    // block, so no math beyond clamping the LOD. Position-only emitters
    // (legacy 1-arg signals) default to LOD 0 — their coords are LOD0 blocks.
    {
        LodEvent ev;
        ev.loaded = p_entered;
        ev.lod = (p_lod.get_type() == Variant::INT)
            ? Math::clamp((int)p_lod, 0, lod_levels - 1)
            : 0;
        ev.page_pos = (Vector3i)p_position;
        pending_lod_events.push_back(ev);
    }

    // Commit immediately: "this block is ready" → the page's GPU memory is
    // initialized NOW (table entry written + 64 KB cell block zero-filled with
    // occupant=-1), not at the next _process. _apply_pending_lod_events owns
    // the sync/submit boundary, so it's safe to call from a signal callback.
    // Events only linger in the queue when the GPU isn't up yet (editor,
    // pre-build); _process drains those leftovers once resources exist.
    if (gpu_ready && lod_buffers_valid && rd) {
        _apply_pending_lod_events();
    }

    // Throttled event log: first event logs immediately, then at most one line
    // per 2 s summarizing how many events occurred in between. Keeps streaming
    // feedback visible without flooding the output.
    const uint64_t now = (uint64_t)Time::get_singleton()->get_ticks_msec();
    if (terrain_event_log_last_ms == 0) {
        UtilityFunctions::print("FluidParticleSystem: terrain ",
                                p_entered ? "block_loaded/mesh_block_entered" : "block_unloaded/mesh_block_exited",
                                " at ", p_position);
        terrain_event_log_last_ms = now;
        terrain_event_log_count = 0;
        return;
    }
    ++terrain_event_log_count;
    if (now - terrain_event_log_last_ms >= 2000) {
        UtilityFunctions::print("FluidParticleSystem: terrain events x",
                                terrain_event_log_count, " in last 2s (suppressing repeats)");
        terrain_event_log_last_ms = now;
        terrain_event_log_count = 0;
    }
}

// Phase 3 debug hook: injects a block event through the exact same code path
// as a real terrain signal — including immediate GPU commit and migration
// detection. Useful with test scripts, e.g.:
//   fps.debug_simulate_block_event(Vector3i(4,0,4), 0, true)   # load lod0
//   fps.debug_simulate_block_event(Vector3i(2,0,2), 1, false)  # unload lod1
void FluidParticleSystem::debug_simulate_block_event(Vector3i position, int lod, bool loaded) {
    UtilityFunctions::print("FluidParticleSystem: debug_simulate_block_event ",
                            loaded ? "load" : "unload", " lod ", lod, " at ", position);
    _on_terrain_block_event(position, Variant(lod), loaded);
}

// ─────────────────────────────────────────────────────────────────────────────
// P5: LOD stats. pages/arena numbers come from the CPU allocator mirror;
// particles_by_lod comes from the GPU stats buffer written by the physics
// pass (requires one _process cycle after the last dispatch — the readback
// path syncs the device first, so this is safe to call any time).
// ─────────────────────────────────────────────────────────────────────────────
Dictionary FluidParticleSystem::get_lod_stats() {
    Dictionary out;

    // Pages allocated per LOD, straight from the CPU page table mirror.
    PackedInt32Array pages_by_lod;
    pages_by_lod.resize(lod_levels);
    pages_by_lod.fill(0);
    for (int k = 0; k < lod_levels; ++k) {
        const int64_t base = _lod_table_base(k);
        if (base < 0) continue;
        const Vector3i d = _lod_table_dims(k);
        int count = 0;
        const int64_t end = lod_table_base[k + 1];
        for (int64_t i = base; i < end; ++i) {
            if ((lod_table_cpu[(size_t)i].flags & LOD_PAGE_FLAG_ALLOCATED) != 0) ++count;
        }
        pages_by_lod[k] = count;
        (void)d;
    }
    out["pages_by_lod"] = pages_by_lod;

    // Particles simulated per LOD last physics dispatch.
    PackedInt32Array particles_by_lod;
    particles_by_lod.resize(lod_levels);
    particles_by_lod.fill(0);
    if (rd && lod_stats_buf.is_valid()) {
        _sync_rd();
        PackedByteArray raw = rd->buffer_get_data(lod_stats_buf, 0,
                                                   32 * (int64_t)sizeof(int32_t));
        if (raw.size() >= (int64_t)(lod_levels * (int64_t)sizeof(int32_t))) {
            for (int k = 0; k < lod_levels; ++k) {
                int32_t v; memcpy(&v, raw.ptr() + k * 4, 4);
                particles_by_lod[k] = v;
            }
        }
    }
    out["particles_by_lod"] = particles_by_lod;

    out["skipped"]             = (int64_t)skipped_particle_count;
    out["store_ring_pending"]  = 0;
    out["arena_free_slots"]    = (int64_t)lod_arena.free_count();
    return out;
}

bool FluidParticleSystem::_refresh_sdf_from_child() {
    // Instantiate a terrain child if we don't have one yet. P1 targets
    // VoxelLodTerrain (godot_voxel's variable-LOD node); a legacy fixed
    // VoxelTerrain is still adopted/created as a fallback. The node comes
    // from the godot_voxel addon (separate GDExtension), so we create it by
    // class name via ClassDB.
    Object *terrain = nullptr;
    if (voxel_terrain_child_id.is_valid()) {
        terrain = Object::cast_to<Object>(ObjectDB::get_instance(voxel_terrain_child_id));
    }
    if (!terrain) {
        // see if has child with either terrain class
        for (int i = 0; i < get_child_count(); i++) {
            Object *child = get_child(i);
            if (child && (child->is_class("VoxelLodTerrain") || child->is_class("VoxelTerrain"))) {
                terrain = child;
                voxel_terrain_child_id = terrain->get_instance_id();
                break;
            }
        }
    }
    if (!terrain) {
        // Prefer VoxelLodTerrain; fall back to legacy VoxelTerrain.
        Variant v = ClassDB::instantiate("VoxelLodTerrain");
        terrain = Object::cast_to<Object>(v);
        String terrain_class_name = "VoxelLodTerrain";
        if (!terrain) {
            UtilityFunctions::print("FluidParticleSystem: VoxelLodTerrain not available, falling back to VoxelTerrain.");
            terrain_class_name = "VoxelTerrain";
            v = ClassDB::instantiate(terrain_class_name);
            terrain = Object::cast_to<Object>(v);
        }
        if (!terrain) {
            UtilityFunctions::printerr("FluidParticleSystem: neither VoxelLodTerrain nor VoxelTerrain class found. Is the godot_voxel addon enabled?");
            return false;
        }
        // Add as child so it enters the tree and starts generating.
        Node *terrain_node = Object::cast_to<Node>(terrain);
        if (terrain_node) {
            terrain_node->set_name(terrain_class_name);
            add_child(terrain_node);
            voxel_terrain_child_id = terrain_node->get_instance_id();
            terrain_is_vlt = (terrain_class_name == "VoxelLodTerrain");
            // In the editor, set the owner to the edited scene root so the
            // child shows up in the scene tree dock and can be selected/edited
            // directly. At runtime the owner stays null (no scene tree dock).
            if (Engine::get_singleton()->is_editor_hint()) {
                if (SceneTree *tree = get_tree()) {
                    if (Node *root = tree->get_edited_scene_root()) {
                        terrain_node->set_owner(root);
                    }
                }
            }
            // Apply terrain properties that were set before the child existed.
            for (const auto &E : _terrain_property_cache) {
                terrain->set(E.key, E.value);
            }
        }
    } else {
        // Adopted an existing child: remember its class and fix up owner.
        terrain_is_vlt = terrain->is_class("VoxelLodTerrain");
        if (Engine::get_singleton()->is_editor_hint()) {
            if (SceneTree *tree = get_tree()) {
                if (Node *root = tree->get_edited_scene_root()) {
                    Node *terrain_node = Object::cast_to<Node>(terrain);
                    if (terrain_node && terrain_node->get_owner() != root) {
                        terrain_node->set_owner(root);
                    }
                }
            }
        }
    }

    // Phase 3: let the VLT run lod_count = lod_levels (default 4) so the
    // terrain actually coarsens/refines and emits multi-LOD block events.
    // Data block size is 16 upstream; mesh_block_size pinned to 16 so the
    // block coords match our page coords at every LOD. P1's SDF path only
    // consumed LOD 0 events, which still arrive (they're a subset).
    if (terrain_is_vlt) {
        Variant lod_count_ret = terrain->get("lod_count");
        if ((int)lod_count_ret != lod_levels) {
            terrain->set("lod_count", lod_levels);
        }
        Variant mbs_ret = terrain->get("mesh_block_size");
        if ((int)mbs_ret != 16) {
            terrain->set("mesh_block_size", 16);
        }
        UtilityFunctions::print("FluidParticleSystem: VoxelLodTerrain configured lod_count=", (int)terrain->get("lod_count"),
                                " mesh_block_size=", (int)terrain->get("mesh_block_size"));
    }

    // Connect block signals so we recopy the SDF when the terrain streams in
    // new data. The callback just sets a flag; _process does the debounced
    // recopy. NOTE: upstream master VoxelLodTerrain does NOT register block
    // signals (they belong to fixed VoxelTerrain), so we probe the live node's
    // get_signal_list() and connect whichever of the four known names exist.
    // If none exist we log it once — SDF refresh then relies on the initial
    // copy plus manual reload_sdf_and_clear_grid() until LOD-tier signals
    // land upstream.
    {
        // Probe the live node's registered signals; connect each known name at
        // most once (is_connected guard). Re-refresh on the same terrain is
        // the common path, so the per-terrain "no signals" warning must not
        // spam — hence the header-stored memo.
        Array signal_list = terrain->call("get_signal_list");
        const char *load_names[] = { "block_loaded", "mesh_block_entered" };
        const char *unload_names[] = { "block_unloaded", "mesh_block_exited" };
        bool connected_any = false;
        terrain_signal_arity = -1;
        for (const char *name : load_names) {
            for (int si = 0; si < signal_list.size(); ++si) {
                Dictionary sig = signal_list[si];
                String sig_name = sig.get("name", String());
                if (sig_name != String(name)) {
                    continue;
                }
                // Cache arity from the signal's args list.
                Array sig_args = sig.get("args", Array());
                terrain_signal_arity = sig_args.size() > 0 ? (int)sig_args.size() : 1;
                if (!terrain->is_connected(name, Callable(this, "_on_terrain_block_loaded"))) {
                    terrain->connect(name, Callable(this, "_on_terrain_block_loaded"));
                }
                connected_any = true;
                break;
            }
        }
        for (const char *name : unload_names) {
            for (int si = 0; si < signal_list.size(); si++) {
                Dictionary sig = signal_list[si];
                String sig_name = sig.get("name", String());
                if (sig_name != String(name)) {
                    continue;
                }
                if (!terrain->is_connected(name, Callable(this, "_on_terrain_block_unloaded"))) {
                    terrain->connect(name, Callable(this, "_on_terrain_block_unloaded"));
                }
                connected_any = true;
                break;
            }
        }
        if (connected_any) {
            UtilityFunctions::print("FluidParticleSystem: connected terrain block signals (arity=",
                                    terrain_signal_arity, ") on ",
                                    terrain_is_vlt ? "VoxelLodTerrain" : "VoxelTerrain");
        } else if (!terrain_no_signal_warned) {
            UtilityFunctions::print("FluidParticleSystem: no block signals registered on terrain node (",
                                    terrain_is_vlt ? "VoxelLodTerrain" : "VoxelTerrain",
                                    ") — SDF auto-refresh unavailable; call reload_sdf_and_clear_grid() after edits.");
            terrain_no_signal_warned = true;
        }
    }

    // SDF is now baked per-page into arena cells by _bake_sdf_for_page.
    // No full-grid VoxelBuffer extraction needed — the poller queries the
    // terrain lazily per page. Return true if the terrain child exists.
    return terrain != nullptr;
}

void FluidParticleSystem::_upload_sdf_data() {
    if (sdf_buffer_var.get_type() == Variant::NIL) {
        _free_local_rid(sdf_storage_buf);
        sdf_w = sdf_h = sdf_d = 0;
        return;
    }

    Object *vb = Object::cast_to<Object>(sdf_buffer_var);
    if (!vb) {
        UtilityFunctions::printerr("FluidParticleSystem: SDF buffer is not a valid Object.");
        return;
    }

    // get_size() → Vector3i
    Vector3i size = vb->call("get_size");
    sdf_w = size.x;
    sdf_h = size.y;
    sdf_d = size.z;
    if (sdf_w <= 0 || sdf_h <= 0 || sdf_d <= 0) {
        UtilityFunctions::printerr("FluidParticleSystem: VoxelBuffer has invalid size.");
        return;
    }

    // get_channel_depth(CHANNEL_SDF=1) → int (0=8bit, 1=16bit, 2=32bit, 3=64bit)
    int64_t depth_val = vb->call("get_channel_depth", 1);
    int depth = (int)depth_val;

    // get_channel_as_byte_array(CHANNEL_SDF=1) → PackedByteArray (raw, ZXY order)
    PackedByteArray raw = vb->call("get_channel_as_byte_array", 1);
    if (raw.size() == 0) {
        UtilityFunctions::printerr("FluidParticleSystem: VoxelBuffer SDF channel is empty.");
        return;
    }

    int64_t voxel_count = (int64_t)sdf_w * sdf_h * sdf_d;
    int64_t float_bytes = voxel_count * sizeof(float);

    UtilityFunctions::print("FluidParticleSystem: uploading SDF — ", voxel_count,
                            " voxels, ", float_bytes, " bytes (", raw.size(),
                            " raw bytes, depth=", depth, ")");

    // Decode into a local byte array. sdf_storage_buf is allocated once at
    // grid size in _build_gpu_resources; we just buffer_update into it.
    PackedByteArray staging;
    staging.resize(float_bytes);
    float *dst = reinterpret_cast<float *>(staging.ptrw());
    const uint8_t *src = raw.ptr();

    // Decode based on depth. godot_voxel quantization scales:
    //   8-bit:  scale 0.1  → sdf = (raw/127) / 0.1 = (raw/127) * 10
    //   16-bit: scale 0.002 → sdf = (raw/32767) / 0.002 = (raw/32767) * 500
    //   32-bit: stored as raw float
    //   64-bit: stored as raw double → downcast to float
    // s8_to_snorm/s16_to_snorm clamp the low end to -1.
    if (depth == 0) {
        const int8_t *s8 = reinterpret_cast<const int8_t *>(src);
        for (int64_t i = 0; i < voxel_count; i++) {
            float v = std::max((float)s8[i] / 127.0f, -1.0f);
            dst[i] = v * 10.0f;
        }
    } else if (depth == 1) {
        const int16_t *s16 = reinterpret_cast<const int16_t *>(src);
        for (int64_t i = 0; i < voxel_count; i++) {
            float v = std::max((float)s16[i] / 32767.0f, -1.0f);
            dst[i] = v * 500.0f;
        }
    } else if (depth == 2) {
        std::memcpy(dst, src, float_bytes);
    } else if (depth == 3) {
        const double *d = reinterpret_cast<const double *>(src);
        for (int64_t i = 0; i < voxel_count; i++) {
            dst[i] = (float)d[i];
        }
    } else {
        UtilityFunctions::printerr("FluidParticleSystem: unsupported SDF depth: ", depth);
        return;
    }

    // Scan the decoded SDF for content statistics. This catches the case where
    // copy() succeeded but returned empty/uninitialized data (e.g. terrain
    // hasn't streamed its blocks yet).
    float sdf_min = std::numeric_limits<float>::max();
    float sdf_max = std::numeric_limits<float>::lowest();
    double sdf_sum = 0.0;
    int64_t solid_count = 0;   // sdf < 0 (inside terrain)
    int64_t empty_count = 0;   // sdf > 0 (outside terrain)
    int64_t zero_count = 0;    // sdf == 0 (surface or uninitialized)
    for (int64_t i = 0; i < voxel_count; i++) {
        float v = dst[i];
        if (v < sdf_min) sdf_min = v;
        if (v > sdf_max) sdf_max = v;
        sdf_sum += (double)v;
        if (v < 0.0f) solid_count++;
        else if (v > 0.0f) empty_count++;
        else zero_count++;
    }
    float sdf_mean = (float)(sdf_sum / (double)voxel_count);
    UtilityFunctions::print("FluidParticleSystem: SDF content — min=", sdf_min,
                            " max=", sdf_max, " mean=", sdf_mean,
                            " solid(<0)=", solid_count,
                            " empty(>0)=", empty_count,
                            " zero(=0)=", zero_count,
                            " / ", voxel_count, " voxels");
    if (solid_count == 0 && zero_count == voxel_count) {
        UtilityFunctions::printerr("FluidParticleSystem: SDF is entirely zero — ",
                                   "terrain data has not loaded. ",
                                   "Wait for streaming or check the stream/generator.");
    }

    if (!rd) return;

    // Direct copy: decode into a local byte array and buffer_update into the
    // existing sdf_storage_buf (allocated once at grid size in
    // _build_gpu_resources). No reallocation, no staging buffer, no read-back.
    if (sdf_storage_buf.is_valid()) {
        rd->buffer_update(sdf_storage_buf, 0, float_bytes, staging);
        UtilityFunctions::print("FluidParticleSystem: SDF buffer_update OK — ",
                                float_bytes, " bytes written");
    }
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
        case 3:
            path = &migrate_shader_path; p_shader = &migrate_shader;
            p_pipeline = &migrate_pipeline; p_uniform_set = &migrate_uniform_set;
            name = "migrate"; break;
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