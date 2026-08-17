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
    float   _pad0, _pad1;
    // Delta transform as 3 vec4 rows (row-major 3x4 matrix):
    //   row0 = (m00, m01, m02, origin_x)
    //   row1 = (m10, m11, m12, origin_y)
    //   row2 = (m20, m21, m22, origin_z)
    // Per-particle displacement: dp = M * p + origin - p
    float   delta_row[12];
};  // 128 bytes


// ─────────────────────────────────────────────────────────────────────────────
FluidParticleSystem::FluidParticleSystem() {}
FluidParticleSystem::~FluidParticleSystem() {}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_grid_width","v"),    &FluidParticleSystem::set_grid_width);
    ClassDB::bind_method(D_METHOD("get_grid_width"),        &FluidParticleSystem::get_grid_width);
    ClassDB::bind_method(D_METHOD("set_grid_height","v"),   &FluidParticleSystem::set_grid_height);
    ClassDB::bind_method(D_METHOD("get_grid_height"),       &FluidParticleSystem::get_grid_height);
    ClassDB::bind_method(D_METHOD("set_grid_depth","v"),    &FluidParticleSystem::set_grid_depth);
    ClassDB::bind_method(D_METHOD("get_grid_depth"),        &FluidParticleSystem::get_grid_depth);
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
    ClassDB::bind_method(D_METHOD("set_blur_radius","v"),          &FluidParticleSystem::set_blur_radius);
    ClassDB::bind_method(D_METHOD("get_blur_radius"),               &FluidParticleSystem::get_blur_radius);
    ClassDB::bind_method(D_METHOD("set_tint_strength","v"),       &FluidParticleSystem::set_tint_strength);
    ClassDB::bind_method(D_METHOD("get_tint_strength"),            &FluidParticleSystem::get_tint_strength);
    ClassDB::bind_method(D_METHOD("set_refraction_strength","v"), &FluidParticleSystem::set_refraction_strength);
    ClassDB::bind_method(D_METHOD("get_refraction_strength"),      &FluidParticleSystem::get_refraction_strength);
    ClassDB::bind_method(D_METHOD("set_absorption_dist","v"),     &FluidParticleSystem::set_absorption_dist);
    ClassDB::bind_method(D_METHOD("get_absorption_dist"),          &FluidParticleSystem::get_absorption_dist);
    ClassDB::bind_method(D_METHOD("set_fluid_tint","v"),          &FluidParticleSystem::set_fluid_tint);
    ClassDB::bind_method(D_METHOD("get_fluid_tint"),               &FluidParticleSystem::get_fluid_tint);
    ClassDB::bind_method(D_METHOD("set_normal_smooth","v"),       &FluidParticleSystem::set_normal_smooth);
    ClassDB::bind_method(D_METHOD("get_normal_smooth"),            &FluidParticleSystem::get_normal_smooth);

    ClassDB::bind_method(D_METHOD("reload_physics_shader"), &FluidParticleSystem::reload_physics_shader);
    ClassDB::bind_method(D_METHOD("get_reload_physics_shader"), &FluidParticleSystem::get_reload_physics_shader);
    // Shader paths group
    ADD_GROUP("Shaders", "");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "clear_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_clear_shader_path",   "get_clear_shader_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "physics_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_physics_shader_path", "get_physics_shader_path");
    // Inspector button: recompiles velocity_spread.glsl and rebuilds the pipeline.
    // PROPERTY_USAGE_BUTTON renders the property as a clickable button in the
    // inspector; clicking it invokes the bound method (no getter needed).
    ADD_PROPERTY(PropertyInfo(Variant::CALLABLE, "reload_physics_shader",
            PROPERTY_HINT_TOOL_BUTTON, "Reload Physics Shader,Reload",
            PROPERTY_USAGE_EDITOR),
            "", "get_reload_physics_shader");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "sortkey_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_sortkey_shader_path", "get_sortkey_shader_path");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "render_material",
        PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial"), "set_render_material", "get_render_material");

    // Initial chunk fill group
    ADD_GROUP("Initial Chunk", "initial_chunk_");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,     "initial_chunk_use"),                             "set_use_initial_chunk",        "get_use_initial_chunk");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3,  "initial_chunk_origin"),                          "set_initial_chunk_origin",     "get_initial_chunk_origin");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "initial_chunk_size"),                            "set_initial_chunk_size",       "get_initial_chunk_size");
    ADD_PROPERTY(PropertyInfo(Variant::COLOR,    "initial_chunk_color"),                           "set_initial_chunk_color",      "get_initial_chunk_color");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT,    "initial_chunk_attraction", PROPERTY_HINT_RANGE, "-2,2,0.01"), "set_initial_chunk_attraction", "get_initial_chunk_attraction");

    ADD_GROUP("Composite", "composite_");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "composite_blur_radius",         PROPERTY_HINT_RANGE, "0,10,0.01"),  "set_blur_radius",          "get_blur_radius");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "composite_tint_strength",       PROPERTY_HINT_RANGE, "0,2,0.01"),   "set_tint_strength",        "get_tint_strength");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "composite_refraction_strength", PROPERTY_HINT_RANGE, "0,0.2,0.001"),"set_refraction_strength",  "get_refraction_strength");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "composite_absorption_dist",     PROPERTY_HINT_RANGE, "0,20,0.01"),  "set_absorption_dist",      "get_absorption_dist");
    ADD_PROPERTY(PropertyInfo(Variant::COLOR, "composite_tint",                PROPERTY_HINT_COLOR_NO_ALPHA),      "set_fluid_tint",           "get_fluid_tint");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "composite_normal_smooth",       PROPERTY_HINT_RANGE, "0,10,0.01"),  "set_normal_smooth",        "get_normal_smooth");

    ADD_GROUP("", "");

    ADD_PROPERTY(PropertyInfo(Variant::INT,   "grid_width"),      "set_grid_width",      "get_grid_width");
    ADD_PROPERTY(PropertyInfo(Variant::INT,   "grid_height"),     "set_grid_height",     "get_grid_height");
    ADD_PROPERTY(PropertyInfo(Variant::INT,   "grid_depth"),      "set_grid_depth",      "get_grid_depth");
    ADD_PROPERTY(PropertyInfo(Variant::INT,   "num_particles"),   "set_num_particles",   "get_num_particles");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity"),         "set_gravity",         "get_gravity");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,    "gravity_local"),   "set_gravity_local",   "get_gravity_local");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "surface_tension"), "set_surface_tension", "get_surface_tension");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "water_viscosity"), "set_water_viscosity", "get_water_viscosity");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attraction_force", PROPERTY_HINT_RANGE, "-2,2,0.01"), "set_attraction_force", "get_attraction_force");
    ADD_PROPERTY(PropertyInfo(Variant::INT,   "neighbor_mode"),   "set_neighbor_mode",   "get_neighbor_mode");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,  "simulation_active"), "set_simulation_active", "get_simulation_active");

}

// ─────────────────────────────────────────────────────────────────────────────
// Composite uniform setters — push to the active composite material.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::set_blur_radius(float v) {
    blur_radius = v;
    Ref<ShaderMaterial> comp = _active_composite_material();
    if (comp.is_valid()) comp->set_shader_parameter("blur_radius", v);
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
void FluidParticleSystem::set_normal_smooth(float v) {
    normal_smooth = v;
    Ref<ShaderMaterial> comp = _active_composite_material();
    if (comp.is_valid()) comp->set_shader_parameter("normal_smooth", v);
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::set_grid_width(int v)   { grid_width  = v; _rebuild_gpu_resources(); }
void FluidParticleSystem::set_grid_height(int v)  { grid_height = v; _rebuild_gpu_resources(); }
void FluidParticleSystem::set_grid_depth(int v)   { grid_depth  = v; _rebuild_gpu_resources(); }
void FluidParticleSystem::set_num_particles(int v){ num_particles = v; _rebuild_gpu_resources(); }

// ─────────────────────────────────────────────────────────────────────────────
// Default render shader path. Used to pre-fill the render_shader resource when
// it has not been set (or has been cleared) in the inspector.
// ─────────────────────────────────────────────────────────────────────────────
static const char *DEFAULT_RENDER_SHADER_PATH =
    "res://addons/fluid_particles/shaders/particle_render.gdshader";

// Loads the default particle render shader code into a new Shader resource.
static Ref<Shader> _load_default_render_shader() {
    Ref<Shader> shader;
    shader.instantiate();
    Ref<FileAccess> sf = FileAccess::open(DEFAULT_RENDER_SHADER_PATH, FileAccess::READ);
    if (sf.is_valid()) {
        shader->set_code(sf->get_as_text());
    } else {
        UtilityFunctions::printerr("FluidParticleSystem: cannot open default render shader: ",
                                    DEFAULT_RENDER_SHADER_PATH);
    }
    return shader;
}

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

// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// Build the RD offscreen render pipeline:
//   FluidParticleSystem (this)
//     └── MeshInstance3D (composite_node)  — fullscreen quad with the
//                                            blur+depth composite shader
//
// Particles are rendered directly via a RenderingDevice render pipeline into
// an RD framebuffer (color RGBA8 + depth D32_SFLOAT). The vertex/fragment
// shader (fluid_depth.glsl) reads the mesh's storage buffers and writes
// packed linear depth + color. The output textures are exposed to the
// composite shader via Texture2DRD.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_ensure_rd_render_pipeline() {
    if (rd_render_pipeline.is_valid()) return;  // already built
    if (!rd) return;
    if (!vertex_buf.is_valid() || !attrib_buf.is_valid()) return;

    // ── Compile the render shader (fluid_depth.glsl: vertex + fragment) ──────
    {
        Ref<RDShaderFile> sf = ResourceLoader::get_singleton()->load(
            "res://addons/fluid_particles/shaders/fluid_depth.glsl", "",
            ResourceLoader::CACHE_MODE_REPLACE);
        if (!sf.is_valid()) {
            UtilityFunctions::printerr(
                "FluidParticleSystem: cannot load fluid_depth.glsl");
            return;
        }
        Ref<RDShaderSPIRV> spirv = sf->get_spirv();
        if (!spirv.is_valid()) {
            UtilityFunctions::printerr("FluidParticleSystem: no SPIR-V in fluid_depth.glsl");
            return;
        }
        String verr = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_VERTEX);
        String ferr = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_FRAGMENT);
        if (!verr.is_empty() || !ferr.is_empty()) {
            UtilityFunctions::printerr("FluidParticleSystem: fluid_depth.glsl compile error:\n",
                                       "  vertex: ", verr, "\n  fragment: ", ferr);
            return;
        }
        rd_render_shader = rd->shader_create_from_spirv(spirv);
        if (!rd_render_shader.is_valid()) {
            UtilityFunctions::printerr("FluidParticleSystem: failed to create render shader RID");
            return;
        }
    }

    // ── Create the color + depth textures + framebuffer ─────────────────────
    // Start at 512x512; _sync_rd_viewport_size() resizes to match the main
    // viewport on the first frame.
    rd_vp_size = Vector2i(512, 512);

    {
        // Color attachment: RGBA8, color-attachment + sampling.
        Ref<RDTextureFormat> cf;
        cf.instantiate();
        cf->set_texture_type(RenderingDevice::TEXTURE_TYPE_2D);
        cf->set_format(RenderingDevice::DATA_FORMAT_R8G8B8A8_UNORM);
        cf->set_width(rd_vp_size.x);
        cf->set_height(rd_vp_size.y);
        cf->set_depth(1);
        cf->set_array_layers(1);
        cf->set_mipmaps(1);
        cf->set_samples(RenderingDevice::TEXTURE_SAMPLES_1);
        cf->set_usage_bits(RenderingDevice::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT |
                           RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT |
                           RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT);
        Ref<RDTextureView> cv;
        cv.instantiate();
        rd_color_tex = rd->texture_create(cf, cv);
    }
    {
        // Depth attachment: D32_SFLOAT, depth-stencil-attachment + sampling.
        Ref<RDTextureFormat> df;
        df.instantiate();
        df->set_texture_type(RenderingDevice::TEXTURE_TYPE_2D);
        df->set_format(RenderingDevice::DATA_FORMAT_D32_SFLOAT);
        df->set_width(rd_vp_size.x);
        df->set_height(rd_vp_size.y);
        df->set_depth(1);
        df->set_array_layers(1);
        df->set_mipmaps(1);
        df->set_samples(RenderingDevice::TEXTURE_SAMPLES_1);
        df->set_usage_bits(RenderingDevice::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                           RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT |
                           RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT);
        Ref<RDTextureView> dv;
        dv.instantiate();
        rd_depth_tex = rd->texture_create(df, dv);
    }

    // Framebuffer: [color, depth].
    {
        TypedArray<RID> fb_textures;
        fb_textures.append(rd_color_tex);
        fb_textures.append(rd_depth_tex);
        rd_framebuffer = rd->framebuffer_create(fb_textures);
        if (!rd_framebuffer.is_valid()) {
            UtilityFunctions::printerr("FluidParticleSystem: failed to create RD framebuffer");
            return;
        }
        rd_framebuffer_format = rd->framebuffer_get_format(rd_framebuffer);
    }

    // ── Vertex array wrapping the mesh's storage buffers ────────────────────
    // The render mesh uses ARRAY_FLAG_USE_STORAGE_BUFFER, so its vertex and
    // attribute buffers are RD storage buffers. We build a vertex array that
    // describes the layout (vec3 position, vec4 color, vec4 custom0) and binds
    // those buffers directly.
    {
        // Describe the vertex format via RDVertexAttribute.
        // Binding 0: vertex buffer (position: vec3 float, stride 12 bytes)
        // Binding 1: attribute buffer (color: vec4 float, custom0: vec4 float)
        TypedArray<Ref<RDVertexAttribute>> attrs;

        {
            Ref<RDVertexAttribute> a;
            a.instantiate();
            a->set_location(0);   // matches layout(location=0) in the GLSL
            a->set_binding(0);
            a->set_offset(0);
            a->set_format(RenderingDevice::DATA_FORMAT_R32G32B32_SFLOAT);
            a->set_stride(vertex_stride_floats * 4);  // bytes (from mesh format)
            a->set_frequency(RenderingDevice::VERTEX_FREQUENCY_VERTEX);
            attrs.append(a);
        }
        {
            // Color: vec4 float at offset 0 in the attribute buffer.
            Ref<RDVertexAttribute> a;
            a.instantiate();
            a->set_location(1);   // matches layout(location=1)
            a->set_binding(1);
            a->set_offset(0);
            a->set_format(RenderingDevice::DATA_FORMAT_R32G32B32A32_SFLOAT);
            a->set_stride(attrib_stride_words * 4);  // bytes
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
            return;
        }

        TypedArray<RID> src_buffers;
        src_buffers.append(vertex_buf);
        src_buffers.append(attrib_buf);
        rd_vertex_array = rd->vertex_array_create(
            num_particles, rd_vertex_format, src_buffers);
        if (!rd_vertex_array.is_valid()) {
            UtilityFunctions::printerr("FluidParticleSystem: failed to create vertex array");
            return;
        }
    }

    // ── Render pipeline ──────────────────────────────────────────────────────
    {
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
        ds->set_depth_compare_operator(RenderingDevice::COMPARE_OP_LESS);

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
            return;
        }
    }

    // ── Camera uniform buffer + uniform set ──────────────────────────────────
    // Camera matrices (5×mat4 + vec3 + pad = 332 bytes) exceed Vulkan's 128-byte
    // push constant limit, so they go in a uniform buffer at binding 0.
    // Layout MUST match CameraUBO in fluid_depth.glsl (std140).
    {
        // std140 layout: 5 mat4 (320 bytes) + vec3 (12) + pad to vec4 (4) = 336.
        // Round up to 352 (multiple of 16) for safety.
        constexpr uint32_t UBO_SIZE = 352;
        PackedByteArray zero;
        zero.resize(UBO_SIZE);
        zero.fill(0);
        rd_camera_ubo = rd->uniform_buffer_create(UBO_SIZE, zero);
        if (!rd_camera_ubo.is_valid()) {
            UtilityFunctions::printerr("FluidParticleSystem: failed to create camera UBO");
            return;
        }

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
            return;
        }
    }

    // ── Share textures to the shared device for the compositor ──────────────
    // The color+depth textures are on the local device. The composite shader
    // (a spatial shader) runs on the shared device, so we create shared-device
    // wrappers around the local device's VkImage handles.
    {
        uint64_t color_vkimage = rd->get_driver_resource(
            RenderingDevice::DRIVER_RESOURCE_TEXTURE, rd_color_tex, 0);
        shared_color_tex = shared_rd->texture_create_from_extension(
            RenderingDevice::TEXTURE_TYPE_2D,
            RenderingDevice::DATA_FORMAT_R8G8B8A8_UNORM,
            RenderingDevice::TEXTURE_SAMPLES_1,
            RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT |
            RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT,
            color_vkimage, rd_vp_size.x, rd_vp_size.y, 1, 1, 1);

        uint64_t depth_vkimage = rd->get_driver_resource(
            RenderingDevice::DRIVER_RESOURCE_TEXTURE, rd_depth_tex, 0);
        shared_depth_tex = shared_rd->texture_create_from_extension(
            RenderingDevice::TEXTURE_TYPE_2D,
            RenderingDevice::DATA_FORMAT_D32_SFLOAT,
            RenderingDevice::TEXTURE_SAMPLES_1,
            RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT |
            RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT,
            depth_vkimage, rd_vp_size.x, rd_vp_size.y, 1, 1, 1);

        if (!shared_color_tex.is_valid() || !shared_depth_tex.is_valid()) {
            UtilityFunctions::printerr("FluidParticleSystem: failed to share textures to shared device");
        }
    }

    // ── Texture2DRD wrappers for the composite shader ────────────────────────
    rd_color_tex_2d.instantiate();
    rd_color_tex_2d->set_texture_rd_rid(shared_color_tex);
    rd_depth_tex_2d.instantiate();
    rd_depth_tex_2d->set_texture_rd_rid(shared_depth_tex);

    // ── Composite quad (fullscreen, samples the RD textures) ────────────────
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

    composite_material.instantiate();
    composite_material->set_shader(composite_shader);

    // Bind the RD textures to the composite shader.
    composite_material->set_shader_parameter("fluid_tex", rd_color_tex_2d);
    composite_material->set_shader_parameter("fluid_depth_tex", rd_depth_tex_2d);
    composite_material->set_shader_parameter(
        "fluid_pixel_size",
        Vector2(1.0f / (float)rd_vp_size.x, 1.0f / (float)rd_vp_size.y));
    composite_material->set_shader_parameter("offscreen_far", 100.0);
    composite_material->set_shader_parameter("offscreen_near", 0.1);

    // Push initial composite uniform values from the node properties.
    composite_material->set_shader_parameter("blur_radius", blur_radius);
    composite_material->set_shader_parameter("tint_strength", tint_strength);
    composite_material->set_shader_parameter("refraction_strength", refraction_strength);
    composite_material->set_shader_parameter("absorption_dist", absorption_dist);
    composite_material->set_shader_parameter("fluid_tint", fluid_tint);
    composite_material->set_shader_parameter("normal_smooth", normal_smooth);

    // Fullscreen quad mesh — a 2-triangle quad in NDC, the vertex shader writes
    // POSITION directly so it covers the whole screen regardless of camera.
    Ref<QuadMesh> quad;
    quad.instantiate();
    quad->set_size(Vector2(2.0, 2.0));   // NDC-sized
    quad->set_orientation(PlaneMesh::FACE_Z);

    composite_node = memnew(MeshInstance3D);
    composite_node->set_mesh(quad);
    composite_node->set_material_override(composite_material);
    composite_node->set_cast_shadows_setting(GeometryInstance3D::SHADOW_CASTING_SETTING_OFF);
    // Prevent frustum culling of the POSITION-rewriting fullscreen quad.
    composite_node->set_custom_aabb(AABB(Vector3(-1e6f, -1e6f, -1e6f), Vector3(2e6f, 2e6f, 2e6f)));
    composite_node->set_extra_cull_margin(1e6f);
    // Park it under this node for now; _sync_offscreen_camera reparents it to
    // the main camera once the camera is resolved.
    add_child(composite_node);

    UtilityFunctions::print("FluidParticleSystem: RD render pipeline ready.");
}

// ─────────────────────────────────────────────────────────────────────────────
// Resize the RD color+depth textures + framebuffer to match the main viewport.
// Called every frame from _sync_offscreen_camera(). Recreates the framebuffer,
// textures, Texture2DRD wrappers, and render pipeline (pipeline is tied to the
// framebuffer format).
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_sync_rd_viewport_size() {
    if (!rd) return;

    Camera3D *main_cam = _resolve_main_camera();
    Vector2i main_size(0, 0);
    if (main_cam) {
        if (Viewport *vp = main_cam->get_viewport())
            main_size = vp->get_visible_rect().size;
    }
    if (main_size.x <= 0 || main_size.y <= 0) return;
    if (main_size == rd_vp_size) return;  // no change

    rd_vp_size = main_size;

    // Free old textures + framebuffer.
    if (rd_color_tex_2d.is_valid()) rd_color_tex_2d->set_texture_rd_rid(RID());
    if (rd_depth_tex_2d.is_valid()) rd_depth_tex_2d->set_texture_rd_rid(RID());
    if (rd_framebuffer.is_valid())  rd->free_rid(rd_framebuffer);
    if (rd_color_tex.is_valid())    rd->free_rid(rd_color_tex);
    if (rd_depth_tex.is_valid())    rd->free_rid(rd_depth_tex);

    // Recreate color texture.
    {
        Ref<RDTextureFormat> cf;
        cf.instantiate();
        cf->set_texture_type(RenderingDevice::TEXTURE_TYPE_2D);
        cf->set_format(RenderingDevice::DATA_FORMAT_R8G8B8A8_UNORM);
        cf->set_width(rd_vp_size.x);
        cf->set_height(rd_vp_size.y);
        cf->set_depth(1);
        cf->set_array_layers(1);
        cf->set_mipmaps(1);
        cf->set_samples(RenderingDevice::TEXTURE_SAMPLES_1);
        cf->set_usage_bits(RenderingDevice::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT |
                           RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT |
                           RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT);
        Ref<RDTextureView> cv;
        cv.instantiate();
        rd_color_tex = rd->texture_create(cf, cv);
    }
    // Recreate depth texture.
    {
        Ref<RDTextureFormat> df;
        df.instantiate();
        df->set_texture_type(RenderingDevice::TEXTURE_TYPE_2D);
        df->set_format(RenderingDevice::DATA_FORMAT_D32_SFLOAT);
        df->set_width(rd_vp_size.x);
        df->set_height(rd_vp_size.y);
        df->set_depth(1);
        df->set_array_layers(1);
        df->set_mipmaps(1);
        df->set_samples(RenderingDevice::TEXTURE_SAMPLES_1);
        df->set_usage_bits(RenderingDevice::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                           RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT |
                           RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT);
        Ref<RDTextureView> dv;
        dv.instantiate();
        rd_depth_tex = rd->texture_create(df, dv);
    }
    // Recreate framebuffer.
    {
        TypedArray<RID> fb_textures;
        fb_textures.append(rd_color_tex);
        fb_textures.append(rd_depth_tex);
        rd_framebuffer = rd->framebuffer_create(fb_textures);
        rd_framebuffer_format = rd->framebuffer_get_format(rd_framebuffer);
    }

    // Recreate the render pipeline (it's tied to the framebuffer format).
    if (rd_render_pipeline.is_valid()) {
        rd->free_rid(rd_render_pipeline);
        rd_render_pipeline = RID();
    }
    {
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
        ds->set_depth_compare_operator(RenderingDevice::COMPARE_OP_LESS);

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
    }

    // Free old shared-device texture wrappers and recreate them.
    if (shared_color_tex.is_valid()) shared_rd->free_rid(shared_color_tex);
    if (shared_depth_tex.is_valid()) shared_rd->free_rid(shared_depth_tex);
    {
        uint64_t color_vkimage = rd->get_driver_resource(
            RenderingDevice::DRIVER_RESOURCE_TEXTURE, rd_color_tex, 0);
        shared_color_tex = shared_rd->texture_create_from_extension(
            RenderingDevice::TEXTURE_TYPE_2D,
            RenderingDevice::DATA_FORMAT_R8G8B8A8_UNORM,
            RenderingDevice::TEXTURE_SAMPLES_1,
            RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT |
            RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT,
            color_vkimage, rd_vp_size.x, rd_vp_size.y, 1, 1, 1);

        uint64_t depth_vkimage = rd->get_driver_resource(
            RenderingDevice::DRIVER_RESOURCE_TEXTURE, rd_depth_tex, 0);
        shared_depth_tex = shared_rd->texture_create_from_extension(
            RenderingDevice::TEXTURE_TYPE_2D,
            RenderingDevice::DATA_FORMAT_D32_SFLOAT,
            RenderingDevice::TEXTURE_SAMPLES_1,
            RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT |
            RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT,
            depth_vkimage, rd_vp_size.x, rd_vp_size.y, 1, 1, 1);
    }

    // Re-wrap textures in Texture2DRD (using shared-device RIDs).
    if (rd_color_tex_2d.is_valid()) {
        rd_color_tex_2d->set_texture_rd_rid(shared_color_tex);
    }
    if (rd_depth_tex_2d.is_valid()) {
        rd_depth_tex_2d->set_texture_rd_rid(shared_depth_tex);
    }

    // Update the composite shader's texture + pixel-size uniforms.
    Ref<ShaderMaterial> comp = _active_composite_material();
    if (comp.is_valid()) {
        comp->set_shader_parameter("fluid_tex", rd_color_tex_2d);
        comp->set_shader_parameter("fluid_depth_tex", rd_depth_tex_2d);
        comp->set_shader_parameter(
            "fluid_pixel_size",
            Vector2(1.0f / (float)rd_vp_size.x, 1.0f / (float)rd_vp_size.y));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Render the particles into the RD framebuffer. Called every frame from
// _process() AFTER the compute dispatches (so the storage buffers are current)
// and AFTER _sync_offscreen_camera() (so the push-constant matrices are set).
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_render_particles_rd() {
    if (!rd || !rd_render_pipeline.is_valid() || !rd_framebuffer.is_valid())
        return;
    if (!rd_vertex_array.is_valid()) return;

    Camera3D *main_cam = _resolve_main_camera();
    if (!main_cam) return;

    // ── Prepare the camera uniform buffer data (main thread) ────────────────
    // Layout MUST match CameraUBO in fluid_depth.glsl (std140):
    //   mat4 view_matrix       (offset   0, 64 bytes)
    //   mat4 proj_matrix       (offset  64, 64 bytes)
    //   mat4 inv_view_matrix   (offset 128, 64 bytes)
    //   mat4 inv_proj_matrix   (offset 192, 64 bytes)
    //   mat4 model_matrix      (offset 256, 64 bytes)
    //   vec3 cam_pos_world     (offset 320, 12 bytes)
    //   float _pad0            (offset 332, 4 bytes)
    // Total: 336 bytes (UBO allocated as 352 for alignment).
    struct CameraUniforms {
        float view_matrix[16];
        float proj_matrix[16];
        float inv_view_matrix[16];
        float inv_proj_matrix[16];
        float model_matrix[16];
        float cam_pos_world[3];
        float _pad0;
    };

    Projection  proj  = main_cam->get_camera_projection();
    Transform3D view_xform = main_cam->get_global_transform();
    Transform3D model_xform = get_global_transform();

    // Camera view matrix = inverse of the camera's global transform.
    Transform3D view_t = view_xform.affine_inverse();
    // inv_view = camera global transform (world ← view).
    Transform3D inv_view_t = view_xform;
    Projection  inv_proj = proj.inverse();

    CameraUniforms ubo{};

    // Projection → float[16] (column-major, matching GLSL mat4).
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

    // Transform3D → mat4 (column-major, matching GLSL mat4 memory layout).
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

    // ── Update UBO + draw (local device — safe from main thread) ────────────
    // The local RenderingDevice is fully under our control, so we can call
    // buffer_update + draw_list_begin/end + submit directly without conflicting
    // with Godot's render thread.
    rd->buffer_update(rd_camera_ubo, 0, rd_ubo_bytes.size(), rd_ubo_bytes);

    PackedColorArray clear_colors;
    clear_colors.append(Color(0, 0, 0, 0));

    int64_t draw_list = rd->draw_list_begin(
        rd_framebuffer,
        RenderingDevice::DRAW_CLEAR_COLOR_0 | RenderingDevice::DRAW_CLEAR_DEPTH,
        clear_colors, 1.0f, 0);
    rd->draw_list_bind_render_pipeline(draw_list, rd_render_pipeline);
    rd->draw_list_bind_uniform_set(draw_list, rd_render_uniform_set, 0);
    rd->draw_list_bind_vertex_array(draw_list, rd_vertex_array);
    rd->draw_list_set_push_constant(draw_list, rd_pc_bytes, rd_pc_bytes.size());
    rd->draw_list_draw(draw_list, false, 1);
    rd->draw_list_end();
    rd->submit();
    rd_submitted = true;
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
    // _ensure_rd_render_pipeline is what actually keeps it from being culled).
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
    _build_gpu_resources();
    _ensure_rd_render_pipeline();
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

void FluidParticleSystem::_exit_tree() {
    _destroy_gpu_resources();
    // composite_node is a child and freed automatically
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_build_gpu_resources() {
    // Create a LOCAL RenderingDevice that we fully control. All compute + render
    // pipelines run on this device, so we can call submit()/sync() without
    // conflicting with Godot's render thread. The output color+depth textures
    // are shared to the shared device for the compositor.
    RenderingServer *rs = RenderingServer::get_singleton();
    shared_rd = rs->get_rendering_device();
    if (!shared_rd) {
        UtilityFunctions::printerr("FluidParticleSystem: Could not get shared RenderingDevice.");
        return;
    }
    rd = rs->create_local_rendering_device();
    if (!rd) {
        UtilityFunctions::printerr("FluidParticleSystem: Could not create local RenderingDevice.");
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

    // ── Load and compile compute shaders ─────────────────────────────────────
    clear_shader   = _compile_compute_shader(clear_shader_path);
    physics_shader = _compile_compute_shader(physics_shader_path);
    sortkey_shader = _compile_compute_shader(sortkey_shader_path);

    if (clear_shader.is_valid())   clear_pipeline   = rd->compute_pipeline_create(clear_shader);
    if (physics_shader.is_valid()) physics_pipeline = rd->compute_pipeline_create(physics_shader);
    if (sortkey_shader.is_valid()) sortkey_pipeline = rd->compute_pipeline_create(sortkey_shader);

    // ── Build uniform sets ────────────────────────────────────────────────────
    // All three pipelines share the same buffer bindings:
    //   binding 0 → vertex buffer (position)
    //   binding 1 → attribute buffer (color + custom0)
    //   binding 2 → chunk_grid
    //   binding 3 → runnable_indices
    //   binding 4 → sort_keys

    TypedArray<RDUniform> uniforms;
    uniforms.append(_make_storage_uniform(vertex_buf,   0));
    uniforms.append(_make_storage_uniform(attrib_buf,   1));
    uniforms.append(_make_storage_uniform(chunk_buf,    2));
    uniforms.append(_make_storage_uniform(runnable_buf, 3));
    uniforms.append(_make_storage_uniform(sort_key_buf, 4));

    if (clear_shader.is_valid()) {
        clear_uniform_set   = rd->uniform_set_create(uniforms, clear_shader,   0);
    }
    if (physics_shader.is_valid()) {
        physics_uniform_set = rd->uniform_set_create(uniforms, physics_shader, 0);
    }
    if (sortkey_shader.is_valid()) {
        sortkey_uniform_set = rd->uniform_set_create(uniforms, sortkey_shader, 0);
    }

    // ── Rendering ──────────────────────────────────────────────────────────
    // Rendering goes through the RD render pipeline, which reads the mesh's
    // vertex/attribute RD storage buffers directly (see _render_particles_rd).
    // No MeshInstance3D or CPU round-trip for the depth pass.
    gpu_ready = true;
    UtilityFunctions::print("FluidParticleSystem: GPU resources ready.");
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_destroy_gpu_resources() {
    if (!rd) return;
    gpu_ready = false;

    // ── RD render pipeline resources ────────────────────────────────────────
    if (rd_color_tex_2d.is_valid()) rd_color_tex_2d->set_texture_rd_rid(RID());
    if (rd_depth_tex_2d.is_valid()) rd_depth_tex_2d->set_texture_rd_rid(RID());
    // Free shared-device texture wrappers (on shared_rd, not rd).
    if (shared_rd) {
        if (shared_color_tex.is_valid()) shared_rd->free_rid(shared_color_tex);
        if (shared_depth_tex.is_valid()) shared_rd->free_rid(shared_depth_tex);
    }
    shared_color_tex = RID();
    shared_depth_tex = RID();
    if (rd_render_uniform_set.is_valid()) rd->free_rid(rd_render_uniform_set);
    if (rd_render_pipeline.is_valid())    rd->free_rid(rd_render_pipeline);
    if (rd_vertex_array.is_valid())       rd->free_rid(rd_vertex_array);
    if (rd_framebuffer.is_valid())        rd->free_rid(rd_framebuffer);
    if (rd_color_tex.is_valid())          rd->free_rid(rd_color_tex);
    if (rd_depth_tex.is_valid())          rd->free_rid(rd_depth_tex);
    if (rd_camera_ubo.is_valid())         rd->free_rid(rd_camera_ubo);
    if (rd_render_shader.is_valid())      rd->free_rid(rd_render_shader);
    rd_render_uniform_set = RID();
    rd_render_pipeline    = RID();
    rd_vertex_array       = RID();
    rd_framebuffer        = RID();
    rd_color_tex          = RID();
    rd_depth_tex          = RID();
    rd_camera_ubo         = RID();
    rd_render_shader      = RID();
    rd_framebuffer_format = -1;
    rd_vertex_format      = -1;

    if (clear_uniform_set.is_valid())   rd->free_rid(clear_uniform_set);
    if (physics_uniform_set.is_valid()) rd->free_rid(physics_uniform_set);
    if (sortkey_uniform_set.is_valid()) rd->free_rid(sortkey_uniform_set);

    if (clear_pipeline.is_valid())   rd->free_rid(clear_pipeline);
    if (physics_pipeline.is_valid()) rd->free_rid(physics_pipeline);
    if (sortkey_pipeline.is_valid()) rd->free_rid(sortkey_pipeline);
    if (clear_shader.is_valid())     rd->free_rid(clear_shader);
    if (physics_shader.is_valid())   rd->free_rid(physics_shader);
    if (sortkey_shader.is_valid())   rd->free_rid(sortkey_shader);

    // All buffers are on the local device and owned by us.
    if (vertex_buf.is_valid())    rd->free_rid(vertex_buf);
    if (attrib_buf.is_valid())    rd->free_rid(attrib_buf);
    if (chunk_buf.is_valid())     rd->free_rid(chunk_buf);
    if (runnable_buf.is_valid())  rd->free_rid(runnable_buf);
    if (sort_key_buf.is_valid())  rd->free_rid(sort_key_buf);

    // rd is a local device — do not delete it (Godot manages its lifetime).
    rd = nullptr;
    shared_rd = nullptr;
    rd_submitted = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Called from property setters (num_particles, grid_*) when a size-affecting
// property changes while the node is in the tree. Tears down all GPU resources
// and rebuilds them from scratch with the new dimensions.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_rebuild_gpu_resources() {
    if (!is_inside_tree()) return;  // _enter_tree will build on first entry
    _destroy_gpu_resources();

    _build_gpu_resources();
    _ensure_rd_render_pipeline();
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_process(double delta) {
    // Wait for the previous frame's GPU work to complete before starting new
    // work. The local device only allows one pending submission at a time.
    if (rd && rd_submitted) {
        rd->sync();
        rd_submitted = false;
    }

    // Sync the RD render pipeline to the main camera (resize textures, push
    // composite uniforms, position the composite quad). This must run BEFORE
    // the early returns so the composite stays positioned even when the
    // simulation is paused.
    _sync_offscreen_camera();

    if (!gpu_ready) {
        _render_particles_rd();  // still render the last sim state
        return;
    }

    if (simulation_active) {
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

        for (int i = 0; i < get_child_count(); i++) {
            FluidSourceBase *ss = Object::cast_to<FluidSourceBase>(get_child(i));
            if (ss) ss->dispatch_into_parent();
        }

        if (frame_count % 60 == 0) _dispatch_sortkey();
        frame_count++;
    }

    // ── Render the particles into the RD framebuffer ─────────────────────────
    // This runs every frame (even when paused) so the composite shader always
    // has a current fluid texture. Must run AFTER the compute dispatches so the
    // storage buffers are up to date.
    _render_particles_rd();
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build and submit a compute list for one pipeline
// ─────────────────────────────────────────────────────────────────────────────
static void dispatch_compute(RenderingDevice *rd,
                             RID pipeline, RID uniform_set,
                             const PushConstants &pc,
                             uint32_t x_groups, uint32_t y_groups = 1, uint32_t z_groups = 1)
{
    if (!pipeline.is_valid() || !uniform_set.is_valid()) return;

    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(PushConstants));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(PushConstants));

    int64_t cl = rd->compute_list_begin();
    rd->compute_list_bind_compute_pipeline(cl, pipeline);
    rd->compute_list_bind_uniform_set(cl, uniform_set, 0);
    rd->compute_list_set_push_constant(cl, pc_bytes, sizeof(PushConstants));
    rd->compute_list_dispatch(cl, x_groups, y_groups, z_groups);
    rd->compute_list_end();
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_dispatch_clear_grid() {
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
                // Write directly into the render mesh's vertex/attribute buffers via the RD storage buffer RIDs. No CPU round-trip.
                // rd->storage_buffer_write(vertex_buf, idx * vertex_stride_floats * sizeof(float), (const uint8_t*)&pos, sizeof(Vector3));
                // Color col = color;
                // rd->storage_buffer_write(attrib_buf, idx * attrib_stride_words * sizeof(uint32_t) + color_offset_words * sizeof(uint32_t), (const uint8_t*)&col, sizeof(Color));
                // float custom0[4] = { attraction, 1.0f, 0.0f, 0.0f };
                // rd->storage_buffer_write(attrib_buf, idx * attrib_stride_words * sizeof(uint32_t) + custom0_offset_words * sizeof(uint32_t), (const uint8_t*)custom0, sizeof(float) * 4);
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
    if (!gpu_ready) return;
    _dispatch_clear_grid();
    rd->submit();
    rd->sync();
    rd_submitted = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Reload the physics compute shader from disk. Frees the old shader RID,
// pipeline RID, and uniform set RID, then recompiles velocity_spread.glsl
// (or whatever physics_shader_path currently points at) and rebuilds the
// pipeline + uniform set against the existing storage buffers.
//
// Safe to call at runtime: if the new shader fails to compile, the old one
// is kept and an error is printed. No-op if the node is not in the tree
// (rd will be null).
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::reload_physics_shader() {
    if (!rd) {
        UtilityFunctions::printerr(
            "FluidParticleSystem::reload_physics_shader: RenderingDevice not available "
            "(node not in tree?).");
        return;
    }

    // Compile the new shader FIRST so a compile failure leaves the old one intact.
    RID new_shader = _compile_compute_shader(physics_shader_path);
    if (!new_shader.is_valid()) {
        UtilityFunctions::printerr(
            "FluidParticleSystem::reload_physics_shader: new shader failed to compile; "
            "keeping previous physics shader.");
        return;
    }

    // Tear down the old physics shader's GPU resources.
    if (physics_uniform_set.is_valid()) {
        rd->free_rid(physics_uniform_set);
        physics_uniform_set = RID();
    }
    if (physics_pipeline.is_valid()) {
        rd->free_rid(physics_pipeline);
        physics_pipeline = RID();
    }
    if (physics_shader.is_valid()) {
        rd->free_rid(physics_shader);
        physics_shader = RID();
    }

    // Install the new shader and rebuild the pipeline + uniform set against
    // the same storage buffers the other pipelines use.
    physics_shader   = new_shader;
    physics_pipeline = rd->compute_pipeline_create(physics_shader);

    TypedArray<RDUniform> uniforms;
    uniforms.append(_make_storage_uniform(vertex_buf,   0));
    uniforms.append(_make_storage_uniform(attrib_buf,   1));
    uniforms.append(_make_storage_uniform(chunk_buf,    2));
    uniforms.append(_make_storage_uniform(runnable_buf, 3));
    uniforms.append(_make_storage_uniform(sort_key_buf, 4));
    physics_uniform_set = rd->uniform_set_create(uniforms, physics_shader, 0);

    UtilityFunctions::print(
        "FluidParticleSystem: physics shader reloaded from ", physics_shader_path);
}

// ─────────────────────────────────────────────────────────────────────────────
// Load a .glsl compute shader resource and compile it into an RD shader RID.
// Uses CACHE_MODE_REPLACE so on-disk edits are picked up on every call (this
// matters for reload_physics_shader(); the initial build path also benefits
// from bypassing any stale cache).
// ─────────────────────────────────────────────────────────────────────────────
RID FluidParticleSystem::_compile_compute_shader(const String &res_path) {
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
// Build a single storage-buffer RDUniform. Static helper shared by the initial
// build path and reload_physics_shader().
// ─────────────────────────────────────────────────────────────────────────────
Ref<RDUniform> FluidParticleSystem::_make_storage_uniform(RID buf, uint32_t binding) {
    Ref<RDUniform> u;
    u.instantiate();
    u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
    u->set_binding(binding);
    u->add_id(buf);
    return u;
}
