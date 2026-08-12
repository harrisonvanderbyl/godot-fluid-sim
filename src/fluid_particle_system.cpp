#include "fluid_particle_system.hpp"
#include "fluid_source_sink.hpp"

#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/classes/immediate_mesh.hpp>
#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/shader.hpp>
#include <godot_cpp/classes/shader_material.hpp>
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

    // Shader paths group
    ADD_GROUP("Shaders", "");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "clear_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_clear_shader_path",   "get_clear_shader_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "physics_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_physics_shader_path", "get_physics_shader_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "sortkey_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_sortkey_shader_path", "get_sortkey_shader_path");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "render_material",
        PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial"), "set_render_material", "get_render_material");
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

    // Initial chunk fill group
    ADD_GROUP("Initial Chunk", "initial_chunk_");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,  "use_initial_chunk"),  "set_use_initial_chunk",  "get_use_initial_chunk");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3,  "origin"),  "set_initial_chunk_origin",  "get_initial_chunk_origin");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "size"),   "set_initial_chunk_size",   "get_initial_chunk_size");
    ADD_PROPERTY(PropertyInfo(Variant::COLOR,  "color"),    "set_initial_chunk_color",   "get_initial_chunk_color");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attraction", PROPERTY_HINT_RANGE, "-2,2,0.01"), "set_initial_chunk_attraction", "get_initial_chunk_attraction");
    ADD_GROUP("", "");
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
        // particle render shader so the user can edit code + uniforms immediately.
        Ref<Shader> shader = render_material->get_shader();
        if (shader.is_null() || shader->get_code().is_empty()) {
            render_material->set_shader(_load_default_render_shader());
        }
    }

    // If the render node already exists, swap the material override live.
    // Using the user's ShaderMaterial directly means shader/uniform edits
    // are reflected instantly by Godot's own resource signaling.
    if (render_node) {
        if (render_material.is_valid()) {
            render_node->set_material_override(render_material);
        } else if (internal_material.is_valid()) {
            render_node->set_material_override(internal_material);
        }
    }
}

Ref<ShaderMaterial> FluidParticleSystem::get_render_material() const {
    return render_material;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ensure the render node exists (lazy-created): a MeshInstance3D that displays
// the render_mesh built in _build_gpu_resources(). The mesh's own vertex and
// attribute RD storage buffers are written to directly by the compute
// shaders — no CPU readback or per-frame rebuild.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_ensure_render_node() {
    if (render_node) return;
    if (!render_mesh.is_valid()) return;

    // If the user hasn't set a render_material, create an internal default
    // material from the addon's particle_render.gdshader. This is NOT exposed
    // as the property — the property stays null so the user knows they're
    // using the built-in default.
    if (render_material.is_null() && internal_material.is_null()) {
        internal_material.instantiate();
        internal_material->set_shader(_load_default_render_shader());
    }

    // Use the user's material if set, otherwise the internal default.
    Ref<ShaderMaterial> mat = render_material.is_valid() ? render_material : internal_material;

    render_node = memnew(MeshInstance3D);
    render_node->set_mesh(render_mesh);
    render_node->set_material_override(mat);
    render_node->set_custom_aabb(get_grid_aabb());
    add_child(render_node);
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_enter_tree() {
    _build_gpu_resources();
    _ensure_render_node();
}

void FluidParticleSystem::_exit_tree() {
    _destroy_gpu_resources();
    // render_node is a child and freed automatically
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_build_gpu_resources() {
    // Use the shared RenderingDevice (RenderingServer's main device). This is
    // required so the RIDs returned by mesh_surface_get_vertex_buffer_rd_rid()/
    // mesh_surface_get_attribute_buffer_rd_rid() can be bound directly in our
    // own compute pipelines — RIDs from a local device are not valid on the
    // shared device (and vice versa), so a local device cannot be used here.
    RenderingServer *rs = RenderingServer::get_singleton();
    rd = rs->get_rendering_device();
    if (!rd) {
        UtilityFunctions::printerr("FluidParticleSystem: Could not get shared RenderingDevice.");
        return;
    }

    // ── Render mesh + storage-buffer-backed vertex/attribute buffers ────────
    // Position lives in the mesh's vertex buffer, color+custom0 (attraction_
    // force, opacity_fade, neighbors_filled) in its attribute buffer. Compute
    // shaders write directly into these buffers — the renderer reads the same
    // memory with zero copies.
    {
        const uint32_t nan_bits = 0x7FC00000u;
        float nan_val;
        std::memcpy(&nan_val, &nan_bits, sizeof(float));

        PackedVector3Array positions;
        PackedColorArray   colors;
        PackedFloat32Array custom0;
        positions.resize(num_particles);
        colors.resize(num_particles);
        custom0.resize(num_particles * 4);
        Vector3 *pp = positions.ptrw();
        Color   *cp = colors.ptrw();
        float   *c0 = custom0.ptrw();

        for (int i = 0; i < num_particles; i++) {
            pp[i] = Vector3(nan_val, 0.0f, 0.0f);  // inactive sentinel
            cp[i] = initial_chunk_color;
            c0[i*4 + 0] = initial_chunk_attraction;
            c0[i*4 + 1] = 1.0f;
            c0[i*4 + 2] = 0.0f;
            c0[i*4 + 3] = 0.0f;
        }

        // If initial chunk is enabled, activate particles in a block.
        // All liquid so they repulse each other and flow.
        if (use_initial_chunk) {
            Color solid_col = Color(0.0f, 0.0f, 0.0f, 17.0f / 255.0f); // alpha=0x11 -> liquid
            int idx = 0;
            for (int iz = 0; iz < initial_chunk_size.z && idx < num_particles; iz++) {
                for (int iy = 0; iy < initial_chunk_size.y && idx < num_particles; iy++) {
                    for (int ix = 0; ix < initial_chunk_size.x && idx < num_particles; ix++, idx++) {
                        pp[idx] = Vector3(initial_chunk_origin.x + ix, initial_chunk_origin.y + iy, initial_chunk_origin.z + iz);
                        cp[idx] = solid_col;
                        c0[idx*4 + 0] = water_viscosity;
                        c0[idx*4 + 1] = 1.0f;
                        c0[idx*4 + 2] = 0.0f;
                        c0[idx*4 + 3] = 0.0f;
                    }
                }
            }
        }

        Array arrays;
        arrays.resize(Mesh::ARRAY_MAX);
        arrays[Mesh::ARRAY_VERTEX]  = positions;
        arrays[Mesh::ARRAY_COLOR]   = colors;
        arrays[Mesh::ARRAY_CUSTOM0] = custom0;

        render_mesh.instantiate();
        render_mesh->add_surface_from_arrays(
            Mesh::PRIMITIVE_POINTS, arrays, TypedArray<Array>(), Dictionary(),
            (BitField<Mesh::ArrayFormat>)(
                ((int64_t)Mesh::ARRAY_CUSTOM_RGBA_FLOAT << Mesh::ARRAY_FORMAT_CUSTOM0_SHIFT)
                | Mesh::ARRAY_FLAG_USE_STORAGE_BUFFER));

        RID mesh_rid = render_mesh->get_rid();
        vertex_buf = rs->mesh_surface_get_vertex_buffer_rd_rid(mesh_rid, 0);
        attrib_buf = rs->mesh_surface_get_attribute_buffer_rd_rid(mesh_rid, 0);
        if (!vertex_buf.is_valid() || !attrib_buf.is_valid()) {
            UtilityFunctions::printerr("FluidParticleSystem: could not fetch mesh RD storage buffers. "
                "Requires Godot 4.8+ with mesh_surface_get_vertex_buffer_rd_rid/attribute_buffer_rd_rid exposed.");
            return;
        }

        uint64_t format = (1ULL << RenderingServer::ARRAY_VERTEX) | (1ULL << RenderingServer::ARRAY_COLOR) | (1ULL << RenderingServer::ARRAY_CUSTOM0);
        format |= (uint64_t)RenderingServer::ARRAY_CUSTOM_RGBA_FLOAT << RenderingServer::ARRAY_FORMAT_CUSTOM0_SHIFT;

        BitField<RenderingServer::ArrayFormat> bformat = (BitField<RenderingServer::ArrayFormat>)format;
        uint32_t vertex_stride_bytes = rs->mesh_surface_get_format_vertex_stride(bformat, num_particles);
        uint32_t attrib_stride_bytes = rs->mesh_surface_get_format_attribute_stride(bformat, num_particles);
        uint32_t color_offset_bytes  = rs->mesh_surface_get_format_offset(bformat, num_particles, RenderingServer::ARRAY_COLOR);
        uint32_t custom0_offset_bytes = rs->mesh_surface_get_format_offset(bformat, num_particles, RenderingServer::ARRAY_CUSTOM0);

        vertex_stride_floats = (int)(vertex_stride_bytes / 4);
        attrib_stride_words  = (int)(attrib_stride_bytes / 4);
        color_offset_words   = (int)(color_offset_bytes / 4);
        custom0_offset_words = (int)(custom0_offset_bytes / 4);
    }

    // ── Chunk grid buffer ────────────────────────────────────────────────────
    {
        int64_t cell_count = (int64_t)grid_width * grid_height * grid_depth;
        int64_t buf_size   = cell_count * sizeof(GPUChunkCell);
        PackedByteArray data;
        data.resize(buf_size);
        data.fill(0);
        // Set all occupants to -1
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
    // Loaded via Godot's resource system (imports .glsl -> RDShaderFile).

    // Load compute shaders via Godot's resource system (imports .glsl -> RDShaderFile)
    auto compile_shader = [&](const String &res_path) -> RID {
        Ref<RDShaderFile> sf = ResourceLoader::get_singleton()->load(res_path);
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
    };

    clear_shader   = compile_shader(clear_shader_path);
    physics_shader = compile_shader(physics_shader_path);
    sortkey_shader = compile_shader(sortkey_shader_path);

    if (clear_shader.is_valid())   clear_pipeline   = rd->compute_pipeline_create(clear_shader);
    if (physics_shader.is_valid()) physics_pipeline = rd->compute_pipeline_create(physics_shader);
    if (sortkey_shader.is_valid()) sortkey_pipeline = rd->compute_pipeline_create(sortkey_shader);

    // ── Build uniform sets ────────────────────────────────────────────────────
    // All three pipelines share the same buffer bindings:
    //   binding 0 → vertex buffer (position, owned by render_mesh)
    //   binding 1 → attribute buffer (color + custom0, owned by render_mesh)
    //   binding 2 → chunk_grid
    //   binding 3 → runnable_indices
    //   binding 4 → sort_keys

    auto make_storage_uniform = [](RID buf, uint32_t binding) -> Ref<RDUniform> {
        Ref<RDUniform> u;
        u.instantiate();
        u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
        u->set_binding(binding);
        u->add_id(buf);
        return u;
    };

    TypedArray<RDUniform> uniforms;
    uniforms.append(make_storage_uniform(vertex_buf,   0));
    uniforms.append(make_storage_uniform(attrib_buf,   1));
    uniforms.append(make_storage_uniform(chunk_buf,    2));
    uniforms.append(make_storage_uniform(runnable_buf, 3));
    uniforms.append(make_storage_uniform(sort_key_buf, 4));

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
    // Rendering goes through a normal MeshInstance3D (render_node) displaying
    // render_mesh, whose vertex/attribute RD buffers are written directly by
    // the compute shaders every frame (see _process). No CPU round-trip.
    gpu_ready = true;
    UtilityFunctions::print("FluidParticleSystem: GPU resources ready.");
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_destroy_gpu_resources() {
    if (!rd) return;
    gpu_ready = false;

    if (clear_uniform_set.is_valid())   rd->free_rid(clear_uniform_set);
    if (physics_uniform_set.is_valid()) rd->free_rid(physics_uniform_set);
    if (sortkey_uniform_set.is_valid()) rd->free_rid(sortkey_uniform_set);

    if (clear_pipeline.is_valid())   rd->free_rid(clear_pipeline);
    if (physics_pipeline.is_valid()) rd->free_rid(physics_pipeline);
    if (sortkey_pipeline.is_valid()) rd->free_rid(sortkey_pipeline);
    if (clear_shader.is_valid())     rd->free_rid(clear_shader);
    if (physics_shader.is_valid())   rd->free_rid(physics_shader);
    if (sortkey_shader.is_valid())   rd->free_rid(sortkey_shader);

    // vertex_buf/attrib_buf are owned by render_mesh (RenderingServer), not by
    // us — freed automatically when render_mesh is released. Do NOT free_rid
    // them here (they belong to the shared device's mesh storage).
    if (chunk_buf.is_valid())     rd->free_rid(chunk_buf);
    if (runnable_buf.is_valid())  rd->free_rid(runnable_buf);
    if (sort_key_buf.is_valid())  rd->free_rid(sort_key_buf);

    // rd is the shared RenderingServer device — do not delete it.
    rd = nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Called from property setters (num_particles, grid_*) when a size-affecting
// property changes while the node is in the tree. Tears down all GPU resources
// and rebuilds them from scratch with the new dimensions.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_rebuild_gpu_resources() {
    if (!is_inside_tree()) return;  // _enter_tree will build on first entry
    _destroy_gpu_resources();

    // render_mesh owns vertex_buf/attrib_buf; release it so the old mesh (and
    // its storage buffers) are freed before we create new ones.
    if (render_mesh.is_valid()) {
        render_mesh.unref();
    }
    // render_node still references the old mesh; update it after rebuild.
    if (render_node) {
        render_node->set_mesh(Ref<ArrayMesh>());
    }

    _build_gpu_resources();
    _ensure_render_node();
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_process(double delta) {
    if (!gpu_ready) return;

    // Simulation can be paused from the editor. The render mesh already
    // reflects the last GPU state, so there's nothing to do when paused.
    if (!simulation_active) return;

    // ── Compute impulses from node movement ───────────────────────────────
    // The fluid lives in the node's local (grid) space. When the node moves or
    // rotates in the world, the fluid — which has its own momentum in grid
    // space — should resist that motion as if it were a separate body.
    //
    // We compute the full delta transform: D = T_cur^-1 * T_prev.
    // For a particle at grid position p, the point where it "would have been"
    // (in the new grid space) is D(p) = D.basis * p + D.origin.
    // The displacement is dp = D(p) - p. The shader subtracts dp from each
    // particle's position, giving it inertia. This correctly handles:
    //   - Pure translation (uniform shift)
    //   - Rotation about the container center (tangential field)
    //   - Rotation about an off-center point (swinging on a string)
    //   - Any combination of translation + rotation
    // Accumulate explicit rotational/transform impulses from script calls.
    // add_velocity_impulse and add_rotational_impulse both compose into the
    // delta transform (pending_delta_basis / pending_delta_origin).
    Basis delta_basis = Basis();  // identity
    Vector3 delta_origin = Vector3();
    bool has_delta = false;

    if (delta_impulse_pending) {
        delta_basis = pending_delta_basis;
        delta_origin = pending_delta_origin;
        has_delta = true;
        delta_impulse_pending = false;
    }

    // Transform-delta impulse from node movement.
    Transform3D cur_xform = get_global_transform();
    if (has_prev_transform) {
        // D = T_prev^-1 * T_cur — the container's forward motion.
        // The shader computes dp = D(p) - p (how the container moved) and
        // subtracts it, so particles resist the motion (inertia).
        // affine_inverse handles scale; plain inverse assumes rotation only.
        Transform3D delta_xform = prev_global_transform.affine_inverse() * cur_xform;

        if (has_delta) {
            // Compose: existing explicit delta * node-movement delta
            delta_basis = delta_xform.basis * delta_basis;
            delta_origin = delta_xform.basis.xform(delta_origin) + delta_xform.origin;
        } else {
            delta_basis = delta_xform.basis;
            delta_origin = delta_xform.origin;
        }
        has_delta = true;
    }
    prev_global_transform = cur_xform;
    has_prev_transform = true;

    // NOTE: No per-frame clear — the chunk grid stores persistent velocity state
    // that is read-consumed by swap0() and re-written by atomic_plus_equals().
    // Clearing every frame would zero all velocities and force firststep=true on
    // every particle, breaking the simulation. The grid is initialized once at
    // _build_gpu_resources() and only cleared explicitly via reset_grid().

    // 1. Physics step (velocity_spread.glsl)
    _dispatch_physics(Vector3(), delta_basis, delta_origin, has_delta);

    // 2. Source/sink children dispatch into the same RD command buffer
    for (int i = 0; i < get_child_count(); i++) {
        FluidSourceBase *ss = Object::cast_to<FluidSourceBase>(get_child(i));
        if (ss) ss->dispatch_into_parent();
    }

    // 3. Sort keys (every 60 frames)
    if (frame_count % 60 == 0) {
        _dispatch_sortkey();
    }

    // rd is the shared RenderingDevice (owned by RenderingServer) — we must
    // NOT call submit()/sync() ourselves. The compute lists recorded above
    // are submitted automatically as part of the engine's own frame, and the
    // render mesh's vertex/attribute buffers are read directly by the
    // renderer with no CPU round-trip.
    frame_count++;
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
}
