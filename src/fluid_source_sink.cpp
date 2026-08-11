#include "fluid_source_sink.hpp"
#include "fluid_particle_system.hpp"

#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <cstring>

using namespace godot;

// ─────────────────────────────────────────────────────────────────────────────
// Push constants for the source/sink shader (48 bytes, 16-byte aligned)
// ─────────────────────────────────────────────────────────────────────────────
struct SourceSinkPC {
    float  world_pos[3];
    float  radius;
    float  color[4];
    float  attraction;
    float  opacity_fade;
    int    mode;
    int    max_count;
    int    num_particles;
    int    grid_w, grid_h, grid_d;
    int32_t vertex_stride_floats;
    int32_t attrib_stride_words;
    int32_t color_offset_words;
    int32_t custom0_offset_words;
};

// ─────────────────────────────────────────────────────────────────────────────
// FluidSourceBase
// ─────────────────────────────────────────────────────────────────────────────
FluidSourceBase::FluidSourceBase() {}
FluidSourceBase::~FluidSourceBase() {}

void FluidSourceBase::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_radius", "v"),  &FluidSourceBase::set_radius);
    ClassDB::bind_method(D_METHOD("get_radius"),       &FluidSourceBase::get_radius);
    ClassDB::bind_method(D_METHOD("set_rate", "v"),    &FluidSourceBase::set_rate);
    ClassDB::bind_method(D_METHOD("get_rate"),         &FluidSourceBase::get_rate);
    ClassDB::bind_method(D_METHOD("set_active", "v"),  &FluidSourceBase::set_active);
    ClassDB::bind_method(D_METHOD("get_active"),       &FluidSourceBase::get_active);
    ClassDB::bind_method(D_METHOD("set_shader_path", "v"), &FluidSourceBase::set_shader_path);
    ClassDB::bind_method(D_METHOD("get_shader_path"),      &FluidSourceBase::get_shader_path);

    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.1,100,0.1"),
        "set_radius", "get_radius");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "rate", PROPERTY_HINT_RANGE, "1,1000,1"),
        "set_rate", "get_rate");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "active"),
        "set_active", "get_active");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "shader_path", PROPERTY_HINT_FILE, "*.glsl"),
        "set_shader_path", "get_shader_path");
}

void FluidSourceBase::_enter_tree() {
    // Just mark parent — actual GPU init deferred to first _process
    Node *p = get_parent();
    while (p) {
        parent_system = Object::cast_to<FluidParticleSystem>(p);
        if (parent_system) break;
        p = p->get_parent();
    }
    if (!parent_system) {
        UtilityFunctions::printerr("FluidSource/Sink: no FluidParticleSystem ancestor found.");
    }
}

void FluidSourceBase::dispatch_into_parent() {
    if (!active || !parent_system) return;

    // Lazy init: wait until parent has GPU resources ready
    if (!gpu_ready) {
        if (tried_init) return;
        rd = parent_system->get_rd();
        if (!rd) return;  // parent not ready yet, try again next frame
        _build_pipeline();
        tried_init = true;
        return;  // don't dispatch on the init frame
    }

    if (!pipeline.is_valid() || !uniform_set.is_valid()) return;
    _dispatch();
}

void FluidSourceBase::_exit_tree() {
    _destroy_pipeline();
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidSourceBase::_build_pipeline() {
    if (!rd) return;

    Ref<RDShaderFile> sf = ResourceLoader::get_singleton()->load(shader_path);
    if (!sf.is_valid()) {
        UtilityFunctions::printerr("FluidSource/Sink: cannot load shader: ", shader_path);
        return;
    }
    Ref<RDShaderSPIRV> spirv = sf->get_spirv();
    if (!spirv.is_valid()) {
        UtilityFunctions::printerr("FluidSource/Sink: no SPIR-V from: ", shader_path);
        return;
    }
    String err = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_COMPUTE);
    if (!err.is_empty()) {
        UtilityFunctions::printerr("FluidSource/Sink: shader error: ", err);
        return;
    }
    shader_rid = rd->shader_create_from_spirv(spirv);
    pipeline   = rd->compute_pipeline_create(shader_rid);

    // Uniform set: binding 0 = particle_buf, binding 1 = chunk_buf (shared with parent)
    RID particle_buf = parent_system->get_vertex_buf();
    RID attribute_buf = parent_system->get_attribute_buf();
    RID chunk_buf    = parent_system->get_chunk_buf();
    if (!particle_buf.is_valid()) {
        UtilityFunctions::printerr("FluidSource/Sink: parent particle_buf is invalid.");
        return;
    }
    if (!attribute_buf.is_valid()) {
        UtilityFunctions::printerr("FluidSource/Sink: parent attribute_buf is invalid.");
        return;
    }
    if (!chunk_buf.is_valid()) {
        UtilityFunctions::printerr("FluidSource/Sink: parent chunk_buf is invalid.");
        return;
    }

    auto make_storage_uniform = [](RID buf, uint32_t binding) -> Ref<RDUniform> {
        Ref<RDUniform> u;
        u.instantiate();
        u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
        u->set_binding(binding);
        u->add_id(buf);
        return u;
    };

    TypedArray<RDUniform> uniforms;
    uniforms.append(make_storage_uniform(particle_buf, 0));
    uniforms.append(make_storage_uniform(attribute_buf, 1));
    if (chunk_buf.is_valid()) {
        uniforms.append(make_storage_uniform(chunk_buf, 2));
    }
    uniform_set = rd->uniform_set_create(uniforms, shader_rid, 0);

    gpu_ready = true;
}

void FluidSourceBase::_destroy_pipeline() {
    if (!rd) return;
    gpu_ready = false;
    if (uniform_set.is_valid()) rd->free_rid(uniform_set);
    if (pipeline.is_valid())    rd->free_rid(pipeline);
    if (shader_rid.is_valid())  rd->free_rid(shader_rid);
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidSourceBase::_dispatch() {
    if (!pipeline.is_valid() || !uniform_set.is_valid()) return;

    // Transform our world position into the parent system's local space
    // (the grid is in parent-local coordinates)
    Vector3 local_pos = parent_system->to_local(get_global_position());

    SourceSinkPC pc{};
    pc.world_pos[0]  = local_pos.x;
    pc.world_pos[1]  = local_pos.y;
    pc.world_pos[2]  = local_pos.z;
    pc.radius        = radius;
    pc.mode          = get_mode();
    pc.max_count     = rate;
    pc.num_particles = parent_system->get_num_particles();
    pc.grid_w        = parent_system->get_grid_w();
    pc.grid_h        = parent_system->get_grid_h();
    pc.grid_d        = parent_system->get_grid_d();
    pc.vertex_stride_floats = parent_system->get_vertex_stride_floats();
    pc.attrib_stride_words  = parent_system->get_attrib_stride_words();
    pc.color_offset_words   = parent_system->get_color_offset_words();
    pc.custom0_offset_words = parent_system->get_custom0_offset_words();

    // Fill source-specific attributes (subclass overrides for sink, but the
    // values are ignored by the shader in sink mode)
    FluidSource *src = Object::cast_to<FluidSource>(this);
    if (src) {
        Color c = src->get_color();
        pc.color[0]      = c.r;
        pc.color[1]      = c.g;
        pc.color[2]      = c.b;
        pc.color[3]      = c.a;
        pc.attraction    = src->get_attraction();
        pc.opacity_fade  = src->get_opacity();
    } else {
        pc.color[0] = 0; pc.color[1] = 0; pc.color[2] = 0; pc.color[3] = 0;
        pc.attraction   = 0;
        pc.opacity_fade = 0;
    }

    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(SourceSinkPC));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(SourceSinkPC));

    // Dispatch enough groups to cover all particles (strided scan)
    int np = pc.num_particles;
    uint32_t groups = (uint32_t)((np + 63) / 64);

    int64_t cl = rd->compute_list_begin();
    rd->compute_list_bind_compute_pipeline(cl, pipeline);
    rd->compute_list_bind_uniform_set(cl, uniform_set, 0);
    rd->compute_list_set_push_constant(cl, pc_bytes, sizeof(SourceSinkPC));
    rd->compute_list_dispatch(cl, groups, 1, 1);
    rd->compute_list_end();
    // No submit — parent owns the submit/sync cycle.
}

// ─────────────────────────────────────────────────────────────────────────────
// FluidSource
// ─────────────────────────────────────────────────────────────────────────────
void FluidSource::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_color", "v"),     &FluidSource::set_color);
    ClassDB::bind_method(D_METHOD("get_color"),          &FluidSource::get_color);
    ClassDB::bind_method(D_METHOD("set_attraction", "v"), &FluidSource::set_attraction);
    ClassDB::bind_method(D_METHOD("get_attraction"),     &FluidSource::get_attraction);
    ClassDB::bind_method(D_METHOD("set_opacity", "v"),   &FluidSource::set_opacity);
    ClassDB::bind_method(D_METHOD("get_opacity"),        &FluidSource::get_opacity);

    ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"),
        "set_color", "get_color");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attraction", PROPERTY_HINT_RANGE, "-2,2,0.01"),
        "set_attraction", "get_attraction");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "opacity", PROPERTY_HINT_RANGE, "0,1,0.01"),
        "set_opacity", "get_opacity");
}

// ─────────────────────────────────────────────────────────────────────────────
// FluidSink
// ─────────────────────────────────────────────────────────────────────────────
void FluidSink::_bind_methods() {}
