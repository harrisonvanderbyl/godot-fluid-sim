#include "fluid_source_sink.hpp"
#include "fluid_particle_system.hpp"

#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <cstring>

using namespace godot;

// ─────────────────────────────────────────────────────────────────────────────
// Push constants for the source/sink shader (16-byte aligned)
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
    int32_t lod_levels;
    int32_t _pad[3];   // pad to 96 bytes (16-byte aligned)
};

static_assert(sizeof(SourceSinkPC) % 16 == 0,
              "SourceSinkPC must be 16-byte aligned for std430");
static_assert(sizeof(SourceSinkPC) <= 128,
              "SourceSinkPC must fit the 128-byte push constant guarantee");

// ─────────────────────────────────────────────────────────────────────────────
// FluidSourceBase
// ─────────────────────────────────────────────────────────────────────────────
FluidSourceBase::FluidSourceBase() {}

FluidSourceBase::~FluidSourceBase() {
    // If we still hold RIDs at this point the parent device is either gone
    // (in which case _destroy_pipeline correctly skips freeing) or still alive
    // (in which case this is the last chance to give them back).
    _destroy_pipeline();
}

void FluidSourceBase::_notification(int p_what) {
    if (p_what == NOTIFICATION_PREDELETE) {
        _destroy_pipeline();
    }
}

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

// Changing the shader path invalidates the built pipeline.
void FluidSourceBase::set_shader_path(const String &v) {
    if (shader_path == v) return;
    shader_path = v;
    _destroy_pipeline();   // resets tried_init, so the next frame rebuilds
}

// ─────────────────────────────────────────────────────────────────────────────
// Resolve the parent system through ObjectID rather than a raw pointer. The old
// code cached FluidParticleSystem* in _enter_tree and never cleared it, so any
// path where the parent was freed first left a dangling dereference.
// ─────────────────────────────────────────────────────────────────────────────
FluidParticleSystem *FluidSourceBase::_parent() const {
    if (!parent_id.is_valid()) return nullptr;
    return Object::cast_to<FluidParticleSystem>(ObjectDB::get_instance(parent_id));
}

void FluidSourceBase::_enter_tree() {
    parent_id = ObjectID();

    Node *p = get_parent();
    while (p) {
        if (FluidParticleSystem *ps = Object::cast_to<FluidParticleSystem>(p)) {
            parent_id = ps->get_instance_id();
            break;
        }
        p = p->get_parent();
    }
    if (!parent_id.is_valid()) {
        UtilityFunctions::printerr("FluidSource/Sink: no FluidParticleSystem ancestor found.");
    }
    // Force a fresh build against whatever generation the parent is on now.
    tried_init = false;
    built_generation = 0;
}

void FluidSourceBase::_exit_tree() {
    // Godot propagates EXIT_TREE to children before the parent, so the parent's
    // device is still alive here and we can hand our RIDs back cleanly.
    _destroy_pipeline();
    parent_id = ObjectID();
}

// ─────────────────────────────────────────────────────────────────────────────
// Called by FluidParticleSystem immediately before it frees the storage buffers
// our uniform set points at. Without this hook, a grid-size or particle-count
// change in the inspector left every source holding a uniform set bound to
// freed buffers plus a raw pointer to a destroyed device — the next dispatch
// bound both.
// ─────────────────────────────────────────────────────────────────────────────
void FluidSourceBase::on_parent_gpu_reset() {
    _destroy_pipeline();
}

void FluidSourceBase::dispatch_into_parent() {
    if (!active) return;

    FluidParticleSystem *ps = _parent();
    if (!ps || !ps->is_gpu_ready()) return;

    RenderingDevice *dev = ps->get_rd();
    if (!dev) return;

    // Detect a parent rebuild we were not told about (device swapped, or the
    // generation counter moved). Cheap insurance on top of on_parent_gpu_reset.
    if (gpu_ready && (dev != rd || built_generation != ps->get_gpu_generation())) {
        _destroy_pipeline();
    }

    if (!gpu_ready) {
        if (tried_init) return;      // build already failed this generation
        tried_init = true;
        rd = dev;
        _build_pipeline(ps);
        return;                      // don't dispatch on the init frame
    }

    if (!pipeline.is_valid() || !uniform_set.is_valid()) return;
    _dispatch(ps);
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidSourceBase::_build_pipeline(FluidParticleSystem *ps) {
    if (!rd || !ps) return;

    RID particle_buf  = ps->get_vertex_buf();
    RID attribute_buf = ps->get_attribute_buf();
    RID chunk_buf     = ps->get_chunk_buf();
    RID arena_buf     = ps->get_cell_arena_buf();
    RID lod_table_buf = ps->get_lod_table_buf();
    if (!particle_buf.is_valid() || !attribute_buf.is_valid() || !chunk_buf.is_valid()) {
        UtilityFunctions::printerr("FluidSource/Sink: parent buffers not ready.");
        return;
    }

    Ref<RDShaderFile> sf = ResourceLoader::get_singleton()->load(
        shader_path, "", ResourceLoader::CACHE_MODE_REPLACE);
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
    if (!shader_rid.is_valid()) {
        UtilityFunctions::printerr("FluidSource/Sink: failed to create shader RID.");
        return;
    }
    pipeline = rd->compute_pipeline_create(shader_rid);
    if (!pipeline.is_valid()) {
        UtilityFunctions::printerr("FluidSource/Sink: failed to create compute pipeline.");
        rd->free_rid(shader_rid);
        shader_rid = RID();
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
    uniforms.append(make_storage_uniform(particle_buf,  0));
    uniforms.append(make_storage_uniform(attribute_buf, 1));
    uniforms.append(make_storage_uniform(chunk_buf,     2));
    // Arena + LOD page table (same bindings as velocity_spread.glsl).
    if (arena_buf.is_valid())     uniforms.append(make_storage_uniform(arena_buf,     6));
    if (lod_table_buf.is_valid()) uniforms.append(make_storage_uniform(lod_table_buf, 7));
    uniform_set = rd->uniform_set_create(uniforms, shader_rid, 0);
    if (!uniform_set.is_valid()) {
        UtilityFunctions::printerr("FluidSource/Sink: failed to create uniform set.");
        rd->free_rid(pipeline);
        rd->free_rid(shader_rid);
        pipeline   = RID();
        shader_rid = RID();
        return;
    }

    // Stamp the generation we built against, so a later parent rebuild is
    // detected even if the reset notification is missed.
    built_generation = ps->get_gpu_generation();
    gpu_ready = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Idempotent teardown. The original never nulled the RIDs and never reset
// tried_init, so a second call double-freed and a re-enter could never rebuild.
// ─────────────────────────────────────────────────────────────────────────────
void FluidSourceBase::_destroy_pipeline() {
    gpu_ready  = false;
    tried_init = false;

    FluidParticleSystem *ps = _parent();

    // Only free if the device we built against is STILL the parent's device.
    // If the parent tore down or swapped its device, these RIDs are already
    // gone and freeing them would hit a destroyed allocator.
    bool device_still_ours = (rd != nullptr) && ps && (ps->get_rd() == rd);

    if (device_still_ours) {
        if (uniform_set.is_valid()) rd->free_rid(uniform_set);
        if (pipeline.is_valid())    rd->free_rid(pipeline);
        if (shader_rid.is_valid())  rd->free_rid(shader_rid);
    }

    uniform_set = RID();
    pipeline    = RID();
    shader_rid  = RID();
    rd          = nullptr;
    built_generation = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidSourceBase::_dispatch(FluidParticleSystem *ps) {
    if (!rd || !ps) return;
    if (!pipeline.is_valid() || !uniform_set.is_valid()) return;

    // Transform our world position into the parent system's local space
    // (the grid is in parent-local coordinates)
    Vector3 local_pos = ps->to_local(get_global_position());

    SourceSinkPC pc{};
    pc.world_pos[0]  = local_pos.x;
    pc.world_pos[1]  = local_pos.y;
    pc.world_pos[2]  = local_pos.z;
    pc.radius        = radius;
    pc.mode          = get_mode();
    pc.max_count     = rate;
    pc.num_particles = ps->get_num_particles();
    pc.grid_w        = ps->get_grid_w();
    pc.grid_h        = ps->get_grid_h();
    pc.grid_d        = ps->get_grid_d();
    pc.vertex_stride_floats = ps->get_vertex_stride_floats();
    pc.attrib_stride_words  = ps->get_attrib_stride_words();
    pc.color_offset_words   = ps->get_color_offset_words();
    pc.custom0_offset_words = ps->get_custom0_offset_words();
    pc.lod_levels    = ps->get_lod_levels();

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

    if (pc.num_particles <= 0) return;

    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(SourceSinkPC));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(SourceSinkPC));

    // Dispatch enough groups to cover all particles (strided scan)
    uint32_t groups = (uint32_t)((pc.num_particles + 63) / 64);

    int64_t cl = rd->compute_list_begin();
    if (cl < 0) return;
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