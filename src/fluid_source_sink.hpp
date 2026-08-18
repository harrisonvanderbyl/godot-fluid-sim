#pragma once
#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace godot {

class FluidParticleSystem;

// ─────────────────────────────────────────────────────────────────────────────
// FluidSourceBase — shared logic for FluidSource and FluidSink.
// Both are Node3D children of a FluidParticleSystem. On _process they dispatch
// a compute shader against the parent's particle buffers to activate (source)
// or deactivate (sink) particles near their world-space position.
//
// Inactive sentinel: position.x = NaN. No extra per-particle field needed.
//
// LIFETIME NOTE: the pipeline here is built against the PARENT's storage
// buffers and the PARENT's local RenderingDevice. Neither is ours. If the
// parent rebuilds (grid resize, particle-count change, tree exit) our
// uniform_set points at freed buffers and `rd` points at a destroyed device.
// Two mechanisms guard that:
//   1. on_parent_gpu_reset() — the parent calls this before it frees anything.
//   2. built_generation      — compared against the parent's gpu_generation on
//                              every dispatch, in case (1) is ever missed.
// _destroy_pipeline() additionally refuses to free RIDs unless the parent's
// device is still the one we built against.
// ─────────────────────────────────────────────────────────────────────────────
class FluidSourceBase : public Node3D {
    GDCLASS(FluidSourceBase, Node3D)

public:
    FluidSourceBase();
    ~FluidSourceBase();

    void _enter_tree() override;
    void _exit_tree() override;

    // Called by parent FluidParticleSystem before it submits its compute list.
    // Adds source/sink compute to the same RD command buffer — no separate submit.
    void dispatch_into_parent();

    // Called by the parent immediately before it frees the buffers we bind.
    // Drops our pipeline so the next dispatch rebuilds against the new generation.
    void on_parent_gpu_reset();

    // Inspector properties
    void set_radius(float v)  { radius = v; }
    float get_radius()  const { return radius; }
    void set_rate(int v)      { rate = v; }
    int  get_rate()     const { return rate; }
    void set_active(bool v)   { active = v; }
    bool get_active()   const { return active; }
    // Non-trivial: tears the pipeline down so it rebuilds from the new path.
    void set_shader_path(const String &v);
    String get_shader_path() const { return shader_path; }

protected:
    static void _bind_methods();

    // Fires on NOTIFICATION_PREDELETE. No `override` — godot-cpp's GDCLASS
    // detects this by name, it is not a virtual on Object.
    void _notification(int p_what);

    // Subclass selects source (0) or sink (1)
    virtual int get_mode() const = 0;

private:
    float  radius       = 5.0f;
    int    rate         = 10;       // particles per frame
    bool   active       = true;
    String shader_path  = "res://addons/fluid_particles/shaders/fluid_source_sink.glsl";

    // Parent held by ObjectID, not a raw pointer: a raw cached pointer survives
    // the parent being freed and turns the next dispatch into a dangling
    // dereference.
    ObjectID             parent_id;
    RenderingDevice     *rd = nullptr;   // borrowed from the parent, never owned

    RID   shader_rid;
    RID   pipeline;
    RID   uniform_set;
    bool  gpu_ready      = false;
    bool  tried_init     = false;  // lazy init flag; reset by _destroy_pipeline
    // The parent's gpu_generation at the time we built. Zero means "not built".
    uint64_t built_generation = 0;

    FluidParticleSystem *_parent() const;

    void _build_pipeline(FluidParticleSystem *ps);
    // Idempotent. Nulls every RID and resets tried_init so a rebuild is possible.
    void _destroy_pipeline();
    void _dispatch(FluidParticleSystem *ps);
};

// ─────────────────────────────────────────────────────────────────────────────
// FluidSource — spawns inactive particles near itself with configurable
// color, attraction, and opacity.
// ─────────────────────────────────────────────────────────────────────────────
class FluidSource : public FluidSourceBase {
    GDCLASS(FluidSource, FluidSourceBase)

public:
    FluidSource() = default;

    void set_color(Color v)     { color = v; }
    Color get_color()     const { return color; }
    void set_attraction(float v) { attraction = v; }
    float get_attraction() const { return attraction; }
    void set_opacity(float v)   { opacity = v; }
    float get_opacity()   const { return opacity; }

protected:
    static void _bind_methods();
    int get_mode() const override { return 0; }

private:
    // Default: water-like liquid. Alpha < 200 means liquid (not solid).
    // Color is a translucent blue. Attraction matches water_viscosity default (1.0).
    Color color      = Color(0.3f, 0.5f, 0.8f, 0.3f);
    float attraction = 1.0f;
    float opacity    = 1.0f;
};

// ─────────────────────────────────────────────────────────────────────────────
// FluidSink — removes (deactivates) active particles near itself.
// ─────────────────────────────────────────────────────────────────────────────
class FluidSink : public FluidSourceBase {
    GDCLASS(FluidSink, FluidSourceBase)

public:
    FluidSink() = default;

protected:
    static void _bind_methods();
    int get_mode() const override { return 1; }
};

}  // namespace godot