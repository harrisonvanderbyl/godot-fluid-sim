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
// a compute shader against the parent's particle_buf to activate (source) or
// deactivate (sink) particles near their world-space position.
//
// Inactive sentinel: position.x = NaN. No extra per-particle field needed.
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

    // Inspector properties
    void set_radius(float v)  { radius = v; }
    float get_radius()  const { return radius; }
    void set_rate(int v)      { rate = v; }
    int  get_rate()     const { return rate; }
    void set_active(bool v)   { active = v; }
    bool get_active()   const { return active; }
    void set_shader_path(const String &v) { shader_path = v; pipeline_ready = false; }
    String get_shader_path() const { return shader_path; }

protected:
    static void _bind_methods();

    // Subclass selects source (0) or sink (1)
    virtual int get_mode() const = 0;

    // For source: the attributes to assign to spawned particles
    void fill_push_constants(PackedByteArray &pc_bytes);

private:
    float  radius       = 5.0f;
    int    rate         = 10;       // particles per frame
    bool   active       = true;
    String shader_path  = "res://addons/fluid_particles/shaders/fluid_source_sink.glsl";

    FluidParticleSystem *parent_system = nullptr;
    RenderingDevice     *rd            = nullptr;

    RID   shader_rid;
    RID   pipeline;
    RID   uniform_set;
    bool  pipeline_ready = false;
    bool  gpu_ready      = false;
    bool  tried_init     = false;  // lazy init flag

    void _build_pipeline();
    void _destroy_pipeline();
    void _dispatch();
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
