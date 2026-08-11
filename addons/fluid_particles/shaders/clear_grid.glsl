#[compute]
#version 450

// ─── Clear grid shader ───────────────────────────────────────────────────────
// Resets every ChunkCell to: occupant = -1, velocity = (0,0,0,0)
// Dispatched with 64 threads/group over all cells.
// ─────────────────────────────────────────────────────────────────────────────

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// ── Push constants ────────────────────────────────────────────────────────────
layout(push_constant, std430) uniform PushConstants {
    int   grid_w, grid_h, grid_d;
    int   num_particles;
    vec3 gravity;
    float surface_tension;
    float water_viscosity;
    float _pad0;
    vec3  global_vel;
    int   frame_count;
    int   neighbor_mode;
    int   num_runnable;
    int   vertex_stride_floats;
    int   attrib_stride_words;
    int   color_offset_words;
    int   custom0_offset_words;
} pc;

// ── Buffers ───────────────────────────────────────────────────────────────────
// Particle position/color/custom0 now live directly inside the render mesh's
// own vertex/attribute storage buffers (bindings 0/1), not a separate buffer.
// This shader doesn't touch them, but declares them to keep the uniform set
// layout identical across clear/physics/sortkey pipelines.
struct ChunkCell {
    int  occupant;
    uint vel_x_bits;
    uint vel_y_bits;
    uint vel_z_bits;
    uint vel_w_bits;
    uint _pad[3];
};

layout(set = 0, binding = 0, std430) buffer VertexBuffer   { float vtx[]; };
layout(set = 0, binding = 1, std430) buffer AttribBuffer   { uint  atr[]; };
layout(set = 0, binding = 2, std430) buffer ChunkBuffer    { ChunkCell cells[]; };
layout(set = 0, binding = 3, std430) buffer RunnableBuffer { int   runnable_indices[]; };
layout(set = 0, binding = 4, std430) buffer SortKeyBuffer  { float sort_keys[]; };

// ─────────────────────────────────────────────────────────────────────────────
void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint cell_count = uint(pc.grid_w) * uint(pc.grid_h) * uint(pc.grid_d);
    if (gid >= cell_count) return;

    cells[gid].occupant   = -1;
    cells[gid].vel_x_bits = 0u;
    cells[gid].vel_y_bits = 0u;
    cells[gid].vel_z_bits = 0u;
    cells[gid].vel_w_bits = 0u;
}
