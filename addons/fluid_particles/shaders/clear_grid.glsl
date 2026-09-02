#[compute]
#version 450

// ─── Clear grid shader ───────────────────────────────────────────────────────
// Resets every ChunkCell to: occupant = -1, velocity = (0,0,0,0)
// Dispatched with 64 threads/group over all cells.
//
// When has_sdf is set, instead of zeroing the velocity field the shader bakes
// the SDF gradient (surface normal) into vel_*_bits. The physics shader then
// reads that velocity each frame, pushing particles out of the terrain — a
// collision-like reaction. The velocity field is persistent: clear_grid is
// only dispatched once (at init or via reset_grid / apply_sdf_to_velocity_field),
// NOT every frame.
// ─────────────────────────────────────────────────────────────────────────────

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// ── Push constants ────────────────────────────────────────────────────────────
// This is a 128-byte blob shared with the physics/sortkey shaders. clear_grid
// only uses grid_w/h/d, num_particles, and (when has_sdf is set) the SDF fields
// packed into the bytes the other shaders leave unused (offset 68+).
layout(push_constant, std430) uniform PushConstants {
    int   grid_w, grid_h, grid_d;    // offset  0, 4, 8
    int   num_particles;               // offset 12
    float _pad0[13];                   // offset 16-67  (unused by clear_grid)
    float has_sdf;                     // offset 68  (1.0 if SDF present)
    float _pad1;                       // offset 72  (unused, was max_occupancy)
    float sdf_strength;                // offset 76
    float sdf_w;                       // offset 80
    float sdf_h;                       // offset 84
    float sdf_d;                       // offset 88
    float sdf_offset_x;                // offset 92
    float sdf_offset_y;                // offset 96
    float sdf_offset_z;                // offset 100
    float sdf_scale;                   // offset 104
    float sdf_keep_occupant;           // offset 108  (1.0 = only update velocity, don't clear)
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
    uint sdf_bits;    // SDF value at this cell (float bits). <0 = inside terrain.
    uint _pad[3];
};

layout(set = 0, binding = 0, std430) buffer VertexBuffer   { float vtx[]; };
layout(set = 0, binding = 1, std430) buffer AttribBuffer   { uint  atr[]; };
layout(set = 0, binding = 2, std430) buffer ChunkBuffer    { ChunkCell cells[]; };
layout(set = 0, binding = 3, std430) buffer RunnableBuffer { int   runnable_indices[]; };
layout(set = 0, binding = 4, std430) buffer SortKeyBuffer  { float sort_keys[]; };

// SDF data: float[] in ZXY order (y + sdf_h * (x + sdf_w * z)), decoded on the
// CPU from the VoxelBuffer CHANNEL_SDF (handles 8/16/32-bit depths + the
// godot_voxel quantization scales). Bound to a dummy buffer when has_sdf=0.
layout(set = 0, binding = 5, std430) readonly buffer SDFBuffer { float sdf_data[]; };

// Sample the SDF at an integer voxel position. Out-of-bounds reads return a
// large positive value (far outside = no terrain), so boundary gradients
// correctly point away from the surface.
float sample_sdf(ivec3 pos) {
    if (pos.x < 0 || pos.x >= int(pc.sdf_w) ||
        pos.y < 0 || pos.y >= int(pc.sdf_h) ||
        pos.z < 0 || pos.z >= int(pc.sdf_d)) {
        return 100.0;  // SDF_FAR_OUTSIDE
    }
    int idx = pos.y + int(pc.sdf_h) * (pos.x + int(pc.sdf_w) * pos.z);
    return sdf_data[idx];
}

// ─────────────────────────────────────────────────────────────────────────────
void main() {
    uint gid = gl_GlobalInvocationID.x;
    // chunk_buf is only LOD_PAGE_CELLS (4096) cells — the LOD arena owns
    // the real cell storage. This shader's writes are dead (physics reads
    // SDF from sdf_storage_buf at binding 5), but we still guard against
    // out-of-bounds writes.
    if (gid >= 4096u) return;

    // Only clear occupant on a full reset, not on a velocity-only reload.
    if (pc.sdf_keep_occupant < 0.5) {
        cells[gid].occupant = -1;
    }

    if (pc.has_sdf > 0.5) {
        // Decode 3D position from the linear cell index (XYZ order).
        int x = int(gid % uint(pc.grid_w));
        int y = int((gid / uint(pc.grid_w)) % uint(pc.grid_h));
        int z = int(gid / (uint(pc.grid_w) * uint(pc.grid_h)));

        // Map fluid-grid position → SDF voxel position.
        vec3 sdf_pos = (vec3(float(x), float(y), float(z)) -
                        vec3(pc.sdf_offset_x, pc.sdf_offset_y, pc.sdf_offset_z)) /
                       pc.sdf_scale;
        ivec3 ipos = ivec3(round(sdf_pos));

        // Store the raw SDF value at this cell. <0 = inside terrain.
        float sdf = sample_sdf(ipos);
        cells[gid].sdf_bits = floatBitsToUint(sdf);

        // Velocity stays zero — the physics shader reads sdf_bits each frame
        // and applies collision directly.
        cells[gid].vel_x_bits = 0u;
        cells[gid].vel_y_bits = 0u;
        cells[gid].vel_z_bits = 0u;
    } else {
        cells[gid].vel_x_bits = 0u;
        cells[gid].vel_y_bits = 0u;
        cells[gid].vel_z_bits = 0u;
        cells[gid].sdf_bits = floatBitsToUint(100.0);  // far outside
    }
}
