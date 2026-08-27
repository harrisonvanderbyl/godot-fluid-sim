#[compute]
#version 450

// ─── Fluid Source / Sink compute shader ──────────────────────────────────────
// One shader handles both operations:
//   mode = 0  → SOURCE: find inactive particles (position.x == NaN) and
//                activate them within radius of the source position, setting
//                their attributes to the source's values.
//   mode = 1  → SINK:   find active particles within radius of the sink
//                position and deactivate them (set position.x = NaN).
//
// Inactive sentinel: position.x = NaN (0x7FC00000 when stored as uint bits).
// This costs zero extra memory — NaN is already representable in the float
// position field.
//
// Dispatch: 1 group of 64 threads. Each thread scans a strided subset of
// particles. Atomic operations ensure only the right number of particles are
// claimed/released.
// ─────────────────────────────────────────────────────────────────────────────

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant, std430) uniform PushConstants {
    vec3  world_pos;     // source/sink position in grid space
    float radius;        // effect radius
    vec4  color;         // color to assign (source mode only)
    float attraction;    // attraction_force to assign (source mode only)
    float opacity_fade;  // opacity_fade to assign (source mode only)
    int   mode;          // 0 = source, 1 = sink
    int   max_count;     // max particles to activate/deactivate this dispatch
    int   num_particles; // total particles in buffer
    int   grid_w, grid_h, grid_d;
    int   vertex_stride_floats;
    int   attrib_stride_words;
    int   color_offset_words;
    int   custom0_offset_words;
} pc;

// MUST match ChunkCell in velocity_spread.glsl (std430, 32 bytes).
struct ChunkCell {
    int  occupant;      // representative particle idx, or -1 if empty
    uint vel_x_bits;
    uint vel_y_bits;
    uint vel_z_bits;
    uint sdf_bits;      // SDF value at this cell (float bits). <0 = inside terrain.
    int  count;         // current occupancy
    uint _pad[2];
};

layout(set = 0, binding = 0, std430) buffer VertexBuffer { float vtx[]; };
layout(set = 0, binding = 1, std430) buffer AttribBuffer { vec4  atr[]; };
layout(set = 0, binding = 2, std430) buffer ChunkBuffer  { ChunkCell cells[]; };

vec3 get_position(int idx) {
    int base = idx * pc.vertex_stride_floats;
    return vec3(vtx[base + 0], vtx[base + 1], vtx[base + 2]);
}
void set_position(int idx, vec3 p) {
    int base = idx * pc.vertex_stride_floats;
    vtx[base + 0] = p.x; vtx[base + 1] = p.y; vtx[base + 2] = p.z;
}
void set_color(int idx, uint r, uint g, uint b, uint a) {
    atr[idx * 2] = vec4(r, g, b, a);
}
void set_custom0(int idx, vec4 v) {
    atr[idx * 2 + 1] = v;
}

int cell_index(ivec3 p) {
    p = clamp(p, ivec3(0), ivec3(pc.grid_w - 1, pc.grid_h - 1, pc.grid_d - 1));
    return p.x + p.y * pc.grid_w + p.z * pc.grid_w * pc.grid_h;
}

// Claim an empty cell for a newly-spawned particle. Mirrors the join half of
// cell_try_join from velocity_spread.glsl: set occupant, then increment count.
bool cell_try_place(int cidx, int pidx) {
    if (atomicCompSwap(cells[cidx].occupant, -1, pidx) != -1) return false;
    atomicAdd(cells[cidx].count, 1);
    return true;
}

// Release a cell from a sunk particle. Mirrors cell_leave from
// velocity_spread.glsl (decrement count, clear representative if the cell
// empties) and additionally drains the pooled velocity field so orphaned
// momentum doesn't create a ghost puff toward the now-empty cell on the next
// physics step.
void cell_leave_and_drain(int cidx) {
    int after = atomicAdd(cells[cidx].count, -1) - 1;
    if (after <= 0) {
        // Cell is empty — clear the representative occupant (CAS against the
        // current value, matching cell_leave in velocity_spread.glsl).
        atomicCompSwap(cells[cidx].occupant, cells[cidx].occupant, -1);
        // Drain the pooled velocity so a neighbor doesn't harvest a velocity
        // spike from an empty cell next frame.
        atomicExchange(cells[cidx].vel_x_bits, 0u);
        atomicExchange(cells[cidx].vel_y_bits, 0u);
        atomicExchange(cells[cidx].vel_z_bits, 0u);
    }
}

// Shared atomic counter — how many particles we've claimed/released this dispatch
shared int s_claimed;

bool is_inactive(int idx) {
    // NaN check: a float is NaN if it != itself
    float x = get_position(idx).x;
    return !(x == x);
}

void set_inactive(int idx) {
    // 0x7FC00000 = quiet NaN
    set_position(idx, vec3(uintBitsToFloat(0x7FC00000u), 0.0, 0.0));
}

void main() {
    if (gl_LocalInvocationIndex == 0u) {
        s_claimed = 0;
    }
    barrier();

    uint gid = gl_GlobalInvocationID.x;
    if (int(gid) >= pc.num_particles) return;

    float r2 = pc.radius * pc.radius;

    if (pc.mode == 0) {
        // ── SOURCE: activate inactive particles near world_pos ──────────────
        if (!is_inactive(int(gid))) return;

        int slot = atomicAdd(s_claimed, 1);
        if (slot >= pc.max_count) return;

        // Place at a random-ish offset within radius using gid as seed
        uint seed = gid * 2654435761u + uint(pc.grid_w);
        float ang1 = float(seed % 1000u) / 1000.0 * 6.2831853;
        float ang2 = float((seed / 1000u) % 1000u) / 1000.0 * 3.14159265;
        float r = pc.radius * float((seed / 1000000u) % 1000u) / 1000.0;

        vec3 new_pos = pc.world_pos + vec3(
            r * sin(ang2) * cos(ang1),
            r * sin(ang2) * sin(ang1),
            r * cos(ang2)
        );

        // Clamp to grid bounds
        new_pos.x = clamp(new_pos.x, 2.0, float(pc.grid_w) - 2.0);
        new_pos.y = clamp(new_pos.y, 2.0, float(pc.grid_h) - 2.0);
        new_pos.z = clamp(new_pos.z, 2.0, float(pc.grid_d) - 2.0);
        set_position(int(gid), new_pos);

        // Pack color
        uint cr = uint(clamp(pc.color.r * 255.0, 0.0, 255.0));
        uint cg = uint(clamp(pc.color.g * 255.0, 0.0, 255.0));
        uint cb = uint(clamp(pc.color.b * 255.0, 0.0, 255.0));
        uint ca = uint(clamp(pc.color.a * 255.0, 0.0, 255.0));
        set_color(int(gid), cr,cg,cb,ca);
        set_custom0(int(gid), vec4(pc.attraction, pc.opacity_fade, 0.0, 0.0));

        // Claim the grid cell so the physics step sees this particle as occupied.
        // If the cell is already taken, release the particle back to the inactive
        // pool rather than overlapping another occupant.
        ivec3 cell_pos = ivec3(round(new_pos));
        if (!cell_try_place(cell_index(cell_pos), int(gid))) {
            set_inactive(int(gid));
        }

    } else {
        // ── SINK: deactivate active particles near world_pos ─────────────────
        if (is_inactive(int(gid))) return;

        vec3 d = get_position(int(gid)) - pc.world_pos;
        if (dot(d, d) > r2) return;

        int slot = atomicAdd(s_claimed, 1);
        if (slot >= pc.max_count) return;

        // Free the grid cell so other particles can move into the vacated
        // location: decrement count, clear the representative occupant if the
        // cell empties, and drain the pooled velocity field.
        ivec3 cell_pos = ivec3(round(get_position(int(gid))));
        int   cidx     = cell_index(cell_pos);
        cell_leave_and_drain(cidx);

        set_inactive(int(gid));
    }
}
