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
    int   lod_levels;    // number of LOD levels in the page table
    int   _pad[3];       // pad to 96 bytes (16-byte aligned)
} pc;

// MUST match ArenaCell in velocity_spread.glsl (std430, 32 bytes).
struct ArenaCell {
    int  occupant;
    uint vel_x_bits;
    uint vel_y_bits;
    uint vel_z_bits;
    uint sdf_bits;
    int  count;
    uint pad0;
    uint pad1;
};

layout(set = 0, binding = 0, std430) buffer VertexBuffer   { float vtx[]; };
layout(set = 0, binding = 1, std430) buffer AttribBuffer   { uint  atr[]; };
// Legacy dense chunk grid — declared for uniform-set layout compat but not read.
layout(set = 0, binding = 2, std430) buffer LegacyChunkBuffer { float legacy_chunk[]; };

// Paged arena + LOD page table (same bindings as velocity_spread.glsl).
layout(set = 0, binding = 6, std430) buffer ArenaBuffer    { ArenaCell acells[]; };
layout(set = 0, binding = 7, std430) buffer LodTableBuffer { uint ltab[]; };

// Page-table field helpers (mirror velocity_spread.glsl).
uint lt_cell_base(int tidx) { return ltab[(tidx << 2) + 0]; }
uint lt_flags(int tidx)     { return ltab[(tidx << 2) + 1]; }

// Page-table dims for LOD k (mirror velocity_spread.glsl table_block_stride).
int table_block_stride(int k) {
    const int pv = 16 << k;
    return ((pc.grid_w + pv - 1) / pv) *
           ((pc.grid_h + pv - 1) / pv) *
           ((pc.grid_d + pv - 1) / pv);
}

// Page-table entry index for a voxel position at LOD k, or -1 if OOB.
int table_index(int k, ivec3 vp) {
    const int pv = 16 << k;
    const int dw = (pc.grid_w + pv - 1) / pv;
    const int dh = (pc.grid_h + pv - 1) / pv;
    const int dd = (pc.grid_d + pv - 1) / pv;
    const ivec3 pp = vp >> (4 + k);
    if (pp.x < 0 || pp.y < 0 || pp.z < 0 || pp.x >= dw || pp.y >= dh || pp.z >= dd) {
        return -1;
    }
    int base = 0;
    for (int q = 0; q < k; ++q) {
        base += table_block_stride(q);
    }
    return base + pp.x + dw * (pp.y + dh * pp.z);
}

// Resolve the arena cell index for a voxel position: finest allocated page wins.
int resolve_cell(ivec3 vp) {
    for (int k = 0; k < pc.lod_levels; ++k) {
        const int tidx = table_index(k, vp);
        if (tidx < 0) continue;
        if ((lt_flags(tidx) & 1u) != 0u) {
            const ivec3 lc = (vp >> k) & 15;
            return int(lt_cell_base(tidx)) + lc.x + lc.y * 16 + lc.z * 256;
        }
    }
    return -1;
}

vec3 get_position(int idx) {
    int base = idx * pc.vertex_stride_floats;
    return vec3(vtx[base + 0], vtx[base + 1], vtx[base + 2]);
}
void set_position(int idx, vec3 p) {
    int base = idx * pc.vertex_stride_floats;
    vtx[base + 0] = p.x; vtx[base + 1] = p.y; vtx[base + 2] = p.z;
}
void set_color(int idx, uint r, uint g, uint b, uint a) {
    atr[idx * pc.attrib_stride_words + pc.color_offset_words] =
        r | (g << 8) | (b << 16) | (a << 24);
}
void set_custom0(int idx, vec4 v) {
    int base = idx * pc.attrib_stride_words + pc.custom0_offset_words;
    atr[base+0] = floatBitsToUint(v.x); atr[base+1] = floatBitsToUint(v.y);
    atr[base+2] = floatBitsToUint(v.z); atr[base+3] = floatBitsToUint(v.w);
}

// Arena cell helpers (mirror velocity_spread.glsl ac_try_join / ac_leave).
int ac_try_join(int cidx, int pidx) {
    return atomicCompSwap(acells[cidx].occupant, -1, pidx);
}
void ac_leave(int cidx, int pidx) {
    atomicCompSwap(acells[cidx].occupant, pidx, -1);
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

        // Claim the arena cell so the physics step sees this particle as
        // occupied. If the cell is already taken or has no allocated page,
        // release the particle back to the inactive pool.
        ivec3 cell_pos = ivec3(round(new_pos));
        int cidx = resolve_cell(cell_pos);
        if (cidx < 0 || ac_try_join(cidx, int(gid)) != -1) {
            set_inactive(int(gid));
        }

    } else {
        // ── SINK: deactivate active particles near world_pos ─────────────────
        if (is_inactive(int(gid))) return;

        vec3 d = get_position(int(gid)) - pc.world_pos;
        if (dot(d, d) > r2) return;

        int slot = atomicAdd(s_claimed, 1);
        if (slot >= pc.max_count) return;

        // Free the arena cell so other particles can move into the vacated
        // location. resolve_cell finds the page; ac_leave clears our occupant
        // slot only if it still points at us.
        ivec3 cell_pos = ivec3(round(get_position(int(gid))));
        int   cidx     = resolve_cell(cell_pos);
        if (cidx >= 0) {
            ac_leave(cidx, int(gid));
        }

        set_inactive(int(gid));
    }
}
