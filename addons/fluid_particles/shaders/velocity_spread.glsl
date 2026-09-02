#[compute]
#version 450

// ─── Physics / velocity spread shader (LOD-aware, Phase 4) ───────────
// Per particle:
//   1. Resolve its sim LOD by walking the LOD page table finest→coarsest for
//      the cell containing its position. First allocated page wins.
//   2. No allocated page anywhere → store-ring path: append the particle
//      index to store_ring_buf (once — bit 30 of custom0.w guards re-appends),
//      then freeze (return before any physics). Render still shows it.
//   3. Strided time stepping: LOD-k particles simulate every 2^k frames with
//      a per-particle phase hash; gravity scales ×stride, smoothing /stride.
//   4. All cell reads/writes hit the ARENA pages (binding 6). Neighbors
//      resolve through the page table, so steps can cross page borders.
//   5. SDF collision samples the dense sdf_storage_buf (binding 5) at the
//      particle's voxel position — identical to the pre-LOD build.
// custom0.w bit layout: bit31 = lod cache valid, bits 0..6 = cached lod+1,
// bit30 = index already pushed to the store ring this gap.
// ─────────────────────────────────────────────────────────────────────

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant, std430) uniform PushConstants {
    int   grid_w, grid_h, grid_d;
    int   num_particles;
    float surface_tension;
    float water_viscosity;
    float attraction_force;
    int   neighbor_mode;
    vec3  gravity;
    int   frame_count;
    int   num_runnable;
    int   vertex_stride_floats;
    int   attrib_stride_words;
    int   color_offset_words;
    int   custom0_offset_words;
    float has_delta;
    int   packed_lod_occ;    // low16 = max_occupancy, high16 = lod_levels
    float back_pressure;     // outward bias strength as a cell fills
    vec4  delta_row[3];
} pc;


int   max_occupancy;
int   lod_levels;
bool  debug_tint;   // decoded from packed_lod_occ bit 31 (inspector toggle)

layout(set = 0, binding = 0, std430) buffer VertexBuffer   { float vtx[]; };
layout(set = 0, binding = 1, std430) buffer AttribBuffer   { uint  atr[]; };
// Legacy dense chunk grid — no longer read by the LOD physics, but it MUST be
// declared: the CPU uniform set binds all 9 buffers 0-8 and a descriptor whose
// layout is missing a binding makes the driver read a garbage/out-of-range
// slot (crash inside libnvidia-glcore).
layout(set = 0, binding = 2, std430) buffer LegacyChunkBuffer { float legacy_chunk[]; };
layout(set = 0, binding = 3, std430) buffer RunnableBuffer { int   runnable_indices[]; };
layout(set = 0, binding = 4, std430) buffer SortKeyBuffer  { float sort_keys[]; };

// The paged arena: every cell read/write in the physics pass goes here.
// One ChunkCell struct per cell (32 B, must match lod_migrate.glsl / CPU).
struct ArenaCell {
    int  occupant;      // representative particle idx, or -1 if empty
    uint vel_x_bits;
    uint vel_y_bits;
    uint vel_z_bits;
    uint sdf_bits;      // SDF value (float bits). <0 = inside terrain
    int  count;         // real occupancy (representative is only a phase hint)
    uint pad0;
    uint pad1;
};
layout(set = 0, binding = 6, std430) buffer ArenaBuffer { ArenaCell acells[]; };

// Page table: one 16-byte entry per (lod, page coord), lod-major blocks.
// Flat uint[4] per entry at tidx*4 + {0=cell_base, 1=flags, 2=hint, 3=pad}.
layout(set = 0, binding = 7, std430) buffer LodTableBuffer { uint ltab[]; };

// Store ring: ring[0] = write_cursor (atomicAdd), ring[1] = overflow counter.
layout(set = 0, binding = 8, std430) buffer StoreRingBuffer { uint ring[]; };

// P5 stats scratch: st[0..7] = particles simulated at LOD k (k<8).
// Zeroed by the CPU each frame before this dispatch.
layout(set = 0, binding = 9, std430) buffer LodStatsBuffer { int lst[]; };

// Render vertex buffer: interpolated display positions. The physics shader
// lerps this toward the real position every frame so LOD-k particles render
// smoothly between their 2^k-spaced full physics steps.
layout(set = 0, binding = 10, std430) buffer RenderVertexBuffer { float rpos[]; };

// Page-table field helpers (hint is int on the CPU; reinterpret here).
uint lt_cell_base(int tidx) { return ltab[(tidx << 2) + 0]; }
uint lt_flags(int tidx)     { return ltab[(tidx << 2) + 1]; }
int  lt_hint(int tidx)      { return int(ltab[(tidx << 2) + 2]); }

vec3 get_position(int idx) {
    int base = idx * pc.vertex_stride_floats;
    return vec3(vtx[base + 0], vtx[base + 1], vtx[base + 2]);
}
void set_position(int idx, vec3 p) {
    int base = idx * pc.vertex_stride_floats;
    vtx[base + 0] = p.x; vtx[base + 1] = p.y; vtx[base + 2] = p.z;
}
vec3 get_render_position(int idx) {
    int base = idx * pc.vertex_stride_floats;
    return vec3(rpos[base + 0], rpos[base + 1], rpos[base + 2]);
}
void set_render_position(int idx, vec3 p) {
    int base = idx * pc.vertex_stride_floats;
    rpos[base + 0] = p.x; rpos[base + 1] = p.y; rpos[base + 2] = p.z;
}
uint get_color(int idx) {
    return atr[idx * pc.attrib_stride_words + pc.color_offset_words];
}
vec4 get_custom0(int idx) {
    int base = idx * pc.attrib_stride_words + pc.custom0_offset_words;
    return vec4(uintBitsToFloat(atr[base+0]), uintBitsToFloat(atr[base+1]), uintBitsToFloat(atr[base+2]), uintBitsToFloat(atr[base+3]));
}
void set_custom0(int idx, vec4 v) {
    int base = idx * pc.attrib_stride_words + pc.custom0_offset_words;
    atr[base+0] = floatBitsToUint(v.x); atr[base+1] = floatBitsToUint(v.y);
    atr[base+2] = floatBitsToUint(v.z); atr[base+3] = floatBitsToUint(v.w);
}
uint raw_custom0_w(int idx) {
    return atr[idx * pc.attrib_stride_words + pc.custom0_offset_words + 3];
}
void set_raw_custom0_w(int idx, uint w) {
    atr[idx * pc.attrib_stride_words + pc.custom0_offset_words + 3] = w;
}

// P5: tint helper. LOD 0 = blue-ish (untinted), 1 = green, 2 = orange,
// 3 = red, 4+ = magenta. Color is a packed rgba8 uint in the attribute.
uint lod_tint(uint packed_color, int sim_lod) {
    vec3 tint;
    if      (sim_lod == 0) tint = vec3(0.45, 0.70, 1.00);
    else if (sim_lod == 1) tint = vec3(0.30, 1.00, 0.35);
    else if (sim_lod == 2) tint = vec3(1.00, 0.75, 0.25);
    else if (sim_lod == 3) tint = vec3(1.00, 0.30, 0.25);
    else                   tint = vec3(1.00, 0.35, 1.00);
    const vec3 rgb = vec3(float(packed_color & 255u),
                          float((packed_color >> 8) & 255u),
                          float((packed_color >> 16) & 255u)) / 255.0;
    const vec3 outc = mix(rgb, tint, 0.75);
    const uint a = (packed_color >> 24) & 255u;
    return uint(clamp(outc.r, 0.0, 1.0) * 255.0)
         | (uint(clamp(outc.g, 0.0, 1.0) * 255.0) << 8)
         | (uint(clamp(outc.b, 0.0, 1.0) * 255.0) << 16)
         | (a << 24);
}

bool in_bounds(ivec3 p) {
    return all(greaterThanEqual(p, ivec3(0))) && all(lessThan(p, ivec3(pc.grid_w, pc.grid_h, pc.grid_d)));
}

// ── LOD page resolution ──────────────────────────────────────────────
int table_block_stride(int k) {
    const int pv = 16 << k;
    return ((pc.grid_w + pv - 1) / pv) *
           ((pc.grid_h + pv - 1) / pv) *
           ((pc.grid_d + pv - 1) / pv);
}

// Page-table entry index for a voxel position at LOD k, or -1 if out of bounds.
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

// Resolve the absolute arena cell index for a voxel position: walk the page
// table finest→coarsest; the first LOD whose covering page is allocated wins.
// Returns -1 when no allocated page covers the position.
int resolve_cell(ivec3 vp) {
    for (int k = 0; k < lod_levels; ++k) {
        const int tidx = table_index(k, vp);
        if (tidx < 0) {
            continue;
        }
        if ((lt_flags(tidx) & 1u) != 0u) {
            const ivec3 lc = (vp >> k) & 15;
            return int(lt_cell_base(tidx)) + lc.x + lc.y * 16 + lc.z * 256;
        }
    }
    return -1;
}

// SDF sample at a voxel position — reads from the arena cell's sdf_bits
// field (baked at page allocation time from the terrain at the page's LOD).
// OOB or unallocated page → far outside (no collision).
// Sentinel: a page can be allocated but not yet SDF-baked (lazy per-poll
// bake); its cells hold exactly this value until baked. MUST match
// SDF_UNSET_VALUE in fluid_particle_system.cpp.
const float SDF_UNSET = -100.0;

float sample_voxel_sdf(ivec3 pos) {
    int cidx = resolve_cell(pos);
    if (cidx < 0) return 100.0;
    return uintBitsToFloat(acells[cidx].sdf_bits);
}

// SDF gradient via central differences (points away from the surface).
// step should match the caller's own cell size (1<<sim_lod) so all 6 taps
// stay inside the caller's LOD page rather than crossing into a
// differently-scaled neighbor page near an LOD boundary.
vec3 sdf_normal_at(ivec3 pos, int step) {
    float xp = sample_voxel_sdf(pos + ivec3(step,0,0));
    float xm = sample_voxel_sdf(pos - ivec3(step,0,0));
    float yp = sample_voxel_sdf(pos + ivec3(0,step,0));
    float ym = sample_voxel_sdf(pos - ivec3(0,step,0));
    float zp = sample_voxel_sdf(pos + ivec3(0,0,step));
    float zm = sample_voxel_sdf(pos - ivec3(0,0,step));
    vec3 g = vec3(xp - xm, yp - ym, zp - zm) * 0.5;
    float l = length(g);
    return (l > 0.0001) ? g / l : vec3(0.0, 1.0, 0.0);
}

// Per-component atomic add on the arena's split velocity words.
void ac_atomic_add(int cidx, vec3 v) {
    uint prev, next;
    do { prev = acells[cidx].vel_x_bits; next = floatBitsToUint(uintBitsToFloat(prev) + v.x); } while (atomicCompSwap(acells[cidx].vel_x_bits, prev, next) != prev);
    do { prev = acells[cidx].vel_y_bits; next = floatBitsToUint(uintBitsToFloat(prev) + v.y); } while (atomicCompSwap(acells[cidx].vel_y_bits, prev, next) != prev);
    do { prev = acells[cidx].vel_z_bits; next = floatBitsToUint(uintBitsToFloat(prev) + v.z); } while (atomicCompSwap(acells[cidx].vel_z_bits, prev, next) != prev);
}

// Harvest pooled momentum WITH zeroing (atomicExchange = momentum exit).
vec3 ac_swap0(int cidx) {
    return vec3(
        uintBitsToFloat(atomicExchange(acells[cidx].vel_x_bits, 0u)),
        uintBitsToFloat(atomicExchange(acells[cidx].vel_y_bits, 0u)),
        uintBitsToFloat(atomicExchange(acells[cidx].vel_z_bits, 0u)));
}

// Try to become the representative of a cell. Returns the previous
// representative: -1 = admitted.
int ac_try_join(int cidx, int pidx) {
    return atomicCompSwap(acells[cidx].occupant, -1, pidx);
}

// Leave a cell: clear the representative only if it still points at us.
void ac_leave(int cidx, int pidx) {
    atomicCompSwap(acells[cidx].occupant, pidx, -1);
}

const ivec3 NEIGHBOR_OFFSETS[15] = ivec3[15](
    ivec3(-1,0,0), ivec3(1,0,0), ivec3(0,-1,0), ivec3(0,1,0),
    ivec3(0,0,-1), ivec3(0,0,1), ivec3(0,0,0),
    ivec3(1,-1,1), ivec3(-1,-1,1), ivec3(1,-1,-1), ivec3(-1,-1,-1),
    ivec3(1,1,1),  ivec3(1,1,-1), ivec3(-1,1,-1), ivec3(-1,1,1)
);

void main() {
    // Decode packed scalars once.
    max_occupancy = pc.packed_lod_occ & 0xFFFF;
    lod_levels    = (pc.packed_lod_occ >> 16) & 0x7FFF;
    debug_tint    = (pc.packed_lod_occ & 0x80000000) != 0;  // P5 debug tint bit
    if (lod_levels <= 0) lod_levels = 1;

    uint gid = gl_GlobalInvocationID.x;
    if (int(gid) >= pc.num_runnable) return;
    int particle_idx = runnable_indices[gid];
    if (particle_idx < 0 || particle_idx >= pc.num_particles) return;

    vec3 position = get_position(particle_idx);
    if (!(position.x == position.x)) return;   // NaN sentinel = inactive

    vec4 c0            = get_custom0(particle_idx);
    float attraction_force_val = c0.x;
    float opacity_fade         = c0.y;
    float neighbors_filled     = c0.z;
    uint  raw_w = floatBitsToUint(c0.w);

    if (opacity_fade < 1.0) opacity_fade += 1.0 / 60.0;

    // ── LOD resolution: finest allocated page wins ────────────────────
    const ivec3 vp0 = ivec3(round(position));
    int sim_lod = -1;
    for (int k = 0; k < lod_levels; ++k) {
        const int tidx = table_index(k, vp0);
        if (tidx >= 0 && (lt_flags(tidx) & 1u) != 0u) {
            sim_lod = k;
            break;
        }
    }

    // A page can be allocated but not yet SDF-baked (lazy per-poll bake).
    // Its cells still hold the SDF_UNSET sentinel — treat exactly like "no
    // page" so particles don't fall through unbaked terrain.
    int vp0_cidx = -1;
    if (sim_lod >= 0) {
        vp0_cidx = resolve_cell(vp0);
        if (vp0_cidx < 0 || uintBitsToFloat(acells[vp0_cidx].sdf_bits) == SDF_UNSET) {
            sim_lod = -1;
        }
    }

    if (sim_lod < 0) {
        // ── Store-ring path: no allocated page. Append once, then freeze. ──
        if ((raw_w & (1u << 30)) == 0u) {   // not yet stored this gap
            const uint slot = atomicAdd(ring[0], 1u);
            if (slot < 65534u) {
                ring[2u + slot] = uint(particle_idx);
                raw_w |= (1u << 30);
                set_raw_custom0_w(particle_idx, raw_w);
            } else {
                atomicAdd(ring[1], 1u);   // overflow counter
                raw_w |= (1u << 30);      // mark anyway so we don't spam
                set_raw_custom0_w(particle_idx, raw_w);
            }
        }
        if (debug_tint) {
            // Frozen: dark grey so skipped particles are obvious.
            const int cbase = particle_idx * pc.attrib_stride_words + pc.color_offset_words;
            atr[cbase] = (atr[cbase] & 0xFF000000u) | 0x00303030u;
        }
        // Snap render position to real so frozen particles don't drift.
        set_render_position(particle_idx, position);
        return;  // freeze: no physics, render still shows the particle
    }

    // P5: count this particle at its sim LOD (stats zeroed each frame by CPU).
    //if(sim_lod != 0) return;
    if (sim_lod < 8) atomicAdd(lst[sim_lod], 1);

    // ── Strided time stepping: LOD-k runs every 2^k frames ────────────
    const int stride = 1 << sim_lod;
    const int phase  = int((uint(particle_idx) * 0x9E3779B1u) & uint(stride - 1));
    const int frame_in_cycle = (pc.frame_count + phase) & (stride - 1);
    if (frame_in_cycle != 0) {
        // Skip frame: lerp render position toward real position.
        // lerp factor = 1 / (stride - frame_in_cycle)
        //   frame 1/4 → 1/4, frame 2/4 → 1/3, frame 3/4 → 1/2, frame 4/4 → 1/1
        // This gives smooth interpolation: 25%, 50%, 75%, 100% across the cycle.
        vec3 rpos_cur = get_render_position(particle_idx);
        float t = 1.0 / float(stride - frame_in_cycle);
        set_render_position(particle_idx, mix(rpos_cur, position, t));
        return;
    }
    const float stride_f   = float(stride);
    const float inv_stride = 1.0 / stride_f;

    // Container rigid-body impulse (inertia vs container motion).
    vec3 container_force = vec3(0.0);
    if (pc.has_delta > 0.5) {
        vec3 dp;
        dp.x = dot(pc.delta_row[0].xyz, position) + pc.delta_row[0].w - position.x;
        dp.y = dot(pc.delta_row[1].xyz, position) + pc.delta_row[1].w - position.y;
        dp.z = dot(pc.delta_row[2].xyz, position) + pc.delta_row[2].w - position.z;
        container_force = dp;
    }
    if (length(container_force) > 0.0) opacity_fade = 1.0;

    bool  firststep    = false;
    vec3 opos = position;
    ivec3 old_cell_pos = vp0;
    int   old_cidx     = vp0_cidx;

    // Ensure we're counted in our current cell.
    if (old_cidx >= 0 && acells[old_cidx].occupant == -1) {
        ac_try_join(old_cidx, particle_idx);
        firststep = true;
    }

    int   n_size    = pc.neighbor_mode;
    float friction  = 1.0 / float(n_size);
    vec3  momentum  = vec3(0.0);
    vec3  cohesion  = vec3(0.0);   // surface-tension direction (toward same-phase mass)
    vec3  crowd     = vec3(0.0);   // back-pressure direction (away from full cells)
    int   allfilled = 0;

    for (int i = 0; i < n_size; i++) {
        ivec3 npos = old_cell_pos + NEIGHBOR_OFFSETS[i] * (1<<sim_lod);
        if (!in_bounds(npos)) continue;
        int ncidx = resolve_cell(npos);
        if (ncidx < 0) continue;   // neighbor outside any allocated page

        // Harvest pooled momentum (atomic exit conserves it across pages;
        // cell distances stay voxel-space, so coarser-LOD neighbors read
        // naturally scaled momentum for their cell volume).
        vec3 pooled = ac_swap0(ncidx);

        int occ = acells[ncidx].occupant;
        if (occ >= 0) {
            allfilled += 1;
            cohesion -= vec3(NEIGHBOR_OFFSETS[i]);
        }
        momentum += pooled;
    }

    float attract = attraction_force_val * pc.attraction_force;
    float mix_f   = 0.98;
    // Smoothing relaxes /stride so visual smoothing stays frame-rate independent.
    float relax = (1.0);
    neighbors_filled = float(allfilled * 15) / float(n_size) * relax + neighbors_filled * (1.0 - relax);

    // Rotate pooled momentum into the new container frame.
    if (pc.has_delta > 0.5) {
        vec3 mrot;
        mrot.x = dot(pc.delta_row[0].xyz, momentum);
        mrot.y = dot(pc.delta_row[1].xyz, momentum);
        mrot.z = dot(pc.delta_row[2].xyz, momentum);
        momentum = mrot;
    }

    momentum -= container_force;

    // ── Stride scaling for the position update ────────────────────────
    // A LOD-k particle runs every 2^k frames, so each step must cover 2^k
    // frames' worth of movement. Scale momentum by stride for the position
    // update, then scale it back before storing so cell velocities stay at
    // per-frame scale (a LOD-0 neighbor harvesting the same cell reads the
    // same velocity regardless of who wrote it).
    momentum += cohesion * pc.surface_tension;
    momentum += crowd    * pc.back_pressure;
    momentum = mix(momentum, momentum * (1.0 - pc.water_viscosity), 0.0);

    // Gravity: one frame's worth. The position update below scales the ENTIRE
    // momentum by stride, so gravity is already multiplied by stride there.
    // DO NOT pre-multiply gravity by stride — that would give stride².
    momentum -= pc.gravity;

    // Move: scale by stride so the particle covers stride frames of distance.
    position += momentum * stride_f;

    // Boundary reflection (mirror fold; rest<1 damps).
    const float sz   = 1.0;
    const float rest = 0.4;
    {float lo = 1.0+sz, hi = float(pc.grid_h)-1.0-sz;
      if (position.y < lo) { position.y = lo + (lo-position.y)*rest; momentum.y = -momentum.y*rest; }
      else if (position.y > hi) { position.y = hi - (position.y-hi)*rest; momentum.y = -momentum.y*rest; } }
    { float lo = 1.0+sz, hi = float(pc.grid_w)-1.0-sz;
      if (position.x < lo) { position.x = lo + (lo-position.x)*rest; momentum.x = -momentum.x*rest; }
      else if (position.x > hi) { position.x = hi - (position.x-hi)*rest; momentum.x = -momentum.x*rest; } }
    { float lo = 1.0+sz, hi = float(pc.grid_d)-1.0-sz;
      if (position.z < lo) { position.z = lo + (lo-position.z)*rest; momentum.z = -momentum.z*rest; }
      else if (position.z > hi) { position.z = hi - (position.z-hi)*rest; momentum.z = -momentum.z*rest; } }

    // ── SDF terrain collision ──────────────────────────────────────────────
    {
        ivec3 pcell = ivec3(round(position));
        pcell = clamp(pcell, ivec3(0), ivec3(pc.grid_w-1, pc.grid_h-1, pc.grid_d-1));
        float sdf = sample_voxel_sdf(pcell);
        if (sdf <= 0.0) {
            vec3 n = sdf_normal_at(pcell, 1 << sim_lod);
            
                position = old_cell_pos;
                float vn = dot(momentum, n);
                if (vn < 0.0) {
                    momentum -= n * vn * 1.5  ;
                }
            
        }
    }

    // Move to new cell if the rounded position changed.
    ivec3 new_cell_pos = ivec3(round(position));
    int   new_cidx     = resolve_cell(new_cell_pos);
    if (new_cidx >= 0 && old_cidx >= 0 && new_cell_pos != old_cell_pos) {
        int oin = ac_try_join(new_cidx, particle_idx);
        if (oin == -1) {
            ac_leave(old_cidx, particle_idx);
        } else {
            position     = opos;
            vec3 to_other = get_position(oin) - position;
            float len_other = length(to_other);
            float len_mom = length(momentum);
            float tootherpos = 0.0;
            if (len_other > 1e-6 && len_mom > 1e-6) {
                tootherpos = dot(to_other / len_other, momentum / len_mom);
            }
            ac_atomic_add(new_cidx, momentum*tootherpos);
            momentum     = momentum * (1.0-tootherpos);
            new_cell_pos = old_cell_pos;
        }
    }

    // Scale momentum back to per-frame scale before spreading to cells.
    // The position update used momentum × stride; cells store per-frame
    // velocity so a LOD-0 neighbor reads the same value regardless of source.
    //momentum *= inv_stride;

    if (firststep) attract = 0.0;
    attract /= 1<<sim_lod;
    vec3 spread = momentum * friction;
    vec3 collected = vec3(0.0);
    for (int i = 0; i < n_size; i++) {
        ivec3 npos = new_cell_pos + NEIGHBOR_OFFSETS[i] * (1 << sim_lod);
        int ncidx = in_bounds(npos) ? resolve_cell(npos) : -1;
        if (ncidx >= 0) {
            vec3 push = vec3(NEIGHBOR_OFFSETS[i]) * attract;
            ac_atomic_add(ncidx, spread + push);
        } else {
            collected += vec3(NEIGHBOR_OFFSETS[i]) * attract + spread;
        }
    }

    if (new_cidx >= 0) {
        ac_atomic_add(new_cidx, collected);
    }

    set_position(particle_idx, position);
    if(sim_lod == 0){
        set_render_position(particle_idx, position);
    }else{
        set_render_position(particle_idx, old_cell_pos);
    }
    set_custom0(particle_idx, vec4(attraction_force_val, opacity_fade, neighbors_filled, uintBitsToFloat(raw_w)));
}