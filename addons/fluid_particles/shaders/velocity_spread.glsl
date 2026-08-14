#[compute]
#version 450

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
    //int   max_occupancy;     // ← NEW: max liquid particles per cell (solids stay 1)
    //float back_pressure;     // ← NEW: outward bias strength as a cell fills
    //float _pad1;
    vec4  delta_row[3];
} pc;

const int max_occupancy = 1;
const float back_pressure = 0.0;


// occupant is now a REPRESENTATIVE (last successful joiner) used only for the
// phase (solid/liquid) check. count is the real occupancy. For exact per-phase
// counts you'd split count into count_solid/count_liquid here.
struct ChunkCell {
    int  occupant;      // representative particle idx, or -1 if empty         // ← NEW: current occupancy
    uint vel_x_bits;
    uint vel_y_bits;
    uint vel_z_bits;
    uint vel_w_bits;
    int count;
    uint _pad[2];
};

layout(set = 0, binding = 0, std430) buffer VertexBuffer   { float vtx[]; };
layout(set = 0, binding = 1, std430) buffer AttribBuffer   { uint  atr[]; };
layout(set = 0, binding = 2, std430) buffer ChunkBuffer    { ChunkCell cells[]; };
layout(set = 0, binding = 3, std430) buffer RunnableBuffer { int   runnable_indices[]; };
layout(set = 0, binding = 4, std430) buffer SortKeyBuffer  { float sort_keys[]; };

vec3 get_position(int idx) {
    int base = idx * pc.vertex_stride_floats;
    return vec3(vtx[base + 0], vtx[base + 1], vtx[base + 2]);
}
void set_position(int idx, vec3 p) {
    int base = idx * pc.vertex_stride_floats;
    vtx[base + 0] = p.x; vtx[base + 1] = p.y; vtx[base + 2] = p.z;
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

int cell_index(ivec3 p) {
    p = clamp(p, ivec3(0), ivec3(pc.grid_w-1, pc.grid_h-1, pc.grid_d-1));
    return p.x + p.y * pc.grid_w + p.z * pc.grid_w * pc.grid_h;
}
bool in_bounds(ivec3 p) {
    return all(greaterThanEqual(p, ivec3(0))) && all(lessThan(p, ivec3(pc.grid_w, pc.grid_h, pc.grid_d)));
}

// Read pooled momentum WITHOUT zeroing (we need to divide by occupancy first).
vec3 cell_read(int cidx) {
    return vec3(
        uintBitsToFloat(cells[cidx].vel_x_bits),
        uintBitsToFloat(cells[cidx].vel_y_bits),
        uintBitsToFloat(cells[cidx].vel_z_bits)
    );
}
vec3 cell_swap0(int cidx) {
    return vec3(
        uintBitsToFloat(atomicExchange(cells[cidx].vel_x_bits, 0u)),
        uintBitsToFloat(atomicExchange(cells[cidx].vel_y_bits, 0u)),
        uintBitsToFloat(atomicExchange(cells[cidx].vel_z_bits, 0u))
    );
}
void cell_atomic_add(int cidx, vec3 v) {
    uint prev, next;
    do { prev = cells[cidx].vel_x_bits; next = floatBitsToUint(uintBitsToFloat(prev) + v.x); } while (atomicCompSwap(cells[cidx].vel_x_bits, prev, next) != prev);
    do { prev = cells[cidx].vel_y_bits; next = floatBitsToUint(uintBitsToFloat(prev) + v.y); } while (atomicCompSwap(cells[cidx].vel_y_bits, prev, next) != prev);
    do { prev = cells[cidx].vel_z_bits; next = floatBitsToUint(uintBitsToFloat(prev) + v.z); } while (atomicCompSwap(cells[cidx].vel_z_bits, prev, next) != prev);
}

// Try to join a cell. Returns true if admitted, and reports how full it was.
// Atomic-add-then-back-out avoids the check-then-increment TOCTOU race:
// unconditionally claim a slot, and if we overshot the max, release it.
bool cell_try_join(int cidx, int pidx, int cap, out int prior_count) {
    prior_count = atomicAdd(cells[cidx].count, 1);
    if (prior_count >= cap) {
        atomicAdd(cells[cidx].count, -1);   // back out, cell was full
        return false;
    }
    // We're in. Become the representative for phase checks (last writer wins;
    // approximate but only matters at phase boundaries).
    atomicExchange(cells[cidx].occupant, pidx);
    return true;
}

// Leave a cell: decrement count. Clear representative only if the cell empties.
void cell_leave(int cidx) {
    int after = atomicAdd(cells[cidx].count, -1) - 1;
    if (after <= 0) atomicCompSwap(cells[cidx].occupant, cells[cidx].occupant, -1);
    // (representative left stale if count>0; harmless — it's only a phase hint)
}

const ivec3 NEIGHBOR_OFFSETS[15] = ivec3[15](
    ivec3(-1,0,0), ivec3(1,0,0), ivec3(0,-1,0), ivec3(0,1,0),
    ivec3(0,0,-1), ivec3(0,0,1), ivec3(0,0,0),
    ivec3(1,-1,1), ivec3(-1,-1,1), ivec3(1,-1,-1), ivec3(-1,-1,-1),
    ivec3(1,1,1),  ivec3(1,1,-1), ivec3(-1,1,-1), ivec3(-1,1,1)
);

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (int(gid) >= pc.num_runnable) return;
    int particle_idx = runnable_indices[gid];
    if (particle_idx < 0 || particle_idx >= pc.num_particles) return;

    vec3 position = get_position(particle_idx);
    if (!(position.x == position.x)) return;   // NaN sentinel = inactive

    uint color_packed = get_color(particle_idx);
    vec4 c0            = get_custom0(particle_idx);
    float attraction_force_val = c0.x;
    float opacity_fade         = c0.y;
    float neighbors_filled     = c0.z;

    if (opacity_fade < 1.0) opacity_fade += 1.0 / 60.0;

    bool me_solid = ((color_packed >> 24) & 0xffu) > 200u;
    int  my_cap   = me_solid ? 1 : max_occupancy;   // solids never share

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
    ivec3 old_cell_pos = ivec3(round(position));
    int   old_cidx     = cell_index(old_cell_pos);

    // Ensure we're counted in our current cell.
    if (cells[old_cidx].occupant == -1 || cells[old_cidx].count == 0) {
        int prior;
        cell_try_join(old_cidx, particle_idx, my_cap, prior);
        firststep = true;
    }

    int   n_size    = pc.neighbor_mode;
    float friction  = 1.0 / float(n_size);
    vec3  momentum  = vec3(0.0);
    vec3  cohesion  = vec3(0.0);   // surface-tension direction (toward same-phase mass)
    vec3  crowd     = vec3(0.0);   // back-pressure direction (away from full cells)
    int   allfilled = 0;

    for (int i = 0; i < n_size; i++) {
        ivec3 npos = old_cell_pos + NEIGHBOR_OFFSETS[i];
        if (!in_bounds(npos)) continue;
        int ncidx = cell_index(npos);

        // Harvest pooled momentum, divided by occupancy so N co-located
        // particles each take their 1/N share — conserves momentum on exit.
        int occ_n = max(cells[ncidx].count, 1);
        vec3 pooled = cell_swap0(ncidx);
        momentum += pooled / float(occ_n);

        int occ = cells[ncidx].occupant;
        if (occ >= 0 && occ < pc.num_particles) {
            bool part_solid = ((get_color(occ) >> 24) & 0xffu) > 200u;
            bool same_phase = (me_solid == part_solid);
            allfilled += same_phase ? 1 : 0;
            if (same_phase) cohesion += vec3(NEIGHBOR_OFFSETS[i]);

            // Back-pressure: neighbor crowding pushes us the OTHER way.
            // Fills toward cap contribute an outward shove.
            float fill = float(cells[ncidx].count) / float(max(my_cap, 1));
            crowd -= vec3(NEIGHBOR_OFFSETS[i]) * max(fill - 1.0 + 1.0/float(max(my_cap,1)), 0.0);
        }
    }

    float attract = attraction_force_val * pc.attraction_force;
    float mix_f   = 0.98;
    neighbors_filled = float(allfilled * 15) / float(n_size) * (1.0 - mix_f) + neighbors_filled * mix_f;

    // Rotate pooled momentum into the new container frame.
    if (pc.has_delta > 0.5) {
        vec3 mrot;
        mrot.x = dot(pc.delta_row[0].xyz, momentum);
        mrot.y = dot(pc.delta_row[1].xyz, momentum);
        mrot.z = dot(pc.delta_row[2].xyz, momentum);
        momentum = mrot;
    }

    momentum -= container_force;

    position += momentum;

    if (firststep) attract = 0.0;

    momentum -= pc.gravity; // assumed to be included in container force
    momentum += cohesion * pc.surface_tension;   // surface tension: pull toward the mass
    momentum += crowd    * back_pressure;     // pressure: push out of crowded cells

    // Isolation damping: lone particles bleed speed so the surface stays crisp.
    //float isolation = 1.0 - clamp(neighbors_filled / float(n_size), 0.0, 1.0);
    //momentum *= mix(1.0, 0.85, isolation);

    // Viscosity: relax toward the (already 1/N-shared) local pooled motion.
    // water_viscosity in [0,1]; 0 = inviscid, higher = stickier flow.
    // momentum already carries neighbor-averaged velocity, so damp deviations.
    momentum = mix(momentum, momentum * (1.0 - pc.water_viscosity), 0.0); // placeholder-safe no-op if unset

    // Boundary reflection (mirror fold; rest<1 damps).
    const float sz   = 1.0;
    const float rest = 0.3;
    {float lo = 1.0+sz, hi = float(pc.grid_h)-1.0-sz;
      if (position.y < lo) { position.y = lo + (lo-position.y)*rest; momentum.y = -momentum.y*rest; }
      else if (position.y > hi) { position.y = hi - (position.y-hi)*rest; momentum.y = -momentum.y*rest; } }
    { float lo = 1.0+sz, hi = float(pc.grid_w)-1.0-sz;
      if (position.x < lo) { position.x = lo + (lo-position.x)*rest; momentum.x = -momentum.x*rest; }
      else if (position.x > hi) { position.x = hi - (position.x-hi)*rest; momentum.x = -momentum.x*rest; } }
    { float lo = 1.0+sz, hi = float(pc.grid_d)-1.0-sz;
      if (position.z < lo) { position.z = lo + (lo-position.z)*rest; momentum.z = -momentum.z*rest; }
      else if (position.z > hi) { position.z = hi - (position.z-hi)*rest; momentum.z = -momentum.z*rest; } }

    // Move to new cell if the rounded position changed.
    ivec3 new_cell_pos = ivec3(round(position));
    int   new_cidx     = cell_index(new_cell_pos);
    if (new_cell_pos != old_cell_pos) {
        int prior;
        if (cell_try_join(new_cidx, particle_idx, my_cap, prior)) {
            cell_leave(old_cidx);            // only after we're admitted elsewhere
        } else {
            // Target cell full: stay put, dump our momentum into it as pressure.
            position     = vec3(old_cell_pos);
            cell_atomic_add(new_cidx, momentum);
            momentum     = vec3(0.0);
            new_cell_pos = old_cell_pos;
        }
    }

    // Spread momentum to neighbors (shared pool → combined-system motion).
    vec3 spread = momentum * friction;
    for (int i = 0; i < n_size; i++) {
        ivec3 npos = new_cell_pos + NEIGHBOR_OFFSETS[i];
        if (!in_bounds(npos)) continue;
        cell_atomic_add(cell_index(npos), spread + vec3(NEIGHBOR_OFFSETS[i]) * attract);
    }

    set_position(particle_idx, position);
    set_custom0(particle_idx, vec4(attraction_force_val, opacity_fade, neighbors_filled, c0.w));
}