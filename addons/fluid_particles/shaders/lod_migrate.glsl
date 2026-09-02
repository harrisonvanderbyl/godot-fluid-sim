#[compute]
#version 450

// ─── LOD migrate shader (Phase 3) ────────────────────────────────────────────
// One dispatch per destination page during a LOD transition.
//
//   mode 0 = COARSEN: destination is the coarser page (lod k); sources are the
//            8 child pages (lod k-1) that tile the same voxel volume.
//            Each destination cell aggregates 8 source cells (all 8 lie in ONE
//            child page — child pages tile the parent 2:1 per axis):
//              vel      = Σ src vel   (pooled momentum is extensive — conserved)
//              count    = Σ src count
//              occupant = first non -1 occupant (phase representative)
//              sdf      = min(src sdf) (most conservative "inside terrain")
//
//   mode 1 = REFINE: destination is one child page (lod k-1); source is the
//            single parent page (lod k). The CPU dispatches 8 refine jobs
//            (one per child page), distinguished by dst_octant:
//              vel      = parent vel / 8  (momentum split evenly; stepped
//                                         integration in P4 re-converges)
//              count    = parent count / 8
//              occupant = parent occupant ONLY in child octant 0; other
//                         children get -1 so cell_join re-populates them.
//              sdf      = copied as-is.
//
// Page geometry (all lods): 16^3 = 4096 cells per page; an LOD-k cell spans
// 2^k voxels. A parent page spans 16·2^k voxels/axis; each child page spans
// half that per axis, so 8 child pages tile one parent page 2:1 per axis.
//
//   COARSEN cell mapping: dst local d ∈ [0,16)³. Its 8 finer sub-cells all
//   lie in ONE child page, the one whose per-axis half index is (d >> 3).
//   Inside that child page the first sub-cell is ((d & 7) * 2); the 8
//   sub-cells add {0,1}³ to it. CPU must order src_base[] by child-page
//   octant o = ((dx>>3)&1) | (((dy>>3)&1)<<1) | (((dz>>3)&1)<<2).
//
//   REFINE cell mapping is the exact inverse: child cell s maps to the parent
//   cell (o_x·8 + (s.x>>1), o_y·8 + (s.y>>1), o_z·8 + (s.z>>1)), o = dst_octant.
//
// Dispatch: local_size 64, 64 groups → 4096 threads = one page.
// ─────────────────────────────────────────────────────────────────────────────

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// 128-byte push constant, one dispatch per destination page.
layout(push_constant, std430) uniform MigrateConstants {
    uint dst_base;      // arena cell offset of the destination page
    uint src_base[8];   // coarsen: 8 child pages in octant order; refine: [0]=parent
    uint mode;          // 0 = coarsen, 1 = refine
    uint dst_octant;    // refine: which child page this dispatch writes (0..7)
    uint _pad[21];      // pad to 128 bytes (4 + 32 + 4 + 4 + 84 = 128)
} mc;

// Same ChunkCell as clear_grid/velocity_spread — MUST stay in sync.
struct ChunkCell {
    int  occupant;
    uint vel_x_bits;
    uint vel_y_bits;
    uint vel_z_bits;
    uint sdf_bits;
    int  count;
    uint _pad[2];
};

// The migration shader touches ONLY the arena — never the dense chunk_buf or
// the particle vertex/attrib buffers.
layout(set = 0, binding = 0, std430) buffer ArenaBuffer { ChunkCell cells[]; };

vec3 cell_vel(uint idx) {
    return vec3(uintBitsToFloat(cells[idx].vel_x_bits),
                uintBitsToFloat(cells[idx].vel_y_bits),
                uintBitsToFloat(cells[idx].vel_z_bits));
}

void main() {
    const uint tid = gl_GlobalInvocationID.x;
    if (tid >= 4096u) {
        return;
    }

    const uint didx = mc.dst_base + tid;

    if (mc.mode == 0u) {
        // ── COARSEN ──
        const uint dx = tid & 15u;
        const uint dy = (tid >> 4) & 15u;
        const uint dz = tid >> 8;

        // which child page holds our 8 sub-cells
        const uint page_oct = ((dx >> 3) & 1u)
                            | (((dy >> 3) & 1u) << 1)
                            | (((dz >> 3) & 1u) << 2);
        const uint page_base = mc.src_base[page_oct];

        // first (lowest) of the 8 sub-cells inside that child page
        const uint sx = (dx & 7u) * 2u;
        const uint sy = (dy & 7u) * 2u;
        const uint sz = (dz & 7u) * 2u;

        vec3 vel = vec3(0.0);
        int  count = 0;
        int  occupant = -1;
        float sdf = 0.0;
        bool first = true;

        for (uint i = 0u; i < 8u; ++i) {
            const uint lx = sx + (i & 1u);
            const uint ly = sy + ((i >> 1) & 1u);
            const uint lz = sz + ((i >> 2) & 1u);
            const uint sidx = page_base + lx + ly * 16u + lz * 256u;

            vel += cell_vel(sidx);
            count += cells[sidx].count;
            const float sv = uintBitsToFloat(cells[sidx].sdf_bits);
            if (first || sv < sdf) {
                sdf = sv;
                first = false;
            }
            if (occupant < 0 && cells[sidx].occupant >= 0) {
                occupant = cells[sidx].occupant;
            }
        }

        ChunkCell o;
        o.occupant   = occupant;
        o.vel_x_bits = floatBitsToUint(vel.x);
        o.vel_y_bits = floatBitsToUint(vel.y);
        o.vel_z_bits = floatBitsToUint(vel.z);
        o.sdf_bits   = floatBitsToUint(sdf);
        o.count      = count;
        o._pad[0]    = 0u;
        o._pad[1]    = 0u;
        cells[didx]  = o;
    } else {
        // ── REFINE ──
        // Each parent cell is split into 8 children. We must distribute the
        // parent's count and velocity correctly:
        //   - count: distribute round-robin. count=8 → each child gets 1.
        //           count=5 → children 0..4 get 1, children 5..7 get 0.
        //           count=3 → children 0..2 get 1, children 3..7 get 0.
        //   - velocity: split by the ACTUAL count, not always /8. If count=3,
        //     each of the 3 occupied children gets parent_vel/3; the other 5
        //     get zero (no phantom velocity in empty cells).
        //   - occupant: the single occupant index goes to the first child that
        //     receives a count (child 0 if count > 0). The other occupied
        //     children get -1 — the physics step's ac_try_join repopulates
        //     them when those particles run their next full step.
        const uint sx = tid & 15u;
        const uint sy = (tid >> 4) & 15u;
        const uint sz = tid >> 8;

        // parent cell (exact inverse of the coarsen mapping)
        const uint px = ((mc.dst_octant & 1u) << 3) + (sx >> 1);
        const uint py = (((mc.dst_octant >> 1) & 1u) << 3) + (sy >> 1);
        const uint pz = (((mc.dst_octant >> 2) & 1u) << 3) + (sz >> 1);
        const uint pidx = mc.src_base[0] + px + py * 16u + pz * 256u;

        const vec3 parent_vel = cell_vel(pidx);
        const int parent_count = cells[pidx].count;

        // Which of the 8 sub-cells is this child? (0..7)
        const uint sub_oct = (sx & 1u) | ((sy & 1u) << 1) | ((sz & 1u) << 2);

        // Round-robin: this child gets a count if sub_oct < parent_count.
        // count=8 → all get 1. count=5 → 0..4 get 1. count=0 → none.
        const int my_count = (int(sub_oct) < parent_count) ? 1 : 0;

        // Velocity: split by parent_count (not 8). Empty children get zero.
        // If parent_count is 0, there's no velocity to split anyway.
        const float inv_count = (parent_count > 0) ? 1.0 / float(parent_count) : 0.0;
        const vec3 my_vel = (my_count > 0) ? parent_vel * inv_count : vec3(0.0);

        // Occupant: first occupied child (sub_oct 0) gets the parent's
        // occupant. Others get -1; physics repopulates via ac_try_join.
        const int my_occupant = (my_count > 0 && sub_oct == 0u)
                                ? cells[pidx].occupant : -1;

        ChunkCell o;
        o.occupant   = my_occupant;
        o.vel_x_bits = floatBitsToUint(my_vel.x);
        o.vel_y_bits = floatBitsToUint(my_vel.y);
        o.vel_z_bits = floatBitsToUint(my_vel.z);
        o.sdf_bits   = cells[pidx].sdf_bits;
        o.count      = my_count;
        o._pad[0]    = 0u;
        o._pad[1]    = 0u;
        cells[didx]  = o;
    }
}
