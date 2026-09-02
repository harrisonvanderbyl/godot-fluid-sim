#pragma once
#include <godot_cpp/variant/vector3i.hpp>
#include <godot_cpp/templates/hash_map.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <cstdint>
#include <vector>

namespace godot {

// ─────────────────────────────────────────────────────────────────────────────
// LOD page constants — MUST match lod_arena.glsl / velocity_spread.glsl copies.
//
// Coordinate spaces (see docs/LOD_UPGRADE_PLAN.md §0):
//   voxel space : integer cell space, [0, grid_w/h/d)
//   page        : one allocated arena block. Page at LOD k covers
//                 (16 << k)³ voxel-space units and holds PAGE_CELLS cells
//                 (every page has the same number of cells; they just get
//                 coarser as k rises — 1 terrain data block ↔ 1 page).
//   page coord  : voxel_pos >> (4 + k)  i.e. voxel_to_data_block_position
// ─────────────────────────────────────────────────────────────────────────────
constexpr int  LOD_PAGE_SIZE       = 16;   // data block size (voxels) at LOD 0
constexpr int  LOD_PAGE_CELLS      = 4096; // 16³ cells per page
constexpr int  LOD_MAX_LEVELS      = 8;    // page-table depth cap
constexpr uint32_t LOD_PAGE_FLAG_ALLOCATED = 1u << 0;
constexpr uint32_t LOD_PAGE_GEN_SHIFT      = 8;   // bits 8..15 = generation

// GPU page-table entry — 16 bytes, std430, mirrored on CPU and written to
// lod_table_buf with a 16-byte buffer_update per lifecycle event.
struct LodPageEntry {
    uint32_t cell_base;      // arena cell offset of this page's first cell
    uint32_t flags;          // bit0 = allocated; bits 8..15 = generation
    int32_t  occupant_hint;  // -1 = page certainly empty (skip fast path); else unknown
    uint32_t reserved;
};
static_assert(sizeof(LodPageEntry) == 16, "LodPageEntry must be 16 bytes (std430)");

/// Key of a (lod, page coord) pair.
struct LodPageKey {
    int32_t lod = 0;
    int32_t x = 0, y = 0, z = 0;
    bool operator==(const LodPageKey &o) const {
        return lod == o.lod && x == o.x && y == o.y && z == o.z;
    }
};

struct LodPageKeyHash {
    static uint32_t hash(const LodPageKey &k) {
        uint32_t h = (uint32_t)k.lod * 0x9E3779B1u;
        h ^= (uint32_t)k.x * 0x85EBCA6Bu;
        h ^= (uint32_t)k.y * 0xC2B2AE35u;
        h ^= (uint32_t)k.z * 0x27D4EB2Fu;
        return h;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// VoxelLodArena — CPU-side page allocator for the cell arena.
//
// Owns the resident-page bookkeeping: which (lod, page coord) pairs map to
// which 4096-cell slot in the GPU arena, plus a freelist for reuse. Purely CPU
// state — GPU table entries are written by the owner via buffer_update after
// each alloc/free.
//
// LIFECYCLE CONTRACT (Phase 2): pages are only created or destroyed by terrain
// signal events routed through alloc()/free() — never by polling or per-frame
// reconciliation. alloc() is idempotent per key (a duplicate load/entered
// event for an already-resident page returns its current slot).
//
// Exhaustion policy: alloc fails (-1) rather than evicting; unallocated pages
// are the safe failure mode (particles there will skip physics / spill to the
// store ring once P4 lands).
// ─────────────────────────────────────────────────────────────────────────────
class VoxelLodArena {
public:
    struct SlotInfo {
        int32_t lod = -1;
        int32_t x = 0, y = 0, z = 0;   // page coord at `lod`
        uint32_t generation = 0;       // bumped on every (re)alloc
    };

    void configure(int32_t p_max_pages) {
        max_pages = p_max_pages;
        slots.clear();
        slots.resize(max_pages);
        freelist.clear();
        // Freelist in reverse so slot 0 hands out first (stable for debugging).
        for (int32_t i = max_pages - 1; i >= 0; --i) {
            freelist.push_back((uint32_t)i);
            slots[i].lod = -1;
            slots[i].generation = 0;
        }
        resident.clear();
    }

    // Allocate (or return the existing slot for) a page. Returns the slot
    // index, or -1 if the arena is exhausted.
    int32_t alloc(int32_t lod, int32_t px, int32_t py, int32_t pz) {
        LodPageKey key{ lod, px, py, pz };
        const uint32_t *existing = resident.getptr(key);
        if (existing) {
            return (int32_t)*existing;
        }
        if (freelist.empty()) {
            return -1;
        }
        const uint32_t slot = freelist.back();
        freelist.pop_back();
        SlotInfo &si = slots[slot];
        si.lod = lod;
        si.x = px; si.y = py; si.z = pz;
        si.generation = (si.generation + 1) & 0xFFu;  // 8-bit gen in table flags
        resident.insert(key, slot);
        return (int32_t)slot;
    }

    // Number of unallocated slots (P5 stats).
    int32_t free_count() const { return (int32_t)freelist.size(); }

    // Free the page at (lod, coord) if resident. Returns the detached slot
    // index so the caller can defer reuse (migration may need to read the
    // page's still-intact GPU data first), or -1 if the key was not resident.
    // The slot is NOT returned to the freelist here — call release_slot().
    int32_t free(int32_t lod, int32_t px, int32_t py, int32_t pz) {
        LodPageKey key{ lod, px, py, pz };
        const uint32_t *slotp = resident.getptr(key);
        if (!slotp) {
            return -1;
        }
        const uint32_t slot = *slotp;
        resident.erase(key);
        slots[slot].lod = -1;  // generation kept — stale readers detect via gen
        return (int32_t)slot;
    }

    // Return a previously freed slot to the freelist for reuse.
    void release_slot(uint32_t slot) {
        if (slot < slots.size()) {
            freelist.push_back(slot);
        }
    }

    // Slot index for a resident page, or -1.
    int32_t find(int32_t lod, int32_t px, int32_t py, int32_t pz) const {
        const uint32_t *slotp = resident.getptr(LodPageKey{ lod, px, py, pz });
        return slotp ? (int32_t)*slotp : -1;
    }

    const SlotInfo &slot(uint32_t i) const { return slots[i]; }

    int32_t used_pages()   const { return (int32_t)resident.size(); }
    int32_t max_page_cap() const { return max_pages; }

private:
    int32_t max_pages = 0;
    std::vector<SlotInfo> slots;
    std::vector<uint32_t> freelist;
    HashMap<LodPageKey, uint32_t, LodPageKeyHash> resident;
};

}  // namespace godot
