#[compute]
#version 450

// ─── Depth Sort Key shader ────────────────────────────────────────────────────
// Computes camera-distance squared for each particle and writes it to
// sort_keys[].  The host then runs a radix/bitonic sort over these keys
// (paired with runnable_indices[]) to achieve back-to-front rendering.
// ─────────────────────────────────────────────────────────────────────────────

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant, std430) uniform PushConstants {
    int   grid_w, grid_h, grid_d;
    int   num_particles;
    vec3 gravity;
    float surface_tension;
    float water_viscosity;
    float _pad0;
    vec3  camera_pos;       // reuses global_vel slot for camera position
    int   frame_count;
    int   neighbor_mode;
    int   num_runnable;
    int   vertex_stride_floats;
    int   attrib_stride_words;
    int   color_offset_words;
    int   custom0_offset_words;
} pc;

layout(set = 0, binding = 0, std430) buffer VertexBuffer   { float vtx[];           };
layout(set = 0, binding = 1, std430) buffer AttribBuffer   { uint  _unused_attr[];  };
layout(set = 0, binding = 2, std430) buffer ChunkBuffer    { int   _unused_chunk[]; };
layout(set = 0, binding = 3, std430) buffer RunnableBuffer { int   runnable_indices[]; };
layout(set = 0, binding = 4, std430) buffer SortKeyBuffer  { float sort_keys[];        };

vec3 get_position(int idx) {
    int base = idx * pc.vertex_stride_floats;
    return vec3(vtx[base + 0], vtx[base + 1], vtx[base + 2]);
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (int(gid) >= pc.num_particles) return;

    vec3 position = get_position(int(gid));

    // Skip inactive particles (NaN sentinel) — push them to back of sort
    if (!(position.x == position.x)) {
        sort_keys[gid] = 1e30;
        return;
    }

    vec3 to_cam = pc.camera_pos - position;
    sort_keys[gid] = dot(to_cam, to_cam);
}
