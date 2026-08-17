// ──────────────────────────────────────────────────────────────────────────────
// fluid_depth.glsl — RD render pipeline shader (vertex + fragment)
//
// Renders particle point-sprites directly into an RD framebuffer with:
//   - Color attachment (RGBA8): rgb = particle color, a = coverage (1.0)
//   - Depth attachment (D32_SFLOAT): hardware depth at the sphere surface
//
// The compositor samples the color texture for shading and the depth texture
// for full-precision depth — no packed depth encoding needed.
//
// Camera matrices are passed via a uniform buffer (binding 0) because they
// exceed Vulkan's 128-byte push constant limit. Small per-frame scalars
// (viewport size, far plane) use a push constant.
// ──────────────────────────────────────────────────────────────────────────────

#[vertex]

#version 450

// Uniform buffer (binding 0, set 0): camera matrices + node transform.
// Layout MUST match CameraUniforms in fluid_particle_system.cpp.
layout(set = 0, binding = 0, std140) uniform CameraUBO {
    mat4 view_matrix;          // camera view (world → view)
    mat4 proj_matrix;          // camera projection (view → clip)
    mat4 inv_view_matrix;      // inverse view (view → world)
    mat4 inv_proj_matrix;      // inverse projection (clip → view)
    mat4 model_matrix;         // FluidParticleSystem world transform
    vec3 cam_pos_world;        // camera position in world space
    float _pad0;               // align to vec4
} cam;

// Push constant: small per-frame scalars only (fits in 128 bytes).
layout(push_constant, std430) uniform PC {
    float viewport_w;          // viewport width in pixels
    float viewport_h;          // viewport height in pixels
    float offscreen_far;       // far plane for linear depth normalization
    float _pad0;
} pc;

// Vertex attributes — match the render mesh's storage-buffer layout:
//   location 0: vec3 position (from vertex buffer)
//   location 1: vec4 color     (from attribute buffer, ARRAY_COLOR)
//   location 2: vec4 custom0   (from attribute buffer, ARRAY_CUSTOM0:
//                               x=attraction, y=opacity_fade,
//                               z=neighbors_filled, w=unused)
layout(location = 0) in vec3  in_position;
layout(location = 1) in vec4  in_color;
layout(location = 2) in vec4  in_custom0;

// Varyings (flat — per-particle, not interpolated across the point sprite)
layout(location = 0) flat out vec4  v_world_pos;
layout(location = 1) flat out vec4  v_color;
layout(location = 2) flat out float v_radius;
layout(location = 3) flat out vec3  v_cam_pos;

void main() {
    float neighbors_filled = in_custom0.z;
    bool  is_solid    = (in_color.a > (200.0 / 255.0));
    float radius      = is_solid ? 1.125 : 2.5;
    float radius_grow = is_solid ? 0.125 : 0.0;
    radius = max(radius + neighbors_filled * radius_grow, 0.0);

    v_radius    = radius;
    v_color     = in_color;
    vec4 world  = cam.model_matrix * vec4(in_position, 1.0);
    v_world_pos = world;
    v_cam_pos   = cam.cam_pos_world;

    vec4 view_pos = cam.view_matrix * world;
    gl_Position   = cam.proj_matrix * view_pos;

    // Point size: project the sphere radius to screen pixels.
    // Matches the original: POINT_SIZE = (radius * VIEWPORT_SIZE.y) / dist
    float dist = length(view_pos.xyz);
    gl_PointSize = (radius * pc.viewport_h) / max(dist, 0.001);
}

#[fragment]

#version 450

layout(set = 0, binding = 0, std140) uniform CameraUBO {
    mat4 view_matrix;
    mat4 proj_matrix;
    mat4 inv_view_matrix;
    mat4 inv_proj_matrix;
    mat4 model_matrix;
    vec3 cam_pos_world;
    float _pad0;
} cam;

layout(push_constant, std430) uniform PC {
    float viewport_w;
    float viewport_h;
    float offscreen_far;
    float _pad0;
} pc;

layout(location = 0) flat in vec4  v_world_pos;
layout(location = 1) flat in vec4  v_color;
layout(location = 2) flat in float v_radius;
layout(location = 3) flat in vec3  v_cam_pos;

// Color output: rgb = particle color, a = coverage (1.0 = opaque).
// Depth is written to gl_FragDepth (the D32 hardware depth texture).
layout(location = 0) out vec4 out_color;

void main() {
    // Discard pixels outside the point-sprite circle.
    vec2 puv = gl_PointCoord - vec2(0.5);
    if (dot(puv, puv) > 0.25) discard;

    // Reconstruct the view ray from the fragment's NDC position.
    vec2 ndc_xy   = (gl_FragCoord.xy / vec2(pc.viewport_w, pc.viewport_h)) * 2.0 - 1.0;
    vec4 view_pos = cam.inv_proj_matrix * vec4(ndc_xy, gl_FragCoord.z, 1.0);
    view_pos /= view_pos.w;
    vec3 world_pos = (cam.inv_view_matrix * view_pos).xyz;
    vec3 ray_dir   = normalize(world_pos - v_cam_pos);

    // Ray-sphere intersection (front face).
    vec3  oc   = v_cam_pos - v_world_pos.xyz;
    float b    = dot(oc, ray_dir);
    float c    = dot(oc, oc) - v_radius * v_radius;
    float disc = b * b - c;
    if (disc < 0.0) discard;
    vec3 hit = v_cam_pos + (-b - sqrt(disc)) * ray_dir;

    // Write the correct depth at the sphere surface.
    vec4 clip_hit = cam.proj_matrix * cam.view_matrix * vec4(hit, 1.0);
    gl_FragDepth = clip_hit.z / clip_hit.w;

    // Output the particle's RGB color. Alpha = 1.0 (coverage).
    out_color = vec4(v_color.rgb, 1.0);
}
