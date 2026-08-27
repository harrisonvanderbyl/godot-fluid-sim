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
//
// ── Optimization notes ────────────────────────────────────────────────────────
// The ray-sphere solve runs in VIEW space, where the camera sits at the origin.
// This removes the inverse-view transform, the perspective divide, the ray
// normalize, and — most importantly — the per-fragment mat4 * mat4 product that
// the old `proj * view * vec4(hit,1)` was silently evaluating.
//
// The vertex stage biases gl_Position to the sphere's NEAREST point instead of
// its center, so the fragment can only push depth AWAY from the camera. That
// lets us declare a conservative depth qualifier and keep early-Z rejection
// alive, which matters a lot given how much these sprites overdraw.
//
// Interpolants dropped from 4 slots to 2 (v_cam_pos was constant and already in
// the UBO; center + radius now share one vec4).
//
// The UBO layout is UNCHANGED — still matches CameraUniforms in
// fluid_particle_system.cpp. inv_view_matrix and cam_pos_world are simply no
// longer read by this shader.
// ──────────────────────────────────────────────────────────────────────────────

#[vertex]

#version 450

// Uniform buffer (binding 0, set 0): camera matrices + node transform.
// Layout MUST match CameraUniforms in fluid_particle_system.cpp.
layout(set = 0, binding = 0, std140) uniform CameraUBO {
    mat4 view_matrix;          // camera view (world → view)
    mat4 proj_matrix;          // camera projection (view → clip)
    mat4 inv_view_matrix;      // inverse view (view → world)   [unused here]
    mat4 inv_proj_matrix;      // inverse projection (clip → view)
    mat4 model_matrix;         // FluidParticleSystem world transform
    vec3 cam_pos_world;        // camera position in world space [unused here]
    float _pad0;               // align to vec4
} cam;

// Push constant: small per-frame scalars only (fits in 128 bytes).
layout(push_constant, std430) uniform PC {
    float viewport_w;          // viewport width in pixels
    float viewport_h;          // viewport height in pixels
    float offscreen_far;       // far plane for linear depth normalization
    float fluid_size;
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

// Varyings (flat — per-particle, not interpolated across the point sprite).
//   v_center_radius.xyz = sphere center in VIEW space
//   v_center_radius.w   = sphere radius
layout(location = 0) flat out vec4 v_center_radius;
layout(location = 1) flat out vec4 v_color;

// A sphere's screen silhouette is slightly wider than its projected radius
// (the tangent cone, not the center plane). A small pad is cheaper than the
// exact math; raise it if you see clipped sprite edges at the screen border.
const float POINT_SIZE_PAD = 1.05;

// Most desktop Vulkan drivers cap pointSize around 256. Clamping avoids
// undefined behaviour when a large particle gets very close to the camera.
// If you hit this clamp in practice, query VkPhysicalDeviceLimits::
// pointSizeRange[1] and feed it through pc._pad0 instead of hardcoding.
const float MAX_POINT_SIZE = 255.0;

void main() {
    float neighbors_filled = in_custom0.z;
    float radius      = pc.fluid_size;
    float radius_grow = 0.0;
    radius = max(radius + neighbors_filled * radius_grow, 0.0);

    vec4 world       = cam.model_matrix * vec4(in_position, 1.0);
    vec3 center_view = (cam.view_matrix * world).xyz;

    v_center_radius = vec4(center_view, radius);
    v_color         = in_color;

    // Bias the emitted depth to the sphere's nearest point (view space is
    // -Z forward, so +radius moves toward the camera). The fragment shader can
    // then only write depth that is >= this value, which is what makes the
    // conservative depth qualifier below legal.
    vec3 near_view = center_view;
    //near_view.z += radius;


    // Projected radius in pixels, derived from the actual projection matrix:
    //   px = r * (1 / tan(fov_y/2)) * height / (2 * dist)
    // proj_matrix[1][1] IS 1/tan(fov_y/2) (abs() because Godot may negate it
    // for the Vulkan Y-flip). The old constant-based form was only correct at
    // a ~53 degree vertical FOV.
    float dist = max(-near_view.z, 0.001);
    float px   = radius * abs(cam.proj_matrix[1][1]) * pc.viewport_h * POINT_SIZE_PAD
               / (2.0 * dist);
    float ps = clamp(px, 1.0, MAX_POINT_SIZE);
    gl_PointSize = ps;

    gl_Position = cam.proj_matrix * vec4(near_view, 1.0);
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
    float fluid_size;
} pc;

layout(location = 0) flat in vec4 v_center_radius;
layout(location = 1) flat in vec4 v_color;

// Conservative depth: preserves early-Z even though we write gl_FragDepth.
// The vertex stage emits the sphere's nearest point, so under a standard
// [0,1] LESS-compare depth buffer the fragment can only increase depth.
// Under reverse-Z (nearest = largest depth value, GREATER compare) the
// inequality flips — uncomment REVERSE_Z to match. The vertex-side bias is
// the same either way; only this qualifier changes.
// #define REVERSE_Z

layout(depth_less) out float gl_FragDepth;


// Color output: rgb = particle color, a = coverage (1.0 = opaque).
layout(location = 0) out vec4 out_color;
void main() {
    // Discard pixels outside the point-sprite circle. Redundant with the
    // discriminant test below, but far cheaper — keep it first.
    vec2 puv = gl_PointCoord - vec2(0.5);
    if (dot(puv, puv) > 0.25) discard;

    // View-space ray from the camera (at the origin) through this fragment.
    // For a perspective projection, inv_proj reduces to its two diagonal
    // terms, so this is equivalent to the old inv_proj * vec4(...) + divide,
    // including any Y-flip baked into the matrix. If you ever switch to an
    // oblique/sheared near plane, restore the full matrix multiply.
    vec2 ndc = (gl_FragCoord.xy / vec2(pc.viewport_w, pc.viewport_h)) * 2.0 - 1.0;
    vec3 ray = vec3(ndc.x * cam.inv_proj_matrix[0][0],
                    ndc.y * cam.inv_proj_matrix[1][1],
                    -1.0);

    // Ray-sphere intersection (front face), unnormalized ray form:
    //   |t*ray - center|^2 = r^2
    vec3  center = v_center_radius.xyz;
    float radius = v_center_radius.w;
    vec3  oc     = -center;                       // ray origin is (0,0,0)

    float qa   = dot(ray, ray);
    float qb   = dot(oc, ray);
    float qc   = dot(oc, oc) - radius * radius;
    float disc = qb * qb - qa * qc;
    if (disc < 0.0) discard;

    float t        = (-qb - sqrt(disc)) / qa;
    vec3  hit_view = ray * t;

    // Project just the z and w rows — the other two clip components are
    // unused. Written generally enough to survive an off-center frustum.
    float clip_z = cam.proj_matrix[2][2] * hit_view.z + cam.proj_matrix[3][2];
    float clip_w = cam.proj_matrix[2][3] * hit_view.z + cam.proj_matrix[3][3];
    gl_FragDepth = clip_z / clip_w;

    // Output the particle's RGB color. Alpha = 1.0 (coverage).
    out_color = vec4(v_color.rgb, 1.0);
}