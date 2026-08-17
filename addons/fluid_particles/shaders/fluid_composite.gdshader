shader_type spatial;
render_mode unshaded, depth_draw_opaque, cull_disabled;

// fluid_tex:       RGBA8 color (rgb = particle color, a = coverage)
// fluid_depth_tex: D32 hardware depth (NDC [0,1], same format as main depth)
// screen_tex:      main scene color
// depth_tex:       main scene depth (NDC [0,1])
uniform sampler2D fluid_tex       : filter_nearest, repeat_disable;
uniform sampler2D fluid_depth_tex : filter_nearest, repeat_disable;
uniform sampler2D screen_tex      : hint_screen_texture, filter_linear;
uniform sampler2D depth_tex       : hint_depth_texture, filter_nearest, repeat_disable;

uniform float blur_radius         = 2.0;
uniform float tint_strength       = 0.6;
uniform float refraction_strength = 0.04;
uniform float absorption_dist     = 4.0;
uniform vec3  fluid_tint          : source_color = vec3(0.7, 0.85, 1.0);
uniform vec2  fluid_pixel_size    = vec2(0.001953, 0.001953);
uniform float normal_smooth       = 2.0;  // depth smoothing radius for normals

uniform float offscreen_far      = 100.0;
uniform float offscreen_near     = 0.1;
uniform mat4  offscreen_inv_view = mat4(1.0);
uniform mat4  offscreen_inv_proj = mat4(1.0);

varying vec2 v_uv;

// Reconstruct world position from a UV + fluid depth sample.
vec3 world_from_fluid(vec2 uv, float d) {
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 vp = offscreen_inv_proj * vec4(ndc, d, 1.0);
    vp /= vp.w;
    return (offscreen_inv_view * vp).xyz;
}

// Smoothed fluid depth at a UV (gaussian-weighted, only where fluid exists).
float smoothed_fluid_depth(vec2 uv, float radius) {
    vec2 px = fluid_pixel_size * radius;
    float sum = 0.0; float wsum = 0.0;
    for (int j = -2; j <= 2; j++)
    for (int i = -2; i <= 2; i++) {
        vec2 ouv = uv + vec2(float(i), float(j)) * px;
        if (texture(fluid_tex, ouv).a < 0.5) continue;
        float d = texture(fluid_depth_tex, ouv).r;
        float w = exp(-float(i*i + j*j) * 0.5);
        sum += d * w;
        wsum += w;
    }
    return (wsum > 0.0001) ? sum / wsum : texture(fluid_depth_tex, uv).r;
}

void vertex() {
    POSITION = vec4(VERTEX.xy, 1.0, 1.0);
    v_uv = VERTEX.xy * 0.5 + 0.5;
}

void fragment() {
    // Sample the fluid color (3x3 average for blur) and depth (nearest, not
    // averaged — averaging depth across edges causes artifacts).
    vec2 px = fluid_pixel_size * blur_radius;
    vec3 col = vec3(0.0); float fd = 0.0; float w = 0.0;
    float fd_center = texture(fluid_depth_tex, v_uv).r;
    for (float j = -1.0; j <= 1.0; j+=0.25)
    for (float i = -1.0; i <= 1.0; i+=0.25) {
        vec2 ouv = v_uv + vec2(float(i), float(j)) * px;
        float present = texture(fluid_tex, ouv).a;
        if (present < 0.5) continue;
        col += texture(fluid_tex, ouv).rgb;
        w   += 1.0;
    }
    if (w < 0.5) discard;
    col /= w;
    fd  = fd_center;

    // Fluid depth is standard-Z (0=near, 1=far). Reconstruct the fluid
    // surface world position, then reproject through the MAIN camera's
    // PROJECTION_MATRIX * VIEW_MATRIX to get a DEPTH value that exactly
    // matches the scene depth buffer format.
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec4 view_pos = offscreen_inv_proj * vec4(ndc, fd, 1.0);
    view_pos /= view_pos.w;
    vec3 fluid_world = (offscreen_inv_view * view_pos).xyz;

    // Reproject through the main camera to get correct NDC depth.
    vec4 clip = PROJECTION_MATRIX * VIEW_MATRIX * vec4(fluid_world, 1.0);
    float fluid_ndc_z = clip.z / clip.w;

    // Main scene depth at this pixel (NDC [0,1], same format).
    float scene_ndc_z = textureLod(depth_tex, SCREEN_UV, 0.0).r;

    // If the scene geometry is closer than the fluid, discard — let the
    // scene show through. (Reverse-Z: higher = closer.)
    if (scene_ndc_z > fluid_ndc_z) discard;

    // ── Depth-based normal reconstruction ────────────────────────────────────
    // Sample smoothed depth at neighbors and reconstruct world positions,
    // then build a normal via cross product of the surface tangents.
    vec2 npx = fluid_pixel_size * normal_smooth;
    float d_r = smoothed_fluid_depth(v_uv + vec2(npx.x, 0.0), normal_smooth);
    float d_l = smoothed_fluid_depth(v_uv - vec2(npx.x, 0.0), normal_smooth);
    float d_u = smoothed_fluid_depth(v_uv + vec2(0.0, npx.y), normal_smooth);
    float d_d = smoothed_fluid_depth(v_uv - vec2(0.0, npx.y), normal_smooth);

    vec3 w_r = world_from_fluid(v_uv + vec2(npx.x, 0.0), d_r);
    vec3 w_l = world_from_fluid(v_uv - vec2(npx.x, 0.0), d_l);
    vec3 w_u = world_from_fluid(v_uv + vec2(0.0, npx.y), d_u);
    vec3 w_d = world_from_fluid(v_uv - vec2(0.0, npx.y), d_d);

    vec3 n_world = normalize(cross(w_r - w_l, w_u - w_d));
    if (dot(n_world, CAMERA_POSITION_WORLD - fluid_world) < 0.0) n_world = -n_world;
    vec3 n_view = normalize((VIEW_MATRIX * vec4(n_world, 0.0)).xyz);

    // Refraction: offset screen UV by the view-space normal's XY.
    vec2 refract_uv = clamp(SCREEN_UV + n_view.xy * refraction_strength,
                            vec2(0.0), vec2(1.0));
    vec3 screen_col = textureLod(screen_tex, refract_uv, 0.0).rgb;

    // Thickness from NDC depth difference.
    float thickness = max(fluid_ndc_z - scene_ndc_z, 0.0);
    float tint = tint_strength * exp(-thickness / max(absorption_dist, 0.001));

    ALBEDO = mix(screen_col, screen_col * fluid_tint + col, tint);
    ALPHA = 1.0;

    // Write the reprojected fluid depth (matches scene depth buffer format).
    DEPTH = fluid_ndc_z;
}