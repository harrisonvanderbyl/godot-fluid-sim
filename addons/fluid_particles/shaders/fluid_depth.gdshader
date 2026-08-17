shader_type spatial;
render_mode unshaded, depth_draw_always, cull_disabled;   // OPAQUE — never write ALPHA

uniform float offscreen_far = 100.0;

varying flat vec4  v_world_pos;
varying flat vec4  v_color;
varying flat float v_radius;
varying flat vec3  v_cam_pos;

// Pack a [0,1] value into two 8-bit channels → ~16-bit precision.
vec2 encode_depth(float v) {
    v = clamp(v, 0.0, 1.0);
    float hi = floor(v * 255.0) / 255.0;
    float lo = fract(v * 255.0);
    return vec2(hi, lo);
}

void vertex() {
    float neighbors_filled = CUSTOM0.z;
    bool  is_solid    = (COLOR.a > (200.0 / 255.0));
    float radius      = is_solid ? 1.125 : 0.5;
    float radius_grow = is_solid ? 0.125 : 0.0;
    radius = max(radius + neighbors_filled * radius_grow, 0.0);

    v_radius    = radius;
    v_color     = COLOR;
    vec4 world  = MODEL_MATRIX * vec4(VERTEX, 1.0);
    v_world_pos = world;
    v_cam_pos   = CAMERA_POSITION_WORLD;

    float dist = length((VIEW_MATRIX * world).xyz);
    POINT_SIZE = (radius * VIEWPORT_SIZE.y) / max(dist, 0.001);
}

void fragment() {
    vec2 puv = POINT_COORD - vec2(0.5);
    if (dot(puv, puv) > 0.25) discard;

    vec2 ndc_xy   = (FRAGCOORD.xy / VIEWPORT_SIZE) * 2.0 - 1.0;
    vec4 view_pos = INV_PROJECTION_MATRIX * vec4(ndc_xy, FRAGCOORD.z, 1.0);
    view_pos /= view_pos.w;
    vec3 world_pos = (INV_VIEW_MATRIX * view_pos).xyz;
    vec3 ray_dir   = normalize(world_pos - v_cam_pos);

    vec3  oc   = v_cam_pos - v_world_pos.xyz;
    float b    = dot(oc, ray_dir);
    float c    = dot(oc, oc) - v_radius * v_radius;
    float disc = b * b - c;
    if (disc < 0.0) discard;
    vec3 hit = v_cam_pos + (-b - sqrt(disc)) * ray_dir;

    vec4 clip_hit = PROJECTION_MATRIX * VIEW_MATRIX * vec4(hit, 1.0);
    DEPTH = clip_hit.z / clip_hit.w;

    float view_z = (VIEW_MATRIX * vec4(hit, 1.0)).z;
    float lin    = clamp(-view_z / offscreen_far, 0.0, 1.0);

    // R,G = packed depth · B = color channel · A untouched (1.0) → opaque + coverage.
    vec2 dpack = encode_depth(lin);
    ALBEDO = vec3(dpack.x, dpack.y, v_color.r);   // B carries one color channel
}