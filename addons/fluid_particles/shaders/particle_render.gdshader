// Particle sphere point-sprite shader for Godot 4
// Translated from ParticleShader (particledisplay.hpp)
//
// Per-vertex inputs (from ArrayMesh PRIMITIVE_POINTS):
//   VERTEX   – particle position (grid coordinates)
//   COLOR    – rgba; alpha > ~0.78 (200/255) → solid, else liquid
//   CUSTOM0  – vec4(attraction_force, opacity_fade, neighbors_filled, 0)
//
// Rendering:
//   Solid particles   : ray-sphere intersection → triplanar texture + directional light
//   Liquid particles  : ray-sphere intersection → screen-space refraction + Fresnel
//   Depth-only pass   : controlled by the "depth_only" uniform (set externally)

shader_type spatial;
render_mode unshaded, depth_draw_always;//, cull_disabled, skip_vertex_transform;
uniform sampler2D depth_texture : hint_depth_texture, repeat_disable, filter_nearest;
// ── Uniforms ──────────────────────────────────────────────────────────────────
uniform sampler2D texture1       : source_color, filter_linear_mipmap, repeat_enable;
uniform sampler2D screen_texture : hint_screen_texture, filter_linear;
uniform vec3 water_tint : source_color;
uniform float     screen_height  = 1024.0;
uniform float     depth_only     = 0.0;  // 1.0 = depth-only pass

// ── Varyings ─────────────────────────────────────────────────────────────────
varying flat vec4  v_world_pos;   // centre of sphere in world space
varying flat vec4  v_color;
varying flat float v_radius;
varying flat vec4  v_cam_pos;

// ─────────────────────────────────────────────────────────────────────────────
void vertex() {
	float neighbors_filled = CUSTOM0.z;

	// Kill invisible or fully-buried particles
	if (neighbors_filled < 1.5) {
		POSITION    = vec4(0.0);
		POINT_SIZE  = 0.0;
	   
	}

	bool is_solid = (COLOR.a > (200.0 / 255.0));

	// Radius calculation (mirrors C++ shader)
	float radius_grow = 0.125;
	float radius      = 1.125;
	vec4  final_color = COLOR;

	if (!is_solid) {
		radius      = 0.5;
		radius_grow = 0.0;
		// Surface-coverage colour blend (liquid look)
		float blend = 0.0f;//pow(1.0 - neighbors_filled / 15.0, 3.0);
		final_color.rgb = mix(COLOR.rgb, vec3(0.5), blend);
	}

	radius += neighbors_filled * radius_grow;
	radius  = max(radius, 0.0);

	v_radius = radius;
	v_color  = final_color;

	// World-space centre
	vec4 world = MODEL_MATRIX * vec4(VERTEX, 1.0);
	v_world_pos = world;

	// Camera world position (extract from inverse view)
	v_cam_pos.xyz = CAMERA_POSITION_WORLD;

	// Eye-space position for point size calculation
	vec4 eye_pos = VIEW_MATRIX * world;
	//POSITION = PROJECTION_MATRIX * eye_pos;
	
	float dist        = length(eye_pos.xyz);
	float focal_len   = PROJECTION_MATRIX[1][1];
	POINT_SIZE = (radius * VIEWPORT_SIZE.y) / max(dist, 0.001);
}
//
//// ─────────────────────────────────────────────────────────────────────────────
void fragment() {
	// Discard outside circle
	vec2  puv    = POINT_COORD - vec2(0.5);
	float dist_sq = dot(puv, puv);
	if (dist_sq > 0.25) discard;

	// ── Depth-only pass ───────────────────────────────────────────────────────
	if (depth_only > 0.5) {
		ALBEDO = vec3(0.0);
		DEPTH  = FRAGCOORD.z;
		
	}else{
	bool is_solid = (v_color.a > (200.0 / 255.0));

	// ── Reconstruct world-space ray ───────────────────────────────────────────
	vec2  screen_size = vec2(VIEWPORT_SIZE);
	vec2  ndc_xy      = (FRAGCOORD.xy / screen_size) * 2.0 - 1.0;
	float ndc_z       = FRAGCOORD.z * 2.0 - 1.0;

	vec4 clip_pos     = vec4(ndc_xy, ndc_z, 1.0);
	vec4 view_pos     = INV_PROJECTION_MATRIX * clip_pos;
	view_pos         /= view_pos.w;
	vec4 world_pos    = INV_VIEW_MATRIX * view_pos;

	vec3 ray_dir      = normalize(world_pos.xyz - v_cam_pos.xyz);

	// ── Ray-sphere intersection ───────────────────────────────────────────────
	vec3  oc           = v_cam_pos.xyz - v_world_pos.xyz;
	float a            = dot(ray_dir, ray_dir);
	float b            = 2.0 * dot(oc, ray_dir);
	float c            = dot(oc, oc) - v_radius * v_radius;
	float discriminant = b * b - 4.0 * a * c;

	if (discriminant < 0.0) discard;

	float t              = (-b - sqrt(discriminant)) / (2.0 * a);
	vec3  hit            = v_cam_pos.xyz + t * ray_dir;
	float depth = textureLod(depth_texture, SCREEN_UV, 0.0).r;
	vec4 upos = INV_PROJECTION_MATRIX * vec4(SCREEN_UV * 2.0 - 1.0, depth, 1.0);
	// Write correct sphere depth
	vec4 clip_hit = PROJECTION_MATRIX * VIEW_MATRIX * vec4(hit, 1.0);
	
	DEPTH = (clip_hit.z / clip_hit.w) ;
	
	vec3 normal = normalize(hit - v_world_pos.xyz);

	// ── Liquid particle — screen-space refraction + Fresnel ──────────────────
	if (!is_solid) {
		float distortion_strength = 0.02;
		vec2  distortion          = normal.xy * distortion_strength;
		vec2  screen_uv           = FRAGCOORD.xy / screen_size;
		vec2  distorted_uv        = clamp(screen_uv + distortion, vec2(0.0), vec2(1.0));

		vec3 bg_color = texture(screen_texture, distorted_uv).rgb;

		vec3  view_dir = normalize(v_cam_pos.xyz - hit);
		float fresnel  = pow(1.0 - abs(dot(view_dir, normal)), 3.0);

		vec3 final_color = bg_color * water_tint + v_color.rgb + vec3(fresnel * 0.143);
		ALBEDO           = final_color;
		ALPHA            = 1.0;
	  
	}else{

	// ── Solid particle — directional light (from above) + triplanar texture ──
	float light = max(dot(normal, vec3(0.0, 1.0, 0.0)), 0.0);

	vec3  abs_n  = abs(normal);
	float tot    = abs_n.x + abs_n.y + abs_n.z + 0.0001;
	vec3  w      = abs_n / tot;

	float dist_to_cam = length(v_cam_pos.xyz - hit);

	vec3 base_color;
	if (dist_to_cam > 50.0) {
		// Simple planar mapping at range
		vec2 uv_far = mod(hit.xz * 0.1, 1.0);
		base_color  = texture(texture1, uv_far).rgb * v_color.rgb;
	} else {
		vec2 uv_yz = mod(hit.yz * 0.1, 1.0);
		vec2 uv_xz = mod(hit.xz * 0.1, 1.0);
		vec2 uv_xy = mod(hit.xy * 0.1, 1.0);
		vec3 t1    = texture(texture1, uv_yz).rgb;
		vec3 t2    = texture(texture1, uv_xz).rgb;
		vec3 t3    = texture(texture1, uv_xy).rgb;
		base_color = (t1 * w.x + t2 * w.y + t3 * w.z) * v_color.rgb;
	}

	ALBEDO = base_color * light;
	ALPHA  = v_color.a;
}}}
